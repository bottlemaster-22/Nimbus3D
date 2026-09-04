//
//  CaptureTimeOffsetCalibrator.swift
//  Capture
//
//  THE CAMERA-TO-IMU TIME OFFSET, MEASURED WHILE THE USER WALKS (F1).
//
//  `CaptureBundle.cameraToIMUTimeOffsetSeconds` is documented as: add this to
//  a camera timestamp to get the IMU timestamp that really corresponds to it.
//  It matters because every motion-derived number in the scan - the blur
//  meter, the QC weight, the pre-pass's re-interpolated poses - reads the gyro
//  at a camera frame's timestamp, and if the two clocks are 15 ms apart then
//  at 200 degrees a second that lookup is three degrees wrong.
//
//  HOW IT IS MEASURED HERE, AND WHY THAT IS HONEST WORK AND NOT A GUESS.
//  Two independent witnesses describe the same rotation:
//
//    the camera   consecutive ARKit poses give an angle and a time, so
//                 angle / dt is how fast the phone turned according to VIO
//    the gyro     `CMDeviceMotion.rotationRate`, sampled at 100 Hz
//
//  The magnitude of a rotation is the same in any frame, so the two series
//  describe the same curve shifted in time by exactly the offset we want. The
//  sweep is CONTRACTS.md section 7's: -50 to +50 ms in 5 ms steps. For each
//  candidate offset the gyro is looked up at `frameTime + offset` and a
//  running Pearson correlation is kept; the winner is the offset whose curve
//  lines up best, refined to sub-step resolution by fitting a parabola through
//  the peak and its two neighbours.
//
//  This is the standard way a visual-inertial system initialises its time
//  offset (it is what Kalibr's initialiser does), and it is cheap: 21 gyro
//  interpolations per evaluated frame, no storage that grows with the length
//  of the walk.
//
//  WHEN IT REFUSES. Three ways, all reported rather than smoothed over:
//  too few samples, a camera that barely turned (a flat signal correlates with
//  everything), or a best correlation too low to mean anything. `result()`
//  returns nil in all three cases and `rejectionReason` says which, because
//  `CaptureBundle` documents nil as a real answer and a plausible-looking
//  zero would be a lie the pre-pass would then build on.
//
//  THE PRE-PASS STILL OWNS THE FINAL WORD. `PoseRefiner.calibrateTimeOffset`
//  re-derives the same number from tracked-feature reprojection error, which
//  is a stronger measurement because it uses the image content and not just
//  the rate magnitude. This one exists so a bundle that never reaches the
//  pre-pass still carries a real answer.
//

import Foundation
import simd

/// Correlates camera turn rate against gyro turn rate over a lag sweep.
///
/// Owned by, and only touched from, the ARKit delegate context (the main
/// actor). It calls into `CaptureGyroSampler`, which is itself thread-safe.
final class CaptureTimeOffsetCalibrator {

    /// Why the last `result()` refused to answer, in one plain sentence. nil
    /// when it did answer, or when it has not been asked yet.
    private(set) var rejectionReason: String?

    private let gyro: CaptureGyroSampler

    /// The candidate offsets, seconds, low to high.
    private let lags: [Double]

    // Running sums for one Pearson correlation per candidate offset. `x` is
    // the camera's turn rate and is shared; `y` is the gyro's, one series per
    // candidate.
    private var sampleCount = 0
    private var sumX: Double = 0
    private var sumXX: Double = 0
    private var sumY: [Double]
    private var sumYY: [Double]
    private var sumXY: [Double]

    private var previousSample: (timestamp: TimeInterval, rotation: Quaternion)?

    /// Camera samples waiting for the gyro to catch up. A candidate offset of
    /// +50 ms asks for a gyro reading that has not happened yet when the frame
    /// arrives, so every sample is held for `timeOffsetGyroSettleSeconds`
    /// before it is correlated. The queue never holds more than two or three
    /// entries at the evaluation rate.
    private var pending: [(timestamp: TimeInterval, speed: Float)] = []

    init(gyro: CaptureGyroSampler) {
        self.gyro = gyro

        var candidates: [Double] = []
        let maximum = CaptureTuning.timeOffsetSweepMaxSeconds
        let step = CaptureTuning.timeOffsetSweepStepSeconds
        var lag = -maximum
        // Built by counting steps rather than by accumulating, so floating
        // point drift cannot leave the last candidate at 0.0499999.
        let steps = Int((2 * maximum / step).rounded())
        for index in 0...Swift.max(0, steps) {
            lag = -maximum + Double(index) * step
            candidates.append(lag)
        }
        self.lags = candidates

        self.sumY = [Double](repeating: 0, count: candidates.count)
        self.sumYY = [Double](repeating: 0, count: candidates.count)
        self.sumXY = [Double](repeating: 0, count: candidates.count)
    }

    func reset() {
        sampleCount = 0
        sumX = 0
        sumXX = 0
        for index in sumY.indices {
            sumY[index] = 0
            sumYY[index] = 0
            sumXY[index] = 0
        }
        previousSample = nil
        pending.removeAll(keepingCapacity: true)
        rejectionReason = nil
    }

    // MARK: - Feeding

    /// Offers one evaluated frame's pose.
    ///
    /// - Parameters:
    ///   - timestamp: `ARFrame.timestamp`, seconds on the device's
    ///     mach-continuous clock - the same clock `CMDeviceMotion.timestamp`
    ///     uses, which is what makes the correlation legal.
    ///   - rotation: the frame's world -> camera rotation. Only the ANGLE
    ///     between consecutive rotations is used, and an angle is the same
    ///     number in any frame of reference, so no convention conversion is
    ///     needed or wanted here.
    func observe(timestamp: TimeInterval, rotation: Quaternion) {
        defer { drainPending(now: timestamp) }

        guard let previous = previousSample else {
            previousSample = (timestamp, rotation)
            return
        }
        previousSample = (timestamp, rotation)

        let dt = timestamp - previous.timestamp
        // A gap outside this range is a dropped run of frames, not a turn
        // rate: dividing by it would invent a spike the gyro never saw.
        guard dt > 0.01, dt < 0.5 else { return }

        let degrees = CaptureKeyframeSelector.angleDegrees(
            between: previous.rotation,
            and: rotation
        )
        let radiansPerSecond = Float(Double(degrees) * .pi / 180 / dt)
        // The camera sample is stamped at the MIDPOINT of the interval it was
        // measured over, because that is where a rate estimated from two
        // endpoints actually applies. Stamping it at either end would bias the
        // answer by half a frame, which is 33 ms at the evaluation rate and
        // would swamp the thing being measured.
        pending.append((timestamp: previous.timestamp + dt / 2, speed: radiansPerSecond))
    }

    private func drainPending(now: TimeInterval) {
        let horizon = now - CaptureTuning.timeOffsetGyroSettleSeconds
        while let first = pending.first, first.timestamp <= horizon {
            correlate(timestamp: first.timestamp, cameraSpeed: first.speed)
            pending.removeFirst()
        }
    }

    private func correlate(timestamp: TimeInterval, cameraSpeed: Float) {
        let x = Double(cameraSpeed)
        sampleCount += 1
        sumX += x
        sumXX += x * x
        for (index, lag) in lags.enumerated() {
            let y = Double(gyro.angularSpeed(at: timestamp + lag))
            sumY[index] += y
            sumYY[index] += y * y
            sumXY[index] += x * y
        }
    }

    // MARK: - Result

    /// The measured offset in seconds, or nil when the sweep had no clear
    /// answer. Safe to call more than once; it recomputes from the running
    /// sums and does not consume them.
    func result() -> Double? {
        guard gyro.isReceivingSamples else {
            rejectionReason =
                "The motion sensor sent nothing during this scan, so the "
                + "camera and motion clocks could not be lined up."
            return nil
        }
        guard sampleCount >= CaptureTuning.timeOffsetMinSamples else {
            rejectionReason =
                "The scan was too short to line up the camera and motion "
                + "clocks. About ten seconds of walking is enough."
            return nil
        }

        let n = Double(sampleCount)
        let meanX = sumX / n
        let varianceX = Swift.max(0, sumXX / n - meanX * meanX)
        guard Float(varianceX.squareRoot())
            >= CaptureTuning.timeOffsetMinCameraSpeedStdRadPerSec
        else {
            rejectionReason =
                "The phone was held too steady for the camera and motion "
                + "clocks to be lined up. This does not affect the scan."
            return nil
        }

        var correlations = [Double](repeating: 0, count: lags.count)
        var bestIndex = 0
        var bestCorrelation = -Double.infinity
        for index in lags.indices {
            let meanY = sumY[index] / n
            let varianceY = Swift.max(0, sumYY[index] / n - meanY * meanY)
            let covariance = sumXY[index] / n - meanX * meanY
            let denominator = (varianceX * varianceY).squareRoot()
            let r = denominator > 1e-9 ? covariance / denominator : 0
            correlations[index] = r
            if r > bestCorrelation {
                bestCorrelation = r
                bestIndex = index
            }
        }

        guard Float(bestCorrelation) >= CaptureTuning.timeOffsetMinCorrelation else {
            rejectionReason =
                "The camera and motion sensor did not agree closely enough to "
                + "line up their clocks, so that measurement is left out."
            return nil
        }

        rejectionReason = nil
        let refined = Self.refinePeak(
            lags: lags,
            correlations: correlations,
            peak: bestIndex
        )

        if bestIndex == 0 || bestIndex == lags.count - 1 {
            let message = "The camera-to-motion offset landed at the edge of the "
                    + "-50 to +50 ms sweep, so the real value may be outside "
                    + "it. Recorded as measured."
            CaptureLog.session.notice("\(message, privacy: .public)")
        }
        let message = "Camera-to-motion offset measured at "
                + "\(String(format: "%.1f", refined * 1000)) ms over "
                + "\(self.sampleCount) samples, correlation "
                + "\(String(format: "%.2f", bestCorrelation))."
        CaptureLog.session.notice("\(message, privacy: .public)")
        return refined
    }

    /// Sub-step refinement: fit a parabola through the peak and its two
    /// neighbours and take its vertex.
    ///
    /// Worth doing because the sweep step is 5 ms and the quantity being
    /// measured is often smaller than that. The vertex is clamped to half a
    /// step either side of the peak, because a parabola through three noisy
    /// points can otherwise place its vertex outside the interval it was fitted
    /// to, which would be an extrapolation dressed up as a measurement.
    static func refinePeak(
        lags: [Double],
        correlations: [Double],
        peak: Int
    ) -> Double {
        guard peak > 0, peak < lags.count - 1 else { return lags[peak] }
        let left = correlations[peak - 1]
        let centre = correlations[peak]
        let right = correlations[peak + 1]
        let denominator = left - 2 * centre + right
        guard abs(denominator) > 1e-12 else { return lags[peak] }
        var shift = 0.5 * (left - right) / denominator
        shift = Swift.max(-0.5, Swift.min(0.5, shift))
        let step = lags[peak + 1] - lags[peak]
        return lags[peak] + shift * step
    }
}
