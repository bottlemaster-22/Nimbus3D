//
//  CaptureGyroSampler.swift
//  Capture
//
//  Angular velocity, sampled fast and read back at a specific instant.
//
//  Why this exists rather than reading `ARFrame`'s motion: the blur meter and
//  the QC weight both want the turn rate at the MIDPOINT of the exposure, not
//  at the moment the frame was delivered. A frame with an 8 ms shutter that
//  arrives 20 ms after the shutter closed is described by the gyro sample from
//  24 ms ago, and at 200 deg/s that difference is most of the answer.
//
//  CMDeviceMotion.timestamp and ARFrame.timestamp are both seconds on the same
//  mach-continuous clock, which is what makes the interpolation below legal
//  and what makes the pre-pass's camera-to-IMU offset sweep (F1) meaningful.
//

import CoreMotion
import Foundation
import simd

/// A small ring buffer of gyro samples, queryable at an arbitrary timestamp.
///
/// Thread-safe: `CMMotionManager` delivers on its own operation queue and the
/// capture pipeline reads from its own serial queue, so every access goes
/// through one lock. The buffer is tiny and the critical section is a handful
/// of comparisons, so contention is not a concern at 100 Hz.
final class CaptureGyroSampler: @unchecked Sendable {

    struct Sample {
        var timestamp: TimeInterval
        var rotationRate: SIMD3<Float>  // rad/s, device body frame
    }

    /// 100 Hz for four seconds. Long enough to cover any plausible camera-to-
    /// IMU offset plus delivery latency, short enough to stay in cache.
    private static let capacity = 400

    private let motionManager = CMMotionManager()
    private let queue: OperationQueue
    private let lock = NSLock()

    private var ring = [Sample](
        repeating: Sample(timestamp: 0, rotationRate: .zero),
        count: CaptureGyroSampler.capacity
    )
    private var writeIndex = 0
    private var filled = 0

    /// True once at least one sample has arrived. `false` means the answers
    /// below are zeros, and the caller should say so rather than pretend the
    /// phone is perfectly still.
    private(set) var isReceivingSamples = false

    init() {
        queue = OperationQueue()
        queue.name = "\(BrandConfig.bundleIdentifier).capture.motion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
    }

    // MARK: - Lifecycle

    /// Starts device motion at 100 Hz.
    ///
    /// Device motion rather than raw gyro: `CMDeviceMotion.rotationRate` is
    /// bias-corrected against the accelerometer and magnetometer, and the bias
    /// on a raw MEMS gyro is the same order as the rates the blur meter cares
    /// about at slow pans.
    ///
    /// - Returns: false when the device has no device-motion support, in which
    ///   case every reading is zero and the caller must not claim otherwise.
    @discardableResult
    func start() -> Bool {
        guard motionManager.isDeviceMotionAvailable else {
            let message = "Device motion is unavailable; angular velocity will be zero "
                    + "and the blur meter cannot work."
            CaptureLog.session.error("\(message, privacy: .public)")
            return false
        }
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: queue
        ) { [weak self] motion, _ in
            guard let self, let motion else { return }
            self.append(
                Sample(
                    timestamp: motion.timestamp,
                    rotationRate: SIMD3<Float>(
                        Float(motion.rotationRate.x),
                        Float(motion.rotationRate.y),
                        Float(motion.rotationRate.z)
                    )
                )
            )
        }
        return true
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
    }

    // MARK: - Query

    /// Angular velocity at `timestamp`, linearly interpolated between the two
    /// bracketing samples. Falls back to the nearest sample when the query is
    /// outside the buffered window (which happens for the first frame, before
    /// any motion has been delivered).
    func angularVelocity(at timestamp: TimeInterval) -> SIMD3<Float> {
        lock.lock()
        defer { lock.unlock() }
        guard filled > 0 else { return .zero }

        var before: Sample?
        var after: Sample?
        for offset in 0..<filled {
            let sample = ring[(writeIndex - 1 - offset + ring.count * 2) % ring.count]
            if sample.timestamp <= timestamp {
                before = sample
                break
            }
            after = sample
        }

        switch (before, after) {
        case let (.some(a), .some(b)) where b.timestamp > a.timestamp:
            let t = Float((timestamp - a.timestamp) / (b.timestamp - a.timestamp))
            let clamped = Swift.max(0, Swift.min(1, t))
            return a.rotationRate + (b.rotationRate - a.rotationRate) * clamped
        case let (.some(a), _):
            return a.rotationRate
        case let (_, .some(b)):
            return b.rotationRate
        default:
            return .zero
        }
    }

    /// Magnitude of the angular velocity at `timestamp`, rad/s. This is what
    /// the bracket guard wants.
    func angularSpeed(at timestamp: TimeInterval) -> Float {
        simd_length(angularVelocity(at: timestamp))
    }

    /// How far the camera actually TURNED between two instants, radians.
    ///
    /// This exists because the blur meter was asking the wrong question.
    /// It sampled the instantaneous rate at the midpoint of the exposure
    /// and multiplied by the shutter time, which is only correct when the
    /// rate is constant across the window. Hand tremor is not constant: it
    /// is an oscillation at roughly 8 to 12 Hz, so within one 1/60 s
    /// shutter the phone swings out and most of the way back again. The
    /// instantaneous rate peaks as it passes through centre, so sampling
    /// there and multiplying reports a large smear for a hand that has
    /// barely moved by the time the shutter closes.
    ///
    /// The owner reported having to hold the phone like a gimbal to stop
    /// the app complaining. This is why. Integrating the rate VECTOR lets
    /// the out and back cancel the way the photons do, while a genuine pan
    /// integrates to exactly what the old formula gave, because for a
    /// constant rate the integral IS rate times duration. So this is a
    /// better estimator of the same physical quantity, not a new quantity:
    /// CONTRACTS.md section 7 fixes the smear-to-pixels conversion, and it
    /// is untouched.
    ///
    /// Trapezoidal, over the interpolated rate. The gyro runs at 100 Hz so
    /// a 1/60 s window holds only about two raw samples; the sub-steps do
    /// not invent resolution the sampler does not have, they just stop the
    /// single midpoint reading from standing in for the whole window.
    func netRotationRadians(from start: TimeInterval, to end: TimeInterval) -> Float {
        guard end > start, isReceivingSamples else { return 0 }
        let steps = 8
        let step = (end - start) / Double(steps)
        var integral = SIMD3<Float>.zero
        var previous = angularVelocity(at: start)
        for i in 1...steps {
            let current = angularVelocity(at: start + step * Double(i))
            integral += (previous + current) * (0.5 * Float(step))
            previous = current
        }
        return simd_length(integral)
    }

    // MARK: - Private

    private func append(_ sample: Sample) {
        lock.lock()
        ring[writeIndex] = sample
        writeIndex = (writeIndex + 1) % ring.count
        filled = Swift.min(filled + 1, ring.count)
        isReceivingSamples = true
        lock.unlock()
    }
}
