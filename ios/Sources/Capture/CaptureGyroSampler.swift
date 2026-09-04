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
            CaptureLog.session.error(
                "Device motion is unavailable; angular velocity will be zero "
                    + "and the blur meter cannot work."
            )
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
    /// the blur meter and the bracket guard both actually want.
    func angularSpeed(at timestamp: TimeInterval) -> Float {
        simd_length(angularVelocity(at: timestamp))
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
