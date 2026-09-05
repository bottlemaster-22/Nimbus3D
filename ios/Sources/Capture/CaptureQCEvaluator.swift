//
//  CaptureQCEvaluator.swift
//  Capture
//
//  The per-frame QC weight (F8), and every input that went into it.
//
//  `FrameQC` deliberately keeps all six inputs, not just the weight, because
//  the QC card has to be able to tell the user WHICH problem cost them
//  coverage. "Low quality" is not actionable; "you were walking too fast
//  through the hallway" is.
//
//  The weight is a product, not an average. That is the important design
//  choice: a frame that is perfectly sharp, perfectly exposed and perfectly
//  tracked but has no depth returns at all is not a three-quarters-good frame,
//  it is a frame with no geometry, and a product says so while a mean does not.
//

import ARKit
import Foundation
import simd

/// Turns raw per-frame measurements into a `FrameQC`.
///
/// Owned by, and only touched from, the capture pipeline's serial queue.
final class CaptureQCEvaluator {

    /// Previous frame's exposure, in EV, for the jump term.
    private var previousExposureEV: Float?

    func reset() {
        previousExposureEV = nil
    }

    /// Builds the QC record for one frame.
    ///
    /// - Parameters:
    ///   - angularVelocity: gyro reading at the exposure midpoint, rad/s.
    ///   - exposureDurationSeconds: `ARCamera.exposureDuration`.
    ///   - exposureOffsetEV: `ARCamera.exposureOffset`.
    ///   - sharpness: normalised variance of Laplacian, 0...1.
    ///   - depthValidFraction: usable returns in the native depth map, 0...1.
    ///   - trackingQuality: ARKit's own verdict.
    ///   - isBracketed: a deliberately dark frame is not penalised for the
    ///     exposure jump WE caused; that jump is signal, not a defect.
    func evaluate(
        angularVelocity: SIMD3<Float>,
        exposureDurationSeconds: Double,
        exposureOffsetEV: Float,
        sharpness: Float,
        depthValidFraction: Float,
        trackingQuality: TrackingQuality,
        isBracketed: Bool
    ) -> FrameQC {
        let angularSpeed = simd_length(angularVelocity)
        let blurPixels = CaptureTuning.motionBlurPixels(
            angularSpeedRadPerSec: angularSpeed,
            exposureDurationSeconds: exposureDurationSeconds
        )

        // Exposure is reported as an offset in EV from the metered target, so
        // the frame-to-frame jump is the difference of those offsets.
        let exposureJumpEV: Float
        if isBracketed {
            exposureJumpEV = 0
        } else if let previous = previousExposureEV {
            exposureJumpEV = abs(exposureOffsetEV - previous)
        } else {
            exposureJumpEV = 0
        }
        if !isBracketed {
            previousExposureEV = exposureOffsetEV
        }

        let weight = Self.weight(
            motionBlurPixels: blurPixels,
            sharpness: sharpness,
            depthValidFraction: depthValidFraction,
            exposureJumpEV: exposureJumpEV,
            trackingQuality: trackingQuality
        )

        return FrameQC(
            angularSpeedRadPerSec: angularSpeed,
            motionBlurPixels: blurPixels,
            sharpness: sharpness,
            depthValidFraction: depthValidFraction,
            exposureJumpEV: exposureJumpEV,
            trackingQuality: trackingQuality,
            weight: weight
        )
    }

    // MARK: - The weight

    /// The five factors, multiplied and clamped to 0...1.
    ///
    /// A zero weight does NOT mean "discard". `FrameQC.weight` is documented as
    /// the multiplier the trainer applies to a frame's photometric loss; the
    /// frame is still written, its pose is still a pose-graph constraint, and
    /// its depth is still geometry. Throwing data away is how you get a hole in
    /// the scan you cannot explain to the user.
    static func weight(
        motionBlurPixels: Float,
        sharpness: Float,
        depthValidFraction: Float,
        exposureJumpEV: Float,
        trackingQuality: TrackingQuality
    ) -> Float {
        // Blur: full marks up to the amber threshold, falling linearly to a
        // floor at `blurWeightFloorPixels`.
        //
        // The floor matters more than the slope, and it is the same 0.35 that
        // sharpness gets two lines below, for the same reason. `blurPixels` is
        // a PREDICTION (gyro rate times shutter, open loop); `sharpness` is a
        // MEASUREMENT of the same defect on the pixels that actually arrived.
        // Multiplying a prediction and a measurement of one problem squares
        // the penalty, so the measured term has to be free to have the last
        // word. Blur used to be the only one of the five factors allowed to
        // reach zero, and because the weight is a product that meant one
        // instantaneous gyro sample could delete a frame's whole photometric
        // contribution. A soft view of a corner nothing else saw is worth far
        // more than no view of it.
        let blurFactor = 0.35 + 0.65 * Self.ramp(
            value: motionBlurPixels,
            fullBelow: CaptureTuning.blurAmberPixels,
            zeroAbove: CaptureTuning.blurWeightFloorPixels
        )

        // Sharpness: normalised against the session, so this is "how sharp for
        // this scene", and the floor is generous because a genuinely
        // low-texture wall scores low while being perfectly usable.
        let sharpnessFactor = 0.35 + 0.65 * Swift.max(0, Swift.min(1, sharpness))

        // Depth validity: a frame with a quarter of its returns is still
        // useful photometrically; a frame with none is not geometry at all.
        let depthFactor = 0.25 + 0.75 * Swift.max(0, Swift.min(1, depthValidFraction))

        // Exposure jump: a full stop of movement between neighbours means a
        // photometric loss should not treat them as directly comparable.
        let exposureFactor = Self.ramp(
            value: exposureJumpEV,
            fullBelow: 0.25,
            zeroAbove: 2.0
        )

        let trackingFactor: Float
        switch trackingQuality {
        case .normal: trackingFactor = 1.0
        case .limitedRelocalizing: trackingFactor = 0.35
        case .limitedExcessiveMotion: trackingFactor = 0.25
        case .limitedInsufficientFeatures: trackingFactor = 0.45
        case .limitedInitializing: trackingFactor = 0.2
        case .notAvailable: trackingFactor = 0.0
        }

        let product =
            blurFactor * sharpnessFactor * depthFactor * exposureFactor
            * trackingFactor
        return Swift.max(0, Swift.min(1, product))
    }

    /// 1 below `fullBelow`, 0 above `zeroAbove`, linear in between.
    private static func ramp(value: Float, fullBelow: Float, zeroAbove: Float) -> Float {
        guard zeroAbove > fullBelow else { return value <= fullBelow ? 1 : 0 }
        if value <= fullBelow { return 1 }
        if value >= zeroAbove { return 0 }
        return 1 - (value - fullBelow) / (zeroAbove - fullBelow)
    }
}

// MARK: - ARKit tracking state bridging

extension TrackingQuality {
    /// Flattens `ARCamera.TrackingState` into the Codable enum Core defines.
    init(_ state: ARCamera.TrackingState) {
        switch state {
        case .notAvailable:
            self = .notAvailable
        case .normal:
            self = .normal
        case .limited(let reason):
            switch reason {
            case .initializing: self = .limitedInitializing
            case .excessiveMotion: self = .limitedExcessiveMotion
            case .insufficientFeatures: self = .limitedInsufficientFeatures
            case .relocalizing: self = .limitedRelocalizing
            @unknown default: self = .limitedInitializing
            }
        }
    }
}
