//
//  CaptureKeyframeSelector.swift
//  Capture
//
//  Which of the 60 frames a second actually get written.
//
//  A splat trainer wants baseline, not frame rate. Two frames 16 ms apart from
//  a hand-held phone are the same ray bundle photographed twice: they double
//  the storage, double the pre-pass's work, and add nothing an optimiser can
//  triangulate against. Sparse POSED keyframes - chosen on motion, gated on
//  quality, with a hard floor so a user standing still still leaves a
//  continuous temporal track for the submap builder - are what this file
//  produces.
//
//  The selector never looks at pixels. It is given the pose and the QC record
//  the pipeline already computed, so it costs a handful of comparisons per
//  frame and can run inside the ARKit delegate callback without stalling it.
//

import Foundation
import simd

/// Decides whether a delivered frame becomes a written keyframe.
///
/// Owned by, and only touched from, the capture pipeline's serial queue.
final class CaptureKeyframeSelector {

    /// Why a frame was accepted. Only used for the log and the HUD's "why is
    /// nothing being recorded" honesty, never written to disk.
    enum Reason: String {
        case first
        case baseline
        case rotation
        case timeout
        case bracket
    }

    private var lastKeyframePose: Pose?
    private var lastKeyframeTimestamp: TimeInterval?

    /// Rate limiter for the evaluation itself, separate from the keyframe
    /// decision: ARKit delivers at 60 Hz and there is no reason to compute a
    /// sharpness variance that often.
    private var lastEvaluationTimestamp: TimeInterval?

    func reset() {
        lastKeyframePose = nil
        lastKeyframeTimestamp = nil
        lastEvaluationTimestamp = nil
    }

    /// Whether this delivered frame is worth measuring at all.
    ///
    /// Call before doing any per-frame work. Returns true at most
    /// `CaptureTuning.frameEvaluationHz` times a second.
    func shouldEvaluate(at timestamp: TimeInterval) -> Bool {
        guard let last = lastEvaluationTimestamp else {
            lastEvaluationTimestamp = timestamp
            return true
        }
        guard timestamp - last >= 1.0 / CaptureTuning.frameEvaluationHz else {
            return false
        }
        lastEvaluationTimestamp = timestamp
        return true
    }

    /// The keyframe decision.
    ///
    /// - Parameters:
    ///   - pose: this frame's world -> camera pose.
    ///   - timestamp: `ARFrame.timestamp`.
    ///   - qc: the QC record already computed for this frame.
    ///   - isBracketRequested: a dark bracket was asked for and must be kept
    ///     whatever the motion says - it is the only correctly-exposed
    ///     evidence the scan has for a bright window.
    ///   - thermalLevel: when the phone is hot the minimum interval stretches,
    ///     because sustained writing is a meaningful share of the heat budget.
    /// - Returns: the reason to accept, or nil to skip.
    func decide(
        pose: Pose,
        timestamp: TimeInterval,
        qc: FrameQC,
        isBracketRequested: Bool,
        thermalLevel: ThermalLevel
    ) -> Reason? {
        guard let lastPose = lastKeyframePose,
            let lastTimestamp = lastKeyframeTimestamp
        else {
            return .first
        }

        let elapsed = timestamp - lastTimestamp

        // The hard floor: never two keyframes closer than the minimum
        // interval, stretched when the phone is hot.
        var minimumInterval = CaptureTuning.keyframeMinIntervalSeconds
        if thermalLevel >= CaptureTuning.thermalDegradeAt {
            minimumInterval *= CaptureTuning.thermalKeyframeIntervalMultiplier
        }

        if isBracketRequested, elapsed >= minimumInterval {
            return .bracket
        }

        guard elapsed >= minimumInterval else { return nil }

        // The ceiling: if nothing has been written for a while, write one
        // regardless of quality. A gap in the temporal track is a gap the
        // pre-pass's submap builder has to bridge with an assumption, and an
        // assumption is exactly what this app is trying not to make.
        if elapsed >= CaptureTuning.keyframeMaxIntervalSeconds {
            return .timeout
        }

        // Below the quality floor, wait for a better frame - the timeout above
        // guarantees the wait is bounded.
        guard qc.weight >= CaptureTuning.keyframeMinQCWeight else { return nil }

        let baseline = simd_distance(pose.center.simd, lastPose.center.simd)
        if baseline >= CaptureTuning.keyframeMinBaselineMeters {
            return .baseline
        }

        let angle = Self.angleDegrees(between: lastPose.rotation, and: pose.rotation)
        if angle >= CaptureTuning.keyframeMinRotationDegrees {
            return .rotation
        }

        return nil
    }

    /// Records that a frame was actually written, so the next decision has a
    /// reference. Separate from `decide` on purpose: a decision the writer
    /// then refuses (out of disk, say) must not move the reference.
    func didAcceptKeyframe(pose: Pose, timestamp: TimeInterval) {
        lastKeyframePose = pose
        lastKeyframeTimestamp = timestamp
    }

    /// Shortest-arc angle between two rotations, degrees.
    static func angleDegrees(between a: Quaternion, and b: Quaternion) -> Float {
        let relative = simd_normalize(b.simd * a.simd.inverse)
        // |w| of the relative rotation is cos(theta/2). Clamped because a
        // normalisation can leave it a hair over 1 and acos would return NaN.
        let w = Swift.min(1, abs(relative.real))
        return 2 * acos(w) * 180 / .pi
    }
}
