//
//  PreviewCameraPathBuilder.swift
//  Viewer
//
//  VIRTUAL CAMERAS ON THE WALKED PATH (F9).
//
//  The rule from the spec, restated so it cannot be softened by accident:
//
//      preview with virtual cameras ON the walked path (within ~20 cm), FOV
//      widened to 100-110 degrees
//
//  Why it matters more than it sounds: a preview camera that leaves the
//  observed set renders the parts of the model that were never supervised.
//  Those are exactly the parts that look worst, and the user cannot do
//  anything about them - they did not walk there, and the fix hint would be
//  "go back and stand somewhere you never stood". So the fly-through is
//  smoothed, not freed: a Catmull-Rom spline through the real camera track,
//  then every sample pulled back to the nearest point on the real polyline if
//  the spline let it drift past `maxDeviationMeters`.
//
//  The widening to 100-110 degrees is the opposite trade and is deliberate.
//  The capture camera is around 68 degrees horizontally; replaying at 68
//  degrees feels like looking through a keyhole and hides context the user
//  needs to judge the scan. Widening changes what is IN frame, not where the
//  camera is, so it stays inside the observed set as long as the extra field
//  of view is drawn honestly - which is what the honesty mask is for.
//

import Foundation
import simd

// MARK: - Building

/// Builds a `PreviewCameraPath` from a capture.
enum PreviewCameraPathBuilder {

    struct Options: Sendable {
        /// Target field of view for the fly-through. The spec's band is
        /// 100-110 degrees; the default sits in the middle of it.
        var horizontalFOVDegrees: Float = 105
        /// Hard limit on how far a preview camera may stray from the walked
        /// path, metres.
        var maxDeviationMeters: Float = 0.20
        /// Minimum distance between two keyframes, metres. A user standing
        /// still for twenty seconds should not produce twenty seconds of
        /// identical fly-through.
        var minKeyframeSpacingMeters: Float = 0.12
        /// Minimum turn between two keyframes, degrees. Catches the case where
        /// the user stood in one spot and panned, which IS motion worth
        /// replaying even though the position did not change.
        var minKeyframeTurnDegrees: Float = 8
        /// Skip frames the capture QC scored below this. A smeared frame is a
        /// bad place to put a review camera.
        var minFrameQCWeight: Float = 0.2
        /// Keyframes are dropped to at most this many; a fifteen-minute house
        /// scan does not need six thousand of them.
        var maxKeyframes: Int = 600
        /// Replay speed. Below 1 the fly-through is slower than the walk,
        /// which is almost always what a reviewer wants.
        var timeScale: Double = 0.75
        /// Quarter turns clockwise the recorded poses need before they are the
        /// right way up on screen: the capture's own
        /// `CaptureSettings.imageQuarterTurnsClockwiseToUpright` where a scan
        /// has one, and `ViewerPoseMath.uprightQuarterTurns(of:)` measured
        /// back out of the poses where it does not.
        ///
        /// ARKit records every pose in the sensor's landscape frame however
        /// the phone was held, so a portrait scan has a quarter turn baked
        /// into it that would otherwise be replayed as a room lying on its
        /// side. Applying it here, once, means the sampler, the deviation
        /// measurement and every consumer of the path all see one consistent
        /// set of poses. The default of 0 replays exactly what was recorded.
        var uprightQuarterTurns: Int = 0

        init() {}
    }

    /// The result, plus the honest reason when there is nothing to build.
    ///
    /// Named `Outcome` rather than `Result` so it cannot shadow `Swift.Result`
    /// for anyone who later adds a throwing call to this file.
    enum Outcome {
        case path(PreviewCameraPath)
        /// No usable path, with a plain sentence saying why.
        case unavailable(String)
    }

    /// Builds the fly-through from the capture's poses.
    ///
    /// Refined poses are used when the pre-pass has produced them, because the
    /// whole point of F1 is that the raw VIO track drifts; replaying the drift
    /// would put the preview camera up to a couple of centimetres off the
    /// track the model was actually trained against.
    static func build(
        bundle: CaptureBundle,
        prePass: PrePassResult?,
        options: Options = Options()
    ) -> Outcome {
        let frames = bundle.frames
        guard !frames.isEmpty else {
            return .unavailable("This scan has no frames recorded, so there is no walk to replay.")
        }

        // 1. Collect usable camera positions along the walk.
        struct Sample {
            var frame: FrameID
            var pose: Pose
            var centre: SIMD3<Float>
            var forward: SIMD3<Float>
            var time: Double
        }

        var samples: [Sample] = []
        samples.reserveCapacity(frames.count)
        for frame in frames {
            guard frame.qc.weight >= options.minFrameQCWeight else { continue }
            let pose = prePass?.refinedPose(for: frame.index)
                ?? frame.refinedPose
                ?? frame.rawPose
            let centre = pose.center.simd
            guard centre.x.isFinite, centre.y.isFinite, centre.z.isFinite else { continue }
            samples.append(
                Sample(
                    frame: frame.index,
                    pose: pose,
                    centre: centre,
                    forward: pose.forward.simd,
                    time: frame.timestampSeconds
                )
            )
        }

        if samples.count < 2 {
            // Falling back to every frame regardless of quality beats refusing
            // to show a preview at all; the QC card already told the user the
            // capture was rough.
            samples = frames.compactMap { frame in
                let pose = prePass?.refinedPose(for: frame.index) ?? frame.rawPose
                let centre = pose.center.simd
                guard centre.x.isFinite, centre.y.isFinite, centre.z.isFinite else { return nil }
                return Sample(
                    frame: frame.index,
                    pose: pose,
                    centre: centre,
                    forward: pose.forward.simd,
                    time: frame.timestampSeconds
                )
            }
        }

        guard samples.count >= 2 else {
            return .unavailable(
                "This scan does not have enough usable camera positions to replay the walk."
            )
        }

        // 2. Thin to keyframes: drop frames that neither moved nor turned.
        var keptSamples: [Sample] = [samples[0]]
        for sample in samples.dropFirst() {
            guard let last = keptSamples.last else { break }
            let moved = simd_distance(sample.centre, last.centre)
            let turned = ViewerMath.angleDegrees(sample.forward, last.forward)
            if moved >= options.minKeyframeSpacingMeters
                || turned >= options.minKeyframeTurnDegrees {
                keptSamples.append(sample)
            }
        }
        if keptSamples.count < 2 { keptSamples = samples }

        // 3. Cap the count by uniform decimation in time.
        if keptSamples.count > options.maxKeyframes {
            // `Double(0)` in the denominator would make this infinite, and
            // `Int(infinity)` is a trapping conversion.
            let stride = Int(
                ceil(Double(keptSamples.count) / Double(Swift.max(options.maxKeyframes, 1)))
            )
            var decimated: [Sample] = []
            var index = 0
            while index < keptSamples.count {
                decimated.append(keptSamples[index])
                index += stride
            }
            if let last = keptSamples.last, decimated.last?.frame != last.frame {
                decimated.append(last)
            }
            keptSamples = decimated
        }

        // 4. Re-time from zero, scaled. Gaps longer than a second (the user
        //    stopped, or tracking dropped) are compressed to a second so the
        //    fly-through does not sit still staring at a wall.
        let startTime = keptSamples[0].time
        var elapsed: Double = 0
        var keyframes: [PreviewCameraPath.Keyframe] = []
        keyframes.reserveCapacity(keptSamples.count)
        var previousTime = startTime
        for sample in keptSamples {
            let gap = Swift.min(Swift.max(sample.time - previousTime, 0), 1.0)
            elapsed += gap * options.timeScale
            previousTime = sample.time
            keyframes.append(
                PreviewCameraPath.Keyframe(
                    // A roll about the camera's own optical axis: it turns the
                    // picture the right way up without moving the camera off
                    // the walked track by so much as a millimetre.
                    pose: sample.pose.rolledForDisplay(
                        quarterTurnsClockwise: options.uprightQuarterTurns
                    ),
                    timeSeconds: elapsed,
                    sourceFrame: sample.frame
                )
            )
        }

        // Degenerate case: every frame had the same timestamp. Space them out
        // evenly rather than producing a zero-length path.
        if elapsed <= 0.001 {
            for index in keyframes.indices {
                keyframes[index].timeSeconds = Double(index) * 0.1
            }
        }

        return .path(
            PreviewCameraPath(
                keyframes: keyframes,
                horizontalFOVDegrees: ViewerMath.clamp(options.horizontalFOVDegrees, 100, 110),
                maxDeviationMeters: options.maxDeviationMeters,
                honestyMaskEnabled: true
            )
        )
    }
}

// MARK: - Sampling

/// Evaluates a `PreviewCameraPath` at a time, with the deviation clamp applied.
///
/// Kept separate from the builder because it runs sixty times a second and the
/// builder runs once.
enum PreviewPathSampler {

    struct Sample {
        var pose: Pose
        var sourceFrame: FrameID?
        /// How far the smoothed camera ended up from the walked polyline,
        /// AFTER clamping. Reported so the review screen can show it honestly
        /// rather than merely promising it is small.
        var deviationMeters: Float
    }

    /// Samples the path at `time` seconds.
    static func sample(_ path: PreviewCameraPath, atTime time: Double) -> Sample {
        let keyframes = path.keyframes
        guard let first = keyframes.first else {
            return Sample(pose: .identity, sourceFrame: nil, deviationMeters: 0)
        }
        guard keyframes.count > 1 else {
            return Sample(
                pose: first.pose,
                sourceFrame: first.sourceFrame,
                deviationMeters: 0
            )
        }

        // Locate the segment. Keyframe times are monotonically non-decreasing
        // by construction, so a binary search is exact.
        let clamped = ViewerMath.clamp(
            time,
            first.timeSeconds,
            keyframes[keyframes.count - 1].timeSeconds
        )
        var lo = 0
        var hi = keyframes.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if keyframes[mid].timeSeconds <= clamped {
                lo = mid
            } else {
                hi = mid
            }
        }

        let span = keyframes[hi].timeSeconds - keyframes[lo].timeSeconds
        let t = span > 1e-6 ? Float((clamped - keyframes[lo].timeSeconds) / span) : 0

        // Catmull-Rom over the four surrounding camera centres. The spline is
        // what removes the hand shake; the clamp below is what stops the
        // spline from cutting a corner the user never cut.
        let p0 = keyframes[Swift.max(lo - 1, 0)].pose.center.simd
        let p1 = keyframes[lo].pose.center.simd
        let p2 = keyframes[hi].pose.center.simd
        let p3 = keyframes[Swift.min(hi + 1, keyframes.count - 1)].pose.center.simd
        var centre = catmullRom(p0, p1, p2, p3, t)

        // Pull back to within the contract's deviation limit. The nearest
        // point on the p1 -> p2 segment IS a point the user occupied, so this
        // can only ever move the camera towards the observed set.
        let limit = Swift.max(path.maxDeviationMeters, 0.01)
        let nearest = closestPointOnSegment(centre, a: p1, b: p2)
        var deviation = simd_distance(centre, nearest)
        if deviation > limit {
            let direction = deviation > 1e-6 ? (centre - nearest) / deviation : SIMD3<Float>(0, 0, 0)
            centre = nearest + direction * limit
            deviation = limit
        }

        // Orientation: slerp between the two keyframes. Rotations are already
        // smooth (the user's head does not jitter the way their hand does) and
        // splining them buys nothing but overshoot at a turn.
        let rotation = simd_slerp(
            keyframes[lo].pose.rotation.simd,
            keyframes[hi].pose.rotation.simd,
            t
        ).normalized

        // The pose is rebuilt from the CLAMPED centre, so the translation is
        // recomputed rather than interpolated: t = -R * C.
        let translation = -(rotation.act(centre))
        let pose = Pose(rotation: Quaternion(rotation), translation: Vector3(translation))

        return Sample(
            pose: pose,
            sourceFrame: t < 0.5 ? keyframes[lo].sourceFrame : keyframes[hi].sourceFrame,
            deviationMeters: deviation
        )
    }

    /// Uniform Catmull-Rom. Uniform rather than centripetal on purpose: the
    /// keyframes are already spaced by distance travelled, which is the same
    /// reparameterisation centripetal Catmull-Rom exists to approximate.
    static func catmullRom(
        _ p0: SIMD3<Float>,
        _ p1: SIMD3<Float>,
        _ p2: SIMD3<Float>,
        _ p3: SIMD3<Float>,
        _ t: Float
    ) -> SIMD3<Float> {
        // Each Catmull-Rom basis term is its own typed local. Written as one
        // expression, the integer literals mixed with SIMD3<Float> operators
        // gave the type checker too many overload combinations to resolve in
        // reasonable time and it refused to compile it. Identical arithmetic.
        let t2: Float = t * t
        let t3: Float = t2 * t

        // The basis is collected into ONE plain Float coefficient per control
        // point, so every vector operation below is a single
        // SIMD3<Float> * Float with exactly one overload. Splitting the
        // original one-liner into four vector terms was not enough: each term
        // still mixed integer literals with SIMD operators, and the type
        // checker still ran out of time. Scalars first, vectors last.
        let c0: Float = -t + 2.0 * t2 - t3
        let c1: Float = 2.0 - 5.0 * t2 + 3.0 * t3
        let c2: Float = t + 4.0 * t2 - 3.0 * t3
        let c3: Float = -t2 + t3

        var sum: SIMD3<Float> = p0 * c0
        sum += p1 * c1
        sum += p2 * c2
        sum += p3 * c3
        return sum * 0.5
    }

    static func closestPointOnSegment(
        _ point: SIMD3<Float>,
        a: SIMD3<Float>,
        b: SIMD3<Float>
    ) -> SIMD3<Float> {
        let ab = b - a
        let lengthSquared = simd_length_squared(ab)
        guard lengthSquared > 1e-12 else { return a }
        let t = ViewerMath.clamp(simd_dot(point - a, ab) / lengthSquared, 0, 1)
        return a + ab * t
    }

    /// The worst deviation anywhere on the path, sampled at `steps` points.
    /// The review screen shows this so the ~20 cm promise is a measurement,
    /// not a claim.
    static func worstDeviation(_ path: PreviewCameraPath, steps: Int = 400) -> Float {
        guard let first = path.keyframes.first,
              let last = path.keyframes.last,
              steps > 1
        else { return 0 }
        let duration = last.timeSeconds - first.timeSeconds
        guard duration > 0 else { return 0 }
        var worst: Float = 0
        for index in 0..<steps {
            let time = first.timeSeconds + duration * Double(index) / Double(steps - 1)
            worst = Swift.max(worst, sample(path, atTime: time).deviationMeters)
        }
        return worst
    }
}
