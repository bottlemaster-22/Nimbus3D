//
//  TrainerSlices.swift
//  Trainer
//
//  TIME-SLICED SLICE-TRAIN-MERGE (F1, F7). SUBMAPS ARE ALWAYS ON.
//
//  A 20 minute house walk is not one optimisation problem. VIO is very nearly
//  perfect inside a 15-30 second window and drifts over minutes, so the
//  pre-pass already cut the capture into overlapping submaps and optimised one
//  rigid placement per submap. The trainer inherits that partition: it trains
//  one slice at a time, at full quality, against that slice's own keyframes,
//  and merges the results.
//
//  ---------------------------------------------------------------------------
//  THE MERGE RULE, AND WHY IT IS NOT A CONCATENATION
//  ---------------------------------------------------------------------------
//  Two slices overlap by 20-30% on purpose: the overlap is what makes the pose
//  graph work. That means the same wall gets Gaussians from two slices, and
//  those Gaussians are TRANSLUCENT. Alpha compositing ADDS coverage: two
//  half-opaque surfaces in the same place read as three-quarters opaque, not
//  as one half-opaque surface. Concatenating overlapping slices therefore
//  produces a model that is visibly heavier, darker and foggier than either
//  slice alone, and no amount of later opacity tuning fixes it, because the
//  duplication is structural.
//
//  So: ONE OWNER PER REGION. Space is partitioned before the merge, every
//  Gaussian is tested against the owner of the region its centre falls in, and
//  a Gaussian whose owner is not the slice that made it is DELETED rather than
//  blended. The partition is a nearest-centre assignment over the submap
//  bounds, which makes it exhaustive (every point in space has exactly one
//  owner) and deterministic (ties break on the lower submap index), so the
//  same capture always merges the same way.
//
//  Optional whole-house co-visibility partitioning - grouping by what sees
//  what rather than by when it was walked - is a different partition of the
//  same problem and is OFF by default, as the spec asks.
//

import Foundation
import simd

/// One unit of training work: a time slice, its keyframes, and the region of
/// space it owns in the merge.
struct TrainerSlice {
    var index: Int
    /// The submap this came from. Nil for the single-slice case, which is what
    /// a short capture with no submaps gets.
    var submap: Submap?
    /// Frames trained on. A frame in the overlap belongs to BOTH neighbouring
    /// slices and is trained by both; only the merge is exclusive.
    var keyframes: [CaptureFrame]
    /// Frames held out of training entirely and used only to report PSNR.
    var heldOutKeyframes: [CaptureFrame]
    /// The centre of this slice's region, for the nearest-centre ownership test.
    var regionCenter: SIMD3<Float>
    var bounds: BoundingBox
    var iterationBudget: Int
    var splatCapShare: Int

    /// The user-facing name of this slice, for the progress message. One-based
    /// because "part 0 of 5" is not a thing anyone says.
    func label(of total: Int) -> String {
        total <= 1 ? "" : "part \(index + 1) of \(total)"
    }
}

enum TrainerSlicePlanner {

    /// Whole-house co-visibility partitioning (group by what sees what, rather
    /// than by when it was walked). OFF, as the spec requires. The time-slice
    /// partition below is what actually runs.
    ///
    /// READ THIS BEFORE FLIPPING IT TO `true`. There is no co-visibility path
    /// behind this flag: `plan()` never consults it, and nothing else does
    /// either. Setting it to `true` changes NOTHING except what this file
    /// appears to promise. It stays here as the written record that the
    /// partitioning strategy was chosen deliberately, not forgotten, and
    /// anyone implementing the other strategy has to wire this into `plan()`
    /// themselves.
    static let coVisibilityPartitioningEnabled = false

    /// Splits the run into slices. A capture with no submaps, or one whose
    /// submaps do not cover enough frames to be worth separating, comes back
    /// as a single slice, which makes the single-slice case exactly the same
    /// code path rather than a special case that rots.
    static func plan(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        keyframes: [CaptureFrame],
        budget: TrainingBudget,
        heldOutFraction: Float
    ) -> [TrainerSlice] {

        guard !keyframes.isEmpty else { return [] }

        let submaps = prePass.submaps
        // One slice when there is nothing to slice by, or when slicing would
        // leave a slice with too few views to train anything meaningful.
        let minimumKeyframesPerSlice = 12
        let useSubmaps = submaps.count > 1
            && keyframes.count >= submaps.count * minimumKeyframesPerSlice

        if !useSubmaps {
            let bounds = boundingBox(of: keyframes, prePass: prePass, fallback: bundle.sceneBounds)
            let split = splitHeldOut(keyframes, fraction: heldOutFraction)
            return [
                TrainerSlice(
                    index: 0,
                    submap: submaps.count == 1 ? submaps[0] : nil,
                    keyframes: split.train,
                    heldOutKeyframes: split.heldOut,
                    regionCenter: center(of: bounds),
                    bounds: bounds,
                    iterationBudget: budget.iterations,
                    splatCapShare: budget.splatCap
                )
            ]
        }

        // Assign keyframes to submaps by frame range. A frame in an overlap
        // lands in more than one slice deliberately: the overlap is what makes
        // the two slices agree about the wall between them.
        var byIndex: [Int: [CaptureFrame]] = [:]
        for frame in keyframes {
            var landed = false
            for (i, submap) in submaps.enumerated()
            where frame.index >= submap.firstFrame && frame.index <= submap.lastFrame {
                byIndex[i, default: []].append(frame)
                landed = true
            }
            if !landed {
                // The frame's own submap tag, if the capture recorded one;
                // otherwise the nearest submap in time. Never dropped.
                if let tagged = frame.submap,
                   let position = submaps.firstIndex(where: { $0.index == tagged })
                {
                    byIndex[position, default: []].append(frame)
                } else {
                    var best = 0
                    var bestGap = Double.greatestFiniteMagnitude
                    for (i, submap) in submaps.enumerated() {
                        let midpoint = (submap.startTimeSeconds + submap.endTimeSeconds) * 0.5
                        let gap = abs(frame.timestampSeconds - midpoint)
                        if gap < bestGap { bestGap = gap; best = i }
                    }
                    byIndex[best, default: []].append(frame)
                }
            }
        }

        let populated = byIndex.filter { !$0.value.isEmpty }
        let totalAssigned = populated.values.reduce(0) { $0 + $1.count }
        guard totalAssigned > 0 else { return [] }

        var slices: [TrainerSlice] = []
        var nextIndex = 0
        for i in populated.keys.sorted() {
            let frames = populated[i] ?? []
            guard !frames.isEmpty else { continue }
            let submap = submaps[i]
            let share = Float(frames.count) / Float(totalAssigned)

            // Iterations and cap are shared out by how much of the capture the
            // slice actually covers. A slice that saw a corridor for six
            // seconds does not get the same budget as one that saw a room for
            // half a minute.
            let iterations = Swift.max(Int(Float(budget.iterations) * share), 200)
            let capShare = Swift.max(Int(Float(budget.splatCap) * share), 20_000)

            let bounds = submap.bounds.sizeMeters.x > 0 || submap.bounds.sizeMeters.y > 0
                ? submap.bounds
                : boundingBox(of: frames, prePass: prePass, fallback: bundle.sceneBounds)

            let split = splitHeldOut(frames, fraction: heldOutFraction)
            slices.append(
                TrainerSlice(
                    index: nextIndex,
                    submap: submap,
                    keyframes: split.train,
                    heldOutKeyframes: split.heldOut,
                    regionCenter: center(of: bounds),
                    bounds: bounds,
                    iterationBudget: iterations,
                    splatCapShare: capShare
                )
            )
            nextIndex += 1
        }

        // Unreachable in practice (an empty `populated` returns above), but a
        // planner that can hand the loop zero slices would silently train
        // nothing, so it falls back to the single-slice plan rather than
        // returning an empty array.
        if slices.isEmpty {
            let bounds = boundingBox(of: keyframes, prePass: prePass, fallback: bundle.sceneBounds)
            let split = splitHeldOut(keyframes, fraction: heldOutFraction)
            return [
                TrainerSlice(
                    index: 0,
                    submap: nil,
                    keyframes: split.train,
                    heldOutKeyframes: split.heldOut,
                    regionCenter: center(of: bounds),
                    bounds: bounds,
                    iterationBudget: budget.iterations,
                    splatCapShare: budget.splatCap
                )
            ]
        }
        return slices
    }

    /// Every Nth keyframe, held out of training and used only for the PSNR
    /// number. Evenly spaced rather than a tail, so the held-out set covers
    /// the whole walk instead of only its last few seconds.
    private static func splitHeldOut(
        _ frames: [CaptureFrame],
        fraction: Float
    ) -> (train: [CaptureFrame], heldOut: [CaptureFrame]) {
        guard fraction > 0, frames.count >= 20 else { return (frames, []) }
        let step = Swift.max(Int(1 / fraction), 2)
        var train: [CaptureFrame] = []
        var heldOut: [CaptureFrame] = []
        for (i, frame) in frames.enumerated() {
            if i % step == step / 2 { heldOut.append(frame) } else { train.append(frame) }
        }
        // Never hold out so much that training suffers for a diagnostic.
        if train.count < frames.count / 2 { return (frames, []) }
        return (train, heldOut)
    }

    private static func boundingBox(
        of frames: [CaptureFrame],
        prePass: PrePassResult,
        fallback: BoundingBox?
    ) -> BoundingBox {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var any = false
        for frame in frames {
            let pose = prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
            let c = pose.center.simd
            lo = simd_min(lo, c)
            hi = simd_max(hi, c)
            any = true
        }
        guard any else {
            return fallback ?? BoundingBox(min: .zero, max: .zero)
        }
        // The cameras walked a path; the surfaces they looked at are further
        // out than the path is. Five metres of padding is the LiDAR's own
        // working range, so this is the sensor's reach rather than a guess.
        let pad = SIMD3<Float>(repeating: 5)
        return BoundingBox(min: Vector3(lo - pad), max: Vector3(hi + pad))
    }

    private static func center(of box: BoundingBox) -> SIMD3<Float> {
        (box.min.simd + box.max.simd) * 0.5
    }
}

// MARK: - Merge

enum TrainerSliceMerger {

    /// Which slice owns a point. Exhaustive and deterministic:
    ///
    ///  * among the slices whose bounds CONTAIN the point, the one whose
    ///    region centre is nearest;
    ///  * if no slice's bounds contain it (it drifted outside every submap),
    ///    the slice whose centre is nearest, so nothing is ever ownerless;
    ///  * ties break on the lower slice index.
    static func owner(of point: SIMD3<Float>, slices: [TrainerSlice]) -> Int {
        guard slices.count > 1 else { return 0 }

        var best = -1
        var bestDistance = Float.greatestFiniteMagnitude
        for slice in slices where contains(slice.bounds, point) {
            let d = simd_distance_squared(point, slice.regionCenter)
            if d < bestDistance { bestDistance = d; best = slice.index }
        }
        if best >= 0 { return best }

        best = slices[0].index
        bestDistance = simd_distance_squared(point, slices[0].regionCenter)
        for slice in slices.dropFirst() {
            let d = simd_distance_squared(point, slice.regionCenter)
            if d < bestDistance { bestDistance = d; best = slice.index }
        }
        return best
    }

    private static func contains(_ box: BoundingBox, _ point: SIMD3<Float>) -> Bool {
        let lo = box.min.simd, hi = box.max.simd
        return point.x >= lo.x && point.x <= hi.x
            && point.y >= lo.y && point.y <= hi.y
            && point.z >= lo.z && point.z <= hi.z
    }

    /// What a merge threw away, so the log can be specific rather than
    /// reporting a count that quietly shrank.
    struct MergeReport {
        var kept = 0
        var droppedToOtherOwners = 0
        var trimmedToCap = 0

        var summary: String {
            "Joined the parts into \(kept) points, dropping \(droppedToOtherOwners) "
                + "that another part already covered."
        }
    }

    /// Concatenates the per-slice clouds under the one-owner rule, then
    /// enforces the global cap. Ordering is by slice, then by the order each
    /// slice produced, so the merge is reproducible.
    static func merge(
        parts: [(slice: TrainerSlice, cloud: SplatCloud)],
        slices: [TrainerSlice],
        splatCap: Int
    ) throws -> (cloud: SplatCloud, report: MergeReport) {

        var report = MergeReport()
        guard !parts.isEmpty else {
            return (SplatCloud.empty(shDegree: .zero), report)
        }
        if parts.count == 1 && slices.count <= 1 {
            report.kept = parts[0].cloud.count
            return (parts[0].cloud, report)
        }

        let degree = parts.first?.cloud.shDegree ?? .zero
        var positions: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var logScales: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var colorDC: [SIMD3<Float>] = []
        var shRest: [[SIMD3<Float>]] = []
        var importance: [Float] = []

        for part in parts {
            let cloud = part.cloud
            // A slice trained at a different SH degree cannot be concatenated
            // with the others without silently reinterpreting its
            // coefficients. Every slice in one run shares a degree, so this is
            // a guard against a future change rather than a case that happens.
            guard cloud.shDegree == degree else {
                throw NimbusError.trainingFailed(
                    "two parts of this scan were built with different colour detail and "
                        + "cannot be joined"
                )
            }
            for i in 0..<cloud.count {
                let position = cloud.positions[i]
                guard owner(of: position, slices: slices) == part.slice.index else {
                    report.droppedToOtherOwners += 1
                    continue
                }
                positions.append(position)
                rotations.append(cloud.rotations[i])
                logScales.append(cloud.logScales[i])
                opacities.append(cloud.opacityLogits[i])
                colorDC.append(cloud.colorDC[i])
                if degree != .zero {
                    shRest.append(i < cloud.shRest.count
                        ? cloud.shRest[i]
                        : [SIMD3<Float>](repeating: .zero, count: degree.restCoefficientCount))
                }
                // Bigger and more opaque contributes more; that is the ranking
                // used if the merged set still overflows the cap.
                let scale = cloud.logScales[i]
                let volume = expf(scale.x) * expf(scale.y) * expf(scale.z)
                importance.append(
                    TrainerMath.sigmoid(cloud.opacityLogits[i]) * cbrtf(Swift.max(volume, 1e-12))
                )
            }
        }

        report.kept = positions.count

        if splatCap > 0, positions.count > splatCap {
            var order = Array(0..<positions.count)
            order.sort { importance[$0] > importance[$1] }
            let keep = Set(order.prefix(splatCap))
            report.trimmedToCap = positions.count - splatCap

            var keptPositions: [SIMD3<Float>] = []
            var keptRotations: [SIMD4<Float>] = []
            var keptScales: [SIMD3<Float>] = []
            var keptOpacities: [Float] = []
            var keptDC: [SIMD3<Float>] = []
            var keptRest: [[SIMD3<Float>]] = []
            keptPositions.reserveCapacity(splatCap)
            for i in 0..<positions.count where keep.contains(i) {
                keptPositions.append(positions[i])
                keptRotations.append(rotations[i])
                keptScales.append(logScales[i])
                keptOpacities.append(opacities[i])
                keptDC.append(colorDC[i])
                if degree != .zero, i < shRest.count { keptRest.append(shRest[i]) }
            }
            positions = keptPositions
            rotations = keptRotations
            logScales = keptScales
            opacities = keptOpacities
            colorDC = keptDC
            shRest = keptRest
            report.kept = positions.count
        }

        var cloud = try SplatCloud(
            shDegree: degree,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacities,
            colorDC: colorDC,
            shRest: degree == .zero ? [] : shRest
        )
        // A fresh cloud starts at `nil`. Every part above was read off the GPU
        // by `MetalSplatTrainer.readCloud`, which folds the trainer's 3D
        // low-pass filter in and says so, and that fact must survive the join
        // or the finished model reports "cannot say" about its own geometry.
        // One unfused part makes the whole model unfaithful, so the rule is
        // the cautious one in `SplatCloud.mergedFilter3DFused`.
        cloud.filter3DFused = SplatCloud.mergedFilter3DFused(
            parts.map { $0.cloud.filter3DFused }
        )
        return (cloud, report)
    }
}
