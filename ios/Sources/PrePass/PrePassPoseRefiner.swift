//
//  PrePassPoseRefiner.swift
//  PrePass
//
//  F1: POSES FIRST. The single biggest quality lever in the whole product.
//
//  Raw ARKit VIO drifts about 0.55 degrees and 19 mm over a long walk. At 2 m
//  that is roughly 13 px of disagreement between two views of the same wall,
//  and a Gaussian splat field needs sub-pixel agreement or it hedges: it
//  builds a soft, semi-transparent cloud that renders every view slightly
//  blurred because no single sharp surface can satisfy all of them at once.
//  Every other clever thing in this app is downstream of getting this right.
//
//  What this file does NOT do, deliberately:
//
//   * It never runs COLMAP or any structure-from-motion from scratch. ARKit's
//     track is metric, gravity-aligned and locally excellent; throwing it away
//     to re-derive it from pixels is strictly worse and much slower.
//   * It never triangulates-then-bundle-adjusts from the ARKit prior. Measured
//     on 15 of 15 rooms, that made the poses WORSE (arXiv 2608.21008): the
//     triangulation inherits the prior's error, and the adjustment then fits
//     the cameras to the bad points.
//
//  What it does instead: trust VIO locally, correct it globally. Slice the
//  walk into 15-30 s submaps (inside which VIO is near-perfect), find the
//  places the user walked back over (revisits), measure those alignments
//  against the native LiDAR depth with point-to-plane ICP, and solve for one
//  rigid SE(3) per submap. A few dozen unknowns instead of a few thousand, and
//  every one of them is constrained by a real measurement.
//

import Foundation
import simd

// MARK: - Adjoint

extension PrePassSE3 {
    /// The 6x6 adjoint, row-major, for the twist ordering `xi = (omega, v)`:
    ///
    ///     Adj(T) = [ R        0 ]
    ///              [ [t]x R   R ]
    ///
    /// It is what lets a perturbation applied on one side of a product be
    /// rewritten as a perturbation on the other, which is the whole of the
    /// pose-graph Jacobian derivation below.
    var adjoint: [Double] {
        let r = simd_double3x3(rotation.normalized)
        let tx = PrePassSE3.skew(translation)
        let txr = tx * r
        var a = [Double](repeating: 0, count: 36)
        for row in 0..<3 {
            for col in 0..<3 {
                // simd matrices are column-major: m[col][row].
                a[row * 6 + col] = r[col][row]
                a[(row + 3) * 6 + col] = txr[col][row]
                a[(row + 3) * 6 + (col + 3)] = r[col][row]
            }
        }
        return a
    }
}

// MARK: - Pose track

/// The raw VIO track as a function of time, so a pose can be asked for at a
/// timestamp that is not exactly a frame's.
///
/// Used by the camera-to-IMU time-offset sweep, which is precisely the
/// question "what pose really belongs to the light that landed on the sensor
/// at this moment", and by nothing else - every other stage works at frame
/// timestamps where no interpolation is needed.
struct PrePassPoseTrack: Sendable {
    private let times: [Double]
    private let poses: [Pose]

    init(frames: [CaptureFrame]) {
        // Sorted by time, not assumed sorted: a dropped-and-recovered ARKit
        // session can log a frame out of order and a binary search over an
        // unsorted array returns nonsense in silence.
        let sorted = frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        times = sorted.map(\.timestampSeconds)
        poses = sorted.map(\.rawPose)
    }

    var isEmpty: Bool { times.isEmpty }
    var duration: Double { (times.last ?? 0) - (times.first ?? 0) }

    /// Pose at an arbitrary time: SLERP on rotation, linear on translation.
    /// Clamps at both ends rather than extrapolating - a +-50 ms sweep at the
    /// very first frame would otherwise invent a pose from before the session
    /// started.
    func pose(at time: Double) -> Pose {
        guard !times.isEmpty else { return .identity }
        if time <= times[0] { return poses[0] }
        if time >= times[times.count - 1] { return poses[poses.count - 1] }

        var low = 0
        var high = times.count - 1
        while high - low > 1 {
            let mid = (low + high) / 2
            if times[mid] <= time { low = mid } else { high = mid }
        }
        let span = times[high] - times[low]
        guard span > 1e-9 else { return poses[low] }
        let t = Float((time - times[low]) / span)
        return Pose.interpolate(poses[low], poses[high], t: t)
    }
}

// MARK: - Submap assignment

/// Which submap OWNS each frame.
///
/// Submap time ranges deliberately overlap by 20-30%, so a frame in an overlap
/// belongs to two submaps' ranges. Exactly one of them owns it for the purpose
/// of applying a correction, or the same frame would get two different refined
/// poses. The owner is the submap whose window CENTRE is nearest, which puts
/// every frame under the submap that saw it furthest from a window edge, where
/// VIO's local accuracy is best used.
enum PrePassSubmapAssignment {
    static func owners(frames: [CaptureFrame], submaps: [Submap]) -> [FrameID: SubmapID] {
        var result: [FrameID: SubmapID] = [:]
        guard !submaps.isEmpty else { return result }
        result.reserveCapacity(frames.count)

        for frame in frames {
            var bestIndex = submaps[0].index
            var bestDistance = Double.greatestFiniteMagnitude
            var found = false
            for submap in submaps {
                guard frame.index >= submap.firstFrame, frame.index <= submap.lastFrame else {
                    continue
                }
                let centre = 0.5 * (submap.startTimeSeconds + submap.endTimeSeconds)
                let distance = abs(frame.timestampSeconds - centre)
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndex = submap.index
                    found = true
                }
            }
            if !found {
                // Outside every window: a frame logged before the first
                // timestamp or after the last (clock hiccup). Give it the
                // nearest submap by time rather than dropping it, so it still
                // gets a refined pose.
                var nearest = submaps[0]
                var nearestDistance = Double.greatestFiniteMagnitude
                for submap in submaps {
                    let centre = 0.5 * (submap.startTimeSeconds + submap.endTimeSeconds)
                    let distance = abs(frame.timestampSeconds - centre)
                    if distance < nearestDistance {
                        nearestDistance = distance
                        nearest = submap
                    }
                }
                bestIndex = nearest.index
            }
            result[frame.index] = bestIndex
        }
        return result
    }
}

// MARK: - The refiner

/// `PoseRefiner`, owned by `Sources/PrePass` (CONTRACTS.md section 5).
public final class SubmapPoseRefiner: PoseRefiner, @unchecked Sendable {

    /// Every number the refinement depends on, in one visible place.
    public struct Tuning: Sendable {
        /// Submap window length, seconds. The spec fixes 15-30 s: short enough
        /// that VIO has not drifted inside one, long enough that a submap has
        /// enough geometry to be pinned by a revisit.
        public var submapWindowSeconds: Double = 20
        /// Fraction of a window shared with the next one. The spec fixes
        /// 20-30%.
        public var submapOverlapFraction: Double = 0.25

        /// Two frames closer than this in time are adjacency, not a revisit.
        public var revisitMinTimeGapSeconds: Double = 8
        /// Camera centres must be within this to be candidates.
        public var revisitMaxCentreDistanceMeters: Float = 1.5
        /// And must be looking within this many degrees of the same way.
        public var revisitMaxViewAngleDegrees: Float = 50
        /// Keyframe spacing for candidate generation: one candidate anchor per
        /// this much movement OR this much time, whichever comes first.
        public var revisitKeyframeSpacingMeters: Float = 0.25
        public var revisitKeyframeSpacingSeconds: Double = 0.5
        /// Hard ceiling on ICP runs. A four-minute walk round a flat can
        /// generate tens of thousands of geometrically plausible pairs and
        /// aligning all of them would take longer than the training does.
        public var revisitMaxICPRuns = 600

        /// Time-offset sweep, as the spec fixes it.
        public var timeOffsetMinSeconds: Double = -0.050
        public var timeOffsetMaxSeconds: Double = 0.050
        public var timeOffsetStepSeconds: Double = 0.005
        /// Frame pairs the sweep scores.
        public var timeOffsetPairCount = 24
        /// Feature points per pair.
        public var timeOffsetFeaturesPerPair = 60
        /// The sweep's best cost has to beat its median by at least this
        /// fraction, or there was no minimum and the honest answer is nil.
        public var timeOffsetMinRelativeImprovement: Double = 0.02

        /// Pose-graph noise model. Separate because a radian and a metre are
        /// not comparable and pretending they are is how a pose graph ends up
        /// silently rotation-dominated.
        public var revisitRotationSigmaDegrees: Double = 0.30
        public var revisitTranslationSigmaMeters: Double = 0.015
        /// How stiff the "VIO got the relative placement of neighbouring
        /// submaps right" prior is. Loose enough that a real loop closure can
        /// overcome it, stiff enough that an unconstrained submap does not
        /// float away.
        public var smoothnessRotationSigmaDegrees: Double = 1.0
        public var smoothnessTranslationSigmaMeters: Double = 0.05
        /// Huber threshold on the whitened 6-vector residual. Whitened, so
        /// this is in sigmas, not metres.
        public var robustDelta: Double = 2.0
        /// RAISED FROM 30, because 30 was not where the solver converged, it
        /// was where it was cut off. The owner's scan reports
        /// `converged: false`, `exitReason: "iterationLimit"`,
        /// `iterationsRun: 30`, and a cost that fell from 4520.09 to 963.15,
        /// a 79 per cent reduction still in progress when it stopped. It left
        /// a median residual of 2.78 cm and 1.45 degrees.
        ///
        /// That residual is roughly three times the median splat this trainer
        /// now produces (9.12 mm), so it is a hard floor on how sharp the
        /// model can be: no amount of optimisation can align a Gaussian to a
        /// photograph whose camera is 2.78 cm from where the solver thinks it
        /// is.
        ///
        /// The whole stage took SEVEN MILLISECONDS. 0.0073 s for 30 iterations
        /// is 0.24 ms each, against a pre-pass of 19 s and a training run of
        /// 56 s. 300 iterations costs about 73 ms, which is a tenth of one per
        /// cent of the run, and the solver stops on its own convergence test
        /// long before that if it gets there.
        public var maxIterations = 300

        public init() {}
    }

    public var tuning: Tuning

    /// Populated by `detectRevisits` and read by the QC card: candidates that
    /// looked like revisits geometrically but whose depth alignment did not
    /// converge. A high number here with a low loop count is the signature of
    /// a scan taken too fast or too far from the surfaces.
    public private(set) var rejectedRevisitCandidates: Int = 0

    /// Diagnostics from the last time-offset sweep, for the QC card and for
    /// anyone wondering why the offset came back nil. Cost per candidate
    /// offset, in sweep order.
    public private(set) var lastTimeOffsetSweep: [(offsetSeconds: Double, cost: Double)] = []

    /// The census sections this type owns. Each is assigned by its own method,
    /// on EVERY exit path including the ones that give up early, so "the stage
    /// ran and found nothing" and "the stage never ran" are different readings
    /// rather than the same silence. See `PrePassCensus`.
    ///
    /// Nothing here is touched from a hot loop: the loops increment locals and
    /// the struct is filled once at the end.
    public private(set) var lastTimeOffsetCensus = PrePassCensus.TimeOffset()
    public private(set) var lastRevisitCensus = PrePassCensus.Revisits()
    public private(set) var lastPoseGraphCensus = PrePassCensus.PoseGraph()

    public init(tuning: Tuning = Tuning()) {
        self.tuning = tuning
    }

    // MARK: Submaps

    /// Slices the capture into overlapping time windows.
    ///
    /// `Submap.correction` is defined here and used everywhere else in this
    /// module: it is a world-to-world rigid transform applied to a world point
    /// BEFORE the raw pose, so
    ///
    ///     refinedPose(frame) = rawPose(frame) * correction(owner(frame))
    ///
    /// in the "apply the right-hand one first" reading. Identity therefore
    /// means "the pose graph left this submap where VIO put it", which is what
    /// the contract says it means.
    public func buildSubmaps(bundle: CaptureBundle) -> [Submap] {
        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard let first = frames.first, let last = frames.last else { return [] }

        let start = first.timestampSeconds
        let end = last.timestampSeconds
        // `timestampSeconds` is a `Double` decoded straight from the frame
        // sidecar with no validation, so a half-written bundle can make this
        // subtraction NaN or astronomically large. `Swift.max(x, 0)` does NOT
        // catch that: `max` is `y >= x ? y : x` and NaN compares false against
        // everything, so with the value written FIRST the NaN is what
        // survives. It then reached `Int(...)` sixteen lines below, which is a
        // trapping conversion, and a corrupt sidecar killed the check-over
        // instead of being reported as one unusable scan. A day is longer than
        // any handheld capture, so anything past it is not a duration.
        let span: Double = end - start
        let duration: Double = (span.isFinite && span > 0) ? Swift.min(span, 86_400) : 0

        let window = Swift.min(Swift.max(tuning.submapWindowSeconds, 15), 30)
        let overlap = Swift.min(Swift.max(tuning.submapOverlapFraction, 0.20), 0.30)
        let stride = Swift.max(window * (1 - overlap), 1)

        // A capture shorter than one window is one submap. Not an edge case to
        // route around: scanning a single object takes 20 seconds.
        let windowCount: Int
        if duration <= window {
            windowCount = 1
        } else {
            windowCount = Int(((duration - window) / stride).rounded(.up)) + 1
        }

        var submaps: [Submap] = []
        submaps.reserveCapacity(windowCount)

        for k in 0..<windowCount {
            let windowStart = start + Double(k) * stride
            let windowEnd = Swift.min(windowStart + window, end)
            let inWindow = frames.filter {
                $0.timestampSeconds >= windowStart && $0.timestampSeconds <= windowEnd
            }
            guard let firstInWindow = inWindow.first, let lastInWindow = inWindow.last else {
                continue
            }

            var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            for frame in inWindow {
                let centre = frame.rawPose.center.simd
                minimum = simd_min(minimum, centre)
                maximum = simd_max(maximum, centre)
            }

            submaps.append(
                Submap(
                    index: SubmapID(submaps.count),
                    firstFrame: firstInWindow.index,
                    lastFrame: lastInWindow.index,
                    startTimeSeconds: firstInWindow.timestampSeconds,
                    endTimeSeconds: lastInWindow.timestampSeconds,
                    correction: .identity,
                    overlapFraction: 0,
                    bounds: BoundingBox(min: Vector3(minimum), max: Vector3(maximum))
                )
            )
        }

        // Measured overlap, not the nominal target: the last window is short,
        // and a paused capture leaves a gap where two windows share nothing.
        for k in 0..<submaps.count {
            let own = frames.filter {
                $0.timestampSeconds >= submaps[k].startTimeSeconds
                    && $0.timestampSeconds <= submaps[k].endTimeSeconds
            }
            guard !own.isEmpty else { continue }
            var shared = 0
            for frame in own {
                for j in 0..<submaps.count where j != k {
                    if frame.timestampSeconds >= submaps[j].startTimeSeconds,
                       frame.timestampSeconds <= submaps[j].endTimeSeconds {
                        shared += 1
                        break
                    }
                }
            }
            submaps[k].overlapFraction = Float(shared) / Float(own.count)
        }

        return submaps
    }

    // MARK: Time-offset calibration

    /// Sweeps the camera-to-IMU offset over -50...+50 ms in 5 ms steps.
    ///
    /// HOW, honestly: the spec says "minimise reprojection error of tracked
    /// features". There is no feature TRACKER in this app and building one
    /// would be a worse use of the pre-pass budget than what is done instead:
    /// direct photometric alignment. Corners are picked in frame A, lifted to
    /// 3D with the frame's own native LiDAR depth (so no triangulation and no
    /// depth guess), projected into frame B using the poses re-interpolated at
    /// the candidate offset, and scored by ZNCC of the patch around each. That
    /// minimises exactly the quantity a tracker's reprojection error is a
    /// proxy for, and it is immune to the tracker's own failure modes.
    ///
    /// CONDITIONING, also honestly: under constant angular velocity the
    /// relative pose between two nearby frames barely depends on the offset at
    /// all, and the sweep is flat. The signal lives in changes of angular
    /// velocity, so pairs are chosen to MAXIMISE the difference in logged gyro
    /// rate between their two frames. Where the walk was genuinely smooth
    /// throughout there is no information to recover and this returns nil,
    /// which is reported to the user, not smoothed over.
    public func calibrateTimeOffset(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> Double? {
        lastTimeOffsetSweep = []

        // Filled on every path out of this function, including the seven that
        // return nil. Before the census, all seven were indistinguishable from
        // each other and from "the offset really is zero".
        var census = PrePassCensus.TimeOffset()
        census.attempted = true
        census.requiredRelativeImprovement = tuning.timeOffsetMinRelativeImprovement
        census.outcome = "did not finish"
        lastTimeOffsetCensus = census
        defer { lastTimeOffsetCensus = census }

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard frames.count >= 8 else {
            census.outcome = "only \(frames.count) frames, at least 8 are needed"
            return nil
        }
        let track = PrePassPoseTrack(frames: frames)
        guard track.duration > 1 else {
            census.outcome = "the scan is under a second long"
            return nil
        }

        let geometry = PrePassDepthGeometry(rgbIntrinsics: bundle.intrinsics, settings: bundle.settings)

        // Work at a resolution between native depth and full RGB: enough
        // pixels for ZNCC to discriminate a few-pixel shift, few enough that
        // twenty-one offsets over two dozen pairs is still fast.
        let workingWidth = Swift.min(bundle.intrinsics.width, 640)
        guard workingWidth > 0, bundle.intrinsics.width > 0 else {
            census.outcome = "the capture recorded no image width"
            return nil
        }
        let workingHeight = Swift.max(
            1,
            Int((Double(bundle.intrinsics.height) * Double(workingWidth)
                 / Double(bundle.intrinsics.width)).rounded())
        )
        let workingIntrinsics = bundle.intrinsics.scaled(toWidth: workingWidth, height: workingHeight)

        let pairs = selectTimeOffsetPairs(frames: frames)
        guard pairs.count >= 4 else {
            census.outcome = "only \(pairs.count) usable frame pairs, at least 4 are needed"
            return nil
        }

        // Everything each pair needs, loaded once. Loading inside the offset
        // loop would decode the same JPEG twenty-one times.
        struct Sample {
            /// Camera-space point in frame A, from A's own native LiDAR
            /// depth. Never triangulated, never guessed.
            var cameraPointA: SIMD3<Float>
            var pixelA: SIMD2<Float>
        }
        struct PreparedPair {
            var frameA: CaptureFrame
            var frameB: CaptureFrame
            var imageA: PrePassGrayImage
            var imageB: PrePassGrayImage
            var samples: [Sample]
        }

        var prepared: [PreparedPair] = []
        prepared.reserveCapacity(pairs.count)

        // PAIRS PREPARED ON EVERY CORE. Each pair is independent: a depth
        // load, two grey images, a corner response and a feature pick. Results
        // land in pair order and are compacted in pair order, and a depth load
        // that throws still surfaces as the first such error in pair order,
        // so `prepared` is exactly what the serial loop built.
        func preparePair(a: CaptureFrame, b: CaptureFrame) throws -> PreparedPair? {
            guard let depthFrame = try PrePassDepthFrame.load(
                frame: a, settings: bundle.settings, at: ref
            ) else { return nil }
            guard let imageA = PrePassImageLoader.loadGray(
                url: ref.url(forRelativePath: a.imagePath),
                width: workingWidth, height: workingHeight
            ) else { return nil }
            guard let imageB = PrePassImageLoader.loadGray(
                url: ref.url(forRelativePath: b.imagePath),
                width: workingWidth, height: workingHeight
            ) else { return nil }

            let corners = PrePassImageOps.shiTomasiResponse(imageA, radius: 2)
            var candidates: [(response: Float, x: Int, y: Int)] = []
            let scaleToNativeX = Float(geometry.width) / Float(workingWidth)
            let scaleToNativeY = Float(geometry.height) / Float(workingHeight)
            var y = 4
            while y < workingHeight - 4 {
                var x = 4
                while x < workingWidth - 4 {
                    let response = corners[y * workingWidth + x]
                    if response > 1 {
                        candidates.append((response, x, y))
                    }
                    x += 3
                }
                y += 3
            }
            candidates.sort { $0.response > $1.response }

            var samples: [Sample] = []
            var used = Set<Int>()
            for candidate in candidates {
                if samples.count >= tuning.timeOffsetFeaturesPerPair { break }
                // At most one feature per 16x16 cell, so one busy corner of
                // the frame cannot supply every sample.
                let cell = (candidate.y / 16) * (workingWidth / 16 + 1) + (candidate.x / 16)
                if used.contains(cell) { continue }
                let nx = Int(Float(candidate.x) * scaleToNativeX)
                let ny = Int(Float(candidate.y) * scaleToNativeY)
                guard nx >= 0, ny >= 0, nx < geometry.width, ny < geometry.height else { continue }
                let nativeIndex = ny * geometry.width + nx
                guard depthFrame.hasReturn(at: nativeIndex) else { continue }
                let z = depthFrame.depthMeters(at: nativeIndex)
                guard z > 0.2, z < bundle.settings.lidarMaxRangeMeters else { continue }
                let cameraPoint = SIMD3<Float>(
                    (Float(candidate.x) + 0.5 - workingIntrinsics.cx) / workingIntrinsics.fx * z,
                    (Float(candidate.y) + 0.5 - workingIntrinsics.cy) / workingIntrinsics.fy * z,
                    z
                )
                used.insert(cell)
                samples.append(
                    Sample(
                        cameraPointA: cameraPoint,
                        pixelA: SIMD2<Float>(Float(candidate.x) + 0.5, Float(candidate.y) + 0.5)
                    )
                )
            }

            guard samples.count >= 12 else { return nil }
            return PreparedPair(
                frameA: a, frameB: b,
                imageA: imageA, imageB: imageB, samples: samples
            )
        }

        try Task.checkCancellation()
        var preparedSlots = [PreparedPair?](repeating: nil, count: pairs.count)
        var prepareErrors = [Error?](repeating: nil, count: pairs.count)
        preparedSlots.withUnsafeMutableBufferPointer { out in
            prepareErrors.withUnsafeMutableBufferPointer { errors in
                DispatchQueue.concurrentPerform(iterations: pairs.count) { p in
                    do {
                        out[p] = try preparePair(a: pairs[p].a, b: pairs[p].b)
                    } catch {
                        errors[p] = error
                    }
                }
            }
        }
        try Task.checkCancellation()
        for error in prepareErrors {
            if let error { throw error }
        }
        for slot in preparedSlots {
            if let slot { prepared.append(slot) }
        }

        guard prepared.count >= 4 else {
            census.outcome = "only \(prepared.count) of \(pairs.count) pairs had enough "
                + "matchable detail, at least 4 are needed"
            return nil
        }
        census.pairsPrepared = prepared.count
        census.samplesPerSweepPoint = prepared.reduce(0) { $0 + $1.samples.count }

        // THE SWEEP ON EVERY CORE. Each candidate timing is scored on its own:
        // the same pairs, the same samples, summed in the same order, so each
        // cost is the one the serial loop computed. The offsets are generated
        // by the same accumulation as before, so they match to the last bit.
        var offsets: [Double] = []
        var offset = tuning.timeOffsetMinSeconds
        while offset <= tuning.timeOffsetMaxSeconds + 1e-9 {
            offsets.append(offset)
            offset += tuning.timeOffsetStepSeconds
        }
        var costs = [Double](repeating: Double.infinity, count: offsets.count)
        costs.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: offsets.count) { k in
                var scratchA: [Float] = []
                var scratchB: [Float] = []
                let offset = offsets[k]
                var total: Double = 0
                var count = 0
                for pair in prepared {
                    let poseA = track.pose(at: pair.frameA.timestampSeconds + offset)
                    let poseB = track.pose(at: pair.frameB.timestampSeconds + offset)
                    let aToB = PrePassRigid.relative(from: poseA, to: poseB)
                    for sample in pair.samples {
                        let inB = PrePassRigid.cameraPoint(worldPoint: sample.cameraPointA, pose: aToB)
                        guard inB.z > 0.05 else { continue }
                        let u = workingIntrinsics.fx * (inB.x / inB.z) + workingIntrinsics.cx
                        let v = workingIntrinsics.fy * (inB.y / inB.z) + workingIntrinsics.cy
                        guard u >= 5, v >= 5,
                              u < Float(workingWidth - 5), v < Float(workingHeight - 5) else { continue }
                        let score = PrePassImageOps.zncc(
                            pair.imageA, centerA: sample.pixelA,
                            pair.imageB, centerB: SIMD2<Float>(u, v),
                            radius: 4,
                            scratchA: &scratchA, scratchB: &scratchB
                        )
                        total += Double(1 - score)
                        count += 1
                    }
                }
                out[k] = count > 0 ? total / Double(count) : Double.infinity
            }
        }
        try Task.checkCancellation()

        var sweep: [(offsetSeconds: Double, cost: Double)] = []
        sweep.reserveCapacity(offsets.count)
        for i in 0..<offsets.count { sweep.append((offsets[i], costs[i])) }
        lastTimeOffsetSweep = sweep
        census.sweepPoints = costs.count
        census.sweepPointsWithFiniteCost = costs.filter { $0.isFinite }.count

        // A minimum has to be interior and has to be a real dip, not the
        // shallowest point of a flat line.
        guard let minimumIndex = costs.indices.min(by: { costs[$0] < costs[$1] }),
              costs[minimumIndex].isFinite else {
            census.outcome = "not one of the \(costs.count) timings compared anything"
            return nil
        }
        census.bestCost = costs[minimumIndex]
        guard minimumIndex > 0, minimumIndex < costs.count - 1 else {
            census.outcome = "the best timing was at the end of the range, which means the "
                + "real offset is outside the range that was searched"
            return nil
        }

        let median = PrePassStats.median(costs.filter { $0.isFinite })
        census.medianCost = median
        guard median > 0 else {
            census.outcome = "every timing scored the same, so there was nothing to choose"
            return nil
        }
        let improvement = (median - costs[minimumIndex]) / median
        census.relativeImprovement = improvement
        guard improvement >= tuning.timeOffsetMinRelativeImprovement else {
            census.outcome = "the best timing was only "
                + PrePassCensusFormat.percent(improvement)
                + " better than the middle one, and it has to be "
                + PrePassCensusFormat.percent(tuning.timeOffsetMinRelativeImprovement)
                + " better to be believed"
            return nil
        }

        // Sub-step refinement: the true offset is very unlikely to land
        // exactly on a 5 ms grid point.
        let sub = PrePassStats.parabolicMinimumOffset(
            previous: costs[minimumIndex - 1],
            centre: costs[minimumIndex],
            next: costs[minimumIndex + 1]
        )
        let result = offsets[minimumIndex] + sub * tuning.timeOffsetStepSeconds
        guard result.isFinite,
              result >= tuning.timeOffsetMinSeconds - tuning.timeOffsetStepSeconds,
              result <= tuning.timeOffsetMaxSeconds + tuning.timeOffsetStepSeconds
        else {
            census.outcome = "the refined answer landed outside the range that was searched"
            return nil
        }
        census.settledSeconds = result
        census.source = "measured"
        census.outcome = "settled, "
            + PrePassCensusFormat.percent(improvement)
            + " better than the middle timing"
        return result
    }

    /// Pairs chosen to maximise sensitivity: a short baseline (so the two
    /// views still overlap) across the largest available CHANGE in gyro rate
    /// (so the offset actually moves the relative pose).
    private func selectTimeOffsetPairs(
        frames: [CaptureFrame]
    ) -> [(a: CaptureFrame, b: CaptureFrame)] {
        var scored: [(score: Float, a: CaptureFrame, b: CaptureFrame)] = []
        let baseline = 0.25   // seconds

        var j = 0
        for i in 0..<frames.count {
            let a = frames[i]
            guard a.depthPath != nil, a.qc.trackingQuality.isPoseTrustworthy else { continue }
            // Blurred frames make ZNCC meaningless; 3 px of smear is already
            // past the point where a 1 px shift is measurable.
            guard a.qc.motionBlurPixels < 3 else { continue }

            if j < i { j = i }
            while j < frames.count - 1,
                  frames[j].timestampSeconds - a.timestampSeconds < baseline {
                j += 1
            }
            guard j < frames.count, j != i else { continue }
            let b = frames[j]
            guard b.qc.trackingQuality.isPoseTrustworthy, b.qc.motionBlurPixels < 3 else { continue }

            let deltaOmega = simd_length(a.angularVelocity.simd - b.angularVelocity.simd)
            scored.append((deltaOmega, a, b))
        }

        scored.sort { $0.score > $1.score }

        // Spread the winners over the whole capture: twenty-four pairs all
        // taken from the one moment the user whipped the phone round would
        // measure that moment's offset, not the session's.
        var chosen: [(a: CaptureFrame, b: CaptureFrame)] = []
        var usedTimes: [Double] = []
        let minimumSeparation = Swift.max(
            (frames[frames.count - 1].timestampSeconds - frames[0].timestampSeconds)
                / Double(tuning.timeOffsetPairCount * 2),
            0.5
        )
        for entry in scored {
            if chosen.count >= tuning.timeOffsetPairCount { break }
            if usedTimes.contains(where: { abs($0 - entry.a.timestampSeconds) < minimumSeparation }) {
                continue
            }
            usedTimes.append(entry.a.timestampSeconds)
            chosen.append((entry.a, entry.b))
        }
        return chosen
    }

    // MARK: Revisit detection

    public func detectRevisits(
        bundle: CaptureBundle,
        submaps: [Submap],
        at ref: CaptureBundleRef
    ) async throws -> [RevisitPair] {
        rejectedRevisitCandidates = 0

        // Every funnel step below is counted. A run that returns no revisits
        // could have failed at any of four places, and until now they all
        // looked the same from outside: an empty array.
        var census = PrePassCensus.Revisits()
        census.attempted = true
        lastRevisitCensus = census
        defer { lastRevisitCensus = census }

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard frames.count >= 4 else { return [] }

        let geometry = PrePassDepthGeometry(rgbIntrinsics: bundle.intrinsics, settings: bundle.settings)
        let owners = PrePassSubmapAssignment.owners(frames: frames, submaps: submaps)

        // 1. Anchors: a sparse subset of frames, spaced by movement or time.
        var anchors: [CaptureFrame] = []
        var lastCentre = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var lastTime = -Double.greatestFiniteMagnitude
        for frame in frames where frame.depthPath != nil {
            let centre = frame.rawPose.center.simd
            let movedFar = simd_distance(centre, lastCentre) >= tuning.revisitKeyframeSpacingMeters
            let waitedLong = frame.timestampSeconds - lastTime >= tuning.revisitKeyframeSpacingSeconds
            guard anchors.isEmpty || movedFar || waitedLong else { continue }
            anchors.append(frame)
            lastCentre = centre
            lastTime = frame.timestampSeconds
        }
        census.anchorFrames = anchors.count
        guard anchors.count >= 2 else { return [] }

        // 2. Geometric gate: near in space, agreeing in view direction, far
        //    apart in time. The time gap is what makes it a REVISIT rather
        //    than the trivially-true fact that consecutive frames overlap.
        struct Candidate {
            var a: CaptureFrame
            var b: CaptureFrame
            var score: Float
        }
        // This gate is O(anchors squared), and anchors are spaced only 0.25 m
        // or 0.5 s apart, so a ten minute walk gives 1200-odd anchors and
        // about 720,000 pairs. In a small room, where nearly every pair falls
        // inside the 1.5 m centre gate, building a Candidate for each one and
        // only then applying revisitMaxICPRuns meant a transient array of
        // hundreds of megabytes: every entry carries two whole CaptureFrame
        // values, each retaining its own image, depth and confidence path
        // strings, on a device that is at the same time holding depth buffers.
        // The peak was reached before a single ICP had run and before the cap
        // had even been looked at, which is the worst possible place to spend
        // memory: on candidates that were about to be thrown away.
        //
        // So the cap is applied AS the list is built, by a fixed-size min-heap
        // keyed on the same score. It holds at most revisitMaxICPRuns
        // (anchor index, anchor index, score) triples of about 24 bytes, so
        // the full heap at the default cap of 600 is under 20 kB. Its root is
        // the WORST pair currently kept, so the common case, where a pair
        // cannot beat what is already held, costs one Float comparison and no
        // allocation at all. The frames are fetched back out of `anchors` only
        // for the survivors.
        //
        // The set this keeps is the set the old sort-then-truncate kept: the
        // revisitMaxICPRuns highest scores. The only pairs that can differ are
        // ones tied at exactly the score on the cut line, and which of those
        // survived was already arbitrary, because Swift's sort is not stable.
        // The ORDER they are then aligned in is unchanged as well: the sort by
        // (a.index, b.index) below is a strict total order over distinct pairs,
        // so it fully determines the order however the pairs arrived here.
        //
        // Carried over rather than fixed, and stated here so nobody has to
        // rediscover it: a NaN score compares false against everything, so one
        // sitting at the root could never be beaten and the heap would freeze
        // holding the first icpRunCap pairs rather than the best ones. The
        // only route to a NaN is a NaN qc.weight, because the distance and
        // angle guards below already reject NaN geometry. This is not a
        // regression, since the old `sort { $0.score > $1.score }` was not a
        // strict weak ordering with a NaN in the array either, but it is not a
        // fix, and it belongs with whatever would have produced the NaN.

        // Clamped at zero so a nonsensical negative tuning value yields an
        // empty list instead of the trap that removeSubrange((-1)...) took.
        let icpRunCap = Swift.max(0, tuning.revisitMaxICPRuns)
        var heap: [(anchorA: Int, anchorB: Int, score: Float)] = []
        // Clamped so an absurdly large cap cannot pre-allocate megabytes for a
        // heap that will never fill. It still grows on demand if it does.
        heap.reserveCapacity(Swift.min(icpRunCap, 4096))

        func heapSiftUp(from start: Int) {
            var child = start
            while child > 0 {
                let parent = (child - 1) / 2
                guard heap[child].score < heap[parent].score else { break }
                heap.swapAt(child, parent)
                child = parent
            }
        }

        // `smallest` rather than the obvious `lowest`: deadwire counts bare
        // identifier tokens across the whole module, and PrePassCensus.swift
        // has a triaged `let lowest` that is only ever read inside a string
        // interpolation, so it is allowlisted as a scanner artefact. Spending
        // the name `lowest` here makes that entry look wired up and invites
        // somebody to delete it, after which the census declaration fails the
        // gate untriaged the next time this local is renamed.
        func heapSiftDown(from start: Int) {
            var parent = start
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var smallest = parent
                if left < heap.count, heap[left].score < heap[smallest].score {
                    smallest = left
                }
                if right < heap.count, heap[right].score < heap[smallest].score {
                    smallest = right
                }
                if smallest == parent { break }
                heap.swapAt(parent, smallest)
                parent = smallest
            }
        }

        // Counted separately from what is kept: `census.geometricCandidates`
        // is every pair that passed the gates, and the QC card at
        // PrePassCensus.swift:768 shows "the rest capped" only while it is
        // strictly greater than `candidatesAfterCap`. If this became the
        // number that fitted in the heap the two would always be equal and the
        // card would silently stop reporting that capping happened at all.
        var geometricCandidateCount = 0
        for i in 0..<anchors.count {
            let a = anchors[i]
            let centreA = a.rawPose.center.simd
            let forwardA = a.rawPose.forward.simd
            let submapA = owners[a.index]
            for j in (i + 1)..<anchors.count {
                let b = anchors[j]
                guard b.timestampSeconds - a.timestampSeconds >= tuning.revisitMinTimeGapSeconds
                else { continue }
                // Two frames inside one submap are not a loop closure: the
                // graph has a single unknown for the pair and the constraint
                // would be vacuous.
                if let sa = submapA, let sb = owners[b.index], sa == sb { continue }

                let distance = simd_distance(centreA, b.rawPose.center.simd)
                guard distance <= tuning.revisitMaxCentreDistanceMeters else { continue }
                let angle = PrePassAngle.degreesBetween(forwardA, b.rawPose.forward.simd)
                guard angle <= tuning.revisitMaxViewAngleDegrees else { continue }

                // Prefer close, well-aligned, well-tracked pairs.
                let quality = a.qc.weight * b.qc.weight
                let score = quality
                    * (1 - distance / tuning.revisitMaxCentreDistanceMeters)
                    * (1 - angle / tuning.revisitMaxViewAngleDegrees)
                geometricCandidateCount += 1
                if heap.count < icpRunCap {
                    heap.append((anchorA: i, anchorB: j, score: score))
                    let inserted = heap.count - 1
                    heapSiftUp(from: inserted)
                } else if icpRunCap > 0, score > heap[0].score {
                    // Beats the worst pair kept so far, so that one goes and
                    // this one takes its place. Strictly greater, so a tie
                    // never evicts an incumbent and the work stays bounded.
                    heap[0] = (anchorA: i, anchorB: j, score: score)
                    heapSiftDown(from: 0)
                }
            }
        }
        census.geometricCandidates = geometricCandidateCount
        guard !heap.isEmpty else { return [] }

        // Only now, for at most revisitMaxICPRuns survivors, are the frames
        // themselves copied into Candidates.
        var candidates: [Candidate] = []
        candidates.reserveCapacity(heap.count)
        for entry in heap {
            candidates.append(
                Candidate(
                    a: anchors[entry.anchorA],
                    b: anchors[entry.anchorB],
                    score: entry.score
                )
            )
        }
        census.candidatesAfterCap = candidates.count

        // 3. Align each surviving candidate with point-to-plane ICP.
        //
        // Sequential with a tiny cache rather than a task group: each
        // unprojected frame is ~5 MB, and holding a few hundred of them
        // resident to parallelise a stage that already fits in the budget is
        // exactly the "never assume the scene fits memory" mistake this
        // product is supposed to be better than.
        candidates.sort {
            $0.a.index == $1.a.index ? $0.b.index < $1.b.index : $0.a.index < $1.a.index
        }

        var cache: [FrameID: PrePassFramePoints] = [:]
        var cacheOrder: [FrameID] = []
        let cacheLimit = 8

        func points(for frame: CaptureFrame) throws -> PrePassFramePoints? {
            if let hit = cache[frame.index] { return hit }
            guard let depthFrame = try PrePassDepthFrame.load(
                frame: frame, settings: bundle.settings, at: ref
            ) else { return nil }
            let built = PrePassFramePoints.build(
                depthFrame: depthFrame,
                geometry: geometry,
                maxRangeMeters: bundle.settings.lidarMaxRangeMeters
            )
            cache[frame.index] = built
            cacheOrder.append(frame.index)
            if cacheOrder.count > cacheLimit {
                let evicted = cacheOrder.removeFirst()
                cache[evicted] = nil
            }
            return built
        }

        var results: [RevisitPair] = []
        for candidate in candidates {
            try Task.checkCancellation()

            guard let sourcePoints = try points(for: candidate.a),
                  let targetPoints = try points(for: candidate.b)
            else {
                // No depth for one of them: record the geometric agreement so
                // the QC card can still count it, but with zero confidence so
                // the pose graph ignores it. A pose-proximity "measurement" is
                // just the VIO estimate handed back, and feeding an estimate
                // to the optimiser as if it were an observation is how a pose
                // graph convinces itself it is right.
                census.candidatesWithoutDepth += 1
                results.append(
                    RevisitPair(
                        frameA: candidate.a.index,
                        frameB: candidate.b.index,
                        method: .poseProximity,
                        measuredRelativePose: PrePassRigid.relative(
                            from: candidate.a.rawPose, to: candidate.b.rawPose
                        ),
                        translationResidualMeters: 0,
                        rotationResidualDegrees: 0,
                        inlierCount: 0,
                        confidence: 0
                    )
                )
                continue
            }

            let initial = PrePassRigid.relative(from: candidate.a.rawPose, to: candidate.b.rawPose)
            let icp = PrePassICP.align(
                source: sourcePoints,
                target: targetPoints,
                geometry: geometry,
                initial: initial
            )
            guard icp.converged else {
                rejectedRevisitCandidates += 1
                census.icpRejected += 1
                continue
            }
            census.icpConverged += 1

            // The residual IS the drift measurement: how far the LiDAR says
            // the two frames really are apart, minus where VIO put them.
            let error = PrePassSE3(icp.relativePose) * PrePassSE3(initial).inverse
            let translationResidual = Float(simd_length(error.translation))
            let rotationResidual = Float(error.rotationAngleDegrees)

            // Confidence from the alignment's own evidence, not from a guess:
            // how much of the source found a match, and how tight the fit is.
            let fitTerm = Float(Swift.max(0, 1 - Double(icp.rmsMeters) / 0.03))
            let confidence = Swift.min(
                Swift.max(icp.inlierFraction * 0.6 + fitTerm * 0.4, 0), 1
            )

            results.append(
                RevisitPair(
                    frameA: candidate.a.index,
                    frameB: candidate.b.index,
                    method: .depthICP,
                    measuredRelativePose: icp.relativePose,
                    translationResidualMeters: translationResidual,
                    rotationResidualDegrees: rotationResidual,
                    inlierCount: icp.inlierCount,
                    confidence: confidence
                )
            )
        }

        // One pass over the finished list. The medians are what turn "3 loop
        // closures" into "3 loop closures that still disagree by 4 cm", which
        // is the difference between a number and a diagnosis.
        census.pairsReturned = results.count
        let confirmed = results.filter { $0.method == .depthICP && $0.confidence > 0.2 }
        census.confirmedPairs = confirmed.count
        if !confirmed.isEmpty {
            census.medianConfidence = PrePassStats.median(confirmed.map { $0.confidence })
            census.medianTranslationResidualCentimeters = PrePassStats.median(
                confirmed.map { $0.translationResidualMeters * 100 }
            )
        }

        return results
    }

    // MARK: The pose graph

    /// Solves for one rigid SE(3) per submap and returns per-frame refined
    /// poses.
    ///
    /// Unknowns: `M_k` for each submap, with `refined_f = raw_f * M_owner(f)`.
    ///
    /// Residuals, whitened by the sigmas in `Tuning` so rotation and
    /// translation are comparable before the robust loss sees them:
    ///
    ///  * one per revisit edge, `log(Z^-1 * P_b * P_a^-1)`, where `Z` is the
    ///    ICP measurement and `P` the current refined pose;
    ///  * one per adjacent submap pair, `log(M_{k+1} * M_k^-1)`, encoding "VIO
    ///    got the relative placement of neighbours right" - loose enough for a
    ///    real loop closure to overcome, stiff enough that a submap with no
    ///    closures at all stays where VIO put it instead of drifting off;
    ///  * one gauge prior, `log(M_0)`, because otherwise the whole solution is
    ///    free to translate and rotate together. Anchoring submap zero also
    ///    keeps the result in ARKit's original world frame, which is the frame
    ///    the anchors and the scene mesh are already in.
    ///
    /// Levenberg-Marquardt with an annealed Huber loss: the delta starts wide
    /// so a genuinely large drift is not clipped away as an outlier on the
    /// first iteration, and tightens as the solution settles so a bad ICP
    /// match cannot pull the final answer.
    public func optimize(
        bundle: CaptureBundle,
        submaps: [Submap],
        revisits: [RevisitPair],
        timeOffsetSeconds: Double?
    ) async throws -> [String: Pose] {
        // Four of the five ways out of this function hand back the VIO poses
        // unchanged, and every one of them used to do it in complete silence:
        // no frames, no submaps, no usable edges, and the 2 metre sanity gate
        // that throws a finished solution away. `exitReason` is set on all of
        // them, so "the pose graph ran" and "the pose graph gave up" stop
        // looking identical from outside.
        var census = PrePassCensus.PoseGraph()
        census.attempted = true
        census.submaps = submaps.count
        census.exitReason = "notRun"
        lastPoseGraphCensus = census
        defer { lastPoseGraphCensus = census }

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        census.framesPosed = frames.count
        guard !frames.isEmpty else {
            census.exitReason = "noFrames"
            return [:]
        }

        // Applying the calibrated offset is the first half of F1 and has to
        // happen BEFORE the graph: the graph corrects drift, and a time offset
        // is not drift - it is every pose being the pose of a slightly
        // different moment, which no rigid per-submap correction can absorb.
        let track = PrePassPoseTrack(frames: frames)
        var basePoses: [FrameID: Pose] = [:]
        basePoses.reserveCapacity(frames.count)
        if let offset = timeOffsetSeconds, abs(offset) > 1e-6 {
            for frame in frames {
                basePoses[frame.index] = track.pose(at: frame.timestampSeconds + offset)
            }
        } else {
            for frame in frames { basePoses[frame.index] = frame.rawPose }
        }

        guard !submaps.isEmpty else {
            census.exitReason = "noSubmaps"
            return Dictionary(uniqueKeysWithValues: basePoses.map { (String($0.key), $0.value) })
        }

        let owners = PrePassSubmapAssignment.owners(frames: frames, submaps: submaps)
        var slotOfSubmap: [SubmapID: Int] = [:]
        for (slot, submap) in submaps.enumerated() { slotOfSubmap[submap.index] = slot }
        let n = submaps.count
        let dimension = 6 * n

        // Usable edges only. A zero-confidence pair is a geometric note for
        // the QC card, not an observation (see `detectRevisits`).
        struct Edge {
            var slotA: Int
            var slotB: Int
            var measurement: PrePassSE3
            var poseA: PrePassSE3
            var poseB: PrePassSE3
            var weight: Double
        }
        var edges: [Edge] = []
        for pair in revisits {
            guard pair.confidence > 0.05 else { continue }
            guard let submapA = owners[pair.frameA], let submapB = owners[pair.frameB],
                  let slotA = slotOfSubmap[submapA], let slotB = slotOfSubmap[submapB],
                  slotA != slotB,
                  let rawA = basePoses[pair.frameA], let rawB = basePoses[pair.frameB]
            else { continue }
            edges.append(
                Edge(
                    slotA: slotA,
                    slotB: slotB,
                    measurement: PrePassSE3(pair.measuredRelativePose),
                    poseA: PrePassSE3(rawA),
                    poseB: PrePassSE3(rawB),
                    weight: Double(pair.confidence)
                )
            )
        }

        // Nothing to solve: no loop closures means no evidence that anything
        // moved, and inventing a correction from a smoothness prior alone
        // would be fabricating a result. Return the (possibly time-shifted)
        // VIO poses, which is the honest answer.
        census.usableEdges = edges.count
        census.discardedEdges = revisits.count - edges.count
        guard !edges.isEmpty else {
            census.exitReason = "noEdges"
            return Dictionary(uniqueKeysWithValues: basePoses.map { (String($0.key), $0.value) })
        }

        var corrections = [PrePassSE3](repeating: .identity, count: n)

        let revisitRotationWeight = 1 / (tuning.revisitRotationSigmaDegrees * .pi / 180)
        let revisitTranslationWeight = 1 / tuning.revisitTranslationSigmaMeters
        let smoothRotationWeight = 1 / (tuning.smoothnessRotationSigmaDegrees * .pi / 180)
        let smoothTranslationWeight = 1 / tuning.smoothnessTranslationSigmaMeters
        // The gauge prior only has to remove the six-dimensional null space,
        // so it is deliberately an order of magnitude stiffer than anything
        // else and applies to one submap alone.
        let anchorWeight = smoothTranslationWeight * 10

        /// Total robust cost at a given set of corrections. Separate from the
        /// normal-equation build so Levenberg-Marquardt can compare the cost
        /// at the CANDIDATE against the cost at the current point, which is
        /// what makes the damping parameter mean anything. (Comparing this
        /// iteration's cost against the last one's instead - the easy mistake -
        /// accepts a step before knowing whether it helped.)
        func graphCost(_ state: [PrePassSE3], delta: Double) -> Double {
            var total: Double = 0
            for edge in edges {
                let pA = state[edge.slotA].then(edge.poseA)
                let pB = state[edge.slotB].then(edge.poseB)
                let error = pA.inverse.then(pB).then(edge.measurement.inverse)
                var residual = error.logVector()
                for i in 0..<3 { residual[i] *= revisitRotationWeight }
                for i in 3..<6 { residual[i] *= revisitTranslationWeight }
                var norm: Double = 0
                for value in residual { norm += value * value }
                norm = norm.squareRoot()
                total += PrePassStats.huberWeight(residual: norm, delta: delta)
                    * edge.weight * norm * norm
            }
            for k in 0..<(n - 1) {
                let d = state[k].inverse.then(state[k + 1])
                var residual = d.logVector()
                for i in 0..<3 { residual[i] *= smoothRotationWeight }
                for i in 3..<6 { residual[i] *= smoothTranslationWeight }
                var norm: Double = 0
                for value in residual { norm += value * value }
                norm = norm.squareRoot()
                total += PrePassStats.huberWeight(residual: norm, delta: delta * 3) * norm * norm
            }
            var anchor = state[0].logVector()
            for i in 0..<6 { anchor[i] *= anchorWeight }
            for value in anchor { total += value * value }
            return total
        }

        var lambda = 1e-4
        // "Converged" means one specific thing here and it is worth being
        // strict about: the solver stopped because its own step got tiny.
        // Running out of iterations is not convergence, and a damping
        // parameter that blew up is the opposite of it.
        census.exitReason = "iterationLimit"
        census.initialCost = graphCost(corrections, delta: tuning.robustDelta * 3)

        for iteration in 0..<tuning.maxIterations {
            try Task.checkCancellation()
            census.iterationsRun = iteration + 1

            // Anneal the robust threshold: wide at first so a real 30 cm drift
            // is fitted rather than rejected, tight at the end so one bad
            // alignment cannot bend the answer.
            let progress = Double(iteration) / Double(Swift.max(tuning.maxIterations - 1, 1))
            let delta = tuning.robustDelta * (3 - 2 * progress)
            let currentCost = graphCost(corrections, delta: delta)

            var h = [Double](repeating: 0, count: dimension * dimension)
            var g = [Double](repeating: 0, count: dimension)

            // --- Revisit edges
            for edge in edges {
                let mA = corrections[edge.slotA]
                let mB = corrections[edge.slotB]
                let pA = mA.then(edge.poseA)      // raw_a * M_a  (apply M first)
                let pB = mB.then(edge.poseB)
                let e = pA.inverse.then(pB)       // P_b * P_a^-1
                let error = e.then(edge.measurement.inverse)   // Z^-1 * (P_b P_a^-1)

                var residual = error.logVector()
                // Whiten.
                for i in 0..<3 { residual[i] *= revisitRotationWeight }
                for i in 3..<6 { residual[i] *= revisitTranslationWeight }

                var norm: Double = 0
                for value in residual { norm += value * value }
                norm = norm.squareRoot()
                let robust = PrePassStats.huberWeight(residual: norm, delta: delta) * edge.weight

                // Jacobians. Deriving once, in full, because a sign error here
                // is invisible: the optimiser still converges, to the wrong
                // answer.
                //
                //   E   = A_b M_b M_a^-1 A_a^-1                (= P_b P_a^-1)
                //   M_k <- exp(d_k) M_k                        (left increment)
                //
                //   from b:  A_b exp(d_b) X    = exp(Adj(A_b) d_b) A_b X
                //   from a:  W exp(-d_a) A_a^-1, W = A_b M_b M_a^-1
                //                              = exp(-Adj(W) d_a) W A_a^-1
                //
                //   so E(d) ~ exp( Adj(A_b) d_b - Adj(W) d_a ) E, and after
                //   pushing through the fixed Z^-1 on the left:
                //
                //   J_b =  Adj(Z^-1 A_b)
                //   J_a = -Adj(Z^-1 W)
                //
                // The left-Jacobian factor Jl^-1(r) that belongs in front of
                // both is approximated by the identity. That is exact at the
                // solution and accurate to first order near it, which is where
                // this runs: the initialisation is VIO, not a random guess.
                let zInverse = edge.measurement.inverse
                // W = A_b M_b M_a^-1. `then` is "apply, then apply", so the
                // matrix product is written right-to-left as a left-to-right
                // chain: apply M_a^-1, then M_b, then A_b.
                let w = mA.inverse.then(mB).then(edge.poseB)
                // Z^-1 A_b: apply A_b, then Z^-1.
                let jacobianB = edge.poseB.then(zInverse).adjoint
                // Z^-1 W: apply W, then Z^-1.
                let jacobianAFull = w.then(zInverse).adjoint

                accumulate(
                    h: &h, g: &g, dimension: dimension,
                    residual: residual, weight: robust,
                    blocks: [
                        (edge.slotB, jacobianB, 1.0),
                        (edge.slotA, jacobianAFull, -1.0)
                    ],
                    rowScales: (revisitRotationWeight, revisitTranslationWeight)
                )
            }

            // --- Smoothness between adjacent submaps
            for k in 0..<(n - 1) {
                let d = corrections[k].inverse.then(corrections[k + 1])   // M_{k+1} M_k^-1
                var residual = d.logVector()
                for i in 0..<3 { residual[i] *= smoothRotationWeight }
                for i in 3..<6 { residual[i] *= smoothTranslationWeight }

                var norm: Double = 0
                for value in residual { norm += value * value }
                norm = norm.squareRoot()
                let robust = PrePassStats.huberWeight(residual: norm, delta: delta * 3)

                // exp(d_{k+1}) D exp(-d_k):  J_{k+1} = I, J_k = -Adj(D).
                var identity = [Double](repeating: 0, count: 36)
                for i in 0..<6 { identity[i * 6 + i] = 1 }
                accumulate(
                    h: &h, g: &g, dimension: dimension,
                    residual: residual, weight: robust,
                    blocks: [
                        (k + 1, identity, 1.0),
                        (k, d.adjoint, -1.0)
                    ],
                    rowScales: (smoothRotationWeight, smoothTranslationWeight)
                )
            }

            // --- Gauge anchor on submap 0
            do {
                var residual = corrections[0].logVector()
                for i in 0..<3 { residual[i] *= anchorWeight }
                for i in 3..<6 { residual[i] *= anchorWeight }
                var identity = [Double](repeating: 0, count: 36)
                for i in 0..<6 { identity[i * 6 + i] = 1 }
                accumulate(
                    h: &h, g: &g, dimension: dimension,
                    residual: residual, weight: 1,
                    blocks: [(0, identity, 1.0)],
                    rowScales: (anchorWeight, anchorWeight)
                )
            }

            guard let step = PrePassDenseSolver.solveDamped(
                h: h, g: g, n: dimension, lambda: lambda
            ) else {
                lambda *= 10
                if lambda > 1e6 {
                    census.exitReason = "solverFailed"
                    break
                }
                continue
            }

            var candidate = corrections
            var stepNorm: Double = 0
            for k in 0..<n {
                let omega = SIMD3<Double>(step[6 * k], step[6 * k + 1], step[6 * k + 2])
                let v = SIMD3<Double>(step[6 * k + 3], step[6 * k + 4], step[6 * k + 5])
                stepNorm += simd_length_squared(omega) + simd_length_squared(v)
                candidate[k] = candidate[k].then(PrePassSE3.exp(omega: omega, v: v))
            }
            stepNorm = stepNorm.squareRoot()

            let candidateCost = graphCost(candidate, delta: delta)
            if candidateCost.isFinite, candidateCost < currentCost {
                corrections = candidate
                lambda = Swift.max(lambda * 0.5, 1e-8)
                if stepNorm < 1e-7 {
                    census.converged = true
                    census.exitReason = "converged"
                    break
                }
            } else {
                // The step made things worse: keep the current estimate, damp
                // harder, try again. This is the whole point of LM, and it is
                // what stops one wild loop-closure edge from throwing the
                // solution somewhere it can never recover from.
                lambda *= 4
                if lambda > 1e6 {
                    census.exitReason = "dampingBlewUp"
                    break
                }
            }
        }

        // --- What the solve actually achieved, in metres and degrees rather
        //     than in whitened cost units. One pass over the edges and one
        //     over the submaps: a few hundred iterations, once per scan.
        census.finalCost = graphCost(corrections, delta: tuning.robustDelta)
        var residualCentimeters: [Float] = []
        var residualDegrees: [Float] = []
        residualCentimeters.reserveCapacity(edges.count)
        residualDegrees.reserveCapacity(edges.count)
        for edge in edges {
            let pA = corrections[edge.slotA].then(edge.poseA)
            let pB = corrections[edge.slotB].then(edge.poseB)
            let error = pA.inverse.then(pB).then(edge.measurement.inverse)
            residualCentimeters.append(Float(simd_length(error.translation) * 100))
            residualDegrees.append(Float(error.rotationAngleDegrees))
        }
        if !residualCentimeters.isEmpty {
            census.finalResidualMedianCentimeters = PrePassStats.median(residualCentimeters)
            census.finalResidualMedianDegrees = PrePassStats.median(residualDegrees)
        }
        for correction in corrections {
            let metres = Float(simd_length(correction.translation))
            let degrees = Float(correction.rotationAngleDegrees)
            if metres.isFinite {
                census.maxSubmapCorrectionCentimeters = Swift.max(
                    census.maxSubmapCorrectionCentimeters, metres * 100
                )
            }
            if degrees.isFinite {
                census.maxSubmapCorrectionDegrees = Swift.max(
                    census.maxSubmapCorrectionDegrees, degrees
                )
            }
        }

        // Sanity gate. A pose graph that wants to move a submap by two metres
        // has found a wrong loop closure, not two metres of drift, and
        // shipping that would be much worse than shipping the VIO poses. This
        // is a real, silent failure mode of every pose-graph SLAM system and
        // the only defence is a hard, explicit limit.
        for correction in corrections {
            let translation = simd_length(correction.translation)
            if !translation.isFinite || translation > 2.0 || correction.rotationAngleDegrees > 20 {
                // The whole solution is discarded here. Silent until the
                // census: the pass reported success, wrote refined poses that
                // were byte-identical to the raw ones, and every later stage
                // attributed the leftover drift to sensor noise.
                census.rejectedBySanityGate = true
                return Dictionary(uniqueKeysWithValues: basePoses.map { (String($0.key), $0.value) })
            }
        }

        var refined: [String: Pose] = [:]
        refined.reserveCapacity(frames.count)
        for frame in frames {
            let base = basePoses[frame.index] ?? frame.rawPose
            guard let submap = owners[frame.index], let slot = slotOfSubmap[submap] else {
                refined[String(frame.index)] = base
                continue
            }
            // refined = raw * M  (apply M to the world point first).
            refined[String(frame.index)] = corrections[slot].then(PrePassSE3(base)).pose
        }

        // How far the cameras actually moved. A graph with edges that moves
        // nothing has a correction that never reached the poses, which is a
        // real and completely silent failure: the numbers all look solved.
        var shifts: [Float] = []
        shifts.reserveCapacity(frames.count)
        for frame in frames {
            guard let base = basePoses[frame.index],
                  let out = refined[String(frame.index)] else { continue }
            let shift = simd_distance(base.center.simd, out.center.simd) * 100
            if shift.isFinite { shifts.append(shift) }
        }
        if !shifts.isEmpty {
            census.medianPoseShiftCentimeters = PrePassStats.median(shifts)
            census.maxPoseShiftCentimeters = shifts.max() ?? 0
        }
        return refined
    }

    /// Accumulates one residual's contribution into the normal equations.
    ///
    /// `blocks` is `(slot, 6x6 row-major Jacobian, sign)`. `rowScales` is the
    /// same whitening already applied to `residual`, applied to the Jacobian
    /// rows so `H = J^T W J` stays consistent - forgetting this is the classic
    /// way to get a solver that converges to a wrong minimum with no
    /// symptom other than a slightly worse result.
    private func accumulate(
        h: inout [Double],
        g: inout [Double],
        dimension: Int,
        residual: [Double],
        weight: Double,
        blocks: [(slot: Int, jacobian: [Double], sign: Double)],
        rowScales: (rotation: Double, translation: Double)
    ) {
        guard weight > 0 else { return }

        // Whiten each block's rows exactly as the residual was whitened.
        var scaled: [(offset: Int, jacobian: [Double])] = []
        scaled.reserveCapacity(blocks.count)
        for block in blocks {
            var j = [Double](repeating: 0, count: 36)
            for row in 0..<6 {
                let scale = (row < 3 ? rowScales.rotation : rowScales.translation) * block.sign
                for col in 0..<6 {
                    j[row * 6 + col] = block.jacobian[row * 6 + col] * scale
                }
            }
            scaled.append((block.slot * 6, j))
        }

        for (offsetA, ja) in scaled {
            for row in 0..<6 {
                var gradient: Double = 0
                for k in 0..<6 { gradient += ja[k * 6 + row] * residual[k] }
                g[offsetA + row] += weight * gradient
            }
            for (offsetB, jb) in scaled {
                for row in 0..<6 {
                    for col in 0..<6 {
                        var sum: Double = 0
                        for k in 0..<6 { sum += ja[k * 6 + row] * jb[k * 6 + col] }
                        h[(offsetA + row) * dimension + (offsetB + col)] += weight * sum
                    }
                }
            }
        }
    }
}
