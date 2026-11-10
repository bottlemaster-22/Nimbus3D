//
//  PrePassBundleAdjuster.swift
//  PrePass
//
//  THE OPTIONAL STRONGER POSE REFINEMENT: LiDAR-DEPTH-ANCHORED, AND
//  TRIANGULATION-FREE.
//
//  ---------------------------------------------------------------------------
//  WHY THIS IS NOT A NORMAL BUNDLE ADJUSTMENT
//  ---------------------------------------------------------------------------
//  A classical bundle adjustment triangulates 3D points from 2D matches and
//  then jointly optimises points and cameras. Started from an ARKit prior,
//  that measurably made the poses WORSE in 15 of 15 test rooms, and the reason
//  is not subtle: the triangulation inherits the prior's error, and the
//  adjustment then happily fits the cameras to the bad points it just made.
//  Nothing in this app triangulates. Nothing in this app runs COLMAP.
//
//  What is different here is that we have a metric depth measurement for every
//  pixel. So the structure is not unknown and does not have to be triangulated:
//  each observation is ONE unknown, the depth along its own ray, and it starts
//  at what the laser actually measured with a prior that holds it there in
//  proportion to how much that particular sample is worth believing.
//
//  ---------------------------------------------------------------------------
//  THE RESIDUALS
//  ---------------------------------------------------------------------------
//  Take one observation: host frame h, native pixel with unit camera-frame ray
//  r, measured range p, and an unknown ray depth d initialised to p.
//
//      X_h = r * d                        the point, in the host's camera frame
//      X_i = T_i T_h^-1 X_h               the same point, in a target frame
//      q   = project(X_i)                 where it lands on the target's grid
//      p_i = the target's OWN LiDAR range at q
//      Y_i = ray_i(q) * p_i               what the target says is really there
//
//  FORWARD residual, in the target:      (|X_i| - p_i) / sigma_i
//  BACKWARD residual, in the host:       (|T_h T_i^-1 Y_i| - d) / sigma_h
//  PRIOR residual, on the ray depth:     (d - p) / sigma_prior
//
//  Both directions, hence "symmetric cross-projection". Only one of them is
//  not enough: the forward residual alone can be driven to zero by pushing
//  every host depth outward until the predictions land in front of whatever
//  the target measured, which is a perfectly good minimum of a wrong cost.
//
//  Nowhere in that is a triangulated point. The geometry is anchored on two
//  independent laser measurements of the same surface, and the cameras move to
//  make those two measurements agree.
//
//  ---------------------------------------------------------------------------
//  THE ROBUST LOSS, AND THE ANNEALING
//  ---------------------------------------------------------------------------
//  Arctan, annealed 1e4 -> 1e3 -> 1e2 on the SQUARED residual, which is the
//  scale convention Ceres uses for the same loss. Arctan rather than Huber
//  because arctan's influence goes to zero for a large residual instead of
//  merely flattening: a sample that landed on the wrong side of a depth
//  discontinuity is not a mildly bad measurement to be down-weighted, it is a
//  measurement of something else entirely and should contribute nothing.
//
//  Wide first so a genuine several-centimetre disagreement is fitted rather
//  than rejected, then twice as tight so a wrong association cannot bend the
//  final answer.
//
//  ---------------------------------------------------------------------------
//  TWO HONEST LIMITS, STATED RATHER THAN HIDDEN
//  ---------------------------------------------------------------------------
//   1. DATA ASSOCIATION IS FIXED WITHIN AN ITERATION. Which target pixel an
//      observation lands on is recomputed at the top of every iteration and
//      then held while that iteration's normal equations are built. That is
//      the same convention ICP uses and for the same reason: the alternative
//      is differentiating through a nearest-neighbour lookup, which is not
//      differentiable.
//   2. IT IS GATED ON MEASURED IMPROVEMENT. The result is kept only if the
//      median revisit residual - the same independent measurement the QC card
//      reports drift from - actually got smaller. Otherwise the pose-graph
//      answer is returned unchanged. An optimiser that reports a lower value
//      of its own cost function has proved nothing about the poses.
//

import Foundation
import simd

// MARK: - Result

struct PrePassBundleAdjustmentResult {
    /// Refined poses keyed by `String(frameIndex)`, covering every frame in
    /// the capture, not only the keyframes that were optimised.
    var poses: [String: Pose]
    /// False means the poses handed back are the input poses, untouched.
    var accepted: Bool
    var medianRevisitResidualBeforeMeters: Float
    var medianRevisitResidualAfterMeters: Float
    var keyframeCount: Int
    var observationCount: Int
    var iterationsRun: Int
    /// One plain sentence for the log and for the module status row.
    var note: String
}

// MARK: - The adjuster

enum PrePassBundleAdjuster {

    struct Settings {
        /// Cameras actually optimised. The dense normal equations are
        /// 6K x 6K, so 80 keyframes is a 480 x 480 Cholesky per iteration,
        /// which is about 37 million multiply-adds: fast, and small enough to
        /// stay well inside a phone's memory during a stage that runs before
        /// training has allocated anything.
        var maxKeyframes = 80
        /// Keyframes are spaced by movement or time, whichever comes first.
        var keyframeSpacingMeters: Float = 0.30
        var keyframeSpacingSeconds: Double = 0.75
        /// Observations sampled per host keyframe.
        var observationsPerKeyframe = 120
        /// Target frames each host is cross-projected into.
        var targetsPerHost = 3
        /// A pair with less baseline than this carries no geometric
        /// information about depth, so optimising against it just adds noise.
        var minBaselineMeters: Float = 0.06
        /// And beyond this the two views have too little in common.
        var maxBaselineMeters: Float = 3.0
        var maxViewAngleDegrees: Float = 60
        /// Arctan loss scales, applied to the squared whitened residual.
        var lossScales: [Double] = [1e4, 1e3, 1e2]
        var iterationsPerScale = 4
        /// How firmly each camera is held at the pose graph's answer. Loose
        /// enough that a real correction happens, stiff enough that a camera
        /// with almost no usable observations stays put.
        var priorRotationSigmaDegrees: Double = 0.5
        var priorTranslationSigmaMeters: Double = 0.03
        /// Floor on the trust weight when it scales the depth prior, so a
        /// sample the trust field distrusts completely still cannot have an
        /// infinite-variance prior and run away.
        var minTrustWeight: Float = 0.05
        /// A single camera correction larger than this is a wrong answer, not
        /// a large one. Same reasoning as the pose graph's own sanity gate.
        var maxCorrectionMeters: Double = 0.30
        var maxCorrectionDegrees: Double = 3.0

        init() {}
    }

    /// One sampled ray of one host keyframe.
    private struct Observation {
        var host: Int
        var ray: SIMD3<Double>
        var measuredRange: Double
        var priorSigma: Double
        var depthSigma: Double
        var targets: [Int]
    }

    /// One keyframe's cached depth, so the inner loops never touch a file.
    private struct KeyframeData {
        var frame: CaptureFrame
        /// True range along each native ray, 0 where there was no usable
        /// return.
        var range: [Float]
        /// Per-sample 1-sigma, metres, physics prior divided by the trust
        /// field's own weight.
        var sigma: [Float]
        var pose: PrePassSE3
        var centre: SIMD3<Float>
        var forward: SIMD3<Float>
    }

    // MARK: Entry point

    static func refine(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        poses: [String: Pose],
        revisits: [RevisitPair],
        trustWeight: (FrameID, Int) -> Float,
        noiseModel: SmartDepthNoiseModel = .default,
        settings: Settings = Settings()
    ) throws -> PrePassBundleAdjustmentResult {

        let before = medianRevisitResidual(revisits: revisits, poses: poses)
        var unchanged = PrePassBundleAdjustmentResult(
            poses: poses,
            accepted: false,
            medianRevisitResidualBeforeMeters: before,
            medianRevisitResidualAfterMeters: before,
            keyframeCount: 0,
            observationCount: 0,
            iterationsRun: 0,
            note: "Not run."
        )

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard frames.count >= 4 else {
            unchanged.note = "Too few frames to refine against."
            return unchanged
        }

        func basePose(of frame: CaptureFrame) -> Pose {
            poses[String(frame.index)] ?? frame.refinedPose ?? frame.rawPose
        }

        // --- Keyframes.
        var chosen: [CaptureFrame] = []
        var lastCentre = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var lastTime = -Double.greatestFiniteMagnitude
        for frame in frames where frame.depthPath != nil {
            guard frame.qc.trackingQuality != .notAvailable else { continue }
            let centre = basePose(of: frame).center.simd
            let movedFar = simd_distance(centre, lastCentre) >= settings.keyframeSpacingMeters
            let waitedLong = frame.timestampSeconds - lastTime >= settings.keyframeSpacingSeconds
            guard chosen.isEmpty || movedFar || waitedLong else { continue }
            chosen.append(frame)
            lastCentre = centre
            lastTime = frame.timestampSeconds
        }
        if chosen.count > settings.maxKeyframes {
            // Even thinning, so the whole walk stays represented rather than
            // just its first minute.
            let interval = Double(chosen.count) / Double(settings.maxKeyframes)
            var thinned: [CaptureFrame] = []
            var position = 0.0
            while Int(position) < chosen.count, thinned.count < settings.maxKeyframes {
                thinned.append(chosen[Int(position)])
                position += interval
            }
            chosen = thinned
        }
        guard chosen.count >= 4 else {
            unchanged.note = "Fewer than four usable keyframes."
            return unchanged
        }

        let geometry = PrePassDepthGeometry(
            rgbIntrinsics: bundle.intrinsics, settings: bundle.settings
        )
        let width = geometry.width
        let height = geometry.height
        let sampleCount = width * height
        let maxRange = Swift.max(bundle.settings.lidarMaxRangeMeters, 0.5)

        // --- Load and pre-whiten every keyframe's depth.
        var keyframes: [KeyframeData] = []
        keyframes.reserveCapacity(chosen.count)
        for frame in chosen {
            try Task.checkCancellation()
            let depthFrame: PrePassDepthFrame?
            do {
                depthFrame = try PrePassDepthFrame.load(
                    frame: frame, settings: bundle.settings, at: ref
                )
            } catch {
                continue
            }
            guard let depthFrame else { continue }

            var range = [Float](repeating: 0, count: sampleCount)
            var sigma = [Float](repeating: 0, count: sampleCount)
            for i in 0..<sampleCount where depthFrame.hasReturn(at: i) {
                let z = depthFrame.depthMeters(at: i)
                let r = geometry.range(index: i, depthMeters: z)
                guard r > 0.2, r <= maxRange else { continue }
                range[i] = r
                // Confidence-weighted LiDAR prior: the physics sigma, widened
                // in inverse proportion to how much the trust field believes
                // this particular sample. A distrusted sample gets a loose
                // prior and is allowed to move; a trusted one is held.
                let w = Swift.max(trustWeight(frame.index, i), settings.minTrustWeight)
                sigma[i] = noiseModel.sigma(rangeMeters: r, incidenceCosine: 1) / w
            }

            let p = basePose(of: frame)
            keyframes.append(
                KeyframeData(
                    frame: frame,
                    range: range,
                    sigma: sigma,
                    pose: PrePassSE3(p),
                    centre: p.center.simd,
                    forward: p.forward.simd
                )
            )
        }
        guard keyframes.count >= 4 else {
            unchanged.note = "Not enough keyframes had readable depth."
            return unchanged
        }

        // --- Co-visibility: who each host is cross-projected into.
        var targetsOf: [[Int]] = Array(repeating: [], count: keyframes.count)
        for h in 0..<keyframes.count {
            var scored: [(score: Float, index: Int)] = []
            for i in 0..<keyframes.count where i != h {
                let baseline = simd_distance(keyframes[h].centre, keyframes[i].centre)
                guard baseline >= settings.minBaselineMeters,
                      baseline <= settings.maxBaselineMeters else { continue }
                let angle = PrePassAngle.degreesBetween(
                    keyframes[h].forward, keyframes[i].forward
                )
                guard angle <= settings.maxViewAngleDegrees else { continue }
                // Prefer the largest baseline that still overlaps: that is
                // where the depth information actually is.
                scored.append((baseline * (1 - angle / (settings.maxViewAngleDegrees + 1)), i))
            }
            scored.sort { $0.score > $1.score }
            targetsOf[h] = scored.prefix(settings.targetsPerHost).map { $0.index }
        }

        // --- Observations.
        var observations: [Observation] = []
        var depths: [Double] = []
        let strideStep = Swift.max(
            1,
            Int((Double(sampleCount) / Double(Swift.max(settings.observationsPerKeyframe, 1)))
                .squareRoot().rounded())
        )
        for h in 0..<keyframes.count {
            guard !targetsOf[h].isEmpty else { continue }
            var taken = 0
            var y = strideStep / 2
            while y < height, taken < settings.observationsPerKeyframe {
                var x = strideStep / 2
                while x < width, taken < settings.observationsPerKeyframe {
                    let index = y * width + x
                    let r = keyframes[h].range[index]
                    if r > 0 {
                        observations.append(
                            Observation(
                                host: h,
                                ray: SIMD3<Double>(geometry.rayDirections[index]),
                                measuredRange: Double(r),
                                priorSigma: Double(Swift.max(keyframes[h].sigma[index], 1e-3)),
                                depthSigma: Double(Swift.max(keyframes[h].sigma[index], 1e-3)),
                                targets: targetsOf[h]
                            )
                        )
                        depths.append(Double(r))
                        taken += 1
                    }
                    x += strideStep
                }
                y += strideStep
            }
        }
        guard observations.count >= 200 else {
            unchanged.note = "Too few laser samples were usable for the finer refinement."
            return unchanged
        }

        // --- Optimise.
        let k = keyframes.count
        let dimension = 6 * k
        var cameras = keyframes.map(\.pose)
        var deltas = [PrePassSE3](repeating: .identity, count: k)

        let priorRotationWeight = 1 / (settings.priorRotationSigmaDegrees * .pi / 180)
        let priorTranslationWeight = 1 / settings.priorTranslationSigmaMeters
        // The gauge: keyframe zero is held an order of magnitude harder than
        // the rest, which removes the six-dimensional null space (the whole
        // solution is otherwise free to translate and rotate together) and
        // keeps the answer in ARKit's original world frame.
        let anchorWeight = priorTranslationWeight * 10

        var lambda = 1e-4
        var iterationsRun = 0

        for scale in settings.lossScales {
            for _ in 0..<settings.iterationsPerScale {
                try Task.checkCancellation()
                iterationsRun += 1

                let currentCost = cost(
                    cameras: cameras, deltas: deltas, depths: depths,
                    observations: observations, keyframes: keyframes,
                    geometry: geometry, scale: scale,
                    priorRotationWeight: priorRotationWeight,
                    priorTranslationWeight: priorTranslationWeight,
                    anchorWeight: anchorWeight
                )

                var h = [Double](repeating: 0, count: dimension * dimension)
                var g = [Double](repeating: 0, count: dimension)
                // Per-observation Schur pieces, kept so the depths can be
                // back-substituted once the cameras are solved.
                var hdd = [Double](repeating: 0, count: observations.count)
                var gd = [Double](repeating: 0, count: observations.count)
                var hdc = [[Int: [Double]]](
                    repeating: [:], count: observations.count
                )

                for (j, observation) in observations.enumerated() {
                    let hostIndex = observation.host
                    let hostPose = cameras[hostIndex]
                    let d = depths[j]
                    let xHost = observation.ray * d

                    var localHDD = 0.0
                    var localGD = 0.0
                    var localHDC: [Int: [Double]] = [:]

                    // --- Prior on the ray depth.
                    do {
                        let sigma = observation.priorSigma
                        let residual = (d - observation.measuredRange) / sigma
                        let jd = 1 / sigma
                        let w = arctanWeight(squared: residual * residual, scale: scale)
                        localHDD += w * jd * jd
                        localGD += w * jd * residual
                    }

                    for targetIndex in observation.targets {
                        let targetPose = cameras[targetIndex]
                        // Host camera -> target camera.
                        let hostToTarget = hostPose.inverse.then(targetPose)
                        let xTarget = hostToTarget.act(xHost)
                        let rangeTarget = simd_length(xTarget)
                        guard rangeTarget > 0.2 else { continue }
                        guard let pixel = geometry.project(
                            cameraPoint: SIMD3<Float>(xTarget)
                        ) else { continue }
                        let targetSample = Int(pixel.y) * width + Int(pixel.x)
                        guard targetSample >= 0, targetSample < sampleCount else { continue }
                        let measuredTarget = keyframes[targetIndex].range[targetSample]
                        guard measuredTarget > 0 else { continue }
                        let sigmaTarget = Double(
                            Swift.max(keyframes[targetIndex].sigma[targetSample], 1e-3)
                        )

                        let rotationHostToTarget = simd_double3x3(
                            hostToTarget.rotation.normalized
                        )
                        let unitTarget = xTarget / rangeTarget

                        // --- FORWARD: (|X_i| - p_i) / sigma_i
                        do {
                            let residual = (rangeTarget - Double(measuredTarget)) / sigmaTarget
                            let w = arctanWeight(squared: residual * residual, scale: scale)

                            // d|X_i| / dX_i = unit(X_i).
                            // Target increment: dX_i/dxi = [-skew(X_i), I].
                            let rowTarget = jacobianRow(
                                gradient: unitTarget / sigmaTarget,
                                point: xTarget,
                                negateRotation: true
                            )
                            // Host increment: X_i = A(X_h - w_h x X_h - v_h),
                            // so dX_i/dxi_h = A [skew(X_h), -I].
                            let gradientInHost = rotationHostToTarget.transpose
                                * (unitTarget / sigmaTarget)
                            let rowHost = jacobianRow(
                                gradient: gradientInHost,
                                point: xHost,
                                negateRotation: false,
                                negateTranslation: true
                            )
                            let jd = simd_dot(
                                unitTarget / sigmaTarget,
                                rotationHostToTarget * observation.ray
                            )

                            accumulate(
                                h: &h, g: &g, dimension: dimension,
                                blocks: [(hostIndex, rowHost), (targetIndex, rowTarget)],
                                residual: residual, weight: w
                            )
                            localHDD += w * jd * jd
                            localGD += w * jd * residual
                            addSchur(&localHDC, slot: hostIndex, row: rowHost, jd: jd, weight: w)
                            addSchur(&localHDC, slot: targetIndex, row: rowTarget, jd: jd, weight: w)
                        }

                        // --- BACKWARD: (|T_h T_i^-1 Y_i| - d) / sigma_h
                        do {
                            let rayTarget = SIMD3<Double>(geometry.rayDirections[targetSample])
                            let yTarget = rayTarget * Double(measuredTarget)
                            let targetToHost = targetPose.inverse.then(hostPose)
                            let yHost = targetToHost.act(yTarget)
                            let rangeHost = simd_length(yHost)
                            guard rangeHost > 0.2 else { continue }
                            let sigmaHost = observation.depthSigma
                            let residual = (rangeHost - d) / sigmaHost
                            let w = arctanWeight(squared: residual * residual, scale: scale)
                            let unitHost = yHost / rangeHost

                            // Host increment: Y_h = T_h Y_w, so
                            // dY_h/dxi_h = [-skew(Y_h), I].
                            let rowHost = jacobianRow(
                                gradient: unitHost / sigmaHost,
                                point: yHost,
                                negateRotation: true
                            )
                            // Target increment: Y_w = T_i^-1(Y_i - w_i x Y_i
                            // - v_i), so dY_h/dxi_i = B [skew(Y_i), -I] with
                            // B the target-to-host rotation.
                            let rotationTargetToHost = simd_double3x3(
                                targetToHost.rotation.normalized
                            )
                            let gradientInTarget = rotationTargetToHost.transpose
                                * (unitHost / sigmaHost)
                            let rowTarget = jacobianRow(
                                gradient: gradientInTarget,
                                point: yTarget,
                                negateRotation: false,
                                negateTranslation: true
                            )
                            let jd = -1 / sigmaHost

                            accumulate(
                                h: &h, g: &g, dimension: dimension,
                                blocks: [(hostIndex, rowHost), (targetIndex, rowTarget)],
                                residual: residual, weight: w
                            )
                            localHDD += w * jd * jd
                            localGD += w * jd * residual
                            addSchur(&localHDC, slot: hostIndex, row: rowHost, jd: jd, weight: w)
                            addSchur(&localHDC, slot: targetIndex, row: rowTarget, jd: jd, weight: w)
                        }
                    }

                    hdd[j] = localHDD
                    gd[j] = localGD
                    hdc[j] = localHDC
                }

                // --- Priors that keep every camera near the pose graph's
                //     answer, and pin the gauge on camera zero.
                for slot in 0..<k {
                    var residual = deltas[slot].logVector()
                    let rotationWeight = slot == 0 ? anchorWeight : priorRotationWeight
                    let translationWeight = slot == 0 ? anchorWeight : priorTranslationWeight
                    for i in 0..<3 { residual[i] *= rotationWeight }
                    for i in 3..<6 { residual[i] *= translationWeight }
                    for row in 0..<6 {
                        var jacobian = [Double](repeating: 0, count: 6)
                        jacobian[row] = row < 3 ? rotationWeight : translationWeight
                        accumulate(
                            h: &h, g: &g, dimension: dimension,
                            blocks: [(slot, jacobian)],
                            residual: residual[row], weight: 1
                        )
                    }
                }

                // --- Schur complement: eliminate the per-observation depths.
                for j in 0..<observations.count {
                    let denominator = hdd[j]
                    guard denominator > 1e-12 else { continue }
                    let inverse = 1 / denominator
                    let entries = Array(hdc[j])
                    for (slotA, rowA) in entries {
                        for i in 0..<6 {
                            g[slotA * 6 + i] -= rowA[i] * gd[j] * inverse
                        }
                        for (slotB, rowB) in entries {
                            for i in 0..<6 {
                                let scaled = rowA[i] * inverse
                                for l in 0..<6 {
                                    h[(slotA * 6 + i) * dimension + (slotB * 6 + l)]
                                        -= scaled * rowB[l]
                                }
                            }
                        }
                    }
                }

                guard let step = PrePassDenseSolver.solveDamped(
                    h: h, g: g, n: dimension, lambda: lambda
                ) else {
                    lambda *= 10
                    if lambda > 1e6 { break }
                    continue
                }

                // --- Candidate cameras.
                var candidateCameras = cameras
                var candidateDeltas = deltas
                for slot in 0..<k {
                    let omega = SIMD3<Double>(
                        step[6 * slot], step[6 * slot + 1], step[6 * slot + 2]
                    )
                    let v = SIMD3<Double>(
                        step[6 * slot + 3], step[6 * slot + 4], step[6 * slot + 5]
                    )
                    guard simd_length(omega) < 0.3, simd_length(v) < 0.5 else {
                        return finish(
                            frames: frames, poses: poses,
                            keyframes: keyframes, deltas: deltas, revisits: revisits,
                            before: before, observationCount: observations.count,
                            iterationsRun: iterationsRun, settings: settings,
                            note: "Stopped early: the finer refinement wanted an implausible step."
                        )
                    }
                    let increment = PrePassSE3.exp(omega: omega, v: v)
                    // Left increment on the camera side: T <- exp(xi) T, which
                    // in `then` (apply-left-first) spelling is T.then(exp).
                    candidateCameras[slot] = candidateCameras[slot].then(increment)
                    candidateDeltas[slot] = candidateDeltas[slot].then(increment)
                }

                // --- Candidate depths, by back-substitution.
                var candidateDepths = depths
                for j in 0..<observations.count {
                    let denominator = hdd[j]
                    guard denominator > 1e-12 else { continue }
                    var numerator = -gd[j]
                    for (slot, row) in hdc[j] {
                        for i in 0..<6 { numerator -= row[i] * step[slot * 6 + i] }
                    }
                    let update = numerator / denominator
                    guard update.isFinite, abs(update) < 0.5 else { continue }
                    candidateDepths[j] = Swift.max(candidateDepths[j] + update, 0.1)
                }

                let candidateCost = cost(
                    cameras: candidateCameras, deltas: candidateDeltas,
                    depths: candidateDepths, observations: observations,
                    keyframes: keyframes, geometry: geometry, scale: scale,
                    priorRotationWeight: priorRotationWeight,
                    priorTranslationWeight: priorTranslationWeight,
                    anchorWeight: anchorWeight
                )

                if candidateCost.isFinite, candidateCost < currentCost {
                    cameras = candidateCameras
                    deltas = candidateDeltas
                    depths = candidateDepths
                    lambda = Swift.max(lambda * 0.5, 1e-8)
                } else {
                    lambda *= 4
                    if lambda > 1e6 { break }
                }
            }
        }

        return finish(
            frames: frames, poses: poses,
            keyframes: keyframes, deltas: deltas, revisits: revisits,
            before: before, observationCount: observations.count,
            iterationsRun: iterationsRun, settings: settings,
            note: ""
        )
    }

    // MARK: Finishing and gating

    /// Spreads the keyframe corrections over every frame, checks the result
    /// against an independent measurement, and keeps it only if it helped.
    private static func finish(
        frames: [CaptureFrame],
        poses: [String: Pose],
        keyframes: [KeyframeData],
        deltas: [PrePassSE3],
        revisits: [RevisitPair],
        before: Float,
        observationCount: Int,
        iterationsRun: Int,
        settings: Settings,
        note: String
    ) -> PrePassBundleAdjustmentResult {

        var result = PrePassBundleAdjustmentResult(
            poses: poses,
            accepted: false,
            medianRevisitResidualBeforeMeters: before,
            medianRevisitResidualAfterMeters: before,
            keyframeCount: keyframes.count,
            observationCount: observationCount,
            iterationsRun: iterationsRun,
            note: note.isEmpty ? "Kept the pose graph's answer." : note
        )

        // Sanity gate, exactly as the pose graph has: an optimiser that wants
        // to move a camera by a third of a metre has found a wrong
        // correspondence, not a third of a metre of error.
        for delta in deltas {
            let translation = simd_length(delta.translation)
            guard translation.isFinite,
                  translation <= settings.maxCorrectionMeters,
                  delta.rotationAngleDegrees <= settings.maxCorrectionDegrees
            else {
                result.note = "Rejected: the finer refinement asked to move a camera "
                    + "further than a real error could be."
                return result
            }
        }

        // Correction as a function of time, interpolated between keyframes, so
        // a frame between two keyframes moves with them instead of snapping to
        // the nearest one and breaking the local smoothness VIO got right.
        let times = keyframes.map(\.frame.timestampSeconds)
        let corrections = deltas.map(\.pose)
        var refined: [String: Pose] = [:]
        refined.reserveCapacity(frames.count)

        for frame in frames {
            let base = poses[String(frame.index)] ?? frame.refinedPose ?? frame.rawPose
            let correction = interpolatedCorrection(
                at: frame.timestampSeconds, times: times, corrections: corrections
            )
            // refined = correction * base, in the camera frame.
            refined[String(frame.index)] = PrePassSE3(base)
                .then(PrePassSE3(correction))
                .pose
        }

        let after = medianRevisitResidual(revisits: revisits, poses: refined)
        result.medianRevisitResidualAfterMeters = after

        // The gate. `before` of 0 means there was nothing independent to check
        // against, and an unchecked refinement is not one worth keeping.
        guard before > 0, after.isFinite, after < before else {
            result.note = before > 0
                ? "Kept the pose graph's answer: the finer refinement did not improve "
                    + "how well revisited surfaces line up."
                : "Kept the pose graph's answer: there were no revisits to check the "
                    + "finer refinement against."
            return result
        }

        result.poses = refined
        result.accepted = true
        let beforeCm = (before * 1000).rounded() / 10
        let afterCm = (after * 1000).rounded() / 10
        result.note = "Revisited surfaces line up to \(afterCm) cm, improved from \(beforeCm) cm."
        return result
    }

    private static func interpolatedCorrection(
        at time: Double,
        times: [Double],
        corrections: [Pose]
    ) -> Pose {
        guard !times.isEmpty else { return .identity }
        if times.count == 1 { return corrections[0] }
        if time <= times[0] { return corrections[0] }
        if time >= times[times.count - 1] { return corrections[corrections.count - 1] }
        var low = 0
        var high = times.count - 1
        while high - low > 1 {
            let mid = (low + high) / 2
            if times[mid] <= time { low = mid } else { high = mid }
        }
        let span = times[high] - times[low]
        guard span > 1e-9 else { return corrections[low] }
        return Pose.interpolate(
            corrections[low], corrections[high], t: Float((time - times[low]) / span)
        )
    }

    /// The independent check: for every confirmed revisit, how far apart the
    /// two frames' poses put a surface the depth alignment said they agreed
    /// on. Returns 0 when there is nothing to measure.
    static func medianRevisitResidual(
        revisits: [RevisitPair],
        poses: [String: Pose]
    ) -> Float {
        var residuals: [Float] = []
        for pair in revisits where pair.method == .depthICP && pair.confidence > 0.2 {
            guard let a = poses[String(pair.frameA)], let b = poses[String(pair.frameB)]
            else { continue }
            let implied = PrePassRigid.relative(from: a, to: b)
            let error = PrePassSE3(pair.measuredRelativePose) * PrePassSE3(implied).inverse
            let distance = Float(simd_length(error.translation))
            if distance.isFinite { residuals.append(distance) }
        }
        guard !residuals.isEmpty else { return 0 }
        return PrePassStats.median(residuals)
    }

    // MARK: Numerics

    /// Arctan loss weight on a squared residual, Ceres' scale convention:
    /// `rho(s) = a * atan(s / a)`, so `rho'(s) = 1 / (1 + (s / a)^2)`.
    @inline(__always)
    static func arctanWeight(squared: Double, scale: Double) -> Double {
        guard squared.isFinite, scale > 0 else { return 0 }
        let ratio = squared / scale
        return 1 / (1 + ratio * ratio)
    }

    /// One 6-vector Jacobian row for a residual whose gradient with respect to
    /// a camera-frame point is `gradient`, where the point transforms as
    /// `P -> P +- (omega x P) +- v` under a left increment.
    ///
    /// `d(omega x P)/d omega = -skew(P)`, so a row that wants `+omega x P`
    /// takes `negateRotation: true` to end up with `-gradient . skew(P)`,
    /// which is `gradient x P` written out. Spelling the sign convention here
    /// once is deliberate: a sign error in a Jacobian still converges, to the
    /// wrong answer, with no symptom.
    @inline(__always)
    static func jacobianRow(
        gradient: SIMD3<Double>,
        point: SIMD3<Double>,
        negateRotation: Bool,
        negateTranslation: Bool = false
    ) -> [Double] {
        // gradient^T * (-skew(P)) == cross(P, gradient)^T
        let rotation = simd_cross(point, gradient)
        let rotationSign: Double = negateRotation ? 1 : -1
        let translationSign: Double = negateTranslation ? -1 : 1
        return [
            rotation.x * rotationSign,
            rotation.y * rotationSign,
            rotation.z * rotationSign,
            gradient.x * translationSign,
            gradient.y * translationSign,
            gradient.z * translationSign
        ]
    }

    /// Adds one residual's contribution to the normal equations.
    private static func accumulate(
        h: inout [Double],
        g: inout [Double],
        dimension: Int,
        blocks: [(slot: Int, row: [Double])],
        residual: Double,
        weight: Double
    ) {
        guard weight > 0, residual.isFinite else { return }
        for (slotA, rowA) in blocks {
            let offsetA = slotA * 6
            for i in 0..<6 {
                g[offsetA + i] += weight * rowA[i] * residual
            }
            for (slotB, rowB) in blocks {
                let offsetB = slotB * 6
                for i in 0..<6 {
                    let scaled = weight * rowA[i]
                    for l in 0..<6 {
                        h[(offsetA + i) * dimension + (offsetB + l)] += scaled * rowB[l]
                    }
                }
            }
        }
    }

    private static func addSchur(
        _ target: inout [Int: [Double]],
        slot: Int,
        row: [Double],
        jd: Double,
        weight: Double
    ) {
        guard weight > 0 else { return }
        var existing = target[slot] ?? [Double](repeating: 0, count: 6)
        for i in 0..<6 { existing[i] += weight * jd * row[i] }
        target[slot] = existing
    }

    /// Total robust cost. Recomputed from scratch, with fresh data
    /// association, so a step is only accepted when it actually helped rather
    /// than when it merely helped the linearisation.
    private static func cost(
        cameras: [PrePassSE3],
        deltas: [PrePassSE3],
        depths: [Double],
        observations: [Observation],
        keyframes: [KeyframeData],
        geometry: PrePassDepthGeometry,
        scale: Double,
        priorRotationWeight: Double,
        priorTranslationWeight: Double,
        anchorWeight: Double
    ) -> Double {
        var total = 0.0
        let width = geometry.width
        let sampleCount = geometry.width * geometry.height

        for (j, observation) in observations.enumerated() {
            let d = depths[j]
            let xHost = observation.ray * d
            let hostPose = cameras[observation.host]

            let priorResidual = (d - observation.measuredRange) / observation.priorSigma
            total += arctanCost(squared: priorResidual * priorResidual, scale: scale)

            for targetIndex in observation.targets {
                let hostToTarget = hostPose.inverse.then(cameras[targetIndex])
                let xTarget = hostToTarget.act(xHost)
                let rangeTarget = simd_length(xTarget)
                guard rangeTarget > 0.2 else { continue }
                guard let pixel = geometry.project(cameraPoint: SIMD3<Float>(xTarget))
                else { continue }
                let targetSample = Int(pixel.y) * width + Int(pixel.x)
                guard targetSample >= 0, targetSample < sampleCount else { continue }
                let measuredTarget = keyframes[targetIndex].range[targetSample]
                guard measuredTarget > 0 else { continue }
                let sigmaTarget = Double(
                    Swift.max(keyframes[targetIndex].sigma[targetSample], 1e-3)
                )

                let forward = (rangeTarget - Double(measuredTarget)) / sigmaTarget
                total += arctanCost(squared: forward * forward, scale: scale)

                let rayTarget = SIMD3<Double>(geometry.rayDirections[targetSample])
                let yTarget = rayTarget * Double(measuredTarget)
                let yHost = cameras[targetIndex].inverse.then(hostPose).act(yTarget)
                let rangeHost = simd_length(yHost)
                guard rangeHost > 0.2 else { continue }
                let backward = (rangeHost - d) / observation.depthSigma
                total += arctanCost(squared: backward * backward, scale: scale)
            }
        }

        for (slot, delta) in deltas.enumerated() {
            var residual = delta.logVector()
            let rotationWeight = slot == 0 ? anchorWeight : priorRotationWeight
            let translationWeight = slot == 0 ? anchorWeight : priorTranslationWeight
            for i in 0..<3 { residual[i] *= rotationWeight }
            for i in 3..<6 { residual[i] *= translationWeight }
            for value in residual { total += value * value }
        }

        return total
    }

    @inline(__always)
    private static func arctanCost(squared: Double, scale: Double) -> Double {
        guard squared.isFinite, scale > 0 else { return 0 }
        return scale * Foundation.atan(squared / scale)
    }
}
