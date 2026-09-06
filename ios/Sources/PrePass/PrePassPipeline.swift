//
//  PrePassPipeline.swift
//  PrePass
//
//  THE NO-TRAINING PASS BETWEEN CAPTURE AND TRAINING. `PrePassService`
//  (CONTRACTS.md section 5).
//
//  This is the single biggest quality lever in the product, and almost all of
//  that leverage is in the ORDER the stages run in.
//
//  ---------------------------------------------------------------------------
//  WHY POSES GO FIRST, ALWAYS
//  ---------------------------------------------------------------------------
//  Every later stage measures a DISAGREEMENT between two views of one surface
//  and attributes it to something. The trust field attributes it to sensor
//  noise. The edge classifier attributes it to a depth step. The carver
//  attributes it to empty air. If the poses are still wrong, every one of them
//  is attributing pose error to the wrong cause, confidently, and writing that
//  wrong attribution to disk as if it were a measurement. A 2 cm pose error
//  reads as 2 cm of sensor noise everywhere, which makes the trust field
//  distrust a perfectly good sensor, which makes the trainer ignore perfectly
//  good depth.
//
//  So: submaps, time offset, revisits, pose graph, and only then anything that
//  measures the world.
//
//  ---------------------------------------------------------------------------
//  WHAT RUNS, IN ORDER
//  ---------------------------------------------------------------------------
//   0. A fast card from poses and LiDAR alone, yielded immediately, so the
//      post-capture screen has something true on it in under three seconds
//      rather than a spinner for two minutes. Its `refinedPoses` is
//      deliberately EMPTY: nothing has been refined yet and handing back raw
//      poses under that name would be a quiet lie.
//   1. F1 TIME-SLICED SUBMAPS, 15-30 s windows with 20-30% overlap. VIO is
//      near-perfect inside a window and only drifts over a long walk, so each
//      window is treated as internally rigid and only its placement is solved.
//   2. F1 CAMERA-TO-IMU TIME OFFSET, swept -50 to +50 ms in 5 ms steps, poses
//      re-interpolated at each step (SLERP on rotation, linear on
//      translation), minimising reprojection error.
//   3. F1 REVISITS, by pose proximity and view-direction agreement, confirmed
//      by point-to-plane ICP on the NATIVE depth.
//   4. F1 SUBMAP POSE GRAPH, one rigid SE(3) per submap, robust annealed loss,
//      Levenberg-Marquardt. Never COLMAP from scratch, never
//      triangulate-then-bundle-adjust from the ARKit prior: that measurably
//      degraded the poses in 15 of 15 rooms.
//   4b. The optional stronger refinement: the LiDAR-depth-anchored,
//      TRIANGULATION-FREE bundle adjustment in `PrePassBundleAdjuster`. Gated
//      on measured improvement, so it can only help.
//   5. F2 FREE-SPACE CARVING. Every LiDAR ray of every keyframe. Passed
//      through is EMPTY, endpoint is SURFACE, past range or behind a no-return
//      stays UNKNOWN and is never marked empty.
//   6. F6 TRUST FIELDS: the coarse bias field and the never-averaged
//      per-sample noise field, plus the recalibrated ARKit confidence.
//   7. F3 EDGE CLASSIFICATION on the native map, dilated by the upsample
//      ratio, into geometric / texture / unknown / band.
//   8. F5 GLASS AND WINDOW REGIONS.
//   9. The initial Gaussian set the trainer seeds from, the refined COLMAP
//      model, the suggested budget, and the finished QC card.
//
//  ---------------------------------------------------------------------------
//  DEGRADING RATHER THAN LOSING
//  ---------------------------------------------------------------------------
//  Stages 5 to 9 each run inside their own error boundary. A carver that runs
//  out of room, or a trust build that hits a corrupt sidecar, must not throw
//  away the poses, which are the expensive part and the part everything else
//  can be rebuilt from. A stage that fails writes a plain-language finding onto
//  the QC card saying which part did not finish and what is missing because of
//  it. Nothing is silently skipped.
//
//  ---------------------------------------------------------------------------
//  WHO OWNS WHAT
//  ---------------------------------------------------------------------------
//  This file orchestrates; it does not re-implement. `SubmapPoseRefiner`,
//  `VoxelFreeSpaceCarver`, `PrePassICP`, `PrePassBundleAdjuster`,
//  `PrePassGlassDetector` and `PrePassInitialSplatBuilder` are this module's.
//  `TwoScaleTrustField` and `NativeDepthEdgeClassifier` belong to
//  `Sources/Smart` (CONTRACTS.md section 5) and are DRIVEN from here, not
//  duplicated: the pre-pass is where they have to run, because the trainer
//  needs their output before its first iteration.
//

import Foundation
import simd
import os

// MARK: - Progress

/// Which part of the pre-pass is running. Named after what the user would say
/// is happening, not after the algorithm.
public enum PrePassStage: String, Codable, Sendable, CaseIterable {
    case preparing
    case firstLook
    case submaps
    case timeOffset
    case revisits
    case poseGraph
    case fineRefinement
    case carving
    case trust
    case edges
    case glass
    case initialSplats
    case writing
    case done

    /// One plain sentence for the UI. Shown verbatim.
    public var message: String {
        switch self {
        case .preparing: return "Getting your scan ready to check over."
        case .firstLook: return "Taking a first look at your scan."
        case .submaps: return "Splitting the walk into short stretches."
        case .timeOffset: return "Lining the camera up with the motion sensors."
        case .revisits: return "Finding the places you walked back over."
        case .poseGraph: return "Straightening out where the phone was."
        case .fineRefinement: return "Fine tuning the camera positions against the laser."
        case .carving: return "Working out which air the laser flew through."
        case .trust: return "Measuring how much to believe each depth reading."
        case .edges: return "Telling real edges apart from flat patterns."
        case .glass: return "Looking for glass and windows."
        case .initialSplats: return "Placing the first set of points to train from."
        case .writing: return "Saving what was worked out."
        case .done: return "Your scan is checked over and ready."
        }
    }

    /// Roughly how far through the whole pass this stage begins. Measured
    /// against a real room scan, not divided evenly: revisit detection and the
    /// trust build genuinely dominate, and a progress bar that pretends
    /// otherwise stalls in the middle and then leaps.
    var startFraction: Double {
        switch self {
        case .preparing: return 0.00
        case .firstLook: return 0.02
        case .submaps: return 0.08
        case .timeOffset: return 0.10
        case .revisits: return 0.22
        case .poseGraph: return 0.48
        case .fineRefinement: return 0.54
        case .carving: return 0.64
        case .trust: return 0.74
        case .edges: return 0.88
        case .glass: return 0.92
        case .initialSplats: return 0.96
        case .writing: return 0.99
        case .done: return 1.00
        }
    }
}

/// One progress tick. Separate from `PrePassResult` because a partial result
/// is a thing on disk and this is only a thing on screen.
public struct PrePassProgress: Sendable {
    public var stage: PrePassStage
    /// 0...1 over the whole pass.
    public var fractionComplete: Double
    /// Plain sentence, shown as-is.
    public var message: String

    public init(stage: PrePassStage, fractionComplete: Double, message: String) {
        self.stage = stage
        self.fractionComplete = fractionComplete
        self.message = message
    }
}

// MARK: - The pipeline

/// `PrePassService`, owned by `Sources/PrePass` (CONTRACTS.md section 5).
public final class PrePassPipeline: PrePassService, @unchecked Sendable {

    /// Everything the orchestration decides, in one visible place. Each stage
    /// has its own tuning on its own type; this is only about which stages run
    /// and at what scale.
    public struct Tuning: Sendable {
        /// Occupancy voxel edge, metres. The carver raises this on its own if
        /// the scene will not fit its cell budget, and reports what it
        /// actually used.
        public var occupancyVoxelSizeMeters: Float = 0.05
        /// Run the optional LiDAR-anchored bundle adjustment after the pose
        /// graph. It is gated on measured improvement, so leaving it on cannot
        /// make the poses worse; it costs a few seconds.
        public var useLiDARAnchoredBundleAdjustment = true
        public var buildOccupancy = true
        public var buildTrustFields = true
        public var classifyEdges = true
        public var detectGlass = true
        public var buildInitialSplats = true
        /// Written into `prepass/sparse_refined/`, so any COLMAP-reading tool
        /// can be pointed straight at the folder.
        public var writeRefinedColmapModel = true

        public init() {}
    }

    // MARK: Collaborators

    public var tuning: Tuning
    /// F1. This module's.
    public let poseRefiner: SubmapPoseRefiner
    /// F2. This module's.
    public let carver: VoxelFreeSpaceCarver
    /// F6. `Sources/Smart`'s, driven from here.
    public private(set) var trustField: TwoScaleTrustField
    /// F3. `Sources/Smart`'s, driven from here. Rebuilt per run because its
    /// dilation band is derived from the capture's own resolution ratio.
    public private(set) var edgeClassifier: NativeDepthEdgeClassifier

    /// The device tier the suggested budget is sized for.
    ///
    /// `Sources/Onboarding` owns the real answer (`DeviceCompatibilityProbe`).
    /// Set this from its report when the app has one; left nil, the budget
    /// falls back to what this process can actually allocate right now, which
    /// is a weaker but honest substitute rather than an assumption.
    public var deviceTier: DeviceTier?

    /// Live stage updates. One consumer: the pre-pass screen.
    public let progress: AsyncStream<PrePassProgress>
    private let progressContinuation: AsyncStream<PrePassProgress>.Continuation

    /// The last completed result, for a caller that missed the stream.
    public private(set) var lastResult: PrePassResult?

    /// THE SPLAT CENSUS: what every stage of the last run counted about
    /// itself. Also written to `prepass/census.json`, and readable later with
    /// `PrePassCensus.read(at:)`.
    ///
    /// It is a separate object from `lastResult` because `PrePassResult` lives
    /// in `Sources/Core`, which this module does not own. Nothing outside this
    /// module had to change to make room for the census, and nothing outside
    /// this module can break by ignoring it.
    ///
    /// Set once, at the end of the pass, from counters the stages kept. See
    /// `PrePassCensus` for why this exists at all.
    public private(set) var lastCensus: PrePassCensus?

    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem, category: "PrePass"
    )

    public init(
        tuning: Tuning = Tuning(),
        poseRefiner: SubmapPoseRefiner = SubmapPoseRefiner(),
        carver: VoxelFreeSpaceCarver = VoxelFreeSpaceCarver()
    ) {
        self.tuning = tuning
        self.poseRefiner = poseRefiner
        self.carver = carver
        self.trustField = TwoScaleTrustField()
        self.edgeClassifier = NativeDepthEdgeClassifier()

        var captured: AsyncStream<PrePassProgress>.Continuation!
        self.progress = AsyncStream<PrePassProgress>(bufferingPolicy: .bufferingNewest(8)) {
            captured = $0
        }
        self.progressContinuation = captured
    }

    deinit {
        progressContinuation.finish()
    }

    // MARK: - PrePassService

    public func run(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) -> AsyncThrowingStream<PrePassResult, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.execute(bundle: bundle, at: ref) { partial in
                        continuation.yield(partial)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Poses and LiDAR only. No carving, no trust field, no image decode.
    ///
    /// The three-second promise is kept by bounding the work rather than by
    /// hoping: 48 depth sidecars (about 4.7 MB) at one native sample in
    /// sixteen. Drift comes from ARKit's own anchors, which is a free
    /// measurement that needs no alignment (F8).
    public func quickQCCard(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> QCCard {
        let survey = try PrePassSurveyor.survey(
            bundle: bundle,
            at: ref,
            poseFor: { $0.refinedPose ?? $0.rawPose }
        )
        let confirmed = bundle.revisitPairs.filter {
            $0.method == .depthICP && $0.confidence > 0.2
        }
        return PrePassQCBuilder.card(
            survey: survey,
            driftCentimeters: driftCentimeters(
                revisits: confirmed, bundle: bundle
            ),
            loopClosureCount: confirmed.count,
            glassAreaFraction: survey.apertureSampleFraction,
            totalFrameCount: bundle.frames.count,
            rejectedRevisitCandidates: 0,
            timeOffsetSeconds: bundle.cameraToIMUTimeOffsetSeconds,
            isFastPath: true
        )
    }

    // MARK: - The pass

    private func execute(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        emit: (PrePassResult) -> Void
    ) async throws {
        report(.preparing)
        let startedAt = Date()
        // The census is built up as the stages run and written ONCE at the
        // end, next to the result. Nothing in a hot loop touches it.
        var census = PrePassCensus()
        census.scanID = bundle.scanID
        census.input = inputCensus(bundle: bundle)
        // Published immediately, so a pass that dies halfway leaves THIS run's
        // input counts and a set of stages marked "not attempted", rather than
        // the previous run's numbers. A stale census would be worse than none.
        lastCensus = census

        try FileManager.default.createDirectory(
            at: ref.url(forRelativePath: PrePassPaths.directory),
            withIntermediateDirectories: true
        )
        guard !bundle.frames.isEmpty else {
            throw NimbusError.prePassFailed("this scan has no frames in it.")
        }

        // Findings contributed by stages that did not finish. Appended to the
        // final card so a missing file is always explained.
        var stageFindings: [QCFinding] = []

        // --- 0. The fast card, first, so the screen has something true on it.
        report(.firstLook)
        let quickSurvey = try PrePassSurveyor.survey(
            bundle: bundle, at: ref, poseFor: { $0.rawPose }
        )
        let capturedRevisits = bundle.revisitPairs.filter {
            $0.method == .depthICP && $0.confidence > 0.2
        }
        let quickCard = PrePassQCBuilder.card(
            survey: quickSurvey,
            driftCentimeters: driftCentimeters(revisits: capturedRevisits, bundle: bundle),
            loopClosureCount: capturedRevisits.count,
            glassAreaFraction: quickSurvey.apertureSampleFraction,
            totalFrameCount: bundle.frames.count,
            rejectedRevisitCandidates: 0,
            timeOffsetSeconds: bundle.cameraToIMUTimeOffsetSeconds,
            isFastPath: true
        )
        emit(
            PrePassResult(
                scanID: bundle.scanID,
                completedAt: Date(),
                submaps: [],
                // Deliberately empty: nothing has been refined yet, and a
                // reader that finds no entry correctly falls back to the raw
                // pose rather than trusting a relabelled one.
                refinedPoses: [:],
                cameraToIMUTimeOffsetSeconds: bundle.cameraToIMUTimeOffsetSeconds,
                occupancy: nil,
                trust: nil,
                edges: nil,
                glassRegions: [],
                initialSplats: nil,
                qcCard: quickCard,
                suggestedBudget: nil
            )
        )

        // --- 1. Submaps.
        try Task.checkCancellation()
        report(.submaps)
        var submaps = poseRefiner.buildSubmaps(bundle: bundle)

        // --- 2. Camera-to-IMU time offset.
        try Task.checkCancellation()
        report(.timeOffset)
        var timeOffset: Double? = nil
        do {
            timeOffset = try await poseRefiner.calibrateTimeOffset(bundle: bundle, at: ref)
        } catch is CancellationError {
            throw NimbusError.cancelled
        } catch {
            timeOffset = nil
            stageFindings.append(
                QCFinding(
                    code: "time_offset_failed",
                    severity: .warning,
                    message: "The camera and motion sensor timing could not be worked out, "
                        + "so it was left exactly as recorded.",
                    fixHint: nil
                )
            )
        }
        census.timeOffset = poseRefiner.lastTimeOffsetCensus
        if timeOffset == nil, let carried = bundle.cameraToIMUTimeOffsetSeconds {
            // The capture measured one and the pre-pass did not. Later stages
            // read it off the bundle, so the census has to say so, otherwise
            // it reports "none" while the pass quietly uses a number.
            census.timeOffset.settledSeconds = carried
            census.timeOffset.source = "carriedFromCapture"
        }

        // --- 3. Revisits.
        try Task.checkCancellation()
        report(.revisits)
        var revisits: [RevisitPair] = []
        var revisitsCameFromCapture = false
        do {
            revisits = try await poseRefiner.detectRevisits(
                bundle: bundle, submaps: submaps, at: ref
            )
        } catch is CancellationError {
            throw NimbusError.cancelled
        } catch {
            revisitsCameFromCapture = true
            // Fall back to whatever the capture itself logged. Losing loop
            // closures costs accuracy; it must not cost the whole pass.
            revisits = bundle.revisitPairs
            stageFindings.append(
                QCFinding(
                    code: "revisits_failed",
                    severity: .warning,
                    message: "The check for places you walked back over could not finish, so "
                        + "the phone's own tracking was used as it was.",
                    fixHint: nil
                )
            )
        }
        if revisits.isEmpty, !bundle.revisitPairs.isEmpty {
            revisits = bundle.revisitPairs
            revisitsCameFromCapture = true
        }
        census.revisits = poseRefiner.lastRevisitCensus
        census.revisits.usedCaptureFallback = revisitsCameFromCapture
        if revisitsCameFromCapture {
            // The counts from the detector describe a run whose answer was
            // then thrown away. What the rest of the pass actually got is the
            // capture's own list, so that is what the census reports.
            census.revisits.pairsReturned = revisits.count
            let fallbackConfirmed = revisits.filter {
                $0.method == .depthICP && $0.confidence > 0.2
            }
            census.revisits.confirmedPairs = fallbackConfirmed.count
            if !fallbackConfirmed.isEmpty {
                census.revisits.medianConfidence = PrePassStats.median(
                    fallbackConfirmed.map { $0.confidence }
                )
                census.revisits.medianTranslationResidualCentimeters = PrePassStats.median(
                    fallbackConfirmed.map { $0.translationResidualMeters * 100 }
                )
            }
        }

        // --- 4. The submap pose graph.
        try Task.checkCancellation()
        report(.poseGraph)
        var refinedPoses = try await poseRefiner.optimize(
            bundle: bundle,
            submaps: submaps,
            revisits: revisits,
            timeOffsetSeconds: timeOffset
        )
        census.poseGraph = poseRefiner.lastPoseGraphCensus

        // --- 4b. Optional LiDAR-anchored, triangulation-free refinement.
        //
        // The trust field does not exist yet, and it must not: it is measured
        // FROM the poses. So every sample's depth prior here is the physics
        // prior alone, weighted equally. That is the honest ordering, and it
        // is why this stage is gated on an independent improvement check
        // rather than trusted on principle.
        if tuning.useLiDARAnchoredBundleAdjustment {
            try Task.checkCancellation()
            report(.fineRefinement)
            census.poseGraph.fineRefinementRan = true
            do {
                let adjusted = try PrePassBundleAdjuster.refine(
                    bundle: bundle,
                    at: ref,
                    poses: refinedPoses,
                    revisits: revisits,
                    trustWeight: { _, _ in 1 }
                )
                if adjusted.accepted { refinedPoses = adjusted.poses }
                census.poseGraph.fineRefinementAccepted = adjusted.accepted
                census.poseGraph.fineRefinementNote = adjusted.note
                log.info("Fine refinement: \(adjusted.note, privacy: .public)")
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                census.poseGraph.fineRefinementNote = "did not finish"
                stageFindings.append(
                    QCFinding(
                        code: "fine_refinement_failed",
                        severity: .warning,
                        message: "The extra fine tuning of the camera positions did not "
                            + "finish, so the straightened positions were kept as they were.",
                        fixHint: nil
                    )
                )
            }
        }

        // --- Fold the refined poses back into a bundle, so every later stage
        //     (all of which read `CaptureFrame.refinedPose`) sees them.
        let owners = PrePassSubmapAssignment.owners(
            frames: bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds },
            submaps: submaps
        )
        var refinedBundle = bundle
        refinedBundle.cameraToIMUTimeOffsetSeconds = timeOffset
            ?? bundle.cameraToIMUTimeOffsetSeconds
        refinedBundle.frames = bundle.frames.map { frame in
            var copy = frame
            copy.refinedPose = refinedPoses[String(frame.index)] ?? frame.rawPose
            copy.submap = owners[frame.index]
            return copy
        }
        refinedBundle.revisitPairs = revisits

        // The rigid correction each submap actually received, recovered from
        // the frame nearest its window centre. It includes the time-offset
        // re-interpolation, because the two are applied together and pulling
        // them apart afterwards would report a number that was never used.
        submaps = fillCorrections(
            submaps: submaps, bundle: refinedBundle, refinedPoses: refinedPoses
        )

        // --- A second survey, now on poses that are actually right. Every
        //     coverage number is worth more here than it was on the fast card.
        try Task.checkCancellation()
        let survey = try PrePassSurveyor.survey(
            bundle: refinedBundle, at: ref,
            poseFor: { $0.refinedPose ?? $0.rawPose }
        )
        let confirmedRevisits = revisits.filter {
            $0.method == .depthICP && $0.confidence > 0.2
        }
        let measuredDrift = driftCentimeters(revisits: confirmedRevisits, bundle: bundle)

        // Hand the carver the MEASURED extent of the scene when the capture
        // did not record one. Without it the carver has to fall back to the
        // camera path grown by the sensor's full reach in every direction,
        // which for a single room is roughly a thousand times the real volume,
        // and it would then coarsen the voxel size to fit that phantom volume
        // into its cell budget. The camera path is unioned in because the
        // carver's rays start at the camera, and a cell outside the grid is
        // simply not carved.
        if refinedBundle.sceneBounds == nil, let surfaceBounds = survey.surfaceBounds {
            var minimum = surfaceBounds.min.simd
            var maximum = surfaceBounds.max.simd
            for frame in refinedBundle.frames {
                let centre = (frame.refinedPose ?? frame.rawPose).center.simd
                minimum = simd_min(minimum, centre)
                maximum = simd_max(maximum, centre)
            }
            refinedBundle.sceneBounds = BoundingBox(
                min: Vector3(minimum), max: Vector3(maximum)
            )
        }

        func card(glassFraction: Float) -> QCCard {
            var built = PrePassQCBuilder.card(
                survey: survey,
                driftCentimeters: measuredDrift,
                loopClosureCount: confirmedRevisits.count,
                glassAreaFraction: glassFraction,
                totalFrameCount: bundle.frames.count,
                rejectedRevisitCandidates: poseRefiner.rejectedRevisitCandidates,
                timeOffsetSeconds: timeOffset,
                isFastPath: false
            )
            built.findings.append(contentsOf: stageFindings)
            return built
        }

        var result = PrePassResult(
            scanID: bundle.scanID,
            completedAt: Date(),
            submaps: submaps,
            refinedPoses: refinedPoses,
            cameraToIMUTimeOffsetSeconds: timeOffset,
            occupancy: nil,
            trust: nil,
            edges: nil,
            glassRegions: [],
            initialSplats: nil,
            qcCard: card(glassFraction: survey.apertureSampleFraction),
            suggestedBudget: nil
        )

        // The refined COLMAP model's poses are worth writing now: they are
        // finished, and a crash in a later stage should not cost them.
        if tuning.writeRefinedColmapModel {
            do {
                try writeRefinedModel(
                    bundle: refinedBundle, poses: refinedPoses, at: ref
                )
            } catch {
                stageFindings.append(
                    QCFinding(
                        code: "refined_model_write_failed",
                        severity: .warning,
                        message: "The corrected camera positions could not be saved in the "
                            + "shareable format, though they are still in the scan itself.",
                        fixHint: nil
                    )
                )
            }
        }
        emit(result)

        // --- 5. F2 free-space carving.
        if tuning.buildOccupancy {
            try Task.checkCancellation()
            report(.carving)
            do {
                let grid = try await carver.carve(
                    bundle: refinedBundle,
                    at: ref,
                    voxelSizeMeters: tuning.occupancyVoxelSizeMeters
                )
                result.occupancy = grid
                if carver.hitCellCap {
                    stageFindings.append(
                        QCFinding(
                            code: "occupancy_capped",
                            severity: .warning,
                            message: "This scan covers a lot of ground, so the map of empty "
                                + "air was built at a coarser scale to fit in memory.",
                            fixHint: "Scanning one room at a time keeps the finest detail."
                        )
                    )
                }
                result.qcCard = card(glassFraction: survey.apertureSampleFraction)
                emit(result)
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                stageFindings.append(failureFinding(
                    code: "carving_failed",
                    what: "working out which air the laser flew through",
                    consequence: "Floating specks will not be cleaned up automatically."
                ))
            }
            // Read on both paths. The carver resets its counters at the start
            // of every carve, so even a carve that threw halfway reports how
            // far it got rather than the previous run's numbers.
            census.carving = carver.lastCensus
        }

        // --- 6. F6 trust fields.
        var trustLoaded = false
        if tuning.buildTrustFields {
            try Task.checkCancellation()
            report(.trust)
            do {
                let refs = try await trustField.build(
                    bundle: refinedBundle, prePassPoses: refinedPoses, at: ref
                )
                result.trust = refs
                // Loaded straight back so the initial splat set can be shaped
                // by it. Re-reading a file we just wrote is cheap next to
                // re-deriving what is in it.
                try await trustField.load(refs, at: ref)
                trustLoaded = true
                result.qcCard = card(glassFraction: survey.apertureSampleFraction)
                emit(result)
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                stageFindings.append(failureFinding(
                    code: "trust_failed",
                    what: "measuring how much to believe each depth reading",
                    consequence: "Every measurement will be treated with the same caution, "
                        + "which is safe but loses some sharpness."
                ))
            }
        }

        // --- 7. F3 edge classification.
        var edgesLoaded = false
        if tuning.classifyEdges {
            try Task.checkCancellation()
            report(.edges)
            do {
                edgeClassifier = NativeDepthEdgeClassifier(
                    settings: edgeSettings(for: bundle)
                )
                let refs = try await edgeClassifier.classify(
                    bundle: refinedBundle, at: ref
                )
                result.edges = refs
                edgesLoaded = true
                result.qcCard = card(glassFraction: survey.apertureSampleFraction)
                emit(result)
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                stageFindings.append(failureFinding(
                    code: "edges_failed",
                    what: "telling real edges apart from flat patterns",
                    consequence: "Patterned surfaces like rugs and posters may come out with "
                        + "bumps that are not really there."
                ))
            }
        }

        // --- 8. F5 glass and windows.
        var glassFraction = survey.apertureSampleFraction
        if tuning.detectGlass {
            try Task.checkCancellation()
            report(.glass)
            do {
                let glass = try PrePassGlassDetector.detect(
                    bundle: refinedBundle,
                    at: ref,
                    poseFor: { $0.refinedPose ?? $0.rawPose }
                )
                result.glassRegions = glass.regions
                glassFraction = glass.areaFraction
                result.qcCard = card(glassFraction: glassFraction)
                emit(result)
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                stageFindings.append(failureFinding(
                    code: "glass_failed",
                    what: "looking for glass and windows",
                    consequence: "Anything seen through a window may be treated as if it were "
                        + "solid."
                ))
            }
        }

        // --- 9. The initial Gaussian set, the point cloud, and the budget.
        let extent = (survey.surfaceBounds ?? bundle.sceneBounds)?.longestEdgeMeters ?? 5
        let budget = suggestedBudget(sceneExtentMeters: extent)
        result.suggestedBudget = budget

        if tuning.buildInitialSplats {
            try Task.checkCancellation()
            report(.initialSplats)
            // Filled by the builder as it goes, and kept even when the builder
            // throws: the run where no seed survived is the run whose funnel
            // matters most, and a thrown error must not take it with it.
            var seedingCensus = PrePassCensus.Seeding()
            seedingCensus.trustFieldLoaded = trustLoaded
            seedingCensus.edgeMapsLoaded = edgesLoaded
            do {
                let area = Float(survey.surfaceCellCount)
                    * survey.voxelSizeMeters * survey.voxelSizeMeters
                // Bound locally so the two non-escaping closures below do not
                // have to reach through `self` on every one of fifty thousand
                // samples.
                let trust = trustField
                let classifier = edgeClassifier
                let splats = try PrePassInitialSplatBuilder.build(
                    bundle: refinedBundle,
                    at: ref,
                    poseFor: { $0.refinedPose ?? $0.rawPose },
                    trustWeight: { frame, sample in
                        trustLoaded ? trust.weight(frame: frame, sampleIndex: sample) : 0
                    },
                    sigmaFor: { frame, sample in
                        trustLoaded ? trust.sigmaMeters(frame: frame, sampleIndex: sample) : nil
                    },
                    edgeMapFor: { frame in
                        edgesLoaded ? classifier.map(for: frame) : []
                    },
                    surfaceAreaSquareMeters: area,
                    targetSplatCount: budget.splatCap,
                    census: &seedingCensus
                )
                result.initialSplats = splats.ref

                if tuning.writeRefinedColmapModel {
                    try PrePassColmapWriter.write(
                        text: PrePassColmapWriter.pointsText(
                            positions: splats.positions,
                            colors: splats.colors,
                            expectedErrorMeters: splats.expectedErrorMeters
                        ),
                        to: ref.url(forRelativePath: PrePassPaths.refinedPoints)
                    )
                }

                if !trustLoaded {
                    stageFindings.append(
                        QCFinding(
                            code: "splats_untrusted",
                            severity: .warning,
                            message: "The starting points were placed without the depth "
                                + "reliability measurements, so they all start out cautious.",
                            fixHint: nil
                        )
                    )
                }
                let summary = "Initial splats: \(splats.ref.splatCount) at "
                    + "\(splats.spacingMeters) m spacing, \(splats.trustedCount) pinned, "
                    + "\(splats.doubtfulCount) free to slide, \(splats.edgeCount) on edges."
                log.info("\(summary, privacy: .public)")
            } catch is CancellationError {
                throw NimbusError.cancelled
            } catch {
                stageFindings.append(failureFinding(
                    code: "initial_splats_failed",
                    what: "placing the first set of points to train from",
                    consequence: "Training will have to work the shape out from scratch, "
                        + "which takes longer and comes out softer."
                ))
            }
            census.seeding = seedingCensus
        }

        // --- The census, finished and turned into findings.
        //
        // The alarms go onto the QC card the owner already looks at, so a
        // stage that produced nothing says so on the screen he opens after
        // every scan, rather than waiting to be found in a JSON file.
        census.recordedAt = Date()
        census.durationSeconds = Date().timeIntervalSince(startedAt)
        // Two alarms would repeat a finding this pipeline already adds in
        // plainer words. One card, one sentence per problem.
        let alreadySaid: [String: String] = [
            "census_carving_capped": "occupancy_capped",
            "census_no_trust_field": "splats_untrusted"
        ]
        let existingCodes = Set(stageFindings.map { $0.code })
        let alarms = census.alarms
        for alarm in alarms {
            if let duplicate = alreadySaid[alarm.code], existingCodes.contains(duplicate) {
                continue
            }
            stageFindings.append(
                QCFinding(
                    code: alarm.code,
                    severity: .problem,
                    message: alarm.message,
                    fixHint: alarm.fixHint
                )
            )
        }
        lastCensus = census
        log.info("Census: \(census.headline, privacy: .public)")
        for alarm in alarms {
            // Built as a plain String first: an os.Logger message is a literal
            // with its own interpolation type and two of them cannot be joined
            // with `+`.
            let line = "Census alarm \(alarm.code): \(alarm.message)"
            log.error("\(line, privacy: .public)")
        }
        do {
            try census.write(at: ref)
        } catch {
            // A report about the work must never be able to fail the work.
            let reason = error.localizedDescription
            log.error("The census could not be written: \(reason, privacy: .public)")
        }

        // --- Write the index and finish.
        try Task.checkCancellation()
        report(.writing)
        result.completedAt = Date()
        result.qcCard = card(glassFraction: glassFraction)
        do {
            let data = try ContractsJSON.encoder().encode(result)
            try PrePassBinary.write(data, to: ref.url(forRelativePath: PrePassPaths.result))
        } catch {
            throw NimbusError.prePassFailed(
                "the results of the check could not be saved: \(error.localizedDescription)"
            )
        }

        lastResult = result
        emit(result)
        report(.done)
    }

    // MARK: - Pieces

    /// What arrived from capture, counted here rather than in `Sources/Capture`.
    ///
    /// These are the numbers THE PRE-PASS SAW, which is the only version that
    /// can explain the pre-pass's own output. A frame the capture wrote whose
    /// depth sidecar never landed on disk is invisible in a capture-side count
    /// and obvious here, and the difference between those two counts is
    /// exactly the kind of thing that used to take a day to find.
    private func inputCensus(bundle: CaptureBundle) -> PrePassCensus.Input {
        var input = PrePassCensus.Input()
        input.framesInBundle = bundle.frames.count
        for frame in bundle.frames {
            if frame.depthPath != nil { input.framesWithDepthPath += 1 }
            if frame.confidencePath != nil { input.framesWithConfidencePath += 1 }
            switch frame.qc.trackingQuality {
            case .notAvailable: input.framesTrackingNotAvailable += 1
            case .normal: input.framesTrackingNormal += 1
            default: break
            }
        }
        // From the extremes rather than from the first and last elements: the
        // array is not guaranteed to be in time order and a negative duration
        // would be a lie rather than a measurement.
        var earliest = Double.greatestFiniteMagnitude
        var latest = -Double.greatestFiniteMagnitude
        for frame in bundle.frames {
            earliest = Swift.min(earliest, frame.timestampSeconds)
            latest = Swift.max(latest, frame.timestampSeconds)
        }
        let span = latest - earliest
        input.captureDurationSeconds = span.isFinite && span > 0 ? span : 0
        input.depthWidth = bundle.settings.depthWidth
        input.depthHeight = bundle.settings.depthHeight
        input.lidarMaxRangeMeters = bundle.settings.lidarMaxRangeMeters
        input.depthSamplesAvailable = input.framesWithDepthPath
            * Swift.max(input.depthWidth, 0) * Swift.max(input.depthHeight, 0)
        return input
    }

    /// Drift, in centimetres, preferring the measured revisit residual and
    /// falling back to the free anchor difference (F8). `nil` when neither
    /// exists, which the card turns into an explicit "could not tell" rather
    /// than a zero.
    private func driftCentimeters(
        revisits: [RevisitPair],
        bundle: CaptureBundle
    ) -> Float? {
        let residuals = revisits.map { $0.translationResidualMeters * 100 }
        if !residuals.isEmpty {
            let median = PrePassStats.median(residuals)
            if median.isFinite { return median }
        }
        return PrePassAnchorDrift.medianCentimeters(bundle: bundle)
    }

    /// F3's dilation band, derived rather than fixed.
    ///
    /// The spec asks for roughly an 8 pixel band at 1920 wide. That band is
    /// expressed at RGB resolution, and the maps are written at NATIVE depth
    /// resolution, so the radius in native pixels is 8 divided by the upsample
    /// ratio. At 1920 against 256 the ratio is 7.5 and the answer is 1, which
    /// is why the default happens to be 1; at any other capture resolution it
    /// is not, and assuming it would put the band in the wrong place.
    private func edgeSettings(for bundle: CaptureBundle) -> SmartLossSettings {
        var settings = SmartLossSettings.default
        let nativeWidth = Swift.max(bundle.settings.depthWidth, 1)
        let rgbWidth = Swift.max(bundle.intrinsics.width, 1)
        let upsampleRatio = Float(rgbWidth) / Float(nativeWidth)
        let bandRGBPixels: Float = 8
        let radius = Int((bandRGBPixels / Swift.max(upsampleRatio, 1)).rounded())
        settings.edgeBandRadiusNativePixels = Swift.max(1, radius)
        return settings
    }

    /// Recovers what the pose graph actually did to each submap.
    ///
    /// `Submap.correction` is defined as the world-side rigid transform with
    /// `refined = raw * correction`, so `correction = raw^-1 * refined`, read
    /// off the frame nearest the window's centre - the frame furthest from an
    /// edge, where the submap's own rigidity assumption is strongest.
    private func fillCorrections(
        submaps: [Submap],
        bundle: CaptureBundle,
        refinedPoses: [String: Pose]
    ) -> [Submap] {
        guard !submaps.isEmpty else { return submaps }
        var framesByIndex: [FrameID: CaptureFrame] = [:]
        framesByIndex.reserveCapacity(bundle.frames.count)
        for frame in bundle.frames { framesByIndex[frame.index] = frame }

        return submaps.map { submap in
            var updated = submap
            let centre = 0.5 * (submap.startTimeSeconds + submap.endTimeSeconds)
            var best: CaptureFrame?
            var bestDistance = Double.greatestFiniteMagnitude
            var index = submap.firstFrame
            while index <= submap.lastFrame {
                if let frame = framesByIndex[index] {
                    let distance = abs(frame.timestampSeconds - centre)
                    if distance < bestDistance {
                        bestDistance = distance
                        best = frame
                    }
                }
                if index == FrameID.max { break }
                index += 1
            }
            guard let frame = best,
                  let refined = refinedPoses[String(frame.index)]
            else { return updated }
            // correction = raw^-1 * refined, which in "apply left first"
            // spelling is refined.then(raw.inverse).
            let raw = PrePassSE3(frame.rawPose)
            updated.correction = PrePassSE3(refined).then(raw.inverse).pose
            return updated
        }
    }

    /// Writes `prepass/sparse_refined/{cameras,images}.txt`. `points3D.txt` is
    /// written later, from the same cloud the initial splats came from, so the
    /// two can never disagree.
    private func writeRefinedModel(
        bundle: CaptureBundle,
        poses: [String: Pose],
        at ref: CaptureBundleRef
    ) throws {
        try PrePassColmapWriter.write(
            text: PrePassColmapWriter.camerasText(bundle.intrinsics),
            to: ref.url(forRelativePath: PrePassPaths.refinedCameras)
        )
        try PrePassColmapWriter.write(
            text: PrePassColmapWriter.imagesText(
                frames: bundle.frames.sorted { $0.index < $1.index },
                poses: poses
            ),
            to: ref.url(forRelativePath: PrePassPaths.refinedImages)
        )
    }

    /// A starting budget for this scene on this device.
    ///
    /// The tier comes from `Sources/Onboarding` when the app set it. Without
    /// it, the fallback is what this process can actually allocate right now,
    /// which is a real measurement rather than a guess about the model name.
    private func suggestedBudget(sceneExtentMeters: Float) -> TrainingBudget {
        var available: UInt64 = 0
        #if canImport(os) && os(iOS)
        let reported = os_proc_available_memory()
        if reported > 0 { available = UInt64(reported) }
        #endif
        var tier = deviceTier
        if available == 0 {
            // Outside a normal app process (a preview, a test). A quarter of
            // physical memory is a deliberately conservative stand-in.
            available = ProcessInfo.processInfo.physicalMemory / 4
            if tier == nil { tier = .limited }
        }
        if tier == nil {
            // 1.5 GB is roughly where a 300k-splat field plus its optimiser
            // state and the frame cache stop fitting.
            tier = available >= 1_500_000_000 ? .full : .limited
        }
        return TrainingBudget.recommended(
            for: tier ?? .limited,
            sceneExtentMeters: sceneExtentMeters,
            availableMemoryBytes: available
        )
    }

    private func failureFinding(
        code: String,
        what: String,
        consequence: String
    ) -> QCFinding {
        QCFinding(
            code: code,
            severity: .warning,
            message: "The step for \(what) did not finish. \(consequence)",
            fixHint: "Everything else in this scan is still usable. Running the check "
                + "again, with the phone cool and plugged in, usually gets through it."
        )
    }

    private func report(_ stage: PrePassStage) {
        progressContinuation.yield(
            PrePassProgress(
                stage: stage,
                fractionComplete: stage.startFraction,
                message: stage.message
            )
        )
    }
}
