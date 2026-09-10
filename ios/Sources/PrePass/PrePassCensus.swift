//
//  PrePassCensus.swift
//  PrePass
//
//  WHERE THE GEOMETRY WENT. THE PRE-PASS COUNTING ITSELF.
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE EXISTS
//  ---------------------------------------------------------------------------
//  The first real scan came out looking like nothing, and finding out why took
//  a day of reading code, because the app records NOTHING about its own
//  behaviour. Four separate bugs had each destroyed most of the model, and not
//  one of them logged, warned, or failed:
//
//   * a densification threshold in the wrong units, so densification made
//     exactly zero splats on every run that has ever happened;
//   * a heat ratchet that cut the splat cap below the live count, deleted real
//     Gaussians, and then disabled densification for the rest of the run;
//   * two pruning fractions that were declared, defaulted, assigned and read
//     by nothing, so pruning ran ungated;
//   * a trust gate at 0.5 that needed a measured sigma under about 2 cm, when
//     2 to 4 cm is ordinary on a handheld scan, so it rejected essentially
//     every sample and every seed came out a stretched blob.
//
//  Every one of those is the same shape: A STAGE PRODUCED NOTHING AND SAID
//  NOTHING. The defence is not more careful code. It is a count. If the app
//  had been able to say "7,840,000 depth samples in, 96,300 seeds out, 0 of
//  them trusted, cut 0.50, median weight 0.04", all four would have been one
//  glance instead of a day.
//
//  So this file is the pre-pass counting itself, stage by stage, at every
//  place where a population shrinks.
//
//  ---------------------------------------------------------------------------
//  THE RULES IT FOLLOWS
//  ---------------------------------------------------------------------------
//  1. ACCUMULATE, WRITE ONCE. Every counter is a plain `Int` on the stack
//     inside the stage that owns it. Nothing here is called from a hot loop:
//     the loops increment locals, the stage hands over one small struct when
//     it finishes, and the pipeline writes ONE file at the end. There is no
//     per-sample logging, no lock, no allocation, and no measurable cost.
//
//  2. A ZERO IS RECORDED, NEVER ABSENT. Every field has a value on every path,
//     including the paths that gave up early. A stage that found nothing
//     writes a zero, and the zero is what this whole file exists to surface.
//     An absent number reads as "not measured"; a zero reads as "measured, and
//     it was nothing", and those are completely different facts.
//
//  3. EVERY SURVIVOR COUNT CARRIES ITS DENOMINATOR. "96,300 seeds" means
//     nothing. "96,300 of 7,840,000 inspected, 1.2 percent" is a bug report.
//
//  4. EVERY THRESHOLD IS RECORDED NEXT TO THE DISTRIBUTION IT CUT. A threshold
//     is only misplaced relative to real data, so the cut value and the median
//     of what it cut are stored side by side. `cut 0.50, median 0.04` is
//     instantly a misplaced threshold. `cut 0.50` on its own is not.
//
//  ---------------------------------------------------------------------------
//  WHERE IT GOES
//  ---------------------------------------------------------------------------
//  `prepass/census.json`, next to `prepass_result.json`, and on the pipeline as
//  `PrePassPipeline.lastCensus`. It is deliberately NOT a field of
//  `PrePassResult`: that type lives in `Sources/Core`, which this module does
//  not own, so the census is a separate file with a separate reader
//  (`PrePassCensus.read(at:)`) and nothing outside this module had to change to
//  make room for it.
//
//  For a screen, use `lines` (every number, in stage order, each already
//  formatted and flagged) and `alarms` (the ones that mean a stage produced
//  nothing). Neither needs a reader that understands the pipeline.
//

import Foundation
import simd

// MARK: - One rendered row

/// One number, ready to put on screen. A UI can render the whole census by
/// walking `PrePassCensus.lines` and never has to know what a submap is.
public struct PrePassCensusLine: Codable, Sendable, Identifiable {
    public var id: String { key }
    /// Stable machine name, e.g. `seeding.trusted`.
    public var key: String
    /// Which stage this belongs to, plain language.
    public var stage: String
    /// What the number means, plain language.
    public var label: String
    /// The number, already formatted.
    public var value: String
    /// The denominator, the threshold, or the unit. Nil when the value speaks
    /// for itself.
    public var detail: String?
    /// True when this number means a stage destroyed or produced nothing.
    /// A UI should make these impossible to miss.
    public var isAlarm: Bool

    public init(
        key: String,
        stage: String,
        label: String,
        value: String,
        detail: String? = nil,
        isAlarm: Bool = false
    ) {
        self.key = key
        self.stage = stage
        self.label = label
        self.value = value
        self.detail = detail
        self.isAlarm = isAlarm
    }
}

/// A stage that produced nothing, in one sentence, with the number in it.
public struct PrePassCensusAlarm: Codable, Sendable, Identifiable {
    public var id: String { code }
    /// Stable machine name, also used as the QC finding code.
    public var code: String
    /// One plain sentence with the measured number in it.
    public var message: String
    /// What to do, when there is anything to do.
    public var fixHint: String?

    public init(code: String, message: String, fixHint: String? = nil) {
        self.code = code
        self.message = message
        self.fixHint = fixHint
    }
}

// MARK: - The census

/// Every population count the pre-pass can measure about itself, in the order
/// the stages run.
public struct PrePassCensus: Codable, Sendable {

    public static let currentFormatVersion = 1

    public var formatVersion: Int = PrePassCensus.currentFormatVersion
    public var scanID: ScanID = ""
    public var recordedAt: Date = Date()
    /// Wall-clock seconds the whole pass took. 0 until the pipeline fills it.
    public var durationSeconds: Double = 0

    /// WHERE THOSE SECONDS WENT, per stage.
    ///
    /// The pre-pass measured 21.89 s on the owner's scan while training had
    /// been ground from 247 s to 77 s, so it is now 22% of the wall clock and
    /// has never been looked at. `durationSeconds` alone cannot say which of
    /// five stages to look at, and guessing wasted a build on the trainer
    /// earlier: every real win there came after the clocks went in, and none
    /// of the changes made before them did anything.
    ///
    /// Seconds, cumulative over the pass. They do not have to sum to
    /// `durationSeconds`: bundle loading and the writes at the end are outside
    /// all of them, and what is left over after subtracting them is itself the
    /// answer to "is the I/O the problem".
    public struct Stages: Codable, Sendable, Equatable {
        /// Sweeping the camera-to-IMU time offset.
        public var timeOffset: Double = 0
        /// Finding and confirming revisited surfaces, ICP included.
        public var revisits: Double = 0
        /// The pose graph, plus the fine refinement that follows it.
        public var poseGraph: Double = 0
        /// Free-space carving. 10.7 million rays over 868 keyframes on the
        /// owner's scan, every one of which needs a depth map loaded.
        public var carving: Double = 0
        /// The depth trust fields. Until these three clocks existed, trust,
        /// edges and glass all ran between the carving mark and the seeding
        /// mark, so all three were billed as "seeding" (12.7 s on build 250).
        public var trust: Double = 0
        /// Edge classification. Lazy since this build, so it should read near
        /// zero; each map's cost now lands wherever it is first asked for.
        public var edges: Double = 0
        /// Glass and window detection.
        public var glass: Double = 0
        /// Building the seed Gaussians from 174 keyframes.
        public var seeding: Double = 0

        public init() {}
    }

    public var stages = Stages()

    public var input = Input()
    public var timeOffset = TimeOffset()
    public var revisits = Revisits()
    public var poseGraph = PoseGraph()
    public var carving = Carving()
    public var seeding = Seeding()

    // MARK: The keys the Viewer's census screen reads
    //
    // `Sources/Viewer` shows one splat census for the whole scan, stitched
    // from a pre-pass sidecar and a trainer sidecar. Its reader is
    // `ScanCensus.Record` in `ios/Sources/Viewer/ScanCensus.swift`, and
    // INTEGRATION_REQUESTS.md asks this module for six keys at the TOP LEVEL
    // of `prepass/census.json`, in camelCase. Four of the six are written
    // below; the note after them says why the other two are not. Its rules,
    // followed here to the letter:
    //
    //   * every key optional, and an ABSENT key means "not counted", which it
    //     renders as "not recorded";
    //   * a key written as 0 is a measured zero, a much stronger claim, so a
    //     number that was not genuinely counted is left absent rather than
    //     written as zero;
    //   * extra keys are ignored, which is why the rest of this file can be as
    //     detailed as it likes without breaking that reader.
    //
    // They duplicate numbers that also live in `seeding`. That is deliberate:
    // one writer fills both from the same source in `fillSharedKeys()`, so
    // they cannot drift, and the other module never has to learn this file's
    // shape.

    public var writtenBy: String? = "prepass"
    public var seedsWritten: Int?
    public var seedsShapedAsConfidentDiscs: Int?
    /// The sigma the trust line works out to, metres. See
    /// `Seeding.trustGateEquivalentSigmaMeters`.
    public var trustGateSigmaMeters: Float?
    /// The sigma the scan actually measured, metres. Written ONLY when the
    /// trust field really measured it; the physics prior is a prediction and
    /// is never reported under this name.
    public var measuredMedianSigmaMeters: Float?

    // `depthSamplesOffered` and `depthSamplesAccepted` are the two keys that
    // reader also asks for, and they are DELIBERATELY NOT WRITTEN. They are
    // defined there as the counts either side of a trust gate that rejects
    // depth readings, and this pipeline has no such gate: trust does not throw
    // a reading away, it decides the SHAPE the point gets, a disc or a smear.
    // The nearest honest numbers are the funnel in `seeding` below
    // (`samplesInspected` -> `samplesWithReturn` -> `samplesInRange` ->
    // `samplesInsideGrid` -> `gaussiansBuilt`), which is in this same file and
    // free for that reader to pick up whenever it wants it.
    //
    // Writing the funnel's numbers under those two names instead would put a
    // true number under a false label, and the reader draws a causal sentence
    // from the pair. A census that explains a scan wrongly is worse than one
    // that says "not counted", so it says "not counted".

    public init() {}

    /// Fills the shared keys from the sections above. Called once, by
    /// `write(at:)`, so the two views of the same number are written by one
    /// piece of code and cannot disagree.
    ///
    /// A stage that did not run leaves its keys absent rather than zero, which
    /// is the difference between "nothing happened" and "we did not look".
    public mutating func fillSharedKeys() {
        writtenBy = "prepass"
        guard seeding.attempted else { return }
        seedsWritten = seeding.splatsWritten
        seedsShapedAsConfidentDiscs = seeding.trustedCount
        if seeding.trustGateEquivalentSigmaMeters > 0 {
            trustGateSigmaMeters = seeding.trustGateEquivalentSigmaMeters
        }
        if seeding.sigmaIsMeasured, seeding.medianSigmaMeters > 0 {
            measuredMedianSigmaMeters = seeding.medianSigmaMeters
        }
    }

    // MARK: Input

    /// What arrived from capture. Counted here rather than in `Sources/Capture`
    /// because these are the numbers the PRE-PASS actually saw, which is the
    /// only version that can explain the pre-pass's own output. A frame the
    /// capture wrote but whose depth sidecar never landed is invisible in a
    /// capture-side count and obvious here.
    public struct Input: Codable, Sendable {
        public var framesInBundle = 0
        public var framesWithDepthPath = 0
        public var framesWithConfidencePath = 0
        public var framesTrackingNotAvailable = 0
        public var framesTrackingNormal = 0
        public var captureDurationSeconds: Double = 0
        public var depthWidth = 0
        public var depthHeight = 0
        public var lidarMaxRangeMeters: Float = 0
        /// Depth samples this capture could in principle offer: frames with a
        /// depth path times the native map size. The ceiling every later
        /// survivor count is measured against.
        public var depthSamplesAvailable = 0

        public init() {}
    }

    // MARK: Time offset

    public struct TimeOffset: Codable, Sendable {
        /// False when the pipeline skipped the stage entirely.
        public var attempted = false
        /// Candidate offsets evaluated in the sweep. Zero means the sweep never
        /// started, which the `outcome` explains.
        public var sweepPoints = 0
        /// Sweep points that actually reprojected something. A point that
        /// reprojected nothing scores infinity by design, and a sweep of all
        /// infinities is a sweep that measured nothing.
        public var sweepPointsWithFiniteCost = 0
        /// Image pairs prepared for the sweep.
        public var pairsPrepared = 0
        /// Depth samples reprojected per sweep point, summed over pairs.
        public var samplesPerSweepPoint = 0
        public var bestCost: Double = 0
        public var medianCost: Double = 0
        /// How much better the winning offset was than the median, as a
        /// fraction, and the fraction it had to clear.
        public var relativeImprovement: Double = 0
        public var requiredRelativeImprovement: Double = 0
        /// The offset the pass settled on and every later stage used, seconds.
        /// Nil means none was settled and raw timestamps were used.
        public var settledSeconds: Double?
        /// `measured`, `carriedFromCapture`, or `none`.
        public var source = "none"
        /// Why it ended where it did, in plain language. Always set.
        public var outcome = "not run"

        public init() {}

        public var settledMilliseconds: Double? {
            settledSeconds.map { $0 * 1000 }
        }
    }

    // MARK: Revisits

    public struct Revisits: Codable, Sendable {
        public var attempted = false
        /// Frames kept as revisit anchors after spacing.
        public var anchorFrames = 0
        /// Anchor pairs that passed the distance, angle and time-gap gates.
        public var geometricCandidates = 0
        /// Candidates left after the ICP-run cap.
        public var candidatesAfterCap = 0
        /// Candidates whose depth could not be loaded. Recorded with zero
        /// confidence, so they count on the card and are ignored by the graph.
        public var candidatesWithoutDepth = 0
        /// ICP runs that converged.
        public var icpConverged = 0
        /// ICP runs that did not. A big number here with a small
        /// `icpConverged` is a scan taken too fast or too far from surfaces.
        public var icpRejected = 0
        /// Pairs returned to the pipeline, of every method.
        public var pairsReturned = 0
        /// The ones the QC card and the drift number actually count: depth ICP
        /// with confidence over 0.2.
        public var confirmedPairs = 0
        /// Whether the pipeline had to fall back to the capture's own pairs.
        public var usedCaptureFallback = false
        public var medianConfidence: Float = 0
        public var medianTranslationResidualCentimeters: Float = 0

        public init() {}
    }

    // MARK: Pose graph

    public struct PoseGraph: Codable, Sendable {
        public var attempted = false
        public var submaps = 0
        public var framesPosed = 0
        /// Revisit pairs that became real constraints: confidence over 0.05,
        /// both ends placed, and the two ends in DIFFERENT submaps.
        public var usableEdges = 0
        /// Pairs that were offered and could not be used. The difference
        /// between this and `usableEdges` is where loop closures go to die.
        public var discardedEdges = 0
        public var iterationsRun = 0
        public var initialCost: Double = 0
        public var finalCost: Double = 0
        /// True only when the solver stopped because its step got tiny, which
        /// is the only exit that means "settled".
        public var converged = false
        /// One of: `notRun`, `noFrames`, `noSubmaps`, `noEdges`, `converged`,
        /// `iterationLimit`, `dampingBlewUp`, `solverFailed`.
        public var exitReason = "notRun"
        /// The hard 2 m / 20 degree sanity gate threw the whole solution away
        /// and handed back the VIO poses. Silent until now.
        public var rejectedBySanityGate = false
        /// What the loop closures still disagree by AFTER the solve. The
        /// number that says whether the graph actually helped.
        public var finalResidualMedianCentimeters: Float = 0
        public var finalResidualMedianDegrees: Float = 0
        /// The largest rigid correction any submap received.
        public var maxSubmapCorrectionCentimeters: Float = 0
        public var maxSubmapCorrectionDegrees: Float = 0
        /// How far the refined poses ended up from the poses that went in. A
        /// graph that "ran" and moved every frame by under a millimetre did
        /// nothing, and that is worth seeing.
        public var medianPoseShiftCentimeters: Float = 0
        public var maxPoseShiftCentimeters: Float = 0
        /// True when the optional LiDAR-anchored refinement ran, and whether
        /// its own improvement check accepted it.
        public var fineRefinementRan = false
        public var fineRefinementAccepted = false
        public var fineRefinementNote = ""

        public init() {}

        /// Whether the solver stopped for a reason that means it did not do
        /// its job. Running out of iterations is ordinary and is not on this
        /// list; giving up before it started is not ordinary and is.
        public var exitIsFailure: Bool {
            switch exitReason {
            case "notRun", "noFrames", "noSubmaps", "noEdges",
                 "solverFailed", "dampingBlewUp":
                return true
            default:
                return false
            }
        }
    }

    // MARK: Carving

    public struct Carving: Codable, Sendable {
        public var attempted = false
        public var keyframesSelected = 0
        public var keyframesWithDepthLoaded = 0
        /// Keyframes the carve could not read depth from, for either reason.
        /// Skipped silently in the carve loop, so this is the only place it
        /// shows.
        public var keyframesDepthMissing = 0
        /// The half of `keyframesDepthMissing` that is DATA LOSS: a depth path
        /// was recorded and the file would not open. The other half is a frame
        /// that never recorded depth at all, which is a different problem with
        /// a different fix, and the two used to be rendered under one label
        /// reading "had no depth to read".
        ///
        /// Optional, and nil is not zero. A census written before the carver
        /// counted the two apart cannot say which half its total was, and a
        /// zero here would be a claim that every miss was a frame with no
        /// laser. Nil reads as "not recorded" on screen, which is the truth.
        public var keyframesDepthUnreadable: Int?
        /// Rays actually walked, after the subsample stride.
        public var raysCast = 0
        public var raysWithReturnInRange = 0
        /// Returned from beyond the sensor's stated reach. Neither the surface
        /// nor the space in front of it is written: UNKNOWN, by design.
        public var raysBeyondMaxRange = 0
        /// Returned from closer than the near cut.
        public var raysTooClose = 0
        public var raysNoReturn = 0
        /// No-return rays whose neighbours gave a free-space bound, so they
        /// carved something.
        public var raysNoReturnBounded = 0
        /// No-return rays where the whole neighbourhood was silent too. These
        /// carve nothing, which is correct and worth counting.
        public var raysNoReturnUnbounded = 0
        public var cellsRecorded = 0
        public var emptyCells = 0
        public var surfaceCells = 0
        /// Cells inside the scene's measured bounding box (`boundsSizeMeters`
        /// at `actualVoxelSizeMeters`) that no ray ever touched. Sparse-hash
        /// semantics make these `.unknown`, and unknown is treated very
        /// differently from empty, so the split matters. Approximate by a
        /// half-metre margin, which the grid adds around the box and this
        /// count does not.
        public var unknownCellsInBounds = 0
        public var requestedVoxelSizeMeters: Float = 0
        /// What the carver used after fitting the scene into its cell budget.
        /// A big gap between requested and actual is a silently coarsened map.
        public var actualVoxelSizeMeters: Float = 0
        public var hitCellCap = false
        public var boundsSizeMeters = SIMD3<Float>(repeating: 0)

        public init() {}

        private enum CodingKeys: String, CodingKey {
            case attempted, keyframesSelected, keyframesWithDepthLoaded
            case keyframesDepthMissing, keyframesDepthUnreadable
            case raysCast, raysWithReturnInRange
            case raysBeyondMaxRange, raysTooClose, raysNoReturn
            case raysNoReturnBounded, raysNoReturnUnbounded
            case cellsRecorded, emptyCells, surfaceCells, unknownCellsInBounds
            case requestedVoxelSizeMeters, actualVoxelSizeMeters, hitCellCap
            case boundsSizeMeters
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            attempted = try c.decode(Bool.self, forKey: .attempted)
            keyframesSelected = try c.decode(Int.self, forKey: .keyframesSelected)
            keyframesWithDepthLoaded = try c.decode(Int.self, forKey: .keyframesWithDepthLoaded)
            keyframesDepthMissing = try c.decode(Int.self, forKey: .keyframesDepthMissing)
            // `decodeIfPresent`: a file written before the split existed has
            // no key here, and that has to come back as nil rather than 0.
            keyframesDepthUnreadable = try c.decodeIfPresent(
                Int.self, forKey: .keyframesDepthUnreadable
            )
            raysCast = try c.decode(Int.self, forKey: .raysCast)
            raysWithReturnInRange = try c.decode(Int.self, forKey: .raysWithReturnInRange)
            raysBeyondMaxRange = try c.decode(Int.self, forKey: .raysBeyondMaxRange)
            raysTooClose = try c.decode(Int.self, forKey: .raysTooClose)
            raysNoReturn = try c.decode(Int.self, forKey: .raysNoReturn)
            raysNoReturnBounded = try c.decode(Int.self, forKey: .raysNoReturnBounded)
            raysNoReturnUnbounded = try c.decode(Int.self, forKey: .raysNoReturnUnbounded)
            cellsRecorded = try c.decode(Int.self, forKey: .cellsRecorded)
            emptyCells = try c.decode(Int.self, forKey: .emptyCells)
            surfaceCells = try c.decode(Int.self, forKey: .surfaceCells)
            unknownCellsInBounds = try c.decode(Int.self, forKey: .unknownCellsInBounds)
            requestedVoxelSizeMeters = try c.decode(Float.self, forKey: .requestedVoxelSizeMeters)
            actualVoxelSizeMeters = try c.decode(Float.self, forKey: .actualVoxelSizeMeters)
            hitCellCap = try c.decode(Bool.self, forKey: .hitCellCap)
            let size = try c.decode([Float].self, forKey: .boundsSizeMeters)
            boundsSizeMeters = SIMD3<Float>(
                size.count > 0 ? size[0] : 0,
                size.count > 1 ? size[1] : 0,
                size.count > 2 ? size[2] : 0
            )
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(attempted, forKey: .attempted)
            try c.encode(keyframesSelected, forKey: .keyframesSelected)
            try c.encode(keyframesWithDepthLoaded, forKey: .keyframesWithDepthLoaded)
            try c.encode(keyframesDepthMissing, forKey: .keyframesDepthMissing)
            try c.encodeIfPresent(keyframesDepthUnreadable, forKey: .keyframesDepthUnreadable)
            try c.encode(raysCast, forKey: .raysCast)
            try c.encode(raysWithReturnInRange, forKey: .raysWithReturnInRange)
            try c.encode(raysBeyondMaxRange, forKey: .raysBeyondMaxRange)
            try c.encode(raysTooClose, forKey: .raysTooClose)
            try c.encode(raysNoReturn, forKey: .raysNoReturn)
            try c.encode(raysNoReturnBounded, forKey: .raysNoReturnBounded)
            try c.encode(raysNoReturnUnbounded, forKey: .raysNoReturnUnbounded)
            try c.encode(cellsRecorded, forKey: .cellsRecorded)
            try c.encode(emptyCells, forKey: .emptyCells)
            try c.encode(surfaceCells, forKey: .surfaceCells)
            try c.encode(unknownCellsInBounds, forKey: .unknownCellsInBounds)
            try c.encode(requestedVoxelSizeMeters, forKey: .requestedVoxelSizeMeters)
            try c.encode(actualVoxelSizeMeters, forKey: .actualVoxelSizeMeters)
            try c.encode(hitCellCap, forKey: .hitCellCap)
            try c.encode(
                [boundsSizeMeters.x, boundsSizeMeters.y, boundsSizeMeters.z],
                forKey: .boundsSizeMeters
            )
        }
    }

    // MARK: Seeding

    /// The initial Gaussian set: the stage where a whole scan quietly became
    /// nothing, twice over, in the bugs that made this file necessary.
    public struct Seeding: Codable, Sendable {
        public var attempted = false
        public var keyframesSelected = 0
        public var keyframesWithDepthLoaded = 0
        /// Depth that would not load. The build loop `continue`s past these
        /// without a word, so this counter is the only evidence they existed.
        public var keyframesDepthMissing = 0
        /// Keyframes whose JPEG would not load. Their samples still become
        /// splats, coloured mid grey, which is a whole scan of grey if this
        /// equals `keyframesSelected`.
        public var keyframesImageMissing = 0

        /// Every native depth sample looked at, over every keyframe read.
        public var samplesInspected = 0
        /// Samples where the laser came back at all.
        public var samplesWithReturn = 0
        /// And that were inside the near and far cuts.
        public var samplesInRange = 0
        /// And that landed in a representable grid cell. A sample outside the
        /// grid is dropped in silence.
        public var samplesInsideGrid = 0
        /// Samples whose trust weight came back as zero. When this equals
        /// `samplesInsideGrid` the trust field contributed nothing, every seed
        /// falls back to the doubtful shape, and the whole scan is blobs.
        public var samplesWithZeroTrustWeight = 0

        /// ARKit's own confidence flag over the in-range samples: low, medium,
        /// high. NOTHING IN THIS STAGE FILTERS ON IT. It is recorded because
        /// it feeds the trust score that the trusted / doubtful line is drawn
        /// on, and because a missing confidence sidecar reads back as medium
        /// for every sample. An all-medium histogram therefore means "the file
        /// was not there", which is a fact worth being able to see rather than
        /// having to infer.
        public var samplesConfidenceLow = 0
        public var samplesConfidenceMedium = 0
        public var samplesConfidenceHigh = 0
        /// Points whose sigma came from the trust field's own cross-frame
        /// measurement rather than from the physics prior. When this is zero
        /// the sigma column is a prediction, not a measurement.
        public var gaussiansWithMeasuredSigma = 0

        /// Distinct occupied cells after voxel downsampling: the number of
        /// Gaussians actually built.
        public var gaussiansBuilt = 0
        /// Of those, how many had a usable surface normal. A trusted disc
        /// needs one, so this is a second, quieter gate on being trusted.
        public var gaussiansWithNormal = 0
        public var trustedCount = 0
        public var doubtfulCount = 0
        public var onEdgeCount = 0

        /// The value that actually decided trusted against doubtful, and the
        /// two settings it came from. Scan-relative since the 0.5 fiasco, so
        /// it is different every run and worth recording every run.
        public var trustCut: Float = 0
        public var trustedFloor: Float = 0
        public var trustedQuantile: Float = 0
        /// The distribution the cut was applied to. Printed next to the cut on
        /// purpose: a cut above the 95th percentile is a misplaced threshold,
        /// and no amount of staring at the cut alone would tell you.
        public var weightMinimum: Float = 0
        public var weightP05: Float = 0
        public var weightMedian: Float = 0
        public var weightP95: Float = 0
        public var weightMaximum: Float = 0

        /// Middle measurement sigma over the points that were built, metres.
        /// The number to hold the gate up against: the trust weight is
        /// `1 / (1 + (sigma / reference)^2)` times a confidence factor, so a
        /// gate and a median sigma together say whether the gate is inside the
        /// data or outside it.
        public var medianSigmaMeters: Float = 0
        /// True when that sigma came from the trust field's cross-frame
        /// measurements. False means it is the physics prior, which is a
        /// prediction and not a measurement, and must not be reported as one.
        public var sigmaIsMeasured = false
        /// The measurement sigma a sample would have to beat to clear
        /// `trustCut`, at perfect confidence, metres. Derived by inverting the
        /// trust weight, so it is directly comparable with
        /// `medianSigmaMeters`. Zero when the cut is not invertible.
        public var trustGateEquivalentSigmaMeters: Float = 0
        /// The reference sigma that inversion used, metres.
        public var trustNoiseReferenceMeters: Float = 0

        /// Whether the two optional inputs were actually available. Both
        /// degrade silently to "no opinion" when they are not.
        public var trustFieldLoaded = false
        public var edgeMapsLoaded = false

        /// Downsample spacing before and after clamping, metres. A scan whose
        /// spacing was clamped got a different density than the budget asked
        /// for, and that is invisible everywhere else.
        public var spacingRequestedMeters: Float = 0
        public var spacingMeters: Float = 0
        public var spacingWasClamped = false
        public var targetSplatCount = 0
        public var measuredSurfaceAreaSquareMeters: Float = 0
        /// Splats written to `init_splats.ply`.
        public var splatsWritten = 0

        public init() {}
    }
}

// MARK: - Reading the census as a list of numbers

extension PrePassCensus {

    /// The whole census as formatted rows, in stage order. A UI can render
    /// this directly; it is the five second read this file exists for.
    public var lines: [PrePassCensusLine] {
        var rows: [PrePassCensusLine] = []
        rows.append(contentsOf: inputLines)
        rows.append(contentsOf: timeOffsetLines)
        rows.append(contentsOf: revisitLines)
        rows.append(contentsOf: poseGraphLines)
        rows.append(contentsOf: carvingLines)
        rows.append(contentsOf: seedingLines)
        return rows
    }

    /// One line for a log or a summary bar. Reads left to right as the funnel
    /// the geometry actually went through.
    ///
    /// Built as separate pieces joined at the end rather than as one long
    /// chain of `+`: a twelve term string expression is exactly the kind of
    /// thing the Swift type checker gives up on, and a diagnostic line is not
    /// worth risking a build over.
    public var headline: String {
        var parts: [String] = []
        parts.append("\(PrePassCensusFormat.count(input.framesInBundle)) frames")
        parts.append("\(PrePassCensusFormat.count(seeding.keyframesWithDepthLoaded)) keyframes")
        parts.append("\(PrePassCensusFormat.count(seeding.samplesInsideGrid)) samples")
        let cut = PrePassCensusFormat.decimal(seeding.trustCut, places: 3)
        let median = PrePassCensusFormat.decimal(seeding.weightMedian, places: 3)
        let trusted = PrePassCensusFormat.count(seeding.trustedCount)
        parts.append(
            "\(PrePassCensusFormat.count(seeding.gaussiansBuilt)) seeds "
                + "(\(trusted) trusted, cut \(cut), median \(median))"
        )
        let funnel = parts.joined(separator: " -> ")

        var tail: [String] = []
        tail.append("carving \(PrePassCensusFormat.count(carving.emptyCells)) empty")
        tail.append("\(PrePassCensusFormat.count(carving.surfaceCells)) surface")
        tail.append("\(PrePassCensusFormat.count(revisits.confirmedPairs)) revisits")
        tail.append("pose graph \(poseGraph.exitReason)")
        return funnel + "; " + tail.joined(separator: "; ")
    }

    private var inputLines: [PrePassCensusLine] {
        let stage = "What came in"
        return [
            PrePassCensusLine(
                key: "input.frames", stage: stage,
                label: "Frames in the scan",
                value: PrePassCensusFormat.count(input.framesInBundle),
                detail: input.captureDurationSeconds > 0
                    ? "over \(PrePassCensusFormat.decimal(Float(input.captureDurationSeconds), places: 1)) s"
                    : nil,
                isAlarm: input.framesInBundle == 0
            ),
            PrePassCensusLine(
                key: "input.framesWithDepth", stage: stage,
                label: "Frames with a depth file",
                value: PrePassCensusFormat.count(input.framesWithDepthPath),
                detail: PrePassCensusFormat.share(
                    input.framesWithDepthPath, of: input.framesInBundle
                ),
                isAlarm: input.framesWithDepthPath == 0
            ),
            PrePassCensusLine(
                key: "input.framesWithConfidence", stage: stage,
                label: "Frames with a confidence file",
                value: PrePassCensusFormat.count(input.framesWithConfidencePath),
                detail: PrePassCensusFormat.share(
                    input.framesWithConfidencePath, of: input.framesInBundle
                ),
                // Capture writes the confidence sidecar in the same block as
                // the depth sidecar, so depth without confidence is a real
                // anomaly rather than an ordinary state.
                isAlarm: input.framesWithDepthPath > 0 && input.framesWithConfidencePath == 0
            ),
            PrePassCensusLine(
                key: "input.trackingLost", stage: stage,
                label: "Frames captured with tracking lost",
                value: PrePassCensusFormat.count(input.framesTrackingNotAvailable),
                detail: PrePassCensusFormat.share(
                    input.framesTrackingNotAvailable, of: input.framesInBundle
                ),
                isAlarm: input.framesInBundle > 0
                    && input.framesTrackingNotAvailable * 2 > input.framesInBundle
            ),
            PrePassCensusLine(
                key: "input.depthSamplesAvailable", stage: stage,
                label: "Depth readings available in total",
                value: PrePassCensusFormat.count(input.depthSamplesAvailable),
                detail: "\(input.depthWidth) by \(input.depthHeight) per frame",
                isAlarm: input.depthSamplesAvailable == 0
            )
        ]
    }

    private var timeOffsetLines: [PrePassCensusLine] {
        let stage = "Camera and motion sensor timing"
        var rows: [PrePassCensusLine] = [
            PrePassCensusLine(
                key: "timeOffset.settled", stage: stage,
                label: "Offset it settled on",
                value: timeOffset.settledMilliseconds
                    .map { "\(PrePassCensusFormat.decimal(Float($0), places: 1)) ms" }
                    ?? "none",
                detail: timeOffset.source,
                isAlarm: timeOffset.settledSeconds == nil
            ),
            PrePassCensusLine(
                key: "timeOffset.outcome", stage: stage,
                label: "What the sweep decided",
                value: timeOffset.outcome,
                detail: nil,
                isAlarm: timeOffset.settledSeconds == nil
            )
        ]
        if timeOffset.attempted {
            rows.append(
                PrePassCensusLine(
                    key: "timeOffset.sweep", stage: stage,
                    label: "Timings tried",
                    value: PrePassCensusFormat.count(timeOffset.sweepPoints),
                    detail: "\(PrePassCensusFormat.count(timeOffset.sweepPointsWithFiniteCost)) of them "
                        + "actually compared anything",
                    isAlarm: timeOffset.sweepPointsWithFiniteCost == 0
                )
            )
            rows.append(
                PrePassCensusLine(
                    key: "timeOffset.improvement", stage: stage,
                    label: "How much better the best timing was",
                    value: PrePassCensusFormat.percent(timeOffset.relativeImprovement),
                    detail: "it had to beat "
                        + PrePassCensusFormat.percent(timeOffset.requiredRelativeImprovement),
                    isAlarm: timeOffset.relativeImprovement
                        < timeOffset.requiredRelativeImprovement
                )
            )
        }
        return rows
    }

    private var revisitLines: [PrePassCensusLine] {
        let stage = "Places you walked back over"
        return [
            PrePassCensusLine(
                key: "revisits.anchors", stage: stage,
                label: "Frames kept as landmarks",
                value: PrePassCensusFormat.count(revisits.anchorFrames),
                detail: PrePassCensusFormat.share(
                    revisits.anchorFrames, of: input.framesInBundle
                ),
                isAlarm: revisits.attempted && revisits.anchorFrames == 0
            ),
            PrePassCensusLine(
                key: "revisits.candidates", stage: stage,
                label: "Pairs that looked like a return visit",
                value: PrePassCensusFormat.count(revisits.geometricCandidates),
                detail: revisits.geometricCandidates > revisits.candidatesAfterCap
                    ? "\(PrePassCensusFormat.count(revisits.candidatesAfterCap)) checked, the rest capped"
                    : nil,
                isAlarm: revisits.attempted && revisits.geometricCandidates == 0
            ),
            PrePassCensusLine(
                key: "revisits.icp", stage: stage,
                label: "Pairs the laser lined up",
                value: PrePassCensusFormat.count(revisits.icpConverged),
                detail: "\(PrePassCensusFormat.count(revisits.icpRejected)) would not line up, "
                    + "\(PrePassCensusFormat.count(revisits.candidatesWithoutDepth)) had no depth",
                isAlarm: revisits.attempted && revisits.icpConverged == 0
            ),
            PrePassCensusLine(
                key: "revisits.confirmed", stage: stage,
                label: "Return visits the rest of the pass believed",
                value: PrePassCensusFormat.count(revisits.confirmedPairs),
                detail: revisits.confirmedPairs > 0
                    ? "typical disagreement "
                        + "\(PrePassCensusFormat.decimal(revisits.medianTranslationResidualCentimeters, places: 1)) cm"
                    : "nothing to correct drift with",
                isAlarm: revisits.confirmedPairs == 0
            )
        ]
    }

    private var poseGraphLines: [PrePassCensusLine] {
        let stage = "Straightening the camera path"
        return [
            PrePassCensusLine(
                key: "poseGraph.submaps", stage: stage,
                label: "Stretches the walk was cut into",
                value: PrePassCensusFormat.count(poseGraph.submaps),
                detail: "\(PrePassCensusFormat.count(poseGraph.framesPosed)) frames placed",
                isAlarm: poseGraph.attempted && poseGraph.submaps == 0
            ),
            PrePassCensusLine(
                key: "poseGraph.edges", stage: stage,
                label: "Return visits it could actually use",
                value: PrePassCensusFormat.count(poseGraph.usableEdges),
                detail: "\(PrePassCensusFormat.count(poseGraph.discardedEdges)) could not be used",
                isAlarm: poseGraph.attempted && poseGraph.usableEdges == 0
            ),
            PrePassCensusLine(
                key: "poseGraph.exit", stage: stage,
                label: "How the solver finished",
                value: poseGraph.exitReason,
                detail: poseGraph.iterationsRun > 0
                    ? "\(poseGraph.iterationsRun) rounds"
                    : nil,
                isAlarm: poseGraph.exitIsFailure
            ),
            PrePassCensusLine(
                key: "poseGraph.cost", stage: stage,
                label: "Disagreement before and after",
                value: "\(PrePassCensusFormat.scientific(poseGraph.initialCost))"
                    + " to \(PrePassCensusFormat.scientific(poseGraph.finalCost))",
                detail: poseGraph.initialCost > 0
                    ? PrePassCensusFormat.percent(
                        1 - poseGraph.finalCost / poseGraph.initialCost
                      ) + " better"
                    : nil,
                isAlarm: poseGraph.attempted
                    && poseGraph.initialCost > 0
                    && poseGraph.finalCost >= poseGraph.initialCost
            ),
            PrePassCensusLine(
                key: "poseGraph.residual", stage: stage,
                label: "What the return visits still disagree by",
                value: "\(PrePassCensusFormat.decimal(poseGraph.finalResidualMedianCentimeters, places: 2)) cm",
                detail: "\(PrePassCensusFormat.decimal(poseGraph.finalResidualMedianDegrees, places: 2)) degrees",
                isAlarm: false
            ),
            PrePassCensusLine(
                key: "poseGraph.moved", stage: stage,
                label: "How far it moved the camera",
                value: "\(PrePassCensusFormat.decimal(poseGraph.medianPoseShiftCentimeters, places: 2)) cm typical",
                detail: "\(PrePassCensusFormat.decimal(poseGraph.maxPoseShiftCentimeters, places: 2)) cm at most",
                isAlarm: poseGraph.usableEdges > 0 && poseGraph.maxPoseShiftCentimeters < 0.1
            ),
            PrePassCensusLine(
                key: "poseGraph.sanityGate", stage: stage,
                label: "Answer thrown away as impossible",
                value: poseGraph.rejectedBySanityGate ? "yes" : "no",
                detail: poseGraph.rejectedBySanityGate
                    ? "over 2 m or 20 degrees, so the phone's own path was kept"
                    : nil,
                isAlarm: poseGraph.rejectedBySanityGate
            ),
            PrePassCensusLine(
                key: "poseGraph.fine", stage: stage,
                label: "Extra fine tuning against the laser",
                value: poseGraph.fineRefinementRan
                    ? (poseGraph.fineRefinementAccepted ? "kept" : "measured no better, dropped")
                    : "not run",
                detail: poseGraph.fineRefinementNote.isEmpty
                    ? nil : poseGraph.fineRefinementNote,
                isAlarm: false
            )
        ]
    }

    /// The second line under "Frames carved from". Built here rather than
    /// inline so the "not recorded" case is a branch and not a printed zero.
    private var carvingKeyframeDetail: String {
        var detail = PrePassCensusFormat.count(carving.keyframesSelected) + " chosen, "
        detail += PrePassCensusFormat.count(carving.keyframesDepthMissing)
        detail += " could not be read"
        if let unreadable = carving.keyframesDepthUnreadable {
            detail += " ("
            detail += PrePassCensusFormat.count(unreadable)
            detail += " of them a depth file that would not open)"
        }
        return detail
    }

    private var carvingLines: [PrePassCensusLine] {
        let stage = "Which air the laser flew through"
        return [
            PrePassCensusLine(
                key: "carving.keyframes", stage: stage,
                label: "Frames carved from",
                value: PrePassCensusFormat.count(carving.keyframesWithDepthLoaded),
                // "could not be read", not "had no depth to read". Every frame
                // the carve tries to open already has a depth path recorded
                // (`keyframes(from:)` filters on it), so the old wording said
                // "this phone had no laser here" about a fact that is "this
                // scan's laser data was lost". The split is appended only when
                // the writer actually measured it.
                detail: carvingKeyframeDetail,
                isAlarm: carving.attempted && carving.keyframesWithDepthLoaded == 0
            ),
            PrePassCensusLine(
                key: "carving.rays", stage: stage,
                label: "Laser rays followed",
                value: PrePassCensusFormat.count(carving.raysCast),
                detail: "\(PrePassCensusFormat.count(carving.raysWithReturnInRange)) hit something in range",
                isAlarm: carving.attempted && carving.raysCast == 0
            ),
            PrePassCensusLine(
                key: "carving.raysLost", stage: stage,
                label: "Rays that proved nothing",
                value: PrePassCensusFormat.count(
                    carving.raysBeyondMaxRange + carving.raysNoReturnUnbounded
                ),
                detail: "\(PrePassCensusFormat.count(carving.raysBeyondMaxRange)) came back from too far, "
                    + "\(PrePassCensusFormat.count(carving.raysNoReturnUnbounded)) came back from nothing",
                isAlarm: false
            ),
            PrePassCensusLine(
                key: "carving.empty", stage: stage,
                label: "Cells proved empty",
                value: PrePassCensusFormat.count(carving.emptyCells),
                detail: PrePassCensusFormat.share(carving.emptyCells, of: carving.cellsRecorded),
                isAlarm: carving.attempted && carving.emptyCells == 0
            ),
            PrePassCensusLine(
                key: "carving.surface", stage: stage,
                label: "Cells with a surface in them",
                value: PrePassCensusFormat.count(carving.surfaceCells),
                detail: PrePassCensusFormat.share(carving.surfaceCells, of: carving.cellsRecorded),
                isAlarm: carving.attempted && carving.surfaceCells == 0
            ),
            PrePassCensusLine(
                key: "carving.unknown", stage: stage,
                label: "Cells never touched by any ray",
                value: PrePassCensusFormat.count(carving.unknownCellsInBounds),
                detail: "still unknown, which is not the same as empty",
                isAlarm: false
            ),
            PrePassCensusLine(
                key: "carving.voxel", stage: stage,
                label: "Cell size used",
                value: "\(PrePassCensusFormat.decimal(carving.actualVoxelSizeMeters * 100, places: 1)) cm",
                detail: "\(PrePassCensusFormat.decimal(carving.requestedVoxelSizeMeters * 100, places: 1)) cm asked for"
                    + (carving.hitCellCap ? ", ran out of cells" : ""),
                isAlarm: carving.hitCellCap
                    || (carving.attempted
                        && carving.actualVoxelSizeMeters > carving.requestedVoxelSizeMeters * 1.5)
            )
        ]
    }

    private var seedingLines: [PrePassCensusLine] {
        let stage = "First points to train from"

        // The two longest sentences are built here rather than inline. Keeping
        // a nested ternary of concatenations out of an array literal is both
        // easier to read and much cheaper for the compiler to type check.
        let lowest = PrePassCensusFormat.decimal(seeding.weightMinimum, places: 3)
        let highest = PrePassCensusFormat.decimal(seeding.weightMaximum, places: 3)
        let middle = PrePassCensusFormat.decimal(seeding.weightMedian, places: 3)
        let floorValue = PrePassCensusFormat.decimal(seeding.trustedFloor, places: 3)
        let cutDetail = "scores ran \(lowest) to \(highest), middle \(middle), floor \(floorValue)"

        let low = PrePassCensusFormat.count(seeding.samplesConfidenceLow)
        let medium = PrePassCensusFormat.count(seeding.samplesConfidenceMedium)
        let high = PrePassCensusFormat.count(seeding.samplesConfidenceHigh)
        let confidenceValue = "\(high) high, \(medium) medium, \(low) low"
        let allMedium = seeding.samplesInRange > 0
            && seeding.samplesConfidenceMedium == seeding.samplesInRange
        let confidenceDetail = allMedium
            ? "every reading came back medium, which is what a missing confidence file looks like"
            : "recorded only, nothing here is thrown away for being low"

        // The trust line and the measurement it judges, both in centimetres.
        // This is the row that would have made the 2 cm gate obvious on day
        // one, so it gets built carefully and says "not measured" rather than
        // printing a prediction under a measurement's name.
        let gateCm = PrePassCensusFormat.decimal(
            seeding.trustGateEquivalentSigmaMeters * 100, places: 1
        )
        let measuredCm = PrePassCensusFormat.decimal(seeding.medianSigmaMeters * 100, places: 1)
        // Half a scan sitting on the wrong side of a line drawn AT the middle
        // of that scan is arithmetic, not a fault, so this needs a margin. Half
        // as tight again as anything the scan reached is a line in the wrong
        // place; a hair either side of the middle is just the middle.
        let gateIsOutsideTheData = seeding.sigmaIsMeasured
            && seeding.trustGateEquivalentSigmaMeters > 0
            && seeding.medianSigmaMeters > seeding.trustGateEquivalentSigmaMeters * 1.5
        let gateValue: String
        let gateDetail: String
        if seeding.trustGateEquivalentSigmaMeters <= 0 {
            gateValue = "no line to work out"
            gateDetail = "every point cleared the line, or none was built"
        } else if seeding.sigmaIsMeasured {
            gateValue = "asked for \(gateCm) cm, measured \(measuredCm) cm"
            gateDetail = gateIsOutsideTheData
                ? "the test is tighter than this scan can reach, so most points fail it"
                : "the test is inside what this scan reached"
        } else {
            gateValue = "asked for \(gateCm) cm, measured nothing"
            gateDetail = "the reliability measurements were missing, so there is nothing "
                + "to hold the test up against"
        }

        let spacingDetail: String
        if seeding.spacingWasClamped {
            let wanted = PrePassCensusFormat.decimal(
                seeding.spacingRequestedMeters * 100, places: 1
            )
            spacingDetail = "the sum wanted \(wanted) cm and was capped"
        } else {
            let area = PrePassCensusFormat.decimal(
                seeding.measuredSurfaceAreaSquareMeters, places: 1
            )
            let budget = PrePassCensusFormat.count(seeding.targetSplatCount)
            spacingDetail = "from \(area) square metres and a budget of \(budget)"
        }

        return [
            PrePassCensusLine(
                key: "seeding.keyframes", stage: stage,
                label: "Frames read",
                value: PrePassCensusFormat.count(seeding.keyframesWithDepthLoaded),
                detail: "\(PrePassCensusFormat.count(seeding.keyframesSelected)) chosen, "
                    + "\(PrePassCensusFormat.count(seeding.keyframesDepthMissing)) had no depth, "
                    + "\(PrePassCensusFormat.count(seeding.keyframesImageMissing)) had no picture",
                isAlarm: seeding.attempted && seeding.keyframesWithDepthLoaded == 0
            ),
            PrePassCensusLine(
                key: "seeding.samplesInspected", stage: stage,
                label: "Depth readings looked at",
                value: PrePassCensusFormat.count(seeding.samplesInspected),
                detail: PrePassCensusFormat.share(
                    seeding.samplesInspected, of: input.depthSamplesAvailable
                ),
                isAlarm: seeding.attempted && seeding.samplesInspected == 0
            ),
            PrePassCensusLine(
                key: "seeding.samplesWithReturn", stage: stage,
                label: "Readings where the laser came back",
                value: PrePassCensusFormat.count(seeding.samplesWithReturn),
                detail: PrePassCensusFormat.share(
                    seeding.samplesWithReturn, of: seeding.samplesInspected
                ),
                isAlarm: seeding.attempted && seeding.samplesWithReturn == 0
            ),
            PrePassCensusLine(
                key: "seeding.samplesInRange", stage: stage,
                label: "Readings inside the usable distance",
                value: PrePassCensusFormat.count(seeding.samplesInRange),
                detail: PrePassCensusFormat.share(
                    seeding.samplesInRange, of: seeding.samplesWithReturn
                ),
                isAlarm: seeding.attempted && seeding.samplesInRange == 0
            ),
            PrePassCensusLine(
                key: "seeding.samplesInsideGrid", stage: stage,
                label: "Readings that landed inside the map",
                value: PrePassCensusFormat.count(seeding.samplesInsideGrid),
                detail: PrePassCensusFormat.share(
                    seeding.samplesInsideGrid, of: seeding.samplesInRange
                ),
                isAlarm: seeding.attempted && seeding.samplesInsideGrid == 0
            ),
            PrePassCensusLine(
                key: "seeding.confidence", stage: stage,
                label: "How sure the sensor said it was",
                value: confidenceValue,
                detail: confidenceDetail,
                isAlarm: seeding.samplesInRange > 0
                    && seeding.samplesConfidenceLow == seeding.samplesInRange
            ),
            PrePassCensusLine(
                key: "seeding.gaussians", stage: stage,
                label: "Points built after thinning",
                value: PrePassCensusFormat.count(seeding.gaussiansBuilt),
                detail: PrePassCensusFormat.shareText(
                    seeding.gaussiansBuilt, of: seeding.samplesInsideGrid
                ) + " at \(PrePassCensusFormat.decimal(seeding.spacingMeters * 100, places: 1)) cm apart",
                isAlarm: seeding.attempted && seeding.gaussiansBuilt == 0
            ),
            PrePassCensusLine(
                key: "seeding.trusted", stage: stage,
                label: "Points solid enough to pin in place",
                value: PrePassCensusFormat.count(seeding.trustedCount),
                detail: PrePassCensusFormat.share(
                    seeding.trustedCount, of: seeding.gaussiansBuilt
                ),
                isAlarm: seeding.attempted && seeding.trustedCount == 0
            ),
            PrePassCensusLine(
                key: "seeding.doubtful", stage: stage,
                label: "Points left free to slide",
                value: PrePassCensusFormat.count(seeding.doubtfulCount),
                detail: PrePassCensusFormat.share(
                    seeding.doubtfulCount, of: seeding.gaussiansBuilt
                ),
                isAlarm: false
            ),
            PrePassCensusLine(
                key: "seeding.trustCut", stage: stage,
                label: "The line between the two",
                value: PrePassCensusFormat.decimal(seeding.trustCut, places: 3),
                detail: cutDetail,
                isAlarm: seeding.attempted
                    && seeding.gaussiansBuilt > 0
                    && seeding.trustCut > seeding.weightP95
            ),
            PrePassCensusLine(
                key: "seeding.sigma", stage: stage,
                label: "How tight a reading had to be, against how tight they were",
                value: gateValue,
                detail: gateDetail,
                isAlarm: gateIsOutsideTheData
            ),
            PrePassCensusLine(
                key: "seeding.normals", stage: stage,
                label: "Points with a usable surface direction",
                value: PrePassCensusFormat.count(seeding.gaussiansWithNormal),
                detail: PrePassCensusFormat.shareText(
                    seeding.gaussiansWithNormal, of: seeding.gaussiansBuilt
                ) + ", needed as well as the score to be pinned",
                isAlarm: seeding.attempted
                    && seeding.gaussiansBuilt > 0
                    && seeding.gaussiansWithNormal == 0
            ),
            PrePassCensusLine(
                key: "seeding.zeroTrust", stage: stage,
                label: "Readings with no reliability score at all",
                value: PrePassCensusFormat.count(seeding.samplesWithZeroTrustWeight),
                detail: PrePassCensusFormat.shareText(
                    seeding.samplesWithZeroTrustWeight, of: seeding.samplesInsideGrid
                ) + (seeding.trustFieldLoaded ? "" : ", reliability measurements were missing"),
                isAlarm: !seeding.trustFieldLoaded && seeding.attempted
            ),
            PrePassCensusLine(
                key: "seeding.spacing", stage: stage,
                label: "Spacing between points",
                value: "\(PrePassCensusFormat.decimal(seeding.spacingMeters * 100, places: 1)) cm",
                detail: spacingDetail,
                isAlarm: seeding.spacingWasClamped
            ),
            PrePassCensusLine(
                key: "seeding.written", stage: stage,
                label: "Points written to the file",
                value: PrePassCensusFormat.count(seeding.splatsWritten),
                detail: seeding.splatsWritten == seeding.gaussiansBuilt
                    ? nil : "does not match the number built",
                isAlarm: seeding.attempted && seeding.splatsWritten == 0
            )
        ]
    }
}

// MARK: - The zeros, called out by name

extension PrePassCensus {

    /// Every stage that produced nothing, as a plain sentence with its number
    /// in it. The pipeline turns these into QC findings so they land on the
    /// screen the owner already looks at, without waiting for a census screen.
    ///
    /// Order is worst first: a stage that found nothing at all before a stage
    /// that found less than it should have.
    public var alarms: [PrePassCensusAlarm] {
        var found: [PrePassCensusAlarm] = []
        // Every message is assembled from named pieces rather than one long
        // chain of `+`. Two reasons: the sentence stays readable in the
        // source, and a very long string expression is the classic way to
        // make the Swift type checker give up on a file.

        if input.framesWithDepthPath == 0 {
            let frames = PrePassCensusFormat.count(input.framesInBundle)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_depth_frames",
                    message: "None of the \(frames) frames in this scan has a depth file, "
                        + "so nothing could be measured from the laser at all.",
                    fixHint: "A fresh scan on a phone with the laser sensor is the only fix."
                )
            )
        }

        if seeding.attempted && seeding.gaussiansBuilt == 0 {
            let looked = PrePassCensusFormat.count(seeding.samplesInspected)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_seeds",
                    message: "Not one starting point survived out of \(looked) depth readings.",
                    fixHint: nil
                )
            )
        } else if seeding.attempted && seeding.trustedCount == 0 && seeding.gaussiansBuilt > 0 {
            let built = PrePassCensusFormat.count(seeding.gaussiansBuilt)
            let cut = PrePassCensusFormat.decimal(seeding.trustCut, places: 3)
            let middle = PrePassCensusFormat.decimal(seeding.weightMedian, places: 3)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_trusted_seeds",
                    message: "None of the \(built) starting points was solid enough to pin "
                        + "down. The line was \(cut) and the middle score was \(middle), "
                        + "so every point starts as a smear instead of a surface.",
                    fixHint: "A slower, steadier pass closer to the surfaces raises the scores."
                )
            )
        }

        // Not zero, but so close to it that the effect is the same. The line is
        // drawn at the middle of this scan's own scores, so roughly half the
        // points are expected to clear it; under one in twenty means something
        // other than the line is doing the rejecting.
        if seeding.attempted, seeding.gaussiansBuilt > 0, seeding.trustedCount > 0,
           seeding.trustedCount * 20 < seeding.gaussiansBuilt {
            let solid = PrePassCensusFormat.count(seeding.trustedCount)
            let built = PrePassCensusFormat.count(seeding.gaussiansBuilt)
            let share = PrePassCensusFormat.percent(
                Double(seeding.trustedCount) / Double(seeding.gaussiansBuilt)
            )
            found.append(
                PrePassCensusAlarm(
                    code: "census_almost_no_trusted_seeds",
                    message: "Only \(solid) of \(built) starting points were solid enough to "
                        + "pin down, which is \(share). About half is normal, so nearly "
                        + "everything is starting as a smear.",
                    fixHint: nil
                )
            )
        }

        if seeding.attempted && seeding.gaussiansBuilt > 0 && seeding.gaussiansWithNormal == 0 {
            let built = PrePassCensusFormat.count(seeding.gaussiansBuilt)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_surface_normals",
                    message: "Not one of the \(built) starting points got a usable surface "
                        + "direction, which on its own stops every point from being pinned "
                        + "no matter how good its score was.",
                    fixHint: nil
                )
            )
        }

        if seeding.attempted && !seeding.trustFieldLoaded {
            let built = PrePassCensusFormat.count(seeding.gaussiansBuilt)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_trust_field",
                    message: "The depth reliability measurements were not available when the "
                        + "starting points were placed, so all \(built) of them were treated "
                        + "as unreliable.",
                    fixHint: nil
                )
            )
        }

        if carving.attempted && carving.emptyCells == 0 {
            let rays = PrePassCensusFormat.count(carving.raysCast)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_empty_cells",
                    message: "Not one pocket of air was proved empty out of \(rays) laser "
                        + "rays, so nothing will clean up floating specks later.",
                    fixHint: nil
                )
            )
        }

        if carving.attempted && carving.surfaceCells == 0 {
            let rays = PrePassCensusFormat.count(carving.raysCast)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_surface_cells",
                    message: "Not one cell was marked as holding a surface out of \(rays) "
                        + "laser rays.",
                    fixHint: nil
                )
            )
        }

        if revisits.confirmedPairs == 0 {
            let candidates = PrePassCensusFormat.count(revisits.geometricCandidates)
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_revisits",
                    message: "No place you walked back over was confirmed, out of "
                        + "\(candidates) that looked like one, so there was nothing to "
                        + "correct drift with and the phone's own path was kept.",
                    fixHint: "Walking back past somewhere you already scanned, once or twice, "
                        + "gives the app the fix it needs."
                )
            )
        }

        if poseGraph.rejectedBySanityGate {
            found.append(
                PrePassCensusAlarm(
                    code: "census_pose_graph_rejected",
                    message: "The straightened camera path wanted to move something by more "
                        + "than 2 metres or 20 degrees, which is a wrong match rather than "
                        + "real drift, so the whole correction was thrown away.",
                    fixHint: nil
                )
            )
        } else if poseGraph.attempted && poseGraph.usableEdges == 0 {
            let offered = PrePassCensusFormat.count(revisits.pairsReturned)
            found.append(
                PrePassCensusAlarm(
                    code: "census_pose_graph_no_edges",
                    message: "The camera path could not be straightened: none of the "
                        + "\(offered) return visits could be used as evidence, so the phone's "
                        + "own path was kept as it was.",
                    fixHint: nil
                )
            )
        } else if poseGraph.usableEdges > 0 && poseGraph.maxPoseShiftCentimeters < 0.1 {
            let used = PrePassCensusFormat.count(poseGraph.usableEdges)
            found.append(
                PrePassCensusAlarm(
                    code: "census_pose_graph_moved_nothing",
                    message: "The camera path was straightened using \(used) return visits "
                        + "and moved nothing by as much as a millimetre, which usually means "
                        + "the corrections never took effect.",
                    fixHint: nil
                )
            )
        }

        if timeOffset.attempted && timeOffset.settledSeconds == nil {
            let why = timeOffset.outcome
            found.append(
                PrePassCensusAlarm(
                    code: "census_no_time_offset",
                    message: "The camera and motion sensor timing could not be settled "
                        + "(\(why)), so timestamps were used exactly as recorded.",
                    fixHint: nil
                )
            )
        }

        if carving.hitCellCap {
            let used = PrePassCensusFormat.decimal(carving.actualVoxelSizeMeters * 100, places: 1)
            let asked = PrePassCensusFormat.decimal(
                carving.requestedVoxelSizeMeters * 100, places: 1
            )
            found.append(
                PrePassCensusAlarm(
                    code: "census_carving_capped",
                    message: "The map of empty air ran out of cells and was built at \(used) cm "
                        + "instead of the \(asked) cm asked for.",
                    fixHint: "Scanning one room at a time keeps the finest detail."
                )
            )
        }

        if seeding.spacingWasClamped {
            let used = PrePassCensusFormat.decimal(seeding.spacingMeters * 100, places: 1)
            let wanted = PrePassCensusFormat.decimal(
                seeding.spacingRequestedMeters * 100, places: 1
            )
            found.append(
                PrePassCensusAlarm(
                    code: "census_spacing_clamped",
                    message: "The starting points were spread \(used) cm apart rather than the "
                        + "\(wanted) cm the budget worked out, so the scan starts coarser than "
                        + "it asked for.",
                    fixHint: nil
                )
            )
        }

        return found
    }
}

// MARK: - Disk

extension PrePassCensus {

    /// Writes `prepass/census.json`. Called once, at the end of the pass.
    ///
    /// Failing to write the census must never fail the pass: it is a report
    /// about the work, not the work. The caller logs and carries on.
    public func write(at ref: CaptureBundleRef) throws {
        // Filled on a copy so the shared keys are always derived from the
        // final state of the sections, whatever order the caller filled them
        // in, and so this stays non-mutating for the caller.
        var finished = self
        finished.fillSharedKeys()
        let data = try ContractsJSON.encoder().encode(finished)
        try PrePassBinary.write(data, to: ref.url(forRelativePath: PrePassPaths.census))
    }

    /// Reads a census back. Nil when there is none, or when it was written by
    /// a version this build does not understand: an unreadable report is not
    /// worth an error, but a half-decoded one would be a lie.
    public static func read(at ref: CaptureBundleRef) -> PrePassCensus? {
        let url = ref.url(forRelativePath: PrePassPaths.census)
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let census = try? ContractsJSON.decoder().decode(PrePassCensus.self, from: data)
        else { return nil }
        guard census.formatVersion == PrePassCensus.currentFormatVersion else { return nil }
        return census
    }
}

// MARK: - Formatting

/// Number formatting for the census, done by hand rather than with
/// `NumberFormatter` so it is identical in a log line, in the JSON, and on
/// screen, and so it costs nothing to call.
public enum PrePassCensusFormat {

    /// `7840000` -> `7,840,000`. Grouping makes an order-of-magnitude mistake
    /// visible at a glance, which is the entire point of this file.
    public static func count(_ value: Int) -> String {
        let negative = value < 0
        var digits = String(value.magnitude)
        var grouped = ""
        var seen = 0
        for character in digits.reversed() {
            if seen > 0 && seen % 3 == 0 { grouped.append(",") }
            grouped.append(character)
            seen += 1
        }
        digits = String(grouped.reversed())
        return negative ? "-" + digits : digits
    }

    /// Fixed decimal places, with non-finite values named rather than printed
    /// as `nan`, which reads like a number and is not one.
    public static func decimal(_ value: Float, places: Int) -> String {
        guard value.isFinite else { return value.isNaN ? "not a number" : "unbounded" }
        return String(format: "%.\(places)f", value)
    }

    public static func decimal(_ value: Double, places: Int) -> String {
        guard value.isFinite else { return value.isNaN ? "not a number" : "unbounded" }
        return String(format: "%.\(places)f", value)
    }

    /// A fraction as a percentage, e.g. `0.012` -> `1.2 percent`. Spelled out
    /// rather than `%` because these strings are read aloud by the UI in
    /// places and a percent sign in a format string is a trap.
    public static func percent(_ fraction: Double) -> String {
        guard fraction.isFinite else { return "unknown" }
        return decimal(fraction * 100, places: 1) + " percent"
    }

    /// `96,300 of 7,840,000, 1.2 percent`. Nil when there is no denominator to
    /// speak of, so a caller never prints "of 0".
    public static func share(_ part: Int, of whole: Int) -> String? {
        guard whole > 0 else { return nil }
        let fraction = Double(part) / Double(whole)
        return "\(count(part)) of \(count(whole)), \(percent(fraction))"
    }

    /// The same, for the places that are building a longer sentence and cannot
    /// take a nil. Falls back to the bare count.
    public static func shareText(_ part: Int, of whole: Int) -> String {
        share(part, of: whole) ?? count(part)
    }

    /// Costs span many orders of magnitude, so they get an exponent rather
    /// than twenty digits.
    public static func scientific(_ value: Double) -> String {
        guard value.isFinite else { return value.isNaN ? "not a number" : "unbounded" }
        if value == 0 { return "0" }
        if abs(value) >= 0.01 && abs(value) < 1_000_000 {
            return String(format: "%.3g", value)
        }
        return String(format: "%.3e", value)
    }
}
