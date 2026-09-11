//
//  TrainerCensus.swift
//  Trainer
//
//  THE SPLAT CENSUS. WHERE EVERY GAUSSIAN WENT, WRITTEN DOWN AS IT HAPPENS.
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE EXISTS
//  ---------------------------------------------------------------------------
//  The first real scan this app ever produced "looked like nothing", and
//  finding out why took a day of reading code and estimating survivor counts
//  stage by stage, because the trainer recorded NOTHING about its own
//  behaviour. Four separate faults had each destroyed most of the model:
//
//    1. `absGradThreshold` was in the wrong units, so densification created
//       ZERO Gaussians on every run that has ever happened.
//    2. `degradeForHeat` cut the splat cap BELOW the live population, which
//       deleted real Gaussians and then left `headroom = cap - count` at 0,
//       switching densification off for the rest of the run.
//    3. `pruneStartFraction` / `pruneEndFraction` were declared, defaulted and
//       assigned but read nowhere, so pruning ran 29 times instead of ~10.
//    4. The seeder's trust gate rejected essentially every realistic sample,
//       so every seed was laid as a stretched blob instead of a solid disc.
//
//  NOT ONE of them logged, warned or failed. The app reported success and
//  produced almost nothing. Every one of the four would have been a single
//  line in this file's output:
//
//    1. "growth was allowed in 18 passes with room for 240,000 more and
//        created 0"
//    2. "the splat cap was cut from 300,000 to 137,000 at iteration 100 while
//        182,000 were alive: 45,000 real points were deleted"
//    3. "faint-or-oversized pruning ran in 17 passes outside its window"
//    4. "0 of 182,340 seeds were laid as solid discs; every one was stretched
//        along the viewing ray"
//
//  ---------------------------------------------------------------------------
//  THE RULES THIS FILE KEEPS
//  ---------------------------------------------------------------------------
//  * CHEAP. Everything here is an integer counter held in memory. There is one
//    disk write, at the very end of the run. There is no per-iteration write
//    and there is not one GPU synchronisation that the trainer would not have
//    done anyway: every number recorded is one the loop already had in a Swift
//    variable.
//  * IT AGGREGATES, IT DOES NOT RE-MEASURE. `TrainerDensifyOutcome` already
//    counts splits, clones, relocations, each prune reason and the carve, and
//    `TrainerBudgetGovernor.changes` already records every reduction. This
//    file collects those, stamps them with when and where, and adds only the
//    context they were missing (which window was open, what the cap was, how
//    much headroom there was).
//  * IT NEVER GUESSES. A number that was not measured is not in the file. The
//    one derived figure, `splatCountAfterGrowth`, is exact arithmetic on two
//    measured counts and says so where it is declared.
//  * IT WRITES EVEN WHEN THE RUN FAILS. The run that throws is the run you
//    most want the census for, so `MetalSplatTrainer.run` writes this from a
//    `defer` rather than from the success path.
//

import Foundation

// MARK: - Budget

/// A training budget flattened to plain numbers, so "asked for" and "actually
/// run" can be read side by side without decoding a nested type.
struct TrainerCensusBudget: Codable {
    var splatCap: Int
    var iterations: Int
    var renderLongEdgePixels: Int
    var shDegree: Int
    var keyframeCount: Int
    var memoryCeilingBytes: UInt64
    /// Recorded as it is written into the model: the GPU structs in
    /// TrainerGPULayouts.swift are fp32 by design whatever was asked for, so
    /// this is false on every real run.
    var useHalfPrecision: Bool

    init(_ budget: TrainingBudget, halfPrecisionActuallyUsed: Bool = false) {
        splatCap = budget.splatCap
        iterations = budget.iterations
        renderLongEdgePixels = budget.renderLongEdgePixels
        shDegree = budget.shDegree.rawValue
        keyframeCount = budget.keyframeCount
        memoryCeilingBytes = budget.memoryCeilingBytes
        useHalfPrecision = halfPrecisionActuallyUsed
    }
}

/// One budget-lowering event, with WHEN it happened and what the population
/// was at that moment. The last two fields are the whole point: a cap cut that
/// lands at or below the live count deletes real geometry and then leaves zero
/// headroom, and without the live count beside it that is invisible.
struct TrainerCensusBudgetReduction: Codable {
    /// "splatCap", "renderLongEdgePixels" or "iterations".
    var what: String
    var from: Int
    var to: Int
    /// "the phone was warm", "memory share", "memory ceiling" or "memory
    /// headroom".
    var reason: String
    /// Thermal level as a number (0 nominal, 3 critical) when the reason was
    /// heat; nil otherwise.
    var thermalLevel: Int?
    var residentBytes: UInt64?
    var comparedAgainstBytes: UInt64?
    /// Slice this happened in, and the iteration counted across the whole run.
    /// Both are -1 for a reduction taken before any loop started (the sizing
    /// pass in `initialSplatCap`), which is a real and different moment.
    var atSliceIndex: Int
    var atIteration: Int
    /// How many Gaussians were alive when this was applied.
    var liveSplatCount: Int
    /// True when a splat-cap cut landed at or below the live population. That
    /// is the one-way ratchet: real Gaussians are deleted to fit, and the
    /// headroom densification needs is then zero for the rest of the run.
    var landedAtOrBelowLivePopulation: Bool
    /// The sentence the user was shown for this reduction.
    var messageShown: String

    /// Straight transcription of what the governor already recorded. Nothing
    /// is recomputed here, so the file and the governor can never disagree.
    init(_ change: TrainerBudgetChange) {
        what = change.changedField
        from = change.fromValue
        to = change.toValue
        reason = change.reason.censusReason
        thermalLevel = change.reason.thermalLevel?.rawValue
        residentBytes = change.reason.residentBytes
        comparedAgainstBytes = change.reason.comparedAgainstBytes
        atSliceIndex = change.atSliceIndex
        atIteration = change.atIteration
        liveSplatCount = change.liveSplatCount
        landedAtOrBelowLivePopulation = change.landedAtOrBelowLivePopulation
        messageShown = change.message
    }
}

// MARK: - The gates, as the code actually read them

/// Every schedule constant the loop READ, recorded from the point of use.
///
/// Bug 3 was a pair of settings that existed in three files and were read by
/// nothing. A value copied out of a struct at start-up would look identical in
/// the working and the broken case, so these are captured inside the slice
/// loop, next to the passes they gate. "The window says 0.15 to 0.80 and
/// pruning ran in 29 passes" is then a contradiction visible on one page.
struct TrainerCensusGates: Codable {
    var densifyStartFraction: Float
    var densifyEndFraction: Float
    var densifyIntervalIterations: Int
    var pruneStartFraction: Float
    var pruneEndFraction: Float
    var carveIntervalIterations: Int
    /// The score a Gaussian must beat to become a densification candidate, as
    /// the loop ACTUALLY applies it. It is `0`: `TrainerDensifier` selects on
    /// `score[i] > 0` and lets the ranked truncation to the cap do the cutting,
    /// which is scale-free and cannot be broken again by a units change.
    ///
    /// This field is deliberately NOT `TrainerTuning.absGradThreshold`. That
    /// constant still exists, at 1e-9, and the densifier no longer reads it.
    /// Recording it here under a name that implies it is the gate would be the
    /// same fault as the pruning window that was declared and read by nothing:
    /// a census whose whole job is to catch a dead setting must not print one
    /// as though it were live.
    var densifyScoreFloor: Float
    var pruneOpacity: Float
    var pruneMaxWorldScaleFraction: Float
    var pruneMaxScreenRadiusPx: Float
    var maxGrowthFractionPerPass: Float
    var maxRelocationFractionPerPass: Float
    var pruneMaxFractionPerPass: Float
    var warmupFraction: Float
    var binarizeLastFraction: Float
    var minimumAuthorityForDepth: Float
}

// MARK: - One densification / prune / carve pass

/// What one call to `TrainerDensifier.run` did, and the context it did it in.
///
/// Growth, pruning and carving all happen inside one pass, so one row covers
/// all three. `splatCountAfterGrowth` is the only derived number here and it
/// is exact rather than estimated: a split replaces its parent in place and
/// appends one child, and a clone appends one, so the population after growth
/// is always `splatCountBefore + addedBySplit + addedByClone`.
struct TrainerCensusDensifyPass: Codable {
    var sliceIndex: Int
    /// Iteration within this slice, which is what the windows are measured
    /// against.
    var iteration: Int
    var progressFraction: Float

    var growthWindowOpen: Bool
    var pruneWindowOpen: Bool
    /// True when the carver actually ran this pass (it is due only every
    /// `carveIntervalIterations`). `carverAvailable` says whether there was an
    /// occupancy grid to run at all, so "no grid" and "not due" stay distinct.
    var carveRan: Bool
    var carverAvailable: Bool

    var capInForce: Int
    var headroom: Int
    var growthAllowance: Int

    /// How many Gaussians were scored, how many scored above zero, and how
    /// many of those survived the "something actually looked at it" filter.
    /// A large population with zero candidates is the fingerprint of a dead
    /// gradient signal or a threshold in the wrong units.
    var splatsScored: Int
    var splatsWithNonZeroScore: Int
    var candidatesAfterVisibilityFilter: Int
    var relocationDonorsAvailable: Int

    var splatCountBefore: Int
    var addedBySplit: Int
    var addedByClone: Int
    var relocated: Int
    /// Derived, exactly: before + split + clone. See the type comment.
    var splatCountAfterGrowth: Int
    var prunedNonFinite: Int
    var prunedLowOpacity: Int
    var prunedOversized: Int
    var carvedFromEmptySpace: Int
    var trimmedToCap: Int
    var splatCountAfter: Int
    /// `TrainerDensifyOutcome.growthVerdict.rawValue`: the one-value answer to
    /// "why did this pass create what it created", decided inside the
    /// densifier from counters that pass actually measured.
    ///
    /// It is copied, not re-derived. The census reconstructing the same
    /// question from seven counters is how `nothingScored` (a dead gradient
    /// signal) and `nothingVisible` (a dead visibility accumulator) came to
    /// read identically, and those are two different faults with two
    /// different fixes.
    var growthVerdict: String

    init(
        sliceIndex: Int,
        iteration: Int,
        progressFraction: Float,
        carverAvailable: Bool,
        outcome: TrainerDensifyOutcome
    ) {
        self.sliceIndex = sliceIndex
        self.iteration = iteration
        self.progressFraction = progressFraction
        self.growthWindowOpen = outcome.growthAllowed
        self.pruneWindowOpen = outcome.pruneAllowed
        self.carveRan = outcome.carveAttempted
        self.carverAvailable = carverAvailable
        self.capInForce = outcome.splatCapInForce
        self.headroom = outcome.headroomAtStart
        self.growthAllowance = outcome.growthAllowance
        self.splatsScored = outcome.splatsScored
        self.splatsWithNonZeroScore = outcome.splatsWithNonZeroScore
        self.candidatesAfterVisibilityFilter = outcome.candidatesAfterVisibilityFilter
        self.relocationDonorsAvailable = outcome.relocationDonorsAvailable
        self.splatCountBefore = outcome.splatCountBefore
        self.addedBySplit = outcome.split
        self.addedByClone = outcome.cloned
        self.relocated = outcome.relocated
        self.splatCountAfterGrowth = outcome.splatCountBefore + outcome.split + outcome.cloned
        self.prunedNonFinite = outcome.prunedNonFinite
        self.prunedLowOpacity = outcome.prunedLowOpacity
        self.prunedOversized = outcome.prunedOversized
        self.carvedFromEmptySpace = outcome.carvedFromEmptySpace
        self.trimmedToCap = outcome.trimmedToCap
        self.splatCountAfter = outcome.splatCountAfter
        self.growthVerdict = outcome.growthVerdict.rawValue
    }
}

// MARK: - One slice

/// One slice's whole life: what it was seeded with, what it ran, what it
/// handed to the merge.
struct TrainerCensusSlice: Codable {
    var index: Int = 0
    var label: String = ""
    var keyframesTrained: Int = 0
    var keyframesHeldOut: Int = 0

    // --- Seeding ------------------------------------------------------------

    /// Either a sidecar path (the pre-pass's own set) or "native depth maps".
    /// The two mean different things by "rejected", which is why the alert
    /// rules below only judge the rejection rate of the second.
    var seedSource: String = ""
    var seedFramesUsed: Int = 0
    var seedSamplesConsidered: Int = 0
    var seedSamplesRejected: Int = 0
    /// How many seeds were laid as SOLID DISCS across the surface because the
    /// depth sample was trusted, and how many were STRETCHED ALONG THE VIEWING
    /// RAY because it was not. A run where the second number is everything is
    /// a run whose trust gate is rejecting realistic input.
    var seedsPinnedAsDiscs: Int = 0
    var seedsStretchedAlongRay: Int = 0
    var seedsBuilt: Int = 0
    /// What actually reached the GPU. Lower than `seedsBuilt` means the buffer
    /// could not hold them all.
    var seedsUploaded: Int = 0
    /// True once `seedsUploaded` above has actually been written by the upload
    /// step. A slice row is opened BEFORE seeding so that a slice which throws
    /// still leaves a row behind, which means an untouched `seedsUploaded` is a
    /// default and not a measurement. Everything downstream that would
    /// otherwise report that default as "the build started with 0 points"
    /// checks this first.
    var seedsUploadedCounted: Bool = false
    /// Optional because 0 mm is not a spacing, it is the absence of one.
    /// `TrainerInitializer` returns 0 when it could not measure a nearest
    /// neighbour at all, and a stored 0 here read back into a sentence would
    /// say the starting points were on top of each other. nil reads as "not
    /// measured", which is what happened.
    var seedMedianSpacingMillimetres: Int?

    // --- The trust line the seeder drew, and the data it drew it on ---------

    /// Copied from `TrainerSeedResult.trustCut`, which the depth-map seeding
    /// path fills in and the pre-pass path leaves nil (that path takes no cut
    /// of its own; `PrePassCensus.seeding` already records the one it did
    /// take). nil here therefore means "this slice took no cut", never "the
    /// cut was zero".
    ///
    /// `seedTrustWasMeasured == false` is a THIRD state and the reason these
    /// exist: it means the scan had no trust field, so no gate was consulted
    /// at all. Without it, "the gate rejected every seed" and "there was no
    /// gate" look identical on screen, and the first sends someone hunting a
    /// threshold that was never read.
    var seedTrustWasMeasured: Bool?
    var seedTrustCut: Float?
    var seedTrustFloor: Float?
    var seedTrustQuantile: Float?
    var seedTrustP05: Float?
    var seedTrustMedian: Float?
    var seedTrustP95: Float?
    var seedTrustCellsConsidered: Int?

    // --- The run ------------------------------------------------------------

    var splatCapAsked: Int = 0
    var splatCapMeasuredAffordable: Int = 0
    var splatCapEffective: Int = 0
    var renderWidth: Int = 0
    var renderHeight: Int = 0
    var iterationsRequested: Int = 0
    var iterationsCompleted: Int = 0
    /// Iterations that took no optimisation step, each for a stated reason.
    /// These are silent skips in the loop, counted here rather than left to be
    /// inferred from a gap in the wall clock.
    var iterationsSkippedNoSupervision: Int = 0
    var iterationsSkippedGrowingTileBuffer: Int = 0

    /// The largest number of (Gaussian, tile) pairs any single iteration of
    /// this slice produced, and the live population at that moment.
    ///
    /// These exist to answer one question with a measurement instead of a
    /// guess: how many tiles does a Gaussian actually touch? The sort
    /// buffers are sized at EIGHT instances per splat, which is 128 of the
    /// 680 bytes each Gaussian costs, and nobody has ever checked whether 8
    /// is right. A median splat in a finished model is about 1.09 cm across
    /// at a 1.29 m stand-off rendered at 720 px with 16x16 tiles, which
    /// suggests most cover ONE tile and that the multiplier is defensive
    /// rather than measured.
    ///
    /// The ratio of these two numbers is that answer. If it is near 1, the
    /// multiplier can come down a long way and every Gaussian gets cheaper.
    /// The count was already computed every iteration and thrown away.
    var peakTileInstances: Int = 0
    var splatCountAtPeakTileInstances: Int = 0
    var iterationsSkippedNothingToRender: Int = 0
    /// Iterations that ran a real forward, backward and Adam step, as opposed
    /// to times round the loop. Measured directly by the loop rather than
    /// inferred, so it can be checked against `iterationsCompleted` minus the
    /// three skip counters above: if the two ever disagree, a skip path
    /// stopped being counted.
    var iterationsWithGradientStep: Int = 0

    // --- How much of each frame the laser actually got a vote on -------------

    /// Summed over the frames below, NOT over every iteration: a frame whose
    /// photo would not decode never reached the loss and is not in either
    /// total. `depthSupervisionFramesMeasured` is the divisor, and it is
    /// stored rather than assumed so that a mean is only ever printed when
    /// there was something to take a mean of.
    ///
    /// This pair is the denominator the depth loss now rests on:
    /// `trainer_loss_depth` divides its five geometry terms by the supervised
    /// count to make them per-sample means. A supervised fraction near zero
    /// means the laser had almost no say in a scan that was meant to be
    /// LiDAR-led, and nothing else on this page would show it.
    var depthSamplesPerFrameTotal: Int = 0
    var depthSamplesSupervisedTotal: Int = 0
    var depthSupervisionFramesMeasured: Int = 0

    var stopReason: String = TrainerCensus.unfinishedOutcome

    // --- What came out -------------------------------------------------------

    var splatCountAtEndOfTraining: Int = 0
    /// Non-finite Gaussians dropped by `readCloud` on the way out. Should
    /// always be zero; if it is not, something is producing NaNs and the prune
    /// is not catching them.
    var droppedNonFiniteOnReadback: Int = 0
    var splatsHandedToMerge: Int = 0
    var heldOutPSNR: Float?
    /// The same held-out frames, scored after a closed-form per-frame gain and
    /// bias fitted to each frame's own render and clamped to the trainer's own
    /// exposure range. Trained frames are scored WITH a fitted exposure and
    /// held-out frames without one, so the difference between this and
    /// `heldOutPSNR` is the part of the train/test gap that was never about
    /// geometry.
    var heldOutPSNRExposureFitted: Float?
    /// STRUCTURAL similarity on the held-out frames, luma, 8x8 blocks, 0 to 1.
    /// PSNR and this disagree exactly when something interesting has happened:
    /// a build can gain half a decibel of PSNR by getting the room's overall
    /// brightness right while smearing every edge, and only this notices.
    var heldOutSSIM: Float?
    /// Raw held-out PSNR per frame from the FINAL evaluation, same frames and
    /// order as `heldOutPSNR`'s mean. Lets two builds compare the frames they
    /// share like for like when their held-out sets differ.
    var heldOutPerFrame: [TrainerHeldOutFrameScore]?
    /// True when the run stopped because held-out PSNR stopped improving,
    /// rather than because it reached its iteration budget.
    var stoppedEarly: Bool = false
    /// The best held-out score any mid-run evaluation saw, and where.
    var bestHeldOutPSNR: Float?
    var bestHeldOutIteration: Int?

    /// PSNR on frames the model DID train on, measured the same way and on
    /// the same number of frames as `heldOutPSNR`, so the two can be compared
    /// directly.
    ///
    /// Without it there is no way to tell two completely different failures
    /// apart, and they need opposite fixes:
    ///
    ///   trained ~25, heldOut ~16   the model reproduces what it was shown
    ///                              and falls apart elsewhere. A coverage or
    ///                              camera-pose problem.
    ///   trained ~16, heldOut ~16   it is wrong even on frames it looked at
    ///                              three thousand times. A fitting problem.
    ///
    /// The owner's own observation is the reason this exists: a short scan of
    /// one wall, viewed from the angle it was shot from, still looked bad.
    /// That is close to a training view, which is the easy case, and it points
    /// at the second failure. This measures it rather than inferring it.
    var trainedPSNR: Float?
    /// How far training moved the cameras it was allowed to move, and whether
    /// they moved TOGETHER (world frame). Held-out frames never get a delta,
    /// so a common drift misregisters every held-out view by the same amount.
    /// The per-step clamps allow at most about 0.9 deg / 6.3 cm over a run.
    var cameraDeltaFrames: Int?
    var cameraDeltaMedianDegrees: Float?
    var cameraDeltaMaxDegrees: Float?
    var cameraDeltaMedianCentimeters: Float?
    var cameraDeltaMaxCentimeters: Float?
    var cameraDeltaCommonCentimeters: Float?
    var cameraDeltaCommonDegrees: Float?
}

// MARK: - The merge

struct TrainerCensusMerge: Codable {
    /// How many Gaussians each slice handed in, in slice order.
    var splatsInPerSlice: [Int]
    var splatsIn: Int
    var kept: Int
    /// Deleted because another slice owns the region their centre fell in.
    var droppedToAnotherOwner: Int
    var trimmedToCap: Int
    var splatsOut: Int
}

// MARK: - Alerts

/// One thing worth looking at, stated with the numbers that make it a fact
/// rather than an opinion.
struct TrainerCensusAlert: Codable {
    /// "loud" means a whole stage did nothing or most of the model was
    /// destroyed. "check" means it is worth a look but may be legitimate.
    var severity: String
    /// A stable machine-readable name, so a screen can match on it without
    /// parsing English.
    var code: String
    var detail: String
}

// MARK: - The census

/// Everything one training run did to its own geometry.
///
/// Written once, at the end of the run, to `model/train_census.json`. See
/// `docs/DATA_FORMAT.md` section 8.
/// WHERE THE TIME ACTUALLY GOES, per run, in seconds.
///
/// Until this existed the census recorded a start time and an end time and
/// nothing in between, so every claim about which part of an iteration was
/// expensive was an inference from which changes happened to help. That
/// inference was wrong repeatedly: four GPU-side optimisations measured
/// zero or negative, one CPU-side allocation fix took 17% off the run, and
/// the owner reasonably asked whether the GPU was being used at all.
///
/// The decisive pair is `gpuBusy` against `wall`. `gpuBusy` is what Metal
/// itself reports the GPU spent executing, summed over every command
/// buffer. If that is a small fraction of `wall`, the GPU is idle most of
/// the run and no amount of shader work can help; the answer is on the CPU.
///
/// `gpuWait` is NOT the same number. It is how long the CPU sat blocked in
/// waitUntilCompleted, which includes queue latency and scheduling as well
/// as execution. gpuWait much larger than gpuBusy means the cost is in
/// round trips, not in the work.
///
/// Everything here is wall-clock seconds accumulated on the training
/// thread. `supervision` and `gpuWait` are disjoint and are the two big
/// blocks; what is left over after both is everything else the CPU does,
/// which is itself a useful number. Only fields that are actually measured
/// appear here, so a zero means zero rather than "not instrumented".
/// One mid-run measurement of how well the model does on frames it is NOT
/// training on. The sequence of these is the answer to "how many rounds is
/// right", which no constant can know in advance because it depends on how
/// many views the capture has and how much parallax they carry.
struct TrainerHeldOutFrameScore: Codable {
    var frameIndex: Int
    var psnr: Float
}

struct TrainerHeldOutSample: Codable {
    var iteration: Int
    var psnr: Float
    var splatCount: Int
    /// Raw held-out PSNR (held-out frames at identity exposure from build 274).
    var psnrRaw: Float? = nil
    /// The same renders after the per-frame gain+bias least-squares fit;
    /// equal to `psnr` while selection runs on the fitted score.
    var psnrExposureFitted: Float? = nil
}

struct TrainerTimings: Codable {
    /// Building one frame of supervision on the CPU: photo decode, ground
    /// truth, background image, depth samples.
    var supervision: Double = 0
    /// Seconds the PREFETCH WORKER spent building supervision, off the
    /// critical path. `supervision` above is what the training loop itself
    /// waited for; this is what was moved out of it. Before the prefetch
    /// existed this was 0 and `supervision` carried the whole cost.
    var supervisionPrefetched: Double = 0
    /// Build 310: supervision builds served from the per-run frame cache (no
    /// decode, no sampling), and the most the cache held, in MB.
    var supervisionCacheHits: Int = 0
    var supervisionCacheMegabytes: Double = 0
    /// CPU time blocked in waitUntilCompleted, every command buffer.
    var gpuWait: Double = 0
    /// What Metal reports the GPU spent EXECUTING, summed over every
    /// command buffer. The honest measure of how busy the GPU is.
    var gpuBusy: Double = 0
    /// The periodic passes, each including its own GPU waits. These are the
    /// contents of the gap between `wall` and (`supervision` + `gpuWait`),
    /// which measured 12.9 ms per iteration on build 102 with nine separate
    /// findings claiming to live inside it. Now they can be checked instead of
    /// believed.
    var densify: Double = 0
    /// Build 320: densify passes whose GPU gather was checked against the
    /// CPU copy path, and the words that differed across them (expected 0).
    var densifyGatherChecks: Int = 0
    var densifyGatherMismatches: Int = 0
    var previewSnapshot: Double = 0
    var filterSweep: Double = 0
    /// runIteration only: seconds from buffer A's completion to buffer B's
    /// commit (instance-count readback, overflow check, CPU encode of B).
    /// The GPU is idle for all of it, so it prices merging A and B.
    var encodeStep: Double = 0
    /// Writing this frame's ground truth, background and depth samples into
    /// the shared Metal buffers.
    var upload: Double = 0
    /// The seed load: PLY parse, record build, thinning. It was in NO
    /// bucket. On scan_20260906_164840 (build 250) 4.03 s of a 77 s training
    /// wall sat in no bucket at all, and this is the largest suspect.
    var prologue: Double = 0
    /// Loading the SMART layer: trust fields, authority, edges, background,
    /// carver. Build 256 still had 3.9 s of training wall in no bucket, and
    /// this is the largest thing between the census opening and the loop
    /// that no clock covered (the pipelines are compiled before it opens).
    var smartLayer: Double = 0
    /// Build 314: the parts of `smartLayer`, each timed on its own. The
    /// trust fields and the free-space map load beside the edge maps, so
    /// these can add up to more than the total.
    var smartLayerTrust: Double = 0
    var smartLayerEdges: Double = 0
    var smartLayerAuthority: Double = 0
    var smartLayerBackground: Double = 0
    var smartLayerCarver: Double = 0

    /// The one command buffer that holds the sort, the forward raster, the
    /// losses, the backward raster and the optimiser. It used to be labelled
    /// "the tile sort", which credited all five stages to `gpuSort` and left
    /// `gpuForward`, `gpuLosses`, `gpuBackward` and `gpuOptimiser` reading
    /// exactly 0.00 in every census. Those four fields still exist and still
    /// work, but they only fill when that buffer is deliberately split into
    /// one buffer per stage for a diagnostic run; the rest of the time this
    /// is the number, and `gpuSort` means the sort alone.
    var gpuStep: Double = 0
    /// Scoring the held-out frames DURING training, to decide when to stop.
    /// Separate from everything else because it is the one cost the run pays
    /// purely to find out whether it should still be running.
    var earlyStopEval: Double = 0

    /// GPU execution split by which command buffer it was in, so the 25.5 ms
    /// the GPU now spends per iteration stops being one opaque number.
    ///
    /// `gpuScan` is command buffer A: clear, preprocess, the tiles-touched
    /// exclusive scan. `gpuSort` is command buffer B, which is everything
    /// else in an iteration: duplicate keys, the eight-pass radix sort, tile
    /// ranges, the forward rasteriser, the losses, the backward rasteriser and
    /// Adam. `gpuOther` is the periodic work, the filter sweep and the resets.
    ///
    /// Coarse on purpose. Per-kernel timing needs counter sample buffers, and
    /// this needs no new Metal objects at all because every command buffer
    /// already passes through `finish` with its stage name. If gpuSort is
    /// almost all of it, which is the expectation, the next question is which
    /// kernel inside it, and THAT is worth the counter buffers.
    var gpuScan: Double = 0
    var gpuSort: Double = 0
    var gpuOther: Double = 0

    /// Command buffer B, opened up. It measured 23.21 ms of a 25.29 ms GPU
    /// iteration, 92% of all GPU time and 76% of the whole run, as ONE number,
    /// which is not something anything can be done about.
    ///
    /// DIAGNOSTIC, AND IT COSTS SOMETHING. Getting these five numbers means
    /// five command buffers where there was one, and the measured overhead of
    /// a commit-and-wait on this device is about 0.29 ms, so roughly 1.2 ms
    /// per iteration of the thing being measured. That is a deliberate trade
    /// for one run, and the split comes back out once it has told us which
    /// kernel to go after.
    var gpuForward: Double = 0
    var gpuLosses: Double = 0
    var gpuBackward: Double = 0
    var gpuOptimiser: Double = 0
    /// Build 286: the five buckets above are filled again, but only on SAMPLED
    /// iterations (every `stageProfileEvery`), where command buffer B is split
    /// five ways. This is how many were sampled: each bucket divided by it is
    /// that stage's GPU milliseconds per iteration. The sampled iterations'
    /// time is NOT in gpuStep.
    var profiledSteps: Int = 0

    /// Backward calibration (build 288): on `backwardCalibrationSteps`
    /// iterations the plain backward rasteriser (A) and the SIMD-summed one
    /// (B) ran on the same inputs. GPU seconds of each summed over those
    /// iterations, the worst relative L1 difference of their gradients (capped
    /// at 1e9 so the census always encodes), and whether B was then used for
    /// the rest of the run (1) or not (0).
    var backwardCalibrationSteps: Int = 0
    var backwardSecondsA: Double = 0
    var backwardSecondsB: Double = 0
    var backwardRelativeDifference: Double = 0
    var backwardSimdSumChosen: Int = 0

    /// Sort calibration (build 290, reworked in 306). Three sorts of the same
    /// frame: the legacy six-pass sort (L, `sortLegacySeconds`), the
    /// splat-order sort with the plain scatter (A, `sortSecondsA`, key
    /// generation included) and with the SIMD-prefix scatter (B,
    /// `sortSecondsB`). Mismatch counts are steps whose order differed from
    /// L's (must be 0: all three are integer rankings). `sortSplatOrderChosen`
    /// 1 means the splat-order sort was used for the rest of the run.
    var sortLegacySeconds: Double = 0
    var sortSplatOrderMismatchSteps: Int = 0
    var sortSplatOrderChosen: Int = 0
    var sortCalibrationSteps: Int = 0
    var sortSecondsA: Double = 0
    var sortSecondsB: Double = 0
    var sortMismatchSteps: Int = 0
    var sortSimdScanChosen: Int = 0

    /// Forward calibration (build 302): the one-pixel rasteriser (A) against
    /// the two-pixel one (B) on the same frame. GPU seconds of each summed, the
    /// largest absolute output difference (capped at 1e9), steps that
    /// disagreed, and whether B was then used (1) or not (0).
    var forwardCalibrationSteps: Int = 0
    var forwardSecondsA: Double = 0
    var forwardSecondsB: Double = 0
    var forwardMaxDifference: Double = 0
    var forwardMismatchSteps: Int = 0
    var forwardTwoPixelChosen: Int = 0

    /// Two-pixel backward calibration (build 304): the backward chosen above
    /// (A) against the two-pixel one (B; plain atomics since 312), same fields as the
    /// first backward calibration.
    var backwardTwoPixelCalibrationSteps: Int = 0
    var backwardTwoPixelSecondsA: Double = 0
    var backwardTwoPixelSecondsB: Double = 0
    var backwardTwoPixelRelativeDifference: Double = 0
    var backwardTwoPixelChosen: Int = 0

    /// Blur calibration (build 318): the two-pass SSIM blur (A) against the
    /// fused one (B) on the same frame. GPU seconds of the whole SSIM stage
    /// each way, steps whose blurred partials differed in ANY bit, and
    /// whether B was then used (1) or not (0).
    var blurCalibrationSteps: Int = 0
    var blurSecondsA: Double = 0
    var blurSecondsB: Double = 0
    var blurMismatchSteps: Int = 0
    var blurFusedChosen: Int = 0

    /// Build 292: steps whose buffer B was left running while the next
    /// iteration's CPU work and buffer A went ahead (completed later by
    /// drainPendingStep).
    var overlappedSteps: Int = 0
    /// Build 316: of those, steps run as ONE command buffer with the sort
    /// sized on the GPU, and how many of them ran short of instance slots
    /// (the sort was clamped to the capacity that step; the buffers grew
    /// before the next). Expected 0.
    var mergedSteps: Int = 0
    var truncatedInstanceSteps: Int = 0

    /// How many command buffers were waited on. gpuWait divided by this is
    /// the average round-trip cost, which is the number that says whether
    /// merging command buffers would be worth anything.
    var commandBuffers: Int = 0
}

struct TrainerCensus: Codable {

    /// Bumped only when a field changes MEANING. Adding an optional field does
    /// not bump it, which is the rule the rest of the format already uses.
    var formatVersion: Int = 1
    var scanID: ScanID

    /// WHICH BUILD TRAINED THIS. "0.1.0 (92)".
    ///
    /// Not derivable from anything else in the bundle. capture_bundle.json
    /// carries an appVersion, but that is stamped when the scan is SHOT,
    /// so a scan captured on one build and re-trained on three later ones
    /// produces four censuses all claiming the capture build. That is
    /// exactly what happened while measuring the speed work: two runs a
    /// day apart both read "0.1.0 (44)" and there was no way to tell from
    /// the file which trainer produced either of them.
    ///
    /// Timings are only comparable between runs of a KNOWN build, so this
    /// is the field that makes every other number in here mean something.
    var appVersion: String = BrandConfig.versionString

    /// Where the time went. See `TrainerTimings`.
    var timings = TrainerTimings()
    /// Held-out PSNR measured DURING the run, in order. This is the curve that
    /// says where the model stopped improving, and it is recorded whether or
    /// not early stopping is switched on, because choosing a fixed iteration
    /// budget without it is guesswork.
    var heldOutCurve: [TrainerHeldOutSample] = []

    /// HOW HOT IT GOT, AND WHEN.
    ///
    /// Everything measured so far is time per iteration. Energy per iteration
    /// is a different quantity and nothing has ever recorded it. They diverge:
    /// making the same work finish sooner draws the same joules over fewer
    /// seconds, which is more watts and more heat. Over this run's 66 s that
    /// is free. Over the 11 minutes a 30,000-iteration run would take at the
    /// current speed, it is the whole problem.
    ///
    /// The census already records `budgetReductions`, but only AFTER the
    /// governor has degraded something, which is the last event in the story
    /// rather than the first. This is the trajectory: seconds spent at each
    /// thermal level, and the iteration at which the run first reached each
    /// one. A run that spends its second half at `serious` is not the same run
    /// as one that never leaves `nominal`, even when both finish in 66 s.
    struct Thermals: Codable, Sendable, Equatable {
        /// Seconds at each level, indexed by ThermalLevel's raw value:
        /// nominal, fair, serious, critical.
        public var secondsAtLevel: [Double] = [0, 0, 0, 0]
        /// The iteration each level was first observed at, -1 if never.
        public var firstReachedAtIteration: [Int] = [-1, -1, -1, -1]
        /// The worst level seen at any point.
        public var peak: Int = 0

        public init() {}
    }

    var thermals = Thermals()

    var startedAt: Date
    var finishedAt: Date?
    /// "completed", "cancelled", "stopped early" or "failed: <reason>". The
    /// default is what a run that vanished mid-flight leaves behind, which is
    /// more honest than a blank.
    var outcome: String = TrainerCensus.unfinishedOutcome

    /// What `outcome` and `TrainerCensusSlice.stopReason` say until something
    /// overwrites them. Named once so a reader cannot mistake a run that died
    /// silently for one that finished.
    static let unfinishedOutcome = "did not reach the end"

    var budgetRequested: TrainerCensusBudget
    var budgetAsRun: TrainerCensusBudget
    var budgetReductions: [TrainerCensusBudgetReduction] = []
    var gates: TrainerCensusGates?

    var keyframesSelected: Int = 0
    /// The frame-index span the chosen keyframes cover, against the capture's
    /// length. The selector walks frames in order and stops at its target, so
    /// on the owner's 868-frame scan the set ended near frame 485 and the
    /// rest of the walk never trained. Build 264's selector change moved that
    /// end (visible only through held_out_frames.json) and cost 0.7 dB. These
    /// make the span a number instead of a reconstruction.
    var keyframeFirstIndex: Int = -1
    var keyframeLastIndex: Int = -1
    var framesInBundle: Int = 0
    var sliceCount: Int = 0
    var slices: [TrainerCensusSlice] = []
    var densifyPasses: [TrainerCensusDensifyPass] = []
    var merge: TrainerCensusMerge?

    var iterationsRequested: Int
    var iterationsCompleted: Int = 0
    var finalSplatCount: Int = 0

    /// Frames whose authority map was built during this run, and how many of
    /// those were at least half CONFIRMED GLASS.
    ///
    /// A confirmed pane multiplies depth authority by zero, so a frame that is
    /// mostly window hands the trainer almost no geometry however good the
    /// capture was. `SmartAuthorityMap` measured that per frame and nothing
    /// carried it anywhere, so a scan of a conservatory could lose most of its
    /// geometry with the census showing only a small final splat count and no
    /// reason for it.
    ///
    /// Both are optional because a run with no SMART authority map measured
    /// neither, and reporting 0 there would claim a measurement that never
    /// happened. Always read them as a pair: the count alone cannot say
    /// whether it is a catastrophe or a footnote.
    var authorityFramesBuilt: Int?
    var glassDominatedFrames: Int?

    /// Whether the 4.5 to 30 m depth band was MEASURED on this run, and in one
    /// sentence what produced it.
    ///
    /// On the phone the monocular depth model is a stub: it reports itself
    /// unavailable and returns nothing, so that band is routed around with
    /// parallax plus the far field rather than measured. A run that fell back
    /// produced different geometry from one that did not, and until this was
    /// recorded no scan said which it got.
    ///
    /// Optional because a run with no background model measured neither.
    var midRegimeIsReal: Bool?
    var midRegimeProvenance: String?

    /// The five second read: one ordered line per stage, from the seeds to the
    /// final count, so a reader sees WHERE the geometry went without adding
    /// anything up. Filled in by `sealed()`.
    var ledger: [String] = []
    /// What is worth looking at, worst first. Filled in by `sealed()`.
    var alerts: [TrainerCensusAlert] = []

    init(scanID: ScanID, requested: TrainingBudget, startedAt: Date = Date()) {
        self.scanID = scanID
        self.startedAt = startedAt
        self.budgetRequested = TrainerCensusBudget(requested)
        self.budgetAsRun = TrainerCensusBudget(requested)
        self.iterationsRequested = requested.iterations
    }

    // MARK: Totals

    /// Plain sums over the recorded passes. Nothing here is stored; it is
    /// recomputed on demand so it can never disagree with the rows above it.
    struct Totals {
        var passes = 0
        var addedBySplit = 0
        var addedByClone = 0
        var relocated = 0
        var prunedNonFinite = 0
        var prunedLowOpacity = 0
        var prunedOversized = 0
        var carved = 0
        var trimmedToCap = 0
        var passesWithGrowthWindowOpen = 0
        var passesWithGrowthWindowOpenAndHeadroom = 0
        var passesWithPruneWindowOpen = 0
        var passesThatPrunedByJudgement = 0
        var passesThatPrunedByJudgementOutsideWindow = 0
        var largestHeadroomWhileGrowthWasAllowed = 0
        /// Passes where pruning of ANY kind removed at least one Gaussian,
        /// non-finite included. The denominator that belongs with
        /// `prunedAtAll`: a pass count taken over one definition of pruning and
        /// a deleted count taken over another is two numbers that look like a
        /// pair and are not.
        var passesThatPrunedAnything = 0
        /// How many Gaussians densification actually looked at as candidates,
        /// summed over every pass. Zero candidates across many passes is a
        /// scoring or units fault; many candidates and nothing created is a
        /// creation fault, and the two need telling apart.
        var densifyCandidates = 0
        /// The longest run of CONSECUTIVE passes that were allowed to add, had
        /// room under the cap, and added nothing. Counted within a slice: two
        /// slices are two separate populations and a streak must not be
        /// stitched across the join.
        ///
        /// Alert 1 only fires when NOTHING was ever created anywhere in the
        /// run, so a run that densified normally and then stopped for two
        /// thousand iterations passes it cleanly. This is the number that
        /// catches that.
        var longestZeroGrowthStreak = 0

        var added: Int { addedBySplit + addedByClone }
        var prunedByJudgement: Int { prunedLowOpacity + prunedOversized }
        /// Everything the prune step removed, however it decided. Carving and
        /// the cap trim are separate steps and are deliberately not in here.
        var prunedAtAll: Int { prunedNonFinite + prunedByJudgement }
    }

    var totals: Totals {
        var t = Totals()
        var streak = 0
        var streakSlice: Int?
        for pass in densifyPasses {
            // The streak is per slice. `densifyPasses` is one flat array in
            // append order, so the slice index changing is the join.
            if streakSlice != pass.sliceIndex {
                streak = 0
                streakSlice = pass.sliceIndex
            }
            if pass.growthWindowOpen, pass.headroom > 0 {
                if pass.addedBySplit + pass.addedByClone > 0 {
                    streak = 0
                } else {
                    streak += 1
                    t.longestZeroGrowthStreak = Swift.max(t.longestZeroGrowthStreak, streak)
                }
            }
            t.passes += 1
            t.addedBySplit += pass.addedBySplit
            t.addedByClone += pass.addedByClone
            t.relocated += pass.relocated
            t.prunedNonFinite += pass.prunedNonFinite
            t.prunedLowOpacity += pass.prunedLowOpacity
            t.prunedOversized += pass.prunedOversized
            t.carved += pass.carvedFromEmptySpace
            t.trimmedToCap += pass.trimmedToCap
            if pass.growthWindowOpen {
                t.passesWithGrowthWindowOpen += 1
                if pass.headroom > 0 {
                    t.passesWithGrowthWindowOpenAndHeadroom += 1
                    t.largestHeadroomWhileGrowthWasAllowed = Swift.max(
                        t.largestHeadroomWhileGrowthWasAllowed, pass.headroom
                    )
                }
            }
            if pass.pruneWindowOpen { t.passesWithPruneWindowOpen += 1 }
            if pass.prunedLowOpacity + pass.prunedOversized > 0 {
                t.passesThatPrunedByJudgement += 1
                if !pass.pruneWindowOpen { t.passesThatPrunedByJudgementOutsideWindow += 1 }
            }
            if pass.prunedNonFinite + pass.prunedLowOpacity + pass.prunedOversized > 0 {
                t.passesThatPrunedAnything += 1
            }
            t.densifyCandidates += pass.candidatesAfterVisibilityFilter
        }
        return t
    }

    /// True once at least one slice has put its seeds on the GPU, which is the
    /// moment training genuinely began.
    ///
    /// Every loop-derived number below is absent until this is true, because a
    /// run that died during allocation has a `densifyPasses` array that is
    /// empty for a reason that has nothing to do with densification. Writing
    /// "0 points created" for that run would be the exact fault this whole
    /// file exists to stop: a number nobody measured, presented as a fact.
    var trainingBegan: Bool {
        slices.contains { $0.seedsUploadedCounted }
    }

    var seedsBuiltTotal: Int { slices.reduce(0) { $0 + $1.seedsBuilt } }
    var seedsUploadedTotal: Int { slices.reduce(0) { $0 + $1.seedsUploaded } }
    var seedsPinnedAsDiscsTotal: Int { slices.reduce(0) { $0 + $1.seedsPinnedAsDiscs } }
    var seedsStretchedAlongRayTotal: Int { slices.reduce(0) { $0 + $1.seedsStretchedAlongRay } }
    var droppedNonFiniteOnReadbackTotal: Int {
        slices.reduce(0) { $0 + $1.droppedNonFiniteOnReadback }
    }
    var iterationsSkippedTotal: Int {
        slices.reduce(0) {
            $0 + $1.iterationsSkippedNoSupervision
                + $1.iterationsSkippedGrowingTileBuffer
                + $1.iterationsSkippedNothingToRender
        }
    }

    // MARK: Sealing

    /// Fills in the ledger and the alerts and stamps the finish time. Call it
    /// once, immediately before writing. Everything it computes is arithmetic
    /// over numbers already in the struct, so it costs nothing and calling it
    /// twice changes nothing.
    func sealed(at when: Date = Date()) -> TrainerCensus {
        var out = self
        out.finishedAt = when
        out.ledger = buildLedger()
        out.alerts = buildAlerts()
        return out
    }

    private func buildLedger() -> [String] {
        let t = totals
        let n: (Int) -> String = TrainerCensusFormat.count
        var lines: [String] = []

        lines.append("seeded \(n(seedsBuiltTotal)) starting points")
        if seedsUploadedTotal != seedsBuiltTotal {
            lines.append("uploaded \(n(seedsUploadedTotal)) of them to the GPU")
        }
        let discs = seedsPinnedAsDiscsTotal
        let stretched = seedsStretchedAlongRayTotal
        if discs + stretched > 0 {
            lines.append(
                "of those, \(n(discs)) were laid as solid discs and \(n(stretched)) "
                    + "were stretched along the viewing ray"
            )
        }
        // Said before densification, because it explains what densification
        // had to work WITH. Only printed when the SMART authority map actually
        // ran; a run without one measured nothing and must not imply zero.
        if let built = authorityFramesBuilt, built > 0, let glassy = glassDominatedFrames {
            lines.append(
                "the depth of \(n(glassy)) of \(n(built)) frame(s) looked at was thrown away "
                    + "over most of the frame because it was confirmed glass"
            )
        }
        // Whether the middle distance was measured or routed around. Said in
        // the ledger and not only in the alerts, because it is a property of
        // how the run was done rather than something that went wrong.
        if let real = midRegimeIsReal, !real {
            lines.append(
                "depth from 4.5 to 30 m was not measured on this device and was worked "
                    + "out from camera movement instead"
            )
        }
        lines.append(
            "densification ran \(t.passes) pass(es) and added \(n(t.added)) "
                + "(\(n(t.addedBySplit)) by split, \(n(t.addedByClone)) by clone)"
        )
        if t.relocated > 0 {
            lines.append("relocated \(n(t.relocated)), which does not change the count")
        }
        lines.append(
            "pruning removed \(n(t.prunedNonFinite + t.prunedByJudgement)) "
                + "(\(n(t.prunedNonFinite)) non-finite, \(n(t.prunedLowOpacity)) too faint, "
                + "\(n(t.prunedOversized)) too big)"
        )
        // How many passes were allowed to prune by judgement, and how many
        // actually did. Bug 3 was pruning running in 29 passes when the
        // configured window covers about 10, so the two counts belong side by
        // side rather than one being inferred from the other.
        lines.append(
            "the prune window was open in \(t.passesWithPruneWindowOpen) of \(t.passes) "
                + "pass(es), and faint-or-oversized pruning removed something in "
                + "\(t.passesThatPrunedByJudgement)"
        )
        // "No occupancy grid" and "a grid that licensed no deletions" are two
        // different runs and both come out as `carved == 0`. `carverAvailable`
        // is already recorded per pass, so this is naming a number that
        // exists rather than measuring a new one.
        if t.carved == 0, t.passes > 0 {
            let withGrid = densifyPasses.filter { $0.carverAvailable }.count
            if withGrid == 0 {
                lines.append(
                    "free-space carving removed 0: there was no map of empty air to carve from"
                )
            } else {
                lines.append(
                    "free-space carving removed 0, from a map of empty air that was loaded for "
                        + "\(withGrid) of \(t.passes) pass(es)"
                )
            }
        } else {
            lines.append("free-space carving removed \(n(t.carved))")
        }
        if t.trimmedToCap > 0 {
            lines.append("the cap trimmed \(n(t.trimmedToCap))")
        }
        if droppedNonFiniteOnReadbackTotal > 0 {
            lines.append(
                "dropped \(n(droppedNonFiniteOnReadbackTotal)) non-finite when reading back"
            )
        }
        if let merge = merge, sliceCount > 1 {
            lines.append(
                "merging \(sliceCount) parts dropped \(n(merge.droppedToAnotherOwner)) "
                    + "to another part's region and trimmed \(n(merge.trimmedToCap))"
            )
        }
        lines.append(
            "ran \(iterationsCompleted) of \(iterationsRequested) requested iterations"
        )
        lines.append("final model: \(n(finalSplatCount)) points")
        return lines
    }

    /// How many consecutive empty densification passes count as a stall
    /// rather than a quiet stretch.
    ///
    /// The training loop has its own copy of this number
    /// (`MetalSplatTrainer.zeroGrowthPassesBeforeSaying`, also 10) because
    /// that one decides when to shout DURING a run and this one decides what
    /// the finished census says. They are the same value on purpose and the
    /// two are checked against each other by nothing, so if one moves, move
    /// the other.
    static let zeroGrowthStreakThatIsAStall = 10

    /// The share of looked-at frames that has to be glass-dominated before the
    /// census says so. A third: below that a few windows in a room is normal
    /// and saying it every time would train the reader to skip the alerts.
    static let glassDominatedRunPercent = 33

    /// The verdict that appears most often across a set of passes, or nil when
    /// there is nothing to report. It never invents one: every value here was
    /// written by the densifier from counters that pass actually measured.
    private static func dominantVerdict(_ passes: [TrainerCensusDensifyPass]) -> String? {
        var tally: [String: Int] = [:]
        for pass in passes { tally[pass.growthVerdict, default: 0] += 1 }
        return tally.max(by: { $0.value < $1.value })?.key
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private func buildAlerts() -> [TrainerCensusAlert] {
        var loud: [TrainerCensusAlert] = []
        var check: [TrainerCensusAlert] = []
        let t = totals
        let n: (Int) -> String = TrainerCensusFormat.count

        // 1. A WHOLE STAGE DID NOTHING. Bug 1's exact signature: growth was
        //    permitted, there was room for it, and not one Gaussian appeared.
        if t.passesWithGrowthWindowOpenAndHeadroom > 0, t.added == 0 {
            let openPasses = densifyPasses.filter { $0.growthWindowOpen }
            let scored = openPasses.reduce(0) { $0 + $1.splatsWithNonZeroScore }
            let examined = openPasses.reduce(0) { $0 + $1.splatsScored }
            // The densifier's own verdict, not the census guessing at it from
            // the counters. "Not one point had a gradient above zero" and
            // "255,035 scored and none of them was visible in any frame" send
            // someone to two completely different places, and the two counts
            // above cannot tell them apart.
            var verdict = ""
            if let dominant = Self.dominantVerdict(openPasses) {
                verdict = " Verdict on most of those passes: " + dominant + "."
            }
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "densification_created_nothing",
                    detail: "Growth was allowed in \(t.passesWithGrowthWindowOpenAndHeadroom) "
                        + "pass(es) with room for up to "
                        + "\(n(t.largestHeadroomWhileGrowthWasAllowed)) more points, and created "
                        + "0. Across those passes \(n(scored)) of \(n(examined)) points scored "
                        + "above zero. A densification stage that adds nothing is a no-op, not a "
                        + "quiet run." + verdict
                )
            )
        }

        // 1a. MOST OF WHAT THE TRAINER LOOKED AT WAS WINDOW. Not a fault in
        //     the code and not a fault in the capture: a confirmed pane
        //     multiplies depth authority by zero, so a room that is mostly
        //     glass genuinely hands the trainer very little to build on. It is
        //     a "check", not a "loud", and it is stated so that a thin model
        //     from a conservatory reads as a measured cause rather than as an
        //     unexplained small number.
        if let built = authorityFramesBuilt, built > 0,
           let glassy = glassDominatedFrames,
           glassy * 100 >= built * Self.glassDominatedRunPercent {
            let percent = glassy * 100 / built
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "glass_dominated_frames",
                    detail: "\(n(glassy)) of \(n(built)) frame(s) the trainer looked at were at "
                        + "least half confirmed glass (\(percent) per cent of them), so their "
                        + "depth was ignored over most of the frame. Expect less geometry than "
                        + "the number of photos suggests."
                )
            )
        }

        // 1b. DENSIFICATION STOPPED PART WAY THROUGH. Alert 1 only fires when
        //     nothing was created ANYWHERE in the run, so a stage that worked
        //     for a while and then went dead for two thousand iterations
        //     passes it cleanly. This is that case.
        if t.longestZeroGrowthStreak >= Self.zeroGrowthStreakThatIsAStall, t.added > 0 {
            var detail = "Densification went "
            detail += String(t.longestZeroGrowthStreak)
            detail += " consecutive pass(es) adding nothing while it was allowed to add and had "
            detail += "room under the cap. It created "
            detail += n(t.added)
            detail += " points in total, so the stage was working and then stopped."
            let stalled = densifyPasses.filter {
                $0.growthWindowOpen && $0.headroom > 0 && $0.addedBySplit + $0.addedByClone == 0
            }
            if let dominant = Self.dominantVerdict(stalled) {
                detail += " Verdict on most of the empty passes: "
                detail += dominant
                detail += "."
            }
            loud.append(
                TrainerCensusAlert(
                    severity: "loud", code: "densification_stalled", detail: detail
                )
            )
        }

        // 2. GROWTH WAS NEVER EVEN OFFERED. The window never opened, which is
        //    a schedule problem rather than a signal problem.
        if t.passes > 0, t.passesWithGrowthWindowOpen == 0 {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "growth_window_never_open",
                    detail: "\(t.passes) densification pass(es) ran and the growth window was "
                        + "shut for every one. Compare densifyStartFraction and "
                        + "densifyEndFraction in the gates block against the progressFraction of "
                        + "each pass."
                )
            )
        }

        // 3. NO ROOM TO GROW, EVER. This is what a splat cap sitting at or
        //    below the live population leaves behind.
        if t.passesWithGrowthWindowOpen > 0, t.passesWithGrowthWindowOpenAndHeadroom == 0 {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "no_headroom_whenever_growth_was_allowed",
                    detail: "The growth window was open in \(t.passesWithGrowthWindowOpen) "
                        + "pass(es) and the headroom (cap minus live count) was 0 in every one, "
                        + "so nothing could be created."
                )
            )
        }

        // 4. THE ONE-WAY RATCHET. A cap cut that lands at or below the live
        //    population deletes real geometry AND zeroes the headroom.
        for reduction in budgetReductions where reduction.landedAtOrBelowLivePopulation {
            let deleted = Swift.max(reduction.liveSplatCount - reduction.to, 0)
            if deleted > 0 {
                // Below the population: real Gaussians were trimmed to fit and
                // the GPU buffer was rebuilt smaller. Unrecoverable.
                loud.append(
                    TrainerCensusAlert(
                        severity: "loud",
                        code: "splat_cap_cut_below_live_population",
                        detail: "At iteration \(reduction.atIteration) the splat cap was cut from "
                            + "\(n(reduction.from)) to \(n(reduction.to)) because "
                            + "\(reduction.reason), while \(n(reduction.liveSplatCount)) points "
                            + "were alive. That deleted \(n(deleted)) real points and left no "
                            + "headroom for densification afterwards."
                    )
                )
            } else {
                // Exactly AT the population: nothing was deleted, which is
                // what the fixed thermal ladder is meant to do, but the
                // headroom is still zero and growth is over until something
                // frees space. Worth seeing, not worth shouting about.
                check.append(
                    TrainerCensusAlert(
                        severity: "check",
                        code: "splat_cap_cut_to_exactly_the_live_population",
                        detail: "At iteration \(reduction.atIteration) the splat cap was cut from "
                            + "\(n(reduction.from)) to \(n(reduction.to)) because "
                            + "\(reduction.reason), which is exactly the live count. Nothing was "
                            + "deleted, but the headroom for densification is now 0."
                    )
                )
            }
        }

        // 5. A GATE THAT IS NOT WIRED. Judgement pruning outside the window it
        //    is supposed to obey means the window is not gating anything.
        if t.passesThatPrunedByJudgementOutsideWindow > 0, let gates = gates {
            let start = String(format: "%.2f", gates.pruneStartFraction)
            let end = String(format: "%.2f", gates.pruneEndFraction)
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "judgement_pruning_outside_its_window",
                    detail: "Faint-or-oversized pruning ran in "
                        + "\(t.passesThatPrunedByJudgementOutsideWindow) pass(es) that fell "
                        + "outside the configured window of \(start) to \(end) of the run. The "
                        + "window is declared but is not gating anything."
                )
            )
        }

        // 6. A THRESHOLD THAT REJECTS EVERYTHING. Only the depth-map seeding
        //    path is judged here: on the pre-pass path "rejected" counts the
        //    spatial thinning down to the cap, which is meant to be large.
        for slice in slices where slice.seedSource == TrainerInitializer.depthSeedSourceName {
            guard slice.seedSamplesConsidered > 0 else { continue }
            let percent = Float(slice.seedSamplesRejected) * 100
                / Float(slice.seedSamplesConsidered)
            guard percent >= 95 else { continue }
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "seeding_rejected_almost_every_sample",
                    detail: "Part \(slice.index + 1) rejected \(n(slice.seedSamplesRejected)) of "
                        + "\(n(slice.seedSamplesConsidered)) depth samples, which is "
                        + String(format: "%.1f", percent)
                        + " percent. A gate that rejects nearly everything is a misplaced "
                        + "threshold, not a high standard."
                )
            )
        }

        // 7. EVERY SEED A BLOB. The seeder's trust gate, made visible.
        let discs = seedsPinnedAsDiscsTotal
        let stretched = seedsStretchedAlongRayTotal
        // "The gate rejected everything" and "there was no gate" produce the
        // same zero. Only the first is a threshold to go and look at, and
        // sending someone to look at a threshold that was never consulted is
        // the exact waste this file exists to prevent.
        // Written as three plain lines rather than one expression with two
        // trailing closures and a negation in it. That shape is the one the
        // Swift type checker gives up on, and CI is this project's only
        // compiler.
        let someSliceConsultedAGate = slices.contains(where: { $0.seedTrustWasMeasured == true })
        let someSliceHadNoTrustField = slices.contains(where: { $0.seedTrustWasMeasured == false })
        let noTrustFieldAnywhere = someSliceHadNoTrustField && !someSliceConsultedAGate
        if discs + stretched > 0, discs == 0, noTrustFieldAnywhere {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "no_depth_reliability_to_trust",
                    detail: "0 of \(n(discs + stretched)) seeds were laid as solid discs across "
                        + "the surface. No gate rejected them: this scan carried no "
                        + "depth-reliability measurements at all, so every starting point was "
                        + "laid the cautious way. Look at the pre-pass trust stage, not at a "
                        + "threshold."
                )
            )
        } else if discs + stretched > 0, discs == 0 {
            var detail = "0 of \(n(discs + stretched)) seeds were laid as solid discs across "
            detail += "the surface; every one was stretched along the viewing ray because its "
            detail += "depth sample was not trusted. A handheld scan is normally a mixture."
            // The cut held up against the distribution it was applied to. A
            // cut at its floor with a 95th percentile below it is the exact
            // fingerprint of a gate nothing could ever clear.
            for slice in slices {
                guard let cut = slice.seedTrustCut, let p95 = slice.seedTrustP95 else { continue }
                detail += " Part \(slice.index + 1) drew the line at "
                detail += String(format: "%.3f", cut)
                detail += " and the most reliable reading in the whole scan was "
                detail += String(format: "%.3f", p95)
                detail += "."
            }
            loud.append(
                TrainerCensusAlert(
                    severity: "loud", code: "no_seed_was_trusted", detail: detail
                )
            )
        } else if discs + stretched > 0, Float(discs) / Float(discs + stretched) < 0.05 {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "almost_no_seed_was_trusted",
                    detail: "\(n(discs)) of \(n(discs + stretched)) seeds were trusted enough to "
                        + "be laid as solid discs, which is under 5 percent."
                )
            )
        }

        // 8. WHERE THE POPULATION ACTUALLY WENT. A model that ends far below
        //    what it started with is worth naming even when no single stage
        //    looks wrong on its own.
        let started = seedsUploadedTotal
        if started > 0, finalSplatCount < started / 2 {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "most_of_the_model_disappeared",
                    detail: "Started with \(n(started)) points and finished with "
                        + "\(n(finalSplatCount)). Pruning removed "
                        + "\(n(t.prunedNonFinite + t.prunedByJudgement)), carving removed "
                        + "\(n(t.carved)), the cap trimmed \(n(t.trimmedToCap)), and "
                        + "densification added \(n(t.added))."
                )
            )
        }
        if budgetAsRun.splatCap > 0, finalSplatCount < budgetAsRun.splatCap / 4, t.added == 0 {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "final_count_far_below_the_cap",
                    detail: "Finished at \(n(finalSplatCount)) points against a cap of "
                        + "\(n(budgetAsRun.splatCap)), having added none. The budget was never "
                        + "the limit here."
                )
            )
        }

        // 9. CARVING AS THE MAIN CAUSE OF DEATH.
        let removedTotal = t.prunedNonFinite + t.prunedByJudgement + t.carved + t.trimmedToCap
        if t.carved > 0, removedTotal > 0,
           Float(t.carved) / Float(removedTotal) > 0.5,
           t.carved > started / 4
        {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "free_space_carving_removed_the_most",
                    detail: "Carving deleted \(n(t.carved)) points, more than half of everything "
                        + "removed and over a quarter of what the run started with. A "
                        + "mis-registered occupancy grid is how a whole room disappears."
                )
            )
        }

        // 10. NON-FINITE GAUSSIANS SURVIVING TO THE READBACK.
        if droppedNonFiniteOnReadbackTotal > 0 {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "non_finite_points_reached_the_readback",
                    detail: "\(n(droppedNonFiniteOnReadbackTotal)) points were non-finite when "
                        + "the model was read off the GPU and were dropped there. The prune is "
                        + "meant to catch these in every pass."
                )
            )
        }

        // 11. THE MERGE ATE THE MODEL.
        if let merge = merge, merge.splatsIn > 0,
           Float(merge.droppedToAnotherOwner) / Float(merge.splatsIn) > 0.5
        {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "merge_dropped_most_of_the_parts",
                    detail: "The merge dropped \(n(merge.droppedToAnotherOwner)) of "
                        + "\(n(merge.splatsIn)) points because another part owned the region they "
                        + "fell in. Above half means region ownership does not match where the "
                        + "geometry actually is."
                )
            )
        }

        // 12. THE RUN WAS CUT SHORT.
        if iterationsRequested > 0, iterationsCompleted < (iterationsRequested * 95) / 100 {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "run_ended_before_its_budget",
                    detail: "Completed \(iterationsCompleted) of \(iterationsRequested) requested "
                        + "iterations. Outcome: \(outcome)."
                )
            )
        }

        // 13. ITERATIONS THAT DID NOTHING.
        //
        // Two rungs, because these are two different events. Over 5 per cent
        // is worth a look. MOST of them is not something to check: it is a
        // failed run wearing a finished run's clothes, which is precisely what
        // the first real scan was.
        let skipped = iterationsSkippedTotal
        if iterationsCompleted > 0, skipped * 2 > iterationsCompleted {
            loud.append(
                TrainerCensusAlert(
                    severity: "loud",
                    code: "most_iterations_did_no_work",
                    detail: "\(skipped) of \(iterationsCompleted) iterations took no optimisation "
                        + "step at all: no supervision could be built, the tile buffer had to "
                        + "grow, or there was nothing to render. More than half a run doing "
                        + "nothing is a failed run that finished on time, not a slow one."
                )
            )
        } else if iterationsCompleted > 0, skipped * 20 > iterationsCompleted {
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "many_iterations_did_no_work",
                    detail: "\(skipped) of \(iterationsCompleted) iterations took no optimisation "
                        + "step: no supervision could be built, the tile buffer had to grow, or "
                        + "there was nothing to render. That is over 5 percent."
                )
            )
        }

        // 14. THE TWO SIDES OF "WORK DONE" DISAGREE. `iterationsWithGradientStep`
        //     is counted by the loop; the subtraction is counted by the three
        //     skip paths. They are the same number by construction, so a
        //     mismatch means an iteration is leaving the loop by a route
        //     nothing counts, and every judgement above that rests on the skip
        //     counters is wrong by that much.
        for slice in slices where slice.iterationsCompleted > 0 {
            let derived = slice.iterationsCompleted
                - slice.iterationsSkippedNoSupervision
                - slice.iterationsSkippedGrowingTileBuffer
                - slice.iterationsSkippedNothingToRender
            guard derived != slice.iterationsWithGradientStep else { continue }
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "gradient_step_count_does_not_reconcile",
                    detail: "Part \(slice.index + 1) counted "
                        + "\(slice.iterationsWithGradientStep) iterations with a real gradient "
                        + "step, but its iteration and skip counters imply \(derived). An "
                        + "iteration is leaving the loop by a path nothing counts."
                )
            )
        }

        // 15. THE LASER BARELY GOT A VOTE. Every geometry term in the depth
        //     loss is divided by the supervised sample count, so this fraction
        //     is what says how much of each frame the depth supervision
        //     actually covered. Only judged for slices that measured it: no
        //     divisor is not a fraction of zero.
        for slice in slices where slice.depthSupervisionFramesMeasured > 0 {
            guard slice.depthSamplesPerFrameTotal > 0 else { continue }
            let percent = Double(slice.depthSamplesSupervisedTotal) * 100
                / Double(slice.depthSamplesPerFrameTotal)
            guard percent < 5 else { continue }
            check.append(
                TrainerCensusAlert(
                    severity: "check",
                    code: "almost_nothing_was_supervised_by_depth",
                    detail: "Part \(slice.index + 1) had depth supervision on "
                        + String(format: "%.1f", percent)
                        + " percent of the samples it read, over "
                        + "\(slice.depthSupervisionFramesMeasured) frame(s). This is a "
                        + "LiDAR-led build, so a figure this low means the trust, authority or "
                        + "photo-quality gates rejected nearly every reading."
                )
            )
        }

        return loud + check
    }
}

// MARK: - Number formatting

/// Thousands separators, built by hand rather than by `NumberFormatter`.
///
/// These strings go into a file and into a log line, not onto a screen, so
/// they must read the same everywhere. A locale-aware formatter would put a
/// full stop in the middle of "182.340" on a German phone and turn a
/// diagnostic into a puzzle.
enum TrainerCensusFormat {
    static func count(_ value: Int) -> String {
        let negative = value < 0
        var digits = String(value.magnitude)
        guard digits.count > 3 else { return negative ? "-\(digits)" : digits }
        var grouped = ""
        while digits.count > 3 {
            let cut = digits.index(digits.endIndex, offsetBy: -3)
            grouped = "," + String(digits[cut...]) + grouped
            digits = String(digits[..<cut])
        }
        grouped = digits + grouped
        return negative ? "-\(grouped)" : grouped
    }
}

// MARK: - The flat record the Viewer reads

/// `model/census.json`: the small, flat slice of this census that the review
/// screen knows how to read.
///
/// TWO FILES, ON PURPOSE, AND THEY ARE NOT THE SAME FILE.
/// `model/train_census.json` is this module's own diagnostic and is as detailed
/// as it likes. `model/census.json` is the shape asked for in
/// INTEGRATION_REQUESTS.md and declared as `ScanCensus.Record` in
/// `ios/Sources/Viewer/ScanCensus.swift`. That type is NOT re-declared here:
/// this is one Xcode target and a duplicate top-level name is a link error, so
/// this is a private local struct with matching key names, which is exactly
/// what that request asks for.
///
/// EVERY FIELD IS OPTIONAL AND THAT IS THE POINT. A key written as `0` is a
/// measured zero, which is a strong claim; a key left out reads as "not
/// recorded" on screen. `sharedRecord` below therefore leaves out every number
/// that would only be a default.
struct TrainerCensusSharedRecord: Codable {
    var formatVersion: Int = 1
    var writtenBy: String = "trainer"

    var splatsAtStart: Int?
    var splatsCreatedByDensification: Int?
    var densificationPassCount: Int?
    var densificationCandidateCount: Int?
    var splatsDeletedByPruning: Int?
    var pruningPassCount: Int?
    var splatsDeletedByHeatCut: Int?
    var heatCutCount: Int?
    var splatsAtEnd: Int?
    var plannedSplatCap: Int?
    var finalSplatCap: Int?
}

extension TrainerCensus {

    /// Every splat-cap reduction the governor made because the phone was warm.
    ///
    /// Matched on `TrainerBudgetChange.Reason.censusReason`, which is a named
    /// constant in TrainerBudget.swift rather than a phrase repeated here, so
    /// a reworded reason cannot silently stop matching.
    private var heatCapCuts: [TrainerCensusBudgetReduction] {
        budgetReductions.filter {
            $0.what == "splatCap" && $0.reason == TrainerCensusBudgetReduction.heatReason
        }
    }

    /// The census flattened into the shape the review screen reads.
    ///
    /// Derived from the rows above rather than accumulated separately, so the
    /// two files physically cannot disagree, and gated on `trainingBegan` so a
    /// run that never got as far as the loop reports "not recorded" rather
    /// than a row of confident zeroes.
    var sharedRecord: TrainerCensusSharedRecord {
        var out = TrainerCensusSharedRecord()
        let t = totals

        if trainingBegan {
            out.splatsAtStart = seedsUploadedTotal
            out.splatsCreatedByDensification = t.added
            out.densificationPassCount = t.passes
            out.densificationCandidateCount = t.densifyCandidates
            out.splatsDeletedByPruning = t.prunedAtAll
            out.pruningPassCount = t.passesThatPrunedAnything
        }

        // These two are measured from the moment the run starts, whether or not
        // it ever reached the loop: the governor records every reduction it
        // makes, so an empty list is a real "no cap cut happened" rather than
        // an absence of counting. Writing them always is what lets the review
        // screen RULE HEAT OUT instead of saying it cannot tell.
        out.heatCutCount = heatCapCuts.count
        // What a warm-phone cap cut cost. Counted at the moment of each cut as
        // the Gaussians that were alive above the new ceiling, which is the
        // number the trim then had to delete. A cut taken before any loop
        // existed has a live count of 0 and contributes nothing.
        out.splatsDeletedByHeatCut = heatCapCuts.reduce(0) {
            $0 + Swift.max($1.liveSplatCount - $1.to, 0)
        }

        // The merge is the only place `finalSplatCount` is written, and it is
        // written in the same breath as `merge`, so this is "the run got to the
        // end" and not a guess.
        if merge != nil {
            out.splatsAtEnd = finalSplatCount
        }
        out.plannedSplatCap = budgetRequested.splatCap
        out.finalSplatCap = budgetAsRun.splatCap
        return out
    }
}

extension TrainerCensusBudgetReduction {
    /// The exact string `TrainerBudgetChange.Reason.censusReason` produces for
    /// heat. Named once so the match above and the phrase there cannot drift.
    static let heatReason = "the phone was warm"
}

// MARK: - Writing it out

enum TrainerCensusWriter {

    /// Where the census lands, relative to the scan folder. Documented in
    /// `docs/DATA_FORMAT.md` section 8.
    static var relativePath: String { "\(BrandConfig.Folder.model)/train_census.json" }

    /// Where the FLAT record lands: the file the review screen actually opens.
    ///
    /// This path is not a choice. `ViewerScanPaths.modelCensusJSON` reads
    /// `model/census.json` and INTEGRATION_REQUESTS.md asks for that name. The
    /// detailed `train_census.json` beside it is this module's own and nothing
    /// outside the Trainer reads it.
    static var sharedRecordRelativePath: String {
        "\(BrandConfig.Folder.model)/census.json"
    }

    /// Seals the census, puts the ledger and the alerts in the log, and writes
    /// the file.
    ///
    /// Best effort by design: a run that produced a model must not fail
    /// because its diagnostic could not be saved. A write that did not happen
    /// is logged, so "there is no census" is never mistaken for "the census
    /// was clean".
    static func write(_ census: TrainerCensus, at ref: CaptureBundleRef) {
        let sealed = census.sealed()

        // `os.Logger` takes an `OSLogMessage`, not a `String`, so every call
        // below is ONE interpolated literal with a pre-built string inside it.
        // A `+` between two pieces here does not compile.
        for line in sealed.ledger {
            TrainerLog.densify.notice("census: \(line, privacy: .public)")
        }
        for alert in sealed.alerts {
            let text = "\(alert.code): \(alert.detail)"
            if alert.severity == "loud" {
                TrainerLog.densify.error("census alert: \(text, privacy: .public)")
            } else {
                TrainerLog.densify.notice("census alert: \(text, privacy: .public)")
            }
        }

        do {
            let data = try ContractsJSON.encoder().encode(sealed)
            try SmartBinary.write(data, to: ref.url(forRelativePath: relativePath))
        } catch {
            let why = error.localizedDescription
            TrainerLog.densify.error(
                "The training census could not be written: \(why, privacy: .public)"
            )
        }

        // The flat record the review screen reads. Written from the SAME sealed
        // census as the file above, in its own `do` block: the detailed
        // diagnostic and the screen's copy must not be able to take each other
        // down, and the screen's copy is the one a person actually sees.
        do {
            let data = try ContractsJSON.encoder().encode(sealed.sharedRecord)
            try SmartBinary.write(
                data, to: ref.url(forRelativePath: sharedRecordRelativePath)
            )
        } catch {
            let why = error.localizedDescription
            TrainerLog.densify.error(
                "model/census.json could not be written: \(why, privacy: .public)"
            )
        }
    }
}
