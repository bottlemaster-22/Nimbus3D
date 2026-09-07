//
//  TrainerSupport.swift
//  Trainer
//
//  SHARED PLUMBING FOR THE ON-DEVICE TRAINER: logging, errors, tunables, and
//  - the important part - the ARGUMENT INDEX TABLE for every kernel in
//  TrainerShaders.metal.
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE HAS ITS OWN BINDING TABLE
//  ---------------------------------------------------------------------------
//  `TrainerBufferIndex` in TrainerGPULayouts.swift is a LOGICAL registry: it
//  names every distinct buffer the trainer owns and gives each a stable id, so
//  a reviewer can talk about "the offsets buffer" without ambiguity. It is not
//  the physical binding table, and it cannot be: MSL numbers a kernel's
//  arguments from 0 within that kernel. `trainer_preprocess` happens to match
//  `TrainerBufferIndex` exactly (splats 0, sh 1, stats 2, draws 3,
//  tilesTouched 4, camera 5), which is where the registry's numbering came
//  from, but `trainer_rasterize_forward` declares `values [[buffer(0)]]` while
//  the registry calls the values buffer 9. Binding that buffer at 9 would
//  leave argument 0 unbound and the rasteriser would read nothing.
//
//  So: `TrainerBind` below is transcribed argument by argument from the
//  `[[buffer(n)]]` attributes in TrainerShaders.metal and is the ONLY thing
//  the encoders use. `TrainerBufferIndex` stays exactly as it was, unused by
//  the encode path, as the vocabulary it was written to be. If a kernel's
//  signature ever changes, this table changes with it and nothing else does.
//

import Foundation
import Metal
import simd
import os

// MARK: - Logging

enum TrainerLog {
    static let general = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Trainer")
    static let gpu = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Trainer.GPU")
    static let budget = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Trainer.Budget")
    static let densify = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Trainer.Densify")
}

// MARK: - Errors

/// Errors raised inside the trainer. Crosses the module boundary as a
/// `NimbusError` via `asNimbusError`, which is what the UI shows.
enum TrainerError: LocalizedError {
    case noMetalDevice
    case noShaderLibrary
    case missingKernel(String)
    case pipelineFailed(kernel: String, reason: String)
    case layoutMismatch(String)
    case threadgroupTooSmall(kernel: String, available: Int, needed: Int)
    case allocationFailed(name: String, bytes: Int)
    case nothingToTrain(String)
    case noKeyframes
    /// A second `train` arrived while the previous run was still on the GPU or
    /// still unwinding after a stop. There is one trainer and one set of GPU
    /// buffers, so this is refused rather than allowed to share them.
    case alreadyRunning
    /// The GPU rejected or failed a batch of work.
    ///
    /// Nothing in this app asked this question before. Six places
    /// committed a command buffer and called `waitUntilCompleted()`, and
    /// not one of them then looked at `buffer.error` or `buffer.status`.
    /// A GPU fault therefore had two ways to present: carry on with
    /// whatever was in the buffers and produce a quietly wrong model, or
    /// have the driver take the process down. The second leaves no crash
    /// report a person can find in Settings, which matches a crash the
    /// owner hit repeatedly with no app-named report and no JetsamEvent
    /// anywhere near it.
    ///
    /// This does not prevent a GPU fault. It makes one say so.
    case gpuFailed(stage: String, detail: String)

    var errorDescription: String? {
        switch self {
        case .gpuFailed(let stage, let detail):
            return "The graphics work for \(stage) did not finish: \(detail)"
        case .noMetalDevice:
            return "This iPhone did not give the app a graphics device to work with."
        case .noShaderLibrary:
            return "The app's graphics code could not be loaded from the bundle."
        case .missingKernel(let name):
            return "A piece of the training code (\(name)) is missing from this build."
        case .pipelineFailed(let kernel, let reason):
            return "The training step \(kernel) could not be prepared: \(reason)"
        case .layoutMismatch(let detail):
            return "The training data layout does not match the graphics code: \(detail)"
        case .threadgroupTooSmall(let kernel, let available, let needed):
            return "This GPU runs \(kernel) with \(available) threads at a time and "
                + "the trainer needs \(needed)."
        case .allocationFailed(let name, let bytes):
            return "There was not enough memory to set aside \(bytes) bytes for \(name)."
        case .nothingToTrain(let why):
            return "There is nothing to build a 3D model from: \(why)"
        case .noKeyframes:
            return "None of the photos in this scan could be used for training."
        case .alreadyRunning:
            return "Your phone is still putting the last one down. Give it a moment "
                + "and start this again."
        }
    }

    /// What a caller outside this module sees.
    var asNimbusError: NimbusError {
        switch self {
        case .allocationFailed:
            let facts = DeviceMemoryFacts.probe()
            return .outOfMemory(needBytes: 0, haveBytes: facts.availableBytes)
        default:
            return .trainingFailed(errorDescription ?? "the trainer stopped unexpectedly")
        }
    }
}

// MARK: - Kernel names
//
// Every entry point in TrainerShaders.metal, in the order the file declares
// them. Compiled one by one by `TrainerPipelines`; a name that is not in the
// metallib is a named error at start-up, never a silent no-op.

enum TrainerKernel {
    static let fillUInt = "trainer_fill_uint"
    static let fillFloat = "trainer_fill_float"
    static let resetVisibility = "trainer_reset_visibility"
    static let resetDensifyStats = "trainer_reset_densify_stats"
    static let scanBlock = "trainer_scan_block"
    static let scanAdd = "trainer_scan_add"
    static let radixHistogram = "trainer_radix_histogram"
    static let radixScatter = "trainer_radix_scatter"
    static let preprocess = "trainer_preprocess"
    static let duplicateKeys = "trainer_duplicate_keys"
    static let tileRanges = "trainer_tile_ranges"
    static let rasterizeForward = "trainer_rasterize_forward"
    static let lossPhotometric = "trainer_loss_photometric"
    static let ssimPrepare = "trainer_ssim_prepare"
    static let blurH = "trainer_blur_h"
    static let blurV = "trainer_blur_v"
    static let ssimStats = "trainer_ssim_stats"
    static let ssimBackward = "trainer_ssim_backward"
    static let lossDepth = "trainer_loss_depth"
    static let lossFinalize = "trainer_loss_finalize"
    static let rasterizeBackward = "trainer_rasterize_backward"
    static let preprocessBackward = "trainer_preprocess_backward"
    static let samplingRateUpdate = "trainer_sampling_rate_update"
    static let filter3DFinalize = "trainer_filter3d_finalize"
    static let regularizer = "trainer_regularizer"
    static let adamSplat = "trainer_adam_splat"
    static let adamSH = "trainer_adam_sh"
    static let extractCenters = "trainer_extract_centers"

    /// All 28, for the compile loop and for the "did we miss one" check.
    static let all: [String] = [
        fillUInt, fillFloat, resetVisibility, resetDensifyStats,
        scanBlock, scanAdd, radixHistogram, radixScatter,
        preprocess, duplicateKeys, tileRanges, rasterizeForward,
        lossPhotometric, ssimPrepare, blurH, blurV, ssimStats, ssimBackward,
        lossDepth, lossFinalize, rasterizeBackward, preprocessBackward,
        samplingRateUpdate, filter3DFinalize, regularizer,
        adamSplat, adamSH, extractCenters
    ]
}

// MARK: - Argument indices, transcribed from TrainerShaders.metal
//
// One nested enum per kernel. Every value below was read off the
// `[[buffer(n)]]` attribute of that kernel's parameter list. Nothing here is
// derived, inferred or shared between kernels, because that is exactly how a
// binding table drifts from the shader it describes.

enum TrainerBind {

    enum FillUInt {
        static let target = 0
        static let args = 1        // constant uint2& (count, value)
    }

    enum FillFloat {
        static let target = 0
        static let count = 1       // constant uint&
        static let value = 2       // constant float&
    }

    enum ResetVisibility {
        static let stats = 0
        static let count = 1
    }

    enum ResetDensifyStats {
        static let stats = 0
        static let count = 1
    }

    enum ScanBlock {
        static let input = 0
        static let output = 1
        static let blockSums = 2
        static let uniforms = 3    // TrainerScanUniforms
    }

    enum ScanAdd {
        static let output = 0
        static let blockOffsets = 1
        static let uniforms = 2    // TrainerScanUniforms
    }

    enum RadixHistogram {
        static let keys = 0
        static let histogram = 1
        static let uniforms = 2    // TrainerRadixUniforms
    }

    enum RadixScatter {
        static let keysIn = 0
        static let valuesIn = 1
        static let keysOut = 2
        static let valuesOut = 3
        static let histogramScan = 4
        static let uniforms = 5    // TrainerRadixUniforms
    }

    enum Preprocess {
        static let splats = 0
        static let sh = 1
        static let stats = 2
        static let draws = 3
        static let tilesTouched = 4
        static let camera = 5
    }

    enum DuplicateKeys {
        static let draws = 0
        static let tilesTouched = 1
        static let offsets = 2
        static let keys = 3
        static let values = 4
        static let camera = 5
        static let instanceCap = 6   // constant uint&
    }

    enum TileRanges {
        static let keys = 0
        static let tileRanges = 1
        static let count = 2         // constant uint&
    }

    enum RasterizeForward {
        static let values = 0
        static let tileRanges = 1
        static let draws = 2
        static let outColor = 3
        static let outAlpha = 4
        static let outDepth = 5
        static let outTFinal = 6
        static let outNContrib = 7
        static let camera = 8
    }

    enum LossPhotometric {
        static let renderColor = 0
        static let renderTFinal = 1
        static let gtColor = 2
        static let bgColor = 3
        static let composited = 4
        static let gradFinal = 5
        static let ssimPlanes = 6
        static let lossAccum = 7
        static let uniforms = 8      // TrainerLossUniforms
    }

    enum SSIMPrepare {
        static let planes = 0
        static let uniforms = 1      // TrainerBlurUniforms
    }

    enum Blur {
        static let src = 0
        static let dst = 1
        static let uniforms = 2      // TrainerBlurUniforms
    }

    enum SSIMStats {
        static let blurred = 0
        static let partials = 1
        static let lossAccum = 2
        static let uniforms = 3      // TrainerLossUniforms
    }

    enum SSIMBackward {
        static let blurredPartials = 0
        static let lumaPlanes = 1
        static let gradFinal = 2
        static let uniforms = 3      // TrainerLossUniforms
    }

    enum LossDepth {
        static let samples = 0
        static let renderDepth = 1
        static let renderAlpha = 2
        static let gradDepth = 3
        static let gradTFinal = 4
        static let unknownMask = 5
        static let lossAccum = 6
        static let uniforms = 7      // TrainerLossUniforms
    }

    enum LossFinalize {
        static let gradFinal = 0
        static let renderColor = 1
        static let renderTFinal = 2
        static let bgColor = 3
        static let gradSplat = 4
        static let gradTFinal = 5
        static let exposureGrad = 6
        static let uniforms = 7      // TrainerLossUniforms
    }

    enum RasterizeBackward {
        static let values = 0
        static let tileRanges = 1
        static let draws = 2
        static let renderTFinal = 3
        static let renderNContrib = 4
        static let gradSplatColor = 5
        static let gradDepth = 6
        static let gradTFinal = 7
        static let bgColor = 8
        static let unknownMask = 9
        static let gradMean2D = 10
        static let gradConic = 11
        static let gradColor = 12
        static let gradOpacity = 13
        static let stats = 14
        static let camera = 15
        static let lossUniforms = 16
    }

    enum PreprocessBackward {
        static let splats = 0
        static let sh = 1
        static let draws = 2
        static let gradMean2D = 3
        static let gradConic = 4
        static let gradColor = 5
        static let gradOpacity = 6
        static let splatGrad = 7
        static let shGrad = 8
        static let cameraGrad = 9
        static let stats = 10
        static let camera = 11
    }

    enum SamplingRateUpdate {
        static let splats = 0
        static let topK = 1
        static let camera = 2
    }

    enum Filter3DFinalize {
        static let topK = 0
        static let stats = 1
        static let count = 2         // constant uint&
        static let filterScale = 3   // constant float&
        static let fallback = 4      // constant float&
    }

    enum Regularizer {
        static let splats = 0
        static let stats = 1
        static let grad = 2
        static let lossAccum = 3
        static let uniforms = 4      // TrainerRegUniforms
    }

    enum AdamSplat {
        static let splats = 0
        static let grad = 1
        static let m = 2
        static let v = 3
        static let stats = 4
        static let uniforms = 5      // TrainerAdamUniforms
    }

    enum AdamSH {
        static let sh = 0
        static let grad = 1
        static let m = 2
        static let v = 3
        static let stats = 4
        static let uniforms = 5      // TrainerAdamUniforms
    }

    enum ExtractCenters {
        static let splats = 0
        static let centers = 1
        static let count = 2         // constant uint&
    }
}

// MARK: - Tunables
//
// Everything the trainer schedules on, in one struct with the reference 3DGS
// numbers as defaults, so an experiment is a value change rather than a hunt
// through five files. Nothing here is read from a global.

struct TrainerTuning: Sendable {

    // --- Adam ---------------------------------------------------------------

    /// Position learning rate at iteration 0, MULTIPLIED by the scene extent.
    /// The reference implementation scales the position step by the size of
    /// the scene, which is what makes the same number work for a chair and a
    /// hallway.
    var positionLRInitialScaled: Float = 0.00016
    /// Position learning rate at the last iteration, same scaling. The decay
    /// between the two is exponential.
    var positionLRFinalScaled: Float = 0.0000016
    var opacityLR: Float = 0.05
    var scaleLR: Float = 0.005
    var rotationLR: Float = 0.001
    var shDCLR: Float = 0.0025
    /// The reference ratio: higher-order SH learns twenty times slower than DC.
    var shRestLRDivisor: Float = 20

    // --- Losses -------------------------------------------------------------

    var lambdaSSIM: Float = 0.2
    var alphaSupervisionWeight: Float = 0.05
    /// Extra multiplier on a sample the edge classifier called GEOMETRIC.
    /// This is the "sharpen" half of F3's WHERE/WHAT split.
    var geometricEdgeBoost: Float = 2.0

    // --- Coarse to fine ------------------------------------------------------

    /// Extra screen-space variance, px^2, at iteration 0. Decays to zero by
    /// `frequencyBlurEndFraction`. Low frequencies first, detail later.
    var frequencyBlurStartVariance: Float = 4.0
    var frequencyBlurEndFraction: Float = 0.2
    /// Fraction of the run after which every SH coefficient is enabled.
    var shFullyEnabledFraction: Float = 0.35

    // --- Schedule ------------------------------------------------------------

    /// Warm-up ends here, as a fraction of the run. Camera pose deltas are
    /// frozen before this point (F1).
    var warmupFraction: Float = 0.15
    /// Densification runs between these two fractions of the run.
    var densifyStartFraction: Float = 0.10
    /// Raised from 0.70 after a measured run showed the budget being freed
    /// and never spent.
    ///
    /// Once the prune learned to test the opacity that actually reaches a
    /// pixel, it started removing the roughly half of the population that
    /// was invisible. Measured on two exports of the same scan: 299,965
    /// splats with a median peak alpha of 0.0038 before, 150,302 with a
    /// median of 0.93 after. The dead half went, correctly.
    ///
    /// But growth stopped at 0.70 while pruning now runs to 1.0, so every
    /// Gaussian that died after iteration 2,100 of 3,000 left a hole
    /// nothing could fill. The run asked for a 300,000 budget and handed
    /// back a 150,000 model. Splats do not die early: a clone starts with
    /// its parent's opacity and fades over hundreds of iterations, so the
    /// deaths are concentrated exactly where growth had already stopped.
    ///
    /// 0.85 leaves 450 iterations to settle. That is later than the 50
    /// percent that photo-only 3DGS pipelines use, and the reason it is
    /// defensible HERE is LiDAR: a clone inherits a parent already sitting
    /// on a measured surface with a measured normal, so it starts in
    /// roughly the right place and refines locally rather than searching
    /// for geometry from scratch the way a photo-only clone must.
    var densifyEndFraction: Float = 0.85
    var densifyIntervalIterations: Int = 100
    /// Opacity binarization runs over the last this-much of the run (F4).
    var binarizeLastFraction: Float = 0.20
    var binarizeWeight: Float = 0.02
    /// Free-space carving deletion sweep interval, iterations.
    var carveIntervalIterations: Int = 250
    /// How often a preview snapshot is read back off the GPU.
    /// WAS 50. Raised on the owner's own observation that the preview was
    /// updating far more often than he needed while the trainer waited.
    ///
    /// The snapshot itself is not free: it copies the whole model off the GPU
    /// and re-boxes it into a SplatCloud on the training thread, so at 300,000
    /// splats it is tens of milliseconds of the training loop each time. At
    /// 0.068 s per iteration, 200 iterations is a preview that refreshes about
    /// every 14 seconds, which for watching a model converge is often enough.
    /// If it feels dead, this is the number to lower.
    var snapshotIntervalIterations: Int = 200
    /// How often the loss accumulator WOULD be read back, if anything read
    /// this.
    ///
    /// NOTHING READS THIS, and that is not a missing feature. The training
    /// loop reads `lossAccum` once per iteration, straight after the
    /// `waitUntilCompleted()` it already has to do, so the read costs four
    /// bytes off a shared buffer that is already synchronised. Honouring an
    /// interval here would not save anything and WOULD change the numbers: the
    /// 0.98 exponential average the progress stream shows is tuned for a
    /// sample every iteration, and feeding it one sample in ten would make it
    /// lag by ten times as much. Left here rather than deleted so that the
    /// next person to wonder why there is no interval finds the answer.
    var lossReadbackIntervalIterations: Int = 10

    // --- Densification --------------------------------------------------------

    /// AbsGS gradient floor, in the PIXEL units this rasteriser actually
    /// accumulates. Effectively "greater than zero": the real cut is the ranked
    /// truncation to the budget in `TrainerDensifier`, never this number.
    ///
    /// It was 0.0006, borrowed from the reference 3DGS implementation, and that
    /// was a units error that silently disabled densification for entire runs.
    /// The reference multiplies its gradient by 0.5 * width before storing it,
    /// so its numbers are in normalised device coordinates. This rasteriser
    /// stores `length(dLdMean2D)` raw, in pixels, which at a 720 px render is
    /// smaller by a factor of a few hundred. Nothing ever cleared 0.0006, so
    /// zero Gaussians were ever created and every model stayed as sparse as its
    /// LiDAR seed.
    ///
    /// Keep this at or near zero. If a real floor is ever wanted, derive it
    /// from the render width so it cannot drift out of units again.
    ///
    /// NOTHING READS THIS TODAY. `TrainerDensifier` selects candidates on
    /// `score[i] > 0` and lets the ranked truncation to the cap do the cutting.
    /// It is written down here rather than deleted because the units trap
    /// above is worth keeping where the next person to want a floor will find
    /// it, and `model/train_census.json` records the floor actually in force
    /// (`gates.densifyScoreFloor`, which is 0) rather than this number, so the
    /// census cannot present a dead setting as a live one.
    var absGradThreshold: Float = 1e-9
    /// A Gaussian larger than this fraction of the scene extent is SPLIT;
    /// smaller ones are CLONED.
    ///
    /// WAS 0.01, WHICH NOTHING EVER REACHED. `addedBySplit` is 0 in all 29
    /// densify passes of the owner's 3000-iteration run, every one of the
    /// ~150,000 splats it added came from clone, and the reason is
    /// arithmetic rather than a bug in the split code: measured on that
    /// run's own PLY, the scene extent is 5.39 m, so 0.01 put the bar at
    /// 5.39 cm, while the population sits at a 1.3 cm median, 1.81 cm p90
    /// and 3.85 cm p99, with the single largest splat in the model at
    /// 7.86 cm. 61 splats out of 151,355 cleared it, 0.04%, and those
    /// still had to rank inside the growth allowance to be chosen.
    ///
    /// The bar sat above the 99.96th percentile of the thing it filters.
    ///
    /// 0.004 is p95 of that measured distribution, about 2.2 cm. Chosen
    /// from the data rather than halved arbitrarily, and deliberately not
    /// lower: split everything and the count explodes into the cap, which
    /// costs time and solves nothing.
    ///
    /// The suspected consequence, not yet confirmed: growth being 100%
    /// clone means an over-large blurry Gaussian gets DUPLICATED at the
    /// same size rather than divided into two sharp ones, and the only
    /// remaining way for the optimiser to remove the error it causes is to
    /// drive both copies transparent. `prunedLowOpacity` on that run goes
    /// 213, then 57,999 at iteration 2500, then 54,888, 19,456, 9,886,
    /// 6,381, taking the model from 299,787 splats to 151,355 in the last
    /// sixth of training. If this setting is the cause, that collapse is a
    /// wall-clock bug as much as a quality one: the most expensive third
    /// of the run trains 300,000 splats in order to delete 148,000 of
    /// them.
    var splitScaleFraction: Float = 0.004
    /// How many children a split produces.
    ///
    /// Descriptive, not read. `TrainerDensifier` splits by shrinking the
    /// parent in place and appending exactly one sibling, which is two
    /// children for one slot, and that two is a property of the code rather
    /// than a number it looks up. Changing this alone changes nothing.
    var splitChildCount: Int = 2
    /// Children are placed at +/- this many standard deviations along the
    /// split axis, and shrunk by `splitShrink`.
    var splitOffsetSigma: Float = 0.8
    var splitShrink: Float = 1.6
    /// Opacity below which a Gaussian is pruned, as a probability.
    var pruneOpacity: Float = 0.005
    /// Screen radius above which a Gaussian is pruned, in pixels.
    var pruneMaxScreenRadiusPx: Float = 0.0
    /// World scale above which a Gaussian is pruned, as a fraction of extent.
    var pruneMaxWorldScaleFraction: Float = 0.1
    /// Fraction of the cap that may be added in one densification pass.
    ///
    /// THESE TWO NUMBERS ALSO SET WHAT DENSIFICATION COSTS IN CPU TIME.
    /// `TrainerDensifier` no longer sorts its candidate list; it selects the
    /// best `max(growthAllowance, relocationLimit)` of it and leaves the rest
    /// unordered, because nothing below the truncation point is ever read.
    /// The size of that selection is exactly these two fractions, so raising
    /// either one raises the per-pass ranking work with it, and setting
    /// `maxGrowthFractionPerPass` to 1.0 restores the full-population sort
    /// this deliberately removed. Counted at a 300,000 point population:
    /// a full sort is 4,242,295 comparisons against 1,104,232 for a bounded
    /// selection of the top 15 percent, and the pass runs every
    /// `densifyIntervalIterations` for the whole run on a phone that is
    /// already thermally limited.
    var maxGrowthFractionPerPass: Float = 0.15
    /// Once the cap is reached, this fraction of the population may be
    /// RELOCATED per pass (MCMC style: a dead Gaussian is moved onto a
    /// high-gradient one, so the count never changes). See the cost note on
    /// `maxGrowthFractionPerPass`: this is the other half of the bound on how
    /// much of the population one pass ranks.
    var maxRelocationFractionPerPass: Float = 0.05
    /// Opacity below which a Gaussian is a relocation donor.
    var relocationDonorOpacity: Float = 0.02

    // --- Mip-Splatting --------------------------------------------------------

    /// Mip-Splatting's filter scale constant.
    var filter3DScale: Float = 0.2
    /// Filter size, world metres, for a Gaussian no camera ever sampled.
    var filter3DFallbackMeters: Float = 0.01
    /// How often the per-Gaussian sampling-rate sweep is re-run.
    var filter3DIntervalIterations: Int = 500

    // --- Exposure (F5, F8) -----------------------------------------------------

    var exposureLearningRate: Float = 0.002
    /// Hard bounds on the learned per-frame gain and bias. Tight on purpose:
    /// exposure is a small nuisance parameter, and a loose one becomes a
    /// licence for the model to explain geometry error as brightness error.
    var exposureGainRange: ClosedRange<Float> = 0.9...1.1
    var exposureBiasRange: ClosedRange<Float> = -0.05...0.05

    // --- Camera pose deltas (F1) ------------------------------------------------

    var cameraRotationLR: Float = 1e-5
    var cameraTranslationLR: Float = 1e-5
    /// Hard per-step clamp so a single bad frame cannot throw a camera.
    var cameraMaxRotationStepRadians: Float = 0.0005
    var cameraMaxTranslationStepMeters: Float = 0.002

    // --- Held-out evaluation ------------------------------------------------------

    /// Fraction of keyframes withheld from training and used only to report
    /// PSNR. Reported whatever it says.
    var heldOutFraction: Float = 0.05

    init() {}
}

// MARK: - Small helpers

enum TrainerMath {

    @inline(__always)
    static func clamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
        simd_clamp(x, lo, hi)
    }

    @inline(__always)
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-simd_clamp(x, -30, 30))) }

    @inline(__always)
    static func logit(_ p: Float) -> Float {
        let c = simd_clamp(p, 1e-6, 1 - 1e-6)
        return logf(c / (1 - c))
    }

    /// Exponential interpolation between two learning rates, the schedule the
    /// reference implementation uses for the position step.
    static func expLerp(_ a: Float, _ b: Float, t: Float) -> Float {
        let tt = simd_clamp(t, 0, 1)
        guard a > 0, b > 0 else { return a + (b - a) * tt }
        return expf(logf(a) * (1 - tt) + logf(b) * tt)
    }

    /// Bytes rounded up to a page, which is what Metal actually reserves.
    ///
    /// Unused today. The memory budget works in bytes-per-splat and
    /// bytes-per-pixel, where per-buffer page rounding is at most a few
    /// hundred kilobytes against hundreds of megabytes, so nothing would
    /// change if it were applied. Kept because it is correct and because the
    /// next person to compare an estimate against Instruments will want it.
    static func pageAligned(_ bytes: Int) -> Int {
        let page = 16 * 1024
        return ((Swift.max(bytes, 1) + page - 1) / page) * page
    }

    /// A quaternion, as (x, y, z, w), that rotates +Z onto `direction`. Used
    /// to lay a Gaussian along a viewing ray (F6) and to lay one across an
    /// edge normal (F4).
    static func quaternionAligningZ(to direction: SIMD3<Float>) -> SIMD4<Float> {
        let d = simd_length(direction) > 1e-6
            ? simd_normalize(direction)
            : SIMD3<Float>(0, 0, 1)
        let z = SIMD3<Float>(0, 0, 1)
        let dot = simd_dot(z, d)
        if dot > 0.999999 { return SIMD4<Float>(0, 0, 0, 1) }
        if dot < -0.999999 { return SIMD4<Float>(1, 0, 0, 0) }  // 180 deg about X
        let axis = simd_cross(z, d)
        let q = simd_quatf(angle: acosf(simd_clamp(dot, -1, 1)), axis: simd_normalize(axis))
        let v = q.normalized.vector
        return SIMD4<Float>(v.x, v.y, v.z, v.w)
    }

    /// Rotation matrix from a (x, y, z, w) quaternion, matching
    /// `trainer_quatToMatrix` in the shader exactly.
    static func rotationMatrix(_ q: SIMD4<Float>) -> simd_float3x3 {
        let n = simd_length(q) > 1e-8 ? q / simd_length(q) : SIMD4<Float>(0, 0, 0, 1)
        let x = n.x, y = n.y, z = n.z, w = n.w
        return simd_float3x3(
            SIMD3<Float>(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
            SIMD3<Float>(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
            SIMD3<Float>(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        )
    }
}

// MARK: - Typed access to a shared MTLBuffer
//
// Every buffer the trainer allocates is `.storageModeShared`, which on Apple
// silicon means one allocation the CPU and GPU both address. These two helpers
// are the only places a raw pointer is taken, so there is one place to look
// when a readback is wrong.

extension MTLBuffer {

    /// Reads ONE element without copying the whole buffer. The training loop
    /// needs exactly two values back from the GPU each iteration (the last
    /// scan offset and the last tile count), and pulling a 300k-element array
    /// across to read its last entry is a megabyte of memcpy per iteration for
    /// four bytes of information.
    func readElement<T>(_ type: T.Type, at index: Int) -> T? {
        let stride = MemoryLayout<T>.stride
        guard index >= 0, (index + 1) * stride <= length else { return nil }
        return contents()
            .advanced(by: index * stride)
            .assumingMemoryBound(to: T.self)
            .pointee
    }

    /// Scoped read-only access to the buffer as a typed array, with no copy.
    /// For the passes that walk a whole per-pixel buffer once and keep
    /// nothing: copying it into a Swift array first would double the work and
    /// the peak memory for no benefit.
    func withElements<T, R>(
        _ type: T.Type,
        count: Int,
        _ body: (UnsafeBufferPointer<T>) -> R
    ) -> R? {
        guard count > 0, length >= count * MemoryLayout<T>.stride else { return nil }
        let raw = contents().bindMemory(to: T.self, capacity: count)
        return body(UnsafeBufferPointer(start: raw, count: count))
    }

    /// Reads `count` elements of `T` from the front of the buffer.
    func readArray<T>(_ type: T.Type, count: Int) -> [T] {
        guard count > 0, length >= count * MemoryLayout<T>.stride else { return [] }
        let raw = contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }

    /// Writes `values` to the front of the buffer. Silently writes as many as
    /// fit, and returns how many that was, so an over-long write is a number
    /// the caller can check rather than a heap corruption.
    @discardableResult
    func writeArray<T>(_ values: [T]) -> Int {
        let fits = Swift.min(values.count, length / Swift.max(MemoryLayout<T>.stride, 1))
        guard fits > 0 else { return 0 }
        let raw = contents().bindMemory(to: T.self, capacity: fits)
        for i in 0..<fits { raw[i] = values[i] }
        return fits
    }

    /// Zeroes the whole allocation.
    func zeroAll() {
        memset(contents(), 0, length)
    }

    /// Zeroes the first `bytes` bytes.
    func zero(bytes: Int) {
        memset(contents(), 0, Swift.min(Swift.max(bytes, 0), length))
    }
}
