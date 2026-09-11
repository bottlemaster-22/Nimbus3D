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
    static let resetDensifyStats = "trainer_reset_densify_stats"
    static let scanBlock = "trainer_scan_block"
    static let scanAdd = "trainer_scan_add"
    static let radixHistogram = "trainer_radix_histogram"
    static let radixScatter = "trainer_radix_scatter"
    static let preprocess = "trainer_preprocess"
    static let duplicateKeys = "trainer_duplicate_keys"
    static let tileRanges = "trainer_tile_ranges"
    static let rasterizeForward = "trainer_rasterize_forward"
    /// Build 302: two pixels per thread. Optional, so not in `all`.
    static let rasterizeForward2 = "trainer_rasterize_forward2"
    /// Build 304: the backward, two pixels per thread (plain atomics since 312). Optional.
    static let rasterizeBackward2 = "trainer_rasterize_backward2"
    /// Build 306: the splat-order tile sort.
    static let depthKeys = "trainer_depth_keys"
    static let gatherTouched = "trainer_gather_touched"
    /// Build 316: sizes the sort on the GPU.
    static let sortSetup = "trainer_sort_setup"
    /// Build 320: applies the densifier's index map to the bulk arrays.
    static let densifyGather = "trainer_densify_gather"
    static let background = "trainer_background"
    static let lossPhotometric = "trainer_loss_photometric"
    static let blurH = "trainer_blur_h"
    static let blurV = "trainer_blur_v"
    /// Build 318: both blur directions in one kernel. Optional, not in `all`.
    static let blurHV = "trainer_blur_hv"
    static let ssimStats = "trainer_ssim_stats"
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

    /// All 25, for the compile loop and for the "did we miss one" check.
    static let all: [String] = [
        fillUInt, fillFloat, resetDensifyStats,
        scanBlock, scanAdd, radixHistogram, radixScatter,
        preprocess, duplicateKeys, depthKeys, gatherTouched, sortSetup, densifyGather, tileRanges,
        rasterizeForward,
        lossPhotometric, blurH, blurV, ssimStats,
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
        /// The compact 32-byte copy the hot loops read. See
        /// TrainerSplatRaster in TrainerShaders.metal.
        static let raster = 8
    }

    enum DuplicateKeys {
        /// The compact record, not the wide one.
        static let raster = 0
        static let tilesTouched = 1
        static let offsets = 2
        static let keys = 3
        static let values = 4
        static let camera = 5
        static let instanceCap = 6   // constant uint&
        static let order = 7         // build 306: depth-sorted splat order
        static let ordered = 8       // constant uint&: 1 in splat-order mode
    }

    enum DepthKeys {
        static let raster = 0
        static let tilesTouched = 1
        static let keys = 2
        static let values = 3
        static let camera = 4
    }

    enum GatherTouched {
        static let order = 0
        static let tilesTouched = 1
        static let sortedTouched = 2
        static let count = 3         // constant uint&
    }

    enum SortSetup {
        static let offsets = 0
        static let touched = 1
        static let args = 2          // TrainerGPUConstants.sortArgSlots x 256 bytes
        static let camera = 3
        static let instanceCap = 4   // constant uint&
        static let groupWidth = 5    // constant uint&
    }

    enum DensifyGather {
        static let src = 0
        static let dst = 1
        static let source = 2
        static let flags = 3
        static let wordsPer = 4      // constant uint&
        static let zeroMask = 5      // constant uint&
        static let count = 6         // constant uint&
    }

    enum TileRanges {
        static let keys = 0
        static let tileRanges = 1
        static let count = 2         // constant uint&
    }

    enum Background {
        static let texels = 0
        static let bgColor = 1
        static let uniforms = 2
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
        static let gtLevels = 9      // build 314: 256 floats, Float(i) / 255
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
        static let blurredPartials = 8
        static let lumaPlanes = 9
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
        static let splatGrad2D = 10
        // 14 WAS `stats`, and is deliberately left as a hole rather than
        // renumbered. The backward rasteriser stopped touching that buffer
        // when its three accumulators moved into splatGrad2D's own cache
        // line; renumbering `camera` and `lossUniforms` down to close the gap
        // would be a silent, invisible way to break every binding in this
        // kernel for nothing.
        static let camera = 15
        static let lossUniforms = 16
    }

    enum PreprocessBackward {
        static let splats = 0
        static let sh = 1
        static let draws = 2
        static let splatGrad2D = 3
        static let splatGrad = 7
        static let shGrad = 8
        static let cameraGrad = 9
        static let stats = 10
        static let camera = 11
        static let tilesTouched = 12
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
        static let tilesTouched = 5
    }

    enum AdamSplat {
        static let splats = 0
        static let grad = 1
        static let m = 2
        static let v = 3
        static let stats = 4
        static let uniforms = 5      // TrainerAdamUniforms
        static let tilesTouched = 6
    }

    enum AdamSH {
        static let sh = 0
        static let grad = 1
        static let m = 2
        static let v = 3
        static let stats = 4
        static let uniforms = 5      // TrainerAdamUniforms
        static let tilesTouched = 6
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
    /// DECAY THE OTHER LEARNING RATES, not just position. Set to 1 to make
    /// every schedule flat again, which is the reference 30,000-iteration
    /// behaviour we inherited.
    ///
    /// `MetalSplatTrainer` already decays the POSITION rate exponentially
    /// across the run and leaves scale, rotation, opacity and SH flat,
    /// because that is what the reference does. But the reference runs
    /// 30,000 iterations and we run 3,000, and a rate tuned to anneal over
    /// ten times as many steps is still near its starting value when our run
    /// ends: the parameter jitters instead of settling. arXiv:2604.28016
    /// measures 26.01 against 27.25 PSNR for exactly this mistake on
    /// position alone, and says it adds "analogous schedules for the
    /// Gaussian scale, rotation, and SH DC coefficients", which 3DGS has
    /// none of. This is the final multiple of each starting rate.
    var lateLRFraction: Float = 0.1

    /// MULTIPLY THE SH DC RATE. 1 restores the reference value.
    ///
    /// Adam here is visibility-masked and sparse, so a Gaussian takes a step
    /// only on iterations where it is drawn. With 114 trained views and
    /// 3,000 iterations each photograph is visited about 26 times, and a
    /// splat drawn in 15 per cent of views gets roughly 450 steps. At
    /// shDCLR 0.0025 that is at most 1.13 in SH coefficient space, which
    /// through colour = 0.5 + 0.282095*dc is at most 0.32 of the 0..1 colour
    /// range, and realistically a third of that once Adam's sign averaging
    /// is accounted for. Against a measured RMSE of 0.117 that is not
    /// obviously enough travel. 4x is the standard remedy for a shortened
    /// schedule. Colour is the safest rate to raise: unlike scale and
    /// opacity it cannot produce a floater, only a wrong colour that the
    /// next gradient corrects.
    var shDCLRMultiplier: Float = 4

    var opacityLR: Float = 0.05
    var scaleLR: Float = 0.005
    var rotationLR: Float = 0.001
    var shDCLR: Float = 0.0025
    /// The reference ratio: higher-order SH learns twenty times slower than DC.
    var shRestLRDivisor: Float = 20

    /// SCOPING NOTE ON SPHERICAL HARMONIC DEGREE, so the next person does not
    /// price the wrong thing.
    ///
    /// The reference capture ships degree 3 and we ship degree 1, and that
    /// looks like a config change. It is not. `TrainerShaders.metal` implements
    /// SH evaluation AND its backward gradient only up to DEGREE 2. There is no
    /// degree-3 path in the training kernels in either direction; the degree-3
    /// code in the repo lives in the VIEWER's render shader, for displaying
    /// assets that were trained elsewhere.
    ///
    /// So degree 2 is config plus allocation and costs 72.0 MB at the 300,000
    /// cap. Degree 3 costs 172.8 MB AND requires writing and validating new
    /// Metal for `trainer_evalSH` and the SH-gradient kernel first. Any earlier
    /// wall-clock estimate for degree 3 cannot have come from a timed run on
    /// this codebase, because there is nothing to time.

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

    /// A CEILING ON WHEN GROWTH MAY START, in iterations, whatever the
    /// fraction above works out to. 0 removes the ceiling.
    ///
    /// `densifyStartFraction` is a FRACTION, and that is fine at 3,000
    /// iterations where it means "start at 300". At 30,000 it means "start at
    /// 3,000", and the first held-out curve this trainer ever recorded shows
    /// exactly what that costs:
    ///
    ///     iter 2000  PSNR 15.14  splats 148,934
    ///     iter 2500  PSNR 14.71  splats 148,727
    ///     iter 3000  PSNR 14.68  splats 148,565
    ///     iter 3500  PSNR 16.19  splats 300,000   <- growth finally happened
    ///
    /// The run spent its first 3,000 iterations optimising a HALF-SIZE
    /// population, because the seeder hands over about 148,000 and nothing was
    /// allowed to add to it yet. Every one of those iterations was training a
    /// model that was never going to be the final one.
    ///
    /// A warm-up before growth is right; scaling that warm-up with the total
    /// budget is not. 300 iterations is enough for the gradients to mean
    /// something, and it is what a 3,000-iteration run was already doing.
    var densifyStartMaxIterations: Int = 300
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
    /// BACK TO 0.85, AND THIS IS WHY BUILDS 244 AND 246 LOOK WORSE THAN 226.
    ///
    /// 0.5 was set for a 30,000-iteration run, where it is right: half the run
    /// to place geometry and half to fit it is what every published 3DGS
    /// number is measured with. When the budget came back to 4,000 it was
    /// never put back, and at 4,000 it means this:
    ///
    ///     growth window: iteration 300 .. 2000
    ///     last densify pass: iteration 3900
    ///     relocated: 0        donors available: 0
    ///
    /// For the last 2,000 iterations, HALF THE RUN, the model cannot split,
    /// clone, prune or relocate. It cannot add a Gaussian, remove one, or make
    /// one smaller. All it can do is push the existing geometry harder at the
    /// training views, which is the exact recipe for a model that scores well
    /// and looks wrong.
    ///
    /// Relocation dies with the window, because it only runs when the growth
    /// window is open and the population is at its cap. That is why the
    /// counter reads exactly 0 in builds 244 and 246 and read 47,775 in 234.
    /// The one mechanism that can still SHRINK a Gaussian once the cap is
    /// reached was switched off for half of every recent run.
    ///
    /// 226, the build the owner still rates highest, ran 0.85 over 3,000
    /// iterations: growth to 2,550, only 15 per cent of the run frozen.
    var densifyEndFraction: Float = 0.85
    var densifyIntervalIterations: Int = 100

    // MARK: - Early stopping, i.e. "how many rounds is actually right"

    /// How often to score the HELD-OUT frames during training, in iterations.
    /// 0 disables early stopping entirely and the run always goes the full
    /// distance.
    ///
    /// WHY THIS EXISTS. A 30,000-iteration run measured trained-view PSNR
    /// rising 21.10 to 25.67 while held-out FELL 16.88 to 14.38, a train/test
    /// gap of 11.29 dB, with 44.4 per cent of the population ending as needles
    /// against 23.4 per cent at 3,000. The optimiser was working perfectly and
    /// the model was memorising 108 training views. Ten minutes of phone
    /// battery bought a worse scan.
    ///
    /// The right number of rounds is not a constant anyone can pick in
    /// advance, because it depends on how many views the capture has and how
    /// much parallax they carry. It IS measurable while training, and the
    /// trainer already holds out frames and already knows how to score them.
    ///
    /// MUST BE A MULTIPLE OF `densifyIntervalIterations`, and the code rounds
    /// it up to one. Scoring runs the preprocess kernel over the held-out
    /// frames, which increments `stats.denom` and sets `visibleFlag`, and
    /// those feed densification. Doing it immediately before a densify pass
    /// means the pass's own `trainer_reset_densify_stats` wipes the pollution
    /// on the very next line. Anywhere else and the AbsGS score quietly
    /// includes frames the model is not training on.
    /// 400, NOT 200, BECAUSE THIS FEATURE HAS NEVER FIRED AND IT IS NOT FREE.
    ///
    /// Build 244: `earlyStopEval` cost 5.67 s of an 80 s training run, 7.1 per
    /// cent, across 16 evaluations of 12 held-out frames at 355 ms each. And
    /// `stoppedEarly` is false with the reason "ran out its iterations": the
    /// held-out curve rises monotonically from 18.95 at iteration 800 to 20.48
    /// at 3,800, so the model is UNDER-trained, not over-trained, and the stop
    /// has never triggered once.
    ///
    /// It still earns its place, because it also chooses which model to
    /// export, and that was worth 0.79 dB on build 234. But it can do that on
    /// a coarser grid. Simulating both grids on builds 244 and 246: the
    /// shipped 200 grid picks 3,400 at 20.432 dB and a 400 grid picks 3,600 at
    /// 20.463 on 244; on 246 the signs reverse, 3,800 at 20.662 against 3,600
    /// at 20.632. Both differences are inside the +/-0.4 dB noise band and the
    /// sign flips between builds, which is the definition of noise.
    var earlyStopEvalIntervalIterations: Int = 400

    /// Stop after this many consecutive evaluations fail to beat the best
    /// held-out score by `earlyStopMinImprovementDB`.
    ///
    /// Four evaluations at 500 iterations is 2,000 iterations of patience,
    /// which is enough to ride out a dip caused by a densification pass
    /// injecting new geometry that has not settled yet.
    // 8 (build 380; was 4): 378 stopped at 25,600 on a curve rising by
    // 0.01 dB an eval; quality mode waits 3,200 iterations of flat.
    var earlyStopPatienceEvals: Int = 8

    /// Never stop before this many iterations, whatever the score does. Early
    /// training is noisy and the densify window has not even opened at 10 per
    /// cent of the run.
    /// WAS 2,000, WHICH WAS ALSO THE FIRST EVALUATION, so the recorded curve
    /// began at its own maximum and declined monotonically from there. "The
    /// model peaks at 2,000 iterations" was not a measurement, it was the
    /// floor of the measuring instrument: nothing before 2,000 was ever
    /// sampled. 700 is just after the population reaches its cap, which the
    /// densify ledger puts at iteration 600 to 900, so the curve now starts
    /// where the model is complete rather than a thousand iterations later.
    var earlyStopMinIterations: Int = 700

    /// How many frames ahead the keyframe selector may look for a SHARPER one
    /// once spacing has already been satisfied. 0 restores the old behaviour
    /// of taking the first frame that clears the gate.
    ///
    /// The selector used `qc.weight > 0.05` as a floor and then never read
    /// sharpness again. Measured on the owner's capture: 93 of the 120 chosen
    /// keyframes carry more than one RENDER pixel of motion blur (native p50
    /// 3.84 px, p90 6.80; at the 720 px render, p50 1.44 and p90 2.55). The
    /// model is being asked to reproduce blurred photographs exactly, and
    /// there is no term anywhere in the loss that knows a photograph is
    /// blurred.
    ///
    /// OFF (0) since build 274. With 5, the walk resumed after the GATE
    /// frame, not after the frame the window picked, so the next frames
    /// passed the turn gate measured against the pick and chose it again.
    /// On scan_20260906_164840 that re-picked 29 of 120 slots (91 distinct
    /// frames), and because the held-out split takes every tenth entry of
    /// that list, 6 of the 12 held-out frames were ALSO trained on
    /// (tools/offline/kf_exact.py, exact against held_out_frames.json).
    /// Every held-out score from build 244 to 272 is flattered by that.
    /// Re-enable only together with a walk that resumes after the pick, and
    /// price the spread that causes (build 264).
    var keyframeSharpnessLookahead: Int = 0

    /// How much better a score has to be to count as an improvement. Below
    /// this it is noise, and waiting for noise to clear is what turns a
    /// stopping rule into a run that never stops.
    var earlyStopMinImprovementDB: Float = 0.05
    /// Opacity binarization runs over the last this-much of the run (F4).
    /// ZERO, AS A NULL TEST. Was 0.2.
    ///
    /// The binarization ramp starts at 80% of the run and the growth window
    /// closes at 85%, so about 90,611 splats are driven to zero opacity and
    /// pruned AFTER densification can no longer replace them. The population
    /// goes 299,787 to 151,355 in the last sixth of training, and the owner's
    /// own Scaniverse comparison has 427,710 splats against our 158,220.
    ///
    /// Before rescheduling the ramp or retuning its weight, find out whether
    /// it earns its cost at all: at 0 the term is skipped entirely
    /// (MetalSplatTrainer applies it only when binarizeLastFraction > 0). If
    /// PSNR improves and the population survives, the answer is to weaken or
    /// remove binarization, not to move it.
    /// 0 DISABLES late opacity binarization entirely, and it is off after
    /// a null test measured what it was costing: at 0.2 the ramp began at
    /// 80 per cent of the run while the growth window closed at 85, so
    /// around 90,000 Gaussians were driven to zero opacity and pruned after
    /// densification could no longer replace them. Turning it off took the
    /// final model from 150,554 splats to 299,883 and held-out PSNR from
    /// 16.00 to 17.30, against a measured noise band of +/-0.4 dB.
    var binarizeLastFraction: Float = 0
    var binarizeWeight: Float = 0.02
    /// Free-space carving deletion sweep interval, iterations.
    /// MUST BE A MULTIPLE OF `densifyIntervalIterations`. The carver only runs
    /// inside a densification pass: MetalSplatTrainer tests
    /// `iteration % carveIntervalIterations == 0` inside a block already gated
    /// on `iteration % densifyIntervalIterations == 0`, so it fires on the
    /// least common multiple of the two. This read 250 against 100, and build
    /// 250's census shows it ran at 500, 1000, ... 3500: seven sweeps. 500 is
    /// what has actually been running and the number now says so. Changing
    /// the real rate (200 would give 19 sweeps) is a separate, measured change.
    var carveIntervalIterations: Int = 500
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
    // 0.002 (build 352; was 0.004). Our points' median long axis is 7.1 mm against
    // 3.4 mm in the Scaniverse reference; at 0.004 of the extent only points
    // over about 2 cm were ever split.
    var splitScaleFraction: Float = 0.002
    /// How many children a split produces.
    ///
    /// Descriptive, not read. `TrainerDensifier` splits by shrinking the
    /// parent in place and appending exactly one sibling, which is two
    /// children for one slot, and that two is a property of the code rather
    /// than a number it looks up. Changing this alone changes nothing.
    var splitChildCount: Int = 2
    /// Children are placed at +/- this many standard deviations along the
    /// split axis, and shrunk by `splitShrink`.
    /// WHAT FRACTION OF EACH GROWTH PASS SPLITS RATHER THAN CLONES.
    ///
    /// The candidates are already ranked by AbsGS score and truncated to the
    /// allowance. This ranks that surviving list a second time, by world
    /// scale, and sends the largest `splitShareOfGrowth` of it down the split
    /// path. 0 falls back to the threshold tests alone.
    ///
    /// WHY A SHARE AND NOT A THRESHOLD. Both thresholds tried so far failed
    /// the same way: each was a magnitude in a unit whose distribution nobody
    /// had measured.
    ///
    ///   `splitScaleFraction` 0.01 put the bar at 5.39 cm against a model
    ///   whose LARGEST Gaussian is 6.18 cm, so split fired 190 times against
    ///   152,801 clones.
    ///
    ///   `splitScreenRadiusPx` 10 put the bar below the 1st percentile of its
    ///   own statistic, so split took 6,836 of the 6,900 slots available.
    ///
    /// A share has no unit to be wrong about, and it is the same fix this
    /// file's densification gate already received when `absGradThreshold`
    /// broke on pixels-versus-NDC: compare against zero and let the ranked
    /// truncation do the cutting.
    ///
    /// 0.2 is the measured reference behaviour. arXiv:2507.20239 reports
    /// roughly 80 per cent clone / 20 per cent split for 3DGS and shows the
    /// two do different jobs: split-dominated Gaussians displace 2.42 / 0.75 /
    /// 2.40 scene units against 0.09 / 0.15 / 0.10 for clone-dominated ones.
    /// Split diffuses, clone refines. We had almost none of the first, and
    /// then briefly almost none of the second.
    /// 0.8, NOT 0.2, AND THE SIMULATION IS WHY.
    ///
    /// A CLONE COPIES THE PARENT'S SCALE EXACTLY (TrainerDensifier, the clone
    /// branch leaves logScale untouched), so every clone is a Gaussian that
    /// cannot move the size distribution. At 0.2, four fifths of every growth
    /// pass was doing nothing about the one thing that is wrong with this
    /// model.
    ///
    /// A simulation of the population's size distribution under the real
    /// schedule, at identical population (298,619) and identical total created
    /// (154,608), sweeping this constant:
    ///
    ///     0.0 -> 1.877x spread     0.6 -> 2.974x
    ///     0.2 -> 2.132x            0.8 -> 3.820x
    ///     0.4 -> 2.399x            1.0 -> 5.183x
    ///
    /// against a target of 8.9x, which is what the Scaniverse capture of the
    /// same room measures. This is the largest lever tested and it costs
    /// nothing: same passes, same allowance, same work.
    ///
    /// WHY NOT 1.0. Coverage. A split replaces one Gaussian with two at 1/1.6
    /// on all three axes, so covered volume goes to 48.8 per cent each time; a
    /// clone ADDS coverage. Measured total disc face area against the 150k
    /// seed set: 4.21x at share 0.2, 2.27x at share 1.0. Still ample either
    /// way, but the margin shrinks, and filling the holes between thinned
    /// seeds is exactly what clones are for. 0.8 keeps a fifth of growth doing
    /// that.
    ///
    /// AND AN HONEST NOTE ON WHERE 0.2 CAME FROM. It matched arXiv:2507.20239's
    /// reported 80/20 clone/split for stock 3DGS. That reference is for a
    /// population whose INITIALISATION already has a size distribution. Ours
    /// does not: every seed in this scan is 7.397 mm, at every percentile from
    /// p1 to p99, because the spacing floor wins for every sample. Copying a
    /// ratio tuned for a varied starting population onto a perfectly uniform
    /// one was the wrong reference class.
    var splitShareOfGrowth: Float = 0.8

    /// SPLIT ON SCREEN SIZE. WITHDRAWN, MEASURED WRONG, LEFT AT 0.
    ///
    /// The idea was that `stats.maxRadiusPxBits` identifies a Gaussian that is
    /// over-reconstructed for the view being fitted, and that 10 px would cut
    /// somewhere in the top few per cent. An offline copy of
    /// `trainer_preprocess`, run over the owner's real model and real poses
    /// and reproducing the census's peakTileInstances to 0.07 per cent, says
    /// otherwise: the median of that statistic is 40 to 49 px depending on how
    /// many views are sampled, and 10 px selects **99.18 per cent** of the
    /// drawn population. 16 px selects 96 per cent. 48 px still selects 52.
    ///
    /// Moving the number cannot fix it. `maxRadiusPxBits` is a MAX over every
    /// view in the interval and the 3-sigma radius scales with 1/z, so one
    /// close-up frame sets it for the whole population, and the median climbs
    /// as more views are sampled. 4.2 per cent of drawn Gaussians record a max
    /// radius wider than the entire 720 px frame. The mean alpha extent, the
    /// obvious alternative, is no better: a fixed 8 px on it selects 99.46 per
    /// cent and 24 px selects 47.89, with no usable knee anywhere.
    ///
    /// Kept at 0 rather than deleted because build 182 ran with it at 10, and
    /// that census only makes sense read next to this note.
    ///
    /// Split fired 190 times against 152,801 clones in the measured run:
    /// 0.12 per cent, where the reference implementation runs about 20 per
    /// cent (arXiv:2507.20239 measures ~80/20 clone/split). That is not a
    /// tuning miss, it is a criterion that cannot fire. The world-scale test
    /// is `largest linear scale > sceneExtent * splitScaleFraction`, and the
    /// owner's own exported model has a median splat of 1.3 cm, p99 of
    /// 3.85 cm and a single largest Gaussian of 7.86 cm in a 5.39 m room.
    /// There is no world-scale tail to select, so the test selects nothing,
    /// and a population that never splits can never get finer than its seed
    /// spacing. Ours is 13.2 mm against Scaniverse's 3.4 mm on the same room.
    ///
    /// Screen radius is the quantity that actually says "this Gaussian is
    /// over-reconstructed for the view it is being fitted to", it is already
    /// measured per splat in `stats.maxRadiusPxBits`, and it selects
    /// near-camera blur that the world-scale test is blind to. The census
    /// gives 763,260 tile instances over 300,000 splats, so 2.545 tiles per
    /// splat on a 16 px grid, which solves to a median projected half-extent
    /// of about 4.8 px. 10 px therefore cuts somewhere in the top few per
    /// cent, which is the right order for a criterion meant to fire on the
    /// worst offenders rather than on everything.
    var splitScreenRadiusPx: Float = 0

    /// SHRINK ALL THREE AXES ON A SPLIT, as the reference does, instead of
    /// only the split axis. Set false to restore the one-axis behaviour.
    ///
    /// This MUST move together with `splitScreenRadiusPx`. Our one-axis
    /// shrink gives two children covering about 2/1.6 = 1.25x the parent's
    /// screen area, so every split makes the tile count WORSE. At 190 splits
    /// that is invisible; at the rate the screen-radius criterion produces
    /// it would add roughly 29,000 tile instances. Shrinking all three axes
    /// by the same 1.6 removes about 25,000 instead. The reference divides
    /// the whole scale vector by 0.8*N = 1.6 for N=2, which is exactly our
    /// splitShrink, so this is the reference geometry and not a new guess.
    ///
    /// THE COST, stated plainly: the one-axis shrink was a deliberate
    /// SAD-GS choice, and it preserves a property we care about, that a disc
    /// which was the right size across a flat measured surface stays the
    /// right size across it. Shrinking all three axes narrows the children
    /// across the surface too, which is why the offset below stops being a
    /// fixed +/- along one axis and becomes a sample from the parent's own
    /// covariance, scattering the children through the parent's volume the
    /// way the reference does. On a LiDAR-measured flat wall that is the
    /// real risk in this batch.
    var splitShrinkAllAxes: Bool = true

    var splitOffsetSigma: Float = 0.8
    var splitShrink: Float = 1.6
    /// Opacity below which a Gaussian is pruned, as a probability.
    /// WAS 0.005, which is 1/196, against a minAlpha of 1/255 = 0.00392.
    /// Those are the same number to within a rounding error, so a Gaussian
    /// had to be at or below the threshold at which it stops being DRAWN
    /// before it became eligible to be pruned and its slot recycled. An
    /// exported 3000-iteration run measured a median peak alpha of 0.0038
    /// with 50.1 per cent of the population below minAlpha: half the model
    /// was invisible, ungraded (the Adam passes gate on visibleFlag) and
    /// un-prunable at the same time, which is a population that can only
    /// grow. 0.02 is five times the render threshold, so a Gaussian is
    /// recycled while it is still fading rather than after it has gone.
    // 0.08 (build 352; was 0.02). The finished 348 model carried 8 % of its
    // points below 0.1 opacity against 1 % in the Scaniverse reference of the
    // same room; those slots are wanted for splits of the large points.
    var pruneOpacity: Float = 0.08
    /// Screen radius above which a Gaussian is pruned, in pixels.
    /// 720, NOT 0. This prune has been fully wired and completely dead.
    ///
    /// `TrainerDensifier` already reads `stats[i].maxRadiusPxBits` and already
    /// deletes anything above this, and 0 disables the test entirely, so the
    /// code has never once run. Measured BEFORE the build-250 tangent clamp: a
    /// 720 px threshold selected 1,060 splats of 298,000, carrying 13.44 per
    /// cent of every tile instance. The clamp removed every one of them. On
    /// build 250's model the clamped max-over-views radius peaks at 243.7 px
    /// over the 108 trained keyframes and 519.7 px over all 868 poses, so this
    /// prune removes nothing (prunedOversized 0) and is kept only as a guard.
    /// Lower cutoffs buy nothing: 120 px is 0.30 per cent of tile instances,
    /// 90 px is 1.27 and costs one held-out view 35 dB against the unpruned
    /// model.
    ///
    /// The margin is not close. The 99th percentile of projected 3-sigma
    /// radius over drawn splats is 79.1 px, so 720 sits nine times above the
    /// p99 and cannot touch anything a person would call a normal Gaussian.
    /// It is the whole render's long edge: a splat claiming more than the
    /// frame is not detail, it is a projection artefact.
    ///
    /// Belt and braces with the tangent clamp in trainer_preprocess, which
    /// stops most of these being created. This catches whatever still gets
    /// through.
    var pruneMaxScreenRadiusPx: Float = 720.0
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
    /// INVARIANT: this MUST stay above `pruneOpacity`, or the donor pool is
    /// empty by construction. A Gaussian below `pruneOpacity` is deleted at
    /// the end of the same pass, so it can never be picked up and reused; the
    /// band between the two thresholds IS the donor pool.
    ///
    /// Both were 0.02, which made that band empty. Measured on build 182's
    /// exported model, only 113 Gaussians of 299,209 (0.04 per cent) sat below
    /// 0.02 at all, so essentially every donor the census recorded came from
    /// the OTHER half of the test, `visAccum <= 0`, meaning "not seen in any
    /// frame this interval" rather than "faint". 24,222 donors across 29
    /// passes, 835 a pass, against a `maxRelocationFractionPerPass` allowance
    /// of 15,000. Donors were the binding limit by a factor of eighteen.
    ///
    /// That matters because relocation is not a side mechanism. It is a SPLIT
    /// that uses a dead Gaussian's slot, and once the population reaches its
    /// cap around iteration 600 it is the ONLY thing still able to make a
    /// Gaussian smaller, for the remaining 80 per cent of the run. Starving it
    /// is what keeps our size distribution 1.8x wide against a good model's
    /// 8.9x.
    ///
    /// 0.05 is chosen so the pool just clears the allowance rather than
    /// dwarfing it: 22,005 Gaussians (7.35 per cent) sit below it, against an
    /// allowance of 5 per cent. A Gaussian at alpha 0.02 to 0.05 contributes
    /// two to five per cent of a pixel, and the relocation's opacity
    /// correction hands its coverage to the target it lands on.
    ///
    /// THE RISK, stated plainly: this raises relocation from about 835 a pass
    /// to as many as 15,000, which is 5 per cent of the population churned per
    /// pass with fresh Adam state each time. That is the 3DGS-MCMC design
    /// working as intended, but it is a large change in how much the model
    /// moves, and it is the first constant to look at if the run becomes
    /// unstable.
    var relocationDonorOpacity: Float = 0.05

    // --- Mip-Splatting --------------------------------------------------------

    /// Mip-Splatting's filter scale constant.
    var filter3DScale: Float = 0.2
    /// Build 356: the share of the training-grid 3D filter width folded into
    /// the EXPORTED sizes. 0.5 is the filter of a render at twice the training
    /// long edge (1,440 px), which is the least any viewer draws at.
    var exportFilter3DScale: Float = 0.75
    /// Build 364: the reference recipe's opacity reset. Every this many
    /// iterations, inside the densify window, every point's opacity is
    /// clamped down to `opacityResetTo` and the optimiser's memory of the
    /// opacity lane is cleared, so a point that was only alive by habit has
    /// to earn its opacity again. 0 turns it off.
    // OFF (build 378). 376's census: the reset at 24,000 (the first at the
    // 1,080-px size) was followed by a prune that took the population from
    // 330,000 to 131,000 and the held-out score from 15.9 to 14.1 dB, and the
    // score before any reset had reached was already below build 338's. With
    // relocation treating every reset point as a donor, the recipe's reset
    // does not transplant.
    var opacityResetIntervalIterations: Int = 0
    var opacityResetTo: Float = 0.01
    /// Faint-point pruning waits this long after a reset, so the points that
    /// are recovering are not the ones it removes (the reference prunes at
    /// 0.005, below the reset value; ours prunes at 0.08 and needs the wait).
    var opacityResetPruneGraceIterations: Int = 500
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

    /// BUILD 330: 1e-4, ten times build 292's 1e-5. At 1e-5 the census
    /// measured the corrections at a median 0.01 cm and a maximum 0.13 cm
    /// over a whole run: the refinement was switched off in all but name,
    /// while the pose graph's own residual is about 3 cm. These rates are
    /// only used in full after the pose check (`poseRefinementCheck`) has
    /// shown on device that one step along the gradient lowers the
    /// photometric loss; until then the trainer multiplies them by 0.1, the
    /// inert value.
    /// 3e-4 since build 332: at 1e-4 the checked refinement moved the
    /// cameras a median 0.08 cm and at most 0.43 cm, against a pose-graph
    /// residual near 3 cm. The per-step clamps and the ceiling are unchanged.
    var cameraRotationLR: Float = 3e-4
    var cameraTranslationLR: Float = 3e-4
    /// Build 330: at the first profiled step past warm-up and the coarse
    /// phase, render the frame at its current camera correction and at that
    /// correction plus one gradient step, both after the step's own Adam
    /// update, and compare the photometric losses. The refinement runs at
    /// its full rate only if the second is lower. Off means the full rate
    /// from the start.
    var poseRefinementCheck: Bool = true
    /// Hard per-step clamp so a single bad frame cannot throw a camera.
    var cameraMaxRotationStepRadians: Float = 0.0005
    var cameraMaxTranslationStepMeters: Float = 0.002

    // --- Held-out evaluation ------------------------------------------------------

    /// Fraction of keyframes withheld from training and used only to report
    /// PSNR. Reported whatever it says.
    /// WAS 0.05, which on this app's 120 keyframes gives a step of 20 and
    /// so exactly SIX held-out frames. Six frames is why the PSNR noise band
    /// is +/-0.4 dB, measured across five runs of identical code, and a
    /// +/-0.4 dB band cannot read a change worth 0.3 dB. 0.10 gives twelve
    /// frames and roughly halves the band. It costs six training views out
    /// of 114, which is the deliberate trade: a diagnostic that cannot
    /// resolve the changes being made is not worth its training data.
    var heldOutFraction: Float = 0.10

    /// Held-out frames FIXED by frame index (build 276): every frame with
    /// index % 40 == 20 is kept out of the keyframe walk and, inside the
    /// trained span, scored. Independent of the walk and of the poses, so a
    /// pose-graph or selector change no longer swaps the test set. 0 or 1
    /// falls back to every tenth chosen keyframe (the old split).
    var heldOutFrameStride: Int = 40

    /// Every this many iterations, command buffer B is split into its five
    /// stages (sort, forward, losses, backward, optimiser) and each is timed
    /// on the GPU (build 286). 16 samples a 4,000-iteration run at about 1 ms
    /// of extra round trips each. 0 turns it off.
    var stageProfileEvery: Int = 250

    /// Build 288: from this iteration, for this many, run both backward
    /// rasterisers on the same inputs, compare and time them, and keep the
    /// SIMD-summed one only if it agrees (relative L1 < 1e-3) and is at least
    /// 3 % faster. Past the resolution levels (build 332: 2,500+, full size
    /// begins at 55 % of a 4,500-round run), so the kernels are compared at
    /// the size they will run at for the rest of the run.
    var backwardCalibrationStart: Int = 24200
    var backwardCalibrationSteps: Int = 6

    /// Build 290: the same for the radix scatter. After the backward window,
    /// so the two never share an iteration. Build 306: this window also checks
    /// the splat-order sort against the legacy one (see `splatOrderSort`).
    var sortCalibrationStart: Int = 24210
    var sortCalibrationSteps: Int = 4

    /// Build 302: the same for the two-pixel forward rasteriser, after the
    /// sort window.
    var forwardCalibrationStart: Int = 24216
    var forwardCalibrationSteps: Int = 4

    /// Build 304: the two-pixel backward against the backward the first
    /// window chose (plain or SIMD-summed), after the forward window. Kept
    /// only if its gradients agree (relative L1 < 1e-3) and it is at least
    /// 3 % faster.
    var backwardTwoPixelCalibrationStart: Int = 24222
    var backwardTwoPixelCalibrationSteps: Int = 6

    /// Build 318: the fused SSIM blur against the two-pass one, after the
    /// two-pixel backward window. Kept only if every blurred plane matched
    /// BIT FOR BIT on every step and it was at least 3 % faster.
    var blurCalibrationStart: Int = 24228
    var blurCalibrationSteps: Int = 2

    /// Build 306: allow the splat-order tile sort (depth-sort the splats, then
    /// sort instances on their tile bits only). Used only after the sort
    /// calibration window has shown it reproduces the legacy order exactly and
    /// is at least 3 % faster; the legacy sort runs until then.
    var splatOrderSort: Bool = true

    /// Build 292: after warm-up, commit the next iteration's buffer A BEFORE
    /// waiting on the previous iteration's buffer B, so the GPU does not sit
    /// idle through the CPU's per-iteration work. See `drainPendingStep`.
    var overlapIterations: Bool = true

    /// Build 316: an overlapped step is ONE command buffer, with the sort
    /// sized by a kernel and dispatched indirectly, so the CPU never waits
    /// between the tile scan and the rest of the step. See
    /// MetalSplatTrainer.runMergedIteration. Off means build 292's two
    /// buffers with the count read back in between.
    var mergedCommandBuffer: Bool = true

    /// Build 324: warm-up steps are overlapped too. Their one extra
    /// read-back, the far field's gradient, is copied into a staging slot by
    /// the step's own command buffer and accumulated when the step completes,
    /// and a step about to apply the accumulated update is completed before
    /// the next frame's supervision is built, so every update lands where
    /// the synchronous path landed it.
    var overlapWarmup: Bool = true

    /// BUILD 328: COARSE TO FINE IN RESOLUTION. The first `coarseResolutionFraction`
    /// of the run renders and is supervised at `coarseResolutionScale` of the
    /// render size (720 x 540 -> 360 x 270): a quarter of the pixels, and
    /// the Gaussians' footprints shrink with them, so a step costs about a
    /// third. This is the same phase the trainer already low-passes with
    /// `frequencyBlurStartVariance` (the blur is scaled by the square of the
    /// scale so it is the same blur in the photograph), and the
    /// densification score is a ranking, so it does not depend on the pixel
    /// unit. The far-field warm-up (warmupFraction 0.15) is the same span.
    /// Everything after it, including every held-out evaluation and every
    /// calibration window, runs at the full size. 0 switches it off.
    /// BUILD 332: LEVELS. Level i trains at `coarseResolutionScales[i]` of
    /// the render size until `coarseResolutionFractions[i]` of the run; after
    /// the last fraction the run is at full size. Build 330 measured a
    /// half-size step at 3.7 ms against 12.5 at full size, and the first
    /// full-size score after 675 half-size steps was the same as build
    /// 320's, so more of the run goes below full size: half until 30 %,
    /// three quarters until 55 %, full for the last 45 % (2,025 rounds of
    /// 4,500). Empty arrays switch it off.
    // Build 340: 330's 4,500 full-size steps scored 19.54 and 338's 2,025
    // scored 19.53, so full-size steps past about 2,000 bought nothing; the
    // last quarter is full size, the rest at 0.5 and 0.75.
    // Build 360: 360 px to 45 %, 720 px to 80 %, then 1,080 px.
    // Build 380: 1,080 px for the last 30 % (378's 1,080 phase was 1,600
    // iterations before early stop, with growth frozen and no cache).
    var coarseResolutionFractions: [Float] = [0.40, 0.70]
    var coarseResolutionScales: [Float] = [0.3333, 0.6667]
    /// BUILD 338: OFF. The background preload (build 332) put a second
    /// builder to work while the active level's prefetch worker built on
    /// another, the first time two frames were ever built at once for the
    /// whole of a level, and builds 334 and 336 died at random points inside
    /// exactly those windows (rounds 20, 200, about 1,300) while build 330,
    /// which never had two builders running, completed every run. Every
    /// level now decodes its frames on first visit, as build 330 did.
    var supervisionPreload: Bool = false


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

    /// `withElements` from `byteOffset` bytes in (build 324: the warm-up
    /// staging holds a step's gradient planes end to end).
    func withElements<T, R>(
        _ type: T.Type,
        count: Int,
        byteOffset: Int,
        _ body: (UnsafeBufferPointer<T>) -> R
    ) -> R? {
        guard count > 0, byteOffset >= 0,
              length >= byteOffset + count * MemoryLayout<T>.stride
        else { return nil }
        let raw = contents().advanced(by: byteOffset).bindMemory(to: T.self, capacity: count)
        return body(UnsafeBufferPointer(start: raw, count: count))
    }

    /// Reads `count` elements of `T` from the front of the buffer.
    func readArray<T>(_ type: T.Type, count: Int) -> [T] {
        guard count > 0, length >= count * MemoryLayout<T>.stride else { return [] }
        let raw = contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }

    /// Reads `count` elements of `T` starting `byteOffset` bytes in (build
    /// 322: the staging buffers hold several arrays end to end). Empty when
    /// the range does not fit, like `readArray(_:count:)`.
    func readArray<T>(_ type: T.Type, count: Int, byteOffset: Int) -> [T] {
        guard count > 0, byteOffset >= 0,
              length >= byteOffset + count * MemoryLayout<T>.stride
        else { return [] }
        let raw = contents().advanced(by: byteOffset).bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }

    /// Writes `values` to the front of the buffer. Silently writes as many as
    /// fit, and returns how many that was, so an over-long write is a number
    /// the caller can check rather than a heap corruption.
    ///
    /// ONE memcpy, NOT an element-at-a-time loop.
    ///
    /// It was a loop, `for i in 0..<fits { raw[i] = values[i] }`, and the
    /// trainer calls this three times per iteration with the frame's ground
    /// truth, its background and its depth samples: about 2.3 million Floats
    /// plus 49,000 structs, every iteration, each one going through Array
    /// subscripting with its bounds check and retain traffic rather than a
    /// bulk copy.
    ///
    /// Measured on the owner's phone at build 108: 35.8 seconds of a 126
    /// second run, 11.95 ms per iteration, 28% of the whole thing. It was the
    /// largest single item left anywhere in the loop and it was invisible
    /// until `timings.upload` existed. An adversarial reviewer had dismissed
    /// this exact finding as "0.3-0.5 ms per iteration, below the bar"; the
    /// measurement says it was wrong by about twenty-five times, which is the
    /// argument for measuring rather than arguing.
    ///
    /// Safe as a raw byte copy: these buffers are storageModeShared, so
    /// `contents()` is the same physical memory the GPU reads, and every T
    /// written here is a trivially-copyable Float or POD struct shared with
    /// the Metal side. `copyFront` elsewhere in this file already does exactly
    /// this for the same reason.
    @discardableResult
    func writeArray<T>(_ values: [T]) -> Int {
        let fits = Swift.min(values.count, length / Swift.max(MemoryLayout<T>.stride, 1))
        guard fits > 0 else { return 0 }
        values.withUnsafeBytes { source in
            guard let base = source.baseAddress else { return }
            memcpy(contents(), base, fits * MemoryLayout<T>.stride)
        }
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
