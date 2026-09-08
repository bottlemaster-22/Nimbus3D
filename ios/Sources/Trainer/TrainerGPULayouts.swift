//
//  TrainerGPULayouts.swift
//  Trainer
//
//  THE SWIFT SIDE OF EVERY BUFFER `TrainerShaders.metal` READS OR WRITES.
//
//  Each struct here is byte-matched to a struct of the same name in
//  `TrainerShaders.metal`. There is no shared header: a single XcodeGen target
//  has no bridging-header hook that both the Swift compiler and the Metal
//  compiler see, so the two declarations are kept in step by hand and checked
//  at run time by `TrainerGPULayouts.verify()`, which `MetalSplatTrainer`
//  calls once before it allocates anything. `Sources/Viewer` does exactly the
//  same thing for its own buffers; this file follows that precedent
//  deliberately.
//
//  RULES THAT KEEP THE LAYOUTS MATCHED (break one and you get plausible
//  garbage, which costs a day to find):
//
//   * Per-element buffer structs use ONLY `Float` and `UInt32` fields on the
//     Swift side and ONLY `packed_*` vector types plus `float` / `uint` on the
//     Metal side. Every field is therefore 4-byte aligned on both sides and
//     Swift's natural field packing matches MSL's exactly.
//   * NOTHING here is `Float16`. `Float16` does not exist on x86_64, so a
//     struct using it refuses to build in an Intel-Mac simulator even though
//     the shipping slice is arm64-only.
//   * The uniform structs are `constant` buffer arguments in MSL, which uses
//     standard (not packed) alignment. Their Swift mirrors therefore use
//     `simd_float4x4` / `SIMD4<Float>` and are padded to a multiple of 16.
//   * `SIMD3<Float>` is NEVER used in a per-element struct: it is 16 bytes in
//     Swift and `packed_float3` is 12 in Metal. Spherical harmonics are held
//     as a flat `[Float]`, three floats per coefficient.
//
//  SIZES, restated so a reviewer can check them without a compiler:
//
//      TrainerSplat            48 bytes, align 4
//      TrainerSplatGrad        48 bytes, align 4
//      TrainerSplatStats       32 bytes, align 4
//      TrainerSplatDraw        64 bytes, align 4
//      TrainerSamplingTopK     16 bytes, align 4
//      TrainerDepthSample      32 bytes, align 4
//      TrainerCameraUniforms  144 bytes, align 16
//      TrainerLossUniforms     68 bytes, align 4
//      TrainerAdamUniforms     64 bytes, align 4
//      TrainerRegUniforms      32 bytes, align 4
//      TrainerScanUniforms     16 bytes, align 4
//      TrainerRadixUniforms    16 bytes, align 4
//      TrainerBlurUniforms     16 bytes, align 4
//

import Foundation
import simd

// MARK: - Fixed GPU constants
//
// These are duplicated as `constant` values at the top of TrainerShaders.metal
// and verified against it by `TrainerGPULayouts.verify()` only in the sense
// that a mismatch here is a review error, not a runtime one: the Metal
// compiler cannot see this file. They are collected in one place so a change
// is a two-line diff in two named places rather than a hunt.

enum TrainerGPUConstants {
    /// Rasteriser tile edge, pixels. 16x16 = 256 threads = one threadgroup.
    static let tileWidth = 16
    static let tileHeight = 16
    static var tileArea: Int { tileWidth * tileHeight }

    /// Threads per block for the prefix-scan and radix-sort kernels.
    static let scanThreads = 256
    /// Elements each scan/sort thread handles.
    static let scanElementsPerThread = 4
    /// Elements one scan/sort block covers.
    static var scanBlockElements: Int { scanThreads * scanElementsPerThread }

    /// Radix digit width. 4 bits keeps the per-thread histogram
    /// (16 bins x 256 threads x 4 bytes = 16 KB) inside the 32 KB threadgroup
    /// allocation every Apple GPU guarantees. 8-bit digits would need 256 KB
    /// and do not fit, which is why the sort is 8 passes and not 4.
    static let radixBits = 4
    static var radixBins: Int { 1 << radixBits }
    /// Sort keys are 32 bits: `(tileID << 16) | quantisedDepth16`.
    static let radixKeyBits = 32
    static var radixPasses: Int { radixKeyBits / radixBits }

    /// Hard ceiling on tile count, because the sort key gives the tile id 16
    /// bits. 65535 tiles is 4096x4096 pixels at 16 px tiles; the trainer
    /// renders at 384-720 px on its long edge, so this is never close.
    static let maxTileCount = 65_535

    /// SSIM window: 11 taps, sigma 1.5, the constants the SSIM paper and every
    /// 3DGS implementation use.
    ///
    /// These two are READ, by `ssimGaussianWindow()` below and by
    /// `TrainerGPULayouts.verify()`. They were decorative until the window
    /// they describe was checked against the one the GPU actually blurs with,
    /// and in that gap the GPU's copy drifted: see `ssimBlurWeights`.
    static let ssimWindowRadius = 5
    static let ssimSigma: Float = 1.5

    /// The eleven taps `trainer_blur_h` and `trainer_blur_v` blur with,
    /// transcribed from TrainerShaders.metal.
    ///
    /// The kernel writes its taps out as literals rather than calling exp() in
    /// a loop with fast-math enabled, which is right, and which is also how
    /// they drifted. The table the GPU shipped with summed to 0.99752 and had
    /// a centre tap of 0.26361500 where the normalised sigma-1.5 Gaussian
    /// wants 0.26601172, so every separable blur lost about half a percent of
    /// its energy and the SSIM half of the loss was computed against slightly
    /// wrong local means. Nothing caught it, because nothing compared the
    /// numbers on the GPU with the sigma written down on the Swift side.
    ///
    /// `verify()` now does exactly that comparison, so this table and the two
    /// constants above cannot disagree without the trainer refusing to start.
    static let ssimBlurWeights: [Float] = [
        0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554,
        0.26601172,
        0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838
    ]

    /// The window `ssimWindowRadius` and `ssimSigma` describe, normalised to
    /// sum to one. This is the truth `ssimBlurWeights` is checked against.
    static func ssimGaussianWindow() -> [Float] {
        let radius: Int = ssimWindowRadius
        let sigma: Float = ssimSigma
        var weights: [Float] = []
        weights.reserveCapacity(radius * 2 + 1)
        var total: Float = 0
        var tap: Int = -radius
        while tap <= radius {
            let x = Float(tap)
            let w: Float = expf(-(x * x) / (2 * sigma * sigma))
            weights.append(w)
            total += w
            tap += 1
        }
        guard total > 0 else { return weights }
        var i: Int = 0
        while i < weights.count {
            weights[i] = weights[i] / total
            i += 1
        }
        return weights
    }

    /// Number of blurred planes the SSIM forward and backward passes carry.
    static let ssimPlaneCount = 5

    /// How many top sampling rates are kept per Gaussian for the Mip-Splatting
    /// 3D filter. The spec asks for "a high percentile of observed sampling
    /// rate, not the max"; keeping the top 4 and using the 4th largest is that
    /// percentile, computed in O(1) memory per Gaussian.
    static let samplingTopK = 4
}

// MARK: - Per-Gaussian parameters

/// One Gaussian's learnable parameters, exactly as the GPU stores them.
///
/// Conventions are `SplatCloud`'s, unchanged, so a trained field can be handed
/// to `Sources/Export` without a single unit conversion:
///
///  * `mean` is world space, metres. The world frame is ARKit's (right-handed,
///    Y up), which is also `SplatCloud`'s RUB frame, so nothing is flipped.
///  * `rotation` is `(x, y, z, w)`, normalised by the Adam kernel after every
///    step.
///  * `logScale` is the raw parameter; `exp()` is applied on the GPU.
///  * `opacityLogit` is pre-sigmoid.
///  * `flags` is the `InitialSplatSetRef` flag byte, widened: bit 0 position
///    pinned, bit 1 elongated along the viewing ray, bit 2 on a detected 3D
///    edge curve (exempt from the disc prior, F4).
struct TrainerSplat {
    var rotX: Float = 0          // offset  0
    var rotY: Float = 0          // offset  4
    var rotZ: Float = 0          // offset  8
    var rotW: Float = 1          // offset 12
    var meanX: Float = 0         // offset 16
    var meanY: Float = 0         // offset 20
    var meanZ: Float = 0         // offset 24
    var opacityLogit: Float = 0  // offset 28
    var logScaleX: Float = 0     // offset 32
    var logScaleY: Float = 0     // offset 36
    var logScaleZ: Float = 0     // offset 40
    var flags: UInt32 = 0        // offset 44
    // stride 48

    var mean: SIMD3<Float> {
        get { SIMD3(meanX, meanY, meanZ) }
        set { meanX = newValue.x; meanY = newValue.y; meanZ = newValue.z }
    }

    var logScale: SIMD3<Float> {
        get { SIMD3(logScaleX, logScaleY, logScaleZ) }
        set { logScaleX = newValue.x; logScaleY = newValue.y; logScaleZ = newValue.z }
    }

    var rotation: SIMD4<Float> {
        get { SIMD4(rotX, rotY, rotZ, rotW) }
        set { rotX = newValue.x; rotY = newValue.y; rotZ = newValue.z; rotW = newValue.w }
    }
}

/// Per-Gaussian flag bits. Same bit assignment as
/// `InitialSplatSetRef.flagsPath` (docs/DATA_FORMAT.md section 7), widened
/// from a byte to a `UInt32` so the GPU can read it without a byte gather.
enum TrainerSplatFlag {
    /// Trusted LiDAR sample: thin, opaque, held in place. The position learning
    /// rate is scaled down for these rather than zeroed, so photometry can
    /// still correct a genuinely wrong prior.
    static let positionPinned: UInt32 = 1 << 0
    /// Doubtful sample: elongated along the viewing ray, free to slide.
    static let elongatedAlongRay: UInt32 = 1 << 1
    /// Lies on a detected 3D edge curve: exempt from the disc / effective-rank
    /// prior, which would otherwise flatten a genuine crease (F4).
    static let onEdgeCurve: UInt32 = 1 << 2
    /// Set by the densifier on a Gaussian it created this run. Diagnostic only.
    static let densified: UInt32 = 1 << 3
}

/// Gradient of the loss with respect to one Gaussian's parameters.
///
/// The backward kernels write this buffer through a parallel Metal struct of
/// `atomic_float` fields with an identical layout, because a Gaussian is
/// touched by every pixel of every tile it covers and by several frames.
/// Twelve floats, in the same order as `TrainerSplat` minus `flags`.
struct TrainerSplatGrad {
    var rot0: Float = 0     // offset  0
    var rot1: Float = 0     // offset  4
    var rot2: Float = 0     // offset  8
    var rot3: Float = 0     // offset 12
    var mean0: Float = 0    // offset 16
    var mean1: Float = 0    // offset 20
    var mean2: Float = 0    // offset 24
    var opacity: Float = 0  // offset 28
    var scale0: Float = 0   // offset 32
    var scale1: Float = 0   // offset 36
    var scale2: Float = 0   // offset 40
    var pad: Float = 0      // offset 44 - keeps the stride at 48 and the
                            //             atomic mirror exactly 12 slots wide
    // stride 48
}

/// Per-Gaussian bookkeeping: densification statistics, visibility, the
/// Mip-Splatting 3D filter size, and the sparse-Adam step counter.
///
/// `maxRadiusPxBits` holds a `Float` bit pattern rather than a `Float` because
/// Metal has no atomic float max. The value is always non-negative, and for
/// non-negative IEEE-754 floats the unsigned integer ordering of the bit
/// pattern is the same as the float ordering, so `atomic_fetch_max_explicit`
/// on the bits is exactly a float max.
struct TrainerSplatStats {
    /// AbsGS: sum over pixels and frames of the MAGNITUDE of the per-pixel
    /// screen-space position gradient, accumulated BEFORE any cancellation.
    /// This is the whole point of AbsGS: the plain 3DGS statistic sums signed
    /// gradients, which cancel across a Gaussian that straddles an edge, and
    /// so it never densifies the one place that needs it most.
    var absGrad2D: Float = 0        // offset  0
    /// Number of (splat, frame) observations that contributed. Pixel-GS's
    /// pixel-area weighting is already in `absGrad2D`, which grows with the
    /// number of pixels covered; dividing by frames rather than by pixels is
    /// what keeps it there.
    var denom: Float = 0            // offset  4
    /// Largest projected radius in pixels seen this interval, as float bits.
    var maxRadiusPxBits: UInt32 = 0 // offset  8
    /// Sum of `alpha * T` over every pixel this Gaussian contributed to.
    var visAccum: Float = 0         // offset 12
    /// The same sum restricted to pixels whose supervision is UNKNOWN (glass,
    /// out of range, low confidence). `unknownAccum / visAccum` is the gate
    /// that switches OFF late opacity binarization for this Gaussian (F4).
    var unknownAccum: Float = 0     // offset 16
    /// Mip-Splatting 3D low-pass filter standard deviation, world metres.
    var filter3D: Float = 0         // offset 20
    /// 1 when this Gaussian was rasterised in the current step. The mask that
    /// makes Adam sparse.
    var visibleFlag: UInt32 = 0     // offset 24
    /// Per-Gaussian Adam step count, for bias correction under sparse updates.
    /// A Gaussian seen in 3 of 3000 steps must be bias-corrected as if it were
    /// on step 3, not step 3000.
    var stepCount: UInt32 = 0       // offset 28
    // stride 32
}

/// What `trainer_preprocess` computed for one Gaussian and one camera, read
/// back by the rasteriser and by both backward kernels.
struct TrainerSplatDraw {
    var meanCamX: Float = 0     // offset  0   camera-space centre, metres
    var meanCamY: Float = 0     // offset  4
    var meanCamZ: Float = 0     // offset  8
    var depth: Float = 0        // offset 12   == meanCamZ, kept explicit
    var mean2DX: Float = 0      // offset 16   pixel coordinates
    var mean2DY: Float = 0      // offset 20
    /// TWO `half`s on the GPU side: the per-axis half-extents of the 3-sigma
    /// ellipse. Declared here as one 32-bit field because nothing in Swift
    /// reads this struct's contents, only its stride, and `Float16` does not
    /// exist on x86_64 so a faithful mirror would not build in an Intel
    /// simulator. See TrainerSplatDraw in TrainerShaders.metal.
    var radiusPxPackedHalf2: UInt32 = 0   // offset 24
    /// Combined Mip-Splatting 2D and 3D opacity compensation, already folded
    /// into `opacity`. Kept so the backward can divide it out to reach
    /// `d alpha / d opacityLogit` without recomputing both determinants.
    var comp: Float = 0         // offset 28
    var conicA: Float = 0       // offset 32   inverse 2D covariance, (a, b, c)
    var conicB: Float = 0       // offset 36
    var conicC: Float = 0       // offset 40
    /// `sigmoid(opacityLogit) * comp`: the alpha multiplier the rasteriser
    /// uses directly.
    var opacity: Float = 0      // offset 44
    var colorR: Float = 0       // offset 48   evaluated SH radiance, linear
    var colorG: Float = 0       // offset 52
    var colorB: Float = 0       // offset 56
    /// Bit i set when channel i was clamped at 0 by the `+0.5` SH offset, so
    /// the backward can zero that channel's gradient instead of pushing on a
    /// value the forward never used.
    var clampedMask: UInt32 = 0 // offset 60
    // stride 64
}

/// The top-K observed sampling rates (pixels per world metre) for one
/// Gaussian, sorted descending. `r3` - the K-th largest - is the "high
/// percentile of observed sampling rate" the Mip-Splatting 3D filter is sized
/// from, rather than the maximum, which a single close-up frame would
/// otherwise dominate.
struct TrainerSamplingTopK {
    var r0: Float = 0   // offset  0
    var r1: Float = 0   // offset  4
    var r2: Float = 0   // offset  8
    var r3: Float = 0   // offset 12
    // stride 16
}

// MARK: - Supervision

/// One native LiDAR depth sample, already resolved against everything the
/// SMART layer knows, ready for the GPU depth-loss kernel.
///
/// Everything expensive and everything that needs a protocol call
/// (`TrustField.weight`, `BackgroundModel.authority`, `EdgeClassifier.map`)
/// happens on the CPU, once per keyframe, and lands here. The GPU kernel is
/// then a flat loop over an array with no branching on module state, which is
/// both faster and far easier to get right.
struct TrainerDepthSample {
    /// `y * renderWidth + x` in the training render grid.
    var pixelIndex: UInt32 = 0    // offset  0
    /// Sensor depth in metres along camera +Z, after the per-frame learned
    /// affine correction has been applied.
    var depth: Float = 0          // offset  4
    /// Product of the trust weight, the per-pixel authority and the frame's QC
    /// weight, 0...1. Zero means "no depth supervision here", which is what an
    /// `EdgeClass.band` or `EdgeClass.unknown` sample gets.
    var weight: Float = 0         // offset  8
    /// F4 bimodal edge supervision: the nearer of the two local depth modes.
    var mode0: Float = 0          // offset 12
    /// The farther of the two local depth modes. Equal to `mode0` away from a
    /// geometric edge, which makes the bimodal term vanish there.
    var mode1: Float = 0          // offset 16
    /// F2 free-space lower bound: rendered depth must be at least this, in
    /// metres, because a beam demonstrably passed through everything nearer.
    /// 0 disables the hinge for this sample.
    var freeSpaceBound: Float = 0 // offset 20
    /// `EdgeClass.rawValue`.
    var edgeClass: UInt32 = 0     // offset 24
    /// Huber transition for this sample, metres. Grows with range, because a
    /// 2 cm error at 5 m is not the same event as a 2 cm error at 0.4 m.
    var huberDelta: Float = 0.02  // offset 28
    // stride 32
}

// MARK: - Uniforms

/// Everything the forward and backward kernels need to know about the camera
/// and the current schedule. `constant` buffer, standard alignment.
struct TrainerCameraUniforms {
    /// World -> camera, with the learned per-camera SE(3) delta already
    /// applied on the left. The shader never composes poses.
    var viewMatrix: simd_float4x4 = matrix_identity_float4x4  // offset   0, 64 bytes
    var fx: Float = 0                    // offset  64
    var fy: Float = 0                    // offset  68
    var cx: Float = 0                    // offset  72
    var cy: Float = 0                    // offset  76
    var imageWidth: UInt32 = 0           // offset  80
    var imageHeight: UInt32 = 0          // offset  84
    var tileCountX: UInt32 = 0           // offset  88
    var tileCountY: UInt32 = 0           // offset  92
    var nearPlane: Float = 0.05          // offset  96
    var farPlane: Float = 100            // offset 100
    var splatCount: UInt32 = 0           // offset 104
    /// `1 + shDegree.restCoefficientCount`: how many SH coefficients each
    /// Gaussian actually stores.
    var shCoeffCount: UInt32 = 1         // offset 108
    var cameraCenterX: Float = 0         // offset 112
    var cameraCenterY: Float = 0         // offset 116
    var cameraCenterZ: Float = 0         // offset 120
    /// Mip-Splatting 2D filter variance in px^2. 0.25 is sigma = 0.5 px.
    /// This REPLACES the 0.3 px dilation of the reference rasteriser: the
    /// dilation inflates every Gaussian without compensating its opacity and
    /// is the direct cause of the erosion and dilation artefacts Mip-Splatting
    /// exists to remove.
    var filter2DVariance: Float = 0.25   // offset 124
    /// Alpha below which a Gaussian contributes nothing and is skipped.
    var minAlpha: Float = 1.0 / 255.0    // offset 128
    /// Coarse-to-fine: how many SH coefficients are currently ENABLED. Starts
    /// at 1 (DC only) and grows; coefficients beyond this are read as zero
    /// and receive no gradient.
    var activeSHCoeffCount: UInt32 = 1   // offset 132
    /// Coarse-to-fine: extra isotropic screen-space variance in px^2 added to
    /// the 2D covariance early in training, decaying to 0. Low frequencies
    /// first, detail later.
    var frequencyBlurVariance: Float = 0 // offset 136
    /// Non-zero when the depth channel is needed. Saves the rasteriser a
    /// multiply-add per contributing Gaussian per pixel when it is not.
    var renderDepth: UInt32 = 1          // offset 140
    // stride 144, align 16
}

/// Photometric and geometric loss weights for one step.
struct TrainerLossUniforms {
    var pixelCount: UInt32 = 0        // offset  0
    var width: UInt32 = 0             // offset  4
    var height: UInt32 = 0            // offset  8
    /// D-SSIM weight. 0.2 is the 3DGS reference value.
    var lambdaSSIM: Float = 0.2       // offset 12
    /// This frame's `FrameQC.weight`, 0...1.
    var frameWeight: Float = 1        // offset 16
    /// Learned per-frame exposure gain and bias (F5, F8).
    var exposureGain: Float = 1       // offset 20
    var exposureBias: Float = 0       // offset 24
    /// Depth-loss multiplier from `TrustField.depthLossScale(iteration:of:)`.
    var depthScale: Float = 1         // offset 28
    var depthSampleCount: UInt32 = 0  // offset 32
    /// F4 bimodal edge weight.
    var bimodalWeight: Float = 1      // offset 36
    /// F4 transition-width penalty weight.
    var transitionWidthWeight: Float = 0.35  // offset 40
    /// F2 free-space hinge weight.
    var freeSpaceWeight: Float = 0.5  // offset 44
    /// Weight on "where LiDAR says there is a surface, accumulated alpha
    /// should be 1".
    var alphaSupervisionWeight: Float = 0.05 // offset 48
    /// 1 when a background image is bound, 0 when the far field is black.
    var hasBackground: UInt32 = 0     // offset 52
    /// SSIM stabilisers, squared, on the 0...1 intensity scale.
    var ssimC1: Float = 0.0001        // offset 56   (0.01^2)
    var ssimC2: Float = 0.0009        // offset 60   (0.03^2)
    /// How many of the `depthSampleCount` samples dispatched this frame
    /// actually carry weight, from
    /// `TrainerFrameSupervision.supervisedSampleCount`.
    ///
    /// This is the divisor `trainer_loss_depth` turns its five geometry terms
    /// into per-sample MEANS with. It is measured, not assumed: the count is
    /// taken over the exact prefix of samples that was uploaded, so a run that
    /// truncated the sample array to the buffer capacity cannot divide by more
    /// samples than the GPU was given.
    ///
    /// 0 means "nothing in this frame was supervised", and the kernel falls
    /// back to `depthSampleCount` for it rather than dividing by one. That is
    /// not a nicety. The free-space hinge does not carry `weight`, so a frame
    /// whose photo QC weight is zero has no supervised samples and thousands
    /// of live hinge terms, and a divisor of one there would put an
    /// unnormalised population SUM straight back into the loss.
    var depthSupervisedCount: UInt32 = 0  // offset 64
    // stride 68
}

/// Adam hyper-parameters and the per-group learning rates for one step.
struct TrainerAdamUniforms {
    var count: UInt32 = 0             // offset  0   splats, or SH floats
    var beta1: Float = 0.9            // offset  4
    var beta2: Float = 0.999          // offset  8
    var epsilon: Float = 1e-15        // offset 12
    var lrMean: Float = 0             // offset 16
    var lrScale: Float = 0            // offset 20
    var lrRotation: Float = 0         // offset 24
    var lrOpacity: Float = 0          // offset 28
    var lrSHDC: Float = 0             // offset 32
    var lrSHRest: Float = 0           // offset 36
    /// 1 to skip Gaussians with `visibleFlag == 0`.
    var sparse: UInt32 = 1            // offset 40
    var shCoeffCount: UInt32 = 1      // offset 44
    /// Position learning-rate multiplier for a `positionPinned` Gaussian.
    /// Scaled down, never zeroed: a trusted prior can still be wrong.
    var pinnedPositionLRScale: Float = 0.1  // offset 48
    var minLogScale: Float = -11.5    // offset 52   ~10 micrometres
    var maxLogScale: Float = 0.7      // offset 56   ~2 metres
    var maxOpacityLogit: Float = 12   // offset 60
    // stride 64
}

/// Regulariser weights: the disc / effective-rank prior and late opacity
/// binarization.
struct TrainerRegUniforms {
    var count: UInt32 = 0                 // offset  0
    var discWeight: Float = 0.01          // offset  4
    var discTargetRank: Float = 2         // offset  8
    var edgeTargetRank: Float = 1         // offset 12
    /// Weight on `sigmoid(o) * (1 - sigmoid(o))`, which is maximal at 0.5 and
    /// zero at both ends: it pushes opacity to a decision. Zero until the last
    /// ~20% of the run.
    var binarizeWeight: Float = 0         // offset 16
    /// Fraction of a Gaussian's observed contribution that has to be UNKNOWN
    /// before binarization is switched off for it entirely.
    var binarizeUnknownCutoff: Float = 0.5  // offset 20
    /// Hinge above this world-space scale, metres, to stop a single Gaussian
    /// swallowing a room.
    var maxScaleMeters: Float = 0.5       // offset 24
    var maxScaleWeight: Float = 0.05      // offset 28
    // stride 32
}

/// Generic exclusive-scan arguments.
struct TrainerScanUniforms {
    var count: UInt32 = 0        // offset  0
    var blockCount: UInt32 = 0   // offset  4
    var pad0: UInt32 = 0         // offset  8
    var pad1: UInt32 = 0         // offset 12
    // stride 16
}

/// Radix-sort pass arguments.
struct TrainerRadixUniforms {
    var count: UInt32 = 0        // offset  0
    var blockCount: UInt32 = 0   // offset  4
    /// Bit position of the digit this pass sorts on.
    var bitShift: UInt32 = 0     // offset  8
    var pad0: UInt32 = 0         // offset 12
    // stride 16
}

/// Separable-blur arguments, shared by the SSIM forward and backward passes.
struct TrainerBlurUniforms {
    var width: UInt32 = 0        // offset  0
    var height: UInt32 = 0       // offset  4
    var planeCount: UInt32 = 1   // offset  8
    var pad0: UInt32 = 0         // offset 12
    // stride 16
}

// MARK: - Buffer binding indices
//
// One enum per kernel family rather than one global table, because a global
// table drifts the moment two kernels want different argument orders and
// nobody notices until a buffer reads as zeros.
//
// READ THIS BEFORE BINDING ANYTHING WITH THESE NUMBERS.
//
// This is a LOGICAL registry: it gives every distinct buffer the trainer owns
// one stable id, so code and reviews can say "the offsets buffer" without
// ambiguity. It is NOT the physical argument table, because MSL numbers each
// kernel's arguments from 0 within that kernel. The numbering below matches
// `trainer_preprocess` exactly (splats 0, sh 1, stats 2, draws 3,
// tilesTouched 4, cameraUniforms 5), which is where it came from, and matches
// nothing else: `trainer_rasterize_forward` declares `values [[buffer(0)]]`
// while `values` is 9 here, and binding it at 9 would leave that kernel's
// argument 0 unbound.
//
// The physical table, transcribed argument by argument from the
// `[[buffer(n)]]` attributes in TrainerShaders.metal, is `TrainerBind` in
// TrainerSupport.swift, and it is the only thing the encoders use.

enum TrainerBufferIndex {
    // trainer_preprocess
    static let splats: Int = 0
    static let sh: Int = 1
    static let stats: Int = 2
    static let draws: Int = 3
    static let tilesTouched: Int = 4
    static let cameraUniforms: Int = 5
    static let samplingTopK: Int = 6

    // binning
    static let offsets: Int = 7
    static let keys: Int = 8
    static let values: Int = 9
    static let tileRanges: Int = 10
    static let instanceCount: Int = 11

    // raster
    static let renderColor: Int = 12
    static let renderAlpha: Int = 13
    static let renderDepth: Int = 14
    static let renderTFinal: Int = 15
    static let renderNContrib: Int = 16

    // loss
    static let gtColor: Int = 17
    static let bgColor: Int = 18
    static let gradFinalColor: Int = 19
    static let gradSplatColor: Int = 20
    static let gradDepthRend: Int = 21
    static let gradTFinal: Int = 22
    static let lossUniforms: Int = 23
    static let lossAccum: Int = 24
    static let depthSamples: Int = 25
    static let unknownMask: Int = 26
    static let composited: Int = 27

    // ssim
    //
    // Three ids, three buffers, all three allocated and all three bound. The
    // middle one is `ssimMid` in `TrainerResources` and `ssimDst` here, which
    // is the same buffer under two names and the only place in this registry
    // where the names differ. `blurUniforms` is the `TrainerBlurUniforms` the
    // blur and prepare kernels take by `setBytes`, not a buffer anyone forgot
    // to allocate.
    static let ssimSrc: Int = 28
    static let ssimDst: Int = 29
    static let ssimTmp: Int = 30
    static let blurUniforms: Int = 31

    // backward
    static let splatGrad: Int = 32
    static let shGrad: Int = 33
    static let cameraGrad: Int = 34
    static let exposureGrad: Int = 35

    // optimiser
    static let adamM: Int = 36
    static let adamV: Int = 37
    static let adamUniforms: Int = 38
    static let shAdamM: Int = 39
    static let shAdamV: Int = 40
    static let regUniforms: Int = 41

    // scan / sort
    static let scanIn: Int = 42
    static let scanOut: Int = 43
    static let scanBlockSums: Int = 44
    static let scanUniforms: Int = 45
    static let radixKeysIn: Int = 46
    static let radixValuesIn: Int = 47
    static let radixKeysOut: Int = 48
    static let radixValuesOut: Int = 49
    static let radixHistogram: Int = 50
    static let radixUniforms: Int = 51

    // utility
    static let fillTarget: Int = 52
    static let fillUniforms: Int = 53
}

// MARK: - Layout verification

enum TrainerGPULayouts {

    /// Everything this file and `TrainerShaders.metal` have to agree about,
    /// checked in one place: the `MemoryLayout.stride` and alignment of every
    /// struct the GPU reads, the eleven SSIM blur taps, and the ceiling the
    /// 16-bit tile id puts on the resolution ladder.
    ///
    /// Called once from `MetalSplatTrainer.prepare()`, before a single buffer
    /// is allocated. In DEBUG a mismatch traps with the offending struct
    /// named; in RELEASE it is returned as a string so the trainer refuses to
    /// start and says exactly why, which is the only honest option when the
    /// alternative is a silently misaligned buffer.
    static func verify() -> String? {
        var problems: [String] = []

        func check(_ name: String, _ actual: Int, _ expected: Int) {
            if actual != expected {
                problems.append("\(name) stride is \(actual), Metal expects \(expected)")
            }
        }

        check("TrainerSplat", MemoryLayout<TrainerSplat>.stride, 48)
        check("TrainerSplatGrad", MemoryLayout<TrainerSplatGrad>.stride, 48)
        check("TrainerSplatStats", MemoryLayout<TrainerSplatStats>.stride, 32)
        check("TrainerSplatDraw", MemoryLayout<TrainerSplatDraw>.stride, 64)
        check("TrainerSamplingTopK", MemoryLayout<TrainerSamplingTopK>.stride, 16)
        check("TrainerDepthSample", MemoryLayout<TrainerDepthSample>.stride, 32)
        check("TrainerCameraUniforms", MemoryLayout<TrainerCameraUniforms>.stride, 144)
        check("TrainerLossUniforms", MemoryLayout<TrainerLossUniforms>.stride, 68)
        check("TrainerAdamUniforms", MemoryLayout<TrainerAdamUniforms>.stride, 64)
        check("TrainerRegUniforms", MemoryLayout<TrainerRegUniforms>.stride, 32)
        check("TrainerScanUniforms", MemoryLayout<TrainerScanUniforms>.stride, 16)
        check("TrainerRadixUniforms", MemoryLayout<TrainerRadixUniforms>.stride, 16)
        check("TrainerBlurUniforms", MemoryLayout<TrainerBlurUniforms>.stride, 16)

        // Alignment matters as much as size: a struct Swift decides to align
        // to 16 while Metal aligns it to 4 will still have the right stride
        // and the wrong element addresses inside an array of it.
        func checkAlign(_ name: String, _ actual: Int, _ expected: Int) {
            if actual != expected {
                problems.append("\(name) alignment is \(actual), Metal expects \(expected)")
            }
        }
        checkAlign("TrainerSplat", MemoryLayout<TrainerSplat>.alignment, 4)
        checkAlign("TrainerSplatGrad", MemoryLayout<TrainerSplatGrad>.alignment, 4)
        checkAlign("TrainerSplatStats", MemoryLayout<TrainerSplatStats>.alignment, 4)
        checkAlign("TrainerSplatDraw", MemoryLayout<TrainerSplatDraw>.alignment, 4)
        checkAlign("TrainerDepthSample", MemoryLayout<TrainerDepthSample>.alignment, 4)
        checkAlign("TrainerCameraUniforms", MemoryLayout<TrainerCameraUniforms>.alignment, 16)

        // The SH buffer is a flat [Float], three floats per coefficient. This
        // is the guard that stops anybody "tidying" it into [SIMD3<Float>],
        // which is 16 bytes in Swift and 12 in Metal.
        check("Float (SH element)", MemoryLayout<Float>.stride, 4)

        // --- The SSIM window, which is written down in two places -----------
        //
        // Strides are not the only thing that drifts between this file and
        // TrainerShaders.metal. The blur taps are a number the GPU carries as
        // eleven literals and Swift carries as a sigma, and until this check
        // existed nothing compared them: the shipped table summed to 0.99752
        // rather than 1, so SSIM ran on means that were half a percent short.
        let derivedWindow = TrainerGPUConstants.ssimGaussianWindow()
        let shippedWindow = TrainerGPUConstants.ssimBlurWeights
        if derivedWindow.count != shippedWindow.count {
            problems.append(
                "SSIM window is \(shippedWindow.count) taps, radius "
                    + "\(TrainerGPUConstants.ssimWindowRadius) wants "
                    + "\(derivedWindow.count)"
            )
        } else {
            var worstTapError: Float = 0
            var shippedTotal: Float = 0
            var i: Int = 0
            while i < shippedWindow.count {
                let difference = Swift.abs(shippedWindow[i] - derivedWindow[i])
                if difference > worstTapError { worstTapError = difference }
                shippedTotal += shippedWindow[i]
                i += 1
            }
            if worstTapError > 1e-6 {
                let formatted = String(format: "%.8f", Double(worstTapError))
                problems.append(
                    "SSIM blur taps are not the sigma "
                        + "\(TrainerGPUConstants.ssimSigma) window: worst tap is off "
                        + "by \(formatted)"
                )
            }
            let totalError = Swift.abs(shippedTotal - 1)
            if totalError > 1e-5 {
                let formatted = String(format: "%.8f", Double(shippedTotal))
                problems.append("SSIM blur taps sum to \(formatted), not 1")
            }
        }

        // --- The tile id is 16 bits wide ------------------------------------
        //
        // `maxTileCount` is the ceiling that fact imposes, and it was written
        // down and never enforced. Above it the sort key `(tileID << 16) |
        // depth` wraps and tiles trade fragments with each other, which does
        // not crash and does not warn. The trainer's own resolution ladder is
        // the only thing that sets the render size, so the honest place to
        // check is the top of that ladder, once, before anything allocates.
        let tallestRung: Int = TrainerBudgetGovernor.resolutionLadder.max() ?? 0
        let squarestRender = TrainerRenderSize(width: tallestRung, height: tallestRung)
        let rungTiles: Int = squarestRender.tileCount
        if tallestRung > 0, rungTiles > TrainerGPUConstants.maxTileCount {
            problems.append(
                "the resolution ladder tops out at \(tallestRung) px, which is "
                    + "\(rungTiles) tiles, above the "
                    + "\(TrainerGPUConstants.maxTileCount) a 16-bit tile id can address"
            )
        }

        guard !problems.isEmpty else { return nil }
        let message = "Trainer GPU contract mismatch: "
            + problems.joined(separator: "; ")
        assertionFailure(message)
        return message
    }
}
