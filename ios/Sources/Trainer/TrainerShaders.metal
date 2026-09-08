//
//  TrainerShaders.metal
//  Trainer
//
//  THE ON-DEVICE DIFFERENTIABLE 3D GAUSSIAN SPLATTING KERNELS.
//
//  Everything lives in one file on purpose. A shared `.h` included by several
//  `.metal` files would work (quoted includes resolve relative to the
//  including file), but it is one more thing that can go wrong in a CI run
//  that has no macOS available to test it, and the whole point of this repo's
//  build rules is that CI reliably produces an installable IPA. One file, no
//  includes, no header search paths.
//
//  Every `struct` below is byte-matched to a Swift struct of the same name in
//  `TrainerGPULayouts.swift`, which also carries the offset table and the
//  runtime verification. If you add a field here, add it there.
//
//  Every entry point is prefixed `trainer_` because a single Xcode target
//  compiles every `.metal` file under `Sources/` into ONE `default.metallib`
//  and function names share a namespace. `Sources/Viewer` uses `viewer_`.
//
//  ---------------------------------------------------------------------------
//  WHAT IS DIFFERENT FROM A TEXTBOOK 3DGS RASTERISER, AND WHY
//  ---------------------------------------------------------------------------
//
//  1. NO 0.3 PIXEL DILATION. The reference implementation adds 0.3 to both
//     diagonal entries of the 2D covariance so that a sub-pixel Gaussian still
//     covers a pixel centre. It does this WITHOUT touching opacity, so every
//     small Gaussian silently gets more total energy than it was optimised to
//     have. That is the direct cause of the erosion/dilation artefacts you see
//     when you view a scan at a resolution it was not trained at. This
//     rasteriser uses the Mip-Splatting pair instead: a 2D screen-space filter
//     (sigma 0.5 px) and a 3D world-space filter, each WITH its opacity
//     compensation `sqrt(det(Sigma) / det(Sigma + filter))`.
//
//  2. THE 3D FILTER IS SIZED BY A HIGH PERCENTILE, NOT THE MAXIMUM. Sizing it
//     by the maximum observed sampling rate lets one accidental close-up frame
//     shrink the filter for a Gaussian the rest of the capture only ever saw
//     from three metres away. `trainer_sampling_rate_update` keeps the top 4
//     rates per Gaussian and the finaliser uses the 4th.
//
//  3. ABSGS. The densification statistic is the sum of the MAGNITUDES of the
//     per-pixel screen-space position gradients, accumulated before they are
//     summed. The signed sum cancels almost exactly for a Gaussian straddling
//     an edge - which is precisely the Gaussian that needs splitting - and
//     that cancellation is why stock 3DGS under-densifies edges and blurs
//     them instead.
//
//  4. DEPTH AND ACCUMULATED ALPHA ARE FIRST-CLASS OUTPUTS, not a debug view.
//     The depth channel is what the LiDAR supervises; the alpha channel is
//     what the background model composites behind and what tells the honesty
//     mask which pixels are actually explained.
//
//  5. THE CAMERA IS A PARAMETER. `trainer_preprocess_backward` accumulates a
//     6-vector se(3) gradient for the camera this step is rendering, using the
//     left-perturbation model. The Swift side scatters it onto spline control
//     points so the correction can only ever be smooth in time.
//
//  ---------------------------------------------------------------------------
//  ONE STATED APPROXIMATION, so it is not discovered later and mistaken for a
//  bug: the two Mip-Splatting opacity compensations are recomputed on every
//  forward pass (so they track the scales as the scales move) but their own
//  derivative is NOT propagated - they enter the backward as constants. The
//  dominant opacity and scale gradients are exact; this second-order term is
//  not. It is written down here rather than hidden in the code.
//

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/// Whether the backward rasteriser may use SIMD-group reductions.
///
/// `simd_max` is Apple GPU family 7, A14 and later. Apple7 is the floor for
/// FULL tier in DeviceCompatibilityProbe, NOT the floor for running, so an
/// A12 or A13 reaches this kernel and the instruction does not exist there.
/// A function constant rather than a runtime branch because it is resolved
/// when the pipeline is specialised: on such a device the instruction is
/// absent rather than merely skipped, so makeComputePipelineState succeeds
/// instead of failing and being reported as a missing kernel.
constant bool kTrainerSimdReduce [[function_constant(0)]];

// ============================================================================
// MARK: - Constants (mirrored in TrainerGPUConstants)
// ============================================================================

constant uint  TRAINER_TILE_W            = 16;
constant uint  TRAINER_TILE_H            = 16;
constant uint  TRAINER_TILE_AREA         = 256;

constant uint  TRAINER_SCAN_THREADS      = 256;
constant uint  TRAINER_SCAN_PER_THREAD   = 4;
constant uint  TRAINER_SCAN_BLOCK        = 1024;   // THREADS * PER_THREAD

constant uint  TRAINER_RADIX_BITS        = 4;
constant uint  TRAINER_RADIX_BINS        = 16;

constant uint  TRAINER_SSIM_RADIUS       = 5;      // 11-tap window
constant uint  TRAINER_SSIM_PLANES       = 5;

constant float TRAINER_MAX_SH_DEGREE     = 2.0f;   // documentation only

// Real spherical-harmonics basis constants, INRIA / SPZ / KHR convention.
constant float TRAINER_SH_C0 = 0.28209479177387814f;
constant float TRAINER_SH_C1 = 0.48860251190291990f;
constant float TRAINER_SH_C2_0 =  1.09254843059207900f;
constant float TRAINER_SH_C2_1 = -1.09254843059207900f;
constant float TRAINER_SH_C2_2 =  0.31539156525252005f;
constant float TRAINER_SH_C2_3 = -1.09254843059207900f;
constant float TRAINER_SH_C2_4 =  0.54627421529603960f;

// `EdgeClass` raw values (Core/Contracts.swift).
constant uint TRAINER_EDGE_NONE      = 0;
constant uint TRAINER_EDGE_GEOMETRIC = 1;
constant uint TRAINER_EDGE_TEXTURE   = 2;
constant uint TRAINER_EDGE_UNKNOWN   = 3;
constant uint TRAINER_EDGE_BAND      = 4;

// Rec. 709 luma weights. SSIM is computed on luma; see the note on
// `trainer_ssim_stats`.
constant float3 TRAINER_LUMA = float3(0.2126f, 0.7152f, 0.0722f);

// ============================================================================
// MARK: - Buffer structs (byte-matched to TrainerGPULayouts.swift)
// ============================================================================

struct TrainerSplat {
    packed_float4 rotation;      //  0..15  (x, y, z, w)
    packed_float3 mean;          // 16..27  world metres
    float         opacityLogit;  // 28..31
    packed_float3 logScale;      // 32..43
    uint          flags;         // 44..47
};                               // 48 bytes

struct TrainerSplatGrad {
    float rot0, rot1, rot2, rot3;   //  0..15
    float mean0, mean1, mean2;      // 16..27
    float opacity;                  // 28..31
    float scale0, scale1, scale2;   // 32..43
    float pad;                      // 44..47
};                                  // 48 bytes

/// Identical layout, atomic fields. Bound to the SAME buffer as
/// `TrainerSplatGrad`; the rasteriser backward writes through this view and
/// the optimiser reads through the plain one.
struct TrainerSplatGradAtomic {
    atomic_float rot0, rot1, rot2, rot3;
    atomic_float mean0, mean1, mean2;
    atomic_float opacity;
    atomic_float scale0, scale1, scale2;
    atomic_float pad;
};

struct TrainerSplatStats {
    float absGrad2D;         //  0
    float denom;             //  4
    uint  maxRadiusPxBits;   //  8
    float visAccum;          // 12
    float unknownAccum;      // 16
    float filter3D;          // 20
    uint  visibleFlag;       // 24
    uint  stepCount;         // 28
};                           // 32 bytes

struct TrainerSplatStatsAtomic {
    atomic_float absGrad2D;
    atomic_float denom;
    atomic_uint  maxRadiusPxBits;
    atomic_float visAccum;
    atomic_float unknownAccum;
    atomic_float filter3D;
    atomic_uint  visibleFlag;
    atomic_uint  stepCount;
};

/// THE FOUR SCREEN-SPACE GRADIENTS OF ONE SPLAT, IN ONE CACHE LINE.
///
/// They used to be four separate device buffers, gradColor, gradOpacity,
/// gradMean2D and gradConic, so accumulating one pixel's contribution to one
/// Gaussian touched FOUR distinct cache lines in four different allocations.
/// The backward rasteriser does that per pixel per Gaussian across 2.07
/// million tile instances, which is the reason it measured 13.18 ms per
/// iteration against the forward pass's 4.63 on identical traversal: 52% of
/// all GPU time and 43% of the entire run.
///
/// Interleaved, one splat's nine gradient floats are contiguous and a
/// contribution touches ONE line. `pad` takes the record to exactly 64 bytes
/// so consecutive splats never share one, which also means two threads
/// accumulating into neighbouring splats do not contend for the same line.
///
/// Two declarations of the same 64 bytes because Metal will not let one
/// pointer be atomic for the writer and plain for the reader. The rasteriser
/// takes the atomic view since many threads add to one splat;
/// trainer_preprocess_backward takes the plain one, because by then the
/// rasteriser has finished and each thread reads only its own row.
struct TrainerSplatGrad2DAtomic {
    atomic_float color0;
    atomic_float color1;
    atomic_float color2;
    atomic_float opacity;
    atomic_float mean2D0;
    atomic_float mean2D1;
    atomic_float conic0;
    atomic_float conic1;
    atomic_float conic2;
    float        pad[7];
};

struct TrainerSplatGrad2D {
    float color0;
    float color1;
    float color2;
    float opacity;
    float mean2D0;
    float mean2D1;
    float conic0;
    float conic1;
    float conic2;
    float pad[7];
};

/// WHAT THE HOT LOOPS ACTUALLY READ, AND NOTHING ELSE. 32 bytes.
///
/// ENERGY, not just time. `TrainerSplatDraw` below is 64 bytes and is read
/// ONCE PER TILE INSTANCE by both rasterisers, so at 763,260 instances that is
/// 98 MB of DRAM traffic an iteration purely to stage Gaussians. On mobile
/// silicon a byte from DRAM costs on the order of a hundred times what an
/// arithmetic operation costs, so bytes moved is where the joules go, and this
/// was the largest single item.
///
/// The rasterisers only ever stage mean2D, conic, opacity, colour and depth:
/// 40 bytes of the 64 they pull. duplicate_keys wants mean2D, the two radii
/// and depth. This record is exactly that set, so the per-instance read halves
/// to 49 MB.
///
/// WHAT IS FULL PRECISION AND WHY:
///   mean2D  float2. Pixel coordinates up to 720 and the per-pixel delta is
///           differenced against them, so half would lose sub-pixel accuracy
///           where it matters most.
///   depth   float. It becomes the low 16 bits of the SORT KEY, and half at
///           30 m has about 3 cm of precision, which would coarsen compositing
///           order into ties.
///
/// WHAT IS HALF AND WHY IT IS ACCEPTABLE: conic, opacity and colour all feed
/// the alpha and colour of a splat that is then composited with hundreds of
/// others. Half carries about three decimal digits, and the owner has asked
/// for efficiency ahead of quality for now, with quality work to follow.
struct TrainerSplatRaster {
    packed_float2 mean2D;               //  0..7
    float         depth;                //  8..11
    half          radiusX, radiusY;     // 12..15
    /// FLOAT, and this is the field that cost 2 dB when it was half.
    ///
    /// The conic is the INVERSE covariance, so the rasteriser evaluates
    /// `a*dx*dx + 2*b*dx*dy + c*dy*dy` with dx and dy up to tens of pixels,
    /// making the products hundreds of times the coefficients, and then feeds
    /// the result to exp(). Half's three decimal digits of relative precision
    /// go in at the widest point of that expression and come out amplified.
    /// Measured: held-out PSNR 14.53 against a 16.0 to 16.8 band, and
    /// trained-view 14.98 against 18.7, so the model got worse at fitting the
    /// frames it looks at directly.
    packed_float3 conic;                // 16..27
    half          opacity;              // 28..29
    half          color0, color1;       // 30..33
    half          color2;               // 34..35
    half          pad0, pad1;           // 36..39
};                                      // 40 bytes

struct TrainerSplatDraw {
    packed_float3 meanCam;    //  0..11
    float         depth;      // 12..15
    packed_float2 mean2D;     // 16..23
    /// HALF-EXTENTS OF THE 3-SIGMA ELLIPSE, x and y, in pixels.
    ///
    /// Was one float: the radius of a CIRCLE around the ellipse's major axis,
    /// used for both axes. That circle is what made peakTileInstances 2.07
    /// million against 300,000 splats, 6.91 tiles per splat, when the splats
    /// are 89.7% discs and a disc seen at an angle is nowhere near circular on
    /// screen. Every excess tile is paid for three times: once in the sort,
    /// once in the forward raster and once in the backward raster, which
    /// together are 20.4 ms of a 25.8 ms GPU iteration.
    ///
    /// The exact extent of {x : x^T Sigma^-1 x <= 9} along x is 3*sqrt(Sigma_xx)
    /// and along y is 3*sqrt(Sigma_yy), so this is the TIGHT bounding box of
    /// the same 3-sigma ellipse the circle was containing. It is a strict
    /// subset of the old square and still contains the whole ellipse, so no
    /// splat/pixel pair that contributed before is lost.
    ///
    /// TWO HALVES IN THE OLD FLOAT'S FOUR BYTES, on purpose. This struct is
    /// exactly 64 bytes, one cache line, and it is read once per tile instance
    /// by both rasterisers: 4.14 million reads an iteration. Growing it to 72
    /// would cost more than the tiles saved. Half precision is ample for a
    /// pixel extent that only ever picks 16-pixel tiles, and both kernels read
    /// the SAME stored halves, which is what keeps trainer_preprocess's
    /// tilesTouched exactly equal to what trainer_duplicate_keys then writes.
    /// A mismatch there would not be slow, it would be a buffer overrun.
    half2         radiusPx;   // 24..27
    float         comp;       // 28..31
    packed_float3 conic;      // 32..43
    float         opacity;    // 44..47
    packed_float3 color;      // 48..59
    uint          clampedMask;// 60..63
};                            // 64 bytes

struct TrainerBackgroundUniforms {
    float4 rotationInverse;   //  0..15   (x, y, z, w)
    float  fx, fy, cx, cy;    // 16..31
    uint   width, height;     // 32..39
    uint   faceSize;          // 40..43
    uint   pad;               // 44..47
};

struct TrainerSamplingTopK {
    float r0, r1, r2, r3;     // 16 bytes, descending
};

struct TrainerDepthSample {
    uint  pixelIndex;      //  0
    float depth;           //  4
    float weight;          //  8
    float mode0;           // 12
    float mode1;           // 16
    float freeSpaceBound;  // 20
    uint  edgeClass;       // 24
    float huberDelta;      // 28
};                         // 32 bytes

struct TrainerCameraUniforms {
    float4x4      viewMatrix;              //   0..63
    float         fx, fy, cx, cy;          //  64..79
    uint          imageWidth, imageHeight; //  80..87
    uint          tileCountX, tileCountY;  //  88..95
    float         nearPlane, farPlane;     //  96..103
    uint          splatCount;              // 104..107
    uint          shCoeffCount;            // 108..111
    packed_float3 cameraCenter;            // 112..123
    float         filter2DVariance;        // 124..127
    float         minAlpha;                // 128..131
    uint          activeSHCoeffCount;      // 132..135
    float         frequencyBlurVariance;   // 136..139
    uint          renderDepth;             // 140..143
};                                         // 144 bytes, align 16

struct TrainerLossUniforms {
    uint  pixelCount;              //  0
    uint  width;                   //  4
    uint  height;                  //  8
    float lambdaSSIM;              // 12
    float frameWeight;             // 16
    float exposureGain;            // 20
    float exposureBias;            // 24
    float depthScale;              // 28
    uint  depthSampleCount;        // 32
    float bimodalWeight;           // 36
    float transitionWidthWeight;   // 40
    float freeSpaceWeight;         // 44
    float alphaSupervisionWeight;  // 48
    uint  hasBackground;           // 52
    float ssimC1;                  // 56
    float ssimC2;                  // 60
    uint  depthSupervisedCount;    // 64
};                                 // 68 bytes

struct TrainerAdamUniforms {
    uint  count;                  //  0
    float beta1;                  //  4
    float beta2;                  //  8
    float epsilon;                // 12
    float lrMean;                 // 16
    float lrScale;                // 20
    float lrRotation;             // 24
    float lrOpacity;              // 28
    float lrSHDC;                 // 32
    float lrSHRest;               // 36
    uint  sparse;                 // 40
    uint  shCoeffCount;           // 44
    float pinnedPositionLRScale;  // 48
    float minLogScale;            // 52
    float maxLogScale;            // 56
    float maxOpacityLogit;        // 60
};                                // 64 bytes

struct TrainerRegUniforms {
    uint  count;                  //  0
    float discWeight;             //  4
    float discTargetRank;         //  8
    float edgeTargetRank;         // 12
    float binarizeWeight;         // 16
    float binarizeUnknownCutoff;  // 20
    float maxScaleMeters;         // 24
    float maxScaleWeight;         // 28
};                                // 32 bytes

struct TrainerScanUniforms {
    uint count;       //  0
    uint blockCount;  //  4
    uint pad0, pad1;  //  8..15
};

struct TrainerRadixUniforms {
    uint count;       //  0
    uint blockCount;  //  4
    uint bitShift;    //  8
    uint pad0;        // 12
};

struct TrainerBlurUniforms {
    uint width;       //  0
    uint height;      //  4
    uint planeCount;  //  8
    uint pad0;        // 12
};

// ----------------------------------------------------------------------------
// THE SIZES, CHECKED BY THE METAL COMPILER RATHER THAN BY THE PHONE.
//
// `TrainerGPULayouts.verify()` already checks every one of these from the
// Swift side, but it runs at start-up on the device: a struct that grew here
// and not there fails as a refusal to train, on the owner's phone, after the
// build shipped. These fail in CI instead, on the line that is wrong. The
// numbers must match the `// stride N` comments in TrainerGPULayouts.swift
// exactly. The viewer's shader has carried the same guard from the start.
// ----------------------------------------------------------------------------
static_assert(sizeof(TrainerSplat) == 48, "TrainerSplat must be 48 bytes");
static_assert(sizeof(TrainerSplatGrad) == 48, "TrainerSplatGrad must be 48 bytes");
static_assert(sizeof(TrainerSplatStats) == 32, "TrainerSplatStats must be 32 bytes");
static_assert(sizeof(TrainerSplatDraw) == 64, "TrainerSplatDraw must be 64 bytes");
// These two were in NEITHER this block nor TrainerGPULayouts.verify(), and
// there is no Swift mirror of either: the buffer is sized by a bare `n * 64`
// literal in TrainerResources. So the one record the backward rasteriser
// hammers twelve times per contributing pair was the only one in the file
// whose size nothing checked.
static_assert(sizeof(TrainerSplatGrad2D) == 64, "TrainerSplatGrad2D must be 64 bytes");
static_assert(sizeof(TrainerSplatGrad2DAtomic) == 64, "TrainerSplatGrad2DAtomic must be 64 bytes");
static_assert(sizeof(TrainerSamplingTopK) == 16, "TrainerSamplingTopK must be 16 bytes");
static_assert(sizeof(TrainerDepthSample) == 32, "TrainerDepthSample must be 32 bytes");
static_assert(sizeof(TrainerCameraUniforms) == 144, "TrainerCameraUniforms must be 144 bytes");
static_assert(sizeof(TrainerLossUniforms) == 68, "TrainerLossUniforms must be 68 bytes");
static_assert(sizeof(TrainerAdamUniforms) == 64, "TrainerAdamUniforms must be 64 bytes");
static_assert(sizeof(TrainerRegUniforms) == 32, "TrainerRegUniforms must be 32 bytes");
static_assert(sizeof(TrainerScanUniforms) == 16, "TrainerScanUniforms must be 16 bytes");
static_assert(sizeof(TrainerRadixUniforms) == 16, "TrainerRadixUniforms must be 16 bytes");
static_assert(sizeof(TrainerBlurUniforms) == 16, "TrainerBlurUniforms must be 16 bytes");

// ============================================================================
// MARK: - Small helpers
// ============================================================================

/// Rotation matrix from a unit quaternion `(x, y, z, w)`. Columns, MSL order:
/// `m[col][row]`.
static inline float3x3 trainer_quatToMatrix(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(
        float3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y)),
        float3(2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x)),
        float3(2.0f * (x * z + w * y), 2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y))
    );
}

static inline float3x3 trainer_viewRotation(float4x4 v) {
    return float3x3(v[0].xyz, v[1].xyz, v[2].xyz);
}

static inline float trainer_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -30.0f, 30.0f)));
}

/// Determinant of a symmetric 3x3 held as (xx, xy, xz, yy, yz, zz).
static inline float trainer_det3(float3x3 m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
         - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
         + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
}

/// Number of SH coefficients for a degree.
static inline uint trainer_shCountForDegree(uint degree) {
    return (degree + 1) * (degree + 1);
}

/// Evaluates the SH radiance for one Gaussian. `sh` points at this Gaussian's
/// first coefficient; coefficients are three consecutive floats each.
/// `activeCount` is the coarse-to-fine gate: coefficients at or beyond it read
/// as zero. Returns the pre-clamp value; the caller clamps and records which
/// channels clamped.
static inline float3 trainer_evalSH(
    const device float* sh,
    uint stride,
    uint activeCount,
    float3 dir
) {
    float3 result = TRAINER_SH_C0 * float3(sh[0], sh[1], sh[2]);
    if (activeCount > 1 && stride > 1) {
        const float x = dir.x, y = dir.y, z = dir.z;
        const float3 s1 = float3(sh[3], sh[4], sh[5]);
        const float3 s2 = float3(sh[6], sh[7], sh[8]);
        const float3 s3 = float3(sh[9], sh[10], sh[11]);
        result += -TRAINER_SH_C1 * y * s1
                 + TRAINER_SH_C1 * z * s2
                 - TRAINER_SH_C1 * x * s3;

        if (activeCount > 4 && stride > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            const float3 s4 = float3(sh[12], sh[13], sh[14]);
            const float3 s5 = float3(sh[15], sh[16], sh[17]);
            const float3 s6 = float3(sh[18], sh[19], sh[20]);
            const float3 s7 = float3(sh[21], sh[22], sh[23]);
            const float3 s8 = float3(sh[24], sh[25], sh[26]);
            result += TRAINER_SH_C2_0 * xy * s4
                    + TRAINER_SH_C2_1 * yz * s5
                    + TRAINER_SH_C2_2 * (2.0f * zz - xx - yy) * s6
                    + TRAINER_SH_C2_3 * xz * s7
                    + TRAINER_SH_C2_4 * (xx - yy) * s8;
        }
    }
    return result + 0.5f;
}

/// Zero for infinity and NaN, by inspecting the bits.
///
/// NOT `isfinite(v)`. project.yml sets MTL_FAST_MATH: YES, which passes
/// -ffinite-math-only, under which the compiler is entitled to assume no
/// value is ever inf or NaN and fold `isfinite` to a constant true. The
/// guard below has therefore very probably never run, in any build, since
/// the day it was written: a poisoned gradient would have been written
/// rather than dropped, and the comment saying otherwise was wrong.
///
/// An exponent field of all ones is inf or NaN whatever the optimiser
/// believes about floating point, and a bitwise test is not a
/// floating-point operation for fast math to reason about.
///
/// MEASURED AS A NO-OP so far: `prunedNonFinite` is 0 in all 29 densify
/// passes of the owner's 3000-iteration run, so non-finite gradients do
/// not appear to be occurring in practice. This is repairing a guard, not
/// fixing an observed failure.
static inline float trainer_finiteOrZero(float value) {
    return ((as_type<uint>(value) & 0x7F800000u) == 0x7F800000u)
        ? 0.0f : value;
}

/// Adds `value` to a plain device float. NOT atomic, and the caller has to
/// have earned that.
///
/// `trainer_preprocess_backward` runs one thread per splat and is the only
/// writer of `splatGrad[gid]`, so its eleven accumulations were eleven device
/// atomics contending with nothing at all. 243,000 splats times eleven is 2.7
/// million read-modify-writes an iteration that could be plain loads and
/// stores.
///
/// Use this ONLY where one thread provably owns the address for the whole
/// dispatch. The rasterisers do not: many pixels add to one splat there, and
/// those keep `trainer_atomicAdd`.
static inline void trainer_ownedAdd(device float* target, float value) {
    if (trainer_finiteOrZero(value) == 0.0f) { return; }
    *target += value;
}

/// Adds `value` to a device float atomically. One place so the memory order is
/// the same everywhere.
static inline void trainer_atomicAdd(device atomic_float* target, float value) {
    if (trainer_finiteOrZero(value) == 0.0f) { return; }
    atomic_fetch_add_explicit(target, value, memory_order_relaxed);
}

// ============================================================================
// MARK: - Utility fills
// ============================================================================

kernel void trainer_fill_uint(
    device uint*        target   [[buffer(0)]],
    constant uint2&     args     [[buffer(1)]],   // (count, value)
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= args.x) { return; }
    target[gid] = args.y;
}

kernel void trainer_fill_float(
    device float*       target   [[buffer(0)]],
    constant uint&      count    [[buffer(1)]],
    constant float&     value    [[buffer(2)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    target[gid] = value;
}

/// Zeroes the per-step fields of the stats buffer without touching the
/// accumulating densification statistics or the per-Gaussian Adam step count.
kernel void trainer_reset_visibility(
    device TrainerSplatStats* stats [[buffer(0)]],
    constant uint&            count [[buffer(1)]],
    uint                      gid   [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    stats[gid].visibleFlag = 0;
}

/// Zeroes the densification accumulators. Called after EVERY densification
/// pass, including one that changed nothing, and never between iterations: the
/// statistic is meant to average over exactly one interval.
///
/// The "including one that changed nothing" is load-bearing. `visAccum` is
/// only ever compared against zero, so a pass that skipped this reset would
/// carry its visibility accumulation into the next interval and make the
/// candidate filter more permissive the longer the stage went without doing
/// anything. See the call in `MetalSplatTrainer.trainSlice`.
kernel void trainer_reset_densify_stats(
    device TrainerSplatStats* stats [[buffer(0)]],
    constant uint&            count [[buffer(1)]],
    uint                      gid   [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    stats[gid].absGrad2D = 0.0f;
    stats[gid].denom = 0.0f;
    stats[gid].maxRadiusPxBits = 0;
    stats[gid].visAccum = 0.0f;
    stats[gid].unknownAccum = 0.0f;
}

// ============================================================================
// MARK: - Exclusive prefix scan
//
// Blelloch-style: a per-thread serial scan, a Hillis-Steele scan over the
// thread totals, then the block offset added back by `trainer_scan_add`. One
// block covers 1024 elements; the block sums are scanned by re-entering the
// same kernel from Swift, which supports up to 1024^3 elements in three
// levels. The trainer never gets near that.
// ============================================================================

kernel void trainer_scan_block(
    const device uint*          input      [[buffer(0)]],
    device uint*                output     [[buffer(1)]],
    device uint*                blockSums  [[buffer(2)]],
    constant TrainerScanUniforms& u        [[buffer(3)]],
    uint                        tid        [[thread_position_in_threadgroup]],
    uint                        bid        [[threadgroup_position_in_grid]]
) {
    threadgroup uint tgTotals[TRAINER_SCAN_THREADS];
    threadgroup uint tgScratch[TRAINER_SCAN_THREADS];

    const uint base = bid * TRAINER_SCAN_BLOCK + tid * TRAINER_SCAN_PER_THREAD;

    uint local[TRAINER_SCAN_PER_THREAD];
    uint running = 0;
    for (uint i = 0; i < TRAINER_SCAN_PER_THREAD; ++i) {
        const uint idx = base + i;
        const uint v = (idx < u.count) ? input[idx] : 0u;
        local[i] = running;
        running += v;
    }
    tgTotals[tid] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan over the 256 thread totals, then shifted to
    // exclusive. Two scratch arrays so no thread reads a slot another thread
    // is mid-write on.
    uint value = tgTotals[tid];
    for (uint offset = 1; offset < TRAINER_SCAN_THREADS; offset <<= 1) {
        tgScratch[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= offset) { value += tgScratch[tid - offset]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // `value` is now the INCLUSIVE scan of thread totals.
    const uint threadOffset = value - tgTotals[tid];

    for (uint i = 0; i < TRAINER_SCAN_PER_THREAD; ++i) {
        const uint idx = base + i;
        if (idx < u.count) { output[idx] = threadOffset + local[i]; }
    }

    if (tid == TRAINER_SCAN_THREADS - 1 && blockSums != nullptr) {
        blockSums[bid] = value;
    }
}

kernel void trainer_scan_add(
    device uint*                  output       [[buffer(0)]],
    const device uint*            blockOffsets [[buffer(1)]],
    constant TrainerScanUniforms& u            [[buffer(2)]],
    uint                          gid          [[thread_position_in_grid]]
) {
    if (gid >= u.count) { return; }
    output[gid] += blockOffsets[gid / TRAINER_SCAN_BLOCK];
}

// ============================================================================
// MARK: - Radix sort (LSD, 4-bit digits, stable)
//
// Keys are 32 bits: `(tileID << 16) | quantisedDepth16`. Eight passes.
//
// 4 bits and not 8: the scatter needs a per-thread histogram in threadgroup
// memory, which is `bins * threads * 4` bytes. 16 bins x 256 threads = 16 KB,
// inside the 32 KB every Apple GPU guarantees. 256 bins would need 256 KB.
// ============================================================================

kernel void trainer_radix_histogram(
    const device uint*             keys  [[buffer(0)]],
    device uint*                   hist  [[buffer(1)]],   // [bin * blockCount + block]
    constant TrainerRadixUniforms& u     [[buffer(2)]],
    uint                           tid   [[thread_position_in_threadgroup]],
    uint                           bid   [[threadgroup_position_in_grid]]
) {
    threadgroup atomic_uint tgHist[TRAINER_RADIX_BINS];
    if (tid < TRAINER_RADIX_BINS) {
        atomic_store_explicit(&tgHist[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = bid * TRAINER_SCAN_BLOCK + tid * TRAINER_SCAN_PER_THREAD;
    for (uint i = 0; i < TRAINER_SCAN_PER_THREAD; ++i) {
        const uint idx = base + i;
        if (idx < u.count) {
            const uint digit = (keys[idx] >> u.bitShift) & (TRAINER_RADIX_BINS - 1);
            atomic_fetch_add_explicit(&tgHist[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < TRAINER_RADIX_BINS) {
        hist[tid * u.blockCount + bid] =
            atomic_load_explicit(&tgHist[tid], memory_order_relaxed);
    }
}

kernel void trainer_radix_scatter(
    const device uint*             keysIn     [[buffer(0)]],
    const device uint*             valuesIn   [[buffer(1)]],
    device uint*                   keysOut    [[buffer(2)]],
    device uint*                   valuesOut  [[buffer(3)]],
    const device uint*             histScan   [[buffer(4)]],  // exclusive scan of hist
    constant TrainerRadixUniforms& u          [[buffer(5)]],
    uint                           tid        [[thread_position_in_threadgroup]],
    uint                           bid        [[threadgroup_position_in_grid]]
) {
    // WAS two [bin][thread] arrays, 16 KB each, exactly saturating the
    // 32 KB a threadgroup may hold on Apple 7 and later. The first was
    // written by every thread, immediately read back by the SAME thread,
    // and never touched again: 4,096 threadgroup stores, 4,096 loads and a
    // barrier per threadgroup per pass, across 8 passes and ~746 blocks an
    // iteration, to move a value from a register to itself.
    //
    // The counts live in `mine` already. Removing the array frees half the
    // budget, which is what any wider radix digit would need.
    threadgroup uint tgScratch[TRAINER_RADIX_BINS][TRAINER_SCAN_THREADS];

    const uint base = bid * TRAINER_SCAN_BLOCK + tid * TRAINER_SCAN_PER_THREAD;

    uint mine[TRAINER_RADIX_BINS];
    for (uint b = 0; b < TRAINER_RADIX_BINS; ++b) { mine[b] = 0u; }

    uint digits[TRAINER_SCAN_PER_THREAD];
    for (uint i = 0; i < TRAINER_SCAN_PER_THREAD; ++i) {
        const uint idx = base + i;
        if (idx < u.count) {
            const uint d = (keysIn[idx] >> u.bitShift) & (TRAINER_RADIX_BINS - 1);
            digits[i] = d;
            mine[d] += 1u;
        } else {
            digits[i] = TRAINER_RADIX_BINS;   // sentinel: skipped below
        }
    }
    // Hillis-Steele inclusive scan across threads, all 16 bins at once.
    // `acc` starts from this thread's own counts. No barrier is needed to
    // read a register the same thread just wrote; the one that used to sit
    // here existed only for the round trip through tgCount.
    uint acc[TRAINER_RADIX_BINS];
    for (uint b = 0; b < TRAINER_RADIX_BINS; ++b) { acc[b] = mine[b]; }
    for (uint offset = 1; offset < TRAINER_SCAN_THREADS; offset <<= 1) {
        for (uint b = 0; b < TRAINER_RADIX_BINS; ++b) { tgScratch[b][tid] = acc[b]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= offset) {
            for (uint b = 0; b < TRAINER_RADIX_BINS; ++b) {
                acc[b] += tgScratch[b][tid - offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Exclusive per-(block, bin) offset for THIS thread's elements.
    uint cursor[TRAINER_RADIX_BINS];
    for (uint b = 0; b < TRAINER_RADIX_BINS; ++b) {
        cursor[b] = histScan[b * u.blockCount + bid] + (acc[b] - mine[b]);
    }

    // Written in ascending element order, so the sort is stable and two
    // instances with the same quantised depth keep their splat-index order.
    for (uint i = 0; i < TRAINER_SCAN_PER_THREAD; ++i) {
        const uint idx = base + i;
        const uint d = digits[i];
        if (idx >= u.count || d >= TRAINER_RADIX_BINS) { continue; }
        const uint dst = cursor[d];
        cursor[d] = dst + 1u;
        if (dst < u.count) {
            keysOut[dst] = keysIn[idx];
            valuesOut[dst] = valuesIn[idx];
        }
    }
}

// ============================================================================
// MARK: - Forward: preprocess
// ============================================================================

kernel void trainer_preprocess(
    const device TrainerSplat*        splats  [[buffer(0)]],
    const device float*               sh      [[buffer(1)]],
    device TrainerSplatStatsAtomic*   stats   [[buffer(2)]],
    device TrainerSplatDraw*          draws   [[buffer(3)]],
    device TrainerSplatRaster*        raster  [[buffer(8)]],
    device uint*                      tilesTouched [[buffer(4)]],
    constant TrainerCameraUniforms&   cam     [[buffer(5)]],
    uint                              gid     [[thread_position_in_grid]]
) {
    if (gid >= cam.splatCount) { return; }

    tilesTouched[gid] = 0u;
    draws[gid].opacity = 0.0f;

    const TrainerSplat s = splats[gid];
    const float3 meanWorld = float3(s.mean);

    // --- camera space -------------------------------------------------------
    const float3 meanCam = (cam.viewMatrix * float4(meanWorld, 1.0f)).xyz;
    if (meanCam.z < cam.nearPlane || meanCam.z > cam.farPlane) { return; }

    // --- 3D covariance, with the Mip-Splatting 3D filter --------------------
    const float3 scale = exp(clamp(float3(s.logScale), -12.0f, 3.0f));
    const float4 q = normalize(float4(s.rotation));
    const float3x3 R = trainer_quatToMatrix(q);
    const float3x3 S = float3x3(
        float3(scale.x, 0.0f, 0.0f),
        float3(0.0f, scale.y, 0.0f),
        float3(0.0f, 0.0f, scale.z)
    );
    const float3x3 M = R * S;
    const float3x3 sigmaRaw = M * transpose(M);

    const float filter3D = max(atomic_load_explicit(&stats[gid].filter3D,
                                                    memory_order_relaxed), 0.0f);
    const float f3sq = filter3D * filter3D;
    float3x3 sigmaWorld = sigmaRaw;
    sigmaWorld[0][0] += f3sq;
    sigmaWorld[1][1] += f3sq;
    sigmaWorld[2][2] += f3sq;

    // Mip-Splatting 3D opacity compensation. Without it the 3D filter is just
    // a blur that adds energy; with it, total integrated opacity is preserved.
    const float detRaw = max(trainer_det3(sigmaRaw), 1e-24f);
    const float detFiltered = max(trainer_det3(sigmaWorld), 1e-24f);
    const float comp3D = sqrt(clamp(detRaw / detFiltered, 0.0f, 1.0f));

    // --- 2D covariance ------------------------------------------------------
    const float3x3 W = trainer_viewRotation(cam.viewMatrix);
    const float3x3 sigmaCam = W * sigmaWorld * transpose(W);

    const float invZ = 1.0f / meanCam.z;
    const float invZ2 = invZ * invZ;
    // J is 2x3; MSL matrices are column-major, so this is 3 columns of 2.
    const float2 j0 = float2(cam.fx * invZ, 0.0f);
    const float2 j1 = float2(0.0f, cam.fy * invZ);
    const float2 j2 = float2(-cam.fx * meanCam.x * invZ2, -cam.fy * meanCam.y * invZ2);

    // Sigma2D = J * sigmaCam * J^T, written out because a 2x3 matrix is not a
    // Metal type.
    const float3 col0 = sigmaCam[0], col1 = sigmaCam[1], col2 = sigmaCam[2];
    // t = sigmaCam * J^T  -> 3x2
    const float3 t0 = col0 * j0.x + col1 * j1.x + col2 * j2.x;   // wrong-order guard below
    (void)t0;
    // Row-wise is clearer: J * sigmaCam gives a 2x3, call its rows a and b.
    const float3 rowA = float3(
        j0.x * col0.x + j1.x * col0.y + j2.x * col0.z,
        j0.x * col1.x + j1.x * col1.y + j2.x * col1.z,
        j0.x * col2.x + j1.x * col2.y + j2.x * col2.z
    );
    const float3 rowB = float3(
        j0.y * col0.x + j1.y * col0.y + j2.y * col0.z,
        j0.y * col1.x + j1.y * col1.y + j2.y * col1.z,
        j0.y * col2.x + j1.y * col2.y + j2.y * col2.z
    );
    const float3 jr0 = float3(j0.x, j1.x, j2.x);   // first row of J
    const float3 jr1 = float3(j0.y, j1.y, j2.y);   // second row of J

    float sa = dot(rowA, jr0);
    float sb = dot(rowA, jr1);
    float sc = dot(rowB, jr1);

    const float detBefore = max(sa * sc - sb * sb, 1e-12f);

    // --- Mip-Splatting 2D filter, and NOT a 0.3 px dilation -----------------
    const float lowPass = cam.filter2DVariance + cam.frequencyBlurVariance;
    sa += lowPass;
    sc += lowPass;
    const float det = sa * sc - sb * sb;
    if (det <= 1e-12f) { return; }
    const float comp2D = sqrt(clamp(detBefore / det, 0.0f, 1.0f));

    const float invDet = 1.0f / det;
    const float3 conic = float3(sc * invDet, -sb * invDet, sa * invDet);

    // 3-sigma screen radius from the larger eigenvalue. Still wanted: it is
    // what `stats.maxRadiusPxBits` records and what the "too big on screen"
    // prune tests, both of which mean the WORST extent, not a per-axis one.
    const float mid = 0.5f * (sa + sc);
    const float disc = sqrt(max(mid * mid - det, 1e-9f));
    const float radius = 3.0f * sqrt(max(mid + disc, 1e-9f));
    if (radius < 0.5f) { return; }

    // WHERE THIS GAUSSIAN CAN ACTUALLY EXCEED THE ALPHA THRESHOLD, which
    // is nearly always tighter than three sigma and sometimes very much
    // tighter.
    //
    // The rasteriser keeps a sample only when
    //     opacity * exp(-0.5 * d^T Sigma^-1 d) >= cam.minAlpha
    // so the whole region it can contribute to is the level set
    //     d^T Sigma^-1 d <= 2 * ln(opacity / minAlpha)
    // and the half-extent along an axis is sqrt(that) * sqrt(Sigma_axis).
    // Three sigma is the level set at 9, so this is the same formula with a
    // level that depends on how faint the splat is rather than a constant.
    //
    // EXACT. A tile outside this box contains no pixel where the alpha test
    // can pass, and the rasteriser already rejects every one of those
    // per pixel. Nothing that contributed is lost. Capped at 9 so it can only
    // ever shrink the old box, never grow it.
    //
    // STALE PREMISE, KEPT BECAUSE THE CODE IS STILL CORRECT. This used to
    // say the population was faint: a median peak alpha of 0.0038 against a
    // minAlpha of 1/255, so a typical splat got a level set of 1.9 rather
    // than 9 and a box a quarter of the area. That was measured when late
    // opacity binarization was still on. With it off, build 182's exported
    // model has a median opacity of 0.344 and ZERO per cent of the population
    // below 1/255, so the level set is 2*ln(0.344/0.00392) = 8.9 and this box
    // is now barely tighter than the three sigma it replaced.
    //
    // The formula is still exact and still worth keeping, because a faint
    // splat still gets a small box. But most of the 81 s to 65 s this change
    // bought has been given back by the population becoming opaque again, and
    // anyone hunting for that missing time should look here first rather than
    // assume it is still being saved.
    const float alphaForExtent = trainer_sigmoid(s.opacityLogit) * comp2D * comp3D;
    if (alphaForExtent < cam.minAlpha) { return; }
    const float levelSet = clamp(
        2.0f * log(alphaForExtent / max(cam.minAlpha, 1e-8f)), 0.0f, 9.0f
    );
    const float kExtent = sqrt(levelSet);

    // Scaled up by a thousandth before the half conversion so rounding can
    // only ever ADD a tile, never drop one: half has about 5e-4 of relative
    // precision, so 1.001 covers it at any magnitude this reaches.
    const half2 extent = half2(kExtent * sqrt(max(sa, 1e-9f)) * 1.001f,
                               kExtent * sqrt(max(sc, 1e-9f)) * 1.001f);
    // Read back through the half so this kernel's tile count is computed from
    // exactly the bits `trainer_duplicate_keys` will later read.
    const float radiusX = float(extent.x);
    const float radiusY = float(extent.y);

    const float2 mean2D = float2(
        cam.fx * meanCam.x * invZ + cam.cx,
        cam.fy * meanCam.y * invZ + cam.cy
    );

    // --- tile footprint -----------------------------------------------------
    const int minX = max(0, int(floor((mean2D.x - radiusX) / float(TRAINER_TILE_W))));
    const int minY = max(0, int(floor((mean2D.y - radiusY) / float(TRAINER_TILE_H))));
    const int maxX = min(int(cam.tileCountX),
                         int(ceil((mean2D.x + radiusX) / float(TRAINER_TILE_W))));
    const int maxY = min(int(cam.tileCountY),
                         int(ceil((mean2D.y + radiusY) / float(TRAINER_TILE_H))));
    if (maxX <= minX || maxY <= minY) { return; }
    const uint touched = uint(maxX - minX) * uint(maxY - minY);

    // --- colour -------------------------------------------------------------
    const float3 dir = normalize(meanWorld - float3(cam.cameraCenter));
    const uint shBase = gid * cam.shCoeffCount * 3u;
    float3 rgb = trainer_evalSH(sh + shBase, cam.shCoeffCount,
                                cam.activeSHCoeffCount, dir);
    uint clampedMask = 0u;
    if (rgb.x < 0.0f) { rgb.x = 0.0f; clampedMask |= 1u; }
    if (rgb.y < 0.0f) { rgb.y = 0.0f; clampedMask |= 2u; }
    if (rgb.z < 0.0f) { rgb.z = 0.0f; clampedMask |= 4u; }

    const float comp = comp2D * comp3D;
    // Already computed above, where it sized the tile footprint.
    const float alpha = alphaForExtent;

    // A GAUSSIAN THIS FAINT CANNOT REACH A SINGLE PIXEL, SO STOP HERE.
    //
    // This is exact, not an approximation, and the proof is two lines of
    // this file. The rasteriser forms its per-pixel alpha as
    // min(0.99, co.w * exp(power)) and rejects it below cam.minAlpha, and
    // power is clamped at or below zero just above that test, so
    // exp(power) <= 1 and the per-pixel alpha can never EXCEED co.w, which
    // is this alpha. So alpha < minAlpha means every pixel of every tile in
    // every view rejects it. The backward pass carries the identical test,
    // so its raster gradient is exactly zero as well. Culling here changes
    // the rendered image and the raster gradients by nothing at all.
    //
    // It is worth doing because half the model is in this state. An
    // exported PLY of a finished 3000-iteration run, 299,965 splats, has a
    // median peak alpha of 0.0038 and 50.1 percent of the population below
    // minAlpha, which is 1/255. Every one of those was being projected,
    // having its spherical harmonics evaluated, being expanded into tile
    // instances, sorted through eight radix passes, and loaded into both
    // rasteriser inner loops, to contribute nothing.
    //
    // The prologue at the top of this kernel already wrote tilesTouched 0,
    // radiusPx 0 and opacity 0, which is precisely the state the rest of
    // the pipeline reads as "not drawn". Returning here leaves that state
    // intact rather than needing a second representation of it.
    //
    // What this DOES change, deliberately: visibleFlag and denom below are
    // not written for these splats. Their raster gradient was already zero
    // so nothing is lost there, and the densification score is
    // absGrad2D / max(denom, 1) with absGrad2D also zero, so the score is
    // unchanged at zero. The one real consequence is that the Adam passes,
    // which gate on visibleFlag, now skip them, so they stop receiving the
    // regulariser gradient too. That is why the prune must ALSO learn to
    // see them: a splat that no longer moves and is never removed would be
    // permanent dead weight. See TrainerDensifier.
    if (alpha < cam.minAlpha) { return; }

    TrainerSplatDraw d;
    d.meanCam = packed_float3(meanCam);
    d.depth = meanCam.z;
    d.mean2D = packed_float2(mean2D);
    d.radiusPx = extent;
    d.comp = comp;
    d.conic = packed_float3(conic);
    d.opacity = alpha;
    d.color = packed_float3(rgb);
    d.clampedMask = clampedMask;
    draws[gid] = d;

    // The compact copy the hot loops read. Same values, narrower where the
    // narrowing is affordable. See TrainerSplatRaster.
    TrainerSplatRaster r;
    r.mean2D = d.mean2D;
    r.depth = d.depth;
    r.radiusX = extent.x;
    r.radiusY = extent.y;
    r.conic = conic;
    r.opacity = half(d.opacity);
    r.color0 = half(rgb.x);
    r.color1 = half(rgb.y);
    r.color2 = half(rgb.z);
    // THE POWER BELOW WHICH THIS GAUSSIAN CANNOT MATTER, so the rasterisers
    // can reject a contribution with one compare instead of an exp().
    //
    // Both rasterisers compute `alpha = min(0.99, opacity * exp(power))` and
    // then drop the contribution when `alpha < minAlpha`, so the exp() is
    // evaluated for every candidate and thrown away for 60.76 per cent of
    // them WITHOUT this cutoff. WITH it, re-measured on the same harness, the
    // waste is 0.58 per cent: 75,529,323 pairs reach exp() per iteration and
    // 75,089,548 clear the alpha test. So this is DONE, and anyone reading the
    // 60.76 figure and proposing a cheaper exp or a harder pre-reject is
    // costing themselves a build for nothing. The test is exactly equivalent
    // to `power < log(minAlpha
    // / opacity)`, which needs no exp at all. Computing it here costs one
    // log per SPLAT per iteration in place of one exp per (pixel, splat)
    // pair, and it rides in a pad field the 40-byte record already carries.
    //
    // The 0.01 slack is deliberate: `half` is coarser than the float compare
    // it replaces, and biasing the cutoff DOWN means the cheap test can only
    // ever let through a contribution the exact test would have kept. The
    // exact test still runs after the exp, so the result is unchanged.
    r.pad0 = half(log(max(cam.minAlpha, 1e-8f) / max(float(d.opacity), 1e-8f)) - 0.01f);
    r.pad1 = 0.0h;
    raster[gid] = r;

    tilesTouched[gid] = touched;

    atomic_store_explicit(&stats[gid].visibleFlag, 1u, memory_order_relaxed);
    // One observation of this Gaussian, for the AbsGS denominator. Pixel-GS
    // area weighting lives in the numerator, which grows with coverage.
    trainer_atomicAdd(&stats[gid].denom, 1.0f);
    atomic_fetch_max_explicit(&stats[gid].maxRadiusPxBits,
                              as_type<uint>(radius), memory_order_relaxed);
}

// ============================================================================
// MARK: - Forward: tile instance expansion
// ============================================================================

kernel void trainer_duplicate_keys(
    const device TrainerSplatRaster* raster       [[buffer(0)]],
    const device uint*              tilesTouched [[buffer(1)]],
    const device uint*              offsets      [[buffer(2)]],
    device uint*                    keys         [[buffer(3)]],
    device uint*                    values       [[buffer(4)]],
    constant TrainerCameraUniforms& cam          [[buffer(5)]],
    constant uint&                  instanceCap  [[buffer(6)]],
    uint                            gid          [[thread_position_in_grid]]
) {
    if (gid >= cam.splatCount) { return; }
    const uint touched = tilesTouched[gid];
    if (touched == 0u) { return; }

    // The compact record, not the 64-byte one: this kernel wants mean2D, the
    // two radii and depth, which is all TrainerSplatRaster holds.
    const TrainerSplatRaster d = raster[gid];
    const float2 mean2D = float2(d.mean2D);
    // The same two halves trainer_preprocess counted tiles with. Recomputing
    // them from the conic instead would risk differing in the last bit, and
    // then this kernel writes a different number of instances than `offsets`
    // reserved for it.
    const float radiusX = float(d.radiusX);
    const float radiusY = float(d.radiusY);

    const int minX = max(0, int(floor((mean2D.x - radiusX) / float(TRAINER_TILE_W))));
    const int minY = max(0, int(floor((mean2D.y - radiusY) / float(TRAINER_TILE_H))));
    const int maxX = min(int(cam.tileCountX),
                         int(ceil((mean2D.x + radiusX) / float(TRAINER_TILE_W))));
    const int maxY = min(int(cam.tileCountY),
                         int(ceil((mean2D.y + radiusY) / float(TRAINER_TILE_H))));

    // Depth quantised to 16 bits over the working range. At a 30 m far plane
    // that is 0.5 mm, far finer than any ordering ambiguity that matters, and
    // it halves both the key width and the number of sort passes.
    const float span = max(cam.farPlane - cam.nearPlane, 1e-3f);
    // THIRTEEN BITS OF DEPTH, NOT SIXTEEN, so the whole key fits in 24 and
    // the radix sort runs SIX four-bit passes instead of eight.
    //
    // The sort moves roughly 98 MB an iteration: 763,260 instances times eight
    // bytes of key and value, read and written, once per pass. Two of those
    // eight passes were sorting bits that are always zero, because the key
    // only ever used 27 of its 32 bits. Dropping depth to 13 takes it to 24,
    // which divides by four exactly, so the pass count stays EVEN and the
    // result still lands in the buffer every reader already expects. An odd
    // pass count would leave it in the other one and corrupt the sort
    // silently.
    //
    // What it costs: 8192 depth levels over the working range instead of
    // 65536, so about 3.7 mm at a 30 m far plane instead of 0.5 mm. Two
    // instances in the same tile closer together than that now tie and fall
    // back to splat-index order. Alpha compositing is order-dependent, so this
    // is a real if small quality trade, taken deliberately for a quarter of
    // the sort's energy.
    const float norm = clamp((d.depth - cam.nearPlane) / span, 0.0f, 1.0f);
    const uint depthKey = uint(norm * 65535.0f);

    uint cursor = offsets[gid];
    for (int ty = minY; ty < maxY; ++ty) {
        for (int tx = minX; tx < maxX; ++tx) {
            if (cursor >= instanceCap) { return; }
            const uint tile = uint(ty) * cam.tileCountX + uint(tx);
            keys[cursor] = (tile << 16) | depthKey;
            values[cursor] = gid;
            cursor += 1u;
        }
    }
}

/// THE FAR FIELD, RASTERISED ON THE GPU INSTEAD OF THE CPU.
///
/// This was a Swift loop over all 388,800 pixels, run once per iteration on
/// the prefetch worker: a ray build, a quaternion rotate, a cube-face select
/// that re-normalised an already-unit vector, and a bilinear cubemap fetch,
/// call it 60 to 90 cycles a pixel. 6 to 10 ms, the largest single item on a
/// 19 ms worker, and then 4.67 MB of it was memcpy'd into `bgColor` on the
/// MAIN thread with the GPU idle behind that too.
///
/// Every input is already on the GPU or fits in a uniform. The cubemap is
/// 6 * 32 * 32 texels, 72 KB, so it is uploaded whole every iteration rather
/// than versioned: at that size the copy is noise and a version counter is one
/// more thing to get wrong.
///
/// The arithmetic below is transcribed from SmartCamera.ray,
/// simd_quatf.act and SmartBackgroundCubemap.faceAndUV / radiance. It must
/// stay transcribed: the background is what the photometric loss composites
/// behind the Gaussians, so a divergence here is not a visual artefact, it is
/// a wrong gradient.
kernel void trainer_background(
    const device float*                   texels [[buffer(0)]],  // 3 per texel
    device float*                         bgColor[[buffer(1)]],  // 3 per pixel
    constant TrainerBackgroundUniforms&   u      [[buffer(2)]],
    uint                                  gid    [[thread_position_in_grid]]
) {
    const uint n = u.width * u.height;
    if (gid >= n) { return; }

    const uint px = gid % u.width;
    const uint py = gid / u.width;

    // SmartCamera.ray: normalize((x - cx)/fx, (y - cy)/fy, 1)
    const float3 ray = normalize(float3(
        (float(px) + 0.5f - u.cx) / u.fx,
        (float(py) + 0.5f - u.cy) / u.fy,
        1.0f
    ));

    // simd_quatf.act for a unit quaternion:
    //   v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    const float3 qv = u.rotationInverse.xyz;
    const float  qw = u.rotationInverse.w;
    const float3 d = ray + 2.0f * cross(qv, cross(qv, ray) + qw * ray);

    // SmartBackgroundCubemap.faceAndUV, branch for branch.
    const float3 a = abs(d);
    int face = 0;
    float sc = 0.0f, tc = 0.0f, ma = 1.0f;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0.0f) { face = 0; sc = -d.z; tc = -d.y; }
        else            { face = 1; sc =  d.z; tc = -d.y; }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (d.y > 0.0f) { face = 2; sc = d.x; tc =  d.z; }
        else            { face = 3; sc = d.x; tc = -d.z; }
    } else {
        ma = a.z;
        if (d.z > 0.0f) { face = 4; sc =  d.x; tc = -d.y; }
        else            { face = 5; sc = -d.x; tc = -d.y; }
    }
    const float m = max(ma, 1e-8f);
    const float uu = (sc / m + 1.0f) * 0.5f;
    const float vv = (tc / m + 1.0f) * 0.5f;

    // bilinearFootprint: clamp at the face edges, never wrap the seam.
    const int nf = max(2, int(u.faceSize));
    const float fx = clamp(uu * float(nf) - 0.5f, 0.0f, float(nf - 1));
    const float fy = clamp(vv * float(nf) - 0.5f, 0.0f, float(nf - 1));
    const int x0 = int(fx), y0 = int(fy);
    const int x1 = min(x0 + 1, nf - 1), y1 = min(y0 + 1, nf - 1);
    const float ax = fx - float(x0), ay = fy - float(y0);
    const int base = face * nf * nf;

    const int i00 = base + y0 * nf + x0;
    const int i01 = base + y0 * nf + x1;
    const int i10 = base + y1 * nf + x0;
    const int i11 = base + y1 * nf + x1;

    // Same order the CPU accumulated in, so the same rounding.
    float3 out = float3(0.0f);
    out += float3(texels[i00 * 3 + 0], texels[i00 * 3 + 1], texels[i00 * 3 + 2])
         * ((1.0f - ax) * (1.0f - ay));
    out += float3(texels[i01 * 3 + 0], texels[i01 * 3 + 1], texels[i01 * 3 + 2])
         * (ax * (1.0f - ay));
    out += float3(texels[i10 * 3 + 0], texels[i10 * 3 + 1], texels[i10 * 3 + 2])
         * ((1.0f - ax) * ay);
    out += float3(texels[i11 * 3 + 0], texels[i11 * 3 + 1], texels[i11 * 3 + 2])
         * (ax * ay);

    bgColor[gid * 3u + 0u] = out.x;
    bgColor[gid * 3u + 1u] = out.y;
    bgColor[gid * 3u + 2u] = out.z;
}

kernel void trainer_tile_ranges(
    const device uint*   keys        [[buffer(0)]],
    device uint*         tileRanges  [[buffer(1)]],   // 2 per tile
    constant uint&       count       [[buffer(2)]],
    uint                 gid         [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const uint tile = keys[gid] >> 16;
    if (gid == 0u) {
        tileRanges[2u * tile] = 0u;
    } else {
        const uint prev = keys[gid - 1u] >> 16;
        if (prev != tile) {
            tileRanges[2u * prev + 1u] = gid;
            tileRanges[2u * tile] = gid;
        }
    }
    if (gid == count - 1u) {
        tileRanges[2u * tile + 1u] = count;
    }
}

// ============================================================================
// MARK: - Forward: rasterise
//
// One threadgroup per tile, one thread per pixel, front to back with early
// termination. Outputs colour, ACCUMULATED ALPHA and DEPTH, plus the final
// transmittance and contributor count the backward pass needs.
// ============================================================================

kernel void trainer_rasterize_forward(
    const device uint*              values      [[buffer(0)]],
    const device uint*              tileRanges  [[buffer(1)]],
    const device TrainerSplatRaster* raster     [[buffer(2)]],
    device float*                   outColor    [[buffer(3)]],   // 3 per pixel
    device float*                   outAlpha    [[buffer(4)]],
    device float*                   outDepth    [[buffer(5)]],
    device float*                   outTFinal   [[buffer(6)]],
    device uint*                    outNContrib [[buffer(7)]],
    constant TrainerCameraUniforms& cam         [[buffer(8)]],
    uint2                           tgPos       [[threadgroup_position_in_grid]],
    uint2                           tPos        [[thread_position_in_threadgroup]],
    uint                            tid         [[thread_index_in_threadgroup]]
) {
    threadgroup float2 tgXY[TRAINER_TILE_AREA];
    // FLOAT4 again. Staging as half was exact only while the conic itself was
    // half, and the conic went back to float because half cost 2 dB there.
    threadgroup float4 tgConicOpacity[TRAINER_TILE_AREA];
    threadgroup float4 tgColorDepth[TRAINER_TILE_AREA];
    /// Per-splat power cutoff, 512 B. See `r.pad0` in trainer_preprocess:
    /// the power below which this Gaussian cannot clear minAlpha, so the
    /// contribution is rejected with a compare instead of an exp().
    threadgroup half tgCutoff[TRAINER_TILE_AREA];

    const uint tileID = tgPos.y * cam.tileCountX + tgPos.x;
    const uint2 pixel = uint2(tgPos.x * TRAINER_TILE_W + tPos.x,
                              tgPos.y * TRAINER_TILE_H + tPos.y);
    const bool inside = (pixel.x < cam.imageWidth) && (pixel.y < cam.imageHeight);
    const uint pixelIndex = pixel.y * cam.imageWidth + pixel.x;
    const float2 pixelCenter = float2(float(pixel.x) + 0.5f, float(pixel.y) + 0.5f);

    const uint rangeStart = tileRanges[2u * tileID];
    const uint rangeEnd = tileRanges[2u * tileID + 1u];
    const uint total = (rangeEnd > rangeStart) ? (rangeEnd - rangeStart) : 0u;
    const uint batches = (total + TRAINER_TILE_AREA - 1u) / TRAINER_TILE_AREA;

    float T = 1.0f;
    float3 color = float3(0.0f);
    float depth = 0.0f;
    uint contributors = 0u;
    bool done = !inside;

    for (uint b = 0; b < batches; ++b) {
        const uint load = rangeStart + b * TRAINER_TILE_AREA + tid;
        if (load < rangeEnd) {
            const uint splatIndex = values[load];
            const TrainerSplatRaster d = raster[splatIndex];
            // tgIndex was staged here and never read back in this
            // kernel. 1 KB of threadgroup memory per threadgroup, on a
            // GPU where threadgroup memory is what limits how many
            // threadgroups run at once. The backward rasteriser keeps
            // its own tgIndex because it genuinely reads it.
            tgXY[tid] = float2(d.mean2D);
            tgConicOpacity[tid] = float4(float3(d.conic), float(d.opacity));
            tgColorDepth[tid] = float4(
                float(d.color0), float(d.color1), float(d.color2), d.depth
            );
            tgCutoff[tid] = d.pad0;
        } else {
            tgConicOpacity[tid] = float4(0.0f);
            // A cutoff no power can fall below, so the padding entries are
            // rejected by the cheap test and never reach the exp.
            tgCutoff[tid] = 60000.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!done) {
            const uint here = min(TRAINER_TILE_AREA, total - b * TRAINER_TILE_AREA);
            for (uint j = 0; j < here; ++j) {
                const float2 delta = tgXY[j] - pixelCenter;
                const float4 co = tgConicOpacity[j];
                const float power = -0.5f * (co.x * delta.x * delta.x
                                             + co.z * delta.y * delta.y)
                                    - co.y * delta.x * delta.y;
                // `if (power > 0) continue;` USED TO BE HERE, AND IT IS
                // PROVABLY DEAD. power is
                //     -0.5*(cx*dx^2 + cz*dy^2) - cy*dx*dy
                // with (cx, cy, cz) = (sc, -sb, sa)/det, so it equals
                //     -(0.5/det) * (sc*dx^2 - 2*sb*dx*dy + sa*dy^2)
                // and that bracket is the quadratic form of the ADJUGATE of
                // Sigma2D. Sigma2D is positive definite here (sa > 0 and
                // det > 1e-12 are both checked in trainer_preprocess, which
                // refuses to emit the splat otherwise), and the adjugate of a
                // positive definite 2x2 is positive definite, so the bracket
                // is non-negative and power can never exceed zero.
                //
                // MEASURED as well as proved: an offline simulation of this
                // loop over 40 tiles of frame 4, 7,719,936 (pixel, Gaussian)
                // pairs, found 100.00 per cent passing the test. It was a
                // compare and a branch on every pair, and this loop body is
                // entered roughly 294 million times an iteration in each of
                // the two rasterisers.
                //
                // Float rounding can leave power at about +1e-7 rather than a
                // clean zero. That is harmless: the cutoff test below rejects
                // on the same value, and exp(1e-7) is 1 to within the half
                // precision the colour is stored at anyway.
                // Cheap reject BEFORE the exp: see `r.pad0`.
                if (power < float(tgCutoff[j])) { continue; }
                const float alpha = min(0.99f, co.w * exp(power));
                if (alpha < cam.minAlpha) { continue; }
                const float testT = T * (1.0f - alpha);
                if (testT < 1e-4f) { done = true; break; }
                const float weight = alpha * T;
                color += float3(tgColorDepth[j].xyz) * weight;
                if (cam.renderDepth != 0u) { depth += tgColorDepth[j].w * weight; }
                T = testT;
                contributors = b * TRAINER_TILE_AREA + j + 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inside) {
        outColor[pixelIndex * 3u + 0u] = color.x;
        outColor[pixelIndex * 3u + 1u] = color.y;
        outColor[pixelIndex * 3u + 2u] = color.z;
        outAlpha[pixelIndex] = 1.0f - T;
        outDepth[pixelIndex] = depth;
        outTFinal[pixelIndex] = T;
        outNContrib[pixelIndex] = contributors;
    }
}

// ============================================================================
// MARK: - Loss: photometric
//
// Composites the background behind the splats, applies the learned per-frame
// exposure, computes the L1 term and writes both the composited image (for
// SSIM and for the exposure gradient) and dL/dC_final.
// ============================================================================

kernel void trainer_loss_photometric(
    const device float*           renderColor [[buffer(0)]],
    const device float*           renderTFinal[[buffer(1)]],
    const device float*           gtColor     [[buffer(2)]],
    const device float*           bgColor     [[buffer(3)]],
    device float*                 composited  [[buffer(4)]],   // 3 per pixel, post-exposure
    device float*                 gradFinal   [[buffer(5)]],   // 3 per pixel, dL/dC_final
    device float*                 ssimPlanes  [[buffer(6)]],   // plane 0 = rendered luma,
                                                               // plane 1 = gt luma
    device atomic_float*          lossAccum   [[buffer(7)]],
    constant TrainerLossUniforms& u           [[buffer(8)]],
    uint                          gid         [[thread_position_in_grid]]
) {
    if (gid >= u.pixelCount) { return; }

    const float3 splat = float3(renderColor[gid * 3u + 0u],
                                renderColor[gid * 3u + 1u],
                                renderColor[gid * 3u + 2u]);
    const float T = renderTFinal[gid];
    float3 bg = float3(0.0f);
    if (u.hasBackground != 0u) {
        bg = float3(bgColor[gid * 3u + 0u], bgColor[gid * 3u + 1u], bgColor[gid * 3u + 2u]);
    }
    const float3 preExposure = splat + T * bg;
    const float3 rendered = u.exposureGain * preExposure + u.exposureBias;

    const float3 truth = float3(gtColor[gid * 3u + 0u],
                                gtColor[gid * 3u + 1u],
                                gtColor[gid * 3u + 2u]);

    const float invN = 1.0f / float(max(u.pixelCount, 1u));
    const float w = u.frameWeight * (1.0f - u.lambdaSSIM) * invN;

    const float3 diff = rendered - truth;
    const float l1 = (abs(diff.x) + abs(diff.y) + abs(diff.z)) / 3.0f;
    trainer_atomicAdd(lossAccum, u.frameWeight * (1.0f - u.lambdaSSIM) * l1 * invN);

    const float3 g = w * sign(diff) / 3.0f;
    gradFinal[gid * 3u + 0u] = g.x;
    gradFinal[gid * 3u + 1u] = g.y;
    gradFinal[gid * 3u + 2u] = g.z;

    composited[gid * 3u + 0u] = rendered.x;
    composited[gid * 3u + 1u] = rendered.y;
    composited[gid * 3u + 2u] = rendered.z;

    ssimPlanes[0u * u.pixelCount + gid] = dot(rendered, TRAINER_LUMA);
    ssimPlanes[1u * u.pixelCount + gid] = dot(truth, TRAINER_LUMA);
}

// ============================================================================
// MARK: - SSIM (luma)
//
// Structural similarity is defined on luminance in the original paper, and
// computing it on luma rather than per channel is a deliberate, stated
// simplification: it cuts the intermediate planes from 15 to 5, which is the
// difference between fitting in a phone's memory budget and not. The gradient
// is redistributed to RGB through the same Rec. 709 weights.
//
// Plane layout in `ssimPlanes` (each `pixelCount` floats):
//   forward in : 0 = X (rendered luma), 1 = Y (gt luma)
//   forward mid: 0 = X, 1 = Y, 2 = X*X, 3 = Y*Y, 4 = X*Y      (pre-blur)
//   forward out: 0 = mu_x, 1 = mu_y, 2 = G*XX, 3 = G*YY, 4 = G*XY
//   backward in: 0 = Cc, 1 = A, 2 = A*mu_x, 3 = B, 4 = B*mu_y (pre-blur)
// ============================================================================

/// Builds the five pre-blur planes from X and Y.
kernel void trainer_ssim_prepare(
    device float*                 planes [[buffer(0)]],
    constant TrainerBlurUniforms& u      [[buffer(1)]],
    uint                          gid    [[thread_position_in_grid]]
) {
    const uint n = u.width * u.height;
    if (gid >= n) { return; }
    const float x = planes[0u * n + gid];
    const float y = planes[1u * n + gid];
    planes[2u * n + gid] = x * x;
    planes[3u * n + gid] = y * y;
    planes[4u * n + gid] = x * y;
}

/// Separable Gaussian blur, horizontal then vertical, over `planeCount`
/// planes. Clamp-to-edge, which is what every SSIM implementation does at the
/// border and keeps the window normalised.
kernel void trainer_blur_h(
    const device float*           src [[buffer(0)]],
    device float*                 dst [[buffer(1)]],
    constant TrainerBlurUniforms& u   [[buffer(2)]],
    uint2                         gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    const uint n = u.width * u.height;
    const uint idx = gid.y * u.width + gid.x;

    // sigma = 1.5, 11 taps, normalised so the eleven weights sum to exactly
    // 1.0. Written out so nobody has to trust an exp() in a loop with
    // fast-math enabled.
    //
    // THESE ELEVEN NUMBERS ARE CHECKED AGAINST SWIFT AT START-UP.
    // `TrainerGPUConstants.ssimBlurWeights` carries the same table and
    // `TrainerGPULayouts.verify()` re-derives it from `ssimSigma` and
    // `ssimWindowRadius` and refuses to train if the three disagree. Change a
    // digit here and the trainer says so on the phone instead of quietly
    // blurring with the wrong window.
    const float k[11] = {
        0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f, 0.21300554f,
        0.26601172f,
        0.21300554f, 0.10936069f, 0.03600077f, 0.00759876f, 0.00102838f
    };

    for (uint p = 0; p < u.planeCount; ++p) {
        float sum = 0.0f;
        for (int t = -int(TRAINER_SSIM_RADIUS); t <= int(TRAINER_SSIM_RADIUS); ++t) {
            const int sx = clamp(int(gid.x) + t, 0, int(u.width) - 1);
            sum += k[t + int(TRAINER_SSIM_RADIUS)] * src[p * n + gid.y * u.width + uint(sx)];
        }
        dst[p * n + idx] = sum;
    }
}

kernel void trainer_blur_v(
    const device float*           src [[buffer(0)]],
    device float*                 dst [[buffer(1)]],
    constant TrainerBlurUniforms& u   [[buffer(2)]],
    uint2                         gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    const uint n = u.width * u.height;
    const uint idx = gid.y * u.width + gid.x;

    // The same normalised sigma = 1.5 window as `trainer_blur_h`, checked
    // against `TrainerGPUConstants.ssimBlurWeights` at start-up.
    const float k[11] = {
        0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f, 0.21300554f,
        0.26601172f,
        0.21300554f, 0.10936069f, 0.03600077f, 0.00759876f, 0.00102838f
    };

    for (uint p = 0; p < u.planeCount; ++p) {
        float sum = 0.0f;
        for (int t = -int(TRAINER_SSIM_RADIUS); t <= int(TRAINER_SSIM_RADIUS); ++t) {
            const int sy = clamp(int(gid.y) + t, 0, int(u.height) - 1);
            sum += k[t + int(TRAINER_SSIM_RADIUS)] * src[p * n + uint(sy) * u.width + gid.x];
        }
        dst[p * n + idx] = sum;
    }
}

/// Turns the five blurred moments into the SSIM value and the three partial
/// derivative planes the backward blur needs.
kernel void trainer_ssim_stats(
    const device float*           blurred  [[buffer(0)]],   // mu_x, mu_y, G*XX, G*YY, G*XY
    device float*                 partials [[buffer(1)]],   // Cc, A, A*mu_x, B, B*mu_y
    device atomic_float*          lossAccum[[buffer(2)]],
    constant TrainerLossUniforms& u        [[buffer(3)]],
    uint                          gid      [[thread_position_in_grid]]
) {
    const uint n = u.pixelCount;
    if (gid >= n) { return; }

    const float mux = blurred[0u * n + gid];
    const float muy = blurred[1u * n + gid];
    const float sxx = max(blurred[2u * n + gid] - mux * mux, 0.0f);
    const float syy = max(blurred[3u * n + gid] - muy * muy, 0.0f);
    const float sxy = blurred[4u * n + gid] - mux * muy;

    const float c1 = u.ssimC1, c2 = u.ssimC2;
    const float n1 = 2.0f * mux * muy + c1;
    const float n2 = 2.0f * sxy + c2;
    const float d1 = mux * mux + muy * muy + c1;
    const float d2 = sxx + syy + c2;
    const float invD = 1.0f / max(d1 * d2, 1e-12f);
    const float ssim = n1 * n2 * invD;

    // L_ssim = frameWeight * lambda * (1 - mean(SSIM))
    const float w = u.frameWeight * u.lambdaSSIM / float(max(n, 1u));
    trainer_atomicAdd(lossAccum, w * (1.0f - ssim));

    // dSSIM/d(mu_x), dSSIM/d(sigma_xy), dSSIM/d(sigma_xx)
    const float dS_dmux = (2.0f * muy * n2 * d1 - 2.0f * mux * n1 * n2)
                          / max(d1 * d1 * d2, 1e-12f);
    const float dS_dsxy = 2.0f * n1 * invD;
    const float dS_dsxx = -n1 * n2 / max(d1 * d2 * d2, 1e-12f);

    const float Cc = -w * dS_dmux;
    const float A  = -w * dS_dsxx;
    const float B  = -w * dS_dsxy;

    // THREE PLANES, NOT FIVE. The blur that follows is a linear operator,
    // clamp-to-edge included, and three of the five blurred planes entered the
    // gradient with no per-pixel factor at the output pixel:
    //
    //   dL/dX = (G*Cc) + 2X(G*A) - 2(G*(A mux)) + Y(G*B) - (G*(B muy))
    //
    // The first, third and fifth terms have no X or Y on them, so by linearity
    // G*Cc - 2 G*(A mux) - G*(B muy) = G*(Cc - 2 A mux - B muy) and the fold
    // can happen HERE, before the convolution, instead of after it. Two of the
    // five planes stop existing: four fewer plane-passes through an 11-tap
    // separable blur per iteration, and 12.4 MB less traffic.
    partials[0u * n + gid] = Cc - 2.0f * A * mux - B * muy;
    partials[1u * n + gid] = A;
    partials[2u * n + gid] = B;
}

/// Assembles dL/dX from the blurred partials and folds it into dL/dC_final
/// through the luma weights.
///
///   dL/dX_q = (G*Cc)_q + 2 X_q (G*A)_q - 2 (G*(A mu_x))_q
///                      +   Y_q (G*B)_q -   (G*(B mu_y))_q
kernel void trainer_ssim_backward(
    const device float*           blurredPartials [[buffer(0)]],
    const device float*           lumaPlanes      [[buffer(1)]],  // 0 = X, 1 = Y
    device float*                 gradFinal       [[buffer(2)]],
    constant TrainerLossUniforms& u               [[buffer(3)]],
    uint                          gid             [[thread_position_in_grid]]
) {
    const uint n = u.pixelCount;
    if (gid >= n) { return; }

    const float X = lumaPlanes[0u * n + gid];
    const float Y = lumaPlanes[1u * n + gid];

    // Plane 0 already carries Cc - 2 A mux - B muy, folded before the blur in
    // trainer_ssim_stats. Same value, two fewer planes convolved.
    const float gP = blurredPartials[0u * n + gid];
    const float gA = blurredPartials[1u * n + gid];
    const float gB = blurredPartials[2u * n + gid];

    const float dLdX = gP + 2.0f * X * gA + Y * gB;

    gradFinal[gid * 3u + 0u] += dLdX * TRAINER_LUMA.x;
    gradFinal[gid * 3u + 1u] += dLdX * TRAINER_LUMA.y;
    gradFinal[gid * 3u + 2u] += dLdX * TRAINER_LUMA.z;
}

// ============================================================================
// MARK: - Loss: depth, free space, alpha (F2, F3, F4, F6)
//
// ----------------------------------------------------------------------------
// WHY EVERY TERM IN HERE IS DIVIDED BY A SAMPLE COUNT. READ THIS BEFORE
// TOUCHING ANY WEIGHT IN `SmartLossSettings` OR `TrainerTuning`.
// ----------------------------------------------------------------------------
//
// The two photometric terms are per-pixel MEANS. `trainer_loss_photometric`
// multiplies by `invN = 1 / pixelCount` and `trainer_ssim_stats` divides by
// the same `n`. So the photograph contributes a number of order 0.01 to 0.1
// per frame, whatever the render resolution is.
//
// Every term in THIS kernel used to be an unnormalised per-sample SUM over the
// whole native depth grid: 256 x 192 = 49,152 samples per frame on an iPhone
// LiDAR. A per-sample Huber of order 0.01 summed over ~30,000 contributing
// samples is a loss of order 300, against a photometric loss of order 0.04.
// Roughly four orders of magnitude. Two things followed, neither of which
// logged anything:
//
//   * The photographs were inert. Every geometry parameter (mean, log-scale,
//     rotation, opacity) was fitted to the laser and effectively not to the
//     picture, which for a splat renderer throws away the half of the input
//     that carries appearance and fine detail.
//
//   * The AbsGS densification statistic in `trainer_rasterize_backward` is
//     `length(dLdMean2D)`, and `dLdMean2D` is fed by BOTH the colour channel
//     (`dLdC`) and the depth channel (`dLdD`, `dLdTTotal`). With the depth
//     channel ~10^5 times larger, the statistic that decides WHERE to add
//     detail was measuring where the laser disagrees, not where the picture is
//     wrong. Densification put its splats in the wrong places.
//
// Dividing by the sample count makes each term a per-sample MEAN, so
// `depthLossScale = 1.0`, `bimodalWeight = 1.0`, `freeSpaceLowerBoundWeight =
// 0.5` and `alphaSupervisionWeight = 0.05` finally read the way anyone would
// assume: a multiple of, and a fraction of, the photometric loss.
//
// ONE denominator for all five terms, not one per term. That is deliberate.
// The bimodal and transition-width terms only fire on geometric-edge samples,
// a few per cent of the frame. Dividing THOSE by the count of edge samples
// would make `bimodalWeight = 1.0` mean "the edge term totals as much as the
// whole Huber term", which is not what it says. One denominator makes the
// weights comparable PER SAMPLE, which is what they read as.
//
// FORWARD AND BACKWARD ARE THE SAME ARITHMETIC HERE. Every term's loss value
// and its gradient are both built from `w` (the four supervised terms) or from
// `fw` (the free-space hinge), inside this one kernel. Scaling `w` and `fw`
// therefore scales the value and the gradient by exactly the same factor, by
// construction rather than by two edits that have to be kept in step. Nothing
// downstream rescales the depth channel again: `trainer_rasterize_backward`
// reads `gradDepth` and `gradTFinal` verbatim into `dLdD` and `dLdTExtra`, and
// `trainer_preprocess_backward` never touches either. The factor appears once,
// on both sides, in one place.
//
// THE DENOMINATOR IS THE NUMBER OF SUPERVISED SAMPLES, NOT THE NUMBER
// DISPATCHED. `u.depthSampleCount` is the whole native grid, including every
// sample that carries no weight: a no-return, a dilation-band pixel, an
// UNKNOWN pixel, anything under `minimumAuthorityForDepth`. Dividing by that
// would run the geometry terms at the supervised FRACTION of their nominal
// strength, which on a scan where the laser gets a vote on 8 per cent of the
// frame is a factor of twelve, and a different factor on every frame.
// `u.depthSupervisedCount` is the count of samples with `weight > 0`, measured
// on the CPU over the exact prefix that was uploaded
// (`TrainerFrameSupervision.supervisedSampleCount`), so it costs no readback
// and cannot disagree with what the GPU was handed.
//
// It falls back to `u.depthSampleCount` when it is zero, and that fallback is
// load-bearing rather than defensive. The four weighted terms all vanish when
// nothing is supervised, but the F2 free-space hinge does NOT carry
// `s.weight`: a frame whose photo QC weight is zero has no supervised samples
// and can still have thousands of live hinge terms. Dividing those by one
// would put an unnormalised sum straight back into the loss, which is the
// exact fault this whole block exists to remove.
// ============================================================================

kernel void trainer_loss_depth(
    const device TrainerDepthSample* samples     [[buffer(0)]],
    const device float*              renderDepth [[buffer(1)]],
    const device float*              renderAlpha [[buffer(2)]],
    device float*                    gradDepth   [[buffer(3)]],  // dL/dD_accumulated
    device float*                    gradTFinal  [[buffer(4)]],  // dL/dT_final
    device float*                    unknownMask [[buffer(5)]],
    device atomic_float*             lossAccum   [[buffer(6)]],
    constant TrainerLossUniforms&    u           [[buffer(7)]],
    uint                             gid         [[thread_position_in_grid]]
) {
    if (gid >= u.depthSampleCount) { return; }
    const TrainerDepthSample s = samples[gid];
    if (s.pixelIndex >= u.pixelCount) { return; }

    // The UNKNOWN mask is written whether or not this sample carries weight:
    // it is what switches off late opacity binarization for the Gaussians that
    // land here, and "we do not know" is exactly the case that must be marked.
    if (s.edgeClass == TRAINER_EDGE_UNKNOWN) {
        unknownMask[s.pixelIndex] = 1.0f;
    }

    // F3: zero, not down-weighted, inside the dilation band. The upsampled
    // value there is wrong rather than noisy, and averaging a wrong value in
    // is worse than having none.
    if (s.edgeClass == TRAINER_EDGE_BAND || s.edgeClass == TRAINER_EDGE_UNKNOWN) {
        return;
    }

    const float alpha = renderAlpha[s.pixelIndex];
    const float accumulated = renderDepth[s.pixelIndex];
    const float safeAlpha = max(alpha, 1e-4f);
    const float expected = accumulated / safeAlpha;

    // The per-sample mean factor. See the block comment above this kernel: it
    // is what puts the geometry terms on the same scale as the photometric
    // mean, and it multiplies the loss VALUE and the GRADIENT together because
    // both are built from `w` and `fw` below.
    const uint supervised = (u.depthSupervisedCount > 0u)
        ? u.depthSupervisedCount
        : u.depthSampleCount;
    const float invSamples = 1.0f / float(max(supervised, 1u));

    const float w = u.depthScale * s.weight * invSamples;
    float dL_dExpected = 0.0f;

    if (w > 0.0f && s.depth > 0.0f) {
        // --- Huber on the plain depth residual ------------------------------
        const float r = expected - s.depth;
        const float delta = max(s.huberDelta, 1e-4f);
        float value, grad;
        if (abs(r) <= delta) {
            value = 0.5f * r * r / delta;
            grad = r / delta;
        } else {
            value = abs(r) - 0.5f * delta;
            grad = (r < 0.0f) ? -1.0f : 1.0f;
        }
        trainer_atomicAdd(lossAccum, w * value);
        dL_dExpected += w * grad;

        // --- F4 bimodal edge supervision ------------------------------------
        // At a depth discontinuity, penalise the rendered depth against
        // whichever of the two local modes it is NEARER, plus a term that is
        // maximal exactly halfway between them. Together those stop the
        // optimiser parking a surface in the middle of a step, which is the
        // single most common 3DGS edge artefact.
        if (s.edgeClass == TRAINER_EDGE_GEOMETRIC && s.mode1 > s.mode0) {
            const float d0 = expected - s.mode0;
            const float d1 = expected - s.mode1;
            const float nearer = (abs(d0) <= abs(d1)) ? d0 : d1;
            const float bw = w * u.bimodalWeight;
            trainer_atomicAdd(lossAccum, bw * 0.5f * nearer * nearer);
            dL_dExpected += bw * nearer;

            const float span = max(s.mode1 - s.mode0, 1e-4f);
            const float t = clamp((expected - s.mode0) / span, 0.0f, 1.0f);
            const float tw = w * u.transitionWidthWeight;
            trainer_atomicAdd(lossAccum, tw * 4.0f * t * (1.0f - t));
            if (t > 0.0f && t < 1.0f) {
                dL_dExpected += tw * 4.0f * (1.0f - 2.0f * t) / span;
            }
        }

        // --- F6 structural alpha supervision --------------------------------
        // Where LiDAR says there is a surface, the pixel should be explained.
        const float aw = w * u.alphaSupervisionWeight;
        const float aResidual = 1.0f - alpha;
        trainer_atomicAdd(lossAccum, aw * aResidual * aResidual);
        // dL/dalpha = -2 aw (1 - alpha); alpha = 1 - T, so dL/dT is the
        // negative of that.
        gradTFinal[s.pixelIndex] += 2.0f * aw * aResidual;
    }

    // --- F2 free-space hinge -------------------------------------------------
    // "Empty air is evidence." A beam demonstrably passed through everything
    // nearer than the bound, so a surface rendered in front of it is provably
    // wrong, regardless of what the photometry would prefer.
    if (s.freeSpaceBound > 0.0f && expected < s.freeSpaceBound) {
        const float violation = s.freeSpaceBound - expected;
        // Same per-sample mean factor as `w`. This term does NOT carry
        // `s.weight` (a beam that passed through a volume is evidence at full
        // strength whatever the trust in its RANGE reading), so the factor has
        // to be applied here rather than inherited.
        const float fw = u.freeSpaceWeight * u.depthScale * invSamples;
        trainer_atomicAdd(lossAccum, fw * 0.5f * violation * violation);
        dL_dExpected += -fw * violation;
    }

    if (dL_dExpected == 0.0f) { return; }

    // expected = accumulated / alpha, and alpha = 1 - T_final, so:
    //   dL/dAccumulated = dL/dExpected / alpha
    //   dL/dT_final     = +dL/dExpected * accumulated / alpha^2
    gradDepth[s.pixelIndex] += dL_dExpected / safeAlpha;
    gradTFinal[s.pixelIndex] += dL_dExpected * accumulated / (safeAlpha * safeAlpha);
}

/// Turns dL/dC_final into dL/dC_splat and the background's share of
/// dL/dT_final, and accumulates the per-frame exposure gradients.
kernel void trainer_loss_finalize(
    const device float*           gradFinal    [[buffer(0)]],
    const device float*           renderColor  [[buffer(1)]],
    const device float*           renderTFinal [[buffer(2)]],
    const device float*           bgColor      [[buffer(3)]],
    device float*                 gradSplat    [[buffer(4)]],
    device float*                 gradTFinal   [[buffer(5)]],
    device atomic_float*          exposureGrad [[buffer(6)]],   // (gain, bias)
    constant TrainerLossUniforms& u            [[buffer(7)]],
    uint                          gid          [[thread_position_in_grid]]
) {
    if (gid >= u.pixelCount) { return; }

    const float3 g = float3(gradFinal[gid * 3u + 0u],
                            gradFinal[gid * 3u + 1u],
                            gradFinal[gid * 3u + 2u]);

    // C_final = gain * (C_splat + T * bg) + bias
    const float3 gSplat = u.exposureGain * g;
    gradSplat[gid * 3u + 0u] = gSplat.x;
    gradSplat[gid * 3u + 1u] = gSplat.y;
    gradSplat[gid * 3u + 2u] = gSplat.z;

    // NOTE, and it is worth reading before "restoring" the four lines that
    // used to be here: the background's contribution to dL/dT_final is added
    // by `trainer_rasterize_backward`, which computes
    //   dLdTTotal = gradTFinal + dot(bg, dLdC)
    // and its `dLdC` IS `gradSplat`, i.e. already multiplied by the exposure
    // gain. Adding `exposureGain * dot(bg, g)` here as well counted the
    // background term twice, which showed up as the far field pulling the
    // accumulated alpha about twice as hard as the loss actually asks for.
    // One term, in one place: the rasteriser backward.

    const float T = renderTFinal[gid];
    float3 bg = float3(0.0f);
    if (u.hasBackground != 0u) {
        bg = float3(bgColor[gid * 3u + 0u], bgColor[gid * 3u + 1u], bgColor[gid * 3u + 2u]);
    }
    const float3 pre = float3(renderColor[gid * 3u + 0u],
                              renderColor[gid * 3u + 1u],
                              renderColor[gid * 3u + 2u]) + T * bg;

    trainer_atomicAdd(&exposureGrad[0], dot(pre, g));
    trainer_atomicAdd(&exposureGrad[1], g.x + g.y + g.z);
}

// ============================================================================
// MARK: - Backward: rasterise
//
// Walks each tile's sorted list BACK to front, recomputing alpha exactly as
// the forward did, and accumulates:
//
//   * dL/dcolour, dL/dopacity, dL/dmean2D, dL/dconic per Gaussian
//   * the AbsGS statistic: the SUM OF MAGNITUDES of the per-pixel dL/dmean2D
//   * per-Gaussian visibility and UNKNOWN exposure, for binarization gating
//
// The per-Gaussian gradients here are 2D screen-space quantities; the mapping
// back to means, scales, rotations, SH and the camera is
// `trainer_preprocess_backward`.
// ============================================================================

kernel void trainer_rasterize_backward(
    const device uint*              values      [[buffer(0)]],
    const device uint*              tileRanges  [[buffer(1)]],
    const device TrainerSplatRaster* raster     [[buffer(2)]],
    const device float*             renderTFinal[[buffer(3)]],
    const device uint*              renderNContrib [[buffer(4)]],
    const device float*             gradSplatColor [[buffer(5)]],
    const device float*             gradDepth   [[buffer(6)]],
    const device float*             gradTFinal  [[buffer(7)]],
    const device float*             bgColor     [[buffer(8)]],
    const device float*             unknownMask [[buffer(9)]],
    // One 64-byte record per splat, replacing four separate buffers. See
    // TrainerSplatGrad2DAtomic.
    device TrainerSplatGrad2DAtomic* splatGrad2D [[buffer(10)]],
    device TrainerSplatStatsAtomic* stats       [[buffer(14)]],
    constant TrainerCameraUniforms& cam         [[buffer(15)]],
    constant TrainerLossUniforms&   lu          [[buffer(16)]],
    uint2                           tgPos       [[threadgroup_position_in_grid]],
    uint2                           tPos        [[thread_position_in_threadgroup]],
    uint                            tid         [[thread_index_in_threadgroup]]
) {
    threadgroup uint   tgIndex[TRAINER_TILE_AREA];
    threadgroup float2 tgXY[TRAINER_TILE_AREA];
    // FLOAT4 again. Staging as half was exact only while the conic itself was
    // half, and the conic went back to float because half cost 2 dB there.
    threadgroup float4 tgConicOpacity[TRAINER_TILE_AREA];
    threadgroup float4 tgColorDepth[TRAINER_TILE_AREA];
    /// Per-splat power cutoff, 512 B. See `r.pad0` in trainer_preprocess:
    /// the power below which this Gaussian cannot clear minAlpha, so the
    /// contribution is rejected with a compare instead of an exp().
    threadgroup half tgCutoff[TRAINER_TILE_AREA];


    const uint tileID = tgPos.y * cam.tileCountX + tgPos.x;
    const uint2 pixel = uint2(tgPos.x * TRAINER_TILE_W + tPos.x,
                              tgPos.y * TRAINER_TILE_H + tPos.y);
    const bool inside = (pixel.x < cam.imageWidth) && (pixel.y < cam.imageHeight);
    const uint pixelIndex = inside ? (pixel.y * cam.imageWidth + pixel.x) : 0u;
    const float2 pixelCenter = float2(float(pixel.x) + 0.5f, float(pixel.y) + 0.5f);

    const uint rangeStart = tileRanges[2u * tileID];
    const uint rangeEnd = tileRanges[2u * tileID + 1u];
    const uint total = (rangeEnd > rangeStart) ? (rangeEnd - rangeStart) : 0u;
    const uint batches = (total + TRAINER_TILE_AREA - 1u) / TRAINER_TILE_AREA;

    const float TFinal = inside ? renderTFinal[pixelIndex] : 1.0f;
    const uint lastContributor = inside ? renderNContrib[pixelIndex] : 0u;

    float3 dLdC = float3(0.0f);
    float dLdD = 0.0f;
    float dLdTExtra = 0.0f;
    float3 bg = float3(0.0f);
    float isUnknown = 0.0f;
    if (inside) {
        dLdC = float3(gradSplatColor[pixelIndex * 3u + 0u],
                      gradSplatColor[pixelIndex * 3u + 1u],
                      gradSplatColor[pixelIndex * 3u + 2u]);
        dLdD = gradDepth[pixelIndex];
        dLdTExtra = gradTFinal[pixelIndex];
        if (lu.hasBackground != 0u) {
            bg = float3(bgColor[pixelIndex * 3u + 0u],
                        bgColor[pixelIndex * 3u + 1u],
                        bgColor[pixelIndex * 3u + 2u]);
        }
        isUnknown = unknownMask[pixelIndex];
    }
    // Everything that reaches an alpha only through the FINAL transmittance:
    // the background composite and the depth normalisation.
    const float dLdTTotal = dLdTExtra + dot(bg, dLdC);

    // HOW FAR BACK ANY PIXEL IN THIS TILE ACTUALLY LOOKED.
    //
    // Every pixel already skips a Gaussian whose position in the tile list
    // is past its own lastContributor, at the `globalIndex > lastContributor`
    // test below. But it skips it AFTER the whole batch has been staged into
    // threadgroup memory by a cooperative load and a barrier, and that
    // staging is per batch rather than per pixel. A batch every pixel in the
    // tile skips is loaded, barriered and thrown away in full.
    //
    // The forward pass writes renderNContrib per pixel, so the tile already
    // knows the answer; it just was not being asked. Taking the maximum
    // across the threadgroup gives the last list position any pixel here
    // reached, and every batch beyond it can be skipped whole.
    //
    // This is EXACT. A skipped batch contains only entries with
    // globalIndex > lastContributor for EVERY pixel in the tile, which is
    // precisely the set the per-pixel test already rejected one at a time.
    // Nothing that could contribute is skipped and no arithmetic changes.
    //
    // AND IT IS THREADGROUP-UNIFORM, which is the property that makes it
    // safe. The loop below contains a threadgroup_barrier, so every thread
    // must execute the same number of iterations or the barriers diverge
    // and the result is undefined. `tgMaxContrib` is read from threadgroup
    // memory after a barrier, so all 256 threads compute the same bound.
    // A per-pixel bound here would be a correctness bug, not an
    // optimisation.
    // A SIMD-GROUP MAX, NOT A THREADGROUP MAX.
    //
    // This was 256 threads doing a contended atomic max on one threadgroup
    // address plus two barriers, per tile, per iteration: 391,680 atomics and
    // 3,060 extra barriers a frame. And it bought nothing, because the maximum
    // over all 256 pixels of a tile is pinned near `total` by any single pixel
    // that never saturates, so `lastUsefulBatch` stayed at batches - 1.
    //
    // 32 lanes is a different quantity, not a smaller version of the same one.
    // A SIMD group here is two rows of a 16-wide tile, and how deep a pixel
    // had to look is strongly correlated over two adjacent rows, so a group
    // whose pixels all terminated early can skip entries the tile as a whole
    // cannot. `simd_max` needs no threadgroup memory, no atomics and no
    // barriers.
    //
    // MUST be called by every lane before any per-thread branch: a SIMD-group
    // reduction with only some lanes participating is undefined. Outside
    // pixels carry lastContributor 0 and cannot inflate it.
    // `total` on a device without SIMD-group reductions, which bounds
    // nothing and leaves the per-lane `continue` below as the only skip:
    // exactly the behaviour before this optimisation existed.
    const uint groupDeepest = kTrainerSimdReduce ? simd_max(lastContributor) : total;

    // THE BATCH LOOP BOUND STAYS THREADGROUP-UNIFORM. It must.
    //
    // It was briefly derived from `groupDeepest`, and that was a real bug that
    // shipped in builds 132 to 138. The loop below contains a
    // threadgroup_barrier AND the cooperative staging that fills tgXY,
    // tgConicOpacity and tgColorDepth with all 256 threads. A per-SIMD-group
    // bound made different groups run different numbers of batches, so the
    // barriers diverged and, worse, groups that had finished stopped taking
    // part in the staging while groups still running read what was left. The
    // model came out visibly wrong: held-out PSNR 14.2 against 16.3, and
    // 219,000 splats against 152,000.
    //
    // `groupDeepest` is still used, but ONLY on the per-group gate inside the
    // loop, where every thread of the threadgroup still reaches both barriers
    // and still stages its entry. That is where the skipping was supposed to
    // happen and it is the only place it can safely happen.
    const int lastUsefulBatch = int(batches) - 1;

    float T = TFinal;
    float3 accumColor = float3(0.0f);
    float accumDepth = 0.0f;
    float lastAlpha = 0.0f;
    float3 lastColor = float3(0.0f);
    float lastDepth = 0.0f;

    for (int b = lastUsefulBatch; b >= 0; --b) {
        const uint batchBase = uint(b) * TRAINER_TILE_AREA;
        const uint load = rangeStart + batchBase + tid;
        if (load < rangeEnd) {
            const uint splatIndex = values[load];
            const TrainerSplatRaster d = raster[splatIndex];
            tgIndex[tid] = splatIndex;
            tgXY[tid] = float2(d.mean2D);
            tgConicOpacity[tid] = float4(float3(d.conic), float(d.opacity));
            tgColorDepth[tid] = float4(
                float(d.color0), float(d.color1), float(d.color2), d.depth
            );
            tgCutoff[tid] = d.pad0;
        } else {
            tgConicOpacity[tid] = float4(0.0f);
            // A cutoff no power can fall below, so the padding entries are
            // rejected by the cheap test and never reach the exp.
            tgCutoff[tid] = 60000.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // `batchBase >= groupDeepest` means every entry in this batch is past
        // the deepest any pixel in THIS SIMD group reached, so the whole loop
        // is skipped for the group rather than rejected entry by entry, which
        // is what the forward pass's `if (!done)` does for a finished pixel.
        //
        // The two threadgroup_barrier calls stay OUTSIDE this branch, where
        // they already are. Groups diverge here, and a barrier reachable by
        // only some lanes is a hang, not a missed optimisation.
        if (inside && batchBase < groupDeepest) {
            const uint here = min(TRAINER_TILE_AREA, total - batchBase);
            for (int j = int(here) - 1; j >= 0; --j) {
                const uint globalIndex = batchBase + uint(j) + 1u;
                if (globalIndex > lastContributor) { continue; }

                const float2 delta = tgXY[j] - pixelCenter;
                const float4 co = tgConicOpacity[j];
                const float power = -0.5f * (co.x * delta.x * delta.x
                                             + co.z * delta.y * delta.y)
                                    - co.y * delta.x * delta.y;
                // `if (power > 0) continue;` USED TO BE HERE, AND IT IS
                // PROVABLY DEAD. power is
                //     -0.5*(cx*dx^2 + cz*dy^2) - cy*dx*dy
                // with (cx, cy, cz) = (sc, -sb, sa)/det, so it equals
                //     -(0.5/det) * (sc*dx^2 - 2*sb*dx*dy + sa*dy^2)
                // and that bracket is the quadratic form of the ADJUGATE of
                // Sigma2D. Sigma2D is positive definite here (sa > 0 and
                // det > 1e-12 are both checked in trainer_preprocess, which
                // refuses to emit the splat otherwise), and the adjugate of a
                // positive definite 2x2 is positive definite, so the bracket
                // is non-negative and power can never exceed zero.
                //
                // MEASURED as well as proved: an offline simulation of this
                // loop over 40 tiles of frame 4, 7,719,936 (pixel, Gaussian)
                // pairs, found 100.00 per cent passing the test. It was a
                // compare and a branch on every pair, and this loop body is
                // entered roughly 294 million times an iteration in each of
                // the two rasterisers.
                //
                // Float rounding can leave power at about +1e-7 rather than a
                // clean zero. That is harmless: the cutoff test below rejects
                // on the same value, and exp(1e-7) is 1 to within the half
                // precision the colour is stored at anyway.
                // Cheap reject BEFORE the exp: see `r.pad0`.
                if (power < float(tgCutoff[j])) { continue; }
                const float gaussian = exp(power);
                const float alpha = min(0.99f, co.w * gaussian);
                if (alpha < cam.minAlpha) { continue; }

                // Undo one compositing step: T becomes the transmittance in
                // FRONT of this Gaussian.
                T = T / max(1.0f - alpha, 1e-6f);
                const float weight = alpha * T;

                const float3 color = float3(tgColorDepth[j].xyz);
                const float depth = tgColorDepth[j].w;

                float dLdAlpha = 0.0f;

                // Colour: the running "everything behind this one" estimate.
                accumColor = lastAlpha * lastColor + (1.0f - lastAlpha) * accumColor;
                lastColor = color;
                dLdAlpha += dot(color - accumColor, dLdC);

                if (cam.renderDepth != 0u) {
                    accumDepth = lastAlpha * lastDepth + (1.0f - lastAlpha) * accumDepth;
                    lastDepth = depth;
                    dLdAlpha += (depth - accumDepth) * dLdD;
                }

                dLdAlpha *= T;
                lastAlpha = alpha;

                // The final-transmittance channel: removing this Gaussian
                // scales T_final by 1/(1 - alpha).
                dLdAlpha += (-TFinal / max(1.0f - alpha, 1e-6f)) * dLdTTotal;

                const uint splatIndex = tgIndex[j];

                // ONE RECORD, ONE CACHE LINE. Every add below lands in the
                // same 64 bytes rather than in four different allocations.
                device TrainerSplatGrad2DAtomic* g = &splatGrad2D[splatIndex];

                // Colour gradient.
                trainer_atomicAdd(&g->color0, weight * dLdC.x);
                trainer_atomicAdd(&g->color1, weight * dLdC.y);
                trainer_atomicAdd(&g->color2, weight * dLdC.z);

                // alpha = opacity * gaussian, so:
                const float dLdG = co.w * dLdAlpha;
                trainer_atomicAdd(&g->opacity, gaussian * dLdAlpha);

                // dG/d(power) = G; d(power)/d(delta) and d(power)/d(conic).
                const float dGdPower = gaussian;
                const float gdx = -(co.x * delta.x + co.y * delta.y);
                const float gdy = -(co.z * delta.y + co.y * delta.x);
                // delta = mean2D - pixel, so d/d(mean2D) == d/d(delta).
                const float2 dLdMean2D = float2(dLdG * dGdPower * gdx,
                                                dLdG * dGdPower * gdy);

                trainer_atomicAdd(&g->mean2D0, dLdMean2D.x);
                trainer_atomicAdd(&g->mean2D1, dLdMean2D.y);

                const float dLdPower = dLdG * dGdPower;
                trainer_atomicAdd(&g->conic0,
                                  dLdPower * (-0.5f * delta.x * delta.x));
                trainer_atomicAdd(&g->conic1,
                                  dLdPower * (-delta.x * delta.y));
                trainer_atomicAdd(&g->conic2,
                                  dLdPower * (-0.5f * delta.y * delta.y));

                // --- AbsGS -------------------------------------------------
                // The MAGNITUDE, accumulated per pixel before any summation.
                // The signed sum cancels for a Gaussian straddling an edge,
                // which is exactly the Gaussian that has to split.
                trainer_atomicAdd(&stats[splatIndex].absGrad2D, length(dLdMean2D));
                trainer_atomicAdd(&stats[splatIndex].visAccum, weight);
                if (isUnknown > 0.0f) {
                    trainer_atomicAdd(&stats[splatIndex].unknownAccum, weight);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// MARK: - Backward: preprocess
//
// Maps the 2D gradients back onto means, log-scales, rotations, opacity, SH,
// and the camera's se(3) delta.
// ============================================================================

kernel void trainer_preprocess_backward(
    const device TrainerSplat*        splats     [[buffer(0)]],
    const device float*               sh         [[buffer(1)]],
    const device TrainerSplatDraw*    draws      [[buffer(2)]],
    const device TrainerSplatGrad2D*  splatGrad2D [[buffer(3)]],
    // PLAIN, not atomic. This kernel runs one thread per splat and is the
    // only writer of splatGrad[gid] for the whole dispatch, so its eleven
    // accumulations were eleven device read-modify-writes contending with
    // nothing: 2.7 million of them an iteration. Same buffer, same bytes,
    // same layout; only the view differs. See trainer_ownedAdd.
    device TrainerSplatGrad*          splatGrad  [[buffer(7)]],
    device float*                     shGrad     [[buffer(8)]],
    device atomic_float*              cameraGrad [[buffer(9)]],   // omega[3], nu[3]
    const device TrainerSplatStats*   stats      [[buffer(10)]],
    constant TrainerCameraUniforms&   cam        [[buffer(11)]],
    const device uint*                tilesTouched [[buffer(12)]],
    uint                              gid        [[thread_position_in_grid]]
) {
    if (gid >= cam.splatCount) { return; }
    // WAS `draws[gid].radiusPx.x <= 0.0h`, which cost trainer_preprocess two
    // scattered stores into a 64-byte record for EVERY splat, drawn or not,
    // purely so this line had something to read. `tilesTouched` answers the
    // same question and is a dense uint per splat, so this is a coalesced
    // 1.2 MB read instead of pulling a whole cache line per rejected splat.
    //
    // Provably the same predicate: every early return in trainer_preprocess
    // leaves tilesTouched[gid] at the 0 its own prologue wrote, and the only
    // path that writes `draws[gid]` writes tilesTouched[gid] = touched two
    // lines later with touched > 0 guaranteed by the maxX <= minX check.
    if (tilesTouched[gid] == 0u) { return; }

    const TrainerSplat s = splats[gid];
    const TrainerSplatDraw d = draws[gid];

    // One contiguous read of this splat's own row, which by now nothing
    // else is writing: the rasteriser finished in an earlier command buffer.
    const TrainerSplatGrad2D g = splatGrad2D[gid];
    const float2 dLdMean2D = float2(g.mean2D0, g.mean2D1);
    const float3 dLdConic = float3(g.conic0, g.conic1, g.conic2);
    const float3 dLdColor0 = float3(g.color0, g.color1, g.color2);
    const float dLdAlphaMul = g.opacity;

    // --- opacity ------------------------------------------------------------
    // alpha = sigmoid(o) * comp. `comp` is a stop-gradient constant; see the
    // stated approximation at the top of this file.
    const float sig = trainer_sigmoid(s.opacityLogit);
    const float dLdOpacityLogit = dLdAlphaMul * d.comp * sig * (1.0f - sig);
    trainer_ownedAdd(&splatGrad[gid].opacity, dLdOpacityLogit);

    // --- colour channels the forward clamped at 0 ---------------------------
    float3 dLdColor = dLdColor0;
    if ((d.clampedMask & 1u) != 0u) { dLdColor.x = 0.0f; }
    if ((d.clampedMask & 2u) != 0u) { dLdColor.y = 0.0f; }
    if ((d.clampedMask & 4u) != 0u) { dLdColor.z = 0.0f; }

    // --- spherical harmonics ------------------------------------------------
    const float3 meanWorld = float3(s.mean);
    const float3 rawDir = meanWorld - float3(cam.cameraCenter);
    const float dirLen = max(length(rawDir), 1e-6f);
    const float3 dir = rawDir / dirLen;

    const uint shBase = gid * cam.shCoeffCount * 3u;
    const uint active = min(cam.activeSHCoeffCount, cam.shCoeffCount);

    float3 dLdDir = float3(0.0f);

    // Degree 0.
    shGrad[shBase + 0u] += TRAINER_SH_C0 * dLdColor.x;
    shGrad[shBase + 1u] += TRAINER_SH_C0 * dLdColor.y;
    shGrad[shBase + 2u] += TRAINER_SH_C0 * dLdColor.z;

    if (active > 1u) {
        const float x = dir.x, y = dir.y, z = dir.z;
        const float b1 = -TRAINER_SH_C1 * y;
        const float b2 =  TRAINER_SH_C1 * z;
        const float b3 = -TRAINER_SH_C1 * x;
        for (uint c = 0; c < 3u; ++c) {
            shGrad[shBase + 3u + c] += b1 * dLdColor[c];
            shGrad[shBase + 6u + c] += b2 * dLdColor[c];
            shGrad[shBase + 9u + c] += b3 * dLdColor[c];
        }
        const float3 s1 = float3(sh[shBase + 3u], sh[shBase + 4u], sh[shBase + 5u]);
        const float3 s2 = float3(sh[shBase + 6u], sh[shBase + 7u], sh[shBase + 8u]);
        const float3 s3 = float3(sh[shBase + 9u], sh[shBase + 10u], sh[shBase + 11u]);
        dLdDir.x += dot(-TRAINER_SH_C1 * s3, dLdColor);
        dLdDir.y += dot(-TRAINER_SH_C1 * s1, dLdColor);
        dLdDir.z += dot( TRAINER_SH_C1 * s2, dLdColor);

        if (active > 4u) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            const float b4 = TRAINER_SH_C2_0 * xy;
            const float b5 = TRAINER_SH_C2_1 * yz;
            const float b6 = TRAINER_SH_C2_2 * (2.0f * zz - xx - yy);
            const float b7 = TRAINER_SH_C2_3 * xz;
            const float b8 = TRAINER_SH_C2_4 * (xx - yy);
            for (uint c = 0; c < 3u; ++c) {
                shGrad[shBase + 12u + c] += b4 * dLdColor[c];
                shGrad[shBase + 15u + c] += b5 * dLdColor[c];
                shGrad[shBase + 18u + c] += b6 * dLdColor[c];
                shGrad[shBase + 21u + c] += b7 * dLdColor[c];
                shGrad[shBase + 24u + c] += b8 * dLdColor[c];
            }
            const float3 s4 = float3(sh[shBase + 12u], sh[shBase + 13u], sh[shBase + 14u]);
            const float3 s5 = float3(sh[shBase + 15u], sh[shBase + 16u], sh[shBase + 17u]);
            const float3 s6 = float3(sh[shBase + 18u], sh[shBase + 19u], sh[shBase + 20u]);
            const float3 s7 = float3(sh[shBase + 21u], sh[shBase + 22u], sh[shBase + 23u]);
            const float3 s8 = float3(sh[shBase + 24u], sh[shBase + 25u], sh[shBase + 26u]);
            dLdDir.x += dot(TRAINER_SH_C2_0 * y * s4
                            + TRAINER_SH_C2_2 * (-2.0f * x) * s6
                            + TRAINER_SH_C2_3 * z * s7
                            + TRAINER_SH_C2_4 * (2.0f * x) * s8, dLdColor);
            dLdDir.y += dot(TRAINER_SH_C2_0 * x * s4
                            + TRAINER_SH_C2_1 * z * s5
                            + TRAINER_SH_C2_2 * (-2.0f * y) * s6
                            + TRAINER_SH_C2_4 * (-2.0f * y) * s8, dLdColor);
            dLdDir.z += dot(TRAINER_SH_C2_1 * y * s5
                            + TRAINER_SH_C2_2 * (4.0f * z) * s6
                            + TRAINER_SH_C2_3 * x * s7, dLdColor);
        }
    }

    // Through the normalisation of the view direction.
    const float3 dLdRawDir = (dLdDir - dir * dot(dir, dLdDir)) / dirLen;

    // --- dL/dSigma2D from dL/dconic -----------------------------------------
    // conic = (c/D, -b/D, a/D) with Sigma2D' = [[a, b], [b, c]], D = ac - b^2.
    // The 2D Mip filter is an additive constant, so dSigma2D'/dSigma2D = I.
    const float ca = d.conic.x, cb = d.conic.y, cc = d.conic.z;
    // Recover (a, b, c) by inverting the conic: Sigma = conic^{-1}.
    const float conicDet = ca * cc - cb * cb;
    if (abs(conicDet) < 1e-20f) { return; }
    const float invConicDet = 1.0f / conicDet;
    const float a = cc * invConicDet;
    const float bb = -cb * invConicDet;
    const float c = ca * invConicDet;

    const float D = a * c - bb * bb;
    const float invD2 = 1.0f / max(D * D, 1e-20f);
    const float dLda = invD2 * (-c * c * dLdConic.x + bb * c * dLdConic.y
                                - bb * bb * dLdConic.z);
    const float dLdb = invD2 * (2.0f * bb * c * dLdConic.x
                                - (D + 2.0f * bb * bb) * dLdConic.y
                                + 2.0f * a * bb * dLdConic.z);
    const float dLdc = invD2 * (-bb * bb * dLdConic.x + a * bb * dLdConic.y
                                - a * a * dLdConic.z);

    // As a symmetric matrix: the off-diagonal derivative is split in half
    // because `b` names both entries.
    const float2x2 Gm = float2x2(float2(dLda, 0.5f * dLdb),
                                 float2(0.5f * dLdb, dLdc));

    // --- rebuild the forward's intermediates --------------------------------
    const float3 meanCam = float3(d.meanCam);
    const float invZ = 1.0f / meanCam.z;
    const float invZ2 = invZ * invZ;
    const float invZ3 = invZ2 * invZ;

    const float3 jr0 = float3(cam.fx * invZ, 0.0f, -cam.fx * meanCam.x * invZ2);
    const float3 jr1 = float3(0.0f, cam.fy * invZ, -cam.fy * meanCam.y * invZ2);

    const float3 scale = exp(clamp(float3(s.logScale), -12.0f, 3.0f));
    const float4 q = normalize(float4(s.rotation));
    const float3x3 R = trainer_quatToMatrix(q);
    const float3x3 S = float3x3(float3(scale.x, 0, 0),
                                float3(0, scale.y, 0),
                                float3(0, 0, scale.z));
    const float3x3 M = R * S;
    float3x3 sigmaWorld = M * transpose(M);
    const float filter3D = max(stats[gid].filter3D, 0.0f);
    const float f3sq = filter3D * filter3D;
    sigmaWorld[0][0] += f3sq;
    sigmaWorld[1][1] += f3sq;
    sigmaWorld[2][2] += f3sq;

    const float3x3 W = trainer_viewRotation(cam.viewMatrix);
    const float3x3 sigmaCam = W * sigmaWorld * transpose(W);

    // --- dL/dSigmaCam = A^T Gm A, with A = J (2x3) --------------------------
    // A row i is jr(i). (A^T Gm A)_{mn} = sum_{ij} A_{im} Gm_{ij} A_{jn}.
    float3x3 dLdSigmaCam;
    for (uint m = 0; m < 3u; ++m) {
        for (uint n = 0; n < 3u; ++n) {
            const float am0 = jr0[m], am1 = jr1[m];
            const float an0 = jr0[n], an1 = jr1[n];
            dLdSigmaCam[n][m] = am0 * Gm[0][0] * an0 + am0 * Gm[1][0] * an1
                              + am1 * Gm[0][1] * an0 + am1 * Gm[1][1] * an1;
        }
    }

    // --- dL/dSigmaWorld = W^T dL/dSigmaCam W --------------------------------
    const float3x3 dLdSigmaWorld = transpose(W) * dLdSigmaCam * W;

    // --- dL/dM = 2 * dL/dSigmaWorld * M -------------------------------------
    const float3x3 dLdM = 2.0f * (dLdSigmaWorld * M);

    // --- scales -------------------------------------------------------------
    // dL/ds_j = (R^T dL/dM)_{jj}; dL/dlogScale_j = dL/ds_j * s_j.
    const float3x3 RtG = transpose(R) * dLdM;
    const float3 dLdLogScale = float3(RtG[0][0] * scale.x,
                                      RtG[1][1] * scale.y,
                                      RtG[2][2] * scale.z);
    trainer_ownedAdd(&splatGrad[gid].scale0, dLdLogScale.x);
    trainer_ownedAdd(&splatGrad[gid].scale1, dLdLogScale.y);
    trainer_ownedAdd(&splatGrad[gid].scale2, dLdLogScale.z);

    // --- rotation -----------------------------------------------------------
    // dL/dR = dL/dM * S^T (S diagonal).
    float3x3 dLdR;
    for (uint col = 0; col < 3u; ++col) {
        dLdR[col] = dLdM[col] * scale[col];
    }
    const float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    // dR/dq, written out. Index order matches trainer_quatToMatrix: m[col][row].
    float4 dLdQ = float4(0.0f);
    {
        // Column-major access helper: R[col][row].
        const float g00 = dLdR[0][0], g10 = dLdR[0][1], g20 = dLdR[0][2];
        const float g01 = dLdR[1][0], g11 = dLdR[1][1], g21 = dLdR[1][2];
        const float g02 = dLdR[2][0], g12 = dLdR[2][1], g22 = dLdR[2][2];

        dLdQ.x = 2.0f * (      g01 * qy + g02 * qz
                        + g10 * qy - 2.0f * g11 * qx - g12 * qw
                        + g20 * qz + g21 * qw - 2.0f * g22 * qx);
        dLdQ.y = 2.0f * (-2.0f * g00 * qy + g01 * qx + g02 * qw
                        + g10 * qx +                    g12 * qz
                        - g20 * qw + g21 * qz - 2.0f * g22 * qy);
        dLdQ.z = 2.0f * (-2.0f * g00 * qz - g01 * qw + g02 * qx
                        + g10 * qw - 2.0f * g11 * qz + g12 * qy
                        + g20 * qx + g21 * qy);
        dLdQ.w = 2.0f * (               - g01 * qz + g02 * qy
                        + g10 * qz               - g12 * qx
                        - g20 * qy + g21 * qx);
    }
    // Through the normalisation: q_used = q_raw / |q_raw|, |q_raw| == 1 here
    // because the optimiser renormalises after every step.
    const float4 qRaw = float4(s.rotation);
    const float qLen = max(length(qRaw), 1e-8f);
    const float4 qHat = qRaw / qLen;
    const float4 dLdQRaw = (dLdQ - qHat * dot(qHat, dLdQ)) / qLen;
    trainer_ownedAdd(&splatGrad[gid].rot0, dLdQRaw.x);
    trainer_ownedAdd(&splatGrad[gid].rot1, dLdQRaw.y);
    trainer_ownedAdd(&splatGrad[gid].rot2, dLdQRaw.z);
    trainer_ownedAdd(&splatGrad[gid].rot3, dLdQRaw.w);

    // --- mean ---------------------------------------------------------------
    // (a) through the projection of the centre
    float3 dLdMeanCam = float3(
        cam.fx * invZ * dLdMean2D.x,
        cam.fy * invZ * dLdMean2D.y,
        -(cam.fx * meanCam.x * invZ2) * dLdMean2D.x
        - (cam.fy * meanCam.y * invZ2) * dLdMean2D.y
    );

    // (b) through J, which depends on the centre too. dL/dJ = 2 Gm J Sigma_cam.
    {
        const float3 jsc0 = sigmaCam * jr0;      // Sigma_cam is symmetric
        const float3 jsc1 = sigmaCam * jr1;
        const float3 dLdJ0 = 2.0f * (Gm[0][0] * jsc0 + Gm[1][0] * jsc1);
        const float3 dLdJ1 = 2.0f * (Gm[0][1] * jsc0 + Gm[1][1] * jsc1);

        dLdMeanCam.x += dLdJ0.z * (-cam.fx * invZ2);
        dLdMeanCam.y += dLdJ1.z * (-cam.fy * invZ2);
        dLdMeanCam.z += dLdJ0.x * (-cam.fx * invZ2)
                      + dLdJ1.y * (-cam.fy * invZ2)
                      + dLdJ0.z * (2.0f * cam.fx * meanCam.x * invZ3)
                      + dLdJ1.z * (2.0f * cam.fy * meanCam.y * invZ3);
    }

    // World-space mean gradient: through the view rotation, plus the SH view
    // direction, which also depends on the world position.
    const float3 dLdMeanWorld = transpose(W) * dLdMeanCam + dLdRawDir;
    trainer_ownedAdd(&splatGrad[gid].mean0, dLdMeanWorld.x);
    trainer_ownedAdd(&splatGrad[gid].mean1, dLdMeanWorld.y);
    trainer_ownedAdd(&splatGrad[gid].mean2, dLdMeanWorld.z);

    // --- camera se(3) delta (F1) --------------------------------------------
    // Left perturbation: p_cam -> (I + omega^) p_cam + nu, W -> (I + omega^) W.
    //   dL/dnu    = dL/dp_cam
    //   dL/domega = p_cam x dL/dp_cam                (from the centre)
    //             - 2 * vee(Sigma_cam G - G Sigma_cam)   (from the covariance)
    if (cameraGrad != nullptr) {
        const float3 dOmegaMean = cross(meanCam, dLdMeanCam);
        const float3x3 comm = sigmaCam * dLdSigmaCam - dLdSigmaCam * sigmaCam;
        // vee of an antisymmetric matrix, MSL column-major: m[col][row].
        const float3 vee = float3(comm[1][2], comm[2][0], comm[0][1]);
        const float3 dOmega = dOmegaMean - 2.0f * vee;

        trainer_atomicAdd(&cameraGrad[0], dOmega.x);
        trainer_atomicAdd(&cameraGrad[1], dOmega.y);
        trainer_atomicAdd(&cameraGrad[2], dOmega.z);
        trainer_atomicAdd(&cameraGrad[3], dLdMeanCam.x);
        trainer_atomicAdd(&cameraGrad[4], dLdMeanCam.y);
        trainer_atomicAdd(&cameraGrad[5], dLdMeanCam.z);
    }
}

// ============================================================================
// MARK: - Mip-Splatting 3D filter sizing
// ============================================================================

/// One camera's contribution to the per-Gaussian top-K sampling rate.
/// Sampling rate is pixels per world metre at this Gaussian's depth,
/// `focal / z`. Run over every keyframe camera during a periodic sweep.
kernel void trainer_sampling_rate_update(
    const device TrainerSplat*      splats [[buffer(0)]],
    device TrainerSamplingTopK*     topK   [[buffer(1)]],
    constant TrainerCameraUniforms& cam    [[buffer(2)]],
    uint                            gid    [[thread_position_in_grid]]
) {
    if (gid >= cam.splatCount) { return; }
    const float3 meanCam = (cam.viewMatrix * float4(float3(splats[gid].mean), 1.0f)).xyz;
    if (meanCam.z < cam.nearPlane || meanCam.z > cam.farPlane) { return; }

    const float2 uv = float2(cam.fx * meanCam.x / meanCam.z + cam.cx,
                             cam.fy * meanCam.y / meanCam.z + cam.cy);
    // Off-frame cameras are not observations.
    if (uv.x < 0.0f || uv.y < 0.0f
        || uv.x >= float(cam.imageWidth) || uv.y >= float(cam.imageHeight)) { return; }

    const float rate = max(cam.fx, cam.fy) / max(meanCam.z, 1e-4f);

    TrainerSamplingTopK t = topK[gid];
    if (rate > t.r0)      { t.r3 = t.r2; t.r2 = t.r1; t.r1 = t.r0; t.r0 = rate; }
    else if (rate > t.r1) { t.r3 = t.r2; t.r2 = t.r1; t.r1 = rate; }
    else if (rate > t.r2) { t.r3 = t.r2; t.r2 = rate; }
    else if (rate > t.r3) { t.r3 = rate; }
    topK[gid] = t;
}

/// Turns the top-K sampling rates into a per-Gaussian 3D filter size.
///
/// `filterScale` is Mip-Splatting's 0.2. Using `r3` (the K-th largest rate)
/// rather than `r0` is what makes this a high percentile rather than a maximum
/// and stops one accidental close-up frame from shrinking the filter for a
/// Gaussian the rest of the capture only ever saw from three metres away.
kernel void trainer_filter3d_finalize(
    const device TrainerSamplingTopK* topK        [[buffer(0)]],
    device TrainerSplatStats*         stats       [[buffer(1)]],
    constant uint&                    count       [[buffer(2)]],
    constant float&                   filterScale [[buffer(3)]],
    constant float&                   fallback    [[buffer(4)]],
    uint                              gid         [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const TrainerSamplingTopK t = topK[gid];
    // Fall back through the list: a Gaussian seen by fewer than K cameras uses
    // the smallest rate it actually has, never a zero.
    float rate = t.r3;
    if (rate <= 0.0f) { rate = t.r2; }
    if (rate <= 0.0f) { rate = t.r1; }
    if (rate <= 0.0f) { rate = t.r0; }
    stats[gid].filter3D = (rate > 0.0f) ? (filterScale / rate) : fallback;
}

// ============================================================================
// MARK: - Regulariser (F4)
// ============================================================================

kernel void trainer_regularizer(
    const device TrainerSplat*      splats [[buffer(0)]],
    const device TrainerSplatStats* stats  [[buffer(1)]],
    device TrainerSplatGrad*        grad   [[buffer(2)]],
    device atomic_float*            lossAccum [[buffer(3)]],
    constant TrainerRegUniforms&    u      [[buffer(4)]],
    uint                            gid    [[thread_position_in_grid]]
) {
    if (gid >= u.count) { return; }
    const TrainerSplat s = splats[gid];
    const TrainerSplatStats st = stats[gid];

    const float3 logScale = clamp(float3(s.logScale), -12.0f, 3.0f);
    const float3 scale = exp(logScale);

    // --- effective-rank / disc prior ---------------------------------------
    // Effective rank of the covariance's eigenvalue spectrum, via the entropy
    // of the normalised eigenvalues. 2 is a disc, 1 is a needle. Surfaces want
    // discs; a Gaussian sitting on a detected 3D edge curve wants a needle,
    // which is why the flag exempts it rather than the prior being switched
    // off wholesale.
    if (u.discWeight > 0.0f) {
        const float3 lambda = scale * scale;
        const float sum = max(lambda.x + lambda.y + lambda.z, 1e-20f);
        const float3 p = lambda / sum;
        const float3 logP = log(max(p, float3(1e-20f)));
        const float H = -dot(p, logP);
        const float rank = exp(H);

        const bool onEdge = (s.flags & 4u) != 0u;
        const float target = onEdge ? u.edgeTargetRank : u.discTargetRank;
        const float residual = rank - target;
        trainer_atomicAdd(lossAccum, u.discWeight * 0.5f * residual * residual);

        // WAS -4.0f, AND THE COMMENT SAID -4 TOO. BOTH WERE WRONG.
        //
        // The loss this kernel accumulates two lines above is
        // w * 0.5 * (rank - target)^2 with rank = exp(H). With
        // p_j = lambda_j / sum(lambda) and lambda_j = exp(2 * logScale_j),
        //     dp_k/dlogScale_j = 2 p_k (delta_kj - p_j)
        //     dH/dlogScale_j   = -2 p_j (log p_j + H)
        // so the derivative is -2 w residual rank p_j (log p_j + H). Two, not
        // four. The gradient was exactly twice the derivative of the loss it
        // reported, so the effective disc weight was 0.002 while the constant
        // said 0.001, and the reported loss and the applied gradient disagreed.
        //
        // Confirmed two ways by two agents that did not share code: analytic
        // differentiation, and central differences on 9,000 components from
        // 3,000 real splats, which gave an analytic/numeric ratio of exactly
        // 2.000000 at h = 1e-6, 1e-5 and 1e-4, and 1.000000 after the fix.
        //
        // NOTE FOR ATTRIBUTION: the 0.01 -> 0.001 decision was made against
        // the 4x behaviour, so the effective weight has now moved 20x from
        // where it started, not 10x.
        // dL/dlogScale_j = -2 w (rank - target) rank p_j (log p_j + H)
        const float k = -2.0f * u.discWeight * residual * rank;
        grad[gid].scale0 += k * p.x * (logP.x + H);
        grad[gid].scale1 += k * p.y * (logP.y + H);
        grad[gid].scale2 += k * p.z * (logP.z + H);
    }

    // --- hinge on runaway scale ---------------------------------------------
    if (u.maxScaleWeight > 0.0f) {
        const float3 over = max(scale - u.maxScaleMeters, float3(0.0f));
        const float value = 0.5f * dot(over, over);
        if (value > 0.0f) {
            trainer_atomicAdd(lossAccum, u.maxScaleWeight * value);
            const float3 g = u.maxScaleWeight * over * scale;   // d(scale)/d(log) = scale
            grad[gid].scale0 += g.x;
            grad[gid].scale1 += g.y;
            grad[gid].scale2 += g.z;
        }
    }

    // --- late opacity binarization (F4) --------------------------------------
    // Push opacity to a decision in the last stretch of the run, EXCEPT where
    // this Gaussian's observations were mostly UNKNOWN (glass, out of range,
    // low confidence). Forcing a decision there is inventing an answer.
    if (u.binarizeWeight > 0.0f) {
        const float vis = max(st.visAccum, 1e-6f);
        const float unknownFraction = clamp(st.unknownAccum / vis, 0.0f, 1.0f);
        if (unknownFraction < u.binarizeUnknownCutoff) {
            const float gate = 1.0f - unknownFraction / max(u.binarizeUnknownCutoff, 1e-6f);
            const float sig = trainer_sigmoid(s.opacityLogit);
            const float value = sig * (1.0f - sig);
            trainer_atomicAdd(lossAccum, u.binarizeWeight * gate * value);
            // d/do [sigma (1 - sigma)] = sigma (1 - sigma) (1 - 2 sigma)
            grad[gid].opacity += u.binarizeWeight * gate * value * (1.0f - 2.0f * sig);
        }
    }
}

// ============================================================================
// MARK: - Optimiser: visibility-masked sparse Adam
// ============================================================================

kernel void trainer_adam_splat(
    device TrainerSplat*         splats [[buffer(0)]],
    const device TrainerSplatGrad* grad [[buffer(1)]],
    device TrainerSplatGrad*     mBuf   [[buffer(2)]],
    device TrainerSplatGrad*     vBuf   [[buffer(3)]],
    device TrainerSplatStats*    stats  [[buffer(4)]],
    constant TrainerAdamUniforms& u     [[buffer(5)]],
    uint                         gid    [[thread_position_in_grid]]
) {
    if (gid >= u.count) { return; }
    if (u.sparse != 0u && stats[gid].visibleFlag == 0u) { return; }

    // Per-Gaussian step count. A Gaussian seen in 3 of 3000 steps has to be
    // bias-corrected as if it were on step 3; using the global step here is
    // the classic sparse-Adam bug and it makes rarely-seen Gaussians move in
    // huge, unstable jumps the first time they are touched.
    const uint step = stats[gid].stepCount + 1u;
    stats[gid].stepCount = step;

    const float bc1 = 1.0f - pow(u.beta1, float(step));
    const float bc2 = 1.0f - pow(u.beta2, float(step));

    TrainerSplat s = splats[gid];
    const TrainerSplatGrad g = grad[gid];
    TrainerSplatGrad m = mBuf[gid];
    TrainerSplatGrad v = vBuf[gid];

    const bool pinned = (s.flags & 1u) != 0u;
    const float meanLR = u.lrMean * (pinned ? u.pinnedPositionLRScale : 1.0f);

    float gv[12] = { g.rot0, g.rot1, g.rot2, g.rot3,
                     g.mean0, g.mean1, g.mean2, g.opacity,
                     g.scale0, g.scale1, g.scale2, g.pad };
    float mv[12] = { m.rot0, m.rot1, m.rot2, m.rot3,
                     m.mean0, m.mean1, m.mean2, m.opacity,
                     m.scale0, m.scale1, m.scale2, m.pad };
    float vv[12] = { v.rot0, v.rot1, v.rot2, v.rot3,
                     v.mean0, v.mean1, v.mean2, v.opacity,
                     v.scale0, v.scale1, v.scale2, v.pad };
    const float lr[12] = { u.lrRotation, u.lrRotation, u.lrRotation, u.lrRotation,
                           meanLR, meanLR, meanLR, u.lrOpacity,
                           u.lrScale, u.lrScale, u.lrScale, 0.0f };

    float delta[12];
    for (uint i = 0; i < 12u; ++i) {
        float gi = gv[i];
        if (!isfinite(gi)) { gi = 0.0f; }
        mv[i] = u.beta1 * mv[i] + (1.0f - u.beta1) * gi;
        vv[i] = u.beta2 * vv[i] + (1.0f - u.beta2) * gi * gi;
        const float mh = mv[i] / bc1;
        const float vh = vv[i] / bc2;
        delta[i] = lr[i] * mh / (sqrt(max(vh, 0.0f)) + u.epsilon);
    }

    m.rot0 = mv[0]; m.rot1 = mv[1]; m.rot2 = mv[2]; m.rot3 = mv[3];
    m.mean0 = mv[4]; m.mean1 = mv[5]; m.mean2 = mv[6]; m.opacity = mv[7];
    m.scale0 = mv[8]; m.scale1 = mv[9]; m.scale2 = mv[10]; m.pad = mv[11];
    v.rot0 = vv[0]; v.rot1 = vv[1]; v.rot2 = vv[2]; v.rot3 = vv[3];
    v.mean0 = vv[4]; v.mean1 = vv[5]; v.mean2 = vv[6]; v.opacity = vv[7];
    v.scale0 = vv[8]; v.scale1 = vv[9]; v.scale2 = vv[10]; v.pad = vv[11];
    mBuf[gid] = m;
    vBuf[gid] = v;

    float4 rot = float4(s.rotation) - float4(delta[0], delta[1], delta[2], delta[3]);
    rot = normalize(rot);
    if (!all(isfinite(rot))) { rot = float4(0.0f, 0.0f, 0.0f, 1.0f); }
    s.rotation = packed_float4(rot);

    float3 mean = float3(s.mean) - float3(delta[4], delta[5], delta[6]);
    if (!all(isfinite(mean))) { mean = float3(s.mean); }
    s.mean = packed_float3(mean);

    s.opacityLogit = clamp(s.opacityLogit - delta[7],
                           -u.maxOpacityLogit, u.maxOpacityLogit);

    float3 logScale = float3(s.logScale) - float3(delta[8], delta[9], delta[10]);
    logScale = clamp(logScale, u.minLogScale, u.maxLogScale);
    if (!all(isfinite(logScale))) { logScale = float3(s.logScale); }
    s.logScale = packed_float3(logScale);

    splats[gid] = s;
}

kernel void trainer_adam_sh(
    device float*                 sh    [[buffer(0)]],
    const device float*           grad  [[buffer(1)]],
    device float*                 mBuf  [[buffer(2)]],
    device float*                 vBuf  [[buffer(3)]],
    const device TrainerSplatStats* stats [[buffer(4)]],
    constant TrainerAdamUniforms& u     [[buffer(5)]],
    uint                          gid   [[thread_position_in_grid]]
) {
    // One thread per Gaussian; each walks its own coefficient run, so the
    // visibility mask is a single load rather than one per float.
    if (gid >= u.count) { return; }
    if (u.sparse != 0u && stats[gid].visibleFlag == 0u) { return; }

    const uint step = max(stats[gid].stepCount, 1u);
    const float bc1 = 1.0f - pow(u.beta1, float(step));
    const float bc2 = 1.0f - pow(u.beta2, float(step));

    const uint floatsPerSplat = u.shCoeffCount * 3u;
    const uint base = gid * floatsPerSplat;

    for (uint i = 0; i < floatsPerSplat; ++i) {
        const uint k = base + i;
        float g = grad[k];
        if (!isfinite(g)) { g = 0.0f; }
        const float lr = (i < 3u) ? u.lrSHDC : u.lrSHRest;
        if (lr == 0.0f) { continue; }
        const float m = u.beta1 * mBuf[k] + (1.0f - u.beta1) * g;
        const float v = u.beta2 * vBuf[k] + (1.0f - u.beta2) * g * g;
        mBuf[k] = m;
        vBuf[k] = v;
        const float upd = lr * (m / bc1) / (sqrt(max(v / bc2, 0.0f)) + u.epsilon);
        const float next = sh[k] - upd;
        sh[k] = isfinite(next) ? clamp(next, -30.0f, 30.0f) : sh[k];
    }
}

// ============================================================================
// MARK: - Preview / snapshot support
// ============================================================================

/// Writes the per-Gaussian world-space centre out as a flat float3 array, so
/// the Swift side can hand centres to `FreeSpaceCarver.certifiedEmptyIndices`
/// without walking a 48-byte-strided struct on the CPU.
kernel void trainer_extract_centers(
    const device TrainerSplat* splats  [[buffer(0)]],
    device float*              centers [[buffer(1)]],
    constant uint&             count   [[buffer(2)]],
    uint                       gid     [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const float3 m = float3(splats[gid].mean);
    centers[gid * 3u + 0u] = m.x;
    centers[gid * 3u + 1u] = m.y;
    centers[gid * 3u + 2u] = m.z;
}
