//
//  SplatRenderShaders.metal
//  Viewer
//
//  THE PREVIEW RENDERER'S GPU SIDE.
//
//  Every symbol in this file is prefixed `viewer_`. The whole app is ONE Xcode
//  target, so every .metal file under Sources/ is compiled into ONE
//  default.metallib and two kernels with the same name anywhere in the project
//  is a link error. Sources/Trainer owns `trainer_*`; this file owns
//  `viewer_*` and nothing else.
//
//  Pipeline, in the order the renderer runs it:
//
//    1. viewer_splat_preprocess   projects every Gaussian, evaluates SH,
//                                 culls, and writes a depth key
//    2. viewer_sort_local /
//       viewer_sort_global        bitonic sort of the keys, front to back
//    3. viewer_prepare_indirect   turns the atomic visible counter into
//                                 MTLDrawPrimitivesIndirectArguments
//    4. viewer_splat_vertex /
//       viewer_splat_fragment     one quad per visible Gaussian, composited
//                                 FRONT to BACK with programmable blending
//    5. viewer_composite_vertex /
//       viewer_composite_fragment honesty mask + artefact heatmap, into the
//                                 drawable
//
//  Why front-to-back with framebuffer fetch rather than back-to-front with
//  fixed-function blending: on Apple GPUs a fragment shader may read the
//  current value of its own colour attachment (`[[color(n)]]` on an input
//  argument), and those reads are ordered per pixel. That gives exact
//  front-to-back compositing, early-out once the pixel is opaque, and - the
//  reason it matters here - a place to accumulate the two extra per-pixel
//  quantities the review UX needs: the alpha-weighted depth (so the honesty
//  mask can rebuild a world point) and the raw splat overlap count (the
//  artefact heatmap's main signal). Neither survives fixed-function blending.
//
//  Coordinate conventions are Core's, unchanged: world is right-handed Y-up
//  metres; the camera frame is +X right, +Y DOWN, +Z FORWARD; a Pose is
//  world -> camera. Nothing here negates an axis.
//

#include <metal_stdlib>
using namespace metal;

// =============================================================================
//  MARK: - Buffer layouts
//
//  Byte-matched to ViewerGPULayouts.swift. Sizes are asserted at run time by
//  ViewerGPULayouts.verify(); the static_asserts below catch the Metal side at
//  compile time.
// =============================================================================

struct ViewerSplatBase {
    packed_float3 position;      //  0
    float         opacityLogit;  // 12
    packed_float4 rotation;      // 16  (x, y, z, w)
    packed_float3 logScale;      // 32
    packed_float3 colorDC;       // 44
    uint          shRestOffset;  // 56
    uint          flags;         // 60
};                               // 64
static_assert(sizeof(ViewerSplatBase) == 64, "ViewerSplatBase must be 64 bytes");

struct ViewerSplatDraw {
    packed_float2 meanPx;   //  0
    packed_float2 axis1Px;  //  8
    packed_float2 axis2Px;  // 16
    packed_float4 color;    // 24  (rgb, alpha)
    float         depth;    // 40
    float         pad0;     // 44
};                          // 48
static_assert(sizeof(ViewerSplatDraw) == 48, "ViewerSplatDraw must be 48 bytes");

struct ViewerSortEntry {
    uint key;
    uint index;
};
static_assert(sizeof(ViewerSortEntry) == 8, "ViewerSortEntry must be 8 bytes");

struct ViewerUniforms {
    float4x4 viewMatrix;       //   0
    float4   camPos;           //  64
    float2   focal;            //  80
    float2   principal;        //  88
    float2   viewportPx;       //  96
    float    nearZ;            // 104
    float    farZ;             // 108
    uint     splatCount;       // 112
    uint     paddedCount;      // 116
    uint     shDegree;         // 120
    uint     shRestCount;      // 124
    float    alphaCutoff;      // 128
    float    scaleBoost;       // 132
    float    filterVariancePx; // 136
    float    pad0;             // 140
};                             // 144
static_assert(sizeof(ViewerUniforms) == 144, "ViewerUniforms must be 144 bytes");

struct ViewerCompositeUniforms {
    float4x4 invView;           //   0
    float4   camPos;            //  64
    float4   gridOrigin;        //  80
    float2   focal;             //  96
    float2   principal;         // 104
    float2   viewportPx;        // 112
    float    voxelSizeMeters;   // 120
    uint     cellCount;         // 124
    uint     flags;             // 128
    float    heatmapGain;       // 132
    float    overlapNormalizer; // 136
    float    hatchPitchPx;      // 140
};                              // 144
static_assert(sizeof(ViewerCompositeUniforms) == 144,
              "ViewerCompositeUniforms must be 144 bytes");

struct ViewerIndirectDrawArgs {
    uint vertexCount;
    uint instanceCount;
    uint vertexStart;
    uint baseInstance;
};

constant uint kViewerHonestyMaskFlag    = 1u << 0;
constant uint kViewerHeatmapFlag        = 1u << 1;
constant uint kViewerFieldLoadedFlag    = 1u << 2;

/// Sentinel key for "culled, or array padding". Sorts behind every real splat
/// because a positive float's bit pattern never reaches 0xFFFFFFFF.
constant uint kViewerSortSentinel = 0xFFFFFFFFu;

// =============================================================================
//  MARK: - Spherical harmonics
//
//  Constants and evaluation order are the INRIA / gsplat / SPZ convention,
//  which is what Sources/Export writes and reads, so a .ply round-trips
//  through this shader unchanged.
// =============================================================================

constant float kSH_C0 = 0.28209479177387814f;
constant float kSH_C1 = 0.4886025119029199f;
constant float kSH_C2[5] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f
};
constant float kSH_C3[7] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f
};

/// Evaluates the SH colour for one splat and one viewing direction.
///
/// `dir` points FROM the camera TOWARDS the splat, normalised - the same sense
/// the reference implementation uses, so imported clouds keep their shading.
/// The returned value is the raw SH sum; the caller adds the 0.5 offset and
/// clamps, exactly as the reference rasteriser does.
static float3 viewer_eval_sh(
    device const float *shRest,
    uint restBase,
    uint restCount,
    uint degree,
    float3 dc,
    float3 dir
) {
    float3 result = kSH_C0 * dc;
    if (degree == 0u || restCount == 0u || restBase == 0xFFFFFFFFu) {
        return result;
    }

    // Coefficient i lives at shRest[(restBase + i) * 3 + channel].
    const uint stride = 3u;

    float x = dir.x, y = dir.y, z = dir.z;

    float3 c[15];
    for (uint i = 0; i < 15u; ++i) { c[i] = float3(0.0f); }
    for (uint i = 0; i < restCount && i < 15u; ++i) {
        uint o = (restBase + i) * stride;
        c[i] = float3(shRest[o + 0], shRest[o + 1], shRest[o + 2]);
    }

    result += kSH_C1 * (-y * c[0] + z * c[1] - x * c[2]);

    if (degree >= 2u && restCount >= 8u) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        result +=
            kSH_C2[0] * xy * c[3] +
            kSH_C2[1] * yz * c[4] +
            kSH_C2[2] * (2.0f * zz - xx - yy) * c[5] +
            kSH_C2[3] * xz * c[6] +
            kSH_C2[4] * (xx - yy) * c[7];

        if (degree >= 3u && restCount >= 15u) {
            result +=
                kSH_C3[0] * y * (3.0f * xx - yy) * c[8] +
                kSH_C3[1] * xy * z * c[9] +
                kSH_C3[2] * y * (4.0f * zz - xx - yy) * c[10] +
                kSH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * c[11] +
                kSH_C3[4] * x * (4.0f * zz - xx - yy) * c[12] +
                kSH_C3[5] * z * (xx - yy) * c[13] +
                kSH_C3[6] * x * (xx - 3.0f * yy) * c[14];
        }
    }

    return result;
}

// =============================================================================
//  MARK: - Small helpers
// =============================================================================

static float3x3 viewer_quat_to_matrix(float4 q) {
    float4 n = normalize(q);
    float x = n.x, y = n.y, z = n.z, w = n.w;
    return float3x3(
        float3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y)),
        float3(2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x)),
        float3(2.0f * (x * z + w * y), 2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y))
    );
}

static float viewer_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

/// Octahedral direction bin, 8x8 = 64 bins.
///
/// MUST stay identical to `ObservedDirectionField.bin(for:)` in Swift - the
/// two are the reader and the writer of the same bitmask and there is no
/// shared header to keep them honest, so any edit here is an edit there.
static uint viewer_direction_bin(float3 dir) {
    float3 d = normalize(dir);
    float denom = max(abs(d.x) + abs(d.y) + abs(d.z), 1e-8f);
    float2 p = d.xy / denom;
    if (d.z < 0.0f) {
        float2 s = float2(p.x >= 0.0f ? 1.0f : -1.0f, p.y >= 0.0f ? 1.0f : -1.0f);
        p = (1.0f - abs(float2(p.y, p.x))) * s;
    }
    float2 uv = clamp(p * 0.5f + 0.5f, 0.0f, 0.999999f);
    uint bx = min(uint(uv.x * 8.0f), 7u);
    uint by = min(uint(uv.y * 8.0f), 7u);
    return by * 8u + bx;
}

/// Spreads 21 bits so they occupy every third bit. The inverse of this lives
/// in Swift as `ViewerMorton.decodePart`, and the pair defines the key layout
/// of `model/observed_directions.bin`.
static ulong viewer_morton_part(uint value) {
    ulong v = (ulong)(value & 0x1FFFFFu);
    v = (v | (v << 32)) & 0x1F00000000FFFFul;
    v = (v | (v << 16)) & 0x1F0000FF0000FFul;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Ful;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ul;
    v = (v | (v <<  2)) & 0x1249249249249249ul;
    return v;
}

static ulong viewer_morton_key(uint3 cell) {
    return viewer_morton_part(cell.x)
         | (viewer_morton_part(cell.y) << 1)
         | (viewer_morton_part(cell.z) << 2);
}

// =============================================================================
//  MARK: - 1. Preprocess
// =============================================================================

kernel void viewer_splat_preprocess(
    device const ViewerSplatBase *bases       [[buffer(0)]],
    device const float           *shRest      [[buffer(1)]],
    device ViewerSplatDraw       *draws       [[buffer(2)]],
    device ViewerSortEntry       *sortEntries [[buffer(3)]],
    device atomic_uint           *visible     [[buffer(4)]],
    constant ViewerUniforms      &u           [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= u.paddedCount) { return; }

    // Padding slots past the real splat count: sentinel, nothing else.
    if (gid >= u.splatCount) {
        sortEntries[gid].key = kViewerSortSentinel;
        sortEntries[gid].index = kViewerSortSentinel;
        return;
    }

    // Pessimistic default: this splat is culled unless proven otherwise.
    sortEntries[gid].key = kViewerSortSentinel;
    sortEntries[gid].index = gid;

    ViewerSplatBase b = bases[gid];
    float3 pWorld = float3(b.position);

    float4 pCam4 = u.viewMatrix * float4(pWorld, 1.0f);
    float3 pCam = pCam4.xyz;

    if (pCam.z < u.nearZ || pCam.z > u.farZ) { return; }

    float invZ = 1.0f / pCam.z;
    float2 meanPx = float2(
        u.focal.x * pCam.x * invZ + u.principal.x,
        u.focal.y * pCam.y * invZ + u.principal.y
    );

    // 3D covariance from rotation and log-scale. `b.logScale` already has the
    // Mip-Splatting 3D low-pass filter folded into it (see the note on the 2D
    // filter below), so this is the trainer's post-filter sigma, not its raw
    // optimiser parameter, and nothing is added to the diagonal here.
    float3 scale = exp(float3(b.logScale)) * u.scaleBoost;
    float3x3 R = viewer_quat_to_matrix(float4(b.rotation));
    float3x3 M = float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    float3x3 sigmaWorld = M * transpose(M);

    // Into camera space. W is the rotation block of the world -> camera matrix.
    float3x3 W = float3x3(u.viewMatrix[0].xyz, u.viewMatrix[1].xyz, u.viewMatrix[2].xyz);
    float3x3 sigmaCam = W * sigmaWorld * transpose(W);

    // Affine (Jacobian) approximation of the projection, with the standard
    // clamp on the off-axis ratio: without it the linearisation blows up at
    // the very edge of a wide preview FOV and splats streak off screen.
    float limX = 1.3f * (u.viewportPx.x * 0.5f) / u.focal.x;
    float limY = 1.3f * (u.viewportPx.y * 0.5f) / u.focal.y;
    float tx = clamp(pCam.x * invZ, -limX, limX) * pCam.z;
    float ty = clamp(pCam.y * invZ, -limY, limY) * pCam.z;

    float3 j0 = float3(u.focal.x * invZ, 0.0f, -u.focal.x * tx * invZ * invZ);
    float3 j1 = float3(0.0f, u.focal.y * invZ, -u.focal.y * ty * invZ * invZ);

    float3 s0 = sigmaCam * j0;
    float3 s1 = sigmaCam * j1;

    float cxx = dot(j0, s0);
    float cxy = dot(j0, s1);
    float cyy = dot(j1, s1);

    // Mip-Splatting 2D screen-space filter (F4), with the opacity compensation
    // that keeps total energy right. This replaces - it is NOT - the
    // unconditional 0.3 px dilation the INRIA reference rasteriser applies and
    // the spec says to remove. The 0.25 below is a variance and a band limit,
    // not that dilation, and the two numbers being close is a coincidence.
    //
    // This is the VIEW-DEPENDENT half of Mip-Splatting and the only half that
    // belongs here. `u.filterVariancePx` is 0.25, the same number the trainer
    // puts in `camera.filter2DVariance`, so from a given camera this kernel
    // computes the same `compensation` the trainer's `comp2D` had.
    //
    // The other half - the 3D filter, a per-Gaussian world-space width the
    // trainer widens the covariance by and then divides the opacity for - is
    // NOT applied here and must not be. It is view-INDEPENDENT and fixed per
    // splat, it lives only in the trainer's stats buffer, and no splat file
    // format has a field for it, so it is folded into the stored log-scale and
    // opacity logit before the cloud ever reaches this kernel
    // (`SplatCloud.fuse3DFilter`). `b.logScale` and `b.opacityLogit` are
    // therefore already the widened, compensated values. Applying the 3D
    // compensation again here would dim and blur every splat twice.
    float detBefore = max(cxx * cyy - cxy * cxy, 1e-12f);
    cxx += u.filterVariancePx;
    cyy += u.filterVariancePx;
    float detAfter = max(cxx * cyy - cxy * cxy, 1e-12f);
    float compensation = sqrt(max(detBefore / detAfter, 0.0f));

    // Eigen-decomposition of the symmetric 2x2 covariance.
    float mid = 0.5f * (cxx + cyy);
    float disc = sqrt(max(mid * mid - detAfter, 0.0f));
    float lambda1 = mid + disc;
    float lambda2 = max(mid - disc, 1e-8f);

    // Eigenvector for lambda1. Both off-diagonal forms are degenerate when the
    // covariance is already axis-aligned, hence the fallback.
    float2 e1;
    if (abs(cxy) > 1e-12f) {
        e1 = normalize(float2(lambda1 - cyy, cxy));
    } else {
        e1 = (cxx >= cyy) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }
    float2 e2 = float2(-e1.y, e1.x);

    float r1 = 3.0f * sqrt(lambda1);
    float r2 = 3.0f * sqrt(lambda2);

    // A splat smaller than a pixel and a splat larger than the screen are both
    // worth rejecting: the first contributes nothing, the second is almost
    // always a diverged Gaussian that would grey out the whole frame.
    float maxRadius = max(u.viewportPx.x, u.viewportPx.y);
    if (r1 < 0.25f || r1 > maxRadius) { return; }

    // Screen-space bound check.
    if (meanPx.x + r1 < 0.0f || meanPx.x - r1 > u.viewportPx.x ||
        meanPx.y + r1 < 0.0f || meanPx.y - r1 > u.viewportPx.y) {
        return;
    }

    float alpha = viewer_sigmoid(b.opacityLogit) * compensation;
    if (alpha < u.alphaCutoff) { return; }

    float3 dir = normalize(pWorld - u.camPos.xyz);
    float3 sh = viewer_eval_sh(
        shRest, b.shRestOffset, u.shRestCount, u.shDegree, float3(b.colorDC), dir
    );
    float3 rgb = max(sh + 0.5f, 0.0f);

    ViewerSplatDraw d;
    d.meanPx = meanPx;
    d.axis1Px = e1 * r1;
    d.axis2Px = e2 * r2;
    d.color = float4(rgb, min(alpha, 1.0f));
    d.depth = pCam.z;
    d.pad0 = 0.0f;
    draws[gid] = d;

    sortEntries[gid].key = as_type<uint>(pCam.z);  // monotonic for z > 0
    sortEntries[gid].index = gid;

    atomic_fetch_add_explicit(visible, 1u, memory_order_relaxed);
}

// =============================================================================
//  MARK: - 2. Bitonic sort
//
//  Ascending by key, i.e. front to back. Two kernels for one algorithm:
//  `viewer_sort_local` does every step whose partner stride fits inside a
//  threadgroup in ONE dispatch out of threadgroup memory; `viewer_sort_global`
//  does the wider strides, one dispatch each. For 2^19 entries that is about
//  64 dispatches instead of 190.
// =============================================================================

kernel void viewer_sort_global(
    device ViewerSortEntry *entries [[buffer(0)]],
    constant uint2         &kj      [[buffer(1)]],  // (k, j)
    uint i [[thread_position_in_grid]]
) {
    uint k = kj.x;
    uint j = kj.y;
    uint ixj = i ^ j;
    if (ixj <= i) { return; }

    ViewerSortEntry a = entries[i];
    ViewerSortEntry b = entries[ixj];
    bool ascending = ((i & k) == 0u);
    bool needsSwap = ascending ? (a.key > b.key) : (a.key < b.key);
    if (needsSwap) {
        entries[i] = b;
        entries[ixj] = a;
    }
}

kernel void viewer_sort_local(
    device ViewerSortEntry    *entries [[buffer(0)]],
    constant uint2            &kj      [[buffer(1)]],  // (k, jStart)
    threadgroup ViewerSortEntry *tile  [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    tile[lid] = entries[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = kj.x;
    for (uint j = kj.y; j > 0u; j >>= 1) {
        uint partner = lid ^ j;
        if (partner > lid) {
            // The direction test uses the GLOBAL index; the threadgroup covers
            // an aligned, contiguous block, so partner-in-tile is exact.
            bool ascending = ((gid & k) == 0u);
            ViewerSortEntry a = tile[lid];
            ViewerSortEntry b = tile[partner];
            bool needsSwap = ascending ? (a.key > b.key) : (a.key < b.key);
            if (needsSwap) {
                tile[lid] = b;
                tile[partner] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    entries[gid] = tile[lid];
}

// =============================================================================
//  MARK: - 3. Indirect draw arguments
// =============================================================================

kernel void viewer_prepare_indirect(
    device const uint            *visible [[buffer(0)]],
    device ViewerIndirectDrawArgs *args   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) { return; }
    args->vertexCount = 4u;          // one triangle strip quad
    args->instanceCount = visible[0];
    args->vertexStart = 0u;
    args->baseInstance = 0u;
}

// =============================================================================
//  MARK: - 4. Splat raster
// =============================================================================

struct ViewerRasterVSOut {
    float4 position [[position]];
    float2 quadUV;   // [-1, 1]^2 in the splat's own eigenbasis
    float4 color;
    float  depth;
};

vertex ViewerRasterVSOut viewer_splat_vertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    device const ViewerSortEntry *sorted [[buffer(0)]],
    device const ViewerSplatDraw *draws  [[buffer(1)]],
    constant ViewerUniforms      &u      [[buffer(2)]]
) {
    const float2 corners[4] = {
        float2(-1.0f, -1.0f), float2(1.0f, -1.0f),
        float2(-1.0f,  1.0f), float2(1.0f,  1.0f)
    };
    float2 corner = corners[vid];

    ViewerRasterVSOut out;
    uint index = sorted[iid].index;

    // A sentinel entry can only appear here if the visible counter and the
    // sorted array ever disagree. Collapse the quad to a point rather than
    // reading past the end of the draw buffer.
    if (sorted[iid].key == kViewerSortSentinel || index >= u.splatCount) {
        out.position = float4(0.0f, 0.0f, 2.0f, 1.0f);  // clipped
        out.quadUV = float2(0.0f);
        out.color = float4(0.0f);
        out.depth = 0.0f;
        return out;
    }

    ViewerSplatDraw d = draws[index];
    float2 px = float2(d.meanPx) + corner.x * float2(d.axis1Px) + corner.y * float2(d.axis2Px);

    // Pixel space (origin top-left, +Y down, matching the camera convention)
    // to clip space.
    float2 ndc = float2(
        (px.x / u.viewportPx.x) * 2.0f - 1.0f,
        1.0f - (px.y / u.viewportPx.y) * 2.0f
    );

    out.position = float4(ndc, 0.0f, 1.0f);
    out.quadUV = corner;
    out.color = float4(d.color);
    out.depth = d.depth;
    return out;
}

struct ViewerRasterFSOut {
    float4 color [[color(0)]];
    float4 aux   [[color(1)]];
};

/// Front-to-back compositing with programmable blending.
///
/// `aux` carries, per pixel:
///   r  sum of depth * weight      (divide by g for the alpha-weighted depth)
///   g  sum of weight              (= 1 - transmittance, the coverage alpha)
///   b  raw number of splats that actually contributed a fragment here
///   a  sum of weight for splats whose contribution was almost invisible,
///      which is the "lots of nearly-transparent haze" artefact signal
fragment ViewerRasterFSOut viewer_splat_fragment(
    ViewerRasterVSOut in [[stage_in]],
    float4 prevColor [[color(0)]],
    float4 prevAux   [[color(1)]],
    constant ViewerUniforms &u [[buffer(0)]]
) {
    ViewerRasterFSOut out;
    out.color = prevColor;
    out.aux = prevAux;

    float transmittance = 1.0f - prevColor.a;
    if (transmittance < 0.00392f) {         // 1/255: the pixel is opaque
        discard_fragment();
        return out;
    }

    float g = exp(-4.5f * dot(in.quadUV, in.quadUV));  // quad spans 3 sigma
    float alpha = in.color.a * g;
    if (alpha < u.alphaCutoff) {
        discard_fragment();
        return out;
    }
    alpha = min(alpha, 0.999f);

    float w = alpha * transmittance;

    out.color = float4(prevColor.rgb + in.color.rgb * w, prevColor.a + w);
    out.aux = float4(
        prevAux.r + in.depth * w,
        prevAux.g + w,
        prevAux.b + 1.0f,
        prevAux.a + (w < 0.02f ? w : 0.0f)
    );
    return out;
}

// =============================================================================
//  MARK: - 5. Composite: honesty mask and artefact heatmap
// =============================================================================

struct ViewerCompositeVSOut {
    float4 position [[position]];
    float2 uv;
};

vertex ViewerCompositeVSOut viewer_composite_vertex(uint vid [[vertex_id]]) {
    // One oversized triangle, no vertex buffer.
    const float2 pos[3] = {
        float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)
    };
    ViewerCompositeVSOut out;
    out.position = float4(pos[vid], 0.0f, 1.0f);
    out.uv = float2(
        (pos[vid].x + 1.0f) * 0.5f,
        1.0f - (pos[vid].y + 1.0f) * 0.5f    // flip: texture rows run downward
    );
    return out;
}

/// Binary search over the sorted `(mortonKey, directionMask)` records of
/// `model/observed_directions.bin`. Returns 0 when the cell is not present,
/// which means "nothing was ever recorded here" - the honest answer, and the
/// one that makes the mask hatch rather than silently pass.
static ulong viewer_lookup_direction_mask(
    device const ulong2 *cells,
    uint cellCount,
    ulong key
) {
    uint lo = 0u;
    uint hi = cellCount;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2u;
        ulong k = cells[mid].x;
        if (k < key) {
            lo = mid + 1u;
        } else if (k > key) {
            hi = mid;
        } else {
            return cells[mid].y;
        }
    }
    return 0ul;
}

static float3 viewer_heat_colour(float t) {
    // Cool blue -> amber -> red. Deliberately not a rainbow: a rainbow ramp is
    // not monotonic in luminance and reads as noise on a small screen.
    float3 cold = float3(0.16f, 0.42f, 0.85f);
    float3 mid  = float3(0.98f, 0.72f, 0.18f);
    float3 hot  = float3(0.90f, 0.16f, 0.16f);
    float s = clamp(t, 0.0f, 1.0f);
    return s < 0.5f ? mix(cold, mid, s * 2.0f) : mix(mid, hot, (s - 0.5f) * 2.0f);
}

fragment float4 viewer_composite_fragment(
    ViewerCompositeVSOut in [[stage_in]],
    texture2d<float, access::sample> colorTex [[texture(0)]],
    texture2d<float, access::sample> auxTex   [[texture(1)]],
    constant ViewerCompositeUniforms &u       [[buffer(0)]],
    device const ulong2 *cells                [[buffer(1)]]
) {
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);

    float4 acc = colorTex.sample(s, in.uv);
    float4 aux = auxTex.sample(s, in.uv);

    float coverage = clamp(acc.a, 0.0f, 1.0f);

    // The background behind the splats. A flat, slightly warm grey: a viewer
    // that paints unknown space black makes holes look like geometry.
    float3 background = float3(0.09f, 0.095f, 0.105f);
    float3 rgb = acc.rgb + background * (1.0f - coverage);

    bool wantHonesty = (u.flags & kViewerHonestyMaskFlag) != 0u;
    bool wantHeatmap = (u.flags & kViewerHeatmapFlag) != 0u;
    bool haveField   = (u.flags & kViewerFieldLoadedFlag) != 0u;

    if (!wantHonesty && !wantHeatmap) {
        return float4(rgb, 1.0f);
    }

    float weightSum = aux.g;
    float overlap = aux.b;
    float hazeWeight = aux.a;

    // Unobserved-ness of this pixel's viewing ray, 0 = seen from here, 1 = the
    // scene was never looked at from anything like this direction.
    float unobserved = 0.0f;
    float sparseDirections = 0.0f;

    if (haveField && u.cellCount > 0u && weightSum > 1e-4f && coverage > 0.02f) {
        float depth = aux.r / weightSum;
        float2 px = in.uv * u.viewportPx;
        float3 rayCam = float3(
            (px.x - u.principal.x) / u.focal.x,
            (px.y - u.principal.y) / u.focal.y,
            1.0f
        );
        float3 pCam = rayCam * depth;
        float3 pWorld = (u.invView * float4(pCam, 1.0f)).xyz;

        float3 rel = (pWorld - u.gridOrigin.xyz) / u.voxelSizeMeters;
        if (rel.x >= 0.0f && rel.y >= 0.0f && rel.z >= 0.0f) {
            uint3 cell = uint3(
                min(uint(rel.x), 0x1FFFFFu),
                min(uint(rel.y), 0x1FFFFFu),
                min(uint(rel.z), 0x1FFFFFu)
            );
            ulong key = viewer_morton_key(cell);
            ulong mask = viewer_lookup_direction_mask(cells, u.cellCount, key);

            float3 toCamera = u.camPos.xyz - pWorld;
            uint bin = viewer_direction_bin(toCamera);
            bool seen = ((mask >> bin) & 1ul) != 0ul;
            unobserved = seen ? 0.0f : 1.0f;

            uint observedCount = popcount(mask);
            sparseDirections = 1.0f - clamp(float(observedCount) / 16.0f, 0.0f, 1.0f);
        } else {
            unobserved = 1.0f;
            sparseDirections = 1.0f;
        }
    }

    if (wantHonesty && unobserved > 0.5f) {
        // 45-degree hatching, in SCREEN pixels so it does not crawl with the
        // geometry - it is an annotation, not a texture. The pitch comes from
        // the renderer, which sizes it from the screen's scale factor: a fixed
        // 8 device pixels is under three points on a 3x display, which
        // shimmers rather than reads.
        //
        // This is deliberately loud. It is the one mark on screen that says
        // "this part is invented", and a subtle version of that message is
        // worse than none: the user would see it, not register it, and come
        // away trusting a surface nothing ever looked at.
        float2 px = in.uv * u.viewportPx;
        float pitch = max(u.hatchPitchPx, 2.0f);
        float stripe = fract((px.x + px.y) / pitch);
        float hatch = smoothstep(0.40f, 0.48f, stripe) * (1.0f - smoothstep(0.92f, 1.0f, stripe));
        float3 hatchColour = float3(0.98f, 0.87f, 0.28f);
        // Drain the colour underneath, so the stripes read over a bright wall
        // and the hatched region is obviously a different KIND of thing from
        // the parts of the picture that are real.
        float luma = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
        float3 drained = mix(rgb, float3(luma * 0.72f), 0.70f);
        rgb = mix(drained, hatchColour, hatch * 0.72f);
    }

    if (wantHeatmap) {
        // Four honest, independent artefact signals:
        //  - coverage deficit: the model never filled this pixel in
        //  - overlap: many Gaussians stacked on one pixel, the signature of
        //    popping and of the "cloud of fog" failure
        //  - haze: most of that overlap arrived as near-invisible slivers
        //  - unobserved / sparse directions: it is guesswork here
        float deficit = 1.0f - coverage;
        float overlapTerm = clamp(overlap * u.overlapNormalizer, 0.0f, 1.0f);
        float hazeTerm = clamp(hazeWeight * 4.0f, 0.0f, 1.0f);
        float score = clamp(
            0.30f * deficit +
            0.30f * overlapTerm +
            0.15f * hazeTerm +
            0.25f * max(unobserved, sparseDirections),
            0.0f, 1.0f
        );
        rgb = mix(rgb, viewer_heat_colour(score), score * u.heatmapGain);
    }

    return float4(rgb, 1.0f);
}
