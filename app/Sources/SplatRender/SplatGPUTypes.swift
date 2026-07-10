//
//  SplatGPUTypes.swift — SplatRender module.
//
//  Byte-exact mirrors of the structs declared in the Metal shaders
//  (Shaders/SplatRasterize.metal, Shaders/SplatSort.metal). These are passed to
//  the GPU via MTLBuffer / setBytes, so their memory layout MUST match the
//  Metal side field-for-field. Do not reorder fields or change types.
//
//  Layout notes:
//   - GPUSplat is 13 x Float = 52 bytes, alignment 4. On the Metal side it is
//     three `packed_float3` + one `packed_float4` (also 52 bytes, alignment 4).
//     We deliberately use loose Floats here instead of SIMD3<Float>, because
//     SIMD3<Float> is 16-byte aligned (occupies 16 bytes) and would NOT match
//     Metal's packed_float3 (12 bytes).
//   - The uniform/param structs use simd_float4x4 (C-compatible, column-major,
//     64 bytes) which matches Metal's float4x4 exactly.
//

import Foundation
import simd

/// One Gaussian, pre-processed into GPU-ready form on load.
/// `cov3dA`/`cov3dB` hold the 6 unique entries of the symmetric 3D covariance
/// Σ = R S Sᵀ Rᵀ (world space): A = (Σ00, Σ01, Σ02), B = (Σ11, Σ12, Σ22).
/// `color` is linear RGB (from the SH DC term) with premultiplied opacity in .a
/// applied per-fragment, opacity stored plain in .a here.
struct GPUSplat: Sendable {
    var px: Float, py: Float, pz: Float          // position (world)
    var a0: Float, a1: Float, a2: Float          // cov3d A (Σ00, Σ01, Σ02)
    var b0: Float, b1: Float, b2: Float          // cov3d B (Σ11, Σ12, Σ22)
    var cr: Float, cg: Float, cb: Float, ca: Float // color rgb + opacity (0..1)
}

/// Sort payload: one per splat (plus power-of-two padding). Key is view-space z
/// (negative in front for a right-handed camera looking down -Z), so an ascending
/// sort yields a far-to-near (back-to-front) draw order for correct "over" blending.
struct SortEntry {
    var key: Float
    var index: UInt32
}

/// Per-frame uniforms for the rasterization pass.
struct SplatUniforms {
    var view: simd_float4x4
    var projection: simd_float4x4
    var viewport: SIMD2<Float>   // pixels (width, height)
    var focal: SIMD2<Float>      // (fx, fy) in pixels, derived from projection + viewport
}

/// Params for the depth-key compute kernel.
struct KeyParams {
    var view: simd_float4x4
    var splatCount: UInt32
    var paddedCount: UInt32
}

/// Params for one bitonic-sort step.
struct SortParams {
    var k: UInt32
    var j: UInt32
    var paddedCount: UInt32
    var _pad: UInt32 = 0
}
