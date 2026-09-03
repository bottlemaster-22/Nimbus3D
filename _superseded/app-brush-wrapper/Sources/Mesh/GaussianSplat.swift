//
//  GaussianSplat.swift — Mesh module (REAL)
//
//  In-memory representation of a single 3D Gaussian used by the mesh extractor's
//  density field. This is a *reduced* view of a full 3DGS primitive: we keep only
//  what the density-field surfacing needs (position, world-space extent, opacity).
//  View-dependent SH colour is intentionally dropped — meshing works on geometry,
//  and per-vertex colour is baked later by the Materials module from the source
//  images, not from the splats.
//

import Foundation
import simd

/// A minimal Gaussian primitive for density accumulation.
struct GaussianSplat: Sendable {
    /// World-space centre (ARKit metres).
    var position: SIMD3<Float>
    /// World-space standard deviation along each principal axis (already exponentiated
    /// out of the PLY's log-scale storage).
    var scale: SIMD3<Float>
    /// Linear opacity 0...1 (already passed through sigmoid out of the PLY's logit storage).
    var opacity: Float

    /// Isotropic support radius used by the density kernel (mean of the three axes,
    /// which is a deliberate simplification — see DensityField for why anisotropy is
    /// dropped at accumulation time).
    var radius: Float { max((scale.x + scale.y + scale.z) / 3.0, 1e-5) }
}

/// A packed, GPU-friendly splat matching the `SplatPoint` struct in SplatDensity.metal.
/// Layout MUST stay in lockstep with the shader.
struct PackedSplat {
    var position: SIMD3<Float>
    var invRadius: Float   // 1 / radius, precomputed
    var weight: Float      // opacity
    // 12 bytes padding to keep 16-byte alignment parity with the MSL struct.
    var _pad0: Float = 0
    var _pad1: Float = 0
    var _pad2: Float = 0
}
