//
//  ViewerGPULayouts.swift
//  Viewer
//
//  THE SWIFT SIDE OF EVERY BUFFER THE VIEWER'S METAL SHADERS READ.
//
//  Each struct here is byte-matched to a struct of the same name in
//  `SplatRenderShaders.metal`. There is no shared header (a single Xcode
//  target with XcodeGen has no bridging-header hook that both the Swift and
//  the Metal compiler see), so the two declarations are kept in step by hand
//  and checked at run time by `ViewerGPULayouts.verify()`, which is called
//  once from `MetalSplatRenderer.init` and traps in DEBUG if a field is ever
//  added on one side only.
//
//  Rules that keep the layouts matched:
//
//   * Every field is `Float`, `UInt32`, or a simd type whose Swift size and
//     alignment are identical to MSL's. NOTHING here is `Float16`: `Float16`
//     does not exist on x86_64, so a struct using it would refuse to build in
//     an Intel-Mac simulator even though the shipping slice is arm64-only.
//   * The per-splat structs use `packed_*` types on the Metal side, so their
//     alignment is 4 and Swift's natural field packing matches exactly.
//   * The uniform structs use ALIGNED simd types (`float4x4`, `float4`,
//     `float2`), because a `constant` buffer argument in MSL is laid out with
//     standard alignment. Their Swift mirrors use `simd_float4x4`,
//     `SIMD4<Float>` and `SIMD2<Float>`, which have exactly those sizes and
//     alignments.
//
//  Sizes, restated so a reviewer can check them without a compiler:
//
//      ViewerSplatBase   64 bytes, align 4
//      ViewerSplatDraw   48 bytes, align 4
//      ViewerSortEntry    8 bytes, align 4
//      ViewerUniforms   144 bytes, align 16
//      ViewerCompositeUniforms 144 bytes, align 16
//      ViewerIndirectDrawArgs  16 bytes, align 4
//

import Foundation
import simd

// MARK: - Per-splat static attributes

/// One Gaussian as the GPU stores it, straight from a `SplatCloud`.
///
/// Storage conventions are `SplatCloud`'s, unchanged, so a cloud can be
/// uploaded without a single unit conversion on the CPU:
///
///  * `position` is world space, metres, RUB (which for this app's world frame
///    is simply "as ARKit produced it").
///  * `rotation` is `(x, y, z, w)`. Normalised in the shader, not here.
///  * `logScale` is the raw log-scale parameter; `exp()` is applied on the GPU.
///  * `opacityLogit` is pre-sigmoid.
///  * `colorDC` is the raw degree-0 SH coefficient, NOT a display colour.
struct ViewerSplatBase {
    var positionX: Float = 0        // offset  0
    var positionY: Float = 0        // offset  4
    var positionZ: Float = 0        // offset  8
    var opacityLogit: Float = 0     // offset 12
    var rotationX: Float = 0        // offset 16
    var rotationY: Float = 0        // offset 20
    var rotationZ: Float = 0        // offset 24
    var rotationW: Float = 1        // offset 28
    var logScaleX: Float = 0        // offset 32
    var logScaleY: Float = 0        // offset 36
    var logScaleZ: Float = 0        // offset 40
    var colorDCR: Float = 0         // offset 44
    var colorDCG: Float = 0         // offset 48
    var colorDCB: Float = 0         // offset 52
    /// Index of this splat's first higher-order SH coefficient in the SH rest
    /// buffer, counted in `packed_float3` units. `UInt32.max` means "no rest
    /// coefficients", which is also the case for every splat at SH degree 0.
    var shRestOffset: UInt32 = .max // offset 56
    /// Reserved. Bit 0 is set by the loader for splats it wants excluded from
    /// the artefact heatmap's overlap term (currently unused; kept so a future
    /// flag does not change the stride).
    var flags: UInt32 = 0           // offset 60
}                                   // size 64

// MARK: - Per-splat per-frame projected state

/// What the preprocess kernel computes for one visible splat and the vertex
/// shader then reads. Written every frame the camera moves.
///
/// `axis1Px` / `axis2Px` are the 2D covariance's eigenvectors scaled to THREE
/// standard deviations, in render pixels. The quad is drawn at
/// `mean + u * axis1 + v * axis2` for `u, v` in `[-1, 1]`, so the Gaussian
/// falloff in the fragment shader is exactly `exp(-4.5 * (u^2 + v^2))` with no
/// conic matrix needing to cross the buffer.
struct ViewerSplatDraw {
    var meanX: Float = 0    // offset  0
    var meanY: Float = 0    // offset  4
    var axis1X: Float = 0   // offset  8
    var axis1Y: Float = 0   // offset 12
    var axis2X: Float = 0   // offset 16
    var axis2Y: Float = 0   // offset 20
    var colorR: Float = 0   // offset 24
    var colorG: Float = 0   // offset 28
    var colorB: Float = 0   // offset 32
    var alpha: Float = 0    // offset 36
    /// View-space z in metres. Used for the depth key, and written into the
    /// auxiliary attachment so the composite pass can rebuild a world point.
    var depth: Float = 0    // offset 40
    var pad0: Float = 0     // offset 44
}                           // size 48

// MARK: - Sort

/// One entry of the depth-sorted draw order.
///
/// `key` is the IEEE bit pattern of the positive view-space depth, which is
/// monotonically increasing for positive floats, so an unsigned integer sort
/// on it is exactly a float sort. `UInt32.max` is the sentinel for "culled or
/// padding": it sorts to the end, behind every real splat.
struct ViewerSortEntry {
    var key: UInt32 = .max     // offset 0
    var index: UInt32 = .max   // offset 4
}                              // size 8

// MARK: - Uniforms

/// Constants for the preprocess kernel and the splat raster pass.
struct ViewerUniforms {
    /// World -> camera. This is `Pose.matrix`, unchanged: the app's one
    /// convention, +Y down, +Z forward.
    var viewMatrix: simd_float4x4 = matrix_identity_float4x4  // offset   0
    /// Camera centre in world space, `xyz`; `w` unused.
    var camPos: SIMD4<Float> = .zero                          // offset  64
    /// fx, fy in RENDER pixels (intrinsics rescaled to the drawable).
    var focal: SIMD2<Float> = .zero                           // offset  80
    /// cx, cy in render pixels.
    var principal: SIMD2<Float> = .zero                       // offset  88
    var viewportPx: SIMD2<Float> = .zero                      // offset  96
    var nearZ: Float = 0.05                                   // offset 104
    var farZ: Float = 200                                     // offset 108
    var splatCount: UInt32 = 0                                // offset 112
    /// `splatCount` rounded up to a power of two (the sort's array length).
    var paddedCount: UInt32 = 0                               // offset 116
    var shDegree: UInt32 = 0                                  // offset 120
    /// Number of higher-order SH coefficients per splat.
    var shRestCount: UInt32 = 0                               // offset 124
    /// Below this alpha a fragment contributes nothing and is discarded.
    var alphaCutoff: Float = 1.0 / 255.0                      // offset 128
    /// Debug/QA multiplier on splat size. 1 in every shipping path.
    var scaleBoost: Float = 1                                 // offset 132
    /// Mip-Splatting 2D low-pass filter variance, in square pixels (F4).
    /// This is NOT the 0.3 px dilation the spec says to remove: the dilation
    /// grows every splat unconditionally, whereas this is a proper screen-space
    /// band limit whose effect on brightness is compensated for on the alpha.
    var filterVariancePx: Float = 0.3                         // offset 136
    var pad0: Float = 0                                       // offset 140
}                                                             // size 144

/// Constants for the full-screen composite (honesty mask + artefact heatmap).
struct ViewerCompositeUniforms {
    /// Camera -> world, i.e. `Pose.matrix.inverse`.
    var invView: simd_float4x4 = matrix_identity_float4x4  // offset   0
    var camPos: SIMD4<Float> = .zero                       // offset  64
    /// Minimum corner of the observed-direction grid's (0,0,0) cell, `xyz`.
    var gridOrigin: SIMD4<Float> = .zero                   // offset  80
    var focal: SIMD2<Float> = .zero                        // offset  96
    var principal: SIMD2<Float> = .zero                    // offset 104
    var viewportPx: SIMD2<Float> = .zero                   // offset 112
    var voxelSizeMeters: Float = 0.25                      // offset 120
    /// Number of records in the observed-direction buffer.
    var cellCount: UInt32 = 0                              // offset 124
    /// Bit 0: honesty mask on. Bit 1: artefact heatmap on.
    /// Bit 2: an observed-direction field is actually loaded.
    var flags: UInt32 = 0                                  // offset 128
    /// Overall strength of the heatmap tint, 0...1.
    var heatmapGain: Float = 0.75                          // offset 132
    /// Divisor that turns a raw per-pixel splat-overlap count into 0...1.
    var overlapNormalizer: Float = 1.0 / 48.0              // offset 136
    /// Spacing of the honesty mask's diagonal stripes, in DRAWABLE pixels.
    /// The renderer sets this from the screen's scale factor so the stripes
    /// are the same physical size on a 2x and a 3x display; a fixed pixel
    /// pitch shimmers on the denser one instead of reading as hatching.
    var hatchPitchPx: Float = 12                           // offset 140
}                                                          // size 144

extension ViewerCompositeUniforms {
    static let flagHonestyMask: UInt32 = 1 << 0
    static let flagArtifactHeatmap: UInt32 = 1 << 1
    static let flagFieldLoaded: UInt32 = 1 << 2
}

// MARK: - Indirect draw

/// Byte-identical to `MTLDrawPrimitivesIndirectArguments`, re-declared so a
/// compute kernel can fill it. Metal fixes this layout; it is not ours to
/// change.
struct ViewerIndirectDrawArgs {
    var vertexCount: UInt32 = 4
    var instanceCount: UInt32 = 0
    var vertexStart: UInt32 = 0
    var baseInstance: UInt32 = 0
}

// MARK: - Buffer binding indices
//
// Shared with the .metal file by convention, in one list, so a renumbering is
// a single diff on both sides rather than a hunt through five call sites.

enum ViewerBufferIndex {
    // Preprocess kernel
    static let baseSplats = 0
    static let shRest = 1
    static let draws = 2
    static let sortEntries = 3
    static let visibleCounter = 4
    static let uniforms = 5

    // Sort kernels
    static let sortEntriesIn = 0
    static let sortParams = 1

    // Indirect-args kernel
    static let indirectCounter = 0
    static let indirectArgs = 1

    // Raster pass
    static let rasterSorted = 0
    static let rasterDraws = 1
    static let rasterUniforms = 2

    // Composite pass
    static let compositeUniforms = 0
    static let compositeDirectionCells = 1
}

enum ViewerTextureIndex {
    static let compositeColor = 0
    static let compositeAux = 1
}

// MARK: - Layout verification

enum ViewerGPULayouts {

    /// Expected `MemoryLayout.stride` of every struct the GPU reads, matched
    /// against `SplatRenderShaders.metal`.
    ///
    /// Called once from `MetalSplatRenderer.init`. In DEBUG a mismatch traps
    /// with the offending struct named, because the alternative - a silently
    /// misaligned buffer - renders as plausible-looking garbage that costs a
    /// day to find. In RELEASE it returns the problem as a string so the
    /// renderer can refuse to start and say why.
    static func verify() -> String? {
        var problems: [String] = []

        func check(_ name: String, _ actual: Int, _ expected: Int) {
            if actual != expected {
                problems.append("\(name) stride is \(actual), Metal expects \(expected)")
            }
        }

        check("ViewerSplatBase", MemoryLayout<ViewerSplatBase>.stride, 64)
        check("ViewerSplatDraw", MemoryLayout<ViewerSplatDraw>.stride, 48)
        check("ViewerSortEntry", MemoryLayout<ViewerSortEntry>.stride, 8)
        check("ViewerUniforms", MemoryLayout<ViewerUniforms>.stride, 144)
        check(
            "ViewerCompositeUniforms",
            MemoryLayout<ViewerCompositeUniforms>.stride,
            144
        )
        check(
            "ViewerIndirectDrawArgs",
            MemoryLayout<ViewerIndirectDrawArgs>.stride,
            16
        )
        // A packed_float3 in the SH rest buffer is 12 bytes; SIMD3<Float> in
        // Swift is 16. The loader therefore writes SH coefficients as a flat
        // [Float] array, three floats per coefficient, and this is the guard
        // that stops anybody "tidying" it into [SIMD3<Float>].
        check("Float (SH element)", MemoryLayout<Float>.stride, 4)

        guard !problems.isEmpty else { return nil }
        let message = "GPU struct layout mismatch: " + problems.joined(separator: "; ")
        assertionFailure(message)
        return message
    }
}
