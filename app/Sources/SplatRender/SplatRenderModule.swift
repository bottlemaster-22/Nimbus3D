//
//  SplatRenderModule.swift — module anchor. Owned by the SplatRender agent.
//
//  This module implements `SplatRenderer` (Sources/Core/Contracts.swift) in
//  Metal: splat loading into GPU buffers, depth sorting (GPU radix or CPU),
//  and rasterization. Put .metal shaders under Sources/SplatRender/Shaders/;
//  they are compiled into default.metallib automatically by the app target.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the SplatRender module.
///
/// Implemented here:
///   - `MetalSplatRenderer` (MetalSplatRenderer.swift) — the `SplatRenderer`
///     contract in Metal: PLY -> GPU buffers, GPU bitonic depth sort, instanced
///     EWA rasterization with premultiplied "over" blending.
///   - `SplatPreviewView` (SplatPreviewView.swift) — SwiftUI orbit/zoom preview
///     driving the renderer through an MTKView.
///   - `SplatPLYParser` (SplatPLYParser.swift) — real 3DGS PLY decoder.
///   - Shaders/SplatRasterize.metal, Shaders/SplatSort.metal.
public enum SplatRenderModule {
    public static let name = "SplatRender"
}
