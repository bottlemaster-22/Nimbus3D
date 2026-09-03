//
//  SplatMeshExtractor.swift — Mesh module (REAL pipeline, one documented frontier gap)
//
//  Concrete `MeshExtractor` (Sources/Core/Contracts.swift). Pipeline:
//
//    PLY splat parse  ->  Gaussian density field (Metal or CPU)  ->  marching-cubes
//    iso-surface  ->  outward winding + normals  ->  QEM decimation  ->  box-projection
//    UV atlas  ->  MeshAsset
//
//  Every stage above is REAL, on-device, and deterministic on the CPU path.
//
//  HONESTY / FRONTIER NOTE (per the owner's no-vibecoding contract):
//  Marching cubes over a Gaussian *density envelope* is a legitimate surfacing method,
//  but it is NOT surface-aligned reconstruction. 3D Gaussians are volumetric blobs, so
//  the iso-surface is watertight-ish yet "puffy" and can bridge thin gaps or fuzz sharp
//  edges. The state-of-the-art clean-mesh methods that fix this — 2DGS, Gaussian Opacity
//  Fields (GOF), MILo — all rely on CUDA-only differentiable tetrahedra / TSDF-from-render
//  passes with no shipping on-device (Metal/iOS) implementation as of this writing. When
//  one exists, it should replace DensityFieldBuilder + MarchingCubes behind this exact
//  protocol; nothing downstream (decimation, UV, export) needs to change. This class is
//  therefore honestly PARTIAL: real and useful, not the frontier-quality result.
//

import Foundation
import simd
import Metal

/// Tunables for the density-field surfacing. Immutable so the extractor stays Sendable.
public struct MeshExtractorConfig: Sendable {
    /// Longest-axis voxel resolution used when `MeshExtractionOptions.lowPoly` is true.
    public var lowPolyGridResolution: Int
    /// Longest-axis voxel resolution for full-detail extraction.
    public var fullGridResolution: Int
    /// Iso-level as a fraction of the field's peak density.
    public var isoFraction: Float
    /// Lower fraction retried once if the first iso-level yields an empty surface.
    public var fallbackIsoFraction: Float

    public init(lowPolyGridResolution: Int = 192,
                fullGridResolution: Int = 256,
                isoFraction: Float = 0.10,
                fallbackIsoFraction: Float = 0.04) {
        self.lowPolyGridResolution = lowPolyGridResolution
        self.fullGridResolution = fullGridResolution
        self.isoFraction = isoFraction
        self.fallbackIsoFraction = fallbackIsoFraction
    }
}

public final class SplatMeshExtractor: MeshExtractor {

    private let config: MeshExtractorConfig

    public init(config: MeshExtractorConfig = MeshExtractorConfig()) {
        self.config = config
    }

    public func extractMesh(from model: SplatModel,
                            options: MeshExtractionOptions,
                            progress: @escaping ProgressHandler) async throws -> MeshAsset {

        func report(_ fraction: Double, _ message: String) {
            progress(PipelineProgress(stage: .meshExtraction,
                                      fractionCompleted: min(max(fraction, 0), 1),
                                      message: message))
        }
        func checkCancel() throws {
            if Task.isCancelled { throw NimbusError.cancelled }
        }

        // --- 0. Format gate. Trainer emits PLY natively; SPZ decode is not wired here. ---
        guard model.format == .ply else {
            // TODO(nimbus): decode Niantic .spz (gzip + fixed-point quantised splat block,
            // spec: github.com/nianticlabs/spz) into [GaussianSplat] and feed the same pipeline.
            throw NimbusError.notImplemented(
                "MeshExtractor currently reads .ply splats only; .spz decoding is not implemented.")
        }

        // --- 1. Parse splats. ---
        report(0.02, "Reading Gaussian splats")
        let splats: [GaussianSplat]
        do {
            splats = try SplatPLYReader.read(contentsOf: model.splatFileURL)
        } catch {
            throw NimbusError.meshExtractionFailed("PLY parse failed: \(error)")
        }
        guard !splats.isEmpty else {
            throw NimbusError.meshExtractionFailed("Splat file contained no vertices.")
        }
        try checkCancel()

        // --- 2. Bounds: derive from splat extents (union with the model's declared box). ---
        let bounds = computeBounds(splats: splats, declared: model.boundingBox)

        // --- 3. Density field. ---
        report(0.12, "Building density field (\(splats.count) splats)")
        let device = MTLCreateSystemDefaultDevice()
        let resolution = options.lowPoly ? config.lowPolyGridResolution : config.fullGridResolution
        guard let field = DensityFieldBuilder.build(splats: splats,
                                                     bounds: bounds,
                                                     longestAxisResolution: resolution,
                                                     device: device) else {
            throw NimbusError.meshExtractionFailed("Failed to build density field (degenerate bounds).")
        }
        guard field.maxValue > 0 else {
            throw NimbusError.meshExtractionFailed("Density field is empty (all splats near-transparent).")
        }
        try checkCancel()

        // --- 4. Marching cubes (retry once at a lower iso-level if empty). ---
        report(0.45, "Extracting surface (marching cubes)")
        var surface = MarchingCubes.surface(from: field,
                                            isoLevel: field.maxValue * config.isoFraction,
                                            isCancelled: { Task.isCancelled })
        try checkCancel()
        if surface.indices.isEmpty {
            surface = MarchingCubes.surface(from: field,
                                            isoLevel: field.maxValue * config.fallbackIsoFraction,
                                            isCancelled: { Task.isCancelled })
        }
        guard !surface.indices.isEmpty else {
            throw NimbusError.meshExtractionFailed("Iso-surface extraction produced no triangles.")
        }
        try checkCancel()

        // --- 5. Consistent outward winding (uses the field's gradient). ---
        report(0.62, "Orienting surface")
        MeshGeometry.orientWindingOutward(positions: surface.positions,
                                          indices: &surface.indices,
                                          field: field)

        var positions = surface.positions
        var indices = surface.indices

        // --- 6. Decimation (QEM) toward the triangle budget. ---
        if options.targetTriangleCount > 0 && indices.count / 3 > options.targetTriangleCount {
            report(0.70, "Decimating to \(options.targetTriangleCount) triangles")
            let decimated = MeshDecimator.decimate(positions: positions,
                                                   indices: indices,
                                                   targetTriangleCount: options.targetTriangleCount,
                                                   isCancelled: { Task.isCancelled })
            positions = decimated.positions
            indices = decimated.indices
        }
        try checkCancel()

        // --- 7. UV unwrap + normals. ---
        let meshPositions: [SIMD3<Float>]
        let meshNormals: [SIMD3<Float>]
        let meshUVs: [SIMD2<Float>]
        let meshIndices: [UInt32]

        if options.unwrapUVs {
            report(0.88, "Unwrapping UV atlas")
            let preNormals = MeshGeometry.computeNormals(positions: positions, indices: indices)
            let unwrapped = UVUnwrapper.unwrap(positions: positions,
                                               normals: preNormals,
                                               indices: indices)
            // Recompute normals on the split mesh so shading stays smooth across the copies.
            meshPositions = unwrapped.positions
            meshIndices = unwrapped.indices
            meshNormals = MeshGeometry.computeNormals(positions: unwrapped.positions, indices: unwrapped.indices)
            meshUVs = unwrapped.uvs
        } else {
            report(0.92, "Computing normals")
            meshPositions = positions
            meshIndices = indices
            meshNormals = MeshGeometry.computeNormals(positions: positions, indices: indices)
            meshUVs = []
        }
        try checkCancel()

        report(1.0, "Mesh ready: \(meshPositions.count) verts, \(meshIndices.count / 3) tris")
        return MeshAsset(positions: meshPositions,
                         normals: meshNormals,
                         uvs: meshUVs,
                         indices: meshIndices,
                         isLowPoly: options.lowPoly,
                         sourceSplatID: model.id)
    }

    // MARK: - Bounds

    private func computeBounds(splats: [GaussianSplat],
                              declared: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox {
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in splats {
            let r = SIMD3<Float>(repeating: s.radius)
            mn = simd_min(mn, s.position - r)
            mx = simd_max(mx, s.position + r)
        }
        // Union with the trainer's declared box when it is finite and non-degenerate.
        let d = declared
        if d.minCorner.x.isFinite, d.maxCorner.x.isFinite, all(d.maxCorner .> d.minCorner) {
            mn = simd_min(mn, d.minCorner)
            mx = simd_max(mx, d.maxCorner)
        }
        return AxisAlignedBoundingBox(minCorner: mn, maxCorner: mx)
    }
}
