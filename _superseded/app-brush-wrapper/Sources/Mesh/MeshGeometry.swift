//
//  MeshGeometry.swift — Mesh module (REAL)
//
//  Small, dependency-free geometry utilities used by the extractor: area-weighted
//  per-vertex normals and density-gradient-based winding correction so triangle
//  fronts face outward (glTF expects CCW front faces).
//

import Foundation
import simd

extension DensityField {
    /// Central-difference gradient of the field at a world-space point (nearest-voxel).
    /// Points toward increasing density, i.e. INTO the solid.
    func gradientWorld(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let local = (p - origin) / voxelSize - SIMD3<Float>(repeating: 0.5)
        let ix = Int(local.x.rounded()), iy = Int(local.y.rounded()), iz = Int(local.z.rounded())
        let x = min(max(ix, 1), dims.x - 2)
        let y = min(max(iy, 1), dims.y - 2)
        let z = min(max(iz, 1), dims.z - 2)
        let gx = value(x + 1, y, z) - value(x - 1, y, z)
        let gy = value(x, y + 1, z) - value(x, y - 1, z)
        let gz = value(x, y, z + 1) - value(x, y, z - 1)
        return SIMD3(gx, gy, gz)
    }
}

enum MeshGeometry {

    /// Area-weighted per-vertex normals; result count == positions count.
    static func computeNormals(positions: [SIMD3<Float>], indices: [UInt32]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var i = 0
        while i + 2 < indices.count {
            let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
            let faceN = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            normals[a] += faceN   // magnitude == 2*area, so this is area-weighted
            normals[b] += faceN
            normals[c] += faceN
            i += 3
        }
        for j in 0..<normals.count {
            let len = simd_length(normals[j])
            normals[j] = len > 1e-9 ? normals[j] / len : SIMD3(0, 1, 0)
        }
        return normals
    }

    /// Reorder each triangle's winding so its geometric normal opposes the density
    /// gradient (outward). Mutates `indices` in place.
    static func orientWindingOutward(positions: [SIMD3<Float>],
                                     indices: inout [UInt32],
                                     field: DensityField) {
        var i = 0
        while i + 2 < indices.count {
            let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
            let faceN = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            let centroid = (positions[a] + positions[b] + positions[c]) / 3
            // Outward = away from higher density = negative gradient.
            let outward = -field.gradientWorld(centroid)
            if simd_dot(faceN, outward) < 0 {
                indices[i + 1] = UInt32(c)
                indices[i + 2] = UInt32(b)
            }
            i += 3
        }
    }
}
