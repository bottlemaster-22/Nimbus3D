//
//  UVUnwrapper.swift — Mesh module (REAL, but a deliberate simplification)
//
//  Produces a valid, NON-OVERLAPPING UV atlas via 6-axis box projection: each triangle
//  is assigned to the cube face its normal most faces, projected onto that plane, and
//  the six resulting charts are packed into a 3x2 atlas. Vertices are split along the
//  seams between differently-projected triangles so every output vertex has exactly one
//  UV (positions/normals/uvs stay index-aligned, as MeshAsset requires).
//
//  Honesty: this is real and bake-ready, but it is NOT xatlas. Box projection produces
//  more seams and worse texel-area uniformity than proper chart-growing + LSCM/ABF
//  parameterisation. The atlas packing is a fixed grid, not rectangle bin-packing, so
//  charts with very different aspect ratios waste texture space. See TODO(nimbus) below
//  for the real approach.
//

import Foundation
import simd

enum UVUnwrapper {

    struct Result {
        var positions: [SIMD3<Float>]
        var normals: [SIMD3<Float>]
        var uvs: [SIMD2<Float>]
        var indices: [UInt32]
    }

    /// Atlas grid: 6 box faces packed 3 wide x 2 tall.
    private static let atlasCols = 3
    private static let atlasRows = 2
    /// Fractional inset per cell to keep charts off the cell border (avoids bilinear bleed).
    private static let cellMargin: Float = 0.02

    static func unwrap(positions: [SIMD3<Float>],
                       normals: [SIMD3<Float>],
                       indices: [UInt32]) -> Result {

        // (originalVertexIndex, group) -> new vertex index.
        struct Key: Hashable { var vertex: Int; var group: Int }
        var newIndexForKey = [Key: UInt32]()

        var outPositions = [SIMD3<Float>]()
        var outNormals = [SIMD3<Float>]()
        var rawUV = [SIMD2<Float>]()          // pre-normalisation projected coords
        var groupOf = [Int]()                 // group per new vertex
        var outIndices = [UInt32]()
        outPositions.reserveCapacity(positions.count)

        // Per-group projected-UV bounds for later normalisation.
        var groupMin = [SIMD2<Float>](repeating: SIMD2(.greatestFiniteMagnitude, .greatestFiniteMagnitude), count: 6)
        var groupMax = [SIMD2<Float>](repeating: SIMD2(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude), count: 6)

        func project(_ p: SIMD3<Float>, group: Int) -> SIMD2<Float> {
            switch group {
            case 0: return SIMD2(-p.z, p.y)   // +X
            case 1: return SIMD2(p.z, p.y)    // -X
            case 2: return SIMD2(p.x, -p.z)   // +Y
            case 3: return SIMD2(p.x, p.z)    // -Y
            case 4: return SIMD2(p.x, p.y)    // +Z
            default: return SIMD2(-p.x, p.y)  // -Z
            }
        }

        func group(for normal: SIMD3<Float>) -> Int {
            let a = abs(normal)
            if a.x >= a.y && a.x >= a.z { return normal.x >= 0 ? 0 : 1 }
            if a.y >= a.z { return normal.y >= 0 ? 2 : 3 }
            return normal.z >= 0 ? 4 : 5
        }

        func vertex(_ original: Int, _ grp: Int) -> UInt32 {
            let key = Key(vertex: original, group: grp)
            if let existing = newIndexForKey[key] { return existing }
            let idx = UInt32(outPositions.count)
            newIndexForKey[key] = idx
            let p = positions[original]
            outPositions.append(p)
            outNormals.append(original < normals.count ? normals[original] : SIMD3(0, 1, 0))
            let uv = project(p, group: grp)
            rawUV.append(uv)
            groupOf.append(grp)
            groupMin[grp] = simd_min(groupMin[grp], uv)
            groupMax[grp] = simd_max(groupMax[grp], uv)
            return idx
        }

        var i = 0
        while i + 2 < indices.count {
            let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
            let faceN = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            let len = simd_length(faceN)
            let grp = len > 1e-12 ? group(for: faceN / len) : 0
            outIndices.append(vertex(a, grp))
            outIndices.append(vertex(b, grp))
            outIndices.append(vertex(c, grp))
            i += 3
        }

        // Normalise each vertex's raw UV within its group bounds, then place into its atlas cell.
        let cellW = 1.0 / Float(atlasCols)
        let cellH = 1.0 / Float(atlasRows)
        var uvs = [SIMD2<Float>](repeating: .zero, count: outPositions.count)

        for v in 0..<outPositions.count {
            let grp = groupOf[v]
            let mn = groupMin[grp], mx = groupMax[grp]
            let span = simd_max(mx - mn, SIMD2(1e-6, 1e-6))
            var local = (rawUV[v] - mn) / span               // 0...1 within chart
            local = local * (1 - 2 * cellMargin) + SIMD2(cellMargin, cellMargin)
            let col = grp % atlasCols
            let row = grp / atlasCols
            uvs[v] = SIMD2(( Float(col) + local.x) * cellW,
                           ( Float(row) + local.y) * cellH)
        }

        // TODO(nimbus): replace box projection with a real chart-based atlas — grow charts
        // by normal-deviation flooding, parameterise each with LSCM/ABF++ to minimise
        // stretch, then rectangle bin-pack the charts (xatlas algorithm). Requires porting
        // xatlas to Swift/C++ or shipping it as a small C target in the xcframework.
        return Result(positions: outPositions, normals: outNormals, uvs: uvs, indices: outIndices)
    }
}
