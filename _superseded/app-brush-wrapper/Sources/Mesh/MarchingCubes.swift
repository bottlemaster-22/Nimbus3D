//
//  MarchingCubes.swift — Mesh module (REAL)
//
//  Lorensen-Cline marching cubes over a DensityField, producing a welded indexed
//  triangle mesh. Corner/edge numbering follows Paul Bourke's reference (see
//  MarchingCubesTables). Vertices shared between adjacent cubes are welded by a
//  quantised-position hash so the output is a connected mesh suitable for QEM
//  decimation and UV unwrap.
//
//  Convention: a corner is flagged "inside" when its density is BELOW the iso-level,
//  matching Bourke's tables. Triangle winding is corrected afterwards in the extractor
//  using the density gradient so normals face outward regardless of table winding.
//

import Foundation
import simd

enum MarchingCubes {

    /// Corner offsets in voxel units (Bourke GRIDCELL order 0...7).
    private static let cornerOffset: [SIMD3<Int>] = [
        SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1),
        SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(0, 1, 1)
    ]

    /// The two corners bounding each of the 12 edges.
    private static let edgeCorners: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    struct Result {
        var positions: [SIMD3<Float>]
        var indices: [UInt32]
    }

    /// Extracts the iso-surface at `isoLevel` from `field`.
    /// - Parameter isCancelled: polled per z-slice so a long extraction can bail out.
    static func surface(from field: DensityField,
                        isoLevel: Float,
                        isCancelled: () -> Bool) -> Result {
        let dims = field.dims
        var positions = [SIMD3<Float>]()
        var indices = [UInt32]()
        positions.reserveCapacity(4096)
        indices.reserveCapacity(8192)

        // Weld vertices by quantised world position.
        let quantScale = 1.0 / (field.voxelSize * 1e-3)
        var vertexMap = [SIMD3<Int64>: UInt32](minimumCapacity: 4096)

        func vertexKey(_ p: SIMD3<Float>) -> SIMD3<Int64> {
            SIMD3(Int64((p.x * quantScale).rounded()),
                  Int64((p.y * quantScale).rounded()),
                  Int64((p.z * quantScale).rounded()))
        }

        func indexFor(_ p: SIMD3<Float>) -> UInt32 {
            let key = vertexKey(p)
            if let existing = vertexMap[key] { return existing }
            let newIndex = UInt32(positions.count)
            positions.append(p)
            vertexMap[key] = newIndex
            return newIndex
        }

        var cornerVal = [Float](repeating: 0, count: 8)
        var cornerPos = [SIMD3<Float>](repeating: .zero, count: 8)
        var edgeVertex = [SIMD3<Float>](repeating: .zero, count: 12)

        for z in 0..<(dims.z - 1) {
            if isCancelled() { break }
            for y in 0..<(dims.y - 1) {
                for x in 0..<(dims.x - 1) {
                    var cubeIndex = 0
                    for i in 0..<8 {
                        let o = cornerOffset[i]
                        let cx = x + o.x, cy = y + o.y, cz = z + o.z
                        let v = field.value(cx, cy, cz)
                        cornerVal[i] = v
                        cornerPos[i] = field.worldPosition(cx, cy, cz)
                        if v < isoLevel { cubeIndex |= (1 << i) }
                    }

                    let edges = MarchingCubesTables.edgeTable[cubeIndex]
                    if edges == 0 { continue }

                    for e in 0..<12 where (edges & (1 << e)) != 0 {
                        let (a, b) = edgeCorners[e]
                        edgeVertex[e] = interpolate(isoLevel,
                                                    cornerPos[a], cornerPos[b],
                                                    cornerVal[a], cornerVal[b])
                    }

                    let tris = MarchingCubesTables.triTable[cubeIndex]
                    var t = 0
                    while t + 2 < tris.count {
                        let e0 = tris[t]
                        if e0 < 0 { break }
                        let e1 = tris[t + 1]
                        let e2 = tris[t + 2]
                        let i0 = indexFor(edgeVertex[Int(e0)])
                        let i1 = indexFor(edgeVertex[Int(e1)])
                        let i2 = indexFor(edgeVertex[Int(e2)])
                        // Drop degenerate triangles that welding may have collapsed.
                        if i0 != i1 && i1 != i2 && i0 != i2 {
                            indices.append(i0); indices.append(i1); indices.append(i2)
                        }
                        t += 3
                    }
                }
            }
        }

        return Result(positions: positions, indices: indices)
    }

    @inline(__always)
    private static func interpolate(_ iso: Float,
                                    _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                                    _ v1: Float, _ v2: Float) -> SIMD3<Float> {
        let denom = v2 - v1
        if abs(denom) < 1e-9 { return p1 }
        let t = (iso - v1) / denom
        return p1 + (p2 - p1) * min(max(t, 0), 1)
    }
}
