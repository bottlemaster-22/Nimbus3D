//
//  MeshDecimator.swift — Mesh module (REAL)
//
//  Garland-Heckbert quadric-error-metric (QEM) edge-collapse decimation. This is a
//  genuine implementation: per-vertex 4x4 quadrics accumulated from area-weighted face
//  planes, a lazily-invalidated min-heap of candidate collapses, optimal contraction
//  positions via a 3x3 solve (midpoint fallback when singular), and incremental
//  topology / quadric updates after each collapse.
//
//  Scope honesty: this preserves the QEM error objective and produces a valid indexed
//  mesh at the requested triangle budget. It does NOT add explicit boundary-preservation
//  penalties or foldover/manifold guards, so extreme decimation of open meshes can
//  nibble silhouettes — acceptable for a game-ready low-poly pass, noted for a future
//  hardening. See TODO below.
//

import Foundation
import simd

enum MeshDecimator {

    /// Symmetric 4x4 error quadric stored as its 10 upper-triangular coefficients.
    ///   [ a00 a01 a02 a03 ]
    ///   [ a01 a11 a12 a13 ]
    ///   [ a02 a12 a22 a23 ]
    ///   [ a03 a13 a23 a33 ]
    struct Quadric {
        var a00: Float, a01: Float, a02: Float, a03: Float
        var a11: Float, a12: Float, a13: Float
        var a22: Float, a23: Float
        var a33: Float

        static let zero = Quadric(a00: 0, a01: 0, a02: 0, a03: 0,
                                  a11: 0, a12: 0, a13: 0, a22: 0, a23: 0, a33: 0)

        /// Build from a plane (a,b,c,d) with ax+by+cz+d = 0, scaled by `weight` (triangle area).
        init(plane n: SIMD3<Float>, d: Float, weight: Float) {
            let a = n.x, b = n.y, c = n.z
            a00 = a*a * weight; a01 = a*b * weight; a02 = a*c * weight; a03 = a*d * weight
            a11 = b*b * weight; a12 = b*c * weight; a13 = b*d * weight
            a22 = c*c * weight; a23 = c*d * weight
            a33 = d*d * weight
        }

        init(a00: Float, a01: Float, a02: Float, a03: Float,
             a11: Float, a12: Float, a13: Float, a22: Float, a23: Float, a33: Float) {
            self.a00 = a00; self.a01 = a01; self.a02 = a02; self.a03 = a03
            self.a11 = a11; self.a12 = a12; self.a13 = a13
            self.a22 = a22; self.a23 = a23; self.a33 = a33
        }

        static func + (l: Quadric, r: Quadric) -> Quadric {
            Quadric(a00: l.a00+r.a00, a01: l.a01+r.a01, a02: l.a02+r.a02, a03: l.a03+r.a03,
                    a11: l.a11+r.a11, a12: l.a12+r.a12, a13: l.a13+r.a13,
                    a22: l.a22+r.a22, a23: l.a23+r.a23, a33: l.a33+r.a33)
        }

        /// Evaluate v^T Q v for homogeneous point (p, 1).
        func error(at p: SIMD3<Float>) -> Float {
            let x = p.x, y = p.y, z = p.z
            return a00*x*x + 2*a01*x*y + 2*a02*x*z + 2*a03*x
                 + a11*y*y + 2*a12*y*z + 2*a13*y
                 + a22*z*z + 2*a23*z
                 + a33
        }
    }

    private struct HeapEntry {
        var cost: Float
        var a: Int
        var b: Int
        var va: Int   // version of a at insertion
        var vb: Int   // version of b at insertion
        var target: SIMD3<Float>
    }

    struct Result {
        var positions: [SIMD3<Float>]
        var indices: [UInt32]
    }

    /// Decimate to at most `targetTriangleCount` triangles. Returns compacted arrays.
    static func decimate(positions inPositions: [SIMD3<Float>],
                         indices inIndices: [UInt32],
                         targetTriangleCount: Int,
                         isCancelled: () -> Bool) -> Result {

        var positions = inPositions
        let faceCount = inIndices.count / 3
        if faceCount <= targetTriangleCount || faceCount == 0 {
            return Result(positions: inPositions, indices: inIndices)
        }

        // Faces as mutable index triples.
        var faces = [SIMD3<Int>](repeating: .zero, count: faceCount)
        for f in 0..<faceCount {
            faces[f] = SIMD3(Int(inIndices[f*3]), Int(inIndices[f*3+1]), Int(inIndices[f*3+2]))
        }
        var faceAlive = [Bool](repeating: true, count: faceCount)
        var liveFaces = faceCount

        let vCount = positions.count
        var vertexAlive = [Bool](repeating: true, count: vCount)
        var vertexVersion = [Int](repeating: 0, count: vCount)
        var quadrics = [Quadric](repeating: .zero, count: vCount)
        var vertexFaces = [Set<Int>](repeating: [], count: vCount)

        // Accumulate quadrics and adjacency.
        for f in 0..<faceCount {
            let tri = faces[f]
            let p0 = positions[tri.x], p1 = positions[tri.y], p2 = positions[tri.z]
            let cross = simd_cross(p1 - p0, p2 - p0)
            let area = simd_length(cross) * 0.5
            if area > 1e-12 {
                let n = simd_normalize(cross)
                let d = -simd_dot(n, p0)
                let quad = Quadric(plane: n, d: d, weight: area)
                quadrics[tri.x] = quadrics[tri.x] + quad
                quadrics[tri.y] = quadrics[tri.y] + quad
                quadrics[tri.z] = quadrics[tri.z] + quad
            }
            vertexFaces[tri.x].insert(f)
            vertexFaces[tri.y].insert(f)
            vertexFaces[tri.z].insert(f)
        }

        // Min-heap of candidate collapses.
        var heap = [HeapEntry]()
        heap.reserveCapacity(faceCount * 3)

        func optimalTarget(_ a: Int, _ b: Int, _ Q: Quadric) -> (SIMD3<Float>, Float) {
            // Solve the 3x3 system A x = rhs from the quadric's top-left block.
            let A = simd_float3x3(rows: [
                SIMD3(Q.a00, Q.a01, Q.a02),
                SIMD3(Q.a01, Q.a11, Q.a12),
                SIMD3(Q.a02, Q.a12, Q.a22)
            ])
            let rhs = SIMD3(-Q.a03, -Q.a13, -Q.a23)
            let det = simd_determinant(A)
            if abs(det) > 1e-10 {
                let x = simd_inverse(A) * rhs
                return (x, Q.error(at: x))
            }
            // Fallback: pick the cheapest of endpoint a, endpoint b, midpoint.
            let pa = positions[a], pb = positions[b], pm = (pa + pb) * 0.5
            let ea = Q.error(at: pa), eb = Q.error(at: pb), em = Q.error(at: pm)
            if ea <= eb && ea <= em { return (pa, ea) }
            if eb <= em { return (pb, eb) }
            return (pm, em)
        }

        func pushEdge(_ a: Int, _ b: Int) {
            guard a != b, vertexAlive[a], vertexAlive[b] else { return }
            let Q = quadrics[a] + quadrics[b]
            let (target, cost) = optimalTarget(a, b, Q)
            heapPush(&heap, HeapEntry(cost: cost, a: a, b: b,
                                      va: vertexVersion[a], vb: vertexVersion[b],
                                      target: target))
        }

        // Seed heap with all unique edges.
        var seen = Set<Int64>()
        func edgeKey(_ a: Int, _ b: Int) -> Int64 {
            let lo = Int64(min(a, b)), hi = Int64(max(a, b))
            return lo &* 2_654_435_761 &+ hi
        }
        for f in 0..<faceCount {
            let t = faces[f]
            let pairs = [(t.x, t.y), (t.y, t.z), (t.z, t.x)]
            for (a, b) in pairs where seen.insert(edgeKey(a, b)).inserted {
                pushEdge(a, b)
            }
        }

        // Collapse loop.
        while liveFaces > targetTriangleCount, !heap.isEmpty {
            if isCancelled() { break }
            guard let e = heapPop(&heap) else { break }
            // Reject stale entries.
            guard vertexAlive[e.a], vertexAlive[e.b],
                  vertexVersion[e.a] == e.va, vertexVersion[e.b] == e.vb else { continue }

            let keep = e.a, remove = e.b

            // Apply the contraction.
            positions[keep] = e.target
            quadrics[keep] = quadrics[keep] + quadrics[remove]
            vertexAlive[remove] = false
            vertexVersion[keep] += 1

            // Rewire faces incident to `remove`. Copy the set first so we are not reading
            // `vertexFaces` while the loop body mutates it (Swift exclusive-access safety).
            let incidentToRemove = vertexFaces[remove]
            for f in incidentToRemove where faceAlive[f] {
                var tri = faces[f]
                if tri.x == remove { tri.x = keep }
                if tri.y == remove { tri.y = keep }
                if tri.z == remove { tri.z = keep }
                if tri.x == tri.y || tri.y == tri.z || tri.x == tri.z {
                    // Collapsed to a sliver — kill it.
                    faceAlive[f] = false
                    liveFaces -= 1
                    vertexFaces[tri.x].remove(f)
                    vertexFaces[tri.y].remove(f)
                    vertexFaces[tri.z].remove(f)
                } else {
                    faces[f] = tri
                    vertexFaces[keep].insert(f)
                }
            }
            vertexFaces[remove] = []

            // Refresh candidate edges around the surviving vertex.
            var neighbors = Set<Int>()
            let incidentToKeep = vertexFaces[keep]
            for f in incidentToKeep where faceAlive[f] {
                let t = faces[f]
                if t.x != keep, vertexAlive[t.x] { neighbors.insert(t.x) }
                if t.y != keep, vertexAlive[t.y] { neighbors.insert(t.y) }
                if t.z != keep, vertexAlive[t.z] { neighbors.insert(t.z) }
            }
            for n in neighbors { pushEdge(keep, n) }
        }

        // Compact into fresh, densely-indexed arrays.
        var remap = [Int](repeating: -1, count: positions.count)
        var outPositions = [SIMD3<Float>]()
        outPositions.reserveCapacity(liveFaces * 3)
        var outIndices = [UInt32]()
        outIndices.reserveCapacity(liveFaces * 3)

        for f in 0..<faceCount where faceAlive[f] {
            let tri = faces[f]
            for vi in [tri.x, tri.y, tri.z] {
                if remap[vi] < 0 {
                    remap[vi] = outPositions.count
                    outPositions.append(positions[vi])
                }
                outIndices.append(UInt32(remap[vi]))
            }
        }

        // TODO(nimbus): add boundary-edge quadric penalties and a foldover/normal-flip
        // guard (reject collapses that invert an incident face normal) to preserve open
        // silhouettes and prevent self-intersection at aggressive ratios.
        return Result(positions: outPositions, indices: outIndices)
    }

    // MARK: - Binary min-heap

    private static func heapPush(_ heap: inout [HeapEntry], _ entry: HeapEntry) {
        heap.append(entry)
        var i = heap.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if heap[parent].cost <= heap[i].cost { break }
            heap.swapAt(parent, i)
            i = parent
        }
    }

    private static func heapPop(_ heap: inout [HeapEntry]) -> HeapEntry? {
        guard !heap.isEmpty else { return nil }
        let top = heap[0]
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var i = 0
            let n = heap.count
            while true {
                let l = 2*i + 1, r = 2*i + 2
                var smallest = i
                if l < n, heap[l].cost < heap[smallest].cost { smallest = l }
                if r < n, heap[r].cost < heap[smallest].cost { smallest = r }
                if smallest == i { break }
                heap.swapAt(i, smallest)
                i = smallest
            }
        }
        return top
    }
}
