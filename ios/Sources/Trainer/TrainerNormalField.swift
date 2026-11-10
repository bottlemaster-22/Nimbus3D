//
//  TrainerNormalField.swift
//
//  Build 464: which way the real surface faces, everywhere the LiDAR saw one.
//
//  WHY: rendered offline, the owner's Nov 9 model showed flat walls and doors
//  as a speckle of splats pointing every which way: on clearly planar patches
//  the splats sat a median 30 degrees off their own surface, one in five more
//  than 60 (random would be 60). Nothing in the trainer ever told a splat which
//  way the surface faces; the disc prior only makes it flat. From any view but
//  the photo's, tilted discs read as bristles and smears.
//
//  The pre-pass already knows: every seed it shaped as a confident disc has its
//  thin axis along the depth map's normal (PrePassInitialSplats). This folds
//  those normals into a coarse world-space grid. Per cell: the sign-aligned
//  mean normal and how well the seeds in it agree. Only cells where they agree
//  (a wall, a door, a floor) carry any weight, so corners, edges and small
//  objects are left to the photographs. trainer_regularizer reads the cell
//  under each splat and pulls the splat's thin axis toward it.
//

import Foundation
import simd

struct TrainerNormalField {
    /// World-space (RUB) corner of cell (0, 0, 0), metres.
    var origin: SIMD3<Float>
    /// Cell edge, metres.
    var voxel: Float
    var dimX: Int
    var dimY: Int
    var dimZ: Int
    /// Per cell, x fastest: (normal.xyz, confidence). Confidence 0 = no say.
    var cells: [SIMD4<Float16>]
    /// Cells with any confidence, for the log.
    var confidentCells: Int

    /// Builds the field from the pre-pass seed cloud. nil when too few seeds
    /// are disc-shaped to say anything.
    static func build(
        from cloud: SplatCloud,
        requestedVoxel: Float = 0.04,
        maxCells: Int = 2_000_000
    ) -> TrainerNormalField? {
        let n = cloud.count
        guard n > 0, cloud.logScales.count == n, cloud.rotations.count == n else { return nil }

        // Disc-shaped seeds only: thinnest axis under 0.4 of the middle one.
        var normals: [SIMD3<Float>] = []
        var points: [SIMD3<Float>] = []
        normals.reserveCapacity(n / 2)
        points.reserveCapacity(n / 2)
        let flatLimit = logf(0.4)
        for i in 0..<n {
            let s = cloud.logScales[i]
            let p = cloud.positions[i]
            guard s.x.isFinite, s.y.isFinite, s.z.isFinite,
                  p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
            let lo = Swift.min(s.x, s.y, s.z)
            let hi = Swift.max(s.x, s.y, s.z)
            let mid = s.x + s.y + s.z - lo - hi
            guard lo - mid < flatLimit else { continue }
            let axis = s.x == lo ? 0 : (s.y == lo ? 1 : 2)
            let q = cloud.rotations[i]
            let length = simd_length(q)
            guard length > 1e-6, length.isFinite else { continue }
            let u = q / length
            let x = u.x, y = u.y, z = u.z, w = u.w
            let normal: SIMD3<Float>
            switch axis {
            case 0: normal = SIMD3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
            case 1: normal = SIMD3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
            default: normal = SIMD3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
            }
            normals.append(normal)
            points.append(p)
        }
        guard points.count >= 1_000 else { return nil }

        // Bounds from the 0.5th to 99.5th percentile per axis, so a few stray
        // far seeds cannot stretch the grid over empty metres.
        func percentileRange(_ values: [Float]) -> (Float, Float) {
            let sorted = values.sorted()
            let a = sorted[sorted.count / 200]
            let b = sorted[sorted.count - 1 - sorted.count / 200]
            return (a, b)
        }
        let (x0, x1) = percentileRange(points.map { $0.x })
        let (y0, y1) = percentileRange(points.map { $0.y })
        let (z0, z1) = percentileRange(points.map { $0.z })
        let lo = SIMD3<Float>(x0, y0, z0) - 0.1
        let extent = simd_min(SIMD3<Float>(x1, y1, z1) + 0.1 - lo, SIMD3<Float>(repeating: 40))
        guard extent.x > 0, extent.y > 0, extent.z > 0 else { return nil }

        var voxel = Swift.max(requestedVoxel, 0.01)
        var dims = SIMD3<Int>(1, 1, 1)
        for _ in 0..<40 {
            let cellsPerAxis = (extent / voxel).rounded(.up)
            dims = SIMD3<Int>(
                Swift.max(Int(cellsPerAxis.x), 1),
                Swift.max(Int(cellsPerAxis.y), 1),
                Swift.max(Int(cellsPerAxis.z), 1)
            )
            if dims.x * dims.y * dims.z <= maxCells { break }
            voxel *= 1.15
        }
        guard dims.x * dims.y * dims.z <= maxCells else { return nil }

        // Sign-aligned sums: each normal is flipped to agree with what the
        // cell has gathered so far, because a depth-map normal's sign says
        // which side the camera stood on, not which way the wall faces.
        var sums: [Int: SIMD4<Float>] = [:]
        sums.reserveCapacity(points.count / 8)
        for i in 0..<points.count {
            let r = (points[i] - lo) / voxel
            guard r.x >= 0, r.y >= 0, r.z >= 0,
                  r.x < Float(dims.x), r.y < Float(dims.y), r.z < Float(dims.z) else { continue }
            let cx = Int(r.x), cy = Int(r.y), cz = Int(r.z)
            let key = (cz * dims.y + cy) * dims.x + cx
            var acc = sums[key] ?? .zero
            var normal = normals[i]
            if simd_dot(SIMD3(acc.x, acc.y, acc.z), normal) < 0 { normal = -normal }
            acc += SIMD4<Float>(normal, 1)
            sums[key] = acc
        }

        var cells = [SIMD4<Float16>](repeating: .zero, count: dims.x * dims.y * dims.z)
        var confident = 0
        for (key, acc) in sums {
            let count = acc.w
            guard count >= 2 else { continue }
            let sum = SIMD3<Float>(acc.x, acc.y, acc.z)
            let length = simd_length(sum)
            guard length > 1e-6 else { continue }
            // 1 when every seed in the cell agrees; ~0.5 or less at a corner.
            let coherence = length / count
            // Build 466: 0.85 -> 0, 0.97 -> 1 (was 0.75 -> 0.95), so a cell that
            // mixes a wall with what hangs on it has no say.
            let confidence = Swift.min(Swift.max((coherence - 0.85) / 0.12, 0), 1)
                * Swift.min(count / 4, 1)
            guard confidence > 0 else { continue }
            let normal = sum / length
            cells[key] = SIMD4<Float16>(
                Float16(normal.x), Float16(normal.y), Float16(normal.z), Float16(confidence)
            )
            confident += 1
        }
        guard confident > 0 else { return nil }
        return TrainerNormalField(
            origin: lo, voxel: voxel, dimX: dims.x, dimY: dims.y, dimZ: dims.z,
            cells: cells, confidentCells: confident
        )
    }
}
