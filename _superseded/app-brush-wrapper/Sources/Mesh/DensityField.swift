//
//  DensityField.swift — Mesh module (REAL)
//
//  A dense scalar occupancy grid built by scattering Gaussian splats. This is the
//  on-device stand-in for a TSDF: instead of fusing posed depth, we accumulate each
//  Gaussian's opacity-weighted density into voxels, then marching cubes extracts the
//  iso-surface of that field. It is a genuine, deterministic surfacing method — but
//  see the honesty note in SplatMeshExtractor: a density envelope over volumetric
//  Gaussians is NOT a surface-aligned reconstruction (2DGS/GOF/MILo), so the result
//  is watertight-ish but "puffy" around thin structure. That gap is the documented
//  frontier stub.
//
//  Two accumulation backends, identical math:
//    - Metal compute (SplatDensity.metal) when a device + default library are present.
//    - Pure-Swift CPU scatter as a guaranteed fallback (CI, simulators, older GPUs).
//

import Foundation
import simd
import Metal

/// Parameters shared with the Metal kernel. Scalar-only layout (no SIMD vectors) so it
/// maps 1:1 onto the MSL `DensityGridParams` with unambiguous 4-byte field alignment.
/// A `SIMD3<Float>` (16-byte) would NOT match MSL `packed_float3` (12-byte), so we avoid it.
struct DensityGridParams {
    var originX: Float
    var originY: Float
    var originZ: Float
    var voxelSize: Float
    var dimX: UInt32
    var dimY: UInt32
    var dimZ: UInt32
    var splatCount: UInt32
    var radiusVoxels: Float
    var fixedPointScale: Float
}

/// A 3D scalar field on a regular voxel grid.
struct DensityField {
    let dims: SIMD3<Int>
    let origin: SIMD3<Float>
    let voxelSize: Float
    /// Row-major x-fastest density values, length dims.x*dims.y*dims.z.
    var values: [Float]
    /// Highest density observed (used to derive a relative iso-level).
    let maxValue: Float

    @inline(__always)
    func linearIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
        (z * dims.y + y) * dims.x + x
    }

    @inline(__always)
    func value(_ x: Int, _ y: Int, _ z: Int) -> Float {
        values[linearIndex(x, y, z)]
    }

    /// World-space centre of voxel (x,y,z).
    @inline(__always)
    func worldPosition(_ x: Int, _ y: Int, _ z: Int) -> SIMD3<Float> {
        origin + (SIMD3<Float>(Float(x), Float(y), Float(z)) + 0.5) * voxelSize
    }
}

enum DensityFieldBuilder {

    /// Support radius of the Gaussian kernel expressed in multiples of the splat's own
    /// std-dev. 2.5σ captures ~99% of the mass while keeping the scatter neighbourhood small.
    static let kernelSigmas: Float = 2.5
    /// Fixed-point quantisation for the atomic-uint GPU path.
    static let fixedPointScale: Float = 4096.0

    /// Builds a density grid sized to `bounds` at the requested longest-axis resolution.
    /// Returns nil only if there are no splats or the bounds are degenerate.
    static func build(splats: [GaussianSplat],
                      bounds: AxisAlignedBoundingBox,
                      longestAxisResolution: Int,
                      device: MTLDevice?) -> DensityField? {
        guard !splats.isEmpty else { return nil }

        let extents = bounds.extents
        let longest = max(extents.x, max(extents.y, extents.z))
        guard longest > 0, longestAxisResolution > 1 else { return nil }

        // Pad the box slightly so surface voxels are not clipped at the border.
        let voxelSize = longest / Float(longestAxisResolution)
        let pad = voxelSize * 2
        let minC = bounds.minCorner - SIMD3<Float>(repeating: pad)
        let maxC = bounds.maxCorner + SIMD3<Float>(repeating: pad)
        let paddedExtents = maxC - minC

        var dims = SIMD3<Int>(
            max(2, Int((paddedExtents.x / voxelSize).rounded(.up))),
            max(2, Int((paddedExtents.y / voxelSize).rounded(.up))),
            max(2, Int((paddedExtents.z / voxelSize).rounded(.up)))
        )
        // Guard against pathological memory: cap total voxels (~64M -> 256MB float readback).
        let maxVoxels = 64_000_000
        while dims.x * dims.y * dims.z > maxVoxels {
            dims = SIMD3<Int>(max(2, dims.x / 2), max(2, dims.y / 2), max(2, dims.z / 2))
        }

        // Representative kernel radius (metres) for GPU neighbourhood sizing. We use the
        // median-ish average radius; the per-splat exact radius is still applied per scatter.
        let avgRadius = splats.reduce(Float(0)) { $0 + $1.radius } / Float(splats.count)
        let radiusVoxels = max(1.0, (kernelSigmas * avgRadius) / voxelSize)

        if let device,
           let field = buildMetal(splats: splats, dims: dims, origin: minC,
                                   voxelSize: voxelSize, radiusVoxels: radiusVoxels,
                                   device: device) {
            return field
        }
        return buildCPU(splats: splats, dims: dims, origin: minC,
                        voxelSize: voxelSize)
    }

    // MARK: - CPU scatter

    private static func buildCPU(splats: [GaussianSplat],
                                 dims: SIMD3<Int>,
                                 origin: SIMD3<Float>,
                                 voxelSize: Float) -> DensityField {
        let count = dims.x * dims.y * dims.z
        var grid = [Float](repeating: 0, count: count)

        grid.withUnsafeMutableBufferPointer { buf in
            for s in splats {
                if s.opacity <= 0 { continue }
                let radius = s.radius
                let support = kernelSigmas * radius
                let supportVox = support / voxelSize
                let local = (s.position - origin) / voxelSize
                let r = Int(supportVox.rounded(.up))
                let bx = Int(local.x.rounded(.down))
                let by = Int(local.y.rounded(.down))
                let bz = Int(local.z.rounded(.down))
                // density = opacity * exp(-0.5 * (worldDist/radius)^2)
                let k = voxelSize / radius
                let coeff = 0.5 * k * k

                let z0 = max(0, bz - r), z1 = min(dims.z - 1, bz + r)
                let y0 = max(0, by - r), y1 = min(dims.y - 1, by + r)
                let x0 = max(0, bx - r), x1 = min(dims.x - 1, bx + r)
                var z = z0
                while z <= z1 {
                    var y = y0
                    while y <= y1 {
                        let rowBase = (z * dims.y + y) * dims.x
                        var x = x0
                        while x <= x1 {
                            let vc = SIMD3<Float>(Float(x), Float(y), Float(z)) + 0.5
                            let d = vc - local
                            let dist2 = simd_dot(d, d)
                            let g = s.opacity * expf(-coeff * dist2)
                            if g > 0 {
                                buf[rowBase + x] += g
                            }
                            x += 1
                        }
                        y += 1
                    }
                    z += 1
                }
            }
        }

        let maxValue = grid.max() ?? 0
        return DensityField(dims: dims, origin: origin, voxelSize: voxelSize,
                            values: grid, maxValue: maxValue)
    }

    // MARK: - Metal scatter

    private static func buildMetal(splats: [GaussianSplat],
                                   dims: SIMD3<Int>,
                                   origin: SIMD3<Float>,
                                   voxelSize: Float,
                                   radiusVoxels: Float,
                                   device: MTLDevice) -> DensityField? {
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "accumulate_density"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            return nil
        }

        let voxelCount = dims.x * dims.y * dims.z

        var packed = [PackedSplat]()
        packed.reserveCapacity(splats.count)
        for s in splats {
            packed.append(PackedSplat(position: s.position,
                                      invRadius: 1.0 / s.radius,
                                      weight: s.opacity))
        }

        guard let splatBuffer = device.makeBuffer(bytes: packed,
                                                  length: MemoryLayout<PackedSplat>.stride * packed.count,
                                                  options: .storageModeShared),
              let gridBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * voxelCount,
                                                 options: .storageModeShared) else {
            return nil
        }
        // Zero the grid (makeBuffer(length:) contents are not guaranteed zeroed).
        memset(gridBuffer.contents(), 0, MemoryLayout<UInt32>.stride * voxelCount)

        var params = DensityGridParams(
            originX: origin.x, originY: origin.y, originZ: origin.z,
            voxelSize: voxelSize,
            dimX: UInt32(dims.x), dimY: UInt32(dims.y), dimZ: UInt32(dims.z),
            splatCount: UInt32(splats.count),
            radiusVoxels: radiusVoxels,
            fixedPointScale: fixedPointScale
        )

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(gridBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<DensityGridParams>.stride, index: 2)

        let threadsPerGroup = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let groups = (splats.count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        if cmd.status != .completed { return nil }

        // Convert fixed-point uint grid back to float density.
        var values = [Float](repeating: 0, count: voxelCount)
        let inv = 1.0 / fixedPointScale
        let ptr = gridBuffer.contents().bindMemory(to: UInt32.self, capacity: voxelCount)
        values.withUnsafeMutableBufferPointer { out in
            for i in 0..<voxelCount { out[i] = Float(ptr[i]) * inv }
        }

        let maxValue = values.max() ?? 0
        return DensityField(dims: dims, origin: origin, voxelSize: voxelSize,
                            values: values, maxValue: maxValue)
    }
}
