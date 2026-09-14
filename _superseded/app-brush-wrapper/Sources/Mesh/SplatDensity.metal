//
//  SplatDensity.metal — Mesh module (REAL)
//
//  GPU density-field accumulation for splat -> mesh surfacing. Each thread owns one
//  Gaussian and scatters its opacity-weighted contribution into a dense scalar voxel
//  grid. Accumulation uses fixed-point atomic_uint adds rather than atomic_float,
//  because 32-bit float atomics are not guaranteed across all Metal GPU families,
//  whereas atomic_uint fetch_add is universally available. The host converts the
//  integer grid back to float density after readback (see DensityField.swift).
//
//  This layout MUST stay in lockstep with `PackedSplat` and `DensityGridParams`
//  in the Swift sources.
//

#include <metal_stdlib>
using namespace metal;

struct SplatPoint {
    float3 position;   // world-space centre (metres)
    float  invRadius;  // 1 / support radius
    float  weight;     // linear opacity
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

// Scalar-only layout (no packed vectors) so it maps 1:1 onto the Swift mirror
// with unambiguous 4-byte field alignment. See DensityGridParams in DensityField.swift.
struct DensityGridParams {
    float originX;
    float originY;
    float originZ;
    float voxelSize;        // world metres per voxel edge
    uint  dimX;
    uint  dimY;
    uint  dimZ;
    uint  splatCount;
    float radiusVoxels;     // isotropic kernel support in voxel units
    float fixedPointScale;  // float density -> uint quantisation factor
};

kernel void accumulate_density(device const SplatPoint*     splats [[buffer(0)]],
                               device atomic_uint*          grid   [[buffer(1)]],
                               constant DensityGridParams&  p      [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= p.splatCount) { return; }

    SplatPoint s = splats[gid];
    if (s.weight <= 0.0f) { return; }

    const float3 origin = float3(p.originX, p.originY, p.originZ);
    const uint3  dims   = uint3(p.dimX, p.dimY, p.dimZ);

    // Splat centre in continuous voxel coordinates.
    float3 local = (s.position - origin) / p.voxelSize;

    int r = int(ceil(p.radiusVoxels));
    int3 base = int3(floor(local));

    // Gaussian falloff coefficient in voxel-distance^2 space:
    //   density = weight * exp(-0.5 * (worldDist / radius)^2)
    // worldDist = voxelDist * voxelSize, and invRadius = 1/radius.
    float k = p.voxelSize * s.invRadius;
    float coeff = 0.5f * k * k;

    for (int dz = -r; dz <= r; ++dz) {
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int3 c = base + int3(dx, dy, dz);
                if (c.x < 0 || c.y < 0 || c.z < 0) { continue; }
                if (uint(c.x) >= dims.x || uint(c.y) >= dims.y || uint(c.z) >= dims.z) { continue; }

                float3 voxelCentre = float3(c) + 0.5f;
                float3 d = voxelCentre - local;      // in voxel units
                float dist2 = dot(d, d);
                float g = s.weight * exp(-coeff * dist2);

                uint add = uint(g * p.fixedPointScale);
                if (add == 0u) { continue; }

                uint idx = (uint(c.z) * dims.y + uint(c.y)) * dims.x + uint(c.x);
                atomic_fetch_add_explicit(&grid[idx], add, memory_order_relaxed);
            }
        }
    }
}
