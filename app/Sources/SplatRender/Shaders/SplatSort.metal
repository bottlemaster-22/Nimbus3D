//
//  SplatSort.metal — SplatRender module.
//
//  GPU depth sort for correct back-to-front alpha compositing. Two kernels:
//
//   1. computeDepthKeys — per splat, write { key = view-space z, index }.
//      Right-handed camera looks down -Z, so a farther splat has a MORE NEGATIVE
//      z. An ASCENDING sort therefore orders farthest -> nearest, which is the
//      back-to-front order the "over" blend needs. Padding entries (index beyond
//      splatCount, added to reach a power of two) get key = +FLT_MAX so they sort
//      to the tail and are never drawn (the renderer issues splatCount instances).
//
//   2. bitonicSortStep — one step of a bitonic sorting network over the padded
//      key/index array. The CPU dispatches this kernel O(log^2 N) times with the
//      (k, j) stage parameters. Bitonic sort is used because it is a simple,
//      in-place, branch-uniform network that maps cleanly to the GPU and needs no
//      scratch buffers.
//

#include <metal_stdlib>
using namespace metal;

struct GPUSplat {
    packed_float3 position;
    packed_float3 cov3d_a;
    packed_float3 cov3d_b;
    packed_float4 color;
};

struct SortEntry {
    float key;
    uint index;
};

struct KeyParams {
    float4x4 view;
    uint splatCount;
    uint paddedCount;
};

struct SortParams {
    uint k;
    uint j;
    uint paddedCount;
    uint _pad;
};

constant float kFarKey = 3.0e38;  // ~FLT_MAX; keeps padding at the tail.

kernel void computeDepthKeys(const device GPUSplat* splats [[buffer(0)]],
                             device SortEntry* entries [[buffer(1)]],
                             constant KeyParams& p [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.paddedCount) { return; }
    if (gid >= p.splatCount) {
        entries[gid].key = kFarKey;
        entries[gid].index = 0;
        return;
    }
    const float3 pos = float3(splats[gid].position);
    const float z = (p.view * float4(pos, 1.0)).z;  // negative in front
    entries[gid].key = z;
    entries[gid].index = gid;
}

kernel void bitonicSortStep(device SortEntry* entries [[buffer(0)]],
                            constant SortParams& sp [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint i = gid;
    if (i >= sp.paddedCount) { return; }

    const uint ixj = i ^ sp.j;
    if (ixj <= i) { return; }  // handle each pair once

    // Ascending sub-sequence when the k-bit of i is 0, descending otherwise.
    const bool ascending = ((i & sp.k) == 0u);

    const SortEntry a = entries[i];
    const SortEntry b = entries[ixj];

    // Swap so that the pair matches the (ascending) direction of its block.
    const bool needSwap = (a.key > b.key) == ascending;
    if (needSwap) {
        entries[i] = b;
        entries[ixj] = a;
    }
}
