"""Our trainer. gsplat supplies rasterisation kernels; the logic is ours.

CONTRACTS.md rule 3: this is not a wrapper around Brush, Nerfstudio, Postshot
or LichtFeld. ``gsplat.rasterization`` is used the way cuBLAS is used - a
library of kernels underneath our own optimiser, densification policy, losses,
pose refinement and background model.

Modules that need **no** torch (they are numpy and run anywhere)::

    geometry      SE(3) / quaternion math, projection, the ARKit convention
    init_lidar    F: normal-aligned disc initialisation from the LiDAR cloud
    pose_graph    F1: submaps, revisits, pose-graph GN, time-offset sweep
    carving       F2: free-space occupancy grid and the prune it licenses
    depth_edges   F3: native-resolution depth edges and the dilated band
    trust         F6: coarse bias field, per-sample noise, decay schedule
    partition     F7: co-visibility clustering for whole-house mode

Modules that **do** need torch (+ gsplat for the rasteriser)::

    pose_delta    F1: constrained per-camera SE(3) delta, spline in time
    densify       F4: AbsGS gradients, budget-capped MCMC relocation
    background    F5: direction-only far field, three-regime authority
    prompt_depth  F5: Prompt Depth Anything mid-field (real, weights on first use)
    train         the optimisation loop that ties them together

:func:`nimbus_booster.trainer.torch_support.probe` reports which half is
available, in plain language, without importing torch.
"""

from __future__ import annotations

__all__ = []
