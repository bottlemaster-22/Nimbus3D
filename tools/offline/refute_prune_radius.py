"""REFUTATION TEST for "set pruneMaxScreenRadiusPx to renderLongEdgePixels/4 = 180 px,
which is above the 17-18 px median and below the 1,780 px near shell".

`pruneMaxScreenRadiusPx` is compared against `stats[i].maxRadiusPxBits`
(TrainerDensifier.swift:815), which trainer_preprocess writes with
atomic_fetch_max over EVERY view in the densify interval
(TrainerShaders.metal:1104).  It is a MAX, not a per-view value, and the 3-sigma
radius scales with 1/z.  TrainerSupport.swift:808 already records that this
statistic destroyed `splitScreenRadiusPx`.  So the question is not "what is the
median per-view radius" but "how much of the population records a MAX above 180
over the views a densify interval actually sees".

A densify interval is 100 iterations, one keyframe per iteration, so up to 100
distinct views feed one max.  This sweeps the number of views.
"""
import io, json, os
import numpy as np
import project as P

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + P.TILE_W - 1)//P.TILE_W, (rh + P.TILE_H - 1)//P.TILE_H
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())
print('splats %d, poses available %d, render %dx%d' % (count, len(keys), rw, rh))

for nviews in (1, 4, 12, 40, 100):
    step = max(len(keys)//nviews, 1)
    use = keys[::step][:nviews]
    mx = np.zeros(count)
    seen = np.zeros(count, bool)
    for f in use:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        mx = np.where(ok, np.maximum(mx, radius), mx)
        seen |= ok
    d = mx[seen]
    print('\n--- max 3-sigma screen radius over %3d views (%d splats ever drawn) ---'
          % (len(use), seen.sum()))
    print('    p50 %8.1f px   p90 %8.1f   p99 %8.1f   max %9.1f'
          % tuple(np.percentile(d, [50, 90, 99, 100])))
    for thr in (10, 48, 180, 360, 720):
        print('      a prune at %4d px would delete %8d of %d drawn splats  (%6.2f%%)'
              % (thr, (d > thr).sum(), d.size, 100*(d > thr).mean()))
