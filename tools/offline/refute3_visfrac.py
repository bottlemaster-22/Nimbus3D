"""How many splats are VISIBLE (tilesTouched>0) per iteration?

trainer_preprocess writes visibleFlag=1 on exactly the same path that writes
tilesTouched=touched>0 (TrainerShaders.metal:1126 and :1124), and
resetVisibility zeroes it every iteration (:623).  So
    visibleFlag == 1  <=>  tilesTouched > 0
exactly, every iteration.

Everything gated on visibleFlag (trainer_adam_splat :2767, trainer_adam_sh
:2852) therefore consumes only this fraction of the per-splat buffers, while
clearPerIteration (TrainerPipelines.swift:941) fills 100% of them and
trainer_regularizer (:2661) computes for 100% of them.
"""
import json, io, os
import numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx = (rw + P.TILE_W - 1)//P.TILE_W
ty = (rh + P.TILE_H - 1)//P.TILE_H
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys(), key=lambda s: int(s))
# 120 keyframes were selected out of 868 frames; sample evenly.
sel = [keys[i] for i in np.linspace(0, len(keys)-1, 120).astype(int)]

vis, ti = [], []
for idx in sel:
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    vis.append(int(ok.sum())); ti.append(int(tiles.sum()))
vis = np.array(vis); ti = np.array(ti)
print('splats in model            %d' % count)
print('views sampled              %d' % len(sel))
print('visible per view  mean     %10.1f  = %.2f%%' % (vis.mean(), 100*vis.mean()/count))
print('                  median   %10.1f  = %.2f%%' % (np.median(vis), 100*np.median(vis)/count))
print('                  p05/p95  %10.1f / %.1f' % (np.percentile(vis,5), np.percentile(vis,95)))
print('                  max      %10d  = %.2f%%' % (vis.max(), 100*vis.max()/count))
print('tile instances    mean     %10.1f   max %d (census peak %d)'
      % (ti.mean(), ti.max(), sl['peakTileInstances']))
