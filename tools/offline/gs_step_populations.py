"""Per-iteration populations the gpuStep cost model needs, measured on the
build-250 model over every training keyframe (not a sample of 4).

  D  splats drawn (tilesTouched > 0)      -> preprocess_backward, adam, sparse rows
  I  tile instances (sum of tilesTouched) -> duplicate_keys, radix sort, tile_ranges,
                                             forward/backward staging loads

Validation: max(I) over the keyframes should land near the census
peakTileInstances (498,745). The exported model is the iteration-3600 best
cloud with the 3D filter fused, so expect a few per cent of drift.
"""
import io
import json
import os

import numpy as np

import project as P

D_ = P.D
census = json.load(io.open(os.path.join(D_, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
TX, TY = (RW + 15) // 16, (RH + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(D_, 'model', 'model.ply'))
poses = P.load_poses()

kf = census.get('keyframesSelected')
if isinstance(kf, list) and kf and isinstance(kf[0], (int, float)):
    frames = [int(k) for k in kf if int(k) in poses]
    src = 'census keyframesSelected (%d)' % len(frames)
else:
    frames = sorted(poses.keys())
    src = 'all refined poses (%d); keyframesSelected=%r' % (len(frames), type(kf).__name__)

rows = []
for f in frames:
    rot, t = poses[f]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, TX, TY)
    rows.append((f, int(ok.sum()), int(tiles.sum())))

a = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
print('frames:', src)
print('splats in model %d, render %dx%d, tiles %dx%d = %d' % (count, RW, RH, TX, TY, TX * TY))
for name, v in (('drawn D', a[:, 0]), ('instances I', a[:, 1])):
    print('%-12s mean %9.0f  median %9.0f  p90 %9.0f  max %9.0f'
          % (name, v.mean(), np.median(v), np.percentile(v, 90), v.max()))
print('census peakTileInstances %d  (offline max / census = %.3f)'
      % (sl['peakTileInstances'], a[:, 1].max() / sl['peakTileInstances']))
print('instances per drawn splat mean %.2f' % (a[:, 1].sum() / a[:, 0].sum()))
print('drawn fraction of population  %.3f' % (a[:, 0].mean() / count))
