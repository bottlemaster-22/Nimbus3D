"""Shrinking splats at fixed SURFACE COVERAGE, not at fixed population.

At fixed population, smaller splats are cheaper - that is already established.
But sizing seeds from image detail is a coverage-preserving change: a patch
held by one splat of radius r is held by (r/r')^2 splats at r'. So the raster
cost of the whole field is  (r/r')^2 * tileInstances(r'), and that is what
this measures, by scaling the real model and re-running project.py's own tile
arithmetic.
"""
import io, json, os
import numpy as np
import project as P

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                           encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
frames = sorted(poses)[:120:15]
s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)

print(' scale   median largest axis   splats for same coverage   tile instances'
      '   per splat   total vs today')
base = None
for k in (2.0, 1.0, 0.5, 0.25, 0.125):
    c2 = dict(col)
    s2 = s + np.log(k)
    c2['scale_0'], c2['scale_1'], c2['scale_2'] = s2[:, 0], s2[:, 1], s2[:, 2]
    ti = 0; vis = 0
    for idx in frames:
        rot, t = poses[idx]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(c2, count, R, t, fx, fy, cx, cy, tx, ty)
        ti += int(tiles.sum()); vis += int(ok.sum())
    total = ti / (k * k)                     # (r/r')^2 copies of the field
    if base is None and k == 2.0:
        pass
    if k == 1.0:
        base = total
    print('%6.3f  %19.2f   %24.0f   %14d   %9.2f   %s'
          % (k, 1000 * k * np.median(np.exp(s).max(1)), count / (k * k), ti,
             ti / max(vis, 1),
             ('%.2fx' % (total / base)) if base else '-'))
