"""Per-tile list length, and how many staging batches the rasterisers really run.

Entry 21 rejected a finding with "With TRAINER_TILE_AREA = 256 and the claim's
own 174-256 instances per tile, `batches` is 1, so there is no list to shorten."
Measure the distribution at build 248's actual render size and population.
"""
import io, json, os, sys
import numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
tx, ty = (RW+15)//16, (RH+15)//16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())[:16]

lens = []
for k in keys:
    rot, t = poses[k]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    idx = np.nonzero(ok & (tiles > 0))[0]
    a = np.maximum(0, np.floor((P.__dict__ and 0) + 0)) # placeholder
    mx = None
    # recompute screen centres
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    iz = 1.0/cam[:, 2]
    mx = fx*cam[:, 0]*iz + cx
    my = fy*cam[:, 1]*iz + cy
    a = np.maximum(0, np.floor((mx[idx]-ex[idx])/16)).astype(np.int64)
    b = np.minimum(tx, np.ceil((mx[idx]+ex[idx])/16)).astype(np.int64)
    c = np.maximum(0, np.floor((my[idx]-ey[idx])/16)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my[idx]+ey[idx])/16)).astype(np.int64)
    grid = np.zeros(tx*ty, dtype=np.int64)
    for i in range(len(idx)):
        for yy in range(c[i], d[i]):
            grid[yy*tx + a[i]:yy*tx + b[i]] += 1
    lens.append(grid)
L = np.concatenate(lens)
print('16x16 tiles, %d views, %d tile-instances total' % (len(keys), L.sum()))
print('per-tile list length: mean %.1f  p10 %d  p50 %d  p90 %d  p99 %d  max %d'
      % (L.mean(), *np.percentile(L, [10, 50, 90, 99]).astype(int), L.max()))
for A in (256, 512, 1024):
    b = np.ceil(L/A)
    print('  stage %4d/round: mean batches %.3f  barriers/tile %.1f  '
          'tiles paying a round for <=32 entries %.1f%%'
          % (A, b.mean(), 2*b.mean(),
             100*np.mean((b > 1) & ((L - (b-1)*A) <= 32))))
occ = L > 0
print('empty tiles %.1f%%' % (100*(1-occ.mean())))
