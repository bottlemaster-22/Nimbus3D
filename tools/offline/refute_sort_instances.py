"""Distribution of tile instances per view, to test whether 763,260 is a
mean and 1,058,989 is a max. Also the tile-id bit occupancy per pass.
"""
import io, json, os, numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx = (rw + 15) // 16
ty = (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())
print('poses %d  tiles %dx%d = %d  splats %d' % (len(keys), tx, ty, tx*ty, count))

# stride across the WHOLE pose set, not just the first 40
STRIDE = 8
sample = keys[::STRIDE]
print('sampling %d poses with stride %d' % (len(sample), STRIDE))

tot = []
for i, idx in enumerate(sample):
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    tot.append(int(tiles.sum()))
    if i % 20 == 0:
        print('  %d/%d idx %d -> %d' % (i, len(sample), idx, tot[-1]), flush=True)

a = np.array(tot, dtype=np.float64)
print()
print('n=%d  mean %.0f  median %.0f  p10 %.0f  p90 %.0f  min %.0f  max %.0f'
      % (len(a), a.mean(), np.median(a), np.percentile(a,10), np.percentile(a,90), a.min(), a.max()))
print('census peakTileInstances %d' % sl['peakTileInstances'])
print('mean / 763260 = %.3f    max / mean = %.3f' % (a.mean()/763260.0, a.max()/a.mean()))
print('census peak / mean = %.3f' % (sl['peakTileInstances']/a.mean()))
np.save(os.path.join(os.path.dirname(__file__), '_sort_tileinst.npy'), a)
