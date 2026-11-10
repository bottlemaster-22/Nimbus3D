"""Tile-instance cost as a function of a UNIFORM splat-size multiplier.

Measured on the real trained model and the real refined poses, with the
validated offline copy of trainer_preprocess. Reports both directions so the
elasticity can be read where it matters (m < 1, i.e. SMALLER splats).
"""
import io, json, os, sys
import numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx = (rw + P.TILE_W - 1) // P.TILE_W
ty = (rh + P.TILE_H - 1) // P.TILE_H
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())
sel = keys[::max(1, len(keys)//24)][:24]
print('splats %d  render %dx%d  tiles %dx%d  views %d' % (count, rw, rh, tx, ty, len(sel)))
print('census peakTileInstances %d at %d splats = %.4f/splat'
      % (sl['peakTileInstances'], sl['splatCountAtPeakTileInstances'],
         sl['peakTileInstances']/sl['splatCountAtPeakTileInstances']))

mults = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.4142]
base = None
rows = []
for m in mults:
    c2 = dict(col)
    lm = np.log(m)
    for k in ('scale_0','scale_1','scale_2'):
        c2[k] = (col[k].astype(np.float64) + lm).astype(np.float32)
    tot = []
    vis = []
    for f in sel:
        rot, t = poses[f]; R = P.quat_to_matrix(rot)
        ok, rad, tiles, ex, ey = P.project(c2, count, R, t, fx, fy, cx, cy, tx, ty)
        tot.append(int(tiles.sum()))
        vis.append(int(ok.sum()))
    tot = np.array(tot, float); vis = np.array(vis, float)
    med = np.median(tot)
    rows.append((m, med, np.median(tot/np.maximum(vis,1)), np.median(vis)))
    if m == 1.00: base = med
print()
print('  mult   median tiles/view   vs m=1     tiles/visible  visible')
for m, med, per, v in rows:
    print('  %6.4f  %14.0f   %8.4fx   %10.4f  %8.0f' % (m, med, med/base, per, v))
print()
# elasticity on each interval
print('local elasticity d(log tiles)/d(log size):')
for i in range(len(rows)-1):
    m0, t0 = rows[i][0], rows[i][1]
    m1, t1 = rows[i+1][0], rows[i+1][1]
    print('  %.4f -> %.4f : %.3f' % (m0, m1, np.log(t1/t0)/np.log(m1/m0)))
