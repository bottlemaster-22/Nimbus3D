"""What a SIMD-group reduction of trainer_rasterize_backward's atomics really buys.

Entry 21's recovery claims "the reduction factor is the FULL 32x ... because in
trainer_rasterize_backward every lane of a SIMD group is a different pixel
processing the SAME j, so all live lanes target the identical address".

The address claim is right. The FACTOR is not 32: it is the mean number of
CONTRIBUTING lanes per (SIMD group, j), because a lane that took one of the three
`continue`s issues no atomic today either. Measure it on the real loop.

A 16x16 tile has tid = y*16 + x, so a 32-lane SIMD group is two adjacent rows.
"""
import io, json, os, sys
import numpy as np
import project as P
import raster as Rst

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()

FRAMES = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['0','4','8','12'])]
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 48
T = 16

def prep(frame):
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    tx, ty = (RW + 15)//16, (RH + 15)//16
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    g = Rst.geometry(col, count, R, t, fx, fy, cx, cy)
    return ok, ex, ey, g

rng = np.random.RandomState(7)
atomics_now = 0        # one per contributing (pixel, splat) pair, per field
groups_hot = 0         # (simd group, j) pairs with at least one contributor
hist = np.zeros(33, dtype=np.int64)
tiles_done = 0

for frame in FRAMES:
    ok, ex, ey, g = prep(frame)
    idx = np.nonzero(ok)[0]
    mx, my = g['mx'], g['my']
    tx, ty = (RW+15)//16, (RH+15)//16
    a = np.maximum(0, np.floor((mx[idx]-ex[idx])/T))
    b = np.minimum((RW+T-1)//T, np.ceil((mx[idx]+ex[idx])/T))
    c = np.maximum(0, np.floor((my[idx]-ey[idx])/T))
    d = np.minimum((RH+T-1)//T, np.ceil((my[idx]+ey[idx])/T))
    picks = rng.choice(tx*ty, size=min(NT, tx*ty), replace=False)
    for p in picks:
        ti, tj = int(p % tx), int(p // tx)
        sel = idx[(a <= ti) & (b > ti) & (c <= tj) & (d > tj)]
        if sel.size == 0:
            continue
        order = sel[np.argsort(g['z'][sel], kind='stable')]
        x0, y0 = ti*T, tj*T
        gx, gy = np.meshgrid(x0 + np.arange(T) + 0.5, y0 + np.arange(T) + 0.5)
        gx, gy = gx.ravel(), gy.ravel()          # tid order: y*16 + x
        inside = (gx < RW) & (gy < RH)
        dx = gx[:, None] - g['mx'][order][None, :]
        dy = gy[:, None] - g['my'][order][None, :]
        power = -0.5*(g['cx'][order][None, :]*dx*dx + g['cz'][order][None, :]*dy*dy) \
                - g['cy'][order][None, :]*dx*dy
        op = g['opacity'][order][None, :]
        cutoff = np.log(np.maximum(P.MIN_ALPHA, 1e-8)/np.maximum(op, 1e-8)) - 0.01
        alpha = np.minimum(0.99, op*np.exp(np.clip(power, -60, 0)))
        keep = (power >= cutoff) & (alpha >= P.MIN_ALPHA)
        # forward transmittance decides lastContributor; backward walks j <= it
        kv = np.where(keep, alpha, 0.0)
        Tr = np.cumprod(1.0 - kv, axis=1)
        dead = Tr < 1e-4
        first = np.where(dead.any(axis=1), dead.argmax(axis=1), kv.shape[1]-1)
        m = np.arange(kv.shape[1])[None, :] <= first[:, None]
        contrib = keep & m & inside[:, None]
        # 8 SIMD groups of 32 consecutive tids
        cg = contrib.reshape(8, 32, contrib.shape[1])
        per = cg.sum(axis=1)                      # (group, j) -> contributing lanes
        atomics_now += int(per.sum())
        groups_hot += int((per > 0).sum())
        hist += np.bincount(per.ravel(), minlength=33)
        tiles_done += 1

print('frames %s, %d sampled 16x16 tiles, %dx%d' % (FRAMES, tiles_done, RW, RH))
print('contributing (pixel, splat) pairs      %d' % atomics_now)
print('(SIMD group, j) pairs with >=1 lane    %d' % groups_hot)
print()
print('ATOMICS PER FIELD, per sampled area:')
print('  today (one per contributing lane)    %d' % atomics_now)
print('  with simd_sum (one per hot group)    %d' % groups_hot)
print('  REDUCTION FACTOR                     %.2fx   (not 32x)' % (atomics_now/max(groups_hot,1)))
print()
nz = hist[1:]
print('distribution of contributing lanes per hot (group, j): mean %.2f, median %d, p90 %d'
      % (atomics_now/max(groups_hot,1),
         np.searchsorted(np.cumsum(nz), 0.5*nz.sum())+1,
         np.searchsorted(np.cumsum(nz), 0.9*nz.sum())+1))
print('  32/32 lanes: %.1f%% of hot groups' % (100*hist[32]/max(groups_hot,1)))
print('  <=4 lanes  : %.1f%% of hot groups' % (100*hist[1:5].sum()/max(groups_hot,1)))
