"""8x8 vs 16x16, measured on the real inner loop rather than a (D+T)^2 model.

Entry 13 rejected 8x8 with: "8x8 does cut inner-loop tests by 23 percent ...
The one thing that would change this is the actual distribution of
draws[].radiusPx, which nothing measures."

This measures it. For a sample of screen tiles it walks the ACTUAL front-to-back
loop of trainer_rasterize_forward at both tile sizes: the same depth order, the
same pad0 cutoff, the same `testT < 1e-4` early termination, and counts the loop
iterations each pixel really executes.
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

def prep(frame):
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    tx, ty = (RW + 15)//16, (RH + 15)//16
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    g = Rst.geometry(col, count, R, t, fx, fy, cx, cy)
    return ok, ex, ey, g

def walk(sel, g, x0, y0, T):
    """One tile. Returns (executed_tests, exp_evals, contributing) summed over pixels."""
    if sel.size == 0:
        return 0, 0, 0
    order = sel[np.argsort(g['z'][sel], kind='stable')]
    px = x0 + np.arange(T) + 0.5
    py = y0 + np.arange(T) + 0.5
    gx, gy = np.meshgrid(px, py)
    gx, gy = gx.ravel(), gy.ravel()
    inside = (gx < RW) & (gy < RH)
    dx = gx[:, None] - g['mx'][order][None, :]
    dy = gy[:, None] - g['my'][order][None, :]
    power = -0.5*(g['cx'][order][None, :]*dx*dx + g['cz'][order][None, :]*dy*dy) \
            - g['cy'][order][None, :]*dx*dy
    op = g['opacity'][order][None, :]
    cutoff = np.log(np.maximum(P.MIN_ALPHA, 1e-8)/np.maximum(op, 1e-8)) - 0.01
    reach_exp = power >= cutoff                       # survives the pad0 compare
    alpha = np.minimum(0.99, op*np.exp(np.clip(power, -60, 0)))
    keep = np.where(reach_exp & (alpha >= P.MIN_ALPHA), alpha, 0.0)
    Tr = np.cumprod(1.0 - keep, axis=1)
    # `done` fires on the FIRST j whose testT < 1e-4; that j is executed, later ones are not.
    dead = Tr < 1e-4
    first = np.where(dead.any(axis=1), dead.argmax(axis=1), keep.shape[1]-1)
    n = first + 1                                     # loop iterations executed
    n = np.where(inside, n, 0)
    m = np.arange(keep.shape[1])[None, :] < n[:, None]
    return int(n.sum()), int((reach_exp & m).sum()), int(((keep > 0) & m).sum())

rng = np.random.RandomState(7)
tot = {8: [0,0,0,0], 16: [0,0,0,0]}
for frame in FRAMES:
    ok, ex, ey, g = prep(frame)
    idx = np.nonzero(ok)[0]
    mx, my = g['mx'], g['my']
    tx, ty = (RW+15)//16, (RH+15)//16
    picks = rng.choice(tx*ty, size=min(NT, tx*ty), replace=False)
    for p in picks:
        bx, by = int(p % tx)*16, int(p // tx)*16
        for T in (16, 8):
            for oy in range(0, 16, T):
                for ox in range(0, 16, T):
                    x0, y0 = bx+ox, by+oy
                    a = np.maximum(0, np.floor((mx[idx]-ex[idx])/T))
                    b = np.minimum((RW+T-1)//T, np.ceil((mx[idx]+ex[idx])/T))
                    c = np.maximum(0, np.floor((my[idx]-ey[idx])/T))
                    d = np.minimum((RH+T-1)//T, np.ceil((my[idx]+ey[idx])/T))
                    ti, tj = x0//T, y0//T
                    sel = idx[(a <= ti) & (b > ti) & (c <= tj) & (d > tj)]
                    n, e, k = walk(sel, g, x0, y0, T)
                    tot[T][0] += n; tot[T][1] += e; tot[T][2] += k; tot[T][3] += sel.size

print('frames %s, %d sampled 16x16 tiles per frame, %dx%d render'
      % (FRAMES, NT, RW, RH))
print()
print('  T   list entries   loop iters executed   reach exp   contributing   iters/entry')
for T in (16, 8):
    n, e, k, s = tot[T]
    print('%3d %14d %21d %11d %14d %12.1f' % (T, s, n, e, k, n/max(s,1)))
n16, e16, k16, s16 = tot[16]
n8, e8, k8, s8 = tot[8]
print()
print('8x8 relative to 16x16 on the SAME screen area:')
print('  loop iterations executed   %.4f   (%.1f%% fewer)' % (n8/n16, 100*(1-n8/n16)))
print('  exp() evaluations          %.4f' % (e8/e16))
print('  contributing pairs         %.6f   (must be 1.000000, the image is identical)' % (k8/k16))
print('  tile-list entries (=instances, sort+gather work)  %.4f' % (s8/s16))
