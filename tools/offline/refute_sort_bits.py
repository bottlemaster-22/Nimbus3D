"""Empirical, not arithmetic: build the ACTUAL 32-bit sort keys for real
views and report, per 4-bit radix pass, how many distinct digit values
occur. A pass is a genuine no-op only if every key has the same digit.
Also sweeps splat scale to show how instance count (= sort work) moves
with the size fix that is the project's actual direction.
"""
import io, json, os, numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()
keys_idx = sorted(poses.keys())

def tile_ids_for_view(col, idx, scale_mul=1.0):
    """Re-derive the per-splat tile box and enumerate the tile ids it covers."""
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    c = dict(col)
    if scale_mul != 1.0:
        lg = np.log(scale_mul)
        for k in ('scale_0','scale_1','scale_2'):
            c[k] = col[k] + np.float32(lg)
    ok, radius, tiles, ex, ey = P.project(c, count, R, t, fx, fy, cx, cy, tx, ty)
    return ok, tiles

# ---- 1. per-pass digit occupancy on the real key encoding -------------------
print('=== pass occupancy, encoding (tile << 16) | depth16, tiles %dx%d = %d ==='
      % (tx, ty, tx*ty))
for idx in [4, 25, keys_idx[len(keys_idx)//2]]:
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    # rebuild the boxes to enumerate tile ids
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:,2]; inv = 1.0/z
    mx = fx*cam[:,0]*inv + cx; my = fy*cam[:,1]*inv + cy
    minx = np.maximum(0, np.floor((mx-ex)/16)).astype(np.int64)
    miny = np.maximum(0, np.floor((my-ey)/16)).astype(np.int64)
    maxx = np.minimum(tx, np.ceil((mx+ex)/16)).astype(np.int64)
    maxy = np.minimum(ty, np.ceil((my+ey)/16)).astype(np.int64)
    sel = ok & (tiles>0)
    # histogram of tile ids, weighted, without materialising 1M instances
    hist = np.zeros(tx*ty, dtype=np.int64)
    a,b,cc,dd = minx[sel], miny[sel], maxx[sel], maxy[sel]
    for i in range(len(a)):
        for yy in range(b[i], dd[i]):
            hist[yy*tx + a[i] : yy*tx + cc[i]] += 1
    n = hist.sum()
    tid = np.arange(tx*ty)
    key_hi = tid << 16
    print('frame %d: %d instances over %d occupied tiles'
          % (idx, n, int((hist>0).sum())))
    for p in range(8):
        dig = (key_hi >> (4*p)) & 0xF
        present = np.unique(dig[hist>0])
        # depth occupies bits 0..15 -> passes 0..3 always multi-valued
        tag = 'DEPTH' if p < 4 else 'tile'
        print('   pass %d bits %2d-%2d (%s): %d distinct digit values among occupied tiles%s'
              % (p, 4*p, 4*p+3, tag, len(present) if p>=4 else -1,
                 '   <-- NO-OP' if (p>=4 and len(present)==1) else ''))
    break_after = True
    print()

# ---- 2. how sort work scales with splat size -------------------------------
print('=== instance count (== sort element count) vs splat scale ===')
sample = keys_idx[::40]
for mul in (1.0, 0.75, 0.5, 0.35, 0.1729):
    tot = []
    for idx in sample:
        ok, tiles = tile_ids_for_view(col, idx, mul)
        tot.append(int(tiles.sum()))
    a = np.array(tot, float)
    print('  scale x%.4f -> mean instances %9.0f  (%.2fx of today)   median largest axis ~%.2f mm'
          % (mul, a.mean(), a.mean()/np.array([0.0]).sum() if False else a.mean()/786967.0, 19.6*mul))
