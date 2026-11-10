"""What gradient-driven sizing does to the RASTER, at the same splat count.

A spread of sizes with the same splat count has a LARGER mean area than one
size (Cauchy-Schwarz), so tile instances go up. This measures by how much, on
the real model, with project.py's own tile-box arithmetic.
"""
import io, json, os
import numpy as np
import project as P, detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
FX = Dt.bundle()['intrinsics']['fx']
NSPLAT = 299209
sp = Z['splat'].astype(np.int64)
w = (Z['sig_px'] ** 2).astype(np.float64)
W = np.bincount(sp, weights=w, minlength=NSPLAT)
g = np.bincount(sp, weights=Z['ngrad'] * w, minlength=NSPLAT) / np.maximum(W, 1e-12)
g[W == 0] = np.median(g[W > 0])

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                           encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()

s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
big = np.exp(s).max(1)

# The rule: radius falls with gradient over a chosen dynamic range, then the
# whole field is rescaled so that sum(1/r^2) - the splat count needed to cover
# the surface - is unchanged. Sweep the range.
print('spread   mean largest axis mm   sum(1/r^2) ratio   tile instances (8 frames)   per splat')
frames = sorted(poses)[:120:15]
base = None
for spread in (1.0, 2.0, 4.0, 8.0, 18.0):
    rank = (np.argsort(np.argsort(g)) / (NSPLAT - 1.0))       # 0 flat .. 1 busy
    f = spread ** (0.5 - rank)                               # geometric spread
    f = f * np.sqrt(np.mean(f ** -2))          # keep mean(1/r^2), i.e. the count
    s2 = s + np.log(f)[:, None]
    c2 = dict(col)
    c2['scale_0'], c2['scale_1'], c2['scale_2'] = s2[:, 0], s2[:, 1], s2[:, 2]
    ti = 0; vis = 0
    for idx in frames:
        rot, t = poses[idx]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(c2, count, R, t, fx, fy, cx, cy, tx, ty)
        ti += int(tiles.sum()); vis += int(ok.sum())
    b2 = np.exp(s2).max(1)
    if base is None:
        base = ti
    print('%5.1fx  %18.2f   %16.3f   %22d   %8.2f   %+6.1f%% tile cost'
          % (spread, 1000 * np.median(b2), np.mean(f ** -2), ti, ti / max(vis, 1),
             100 * (ti / base - 1)))
