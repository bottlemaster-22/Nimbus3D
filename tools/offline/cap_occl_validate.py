"""Does the cheap z-buffer occlusion test agree with the real front-to-back
compositor? Compares, on sampled tiles of sampled keyframes, the z-buffer's
`front` flag against the max blending weight alpha*T the splat actually gets.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

TILE = 16
STAMP = 2
Z_FLOOR = 0.02
WEIGHT_CUT = 0.01     # a splat "contributes" if some pixel gives it >= 1% weight

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
b, r, frames, pool, kf, info = keyframes()
kfidx = [f['index'] for f in kf]

rng = np.random.RandomState(3)
views = [kfidx[i] for i in rng.choice(len(kfidx), 6, replace=False)]
agree = tp = fp = fn = tn = 0
for fi in views:
    rot, tv = poses[fi]
    R = P.quat_to_matrix(rot)
    ok, rad, tiles, ex, ey = P.project(col, count, R, tv, fx, fy, cx, cy, tx, ty)
    g = __import__('raster').geometry(col, count, R, tv, fx, fy, cx, cy)
    z, mx, my = g['z'], g['mx'], g['my']
    idx = np.flatnonzero(ok)

    # --- the cheap z-buffer, identical to cap_occlusion.py -----------------
    scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                    axis=1).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q/np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1-2*(y_*y_+z_*z_); Rm[:, 0, 1] = 2*(x_*y_-w_*z_); Rm[:, 0, 2] = 2*(x_*z_+w_*y_)
    Rm[:, 1, 0] = 2*(x_*y_+w_*z_); Rm[:, 1, 1] = 1-2*(x_*x_+z_*z_); Rm[:, 1, 2] = 2*(y_*z_-w_*x_)
    Rm[:, 2, 0] = 2*(x_*z_-w_*y_); Rm[:, 2, 1] = 2*(y_*z_+w_*x_); Rm[:, 2, 2] = 1-2*(x_*x_+y_*y_)
    M = Rm*scale[:, None, :]
    sw = M @ np.transpose(M, (0, 2, 1))
    rz = R[2]
    szz = np.einsum('i,nij,j->n', rz, sw[idx], rz)
    tol = 3.0*np.sqrt(np.maximum(szz, 1e-12)) + Z_FLOOR
    zb = np.full(rw*rh, np.inf)
    pxi = np.round(mx[idx]).astype(np.int64); pyi = np.round(my[idx]).astype(np.int64)
    order = np.argsort(-z[idx])
    for dy in range(-STAMP, STAMP+1):
        for dx in range(-STAMP, STAMP+1):
            X = pxi[order]+dx; Y = pyi[order]+dy
            m = (X >= 0) & (X < rw) & (Y >= 0) & (Y < rh)
            zb[Y[m]*rw+X[m]] = z[idx][order][m]
    m = (pxi >= 0) & (pxi < rw) & (pyi >= 0) & (pyi < rh)
    front = np.zeros(count, dtype=bool)
    front[idx[m]] = zb[pyi[m]*rw+pxi[m]] >= (z[idx][m]-tol[m])

    # --- the real compositor over sampled tiles ---------------------------
    min_x = np.maximum(0, np.floor((mx-ex)/TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my-ey)/TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx+ex)/TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my+ey)/TILE)).astype(np.int64)
    picks = rng.choice(tx*ty, size=90, replace=False)
    for p in picks:
        tX, tY = int(p % tx), int(p//tx)
        sel = idx[(min_x[idx] <= tX) & (max_x[idx] > tX)
                  & (min_y[idx] <= tY) & (max_y[idx] > tY)]
        if sel.size == 0:
            continue
        o = sel[np.argsort(z[sel])]
        px = tX*TILE+np.arange(TILE)+0.5
        py = tY*TILE+np.arange(TILE)+0.5
        gx, gy = np.meshgrid(px, py); gx = gx.ravel(); gy = gy.ravel()
        dx = gx[:, None]-mx[o][None, :]; dy = gy[:, None]-my[o][None, :]
        power = -0.5*(g['cx'][o][None, :]*dx*dx + g['cz'][o][None, :]*dy*dy) \
            - g['cy'][o][None, :]*dx*dy
        alpha = np.where(power <= 0, np.minimum(0.99, g['opacity'][o][None, :]
                                                * np.exp(np.clip(power, -60, 0))), 0.0)
        alpha = np.where(alpha >= P.MIN_ALPHA, alpha, 0.0)
        T = np.cumprod(1.0-alpha, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        w = (alpha*before).max(axis=0)
        # only judge splats whose CENTRE is inside this tile: that is the pixel
        # the z-buffer test used.
        own = (np.floor(mx[o]/TILE).astype(int) == tX) & (np.floor(my[o]/TILE).astype(int) == tY)
        truth = w >= WEIGHT_CUT
        pred = front[o]
        s = own
        tp += int((pred[s] & truth[s]).sum()); tn += int((~pred[s] & ~truth[s]).sum())
        fp += int((pred[s] & ~truth[s]).sum()); fn += int((~pred[s] & truth[s]).sum())

n = tp+tn+fp+fn
print('validated on %d views x 90 tiles, %d splats judged' % (len(views), n))
print('  agreement                 %.2f%%' % (100*(tp+tn)/n))
print('  z-buffer says FRONT and compositor agrees (TP) %6d' % tp)
print('  z-buffer says OCCLUDED and compositor agrees (TN) %6d' % tn)
print('  z-buffer FRONT, compositor says no weight (FP)  %6d' % fp)
print('  z-buffer OCCLUDED, compositor says it counts (FN) %6d' % fn)
print('  compositor: %.2f%% of centre-owning splats contribute; z-buffer says %.2f%%'
      % (100*(tp+fn)/n, 100*(tp+fp)/n))
