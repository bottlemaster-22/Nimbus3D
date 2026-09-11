"""Calibrate the cheap z-buffer occlusion test against the exact front-to-back
compositor, sweeping stamp radius and depth slack."""
import io, json, os, itertools
import numpy as np
import project as P
import raster
from cap_keyframes import keyframes

TILE = 16
WEIGHT_CUT = 0.01
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
b, r, frames, pool, kf, info = keyframes()
kfidx = [f['index'] for f in kf]

scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], axis=1).astype(np.float64)
q = q/np.linalg.norm(q, axis=1, keepdims=True)
x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
Rm = np.empty((count, 3, 3))
Rm[:, 0, 0] = 1-2*(y_*y_+z_*z_); Rm[:, 0, 1] = 2*(x_*y_-w_*z_); Rm[:, 0, 2] = 2*(x_*z_+w_*y_)
Rm[:, 1, 0] = 2*(x_*y_+w_*z_); Rm[:, 1, 1] = 1-2*(x_*x_+z_*z_); Rm[:, 1, 2] = 2*(y_*z_-w_*x_)
Rm[:, 2, 0] = 2*(x_*z_-w_*y_); Rm[:, 2, 1] = 2*(y_*z_+w_*x_); Rm[:, 2, 2] = 1-2*(x_*x_+y_*y_)
M = Rm*scale[:, None, :]
SW = M @ np.transpose(M, (0, 2, 1))

rng = np.random.RandomState(3)
views = [kfidx[i] for i in rng.choice(len(kfidx), 6, replace=False)]

cache = []
for fi in views:
    rot, tv = poses[fi]
    R = P.quat_to_matrix(rot)
    ok, rad, tiles, ex, ey = P.project(col, count, R, tv, fx, fy, cx, cy, tx, ty)
    g = raster.geometry(col, count, R, tv, fx, fy, cx, cy)
    idx = np.flatnonzero(ok)
    szz = np.einsum('i,nij,j->n', R[2], SW[idx], R[2])
    min_x = np.maximum(0, np.floor((g['mx']-ex)/TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((g['my']-ey)/TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((g['mx']+ex)/TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((g['my']+ey)/TILE)).astype(np.int64)
    picks = rng.choice(tx*ty, size=90, replace=False)
    truth = {}
    for p in picks:
        tX, tY = int(p % tx), int(p//tx)
        sel = idx[(min_x[idx] <= tX) & (max_x[idx] > tX) & (min_y[idx] <= tY) & (max_y[idx] > tY)]
        if sel.size == 0:
            continue
        o = sel[np.argsort(g['z'][sel])]
        px = tX*TILE+np.arange(TILE)+0.5; py = tY*TILE+np.arange(TILE)+0.5
        gx, gy = np.meshgrid(px, py); gx = gx.ravel(); gy = gy.ravel()
        dx = gx[:, None]-g['mx'][o][None, :]; dy = gy[:, None]-g['my'][o][None, :]
        power = -0.5*(g['cx'][o][None, :]*dx*dx+g['cz'][o][None, :]*dy*dy)-g['cy'][o][None, :]*dx*dy
        alpha = np.where(power <= 0, np.minimum(0.99, g['opacity'][o][None, :]
                                                * np.exp(np.clip(power, -60, 0))), 0.0)
        alpha = np.where(alpha >= P.MIN_ALPHA, alpha, 0.0)
        T = np.cumprod(1.0-alpha, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        w = (alpha*before).max(axis=0)
        own = (np.floor(g['mx'][o]/TILE).astype(int) == tX) & (np.floor(g['my'][o]/TILE).astype(int) == tY)
        for k, s in zip(o[own], (w >= WEIGHT_CUT)[own]):
            truth[int(k)] = bool(s)
    cache.append((g, idx, szz, truth))

print('%-6s %-8s %8s %8s %8s' % ('stamp', 'slack', 'agree%', 'pred+%', 'true+%'))
for STAMP, SLACK in itertools.product([0, 1, 2], [0.02, 0.05, 0.10, 0.20]):
    tp = tn = fp = fn = 0
    for g, idx, szz, truth in cache:
        tol = 3.0*np.sqrt(np.maximum(szz, 1e-12))+SLACK
        zb = np.full(rw*rh, np.inf)
        pxi = np.round(g['mx'][idx]).astype(np.int64); pyi = np.round(g['my'][idx]).astype(np.int64)
        order = np.argsort(-g['z'][idx])
        for dy in range(-STAMP, STAMP+1):
            for dx in range(-STAMP, STAMP+1):
                X = pxi[order]+dx; Y = pyi[order]+dy
                m = (X >= 0) & (X < rw) & (Y >= 0) & (Y < rh)
                zb[Y[m]*rw+X[m]] = g['z'][idx][order][m]
        m = (pxi >= 0) & (pxi < rw) & (pyi >= 0) & (pyi < rh)
        front = np.zeros(count, dtype=bool)
        front[idx[m]] = zb[pyi[m]*rw+pxi[m]] >= (g['z'][idx][m]-tol[m])
        ks = np.fromiter(truth.keys(), dtype=np.int64)
        vs = np.fromiter(truth.values(), dtype=bool)
        pr = front[ks]
        tp += int((pr & vs).sum()); tn += int((~pr & ~vs).sum())
        fp += int((pr & ~vs).sum()); fn += int((~pr & vs).sum())
    n = tp+tn+fp+fn
    print('%-6d %-8.2f %8.2f %8.2f %8.2f'
          % (STAMP, SLACK, 100*(tp+tn)/n, 100*(tp+fp)/n, 100*(tp+fn)/n))
