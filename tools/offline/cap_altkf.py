"""Same budget, whole walk. What does the capture offer that the trainer's
keyframe selector never asked for?

Set A: the 120 keyframes the trainer actually used (greedy, stops at frame 485).
Set B: 120 frames at an even stride over all 868 usable frames.
Everything else identical: same occlusion test, same model, same budget.
"""
import io, json, os, time
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP, SLACK = 2, 0.10
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
b, r, frames, pool, kf, info = keyframes()
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)

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


def az(v):
    return np.degrees(np.arctan2(v[..., 2], v[..., 0])) % 360


def evaluate(frame_list, label):
    n = len(frame_list)
    vis = np.zeros((count, n), bool)
    azsc = np.zeros((count, n), np.float32)
    t0 = time.time()
    for j, fi in enumerate(frame_list):
        rot, tv = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, rad, tiles, ex, ey = P.project(col, count, R, tv, fx, fy, cx, cy, tx, ty)
        cam = mean @ R.T + tv
        z = cam[:, 2]
        inv = 1.0/np.where(np.abs(z) > 1e-9, z, 1e-9)
        mx = fx*cam[:, 0]*inv+cx; my = fy*cam[:, 1]*inv+cy
        idx = np.flatnonzero(ok)
        szz = np.einsum('i,nij,j->n', R[2], SW[idx], R[2])
        tol = 3.0*np.sqrt(np.maximum(szz, 1e-12))+SLACK
        zb = np.full(rw*rh, np.inf)
        pxi = np.round(mx[idx]).astype(np.int64); pyi = np.round(my[idx]).astype(np.int64)
        o = np.argsort(-z[idx])
        for dy in range(-STAMP, STAMP+1):
            for dx in range(-STAMP, STAMP+1):
                X = pxi[o]+dx; Y = pyi[o]+dy
                m = (X >= 0) & (X < rw) & (Y >= 0) & (Y < rh)
                zb[Y[m]*rw+X[m]] = z[idx][o][m]
        m = (pxi >= 0) & (pxi < rw) & (pyi >= 0) & (pyi < rh)
        fr = np.zeros(count, bool)
        fr[idx[m]] = zb[pyi[m]*rw+pxi[m]] >= (z[idx][m]-tol[m])
        vis[:, j] = fr
        C = -R.T @ tv
        azsc[:, j] = az(C - mean)
    nv = vis.sum(axis=1)
    BW = 10.0; nb = 36
    bins = (azsc/BW).astype(np.int16)
    cover = np.zeros((count, nb), bool)
    for j in range(n):
        mm = vis[:, j]
        cover[np.flatnonzero(mm), bins[mm, j]] = True
    nd = cover.sum(axis=1)
    print('--- %s (%d views, %.0fs) ---' % (label, n, time.time()-t0))
    print('  occlusion-aware views per splat: p10 %d  MEDIAN %d  p90 %d  mean %.2f'
          % (np.percentile(nv, 10), np.median(nv), np.percentile(nv, 90), nv.mean()))
    print('  never seen %.2f%%   < 2 views %.2f%%   < 3 views %.2f%%'
          % (100*(nv == 0).mean(), 100*(nv < 2).mean(), 100*(nv < 3).mean()))
    print('  distinct 10-deg azimuth sectors: p10 %d  MEDIAN %d  p90 %d  mean %.2f'
          % (np.percentile(nd, 10), np.median(nd), np.percentile(nd, 90), nd.mean()))
    print('  splats seen from <=1 sector %.2f%%  <=3 sectors %.2f%%'
          % (100*(nd <= 1).mean(), 100*(nd <= 3).mean()))
    F = np.array([-P.quat_to_matrix(poses[fi][0]).T @ np.array([0., 0., -1.]) for fi in frame_list])
    a = az(F)
    h, _ = np.histogram(a, bins=12, range=(0, 360))
    print('  optical-axis azimuth, 30-deg bins: %s' % ' '.join('%3d' % v for v in h))
    print('  empty 30-deg azimuth bins: %d of 12' % (h == 0).sum())
    return nv, nd


kfidx = [f['index'] for f in kf]
allidx = [f['index'] for f in pool]
step = len(allidx)/120.0
alt = [allidx[int(i*step)] for i in range(120)]
print('set A last frame %d ; set B last frame %d\n' % (kfidx[-1], alt[-1]))
nvA, ndA = evaluate(kfidx, 'A: the 120 keyframes the trainer used')
print()
nvB, ndB = evaluate(alt, 'B: 120 frames evenly strided over the whole walk')
print('\n=== A -> B ===')
print('  mean views per splat      %.2f -> %.2f  (%+.1f%%)'
      % (nvA.mean(), nvB.mean(), 100*(nvB.mean()/nvA.mean()-1)))
print('  mean azimuth sectors      %.2f -> %.2f  (%+.1f%%)'
      % (ndA.mean(), ndB.mean(), 100*(ndB.mean()/ndA.mean()-1)))
print('  splats gaining >=1 view   %.1f%%' % (100*(nvB > nvA).mean()))
print('  splats gaining >=1 sector %.1f%%' % (100*(ndB > ndA).mean()))
np.save(os.path.join(OUT, '_cap_nvA.npy'), nvA); np.save(os.path.join(OUT, '_cap_nvB.npy'), nvB)
np.save(os.path.join(OUT, '_cap_ndA.npy'), ndA); np.save(os.path.join(OUT, '_cap_ndB.npy'), ndB)
