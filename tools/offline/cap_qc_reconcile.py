"""Why does the QC card say 10.28 deg when the splats say ~33?

Emulates PrePassSurveyor exactly on the frame set it actually used: 49 frames
taken at an even stride over all 868, angle measured from the FIRST of those
frames to hit each 10 cm voxel. Then adds back one difference at a time.
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
mean_all = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)

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

MAXR = 5.0*0.90   # settings.usableRangeFraction * lidarMaxRangeMeters
MINR = 0.25


def run(frame_list, label, range_gate=True):
    nv = len(frame_list)
    vis = np.zeros((count, nv), bool)
    rng_m = np.zeros((count, nv), np.float32)
    dirs = np.zeros((count, nv, 3), np.float32)
    for j, fi in enumerate(frame_list):
        rot, tv = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, rad, tiles, ex, ey = P.project(col, count, R, tv, fx, fy, cx, cy, tx, ty)
        cam = mean_all @ R.T + tv
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
        C = -R.T @ tv
        v = mean_all - C
        d = np.linalg.norm(v, axis=1)
        if range_gate:
            fr &= (d >= MINR) & (d <= MAXR)
        vis[:, j] = fr
        rng_m[:, j] = d
        dirs[:, j, :] = (v/d[:, None]).astype(np.float32)
    # per 10 cm voxel, max angle from the first frame in list order to hit it
    key = np.floor(mean_all/0.10).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nvox = inv.max()+1
    first = np.zeros((nvox, 3)); hf = np.zeros(nvox, bool)
    spread = np.zeros(nvox); hits = np.zeros(nvox, np.int64)
    distinct = np.zeros(nvox, np.int64)
    for j in range(nv):
        m = vis[:, j]
        if not m.any():
            continue
        vi = inv[m]; d = dirs[m, j].astype(np.float64)
        new = ~hf[vi]
        first[vi[new]] = d[new]; hf[vi[new]] = True
        np.add.at(hits, vi, 1)
        touched = np.unique(vi); distinct[touched] += 1
        a = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', d, first[vi]), -1, 1)))
        np.maximum.at(spread, vi, a)
    use = distinct >= 2
    print('%-52s voxels %6d (>=2 frames %6d)  spread p25 %5.2f MEDIAN %5.2f p75 %5.2f'
          % (label, nvox, use.sum(), *np.percentile(spread[use], [25, 50, 75])))
    return spread, distinct, use


allf = sorted(poses.keys())
step = max(1, len(allf)//48)
survey49 = allf[::step]
print('survey frame set: %d frames, stride %d, first %d last %d'
      % (len(survey49), step, survey49[0], survey49[-1]))
kfidx = [f['index'] for f in kf]
t0 = time.time()
run(kfidx, '120 trainer keyframes, range gated 0.25-4.5 m')
run(survey49, '49 stride frames over all 868 (what the QC card used)')
run(kfidx, '120 trainer keyframes, NO range gate', range_gate=False)
print('elapsed %.0fs' % (time.time()-t0))
