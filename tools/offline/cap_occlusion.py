"""Occlusion-aware visibility: a Gaussian counts as SEEN by a keyframe only if
nothing else in the model is in front of it at its own pixel.

Depth-buffer built from the model's own splats (the LiDAR depth maps are not on
this machine). Each splat stamps a 5x5 pixel neighbourhood, min-z wins; the
smallest z is written last so plain fancy indexing gives a min-buffer.
Validated against raster.composite in cap_occl_validate.py.
"""
import io, json, os, sys, time
import numpy as np
import project as P
from cap_keyframes import keyframes, split_held_out

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = 2          # calibrated in cap_occl_calib.py # +/- pixels each splat writes into the z-buffer
Z_FLOOR = 0.10     # calibrated: matches the exact compositor's base rate to 1.1 pp


def view_geometry(col, count, R, t, fx, fy, cx, cy):
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(np.abs(z) > 1e-9, z, 1e-9)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    return cam, z, mx, my


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15)//16, (rh + 15)//16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    b, r, frames, pool, kf, info = keyframes()
    kfidx = [f['index'] for f in kf]

    # per-splat along-ray extent needs sigma_cam[2,2]; build world sigma once
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1-2*(y_*y_+z_*z_); Rm[:, 0, 1] = 2*(x_*y_-w_*z_); Rm[:, 0, 2] = 2*(x_*z_+w_*y_)
    Rm[:, 1, 0] = 2*(x_*y_+w_*z_); Rm[:, 1, 1] = 1-2*(x_*x_+z_*z_); Rm[:, 1, 2] = 2*(y_*z_-w_*x_)
    Rm[:, 2, 0] = 2*(x_*z_-w_*y_); Rm[:, 2, 1] = 2*(y_*z_+w_*x_); Rm[:, 2, 2] = 1-2*(x_*x_+y_*y_)
    M = Rm * scale[:, None, :]
    sigma_w = (M @ np.transpose(M, (0, 2, 1))).astype(np.float64)

    vis_f = np.zeros((count, len(kf)), dtype=bool)   # frustum only
    vis_o = np.zeros((count, len(kf)), dtype=bool)   # frustum AND unoccluded
    npix = rw * rh
    t0 = time.time()
    for j, fi in enumerate(kfidx):
        rot, tv = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, rad, tiles, ex, ey = P.project(col, count, R, tv, fx, fy, cx, cy, tx, ty)
        vis_f[:, j] = ok
        cam, z, mx, my = view_geometry(col, count, R, tv, fx, fy, cx, cy)
        idx = np.flatnonzero(ok)
        # along-ray 1-sigma of each splat in this camera
        rz = R[2]
        szz = np.einsum('i,nij,j->n', rz, sigma_w[idx], rz)
        tol = 3.0*np.sqrt(np.maximum(szz, 1e-12)) + Z_FLOOR

        zb = np.full(npix, np.inf)
        pxi = np.round(mx[idx]).astype(np.int64)
        pyi = np.round(my[idx]).astype(np.int64)
        order = np.argsort(-z[idx])        # far first, near last -> min wins
        for dy in range(-STAMP, STAMP+1):
            for dx in range(-STAMP, STAMP+1):
                X = pxi[order]+dx; Y = pyi[order]+dy
                m = (X >= 0) & (X < rw) & (Y >= 0) & (Y < rh)
                zb[Y[m]*rw + X[m]] = z[idx][order][m]
        m = (pxi >= 0) & (pxi < rw) & (pyi >= 0) & (pyi < rh)
        front = np.zeros(len(idx), dtype=bool)
        front[m] = zb[pyi[m]*rw + pxi[m]] >= (z[idx][m] - tol[m])
        vis_o[idx, j] = front
        if (j+1) % 30 == 0:
            print('  ...%d/%d  %.0fs' % (j+1, len(kf), time.time()-t0))

    samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
    rays = np.load(os.path.join(OUT, '_cap_rays.npy'))
    rays[~vis_o[samp]] = 0
    np.save(os.path.join(OUT, '_cap_rays_occ.npy'), rays)
    np.save(os.path.join(OUT, '_cap_viso.npy'), np.packbits(vis_o, axis=1))
    nf = vis_f.sum(axis=1); no = vis_o.sum(axis=1)
    np.save(os.path.join(OUT, '_cap_nviews_occ.npy'), no.astype(np.int16))
    print('\n=== view count per Gaussian, OCCLUSION AWARE (%d keyframes) ===' % len(kf))
    print('  frustum-only mean %.2f  ->  unoccluded mean %.2f' % (nf.mean(), no.mean()))
    for k in (1, 2, 3, 4, 5, 8, 10):
        print('  seen in < %-2d views: %8d (%5.2f%%)' % (k, (no < k).sum(), 100*(no < k).mean()))
    for q in (1, 5, 10, 25, 50, 75, 90):
        print('  p%-3d %5d' % (q, np.percentile(no, q)))
    tr, ho = split_held_out(len(kf))
    ntr = vis_o[:, tr].sum(axis=1)
    print('\n  restricted to the %d TRAINED keyframes:' % len(tr))
    for k in (1, 2, 3):
        print('    seen in < %d: %8d (%5.2f%%)' % (k, (ntr < k).sum(), 100*(ntr < k).mean()))
    print('    median %d mean %.2f' % (np.median(ntr), ntr.mean()))
    np.save(os.path.join(OUT, '_cap_nviews_occ_train.npy'), ntr.astype(np.int16))


if __name__ == '__main__':
    main()
