"""BUILD 250 CHANGE #1 AUDIT: the EWA tangent clamp, forward vs backward.

The forward (TrainerShaders.metal ~908) now evaluates J at a CLAMPED tangent.
trainer_preprocess_backward (~2455) rebuilds jr0/jr1 from the UNCLAMPED
meanCam.x/y, and block (b) differentiates J as if the clamp were not there.
This measures how much of a real frame that mismatch covers.

Run: python -u clampaudit.py [frames...]
"""
import io, json, os, sys
import numpy as np
import project as P

LIM = 1.3


def project_clamped(col, count, R, t, fx, fy, cx, cy, tx, ty, clamped=True):
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_); Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_);     Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_); Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_);     Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sigma_cam = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    inv_z = 1.0 / np.where(z != 0, z, 1)
    inv_z2 = inv_z * inv_z

    limx = LIM * (0.5 * (2 * cx)) / fx if False else LIM * (0.5 * WIDTH) / fx
    limy = LIM * (0.5 * HEIGHT) / fy
    txtz = cam[:, 0] * inv_z
    tytz = cam[:, 1] * inv_z
    bindx = np.abs(txtz) > limx
    bindy = np.abs(tytz) > limy
    if clamped:
        tcx = np.clip(txtz, -limx, limx) * z
        tcy = np.clip(tytz, -limy, limy) * z
    else:
        tcx, tcy = cam[:, 0], cam[:, 1]

    j00, j11 = fx * inv_z, fy * inv_z
    j02 = -fx * tcx * inv_z2
    j12 = -fy * tcy * inv_z2
    s00, s01, s02 = sigma_cam[:, 0, 0], sigma_cam[:, 0, 1], sigma_cam[:, 0, 2]
    s11, s12, s22 = sigma_cam[:, 1, 1], sigma_cam[:, 1, 2], sigma_cam[:, 2, 2]
    a0 = j00 * s00 + j02 * s02; a1 = j00 * s01 + j02 * s12; a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12; b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02; sb = a1 * j11 + a2 * j12; sc = b1 * j11 + b2 * j12
    det_before = np.maximum(sa * sc - sb * sb, 1e-12)
    lp = P.FILTER_2D_VARIANCE + P.FREQ_BLUR_VARIANCE
    sa = sa + lp; sc = sc + lp
    det = sa * sc - sb * sb
    comp2d = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))
    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    alpha = opacity * comp2d
    level = np.clip(2.0 * np.log(alpha / max(P.MIN_ALPHA, 1e-8)), 0, 9)
    k = np.sqrt(level)
    ex = k * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = k * np.sqrt(np.maximum(sc, 1e-9)) * 1.001
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    mnx = np.maximum(0, np.floor((mx - ex) / 16)).astype(np.int64)
    mny = np.maximum(0, np.floor((my - ey) / 16)).astype(np.int64)
    mxx = np.minimum(tx, np.ceil((mx + ex) / 16)).astype(np.int64)
    mxy = np.minimum(ty, np.ceil((my + ey) / 16)).astype(np.int64)
    tiles = np.maximum(mxx - mnx, 0) * np.maximum(mxy - mny, 0)
    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5) & (alpha >= P.MIN_ALPHA) & (tiles > 0)
    return ok, radius, tiles, ex, ey, (bindx | bindy), bindx, bindy


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = sl['renderWidth'], sl['renderHeight']
    tx, ty = (WIDTH + 15) // 16, (HEIGHT + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(WIDTH, HEIGHT)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = [int(a) for a in sys.argv[1:]] or sorted(poses.keys())[::7]
    print('frames sampled: %d   render %dx%d  fx %.1f fy %.1f' % (len(frames), WIDTH, HEIGHT, fx, fy))
    print('  limX = 1.3*(W/2)/fx = %.4f  -> a splat binds when |x/z| exceeds it'
          % (LIM * (0.5 * WIDTH) / fx))
    tot = dict(drawn=0, bind=0, inst=0, inst_bind=0, rad=[], radu=[], dtile=0)
    for fi in frames:
        rot, t3 = poses[fi]
        Rm = P.quat_to_matrix(rot)
        okc, rc, tic, _, _, bind, bx, by = project_clamped(col, count, Rm, t3, fx, fy, cx, cy, tx, ty, True)
        oku, ru, tiu, _, _, _, _, _ = project_clamped(col, count, Rm, t3, fx, fy, cx, cy, tx, ty, False)
        d = okc & bind
        tot['drawn'] += int(okc.sum()); tot['bind'] += int(d.sum())
        tot['inst'] += int(tic[okc].sum()); tot['inst_bind'] += int(tic[d].sum())
        tot['dtile'] += int(tiu[oku].sum()) - int(tic[okc].sum())
        if d.sum():
            tot['rad'].append(rc[d]); tot['radu'].append(ru[d])
    print('\nDRAWN splats over the sample     : %d' % tot['drawn'])
    print('  of those, the clamp BINDS on   : %d  (%.3f%%)'
          % (tot['bind'], 100 * tot['bind'] / max(tot['drawn'], 1)))
    print('TILE INSTANCES over the sample   : %d' % tot['inst'])
    print('  carried by clamp-bound splats  : %d  (%.2f%%)'
          % (tot['inst_bind'], 100 * tot['inst_bind'] / max(tot['inst'], 1)))
    print('  instances the clamp REMOVED    : %d  (%.2f%% of the unclamped total)'
          % (tot['dtile'], 100 * tot['dtile'] / max(tot['inst'] + tot['dtile'], 1)))
    if tot['rad']:
        rc = np.concatenate(tot['rad']); ru = np.concatenate(tot['radu'])
        print('\nON THE BOUND SPLATS, 3-sigma screen radius px:')
        print('  clamped    p50 %8.1f  p90 %8.1f  max %8.1f' % (*np.percentile(rc, [50, 90]), rc.max()))
        print('  unclamped  p50 %8.1f  p90 %8.1f  max %8.1f' % (*np.percentile(ru, [50, 90]), ru.max()))
        print('  -> the forward changed for EVERY one of these; the backward did not.')


if __name__ == '__main__':
    main()
