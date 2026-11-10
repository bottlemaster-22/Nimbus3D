"""DIMENSION: THE OVERSIZED PRUNE THAT NEVER FIRES.

Measures the exact quantity TrainerDensifier compares against
pruneMaxScreenRadiusPx: stats[i].maxRadiusPxBits, which trainer_preprocess
writes as
    atomic_fetch_max(&stats[gid].maxRadiusPxBits, as_type<uint>(radius))
with radius = 3 * sqrt(largest eigenvalue of Sigma2D + lowPass), in PIXELS,
written ONLY for splats that survived every early return (i.e. that drew).

Run twice: with the build-250 tangent clamp on the EWA Jacobian, and without,
so the clamp's effect on this statistic is isolated. World covariance is
precomputed once; only the view transform is per-frame.
"""
import io
import json
import os
import sys

import numpy as np

import project as P

CLAMP_FACTOR = 1.3


def world_covariance(col, count):
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
    Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    return M @ np.transpose(M, (0, 2, 1)), scale


def per_view(mean, sigma, opacity, R, t, fx, fy, cx, cy, tx, ty, rw, rh, clamp):
    cam = mean @ R.T + t
    z = cam[:, 2]
    sigma_cam = np.matmul(np.matmul(R, sigma), R.T)

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    if clamp:
        lim_x = CLAMP_FACTOR * (0.5 * rw / fx)
        lim_y = CLAMP_FACTOR * (0.5 * rh / fy)
        txc = np.clip(cam[:, 0] * inv_z, -lim_x, lim_x) * z
        tyc = np.clip(cam[:, 1] * inv_z, -lim_y, lim_y) * z
    else:
        txc = cam[:, 0]
        tyc = cam[:, 1]

    j00 = fx * inv_z
    j11 = fy * inv_z
    j02 = -fx * txc * inv_z2
    j12 = -fy * tyc * inv_z2

    s00 = sigma_cam[:, 0, 0]
    s01 = sigma_cam[:, 0, 1]
    s02 = sigma_cam[:, 0, 2]
    s11 = sigma_cam[:, 1, 1]
    s12 = sigma_cam[:, 1, 2]
    s22 = sigma_cam[:, 2, 2]

    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12

    det_before = np.maximum(sa * sc - sb * sb, 1e-12)
    low_pass = P.FILTER_2D_VARIANCE + P.FREQ_BLUR_VARIANCE
    sa = sa + low_pass
    sc = sc + low_pass
    det = sa * sc - sb * sb
    comp2d = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))

    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))

    alpha = opacity * comp2d
    level = np.clip(2.0 * np.log(alpha / max(P.MIN_ALPHA, 1e-8)), 0, 9)
    k = np.sqrt(level)
    ex = k * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = k * np.sqrt(np.maximum(sc, 1e-9)) * 1.001

    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    min_x = np.maximum(0, np.floor((mx - ex) / P.TILE_W)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / P.TILE_H)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / P.TILE_W)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / P.TILE_H)).astype(np.int64)
    tiles = np.maximum(max_x - min_x, 0) * np.maximum(max_y - min_y, 0)

    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5)
    ok &= alpha >= P.MIN_ALPHA
    ok &= tiles > 0
    return ok, radius, np.where(ok, tiles, 0)


def run(mean, sigma, opacity, poses, keys, fx, fy, cx, cy, tx, ty, rw, rh,
        clamp, label):
    count = mean.shape[0]
    mxr = np.zeros(count)
    seen = np.zeros(count, bool)
    inst = np.zeros(count, np.int64)
    hist = np.zeros(2048, np.int64)
    edges = np.geomspace(0.5, 20000.0, 2049)
    for f in keys:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles = per_view(mean, sigma, opacity, R, t, fx, fy, cx, cy,
                                     tx, ty, rw, rh, clamp)
        mxr = np.where(ok, np.maximum(mxr, radius), mxr)
        seen |= ok
        inst += tiles
        hist += np.histogram(radius[ok], bins=edges)[0]
    total = inst.sum()
    d = mxr[seen]
    ndraw = int(hist.sum())
    cum = np.cumsum(hist)
    print('')
    print('===== ' + label)
    print('  views %d | splats ever drawn %d of %d | tile instances %d'
          % (len(keys), int(seen.sum()), count, int(total)))
    print('  PER-VIEW 3-sigma radius over %d (splat,view) draws, px:' % ndraw)
    for qq in (50, 90, 99, 99.9, 99.99):
        i = int(np.searchsorted(cum, qq / 100.0 * ndraw))
        print('    p%-6s %10.2f' % (qq, edges[min(i, 2047)]))
    print('  MAX-OVER-VIEWS radius (== stats.maxRadiusPxBits), px:')
    print('    p50 %.2f  p90 %.2f  p99 %.2f  p99.9 %.2f  max %.2f'
          % tuple(np.percentile(d, [50, 90, 99, 99.9, 100])))
    print('  cutoff  splats>cut   pctOfDrawn   tileInstCarried   pctOfAllInst')
    for thr in (720, 540, 360, 180, 90, 45, 24, 12):
        m = seen & (mxr > thr)
        n = int(m.sum())
        ti = int(inst[m].sum())
        print('  %6d  %10d   %10.4f   %15d   %12.4f'
              % (thr, n, 100.0 * n / max(int(seen.sum()), 1), ti,
                 100.0 * ti / max(int(total), 1)))
    sys.stdout.flush()
    return mxr, seen, inst, total


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx = (rw + P.TILE_W - 1) // P.TILE_W
    ty = (rh + P.TILE_H - 1) // P.TILE_H
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    allkeys = sorted(poses.keys())
    print('render %dx%d  tiles %dx%d = %d  fx %.3f fy %.3f cx %.2f cy %.2f'
          % (rw, rh, tx, ty, tx * ty, fx, fy, cx, cy))
    print('splats %d  poses %d  keyframesTrained %d heldOut %d'
          % (count, len(allkeys), sl['keyframesTrained'], sl['keyframesHeldOut']))
    print('census peakTileInstances %d at %d splats = %.3f per splat'
          % (sl['peakTileInstances'], sl['splatCountAtPeakTileInstances'],
             sl['peakTileInstances'] / sl['splatCountAtPeakTileInstances']))
    print('tangent clamp limits: limX %.5f  limY %.5f  (units of x/z)'
          % (1.3 * 0.5 * rw / fx, 1.3 * 0.5 * rh / fy))
    sys.stdout.flush()

    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, scale = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))

    step = max(len(allkeys) // 120, 1)
    sub = allkeys[::step][:120]

    out = {}
    for clamp in (True, False):
        tag = ('WITH build-250 tangent clamp' if clamp
               else 'WITHOUT clamp (pre-250 projection)')
        out[clamp] = run(mean, sigma, opacity, poses, sub, fx, fy, cx, cy,
                         tx, ty, rw, rh, clamp,
                         tag + ' | 120 evenly spaced views')

    a = out[True][0]
    b = out[False][0]
    both = out[True][1] | out[False][1]
    print('')
    print('===== CLAMP EFFECT, per splat, max-over-views radius')
    changed = int((np.abs(a - b) > 0.01 * np.maximum(b, 1e-9)).sum())
    print('  splats whose max radius the clamp changed by >1 pct: %d of %d drawn'
          % (changed, int(both.sum())))
    ratio = a[both] / np.maximum(b[both], 1e-9)
    print('  clamped/unclamped ratio: min %.4f p1 %.4f p50 %.4f p99 %.4f'
          % tuple(np.percentile(ratio, [0, 1, 50, 99])))
    print('  largest 12 unclamped radii and their clamped counterparts:')
    order = np.argsort(-b)[:12]
    for i in order:
        print('    splat %7d  unclamped %10.2f px  clamped %10.2f px  inst %d'
              % (i, b[i], a[i], out[True][2][i]))
    sys.stdout.flush()

    run(mean, sigma, opacity, poses, allkeys, fx, fy, cx, cy, tx, ty, rw, rh,
        True, 'WITH clamp | ALL %d poses (upper bound)' % len(allkeys))


if __name__ == '__main__':
    main()
