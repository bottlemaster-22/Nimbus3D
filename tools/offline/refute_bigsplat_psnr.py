"""What does deleting the giant near-camera splats actually cost in PSNR?

refute_bigsplat.py measured that 155 splats of 298,965 (0.052%) project to a
screen radius over 512 px on at least one view and generate 8.18% of every
view's tile instances. This renders the model with and without them and scores
both against the real photographs with the same closed-form gain+bias fit
evaluateHeldOut uses, so the trade is a number rather than an argument.

A splat is cut if its screen radius exceeds the threshold on ANY of the sampled
poses, which is what a screen-radius prune running through the whole training
run would do.

Run: python -u refute_bigsplat_psnr.py [threshold_px]
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import detail as Dt
import refute_earlyout as E

TILE = 16


def render_full(col, count, R, t, fx, fy, cx, cy, rw, rh):
    g = E.prep(col, count, R, t, fx, fy, cx, cy, rw, rh)
    mx, my = g['mx'], g['my']
    cxk, cyk, czk = g['conic']
    opacity, rgb = g['opacity'], g['rgb']
    splat, bounds = g['splat'], g['bounds']
    tx, ty = g['tx'], g['ty']
    out = np.zeros((rh, rw, 3), np.float64)
    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat[a:b]
        cxt, cyt = tid % tx, tid // tx
        px = cxt * TILE + np.arange(TILE) + 0.5
        py = cyt * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx, gy = gx.ravel(), gy.ravel()
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = -0.5 * (cxk[sel][None, :] * dx * dx + czk[sel][None, :] * dy * dy) \
            - cyk[sel][None, :] * dx * dy
        al = np.minimum(0.99, opacity[sel][None, :] * np.exp(np.clip(power, -60, 0)))
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        ta = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((256, 1)), ta[:, :-1]], axis=1)
        colr = (al * before) @ rgb[sel]
        y0, y1 = cyt * TILE, min(cyt * TILE + TILE, rh)
        x0, x1 = cxt * TILE, min(cxt * TILE + TILE, rw)
        out[y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
    return out, g['total']


def fit(x, y, clamp=True):
    x = x.ravel()
    y = y.ravel()
    n = float(x.size)
    den = n * (x * x).sum() - x.sum() ** 2
    gn = (n * (x * y).sum() - x.sum() * y.sum()) / den
    bs = (y.sum() - gn * x.sum()) / n
    if clamp:
        gn = min(max(gn, 0.9), 1.1)
        bs = min(max(bs, -0.05), 0.05)
    return gn, bs


def main():
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 512.0
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    tx, ty = (rw + 15) // 16, (rh + 15) // 16

    # Which splats ever exceed the threshold, over the WHOLE pose set.
    over = np.zeros(count, bool)
    keys = sorted(poses.keys())
    for f in keys[::4]:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        over |= ok & (radius > th)
    print('threshold %.0f px: %d of %d splats (%.4f%%) exceed it on at least one '
          'of %d poses' % (th, int(over.sum()), count, 100.0 * over.mean(),
                           len(keys[::4])))
    keep = ~over
    sub = dict((k, v[keep]) for k, v in col.items())
    nsub = int(keep.sum())

    rows = []
    for f in sorted(frames):
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        a, ia = render_full(col, count, R, t, fx, fy, cx, cy, rw, rh)
        b, ib = render_full(sub, nsub, R, t, fx, fy, cx, cy, rw, rh)
        _, gt = Dt.luma_of(frames[f], rw, rh)
        gt = gt.astype(np.float64)
        la = a
        lb = b
        ga, ba = fit(la, gt)
        gb, bb = fit(lb, gt)
        pa = E.psnr(np.clip(ga * la + ba, 0, 1), gt)
        pb = E.psnr(np.clip(gb * lb + bb, 0, 1), gt)
        rows.append((f, ia, ib, pa, pb))
        print('  frame %2d  instances %7d -> %7d (%+6.2f%%)   clamped-fit PSNR '
              '%7.3f -> %7.3f  (%+.3f dB)'
              % (f, ia, ib, 100.0 * (ib - ia) / ia, pa, pb, pb - pa))
        sys.stdout.flush()

    ia = sum(r[1] for r in rows)
    ib = sum(r[2] for r in rows)
    pa = np.mean([r[3] for r in rows])
    pb = np.mean([r[4] for r in rows])
    print('')
    print('OVER %d PHOTOGRAPHED FRAMES' % len(rows))
    print('  tile instances %d -> %d  (%.2f%% removed)'
          % (ia, ib, 100.0 * (ia - ib) / ia))
    print('  mean clamped-fit PSNR %.3f -> %.3f dB  (%+.3f dB)' % (pa, pb, pb - pa))
    print('  splats %d -> %d (%.4f%% removed)'
          % (count, nsub, 100.0 * (count - nsub) / count))


if __name__ == '__main__':
    main()
