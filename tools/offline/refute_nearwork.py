"""Where does the rasteriser's work go, by camera distance and by how much a
splat actually contributes?

For each sampled view this reports the drawn splats and the TILE INSTANCES they
generate, split by camera-space depth, plus the share of composited weight
(sum of alpha*T over pixels) each band carries. A band that is a large share of
the instances and a small share of the weight is work being paid for and thrown
away, which is the only kind of speed-up allowed here (same seeds, same cap,
same rounds).

Run: python -u refute_nearwork.py [nframes]
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
BANDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    keys = sorted(poses.keys())
    sample = keys[::max(1, len(keys) // nf)][:nf]

    inst_band = np.zeros(len(BANDS) - 1)
    wgt_band = np.zeros(len(BANDS) - 1)
    drawn_band = np.zeros(len(BANDS) - 1)
    per_splat_w = np.zeros(count)
    per_splat_inst = np.zeros(count)
    for f in sample:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        tx, ty = (rw + 15) // 16, (rh + 15) // 16
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
        g = E.prep(col, count, R, t, fx, fy, cx, cy, rw, rh)
        splat, bounds = g['splat'], g['bounds']
        mxx, myy = g['mx'], g['my']
        cxk, cyk, czk = g['conic']
        opacity = g['opacity']
        w_acc = np.zeros(count)
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
            dx = gx[:, None] - mxx[sel][None, :]
            dy = gy[:, None] - myy[sel][None, :]
            power = -0.5 * (cxk[sel][None, :] * dx * dx
                            + czk[sel][None, :] * dy * dy) \
                - cyk[sel][None, :] * dx * dy
            al = np.minimum(0.99, opacity[sel][None, :]
                            * np.exp(np.clip(power, -60, 0)))
            al = np.where(al >= P.MIN_ALPHA, al, 0.0)
            ta = np.cumprod(1.0 - al, axis=1)
            before = np.concatenate([np.ones((256, 1)), ta[:, :-1]], axis=1)
            np.add.at(w_acc, sel, (al * before).sum(axis=0))
        per_splat_w += w_acc
        per_splat_inst += np.where(ok, tiles, 0)
        for i in range(len(BANDS) - 1):
            m = ok & (z >= BANDS[i]) & (z < BANDS[i + 1])
            drawn_band[i] += m.sum()
            inst_band[i] += tiles[m].sum()
            wgt_band[i] += w_acc[m].sum()
        print('  view %d done (%d drawn, %d instances, weight %.0f)'
              % (f, int(ok.sum()), int(tiles[ok].sum()), w_acc.sum()))
        sys.stdout.flush()

    print('')
    print('%d views. BY CAMERA DEPTH:' % len(sample))
    print('  %-14s %10s %10s %12s %10s %10s' %
          ('band (m)', 'drawn', '%drawn', 'instances', '%inst', '%weight'))
    ti, tw, td = inst_band.sum(), wgt_band.sum(), drawn_band.sum()
    for i in range(len(BANDS) - 1):
        hi = BANDS[i + 1]
        print('  %-14s %10d %9.2f%% %12d %9.2f%% %9.2f%%'
              % ('%.2f-%s' % (BANDS[i], 'inf' if hi > 1e8 else '%.2f' % hi),
                 drawn_band[i], 100 * drawn_band[i] / td, inst_band[i],
                 100 * inst_band[i] / ti, 100 * wgt_band[i] / tw))

    print('')
    print('PER SPLAT over the same views: total composited weight vs instances')
    order = np.argsort(per_splat_w)
    cw = np.cumsum(per_splat_w[order]) / max(per_splat_w.sum(), 1e-9)
    ci = np.cumsum(per_splat_inst[order]) / max(per_splat_inst.sum(), 1e-9)
    for frac in (0.10, 0.25, 0.50):
        k = int(frac * count)
        print('  the %5.1f%% of splats with the LOWEST weight carry %6.3f%% of the '
              'composited weight and %6.2f%% of the tile instances'
              % (100 * frac, 100 * cw[k - 1], 100 * ci[k - 1]))
    zero = per_splat_w <= 0
    print('  splats with EXACTLY zero composited weight in all %d views: %d (%.2f%%), '
          'costing %.2f%% of the tile instances'
          % (len(sample), int(zero.sum()), 100.0 * zero.mean(),
             100.0 * per_splat_inst[zero].sum() / max(per_splat_inst.sum(), 1)))


if __name__ == '__main__':
    main()
