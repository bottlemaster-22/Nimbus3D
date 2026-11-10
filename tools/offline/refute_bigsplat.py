"""Both oversize controls in this trainer are inert, and the splats they would
have caught are the ones the rasteriser spends its instances on.

  * trainer_regularizer's world hinge: MetalSplatTrainer.swift:2334 sets
    reg.maxScaleMeters = 0.5. On model.ply, 0 of 896,895 scale components
    exceed 0.5 m, so the hinge fires never.
  * the screen-size prune: census gates say pruneMaxScreenRadiusPx = 0, which
    is off.

This sweeps a screen-radius threshold over real views and reports, for each,
the splats it would catch, the TILE INSTANCES they generate (the sort's input
and the rasteriser's inner-loop length) and the composited weight they carry
(the light that would be lost).

Run: python -u refute_bigsplat.py [nframes]
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
THRESH = [512, 384, 256, 192, 128, 96, 64, 48, 32]


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    keys = sorted(poses.keys())
    sample = keys[::max(1, len(keys) // nf)][:nf]

    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    caught = np.zeros((len(THRESH),))
    inst_cut = np.zeros((len(THRESH),))
    wgt_cut = np.zeros((len(THRESH),))
    inst_all = 0.0
    wgt_all = 0.0
    ever = np.zeros(count, bool)
    ever_over = dict((t, np.zeros(count, bool)) for t in THRESH)

    for f in sample:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
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
        inst_all += float(tiles[ok].sum())
        wgt_all += float(w_acc.sum())
        ever |= ok
        for i, th in enumerate(THRESH):
            m = ok & (radius > th)
            ever_over[th] |= m
            caught[i] += m.sum()
            inst_cut[i] += float(tiles[m].sum())
            wgt_cut[i] += float(w_acc[m].sum())
        print('  view %d: %d drawn, %d instances, radius p50 %.1f p99 %.1f max %.1f px'
              % (f, int(ok.sum()), int(tiles[ok].sum()),
                 float(np.percentile(radius[ok], 50)),
                 float(np.percentile(radius[ok], 99)),
                 float(radius[ok].max())))
        sys.stdout.flush()

    print('')
    print('%d views, %d drawn-splat-views, %d tile instances total'
          % (len(sample), int(ever.sum()), int(inst_all)))
    print('  %-10s %14s %10s %12s %12s' %
          ('radius >', 'splat-views', '%drawn', '%instances', '%weight'))
    for i, th in enumerate(THRESH):
        print('  %-10d %14d %9.3f%% %11.2f%% %11.3f%%   distinct splats ever over: %d (%.3f%%)'
              % (th, caught[i], 100.0 * caught[i] / max(1.0, float(ever.sum()) * len(sample)),
                 100.0 * inst_cut[i] / inst_all, 100.0 * wgt_cut[i] / wgt_all,
                 int(ever_over[th].sum()), 100.0 * ever_over[th].mean()))


if __name__ == '__main__':
    main()
