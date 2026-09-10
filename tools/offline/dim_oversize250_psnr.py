"""QUALITY COST of a screen-radius prune at each candidate cutoff, build 250.

Selects splats by the clamped max-over-trained-views radius (exactly what
stats.maxRadiusPxBits would hold at the end of an interval that saw every
trained keyframe), deletes them, and renders the held-out poses with the full
model and with each pruned model. Reports PSNR of pruned-vs-full (self
consistency: how much the held-out picture changes), and the tile instances
removed at those same poses.

CAVEAT: render.py uses project.py's Jacobian, which has NO tangent clamp. The
clamp binds on 151 of 1,892 (splat,view) draws above 90 px, so the rendered
footprint of the candidate splats is almost entirely unaffected.
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import render as Rn
from dim_oversize2 import world_covariance
from dim_oversize250 import keyframes, sweep

CUTS = (180, 120, 90)
N_HELD = 6


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx = (rw + P.TILE_W - 1) // P.TILE_W
    ty = (rh + P.TILE_H - 1) // P.TILE_H
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    _, trained, held = keyframes(poses)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, _ = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    r = sweep(mean, sigma, opacity, poses, trained, (fx, fy, cx, cy), (tx, ty, rw, rh), True)
    masks = {c: ~(r['seen'] & (r['mxr'] > c)) for c in CUTS}
    for c in CUTS:
        print('cut %d px removes %d splats' % (c, int((~masks[c]).sum())))
    sys.stdout.flush()

    use = held[:: max(len(held) // N_HELD, 1)][:N_HELD]
    res = {c: [] for c in CUTS}
    inst_full = 0
    inst_cut = {c: 0 for c in CUTS}
    for f in use:
        R = P.quat_to_matrix(poses[f][0])
        t = poses[f][1]
        ok, _, tiles, _, _ = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        inst_full += int(tiles.sum())
        full, _ = Rn.render(col, count, R, t, fx, fy, cx, cy, rw, rh)
        for c in CUTS:
            keep = masks[c]
            inst_cut[c] += int(tiles[~keep].sum())
            sub = {k: v[keep] for k, v in col.items()}
            img, _ = Rn.render(sub, int(keep.sum()), R, t, fx, fy, cx, cy, rw, rh)
            p = psnr(img, full)
            res[c].append(p)
            print('  held-out frame %3d  cut %3d px  PSNR pruned-vs-full %.2f dB  mean|err| %.5f'
                  % (f, c, p, float(np.abs(img - full).mean())))
            sys.stdout.flush()
    print('')
    print('held-out poses rendered: %s' % use)
    for c in CUTS:
        print('cut %3d px: %6d splats removed | held-out tile instances removed %.3f%% | '
              'mean PSNR pruned-vs-full %.2f dB (min %.2f)'
              % (c, int((~masks[c]).sum()), 100.0 * inst_cut[c] / inst_full,
                 float(np.mean(res[c])), float(np.min(res[c]))))


if __name__ == '__main__':
    main()
