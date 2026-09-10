"""STRICT UPPER BOUND for dim_oversize250: max-over-views screen radius using
EVERY refined pose (868), clamped as build 250 runs. The trainer renders at
most 120 of these (108 trained + 12 held out), and one densify interval at
most ~100 + 12, so no interval can record a larger maxRadiusPxBits on this
model than the max printed here. Also reports the unclamped counterpart.
"""
import io
import json
import os
import sys

import numpy as np

import project as P
from dim_oversize2 import world_covariance
from dim_oversize250 import sweep


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx = (rw + P.TILE_W - 1) // P.TILE_W
    ty = (rh + P.TILE_H - 1) // P.TILE_H
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    keys = sorted(poses.keys())
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, _ = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    for clamp in (True, False):
        r = sweep(mean, sigma, opacity, poses, keys, (fx, fy, cx, cy), (tx, ty, rw, rh), clamp)
        d = r['mxr'][r['seen']]
        tot = r['inst'].sum()
        print('ALL %d poses, %s: drawn %d  max-over-views p50 %.2f p90 %.2f p99 %.2f p99.9 %.2f max %.2f'
              % ((len(keys), 'CLAMPED' if clamp else 'UNCLAMPED', int(r['seen'].sum()))
                 + tuple(np.percentile(d, [50, 90, 99, 99.9, 100]))))
        for thr in (720, 360, 180, 90):
            m = r['seen'] & (r['mxr'] > thr)
            print('   >%4d px: %6d splats  instShare %.4f%%' % (thr, int(m.sum()), 100.0 * r['inst'][m].sum() / tot))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
