"""Follow-up to dim_oversize2: WHAT are the splats a lower screen-radius cut
would delete, and how much raster work is really behind each cutoff.

Adds, on top of dim_oversize2's distribution:
  * per-view tile-instance totals (mean / max) so the offline number can be
    compared with the census peakTileInstances honestly.
  * for each candidate cutoff, the WORLD size (mm) and the nearest camera
    distance of the splats it selects, so "projection artefact" can be told
    apart from "ordinary splat the camera walked up to".
"""
import io
import json
import os
import sys

import numpy as np

import project as P
from dim_oversize2 import per_view, world_covariance


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
    step = max(len(allkeys) // 120, 1)
    sub = allkeys[::step][:120]

    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, scale = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    # World size: the largest axis, in millimetres, and the 2-sigma diameter
    # used by analyse.py's size report is 2*sigma; keep to sigma here and say so.
    largest_mm = 1000.0 * scale.max(axis=1)

    mxr = np.zeros(count)
    seen = np.zeros(count, bool)
    inst = np.zeros(count, np.int64)
    nearest = np.full(count, np.inf)
    per_view_inst = []
    for f in sub:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles = per_view(mean, sigma, opacity, R, t, fx, fy, cx, cy,
                                     tx, ty, rw, rh, True)
        camz = (mean @ R.T + t)[:, 2]
        mxr = np.where(ok, np.maximum(mxr, radius), mxr)
        nearest = np.where(ok, np.minimum(nearest, camz), nearest)
        seen |= ok
        inst += tiles
        per_view_inst.append(int(tiles.sum()))
    pvi = np.array(per_view_inst)
    total = int(inst.sum())
    print('views %d  tile instances: mean %d  max %d  min %d'
          % (len(sub), int(pvi.mean()), int(pvi.max()), int(pvi.min())))
    print('census peakTileInstances %d (a MAX over 4000 iterations)'
          % sl['peakTileInstances'])
    print('offline max view / census peak = %.3f' % (pvi.max() / sl['peakTileInstances']))
    print('')
    print(' cutoff   splats   pctPop   instShare   largestAxis_mm p50/p90/max'
          '    nearest_z_m p10/p50')
    for thr in (720, 540, 360, 180, 120, 90, 60, 45):
        m = seen & (mxr > thr)
        n = int(m.sum())
        if n == 0:
            print(' %6d %8d   %6.3f   %8.4f%%   -' % (thr, 0, 0.0, 0.0))
            continue
        mm = largest_mm[m]
        nz = nearest[m]
        print(' %6d %8d   %6.3f   %8.4f%%   %8.2f %8.2f %8.2f   %7.3f %7.3f'
              % (thr, n, 100.0 * n / count, 100.0 * inst[m].sum() / total,
                 np.percentile(mm, 50), np.percentile(mm, 90), mm.max(),
                 np.percentile(nz, 10), np.percentile(nz, 50)))
    print('')
    print('whole drawn population for reference:')
    mm = largest_mm[seen]
    print('  largest axis mm: p50 %.2f  p90 %.2f  p99 %.2f  max %.2f'
          % tuple(np.percentile(mm, [50, 90, 99, 100])))
    nz = nearest[seen]
    print('  nearest camera z m: p10 %.3f  p50 %.3f  p90 %.3f'
          % tuple(np.percentile(nz, [10, 50, 90])))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
