"""What the split criterion actually selects, and what splitting costs.

Uses the validated offline projection in project.py. Everything here is
measured on the owner's real model and real poses, so it is evidence rather
than an estimate.
"""
import io
import json
import os

import numpy as np

import project as P

D = P.D
STRIDE = 7          # ~124 of the 868 frames, standing in for the 120 keyframes


def gather():
    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()

    frames = sorted(poses.keys())[::STRIDE]
    max_radius = np.zeros(count)
    total_tiles = np.zeros(count, dtype=np.int64)
    seen = np.zeros(count, dtype=np.int64)
    for idx in frames:
        rot, t = poses[idx]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(
            col, count, R, t, fx, fy, cx, cy, tx, ty
        )
        max_radius = np.maximum(max_radius, np.where(ok, radius, 0))
        total_tiles += tiles
        seen += ok.astype(np.int64)
    return col, count, max_radius, total_tiles, seen, len(frames)


def main():
    col, count, max_radius, total_tiles, seen, nframes = gather()
    print('model: %d splats, %d frames sampled\n' % (count, nframes))

    # ---------------------------------------------------------- world scale
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    largest = scale.max(axis=1)
    print('WORLD SCALE of the largest axis, metres')
    for q in (50, 90, 95, 99, 99.9):
        print('  p%-5s %8.4f m  (%6.1f mm)'
              % (q, np.percentile(largest, q), np.percentile(largest, q) * 1000))
    print('  max   %8.4f m  (%6.1f mm)' % (largest.max(), largest.max() * 1000))
    print('  Scaniverse median splat, same room: 3.4 mm')

    ever = seen > 0
    print('\nsplats drawn in at least one sampled frame: %d (%.1f%%)'
          % (ever.sum(), 100 * ever.mean()))

    # ------------------------------------------------- what split selects
    print('\nMAX SCREEN RADIUS in px, over the splats that are ever drawn')
    r = max_radius[ever]
    for q in (50, 75, 90, 95, 99, 99.9):
        print('  p%-5s %8.2f px' % (q, np.percentile(r, q)))
    print('  max   %8.2f px' % r.max())

    print('\nWHAT EACH splitScreenRadiusPx THRESHOLD WOULD SELECT')
    print('  %8s %12s %8s %14s %8s' %
          ('px', 'splats', '% of', 'tileInstances', '% of'))
    tt = total_tiles.sum()
    for thr in (4, 6, 8, 10, 12, 16, 24, 32, 48):
        sel = ever & (max_radius > thr)
        print('  %8.0f %12d %7.2f%% %14d %7.2f%%'
              % (thr, sel.sum(), 100 * sel.sum() / max(ever.sum(), 1),
                 total_tiles[sel].sum(), 100 * total_tiles[sel].sum() / tt))

    # -------------------------------------- where the tile instances live
    print('\nWHERE THE TILE INSTANCES COME FROM, splats ranked by radius')
    order = np.argsort(-max_radius)
    cum = np.cumsum(total_tiles[order])
    for pct in (0.1, 0.5, 1, 2, 5, 10, 25, 50):
        n = max(1, int(count * pct / 100))
        print('  top %5.1f%% of splats (%7d) carry %6.2f%% of tile instances'
              % (pct, n, 100 * cum[n - 1] / tt))

    # ---------------------------- what a split does to the tile footprint
    #
    # A split replaces one Gaussian with two. The tile box scales with the
    # projected sigma, so shrinking every axis by `k` shrinks each child's
    # half-extent by k, and two children cover roughly
    #     2 * (w/k + 1)(h/k + 1) tiles
    # against the parent's (w + 1)(h + 1), where w and h are the parent's
    # extents in TILES. Shrinking ONE axis only divides one of the two.
    print('\nTILE COST OF ONE SPLIT, measured on the selected population')
    print('  (parent tile box from the real projection, children modelled at'
          ' the same centre)')
    for thr in (8, 10, 12, 16):
        sel = ever & (max_radius > thr)
        if sel.sum() == 0:
            continue
        # Mean tile box per frame for the selected splats.
        parent = total_tiles[sel].sum() / max(seen[sel].sum(), 1)
        side = np.sqrt(max(parent, 1.0))          # w+1 = h+1 = sqrt(tiles)
        w = max(side - 1, 0.0)
        one = 2 * (w / 1.6 + 1) * (w + 1)          # one axis shrunk
        three = 2 * (w / 1.6 + 1) * (w / 1.6 + 1)  # all three shrunk
        print('  >%2d px: parent %6.2f tiles -> one-axis %6.2f (%+.0f%%),'
              '  three-axis %6.2f (%+.0f%%)'
              % (thr, parent, one, 100 * (one / parent - 1),
                 three, 100 * (three / parent - 1)))


if __name__ == '__main__':
    main()
