"""Why the backward raster costs 11 ms against the forward's 3 ms.

The forward stops a pixel as soon as transmittance runs out. The backward
cannot: it walks the tile list from the far end, and its skip is bounded by the
DEEPEST pixel in a SIMD group, because the loop holds a threadgroup barrier and
so must run a threadgroup-uniform number of times.

This measures how much that costs, on real tiles, with the real contributor
depths, so the answer is a number rather than an argument.

Run:  python tools/offline/backward.py [frame] [tile_sample]
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import raster as Rz

D = P.D
TILE = 16
BATCH = 256          # TRAINER_TILE_AREA: one staged batch
SIMD = 32            # a SIMD group is two rows of a 16-wide tile


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_tiles = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)

    g = Rz.geometry(col, count, R, t, fx, fy, cx, cy)
    ok, radius, tiles, ex, ey = P.project(
        col, count, R, t, fx, fy, cx, cy, tx, ty
    )
    idx = np.nonzero(ok)[0]
    mx, my, z = g['mx'], g['my'], g['z']
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)

    rng = np.random.RandomState(0)
    picks = rng.choice(tx * ty, size=min(n_tiles, tx * ty), replace=False)

    fwd_pairs = 0            # what the forward evaluates (stops at T < 1e-4)
    bwd_pairs_perpixel = 0   # ideal: each pixel stops at its own contributor
    bwd_pairs_simd = 0       # real: bounded by the deepest pixel in 32
    bwd_pairs_tile = 0       # what a threadgroup-wide max would give
    bwd_pairs_none = 0       # no gate at all
    batches_run = 0
    batches_total = 0
    for p in picks:
        tile_x, tile_y = int(p % tx), int(p // tx)
        sel = idx[(min_x[idx] <= tile_x) & (max_x[idx] > tile_x)
                  & (min_y[idx] <= tile_y) & (max_y[idx] > tile_y)]
        if sel.size == 0:
            continue
        order = sel[np.argsort(z[sel])]
        total = order.size
        alpha_ok, before = Rz.composite(g, order, tile_x, tile_y)
        # lastContributor: 1-based index of the last entry this pixel used.
        used = alpha_ok & (before > 1e-4)
        anyused = used.any(axis=1)
        last = np.where(anyused, used.shape[1] - np.argmax(used[:, ::-1], axis=1),
                        0)
        npix = last.size

        fwd_pairs += int(last.sum())
        bwd_pairs_perpixel += int(last.sum())
        # SIMD groups: pixels are laid out row-major inside the tile, so 32
        # consecutive thread indices are two rows.
        groups = last.reshape(-1, SIMD)
        bwd_pairs_simd += int(groups.max(axis=1).sum() * SIMD)
        bwd_pairs_tile += int(last.max() * npix)
        bwd_pairs_none += int(total * npix)

        nb = (total + BATCH - 1) // BATCH
        batches_total += nb
        deepest = int(last.max())
        batches_run += min(nb, (deepest + BATCH - 1) // BATCH)

    print('frame %d, %d tiles\n' % (frame, len(picks)))
    print('(pixel, Gaussian) PAIRS THE BACKWARD ACTUALLY WALKS')
    base = bwd_pairs_perpixel
    for name, v in (
        ('forward (stops at T<1e-4)', fwd_pairs),
        ('backward, ideal per-pixel', bwd_pairs_perpixel),
        ('backward, SIMD-group gate (SHIPPED)', bwd_pairs_simd),
        ('backward, threadgroup gate', bwd_pairs_tile),
        ('backward, no gate at all', bwd_pairs_none),
    ):
        print('  %-38s %12d  %5.2fx the ideal' % (name, v, v / max(base, 1)))
    print('\nBATCH STAGING (256 Gaussians per staged batch)')
    print('  batches in the tile lists      %8d' % batches_total)
    print('  batches the loop actually runs %8d  (%.1f%%)'
          % (batches_run, 100 * batches_run / max(batches_total, 1)))


if __name__ == '__main__':
    main()
