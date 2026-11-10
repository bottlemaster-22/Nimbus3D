"""How much of the rasterisers' inner loop a per-SIMD-group row test would skip.

A 16x16 tile is 8 SIMD groups of 32 lanes, each covering TWO pixel rows. For
every splat in the tile's list every group runs the per-pixel power test, even
when the splat's alpha-threshold box does not reach either of its two rows.
A group can skip splat j with two compares when the splat's y extent misses
its rows: exact (the box is where the cutoff test could pass at all).

Counts, over the instances that survive build 286's exact tile test:
  strip-iterations  = instances * 8 (what every group runs now)
  y-skippable       = those whose box y-range misses the strip's 2 rows
  exact-skippable   = those where no pixel centre of the strip passes the cutoff
                      (the best any strip test could do)
"""
import os
import sys

import numpy as np

import project as P
import tile_cull as TC


def strips(ok, ex, ey, mx, my, conic, level, tiles_x, tiles_y):
    idx = np.nonzero(ok)[0]
    mx, my, ex, ey = mx[idx], my[idx], ex[idx], ey[idx]
    a, b, c = conic[idx, 0], conic[idx, 1], conic[idx, 2]
    L = level[idx]
    x0 = np.maximum(0, np.floor((mx - ex) / 16)).astype(np.int64)
    y0 = np.maximum(0, np.floor((my - ey) / 16)).astype(np.int64)
    x1 = np.minimum(tiles_x, np.ceil((mx + ex) / 16)).astype(np.int64)
    y1 = np.minimum(tiles_y, np.ceil((my + ey) / 16)).astype(np.int64)
    nx, ny = np.maximum(x1 - x0, 0), np.maximum(y1 - y0, 0)
    n = nx * ny
    total = int(n.sum())
    rep = np.repeat(np.arange(len(idx)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(total) - start
    tx = x0[rep] + k % nx[rep]
    ty = y0[rep] + k // nx[rep]
    A, B, C, LL = a[rep], b[rep], c[rep], L[rep]
    MX, MY, EY = mx[rep], my[rep], ey[rep]
    # Keep only instances the exact tile test keeps (build 286).
    X0 = tx * 16 + 0.5 - MX; X1 = X0 + 15
    Y0t = ty * 16 + 0.5 - MY; Y1t = Y0t + 15
    inside = (X0 <= 0) & (X1 >= 0) & (Y0t <= 0) & (Y1t >= 0)
    m = np.minimum(np.minimum(TC.edge_min(A, B, C, X0, Y0t, Y1t), TC.edge_min(A, B, C, X1, Y0t, Y1t)),
                   np.minimum(TC.edge_min(C, B, A, Y0t, X0, X1), TC.edge_min(C, B, A, Y1t, X0, X1)))
    keep = inside | (m <= LL)
    A, B, C, LL, MX, MY, EY, X0, X1, ty = A[keep], B[keep], C[keep], LL[keep], MX[keep], MY[keep], EY[keep], X0[keep], X1[keep], ty[keep]
    inst = int(keep.sum())
    y_skip = 0
    exact_skip = 0
    for g in range(8):
        # Pixel-centre rows of group g: tile row 2g and 2g+1.
        r0 = ty * 16 + 2 * g + 0.5
        r1 = r0 + 1.0
        # Box y-range test (what the kernel would do with the staged extent).
        miss = (r1 < MY - EY) | (r0 > MY + EY)
        y_skip += int(miss.sum())
        # Exact strip test on the 16x2 rectangle of pixel centres.
        Y0 = r0 - MY; Y1 = r1 - MY
        ins = (X0 <= 0) & (X1 >= 0) & (Y0 <= 0) & (Y1 >= 0)
        mm = np.minimum(np.minimum(TC.edge_min(A, B, C, X0, Y0, Y1), TC.edge_min(A, B, C, X1, Y0, Y1)),
                        np.minimum(TC.edge_min(C, B, A, Y0, X0, X1), TC.edge_min(C, B, A, Y1, X0, X1)))
        exact_skip += int((~(ins | (mm <= LL))).sum())
    return inst, inst * 8, y_skip, exact_skip


def main():
    rw, rh = 720, 540
    tiles_x, tiles_y = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = [int(a) for a in sys.argv[1:]] or list(range(0, 435, 72))
    tot = np.zeros(4, dtype=np.int64)
    for f in frames:
        rot, t = poses[f]
        ok, tiles, ex, ey, mx, my, conic, level = TC.project_full(
            col, count, P.quat_to_matrix(rot), t, fx, fy, cx, cy, tiles_x, tiles_y)
        r = strips(ok, ex, ey, mx, my, conic, level, tiles_x, tiles_y)
        tot += np.array(r)
        print('frame %4d  instances %8d  strip-iters %9d  y-skip %5.1f%%  exact-skip %5.1f%%'
              % (f, r[0], r[1], 100 * r[2] / max(r[1], 1), 100 * r[3] / max(r[1], 1)))
    print('TOTAL strip-iterations %d   y-range skip %.1f%%   exact strip skip %.1f%%'
          % (tot[1], 100 * tot[2] / max(tot[1], 1), 100 * tot[3] / max(tot[1], 1)))


if __name__ == '__main__':
    main()
