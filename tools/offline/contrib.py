"""How many of the 300,000 splats are actually contributing to any rendered
view of the room, in the FINAL trained model?

For each sampled frame this does the same tile-based front-to-back composite
as render.py, but instead of assembling an RGB image it accumulates, per
splat, the sum over all pixels of (alpha * T_before) -- exactly the weight
that splat's colour and opacity gradient would have received from that pixel
during backward. A splat with zero accumulated weight over every sampled view
was invisible in the model's current geometry: nothing drew it, so nothing
could have supervised it either.

View sample: the 12 held-out frames (known exactly, from held_out_frames.json)
unioned with an every-8th-frame stride across the full 0..867 capture range.
This is deliberately a SUPERSET of the ~120 keyframes actually used in
training (all real keyframes fall inside the 0..~467 span used by the held-out
set; the stride covers that span roughly 8x denser than the ~4-frame keyframe
spacing would need), so it can only find MORE contributing splats than the
real training view set, not fewer. That makes "still zero after this" a safe,
conservative floor, not an inflated one.

Run: python -u contrib.py [stride]
"""
import io
import json
import os
import sys
import time

import numpy as np

import project as P
import detail as Dt

TILE = 16


def geometry_and_weights(col, count, R, t, fx, fy, cx, cy):
    mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
    sa = sa + P.FILTER_2D_VARIANCE
    sc = sc + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    inv_det = 1.0 / det
    conic_x = sc * inv_det
    conic_y = -sb * inv_det
    conic_z = sa * inv_det
    det_before = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE)
                            - sb * sb, 1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d
    return mx, my, z, conic_x, conic_y, conic_z, opacity


def accumulate(col, count, R, t, fx, fy, cx, cy, tx, ty, rw, rh, contrib, ever_drawn):
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mx, my, z, conic_x, conic_y, conic_z, opacity = geometry_and_weights(
        col, count, R, t, fx, fy, cx, cy)

    idx = np.nonzero(ok)[0]
    ever_drawn[idx] = True
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)[idx]
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)[idx]
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)[idx]
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)[idx]
    w = np.maximum(max_x - min_x, 0)
    h = np.maximum(max_y - min_y, 0)
    n = w * h
    keep = n > 0
    idx, min_x, min_y, w, h, n = idx[keep], min_x[keep], min_y[keep], w[keep], h[keep], n[keep]

    total = int(n.sum())
    splat = np.repeat(idx, n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_x = np.repeat(min_x, n) + (j % lw)
    tile_y = np.repeat(min_y, n) + (j // lw)
    tile_id = tile_y * tx + tile_x

    order = np.lexsort((z[splat], tile_id))
    splat_s = splat[order]
    tile_id_s = tile_id[order]
    bounds = np.searchsorted(tile_id_s, np.arange(tx * ty + 1))

    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat_s[a:b]
        cxt, cyt = tid % tx, tid // tx
        px = cxt * TILE + np.arange(TILE) + 0.5
        py = cyt * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx, gy = gx.ravel(), gy.ravel()
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = -0.5 * (conic_x[sel][None, :] * dx * dx
                        + conic_z[sel][None, :] * dy * dy) \
            - conic_y[sel][None, :] * dx * dy
        al = np.where(power <= 0,
                      np.minimum(0.99, opacity[sel][None, :]
                                 * np.exp(np.clip(power, -60, 0))), 0.0)
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        wgt = (al * before).sum(axis=0)          # per splat, summed over pixels
        np.add.at(contrib, sel, wgt)


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    held_out = json.load(io.open(os.path.join(P.D, 'model', 'held_out_frames.json'),
                                 encoding='utf-8'))

    all_idx = sorted(poses.keys())
    frames = sorted(set(held_out) | set(all_idx[::stride]))
    print('splats %d   frames sampled %d (stride %d, %d held-out unioned in)'
          % (count, len(frames), stride, len(held_out)))

    contrib = np.zeros(count, dtype=np.float64)
    ever_drawn = np.zeros(count, dtype=bool)

    t0 = time.time()
    for n, f in enumerate(frames):
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        accumulate(col, count, R, t, fx, fy, cx, cy, tx, ty, rw, rh, contrib, ever_drawn)
        if (n + 1) % 10 == 0 or n == len(frames) - 1:
            elapsed = time.time() - t0
            print('  %d/%d frames, %.1fs elapsed, %.2fs/frame'
                  % (n + 1, len(frames), elapsed, elapsed / (n + 1)))

    np.save(os.path.join(Dt.SCRATCH, 'contrib.npy'), contrib)
    np.save(os.path.join(Dt.SCRATCH, 'ever_drawn.npy'), ever_drawn)
    with open(os.path.join(Dt.SCRATCH, 'contrib_frames.json'), 'w') as fh:
        json.dump(frames, fh)

    never = ~ever_drawn
    print('\nnever drawn in any sampled view: %d (%.2f%%)'
          % (never.sum(), 100 * never.sum() / count))

    # "Negligible accumulated alpha" -- contrib is a sum of (alpha*T) over all
    # sampled pixels across all sampled views. A splat that peaks at, say,
    # alpha 0.05 in one 3x3-pixel footprint in one view accumulates roughly
    # 0.05 * 9 =~ 0.45. Splats that never clear the compositor's own
    # MIN_ALPHA (1/255) anywhere contribute exactly 0.
    zero = contrib <= 0
    print('exactly zero accumulated weight: %d (%.2f%%)'
          % (zero.sum(), 100 * zero.sum() / count))
    for thr in (1e-6, 1e-3, 1e-1, 1.0, 3.0):
        below = contrib < thr
        print('  accumulated weight < %-8g : %7d  (%5.2f%%)'
              % (thr, below.sum(), 100 * below.sum() / count))

    order = np.argsort(contrib)
    print('\ncontrib percentiles (sorted ascending):')
    for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        i = min(int(q / 100 * (count - 1)), count - 1)
        print('  p%-3d %12.6f' % (q, contrib[order[i]]))


if __name__ == '__main__':
    main()
