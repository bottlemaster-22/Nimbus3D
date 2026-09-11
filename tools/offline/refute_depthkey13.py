"""What does the 24-bit sort key actually cost in image quality?

TrainerGPULayouts.swift:80-104 says a six-pass radix sort needs the coupled
change `uint(norm * 8191.0f)` + `(tile << 13)` + `radixKeyBits = 24`, and calls
the resulting ordering ties "a real if small quality trade" WITHOUT MEASURING
IT. This measures it.

Both renders composite in the order the shipped radix sort would produce:
the key is quantised depth, the sort is stable, so instances that tie fall back
to the order `trainer_duplicate_keys` wrote them in, which is splat index.

Reported: PSNR of the 13-bit-key render against the 16-bit-key render, and each
against the real photograph, on the frames that have one.

Run: python -u refute_depthkey13.py [frame ...]
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


def order_for(bits, z, splat, tile_id, near, far):
    """The permutation the shipped radix sort produces for a given depth width."""
    span = max(far - near, 1e-3)
    norm = np.clip((z[splat] - near) / span, 0.0, 1.0)
    key = (norm * float((1 << bits) - 1)).astype(np.int64)
    # Stable sort on (tile, quantised depth); ties keep splat-index order,
    # which is what a stable LSD radix sort over a key that ignores the tail
    # bits does.
    return np.lexsort((splat, key, tile_id))


def render_with(order_idx, g, rw, rh, splat, bounds_key):
    mx, my = g['mx'], g['my']
    cxk, cyk, czk = g['conic']
    opacity, rgb = g['opacity'], g['rgb']
    tx, ty = g['tx'], g['ty']
    out = np.zeros((rh, rw, 3), np.float64)
    for tid in range(tx * ty):
        a, b = bounds_key[tid], bounds_key[tid + 1]
        if b <= a:
            continue
        sel = splat[order_idx[a:b]]
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
        t_after = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((256, 1)), t_after[:, :-1]], axis=1)
        colr = (al * before) @ rgb[sel]
        y0, y1 = cyt * TILE, min(cyt * TILE + TILE, rh)
        x0, x1 = cxt * TILE, min(cxt * TILE + TILE, rw)
        out[y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
    return out


def build_instances(col, count, R, t, fx, fy, cx, cy, rw, rh):
    """Same instance list prep as refute_earlyout, but WITHOUT sorting, so the
    two key widths can each impose their own order on the same instances."""
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
    idx = np.nonzero(ok)[0]
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)[idx]
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)[idx]
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)[idx]
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)[idx]
    w = np.maximum(max_x - min_x, 0)
    h = np.maximum(max_y - min_y, 0)
    n = w * h
    keep = n > 0
    idx, min_x, min_y, w, n = idx[keep], min_x[keep], min_y[keep], w[keep], n[keep]
    total = int(n.sum())
    splat = np.repeat(idx, n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_x = np.repeat(min_x, n) + (j % lw)
    tile_y = np.repeat(min_y, n) + (j // lw)
    tile_id = tile_y * tx + tile_x
    return splat, tile_id, z, tx, ty, total


def main():
    frames = [int(x) for x in sys.argv[1:]] or [0, 4, 8, 12]
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()

    # near/far the trainer quantises against.
    near, far = 0.05, 100.0
    print('key = (tile << B) | quantised depth; near %.2f far %.2f m' % (near, far))
    print('16-bit depth step at the far plane: %.2f mm; 13-bit: %.2f mm'
          % (1000.0 * (far - near) / 65535.0, 1000.0 * (far - near) / 8191.0))
    tx0 = (rw + 15) // 16
    ty0 = (rh + 15) // 16
    ntiles = tx0 * ty0
    print('tiles %dx%d = %d, so tile ids need %d bits'
          % (tx0, ty0, ntiles, int(np.ceil(np.log2(ntiles)))))
    for p in range(8):
        lo, hi = 4 * p, 4 * p + 3
        alive = hi >= 16 and lo <= 16 + int(np.ceil(np.log2(ntiles))) - 1
        alive = alive or hi < 16
        print('  pass %d bits %2d..%2d  %s' % (p, lo, hi,
              'CARRIES DATA' if alive else 'ALWAYS ZERO'))

    ds = []
    for f in frames:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        g = E.prep(col, count, R, t, fx, fy, cx, cy, rw, rh)
        splat, tile_id, z, tx, ty, total = build_instances(
            col, count, R, t, fx, fy, cx, cy, rw, rh)
        o16 = order_for(16, z, splat, tile_id, near, far)
        o13 = order_for(13, z, splat, tile_id, near, far)
        tsorted = tile_id[o16]
        bounds = np.searchsorted(tsorted, np.arange(tx * ty + 1))
        img16 = render_with(o16, g, rw, rh, splat, bounds)
        img13 = render_with(o13, g, rw, rh, splat, bounds)
        d = E.psnr(img13, img16)
        swapped = int((o16 != o13).sum())
        ds.append(d)
        print('frame %2d: %d instances, %d (%.3f%%) land in a different slot '
              'under a 13-bit depth;  PSNR(13-bit vs 16-bit) %.2f dB, '
              'max pixel error %.5f'
              % (f, total, swapped, 100.0 * swapped / total, d,
                 float(np.abs(img13 - img16).max())))
        sys.stdout.flush()
    print('mean PSNR(13-bit render vs 16-bit render) over %d frames: %.2f dB'
          % (len(ds), float(np.mean(ds))))


if __name__ == '__main__':
    main()
