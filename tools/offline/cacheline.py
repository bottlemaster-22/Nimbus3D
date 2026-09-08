"""How many (pixel, Gaussian) pairs actually reach the atomic block in
trainer_rasterize_backward, how many DISTINCT splat records they touch, and
therefore what the two-cache-line layout costs in bytes per iteration.

Exact backward gating is reproduced:
  forward:  reject power < cutoff, reject alpha < minAlpha,
            BREAK when T*(1-alpha) < 1e-4 -> contributors = j (0-based count)
  backward: walk j < contributors, same two rejects, then 9 atomics into
            splatGrad2D[i] (64 B, one line) and 2-3 into stats[i] (32 B, so
            two splats share one 64 B line).

Run: cd tools/offline && python -u cacheline.py [n_frames] [tile_stride]
"""
import io
import json
import os
import sys
import time

import numpy as np

import project as P

D = P.D
TILE = 16
LINE = 64


def geometry(col, count, R, t, fx, fy, cx, cy):
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(z != 0, z, 1)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
    Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sc3 = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    j00 = fx * inv_z
    j11 = fy * inv_z
    j02 = -fx * cam[:, 0] * inv_z * inv_z
    j12 = -fy * cam[:, 1] * inv_z * inv_z
    s00, s01, s02 = sc3[:, 0, 0], sc3[:, 0, 1], sc3[:, 0, 2]
    s11, s12, s22 = sc3[:, 1, 1], sc3[:, 1, 2], sc3[:, 2, 2]
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    inv_det = 1.0 / det
    det_before = np.maximum(
        (sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb, 1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d
    return z, mx, my, sc * inv_det, -sb * inv_det, sa * inv_det, opacity


def instances(ok, mx, my, ex, ey, tx, ty):
    """duplicate_keys: one (tileID, splat) pair per tile the box covers."""
    idx = np.nonzero(ok)[0]
    min_x = np.maximum(0, np.floor((mx[idx] - ex[idx]) / TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my[idx] - ey[idx]) / TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx[idx] + ex[idx]) / TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my[idx] + ey[idx]) / TILE)).astype(np.int64)
    w = np.maximum(max_x - min_x, 0)
    h = np.maximum(max_y - min_y, 0)
    n = w * h
    keep = n > 0
    idx, min_x, min_y, w, h, n = (idx[keep], min_x[keep], min_y[keep],
                                  w[keep], h[keep], n[keep])
    total = int(n.sum())
    splat = np.repeat(idx, n)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    within = np.arange(total) - np.repeat(starts, n)
    ww = np.repeat(w, n)
    txx = np.repeat(min_x, n) + (within % ww)
    tyy = np.repeat(min_y, n) + (within // ww)
    return splat, (tyy * tx + txx).astype(np.int64), total


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 and "," not in sys.argv[1] else 6
    tile_stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # NULL TEST. `shuffle` relabels every Gaussian with a random index, which
    # leaves the geometry, the tiles and the compositing untouched and destroys
    # only the correlation between index and position. Any stats-line sharing
    # that survives it is chance; anything above it is real index locality.
    shuffle = len(sys.argv) > 3 and sys.argv[3] == 'shuffle'

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    keys = sorted(poses.keys())
    if len(sys.argv) > 1 and ',' in sys.argv[1]:
        picked = [int(x) for x in sys.argv[1].split(',')]
    else:
        step = max(1, len(keys) // n_frames)
        picked = keys[::step][:n_frames]
    print('splats %d  render %dx%d  tiles %dx%d (%d)  frames %s  tile_stride %d'
          '  SHUFFLED=%s' % (count, rw, rh, tx, ty, tx * ty, picked,
                             tile_stride, shuffle))
    relabel = np.random.RandomState(1).permutation(count)

    rows = []
    for fi in picked:
        t0 = time.time()
        rot, t = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        z, mx, my, cxx, cyy, czz, opac = geometry(col, count, R, t, fx, fy, cx, cy)
        splat, tid, total = instances(ok, mx, my, ex, ey, tx, ty)
        order = np.lexsort((z[splat], tid))
        splat, tid = splat[order], tid[order]
        bounds = np.searchsorted(tid, np.arange(tx * ty + 1))

        px = (np.arange(TILE) + 0.5)
        contributed = np.zeros(count, dtype=bool)
        pairs_eval = 0
        pairs_alpha = 0
        pairs_contrib = 0
        pairs_shared = 0     # contributing pairs whose 32B stats record shares
                             # its 64B line with ANOTHER splat live in the SAME
                             # tile: false sharing the 64B grad2D record has not
        tile_splats = 0
        tile_statslines = 0
        tiles_done = 0
        for tile in range(0, tx * ty, tile_stride):
            s, e = bounds[tile], bounds[tile + 1]
            tiles_done += 1
            if e <= s:
                continue
            sel = splat[s:e]
            n = sel.size
            ty_, tx_ = divmod(tile, tx)
            gx = np.tile(tx_ * TILE + px, TILE)
            gy = np.repeat(ty_ * TILE + px, TILE)
            dx = gx[:, None] - mx[sel][None, :]
            dy = gy[:, None] - my[sel][None, :]
            power = -0.5 * (cxx[sel][None, :] * dx * dx
                            + czz[sel][None, :] * dy * dy) \
                - cyy[sel][None, :] * dx * dy
            alpha = np.minimum(0.99, opac[sel][None, :] * np.exp(np.clip(power, -60, 0)))
            aok = alpha >= P.MIN_ALPHA
            keepv = np.where(aok, alpha, 0.0)
            T = np.cumprod(1.0 - keepv, axis=1)
            brk = aok & (T < 1e-4)
            has = brk.any(axis=1)
            first = np.argmax(brk, axis=1)
            limit = np.where(has, first, n)
            live = aok & (np.arange(n)[None, :] < limit[:, None])
            pairs_eval += 256 * n
            pairs_alpha += int(aok.sum())
            pairs_contrib += int(live.sum())
            any_live = live.any(axis=0)
            contributed[sel[any_live]] = True
            if shuffle:
                sel = relabel[sel]
            # False sharing inside ONE tile: two live splats whose indices
            # differ only in the low bit land in the same 64 B stats line.
            live_idx = np.unique(sel[any_live])
            tile_splats += live_idx.size
            lines, inv, cnt = np.unique(live_idx // 2, return_inverse=True,
                                        return_counts=True)
            tile_statslines += lines.size
            shared_splats = live_idx[cnt[inv] > 1]
            if shared_splats.size:
                mask = np.isin(sel, shared_splats)
                pairs_shared += int(live[:, mask].sum())
        scale = (tx * ty) / float(tiles_done)
        d_splats = int(contributed.sum())
        d_lines_stats = int(np.unique(np.nonzero(contributed)[0] // 2).size)
        rows.append(dict(frame=fi, drawn=int(ok.sum()), inst=total,
                         ev=pairs_eval * scale, al=pairs_alpha * scale,
                         co=pairs_contrib * scale, dsplat=d_splats,
                         dstats=d_lines_stats, scale=scale,
                         sh=pairs_shared * scale,
                         ts=tile_splats * scale, tl=tile_statslines * scale))
        print('frame %4d  drawn %7d  inst %8d  pairs %12.0f  alphaOK %11.0f  '
              'CONTRIB %11.0f  distinct splats %7d  stats lines %7d  '
              'falseShared pairs %11.0f (%.1f%%)  [%.1fs, x%.2f]'
              % (fi, ok.sum(), total, pairs_eval * scale, pairs_alpha * scale,
                 pairs_contrib * scale, d_splats, d_lines_stats,
                 pairs_shared * scale,
                 100.0 * pairs_shared / max(pairs_contrib, 1),
                 time.time() - t0, scale))

    print()
    c = np.array([r['co'] for r in rows], float)
    ds = np.array([r['dsplat'] for r in rows], float)
    dl = np.array([r['dstats'] for r in rows], float)
    ev = np.array([r['ev'] for r in rows], float)
    al = np.array([r['al'] for r in rows], float)
    print('MEAN OVER %d FRAMES' % len(rows))
    print('  pairs evaluated                %14.0f' % ev.mean())
    print('  clear the alpha test           %14.0f' % al.mean())
    print('  CONTRIBUTING pairs (atomics)   %14.0f' % c.mean())
    print('  min / max contributing         %14.0f / %.0f' % (c.min(), c.max()))
    print('  distinct splats written        %14.0f  (%.1f%% of %d)'
          % (ds.mean(), 100 * ds.mean() / count, count))
    print('  distinct 64B stats lines       %14.0f' % dl.mean())
    print('  contributing pairs per splat   %14.1f' % (c.mean() / ds.mean()))
    print()
    print('ATOMIC OPS PER ITERATION (one frame per iteration)')
    print('  into splatGrad2D  9 x %.0f = %.0f' % (c.mean(), 9 * c.mean()))
    print('  into stats        2 x %.0f = %.0f (3 x on UNKNOWN pixels)'
          % (c.mean(), 2 * c.mean()))
    print()
    print('COMPULSORY LINE TRAFFIC PER ITERATION, backward raster only')
    g_lines = ds.mean()
    s_lines = dl.mean()
    print('  splatGrad2D lines touched      %14.0f  -> %8.2f MB in + %8.2f MB out'
          % (g_lines, g_lines * LINE / 1e6, g_lines * LINE / 1e6))
    print('  stats       lines touched      %14.0f  -> %8.2f MB in + %8.2f MB out'
          % (s_lines, s_lines * LINE / 1e6, s_lines * LINE / 1e6))
    print('  MERGING REMOVES the stats half:  %8.3f MB per iteration'
          % (2 * s_lines * LINE / 1e6))
    print('  over 3000 iterations:            %8.3f GB'
          % (3000 * 2 * s_lines * LINE / 1e9))
    sh = np.array([r['sh'] for r in rows], float)
    ts = np.array([r['ts'] for r in rows], float)
    tl = np.array([r['tl'] for r in rows], float)
    print()
    print('FALSE SHARING ON THE 32-BYTE stats RECORD (per tile)')
    print('  live (tile, splat) rows        %14.0f' % ts.mean())
    print('  distinct 64B stats lines       %14.0f' % tl.mean())
    print('  splats per stats line          %14.3f' % (ts.mean() / tl.mean()))
    print('  contributing pairs whose stats line is shared with another live')
    print('  splat in the SAME tile         %14.0f  (%.2f%% of contributing)'
          % (sh.mean(), 100 * sh.mean() / c.mean()))
    print('  -> stats atomics that collide on a line the 64B grad2D record')
    print('     never collides on:          %14.0f per iteration'
          % (2 * sh.mean()))
    print()
    print('THE CLEAR, which the merge breaks (arithmetic, not a sample)')
    n = 299209
    print('  blit fill of splatGrad2D       %14d B  = %.2f MB per iteration'
          % (n * 64, n * 64 / 1e6))
    print('  a 36-of-64 partial clear would write %.2f MB and, if the GPU has'
          % (n * 36 / 1e6))
    print('  to fetch the line to preserve the other 28 B, read %.2f MB too.'
          % (n * 64 / 1e6))


if __name__ == '__main__':
    main()
