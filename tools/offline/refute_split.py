"""How much does the tile sort's depth-key width actually cost in quality?

The radix sort keys are (tile << 16) | depth16.  32 bits, eight 4-bit passes.
A stale comment in trainer_duplicate_keys describes a 13-bit depth / 24-bit
key / six-pass design that the code has never implemented, and justifies it
with "3.7 mm at a 30 m far plane".  The configured far plane is 100 m, not 30,
so the real quantum is 12.2 mm, not 3.7.  This measures whether that matters.

Renders one frame with several depth-key widths and spans, all with the GPU's
stable tie-break (increasing splat id within a tile), and reports PSNR against
the exact-order render and against the real photograph.
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt

TILE = 16


def build(col, count, R, t, fx, fy, cx, cy, rw, rh):
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
    sa = sa + P.FILTER_2D_VARIANCE
    sc = sc + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    inv_det = 1.0 / det
    conic = (sc * inv_det, -sb * inv_det, sa * inv_det)
    det_before = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE)
                            - sb * sb, 1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d

    centre = -R.T @ t
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    d = mean - centre
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1).astype(np.float64)
    rgb = 0.5 + Dt.C0 * dc
    if 'f_rest_0' in col:
        s1 = np.stack([col['f_rest_0'], col['f_rest_3'], col['f_rest_6']], axis=1)
        s2 = np.stack([col['f_rest_1'], col['f_rest_4'], col['f_rest_7']], axis=1)
        s3 = np.stack([col['f_rest_2'], col['f_rest_5'], col['f_rest_8']], axis=1)
        rgb += (-Dt.C1 * d[:, 1:2] * s1 + Dt.C1 * d[:, 2:3] * s2
                - Dt.C1 * d[:, 0:1] * s3)
    rgb = np.clip(rgb, 0, 1)

    idx = np.nonzero(ok)[0]
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
    return dict(tx=tx, ty=ty, splat=splat, tile_id=tile_id, mx=mx, my=my, z=z,
                conic=conic, opacity=opacity, rgb=rgb, total=total)


def paint(g, order, rw, rh):
    tx, ty = g['tx'], g['ty']
    splat = g['splat'][order]
    tile_id = g['tile_id'][order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))
    mx, my = g['mx'], g['my']
    cxk, cyk, czk = g['conic']
    opacity, rgb = g['opacity'], g['rgb']
    out = np.zeros((rh, rw, 3), np.float32)
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
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = -0.5 * (cxk[sel][None, :] * dx * dx + czk[sel][None, :] * dy * dy) \
            - cyk[sel][None, :] * dx * dy
        al = np.where(power <= 0,
                      np.minimum(0.99, opacity[sel][None, :]
                                 * np.exp(np.clip(power, -60, 0))), 0.0)
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        colr = (al * before) @ rgb[sel]
        y0, y1 = cyt * TILE, min(cyt * TILE + TILE, rh)
        x0, x1 = cxt * TILE, min(cxt * TILE + TILE, rw)
        out[y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
    return out


def key_order(g, bits, near, far):
    z = g['z'][g['splat']]
    span = max(far - near, 1e-3)
    norm = np.clip((z - near) / span, 0.0, 1.0)
    q = np.floor(norm * float((1 << bits) - 1)).astype(np.int64)
    return np.lexsort((np.arange(g['total']), q, g['tile_id'])), q


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    frames = [int(x) for x in sys.argv[1:]] or [4, 8, 12]
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    photos = dict(Dt.frames_with_images())

    cases = [
        ('16b span 0.05-100 SHIPPED    ', 16, 0.05, 100.0),
        ('13b span 0.05-10  tile11bits ', 13, 0.05, 10.0),
        ('12b span 0.05-10  tile12bits ', 12, 0.05, 10.0),
        ('12b span 0.05-6   tile12bits ', 12, 0.05, 6.0),
        ('12b span 0.05-4   tile12bits ', 12, 0.05, 4.0),
        ('13b span 0.05-6   tile11bits ', 13, 0.05, 6.0),
    ]
    for f in frames:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        g = build(col, count, R, t, fx, fy, cx, cy, rw, rh)
        zz = g['z'][g['splat']]
        exact = np.lexsort((np.arange(g['total']), zz, g['tile_id']))
        base = paint(g, exact, rw, rh)
        _, photo = Dt.luma_of(photos[f], rw, rh)
        print('')
        print('frame %d   instances %d   depth p1 %.2f p50 %.2f p99 %.2f max %.2f m'
              % (f, g['total'], np.percentile(zz, 1), np.percentile(zz, 50),
                 np.percentile(zz, 99), zz.max()))
        print('  exact float order         : vs photo %.4f' % psnr(base, photo))
        for name, bits, near, far in cases:
            o, q = key_order(g, bits, near, far)
            img = paint(g, o, rw, rh)
            same_tile = g['tile_id'][o][1:] == g['tile_id'][o][:-1]
            tied = same_tile & (q[o][1:] == q[o][:-1])
            swapped = int((np.asarray(o) != np.asarray(exact)).sum())
            print('  %s: vs photo %.4f  vs exact %6.2f dB  maxpix %.4f  ties %5.2f%%  moved %d/%d'
                  % (name, psnr(img, photo), psnr(img, base),
                     np.abs(img - base).max(),
                     100.0 * tied.sum() / max(same_tile.sum(), 1),
                     swapped, g['total']))


if __name__ == '__main__':
    main()
