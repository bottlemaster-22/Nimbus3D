"""Measure what the forward rasteriser's transmittance early-out actually saves,
and what raising it would save, on the REAL build-244 model.

The shipped loop (TrainerShaders.metal:1406-1412) is

    alpha = min(0.99, o*exp(power));
    if (alpha < cam.minAlpha) continue;
    testT = T * (1 - alpha);
    if (testT < 1e-4f) { done = true; break; }

so a pixel keeps iterating until its transmittance would fall under 1e-4.
`done` is PER LANE. The batch loop above it still runs `batches` times and
still performs the cooperative load of 256 raster records into threadgroup
memory for every batch, because nothing tests whether the whole threadgroup
has finished.

Measured here, per threshold tau, over full frames:
  * inner-loop (pixel, Gaussian) visits   -> the ALU/exp work
  * batches whose cooperative load is still needed if a group-wide all-done
    test were added                       -> the device-memory traffic
  * the rendered image, so the quality cost of tau is a measured PSNR.

Run: python -u refute_earlyout.py [frame ...]
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import detail as Dt

TILE = 16
AREA = 256
TAUS = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def prep(col, count, R, t, fx, fy, cx, cy, rw, rh):
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
    dist = np.linalg.norm(d, axis=1)
    d = d / np.maximum(dist[:, None], 1e-9)
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1).astype(np.float64)
    rgb = 0.5 + Dt.C0 * dc
    if 'f_rest_0' in col:
        s1 = np.stack([col['f_rest_0'], col['f_rest_3'], col['f_rest_6']], axis=1)
        s2 = np.stack([col['f_rest_1'], col['f_rest_4'], col['f_rest_7']], axis=1)
        s3 = np.stack([col['f_rest_2'], col['f_rest_5'], col['f_rest_8']], axis=1)
        rgb = rgb + (-Dt.C1 * d[:, 1:2] * s1 + Dt.C1 * d[:, 2:3] * s2
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
    idx = idx[keep]
    min_x = min_x[keep]
    min_y = min_y[keep]
    w = w[keep]
    n = n[keep]
    total = int(n.sum())
    splat = np.repeat(idx, n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_x = np.repeat(min_x, n) + (j % lw)
    tile_y = np.repeat(min_y, n) + (j // lw)
    tile_id = tile_y * tx + tile_x
    order = np.lexsort((z[splat], tile_id))
    splat = splat[order]
    tile_id = tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))
    return dict(tx=tx, ty=ty, mx=mx, my=my, conic=conic, opacity=opacity,
                rgb=rgb, splat=splat, bounds=bounds, total=total, dist=dist)


def run(g, rw, rh):
    tx, ty = g['tx'], g['ty']
    mx, my = g['mx'], g['my']
    cxk, cyk, czk = g['conic']
    opacity, rgb = g['opacity'], g['rgb']
    splat, bounds = g['splat'], g['bounds']

    ntau = len(TAUS)
    imgs = np.zeros((ntau + 1, rh, rw, 3), np.float64)
    visits = np.zeros(ntau + 1, np.int64)
    batches_used = np.zeros(ntau + 1, np.int64)
    batches_total = 0
    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat[a:b]
        K = int(sel.size)
        batches_total += (K + AREA - 1) // AREA
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
        wfull = al * before
        y0, y1 = cyt * TILE, min(cyt * TILE + TILE, rh)
        x0, x1 = cxt * TILE, min(cxt * TILE + TILE, rw)
        ar = np.arange(K)[None, :]
        for ti, tau in enumerate(TAUS + [0.0]):
            if tau <= 0:
                stop = np.full(256, K, np.int64)
                ncomp = stop
            else:
                bad = t_after < tau
                any_bad = bad.any(axis=1)
                first = np.argmax(bad, axis=1)
                stop = np.where(any_bad, first + 1, K)
                ncomp = np.where(any_bad, first, K)
            visits[ti] += int(stop.sum())
            batches_used[ti] += (int(stop.max()) + AREA - 1) // AREA
            colr = (wfull * (ar < ncomp[:, None])) @ rgb[sel]
            imgs[ti, y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
    return imgs, visits, batches_used, batches_total


def psnr(a, b):
    m = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(m, 1e-14))


def main():
    frames = [int(x) for x in sys.argv[1:]] or [0, 4, 8, 12]
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()

    nt = len(TAUS)
    agg_v = np.zeros(nt + 1, np.int64)
    agg_b = np.zeros(nt + 1, np.int64)
    agg_bt = 0
    dbs = dict((t, []) for t in TAUS)
    for f in frames:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        g = prep(col, count, R, t, fx, fy, cx, cy, rw, rh)
        imgs, v, bu, bt = run(g, rw, rh)
        agg_v += v
        agg_b += bu
        agg_bt += bt
        ref = imgs[-1]
        print('frame %d: %d tile instances, %d batch loads shipped'
              % (f, g['total'], bt))
        for ti, tau in enumerate(TAUS):
            d = psnr(imgs[ti], ref)
            dbs[tau].append(d)
            print('   tau %-7g visits %11d (%6.2f%% of shipped)  '
                  'loads %6d (%6.2f%%)  PSNR vs exact %7.2f dB  maxerr %.5f'
                  % (tau, v[ti], 100.0 * v[ti] / v[0], bu[ti],
                     100.0 * bu[ti] / bt, d, float(np.abs(imgs[ti] - ref).max())))
        print('   no early out at all: visits %d (%.2f%% of shipped)'
              % (v[-1], 100.0 * v[-1] / v[0]))
        sys.stdout.flush()

    print('')
    print('AGGREGATE over %d frames' % len(frames))
    for ti, tau in enumerate(TAUS):
        print('  tau %-7g  visits %12d = %6.2f%% of SHIPPED inner work;  '
              'group-all-done loads %8d = %6.2f%% of shipped loads;  '
              'mean PSNR vs exact %6.2f dB'
              % (tau, agg_v[ti], 100.0 * agg_v[ti] / agg_v[0],
                 agg_b[ti], 100.0 * agg_b[ti] / agg_bt,
                 float(np.mean(dbs[tau]))))
    print('  no early out   visits %12d = %6.2f%% of shipped'
          % (agg_v[-1], 100.0 * agg_v[-1] / agg_v[0]))
    print('  shipped batch loads (no group all-done test): %d' % agg_bt)


if __name__ == '__main__':
    main()
