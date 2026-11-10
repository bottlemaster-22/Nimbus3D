"""What the frequency-blur schedule does to the render the loss is computed on,
and how far that render is from the one the held-out score is taken on.

trainer_preprocess:  lowPass = cam.filter2DVariance + cam.frequencyBlurVariance
                     sa += lowPass; sc += lowPass
                     comp2D = sqrt(detBefore / det)          <- opacity paid back
                     alpha  = sigmoid(logit) * comp2D * comp3D
                     if (alpha < cam.minAlpha) return;       <- CULLED, no gradient

Run: python -u lossq_blur.py
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt

TILE = 16
C0, C1 = Dt.C0, Dt.C1


def draw_state(col, count, R, t, fx, fy, cx, cy, low_pass):
    """Everything trainer_preprocess computes, at this lowPass."""
    mx, my, z, sa0, sb, sc0, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
    det_before = np.maximum(sa0 * sc0 - sb * sb, 1e-12)
    sa = sa0 + low_pass
    sc = sc0 + low_pass
    det = sa * sc - sb * sb
    comp2d = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))
    inv_det = 1.0 / np.maximum(det, 1e-12)
    sig = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    alpha = sig * comp2d
    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))
    level = np.clip(2.0 * np.log(np.maximum(alpha, 1e-12) / P.MIN_ALPHA), 0, 9)
    k = np.sqrt(level)
    ex = k * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = k * np.sqrt(np.maximum(sc, 1e-9)) * 1.001
    return dict(mx=mx, my=my, z=z, cx=sc * inv_det, cy=-sb * inv_det, cz=sa * inv_det,
                alpha=alpha, comp2d=comp2d, sigmoid=sig, radius=radius,
                ex=ex, ey=ey, det=det)


def colours(col, R, t, active_sh):
    centre = -R.T @ t
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    d = mean - centre
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1).astype(np.float64)
    rgb = 0.5 + C0 * dc
    s = [np.stack([col['f_rest_%d' % i], col['f_rest_%d' % (i + 3)],
                   col['f_rest_%d' % (i + 6)]], axis=1).astype(np.float64)
         for i in (0, 1, 2)]
    if active_sh > 1:
        rgb = rgb + (-C1 * d[:, 1:2] * s[0])
    if active_sh > 2:
        rgb = rgb + (C1 * d[:, 2:3] * s[1])
    if active_sh > 3:
        rgb = rgb + (-C1 * d[:, 0:1] * s[2])
    return np.clip(rgb, 0, 1)


def render(col, count, R, t, fx, fy, cx, cy, rw, rh, low_pass, active_sh):
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    g = draw_state(col, count, R, t, fx, fy, cx, cy, low_pass)
    rgb = colours(col, R, t, active_sh)
    ok = ((g['z'] > 0.05) & (g['z'] < 100) & (g['det'] > 1e-12)
          & (g['radius'] >= 0.5) & (g['alpha'] >= P.MIN_ALPHA))
    idx = np.nonzero(ok)[0]
    mx, my, z = g['mx'], g['my'], g['z']
    min_x = np.maximum(0, np.floor((mx - g['ex']) / TILE)).astype(np.int64)[idx]
    min_y = np.maximum(0, np.floor((my - g['ey']) / TILE)).astype(np.int64)[idx]
    max_x = np.minimum(tx, np.ceil((mx + g['ex']) / TILE)).astype(np.int64)[idx]
    max_y = np.minimum(ty, np.ceil((my + g['ey']) / TILE)).astype(np.int64)[idx]
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
    tile_id = (np.repeat(min_y, n) + (j // lw)) * tx + (np.repeat(min_x, n) + (j % lw))
    order = np.lexsort((z[splat], tile_id))
    splat, tile_id = splat[order], tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))

    out = np.zeros((rh, rw, 3), np.float64)
    Tf = np.ones((rh, rw), np.float64)
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
        power = (-0.5 * (g['cx'][sel][None, :] * dx * dx
                         + g['cz'][sel][None, :] * dy * dy)
                 - g['cy'][sel][None, :] * dx * dy)
        al = np.where(power <= 0, np.minimum(0.99, g['alpha'][sel][None, :]
                                             * np.exp(np.clip(power, -60, 0))), 0.0)
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        colr = (al * before) @ rgb[sel]
        y1 = min(cyt * TILE + TILE, rh)
        x1 = min(cxt * TILE + TILE, rw)
        y0, x0 = cyt * TILE, cxt * TILE
        out[y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
        Tf[y0:y1, x0:x1] = T[:, -1].reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
    return out, Tf, ok, g, total


def fit_exposure(x, y):
    """The exact least squares evaluateHeldOut solves, with the same clamps."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    n = float(x.size)
    den = n * (x * x).sum() - x.sum() ** 2
    if den <= 1e-9:
        return 1.0, 0.0
    gain = (n * (x * y).sum() - x.sum() * y.sum()) / den
    bias = (y.sum() - gain * x.sum()) / n
    return float(np.clip(gain, 0.9, 1.1)), float(np.clip(bias, -0.05, 0.05))


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(m, 1e-12))


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    use = sorted(frames)[:8]

    configs = [
        ('training at iteration 2000', 0.25 + 2.66667, 1),
        ('training at iteration 4000', 0.25 + 1.33333, 2),
        ('held-out eval and viewer', 0.25, 4),
        ('eval lowPass, train SH', 0.25, 2),
        ('train lowPass 4000, SH 4', 0.25 + 1.33333, 4),
    ]

    print('=== PER-SPLAT DRAW STATE, frame %d, %d splats ===' % (use[0], count))
    rot, t = poses[use[0]]
    R = P.quat_to_matrix(rot)
    print('%-28s %8s %9s %20s %11s %10s %10s'
          % ('config', 'lowPass', 'drawn', 'culled a<1/255', 'med comp2D', 'med alpha', 'med radius'))
    for name, lp, _ in configs[:3]:
        g = draw_state(col, count, R, t, fx, fy, cx, cy, lp)
        infront = (g['z'] > 0.05) & (g['z'] < 100)
        drawn = infront & (g['alpha'] >= P.MIN_ALPHA) & (g['radius'] >= 0.5)
        culled = infront & (g['alpha'] < P.MIN_ALPHA)
        print('%-28s %8.4f %9d %11d (%5.2f%%) %11.4f %10.4f %10.3f'
              % (name, lp, drawn.sum(), culled.sum(),
                 100 * culled.sum() / max(infront.sum(), 1),
                 np.median(g['comp2d'][infront]), np.median(g['alpha'][infront]),
                 np.median(g['radius'][infront])))

    print('\n=== PSNR OF THE SAME MODEL UNDER EACH RENDER CONFIG ===')
    print('(no learned exposure on disk; raw and least-squares-fitted both shown)')
    print('%-28s %8s %4s %11s %12s %12s'
          % ('config', 'lowPass', 'SH', 'PSNR raw', 'PSNR expfit', 'tile inst'))
    res = {}
    for name, lp, ash in configs:
        raws, fits, insts = [], [], []
        for f in use:
            rot, t = poses[f]
            R = P.quat_to_matrix(rot)
            img, Tf, ok, g, inst = render(col, count, R, t, fx, fy, cx, cy, rw, rh, lp, ash)
            _, gt = Dt.luma_of(frames[f], rw, rh)
            gt = gt.astype(np.float64)
            raws.append(psnr(img, gt))
            gn, bs = fit_exposure(img, gt)
            fits.append(psnr(np.clip(gn * img + bs, 0, 1), gt))
            insts.append(inst)
        res[name] = (float(np.mean(raws)), float(np.mean(fits)))
        print('%-28s %8.4f %4d %11.3f %12.3f %12d'
              % (name, lp, ash, np.mean(raws), np.mean(fits), int(np.mean(insts))))

    a = res['training at iteration 4000'][1]
    b = res['held-out eval and viewer'][1]
    print('\n  render the loss optimised at iteration 4000 scores %.3f dB' % a)
    print('  render the held-out score is taken on scores       %.3f dB' % b)
    print('  MISMATCH %+.3f dB, mean over %d real frames' % (b - a, len(use)))


if __name__ == '__main__':
    main()
