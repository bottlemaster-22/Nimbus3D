"""Adam divides by the running RMS, so the term that decides a parameter's
direction is the one whose SIGN does not change between views, not the one with
the biggest number.  This measures the sign persistence of the PHOTOMETRIC
gradient per Gaussian across real views.

Per splat, per frame, the exact quantity trainer_rasterize_backward accumulates
into grad.opacity from the colour channel:

    dLdAlpha_i   = T_i * dot(colour_i - colourBehind_i, dLdC)
    dLdOpacity_i = sum over pixels of  gaussian_i * dLdAlpha_i

colourBehind_i is exact rather than sampled: with C the finished composite and
prefix_i the front-to-back partial sum, colourBehind_i = (C - prefix_i)/T_{i+1}.

Coherence C = |sum over frames| / sum of |per frame|.  Adam's mhat/sqrt(vhat)
tends to that ratio for a stationary gradient, and to +/-1 for a term whose sign
never changes.  The disc prior has coherence 1 by construction: trainer_regularizer
takes no view argument.

Run: python -u lossq_coherence.py [nframes]
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B
import lossq_ssim as S

LUMA = S.LUMA
TILE = 16


def frame_opacity_grad(col, count, R, t, fx, fy, cx, cy, rw, rh, gt, qcw, low_pass, active_sh):
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    g = B.draw_state(col, count, R, t, fx, fy, cx, cy, low_pass)
    rgb = B.colours(col, R, t, active_sh)
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
    jj = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_id = (np.repeat(min_y, n) + (jj // lw)) * tx + (np.repeat(min_x, n) + (jj % lw))
    order = np.lexsort((z[splat], tile_id))
    splat, tile_id = splat[order], tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))

    # pass 1: the finished composite, needed for the exposure fit and dLdC
    C = np.zeros((rh, rw, 3))
    cache = {}
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
        gauss = np.exp(np.clip(power, -60, 0))
        al = np.where(power <= 0, np.minimum(0.99, g['alpha'][sel][None, :] * gauss), 0.0)
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        wg = al * before
        colr = wg @ rgb[sel]
        y1 = min(cyt * TILE + TILE, rh)
        x1 = min(cxt * TILE + TILE, rw)
        y0, x0 = cyt * TILE, cxt * TILE
        C[y0:y1, x0:x1] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
        cache[tid] = (sel, al, before, T, gauss, wg, (y0, y1, x0, x1))

    gain, bias = B.fit_exposure(C, gt)
    rendered = gain * C + bias
    diff = rendered - gt
    N = rw * rh
    g_l1 = qcw * 0.8 / N * np.sign(diff) / 3.0
    X = (rendered * LUMA).sum(2)
    Y = (gt * LUMA).sum(2)
    SS = S.ssim_terms(X, Y, qcw * 0.2 / N)
    g_ss = SS['dLdX'][:, :, None] * LUMA[None, None, :]
    dLdC = gain * (g_l1 + g_ss)

    out = np.zeros(count)
    for tid, (sel, al, before, T, gauss, wg, box) in cache.items():
        y0, y1, x0, x1 = box
        d = np.zeros((TILE * TILE, 3))
        blk = dLdC[y0:y1, x0:x1]
        d.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0] = blk
        Cb = np.zeros((TILE * TILE, 3))
        Ct = np.zeros((TILE * TILE, 3))
        # the tile's own composite, per pixel
        colr = wg @ rgb[sel]
        Ct[:] = colr
        pref = np.cumsum(wg[:, :, None] * rgb[sel][None, :, :], axis=1)
        behind = (Ct[:, None, :] - pref) / np.maximum(T[:, :, None], 1e-6)
        dLdAlpha = before * ((rgb[sel][None, :, :] - behind) * d[:, None, :]).sum(2)
        contrib = (gauss * dLdAlpha * (al > 0)).sum(axis=0)
        np.add.at(out, sel, contrib)
    return out, ok


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    bundle = Dt.bundle()
    qc = {int(f['index']): float(f['qc']['weight']) for f in bundle['frames']}
    use = sorted(frames)[:nf]

    G = np.zeros((len(use), count))
    V = np.zeros((len(use), count), bool)
    for k, f in enumerate(use):
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        _, gt = Dt.luma_of(frames[f], rw, rh)
        g, ok = frame_opacity_grad(col, count, R, t, fx, fy, cx, cy, rw, rh,
                                   gt.astype(np.float64), qc[f],
                                   0.25 + 1.33333, 2)
        G[k] = g
        V[k] = ok
        print('  frame %2d done, %d drawn, sum|g| %.4e' % (f, ok.sum(), np.abs(g).sum()))

    np.save(os.path.join(Dt.SCRATCH, 'coh_G.npy'), G)
    # "seen" must mean "actually received a gradient", not "was projected":
    # nearly every splat projects into nearly every frame, so V is not a filter.
    seen = (G != 0).sum(0)
    print('\n  frames with a NON-ZERO photometric gradient, per splat:')
    print('    p10 %d p50 %d p90 %d max %d' % tuple(np.percentile(seen, [10, 50, 90, 100])))
    for minseen in (4, 6, 10, 12):
        m = seen >= minseen
        s_abs = np.abs(G).sum(0)[m]
        s_net = np.abs(G.sum(0))[m]
        live = s_abs > 0
        coh = s_net[live] / s_abs[live]
        print('\n=== PHOTOMETRIC dL/dOpacity, splats visible in >= %d of %d views (%d splats) ==='
              % (minseen, len(use), int(live.sum())))
        print('  coherence |sum| / sum|.|  : p10 %.3f  p25 %.3f  MEDIAN %.3f  p75 %.3f  p90 %.3f'
              % tuple(np.percentile(coh, [10, 25, 50, 75, 90])))
        print('  a random-sign gradient over %d views would average %.3f'
              % (minseen, np.mean([np.abs(np.random.RandomState(q).choice([-1,1],minseen).sum())/minseen for q in range(400)])))
        print('  mean %.3f' % coh.mean())
        print('  a perfectly view-consistent term has coherence 1.000')
        print('  fraction of splats below coherence 0.5: %.1f%%' % (100 * (coh < 0.5).mean()))
        print('  Adam step ratio a coherent term needs to match this photometric term:')
        print('    a term with coherence 1.0 and magnitude m matches a photometric term')
        print('    of magnitude m / %.3f (the median), i.e. it can be %.1fx SMALLER'
              % (np.median(coh), 1.0 / max(np.median(coh), 1e-9)))
    print('\n  NOTE these %d views are consecutive frames from one %.0f-second window of'
          % (len(use), 8))
    print('  the capture, so their disagreement is the SMALLEST in the scan. Coherence')
    print('  measured over the 108 keyframes spanning 276 s can only be lower.')


if __name__ == '__main__':
    main()
