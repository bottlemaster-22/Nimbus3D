"""The per-term gradient ledger at a real pixel, in ONE unit: dL/dalpha of the
FRONT-MOST contributing Gaussian, which is what trainer_rasterize_backward
computes and what every downstream gradient (opacity, mean2D, conic) is built
from.

  dLdAlpha  = T_i * dot(colour_i - colourBehind_i, dLdC)          <- photometric
            + T_i * (depth_i - depthBehind_i) * dLdD              <- depth channel
            + (-TFinal / (1 - alpha_i)) * (gradTFinal + dot(bg, dLdC))

with, from trainer_loss_depth,
  w             = depthScale * s.weight * invSamples
  dL/dExpected  = w*huberGrad  +  w*bimodal*nearer  +  w*trans*4(1-2t)/span
                  - freeSpaceWeight*depthScale*invSamples*violation
  dLdD          = dL/dExpected / max(alpha, 1e-4)
  gradTFinal   += 2 * w * alphaSupervisionWeight * (1 - alpha)     <- F6
               +  dL/dExpected * accumulated / alpha^2

For the front-most Gaussian T_i = 1 and the "behind" composite is exact from the
finished render:  colourBehind = (C - alpha_0 c_0) / (1 - alpha_0).

Run: python -u lossq_ledger.py [frame ...]
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B
import lossq_ssim as S

LUMA = S.LUMA
TILE = 16

DEPTH_SCALE = 1.0          # measured: the schedule is 1.0 for every iteration
                           # of a 4000-iteration run inside a 30000 budget
ALPHA_SUP_W = 0.05         # TrainerTuning.alphaSupervisionWeight
FREESPACE_W = 0.5          # SmartLossSettings.freeSpaceLowerBoundWeight
BIMODAL_W = 1.0
TRANS_W = 0.35
EDGE_BOOST = 2.0
HUBER_FLOOR = 0.004
SIGMA_MEDIAN = 0.012980087  # prepass/census.json measuredMedianSigmaMeters


def render_with_depth(col, count, R, t, fx, fy, cx, cy, rw, rh, low_pass, active_sh):
    """B.render, plus the depth channel and the front-most contributor."""
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
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_id = (np.repeat(min_y, n) + (j // lw)) * tx + (np.repeat(min_x, n) + (j % lw))
    order = np.lexsort((z[splat], tile_id))
    splat, tile_id = splat[order], tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))

    C = np.zeros((rh, rw, 3))
    D = np.zeros((rh, rw))
    Tf = np.ones((rh, rw))
    a0 = np.zeros((rh, rw))          # front-most contributor's alpha
    c0 = np.zeros((rh, rw, 3))       # its colour
    d0 = np.zeros((rh, rw))          # its depth
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
        wgt = al * before
        colr = wgt @ rgb[sel]
        dep = wgt @ z[sel]
        live = al > 0
        first = np.where(live.any(1), np.argmax(live, axis=1), -1)
        fa = np.where(first >= 0, al[np.arange(al.shape[0]), np.maximum(first, 0)], 0.0)
        fc = rgb[sel][np.maximum(first, 0)] * (first >= 0)[:, None]
        fd = z[sel][np.maximum(first, 0)] * (first >= 0)

        y1 = min(cyt * TILE + TILE, rh)
        x1 = min(cxt * TILE + TILE, rw)
        y0, x0 = cyt * TILE, cxt * TILE
        sl = (slice(y0, y1), slice(x0, x1))
        C[sl] = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
        D[sl] = dep.reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
        Tf[sl] = T[:, -1].reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
        a0[sl] = fa.reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
        c0[sl] = fc.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
        d0[sl] = fd.reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
    return C, D, Tf, a0, c0, d0


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    N = rw * rh
    SUP = sl['depthSamplesSupervisedTotal'] / sl['depthSupervisionFramesMeasured']
    ALL = sl['depthSamplesPerFrameTotal'] / sl['depthSupervisionFramesMeasured']
    invS = 1.0 / SUP
    invN = 1.0 / N
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    bundle = Dt.bundle()
    qc = {int(f['index']): float(f['qc']['weight']) for f in bundle['frames']}
    allqc = np.array([float(f['qc']['weight']) for f in bundle['frames']])
    want = [int(a) for a in sys.argv[1:]] or [0, 8]

    print('=== MEASURED SUPERVISION GEOMETRY (from model/train_census.json) ===')
    print('  depth samples dispatched per frame   %.0f   (256 x 192 = %d)' % (ALL, 256 * 192))
    print('  of those, SUPERVISED (weight > 0)    %.0f   = %.2f%%' % (SUP, 100 * SUP / ALL))
    print('  render pixels                        %d' % N)
    print('  so the depth terms reach at most     %.2f%% of pixels, %.2f%% carry weight'
          % (100 * ALL / N, 100 * SUP / N))
    print('  invSamples = 1/%.0f = %.4e     invN = 1/%d = %.4e' % (SUP, invS, N, invN))
    print('  invSamples / invN = %.2f  (a supervised sample is worth this many pixels)'
          % (invS / invN))
    print('  qc.weight over all %d bundle frames: p10 %.3f p50 %.3f p90 %.3f'
          % (allqc.size, *np.percentile(allqc, [10, 50, 90])))

    for frame in want:
        rot, t = poses[frame]
        R = P.quat_to_matrix(rot)
        for label, lp, ash in (('iteration 2000', 0.25 + 2.66667, 1),
                               ('iteration 4000', 0.25 + 1.33333, 2)):
            C, D, Tf, a0, c0, d0 = render_with_depth(
                col, count, R, t, fx, fy, cx, cy, rw, rh, lp, ash)
            _, gt = Dt.luma_of(frames[frame], rw, rh)
            gt = gt.astype(np.float64)
            gain, bias = B.fit_exposure(C, gt)
            rendered = gain * C + bias
            qcw = qc[frame]

            # ---- photometric gradFinal ------------------------------------
            diff = rendered - gt
            g_l1 = qcw * (1 - S.LAMBDA_SSIM) * invN * np.sign(diff) / 3.0
            X = (rendered * LUMA).sum(2)
            Y = (gt * LUMA).sum(2)
            SS = S.ssim_terms(X, Y, qcw * S.LAMBDA_SSIM / N)
            g_ss = SS['dLdX'][:, :, None] * LUMA[None, None, :]
            dLdC = gain * (g_l1 + g_ss)              # trainer_loss_finalize

            alpha_px = 1.0 - Tf
            behind = np.where((1 - a0)[:, :, None] > 1e-6,
                              (C - a0[:, :, None] * c0) / np.maximum((1 - a0)[:, :, None], 1e-6),
                              0.0)
            dBehind = np.where(1 - a0 > 1e-6, (D - a0 * d0) / np.maximum(1 - a0, 1e-6), 0.0)

            live = a0 > 0
            dA_col = ((c0 - behind) * dLdC).sum(2)                   # T_i = 1

            # ---- depth channel, per unit dL/dExpected ---------------------
            unit_dLdD = 1.0 / np.maximum(alpha_px, 1e-4)
            dA_depth_per_unit = (d0 - dBehind) * unit_dLdD
            # the T channel that the depth normalisation also feeds
            dA_T_per_unit = (-Tf / np.maximum(1 - a0, 1e-6)) * (D / np.maximum(alpha_px, 1e-4) ** 2)

            # ---- the exact coefficients -----------------------------------
            # s.weight = trustWeight * authority * qc, clamped to 0..4, x2 on a
            # geometric edge.  Only its RANGE is on disk, so it is swept.
            print('\n=========== frame %d, %s (lowPass %.4f, SH %d) ==========='
                  % (frame, label, lp, ash))
            print('  render PSNR %.3f dB (exposure gain %.4f bias %+.4f), qc.weight %.4f'
                  % (10 * np.log10(1 / max(((rendered - gt) ** 2).mean(), 1e-12)),
                     gain, bias, qcw))
            print('  pixels with a contributor %.2f%%, median alpha %.4f, median T_final %.5f'
                  % (100 * live.mean(), np.median(alpha_px), np.median(Tf)))
            m = live
            print('  |dL/dalpha| from PHOTOMETRY at the front-most Gaussian:')
            print('     L1   only  p50 %.4e  p90 %.4e' %
                  tuple(np.percentile(np.abs(((c0 - behind) * (gain * g_l1)).sum(2))[m], [50, 90])))
            print('     SSIM only  p50 %.4e  p90 %.4e' %
                  tuple(np.percentile(np.abs(((c0 - behind) * (gain * g_ss)).sum(2))[m], [50, 90])))
            print('     both       p50 %.4e  p90 %.4e' %
                  tuple(np.percentile(np.abs(dA_col)[m], [50, 90])))
            print('  |colour_i - colourBehind| p50 %.4f   |depth_i - depthBehind| p50 %.4f m'
                  % (np.median(np.abs(c0 - behind)[m]), np.median(np.abs(d0 - dBehind)[m])))

            print('  DEPTH CHANNEL, per unit dL/dExpected:')
            print('     |dLdalpha| p50 %.4e  p90 %.4e (depth term)'
                  % tuple(np.percentile(np.abs(dA_depth_per_unit)[m], [50, 90])))
            print('     |dLdalpha| p50 %.4e  p90 %.4e (T normalisation term)'
                  % tuple(np.percentile(np.abs(dA_T_per_unit)[m], [50, 90])))

            print('  RATIO depth / photometric at a supervised pixel, by s.weight:')
            print('     %-46s %12s %12s %12s' % ('term', 'dL/dExpected', 'vs L1', 'vs L1+SSIM'))
            ph_l1 = np.median(np.abs(((c0 - behind) * (gain * g_l1)).sum(2))[m])
            ph_all = np.median(np.abs(dA_col)[m])
            dep_unit = np.median(np.abs(dA_depth_per_unit + dA_T_per_unit)[m])
            rows = [
                ('Huber, s.weight 0.15 (0.5 trust x 0.5 auth x 0.6 qc), |hg|=1',
                 DEPTH_SCALE * 0.15 * invS),
                ('Huber, s.weight 0.5, |hg|=1', DEPTH_SCALE * 0.5 * invS),
                ('Huber, s.weight clamp 4.0, |hg|=1', DEPTH_SCALE * 4.0 * invS),
                ('F6 alpha supervision, s.weight 0.5, alpha 0.98',
                 DEPTH_SCALE * 0.5 * invS * ALPHA_SUP_W * 2 * 0.02),
                ('F2 free space, violation 0.05 m (the margin itself)',
                 FREESPACE_W * DEPTH_SCALE * invS * 0.05),
                ('F2 free space, violation 0.50 m', FREESPACE_W * DEPTH_SCALE * invS * 0.50),
                ('F2 free space, violation 2.00 m', FREESPACE_W * DEPTH_SCALE * invS * 2.00),
                ('F4 transition width, span 0.30 m, s.weight 0.5 x edge boost 2',
                 DEPTH_SCALE * 1.0 * invS * TRANS_W * 4 / 0.30),
                ('F4 transition width, span 0.04 m (the edge threshold)',
                 DEPTH_SCALE * 1.0 * invS * TRANS_W * 4 / 0.04),
                ('F4 transition width, span 0.001 m', DEPTH_SCALE * 1.0 * invS * TRANS_W * 4 / 0.001),
                ('F4 transition width, span at the 1e-4 floor',
                 DEPTH_SCALE * 1.0 * invS * TRANS_W * 4 / 1e-4),
            ]
            for name, coeff in rows:
                print('     %-46s %12.4e %12.2f %12.2f'
                      % (name, coeff, coeff * dep_unit / ph_l1, coeff * dep_unit / ph_all))

            # Huber saturation: is |r| bigger than delta in practice?
            print('  Huber delta = max(sigma, 0.004); measured median sigma %.4f m,'
                  % SIGMA_MEDIAN)
            print('    so |r| > %.4f m saturates the Huber to a constant +/-w.' % SIGMA_MEDIAN)
            print('    the model\'s own front-to-behind depth spread is p50 %.4f m,'
                  % np.median(np.abs(d0 - dBehind)[m]))
            print('    i.e. %.1fx the Huber knee: the Huber is in its LINEAR regime,'
                  % (np.median(np.abs(d0 - dBehind)[m]) / SIGMA_MEDIAN))
            print('    where its gradient is a constant sign and carries no magnitude.')


if __name__ == '__main__':
    main()
