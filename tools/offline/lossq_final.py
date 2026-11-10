"""Four closing measurements.

1. What the two free per-frame exposure scalars are worth, and whether the
   optimiser is pinned against their clamps.
2. How thick the model's surfaces are along a ray, which is the lever the
   depth channel multiplies by (depth_i - depthBehind_i).
3. Frame-wide gradient mass by term, which is what decides where geometry goes.
4. The effective-rank spike, at a resolution fine enough to see whether it is
   a spike or a wide bin.

Run: python -u lossq_final.py
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B
import lossq_ssim as S
import lossq_ledger as L

LUMA = S.LUMA
TILE = 16


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(m, 1e-12))


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    N = rw * rh
    SUP = sl['depthSamplesSupervisedTotal'] / sl['depthSupervisionFramesMeasured']
    invS, invN = 1.0 / SUP, 1.0 / N
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    bundle = Dt.bundle()
    qc = {int(f['index']): float(f['qc']['weight']) for f in bundle['frames']}
    use = sorted(frames)

    print('=== 1. WHAT THE PER-FRAME EXPOSURE (gain, bias) IS WORTH ===')
    print('  exposureGainRange 0.9...1.1, exposureBiasRange -0.05...0.05 '
          '(TrainerSupport.swift:953-954)')
    print('%5s %10s %10s %10s %10s %10s %9s %7s' %
          ('frame', 'PSNR id', 'PSNR fit', 'gain', 'bias', 'gain uncl', 'bias uncl', 'gain dB'))
    pinned = 0
    gains = []
    for f in use:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        img, Tf, ok, g, inst = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
        _, gt = Dt.luma_of(frames[f], rw, rh)
        gt = gt.astype(np.float64)
        x = img.ravel()
        y = gt.ravel()
        n = float(x.size)
        den = n * (x * x).sum() - x.sum() ** 2
        gu = (n * (x * y).sum() - x.sum() * y.sum()) / den
        bu = (y.sum() - gu * x.sum()) / n
        gn, bs = float(np.clip(gu, 0.9, 1.1)), float(np.clip(bu, -0.05, 0.05))
        p_id = psnr(img, gt)
        p_ft = psnr(np.clip(gn * img + bs, 0, 1), gt)
        hit = (abs(gn - gu) > 1e-9) or (abs(bs - bu) > 1e-9)
        pinned += 1 if hit else 0
        gains.append(p_ft - p_id)
        print('%5d %10.3f %10.3f %10.4f %10.4f %10.4f %9.4f %7.3f%s'
              % (f, p_id, p_ft, gn, bs, gu, bu, p_ft - p_id, '  CLAMPED' if hit else ''))
    print('  mean PSNR bought by two per-frame scalars: %+.3f dB (max %+.3f)'
          % (np.mean(gains), np.max(gains)))
    print('  frames whose best (gain, bias) is OUTSIDE the allowed box: %d of %d'
          % (pinned, len(use)))
    print('  census: trained views are scored WITH their learned exposure,')
    print('          held-out raw is scored at gain 1 bias 0 (16.000 dB) and')
    print('          held-out fitted at the same clamped least squares (16.335 dB).')

    print('\n=== 2. HOW THICK A SURFACE IS ALONG A RAY ===')
    print('  alpha-weighted depth mean and sd per pixel, from the finished composite')
    for f in (0, 8):
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        C, D, Tf, a0, c0, d0 = L.render_with_depth(
            col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25 + 1.33333, 2)
        # second moment needs another pass; recompute with a depth^2 channel
        alpha = 1 - Tf
        expected = D / np.maximum(alpha, 1e-4)
        live = a0 > 0
        print('  frame %d: front-most contributor alpha p50 %.4f, its depth p50 %.3f m,'
              % (f, np.median(a0[live]), np.median(d0[live])))
        print('           alpha-weighted expected depth p50 %.3f m,'
              % np.median(expected[live]))
        print('           |front-most - everything behind| p50 %.3f m p90 %.3f m'
              % tuple(np.percentile(np.abs(d0 - np.where(1 - a0 > 1e-6,
                                                         (D - a0 * d0) / np.maximum(1 - a0, 1e-6), 0))[live],
                                    [50, 90])))
        print('           expected - front-most depth: p50 %+.3f m  (positive = the'
              % np.median((expected - d0)[live]))
        print('           first thing the ray hits is IN FRONT of where the mass is)')

    print('\n=== 3. FRAME-WIDE GRADIENT MASS BY TERM ===')
    print('  sum |dL/dalpha| over the front-most Gaussian of every pixel.')
    print('  Photometric reaches every pixel; the depth terms reach %.2f%% of them.'
          % (100 * SUP / N))
    for f in (0, 8):
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        C, D, Tf, a0, c0, d0 = L.render_with_depth(
            col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25 + 1.33333, 2)
        _, gt = Dt.luma_of(frames[f], rw, rh)
        gt = gt.astype(np.float64)
        gain, bias = B.fit_exposure(C, gt)
        rendered = gain * C + bias
        qcw = qc[f]
        diff = rendered - gt
        g_l1 = qcw * 0.8 * invN * np.sign(diff) / 3.0
        X = (rendered * LUMA).sum(2)
        Y = (gt * LUMA).sum(2)
        SS = S.ssim_terms(X, Y, qcw * 0.2 / N)
        g_ss = SS['dLdX'][:, :, None] * LUMA[None, None, :]
        alpha = 1 - Tf
        behind = (C - a0[:, :, None] * c0) / np.maximum((1 - a0)[:, :, None], 1e-6)
        dBehind = (D - a0 * d0) / np.maximum(1 - a0, 1e-6)
        live = a0 > 0
        m_l1 = np.abs(((c0 - behind) * (gain * g_l1)).sum(2))[live].sum()
        m_ss = np.abs(((c0 - behind) * (gain * g_ss)).sum(2))[live].sum()
        # depth: only SUP of N pixels carry it. Sample that fraction at random
        # so the mass is comparable, and use s.weight 0.5, |huberGrad| = 1.
        rng = np.random.RandomState(3)
        flat = np.nonzero(live.ravel())[0]
        pick = rng.choice(flat, size=min(int(SUP), flat.size), replace=False)
        lever = np.abs((d0 - dBehind) / np.maximum(alpha, 1e-4)).ravel()[pick]
        for wname, wval in (('s.weight 0.15', 0.15), ('s.weight 0.5', 0.5)):
            m_dp = (1.0 * wval * invS) * lever.sum()
            tot = m_l1 + m_ss + m_dp
            print('  frame %d, %s: L1 %5.1f%%   SSIM %5.1f%%   depth Huber %5.1f%%'
                  % (f, wname, 100 * m_l1 / tot, 100 * m_ss / tot, 100 * m_dp / tot))

    print('\n=== 4. THE EFFECTIVE-RANK SPIKE, FINE BINS ===')
    logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1
                            ).astype(np.float64), -12, 3)
    s = np.exp(logs)
    lam = s * s
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-20)
    lp = np.log(np.maximum(p, 1e-20))
    r = np.exp(-(p * lp).sum(1))
    for w in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        print('  |exp(H) - 2.000| <= %.3f : %7d  %6.2f%%    |exp(H) - 1.000| <= %.3f : %7d  %6.2f%%'
              % (w, (np.abs(r - 2) <= w).sum(), 100 * (np.abs(r - 2) <= w).mean(),
                 w, (np.abs(r - 1) <= w).sum(), 100 * (np.abs(r - 1) <= w).mean()))
    print('  a uniform population over [1,3] would put %.2f%% inside +/-0.01 of any value'
          % (100 * 0.02 / 2.0))
    print('  observed inside +/-0.01 of 2.000: %.2f%%  -> %.0fx the uniform rate'
          % (100 * (np.abs(r - 2) <= 0.01).mean(),
             (np.abs(r - 2) <= 0.01).mean() / (0.02 / 2.0)))


if __name__ == '__main__':
    main()
