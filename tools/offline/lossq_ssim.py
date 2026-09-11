"""The photometric objective, term by term, in the exact arithmetic of the
kernels, on a real render of a real frame against the real photograph.

Reproduced line for line from TrainerShaders.metal:
  trainer_loss_photometric   L1 branch and gradFinal
  trainer_ssim_prepare / trainer_blur_h / trainer_blur_v
  trainer_ssim_stats         (the three-plane fold)
  trainer_ssim_backward      dLdX -> gradFinal through the luma weights

Run: python -u lossq_ssim.py [frame ...]
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B

# TrainerLossUniforms defaults / MetalSplatTrainer assignment
LAMBDA_SSIM = 0.2
SSIM_C1 = 0.0001
SSIM_C2 = 0.0009
LUMA = np.array([0.2126, 0.7152, 0.0722])
K11 = np.array([0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554,
                0.26601172,
                0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838])


def blur(plane):
    """trainer_blur_h then trainer_blur_v, clamp to edge, one plane."""
    h, w = plane.shape
    out = np.zeros_like(plane)
    for t in range(-5, 6):
        idx = np.clip(np.arange(w) + t, 0, w - 1)
        out += K11[t + 5] * plane[:, idx]
    fin = np.zeros_like(plane)
    for t in range(-5, 6):
        idy = np.clip(np.arange(h) + t, 0, h - 1)
        fin += K11[t + 5] * out[idy, :]
    return fin


def ssim_terms(X, Y, w_ssim):
    """Everything trainer_ssim_stats and trainer_ssim_backward compute."""
    mux, muy = blur(X), blur(Y)
    gxx, gyy, gxy = blur(X * X), blur(Y * Y), blur(X * Y)
    sxx = np.maximum(gxx - mux * mux, 0.0)
    syy = np.maximum(gyy - muy * muy, 0.0)
    sxy = gxy - mux * muy
    n1 = 2.0 * mux * muy + SSIM_C1
    n2 = 2.0 * sxy + SSIM_C2
    d1 = mux * mux + muy * muy + SSIM_C1
    d2 = sxx + syy + SSIM_C2
    invD = 1.0 / np.maximum(d1 * d2, 1e-12)
    ssim = n1 * n2 * invD
    dS_dmux = (2.0 * muy * n2 * d1 - 2.0 * mux * n1 * n2) / np.maximum(d1 * d1 * d2, 1e-12)
    dS_dsxy = 2.0 * n1 * invD
    dS_dsxx = -n1 * n2 / np.maximum(d1 * d2 * d2, 1e-12)
    Cc = -w_ssim * dS_dmux
    A = -w_ssim * dS_dsxx
    Bp = -w_ssim * dS_dsxy
    p0 = blur(Cc - 2.0 * A * mux - Bp * muy)
    pA = blur(A)
    pB = blur(Bp)
    dLdX = p0 + 2.0 * X * pA + Y * pB
    return dict(ssim=ssim, dLdX=dLdX, sxx=sxx, syy=syy, d2=d2, d1=d1,
                mux=mux, muy=muy, dS_dsxy=dS_dsxy, clampD=(d1 * d2 <= 1e-12),
                clampD1=(d1 * d1 * d2 <= 1e-12), clampD2=(d1 * d2 * d2 <= 1e-12))


def pct(a, qs=(1, 10, 50, 90, 99, 100)):
    return ' '.join('%s %.4g' % (('p%d' % q) if q < 100 else 'max',
                                 np.percentile(a, q)) for q in qs)


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    N = rw * rh
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    bundle = Dt.bundle()
    qc = {int(f['index']): float(f['qc']['weight']) for f in bundle['frames']}
    want = [int(a) for a in sys.argv[1:]] or [0, 4, 8, 12]

    print('11-tap window check: sum %.9f, radius 5, sigma implied from k[5]/k[4]'
          % K11.sum())
    ref = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
    ref /= ref.sum()
    print('  max |shipped - Gaussian(sigma=1.5, 11 taps, normalised)| = %.3e'
          % np.abs(K11 - ref).max())
    print('  separable 11x11 window on a %dx%d render = %.4f%% of the frame per window'
          % (rw, rh, 100 * 121.0 / N))

    for frame in want:
        rot, t = poses[frame]
        R = P.quat_to_matrix(rot)
        # The render the loss was computed on at iteration 4000.
        img, Tf, ok, g, inst = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh,
                                        0.25 + 1.33333, 2)
        _, gt = Dt.luma_of(frames[frame], rw, rh)
        gt = gt.astype(np.float64)
        gain, bias = B.fit_exposure(img, gt)
        rendered = gain * img + bias
        qcw = qc[frame]

        # ---- trainer_loss_photometric ------------------------------------
        invN = 1.0 / N
        w_l1 = qcw * (1.0 - LAMBDA_SSIM) * invN
        diff = rendered - gt
        l1 = np.abs(diff).mean(axis=2)
        L_l1 = float((qcw * (1.0 - LAMBDA_SSIM) * l1 * invN).sum())
        g_l1 = w_l1 * np.sign(diff) / 3.0                       # per channel

        # ---- SSIM --------------------------------------------------------
        X = (rendered * LUMA).sum(axis=2)
        Y = (gt * LUMA).sum(axis=2)
        w_ssim = qcw * LAMBDA_SSIM / N
        S = ssim_terms(X, Y, w_ssim)
        L_ssim = float((w_ssim * (1.0 - S['ssim'])).sum())
        g_ss = S['dLdX'][:, :, None] * LUMA[None, None, :]      # per channel

        tot = g_l1 + g_ss
        agree = np.sign(g_l1) == np.sign(g_ss)
        flip = np.sign(tot) != np.sign(g_l1)
        m_l1 = np.abs(g_l1).sum()
        m_ss = np.abs(g_ss).sum()

        print('\n================ frame %d  (qc.weight %.4f, exposure fit gain %.4f bias %+.4f) '
              % (frame, qcw, gain, bias))
        print('  render PSNR %.3f dB, mean SSIM %.4f' %
              (10 * np.log10(1.0 / max(((rendered - gt) ** 2).mean(), 1e-12)), S['ssim'].mean()))
        print('  LOSS VALUE   L1 term %.6e   SSIM term %.6e   ssim share %.1f%%'
              % (L_l1, L_ssim, 100 * L_ssim / (L_l1 + L_ssim)))
        print('  GRADIENT MASS into gradFinal, sum|g| over %d pixels x 3 channels:' % N)
        print('    L1   %.6e   (%.1f%%)' % (m_l1, 100 * m_l1 / (m_l1 + m_ss)))
        print('    SSIM %.6e   (%.1f%%)' % (m_ss, 100 * m_ss / (m_l1 + m_ss)))
        print('    ratio SSIM/L1 by mass  %.2fx   (lambda ratio 0.2/0.8 = 0.25x)'
              % (m_ss / m_l1))
        print('  |g| per (pixel,channel):')
        print('    L1   %s' % pct(np.abs(g_l1)))
        print('    SSIM %s' % pct(np.abs(g_ss)))
        r = np.abs(g_ss) / np.maximum(np.abs(g_l1), 1e-300)
        print('    ratio %s' % pct(r))
        print('    SSIM larger than L1 on %.2f%% of (pixel,channel)' % (100 * (r > 1).mean()))
        print('  DIRECTION:')
        print('    signs agree                       %.2f%%' % (100 * agree.mean()))
        print('    SSIM OVERRIDES the L1 direction   %.2f%%  (sign(L1+SSIM) != sign(L1))'
              % (100 * flip.mean()))
        print('  UNBOUNDEDNESS: dSSIM/dsigma_xy = 2 n1 / (d1 d2), d2 = sxx+syy+c2, c2 = %.4g' % SSIM_C2)
        print('    d2 (local variance + c2)  %s' % pct(S['d2']))
        print('    dS/dsigma_xy              %s' % pct(np.abs(S['dS_dsxy'])))
        print('    the 1e-12 clamps fire on d1*d2 %.4f%%, d1^2 d2 %.4f%%, d1 d2^2 %.4f%%'
              % (100 * S['clampD'].mean(), 100 * S['clampD1'].mean(), 100 * S['clampD2'].mean()))
        flat = S['d2'] < np.percentile(S['d2'], 10)
        text = S['d2'] > np.percentile(S['d2'], 90)
        for nm, msk in (('flattest 10% of the frame', flat), ('most textured 10%', text)):
            rr = np.abs(g_ss)[msk] / np.maximum(np.abs(g_l1)[msk], 1e-300)
            print('    %-26s median |gSSIM|/|gL1| %8.2f   median d2 %.5f'
                  % (nm, np.median(rr), np.median(S['d2'][msk])))


if __name__ == '__main__':
    main()
