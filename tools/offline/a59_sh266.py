"""A59, re-measured on build 266: how strong is our SH band 1, how strong is
Scaniverse's, and how far could Adam have moved it in a 4,000-iteration run?

Run: cd tools/offline && python -u a59_sh266.py

Reads model/model.ply (266, degree 1, f_rest channel-major 3 per channel) and
the scratchpad sv.npz (Scaniverse, degree 3, 15 per channel). No renders.
"""
import os
import numpy as np
import project as P
import detail as Dt

C1 = 0.4886025119029199
SV = os.path.join(Dt.SCRATCH, 'sv.npz')


def band1(col, per_channel):
    """(N, 3 channels, 3 coefficients) band-1 block, INRIA channel-major."""
    return np.stack([np.stack([np.asarray(col['f_rest_%d' % (c * per_channel + k)],
                                          np.float64) for k in range(3)], 1)
                     for c in range(3)], 1)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -30, 30)))


def report(name, col, per_channel):
    b1 = band1(col, per_channel)
    dc = np.stack([np.asarray(col['f_dc_%d' % c], np.float64) for c in range(3)], 1)
    op = sig(col['opacity'])
    # Coefficient rms over every band-1 coefficient (the 0.01456 / 0.07022 statistic).
    coef_rms = float(np.sqrt((b1 ** 2).mean()))
    # Colour swing: per splat per channel, rms over the uniform sphere of
    # C1*(-y r0 + z r1 - x r2) = C1*|r|/sqrt(3).
    swing = C1 * np.linalg.norm(b1, axis=2) / np.sqrt(3.0)       # (N, 3)
    swing_s = swing.mean(1)
    w = op / op.sum()
    print('%-11s n=%d' % (name, b1.shape[0]))
    print('  DC coefficient rms              %.5f' % float(np.sqrt((dc ** 2).mean())))
    print('  band-1 coefficient rms          %.5f' % coef_rms)
    print('  band-1 |coef| p50 %.5f p90 %.5f p99 %.5f p99.9 %.5f max %.4f'
          % tuple(np.percentile(np.abs(b1), [50, 90, 99, 99.9, 100])))
    print('  band-1 colour swing (0..1 units, rms over sphere) p50 %.5f p90 %.5f '
          'opacity-weighted mean %.5f' % (np.median(swing_s), np.percentile(swing_s, 90),
                                          float((swing_s * w).sum())))
    print('  in 8-bit levels: p50 %.2f p90 %.2f' % (255 * np.median(swing_s),
                                                   255 * np.percentile(swing_s, 90)))
    print('  fraction of band-1 coefficients exactly 0: %.2f%%'
          % (100 * (b1 == 0).mean()))
    return coef_rms, np.abs(b1)


def lr_budget():
    """Upper bound on how far one SH-rest coefficient can travel under Adam
    (|step| <= lr per step drawn), per unit of 'fraction of iterations drawn'.

    Ours: MetalSplatTrainer.adamUniforms: lrSHRest = shDCLR*shDCLRMultiplier*
    lrDecay/shRestLRDivisor, lrDecay = expLerp(1, lateLRFraction=0.1, t).
    Band 1 fully active from about iteration 467 (ledger A14/A32).
    Reference 3DGS: f_rest lr 0.0025/20 flat for 30,000 iterations, band 1
    switched on at iteration 1,000.
    """
    it = np.arange(4000)
    t = it / 4000.0
    decay = 0.1 ** t
    active = it >= 467
    out = {}
    ref = 0.0025 / 20 * (30000 - 1000)
    print('\nLR TRAVEL BUDGET for one band-1 coefficient, per unit drawn fraction')
    print('  reference 3DGS (30k iterations, band 1 from 1,000):  %.3f' % ref)
    for div in (20, 10, 6, 5, 4, 3):
        lr = 0.0025 * 4 * decay / div
        b = float(lr[active].sum())
        out[div] = b
        print('  ours, shRestLRDivisor %2d:                             %.3f  '
              '(%.2fx the reference)' % (div, b, b / ref))
    return out, ref


def main():
    col, n = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    ours_rms, ours_abs = report('OURS (266)', col, 3)
    z = np.load(SV)
    svc = {k: z[k] for k in z.files}
    sv_rms, sv_abs = report('SCANIVERSE', svc, 15)
    print('\n  RATIO band-1 coefficient rms Scaniverse / ours = %.3fx' % (sv_rms / ours_rms))
    budget, ref = lr_budget()
    print('\nIS OURS AT ITS TRAVEL LIMIT? p99 |coef| as a fraction of the budget, '
          'for a splat drawn in f of iterations:')
    p99 = float(np.percentile(ours_abs, 99))
    p999 = float(np.percentile(ours_abs, 99.9))
    for f in (0.05, 0.15, 0.30):
        print('  f = %.2f  budget(div 20) %.4f   p99/budget %.2f   p99.9/budget %.2f'
              % (f, budget[20] * f, p99 / (budget[20] * f), p999 / (budget[20] * f)))


if __name__ == '__main__':
    main()
