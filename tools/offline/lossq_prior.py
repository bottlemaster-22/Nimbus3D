"""Which term decided the SHAPE of the trained Gaussians.

The disc / effective-rank prior in trainer_regularizer is DETERMINISTIC: it
depends only on that Gaussian's own three log-scales, so it pushes the same
direction on every iteration the Gaussian is visible.  The photometric gradient
on the same three parameters changes sign with the view.  Adam divides by the
running RMS, so a small persistent push beats a large alternating one; the term
that decides the direction is the one whose sign does not change.

If the prior decided, the trained population sits AT its fixed points and
nowhere else: exp(H) = discTargetRank on ordinary Gaussians and edgeTargetRank
on flagged ones.  That is a histogram, and the model on disk is the evidence.

Constants, all read from source:
  discPriorWeight            0.005  SmartCore.swift:244
  discTargetEffectiveRank    2.0    SmartCore.swift:245
  edgeTargetEffectiveRank    1.0    SmartCore.swift:246
  gradient factor            -2 w residual rank p_j (log p_j + H)
                                    TrainerShaders.metal:2664-2672
  maxScaleMeters 0.5, maxScaleWeight 0.05  MetalSplatTrainer.swift:2301-2302
  lrScale 0.005 * lrDecay, Adam eps 1e-15  TrainerSupport.swift:461, GPULayouts:526

Run: python -u lossq_prior.py
"""
import io, json, os
import numpy as np

import project as P

W_DISC = 0.005
TARGET_DISC = 2.0
TARGET_EDGE = 1.0
MAXSCALE_M = 0.5
MAXSCALE_W = 0.05


def rank_of(logs):
    s = np.exp(np.clip(logs, -12, 3))
    lam = s * s
    tot = np.maximum(lam.sum(1, keepdims=True), 1e-20)
    p = lam / tot
    lp = np.log(np.maximum(p, 1e-20))
    H = -(p * lp).sum(1)
    return np.exp(H), p, lp, H, s


def prior_grad(logs, target):
    r, p, lp, H, s = rank_of(logs)
    k = -2.0 * W_DISC * (r - target) * r
    return k[:, None] * p * (lp + H[:, None]), r


def main():
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    logs = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
    logs = np.clip(logs, -12, 3)
    r, p, lp, H, s = rank_of(logs)

    print('=== EFFECTIVE RANK exp(H) OF THE TRAINED POPULATION, %d splats ===' % count)
    print('  target on an ordinary Gaussian %.2f, on an edge-flagged one %.2f'
          % (TARGET_DISC, TARGET_EDGE))
    print('  min %.4f  p1 %.4f  p10 %.4f  p25 %.4f  MEDIAN %.4f  p75 %.4f  p90 %.4f  p99 %.4f  max %.4f'
          % tuple(np.percentile(r, [0, 1, 10, 25, 50, 75, 90, 99, 100])))
    print('  mean %.4f  sd %.4f' % (r.mean(), r.std()))
    for band in (0.01, 0.02, 0.05, 0.10, 0.20):
        n2 = np.abs(r - TARGET_DISC) <= band
        n1 = np.abs(r - TARGET_EDGE) <= band
        print('    within +/-%.2f of 2.00: %7d (%5.2f%%)    of 1.00: %7d (%5.2f%%)'
              % (band, n2.sum(), 100 * n2.mean(), n1.sum(), 100 * n1.mean()))

    hist, edges = np.histogram(r, bins=60, range=(1.0, 3.0))
    print('\n  histogram of exp(H), 60 bins over [1,3]; * = 1%% of the population')
    for i in range(60):
        frac = hist[i] / count
        if frac < 0.002 and i % 3:
            continue
        print('   %5.3f %7d %6.2f%% %s' % (edges[i], hist[i], 100 * frac,
                                           '*' * int(round(100 * frac))))

    # --- how hard the prior is pushing, and where its zero is ---------------
    g2, _ = prior_grad(logs, TARGET_DISC)
    g1, _ = prior_grad(logs, TARGET_EDGE)
    print('\n=== THE PRIOR GRADIENT ON log-scale, disc target ===')
    print('  |g| per component: p50 %.4e  p90 %.4e  p99 %.4e  max %.4e'
          % tuple(np.percentile(np.abs(g2), [50, 90, 99, 100])))
    print('  it is the SAME every iteration for a given Gaussian: it has no view argument.')
    print('  Adam step is lr * mhat/(sqrt(vhat)+1e-15). A gradient whose sign never')
    print('  changes drives mhat/sqrt(vhat) to +/-1, so its step saturates at lr.')
    lr0, lr4000 = 0.005 * 1.0, 0.005 * 0.7356
    print('  lrScale at iteration 0 = %.5f, at 4000 = %.5f' % (lr0, lr4000))
    print('  4000 saturated steps would move log-scale by up to %.1f in log space'
          % (4000 * lr4000))
    need = np.abs(np.log(np.maximum(r, 1e-9) / TARGET_DISC))
    print('  |log(rank/2)| still outstanding at the end: p50 %.4f p90 %.4f'
          % (np.percentile(need, 50), np.percentile(need, 90)))

    # --- the runaway-scale hinge -------------------------------------------
    over = np.maximum(s - MAXSCALE_M, 0.0)
    print('\n=== RUNAWAY-SCALE HINGE, %.2f m ===' % MAXSCALE_M)
    print('  largest axis: p50 %.5f m  p90 %.5f  p99 %.5f  max %.5f'
          % tuple(np.percentile(s.max(1), [50, 90, 99, 100])))
    print('  components over the hinge: %d of %d (%.4f%%) -> the term is inert'
          % ((over > 0).sum(), over.size, 100 * (over > 0).mean()))

    # --- opacity ------------------------------------------------------------
    o = col['opacity'].astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-o))
    print('\n=== OPACITY (binarization is OFF, binarizeLastFraction 0) ===')
    print('  sigmoid(logit): p1 %.4f p10 %.4f p50 %.4f p90 %.4f p99 %.4f'
          % tuple(np.percentile(sig, [1, 10, 50, 90, 99])))
    print('  fraction below 1/255 %.4f%%,  above 0.9 %.2f%%'
          % (100 * (sig < 1 / 255).mean(), 100 * (sig > 0.9).mean()))
    print('  sigma(1-sigma), which binarization would penalise: mean %.4f' % (sig * (1 - sig)).mean())

    # --- axis ratios, for the record ---------------------------------------
    ss = np.sort(s, axis=1)[:, ::-1]
    print('\n=== AXIS RATIOS ===')
    print('  s2/s1 median %.4f   s3/s2 median %.4f'
          % (np.median(ss[:, 1] / ss[:, 0]), np.median(ss[:, 2] / ss[:, 1])))
    print('  s3/s1 median %.4f' % np.median(ss[:, 2] / ss[:, 0]))


if __name__ == '__main__':
    main()
