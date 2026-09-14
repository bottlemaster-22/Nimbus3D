"""Turn the measured per-unit responses into the actual loss balance, with
every constant named and sourced.
"""
import io, json, os, sys
import numpy as np
import project as P

frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4
b = np.load(os.path.join(os.path.dirname(__file__), '_bwd_%d.npz' % frame))
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
N = sl['renderWidth'] * sl['renderHeight']
SUP = sl['depthSamplesSupervisedTotal'] / sl['depthSupervisionFramesMeasured']
INV_N, INV_S = 1.0 / N, 1.0 / SUP
qcw = float(b['qcw'])

# ---- the exact depth coefficient ------------------------------------------
DEPTH_SCALE = 1.0        # TwoScaleTrustField.depthLossScale, i <= 0.35*total
EDGE_BOOST = 2.0         # TrainerTuning.geometricEdgeBoost
print('=== CONSTANTS (all read from source) ===')
print('  lambdaSSIM 0.2   frameWeight = qc.weight = %.4f (this frame)' % qcw)
print('  invN        = 1/%d       = %.6e' % (N, INV_N))
print('  invSamples  = 1/%.0f     = %.6e   (MEASURED supervised count)' % (SUP, INV_S))
print('  invSamples / invN = %.3f' % (INV_S / INV_N))

mass = b['mass']
print('\n=== (1) DEPTH vs PHOTOMETRIC dL/dalpha ===')
print('  measured on frame %d, %d live (pixel,Gaussian) pairs' % (frame, mass[3]))
print('  photometric |dL/dalpha| summed over the frame          %.4e' % mass[0])
print('  ... the part landing on depth-sampled pixels           %.4e' % mass[2])
print('  depth |dL/dalpha| summed, per unit dL/dExpected        %.4e' % mass[1])
print('\n  dL/dExpected = depthScale * s.weight * invSamples * huberGrad')
print('    s.weight = trustWeight * authority * qcWeight,  huberGrad in [-1,1]')
rows = [
    ('full trust, saturated Huber', 1.0, 1.0, 1.0, 1.0),
    ('trust 0.91 (noise model @1.27m), auth 1, |hg|=1', 0.91, 1.0, 1.0, 1.0),
    ('geometric edge (boost 2), |hg|=1', 0.91, 1.0, 1.0, EDGE_BOOST),
    ('trust 0.5, |hg|=0.5 (inside the Huber knee)', 0.5, 1.0, 0.5, 1.0),
    ('late run: depthScale floor 0.05', 0.91, 1.0, 1.0, 1.0),
]
print('\n  %-48s %10s %10s %10s' % ('case', 'cD', 'depth/photo', 'at sup px'))
for name, tw, auth, hg, boost in rows:
    ds = 0.05 if 'floor' in name else DEPTH_SCALE
    cD = ds * min(tw * auth * qcw * boost, 4.0) * INV_S * hg
    dmass = mass[1] * cD
    print('  %-48s %10.4e %10.3f %10.3f'
          % (name, cD, dmass / mass[0], dmass / mass[2]))

cD1 = DEPTH_SCALE * (0.91 * 1.0 * qcw) * INV_S * 1.0
print('\n  HEADLINE: at a supervised pixel, depth dL/dalpha is %.2fx the photometric one.'
      % (mass[1] * cD1 / mass[2]))
print('  Frame-wide, depth carries %.1f%% of all dL/dalpha mass (it reaches %.2f%% of pixels).'
      % (100 * mass[1] * cD1 / (mass[1] * cD1 + mass[0]), 100 * mass[2] / mass[0]))

# ---- the transition-width term, which nobody has priced --------------------
print('\n  The F4 transition-width term is NOT bounded like the Huber:')
print('    dL/dExpected += w * transitionWidthWeight * 4 * (1-2t) / span,  span in METRES')
for span in (0.30, 0.10, 0.04, 0.01):
    print('    span %5.2f m -> %6.1f x the Huber term (which is w * 1)' % (span, 0.35 * 4 / span))
print('    fires only on GEOMETRIC edges with 0 < t < 1, i.e. rendered depth strictly')
print('    between the two local modes; those samples also carry the 2.0 edge boost.')

# ---- (2) prior vs photometric dL/dlogScale ---------------------------------
gC_ph, gC_dp = b['gC_ph'], b['gC_dp']
cA, cB, cC = b['cA'], b['cB'], b['cC']
camv, idx = b['camv'], b['idx']
scale, Rm, Rw = b['scale'], b['Rm'], b['Rw']
fx, fy = float(b['fx']), float(b['fy'])


def conic_to_logscale(dLdConic):
    """trainer_preprocess_backward, dL/dconic -> dL/dlogScale, verbatim."""
    n = dLdConic.shape[0]
    conicDet = cA * cC - cB * cB
    good = np.abs(conicDet) >= 1e-20
    inv = np.where(good, 1.0 / np.where(good, conicDet, 1), 0.0)
    a = cC * inv
    bb = -cB * inv
    c = cA * inv
    D = a * c - bb * bb
    invD2 = 1.0 / np.maximum(D * D, 1e-20)
    dLda = invD2 * (-c * c * dLdConic[:, 0] + bb * c * dLdConic[:, 1] - bb * bb * dLdConic[:, 2])
    dLdb = invD2 * (2 * bb * c * dLdConic[:, 0] - (D + 2 * bb * bb) * dLdConic[:, 1]
                    + 2 * a * bb * dLdConic[:, 2])
    dLdc = invD2 * (-bb * bb * dLdConic[:, 0] + a * bb * dLdConic[:, 1] - a * a * dLdConic[:, 2])
    Gm = np.zeros((n, 2, 2))
    Gm[:, 0, 0] = dLda
    Gm[:, 0, 1] = 0.5 * dLdb
    Gm[:, 1, 0] = 0.5 * dLdb
    Gm[:, 1, 1] = dLdc
    iz = 1.0 / camv[:, 2]
    A = np.zeros((n, 2, 3))
    A[:, 0, 0] = fx * iz
    A[:, 0, 2] = -fx * camv[:, 0] * iz * iz
    A[:, 1, 1] = fy * iz
    A[:, 1, 2] = -fy * camv[:, 1] * iz * iz
    dLdSigmaCam = np.transpose(A, (0, 2, 1)) @ Gm @ A
    dLdSigmaWorld = Rw.T @ dLdSigmaCam @ Rw
    M = Rm * scale[:, None, :]
    dLdM = 2.0 * (dLdSigmaWorld @ M)
    RtG = np.transpose(Rm, (0, 2, 1)) @ dLdM
    return np.stack([RtG[:, 0, 0] * scale[:, 0],
                     RtG[:, 1, 1] * scale[:, 1],
                     RtG[:, 2, 2] * scale[:, 2]], 1)


g_ph = conic_to_logscale(gC_ph)
g_dp_unit = conic_to_logscale(gC_dp)
prior = np.load(os.path.join(os.path.dirname(__file__), '_prior_grad.npy'))
rank = np.load(os.path.join(os.path.dirname(__file__), '_rank.npy'))

drawn = np.zeros(g_ph.shape[0], bool)
drawn[idx] = True
touched = drawn & (np.abs(g_ph).sum(1) > 0)
print('\n=== (2) PRIOR vs PHOTOMETRIC dL/dlogScale ===')
print('  %d splats drawn this frame; %d received a non-zero photometric scale gradient'
      % (drawn.sum(), touched.sum()))
mag_ph = np.abs(g_ph[touched])
mag_pr = np.abs(prior[touched])
for nm, v in (('photometric', mag_ph), ('disc prior w=0.001', mag_pr),
              ('disc prior w=0.01 (old)', mag_pr * 10)):
    print('  |dL/dlogScale| %-24s p10 %.4e p50 %.4e p90 %.4e'
          % (nm, *np.percentile(v, [10, 50, 90])))
r = mag_pr / np.maximum(mag_ph, 1e-300)
print('  per-component ratio prior/photometric   p10 %.3f p50 %.3f p90 %.3f  mean %.3f'
      % (*np.percentile(r, [10, 50, 90]), r.mean()))
print('  fraction of components where the prior is the larger term: %.4f' % (r > 1).mean())

tot = g_ph + prior
flip = (np.sign(tot) != np.sign(g_ph)) & (g_ph != 0)
print('\n  ADAM IS SCALE-INVARIANT (epsilon 1e-15), so what matters is the SIGN,')
print('  not the magnitude: the step is lr * mhat/sqrt(vhat), never lr * gradient.')
print('  fraction of scale components whose SIGN the prior flips: %.4f' % flip[touched].mean())
print('  ... on splats with |rank-2| > 0.1 (the ones it still has an opinion about): %.4f'
      % flip[touched & (np.abs(rank - 2) > 0.1)].mean())
print('  ... on splats with |rank-2| < 0.02: %.4f'
      % flip[touched & (np.abs(rank - 2) < 0.02)].mean())

# depth channel's own scale gradient
cDn = cD1
mag_dp = np.abs(g_dp_unit[touched]) * cDn
print('\n  For reference, the DEPTH channel dL/dlogScale at cD = %.3e:' % cDn)
print('  |dL/dlogScale| depth                     p10 %.4e p50 %.4e p90 %.4e'
      % tuple(np.percentile(mag_dp, [10, 50, 90])))
print('  depth / photometric on scale (medians)   %.2f'
      % (np.median(mag_dp) / np.median(mag_ph)))

# ---- (3) AbsGS: which channel decides WHERE to densify ---------------------
ap, ad = b['abs_ph'], b['abs_dp']
m = drawn & (ap > 0)
print('\n=== BONUS: the AbsGS densification statistic, per channel ===')
print('  sum|dL/dmean2D| photometric  p50 %.4e' % np.median(ap[m]))
print('  sum|dL/dmean2D| depth * cD   p50 %.4e' % np.median(ad[m] * cDn))
print('  depth / photometric          p50 %.2f   mean %.2f'
      % (np.median(ad[m] * cDn / ap[m]), (ad[m] * cDn / ap[m]).mean()))
print('  Spearman-ish: fraction of the top-10%% by TOTAL score that are also')
sc_tot = ap[m] + ad[m] * cDn
sc_ph = ap[m]
k = int(0.1 * m.sum())
top_t = set(np.argsort(-sc_tot)[:k].tolist())
top_p = set(np.argsort(-sc_ph)[:k].tolist())
print('  in the top-10%% by the PHOTOMETRIC score alone: %.4f' % (len(top_t & top_p) / k))
