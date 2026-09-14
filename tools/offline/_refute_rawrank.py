"""Refutation check for the shape-agent's item 3.

shape250.py computes the disc prior on the PLY's scales, which have filter3D
FUSED in (sigma_ply = sqrt(sigma_raw^2 + f^2)). trainer_regularizer applies
the prior to the RAW logScale (TrainerShaders.metal:2673, no filter3D). This
recovers f per splat (same replay as reloc_comp3d.py), unfuses, and redoes the
prior-vs-photometric comparison on the quantity the kernel actually sees.
Prints numbers, writes no arrays.
"""
import sys, time
import numpy as np
import shape250 as S
import project as PJ

frames = [int(a) for a in sys.argv[1:]] or [0, 8]

# --- recover filter3D (trainer_sampling_rate_update + filter3d_finalize) ---
n = S.COUNT
rate_top = np.zeros((n, 4))
for idx in sorted(S.POSES):
    q, t = S.POSES[idx]
    R = PJ.quat_to_matrix(q)
    cam = S.MEAN @ R.T + t
    z = cam[:, 2]
    ok = z > 0.2
    with np.errstate(divide='ignore', invalid='ignore'):
        u = S.FX * cam[:, 0] / z + S.CX
        v = S.FY * cam[:, 1] / z + S.CY
    ok &= (u >= 0) & (u < S.RW) & (v >= 0) & (v < S.RH)
    r = np.where(ok, max(S.FX, S.FY) / np.maximum(z, 1e-4), 0.0)
    for j in range(4):
        hit = r > rate_top[:, j]
        if not hit.any():
            continue
        rate_top[hit, j + 1:] = rate_top[hit, j:3]
        rate_top[hit, j] = r[hit]
        r = np.where(hit, 0.0, r)
rate = rate_top[:, 3].copy()
for j in [2, 1, 0]:
    rate = np.where(rate <= 0, rate_top[:, j], rate)
f3 = np.where(rate > 0, 0.2 / np.maximum(rate, 1e-30), 0.01)

sig_f = np.exp(S.LOGS)
v = sig_f ** 2 - (f3 ** 2)[:, None]
inv = (v > 0).all(1)
sig_r = np.where(inv[:, None], np.sqrt(np.maximum(v, 1e-30)), sig_f)
LOGR = np.clip(np.log(sig_r), -12, 3)

print('filter3D mm p10 %.3f p50 %.3f p90 %.3f ; invertible %.2f%%'
      % (*np.percentile(f3 * 1000, [10, 50, 90]), 100 * inv.mean()))
needle, disc, blob = S.shape_classes(S.LOGS)
srt = np.sort(sig_f, 1)[:, ::-1]
srr = np.sort(sig_r, 1)[:, ::-1]
print('discs thin axis mm   fused p50 %.3f   raw p50 %.3f   f3 p50 %.3f'
      % (np.median(srt[disc, 2]) * 1000, np.median(srr[disc, 2]) * 1000, np.median(f3[disc]) * 1000))
print('discs s3/s2          fused p50 %.3f   raw p50 %.3f'
      % (np.median(srt[disc, 2] / srt[disc, 1]), np.median(srr[disc, 2] / srr[disc, 1])))
rf = S.eff_rank(S.LOGS)[0]
rr = S.eff_rank(LOGR)[0]
for nm, m in (('discs', disc), ('needles', needle), ('blobs', blob)):
    print('%-8s rank fused p10 %.3f p50 %.3f p90 %.3f | RAW p10 %.3f p50 %.3f p90 %.3f | raw<2 %.1f%%'
          % (nm, *np.percentile(rf[m], [10, 50, 90]), *np.percentile(rr[m], [10, 50, 90]),
             100 * (rr[m] < 2).mean()))

# --- photometric gradient, fused -> raw by the chain rule ---
chain = (sig_r ** 2) / np.maximum(sig_f ** 2, 1e-30)   # dlog fused / dlog raw
gs = []
t0 = time.time()
for f in frames:
    gt, qcw = S.photo(f)
    g = S.geometry(f, S.LOGS, True)
    gC = S.photometric_dconic(g, gt, qcw)
    gl = S.conic_to_logscale(gC, g, False)
    gs.append(np.where((g['ok'] & (np.abs(gC).sum(1) > 0))[:, None], gl * chain, np.nan))
    print('f%2d done %.0f s' % (f, time.time() - t0), flush=True)
Gs = np.stack(gs, 0)
seen = np.isfinite(Gs[:, :, 0]); ns = seen.sum(0)
Gz = np.where(np.isfinite(Gs), Gs, 0.0)
mean = Gz.sum(0) / np.maximum(ns, 1)[:, None]
rms = np.sqrt((Gz ** 2).sum(0) / np.maximum(ns, 1)[:, None])
use = ns >= 2
ii = np.nonzero(use)[0]
th = np.argsort(-sig_r, 1)[:, 2][ii]
base = mean[ii, th] / np.maximum(rms[ii, th], 1e-300)
print('\nsplats with gradient in >=2 frames: %d' % use.sum())
print('photometric-only thin-axis (RAW) E[g]/rms p10 %.3f p50 %.3f p90 %.3f'
      % tuple(np.percentile(base, [10, 50, 90])))
cls = {'discs': disc[ii], 'blobs': blob[ii], 'needles': needle[ii]}
for w, T in ((0.001, 2.0), (0.002, 2.0), (0.005, 2.0), (0.000001, 2.0), (0.001, 3.0), (0.0, 2.0)):
    pr = S.prior_grad(LOGR, w, T)
    m_ = mean[ii, th] + pr[ii, th]
    v_ = rms[ii, th] ** 2 + 2 * mean[ii, th] * pr[ii, th] + pr[ii, th] ** 2
    d = m_ / np.sqrt(np.maximum(v_, 1e-300))
    ratio = np.abs(pr[ii, th]) / np.maximum(rms[ii, th], 1e-300)
    for cn, cm in cls.items():
        print('  RAW w=%.6f T=%.1f %-7s |pr|/rms p50 %10.3f  pr>rms %5.1f%%  thin grows %5.1f%%'
              % (w, T, cn, np.median(ratio[cm]), 100 * (ratio[cm] > 1).mean(), 100 * (d[cm] < 0).mean()))
