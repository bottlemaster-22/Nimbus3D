"""Two rival explanations for 'big splats are seen from more views and fade':
   A. TEXTURE. An untextured wall gives no gradient, so nothing splits it; it
      stays big.  Walls are also what many cameras see.  Proxy for texture:
      local colour variance among the 8 nearest neighbours in the model.
   B. OVERLAP. Front-to-back alpha compositing needs each of n overlapping
      splats to carry roughly 1-(1-A)^(1/n).  Big splats overlap more, so a
      falling per-splat opacity is what compositing REQUIRES, with or without
      any pose disagreement.
"""
import io, json, os
import numpy as np
from scipy.spatial import cKDTree
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls_m = scale.max(axis=1)
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
maxang = np.load(os.path.join(OUT, '_cap_maxang_occ.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))
nv = np.load(os.path.join(OUT, '_cap_nviews_occ.npy')).astype(float)[samp]
spacing_mm = np.load(os.path.join(OUT, '_v_spacing.npy'))
fx, fy, cx, cy = P.load_intrinsics(720, 540)
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
tree = cKDTree(mean)

def pcorr(a, b, ctrls):
    X = np.column_stack([np.ones(len(a))] + list(ctrls))
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]

# --- A. local colour variance -------------------------------------------
dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1).astype(np.float64)
d, idx = tree.query(mean[samp], k=9, workers=-1)
nb = dc[idx]                               # n x 9 x 3
cvar = nb.var(axis=1).sum(axis=1)          # summed per-channel variance of the neighbourhood
lcv = np.log(cvar + 1e-6)

# --- B. overlap count: neighbours inside the splat's own largest axis ----
K = 64
d64, _ = tree.query(mean[samp], k=K, workers=-1)
nover = (d64 <= ls_m[samp][:, None]).sum(axis=1).astype(float)   # includes self

ok = ~np.isnan(maxang) & ~np.isnan(medrng)
m = ok & (medrng >= 0.9) & (medrng < 1.6)
lw = np.log(ls_m[samp][m]*1000.0)
lr = np.log(medrng[m]); ls = np.log(spacing_mm[m])
print('n = %d' % m.sum())
print('\n--- A. texture proxy ---')
print('  local colour variance: p10 %.4f MEDIAN %.4f p90 %.4f'
      % tuple(np.percentile(cvar[m], [10, 50, 90])))
print('  r(log colour var, log WORLD size) = %+.3f' % np.corrcoef(lcv[m], lw)[0, 1])
print('  r(log colour var, view count    ) = %+.3f' % np.corrcoef(lcv[m], nv[m])[0, 1])
print('  r(view count, log WORLD size)                          = %+.3f'
      % np.corrcoef(nv[m], lw)[0, 1])
print('  partial r(view count, log WORLD | colour var, spacing, range) = %+.3f'
      % pcorr(nv[m], lw, [lcv[m], ls, lr]))
print('  median WORLD size mm by colour-variance quartile:', end=' ')
qs = np.percentile(cvar[m], [25, 50, 75])
for lo, hi in [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]:
    s = (cvar[m] > lo) & (cvar[m] <= hi)
    print('%.2f' % np.median(ls_m[samp][m][s]*1000), end='  ')
print()

print('\n--- B. overlap and opacity ---')
op = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))[samp]
srt = np.sort(scale[samp], axis=1)[:, ::-1]
p1 = fx*srt[:, 0]/medrng; p2 = fx*srt[:, 1]/medrng
comp2d = np.sqrt(np.clip((p1*p1*p2*p2)/((p1*p1+.25)*(p2*p2+.25)), 0, 1))
eff = np.clip(op*comp2d, 1e-6, 1-1e-6)
px = ls_m[samp]*fx/medrng
BAND = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]
print('  %-10s %7s %9s %9s %10s %14s' % ('px size', 'n', 'opacity', 'eff a', 'overlap n',
                                         '-n*ln(1-a)'))
for lo, hi in BAND:
    s = ok & (px >= lo) & (px < hi)
    if s.sum() < 50:
        continue
    inv = -nover[s]*np.log(1-eff[s])
    print('  %-10s %7d %9.3f %9.3f %10.1f %14.2f'
          % ('%d-%d px' % (lo, hi), s.sum(), np.median(op[s]), np.median(eff[s]),
             np.median(nover[s]), np.median(inv)))
print('  r(log overlap n, log eff alpha) = %+.3f'
      % np.corrcoef(np.log(nover[m]+1), np.log(eff[m]))[0, 1])
