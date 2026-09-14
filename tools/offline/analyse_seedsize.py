"""What seedsize.py measured, turned into decisions.

Three questions, in order:
  1. Does the gradient the pre-pass already computes predict the footprint
     colour variance, which is the error a one-colour Gaussian cannot avoid?
  2. How small would each splat have to be, in millimetres, to get that
     variance under a target - and does that number vary across the room, or
     is it the same everywhere (in which case adaptive sizing buys nothing)?
  3. At the SAME splat budget, how much error does gradient-driven sizing
     remove compared with one size for everything?
"""
import io, json, os, sys
import numpy as np
import project as P, detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
SIG = Z['sigmas']
b = Dt.bundle()
FX = b['intrinsics']['fx']
NSPLAT = 299209

frame, splat = Z['frame'], Z['splat']
z, sig_px, own_var = Z['z'], Z['sig_px'], Z['own_var']
ladder, ngrad, fgrad = Z['var_ladder'], Z['ngrad'], Z['fgrad']
own_std = np.sqrt(np.maximum(own_var, 0))
n = frame.size
print('records %d   distinct splats %d of %d (%.1f%%)   frames %d'
      % (n, np.unique(splat).size, NSPLAT, 100 * np.unique(splat).size / NSPLAT,
         np.unique(frame).size))

# ---------------------------------------------------------------- 1. sizes
r_mm = sig_px * z / FX * 1000.0          # 1-sigma footprint, world mm
print('\nCURRENT FIELD, as seen from these 16 photographs')
for q in (10, 25, 50, 75, 90):
    print('  p%-2d  range %.2f m   footprint sigma %6.2f px full-res  = %6.2f mm world'
          % (q, np.percentile(z, q), np.percentile(sig_px, q), np.percentile(r_mm, q)))
print('  footprint luma std: p10 %.4f  median %.4f  p90 %.4f   mean %.4f'
      % (*[np.percentile(own_std, q) for q in (10, 50, 90)], own_std.mean()))
rms = np.sqrt((own_var).mean())
print('  RMS over all footprints %.4f  ->  ceiling %.2f dB if each splat paints one colour'
      % (rms, 20 * np.log10(1 / rms)))

# ------------------------------------------------- 2. does gradient predict?
def spearman(a, bb):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(bb)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

print('\nDOES THE GRADIENT ALREADY IN THE PRE-PASS PREDICT FOOTPRINT VARIANCE?')
print('  Spearman(native 256x192 Sobel, footprint luma std) = %.4f'
      % spearman(ngrad, own_std))
print('  Spearman(full 1920x1440 Sobel, footprint luma std) = %.4f'
      % spearman(fgrad, own_std))
print('  Spearman(range z,             footprint luma std) = %.4f  (control)'
      % spearman(z, own_std))
print('  Spearman(footprint sigma px,  footprint luma std) = %.4f  (control)'
      % spearman(sig_px, own_std))

order = np.argsort(ngrad)
edges = np.linspace(0, n, 11).astype(int)
print('\n  native-gradient decile -> what is actually under those splats')
print('  decile  gradient      footprint std   median required radius (mm) at std<=0.05')
tau = 0.05
r_ladder = SIG[None, :] * (z / FX * 1000.0)[:, None]
under = ladder <= tau * tau
big = np.where(under, r_ladder, 0.0).max(axis=1)
for k in range(10):
    s = order[edges[k]:edges[k + 1]]
    print('   %2d     %.4f        %.4f          %6.2f'
          % (k + 1, np.median(ngrad[s]), np.median(own_std[s]), np.median(big[s])))

t = 0.18   # SmartLossSettings.textureEdgeGradient, the existing threshold
m = ngrad > t
print('\n  the EXISTING texture threshold (%.2f) splits the population %.1f%% / %.1f%%'
      % (t, 100 * m.mean(), 100 * (~m).mean()))
print('  footprint std   above %.4f   below %.4f   ratio %.2fx'
      % (own_std[m].mean(), own_std[~m].mean(), own_std[m].mean() / own_std[~m].mean()))

# ------------------------------------------- 3. per splat, across all frames
GRID = np.exp(np.linspace(np.log(0.4), np.log(80.0), 22))     # world mm, 1 sigma
lg = np.log(GRID)
V = np.zeros((NSPLAT, GRID.size), np.float32)
seen = np.zeros(NSPLAT, bool)
lr = np.log(np.maximum(r_ladder, 1e-6))
for k in range(GRID.size):
    j = np.clip(np.searchsorted(lr[0], 0) * 0 + np.sum(lr < lg[k], axis=1) - 1, 0, SIG.size - 2)
    a = lr[np.arange(n), j]; bq = lr[np.arange(n), j + 1]
    w = np.clip((lg[k] - a) / (bq - a), 0, 1)
    v = ladder[np.arange(n), j] * (1 - w) + ladder[np.arange(n), j + 1] * w
    np.maximum.at(V[:, k], splat, v)          # the most demanding view wins
seen[splat] = True
cur = np.zeros(NSPLAT, np.float32); np.maximum.at(cur, splat, r_mm)
g = np.zeros(NSPLAT, np.float32);   np.maximum.at(g, splat, ngrad)
V, cur, g = V[seen], cur[seen], g[seen]
M = V.shape[0]
print('\n%d splats seen in at least one photograph; median current radius %.2f mm'
      % (M, np.median(cur)))

cost = (cur[:, None] / GRID[None, :]) ** 2      # splats needed per current splat
BASE = float(cost[np.arange(M), np.abs(np.log(GRID[None, :] / cur[:, None])).argmin(axis=1)].sum())

def report(name, pick):
    N = cost[np.arange(M), pick].sum()
    err = V[np.arange(M), pick].mean()
    r = np.sqrt(err)
    print('  %-26s N = %8.0f (%5.2fx)   RMS %.4f   ceiling %.2f dB   median r %.2f mm'
          % (name, N, N / M, r, 20 * np.log10(1 / r), np.median(GRID[pick])))
    return N, r

print('\nWHAT SIZING BUYS, at equal splat count. Cost model: a patch now held by'
      '\none splat of radius r_cur needs (r_cur/r)^2 splats at radius r.')
here = np.abs(np.log(GRID[None, :] / cur[:, None])).argmin(axis=1)
report('as built (one size)', here)

# uniform: the same radius everywhere, swept to the same budget
best = None
for k in range(GRID.size):
    pick = np.full(M, k)
    N = cost[np.arange(M), pick].sum()
    if best is None or abs(N - M) < abs(best[0] - M):
        best = (N, k)
uni = np.full(M, best[1])
report('uniform, same budget', uni)

# oracle water-filling: minimise mean var + lambda * count
def waterfill(Vm, lam):
    return (Vm + lam * cost).argmin(axis=1)

lam = 1.0
for _ in range(60):
    pick = waterfill(V, lam)
    N = cost[np.arange(M), pick].sum()
    if N > M: lam *= 1.3
    else: lam /= 1.15
    if abs(N - M) / M < 0.01: break
report('oracle per-splat sizing', pick)

# the DEPLOYABLE rule: radius is a function of the gradient decile only
q = np.searchsorted(np.percentile(g, np.arange(10, 100, 10)), g)
lam2 = lam
for _ in range(80):
    pick2 = np.zeros(M, np.int64)
    for d in range(10):
        s = q == d
        tot = V[s].mean(axis=0) * s.sum() + lam2 * cost[s].sum(axis=0)
        pick2[s] = tot.argmin()
    N = cost[np.arange(M), pick2].sum()
    if N > M: lam2 *= 1.3
    else: lam2 /= 1.15
    if abs(N - M) / M < 0.01: break
report('gradient decile sizing', pick2)
print('\n  radius per gradient decile under that rule:')
for d in range(10):
    s = q == d
    print('    decile %2d  gradient median %.4f  radius %6.2f mm  splats %6d -> %8.0f'
          % (d + 1, np.median(g[s]), GRID[pick2[s][0]], s.sum(),
             cost[np.nonzero(s)[0], pick2[s]].sum()))
