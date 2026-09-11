"""At a FIXED splat budget, what does sizing by image detail actually buy?

Cost model, and it is the only modelling step in the whole investigation:
a patch of surface now held by ONE splat of radius r_cur needs (r_cur/r)^2
splats at radius r. Error model: a Gaussian paints one colour over its
footprint, so its squared error is the photograph's variance under that
footprint - which is MEASURED per splat per frame, at seven scales, by
seedsize.py, not assumed.

Weighting is what PSNR weights by: the pixels a patch covers, which is its
current footprint area and does NOT change when the patch is re-tiled with
more, smaller splats.

Three allocations are compared at the same total count:
  uniform     one radius everywhere (what the app does now)
  oracle      each splat its own radius, chosen by water-filling. Not
              shippable; it is the ceiling on what any sizing rule can win.
  decile rule radius is a function of the native-resolution gradient decile
              ONLY - a number the pre-pass already computes. Shippable.
"""
import os
import numpy as np
import project as P, detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
SIG = Z['sigmas']
FX = Dt.bundle()['intrinsics']['fx']
NSPLAT = 299209
RMAX_MM = 40.0                                  # physical cap; see below
GRID = np.exp(np.linspace(np.log(0.4), np.log(RMAX_MM), 24))

frame, splat = Z['frame'], Z['splat']
z, sig_px, ladder, ngrad = Z['z'], Z['sig_px'], Z['var_ladder'], Z['ngrad']
n = frame.size
r_mm = sig_px * z / FX * 1000.0
r_ladder = SIG[None, :] * (z / FX * 1000.0)[:, None]
lr = np.log(r_ladder)
lg = np.log(GRID)

# Per record, the variance at each world radius on the common grid.
Vrec = np.empty((n, GRID.size), np.float32)
ar = np.arange(n)
for k in range(GRID.size):
    j = np.clip((lr < lg[k]).sum(axis=1) - 1, 0, SIG.size - 2)
    a, b = lr[ar, j], lr[ar, j + 1]
    w = np.clip((lg[k] - a) / (b - a), 0, 1)
    Vrec[:, k] = ladder[ar, j] * (1 - w) + ladder[ar, j + 1] * w

# Per splat: pixel-weighted mean over the frames that saw it (what PSNR sums).
wrec = (sig_px ** 2).astype(np.float64)
sp = splat.astype(np.int64)
W = np.bincount(sp, weights=wrec, minlength=NSPLAT)
V = np.empty((NSPLAT, GRID.size))
for k in range(GRID.size):
    V[:, k] = np.bincount(sp, weights=Vrec[:, k] * wrec, minlength=NSPLAT)
seen = W > 0
V = V[seen] / W[seen][:, None]
Wt = W[seen]
cur = (np.bincount(sp, weights=r_mm * wrec, minlength=NSPLAT) / np.maximum(W, 1e-12))[seen]
g = (np.bincount(sp, weights=ngrad * wrec, minlength=NSPLAT) / np.maximum(W, 1e-12))[seen]
M = V.shape[0]
cost = (cur[:, None] / GRID[None, :]) ** 2
here = np.abs(np.log(GRID[None, :] / cur[:, None])).argmin(axis=1)
N0 = float(cost[np.arange(M), here].sum())
print('%d splats seen; budget normalised to the %.0f they hold now' % (M, N0))
print('current radius mm: p10 %.1f  p50 %.1f  p90 %.1f'
      % tuple(np.percentile(cur, q) for q in (10, 50, 90)))


def score(pick):
    N = float(cost[np.arange(M), pick].sum())
    e = float((V[np.arange(M), pick] * Wt).sum() / Wt.sum())
    return N, np.sqrt(e)


def show(name, pick):
    N, r = score(pick)
    print('  %-24s N %8.0f (%4.2fx)  RMS %.4f  %5.2f dB  radius p10/p50/p90 %.1f/%.1f/%.1f mm'
          % (name, N, N / N0, r, 20 * np.log10(1 / r),
             *[np.percentile(GRID[pick], q) for q in (10, 50, 90)]))
    return N, r


def uniform_at(budget):
    best = None
    for k in range(GRID.size):
        N, r = score(np.full(M, k))
        if best is None or abs(N - budget) < abs(best[1] - budget):
            best = (k, N)
    return np.full(M, best[0])


def oracle_at(budget):
    lam = 1e-3
    for _ in range(200):
        pick = (V + lam * cost).argmin(axis=1)
        N, _ = score(pick)
        if abs(N - budget) / budget < 0.02:
            break
        lam = lam * (1.25 if N > budget else 1 / 1.15)
    return pick


def decile_at(budget, nq=10):
    q = np.searchsorted(np.percentile(g, np.arange(100 / nq, 100, 100 / nq)), g)
    lam = 1e-3
    for _ in range(200):
        pick = np.zeros(M, np.int64)
        for d in range(nq):
            s = q == d
            tot = (V[s] * Wt[s][:, None]).sum(0) + lam * cost[s].sum(0)
            pick[s] = tot.argmin()
        N, _ = score(pick)
        if abs(N - budget) / budget < 0.02:
            break
        lam = lam * (1.25 if N > budget else 1 / 1.15)
    return pick, q


print('\nAT THE SAME BUDGET')
show('as built', here)
show('uniform', uniform_at(N0))
op = oracle_at(N0); show('oracle per splat', op)
dp, q = decile_at(N0); show('gradient decile rule', dp)

print('\n  what the decile rule does')
print('   decile   gradient   radius mm   splats now -> after')
for d in range(10):
    s = q == d
    print('     %2d     %.4f      %6.2f      %7.0f -> %8.0f'
          % (d + 1, np.median(g[s]), GRID[dp[s][0]],
             cost[np.nonzero(s)[0], here[s]].sum(),
             cost[np.nonzero(s)[0], dp[s]].sum()))

print('\nTHE FRONTIER: splats needed to reach a target RMS luma error')
print('  target RMS   dB     uniform N      oracle N    decile-rule N   saving')
for tgt in (0.12, 0.10, 0.08, 0.06, 0.05, 0.04):
    def find(fn):
        lo, hi = N0 / 40, N0 * 400
        for _ in range(28):
            mid = np.sqrt(lo * hi)
            _, r = score(fn(mid))
            if r > tgt: lo = mid
            else: hi = mid
        return hi
    nu = find(uniform_at); no = find(oracle_at); nd = find(lambda b: decile_at(b)[0])
    print('    %.2f      %5.1f  %11.0f  %12.0f  %13.0f   %5.2fx'
          % (tgt, 20 * np.log10(1 / tgt), nu, no, nd, nu / max(nd, 1)))
