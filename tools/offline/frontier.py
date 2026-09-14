"""The (splat count, achievable error) frontier for three sizing rules.

Same cost and error models as budget.py, but swept rather than bisected, so
the whole curve is visible instead of six points on it.
"""
import os
import numpy as np
import detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
SIG = Z['sigmas']; FX = Dt.bundle()['intrinsics']['fx']; NSPLAT = 299209
RMAX = 40.0
GRID = np.exp(np.linspace(np.log(0.4), np.log(RMAX), 24))
frame, splat = Z['frame'], Z['splat']
z, sig_px, ladder, ngrad = Z['z'], Z['sig_px'], Z['var_ladder'], Z['ngrad']
n = frame.size
r_mm = sig_px * z / FX * 1000.0
lr = np.log(SIG[None, :] * (z / FX * 1000.0)[:, None]); lg = np.log(GRID)
ar = np.arange(n)
Vrec = np.empty((n, GRID.size), np.float32)
for k in range(GRID.size):
    j = np.clip((lr < lg[k]).sum(1) - 1, 0, SIG.size - 2)
    w = np.clip((lg[k] - lr[ar, j]) / (lr[ar, j + 1] - lr[ar, j]), 0, 1)
    Vrec[:, k] = ladder[ar, j] * (1 - w) + ladder[ar, j + 1] * w
sp = splat.astype(np.int64)
wrec = (sig_px ** 2).astype(np.float64)
W = np.bincount(sp, weights=wrec, minlength=NSPLAT)
V = np.stack([np.bincount(sp, weights=Vrec[:, k] * wrec, minlength=NSPLAT)
              for k in range(GRID.size)], 1)
seen = W > 0
V = V[seen] / W[seen][:, None]; Wt = W[seen]
cur = (np.bincount(sp, weights=r_mm * wrec, minlength=NSPLAT) / np.maximum(W, 1e-12))[seen]
g = (np.bincount(sp, weights=ngrad * wrec, minlength=NSPLAT) / np.maximum(W, 1e-12))[seen]
M = V.shape[0]
cost = (cur[:, None] / GRID[None, :]) ** 2
here = np.abs(np.log(GRID[None, :] / cur[:, None])).argmin(1)
N0 = cost[np.arange(M), here].sum()
SCALE = NSPLAT / N0                 # report in whole-model splats
q = np.searchsorted(np.percentile(g, np.arange(10, 100, 10)), g)


def sc(pick):
    N = cost[np.arange(M), pick].sum() * SCALE
    e = (V[np.arange(M), pick] * Wt).sum() / Wt.sum()
    return N, np.sqrt(e)


N, r = sc(here)
print('as built              %9.0f splats   RMS %.4f   %.2f dB'
      % (N, r, 20 * np.log10(1 / r)))

uni = [sc(np.full(M, k)) for k in range(GRID.size)]
orc, dec = [], []
for lam in np.exp(np.linspace(np.log(1e-7), np.log(1e-1), 40)):
    orc.append(sc((V + lam * cost).argmin(1)))
    pick = np.zeros(M, np.int64)
    for d in range(10):
        s = q == d
        pick[s] = ((V[s] * Wt[s][:, None]).sum(0) + lam * cost[s].sum(0)).argmin()
    dec.append(sc(pick))


def at(curve, target):
    c = sorted(curve)
    for N, r in c:
        if r <= target:
            return N
    return float('nan')


print('\nsplats needed to reach a target RMS luma error (whole model, 299,209 today)')
print('  target RMS    dB     uniform      oracle    decile rule   oracle saving')
for t in (0.14, 0.12, 0.10, 0.08, 0.06):
    a, b, c = at(uni, t), at(orc, t), at(dec, t)
    print('    %.2f      %5.1f  %10.0f  %10.0f  %10.0f    %6.2fx'
          % (t, 20 * np.log10(1 / t), a, b, c, a / b if b == b else float('nan')))

print('\nerror achievable at a given budget')
print('   budget      uniform RMS    oracle RMS    decile RMS')
for B in (150e3, 299209, 600e3, 1.2e6, 3e6):
    def best(curve):
        ok = [r for N, r in curve if N <= B]
        return min(ok) if ok else float('nan')
    print('  %9.0f      %.4f        %.4f        %.4f'
          % (B, best(uni), best(orc), best(dec)))
