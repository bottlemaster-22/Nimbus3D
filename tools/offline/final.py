"""The numbers the conclusion rests on, in one place."""
import os
import numpy as np
import detail as Dt
exec(open('frontier.py').read().split("print('as built")[0])

def sc(pick):
    N = cost[np.arange(M), pick].sum() * SCALE
    e = (V[np.arange(M), pick] * Wt).sum() / Wt.sum()
    return N, np.sqrt(e)

def db(r): return 20 * np.log10(1 / r)

N0s, r0 = sc(here)
print('as built:            %8.0f splats   RMS %.4f  %.2f dB' % (N0s, r0, db(r0)))

# oracle at several budgets, to price "budget freed from flat surfaces"
lams = np.exp(np.linspace(np.log(1e-9), np.log(1e4), 160))
curve = []
for lam in lams:
    curve.append(sc((V + lam * cost).argmin(1)))
curve.sort()
def best_at(B):
    ok = [r for N, r in curve if N <= B]
    return min(ok) if ok else float('nan')
print('\nORACLE sizing, what extra budget is worth')
for mult in (1.0, 1.1, 1.25, 1.5, 2.0, 4.0, 10.0):
    r = best_at(N0s * mult)
    print('   budget %5.2fx (%8.0f)   RMS %.4f   %.2f dB   (+%.2f dB over as built)'
          % (mult, N0s * mult, r, db(r), db(r) - db(r0)))

# equal-error allocation: every splat as large as it can be while holding
# footprint std under a target, clamped by the 720x540 render floor and 40 mm
FLOOR = 3.3
for tau in (0.04, 0.05, 0.07, 0.10):
    ok = V <= tau * tau
    pick = np.where(ok.any(1), np.where(ok, np.arange(V.shape[1])[None, :], -1).max(1), 0)
    r = GRID[pick]
    r = np.clip(r, FLOOR, RMAX)
    p2 = np.abs(np.log(GRID[None, :] / r[:, None])).argmin(1)
    N, e = sc(p2)
    print('equal-error tau %.2f: %10.0f splats   RMS %.4f   %.2f dB   radius p10/p50/p90 %.1f/%.1f/%.1f mm'
          % (tau, N, e, db(e), *[np.percentile(GRID[p2], q) for q in (10, 50, 90)]))

# how much of the budget do the flat regions hold, and what does releasing it buy
flat = g < np.percentile(g, 20)
share = cost[np.nonzero(flat)[0], here[flat]].sum() / cost[np.arange(M), here].sum()
print('\nthe flattest 20%% of splats hold %.1f%% of the budget; releasing ALL of it'
      '\ngives the rest %.2fx more splats, worth %.2f dB by the curve above.'
      % (100 * share, 1 / (1 - share), db(best_at(N0s / (1 - share))) - db(r0)))
