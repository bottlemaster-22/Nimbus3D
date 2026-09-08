"""The deployable rule, done properly: radius is a function of the native
gradient bucket only, swept over a wide enough lambda to see the whole curve,
and also in a MONOTONE variant (radius may not increase with gradient), which
is what anyone would actually ship."""
import os
import numpy as np
import detail as Dt

exec(open('frontier.py').read().split("print('as built")[0])   # reuse the setup

def sc2(pick):
    N = cost[np.arange(M), pick].sum() * SCALE
    e = (V[np.arange(M), pick] * Wt).sum() / Wt.sum()
    return N, np.sqrt(e)

NQ = 16
qq = np.searchsorted(np.percentile(g, np.arange(100 / NQ, 100, 100 / NQ)), g)
Vg = np.stack([(V[qq == d] * Wt[qq == d][:, None]).sum(0) for d in range(NQ)])
Cg = np.stack([cost[qq == d].sum(0) for d in range(NQ)])

lams = np.exp(np.linspace(np.log(1e-9), np.log(1e4), 120))
free, mono = [], []
for lam in lams:
    tot = Vg + lam * Cg
    p = tot.argmin(1)
    pick = p[qq]
    free.append((sc2(pick), p.copy()))
    # monotone: radius non-increasing as gradient rises. DP over buckets.
    K = V.shape[1]
    best = np.full((NQ, K), np.inf); back = np.zeros((NQ, K), int)
    best[0] = tot[0]
    for d in range(1, NQ):
        run = np.inf; arg = 0
        row = np.empty(K); rowa = np.empty(K, int)
        for k in range(K - 1, -1, -1):          # k increases with radius
            if best[d - 1, k] < run:
                run = best[d - 1, k]; arg = k
            row[k] = tot[d, k] + run; rowa[k] = arg
        best[d] = row; back[d] = rowa
    k = int(best[NQ - 1].argmin()); pm = np.zeros(NQ, int); pm[NQ - 1] = k
    for d in range(NQ - 1, 0, -1):
        k = back[d, k]; pm[d - 1] = k
    mono.append((sc2(pm[qq]), pm.copy()))

def at(curve, t):
    pts = sorted([c[0] for c in curve])
    for N, r in pts:
        if r <= t:
            return N
    return float('nan')

print('splats needed to reach a target RMS (whole model; today 299,209 at RMS %.4f)'
      % sc2(here)[1])
print('  target   free bucket rule   monotone bucket rule')
for t in (0.14, 0.12, 0.10, 0.08):
    print('   %.2f      %12.0f       %12.0f' % (t, at(free, t), at(mono, t)))

print('\nbest RMS at a given budget')
print('   budget      free rule    monotone rule')
for B in (150e3, 299209, 600e3, 1.2e6, 3e6):
    def best(c):
        ok = [x[0][1] for x in c if x[0][0] <= B]
        return min(ok) if ok else float('nan')
    print('  %9.0f      %.4f        %.4f' % (B, best(free), best(mono)))

# the shippable table at the current budget
cand = [x for x in mono if x[0][0] <= 299209 * 1.02]
pickd = min(cand, key=lambda x: x[0][1])
print('\nMONOTONE RULE AT TODAY\'S BUDGET  (N %.0f, RMS %.4f, %.2f dB vs %.2f dB now)'
      % (pickd[0][0], pickd[0][1], 20 * np.log10(1 / pickd[0][1]),
         20 * np.log10(1 / sc2(here)[1])))
print('  bucket  gradient median   radius mm   share of budget')
tot = cost[np.arange(M), pickd[1][qq]].sum()
for d in range(NQ):
    s = qq == d
    print('    %2d      %.4f          %6.2f        %5.1f%%'
          % (d + 1, np.median(g[s]), GRID[pickd[1][d]],
             100 * cost[np.nonzero(s)[0], pickd[1][d]].sum() / tot))
