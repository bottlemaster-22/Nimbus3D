"""REFUTATION TEST for "the disc prior has set the shape of a third of the model".

Two questions the original finding did not ask:
  1. Is "uniform over [1,3]" a sane null for exp(H)?  Build a real null instead:
     permute the three log-scale COLUMNS independently across splats.  That keeps
     every axis' marginal distribution and destroys all per-splat shape coupling.
  2. WHAT SHAPE is in the spike?  If it is (r, r, t) with s2/s1 == 1 EXACTLY, the
     spike is the seeder's signature (it lays radius,radius,thickness) carried
     through clone/split, which scale all three axes by one factor and therefore
     preserve exp(H) EXACTLY.  That would make the spike inherited, not learned.
"""
import os
import numpy as np
import project as P

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
logs = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
logs = np.clip(logs, -12, 3)
s = np.exp(logs)

def rank(sc):
    lam = sc * sc
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-300)
    H = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    return np.exp(H)

r = rank(s)
band = np.abs(r - 2.0) <= 0.01
print('=== 1. IS "30x THE UNIFORM RATE" A REAL EXCESS? ===')
print('  measured   |exp(H)-2| <= 0.01 : %7d  %6.2f%%' % (band.sum(), 100*band.mean()))
print('  claimed null (uniform on [1,3]):          1.00%%   -> 30x')

rng = np.random.default_rng(0)
for trial in range(3):
    perm = np.stack([s[rng.permutation(count), j] for j in range(3)], 1)
    rp = rank(perm)
    b = np.abs(rp - 2.0) <= 0.01
    print('  column-permuted null, trial %d          : %7d  %6.2f%%  -> excess %.1fx'
          % (trial, b.sum(), 100*b.mean(), band.mean()/max(b.mean(), 1e-9)))

# lognormal null matched to the pooled marginal of all 900k components
mu, sd = logs.mean(), logs.std()
ln = rng.normal(mu, sd, size=(count, 3))
rl = rank(np.exp(ln))
b = np.abs(rl - 2.0) <= 0.01
print('  iid lognormal null (mu %.3f sd %.3f)   : %7d  %6.2f%%  -> excess %.1fx'
      % (mu, sd, b.sum(), 100*b.mean(), band.mean()/max(b.mean(), 1e-9)))

print()
print('=== 2. WHAT SHAPE IS IN THE SPIKE? ===')
ss = np.sort(s, 1)[:, ::-1]          # s1 >= s2 >= s3
r21, r31 = ss[:, 1]/ss[:, 0], ss[:, 2]/ss[:, 0]
for name, m in (('WHOLE MODEL', np.ones(count, bool)),
                ('spike bin [2.000,2.033)', (r >= 2.0) & (r < 2.0 + 2/60)),
                ('|exp(H)-2|<=0.01', band),
                ('needle mode exp(H)<1.3', r < 1.3)):
    if not m.any():
        continue
    print('  %-26s n=%7d (%5.2f%%)' % (name, m.sum(), 100*m.mean()))
    print('     s2/s1  p1 %.5f p25 %.5f p50 %.5f p75 %.5f p99 %.5f' %
          tuple(np.percentile(r21[m], [1, 25, 50, 75, 99])))
    print('     s3/s1  p1 %.5f p25 %.5f p50 %.5f p75 %.5f p99 %.5f' %
          tuple(np.percentile(r31[m], [1, 25, 50, 75, 99])))
    print('     s2/s1 == 1 to 1e-6: %6.2f%%   to 1e-3: %6.2f%%' %
          (100*(np.abs(r21[m]-1) < 1e-6).mean(), 100*(np.abs(r21[m]-1) < 1e-3).mean()))
    print('     log-scale at the -11.5 floor (any axis): %6.3f%%' %
          (100*(logs[m].min(1) <= -11.49).mean()))

print()
print('=== 3. IS THE SPIKE AT THE PRIOR ZERO (2.000) OR OFFSET FROM IT? ===')
sel = np.abs(r - 2.0) <= 0.05
print('  within +/-0.05: median exp(H) %.6f, mean %.6f' % (np.median(r[sel]), r[sel].mean()))
print('  share of that band strictly BELOW 2.000: %.2f%%' % (100*(r[sel] < 2.0).mean()))
print('  share strictly ABOVE                    : %.2f%%' % (100*(r[sel] > 2.0).mean()))
print('  a prior with a zero AT 2.000 and no other force would be symmetric.')

print()
print('=== 4. THE (r, r, t) FAMILY THE SEEDER LAYS ===')
for k in (0.0, 0.02, 0.04, 0.06, 0.10, 0.20, 0.35, 0.50):
    a = np.array([[1.0, 1.0, k]])
    print('   seeder disc (r, r, %.2f r) -> exp(H) = %.5f' % (k, rank(a)[0]))
print('  the seeder lays (radius, radius, thickness) with thickness in')
print('  [0.10, 0.35] x radius -> exp(H) in [2.057, 2.396], NOT 2.000.')
