"""Is the exp(H)=2 spike an ATTRACTOR, or is exp(H) simply a flat, compressive
function over the shapes this model actually has?

Test A: how WIDE is the shape band that maps into |exp(H)-2| <= 0.01?
Test B: a shape-marginal-preserving null - permute s2/s1 and s3/s1 independently
        across splats, so no splat keeps its own joint shape, then re-measure.
Test C: how much of the model is within +/-0.01 of OTHER exp(H) values, not 2?
"""
import os
import numpy as np
import project as P

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64), -12, 3)
s = np.sort(np.exp(logs), 1)[:, ::-1]
r21, r31 = s[:, 1]/s[:, 0], s[:, 2]/s[:, 0]

def rank_from_ratios(a, b):
    lam = np.stack([np.ones_like(a), a*a, b*b], 1)
    p = lam / lam.sum(1, keepdims=True)
    return np.exp(-(p*np.log(np.maximum(p, 1e-300))).sum(1))

r = rank_from_ratios(r21, r31)

print('=== A. HOW WIDE IS THE SHAPE BAND THAT LANDS IN |exp(H)-2| <= 0.01? ===')
print('    (holding s3/s1 at the model median 0.1186 and sweeping s2/s1)')
grid = np.linspace(0.3, 1.0, 70001)
rg = rank_from_ratios(grid, np.full_like(grid, 0.1186))
ok = np.abs(rg - 2.0) <= 0.01
print('    s2/s1 in [%.5f, %.5f]  -> width %.5f  (%.1f%% of the [0.3,1.0] sweep)'
      % (grid[ok].min(), grid[ok].max(), grid[ok].max()-grid[ok].min(), 100*ok.mean()))
print('    model s2/s1 p25..p75 is [%.5f, %.5f] (width %.5f)'
      % (np.percentile(r21, 25), np.percentile(r21, 75), np.percentile(r21, 75)-np.percentile(r21, 25)))
print('    -> the exp(H)=2 band is %.2fx as wide as the model\'s own interquartile shape range.'
      % ((grid[ok].max()-grid[ok].min()) / (np.percentile(r21,75)-np.percentile(r21,25))))

print()
print('=== B. SHAPE-MARGINAL-PRESERVING NULL ===')
rng = np.random.default_rng(1)
base = np.abs(r - 2.0) <= 0.01
print('    measured                              %6.2f%%' % (100*base.mean()))
for t in range(3):
    rr = rank_from_ratios(r21[rng.permutation(count)], r31[rng.permutation(count)])
    b = np.abs(rr - 2.0) <= 0.01
    print('    s2/s1 and s3/s1 shuffled apart, trial %d  %6.2f%%   excess %.2fx'
          % (t, 100*b.mean(), base.mean()/max(b.mean(),1e-9)))

print()
print('=== C. +/-0.01 DENSITY AT OTHER exp(H) VALUES (same window width) ===')
for c in (1.10, 1.50, 1.90, 1.99, 2.00, 2.01, 2.05, 2.15, 2.40):
    m = np.abs(r - c) <= 0.01
    print('    within +/-0.01 of %.2f : %7d  %6.2f%%' % (c, m.sum(), 100*m.mean()))

print()
print('=== D. THE PRIOR PULLS DOWN FROM ABOVE 2, UP FROM BELOW. WHERE IS THE MASS? ===')
print('    exp(H) > 2.000 : %6.2f%%   (prior pushes these DOWN toward 2)' % (100*(r > 2.0).mean()))
print('    exp(H) < 2.000 : %6.2f%%   (prior pushes these UP toward 2)' % (100*(r < 2.0).mean()))
print('    exp(H) in [1.30, 1.99] (between the two modes, where the prior is')
print('    strongest and nothing should linger): %6.2f%%' % (100*((r>=1.30)&(r<=1.99)).mean()))
