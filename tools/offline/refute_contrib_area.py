"""REFUTATION TEST 1: is `contrib` (sum over pixels of alpha*T) a
CONTRIBUTION metric or just an AREA metric?

If contrib is dominated by projected footprint, then "delete the bottom 50%
by contrib" == "delete the smaller half of the model", which is precisely the
population this project is trying to GROW (median 9.12mm vs Scaniverse
3.39mm).  Measures rank correlation and the size composition of the cut.
"""
import io, json, os
import numpy as np
import project as P
import detail as Dt

SCRATCH = Dt.SCRATCH

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
contrib = np.load(os.path.join(SCRATCH, 'contrib.npy'))
assert contrib.shape[0] == count

scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
s = np.sort(scale, axis=1)[:, ::-1]
largest_mm = s[:, 0] * 1000
# world-space ellipse area of the two dominant axes, mm^2
area_mm2 = np.pi * (s[:, 0] * 1000) * (s[:, 1] * 1000)
opac = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

print('n = %d' % count)
print('Spearman(contrib, largest axis mm)      = %+.4f' % spearman(contrib, largest_mm))
print('Spearman(contrib, world ellipse area)   = %+.4f' % spearman(contrib, area_mm2))
print('Spearman(contrib, opacity)              = %+.4f' % spearman(contrib, opac))
print('Spearman(contrib, largest*opacity)      = %+.4f' % spearman(contrib, largest_mm * opac))
print('Spearman(largest axis, opacity)         = %+.4f' % spearman(largest_mm, opac))

order = np.argsort(contrib)
n = count
for frac in (0.25, 0.50, 0.75):
    cut = order[:int(frac * n)]
    keep = np.ones(n, bool); keep[cut] = False
    print('\n--- drop bottom %.0f%% by contrib ---' % (frac * 100))
    print('  median largest axis: cut %.3f mm   kept %.3f mm   (whole model %.3f mm)'
          % (np.median(largest_mm[cut]), np.median(largest_mm[keep]),
             np.median(largest_mm)))
    print('  p10/p90 largest axis kept: %.3f / %.3f mm  (whole model %.3f / %.3f)'
          % (np.percentile(largest_mm[keep], 10), np.percentile(largest_mm[keep], 90),
             np.percentile(largest_mm, 10), np.percentile(largest_mm, 90)))

# Where do the SMALL splats go?
print('\n=== survival rate of the drop-50%% cut, by splat size decile ===')
drop50 = np.zeros(n, bool); drop50[order[:n // 2]] = True
edges = np.percentile(largest_mm, np.arange(0, 101, 10))
print('%-22s %8s %10s' % ('size decile (mm)', 'n', 'deleted%'))
for i in range(10):
    lo, hi = edges[i], edges[i + 1]
    m = (largest_mm >= lo) & (largest_mm <= hi if i == 9 else largest_mm < hi)
    print('%-22s %8d %9.1f%%' % ('%.2f - %.2f' % (lo, hi), m.sum(),
                                 100 * drop50[m].sum() / max(m.sum(), 1)))

# The headline: what happens to the size statistic the project cares about
print('\n=== the project bar is MEDIAN SPLAT SIZE (Scaniverse 3.39 mm) ===')
print('  full model              median %.3f mm' % np.median(largest_mm))
for frac in (0.25, 0.50, 0.75, 0.90):
    keep = np.ones(n, bool); keep[order[:int(frac * n)]] = False
    print('  after dropping %5.0f%%   median %.3f mm   (moves %+.3f mm, %s)'
          % (frac * 100, np.median(largest_mm[keep]),
             np.median(largest_mm[keep]) - np.median(largest_mm),
             'AWAY from the bar' if np.median(largest_mm[keep]) > np.median(largest_mm) else 'toward'))

# how much of the sub-5mm "detail" population survives
for thr in (3.39, 5.0, 6.8):
    small = largest_mm < thr
    print('  splats < %.2f mm: %d (%.1f%%) -> after drop50 %d survive (%.1f%% of them)'
          % (thr, small.sum(), 100 * small.sum() / n, (small & ~drop50).sum(),
             100 * (small & ~drop50).sum() / max(small.sum(), 1)))
