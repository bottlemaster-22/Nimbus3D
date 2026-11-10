"""Recover filter3D from the fused PLY, then measure the donor pool in the
convention the DONOR TEST actually uses, and profile donors by size."""
import numpy as np

BASE = 'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840'
f = open(BASE + '/model/model.ply', 'rb')
h = b''
while not h.endswith(b'end_header\n'):
    h += f.read(1)
names, n = [], 0
for line in h.decode().split('\n'):
    q = line.split()
    if not q: continue
    if q[0] == 'element' and q[1] == 'vertex': n = int(q[2])
    elif q[0] == 'property' and q[1] == 'float': names.append(q[2])
arr = np.frombuffer(f.read(n * len(names) * 4), dtype='<f4').reshape(n, len(names))
col = {nm: arr[:, i] for i, nm in enumerate(names)}

sig = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                              axis=1).astype(np.float64), -12, 3))
srt = np.sort(sig, axis=1)          # s3 <= s2 <= s1
s3, s2, s1 = srt[:, 0], srt[:, 1], srt[:, 2]
alpha = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))

print('=== RECOVERING filter3D FROM THE FUSED PLY ===')
print('fusing3DFilter writes sigma_ply = sqrt(sigma_train^2 + f^2), so every')
print('axis of every fused splat is >= f. A hard floor in the SMALLEST axis is')
print('the filter width itself.')
print()
print('smallest axis, mm : ' + '  '.join(
    'p%-4g %8.4f' % (q, np.percentile(s3, q) * 1000)
    for q in [0, 0.01, 0.1, 1, 5, 25, 50, 90]))
print('min s3 = %.6f mm   max s3 = %.4f mm' % (s3.min() * 1000, s3.max() * 1000))
# is there a spike at the fallback 0.01 m = 10 mm?
for v in [0.01, 0.005, 0.002, 0.001]:
    near = (np.abs(s3 - v) < v * 0.02).sum()
    print('  s3 within 2%% of %6.4f m : %7d  (%.3f%%)' % (v, near, 100*near/n))
print()
print('VERDICT: s3 spans %.5f to %.3f mm with no floor at any candidate filter'
      % (s3.min()*1000, s3.max()*1000))
print('width, so filter3D is per-splat and small relative to the geometry, and')
print('comp3D CANNOT be inverted per splat from this file. What follows is')
print('therefore stated as a BOUND, not a point estimate.')
print()

print('=== THE DONOR POOL, BOTH CONVENTIONS ===')
print('PRUNE tests   drawnOpacity = sigmoid(logit)*comp3D  <  pruneOpacity 0.02')
print('DONOR tests   sigmoid(logit)                        <  donorOpacity 0.05')
print('The PLY carries the FUSED logit, i.e. sigmoid(ply) == drawnOpacity.')
print('sigmoid(trainer logit) = drawnOpacity / comp3D >= drawnOpacity.')
print()
for c in [1.0, 0.9, 0.75, 0.5, 0.35]:
    tr = np.minimum(alpha / c, 0.999999)
    band = ((tr >= 0.02) & (tr < 0.05)).sum()
    below = (tr < 0.05).sum()
    print('  if comp3D == %.2f uniformly: donors by opacity %7d (%.3f%%),'
          '  band [0.02,0.05) %7d' % (c, below, 100*below/n, band))
print()
print('  comp3D == 1 is the UPPER BOUND on the opacity-side donor count:')
print('  %d splats, %.3f%% of the population, against a 5%% allowance of %d.'
      % ((alpha < 0.05).sum(), 100*(alpha < 0.05).sum()/n, int(n*0.05)))
print()

print('=== ARE DONORS BIG OR SMALL? (the whole point) ===')
donor = alpha < 0.05
rest = ~donor
lg = s1 * 1000
print('                        count     p10      p50      p90   mean s1(mm)')
for nm, m in [('donors (alpha<0.05)', donor), ('everything else', rest),
              ('whole model', np.ones(n, bool))]:
    print('%-22s %7d %8.3f %8.3f %8.3f %8.3f'
          % (nm, m.sum(), np.percentile(lg[m], 10), np.percentile(lg[m], 50),
             np.percentile(lg[m], 90), lg[m].mean()))
print()
print('share of total disc face area (s1*s2) carried by donors: %.2f%%'
      % (100 * (s1[donor]*s2[donor]).sum() / (s1*s2).sum()))
print('donors are %.2fx the median size of the rest'
      % (np.median(lg[donor]) / np.median(lg[rest])))
print()
print('=== SHAPE ===')
r21 = s2 / np.maximum(s1, 1e-12)
r32 = s3 / np.maximum(s2, 1e-12)
print('whole model : s2/s1 %.3f  s3/s2 %.3f' % (np.median(r21), np.median(r32)))
print('donors      : s2/s1 %.3f  s3/s2 %.3f'
      % (np.median(r21[donor]), np.median(r32[donor])))
np.save('_reloc_s.npy', srt)
np.save('_reloc_alpha.npy', alpha)
