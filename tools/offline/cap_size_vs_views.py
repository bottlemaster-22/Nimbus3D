"""If the CAPTURE geometry were the ceiling, splats with more views and more
parallax would be sharper (smaller). If POSE INCONSISTENCY is the ceiling, more
views means more mutually contradictory evidence, and the splat must grow to
stay consistent with all of them. Opposite predictions; measure which happens.
"""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
largest_mm = scale.max(axis=1)*1000.0
opacity = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))
nv = np.load(os.path.join(OUT, '_cap_nviews_occ.npy')).astype(int)
nd = np.load(os.path.join(OUT, '_cap_azsectors.npy')).astype(int)
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
maxang = np.load(os.path.join(OUT, '_cap_maxang_occ.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))

print('model: median largest axis %.2f mm  (census says 9.12)' % np.median(largest_mm))
print()
print('=== splat size vs OCCLUSION-AWARE view count ===')
print('%-12s %8s %10s %10s %10s' % ('views', 'splats', 'median mm', 'p25 mm', 'median op'))
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 21), (21, 30), (30, 999)]:
    m = (nv >= lo) & (nv < hi)
    if m.sum() < 200:
        continue
    print('%-12s %8d %10.2f %10.2f %10.3f'
          % ('%d-%d' % (lo, hi-1), m.sum(), np.median(largest_mm[m]),
             np.percentile(largest_mm[m], 25), np.median(opacity[m])))
r = np.corrcoef(nv, np.log(largest_mm))[0, 1]
print('  Pearson r(view count, log size) = %+.3f' % r)

print('\n=== splat size vs distinct 10-deg AZIMUTH sectors ===')
print('%-12s %8s %10s' % ('sectors', 'splats', 'median mm'))
for k in range(0, 12):
    m = nd == k
    if m.sum() < 200:
        continue
    print('%-12d %8d %10.2f' % (k, m.sum(), np.median(largest_mm[m])))
print('  Pearson r(sectors, log size) = %+.3f' % np.corrcoef(nd, np.log(largest_mm))[0, 1])

print('\n=== splat size vs TRIANGULATION ANGLE (sampled splats) ===')
ls = largest_mm[samp]
ok = ~np.isnan(maxang)
print('%-14s %8s %10s %10s' % ('max-pair deg', 'splats', 'median mm', 'median range m'))
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 70), (70, 110), (110, 181)]:
    m = ok & (maxang >= lo) & (maxang < hi)
    if m.sum() < 100:
        continue
    print('%-14s %8d %10.2f %10.2f' % ('%d-%d' % (lo, hi), m.sum(),
                                       np.median(ls[m]), np.median(medrng[m])))
print('  Pearson r(max-pair angle, log size) = %+.3f'
      % np.corrcoef(maxang[ok], np.log(ls[ok]))[0, 1])

print('\n=== control: splat size vs RANGE (a splat far away is bigger in world units')
print('    simply because the pixel it must cover is bigger) ===')
print('%-14s %8s %10s %12s' % ('range m', 'splats', 'median mm', 'mm per px'))
for lo, hi in [(0, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.6), (2.6, 10)]:
    m = ok & (medrng >= lo) & (medrng < hi)
    if m.sum() < 100:
        continue
    fx, fy, cx, cy = P.load_intrinsics(720, 540)
    print('%-14s %8d %10.2f %12.2f' % ('%.1f-%.1f' % (lo, hi), m.sum(),
                                       np.median(ls[m]), 1000*np.median(medrng[m])/fx))
# size in PIXELS is the scale-free version
fx, fy, cx, cy = P.load_intrinsics(720, 540)
px = (ls[ok]/1000.0)*fx/medrng[ok]
print('\nsplat largest axis in PIXELS at the render size the trainer used:')
print('  p10 %.2f  p25 %.2f  MEDIAN %.2f  p75 %.2f  p90 %.2f px'
      % tuple(np.percentile(px, [10, 25, 50, 75, 90])))
print('  Scaniverse 3.39 mm at 1.27 m would be %.2f px at this fx'
      % (0.00339*fx/1.27))
