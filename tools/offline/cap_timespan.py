"""Discriminating test.

Two rival explanations for "more views -> bigger splats":
  (1) the extra views come from a wide baseline, and wide baselines are hard;
  (2) the extra views come from LATER IN THE WALK, and the pose graph put the
      later visit ~10 cm away from the earlier one, so the splat must grow to
      straddle both.
(2) predicts size tracks the TIME SPAN of the views that see the splat.
(1) predicts size tracks the ANGLE and not the time span.
Both are measured here, with range and view count held in a band.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls_mm = scale.max(axis=1)*1000.0
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))
maxang = np.load(os.path.join(OUT, '_cap_maxang_occ.npy'))
rays = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))
seen = np.abs(rays).sum(axis=2) > 0
b, r, frames, pool, kf, info = keyframes()
ts = np.array([f['ts'] for f in kf]); ts -= ts[0]
fx, fy, cx, cy = P.load_intrinsics(720, 540)
px = (ls_mm[samp]/1000.0)*fx/medrng
nv = seen.sum(axis=1)

span = np.full(len(px), np.nan)
for i in range(len(px)):
    m = seen[i]
    if m.sum() >= 2:
        t = ts[m]
        span[i] = t.max()-t.min()

base = ~np.isnan(span) & ~np.isnan(maxang) & (medrng >= 0.9) & (medrng < 1.6)
print('n = %d splats, range 0.9-1.6 m' % base.sum())
print('view time span: p10 %.0f  p50 %.0f  p90 %.0f s  (walk is 276 s, keyframes span 153 s)'
      % tuple(np.percentile(span[base], [10, 50, 90])))

lp = np.log(px[base])
for nm, v in (('time span (s)', span[base]), ('max-pair angle', maxang[base]),
              ('view count', nv[base])):
    print('  r(%-16s, log px size) = %+.3f' % (nm, np.corrcoef(v, lp)[0, 1]))

print('\n=== size in px: TIME SPAN vs ANGLE, both held (range 0.9-1.6 m) ===')
SP = [(0, 20), (20, 60), (60, 110), (110, 160)]
AN = [(0, 25), (25, 55), (55, 95), (95, 181)]
print('%-16s' % 'span \ angle', end='')
for a in AN:
    print('%12s' % ('%d-%d deg' % a), end='')
print()
for sl, sh in SP:
    print('%-16s' % ('%d-%d s' % (sl, sh)), end='')
    for al, ah in AN:
        m = base & (span >= sl) & (span < sh) & (maxang >= al) & (maxang < ah)
        print('%12s' % ('%.2f px' % np.median(px[m]) if m.sum() >= 60 else '(%d)' % m.sum()), end='')
    print()

print('\n=== the same table but for VIEW COUNT held near the median (8-16 views) ===')
b2 = base & (nv >= 8) & (nv <= 16)
print('  n = %d' % b2.sum())
print('%-16s' % 'span \ angle', end='')
for a in AN:
    print('%12s' % ('%d-%d deg' % a), end='')
print()
for sl, sh in SP:
    print('%-16s' % ('%d-%d s' % (sl, sh)), end='')
    for al, ah in AN:
        m = b2 & (span >= sl) & (span < sh) & (maxang >= al) & (maxang < ah)
        print('%12s' % ('%.2f px' % np.median(px[m]) if m.sum() >= 40 else '(%d)' % m.sum()), end='')
    print()
