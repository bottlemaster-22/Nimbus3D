"""Same test, range confound removed: splat size in PIXELS at the render size
the trainer used, versus view count and triangulation angle, within range bands."""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls_mm = scale.max(axis=1)*1000.0
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
maxang = np.load(os.path.join(OUT, '_cap_maxang_occ.npy'))
medang = np.load(os.path.join(OUT, '_cap_medang_occ.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))
nv = np.load(os.path.join(OUT, '_cap_nviews_occ.npy')).astype(int)[samp]
fx, fy, cx, cy = P.load_intrinsics(720, 540)
ok = ~np.isnan(maxang) & ~np.isnan(medrng)
px = (ls_mm[samp]/1000.0)*fx/medrng
print('splat largest axis in pixels: median %.2f px   (Scaniverse-equivalent 1.38 px)'
      % np.median(px[ok]))
print('overall median range %.2f m' % np.median(medrng[ok]))

BANDS = [(0.6, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 2.2)]
print('\n=== median splat size in PIXELS, by view count, WITHIN a range band ===')
print('%-12s' % 'range m', end='')
for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 21), (21, 40)]:
    print('%9s' % ('%d-%d v' % (lo, hi-1)), end='')
print('%9s' % 'n')
for rl, rh in BANDS:
    rm = ok & (medrng >= rl) & (medrng < rh)
    print('%-12s' % ('%.1f-%.1f' % (rl, rh)), end='')
    for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 21), (21, 40)]:
        m = rm & (nv >= lo) & (nv < hi)
        print('%9s' % ('%.2f' % np.median(px[m]) if m.sum() >= 80 else '-'), end='')
    print('%9d' % rm.sum())

print('\n=== median splat size in PIXELS, by MAX-PAIR triangulation angle, within a range band ===')
ANG = [(0, 15), (15, 30), (30, 60), (60, 100), (100, 181)]
print('%-12s' % 'range m', end='')
for a in ANG:
    print('%9s' % ('%d-%d' % a), end='')
print()
for rl, rh in BANDS:
    rm = ok & (medrng >= rl) & (medrng < rh)
    print('%-12s' % ('%.1f-%.1f' % (rl, rh)), end='')
    for lo, hi in ANG:
        m = rm & (maxang >= lo) & (maxang < hi)
        print('%9s' % ('%.2f' % np.median(px[m]) if m.sum() >= 80 else '-'), end='')
    print()

print('\npartial correlations of log(size in px), range band 0.9-1.6 m:')
m = ok & (medrng >= 0.9) & (medrng < 1.6)
lp = np.log(px[m])
for name, v in (('view count', nv[m]), ('max-pair angle', maxang[m]),
                ('median-pair angle', medang[m]), ('range', medrng[m])):
    print('  r(%-18s, log px size) = %+.3f   (n=%d)' % (name, np.corrcoef(v, lp)[0, 1], m.sum()))
