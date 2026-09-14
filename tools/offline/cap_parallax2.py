"""Triangulation angle using the OCCLUSION-AWARE visibility mask."""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
rays = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))
rngm = np.load(os.path.join(OUT, '_cap_range.npy'))
S, V, _ = rays.shape
seen = np.abs(rays).sum(axis=2) > 0
nv = seen.sum(axis=1)
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
fx, fy, cx, cy = P.load_intrinsics(sl['renderWidth'], sl['renderHeight'])
print('sample %d splats; occlusion-aware view count median %d mean %.2f'
      % (S, np.median(nv), nv.mean()))

maxang = np.full(S, np.nan); medang = np.full(S, np.nan)
cone = np.full(S, np.nan); medrng = np.full(S, np.nan)
for i in range(S):
    m = seen[i]
    if m.sum():
        medrng[i] = np.median(rngm[i, m])
    if m.sum() < 2:
        continue
    Rv = rays[i, m].astype(np.float64)
    a = np.degrees(np.arccos(np.clip(Rv @ Rv.T, -1, 1)))
    iu = np.triu_indices(len(Rv), 1)
    maxang[i] = a[iu].max(); medang[i] = np.median(a[iu])
    mu = Rv.mean(axis=0); mu /= np.linalg.norm(mu)
    cone[i] = np.degrees(np.arccos(np.clip(Rv @ mu, -1, 1))).max()

ok = ~np.isnan(maxang)
print('\n=== triangulation angle per Gaussian, occlusion aware (n=%d) ===' % ok.sum())
print('%-8s %8s %8s %8s' % ('pct', 'max-pair', 'med-pair', 'cone-half'))
for q in (1, 5, 10, 25, 50, 75, 90, 99):
    print('p%-7d %8.2f %8.2f %8.2f'
          % (q, np.percentile(maxang[ok], q), np.percentile(medang[ok], q),
             np.percentile(cone[ok], q)))
print('mean     %8.2f %8.2f %8.2f' % (maxang[ok].mean(), medang[ok].mean(), cone[ok].mean()))
print()
for th in (2, 5, 10, 15, 20, 30):
    print('  widest pair under %2d deg: %5.2f%%   |  median pair under %2d deg: %5.2f%%'
          % (th, 100*(maxang[ok] < th).mean(), th, 100*(medang[ok] < th).mean()))

# ---- depth precision the parallax buys ----------------------------------
z = medrng[ok]
print('\ncamera-to-splat range: p10 %.2f  p50 %.2f  p90 %.2f m'
      % tuple(np.percentile(z, [10, 50, 90])))
for label, angdeg in (('widest pair', maxang[ok]), ('median pair', medang[ok])):
    th = np.radians(np.maximum(angdeg, 1e-6))
    for spx in (0.5, 1.0):
        sz = z*spx/(fx*th)
        print('\n  depth sigma from the %s at %.1f px matching error:' % (label, spx))
        print('    p10 %.2f  p25 %.2f  p50 %.2f  p75 %.2f  p90 %.2f mm'
              % tuple(1000*np.percentile(sz, [10, 25, 50, 75, 90])))
        print('    worse than LiDAR sigma 13.0 mm: %5.1f%%   worse than 3.39 mm: %5.1f%%'
              % (100*(sz > 0.0130).mean(), 100*(sz > 0.00339).mean()))
np.save(os.path.join(OUT, '_cap_maxang_occ.npy'), maxang)
np.save(os.path.join(OUT, '_cap_medang_occ.npy'), medang)
np.save(os.path.join(OUT, '_cap_medrng.npy'), medrng)
