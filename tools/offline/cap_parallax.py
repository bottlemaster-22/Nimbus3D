"""Real triangulation angle: the angle between the viewing rays of the cameras
that ACTUALLY see the same Gaussian, not all-pairs of cameras."""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
rays = np.load(os.path.join(OUT, '_cap_rays.npy'))     # S x 120 x 3, zero when unseen
rngm = np.load(os.path.join(OUT, '_cap_range.npy'))    # S x 120
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
S, V, _ = rays.shape
seen = np.abs(rays).sum(axis=2) > 0
nv = seen.sum(axis=1)
print('sample %d splats, %d views; view count median %d' % (S, V, np.median(nv)))

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
fx, fy, cx, cy = P.load_intrinsics(sl['renderWidth'], sl['renderHeight'])
print('render %dx%d  fx %.2f px  -> 1 px subtends %.4f deg'
      % (sl['renderWidth'], sl['renderHeight'], fx, np.degrees(1.0/fx)))

maxang = np.zeros(S)
p90ang = np.zeros(S)
medrng = np.zeros(S)
for i in range(S):
    m = seen[i]
    if m.sum() < 2:
        maxang[i] = np.nan; p90ang[i] = np.nan
        medrng[i] = rngm[i, m].mean() if m.sum() else np.nan
        continue
    Rv = rays[i, m].astype(np.float64)
    c = np.clip(Rv @ Rv.T, -1, 1)
    a = np.degrees(np.arccos(c))
    iu = np.triu_indices(len(Rv), 1)
    maxang[i] = a[iu].max()
    p90ang[i] = np.percentile(a[iu], 90)
    medrng[i] = np.median(rngm[i, m])

ok = ~np.isnan(maxang)
print('\n=== MAX triangulation angle over the cameras that see each Gaussian ===')
for q in (1, 5, 10, 25, 50, 75, 90, 99):
    print('  p%-3d %6.2f deg' % (q, np.percentile(maxang[ok], q)))
print('  mean %.2f deg' % maxang[ok].mean())
for th in (2, 5, 10, 15, 20, 30):
    print('  splats whose WIDEST pair is under %2d deg: %5.2f%%'
          % (th, 100*(maxang[ok] < th).mean()))

print('\n=== camera-to-splat range ===')
for q in (10, 50, 90):
    print('  p%-3d %.3f m' % (q, np.percentile(medrng[ok], q)))

# ---- what depth precision does that parallax buy? -----------------------
# Two views separated by parallax angle a, point at range z, matching to
# sigma_px pixels: sigma_z = z * sigma_px / (fx * a_radians)  (small angle).
th = np.radians(maxang[ok])
z = medrng[ok]
for spx in (0.5, 1.0):
    sz = z * spx / (fx * th)
    print('\n=== photometric depth precision at %.1f px matching error ===' % spx)
    for q in (10, 25, 50, 75, 90):
        print('  p%-3d %8.1f mm' % (q, 1000*np.percentile(sz, q)))
    print('  fraction WORSE than the LiDAR sigma (13.0 mm): %.1f%%'
          % (100*(sz > 0.0130).mean()))
    print('  fraction WORSE than the median splat size (9.12 mm): %.1f%%'
          % (100*(sz > 0.00912).mean()))
    print('  fraction WORSE than the Scaniverse splat size (3.39 mm): %.1f%%'
          % (100*(sz > 0.00339).mean()))
np.save(os.path.join(OUT, '_cap_maxang.npy'), maxang)
np.save(os.path.join(OUT, '_cap_medrng.npy'), medrng)
