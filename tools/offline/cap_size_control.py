"""Control for the obvious confound: a bigger splat is easier to see, and my
occlusion tolerance is itself proportional to splat size. Redo the size-vs-
evidence correlation with the FRUSTUM-ONLY mask, whose gates (radius >= 0.5 px,
alpha >= 1/255, on screen) barely bind over the 1.6-6.4 px range in question.
"""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls_mm = scale.max(axis=1)*1000.0
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))
fx, fy, cx, cy = P.load_intrinsics(720, 540)
px = (ls_mm[samp]/1000.0)*fx/medrng

raysF = np.load(os.path.join(OUT, '_cap_rays.npy'))       # frustum only
raysO = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))   # + occlusion
S, V, _ = raysF.shape

for tag, rays in (('FRUSTUM ONLY (no occlusion, no size-dependent tolerance)', raysF),
                  ('OCCLUSION AWARE', raysO)):
    seen = np.abs(rays).sum(axis=2) > 0
    nv = seen.sum(axis=1)
    mx = np.full(S, np.nan)
    for i in range(S):
        m = seen[i]
        if m.sum() < 2:
            continue
        Rv = rays[i, m].astype(np.float64)
        a = np.degrees(np.arccos(np.clip(Rv @ Rv.T, -1, 1)))
        iu = np.triu_indices(len(Rv), 1)
        mx[i] = a[iu].max()
    ok = ~np.isnan(mx) & ~np.isnan(medrng) & (medrng >= 0.9) & (medrng < 1.6)
    lp = np.log(px[ok])
    print('=== %s ===' % tag)
    print('  n=%d   r(view count, log px) %+.3f   r(max-pair angle, log px) %+.3f'
          % (ok.sum(), np.corrcoef(nv[ok], lp)[0, 1], np.corrcoef(mx[ok], lp)[0, 1]))
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 100), (100, 181)]:
        m = ok & (mx >= lo) & (mx < hi)
        if m.sum() >= 80:
            print('    max-pair %3d-%3d deg: %6d splats, median size %.2f px'
                  % (lo, hi, m.sum(), np.median(px[m])))
    print()

# and the reverse-causation check: does the frustum gate actually bind?
print('does the visibility gate depend on size over this range?')
print('  a 1.6 px splat and a 6.4 px splat both pass radius>=0.5 px; the gate that')
print('  can differ is the alpha floor. Opacity by size band, range 0.9-1.6 m:')
opacity = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))[samp]
ok = (medrng >= 0.9) & (medrng < 1.6) & ~np.isnan(medrng)
for lo, hi in [(0, 2), (2, 3), (3, 5), (5, 8), (8, 100)]:
    m = ok & (px >= lo) & (px < hi)
    if m.sum() >= 80:
        print('    %4.1f-%4.1f px: %6d splats, median opacity %.3f'
              % (lo, hi, m.sum(), np.median(opacity[m])))
