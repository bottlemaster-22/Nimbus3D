"""REFUTE 'splat size grows with evidence'.

Three attacks:
 1. px = worldsize * fx / medrng.  Correlating px against view count inside a
    range BAND still leaves 1/medrng in the numerator.  Redo it on WORLD size.
 2. Local model spacing (kNN) is the obvious confound: sparse regions hold big
    splats and sparse regions are the open middle of the room, which is what
    many cameras see.  Partial-correlate.
 3. Opacity: the Mip-Splatting 2D filter multiplies opacity by
    comp2d = sqrt(det/det+filter).  Small splats get compensated HARDEST, so
    the optimiser has to store a HIGHER logit for them.  Compare stored
    opacity against EFFECTIVE alpha = opacity * comp2d.
"""
import io, json, os
import numpy as np
from scipy.spatial import cKDTree
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls_mm = scale.max(axis=1)*1000.0
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
maxang = np.load(os.path.join(OUT, '_cap_maxang_occ.npy'))
medrng = np.load(os.path.join(OUT, '_cap_medrng.npy'))
nv = np.load(os.path.join(OUT, '_cap_nviews_occ.npy')).astype(float)[samp]
nsub = np.load(os.path.join(OUT, '_v_nsub.npy')).astype(float)
fx, fy, cx, cy = P.load_intrinsics(720, 540)
px = (ls_mm[samp]/1000.0)*fx/medrng
world = ls_mm[samp]

def pcorr(a, b, ctrls):
    """partial correlation of a,b controlling for the columns of ctrls"""
    X = np.column_stack([np.ones(len(a))] + list(ctrls))
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]

ok = ~np.isnan(maxang) & ~np.isnan(medrng)
m = ok & (medrng >= 0.9) & (medrng < 1.6)
print('n = %d in the 0.9-1.6 m band' % m.sum())
lpx, lw, lr = np.log(px[m]), np.log(world[m]), np.log(medrng[m])

print('\n--- 1. PIXEL size vs WORLD size (the 1/range term) ---')
print('  r(view count , log PX    size) = %+.3f   <- the finding' % np.corrcoef(nv[m], lpx)[0, 1])
print('  r(view count , log WORLD size) = %+.3f' % np.corrcoef(nv[m], lw)[0, 1])
print('  r(view count , log range     ) = %+.3f' % np.corrcoef(nv[m], lr)[0, 1])
print('  r(max-pair   , log PX    size) = %+.3f   <- the finding' % np.corrcoef(maxang[m], lpx)[0, 1])
print('  r(max-pair   , log WORLD size) = %+.3f' % np.corrcoef(maxang[m], lw)[0, 1])
print('  r(max-pair   , log range     ) = %+.3f' % np.corrcoef(maxang[m], lr)[0, 1])
print('  partial r(view count, log PX | log range) = %+.3f' % pcorr(nv[m], lpx, [lr]))

print('\n--- 2. local model spacing as a confound ---')
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
tree = cKDTree(mean)
d, _ = tree.query(mean[samp], k=9, workers=-1)
spacing_mm = d[:, 8]*1000.0                 # distance to the 8th neighbour
ls = np.log(spacing_mm[m])
print('  local 8-NN spacing mm: p10 %.2f MEDIAN %.2f p90 %.2f' %
      tuple(np.percentile(spacing_mm[m], [10, 50, 90])))
print('  r(log spacing, log WORLD size) = %+.3f' % np.corrcoef(ls, lw)[0, 1])
print('  r(view count , log spacing   ) = %+.3f' % np.corrcoef(nv[m], ls)[0, 1])
print('  partial r(view count, log WORLD | log spacing, log range) = %+.3f'
      % pcorr(nv[m], lw, [ls, lr]))
print('  partial r(max-pair  , log WORLD | log spacing, log range) = %+.3f'
      % pcorr(maxang[m], lw, [ls, lr]))
print('  ratio world/spacing (how much bigger than its own neighbourhood a splat is):')
ratio = world[m]/spacing_mm[m]
print('     r(view count, log ratio) = %+.3f    r(max-pair, log ratio) = %+.3f'
      % (np.corrcoef(nv[m], np.log(ratio))[0, 1], np.corrcoef(maxang[m], np.log(ratio))[0, 1]))
print('     median ratio by view-count band:', end=' ')
for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 21), (21, 40)]:
    sel = (nv[m] >= lo) & (nv[m] < hi)
    print('%d-%d:%.2f' % (lo, hi-1, np.median(ratio[sel])) if sel.sum() >= 80 else '-', end='  ')
print()

print('\n--- 3. opacity: stored logit vs EFFECTIVE alpha after the 2D filter ---')
op = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))[samp]
# comp2d for a face-on disc of the two largest world axes at its median range
srt = np.sort(scale[samp], axis=1)[:, ::-1]
p1 = fx*srt[:, 0]/medrng
p2 = fx*srt[:, 1]/medrng
det_b = (p1*p1)*(p2*p2)
det_a = (p1*p1+0.25)*(p2*p2+0.25)
comp2d = np.sqrt(np.clip(det_b/det_a, 0, 1))
eff = op*comp2d
BAND = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]
print('  %-12s %8s %8s %8s %8s' % ('px size', 'n', 'opacity', 'comp2d', 'eff alpha'))
for lo, hi in BAND:
    s = ok & (px >= lo) & (px < hi)
    if s.sum() < 50:
        continue
    print('  %-12s %8d %8.3f %8.3f %8.3f'
          % ('%d-%d px' % (lo, hi), s.sum(), np.median(op[s]), np.median(comp2d[s]), np.median(eff[s])))

print('\n--- 4. submap span, the pose-relevant predictor ---')
print('  r(distinct submaps, log WORLD size) = %+.3f' % np.corrcoef(nsub[m], lw)[0, 1])
print('  partial r(submaps, log WORLD | view count, log spacing, log range) = %+.3f'
      % pcorr(nsub[m], lw, [nv[m], ls, lr]))
print('  partial r(view count, log WORLD | submaps, log spacing, log range) = %+.3f'
      % pcorr(nv[m], lw, [nsub[m], ls, lr]))
np.save(os.path.join(OUT, '_v_spacing.npy'), spacing_mm)
