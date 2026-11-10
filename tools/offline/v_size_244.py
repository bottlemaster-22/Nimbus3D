"""Entry 12's decisive test, re-run on the BUILD 244 model (the one whose
pose graph ran 300 iterations instead of 30). Frustum-only visibility, which
is the variant with no occlusion tolerance to confound the size term."""
import io, json, os
import numpy as np
import project as P

D = P.D
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
world_mm = scale.max(axis=1) * 1000.0
opac = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))
print('build 244 model: %d splats, median largest axis %.2f mm, p10 %.2f p90 %.2f (spread %.2fx)'
      % (count, np.median(world_mm), np.percentile(world_mm,10), np.percentile(world_mm,90),
         np.percentile(world_mm,90)/np.percentile(world_mm,10)))

poses = P.load_poses()
b = json.load(io.open(os.path.join(D,'capture_bundle.json'), encoding='utf-8'))
fx, fy, cx, cy = P.load_intrinsics(720, 540)
keys = sorted(poses.keys())
views = keys[::max(1, len(keys)//120)][:120]
print('views used: %d' % len(views))

rng = np.random.default_rng(0)
samp = rng.choice(count, size=min(60000, count), replace=False)
samp.sort()
pts = mean[samp]

nv = np.zeros(len(samp), np.int32)
rsum = np.zeros(len(samp))
rlist = []
dirs = []
for f in views:
    rot, t = poses[f]
    R = P.quat_to_matrix(rot)
    cam = pts @ R.T + t
    z = cam[:,2]
    u = fx*cam[:,0]/np.where(z==0,1e-9,z) + cx
    v = fy*cam[:,1]/np.where(z==0,1e-9,z) + cy
    vis = (z>0.05)&(z<5.0)&(u>=0)&(u<720)&(v>=0)&(v<540)
    centre = -R.T @ t
    d = pts - centre
    rr = np.linalg.norm(d, axis=1)
    nv += vis
    rlist.append(np.where(vis, rr, np.nan))
    dirs.append(np.where(vis[:,None], d/np.maximum(rr,1e-9)[:,None], np.nan))

Rm = np.array(rlist)                      # views x samples
Dm = np.array(dirs)                       # views x samples x 3
medrng = np.nanmedian(Rm, axis=0)
# max pairwise angle between viewing directions, per splat
maxang = np.full(len(samp), np.nan)
for i in range(len(samp)):
    if nv[i] < 2: continue
    dd = Dm[:, i, :]
    dd = dd[~np.isnan(dd[:,0])]
    g = np.clip(dd @ dd.T, -1, 1)
    maxang[i] = np.degrees(np.arccos(g.min()))

ok = (nv >= 1) & ~np.isnan(medrng)
px = (world_mm[samp]/1000.0)*fx/medrng
print('median splat %.2f render px  (Scaniverse-equivalent 1.38 px)' % np.median(px[ok]))

# 8-NN local spacing on the sampled subset
from scipy.spatial import cKDTree
tree = cKDTree(mean)
dist, _ = tree.query(pts, k=9)
spacing_mm = dist[:,1:].mean(axis=1)*1000.0
print('8-NN spacing mm: p10 %.2f p50 %.2f p90 %.2f' % tuple(np.percentile(spacing_mm,[10,50,90])))

def partial(y, x, controls):
    A = np.column_stack([np.ones_like(x)] + controls)
    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    return np.corrcoef(resid(x), resid(y))[0,1]

m = ok & (medrng>=0.9) & (medrng<1.6) & ~np.isnan(maxang)
lw = np.log(world_mm[samp][m]); lp = np.log(px[m])
c = [np.log(spacing_mm[m]), np.log(medrng[m])]
print('\nBUILD 244, range band 0.9-1.6 m, n=%d (frustum-only visibility)' % m.sum())
print('  r(view count,     log PX   size) = %+.3f     [build 234 refutation: +0.544]'
      % np.corrcoef(nv[m], lp)[0,1])
print('  r(view count,     log WORLD size) = %+.3f    [build 234 refutation: +0.219]'
      % np.corrcoef(nv[m], lw)[0,1])
print('  r(max-pair angle, log WORLD size) = %+.3f    [build 234 refutation: +0.198]'
      % np.corrcoef(maxang[m], lw)[0,1])
print('  partial r(view count, log WORLD | spacing, range) = %+.3f   [build 234 refutation: +0.338]'
      % partial(lw, nv[m].astype(float), c))
print('  r(view count, log opacity) = %+.3f' % np.corrcoef(nv[m], np.log(opac[samp][m]))[0,1])
print('\nworld/spacing ratio by view-count band:')
for lo,hi in [(1,3),(3,6),(6,10),(10,15),(15,21),(21,200)]:
    mm = m & (nv>=lo) & (nv<hi)
    if mm.sum() < 80: continue
    print('   %3d-%-3d v  n=%6d  median world %.2f mm  ratio %.3f  median opacity %.3f'
          % (lo,hi-1,mm.sum(), np.median(world_mm[samp][mm]),
             np.median(world_mm[samp][mm]/spacing_mm[mm]), np.median(opac[samp][mm])))
