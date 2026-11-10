"""Null test: recompute the QC card's medianAngularSpreadDegrees the way
PrePassSurveyor does it (per 10 cm surface voxel, MAX angle from the FIRST
ray that hit it) but from the splat model + keyframes, and see whether the
per-splat number and the per-voxel number really are the same population.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
rays = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))     # S x 120 x 3
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
S, V, _ = rays.shape
seen = np.abs(rays).sum(axis=2) > 0

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)[samp]

for vox in (0.05, 0.10, 0.25):
    key = np.floor(mean/vox).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nvox = inv.max()+1
    firstdir = np.zeros((nvox, 3)); havefirst = np.zeros(nvox, bool)
    spread = np.zeros(nvox); hits = np.zeros(nvox, np.int64)
    for j in range(V):
        m = seen[:, j]
        vi = inv[m]; d = rays[m, j].astype(np.float64)
        new = ~havefirst[vi]
        firstdir[vi[new]] = d[new]; havefirst[vi[new]] = True
        hits[vi] += 1
        c = np.clip(np.einsum('ij,ij->i', d, firstdir[vi]), -1, 1)
        a = np.degrees(np.arccos(c))
        np.maximum.at(spread, vi, a)
    use = hits >= 2
    print('voxel %.2f m: %6d voxels, %6d with >=2 hits' % (vox, nvox, use.sum()))
    print('   per-VOXEL max-from-first spread: p25 %.2f  MEDIAN %.2f  p75 %.2f  mean %.2f deg'
          % tuple(np.percentile(spread[use], [25, 50, 75]).tolist() + [spread[use].mean()]))
    # per-splat version of the SAME statistic, for comparison
    ps = np.zeros(S); first = np.zeros((S, 3)); hf = np.zeros(S, bool); hc = np.zeros(S, np.int64)
    for j in range(V):
        m = seen[:, j]; d = rays[m, j].astype(np.float64)
        idx = np.flatnonzero(m); new = ~hf[idx]
        first[idx[new]] = d[new]; hf[idx[new]] = True; hc[idx] += 1
        a = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', d, first[idx]), -1, 1)))
        ps[idx] = np.maximum(ps[idx], a)
    u2 = hc >= 2
    print('   per-SPLAT same statistic:        p25 %.2f  MEDIAN %.2f  p75 %.2f  mean %.2f deg'
          % tuple(np.percentile(ps[u2], [25, 50, 75]).tolist() + [ps[u2].mean()]))
    print()
print('QC card, measured by the app over 49 stride-sampled frames and LiDAR')
print('returns at 10 cm voxels: medianAngularSpreadDegrees = 10.28')
