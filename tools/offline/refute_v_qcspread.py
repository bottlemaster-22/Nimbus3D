"""REFUTATION of the 'UNRESOLVED 3x angular-spread disagreement'.

PrePassSurvey.swift:387 is  result.medianAngularSpreadDegrees = median(spreads)
where `spreads` is appended for EVERY hash entry (line 352), i.e. over ALL
surface voxels including the ones only one keyframe ever hit (their spread
stays at the 0 it was inserted with, or the small within-frame ray fan).

Both replications gate the median on `hits >= 2` / `distinct >= 2`
(cap_voxel_spread.py line 39, cap_qc_reconcile.py line 88). That is a
different population. Recompute WITHOUT the gate.
"""
import io, json, os
import numpy as np
import project as P

OUT = os.path.dirname(os.path.abspath(__file__))
rays = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))     # S x 120 x 3
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
S, V, _ = rays.shape
seen = np.abs(rays).sum(axis=2) > 0
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)[samp]

for vox in (0.10,):
    key = np.floor(mean/vox).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nvox = inv.max()+1
    firstdir = np.zeros((nvox, 3)); havefirst = np.zeros(nvox, bool)
    spread = np.zeros(nvox); hits = np.zeros(nvox, np.int64)
    distinct = np.zeros(nvox, np.int64)
    for j in range(V):
        m = seen[:, j]
        if not m.any():
            continue
        vi = inv[m]; d = rays[m, j].astype(np.float64)
        new = ~havefirst[vi]
        firstdir[vi[new]] = d[new]; havefirst[vi[new]] = True
        np.add.at(hits, vi, 1)
        distinct[np.unique(vi)] += 1
        a = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', d, firstdir[vi]), -1, 1)))
        np.maximum.at(spread, vi, a)
    touched = hits > 0
    print('voxel %.2f m   touched voxels %d' % (vox, touched.sum()))
    print('  distinct-frame count per voxel: p10 %d p25 %d MEDIAN %d p75 %d  frac==1 %.1f%%'
          % (*np.percentile(distinct[touched], [10, 25, 50, 75]).astype(int),
             100*np.mean(distinct[touched] == 1)))
    for label, mask in (('ALL touched voxels  (what the app medians over)', touched),
                        ('distinct >= 2       (what the replication used)', distinct >= 2),
                        ('distinct >= 3', distinct >= 3)):
        s = spread[mask]
        print('  %-46s n=%6d  p25 %5.2f MEDIAN %5.2f p75 %5.2f'
              % (label, mask.sum(), *np.percentile(s, [25, 50, 75])))
    # what fraction would clear the app's own covered-gate spread threshold of 15 deg
    print('  fraction of ALL touched voxels with spread >= 15 deg (app coverage gate): %.1f%%'
          % (100*np.mean(spread[touched] >= 15)))
    print('  app publishes coverageFraction 0.4242 (needs hits>=3 AND distinct>=2 AND spread>=15)')
