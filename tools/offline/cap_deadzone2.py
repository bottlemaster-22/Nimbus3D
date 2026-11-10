"""Locate the dead zone and price it: how much of the model sits in it, how
badly is that part covered, and does the discarded tail of the walk fix it?"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
cen = mean.mean(axis=0)
b, r, frames, pool, kf, info = keyframes()
Ckf = np.array([f['C'] for f in kf])
Call = np.array([f['C'] for f in frames])
last_kf = max(f['index'] for f in kf)


def az(v):
    return np.degrees(np.arctan2(v[..., 2], v[..., 0])) % 360


print('=== camera POSITION azimuth about the model centroid, 10-deg bins ===')
a_kf = az(Ckf-cen); a_all = az(Call-cen)
a_tail = az(np.array([f['C'] for f in frames if f['index'] > last_kf])-cen)
print('%-10s %8s %8s %8s' % ('azimuth', 'kf(120)', 'all(868)', 'tail(382)'))
for i in range(36):
    lo, hi = i*10, i*10+10
    k = ((a_kf >= lo) & (a_kf < hi)).sum()
    al = ((a_all >= lo) & (a_all < hi)).sum()
    tl = ((a_tail >= lo) & (a_tail < hi)).sum()
    mark = '   <-- DEAD in the keyframe set' if k == 0 else ''
    print('%-10s %8d %8d %8d%s' % ('%d-%d' % (lo, hi), k, al, tl, mark))

DZ = (140, 200)
print('\n=== the %d-degree dead wedge, azimuth %d-%d about the model centroid ==='
      % (DZ[1]-DZ[0], *DZ))
am = az(mean-cen)
inz = (am >= DZ[0]) & (am < DZ[1])
print('  model splats in the wedge: %d of %d (%.2f%%)' % (inz.sum(), count, 100*inz.mean()))
print('  keyframe cameras in the wedge: %d of 120' % ((a_kf >= DZ[0]) & (a_kf < DZ[1])).sum())
print('  all-capture cameras in the wedge: %d of 868' % ((a_all >= DZ[0]) & (a_all < DZ[1])).sum())
print('  discarded-tail cameras in the wedge: %d of 382'
      % ((a_tail >= DZ[0]) & (a_tail < DZ[1])).sum())

nvA = np.load(os.path.join(OUT, '_cap_nvA.npy')); nvB = np.load(os.path.join(OUT, '_cap_nvB.npy'))
ndA = np.load(os.path.join(OUT, '_cap_ndA.npy')); ndB = np.load(os.path.join(OUT, '_cap_ndB.npy'))
print('\n                          in the wedge      rest of the model')
print('  median views,  set A %10d %20d' % (np.median(nvA[inz]), np.median(nvA[~inz])))
print('  median views,  set B %10d %20d' % (np.median(nvB[inz]), np.median(nvB[~inz])))
print('  median sectors,set A %10d %20d' % (np.median(ndA[inz]), np.median(ndA[~inz])))
print('  median sectors,set B %10d %20d' % (np.median(ndB[inz]), np.median(ndB[~inz])))
print('  never seen, set A    %9.2f%% %19.2f%%' % (100*(nvA[inz] == 0).mean(), 100*(nvA[~inz] == 0).mean()))
print('  never seen, set B    %9.2f%% %19.2f%%' % (100*(nvB[inz] == 0).mean(), 100*(nvB[~inz] == 0).mean()))

scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))
ls = scale.max(axis=1)*1000
print('  median splat size    %8.2f mm %17.2f mm' % (np.median(ls[inz]), np.median(ls[~inz])))
