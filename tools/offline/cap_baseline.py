"""Camera baselines and viewing-direction spread over the 120 selected keyframes."""
import numpy as np
from cap_keyframes import keyframes, split_held_out

b, r, frames, pool, kf, info = keyframes()
C = np.array([f['C'] for f in kf])          # 120 x 3 camera centres
F = np.array([f['F'] for f in kf])
F = F / np.linalg.norm(F, axis=1, keepdims=True)
n = len(kf)

# ---- pairwise baselines --------------------------------------------------
d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
iu = np.triu_indices(n, 1)
pw = d[iu]
print('=== BASELINES between the %d keyframe camera centres (%d pairs) ===' % (n, pw.size))
for q in (0, 5, 10, 25, 50, 75, 90, 95, 100):
    print('  p%-3d %7.3f m' % (q, np.percentile(pw, q)))
print('  mean %7.3f m' % pw.mean())

# consecutive keyframe baseline (the stride the greedy selector actually walked)
cons = np.linalg.norm(np.diff(C, axis=0), axis=1)
print('\n=== consecutive-keyframe baseline ===')
for q in (10, 50, 90):
    print('  p%-3d %7.3f m' % (q, np.percentile(cons, q)))
print('  min %.4f  max %.4f  mean %.4f m' % (cons.min(), cons.max(), cons.mean()))

# ---- viewing direction spread -------------------------------------------
cosang = np.clip(F @ F.T, -1, 1)
ang = np.degrees(np.arccos(cosang))[iu]
print('\n=== ANGLE between keyframe optical axes (all pairs) ===')
for q in (5, 10, 25, 50, 75, 90, 95, 100):
    print('  p%-3d %7.2f deg' % (q, np.percentile(ang, q)))
consang = np.degrees(np.arccos(np.clip(np.sum(F[:-1]*F[1:], axis=1), -1, 1)))
print('  consecutive: p10 %.2f  p50 %.2f  p90 %.2f  max %.2f deg'
      % tuple(np.percentile(consang, [10, 50, 90]).tolist() + [consang.max()]))

# ---- parallax at the typical stand-off ----------------------------------
STANDOFF = 1.29
# Naive geometric parallax for a point at STANDOFF straight ahead of BOTH
# cameras: angle subtended at the point by the baseline.
par = np.degrees(2*np.arcsin(np.clip(pw/(2*STANDOFF), 0, 1)))
print('\n=== NAIVE parallax at %.2f m stand-off, all keyframe pairs ===' % STANDOFF)
for q in (5, 10, 25, 50, 75, 90, 95):
    print('  p%-3d %7.2f deg' % (q, np.percentile(par, q)))
parc = np.degrees(2*np.arcsin(np.clip(cons/(2*STANDOFF), 0, 1)))
print('  consecutive keyframes: median %.2f deg, mean %.2f deg'
      % (np.median(parc), parc.mean()))

# ---- how much of the room did the rig actually orbit? -------------------
cen = C.mean(axis=0)
print('\n=== camera path extent ===')
print('  centroid  %s' % np.round(cen, 3))
print('  bbox min  %s' % np.round(C.min(axis=0), 3))
print('  bbox max  %s' % np.round(C.max(axis=0), 3))
print('  bbox size %s m' % np.round(C.max(axis=0)-C.min(axis=0), 3))
print('  radius from centroid: p50 %.3f  p90 %.3f  max %.3f m'
      % tuple(np.percentile(np.linalg.norm(C-cen, axis=1), [50, 90, 100])))

# ---- the discarded tail --------------------------------------------------
last_kf = kf[-1]['index']
tail = [f for f in frames if f['index'] > last_kf]
Ct = np.array([f['C'] for f in tail])
print('\n=== the %d frames AFTER the last keyframe (index %d) ==='
      % (len(tail), last_kf))
print('  tail bbox size %s m' % np.round(Ct.max(axis=0)-Ct.min(axis=0), 3))
# how far is each tail camera from the NEAREST keyframe camera?
dt = np.linalg.norm(Ct[:, None, :] - C[None, :, :], axis=2).min(axis=1)
print('  distance from each tail camera to the nearest keyframe camera:')
for q in (50, 75, 90, 95, 100):
    print('    p%-3d %6.3f m' % (q, np.percentile(dt, q)))
print('  tail cameras more than 0.33 m (one keyframe spacing) from any keyframe: %d (%.1f%%)'
      % ((dt > info['spacing']).sum(), 100*(dt > info['spacing']).mean()))
