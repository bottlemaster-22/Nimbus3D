"""Find the azimuth dead zone in the pose data, and measure how much of the
model sits in it.

World frame is ARKit's (Y up); Pose.center/forward are in it. Azimuth is
atan2(z, x) about the vertical axis.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes, split_held_out

OUT = os.path.dirname(os.path.abspath(__file__))
b, r, frames, pool, kf, info = keyframes()
kfidx = set(f['index'] for f in kf)
last_kf = max(kfidx)

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
cen = mean.mean(axis=0)
print('model centroid %s   splats %d' % (np.round(cen, 3), count))
print('model bbox %s .. %s' % (np.round(mean.min(0), 2), np.round(mean.max(0), 2)))
Ckf = np.array([f['C'] for f in kf])
print('keyframe camera centroid %s' % np.round(Ckf.mean(0), 3))


def az(v):
    return np.degrees(np.arctan2(v[..., 2], v[..., 0])) % 360


def biggest_gap(angles, label, binw=2.0):
    """Largest run of empty 2-degree bins, wrapping."""
    nb = int(360/binw)
    h, _ = np.histogram(angles % 360, bins=nb, range=(0, 360))
    occ = h > 0
    if occ.all():
        print('  %-46s no empty bin; min bin count %d' % (label, h.min()))
        return None
    d = np.concatenate([occ, occ])
    best = bestat = 0
    run = 0
    for i, o in enumerate(d):
        if not o:
            run += 1
            if run > best:
                best, bestat = run, i-run+1
        else:
            run = 0
    best = min(best, nb)
    lo = (bestat % nb)*binw
    print('  %-46s biggest empty wedge %5.1f deg, from %5.1f to %5.1f'
          % (label, best*binw, lo, (lo+best*binw) % 360))
    return (lo, best*binw)


print('\n=== A. azimuth of the camera OPTICAL AXIS ===')
F_all = np.array([f['F'] for f in frames]); a_all = az(F_all)
F_kf = np.array([f['F'] for f in kf]); a_kf = az(F_kf)
tail = [f for f in frames if f['index'] > last_kf]
F_t = np.array([f['F'] for f in tail]); a_t = az(F_t)
biggest_gap(a_all, 'all 868 frames')
biggest_gap(a_kf, 'the 120 trainer keyframes')
biggest_gap(a_t, 'the %d frames the trainer discarded' % len(tail))
for lo in range(0, 360, 30):
    hi = lo+30
    print('    az %3d-%3d : all %4d   keyframes %3d   discarded tail %4d'
          % (lo, hi, ((a_all >= lo) & (a_all < hi)).sum(),
             ((a_kf >= lo) & (a_kf < hi)).sum(), ((a_t >= lo) & (a_t < hi)).sum()))

print('\n=== B. azimuth of the camera POSITION about the model centroid ===')
Call = np.array([f['C'] for f in frames])
biggest_gap(az(Call-cen), 'all 868 frames')
biggest_gap(az(Ckf-cen), 'the 120 trainer keyframes')

print('\n=== C. per-splat azimuth coverage (occlusion aware) ===')
viso = np.unpackbits(np.load(os.path.join(OUT, '_cap_viso.npy')), axis=1)[:, :len(kf)].astype(bool)
# azimuth of splat->camera direction, per keyframe
azsc = np.zeros((count, len(kf)), np.float32)
for j, f in enumerate(kf):
    v = f['C'] - mean
    azsc[:, j] = az(v)
BW = 10.0
nb = int(360/BW)
bins = (azsc/BW).astype(np.int16)
cover = np.zeros((count, nb), bool)
for j in range(len(kf)):
    m = viso[:, j]
    cover[np.flatnonzero(m), bins[m, j]] = True
ndir = cover.sum(axis=1)
print('  distinct 10-deg azimuth sectors each splat is seen from:')
for q in (5, 10, 25, 50, 75, 90):
    print('    p%-3d %2d sectors (%5.1f deg of azimuth)' % (q, np.percentile(ndir, q),
                                                            BW*np.percentile(ndir, q)))
print('  splats seen from only ONE 10-deg sector: %6d (%.2f%%)'
      % ((ndir <= 1).sum(), 100*(ndir <= 1).mean()))
print('  splats seen from <= 3 sectors (<=30 deg): %6d (%.2f%%)'
      % ((ndir <= 3).sum(), 100*(ndir <= 3).mean()))
# widest contiguous EMPTY azimuth wedge per splat
pad = np.concatenate([cover, cover], axis=1)
run = np.zeros(count, np.int16); best = np.zeros(count, np.int16)
for k in range(2*nb):
    e = ~pad[:, k]
    run = np.where(e, run+1, 0)
    best = np.maximum(best, run)
best = np.minimum(best, nb)
print('  widest UNSEEN azimuth wedge per splat: p10 %.0f  p50 %.0f  p90 %.0f deg'
      % tuple(BW*np.percentile(best, [10, 50, 90])))
np.save(os.path.join(OUT, '_cap_azsectors.npy'), ndir.astype(np.int16))

print('\n=== D. where the model sits, by azimuth about the camera path ===')
ccam = Ckf.mean(0)
am = az(mean-ccam)
h, _ = np.histogram(am, bins=36, range=(0, 360))
for i in range(36):
    print('    az %3d-%3d : %7d splats (%5.2f%%)' % (i*10, i*10+10, h[i], 100*h[i]/count))
