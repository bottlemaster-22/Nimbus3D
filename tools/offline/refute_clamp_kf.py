"""Decisive test for the tangent-clamp finding.

The census recorded peakTileInstances = 1,058,989 as a MAX over 3000 iterations,
at splatCount 299,352 -- i.e. between iterations 2800 and 2900, when the model
was within 143 splats of the exported model.ply. So the population confound is
dead: the device measured essentially THIS model.

If trainer_preprocess really is unclamped, the device's max over its ~108
trained keyframes should equal the max of the UNCLAMPED offline distribution
over those keyframes. Held-out frame indices run 17..460, so the keyframes
live in the pose range 0..~470. Sample that range densely and ask: how many
frames exceed the census peak, unclamped and clamped?
"""
import numpy as np, project as P, sv_clamp as C, os

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
fx, fy, cx, cy = P.load_intrinsics(720, 540)
tx, ty = 45, 34
CENSUS_PEAK = 1058989

ks = [k for k in sorted(poses) if 0 <= k <= 470][::4]
print('keyframe-range sample: %d frames in pose range 0..470 (census: 120 selected, 108 trained)' % len(ks))
u, c, pathn, patht = [], [], [], []
for k in ks:
    R = P.quat_to_matrix(poses[k][0]); t = poses[k][1]
    ok_u, ti_u, rad_u, off_u = C.run(col, count, R, t, fx, fy, cx, cy, tx, ty, None)
    ok_c, ti_c, _, _ = C.run(col, count, R, t, fx, fy, cx, cy, tx, ty, 1.3)
    m = ok_u & (off_u > 1.3)
    u.append(ti_u.sum()); c.append(ti_c.sum())
    pathn.append(m.sum()); patht.append(ti_u[m].sum())
u = np.array(u, float); c = np.array(c, float)
pathn = np.array(pathn, float); patht = np.array(patht, float)

print()
print('UNCLAMPED: p50 %d  p90 %d  MAX %d  mean %d' % (np.percentile(u,50), np.percentile(u,90), u.max(), u.mean()))
print('CLAMPED  : p50 %d  p90 %d  MAX %d  mean %d' % (np.percentile(c,50), np.percentile(c,90), c.max(), c.mean()))
print('mean reduction %.1f%%   MAX reduction %.1f%%' % (100*(1-c.mean()/u.mean()), 100*(1-c.max()/u.max())))
print()
print('CENSUS PEAK %d  (device MAX, recorded at iter ~2850, 299,352 splats)' % CENSUS_PEAK)
print('  UNCLAMPED offline MAX / census peak = %.3fx   (%d frames of %d exceed the census peak)'
      % (u.max()/CENSUS_PEAK, (u>CENSUS_PEAK).sum(), len(u)))
print('  CLAMPED   offline MAX / census peak = %.3fx   (%d frames of %d exceed the census peak)'
      % (c.max()/CENSUS_PEAK, (c>CENSUS_PEAK).sum(), len(c)))
pe = (u > CENSUS_PEAK).mean()
if pe > 0:
    print('  P(max over 108 unclamped keyframes <= census peak) = (1-%.3f)^108 = %.3g' % (pe, (1-pe)**108))
print()
print('pathological population (|x/z| > 1.3 x half-FOV tangent), per frame:')
print('  count : p10 %.0f p50 %.0f p90 %.0f max %.0f' % tuple(np.percentile(pathn,[10,50,90,100])))
print('  share of that frame tile instances: p10 %.1f%% p50 %.1f%% p90 %.1f%%'
      % tuple(100*np.percentile(patht/u,[10,50,90])))
print('  frames where it is under 5%% of instances: %d of %d' % (((patht/u)<0.05).sum(), len(u)))
