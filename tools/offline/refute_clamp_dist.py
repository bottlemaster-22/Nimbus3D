"""Independent check of the tangent-clamp finding.

Question: the census recorded peakTileInstances 1,058,989 as a MAX over 3000
iterations of the 108 trained keyframes. The offline UNCLAMPED model says the
mean frame costs 787k and the max 1.38M. If ~50% of every frame's instances
really came from off-axis pathological splats, the census MAX could not be
equal to the CLAMPED max. So: what does the per-frame distribution look like,
and where does the census peak sit in it?
"""
import numpy as np, project as P, sv_clamp as C, json, io, os

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
fx, fy, cx, cy = P.load_intrinsics(720, 540)
tx, ty = 45, 34
CENSUS_PEAK = 1058989

ks = sorted(poses)[::5]
print('sampling %d of %d poses' % (len(ks), len(poses)))
rows = []
for k in ks:
    R = P.quat_to_matrix(poses[k][0]); t = poses[k][1]
    ok_u, ti_u, rad_u, off_u = C.run(col, count, R, t, fx, fy, cx, cy, tx, ty, None)
    ok_c, ti_c, rad_c, off_c = C.run(col, count, R, t, fx, fy, cx, cy, tx, ty, 1.3)
    # how many splats are past the clamp in EITHER axis
    rows.append((k, int(ti_u.sum()), int(ti_c.sum()), int(ok_u.sum()), int(ok_c.sum())))
np.save('_clampdist.npy', np.array(rows, dtype=np.int64))

a = np.array(rows, dtype=np.float64)
u, c = a[:,1], a[:,2]
print()
print('UNCLAMPED per-frame tile instances: min %d p10 %d p50 %d p90 %d max %d mean %d'
      % (u.min(), np.percentile(u,10), np.percentile(u,50), np.percentile(u,90), u.max(), u.mean()))
print('CLAMPED   per-frame tile instances: min %d p10 %d p50 %d p90 %d max %d mean %d'
      % (c.min(), np.percentile(c,10), np.percentile(c,50), np.percentile(c,90), c.max(), c.mean()))
print('mean reduction %.1f%%' % (100*(1-c.mean()/u.mean())))
print()
print('CENSUS PEAK = %d (a MAX over 3000 iterations of 108 keyframes)' % CENSUS_PEAK)
print('  frames whose UNCLAMPED total exceeds the census peak: %d of %d (%.1f%%)'
      % ((u > CENSUS_PEAK).sum(), len(u), 100*(u > CENSUS_PEAK).mean()))
print('  frames whose CLAMPED   total exceeds the census peak: %d of %d (%.1f%%)'
      % ((c > CENSUS_PEAK).sum(), len(c), 100*(c > CENSUS_PEAK).mean()))
print('  census peak as a percentile of the UNCLAMPED distribution: %.1f'
      % (100*(u < CENSUS_PEAK).mean()))
print('  census peak as a percentile of the CLAMPED   distribution: %.1f'
      % (100*(c < CENSUS_PEAK).mean()))
