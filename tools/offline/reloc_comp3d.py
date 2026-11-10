"""Replay trainer_filter3d_sweep + finalize over the real poses to recover
filter3D per splat, then invert the export fuse to get the TRAINER-SIDE
opacity the donor test actually compares against."""
import json, numpy as np, project as PJ

col, n = PJ.load_ply(PJ.D + r'\model\model.ply')
poses = PJ.load_poses()
cen = json.load(open(PJ.D + r'\model\train_census.json', encoding='utf-8'))
RW = 720
b = json.load(open(PJ.D + r'\capture_bundle.json', encoding='utf-8'))
k = b['intrinsics']
if k['width'] >= k['height']:
    rw, rh = RW, int(round(RW * k['height'] / k['width']))
else:
    rh, rw = RW, int(round(RW * k['width'] / k['height']))
fx, fy, cx, cy = PJ.load_intrinsics(rw, rh)
print('render %dx%d  fx %.2f fy %.2f  frames %d' % (rw, rh, fx, fy, len(poses)))

mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
FS, FB = 0.2, 0.01
rate_top = np.zeros((n, 4))         # r0..r3, descending
for idx in sorted(poses):
    q, t = poses[idx]
    R = PJ.quat_to_matrix(q)
    cam = mean @ R.T + t
    z = cam[:, 2]
    ok = z > 0.2
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy
    ok &= (u >= 0) & (u < rw) & (v >= 0) & (v < rh)
    r = np.where(ok, max(fx, fy) / np.maximum(z, 1e-4), 0.0)
    # insert into the descending top-4, exactly as the kernel does
    for j in range(4):
        hit = r > rate_top[:, j]
        if not hit.any():
            continue
        rate_top[hit, j + 1:] = rate_top[hit, j:3]
        rate_top[hit, j] = r[hit]
        r = np.where(hit, 0.0, r)

rate = rate_top[:, 3].copy()
for j in [2, 1, 0]:
    rate = np.where(rate <= 0, rate_top[:, j], rate)
f3 = np.where(rate > 0, FS / np.maximum(rate, 1e-30), FB)
print()
print('filter3D mm : ' + '  '.join('p%-3g %6.3f' % (p, np.percentile(f3, p)*1000)
                                   for p in [1, 10, 50, 90, 99]))
print('splats never seen by any keyframe (fallback 10 mm): %d (%.3f%%)'
      % ((rate <= 0).sum(), 100*(rate <= 0).sum()/n))

sig_ply = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'],
                                   col['scale_2']], axis=1).astype(np.float64), -12, 3))
# sigma_ply = sqrt(sigma_train^2 + f^2)  ->  invert
v = sig_ply**2 - (f3**2)[:, None]
invertible = (v > 0).all(axis=1)
sig_tr = np.sqrt(np.maximum(v, 1e-24))
comp = np.prod(sig_tr / sig_ply, axis=1)
print()
print('splats where every PLY axis exceeds its own filter width : %d (%.2f%%)'
      % (invertible.sum(), 100*invertible.mean()))
print('  the rest were never fused (fusing3DFilter returns unchanged when the')
print('  compensation is not < 1 or not finite), so comp3D = 1 for them.')
comp = np.where(invertible, comp, 1.0)
print('comp3D      : ' + '  '.join('p%-3g %6.4f' % (p, np.percentile(comp, p))
                                   for p in [1, 10, 50, 90, 99]))

alpha_drawn = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
alpha_train = np.clip(alpha_drawn / np.maximum(comp, 1e-6), 0, 0.999999)
print()
print('=== THE DONOR POOL, IN THE DONOR TEST\'S OWN CONVENTION ===')
for nm, a, thr in [('drawn (what the PRUNE sees)  < 0.02', alpha_drawn, 0.02),
                   ('trainer (what the DONOR sees) < 0.05', alpha_train, 0.05)]:
    print('  %-38s %7d  (%.3f%%)' % (nm, (a < thr).sum(), 100*(a < thr).mean()))
band = (alpha_train >= 0.02) & (alpha_train < 0.05)
print()
print('  donors by opacity, trainer convention : %7d  (%.3f%%)'
      % ((alpha_train < 0.05).sum(), 100*(alpha_train < 0.05).mean()))
print('  maxRelocationFractionPerPass allowance: %7d  (5%% of %d)'
      % (int(n*0.05), n))
print('  build 182 measured 22,005 (7.35%%) in the band.')
print()
print('=== THE INVARIANT, CHECKED IN THE UNITS THAT ACTUALLY APPLY ===')
print('The comment on relocationDonorOpacity says the donor pool IS the band')
print('between pruneOpacity 0.02 and donorOpacity 0.05. But the prune tests a')
print('COMPENSATED opacity and the donor test tests a RAW one, so in')
print('trainer-side units the prune line sits at 0.02 / comp3D:')
eff = 0.02 / np.maximum(comp, 1e-6)
print('  effective prune line, trainer units : ' + '  '.join(
    'p%-3g %6.4f' % (p, np.percentile(eff, p)) for p in [10, 50, 90, 99]))
above = (eff > 0.05)
print('  splats whose prune line is ABOVE the donor line: %d (%.2f%%)'
      % (above.sum(), 100*above.mean()))
print('  -> for those the band is INVERTED: they are deleted at step 4 of the')
print('     same pass at an opacity where they were never eligible donors.')
d = (alpha_train < 0.05)
print()
print('=== DONOR SIZE, using the recovered pool ===')
s = np.sort(sig_ply, axis=1)*1000
print('  donors p50 largest axis %.3f mm  | rest %.3f mm  | ratio %.2fx'
      % (np.median(s[d, 2]), np.median(s[~d, 2]),
         np.median(s[d, 2])/np.median(s[~d, 2])))
np.save('_reloc_donor_mask.npy', d)
np.save('_reloc_comp3d.npy', comp)
