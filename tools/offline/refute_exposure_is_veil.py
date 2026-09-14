"""INTERACTION TEST: is the "exposure" the least-squares fit wants actually the
NEAR VEIL?  The capture bundle says the true per-frame exposure ratio over these
16 frames is exactly 1.0000 (identical exposureDurationSeconds, exposureOffsetEV
0 on all 868, qc.exposureJumpEV max 0.0000), so every departure from gain 1.0 /
bias 0.0 is render error.  If the near-field Gaussians are what the fit is
chasing, deleting them should collapse the fitted gain toward 1.
"""
import io, json, os
import numpy as np
import project as P, detail as Dt, lossq_blur as B

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]; rw, rh = sl['renderWidth'], sl['renderHeight']
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses(); frames = dict(Dt.frames_with_images())

def raw_fit(x, y):
    x = x.ravel(); y = y.ravel(); n = float(x.size)
    den = n*(x*x).sum() - x.sum()**2
    g = (n*(x*y).sum() - x.sum()*y.sum())/den
    return g, (y.sum() - g*x.sum())/n

rows = []
for f in sorted(frames):
    rot, t = poses[f]; R = P.quat_to_matrix(rot)
    img, Tf, ok, g, _ = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    _, gt = Dt.luma_of(frames[f], rw, rh); gt = gt.astype(np.float64)
    ga, ba = raw_fit(img, gt)
    gc, bc = B.fit_exposure(img, gt); p1 = B.psnr(gc*img+bc, gt)
    keep = ~(ok & (g['z'] < 0.5))
    sub = {k: v[keep] for k, v in col.items()}
    img2, _, _, _, _ = B.render(sub, int(keep.sum()), R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    ga2, ba2 = raw_fit(img2, gt)
    gc2, bc2 = B.fit_exposure(img2, gt); p2 = B.psnr(gc2*img2+bc2, gt)
    outside1 = not (0.9 <= ga <= 1.1 and -0.05 <= ba <= 0.05)
    outside2 = not (0.9 <= ga2 <= 1.1 and -0.05 <= ba2 <= 0.05)
    n_near = int((ok & (g['z'] < 0.5)).sum())
    rows.append((f, n_near, ga, ba, outside1, ga2, ba2, outside2, p1, p2))
    print('  frame %2d  near<0.5m %6d   FULL gain %7.4f bias %+7.4f %-8s'
          '  VEIL REMOVED gain %7.4f bias %+7.4f %-8s   PSNR %6.3f -> %6.3f'
          % (f, n_near, ga, ba, 'OUTSIDE' if outside1 else 'inside',
             ga2, ba2, 'OUTSIDE' if outside2 else 'inside', p1, p2))

a = np.array([[r[2], r[3], r[5], r[6], r[8], r[9]] for r in rows])
print('\n  frames wanting a gain/bias OUTSIDE the clamp box:')
print('    with the near veil    : %d of 16' % sum(r[4] for r in rows))
print('    with it removed       : %d of 16' % sum(r[7] for r in rows))
print('  |gain - 1| mean: %.4f -> %.4f   |bias| mean: %.4f -> %.4f'
      % (np.abs(a[:,0]-1).mean(), np.abs(a[:,2]-1).mean(),
         np.abs(a[:,1]).mean(), np.abs(a[:,3]).mean()))
print('  exposure-fitted PSNR mean: %.3f -> %.3f dB  (%+.3f)'
      % (a[:,4].mean(), a[:,5].mean(), a[:,5].mean()-a[:,4].mean()))
