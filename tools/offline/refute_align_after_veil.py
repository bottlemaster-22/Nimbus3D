"""INTERACTION TEST: lossq_align concluded "pose error is NOT the ceiling" from
a +0.326 dB mean gain for the best integer 2D shift.  But that shift search was
run against a residual DOMINATED by the near-field veil (measured: deleting the
splats nearer than 0.50 m is worth +1.835 dB mean on the same 16 frames).  A
correlator cannot see a 2 px misregistration through a full-frame colour cast.
Re-run the same search on the veil-removed render.
"""
import io, json, os
import numpy as np
import project as P, detail as Dt, lossq_blur as B
from lossq_align import best_shift, shift_img

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]; rw, rh = sl['renderWidth'], sl['renderHeight']
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses(); frames = dict(Dt.frames_with_images())
LUMA = np.array([0.2126, 0.7152, 0.0722])

print('%5s | %20s | %20s' % ('frame', 'FULL MODEL', 'VEIL (<0.5 m) REMOVED'))
print('%5s | %5s %5s %8s | %5s %5s %8s' % ('', 'dy', 'dx', 'gain dB', 'dy', 'dx', 'gain dB'))
A, Bb = [], []
for f in sorted(frames):
    rot, t = poses[f]; R = P.quat_to_matrix(rot)
    img, Tf, ok, g, _ = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    _, gt = Dt.luma_of(frames[f], rw, rh); gt = gt.astype(np.float64)
    keep = ~(ok & (g['z'] < 0.5))
    sub = {k: v[keep] for k, v in col.items()}
    img2, _, _, _, _ = B.render(sub, int(keep.sum()), R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    out = []
    for im in (img, img2):
        X = (im * LUMA).sum(2); Y = (gt * LUMA).sum(2)
        dy, dx, _, _ = best_shift(X, Y)
        ga, ba = B.fit_exposure(im, gt); p0 = B.psnr(ga*im+ba, gt)
        sh, (y0, y1, x0, x1) = shift_img(im, dy, dx)
        cut = (slice(y0, y1), slice(x0, x1))
        gs, bs = B.fit_exposure(sh[cut], gt[cut])
        p1 = B.psnr(gs*sh[cut]+bs, gt[cut])
        p0m = B.psnr((ga*im+ba)[cut], gt[cut])
        out.append((dy, dx, p1-p0m))
    A.append(out[0][2]); Bb.append(out[1][2])
    print('%5d | %5d %5d %8.3f | %5d %5d %8.3f'
          % (f, out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2]))
print('\n  PSNR bought by the best integer translation:')
print('    on the full model        mean %+.3f dB  max %+.3f' % (np.mean(A), np.max(A)))
print('    with the near veil gone  mean %+.3f dB  max %+.3f' % (np.mean(Bb), np.max(Bb)))
