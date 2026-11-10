"""Is the photometric target the loss is fitting actually registered?

For each frame with an image on disk: render the trained model through that
frame's refined pose, then find the 2D shift that best aligns the render's luma
with the photograph's luma.  A shift of zero on every frame means the poses
agree with the model.  A shift that is LARGE and DIFFERENT PER FRAME means the
photometric loss is being handed a target that moves, and no amount of
optimisation can satisfy two views at once.

A constant shift across all frames would be an intrinsics or convention error
instead, so both the per-frame shift and its spread are reported.

Run: python -u lossq_align.py
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B

LUMA = np.array([0.2126, 0.7152, 0.0722])
MAXSHIFT = 40


def best_shift(X, Y, maxshift=MAXSHIFT):
    """Integer (dy, dx) maximising normalised cross-correlation of gradient
    magnitude, by FFT.  Gradient magnitude rather than intensity so a global
    exposure difference cannot bias it."""
    def gm(a):
        gy, gx = np.gradient(a)
        g = np.sqrt(gx * gx + gy * gy)
        g = g - g.mean()
        s = g.std()
        return g / (s if s > 1e-12 else 1.0)
    a, b = gm(X), gm(Y)
    F = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    c = np.fft.irfft2(F, s=a.shape)
    h, w = a.shape
    # search only |shift| <= maxshift, in wrapped coordinates
    ys = np.concatenate([np.arange(0, maxshift + 1), np.arange(h - maxshift, h)])
    xs = np.concatenate([np.arange(0, maxshift + 1), np.arange(w - maxshift, w)])
    sub = c[np.ix_(ys, xs)]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    dy = ys[k[0]]
    dx = xs[k[1]]
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    peak = sub.max() / (h * w)
    zero = c[0, 0] / (h * w)
    return int(dy), int(dx), float(peak), float(zero)


def shift_img(img, dy, dx):
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out, (ys0, ys1, xs0, xs1)


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(m, 1e-12))


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    use = sorted(frames)

    pre = json.load(io.open(os.path.join(P.D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
    print('prepass pose graph, from prepass/census.json:')
    pc = json.load(io.open(os.path.join(P.D, 'prepass', 'census.json'), encoding='utf-8'))['poseGraph']
    for k in ('converged', 'exitReason', 'iterationsRun', 'finalResidualMedianCentimeters',
              'finalResidualMedianDegrees', 'medianPoseShiftCentimeters',
              'maxPoseShiftCentimeters', 'usableEdges', 'framesPosed'):
        print('  %-34s %s' % (k, pc[k]))
    print('  fx at the render size = %.2f px, so 1 degree of pose error = %.2f px'
          % (fx, fx * np.tan(np.deg2rad(1.0))))
    print('  %.4f deg (the graph median) = %.2f px at the render size'
          % (pc['finalResidualMedianDegrees'],
             fx * np.tan(np.deg2rad(pc['finalResidualMedianDegrees']))))
    print('  %.4f cm (the graph median) at 2 m range = %.2f px'
          % (pc['finalResidualMedianCentimeters'],
             fx * (pc['finalResidualMedianCentimeters'] / 100.0) / 2.0))

    print('\n%5s %7s %7s %9s %11s %11s %9s' %
          ('frame', 'dy px', 'dx px', '|shift|', 'PSNR at 0', 'PSNR shifted', 'gain dB'))
    rows = []
    for f in use:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        img, Tf, ok, g, inst = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
        _, gt = Dt.luma_of(frames[f], rw, rh)
        gt = gt.astype(np.float64)
        X = (img * LUMA).sum(axis=2)
        Y = (gt * LUMA).sum(axis=2)
        dy, dx, peak, zero = best_shift(X, Y)
        sh, (y0, y1, x0, x1) = shift_img(img, dy, dx)
        a0, b0 = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        gn, bs = B.fit_exposure(a0, b0)
        p0 = psnr(np.clip(gn * a0 + bs, 0, 1), b0)
        a1 = sh[y0:y1, x0:x1]
        gn2, bs2 = B.fit_exposure(a1, b0)
        p1 = psnr(np.clip(gn2 * a1 + bs2, 0, 1), b0)
        rows.append((f, dy, dx, p0, p1))
        print('%5d %7d %7d %9.2f %11.3f %11.3f %9.3f'
              % (f, dy, dx, np.hypot(dy, dx), p0, p1, p1 - p0))

    a = np.array([(r[1], r[2]) for r in rows], float)
    gains = np.array([r[4] - r[3] for r in rows])
    print('\n  shift dy: mean %+.2f  sd %.2f   dx: mean %+.2f  sd %.2f'
          % (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std()))
    print('  |shift| median %.2f px, max %.2f px' %
          (np.median(np.hypot(a[:, 0], a[:, 1])), np.hypot(a[:, 0], a[:, 1]).max()))
    print('  PSNR bought by a per-frame integer translation: mean %+.3f dB, max %+.3f dB'
          % (gains.mean(), gains.max()))
    print('  (a per-frame shift that is not the same for every frame is pose error,')
    print('   not an intrinsics or convention error, which would be constant)')


if __name__ == '__main__':
    main()
