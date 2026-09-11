"""A27 re-measured with controls.  lossq_align.py picks the best integer shift
on the WHOLE image and scores it on the same pixels, which can only flatter
the gain.  This version:

  1. reproduces lossq_align's whole-image number (same renderer, clamp on);
  2. CROSS-VALIDATES it: the shift is chosen on one half of the image and the
     PSNR gain is scored on the OTHER half (left/right and top/bottom);
  3. measures the shift separately in the four quadrants: a pure small
     rotation moves every quadrant alike, a translation error at mixed depth
     or a lens/rolling-shutter error does not;
  4. refines the peak to sub-pixel and scores a Fourier sub-pixel shift;
  5. POSITIVE CONTROL: shifts the photo by a known (+3, -2) and checks the
     search recovers it.

Renders are cached (float32) in the scratchpad, not under tools/offline.
Run: python -u lossq_align_cv.py
"""
import io, json, os
import numpy as np
import project as P
import detail as Dt
import lossq_blur as B
from lossq_align import best_shift, shift_img, LUMA

CACHE = os.path.join(os.environ.get('LIKOVA_SCRATCH', '.'), 'align_renders_clamped.npz')


def luma(a):
    return (a * LUMA).sum(axis=2)


def expo_psnr(a, gt):
    gn, bs = B.fit_exposure(a, gt)
    return B.psnr(np.clip(gn * a + bs, 0, 1), gt)


def gain_in(img, gt, dy, dx, rows, cols):
    """PSNR gain of shifting the WHOLE render by (dy, dx), scored only on the
    given rows/cols window, intersected with the pixels the shift keeps."""
    sh, (y0, y1, x0, x1) = shift_img(img, dy, dx)
    rs = slice(max(y0, rows.start), min(y1, rows.stop))
    cs = slice(max(x0, cols.start), min(x1, cols.stop))
    return expo_psnr(sh[rs, cs], gt[rs, cs]) - expo_psnr(img[rs, cs], gt[rs, cs])


def subpixel_peak(X, Y, dy, dx):
    def gm(a):
        gy, gx = np.gradient(a)
        g = np.sqrt(gx * gx + gy * gy)
        g = g - g.mean()
        return g / max(g.std(), 1e-12)
    a, b = gm(X), gm(Y)
    c = np.fft.irfft2(np.fft.rfft2(b) * np.conj(np.fft.rfft2(a)), s=a.shape)
    h, w = a.shape

    def at(y, x):
        return c[y % h, x % w]

    def para(m, z, p):
        den = m - 2 * z + p
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (m - p) / den, -0.5, 0.5))
    fy = para(at(dy - 1, dx), at(dy, dx), at(dy + 1, dx))
    fx = para(at(dy, dx - 1), at(dy, dx), at(dy, dx + 1))
    return dy + fy, dx + fx


def fourier_shift(img, dy, dx):
    h, w = img.shape[:2]
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    ramp = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    out = np.empty_like(img)
    for ch in range(img.shape[2]):
        out[..., ch] = np.real(np.fft.ifft2(np.fft.fft2(img[..., ch]) * ramp))
    return out


def renders():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return list(z['frames']), z['imgs'].astype(np.float64), z['gts'].astype(np.float64), float(z['fx'])
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    fr = dict(Dt.frames_with_images())
    use = sorted(fr)
    imgs, gts = [], []
    for f in use:
        rot, t = poses[f]
        img, _, _, _, _ = B.render(col, count, P.quat_to_matrix(rot), t, fx, fy, cx, cy, rw, rh, 0.25, 4)
        _, gt = Dt.luma_of(fr[f], rw, rh)
        imgs.append(img.astype(np.float32))
        gts.append(gt.astype(np.float32))
        print('  rendered frame %d' % f, flush=True)
    np.savez(CACHE, frames=np.array(use), imgs=np.array(imgs), gts=np.array(gts), fx=fx)
    return use, np.array(imgs, np.float64), np.array(gts, np.float64), fx


def main():
    use, imgs, gts, fx = renders()
    h, w = imgs.shape[1:3]
    H, W = slice(0, h), slice(0, w)
    L, R = slice(0, w // 2), slice(w // 2, w)
    T, Bo = slice(0, h // 2), slice(h // 2, h)
    print('%5s %9s %8s | %8s %8s | %22s %6s | %13s %8s | %s'
          % ('frame', 'shift', 'whole', 'CV L/R', 'CV T/B', 'quadrant shifts', 'spread',
             'subpixel', 'sub gain', 'control'))
    whole, cvlr, cvtb, subg, spreads, mags, ok = [], [], [], [], [], [], 0
    for k, f in enumerate(use):
        img, gt = imgs[k], gts[k]
        X, Y = luma(img), luma(gt)
        dy, dx, _, _ = best_shift(X, Y)
        g_whole = gain_in(img, gt, dy, dx, H, W)
        # cross-validation: choose on one half, score on the other
        l = best_shift(X[:, L], Y[:, L])[:2]
        r = best_shift(X[:, R], Y[:, R])[:2]
        g_lr = 0.5 * (gain_in(img, gt, *l, H, R) + gain_in(img, gt, *r, H, L))
        t = best_shift(X[T, :], Y[T, :])[:2]
        bb = best_shift(X[Bo, :], Y[Bo, :])[:2]
        g_tb = 0.5 * (gain_in(img, gt, *t, Bo, W) + gain_in(img, gt, *bb, T, W))
        quads = [best_shift(X[rs, cs], Y[rs, cs], 20)[:2] for rs in (T, Bo) for cs in (L, R)]
        qa = np.array(quads, float)
        spread = max(np.hypot(*(qa[i] - qa[j])) for i in range(4) for j in range(4))
        sy, sx = subpixel_peak(X, Y, dy, dx)
        m = int(np.ceil(max(abs(sy), abs(sx)))) + 2
        crop = (slice(m, h - m), slice(m, w - m))
        g_sub = expo_psnr(fourier_shift(img, sy, sx)[crop], gt[crop]) - expo_psnr(img[crop], gt[crop])
        gt_moved, _ = shift_img(gt, 3, -2)
        cy_, cx_ = best_shift(X, luma(gt_moved))[:2]
        good = (cy_, cx_) == (dy + 3, dx - 2)
        ok += good
        whole.append(g_whole); cvlr.append(g_lr); cvtb.append(g_tb); subg.append(g_sub)
        spreads.append(spread); mags.append(np.hypot(sy, sx))
        print('%5d %4d,%3d %8.3f | %8.3f %8.3f | %22s %6.1f | %6.2f,%6.2f %8.3f | %s'
              % (f, dy, dx, g_whole, g_lr, g_tb,
                 ' '.join('%d,%d' % q for q in quads), spread, sy, sx, g_sub,
                 'recovered' if good else 'MISSED (%d,%d)' % (cy_, cx_)))
    print('\n  whole-image gain (as lossq_align)   mean %+.3f dB  median %+.3f' % (np.mean(whole), np.median(whole)))
    print('  cross-validated gain, left/right    mean %+.3f dB  median %+.3f' % (np.mean(cvlr), np.median(cvlr)))
    print('  cross-validated gain, top/bottom    mean %+.3f dB  median %+.3f' % (np.mean(cvtb), np.median(cvtb)))
    print('  sub-pixel Fourier shift gain        mean %+.3f dB  median %+.3f' % (np.mean(subg), np.median(subg)))
    print('  sub-pixel |shift| median %.2f px = %.3f deg of rotation at fx %.1f'
          % (np.median(mags), np.degrees(np.arctan(np.median(mags) / fx)), fx))
    print('  quadrant spread median %.1f px (0 = one rigid shift explains the frame)' % np.median(spreads))
    print('  positive control recovered on %d of %d frames' % (ok, len(use)))


if __name__ == '__main__':
    main()
