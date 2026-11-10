"""THE SHAPE GAP, MEASURED.

1. Effective-rank distribution of ours vs Scaniverse, and what target each
   population is actually sitting at.
2. What the disc prior DOES to the smallest axis at each candidate target.
3. Re-render a real frame with the model isotropised at CONSTANT VOLUME
   (which is exactly what the prior's gradient conserves: it sums to zero
   across the three log-scale components) and score PSNR + trainer SSIM
   against the real photograph.

Run: python -u shape_iso.py [frames...]
"""
import io, json, os, sys
import numpy as np

import project as P
import raster as R
import showme as S
import lossq_ssim as Q

SV = os.path.join(
    r"C:\Users\Undea\AppData\Local\Temp\claude"
    r"\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f"
    r"\scratchpad", 'sv.npz')


def rank_of(scale):
    lam = scale * scale
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-300)
    lp = np.log(np.maximum(p, 1e-300))
    H = -(p * lp).sum(1)
    return np.exp(H), p, lp, H


def main():
    frames = [int(a) for a in sys.argv[1:]] or [4]
    col0, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    col = {k: np.array(v, dtype=np.float64) for k, v in col0.items()}
    logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1), -12, 3)
    scale = np.exp(logs)
    rank, p, lp, H = rank_of(scale)

    print('=== 1. EFFECTIVE RANK, OURS vs SCANIVERSE ===')
    print('ours   n=%d' % count)
    print('  rank  p10 %.4f  p25 %.4f  p50 %.4f  p75 %.4f  p90 %.4f  mean %.4f'
          % (*np.percentile(rank, [10, 25, 50, 75, 90]), rank.mean()))
    z = np.load(SV)
    sls = np.clip(np.stack([z['scale_0'], z['scale_1'], z['scale_2']], 1).astype(np.float64), -12, 3)
    svs = np.exp(sls)
    svr, _, _, _ = rank_of(svs)
    print('scaniverse n=%d' % svs.shape[0])
    print('  rank  p10 %.4f  p25 %.4f  p50 %.4f  p75 %.4f  p90 %.4f  mean %.4f'
          % (*np.percentile(svr, [10, 25, 50, 75, 90]), svr.mean()))
    print('  -> the target that matches THEIR population is their MEDIAN rank: %.3f'
          % np.median(svr))

    print('\n=== 2. WHAT THE PRIOR DOES TO THE SMALLEST AXIS, PER TARGET ===')
    print('  (kernel: dL/dlogScale_j = -2 w (rank-target) rank p_j (log p_j + H);')
    print('   Adam steps DOWN the gradient, so grad>0 SHRINKS that axis.)')
    order = np.argsort(scale, axis=1)          # ascending
    jmin = order[:, 0]
    rows = np.arange(count)
    W = 0.001
    print('  %-8s %8s %8s %9s  %-22s %s'
          % ('target', 'frac>t', 'med res', 'med|g_min|', 'smallest axis is', 'sum g'))
    for tgt in (1.0, 2.0, 2.3, 2.61, 3.0):
        res = rank - tgt
        k = -2.0 * W * res * rank
        g = k[:, None] * p * (lp + H[:, None])
        gmin = g[rows, jmin]
        frac_shrink = (gmin > 0).mean()
        print('  %-8.2f %7.1f%% %+8.4f %9.2e  %-22s %.2e'
              % (tgt, 100 * (rank > tgt).mean(), np.median(res),
                 np.median(np.abs(gmin)),
                 'SHRUNK %.1f%% of model' % (100 * frac_shrink),
                 np.abs(g.sum(1)).max()))

    print('\n  volume check: the prior gradient sums to zero across the three')
    print('  components for every splat (max |sum| printed above), so the prior')
    print('  can only ROTATE shape at constant geometric-mean scale. It cannot')
    print('  make a splat smaller. Size and shape are separate levers.')

    # ---------------------------------------------------------------- render
    print('\n=== 3. RENDER, ISOTROPISED AT CONSTANT VOLUME ===')
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    poses = P.load_poses()
    bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
    paths = {f['index']: f['imagePath'] for f in bundle['frames']}
    from PIL import Image
    LUMA = Q.LUMA

    TS = [-0.15, 0.0, 0.10, 0.20, 0.35, 0.60]
    mean_log = logs.mean(axis=1, keepdims=True)

    acc = {t: [] for t in TS}
    for frame in frames:
        src = os.path.join(r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840",
                           paths[frame])
        truth = np.asarray(Image.open(src).convert('RGB').resize((rw, rh), Image.LANCZOS),
                           dtype=np.float64) / 255.0
        ty_l = (truth * LUMA).sum(2)
        rot, t3 = poses[frame]
        Rm = P.quat_to_matrix(rot)
        for tt in TS:
            newlogs = mean_log + (1.0 - tt) * (logs - mean_log)
            col['scale_0'], col['scale_1'], col['scale_2'] = newlogs[:, 0], newlogs[:, 1], newlogs[:, 2]
            ok, radius, tiles, ex, ey = P.project(col, count, Rm, t3, fx, fy, cx, cy, tx, ty)
            g = R.geometry(col, count, Rm, t3, fx, fy, cx, cy)
            img = S.render(col, count, g, ok, ex, ey, g['mx'], g['my'], tx, ty, rw, rh, None)
            img = np.clip(img, 0, 1)
            mse = float(np.mean((img - truth) ** 2))
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            x, y = img.ravel(), truth.ravel()
            n = x.size
            den = n * (x * x).sum() - x.sum() ** 2
            gain = (n * (x * y).sum() - x.sum() * y.sum()) / den if den > 1e-9 else 1.0
            bias = (y.sum() - gain * x.sum()) / n
            fit = np.clip(gain * img + bias, 0, 1)
            psnr_f = 10 * np.log10(1.0 / max(float(np.mean((fit - truth) ** 2)), 1e-12))
            ss = Q.ssim_terms((img * LUMA).sum(2), ty_l, Q.LAMBDA_SSIM)['ssim'].mean()
            sc2 = np.exp(np.clip(newlogs, -12, 3))
            rk, _, _, _ = rank_of(sc2)
            acc[tt].append((psnr, psnr_f, ss))
            print('  frame %2d  iso t=%+.2f  rank p50 %.3f  aspect p50 %5.2f  '
                  'PSNR %6.3f  fitted %6.3f  SSIM %.5f'
                  % (frame, tt, np.median(rk),
                     np.median(np.sort(sc2, 1)[:, 2] / np.maximum(np.sort(sc2, 1)[:, 0], 1e-12)),
                     psnr, psnr_f, ss))
            sys.stdout.flush()

    print('\n  --- mean over %d frame(s) ---' % len(frames))
    base = np.mean(acc[0.0], axis=0)
    for tt in TS:
        m = np.mean(acc[tt], axis=0)
        print('  t=%+.2f  PSNR %6.3f (%+6.3f)  fitted %6.3f (%+6.3f)  SSIM %.5f (%+.5f)'
              % (tt, m[0], m[0] - base[0], m[1], m[1] - base[1], m[2], m[2] - base[2]))


if __name__ == '__main__':
    main()
