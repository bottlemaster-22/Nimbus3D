"""A26 / A38: is the per-frame exposure correction inert, what is it fitting,
and is heldOutPSNRExposureFitted a fair model-selection score?

Frames 0-15 (the only photographed frames), build-266 model, DEVICE-MATCHED
clamped renderer (project.TANGENT_CLAMP), held-out eval config
(lowPass 0.25, all degree-1 SH bands), no background (mean T ~0.001).

Per frame:
  raw PSNR
  device LS fit (evaluateHeldOut 2995-3019: joint gain+bias over RGB, clamped)
  unclamped LS gain/bias, and the slope decomposition g = cov(R,T)/var(R)
  first-moment gain gm = mean(T)/mean(R)   (unaffected by zero-mean render error)
  low-pass LS gain (both images blurred sigma 16 px)
  per-channel mean ratio (white-balance drift proxy)
  duration-only prior: frames 0-15 are 1/60 s, the trained bulk 1/100 s
  device update rule (MetalSplatTrainer 2209-2221, lr 0.002, clamps) for
    37 visits (4000 iterations / 108 trained keyframes), L1+SSIM, on the
    FIXED final render; plus an Adam alternative sized to converge.
Degraded-model proxies (for the model-selection question):
  B = training-at-2000 render config (lowPass 2.9167, SH band 0 only)
  C = eval config with a seeded random 25% of splats dropped
Run: python -u exp_a26_a38.py
"""
import io, json, os, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter

import project as P
import detail as Dt
import lossq_blur as LB

assert P.TANGENT_CLAMP, "must use the device-matched clamped projection"
LUMA = np.array([0.2126, 0.7152, 0.0722])        # TrainerShaders.metal:121
LR, LAM = 0.002, 0.2                               # TrainerSupport 1019, 473
GMIN, GMAX, BMIN, BMAX = 0.9, 1.1, -0.05, 0.05     # TrainerSupport 1023-1024
C1S, C2S = 0.01 ** 2, 0.03 ** 2
VISITS = int(round(4000 / 108))                    # 37
TRAINED = {0, 3, 8, 11, 14}                        # exp_gap.py keyframe reproduction
OUT = os.path.join(Dt.SCRATCH, 'exp_a26_a38.npz')


def blur(x):
    return gaussian_filter(x, sigma=1.5, mode='nearest', truncate=5.0 / 1.5)


def ssim_dLdX(X, Y, n, fw):
    mux = blur(X); muy = blur(Y)
    sxx = np.maximum(blur(X * X) - mux * mux, 0); syy = np.maximum(blur(Y * Y) - muy * muy, 0)
    sxy = blur(X * Y) - mux * muy
    n1 = 2 * mux * muy + C1S; n2 = 2 * sxy + C2S
    d1 = mux * mux + muy * muy + C1S; d2 = sxx + syy + C2S
    invD = 1.0 / np.maximum(d1 * d2, 1e-12)
    w = fw * LAM / n
    dmu = (2 * muy * n2 * d1 - 2 * mux * n1 * n2) / np.maximum(d1 * d1 * d2, 1e-12)
    dsxy = 2 * n1 * invD
    dsxx = -n1 * n2 / np.maximum(d1 * d2 * d2, 1e-12)
    Cc = -w * dmu; A = -w * dsxx; B = -w * dsxy
    return blur(Cc - 2 * A * mux - B * muy) + 2 * X * blur(A) + Y * blur(B)


def exposure_grad(pre, T, g, b, fw):
    """exactly trainer_loss_photometric + ssim + trainer_loss_finalize."""
    H, W, _ = T.shape; n = H * W
    rend = g * pre + b
    gpx = fw * (1 - LAM) / n * np.sign(rend - T) / 3.0
    X = (rend * LUMA).sum(-1); Y = (T * LUMA).sum(-1)
    gpx = gpx + ssim_dLdX(X, Y, n, fw)[..., None] * LUMA[None, None, :]
    return float((pre * gpx).sum()), float(gpx.sum())


def device_travel(pre, T, fw, steps):
    g, b = 1.0, 0.0; mg = []; mb = []
    for _ in range(steps):
        gg, gb = exposure_grad(pre, T, g, b, fw)
        mg.append(abs(gg)); mb.append(abs(gb))
        g = min(max(g - LR * gg, GMIN), GMAX)
        b = min(max(b - LR * gb, BMIN), BMAX)
    return g, b, float(np.mean(mg)), float(np.mean(mb))


def adam_travel(pre, T, fw, steps, lr):
    p = np.array([1.0, 0.0]); m = np.zeros(2); v = np.zeros(2)
    for k in range(1, steps + 1):
        gr = np.array(exposure_grad(pre, T, p[0], p[1], fw))
        m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
        p = p - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
        p[0] = min(max(p[0], GMIN), GMAX); p[1] = min(max(p[1], BMIN), BMAX)
    return float(p[0]), float(p[1])


def ls(x, y):
    x = x.ravel(); y = y.ravel()
    vx = x.var(); c = ((x - x.mean()) * (y - y.mean())).mean()
    g = c / vx; return g, y.mean() - g * x.mean()


def psnr(a, b):
    return 10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-12))


def scores(img, gt):
    """raw, device-fitted (clamped LS), mean-matched gain (clamped)."""
    g, b = LB.fit_exposure(img, gt)
    gm = float(np.clip(gt.mean() / max(img.mean(), 1e-9), GMIN, GMAX))
    # evaluateHeldOut does NOT clip the fitted value to 0..1 (3035); nor here
    return psnr(img, gt), psnr(g * img + b, gt), psnr(gm * img, gt)


def main():
    t0 = time.time()
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]; rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
    byidx = {f['index']: f for f in bundle['frames']}
    frames = sorted(Dt.frames_with_images())
    rng = np.random.default_rng(12345)
    keep = rng.random(count) >= 0.25
    colC = {k: v[keep] for k, v in col.items()}; countC = int(keep.sum())
    print('render %dx%d  splats %d (drop-25%% proxy keeps %d)  visits/frame %d' % (rw, rh, count, countC, VISITS))
    rows = []
    for idx, path in frames:
        f = byidx[idx]; fw = f['qc']['weight']
        rot, t = poses[idx]; R = P.quat_to_matrix(rot)
        img, Tf, *_ = LB.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
        gt = Dt.luma_of(path, rw, rh)[1].astype(np.float64)
        imgB, *_ = LB.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25 + 2.66667, 1)
        imgC, *_ = LB.render(colC, countC, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
        sA, sB, sC = scores(img, gt), scores(imgB, gt), scores(imgC, gt)
        gls, bls = ls(img, gt)
        e = img - gt
        covRe = ((img - img.mean()) * (e - e.mean())).mean() / img.var()
        rho = np.corrcoef(img.ravel(), gt.ravel())[0, 1]
        gm = gt.mean() / img.mean()
        chan = gt.reshape(-1, 3).mean(0) / img.reshape(-1, 3).mean(0)
        glp, blp = ls(gaussian_filter(img, (16, 16, 0), mode='nearest'),
                      gaussian_filter(gt, (16, 16, 0), mode='nearest'))
        dprior = f['exposureDurationSeconds'] / 0.01          # photo brighter by this if ISO fixed
        p_prior = psnr(img * dprior, gt)                       # scale the RENDER to the photo's exposure
        dg, db, mgg, mgb = device_travel(img, gt, fw, VISITS)
        ag, ab = adam_travel(img, gt, fw, VISITS, 0.005)
        l1_bound = fw * (1 - LAM) * img.sum(-1).mean() / 3.0
        r = dict(idx=idx, trained=idx in TRAINED, fw=fw, blur=f['qc']['motionBlurPixels'],
                 dur=f['exposureDurationSeconds'], meanT=float(Tf.mean()),
                 rawA=sA[0], fitA=sA[1], mmA=sA[2], rawB=sB[0], fitB=sB[1], mmB=sB[2],
                 rawC=sC[0], fitC=sC[1], mmC=sC[2],
                 gls=gls, bls=bls, covRe=covRe, rho=rho, gm=gm, glp=glp, blp=blp,
                 cR=chan[0], cG=chan[1], cB=chan[2], p_prior=p_prior,
                 dev_g=dg, dev_b=db, dev_mg=mgg, dev_mb=mgb, l1_bound=l1_bound,
                 adam_g=ag, adam_b=ab, p_dev=psnr(dg * img + db, gt), p_adam=psnr(ag * img + ab, gt))
        rows.append(r)
        print(('f%2d %s w%.2f blur%5.2f | raw %.2f fit %.2f mm %.2f | LS g %.3f b %+.3f (covRe/varR %.3f rho %.3f) '
               '| gm %.3f lp-g %.3f | ch %.3f %.3f %.3f | prior %.2f | dev g %.4f b %+.4f |gg| %.4f bound %.3f '
               '| adam g %.3f b %+.3f p %.2f | B %.2f/%.2f C %.2f/%.2f  (%.0fs)')
              % (idx, 'T' if r['trained'] else '-', fw, r['blur'], sA[0], sA[1], sA[2], gls, bls, covRe, rho,
                 gm, glp, chan[0], chan[1], chan[2], p_prior, dg, db, mgg, l1_bound, ag, ab, r['p_adam'],
                 sB[0], sB[1], sC[0], sC[1], time.time() - t0), flush=True)
    np.savez(OUT, rows=np.array([json.dumps(r, default=float) for r in rows]))
    K = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    c = lambda a, b: float(np.corrcoef(K[a], K[b])[0, 1])
    print('\n=== SUMMARY over %d frames ===' % len(rows))
    print('duration of all 16: %s s' % sorted(set(np.round(K['dur'], 5))))
    print('LS gain %.3f..%.3f  bias %+.3f..%+.3f ; corr(gain, rawPSNR) %+.2f corr(gain, blur) %+.2f'
          % (K['gls'].min(), K['gls'].max(), K['bls'].min(), K['bls'].max(), c('gls', 'rawA'), c('gls', 'blur')))
    print('first-moment gain %.3f..%.3f (mean %.3f sd %.3f) ; corr(gm, rawPSNR) %+.2f'
          % (K['gm'].min(), K['gm'].max(), K['gm'].mean(), K['gm'].std(), c('gm', 'rawA')))
    print('low-pass LS gain %.3f..%.3f ; corr(glp, rawPSNR) %+.2f' % (K['glp'].min(), K['glp'].max(), c('glp', 'rawA')))
    print('cov(R,e)/var(R) %.3f..%.3f  (LS gain = 1 - this) ; rho %.3f..%.3f'
          % (K['covRe'].min(), K['covRe'].max(), K['rho'].min(), K['rho'].max()))
    print('channel ratio R %.3f..%.3f G %.3f..%.3f B %.3f..%.3f'
          % (K['cR'].min(), K['cR'].max(), K['cG'].min(), K['cG'].max(), K['cB'].min(), K['cB'].max()))
    print('PSNR raw %.3f | device-fit %.3f (%+.3f) | mean-matched %.3f (%+.3f) | duration prior x1.667 %.3f (%+.3f)'
          % (K['rawA'].mean(), K['fitA'].mean(), K['fitA'].mean() - K['rawA'].mean(), K['mmA'].mean(),
             K['mmA'].mean() - K['rawA'].mean(), K['p_prior'].mean(), K['p_prior'].mean() - K['rawA'].mean()))
    print('device update after %d visits: gain %.5f..%.5f bias %+.5f..%+.5f ; mean |dL/dgain| %.4f (L1 bound %.3f) |dL/dbias| %.4f'
          % (VISITS, K['dev_g'].min(), K['dev_g'].max(), K['dev_b'].min(), K['dev_b'].max(),
             K['dev_mg'].mean(), K['l1_bound'].mean(), K['dev_mb'].mean()))
    print('   -> max travel %.2f%% of gain half-width, PSNR change %+.4f dB'
          % (100 * np.abs(K['dev_g'] - 1).max() / 0.1, (K['p_dev'] - K['rawA']).mean()))
    print('Adam lr 0.005, %d visits: gain %.3f..%.3f bias %+.3f..%+.3f ; PSNR %+.3f dB vs raw ; corr(adam_g, rawPSNR) %+.2f'
          % (VISITS, K['adam_g'].min(), K['adam_g'].max(), K['adam_b'].min(), K['adam_b'].max(),
             (K['p_adam'] - K['rawA']).mean(), c('adam_g', 'rawA')))
    for tag, sel in (('trained', K['trained'] > 0), ('never-trained', K['trained'] == 0)):
        print('  %-13s n=%d raw %.2f fit %.2f gm %.3f LSg %.3f' % (tag, sel.sum(), K['rawA'][sel].mean(),
              K['fitA'][sel].mean(), K['gm'][sel].mean(), K['gls'][sel].mean()))
    print('\n=== MODEL SELECTION: does the fitted score preserve differences? ===')
    for v, name in (('B', 'blurrier (train-2000 config)'), ('C', 'drop 25% splats')):
        dr = K['raw' + v] - K['rawA']; df = K['fit' + v] - K['fitA']; dm = K['mm' + v] - K['mmA']
        print('%-30s delta raw %+.3f  fitted %+.3f  mean-matched %+.3f ; frames where fitted ranks A<=%s: %d/16 (raw: %d/16)'
              % (name, dr.mean(), df.mean(), dm.mean(), v, int((df >= 0).sum()), int((dr >= 0).sum())))
        print('%-30s fit uplift: A %+.3f  %s %+.3f' % ('', (K['fitA'] - K['rawA']).mean(), v,
              (K['fit' + v] - K['raw' + v]).mean()))
    print('total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
