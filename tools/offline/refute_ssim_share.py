"""REFUTATION TEST for "lambdaSSIM 0.2 really delivers 74% of the gradient, so
set it to 0.2/2.87 = 0.07".

L1's total gradient mass is EXACTLY frameWeight*(1-lambda) = qc*0.8, whatever the
image is: the kernel emits w*sign(diff)/3 per channel, a constant magnitude.  SSIM's
mass is NOT bounded and shrinks as the render approaches the photograph.  So the
"2.87x" is a measurement of how wrong THIS render is, not a property of the weights.
If that is right, the ratio collapses as the render improves, and a lambda picked to
cancel it today is wrong at every other quality level.

Also: is a 14% "override" rate anomalous?  Compare it with the rate two INDEPENDENT
sign fields of the same marginal magnitudes would produce.
"""
import io, json, os, sys
import numpy as np
import project as P
import detail as Dt
import lossq_blur as B
from lossq_ssim import ssim_terms, LUMA, LAMBDA_SSIM

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
N = rw * rh
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
frames = dict(Dt.frames_with_images())
qc = {int(f['index']): float(f['qc']['weight']) for f in Dt.bundle()['frames']}

for frame in [int(a) for a in sys.argv[1:]] or [0, 4]:
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    img, Tf, ok, g, inst = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25 + 1.33333, 2)
    _, gt = Dt.luma_of(frames[frame], rw, rh)
    gt = gt.astype(np.float64)
    gain, bias = B.fit_exposure(img, gt)
    base = gain * img + bias
    qcw = qc[frame]
    print('\n=== frame %d (qc %.4f) ===' % (frame, qcw))
    print('  a   PSNR dB   sum|gL1|   sum|gSSIM|  SSIM share  ratio  lambda that would')
    print('                                                          give a 0.25x ratio')
    for a in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99):
        rend = (1 - a) * base + a * gt
        diff = rend - gt
        w_l1 = qcw * (1 - LAMBDA_SSIM) / N
        g_l1 = w_l1 * np.sign(diff) / 3.0
        X = (rend * LUMA).sum(2); Y = (gt * LUMA).sum(2)
        S = ssim_terms(X, Y, qcw * LAMBDA_SSIM / N)
        g_ss = S['dLdX'][:, :, None] * LUMA[None, None, :]
        m1, ms = np.abs(g_l1).sum(), np.abs(g_ss).sum()
        psnr = 10 * np.log10(1.0 / max((diff ** 2).mean(), 1e-12))
        # lambda' such that mass(SSIM)/mass(L1) == 0.25, holding the fields fixed
        # mass scales linearly in lambda for SSIM and in (1-lambda) for L1
        k = ms / max(m1, 1e-30) * (1 - LAMBDA_SSIM) / LAMBDA_SSIM   # per unit lambda ratio
        lam = 0.25 / (k + 0.25)
        print('  %.2f  %7.3f  %.4e  %.4e   %5.1f%%   %5.2fx   %.4f'
              % (a, psnr, m1, ms, 100 * ms / (m1 + ms), ms / m1, lam))

    # override rate against an independent-signs null
    diff = base - gt
    g_l1 = qcw * (1 - LAMBDA_SSIM) / N * np.sign(diff) / 3.0
    X = (base * LUMA).sum(2); Y = (gt * LUMA).sum(2)
    S = ssim_terms(X, Y, qcw * LAMBDA_SSIM / N)
    g_ss = S['dLdX'][:, :, None] * LUMA[None, None, :]
    bigger = (np.abs(g_ss) > np.abs(g_l1))
    disagree = (np.sign(g_ss) != np.sign(g_l1))
    flip = np.sign(g_l1 + g_ss) != np.sign(g_l1)
    print('  OVERRIDE RATE vs an independent-fields null:')
    print('    P(|gSSIM|>|gL1|) %.4f  x  P(signs disagree) %.4f  =  %.4f expected if independent'
          % (bigger.mean(), disagree.mean(), bigger.mean() * disagree.mean()))
    print('    measured override                                  =  %.4f' % flip.mean())
    print('    -> measured / independent-null = %.2fx' % (flip.mean() / (bigger.mean() * disagree.mean())))
