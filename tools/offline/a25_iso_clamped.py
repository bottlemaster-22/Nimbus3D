"""A25/A69: the post-hoc isotropy test, RE-RUN WITH THE TANGENT CLAMP, on the
full renderer (lossq_blur.render: clamped EWA, Mip 2D filter at the eval
low-pass 0.25, SH degree 1 colour), on build 266's model, on all 16 frames
that have photographs. Also one extra config for A59: the same model drawn
with DC colour only (band 1 switched off).

Isotropy at constant volume: new = mean_log + (1 - t)(log - mean_log), per
splat; t = 0 is the model as trained, t = 1 is a sphere of the same
geometric-mean radius. This is a PERTURBATION of a trained model, not a
retrain; it bounds how much the image cares about shape, nothing more.

Run: cd tools/offline && python -u a25_iso_clamped.py [workers]
Writes a25_iso_clamped.json (small).
"""
import io, json, os, sys, time
from multiprocessing import Pool
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B
import lossq_ssim as Q

CONFIGS = [('t=0.00 (as trained)', 0.0, 4), ('t=0.25', 0.25, 4), ('t=0.50', 0.5, 4),
           ('t=1.00 (spheres)', 1.0, 4), ('t=0.00, DC colour only', 0.0, 1)]
LOW_PASS = 0.25          # held-out eval and viewer (lossq_blur configs)
_G = {}


def _init():
    assert P.TANGENT_CLAMP, 'this test must run with the device tangent clamp'
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    col0, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    col = {k: np.array(v, dtype=np.float64) for k, v in col0.items()}
    logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1), -12, 3)
    _G.update(rw=rw, rh=rh, col=col, count=count, logs=logs,
              intr=P.load_intrinsics(rw, rh), poses=P.load_poses(),
              frames=dict(Dt.frames_with_images()))


def job(args):
    frame, ci = args
    name, t, ash = CONFIGS[ci]
    g = _G
    col = dict(g['col'])
    logs = g['logs']
    m = logs.mean(1, keepdims=True)
    new = m + (1.0 - t) * (logs - m)
    col['scale_0'], col['scale_1'], col['scale_2'] = new[:, 0], new[:, 1], new[:, 2]
    fx, fy, cx, cy = g['intr']
    rot, tr = g['poses'][frame]
    R = P.quat_to_matrix(rot)
    img, Tf, ok, gg, inst = B.render(col, g['count'], R, tr, fx, fy, cx, cy,
                                     g['rw'], g['rh'], LOW_PASS, ash)
    img = np.clip(img, 0, 1)
    _, gt = Dt.luma_of(g['frames'][frame], g['rw'], g['rh'])
    gt = gt.astype(np.float64)
    raw = B.psnr(img, gt)
    gn, bs = B.fit_exposure(img, gt)
    fit = np.clip(gn * img + bs, 0, 1)
    pf = B.psnr(fit, gt)
    ss = float(Q.ssim_terms((fit * Q.LUMA).sum(2), (gt * Q.LUMA).sum(2),
                            Q.LAMBDA_SSIM)['ssim'].mean())
    return frame, ci, raw, pf, ss, int(inst)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    frames = sorted(dict(Dt.frames_with_images()))
    jobs = [(f, ci) for f in frames for ci in range(len(CONFIGS))]
    t0 = time.time()
    res = []
    with Pool(workers, initializer=_init) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print('  frame %2d  %-24s PSNR raw %6.3f  fitted %6.3f  SSIM %.4f  inst %d  (%.0fs)'
                  % (r[0], CONFIGS[r[1]][0], r[2], r[3], r[4], r[5], time.time() - t0))
            sys.stdout.flush()
    A = {ci: np.array([[r[2], r[3], r[4]] for r in sorted(res) if r[1] == ci])
         for ci in range(len(CONFIGS))}
    base = A[0].mean(0)
    print('\n=== MEAN OVER %d FRAMES (clamped, eval low-pass %.2f) ===' % (len(frames), LOW_PASS))
    print('%-26s %9s %9s %9s %9s %8s %9s  %s' % ('config', 'PSNR raw', 'd', 'fitted', 'd',
                                                 'SSIM', 'd', 'frames worse (fitted)'))
    out = {}
    for ci, (name, t, ash) in enumerate(CONFIGS):
        m = A[ci].mean(0)
        worse = int((A[ci][:, 1] < A[0][:, 1]).sum())
        print('%-26s %9.3f %+9.3f %9.3f %+9.3f %8.4f %+9.4f  %d/%d'
              % (name, m[0], m[0] - base[0], m[1], m[1] - base[1], m[2], m[2] - base[2],
                 worse, len(frames)))
        d = A[ci][:, 1] - A[0][:, 1]
        out[name] = dict(psnr_raw=float(m[0]), psnr_fit=float(m[1]), ssim=float(m[2]),
                         d_fit_mean=float(d.mean()), d_fit_min=float(d.min()),
                         d_fit_max=float(d.max()), per_frame_fit=A[ci][:, 1].tolist())
    json.dump(out, open('a25_iso_clamped.json', 'w'), indent=1)
    print('wrote a25_iso_clamped.json')


if __name__ == '__main__':
    main()
