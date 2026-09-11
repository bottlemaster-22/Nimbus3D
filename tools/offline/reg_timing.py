"""Is the per-frame render-vs-photo shift on frames 0-15 a POSE error or a
TIMING error?

If the photo was exposed at t + dt while its pose is the pose at t, the photo
is the render displaced by (image motion per second) * dt.  So regress the
measured sub-pixel shift on each frame's own image-motion rate, computed from
the refined poses of its neighbours (central difference, points 2 m ahead).
A single dt explaining most of the variance says timing; no fit says pose.

Only UNTRAINED frames are fitted: a trained frame's error has been absorbed by
the model and its camera delta (frames 0, 4, 11 are the build-266 keyframes in
0-15, from the kf_lookahead replica that reproduces held_out_frames.json).

Needs the render cache written by lossq_align_cv.py (LIKOVA_SCRATCH).
Run: python -u reg_timing.py
"""
import io, json, os
import numpy as np
import project as P
from lossq_align import best_shift
from lossq_align_cv import CACHE, luma, subpixel_peak

TRAINED = {0, 4, 11}


def mat(p):
    M = np.eye(4)
    M[:3, :3] = P.quat_to_matrix(p[0])
    M[:3, 3] = p[1]
    return M


def main():
    z = np.load(CACHE)
    use, imgs, gts, fx = list(z['frames']), z['imgs'], z['gts'], float(z['fx'])
    h, w = imgs.shape[1:3]
    cx, cy = w / 2.0, h / 2.0
    b = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
    fr = {f['index']: f for f in b['frames']}
    ref = P.load_poses()

    u, v = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.4, 0.4, 5))
    pts_cam = np.stack([u.ravel() * 2, v.ravel() * 2, np.full(u.size, 2.0), np.ones(u.size)])

    def flow_rate(i):
        lo, hi = max(i - 1, 0), i + 1
        Mi = mat(ref[i])
        world = np.linalg.inv(Mi) @ pts_cam
        out = []
        for j in (lo, hi):
            q = mat(ref[j]) @ world
            out.append(np.stack([fx * q[0] / q[2] + cx, fx * q[1] / q[2] + cy]))
        dt = fr[hi]['timestampSeconds'] - fr[lo]['timestampSeconds']
        f = (out[1] - out[0]) / dt           # px per second, (x, y) per point
        return f[1].mean(), f[0].mean()      # (dy, dx) order, as the shifts

    rows = []
    print('%5s %7s %8s %8s | %9s %9s | %6s %6s' %
          ('frame', 'trained', 'sub dy', 'sub dx', 'flow dy/s', 'flow dx/s', 'blur', 'w r/s'))
    for k, f in enumerate(use):
        X, Y = luma(imgs[k].astype(np.float64)), luma(gts[k].astype(np.float64))
        dy, dx, _, _ = best_shift(X, Y)
        sy, sx = subpixel_peak(X, Y, dy, dx)
        fy_, fx_ = flow_rate(f)
        q = fr[f]['qc']
        rows.append((f, f in TRAINED, sy, sx, fy_, fx_))
        print('%5d %7s %8.2f %8.2f | %9.1f %9.1f | %6.2f %6.3f'
              % (f, 'yes' if f in TRAINED else '', sy, sx, fy_, fx_,
                 q['motionBlurPixels'], q['angularSpeedRadPerSec']))

    un = [r for r in rows if not r[1]]
    s = np.concatenate([[r[2] for r in un], [r[3] for r in un]])
    g = np.concatenate([[r[4] for r in un], [r[5] for r in un]])
    dt = float(g @ s / (g @ g))
    res = s - dt * g
    r2 = 1 - (res @ res) / (s @ s)
    print('\nuntrained frames %d: shift = dt * flow, dt = %+.1f ms, R^2 (about zero) = %.3f'
          % (len(un), 1000 * dt, r2))
    print('  |shift| median before %.2f px, after removing dt*flow %.2f px'
          % (np.median(np.hypot(s[:len(un)], s[len(un):])),
             np.median(np.hypot(res[:len(un)], res[len(un):]))))
    # leave-one-frame-out prediction
    press = 0.0
    for i in range(len(un)):
        keep = [j for j in range(len(un)) if j != i]
        ss = np.concatenate([[un[j][2] for j in keep], [un[j][3] for j in keep]])
        gg = np.concatenate([[un[j][4] for j in keep], [un[j][5] for j in keep]])
        d = gg @ ss / (gg @ gg)
        e = np.array([un[i][2] - d * un[i][4], un[i][3] - d * un[i][5]])
        press += e @ e
    print('  leave-one-frame-out predictive R^2 = %.3f' % (1 - press / (s @ s)))
    # null: shuffle which frame's flow goes with which shift
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(2000):
        p = rng.permutation(len(un))
        gp = np.concatenate([[un[j][4] for j in p], [un[j][5] for j in p]])
        d = gp @ s / (gp @ gp)
        rr = s - d * gp
        nulls.append(1 - (rr @ rr) / (s @ s))
    nulls = np.array(nulls)
    print('  null (flows shuffled across frames, 2000 draws): R^2 p50 %.3f p95 %.3f p99 %.3f; '
          'observed beats %.1f%% of shuffles' % (np.median(nulls), np.percentile(nulls, 95),
                                                 np.percentile(nulls, 99), 100 * (nulls < r2).mean()))
    t = [r for r in rows if r[1]]
    print('trained frames: ' + '  '.join('%d: shift %.2f,%.2f predicted %.2f,%.2f'
                                         % (r[0], r[2], r[3], dt * r[4], dt * r[5]) for r in t))


if __name__ == '__main__':
    main()
