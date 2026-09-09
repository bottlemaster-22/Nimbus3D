"""How many of the 120 selected keyframes does each Gaussian appear in?

Uses tools/offline/project.py (the validated offline trainer_preprocess) once
per keyframe, and records the full 300k x 120 visibility mask plus, for a
random sample of splats, the world-space viewing ray from each camera that
sees it, so the real triangulation angle can be measured on views that
actually share the point.
"""
import io, json, os, time
import numpy as np
import project as P
from cap_keyframes import keyframes, split_held_out

OUT = os.path.dirname(os.path.abspath(__file__))
SAMPLE = 30000
rng = np.random.default_rng(7)


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15)//16, (rh + 15)//16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()

    b, r, frames, pool, kf, info = keyframes()
    kfidx = [f['index'] for f in kf]
    tr, ho = split_held_out(len(kf))
    print('keyframes %d  (train %d, held out %d)' % (len(kf), len(tr), len(ho)))

    # world-space splat means in the SAME frame as the poses (project.py's flip)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)

    samp = np.sort(rng.choice(count, SAMPLE, replace=False))
    vis = np.zeros((count, len(kf)), dtype=bool)
    rays = np.zeros((SAMPLE, len(kf), 3), dtype=np.float32)
    rng_m = np.zeros((SAMPLE, len(kf)), dtype=np.float32)   # camera->splat range

    t0 = time.time()
    for j, fi in enumerate(kfidx):
        rot, t = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, rad, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        vis[:, j] = ok
        C = -R.T @ t
        v = mean[samp] - C
        d = np.linalg.norm(v, axis=1)
        rays[:, j, :] = (v / d[:, None]).astype(np.float32)
        rng_m[:, j] = d.astype(np.float32)
        rays[~ok[samp], j, :] = 0
        if (j + 1) % 30 == 0:
            print('  ...%d/%d  %.0fs' % (j+1, len(kf), time.time()-t0))

    nviews = vis.sum(axis=1)
    np.save(os.path.join(OUT, '_cap_vis.npy'), np.packbits(vis, axis=1))
    np.save(os.path.join(OUT, '_cap_nviews.npy'), nviews.astype(np.int16))
    np.save(os.path.join(OUT, '_cap_samp.npy'), samp)
    np.save(os.path.join(OUT, '_cap_rays.npy'), rays)
    np.save(os.path.join(OUT, '_cap_range.npy'), rng_m)

    print('\n=== VIEW COUNT per Gaussian, over the %d selected keyframes ===' % len(kf))
    print('  splats %d   never seen in any keyframe: %d (%.2f%%)'
          % (count, (nviews == 0).sum(), 100*(nviews == 0).mean()))
    for k in (1, 2, 3, 4, 5, 8, 10, 16):
        print('  seen in < %-2d views: %8d  (%5.2f%%)'
              % (k, (nviews < k).sum(), 100*(nviews < k).mean()))
    print('  percentiles of view count:')
    for q in (1, 5, 10, 25, 50, 75, 90, 99):
        print('    p%-3d %5d' % (q, np.percentile(nviews, q)))
    print('  mean %.2f  max %d' % (nviews.mean(), nviews.max()))

    # the same, restricted to the 108 TRAINED views (held-out never supervise)
    vtr = vis[:, tr].sum(axis=1)
    print('\n  restricted to the %d TRAINED keyframes:' % len(tr))
    for k in (1, 2, 3):
        print('    seen in < %d trained views: %8d (%5.2f%%)'
              % (k, (vtr < k).sum(), 100*(vtr < k).mean()))
    print('    median %d  mean %.2f' % (np.median(vtr), vtr.mean()))
    np.save(os.path.join(OUT, '_cap_nviews_train.npy'), vtr.astype(np.int16))

    print('  saved _cap_vis.npy / _cap_nviews.npy / _cap_rays.npy')


if __name__ == '__main__':
    main()
