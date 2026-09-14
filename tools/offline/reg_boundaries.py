"""Does the per-submap rigid correction tear the refined trajectory at the
submap OWNER boundaries?

refined_f = raw_f * M_owner(f).  Inside one owner the relative pose of two
consecutive frames is exactly the raw VIO relative pose (M cancels). Across an
owner change it is not: the two frames see the same surfaces from almost the
same place, but get corrections M_k and M_k+1.  This measures that tear
directly from prepass_result.json refinedPoses against the bundle's rawPoses.

Run: python -u reg_boundaries.py
"""
import io, json, os
import numpy as np
import project as P
import reg_candidates as RC


def mat(rot, t):
    M = np.eye(4)
    M[:3, :3] = P.quat_to_matrix(rot)
    M[:3, 3] = np.asarray(t, float)
    return M


def main():
    b, r = RC.load()
    frames = sorted(b['frames'], key=lambda f: f['timestampSeconds'])
    own = RC.owners(frames, r['submaps'])
    ref = P.load_poses()
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    fx = P.load_intrinsics(sl['renderWidth'], sl['renderHeight'])[0]
    held = json.load(io.open(os.path.join(P.D, 'model', 'held_out_frames.json'), encoding='utf-8'))

    # a grid of points 2 m in front of camera b, to turn the tear into pixels
    u, v = np.meshgrid(np.linspace(-0.6, 0.6, 7), np.linspace(-0.45, 0.45, 5))
    pts = np.stack([u.ravel() * 2, v.ravel() * 2, np.full(u.size, 2.0), np.ones(u.size)])

    within, across = [], []
    for a, c in zip(frames[:-1], frames[1:]):
        Ra = mat(a['rawPose']['rotation'], a['rawPose']['translation'])
        Rb = mat(c['rawPose']['rotation'], c['rawPose']['translation'])
        Fa = mat(*ref[a['index']]); Fb = mat(*ref[c['index']])
        E = (Fb @ np.linalg.inv(Fa)) @ np.linalg.inv(Rb @ np.linalg.inv(Ra))
        ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
        tr = 100 * np.linalg.norm(E[:3, 3])
        q = E @ pts
        px = np.hypot(fx * (q[0] / q[2] - pts[0] / pts[2]), fx * (q[1] / q[2] - pts[1] / pts[2]))
        row = (a['index'], c['index'], own[a['index']], own[c['index']], ang, tr, px.mean())
        (across if own[a['index']] != own[c['index']] else within).append(row)

    w = np.array([x[4:] for x in within])
    print('consecutive pairs inside one owner: %d   max tear %.4f deg %.4f cm %.3f px  (validation: ~0, '
          'only the 1.3 ms time-offset re-interpolation differs)' % (len(within), *w.max(axis=0)))
    print('\nOWNER BOUNDARIES (frame a -> frame b, submap -> submap): tear between two consecutive frames')
    print('%6s %6s %9s %8s %8s %12s' % ('a', 'b', 'submaps', 'deg', 'cm', 'px @2 m mean'))
    for x in across:
        print('%6d %6d %4d->%-4d %8.3f %8.2f %12.2f' % x)
    t = np.array([x[4:] for x in across])
    print('\nboundary tears: median %.2f deg %.2f cm %.1f px | max %.2f deg %.2f cm %.1f px'
          % (*np.median(t, axis=0), *t.max(axis=0)))
    bounds = np.array([x[1] for x in across])
    print('\nheld-out frame -> distance (frames) to nearest owner boundary:')
    print('  ' + '  '.join('%d:%d' % (h, np.abs(bounds - h).min()) for h in held))
    n = len(frames)
    for k in (3, 5, 10):
        near = sum(1 for f in frames if np.abs(bounds - f['index']).min() <= k)
        print('frames within %2d frames of a boundary: %3d of %d (%.0f%%)' % (k, near, n, 100 * near / n))


if __name__ == '__main__':
    main()
