"""The SEED size distribution, from the pre-pass rule and the scan's own
measured camera-to-surface ranges.

PrePassInitialSplats builds every seed as (radius, radius, third) with
    radius = 0.5 * max(downsampleSpacing, range / depthFX)
so the seed's LARGEST axis is `radius` for every seed whose third axis is
capped below it (0.35*radius trusted, 0.7*radius doubtful - both are).
The only source of spread in the seed's largest axis is therefore `range`,
and only for the ranges where range/depthFX exceeds the downsample spacing.

Ranges are measured, not assumed: distance from each trained splat to the
nearest refined camera centre.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project import load_ply, quat_to_matrix
import numpy as np

D = r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840'
SPACING_M = 0.014793017          # prepass/census.json seeding.spacingMeters
RGB_FX = 1381.8971               # capture_bundle.json intrinsics.fx at 1920
DEPTH_W, RGB_W = 256, 1920
DEPTH_FX = RGB_FX * DEPTH_W / RGB_W


def camera_centres():
    r = json.load(open(os.path.join(D, 'prepass', 'prepass_result.json')))
    poses = r['refinedPoses']
    cs = []
    for k in sorted(poses, key=lambda s: int(s)):
        p = poses[k]
        R = quat_to_matrix(p['rotation'])
        t = np.asarray(p['translation'], float)
        cs.append(-R.T @ t)          # world->camera, so centre is -R^T t
    return np.array(cs)


def main():
    col, n = load_ply(os.path.join(D, 'model', 'model.ply'))
    # PLY is RDF, trainer is RUB: negate y and z on position.
    P = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    C = camera_centres()
    print('cameras', C.shape, 'splats', n, 'depth fx', round(DEPTH_FX, 3))
    print('spacing floor radius = %.3f mm; sensor spacing beats it beyond %.3f m'
          % (500 * SPACING_M, SPACING_M * DEPTH_FX))
    # nearest camera per splat, chunked
    best = np.full(n, np.inf)
    for i in range(0, n, 20000):
        d = np.linalg.norm(P[i:i + 20000, None, :] - C[None, :, :], axis=2)
        best[i:i + 20000] = d.min(axis=1)
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print('range to nearest camera (m):',
          ' '.join('p%d %.2f' % (a, b) for a, b in zip(q, np.percentile(best, q))))
    radius_mm = 1000 * 0.5 * np.maximum(SPACING_M, best / DEPTH_FX)
    print('SEED largest axis (mm):',
          ' '.join('p%d %.3f' % (a, b) for a, b in zip(q, np.percentile(radius_mm, q))))
    p10, p50, p90 = np.percentile(radius_mm, [10, 50, 90])
    print('SEED spread p90/p10 = %.4fx   sd(log10) = %.4f decades'
          % (p90 / p10, np.std(np.log10(radius_mm))))
    print('share of seeds pinned at the spacing floor: %.2f%%'
          % (100 * np.mean(best / DEPTH_FX <= SPACING_M)))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed_radius_mm.npy'),
            radius_mm)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splat_range_m.npy'), best)


if __name__ == '__main__':
    main()
