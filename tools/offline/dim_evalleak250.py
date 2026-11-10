"""How much does evaluateHeldOut dilute the AbsGS denominator? (build 250)

evaluateHeldOut runs trainer_preprocess on the live stats buffer, so every
held-out view that DRAWS a splat adds denom += 1 with no matching absGrad2D
(no backward runs). The densify pass that follows reads absGrad2D / denom.

Per splat: nH = held-out views drawing it (of 12), nT = trained views drawing
it (of 108). One 100-iteration interval draws about nT * 100/108 trained
observations, so the score is scaled by d / (d + nH), d = nT * 100 / 108.
Uses the exported iteration-3600 model for every pass (only model on disk).
"""
import numpy as np

import project as P
import dim_oversize250 as O
from dim_oversize2 import world_covariance
import io, json, os


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx = (rw + P.TILE_W - 1) // P.TILE_W
    ty = (rh + P.TILE_H - 1) // P.TILE_H
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    _, trained, held = O.keyframes(poses)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, _ = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))

    def counts(keys):
        n = np.zeros(count, np.int32)
        for f in keys:
            R = P.quat_to_matrix(poses[f][0])
            ok = O.per_view(mean, sigma, opacity, R, poses[f][1],
                            fx, fy, cx, cy, tx, ty, rw, rh, True)[0]
            n += ok
        return n
    nT = counts(trained)
    nH = counts(held)
    d = nT * (100.0 / len(trained))
    live = nT > 0
    factor = d[live] / (d[live] + nH[live])
    print('splats drawn by >=1 trained view: %d of %d' % (int(live.sum()), count))
    print('held-out views drawing a splat (of %d): p50 %d p90 %d max %d'
          % (len(held), *np.percentile(nH[live], [50, 90, 100]).astype(int)))
    print('score scale factor d/(d+nH): p50 %.4f  p10 %.4f  p1 %.4f  min %.4f'
          % tuple(np.percentile(factor, [50, 10, 1, 0])))
    for thr in (0.95, 0.9, 0.8, 0.5):
        print('  share of drawn splats with factor < %.2f: %.3f%%'
              % (thr, 100.0 * (factor < thr).mean()))
    heldonly = (nT == 0) & (nH > 0)
    print('splats drawn ONLY by held-out views: %d' % int(heldonly.sum()))


if __name__ == '__main__':
    main()
