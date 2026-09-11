"""Tile instances per frame (clamped projection, build 266 model), all 868
frames, then the mean per TRAINING ENTRY for each keyframe set: a measured
proxy for per-iteration raster/sort work, not a timing."""
import io, json, os
import numpy as np
import project as P
import kf_exact as K

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
out = os.path.join(os.path.dirname(__file__), '_kf_tiles.npy')
if os.path.exists(out):
    T = np.load(out)
else:
    T = np.zeros(max(poses) + 1, np.int64)
    for idx in sorted(poses):
        R = P.quat_to_matrix(poses[idx][0])
        T[idx] = int(P.project(col, count, R, poses[idx][1], fx, fy, cx, cy, tx, ty)[2].sum())
    np.save(out, T)          # 868 int64, 7 KB
print('census peakTileInstances %d; offline mean over all frames %.0f, max %d'
      % (sl['peakTileInstances'], T.mean(), T.max()))


def per_entry(ch):
    kf = [int(K.IDX[b]) for b, _, _ in ch]
    tr, _ = K.split_held_out(kf)
    return np.mean([T[k] for k in tr])


if __name__ == '__main__':
    import kf_price_sets as S
    base = None
    for name, ch in S.sets():
        m = per_entry(ch)
        base = base or m
        print('%-40s tile instances per training entry %8.0f  (%+.1f%% vs 266)' % (name, m, 100 * (m / base - 1)))
