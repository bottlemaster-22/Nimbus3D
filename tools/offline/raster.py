"""Simulate the rasteriser's INNER LOOP on a sample of tiles.

`project.py` answers questions about tile footprints. This answers questions
about what happens inside a tile, which is where 11 of the 21 ms an iteration
costs actually goes:

  * how many (pixel, Gaussian) pairs are evaluated
  * how many of those are killed by `power > 0` before any exp
  * how many reach the exp and are then thrown away by the alpha test, which
    is the population the `pad0` cutoff removes
  * how far transmittance actually falls, i.e. whether early termination can
    ever fire
  * how many Gaussians genuinely contribute to a pixel

Run:  python tools/offline/raster.py [frame] [tile_sample]
"""
import io
import json
import os
import sys

import numpy as np

import project as P

D = P.D
TILE = 16


def geometry(col, count, R, t, fx, fy, cx, cy):
    """Per-splat screen position, 2D conic and drawn opacity.

    The same quantities trainer_preprocess writes into TrainerSplatRaster,
    kept rather than thrown away so the inner-loop simulations can use them.
    """
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(z != 0, z, 1)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy

    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
    Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sigma_cam = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    j00 = fx * inv_z
    j11 = fy * inv_z
    if P.TANGENT_CLAMP:
        limx, limy = 1.3 * cx / fx, 1.3 * cy / fy
        txc = np.clip(cam[:, 0] * inv_z, -limx, limx) * z
        tyc = np.clip(cam[:, 1] * inv_z, -limy, limy) * z
    else:
        txc, tyc = cam[:, 0], cam[:, 1]
    j02 = -fx * txc * inv_z * inv_z
    j12 = -fy * tyc * inv_z * inv_z
    s00, s01, s02 = sigma_cam[:, 0, 0], sigma_cam[:, 0, 1], sigma_cam[:, 0, 2]
    s11, s12, s22 = sigma_cam[:, 1, 1], sigma_cam[:, 1, 2], sigma_cam[:, 2, 2]
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    inv_det = 1.0 / det
    det_before = np.maximum(
        (sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb,
        1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d
    return {
        'z': z, 'mx': mx, 'my': my,
        'cx': sc * inv_det, 'cy': -sb * inv_det, 'cz': sa * inv_det,
        'opacity': opacity,
    }


def composite(g, order, tile_x, tile_y):
    """Front-to-back over one tile. Returns (alpha_passes, T_before_entry)."""
    px = tile_x * TILE + np.arange(TILE) + 0.5
    py = tile_y * TILE + np.arange(TILE) + 0.5
    gx, gy = np.meshgrid(px, py)
    gx, gy = gx.ravel(), gy.ravel()
    dx = gx[:, None] - g['mx'][order][None, :]
    dy = gy[:, None] - g['my'][order][None, :]
    power = -0.5 * (g['cx'][order][None, :] * dx * dx
                    + g['cz'][order][None, :] * dy * dy)         - g['cy'][order][None, :] * dx * dy
    alpha = np.minimum(0.99, g['opacity'][order][None, :]
                       * np.exp(np.clip(power, -60, 0)))
    alpha_ok = alpha >= P.MIN_ALPHA
    keep = np.where(alpha_ok, alpha, 0.0)
    T = np.cumprod(1.0 - keep, axis=1)
    before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
    return alpha_ok, before


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_tiles = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)

    ok, radius, tiles, ex, ey = P.project(
        col, count, R, t, fx, fy, cx, cy, tx, ty
    )
    print('frame %d: %d drawn, %d tile instances (%.2f per drawn splat)'
          % (frame, ok.sum(), tiles.sum(), tiles.sum() / max(ok.sum(), 1)))

    # Rebuild the per-splat 2D conic and screen position for the drawn set.
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(z != 0, z, 1)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy

    # sa, sb, sc again, this time kept (project.py does not return them).
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
    Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sigma_cam = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    j00 = fx * inv_z
    j11 = fy * inv_z
    if P.TANGENT_CLAMP:
        limx, limy = 1.3 * cx / fx, 1.3 * cy / fy
        txc = np.clip(cam[:, 0] * inv_z, -limx, limx) * z
        tyc = np.clip(cam[:, 1] * inv_z, -limy, limy) * z
    else:
        txc, tyc = cam[:, 0], cam[:, 1]
    j02 = -fx * txc * inv_z * inv_z
    j12 = -fy * tyc * inv_z * inv_z
    s00, s01, s02 = sigma_cam[:, 0, 0], sigma_cam[:, 0, 1], sigma_cam[:, 0, 2]
    s11, s12, s22 = sigma_cam[:, 1, 1], sigma_cam[:, 1, 2], sigma_cam[:, 2, 2]
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
    det = sa * sc - sb * sb
    det = np.where(det > 1e-12, det, 1e-12)
    inv_det = 1.0 / det
    conic_x = sc * inv_det
    conic_y = -sb * inv_det
    conic_z = sa * inv_det

    det_before = np.maximum(
        (sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb,
        1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d

    idx = np.nonzero(ok)[0]
    # Tile ranges, exactly as duplicate_keys builds them.
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)

    rng = np.random.RandomState(0)
    picks = rng.choice(tx * ty, size=min(n_tiles, tx * ty), replace=False)

    tot_pairs = tot_power_ok = tot_alpha_ok = tot_contrib = 0
    t_finals = []
    contrib_counts = []
    for p in picks:
        tile_x, tile_y = int(p % tx), int(p // tx)
        sel = idx[(min_x[idx] <= tile_x) & (max_x[idx] > tile_x)
                  & (min_y[idx] <= tile_y) & (max_y[idx] > tile_y)]
        if sel.size == 0:
            continue
        order = sel[np.argsort(z[sel])]              # front to back, as the sort gives
        px = tile_x * TILE + np.arange(TILE) + 0.5
        py = tile_y * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx = gx.ravel()
        gy = gy.ravel()

        dx = gx[:, None] - mx[order][None, :]
        dy = gy[:, None] - my[order][None, :]
        power = -0.5 * (conic_x[order][None, :] * dx * dx
                        + conic_z[order][None, :] * dy * dy) \
            - conic_y[order][None, :] * dx * dy
        pairs = power.size
        power_ok = power <= 0
        alpha = np.where(power_ok,
                         np.minimum(0.99, opacity[order][None, :]
                                    * np.exp(np.clip(power, -60, 0))), 0.0)
        alpha_ok = power_ok & (alpha >= P.MIN_ALPHA)

        # Front-to-back compositing, so transmittance and contributor counts
        # are the real ones rather than an upper bound.
        keep = np.where(alpha_ok, alpha, 0.0)
        one_minus = 1.0 - keep
        T = np.cumprod(one_minus, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        contributed = alpha_ok & (before > 1e-4)
        tot_pairs += pairs
        tot_power_ok += int(power_ok.sum())
        tot_alpha_ok += int(alpha_ok.sum())
        tot_contrib += int(contributed.sum())
        t_finals.append(T[:, -1])
        contrib_counts.append(contributed.sum(axis=1))

    t_final = np.concatenate(t_finals)
    contrib = np.concatenate(contrib_counts)
    print('\n%d tiles sampled, %d pixels\n' % (len(picks), t_final.size))
    print('PER (PIXEL, GAUSSIAN) PAIR')
    print('  pairs evaluated               %12d' % tot_pairs)
    print('  survive power <= 0            %12d  %6.2f%%'
          % (tot_power_ok, 100 * tot_power_ok / tot_pairs))
    print('  ... and clear the alpha test  %12d  %6.2f%%'
          % (tot_alpha_ok, 100 * tot_alpha_ok / tot_pairs))
    wasted = tot_power_ok - tot_alpha_ok
    print('  exp() evaluated then DISCARDED%12d  %6.2f%% of all exps'
          % (wasted, 100 * wasted / max(tot_power_ok, 1)))
    print('  ^ this is what the pad0 cutoff removes')
    print('  genuinely composited          %12d  %6.2f%%'
          % (tot_contrib, 100 * tot_contrib / tot_pairs))

    print('\nEARLY TERMINATION: the loop stops when T < 1e-4')
    print('  median final T   %.4f' % np.median(t_final))
    print('  pixels reaching T < 1e-4      %6.2f%%'
          % (100 * (t_final < 1e-4).mean()))
    print('  pixels still above T = 0.5    %6.2f%%'
          % (100 * (t_final > 0.5).mean()))
    print('\nCONTRIBUTORS PER PIXEL')
    for q in (50, 90, 99):
        print('  p%-3s %8.1f' % (q, np.percentile(contrib, q)))
    print('  max  %8.1f' % contrib.max())


if __name__ == '__main__':
    main()
