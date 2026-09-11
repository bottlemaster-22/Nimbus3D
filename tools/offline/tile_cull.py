"""How many tile instances an EXACT ellipse-tile test would remove.

trainer_preprocess gives each splat every tile in the axis-aligned box of its
alpha-threshold ellipse. For an elongated, rotated splat most of that box is
empty. The rasteriser still loads every (splat, tile) instance and runs all
256 pixel threads over it, rejecting the empty ones with the power cutoff.

A tile can be dropped with NO change to the output when no pixel centre in it
can pass the rasteriser's cutoff test:  power >= log(minAlpha / opacity) - 0.01
i.e.  q(d) = a dx^2 + 2 b dx dy + c dy^2 <= 2 ln(opacity / minAlpha) + 0.02,
with (a, b, c) the conic. Tested here against the rectangle spanning the tile's
pixel CENTRES (conservative), with a further safety margin on the level.

Reports, per view and in total: instances now, instances after the exact test,
and the reduction. Uses the clamped projection copied from project.py.
"""
import os
import sys

import numpy as np

import project as P

MARGIN = 0.05          # extra level-set slack on top of the device's 0.01


def project_full(col, count, R, t, fx, fy, cx, cy, tiles_x, tiles_y):
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tiles_x, tiles_y)
    # Recompute the pieces project() does not return (same arithmetic).
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], axis=1)
                           .astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q.T
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_); Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_); Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_); Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_); Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_); Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_); Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sc_ = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    inv_z = 1.0 / z
    j00, j11 = fx * inv_z, fy * inv_z
    limx, limy = 1.3 * cx / fx, 1.3 * cy / fy
    txc = np.clip(cam[:, 0] * inv_z, -limx, limx) * z
    tyc = np.clip(cam[:, 1] * inv_z, -limy, limy) * z
    j02, j12 = -fx * txc * inv_z * inv_z, -fy * tyc * inv_z * inv_z
    s00, s01, s02, s11, s12, s22 = sc_[:, 0, 0], sc_[:, 0, 1], sc_[:, 0, 2], sc_[:, 1, 1], sc_[:, 1, 2], sc_[:, 2, 2]
    a0 = j00 * s00 + j02 * s02; a1 = j00 * s01 + j02 * s12; a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12; b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
    sb = a1 * j11 + a2 * j12
    scc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
    det_before = np.maximum((sa - P.FILTER_2D_VARIANCE) * (scc - P.FILTER_2D_VARIANCE) - sb * sb, 1e-12)
    det = sa * scc - sb * sb
    comp = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))
    alpha = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64))) * comp
    conic = np.stack([scc / det, -sb / det, sa / det], axis=1)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    level = 2.0 * np.log(np.maximum(alpha, 1e-30) / P.MIN_ALPHA) + 0.02 + MARGIN
    return ok, tiles, ex, ey, mx, my, conic, level


def edge_min(a, b, c, X, Y0, Y1):
    """min over dy in [Y0,Y1] of a X^2 + 2 b X dy + c dy^2."""
    dy = np.clip(-b * X / c, Y0, Y1)
    return a * X * X + 2 * b * X * dy + c * dy * dy


def exact_count(ok, ex, ey, mx, my, conic, level, tiles_x, tiles_y):
    idx = np.nonzero(ok)[0]
    mx, my, ex, ey = mx[idx], my[idx], ex[idx], ey[idx]
    a, b, c = conic[idx, 0], conic[idx, 1], conic[idx, 2]
    L = level[idx]
    x0 = np.maximum(0, np.floor((mx - ex) / 16)).astype(np.int64)
    y0 = np.maximum(0, np.floor((my - ey) / 16)).astype(np.int64)
    x1 = np.minimum(tiles_x, np.ceil((mx + ex) / 16)).astype(np.int64)
    y1 = np.minimum(tiles_y, np.ceil((my + ey) / 16)).astype(np.int64)
    nx, ny = np.maximum(x1 - x0, 0), np.maximum(y1 - y0, 0)
    n = nx * ny
    total = int(n.sum())
    # Expand to (splat, tile) pairs.
    rep = np.repeat(np.arange(len(idx)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(total) - start
    tx = x0[rep] + k % nx[rep]
    ty = y0[rep] + k // nx[rep]
    # Pixel-centre rectangle of the tile, relative to the mean.
    X0 = tx * 16 + 0.5 - mx[rep]; X1 = tx * 16 + 15.5 - mx[rep]
    Y0 = ty * 16 + 0.5 - my[rep]; Y1 = ty * 16 + 15.5 - my[rep]
    A, B, C = a[rep], b[rep], c[rep]
    inside = (X0 <= 0) & (X1 >= 0) & (Y0 <= 0) & (Y1 >= 0)
    m = np.minimum(np.minimum(edge_min(A, B, C, X0, Y0, Y1), edge_min(A, B, C, X1, Y0, Y1)),
                   np.minimum(edge_min(C, B, A, Y0, X0, X1), edge_min(C, B, A, Y1, X0, X1)))
    keep = inside | (m <= L[rep])
    return total, int(keep.sum())


def main():
    rw, rh = 720, 540
    tiles_x, tiles_y = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = [int(a) for a in sys.argv[1:]] or list(range(0, 435, 36))
    print('splats %d  render %dx%d  views %s' % (count, rw, rh, frames))
    T0 = T1 = 0
    for f in frames:
        rot, t = poses[f]
        ok, tiles, ex, ey, mx, my, conic, level = project_full(
            col, count, P.quat_to_matrix(rot), t, fx, fy, cx, cy, tiles_x, tiles_y)
        n0, n1 = exact_count(ok, ex, ey, mx, my, conic, level, tiles_x, tiles_y)
        T0 += n0; T1 += n1
        print('frame %4d  visible %7d  instances %9d -> %9d  (-%.1f%%)'
              % (f, ok.sum(), n0, n1, 100 * (1 - n1 / max(n0, 1))))
    print('TOTAL instances %d -> %d   reduction %.1f%%' % (T0, T1, 100 * (1 - T1 / max(T0, 1))))


if __name__ == '__main__':
    main()
