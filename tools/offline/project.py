"""An OFFLINE copy of trainer_preprocess, so densification changes can be
measured on a real model without waiting for a device run.

It reproduces, in numpy, exactly what TrainerShaders.metal does per splat per
view: the 3D covariance from log-scale and quaternion, the EWA projection to a
2D conic, the Mip-Splatting 2D low-pass and its opacity compensation, the
alpha-threshold tile box, and the tile count. If the tile-instance total it
computes matches the census's peakTileInstances, the projection is right and
anything else it reports can be trusted.

Inputs are all on disk already:
  model/model.ply            the trained Gaussians
  prepass/prepass_result.json  refined world->camera poses
  capture_bundle.json        intrinsics at 1920x1440
  model/train_census.json    render size, splat count, peakTileInstances
"""
import io
import json
import os
import struct

import numpy as np

D = r"C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840"
TILE_W = TILE_H = 16
MIN_ALPHA = 1.0 / 255.0
FILTER_2D_VARIANCE = 0.25          # MetalSplatTrainer.cameraUniforms
FREQ_BLUR_VARIANCE = 0.0           # only non-zero during the blur schedule


# ----------------------------------------------------------------- the model
def load_ply(path):
    f = open(path, 'rb')
    header = b''
    while not header.endswith(b'end_header\n'):
        header += f.read(1)
    names, count = [], 0
    for line in header.decode().split('\n'):
        p = line.split()
        if not p:
            continue
        if p[0] == 'element' and p[1] == 'vertex':
            count = int(p[2])
        elif p[0] == 'property' and p[1] == 'float':
            names.append(p[2])
    data = np.frombuffer(f.read(count * len(names) * 4), dtype='<f4')
    data = data.reshape(count, len(names))
    col = {n: data[:, i] for i, n in enumerate(names)}
    return col, count


# ------------------------------------------------------------------ the views
def quat_to_matrix(q):
    """(x, y, z, w) -> 3x3, normalised first, matching trainer_quatToMatrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_poses():
    r = json.load(io.open(os.path.join(D, 'prepass', 'prepass_result.json'),
                          encoding='utf-8'))
    # `refinedPoses` is a dict keyed by frame index as a STRING.
    out = {}
    for key, p in r['refinedPoses'].items():
        out[int(key)] = (np.asarray(p['rotation'], float),
                         np.asarray(p['translation'], float))
    return out


def load_intrinsics(render_w, render_h):
    b = json.load(io.open(os.path.join(D, 'capture_bundle.json'),
                          encoding='utf-8'))
    k = b['intrinsics']
    sx = render_w / float(k['width'])
    sy = render_h / float(k['height'])
    return (k['fx'] * sx, k['fy'] * sy, k['cx'] * sx, k['cy'] * sy)


# ------------------------------------------------------- the projection itself
def project(col, count, R, t, fx, fy, cx, cy, tiles_x, tiles_y):
    """Returns (visible, radius3sigma_px, tiles_touched, extent_x, extent_y).

    Everything is per splat, with `visible` false where trainer_preprocess
    would have returned early.
    """
    # UNDO THE PLY EXPORT'S FRAME FLIP. PLYCodec writes RDF (right, down,
    # front, the INRIA convention) while the trainer and the refined poses are
    # in RUB (right, up, back), and the flip is exactly "negate y and z" on
    # both the position and the quaternion's imaginary part.
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]

    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    # PLYCodec: rot_0 = w, rot_1 = x, rot_2 = -y, rot_3 = -z.
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

    M = Rm * scale[:, None, :]                    # R * S
    sigma = M @ np.transpose(M, (0, 2, 1))        # world covariance

    # The exported model is draw-ready: fuse3DFilter has already folded the
    # Mip-Splatting 3D filter into logScales and opacity, so comp3D is 1 here.
    sigma_cam = R @ sigma @ R.T

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    j00 = fx * inv_z
    j11 = fy * inv_z
    j02 = -fx * cam[:, 0] * inv_z2
    j12 = -fy * cam[:, 1] * inv_z2

    s00 = sigma_cam[:, 0, 0]
    s01 = sigma_cam[:, 0, 1]
    s02 = sigma_cam[:, 0, 2]
    s11 = sigma_cam[:, 1, 1]
    s12 = sigma_cam[:, 1, 2]
    s22 = sigma_cam[:, 2, 2]

    # rowA = J[0] * sigma_cam, rowB = J[1] * sigma_cam
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b0 = j11 * s01 + j12 * s02
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22

    sa = a0 * j00 + a2 * j02
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12

    det_before = np.maximum(sa * sc - sb * sb, 1e-12)
    low_pass = FILTER_2D_VARIANCE + FREQ_BLUR_VARIANCE
    sa = sa + low_pass
    sc = sc + low_pass
    det = sa * sc - sb * sb
    comp2d = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))

    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))

    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    alpha = opacity * comp2d
    level = np.clip(2.0 * np.log(alpha / max(MIN_ALPHA, 1e-8)), 0, 9)
    k = np.sqrt(level)
    ex = k * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = k * np.sqrt(np.maximum(sc, 1e-9)) * 1.001

    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy

    min_x = np.maximum(0, np.floor((mx - ex) / TILE_W)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / TILE_H)).astype(np.int64)
    max_x = np.minimum(tiles_x, np.ceil((mx + ex) / TILE_W)).astype(np.int64)
    max_y = np.minimum(tiles_y, np.ceil((my + ey) / TILE_H)).astype(np.int64)
    tiles = np.maximum(max_x - min_x, 0) * np.maximum(max_y - min_y, 0)

    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5)
    ok &= alpha >= MIN_ALPHA
    ok &= tiles > 0
    tiles = np.where(ok, tiles, 0)
    return ok, radius, tiles, ex, ey


def main():
    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tiles_x = (rw + TILE_W - 1) // TILE_W
    tiles_y = (rh + TILE_H - 1) // TILE_H
    fx, fy, cx, cy = load_intrinsics(rw, rh)
    col, count = load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = load_poses()
    print('splats %d   render %dx%d   tiles %dx%d   fx %.2f cx %.2f cy %.2f'
          % (count, rw, rh, tiles_x, tiles_y, fx, cx, cy))
    print('census peakTileInstances %d at %d splats = %.2f per splat'
          % (sl['peakTileInstances'], sl['splatCountAtPeakTileInstances'],
             sl['peakTileInstances'] / sl['splatCountAtPeakTileInstances']))
    print('poses: %d' % len(poses))

    keys = sorted(poses.keys())[:40]
    totals = []
    for idx in keys:
        rot, t = poses[idx]
        R = quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = project(
            col, count, R, t, fx, fy, cx, cy, tiles_x, tiles_y
        )
        totals.append((idx, int(ok.sum()), int(tiles.sum())))
    totals.sort(key=lambda r: -r[2])
    print('\nframe   visible    tileInstances   per visible splat')
    for idx, vis, ti in totals[:8]:
        print('%6s %9d %14d %9.2f' % (idx, vis, ti, ti / max(vis, 1)))
    print('...')
    for idx, vis, ti in totals[-3:]:
        print('%6s %9d %14d %9.2f' % (idx, vis, ti, ti / max(vis, 1)))


if __name__ == '__main__':
    main()
