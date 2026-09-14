"""Render the model from a real camera pose and put it beside the photograph.

The census reports a held-out PSNR. That number is computed on frames whose
photographs are NOT on this machine (held-out are 18, 47, 94, ...; only frames
0 to 15 have images), so it cannot be checked here directly. What CAN be
checked is whether the model, rendered from a pose we do have a photo for,
looks like the photo, and what PSNR that pairing scores.

If the render looks bad and still scores ~20 dB, the metric is not measuring
what a person sees, and that is worth knowing before another constant is tuned
against it.

Run:  python tools/offline/showme.py [frame]
Writes compare_<frame>.png: photo on the left, render on the right.
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import raster as R

TILE = 16


def render(col, count, g, ok, ex, ey, mx, my, tx, ty, rw, rh, order_all):
    """Front-to-back composite of every tile. Returns an (rh, rw, 3) image."""
    out = np.zeros((rh, rw, 3), dtype=np.float64)
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1)
    rgb = np.clip(0.5 + 0.282095017 * dc.astype(np.float64), 0, 1)

    idx = np.nonzero(ok)[0]
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)

    for ty_i in range(ty):
        for tx_i in range(tx):
            sel = idx[(min_x[idx] <= tx_i) & (max_x[idx] > tx_i)
                      & (min_y[idx] <= ty_i) & (max_y[idx] > ty_i)]
            if sel.size == 0:
                continue
            order = sel[np.argsort(g['z'][sel])]
            alpha_ok, before = R.composite(g, order, tx_i, ty_i)
            px = tx_i * TILE + np.arange(TILE)
            py = ty_i * TILE + np.arange(TILE)
            gx, gy = np.meshgrid(px, py)
            gx, gy = gx.ravel(), gy.ravel()

            dx = (gx[:, None] + 0.5) - g['mx'][order][None, :]
            dy = (gy[:, None] + 0.5) - g['my'][order][None, :]
            power = -0.5 * (g['cx'][order][None, :] * dx * dx
                            + g['cz'][order][None, :] * dy * dy) \
                - g['cy'][order][None, :] * dx * dy
            alpha = np.minimum(0.99, g['opacity'][order][None, :]
                               * np.exp(np.clip(power, -60, 0)))
            w = np.where(alpha_ok, alpha, 0.0) * before          # alpha * T
            tile = w @ rgb[order]                                # (256, 3)
            inside = (gy < rh) & (gx < rw)
            out[gy[inside], gx[inside]] = tile[inside]
    return out


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    plypath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P.D, 'model', 'model.ply')
    col, count = P.load_ply(plypath)
    poses = P.load_poses()
    rot, t = poses[frame]
    Rm = P.quat_to_matrix(rot)

    ok, radius, tiles, ex, ey = P.project(
        col, count, Rm, t, fx, fy, cx, cy, tx, ty
    )
    g = R.geometry(col, count, Rm, t, fx, fy, cx, cy)
    img = render(col, count, g, ok, ex, ey, g['mx'], g['my'], tx, ty, rw, rh, None)

    bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'),
                               encoding='utf-8'))
    path = None
    for f in bundle['frames']:
        if f['index'] == frame:
            path = f['imagePath']
    src = os.path.join(r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840", path)
    from PIL import Image
    photo = Image.open(src).convert('RGB').resize((rw, rh), Image.LANCZOS)
    truth = np.asarray(photo, dtype=np.float64) / 255.0

    mse = float(np.mean((img - truth) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    # The same closed-form gain/bias the trainer fits for held-out frames.
    x, y = img.ravel(), truth.ravel()
    n = x.size
    den = n * (x * x).sum() - x.sum() ** 2
    gain = (n * (x * y).sum() - x.sum() * y.sum()) / den if den > 1e-9 else 1.0
    bias = (y.sum() - gain * x.sum()) / n
    fit = np.clip(gain * img + bias, 0, 1)
    mse_f = float(np.mean((fit - truth) ** 2))
    psnr_f = 10 * np.log10(1.0 / max(mse_f, 1e-12))

    print('frame %d   %d splats drawn of %d' % (frame, int(ok.sum()), count))
    print('  raw PSNR            %6.2f dB' % psnr)
    print('  exposure-fitted     %6.2f dB   (gain %.3f bias %+.3f)'
          % (psnr_f, gain, bias))
    print('  mean render / photo  %.3f / %.3f' % (img.mean(), truth.mean()))

    tag = sys.argv[3] if len(sys.argv) > 3 else 'cur'
    out = os.path.join(os.path.dirname(__file__), 'r_%s_%d.npy' % (tag, frame))
    np.save(out, np.clip(img, 0, 1))
    np.save(os.path.join(os.path.dirname(__file__), 'r_truth_%d.npy' % frame), truth)
    print('  saved %s' % out)


if __name__ == '__main__':
    main()
