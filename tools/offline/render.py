"""Full front-to-back render of the trained model into a real frame, in numpy,
so the render can be differenced against the actual photograph.

This exists to VALIDATE the projection before anything is correlated with it:
if the rendered image lines up with the JPEG and the PSNR is in the same
region as the census's own trainedPSNR, then a splat's projection really is
landing on the image content it was fitted to.

Run: python -u render.py [frame]
"""
import io, json, os, sys
import numpy as np
import cv2

import project as P
import detail as Dt

TILE = 16


def render(col, count, R, t, fx, fy, cx, cy, rw, rh):
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)

    mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fx, fy, cx, cy)
    sa = sa + P.FILTER_2D_VARIANCE
    sc = sc + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    inv_det = 1.0 / det
    conic_x = sc * inv_det
    conic_y = -sb * inv_det
    conic_z = sa * inv_det
    det_before = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE)
                            - sb * sb, 1e-12)
    comp2d = np.sqrt(np.clip(det_before / det, 0, 1))
    opacity = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d

    # --- colour: degree-1 SH, dir = normalize(meanWorld - cameraCenter), RUB.
    centre = -R.T @ t
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    d = mean - centre
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1).astype(np.float64)
    rgb = 0.5 + Dt.C0 * dc
    if 'f_rest_0' in col:
        # CHANNEL-MAJOR in the PLY: all R coefficients, then G, then B.
        s1 = np.stack([col['f_rest_0'], col['f_rest_3'], col['f_rest_6']], axis=1)
        s2 = np.stack([col['f_rest_1'], col['f_rest_4'], col['f_rest_7']], axis=1)
        s3 = np.stack([col['f_rest_2'], col['f_rest_5'], col['f_rest_8']], axis=1)
        rgb += (-Dt.C1 * d[:, 1:2] * s1 + Dt.C1 * d[:, 2:3] * s2
                - Dt.C1 * d[:, 0:1] * s3)
    rgb = np.clip(rgb, 0, 1)

    idx = np.nonzero(ok)[0]
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)[idx]
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)[idx]
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)[idx]
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)[idx]
    w = np.maximum(max_x - min_x, 0)
    h = np.maximum(max_y - min_y, 0)
    n = w * h
    keep = n > 0
    idx, min_x, min_y, w, h, n = idx[keep], min_x[keep], min_y[keep], w[keep], h[keep], n[keep]

    total = int(n.sum())
    splat = np.repeat(idx, n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_x = np.repeat(min_x, n) + (j % lw)
    tile_y = np.repeat(min_y, n) + (j // lw)
    tile_id = tile_y * tx + tile_x

    order = np.lexsort((z[splat], tile_id))
    splat = splat[order]
    tile_id = tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))

    out = np.zeros((rh, rw, 3), np.float32)
    alpha_img = np.zeros((rh, rw), np.float32)
    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat[a:b]
        cxt, cyt = tid % tx, tid // tx
        px = cxt * TILE + np.arange(TILE) + 0.5
        py = cyt * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx, gy = gx.ravel(), gy.ravel()
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = -0.5 * (conic_x[sel][None, :] * dx * dx
                        + conic_z[sel][None, :] * dy * dy) \
            - conic_y[sel][None, :] * dx * dy
        al = np.where(power <= 0,
                      np.minimum(0.99, opacity[sel][None, :]
                                 * np.exp(np.clip(power, -60, 0))), 0.0)
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        wgt = al * before
        colr = wgt @ rgb[sel]
        y0, y1 = cyt * TILE, min(cyt * TILE + TILE, rh)
        x0, x1 = cxt * TILE, min(cxt * TILE + TILE, rw)
        block = colr.reshape(TILE, TILE, 3)[:y1 - y0, :x1 - x0]
        out[y0:y1, x0:x1] = block
        alpha_img[y0:y1, x0:x1] = (1 - T[:, -1]).reshape(TILE, TILE)[:y1 - y0, :x1 - x0]
    return out, alpha_img


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    img, alpha = render(col, count, R, t, fx, fy, cx, cy, rw, rh)

    path = dict(Dt.frames_with_images())[frame]
    luma, rgb = Dt.luma_of(path, rw, rh)
    err = np.abs(img - rgb).mean(axis=2)
    mse = ((img - rgb) ** 2).mean()
    print('frame %d  mean|err| %.4f  PSNR %.2f dB  mean alpha %.3f'
          % (frame, err.mean(), 10 * np.log10(1.0 / max(mse, 1e-12)), alpha.mean()))
    np.savez(os.path.join(Dt.SCRATCH, 'render_%d.npz' % frame),
             img=img, rgb=rgb, luma=luma, err=err, alpha=alpha)
    side = np.concatenate([img, rgb], axis=1)
    cv2.imwrite(os.path.join(Dt.SCRATCH, 'render_%d.png' % frame),
                (side[:, :, ::-1] * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
