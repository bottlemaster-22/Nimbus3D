"""DOES IMAGE DETAIL PREDICT WHERE SMALL SPLATS ARE NEEDED?

For every splat, in every frame that has a photograph, this measures:

  footprintStd  the standard deviation of the photograph's luma UNDER the
                splat's own Gaussian footprint. This is the error a splat
                cannot avoid: a Gaussian paints ONE colour over its footprint,
                so whatever the image varies by inside it is error by
                construction. It is the honest definition of "this splat is
                too big".

  nativeGrad    the Sobel gradient the pre-pass ALREADY computes, on the
                256x192 native-resolution luma, sampled at the splat's
                projection. NativeDepthEdgeClassifier.classifyFrame computes
                exactly this number and then throws the magnitude away,
                keeping one bit (`texture`) of it.

  fullGrad      the same Sobel on the full 1920x1440 luma, for reference.

  requiredSigma the largest footprint sigma whose footprintStd stays under a
                target. This is "how small would this splat have to be".

Occlusion is handled with a splat-centre z-buffer at render resolution, so a
splat hidden behind a wall is not credited with the wall's texture.

Run: python -u seedsize.py
"""
import io, json, os, sys
import numpy as np
import cv2

import project as P
import detail as Dt

OUT = os.path.join(Dt.SCRATCH, 'seedsize.npz')
# Footprint scales sampled, in FULL-RESOLUTION pixels. The current field's
# footprints land around 8 px, Scaniverse's would land near 2.
SIGMAS = np.array([0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4])


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    b = Dt.bundle()
    W, H = b['intrinsics']['width'], b['intrinsics']['height']
    NW, NH = b['settings']['depthWidth'], b['settings']['depthHeight']
    FX = b['intrinsics']['fx']
    fxF, fyF, cxF, cyF = P.load_intrinsics(W, H)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    print('full res %dx%d  fx %.1f   native %dx%d  fx %.2f   render %dx%d'
          % (W, H, fxF, NW, NH, fxF * NW / W, rw, rh))

    rows = []
    for idx, path in Dt.frames_with_images():
        rot, t = poses[idx]
        R = P.quat_to_matrix(rot)
        mx, my, z, sa, sb, sc, cam = Dt.geometry_full(
            col, count, R, t, fxF, fyF, cxF, cyF)
        det_before = np.maximum(sa * sc - sb * sb, 1e-16)
        sig_px = det_before ** 0.25            # geometric-mean 1-sigma, px

        inside = (z > 0.2) & (z < 8) & (mx > 2) & (my > 2) & (mx < W - 2) & (my < H - 2)

        # --- occlusion: a splat-centre z-buffer at render resolution --------
        s = rw / float(W)
        zb = np.full((rh, rw), np.inf)
        gx = np.clip((mx[inside] * s).astype(np.int64), 0, rw - 1)
        gy = np.clip((my[inside] * s).astype(np.int64), 0, rh - 1)
        np.minimum.at(zb, (gy, gx), z[inside])
        front = np.full(count, False)
        near = zb[gy, gx]
        front[np.nonzero(inside)[0]] = z[inside] <= near * 1.03 + 0.03
        vis = inside & front

        # --- the photograph -------------------------------------------------
        luma, rgb = Dt.luma_of(path)
        ss = Dt.scale_space(luma, SIGMAS)
        ngrad, nsmall = Dt.native_sobel(luma, NW, NH)
        fgx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        fgy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        fgrad = np.sqrt(fgx * fgx + fgy * fgy) * 0.125

        vx, vy = mx[vis], my[vis]
        # variance of the image at every sampled footprint scale
        var = np.stack([Dt.bilinear(v, vx, vy) for (_, v) in ss], axis=1)
        # this splat's own footprint variance, log-interpolated in sigma
        lg = np.log(np.clip(sig_px[vis], SIGMAS[0], SIGMAS[-1]))
        lgs = np.log(SIGMAS)
        j = np.clip(np.searchsorted(lgs, lg) - 1, 0, len(SIGMAS) - 2)
        w1 = (lg - lgs[j]) / (lgs[j + 1] - lgs[j])
        own = var[np.arange(var.shape[0]), j] * (1 - w1) \
            + var[np.arange(var.shape[0]), j + 1] * w1

        rows.append(dict(
            frame=np.full(vis.sum(), idx, np.int32),
            splat=np.nonzero(vis)[0].astype(np.int32),
            z=z[vis].astype(np.float32),
            sig_px=sig_px[vis].astype(np.float32),
            own_var=own.astype(np.float32),
            var_ladder=var.astype(np.float32),
            ngrad=Dt.bilinear(ngrad, vx * NW / W, vy * NH / H).astype(np.float32),
            fgrad=Dt.bilinear(fgrad, vx, vy).astype(np.float32),
        ))
        print('frame %3d  visible %7d of %d  median sigma %.2f px  median foot std %.4f'
              % (idx, vis.sum(), count, np.median(sig_px[vis]),
                 np.median(np.sqrt(own))), flush=True)

    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out['sigmas'] = SIGMAS
    np.savez_compressed(OUT, **out)
    print('wrote', OUT, out['frame'].shape)


if __name__ == '__main__':
    main()
