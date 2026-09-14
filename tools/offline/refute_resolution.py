"""Is the error model's floor measured at a resolution the trainer can see?

seedsize.py measures the footprint variance on the FULL 1920x1440 photograph.
The trainer optimises against a 720x540 render. Downsampling removes exactly
the high frequencies the footprint-variance model counts as unavoidable error.
This recomputes the same floor on the 720x540 image the loss actually sees.
"""
import io, json, os
import numpy as np
import cv2
import project as P, detail as Dt

SIGMAS = np.array([0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4])

b = Dt.bundle()
W, H = b['intrinsics']['width'], b['intrinsics']['height']
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]; rw, rh = sl['renderWidth'], sl['renderHeight']
fxF, fyF, cxF, cyF = P.load_intrinsics(W, H)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
s = rw / float(W)

acc = {}
for idx, path in Dt.frames_with_images():
    rot, t = poses[idx]; R = P.quat_to_matrix(rot)
    mx, my, z, sa, sb, sc, cam = Dt.geometry_full(col, count, R, t, fxF, fyF, cxF, cyF)
    sig_px = np.maximum(sa * sc - sb * sb, 1e-16) ** 0.25
    inside = (z > 0.2) & (z < 8) & (mx > 2) & (my > 2) & (mx < W - 2) & (my < H - 2)
    zb = np.full((rh, rw), np.inf)
    gx = np.clip((mx[inside] * s).astype(np.int64), 0, rw - 1)
    gy = np.clip((my[inside] * s).astype(np.int64), 0, rh - 1)
    np.minimum.at(zb, (gy, gx), z[inside])
    front = np.full(count, False)
    front[np.nonzero(inside)[0]] = z[inside] <= zb[gy, gx] * 1.03 + 0.03
    vis = inside & front

    luma_full, _ = Dt.luma_of(path)
    luma_rnd = cv2.resize(luma_full, (rw, rh), interpolation=cv2.INTER_AREA)

    for tag, img, sc_px, xx, yy in (
        ('full 1920x1440', luma_full, sig_px[vis], mx[vis], my[vis]),
        ('render 720x540', luma_rnd, sig_px[vis] * s, mx[vis] * s, my[vis] * s)):
        ss = Dt.scale_space(img, SIGMAS * (1.0 if tag.startswith('full') else s))
        var = np.stack([Dt.bilinear(v, xx, yy) for (_, v) in ss], 1)
        SG = SIGMAS * (1.0 if tag.startswith('full') else s)
        lg = np.log(np.clip(sc_px, SG[0], SG[-1])); lgs = np.log(SG)
        j = np.clip(np.searchsorted(lgs, lg) - 1, 0, len(SG) - 2)
        w1 = (lg - lgs[j]) / (lgs[j + 1] - lgs[j])
        own = var[np.arange(var.shape[0]), j] * (1 - w1) + var[np.arange(var.shape[0]), j + 1] * w1
        wt = sig_px[vis] ** 2
        a = acc.setdefault(tag, [0.0, 0.0])
        a[0] += float((own * wt).sum()); a[1] += float(wt.sum())
    print('frame %d done' % idx, flush=True)

print()
for tag, (num, den) in acc.items():
    r = np.sqrt(num / den)
    print('%-16s  model floor RMS %.4f  = %.2f dB' % (tag, r, 20 * np.log10(1 / r)))
print('census trainedPSNR %.2f dB (720x540), heldOutPSNR %.2f dB'
      % (sl['trainedPSNR'], sl['heldOutPSNR']))
