"""How well does the exported model actually fit each of the frames whose
image survived on disk? Needed to bracket the photometric gradient magnitude:
a better fit means a smaller SSIM gradient and therefore a larger depth ratio.
"""
import io, json, os, sys
import numpy as np
from PIL import Image
import project as P

TILE = 16
SH_C0, SH_C1 = 0.28209479177387814, 0.48860251190291990
INC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"
RW, RH = 720, 540
N = RW * RH
tx, ty = (RW + 15) // 16, (RH + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
mean_all = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1
                                ).astype(np.float64), -12, 3))
q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)
q /= np.linalg.norm(q, axis=1, keepdims=True)
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
MM = Rm * scale[:, None, :]
sigW = MM @ np.transpose(MM, (0, 2, 1))
op_logit = col['opacity'].astype(np.float64)
dcv = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1).astype(np.float64)
rest = np.stack([col['f_rest_%d' % i] for i in range(9)], 1).astype(np.float64)

frames = [int(a) for a in sys.argv[1:]] or [0, 2, 6, 8, 10, 14]
print('frame   qc      PSNR(gain=1)   PSNR(best gain+bias)   gain   bias')
for frame in frames:
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    cc = -R.T @ t
    camv = mean_all @ R.T + t
    z = camv[:, 2]
    iz = 1.0 / np.where(z != 0, z, 1)
    mx = fx * camv[:, 0] * iz + cx
    my = fy * camv[:, 1] * iz + cy
    sigC = R @ sigW @ R.T
    j00 = fx * iz
    j11 = fy * iz
    j02 = -fx * camv[:, 0] * iz * iz
    j12 = -fy * camv[:, 1] * iz * iz
    a0 = j00 * sigC[:, 0, 0] + j02 * sigC[:, 0, 2]
    a1 = j00 * sigC[:, 0, 1] + j02 * sigC[:, 1, 2]
    a2 = j00 * sigC[:, 0, 2] + j02 * sigC[:, 2, 2]
    b1 = j11 * sigC[:, 1, 1] + j12 * sigC[:, 1, 2]
    b2 = j11 * sigC[:, 1, 2] + j12 * sigC[:, 2, 2]
    sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
    det = np.maximum(sa * sc - sb * sb, 1e-12)
    cA, cB, cC = sc / det, -sb / det, sa / det
    detb = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb, 1e-12)
    opac = (1.0 / (1.0 + np.exp(-op_logit))) * np.sqrt(np.clip(detb / det, 0, 1))
    dv = mean_all - cc
    dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-6)
    rgb = np.maximum(SH_C0 * dcv + (-SH_C1 * dv[:, 1:2] * rest[:, 0:3]
                                    + SH_C1 * dv[:, 2:3] * rest[:, 3:6]
                                    - SH_C1 * dv[:, 0:1] * rest[:, 6:9]) + 0.5, 0.0)
    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))
    lev = np.clip(2.0 * np.log(opac / max(P.MIN_ALPHA, 1e-8)), 0, 9)
    kk = np.sqrt(lev)
    ex = kk * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = kk * np.sqrt(np.maximum(sc, 1e-9)) * 1.001
    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5) & (opac >= P.MIN_ALPHA)
    minx = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    miny = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    maxx = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
    maxy = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)
    ok &= (maxx > minx) & (maxy > miny)
    idx = np.nonzero(ok)[0]
    out = np.zeros((N, 3))
    for ti in range(tx * ty):
        ttx, tty = ti % tx, ti // tx
        s2 = idx[(minx[idx] <= ttx) & (maxx[idx] > ttx) & (miny[idx] <= tty) & (maxy[idx] > tty)]
        if s2.size == 0:
            continue
        o = s2[np.argsort(z[s2], kind='stable')]
        gx = ttx * TILE + np.arange(TILE) + 0.5
        gy = tty * TILE + np.arange(TILE) + 0.5
        GX, GY = np.meshgrid(gx, gy)
        GX = GX.ravel()
        GY = GY.ravel()
        dx = GX[:, None] - mx[o][None, :]
        dy = GY[:, None] - my[o][None, :]
        pw = -0.5 * (cA[o][None, :] * dx * dx + cC[o][None, :] * dy * dy) - cB[o][None, :] * dx * dy
        al = np.minimum(0.99, opac[o][None, :] * np.exp(np.clip(pw, -60, 0)))
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        Tc = np.cumprod(1.0 - al, axis=1)
        done = Tc < 1e-4
        fd = np.where(done.any(1), done.argmax(1), al.shape[1])
        live = np.arange(al.shape[1])[None, :] <= fd[:, None]
        al = np.where(live, al, 0.0)
        Tc = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((256, 1)), Tc[:, :-1]], axis=1)
        pix = (GY - 0.5).astype(np.int64) * RW + (GX - 0.5).astype(np.int64)
        ins = ((GX - 0.5).astype(np.int64) < RW) & ((GY - 0.5).astype(np.int64) < RH)
        cvals = (al * before) @ rgb[o]
        out[pix[ins]] = cvals[ins]
    fr = [f for f in bundle['frames'] if f['index'] == frame][0]
    gt = np.asarray(Image.open(os.path.join(INC, fr['imagePath'].replace('/', os.sep))
                               ).convert('RGB').resize((RW, RH), Image.BOX),
                    dtype=np.float64).reshape(N, 3) / 255.0
    p1 = 10 * np.log10(1.0 / ((out - gt) ** 2).mean())
    A = np.stack([out.ravel(), np.ones(out.size)], 1)
    sol, *_ = np.linalg.lstsq(A, gt.ravel(), rcond=None)
    g = np.clip(sol[0], 0.9, 1.1)
    b = sol[1]
    p2 = 10 * np.log10(1.0 / ((g * out + b - gt) ** 2).mean())
    print('%5d  %.4f   %8.2f dB      %8.2f dB        %.3f  %+.3f'
          % (frame, fr['qc']['weight'], p1, p2, g, b))
