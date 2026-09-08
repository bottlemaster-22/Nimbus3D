"""trainer_rasterize_backward + the conic half of trainer_preprocess_backward,
in numpy, on a real render, with the photometric and depth channels kept apart.

Depth is carried at UNIT dL/dExpected so the result is a per-unit response; the
real coefficient is exact and is multiplied in by lossbal_report.py.
"""
import io, json, os, sys, time
import numpy as np
from PIL import Image
import project as P

TILE = 16
LUMA = np.array([0.2126, 0.7152, 0.0722])
SH_C0, SH_C1 = 0.28209479177387814, 0.48860251190291990
K11 = np.array([0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554,
                0.26601172, 0.21300554, 0.10936069, 0.03600077, 0.00759876,
                0.00102838])
LAM = 0.2
C1 = 1e-4
C2 = 9e-4
INC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"

frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                           encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
N = RW * RH
SUP = sl['depthSamplesSupervisedTotal'] / sl['depthSupervisionFramesMeasured']
INV_N = 1.0 / N
INV_S = 1.0 / SUP
tx, ty = (RW + 15) // 16, (RH + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
rot, t = poses[frame]
R = P.quat_to_matrix(rot)
cam_center = -R.T @ t

mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
camv = mean @ R.T + t
z = camv[:, 2]
invz = 1.0 / np.where(z != 0, z, 1)
mx = fx * camv[:, 0] * invz + cx
my = fy * camv[:, 1] * invz + cy
scale = np.exp(np.clip(np.stack(
    [col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64), -12, 3))
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
M = Rm * scale[:, None, :]
sigW = M @ np.transpose(M, (0, 2, 1))
sigC = R @ sigW @ R.T
j00 = fx * invz
j11 = fy * invz
j02 = -fx * camv[:, 0] * invz * invz
j12 = -fy * camv[:, 1] * invz * invz
s00, s01, s02 = sigC[:, 0, 0], sigC[:, 0, 1], sigC[:, 0, 2]
s11, s12, s22 = sigC[:, 1, 1], sigC[:, 1, 2], sigC[:, 2, 2]
a0 = j00 * s00 + j02 * s02
a1 = j00 * s01 + j02 * s12
a2 = j00 * s02 + j02 * s22
b1 = j11 * s11 + j12 * s12
b2 = j11 * s12 + j12 * s22
sa = a0 * j00 + a2 * j02 + P.FILTER_2D_VARIANCE
sb = a1 * j11 + a2 * j12
sc = b1 * j11 + b2 * j12 + P.FILTER_2D_VARIANCE
det = np.maximum(sa * sc - sb * sb, 1e-12)
invdet = 1.0 / det
cA, cB, cC = sc * invdet, -sb * invdet, sa * invdet
detb = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb, 1e-12)
comp2d = np.sqrt(np.clip(detb / det, 0, 1))
opac = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * comp2d
dirv = mean - cam_center
dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1).astype(np.float64)
rest = np.stack([col['f_rest_%d' % i] for i in range(9)], 1).astype(np.float64)
rgb = SH_C0 * dc + (-SH_C1 * dirv[:, 1:2] * rest[:, 0:3]
                    + SH_C1 * dirv[:, 2:3] * rest[:, 3:6]
                    - SH_C1 * dirv[:, 0:1] * rest[:, 6:9]) + 0.5
rgb = np.maximum(rgb, 0.0)
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

fwd = np.load(os.path.join(os.path.dirname(__file__), '_fwd_%d.npz' % frame))
rc, rd, rT = fwd['c'], fwd['d'], fwd['T']
alpha_px = 1.0 - rT
safeA = np.maximum(alpha_px, 1e-4)

bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
fr = [f for f in bundle['frames'] if f['index'] == frame][0]
qcw = float(fr['qc']['weight'])
imgp = os.path.join(INC, fr['imagePath'].replace('/', os.sep))
gt = np.asarray(Image.open(imgp).convert('RGB').resize((RW, RH), Image.BOX),
                dtype=np.float64) / 255.0
gt = gt.reshape(N, 3)
print('frame %d  qc.weight %.4f  image %s' % (frame, qcw, os.path.basename(imgp)))

GAIN, BIAS = 1.0, 0.0
rendered = GAIN * rc + BIAS
diff = rendered - gt
mse = ((rendered - gt) ** 2).mean()
print('rendered vs gt: L1 %.4f  PSNR %.2f dB (census trainedPSNR %.2f)'
      % ((np.abs(diff).sum(1) / 3).mean(), 10 * np.log10(1.0 / mse), sl['trainedPSNR']))

wph = qcw * (1.0 - LAM) * INV_N
gL1 = wph * np.sign(diff) / 3.0


def blur(planes):
    p = planes.reshape(-1, RH, RW)
    pad = np.pad(p, ((0, 0), (0, 0), (5, 5)), mode='edge')
    out = np.zeros_like(p)
    for i, k in enumerate(K11):
        out += k * pad[:, :, i:i + RW]
    pad = np.pad(out, ((0, 0), (5, 5), (0, 0)), mode='edge')
    out2 = np.zeros_like(p)
    for i, k in enumerate(K11):
        out2 += k * pad[:, i:i + RH, :]
    return out2.reshape(planes.shape)


X = (rendered * LUMA).sum(1)
Y = (gt * LUMA).sum(1)
bl = blur(np.stack([X, Y, X * X, Y * Y, X * Y], 0))
mux, muy = bl[0], bl[1]
sxx = np.maximum(bl[2] - mux * mux, 0)
syy = np.maximum(bl[3] - muy * muy, 0)
sxy = bl[4] - mux * muy
n1 = 2 * mux * muy + C1
n2 = 2 * sxy + C2
d1 = mux * mux + muy * muy + C1
d2 = sxx + syy + C2
invD = 1.0 / np.maximum(d1 * d2, 1e-12)
ssim = n1 * n2 * invD
wss = qcw * LAM * INV_N
dS_dmux = (2 * muy * n2 * d1 - 2 * mux * n1 * n2) / np.maximum(d1 * d1 * d2, 1e-12)
dS_dsxy = 2 * n1 * invD
dS_dsxx = -n1 * n2 / np.maximum(d1 * d2 * d2, 1e-12)
Cc = -wss * dS_dmux
A = -wss * dS_dsxx
B = -wss * dS_dsxy
bp = blur(np.stack([Cc - 2 * A * mux - B * muy, A, B], 0))
dLdX = bp[0] + 2 * X * bp[1] + Y * bp[2]
gSS = dLdX[:, None] * LUMA[None, :]
import os as _os
gradFinal = gL1 + (0.0 if _os.environ.get('NOSSIM') else 1.0) * gSS
print('mean SSIM %.4f' % ssim.mean())
for nm, g in (('L1', gL1), ('SSIM', gSS), ('total', gradFinal)):
    a = np.abs(g)
    print('  |dL/dC_final| %-6s per channel  p50 %.4e  p90 %.4e  mean %.4e'
          % (nm, np.percentile(a, 50), np.percentile(a, 90), a.mean()))
dLdC = GAIN * gradFinal

DW, DH = 256, 192
uu, vv = np.meshgrid(np.arange(DW), np.arange(DH))
ppx = ((uu + 0.5) * RW / DW).astype(np.int64)
ppy = ((vv + 0.5) * RH / DH).astype(np.int64)
sup_pix = np.unique(ppy.ravel() * RW + ppx.ravel())
supmask = np.zeros(N, bool)
supmask[sup_pix] = True
print('native grid %d samples -> %d distinct render pixels (%.2f%% of the grid); '
      'measured %.0f carry weight' % (DW * DH, sup_pix.size, 100 * sup_pix.size / N, SUP))

cD = supmask.astype(np.float64)          # UNIT dL/dExpected
dLdD = cD / safeA
dLdT_depth = cD * rd / (safeA * safeA)
BG = np.zeros((N, 3))
dLdT_photo = (BG * dLdC).sum(1)

gC_ph = np.zeros((count, 3))
gC_dp = np.zeros((count, 3))
abs_ph = np.zeros(count)
abs_dp = np.zeros(count)
vis = np.zeros(count)
mass = np.zeros(4)          # ph_all, dp_all, ph_on_sup, pairs
nph, ndp, nlev = [], [], []
t0 = time.time()
for ti in range(tx * ty):
    ttx, tty = ti % tx, ti // tx
    sel = idx[(minx[idx] <= ttx) & (maxx[idx] > ttx)
              & (miny[idx] <= tty) & (maxy[idx] > tty)]
    if sel.size == 0:
        continue
    o = sel[np.argsort(z[sel], kind='stable')]
    gxi = ttx * TILE + np.arange(TILE)
    gyi = tty * TILE + np.arange(TILE)
    GX, GY = np.meshgrid(gxi, gyi)
    GX = GX.ravel()
    GY = GY.ravel()
    ins = (GX < RW) & (GY < RH)
    pix = GY * RW + GX
    dx = (GX + 0.5)[:, None] - mx[o][None, :]
    dy = (GY + 0.5)[:, None] - my[o][None, :]
    pw = -0.5 * (cA[o][None, :] * dx * dx + cC[o][None, :] * dy * dy) - cB[o][None, :] * dx * dy
    gsn = np.exp(np.clip(pw, -60, 0))
    al = np.minimum(0.99, opac[o][None, :] * gsn)
    al = np.where(al >= P.MIN_ALPHA, al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    done = Tc < 1e-4
    fd = np.where(done.any(1), done.argmax(1), al.shape[1])
    live = np.arange(al.shape[1])[None, :] <= fd[:, None]
    al = np.where(live, al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    before = np.concatenate([np.ones((256, 1)), Tc[:, :-1]], axis=1)
    wgt = al * before
    Pj = np.maximum(Tc, 1e-12)
    sufC = np.cumsum((wgt[:, :, None] * rgb[o][None, :, :])[:, ::-1, :], axis=1)[:, ::-1, :]
    sufC = np.concatenate([sufC[:, 1:, :], np.zeros((256, 1, 3))], axis=1)
    accC = sufC / Pj[:, :, None]
    sufD = np.cumsum((wgt * z[o][None, :])[:, ::-1], axis=1)[:, ::-1]
    sufD = np.concatenate([sufD[:, 1:], np.zeros((256, 1))], axis=1)
    accD = sufD / Pj
    p = pix.copy()
    p[~ins] = 0
    dc_ = dLdC[p]
    dd_ = dLdD[p]
    tf_ = rT[p]
    tph = dLdT_photo[p]
    tdp = dLdT_depth[p]
    dA_ph = (before * ((rgb[o][None, :, :] - accC) * dc_[:, None, :]).sum(2)
             + (-tf_[:, None] / np.maximum(1.0 - al, 1e-6)) * tph[:, None])
    dA_dp = (before * (z[o][None, :] - accD) * dd_[:, None]
             + (-tf_[:, None] / np.maximum(1.0 - al, 1e-6)) * tdp[:, None])
    live2 = (al > 0) & ins[:, None]
    dA_ph = np.where(live2, dA_ph, 0.0)
    dA_dp = np.where(live2, dA_dp, 0.0)
    if ti % 61 == 0:
        s = live2 & supmask[p][:, None]
        if s.any():
            nph.append(np.abs(dA_ph[s]))
            ndp.append(np.abs(dA_dp[s]))
            nlev.append(np.abs(np.broadcast_to(z[o][None, :], al.shape)[s] - accD[s]))
    mass[0] += np.abs(dA_ph).sum()
    mass[1] += np.abs(dA_dp).sum()
    mass[2] += np.abs(dA_ph[live2 & supmask[p][:, None]]).sum()
    mass[3] += live2.sum()
    dP_ph = opac[o][None, :] * dA_ph * gsn
    dP_dp = opac[o][None, :] * dA_dp * gsn
    k0 = -0.5 * dx * dx
    k1 = -dx * dy
    k2 = -0.5 * dy * dy
    for g, dP in ((gC_ph, dP_ph), (gC_dp, dP_dp)):
        np.add.at(g, (o, 0), (dP * k0).sum(0))
        np.add.at(g, (o, 1), (dP * k1).sum(0))
        np.add.at(g, (o, 2), (dP * k2).sum(0))
    # AbsGS: sum over pixels of |dL/dmean2D|, per channel
    gdx = -(cA[o][None, :] * dx + cB[o][None, :] * dy)
    gdy = -(cC[o][None, :] * dy + cB[o][None, :] * dx)
    for g, dP in ((abs_ph, dP_ph), (abs_dp, dP_dp)):
        np.add.at(g, o, np.hypot(dP * gdx, dP * gdy).sum(0))
    np.add.at(vis, o, wgt.sum(0))
print('backward %.1f s' % (time.time() - t0))
print('\n=== dL/dalpha MASS over the whole frame (sum of |.| over all live pairs) ===')
print('  live pairs                       %d' % mass[3])
print('  photometric, all pixels          %.6e' % mass[0])
print('  photometric, depth-sampled px    %.6e  (%.2f%% of it)' % (mass[2], 100 * mass[2] / mass[0]))
print('  depth, per UNIT dL/dExpected     %.6e' % mass[1])
np.savez(os.path.join(os.path.dirname(__file__), '_bwd_%d%s.npz' % (frame, '_nossim' if _os.environ.get('NOSSIM') else '')),
         gC_ph=gC_ph, gC_dp=gC_dp, cA=cA, cB=cB, cC=cC, idx=idx, camv=camv,
         qcw=qcw, fx=fx, fy=fy, scale=scale, Rm=Rm, Rw=R, sup=sup_pix.size,
         abs_ph=abs_ph, abs_dp=abs_dp, vis=vis, mass=mass,
         nph=np.concatenate(nph), ndp=np.concatenate(ndp))

nph = np.concatenate(nph)
ndp = np.concatenate(ndp)
nlev = np.concatenate(nlev)
print('\n=== (1) dL/dalpha at a DEPTH-SAMPLED pixel, per contributing pair ===')
print('    %d pairs sampled' % nph.size)
print('  |dL/dalpha| photometric               p10 %.4e p50 %.4e p90 %.4e'
      % tuple(np.percentile(nph, [10, 50, 90])))
print('  |dL/dalpha| depth per UNIT dL/dExp    p10 %.4e p50 %.4e p90 %.4e'
      % tuple(np.percentile(ndp, [10, 50, 90])))
print('  |depth - accumDepth| metres           p10 %.4f p50 %.4f p90 %.4f'
      % tuple(np.percentile(nlev, [10, 50, 90])))
print('  ratio of medians per unit dL/dExpected  %.4e' % (np.median(ndp) / np.median(nph)))
