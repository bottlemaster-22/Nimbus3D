"""End-to-end finite-difference check of the whole photometric
dL/dlogScale path: rasterize_backward accumulation AND the conic chain.

Renders one real tile, forms the linearised photometric objective
L = sum_px dot(C_rendered_px, dLdC_px) with dLdC held fixed (which is exactly
what the backward pass differentiates, since sign(diff) is piecewise constant),
then finite-differences L with respect to one splat's logScale and compares
against the analytic value the backward produces.
"""
import io, json, os
import numpy as np
import project as P

TILE = 16
SH_C0, SH_C1 = 0.28209479177387814, 0.48860251190291990
FRAME = 4
RW, RH = 720, 540
tx, ty = (RW + 15) // 16, (RH + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
rot, t = poses[FRAME]
R = P.quat_to_matrix(rot)
cam_center = -R.T @ t

mean_all = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
logs_all = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1
                            ).astype(np.float64), -12, 3)
q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)
q /= np.linalg.norm(q, axis=1, keepdims=True)
x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
Rm_all = np.empty((count, 3, 3))
Rm_all[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
Rm_all[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
Rm_all[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
Rm_all[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
Rm_all[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
Rm_all[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
Rm_all[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
Rm_all[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
Rm_all[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
op_logit = col['opacity'].astype(np.float64)
dcv = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1).astype(np.float64)
rest = np.stack([col['f_rest_%d' % i] for i in range(9)], 1).astype(np.float64)


def geometry(logs):
    camv = mean_all @ R.T + t
    z = camv[:, 2]
    iz = 1.0 / np.where(z != 0, z, 1)
    mx = fx * camv[:, 0] * iz + cx
    my = fy * camv[:, 1] * iz + cy
    s = np.exp(logs)
    M = Rm_all * s[:, None, :]
    sigC = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    j00 = fx * iz
    j11 = fy * iz
    j02 = -fx * camv[:, 0] * iz * iz
    j12 = -fy * camv[:, 1] * iz * iz
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
    detb = np.maximum((sa - P.FILTER_2D_VARIANCE) * (sc - P.FILTER_2D_VARIANCE) - sb * sb, 1e-12)
    comp2d = np.sqrt(np.clip(detb / det, 0, 1))
    opac = (1.0 / (1.0 + np.exp(-op_logit))) * comp2d
    return z, mx, my, np.stack([sc / det, -sb / det, sa / det], 1), opac, camv


dirv = mean_all - cam_center
dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
rgb = SH_C0 * dcv + (-SH_C1 * dirv[:, 1:2] * rest[:, 0:3]
                     + SH_C1 * dirv[:, 2:3] * rest[:, 3:6]
                     - SH_C1 * dirv[:, 0:1] * rest[:, 6:9]) + 0.5
rgb = np.maximum(rgb, 0.0)

z, mx, my, conic, opac, camv = geometry(logs_all)
# pick a busy tile
ttx, tty = 22, 17
gxi = ttx * TILE + np.arange(TILE)
gyi = tty * TILE + np.arange(TILE)
GX, GY = np.meshgrid(gxi, gyi)
GX = GX.ravel().astype(np.float64) + 0.5
GY = GY.ravel().astype(np.float64) + 0.5
near = np.nonzero((np.abs(mx - GX.mean()) < 60) & (np.abs(my - GY.mean()) < 60)
                  & (z > 0.05) & (opac >= P.MIN_ALPHA))[0]
order = near[np.argsort(z[near], kind='stable')]
print('tile (%d,%d): %d candidate splats' % (ttx, tty, order.size))

rng = np.random.RandomState(3)
dLdC = rng.randn(256, 3) * 1e-6           # stand-in for the real per-pixel dL/dC


def render_tile(logs, o):
    z2, mx2, my2, conic2, opac2, _ = geometry(logs)
    dx = GX[:, None] - mx2[o][None, :]
    dy = GY[:, None] - my2[o][None, :]
    pw = -0.5 * (conic2[o, 0][None, :] * dx * dx + conic2[o, 2][None, :] * dy * dy) \
        - conic2[o, 1][None, :] * dx * dy
    gsn = np.exp(np.clip(pw, -60, 0))
    al = np.minimum(0.99, opac2[o][None, :] * gsn)
    al = np.where(al >= P.MIN_ALPHA, al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    done = Tc < 1e-4
    fd = np.where(done.any(1), done.argmax(1), al.shape[1])
    live = np.arange(al.shape[1])[None, :] <= fd[:, None]
    al = np.where(live, al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    before = np.concatenate([np.ones((256, 1)), Tc[:, :-1]], axis=1)
    wgt = al * before
    return wgt @ rgb[o], al, before, wgt, Tc


C0, al, before, wgt, Tc = render_tile(logs_all, order)
Pj = np.maximum(Tc, 1e-12)
sufC = np.cumsum((wgt[:, :, None] * rgb[order][None, :, :])[:, ::-1, :], axis=1)[:, ::-1, :]
sufC = np.concatenate([sufC[:, 1:, :], np.zeros((256, 1, 3))], axis=1)
accC = sufC / Pj[:, :, None]
dx = GX[:, None] - mx[order][None, :]
dy = GY[:, None] - my[order][None, :]
pw = -0.5 * (conic[order, 0][None, :] * dx * dx + conic[order, 2][None, :] * dy * dy) \
    - conic[order, 1][None, :] * dx * dy
gsn = np.exp(np.clip(pw, -60, 0))
dA = before * ((rgb[order][None, :, :] - accC) * dLdC[:, None, :]).sum(2)
dA = np.where(al > 0, dA, 0.0)
dP = opac[order][None, :] * dA * gsn
gC = np.stack([(dP * (-0.5 * dx * dx)).sum(0),
               (dP * (-dx * dy)).sum(0),
               (dP * (-0.5 * dy * dy)).sum(0)], 1)

cA_, cB_, cC_ = conic[order, 0], conic[order, 1], conic[order, 2]
nn = order.size
conicDet = cA_ * cC_ - cB_ * cB_
inv = 1.0 / conicDet
a = cC_ * inv
bb = -cB_ * inv
c = cA_ * inv
D = a * c - bb * bb
invD2 = 1.0 / np.maximum(D * D, 1e-20)
dLda = invD2 * (-c * c * gC[:, 0] + bb * c * gC[:, 1] - bb * bb * gC[:, 2])
dLdb = invD2 * (2 * bb * c * gC[:, 0] - (D + 2 * bb * bb) * gC[:, 1] + 2 * a * bb * gC[:, 2])
dLdc = invD2 * (-bb * bb * gC[:, 0] + a * bb * gC[:, 1] - a * a * gC[:, 2])
Gm = np.zeros((nn, 2, 2))
Gm[:, 0, 0] = dLda
Gm[:, 0, 1] = 0.5 * dLdb
Gm[:, 1, 0] = 0.5 * dLdb
Gm[:, 1, 1] = dLdc
cv = camv[order]
iz = 1.0 / cv[:, 2]
A = np.zeros((nn, 2, 3))
A[:, 0, 0] = fx * iz
A[:, 0, 2] = -fx * cv[:, 0] * iz * iz
A[:, 1, 1] = fy * iz
A[:, 1, 2] = -fy * cv[:, 1] * iz * iz
dSC = np.transpose(A, (0, 2, 1)) @ Gm @ A
dSW = R.T @ dSC @ R
sO = np.exp(logs_all[order])
Mo = Rm_all[order] * sO[:, None, :]
dM = 2.0 * (dSW @ Mo)
RtG = np.transpose(Rm_all[order], (0, 2, 1)) @ dM
ana = np.stack([RtG[:, 0, 0] * sO[:, 0], RtG[:, 1, 1] * sO[:, 1], RtG[:, 2, 2] * sO[:, 2]], 1)

L0 = (C0 * dLdC).sum()
h = 1e-5
picks = np.argsort(-np.abs(ana).sum(1))[:12]
print('\n  splat        axis   analytic        finite-diff     rel.err')
errs = []
for pi in picks:
    for j in range(3):
        lp = logs_all.copy()
        lp[order[pi], j] += h
        lm = logs_all.copy()
        lm[order[pi], j] -= h
        Lp = (render_tile(lp, order)[0] * dLdC).sum()
        Lm = (render_tile(lm, order)[0] * dLdC).sum()
        fd = (Lp - Lm) / (2 * h)
        e = abs(ana[pi, j] - fd) / max(abs(fd), 1e-30)
        errs.append(e)
        print('  %8d      %d    %+.6e   %+.6e   %.2e' % (order[pi], j, ana[pi, j], fd, e))
print('\n  median relative error over %d components: %.3e' % (len(errs), np.median(errs)))
