"""Finite-difference check of the dL/dconic -> dL/dlogScale chain that
lossbal_report.py uses, on real splats from the real model and pose.

If the analytic chain is wrong the whole prior-vs-photometric number is wrong,
so this is checked rather than assumed.
"""
import os
import numpy as np
import project as P

FRAME = 4
RW, RH = 720, 540
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
rot, t = poses[FRAME]
R = P.quat_to_matrix(rot)

rng = np.random.RandomState(7)
sel = rng.choice(count, 400, replace=False)

mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)[sel]
camv = mean @ R.T + t
logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1
                        ).astype(np.float64)[sel], -12, 3)
q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)[sel]
q /= np.linalg.norm(q, axis=1, keepdims=True)
x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
n = sel.size
Rm = np.empty((n, 3, 3))
Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)


def conic_of(logscale):
    s = np.exp(logscale)
    M = Rm * s[:, None, :]
    sigC = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    iz = 1.0 / camv[:, 2]
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
    return np.stack([sc / det, -sb / det, sa / det], 1)


base = conic_of(logs)
cA, cB, cC = base[:, 0], base[:, 1], base[:, 2]
dLdConic = rng.randn(n, 3) * 1e-6          # arbitrary upstream gradient


def analytic(dLdConic):
    conicDet = cA * cC - cB * cB
    inv = 1.0 / conicDet
    a = cC * inv
    bb = -cB * inv
    c = cA * inv
    D = a * c - bb * bb
    invD2 = 1.0 / np.maximum(D * D, 1e-20)
    dLda = invD2 * (-c * c * dLdConic[:, 0] + bb * c * dLdConic[:, 1] - bb * bb * dLdConic[:, 2])
    dLdb = invD2 * (2 * bb * c * dLdConic[:, 0] - (D + 2 * bb * bb) * dLdConic[:, 1]
                    + 2 * a * bb * dLdConic[:, 2])
    dLdc = invD2 * (-bb * bb * dLdConic[:, 0] + a * bb * dLdConic[:, 1] - a * a * dLdConic[:, 2])
    Gm = np.zeros((n, 2, 2))
    Gm[:, 0, 0] = dLda
    Gm[:, 0, 1] = 0.5 * dLdb
    Gm[:, 1, 0] = 0.5 * dLdb
    Gm[:, 1, 1] = dLdc
    iz = 1.0 / camv[:, 2]
    A = np.zeros((n, 2, 3))
    A[:, 0, 0] = fx * iz
    A[:, 0, 2] = -fx * camv[:, 0] * iz * iz
    A[:, 1, 1] = fy * iz
    A[:, 1, 2] = -fy * camv[:, 1] * iz * iz
    dLdSigmaCam = np.transpose(A, (0, 2, 1)) @ Gm @ A
    dLdSigmaWorld = R.T @ dLdSigmaCam @ R
    s = np.exp(logs)
    M = Rm * s[:, None, :]
    dLdM = 2.0 * (dLdSigmaWorld @ M)
    RtG = np.transpose(Rm, (0, 2, 1)) @ dLdM
    return np.stack([RtG[:, 0, 0] * s[:, 0], RtG[:, 1, 1] * s[:, 1], RtG[:, 2, 2] * s[:, 2]], 1)


an = analytic(dLdConic)
num = np.zeros((n, 3))
h = 1e-6
for j in range(3):
    lp = logs.copy()
    lm = logs.copy()
    lp[:, j] += h
    lm[:, j] -= h
    num[:, j] = ((conic_of(lp) - conic_of(lm)) * dLdConic).sum(1) / (2 * h)

good = np.isfinite(an).all(1) & np.isfinite(num).all(1) & (np.abs(num).max(1) > 1e-14)
rel = np.abs(an[good] - num[good]) / np.maximum(np.abs(num[good]), 1e-30)
print('finite-difference check of dL/dconic -> dL/dlogScale on %d real splats' % good.sum())
print('  relative error: p50 %.3e  p90 %.3e  p99 %.3e  max %.3e'
      % (*np.percentile(rel, [50, 90, 99]), rel.max()))
print('  correlation analytic vs numeric: %.10f'
      % np.corrcoef(an[good].ravel(), num[good].ravel())[0, 1])
