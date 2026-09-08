"""Last three measurements.

1. Finite-difference check of the trainer_regularizer disc-prior gradient.
2. The alpha-weighted depth SPREAD along a ray, which sets the scale the Huber
   residual actually lives on.
3. What the disc target does, at 2.0 (ours) and 2.61 (the Scaniverse median).
"""
import io, json, os
import numpy as np
import project as P

# ---------- 1. the prior gradient, finite-differenced -----------------------
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1
                        ).astype(np.float64), -12, 3)
rng = np.random.RandomState(11)
sel = rng.choice(count, 2000, replace=False)
L = logs[sel]
W, T = 0.001, 2.0


def loss(l):
    s = np.exp(l)
    lam = s * s
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-20)
    lp = np.log(np.maximum(p, 1e-20))
    H = -(p * lp).sum(1)
    return W * 0.5 * (np.exp(H) - T) ** 2


def grad(l):
    s = np.exp(l)
    lam = s * s
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-20)
    lp = np.log(np.maximum(p, 1e-20))
    H = -(p * lp).sum(1)
    r = np.exp(H)
    k = -4.0 * W * (r - T) * r
    return k[:, None] * p * (lp + H[:, None])


an = grad(L)
h = 1e-6
num = np.zeros_like(an)
for j in range(3):
    lp_ = L.copy()
    lm_ = L.copy()
    lp_[:, j] += h
    lm_[:, j] -= h
    num[:, j] = (loss(lp_) - loss(lm_)) / (2 * h)
ok = np.abs(num) > 1e-16
rel = np.abs(an[ok] - num[ok]) / np.abs(num[ok])
print('1. disc-prior gradient, finite-difference check on %d real splats' % sel.size)
print('   relative error p50 %.3e  p99 %.3e' % (np.percentile(rel, 50), np.percentile(rel, 99)))

# ---------- 2. depth spread along a ray -------------------------------------
census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
tx, ty = (RW + 15) // 16, (RH + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
poses = P.load_poses()
FRAME = 4
rot, t = poses[FRAME]
R = P.quat_to_matrix(rot)
mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
camv = mean @ R.T + t
z = camv[:, 2]
iz = 1.0 / np.where(z != 0, z, 1)
mx = fx * camv[:, 0] * iz + cx
my = fy * camv[:, 1] * iz + cy
scale = np.exp(logs)
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
sigC = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
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
opac = (1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))) * np.sqrt(np.clip(detb / det, 0, 1))
mid = 0.5 * (sa + sc)
disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))
lev = np.clip(2.0 * np.log(opac / max(P.MIN_ALPHA, 1e-8)), 0, 9)
kk = np.sqrt(lev)
ex = kk * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
ey = kk * np.sqrt(np.maximum(sc, 1e-9)) * 1.001
ok2 = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5) & (opac >= P.MIN_ALPHA)
minx = np.maximum(0, np.floor((mx - ex) / 16)).astype(np.int64)
miny = np.maximum(0, np.floor((my - ey) / 16)).astype(np.int64)
maxx = np.minimum(tx, np.ceil((mx + ex) / 16)).astype(np.int64)
maxy = np.minimum(ty, np.ceil((my + ey) / 16)).astype(np.int64)
ok2 &= (maxx > minx) & (maxy > miny)
idx = np.nonzero(ok2)[0]

rs = np.random.RandomState(0)
picks = rs.choice(tx * ty, 120, replace=False)
sd, ex_all, front = [], [], []
for pk in picks:
    ttx, tty = int(pk % tx), int(pk // tx)
    s2 = idx[(minx[idx] <= ttx) & (maxx[idx] > ttx) & (miny[idx] <= tty) & (maxy[idx] > tty)]
    if s2.size == 0:
        continue
    o = s2[np.argsort(z[s2], kind='stable')]
    gx = ttx * 16 + np.arange(16) + 0.5
    gy = tty * 16 + np.arange(16) + 0.5
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
    wg = al * before
    ws = wg.sum(1)
    m1 = wg @ z[o]
    m2 = wg @ (z[o] ** 2)
    good = ws > 0.05
    e = m1[good] / ws[good]
    v = np.maximum(m2[good] / ws[good] - e * e, 0)
    sd.append(np.sqrt(v))
    ex_all.append(e)
    # depth of the first contributor carrying >5% of the weight
    cum = np.cumsum(wg, axis=1)
    first = (cum >= 0.05 * ws[:, None]).argmax(1)
    front.append(np.broadcast_to(z[o][None, :], al.shape)[np.arange(256), first][good])
sd = np.concatenate(sd)
ex_all = np.concatenate(ex_all)
front = np.concatenate(front)
print('\n2. depth structure along a ray, frame %d, %d pixels over %d tiles'
      % (FRAME, sd.size, len(picks)))
print('   alpha-weighted depth STD per pixel (m)  p10 %.4f p50 %.4f p90 %.4f'
      % tuple(np.percentile(sd, [10, 50, 90])))
print('   expected depth (m)                      p10 %.4f p50 %.4f p90 %.4f'
      % tuple(np.percentile(ex_all, [10, 50, 90])))
print('   expected - front-surface depth (m)      p10 %.4f p50 %.4f p90 %.4f'
      % tuple(np.percentile(ex_all - front, [10, 50, 90])))
sig0, kr = 0.004, 0.0015
sigma_at = sig0 + kr * ex_all * ex_all
print('   Huber delta = max(sigma, 4mm), sigma from the noise model at that range:')
print('     delta (mm)                            p10 %.2f p50 %.2f p90 %.2f'
      % tuple(np.percentile(np.maximum(sigma_at, 0.004) * 1000, [10, 50, 90])))
print('     depth STD / delta                     p10 %.1f p50 %.1f p90 %.1f'
      % tuple(np.percentile(sd / np.maximum(sigma_at, 0.004), [10, 50, 90])))

# ---------- 3. what the target does -----------------------------------------
print('\n3. the disc target, against the Scaniverse bar')
s = np.exp(logs)
lam = s * s
p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-20)
lp = np.log(np.maximum(p, 1e-20))
H = -(p * lp).sum(1)
rank = np.exp(H)
for T2 in (2.0, 2.61, 3.0):
    r2 = rank - T2
    k2 = -4.0 * 0.001 * r2 * rank
    g2 = k2[:, None] * p * (lp + H[:, None])
    order = np.argsort(s, axis=1)
    gs = np.take_along_axis(g2, order, axis=1)
    print('   target %.2f : |grad| p50 %.3e   median residual %+.4f   '
          'smallest axis pushed %s on %.1f%% of splats'
          % (T2, np.median(np.abs(g2)), np.median(r2),
             'UP' if np.median(gs[:, 0]) < 0 else 'DOWN',
             100 * (gs[:, 0] > 0).mean() if np.median(gs[:, 0]) > 0 else 100 * (gs[:, 0] < 0).mean()))
print('   Scaniverse median effective rank 2.6101, ours 2.0405, our target 2.0')
