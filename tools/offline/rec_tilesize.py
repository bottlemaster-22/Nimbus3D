"""Measure what tile size and what 2D low-pass actually cost.

Answers three questions that were REJECTED as unmeasurable:
  1. entry 13: "the actual distribution of draws[].radiusPx, which nothing
     measures; if the median projected radius is nearer 10 px than 24, 8x8
     becomes arguable on time alone."
  2. entry 14: filter2DVariance 0.25 -> 0.1 as a COST lever.
  3. entry 25: the real visible fraction f, which its cache-line arithmetic
     assumed was 0.25-0.5.

Everything comes out of project.project(), which is validated against the
census's peakTileInstances.
"""
import io, json, os, sys
import numpy as np
import project as P

D = P.D

def geom(col, count, R, t, fx, fy, cx, cy, low_pass):
    """project.project() but with the low-pass as a parameter and the tile
    box deferred, so one projection can be scored at several tile sizes."""
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2*(y_*y_ + z_*z_); Rm[:, 0, 1] = 2*(x_*y_ - w_*z_); Rm[:, 0, 2] = 2*(x_*z_ + w_*y_)
    Rm[:, 1, 0] = 2*(x_*y_ + w_*z_); Rm[:, 1, 1] = 1 - 2*(x_*x_ + z_*z_); Rm[:, 1, 2] = 2*(y_*z_ - w_*x_)
    Rm[:, 2, 0] = 2*(x_*z_ - w_*y_); Rm[:, 2, 1] = 2*(y_*z_ + w_*x_); Rm[:, 2, 2] = 1 - 2*(x_*x_ + y_*y_)
    M = Rm * scale[:, None, :]
    sigma = M @ np.transpose(M, (0, 2, 1))
    sigma_cam = R @ sigma @ R.T
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    # tangent clamp, SHIPPED since these findings were written
    limX = 1.3 * (0.5 * P_W / fx)
    limY = 1.3 * (0.5 * P_H / fy)
    txc = np.clip(cam[:, 0] * inv_z, -limX, limX) * z
    tyc = np.clip(cam[:, 1] * inv_z, -limY, limY) * z
    j00 = fx * inv_z; j11 = fy * inv_z
    j02 = -fx * txc * inv_z2; j12 = -fy * tyc * inv_z2
    s00 = sigma_cam[:, 0, 0]; s01 = sigma_cam[:, 0, 1]; s02 = sigma_cam[:, 0, 2]
    s11 = sigma_cam[:, 1, 1]; s12 = sigma_cam[:, 1, 2]; s22 = sigma_cam[:, 2, 2]
    a0 = j00*s00 + j02*s02; a1 = j00*s01 + j02*s12; a2 = j00*s02 + j02*s22
    b0 = j11*s01 + j12*s02; b1 = j11*s11 + j12*s12; b2 = j11*s12 + j12*s22
    sa = a0*j00 + a2*j02
    sb = a1*j11 + a2*j12
    sc = b1*j11 + b2*j12
    det_before = np.maximum(sa*sc - sb*sb, 1e-12)
    sa = sa + low_pass; sc = sc + low_pass
    det = sa*sc - sb*sb
    comp2d = np.sqrt(np.clip(det_before/np.maximum(det, 1e-12), 0, 1))
    mid = 0.5*(sa + sc)
    disc = np.sqrt(np.maximum(mid*mid - det, 1e-9))
    radius = 3.0*np.sqrt(np.maximum(mid + disc, 1e-9))
    opacity = 1.0/(1.0 + np.exp(-col['opacity'].astype(np.float64)))
    alpha = opacity * comp2d
    level = np.clip(2.0*np.log(alpha/max(P.MIN_ALPHA, 1e-8)), 0, 9)
    k = np.sqrt(level)
    ex = k*np.sqrt(np.maximum(sa, 1e-9))*1.001
    ey = k*np.sqrt(np.maximum(sc, 1e-9))*1.001
    mx = fx*cam[:, 0]*inv_z + cx
    my = fy*cam[:, 1]*inv_z + cy
    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5) & (alpha >= P.MIN_ALPHA)
    return ok, mx, my, ex, ey, radius, alpha


def boxes(ok, mx, my, ex, ey, T, W, H):
    tx = (W + T - 1)//T; ty = (H + T - 1)//T
    a = np.maximum(0, np.floor((mx - ex)/T)).astype(np.int64)
    b = np.maximum(0, np.floor((my - ey)/T)).astype(np.int64)
    c = np.minimum(tx, np.ceil((mx + ex)/T)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my + ey)/T)).astype(np.int64)
    n = np.maximum(c - a, 0) * np.maximum(d - b, 0)
    n = np.where(ok, n, 0)
    return n, tx*ty


census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
P_W, P_H = sl['renderWidth'], sl['renderHeight']
fx, fy, cx, cy = P.load_intrinsics(P_W, P_H)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())
print('splats %d  render %dx%d  census peakTileInstances %d' % (count, P_W, P_H, sl['peakTileInstances']))

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 24
use = keys[:NF]
acc = {}
vis_frac = []
allex, alley, allrad = [], [], []
vis_masks = []
for idx in use:
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    for lp_name, lp in (('0.25', 0.25), ('0.10', 0.10)):
        ok, mx, my, ex, ey, rad, alpha = geom(col, count, R, t, fx, fy, cx, cy, lp)
        if lp_name == '0.25':
            n16, _ = boxes(ok, mx, my, ex, ey, 16, P_W, P_H)
            okv = ok & (n16 > 0)
            vis_masks.append(okv.copy())
            allex.append(ex[okv]); alley.append(ey[okv]); allrad.append(rad[okv])
        for T in (8, 16, 32):
            n, ntiles = boxes(ok, mx, my, ex, ey, T, P_W, P_H)
            key = (lp_name, T)
            inst = int(n.sum())
            a = acc.setdefault(key, [0, 0, 0, 0])
            a[0] += inst
            a[1] += int((n > 0).sum())
            a[2] += 1
            a[3] = max(a[3], inst)
        if lp_name == '0.25':
            vis_frac.append(okv.mean())

print()
print('per-frame means over %d frames, splatCount %d' % (len(use), count))
print('lowPass  T   instances   inst/splat   pairTests(T^2*inst)   rel tests   rel inst')
base = None
for lp_name in ('0.25', '0.10'):
    for T in (8, 16, 32):
        inst, vis, nf, mx_ = acc[(lp_name, T)]
        inst /= nf; vis /= nf
        tests = T*T*inst
        if base is None: base = (tests, inst)
        print('%6s %3d %11.0f %10.3f %19.3e %10.3f %9.3f   peakframe %d'
              % (lp_name, T, inst, inst/vis, tests, tests/base[0], inst/base[1], mx_))

ex = np.concatenate(allex); ey = np.concatenate(alley); rad = np.concatenate(allrad)
big = np.maximum(ex, ey); small = np.minimum(ex, ey)
print()
print('projected extents over drawn splats (px, half-extent):')
for nm, v in (('extentX', ex), ('extentY', ey), ('max axis', big), ('min axis', small), ('3sig radius', rad)):
    print('  %-11s p10 %6.2f  p50 %6.2f  p90 %7.2f  p99 %8.2f  mean %7.2f'
          % (nm, *np.percentile(v, [10, 50, 90, 99]), v.mean()))
print('  axis ratio (max/min): p50 %.2f  p90 %.2f' % tuple(np.percentile(big/np.maximum(small,1e-9), [50, 90])))

vm = np.stack(vis_masks)
f = vm.mean()
print()
print('visible fraction f: mean %.4f  min %.4f  max %.4f  (over %d views)'
      % (f, vm.mean(axis=1).min(), vm.mean(axis=1).max(), vm.shape[0]))
print('union over all views %.4f   never visible in any of them %.4f'
      % (vm.any(axis=0).mean(), 1 - vm.any(axis=0).mean()))
for L in (64, 128):
    g = L//48
    # empirical: probability that a run of g consecutive indices is all-invisible
    per = []
    for r in range(vm.shape[0]):
        m = vm[r]
        k = (len(m)//max(g,1))*max(g,1)
        blk = m[:k].reshape(-1, max(g,1))
        per.append((~blk.any(axis=1)).mean())
    print('  %3d-byte line spans %d gaussians: measured P(line fully masked) %.4f  '
          'independent model (1-f)^%d = %.4f' % (L, g, float(np.mean(per)), g, (1-f)**g))
