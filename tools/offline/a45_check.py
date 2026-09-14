"""A45: verify, on real splat geometry, that accumulating
    p = dLdPower * dx,   q = dLdPower * dy
per pair and rebuilding once per splat
    dLdMean2D = -( cx*P + cy*Q ,  cy*P + cz*Q )        (P = sum p, Q = sum q)
    dLdConic  = ( -0.5*sum(p*dx), -sum(p*dy), -0.5*sum(q*dy) )
gives the same answer as the kernel's per-pair form, and that float32
accumulation error does not grow.

Geometry is REAL: build 266's model projected into frame 4 with the clamped
projection and the eval 2D filter (lossq_blur.draw_state), every pixel of each
splat's tight tile box, the kernel's cull (alpha >= 1/255, alpha <= 0.99).
dLdAlpha is not on disk, so two upstream fields are used: random N(0,1) per
pixel, and a constant 1 (the maximum-cancellation case: a symmetric footprint
sums its mean gradient to about zero).

Float32 accumulation is simulated sequentially in a random order per splat,
i.e. one arbitrary atomic order. Reference is float64.

Run: cd tools/offline && python -u a45_check.py
"""
import io, json, os
import numpy as np
import project as P
import lossq_blur as B

f32 = np.float32


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    rot, t = P.load_poses()[4]
    R = P.quat_to_matrix(rot)
    g = B.draw_state(col, count, R, t, fx, fy, cx, cy, 0.25)
    ok = ((g['z'] > 0.05) & (g['z'] < 100) & (g['det'] > 1e-12)
          & (g['radius'] >= 0.5) & (g['alpha'] >= P.MIN_ALPHA)
          & (g['mx'] > 0) & (g['mx'] < rw) & (g['my'] > 0) & (g['my'] < rh))
    idx = np.nonzero(ok)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(idx, size=min(800, idx.size), replace=False)
    # conic as float32, exactly as tgConicOpacity holds it (d.conic -> float3)
    rows = {'rand': [], 'const': []}
    conds = []
    for i in pick:
        ccx, ccy, ccz = f32(g['cx'][i]), f32(g['cy'][i]), f32(g['cz'][i])
        op = f32(g['alpha'][i])
        x0 = int(max(0, np.floor(g['mx'][i] - g['ex'][i])))
        x1 = int(min(rw, np.ceil(g['mx'][i] + g['ex'][i])))
        y0 = int(max(0, np.floor(g['my'][i] - g['ey'][i])))
        y1 = int(min(rh, np.ceil(g['my'][i] + g['ey'][i])))
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        dx = (f32(g['mx'][i]) - gx.ravel().astype(f32)).astype(f32)   # delta = xy - pixel
        dy = (f32(g['my'][i]) - gy.ravel().astype(f32)).astype(f32)
        power = (f32(-0.5) * (ccx * dx * dx + ccz * dy * dy) - ccy * dx * dy).astype(f32)
        gau = np.exp(np.minimum(power, 0)).astype(f32)
        alpha = np.minimum(f32(0.99), op * gau)
        keep = alpha >= f32(P.MIN_ALPHA)
        if keep.sum() < 2:
            continue
        dx, dy, gau = dx[keep], dy[keep], gau[keep]
        C = np.array([[g['cx'][i], g['cy'][i]], [g['cy'][i], g['cz'][i]]])
        ev = np.linalg.eigvalsh(C)
        conds.append(ev.max() / max(ev.min(), 1e-300))
        for kind in ('rand', 'const'):
            dLdAlpha = (rng.standard_normal(dx.size) if kind == 'rand'
                        else np.ones(dx.size)).astype(f32)
            dLdPower = (op * dLdAlpha * gau).astype(f32)            # dLdG * dGdPower
            order = rng.permutation(dx.size)
            # ---- float64 reference
            D = dLdPower.astype(np.float64); X = dx.astype(np.float64); Y = dy.astype(np.float64)
            ref_m = np.array([(D * -(g['cx'][i] * X + g['cy'][i] * Y)).sum(),
                              (D * -(g['cz'][i] * Y + g['cy'][i] * X)).sum()])
            ref_c = np.array([(D * -0.5 * X * X).sum(), (D * -X * Y).sum(), (D * -0.5 * Y * Y).sum()])
            scale_m = np.array([np.abs(D * (g['cx'][i] * X + g['cy'][i] * Y)).sum(),
                                np.abs(D * (g['cz'][i] * Y + g['cy'][i] * X)).sum()])
            ref_abs = np.sqrt((D * (g['cx'][i] * X + g['cy'][i] * Y)) ** 2
                              + (D * (g['cz'][i] * Y + g['cy'][i] * X)) ** 2).sum()
            # ---- current kernel, float32, arbitrary order
            m0 = f32(0); m1 = f32(0); c0 = f32(0); c1 = f32(0); c2 = f32(0); a_old = f32(0)
            P_ = f32(0); Q_ = f32(0); A_ = f32(0); Bc = f32(0); Cc = f32(0); a_new = f32(0)
            for j in order:
                d, ex_, ey_ = dLdPower[j], dx[j], dy[j]
                gdx = f32(-(ccx * ex_ + ccy * ey_)); gdy = f32(-(ccz * ey_ + ccy * ex_))
                mx_ = f32(d * gdx); my_ = f32(d * gdy)
                m0 = f32(m0 + mx_); m1 = f32(m1 + my_)
                c0 = f32(c0 + f32(d * f32(f32(-0.5) * ex_ * ex_)))
                c1 = f32(c1 + f32(d * f32(-ex_ * ey_)))
                c2 = f32(c2 + f32(d * f32(f32(-0.5) * ey_ * ey_)))
                a_old = f32(a_old + f32(np.sqrt(mx_ * mx_ + my_ * my_)))
                # ---- proposed
                p = f32(d * ex_); q = f32(d * ey_)
                P_ = f32(P_ + p); Q_ = f32(Q_ + q)
                A_ = f32(A_ + f32(p * ex_)); Bc = f32(Bc + f32(p * ey_)); Cc = f32(Cc + f32(q * ey_))
                u = f32(ccx * p + ccy * q); v = f32(ccy * p + ccz * q)
                a_new = f32(a_new + f32(np.sqrt(u * u + v * v)))
            new_m = np.array([-(ccx * P_ + ccy * Q_), -(ccy * P_ + ccz * Q_)], np.float64)
            new_c = np.array([f32(-0.5) * A_, -Bc, f32(-0.5) * Cc], np.float64)
            old_m = np.array([m0, m1], np.float64)
            old_c = np.array([c0, c1, c2], np.float64)
            rows[kind].append((
                np.abs(old_m - ref_m).max() / max(scale_m.max(), 1e-30),
                np.abs(new_m - ref_m).max() / max(scale_m.max(), 1e-30),
                np.abs(old_c - ref_c).max() / max(np.abs(ref_c).max(), 1e-30),
                np.abs(new_c - ref_c).max() / max(np.abs(ref_c).max(), 1e-30),
                abs(a_old - ref_abs) / max(ref_abs, 1e-30),
                abs(a_new - ref_abs) / max(ref_abs, 1e-30),
                dx.size))
    conds = np.array(conds)
    print('splats checked %d, pairs per splat p50 %d max %d, conic condition p50 %.1f p99 %.1f max %.1f'
          % (len(rows['rand']), np.median([r[6] for r in rows['rand']]),
             max(r[6] for r in rows['rand']), np.median(conds), np.percentile(conds, 99), conds.max()))
    for kind in ('rand', 'const'):
        a = np.array(rows[kind])
        print('\n upstream dLdAlpha = %s' % ('N(0,1) per pixel' if kind == 'rand' else 'constant 1'))
        for k, name in ((0, 'mean2D current '), (1, 'mean2D p/q     '), (2, 'conic current  '),
                        (3, 'conic hoisted  '), (4, 'AbsGS current  '), (5, 'AbsGS p/q      ')):
            print('   %s error (normalised) p50 %.2e  p99 %.2e  max %.2e'
                  % (name, np.median(a[:, k]), np.percentile(a[:, k], 99), a[:, k].max()))
    print('\n mean2D errors are normalised by sum|per-pair term|, the scale of float32 atomic'
          ' error; conic by |reference|; AbsGS by the reference sum.')


if __name__ == '__main__':
    main()
