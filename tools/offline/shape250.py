"""Build 250 shape-gap experiments. Prints numbers, writes no arrays.

  python shape250.py render [frames...]   item 2 + item 5 (forward, vs photo)
  python shape250.py grad   [frames...]   item 3 + clamp backward mismatch

Forward is the trainer's: EWA projection with the build-250 tangent clamp
(switchable), 0.25 px^2 Mip 2D filter + comp2D, alpha-threshold tile box,
front-to-back composite with T < 1e-4 termination, SH degree 1, no background
(identical for every variant, so deltas are comparable). The exported PLY has
the 3D filter fused in, so comp3D = 1, as in project.py.

Metrics against the real photograph (BOX-resized to 720x540):
  PSNR raw, PSNR after the trainer's closed-form gain/bias fit,
  SSIM8 = MetalSplatTrainer.evaluateHeldOut's 8x8 block luma SSIM, verbatim,
  SSIMg = 11x11 Gaussian luma SSIM (the training loss's window).
"""
import io, json, os, sys, time
import numpy as np
from PIL import Image
import project as P

TILE = 16
SH_C0, SH_C1 = 0.28209479177387814, 0.48860251190291990
INC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"
K11 = np.array([0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554,
                0.26601172, 0.21300554, 0.10936069, 0.03600077, 0.00759876,
                0.00102838])
LUMA_T = np.array([0.2126, 0.7152, 0.0722])      # training SSIM luma
LUMA_E = np.array([0.299, 0.587, 0.114])         # evaluateHeldOut luma

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
SL = census['slices'][0]
RW, RH = SL['renderWidth'], SL['renderHeight']
N = RW * RH
TX, TY = (RW + 15) // 16, (RH + 15) // 16
FX, FY, CX, CY = P.load_intrinsics(RW, RH)
COL, COUNT = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
POSES = P.load_poses()
BUNDLE = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))

MEAN = np.stack([COL['x'], -COL['y'], -COL['z']], 1).astype(np.float64)
LOGS = np.clip(np.stack([COL['scale_0'], COL['scale_1'], COL['scale_2']], 1).astype(np.float64), -12, 3)
Q = np.stack([COL['rot_1'], -COL['rot_2'], -COL['rot_3'], COL['rot_0']], 1).astype(np.float64)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
OPL = COL['opacity'].astype(np.float64)
DC = np.stack([COL['f_dc_0'], COL['f_dc_1'], COL['f_dc_2']], 1).astype(np.float64)
REST = np.stack([COL['f_rest_%d' % i] for i in range(9)], 1).astype(np.float64)


def rotmats(q):
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((q.shape[0], 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_); Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_);     Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_); Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_);     Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    return Rm


RMS = rotmats(Q)


def shape_classes(logs):
    s = np.sort(np.exp(logs), 1)[:, ::-1]
    r21 = s[:, 1] / s[:, 0]
    r32 = s[:, 2] / s[:, 1]
    needle = r21 < 0.4
    disc = (~needle) & (r32 < 0.4)
    return needle, disc, ~(needle | disc)


def eff_rank(logs):
    lam = np.exp(2 * logs)
    p = lam / lam.sum(1, keepdims=True)
    lp = np.log(np.maximum(p, 1e-20))
    H = -(p * lp).sum(1)
    return np.exp(H), p, lp, H


def prior_grad(logs, w, target):
    """trainer_regularizer, verbatim: -2 w (rank-target) rank p_j (log p_j + H)."""
    rank, p, lp, H = eff_rank(logs)
    k = -2.0 * w * (rank - target) * rank
    return k[:, None] * p * (lp + H[:, None])


def geometry(frame, logs, clamp=True):
    rot, t = POSES[frame]
    R = P.quat_to_matrix(rot)
    cam = MEAN @ R.T + t
    z = cam[:, 2]
    iz = 1.0 / np.where(z != 0, z, 1)
    mx = FX * cam[:, 0] * iz + CX
    my = FY * cam[:, 1] * iz + CY
    sc = np.exp(logs)
    M = RMS * sc[:, None, :]
    sig = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    limx = 1.3 * (0.5 * RW / FX)
    limy = 1.3 * (0.5 * RH / FY)
    tx_ = cam[:, 0] * iz
    ty_ = cam[:, 1] * iz
    bind = (np.abs(tx_) > limx) | (np.abs(ty_) > limy)
    if clamp:
        tx_ = np.clip(tx_, -limx, limx)
        ty_ = np.clip(ty_, -limy, limy)
    A = np.zeros((COUNT, 2, 3))
    A[:, 0, 0] = FX * iz
    A[:, 0, 2] = -FX * tx_ * iz
    A[:, 1, 1] = FY * iz
    A[:, 1, 2] = -FY * ty_ * iz
    S2 = A @ sig @ np.transpose(A, (0, 2, 1))
    sa = S2[:, 0, 0]; sb = S2[:, 0, 1]; scc = S2[:, 1, 1]
    detb = np.maximum(sa * scc - sb * sb, 1e-12)
    sa = sa + P.FILTER_2D_VARIANCE
    scc = scc + P.FILTER_2D_VARIANCE
    det = sa * scc - sb * sb
    comp2d = np.sqrt(np.clip(detb / np.maximum(det, 1e-12), 0, 1))
    dsafe = np.maximum(det, 1e-12)
    mid = 0.5 * (sa + scc)
    radius = 3.0 * np.sqrt(np.maximum(mid + np.sqrt(np.maximum(mid * mid - det, 1e-9)), 1e-9))
    opac = 1.0 / (1.0 + np.exp(-OPL)) * comp2d
    lev = np.clip(2.0 * np.log(np.maximum(opac, 1e-30) / P.MIN_ALPHA), 0, 9)
    ex = np.sqrt(lev) * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = np.sqrt(lev) * np.sqrt(np.maximum(scc, 1e-9)) * 1.001
    minx = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    miny = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    maxx = np.minimum(TX, np.ceil((mx + ex) / TILE)).astype(np.int64)
    maxy = np.minimum(TY, np.ceil((my + ey) / TILE)).astype(np.int64)
    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5) & (opac >= P.MIN_ALPHA)
    ok &= (maxx > minx) & (maxy > miny)
    cam_c = -R.T @ t
    d = MEAN - cam_c
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    rgb = SH_C0 * DC + (-SH_C1 * d[:, 1:2] * REST[:, 0:3] + SH_C1 * d[:, 2:3] * REST[:, 3:6]
                        - SH_C1 * d[:, 0:1] * REST[:, 6:9]) + 0.5
    rgb = np.maximum(rgb, 0.0)
    return dict(z=z, mx=mx, my=my, cA=scc / dsafe, cB=-sb / dsafe, cC=sa / dsafe,
                opac=opac, rgb=rgb, ok=ok, minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                bind=bind, cam=cam, A=A, R=R, sig=sig)


def tile_lists(g):
    idx = np.nonzero(g['ok'])[0]
    w = (g['maxx'] - g['minx'])[idx]
    h = (g['maxy'] - g['miny'])[idx]
    n = w * h
    rep = np.repeat(idx, n)
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(rep.size) - start
    ww = np.repeat(w, n)
    tile = (g['miny'][rep] + k // ww) * TX + (g['minx'][rep] + k % ww)
    order = np.lexsort((g['z'][rep], tile))
    rep, tile = rep[order], tile[order]
    bounds = np.searchsorted(tile, np.arange(TX * TY + 1))
    return rep, bounds


def tile_eval(g, o, ti):
    ttx, tty = ti % TX, ti // TX
    GX, GY = np.meshgrid(ttx * TILE + np.arange(TILE), tty * TILE + np.arange(TILE))
    GX, GY = GX.ravel(), GY.ravel()
    dx = (GX + 0.5)[:, None] - g['mx'][o][None, :]
    dy = (GY + 0.5)[:, None] - g['my'][o][None, :]
    pw = -0.5 * (g['cA'][o][None, :] * dx * dx + g['cC'][o][None, :] * dy * dy) \
        - g['cB'][o][None, :] * dx * dy
    gs = np.exp(np.clip(pw, -60, 0))
    al = np.minimum(0.99, g['opac'][o][None, :] * gs)
    al = np.where(al >= P.MIN_ALPHA, al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    done = Tc < 1e-4
    fd = np.where(done.any(1), done.argmax(1), al.shape[1])
    al = np.where(np.arange(al.shape[1])[None, :] <= fd[:, None], al, 0.0)
    Tc = np.cumprod(1.0 - al, axis=1)
    before = np.concatenate([np.ones((al.shape[0], 1)), Tc[:, :-1]], axis=1)
    return GX, GY, dx, dy, gs, al, Tc, before


def render(g):
    rep, bounds = tile_lists(g)
    img = np.zeros((RH, RW, 3))
    T = np.ones((RH, RW))
    for ti in range(TX * TY):
        a, b = bounds[ti], bounds[ti + 1]
        if b <= a:
            continue
        o = rep[a:b]
        GX, GY, dx, dy, gs, al, Tc, before = tile_eval(g, o, ti)
        c = (al * before) @ g['rgb'][o]
        ins = (GX < RW) & (GY < RH)
        img[GY[ins], GX[ins]] = c[ins]
        T[GY[ins], GX[ins]] = Tc[ins, -1]
    return img, T, int(b - 0) if False else int(rep.size)


def photo(frame):
    fr = [f for f in BUNDLE['frames'] if f['index'] == frame][0]
    p = os.path.join(INC, fr['imagePath'].replace('/', os.sep))
    im = Image.open(p).convert('RGB').resize((RW, RH), Image.BOX)
    return np.asarray(im, dtype=np.float64) / 255.0, float(fr['qc']['weight'])


def gblur(p):
    out = np.zeros_like(p)
    for t in range(-5, 6):
        out += K11[t + 5] * p[:, np.clip(np.arange(RW) + t, 0, RW - 1)]
    fin = np.zeros_like(p)
    for t in range(-5, 6):
        fin += K11[t + 5] * out[np.clip(np.arange(RH) + t, 0, RH - 1), :]
    return fin


def metrics(img, gt):
    mse = np.mean((img - gt) ** 2)
    x, y = img.ravel(), gt.ravel()
    n = x.size
    den = n * (x * x).sum() - x.sum() ** 2
    gain = (n * (x * y).sum() - x.sum() * y.sum()) / den
    bias = (y.sum() - gain * x.sum()) / n
    gain = min(max(gain, 0.5), 2.0)          # generous; trainer clamps to its range
    fit = gain * img + bias
    msef = np.mean((fit - gt) ** 2)
    # evaluateHeldOut SSIM, verbatim: exposure-applied luma, 8x8 blocks, population stats
    lr = (fit * LUMA_E).sum(2)
    lt = (gt * LUMA_E).sum(2)
    bh, bw = RH // 8, RW // 8
    L = lr[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8)
    Tt = lt[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8)
    mr, mt = L.mean((1, 3)), Tt.mean((1, 3))
    vr = np.maximum((L * L).mean((1, 3)) - mr * mr, 0)
    vt = np.maximum((Tt * Tt).mean((1, 3)) - mt * mt, 0)
    cv = (L * Tt).mean((1, 3)) - mr * mt
    c1, c2 = 1e-4, 9e-4
    s8 = ((2 * mr * mt + c1) * (2 * cv + c2) / ((mr * mr + mt * mt + c1) * (vr + vt + c2))).mean()
    X = (fit * LUMA_T).sum(2); Y = (gt * LUMA_T).sum(2)
    mux, muy = gblur(X), gblur(Y)
    sxx = np.maximum(gblur(X * X) - mux * mux, 0); syy = np.maximum(gblur(Y * Y) - muy * muy, 0)
    sxy = gblur(X * Y) - mux * muy
    sg = ((2 * mux * muy + c1) * (2 * sxy + c2) / ((mux * mux + muy * muy + c1) * (sxx + syy + c2))).mean()
    return 10 * np.log10(1 / mse), 10 * np.log10(1 / msef), s8, sg


def variants():
    needle, disc, blob = shape_classes(LOGS)
    mu = LOGS.mean(1, keepdims=True)
    toward = mu - LOGS                      # volume-preserving direction to isotropy
    out = [('base (build 250, clamp on)', LOGS, True), ('clamp OFF (build 246 fwd)', LOGS, False)]
    for t in (0.25, 0.5, 1.0):
        L = LOGS.copy(); L[disc] += t * toward[disc]
        out.append(('discs -> iso t=%.2f (vol kept)' % t, L, True))
    L = LOGS.copy(); L[disc] -= 0.25 * toward[disc]
    out.append(('discs -> MORE aniso t=-0.25 (mirror)', L, True))
    rng = np.random.RandomState(7)
    r = rng.normal(size=LOGS.shape); r -= r.mean(1, keepdims=True)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    mag = np.linalg.norm(0.25 * toward, axis=1, keepdims=True)
    L = LOGS.copy(); L[disc] += (r * mag)[disc]
    out.append(('discs NULL random dir, |d| of t=0.25', L, True))
    s = np.exp(LOGS); srt = np.argsort(-s, 1)
    L = LOGS.copy()
    ii = np.nonzero(disc)[0]
    i1, i2 = srt[ii, 1], srt[ii, 2]
    L[ii, i2] = np.maximum(LOGS[ii, i2], LOGS[ii, i1] + np.log(0.78))
    out.append(('discs thicken s3 -> 0.78*s2 (vol grows)', L, True))
    return out, disc


def cmd_render(frames):
    vs, disc = variants()
    print('model %d splats, discs %.1f%%, render %dx%d, frames %s' % (COUNT, 100 * disc.mean(), RW, RH, frames))
    res = {v[0]: [] for v in vs}
    for f in frames:
        gt, _ = photo(f)
        for name, L, cl in vs:
            t0 = time.time()
            g = geometry(f, L, cl)
            img, T, ninst = render(g)
            m = metrics(img, gt)
            res[name].append(m)
            extra = ''
            if name.startswith('base'):
                b = g['ok'] & g['bind']
                w = (g['maxx'] - g['minx']) * (g['maxy'] - g['miny'])
                extra = '  clamp binds on %d drawn splats, %.2f%% of tile instances' % (
                    b.sum(), 100 * w[b].sum() / max(w[g['ok']].sum(), 1))
            if name.startswith('clamp OFF'):
                w = (g['maxx'] - g['minx']) * (g['maxy'] - g['miny'])
                b = g['ok'] & g['bind']
                extra = '  (unclamped: %d bind-splats hold %.2f%% of instances)' % (
                    b.sum(), 100 * w[b].sum() / max(w[g['ok']].sum(), 1))
            print('f%2d %-42s PSNR %6.3f fit %6.3f SSIM8 %.4f SSIMg %.4f  inst %7d %4.0fs%s'
                  % (f, name, m[0], m[1], m[2], m[3], ninst, time.time() - t0, extra), flush=True)
    print('\nMEAN over frames %s (delta vs base)' % frames)
    base = np.mean(res[vs[0][0]], 0)
    for name, _, _ in vs:
        m = np.mean(res[name], 0)
        print('  %-42s PSNR %6.3f (%+.3f) fit %6.3f (%+.3f) SSIM8 %.4f (%+.4f) SSIMg %.4f (%+.4f)'
              % (name, m[0], m[0] - base[0], m[1], m[1] - base[1], m[2], m[2] - base[2], m[3], m[3] - base[3]))


def conic_to_logscale(gC, g, clampJ):
    """trainer_preprocess_backward conic->logScale, with J clamped or not."""
    sa = g['cC']; scc = g['cA']; sb = -g['cB']     # conic -> (sc,-sb,sa)/det
    cA, cB, cC = g['cA'], g['cB'], g['cC']
    cd = cA * cC - cB * cB
    inv = 1.0 / np.where(np.abs(cd) >= 1e-20, cd, 1)
    a = cC * inv; bb = -cB * inv; c = cA * inv
    D = a * c - bb * bb
    i2 = 1.0 / np.maximum(D * D, 1e-20)
    dLda = i2 * (-c * c * gC[:, 0] + bb * c * gC[:, 1] - bb * bb * gC[:, 2])
    dLdb = i2 * (2 * bb * c * gC[:, 0] - (D + 2 * bb * bb) * gC[:, 1] + 2 * a * bb * gC[:, 2])
    dLdc = i2 * (-bb * bb * gC[:, 0] + a * bb * gC[:, 1] - a * a * gC[:, 2])
    Gm = np.zeros((COUNT, 2, 2))
    Gm[:, 0, 0] = dLda; Gm[:, 0, 1] = Gm[:, 1, 0] = 0.5 * dLdb; Gm[:, 1, 1] = dLdc
    cam = g['cam']; iz = 1.0 / np.where(cam[:, 2] != 0, cam[:, 2], 1)
    if clampJ:
        A = g['A']
    else:
        A = np.zeros((COUNT, 2, 3))
        A[:, 0, 0] = FX * iz; A[:, 0, 2] = -FX * cam[:, 0] * iz * iz
        A[:, 1, 1] = FY * iz; A[:, 1, 2] = -FY * cam[:, 1] * iz * iz
    dSc = np.transpose(A, (0, 2, 1)) @ Gm @ A
    R = g['R']
    dSw = R.T @ dSc @ R
    sc = np.exp(LOGS)
    M = RMS * sc[:, None, :]
    RtG = np.transpose(RMS, (0, 2, 1)) @ (2.0 * dSw @ M)
    return np.stack([RtG[:, 0, 0] * sc[:, 0], RtG[:, 1, 1] * sc[:, 1], RtG[:, 2, 2] * sc[:, 2]], 1)


def photometric_dconic(g, gt, qcw):
    """L1 (0.8) + SSIM (0.2) photometric loss -> dL/dconic per splat."""
    img, T, _ = render(g)
    rc = img.reshape(N, 3); G = gt.reshape(N, 3)
    diff = rc - G
    gL1 = qcw * 0.8 / N * np.sign(diff) / 3.0
    X = (rc * LUMA_T).sum(1).reshape(RH, RW); Y = (G * LUMA_T).sum(1).reshape(RH, RW)
    c1, c2 = 1e-4, 9e-4
    mux, muy = gblur(X), gblur(Y)
    sxx = np.maximum(gblur(X * X) - mux * mux, 0); syy = np.maximum(gblur(Y * Y) - muy * muy, 0)
    sxy = gblur(X * Y) - mux * muy
    n1 = 2 * mux * muy + c1; n2 = 2 * sxy + c2; d1 = mux * mux + muy * muy + c1; d2 = sxx + syy + c2
    iD = 1.0 / np.maximum(d1 * d2, 1e-12)
    w = qcw * 0.2 / N
    Cc = -w * (2 * muy * n2 * d1 - 2 * mux * n1 * n2) / np.maximum(d1 * d1 * d2, 1e-12)
    Aa = -w * (-n1 * n2 / np.maximum(d1 * d2 * d2, 1e-12))
    Bb = -w * 2 * n1 * iD
    dLdX = gblur(Cc - 2 * Aa * mux - Bb * muy) + 2 * X * gblur(Aa) + Y * gblur(Bb)
    dLdC = gL1 + dLdX.reshape(N, 1) * LUMA_T[None, :]
    rep, bounds = tile_lists(g)
    gC = np.zeros((COUNT, 3))
    rgb = g['rgb']
    for ti in range(TX * TY):
        a, b = bounds[ti], bounds[ti + 1]
        if b <= a:
            continue
        o = rep[a:b]
        GX, GY, dx, dy, gs, al, Tc, before = tile_eval(g, o, ti)
        ins = (GX < RW) & (GY < RH)
        pix = np.where(ins, GY * RW + GX, 0)
        wgt = al * before
        Pj = np.maximum(Tc, 1e-12)
        suf = np.cumsum((wgt[:, :, None] * rgb[o][None])[:, ::-1], axis=1)[:, ::-1]
        suf = np.concatenate([suf[:, 1:], np.zeros((al.shape[0], 1, 3))], axis=1)
        acc = suf / Pj[:, :, None]
        dA = before * ((rgb[o][None] - acc) * dLdC[pix][:, None, :]).sum(2)
        dA = np.where((al > 0) & ins[:, None], dA, 0.0)
        dP = g['opac'][o][None, :] * dA * gs
        np.add.at(gC, (o, 0), (dP * (-0.5 * dx * dx)).sum(0))
        np.add.at(gC, (o, 1), (dP * (-dx * dy)).sum(0))
        np.add.at(gC, (o, 2), (dP * (-0.5 * dy * dy)).sum(0))
    return gC


def cmd_grad(frames):
    needle, disc, blob = shape_classes(LOGS)
    rank = eff_rank(LOGS)[0]
    print('EFFECTIVE RANK of the exported model (fused scales)')
    for nm, m in (('all', np.ones(COUNT, bool)), ('discs', disc), ('needles', needle), ('blobs', blob)):
        print('  %-8s n %6d  rank p10 %.3f p50 %.3f p90 %.3f' % (nm, m.sum(), *np.percentile(rank[m], [10, 50, 90])))
    srt = np.argsort(-np.exp(LOGS), 1)
    thin = srt[:, 2]                         # index of the smallest axis
    gs_list = []
    t0 = time.time()
    for f in frames:
        gt, qcw = photo(f)
        g = geometry(f, LOGS, True)
        gC = photometric_dconic(g, gt, qcw)
        gl_c = conic_to_logscale(gC, g, True)    # what a consistent backward would give
        gl_u = conic_to_logscale(gC, g, False)   # what build 250's backward computes
        b = g['ok'] & g['bind'] & (np.abs(gC).sum(1) > 0)
        if b.any():
            num = (gl_c[b] * gl_u[b]).sum(1)
            den = np.linalg.norm(gl_c[b], axis=1) * np.linalg.norm(gl_u[b], axis=1) + 1e-300
            rel = np.linalg.norm(gl_u[b] - gl_c[b], axis=1) / np.maximum(np.linalg.norm(gl_c[b], axis=1), 1e-300)
            print('f%2d clamp-bound splats with gradient: %d; cos(consistent, shipped) p10 %.3f p50 %.3f; '
                  '|err|/|grad| p50 %.2f p90 %.2f; sign-disagree %.1f%% of components'
                  % (f, b.sum(), np.percentile(num / den, 10), np.median(num / den),
                     np.median(rel), np.percentile(rel, 90),
                     100 * (np.sign(gl_c[b]) != np.sign(gl_u[b])).mean()), flush=True)
        gs_list.append(np.where((g['ok'] & (np.abs(gC).sum(1) > 0))[:, None], gl_u, np.nan))
        print('f%2d backward done  %.0f s' % (f, time.time() - t0), flush=True)
    Gs = np.stack(gs_list, 0)                      # frames x splats x 3
    seen = np.isfinite(Gs[:, :, 0])
    nseen = seen.sum(0)
    Gz = np.where(np.isfinite(Gs), Gs, 0.0)
    mean = Gz.sum(0) / np.maximum(nseen, 1)[:, None]
    rms = np.sqrt((Gz ** 2).sum(0) / np.maximum(nseen, 1)[:, None])
    use = nseen >= 2
    print('\nsplats with photometric scale gradient in >=2 of %d frames: %d' % (len(frames), use.sum()))
    ii = np.nonzero(use)[0]
    th = thin[ii]
    # Adam in steady state: step ~ lr * E[g] / sqrt(E[g^2]).  Evaluate the
    # expected step on the THIN axis (the one the prior targets) with and
    # without the prior, over the gradient sample across frames.
    def adam_dir(pr):
        m = mean[ii, th] + pr[ii, th]
        v = rms[ii, th] ** 2 + 2 * mean[ii, th] * pr[ii, th] + pr[ii, th] ** 2
        return m / np.sqrt(np.maximum(v, 1e-300))
    base_dir = mean[ii, th] / np.maximum(rms[ii, th], 1e-300)
    print('  photometric-only Adam direction on thin axis (E[g]/rms): p10 %.3f p50 %.3f p90 %.3f'
          % tuple(np.percentile(base_dir, [10, 50, 90])))
    print('  (negative = photometric wants the thin axis to GROW)')
    print('\n  %-22s %-7s %9s %9s %11s %11s %11s' % ('setting', 'class', '|pr|/rms', 'pr>rms %', 'dAdam p50', 'dAdam mean', 'thin grows%'))
    cls = {'discs': disc[ii], 'blobs': blob[ii], 'needles': needle[ii]}
    for w, T in ((0.001, 2.0), (0.002, 2.0), (0.005, 2.0), (0.001, 2.5), (0.001, 3.0), (0.0, 2.0)):
        pr = prior_grad(LOGS, w, T)
        d = adam_dir(pr)
        ratio = np.abs(pr[ii, th]) / np.maximum(rms[ii, th], 1e-300)
        for cn, cm in cls.items():
            print('  w=%.3f target=%.1f    %-7s %9.3f %8.1f%% %+11.4f %+11.4f %10.1f%%'
                  % (w, T, cn, np.median(ratio[cm]), 100 * (ratio[cm] > 1).mean(),
                     np.median((d - base_dir)[cm]), np.mean((d - base_dir)[cm]),
                     100 * (d[cm] < 0).mean()))


if __name__ == '__main__':
    mode = sys.argv[1]
    frames = [int(a) for a in sys.argv[2:]] or [0, 4, 8, 12, 15]
    {'render': cmd_render, 'grad': cmd_grad}[mode](frames)
