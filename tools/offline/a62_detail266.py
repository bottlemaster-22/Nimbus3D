"""A62/A64 on build 266: does local IMAGE detail predict the final trained
splat size, and does it do better than what the seeder's lattice routes on?

What is on disk decides what can be measured:
  - model.ply (266), poses, the 16 photographs of frames 0-15.
  - NOT the seeds (init_splats.ply / .flags stay on the phone), NOT the depth
    or confidence maps, NOT prepass/edges. So each splat's lattice level
    cannot be recovered. The lattice routes on DEPTH geometry (geometric edge
    -> fine/2, confident normal and flat -> coarse x2), so the stand-in here
    is a depth-edge measure taken from the model's own rendered depth.
    It is a PROXY and is labelled as one.

Per frame, the model is rendered (clamped, eval low-pass) and every splat's
contribution sum(alpha*T) is kept. A record is a (frame, splat) pair with
contribution >= 1 pixel; each splat is kept once, from the frame where it
contributes most. Features at the splat's projected centre (720x540):
  sob     Sobel magnitude of the photo's luma
  nsob    the pre-pass's own gradient: Sobel on 256x192 native luma x0.125
  std4    local luma std, Gaussian sigma 4 px (a footprint-scale detail)
  dedge   |grad z| / z of the rendered depth (the lattice's input, proxied)
Target: log10 world largest axis (mm). Controls: range z.

Run: cd tools/offline && python -u a62_detail266.py [workers]
"""
import io, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import cv2

import project as P
import detail as Dt
import lossq_blur as B

TILE = 16
_G = {}


def _init():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    _G.update(rw=rw, rh=rh, col=col, count=count, intr=P.load_intrinsics(rw, rh),
              poses=P.load_poses(), frames=dict(Dt.frames_with_images()))


def render_contrib(col, count, R, t, fx, fy, cx, cy, rw, rh):
    """lossq_blur.render, plus per-splat sum(alpha*T) and an expected-depth image."""
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE
    g = B.draw_state(col, count, R, t, fx, fy, cx, cy, 0.25)
    ok = ((g['z'] > 0.05) & (g['z'] < 100) & (g['det'] > 1e-12)
          & (g['radius'] >= 0.5) & (g['alpha'] >= P.MIN_ALPHA))
    idx = np.nonzero(ok)[0]
    mx, my, z = g['mx'], g['my'], g['z']
    min_x = np.maximum(0, np.floor((mx - g['ex']) / TILE)).astype(np.int64)[idx]
    min_y = np.maximum(0, np.floor((my - g['ey']) / TILE)).astype(np.int64)[idx]
    max_x = np.minimum(tx, np.ceil((mx + g['ex']) / TILE)).astype(np.int64)[idx]
    max_y = np.minimum(ty, np.ceil((my + g['ey']) / TILE)).astype(np.int64)[idx]
    w = np.maximum(max_x - min_x, 0); h = np.maximum(max_y - min_y, 0)
    n = w * h
    keep = n > 0
    idx, min_x, min_y, w, n = idx[keep], min_x[keep], min_y[keep], w[keep], n[keep]
    total = int(n.sum())
    splat = np.repeat(idx, n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    j = np.arange(total) - np.repeat(start, n)
    lw = np.repeat(w, n)
    tile_id = (np.repeat(min_y, n) + (j // lw)) * tx + (np.repeat(min_x, n) + (j % lw))
    order = np.lexsort((z[splat], tile_id))
    splat, tile_id = splat[order], tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(tx * ty + 1))
    contrib = np.zeros(count)
    depth = np.zeros((rh, rw)); acc = np.zeros((rh, rw))
    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat[a:b]
        cxt, cyt = tid % tx, tid // tx
        px = cxt * TILE + np.arange(TILE) + 0.5
        py = cyt * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx, gy = gx.ravel(), gy.ravel()
        inside = (gx < rw) & (gy < rh)
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = (-0.5 * (g['cx'][sel][None, :] * dx * dx + g['cz'][sel][None, :] * dy * dy)
                 - g['cy'][sel][None, :] * dx * dy)
        al = np.minimum(0.99, g['alpha'][sel][None, :] * np.exp(np.clip(power, -60, 0)))
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        T = np.cumprod(1.0 - al, axis=1)
        before = np.concatenate([np.ones((T.shape[0], 1)), T[:, :-1]], axis=1)
        wgt = al * before * inside[:, None]
        np.add.at(contrib, sel, wgt.sum(0))
        y1 = min(cyt * TILE + TILE, rh); x1 = min(cxt * TILE + TILE, rw)
        y0, x0 = cyt * TILE, cxt * TILE
        dimg = (wgt @ z[sel]).reshape(TILE, TILE)
        aimg = wgt.sum(1).reshape(TILE, TILE)
        depth[y0:y1, x0:x1] = dimg[:y1 - y0, :x1 - x0]
        acc[y0:y1, x0:x1] = aimg[:y1 - y0, :x1 - x0]
    return contrib, depth / np.maximum(acc, 1e-6), g


def job(frame):
    G = _G
    fx, fy, cx, cy = G['intr']
    rot, t = G['poses'][frame]
    R = P.quat_to_matrix(rot)
    contrib, dimg, g = render_contrib(G['col'], G['count'], R, t, fx, fy, cx, cy, G['rw'], G['rh'])
    rw, rh = G['rw'], G['rh']
    luma, _ = Dt.luma_of(G['frames'][frame], rw, rh)
    luma = luma.astype(np.float32)
    lb = cv2.GaussianBlur(luma, (0, 0), 1.0)
    sx = cv2.Sobel(lb, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(lb, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(sx * sx + sy * sy) * 0.125
    full, _ = Dt.luma_of(G['frames'][frame])
    ns, _ = Dt.native_sobel(full.astype(np.float32), 256, 192)
    nsob = cv2.resize(ns, (rw, rh), interpolation=cv2.INTER_NEAREST)
    (_, v4), = Dt.scale_space(luma.astype(np.float64), [4.0])
    std4 = np.sqrt(v4)
    d = cv2.GaussianBlur(dimg.astype(np.float32), (0, 0), 1.0)
    dxz = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3); dyz = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    dedge = np.sqrt(dxz * dxz + dyz * dyz) * 0.125 / np.maximum(d, 0.05)
    sel = np.nonzero((contrib >= 1.0) & (g['mx'] >= 0) & (g['mx'] < rw)
                     & (g['my'] >= 0) & (g['my'] < rh))[0]
    X, Y = g['mx'][sel], g['my'][sel]
    return dict(frame=frame, splat=sel, contrib=contrib[sel], z=g['z'][sel],
                sob=Dt.bilinear(sob.astype(np.float64), X, Y),
                nsob=Dt.bilinear(nsob.astype(np.float64), X, Y),
                std4=Dt.bilinear(std4, X, Y),
                dedge=Dt.bilinear(dedge.astype(np.float64), X, Y))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def r2_binned(y, x, nb=10):
    """Variance of y explained by the decile of x (any monotone or not rule)."""
    q = np.searchsorted(np.percentile(x, np.linspace(0, 100, nb + 1)[1:-1]), x)
    m = np.array([y[q == k].mean() if (q == k).any() else 0 for k in range(nb)])
    return 1 - ((y - m[q]) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    frames = sorted(dict(Dt.frames_with_images()))
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(job, frames)
    print('rendered %d frames in %.0f s' % (len(res), time.time() - t0))
    cat = {k: np.concatenate([r[k] if k != 'frame' else np.full(r['splat'].size, r['frame'])
                              for r in res]) for k in ('frame', 'splat', 'contrib', 'z', 'sob',
                                                       'nsob', 'std4', 'dedge')}
    # one record per splat: the frame where it contributes most
    o = np.lexsort((-cat['contrib'], cat['splat']))
    first = np.concatenate([[True], cat['splat'][o][1:] != cat['splat'][o][:-1]])
    k = o[first]
    rec = {n: v[k] for n, v in cat.items()}
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
    big = np.log10(1000 * np.exp(s.max(1)))[rec['splat']]
    fx = P.load_intrinsics(1920, 1440)[0]
    print('records (unique splats contributing >= 1 px in some photo): %d of %d' % (big.size, count))
    print('  largest axis mm p10 %.2f p50 %.2f p90 %.2f   range z p50 %.2f m'
          % (*10 ** np.percentile(big, [10, 50, 90]), np.median(rec['z'])))
    logz = np.log10(rec['z'])
    feats = [('sob  (photo Sobel, 720)', rec['sob']), ('nsob (pre-pass native Sobel)', rec['nsob']),
             ('std4 (luma std, sigma 4 px)', rec['std4']), ('dedge (depth-edge PROXY)', rec['dedge']),
             ('range z (control)', rec['z'])]
    print('\n%-30s %9s %13s %11s' % ('feature', 'Spearman', 'within-z p50', 'R2 decile'))
    zq = np.searchsorted(np.percentile(logz, [25, 50, 75]), logz)
    for name, f in feats:
        within = [spearman(big[zq == q], f[zq == q]) for q in range(4)]
        print('%-30s %+9.4f %+13.4f %11.4f' % (name, spearman(big, f), np.median(within),
                                              r2_binned(big, f)))
    print('\nFINAL SIZE BY IMAGE-DETAIL QUINTILE, within range shells (mm, median largest axis)')
    for name, f in feats[:4]:
        row = []
        for q in range(4):
            m = zq == q
            fq = np.searchsorted(np.percentile(f[m], [20, 40, 60, 80]), f[m])
            row.append([10 ** np.median(big[m][fq == j]) for j in range(5)])
        row = np.array(row)
        print('  %-28s ' % name + '  '.join('%5.2f' % v for v in np.median(row, 0))
              + '   (low detail -> high)')
    print('\n  Scaniverse reference (ledger A64, colour-variation quintiles): 4.93 -> 2.64 mm')
    np.savez_compressed(os.path.join(Dt.SCRATCH, 'a62_records.npz'), big=big, **rec)


if __name__ == '__main__':
    main()
