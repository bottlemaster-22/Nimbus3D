"""Does deleting the >Npx splats actually shorten the rasteriser's inner loop?

refute_bigsplat_psnr.py measured that they carry 8.30% of TILE INSTANCES. That
is the sort's input, not the raster's work. A giant near-camera splat is sorted
FIRST in every tile it touches and is opaque, so it terminates other pixels'
loops early. Deleting it removes its own visits but UNCOVERS everything behind
it. This measures the net: inner-loop entries with and without, at the shipped
1e-4 early-out, and the interaction with a raised cutoff.

Run: python -u check_bigsplat_visits.py [threshold_px] [frame ...]
"""
import io, json, os, sys
import numpy as np
import project as P
import refute_earlyout as E

TILE, AREA = 16, 256


def visits(g, taus):
    splat, bounds = g['splat'], g['bounds']
    mx, my = g['mx'], g['my']
    cxk, cyk, czk = g['conic']
    opacity = g['opacity']
    tx, ty = g['tx'], g['ty']
    out = np.zeros(len(taus), np.int64)
    for tid in range(tx * ty):
        a, b = bounds[tid], bounds[tid + 1]
        if b <= a:
            continue
        sel = splat[a:b]
        K = int(sel.size)
        cxt, cyt = tid % tx, tid // tx
        px = cxt * TILE + np.arange(TILE) + 0.5
        py = cyt * TILE + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(px, py)
        gx, gy = gx.ravel(), gy.ravel()
        dx = gx[:, None] - mx[sel][None, :]
        dy = gy[:, None] - my[sel][None, :]
        power = -0.5 * (cxk[sel][None, :] * dx * dx + czk[sel][None, :] * dy * dy) \
            - cyk[sel][None, :] * dx * dy
        al = np.minimum(0.99, opacity[sel][None, :] * np.exp(np.clip(power, -60, 0)))
        al = np.where(al >= P.MIN_ALPHA, al, 0.0)
        ta = np.cumprod(1.0 - al, axis=1)
        for i, tau in enumerate(taus):
            bad = ta < tau
            any_bad = bad.any(axis=1)
            first = np.argmax(bad, axis=1)
            out[i] += int(np.where(any_bad, first + 1, K).sum())
    return out


def main():
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 512.0
    frames = [int(x) for x in sys.argv[2:]] or [0, 4, 8, 12]
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    tx, ty = (rw + 15) // 16, (rh + 15) // 16

    over = np.zeros(count, bool)
    keys = sorted(poses.keys())
    for f in keys[::4]:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        over |= ok & (radius > th)
    print('threshold %.0f px: %d of %d splats (%.4f%%) over on >=1 of %d poses'
          % (th, int(over.sum()), count, 100.0 * over.mean(), len(keys[::4])))
    keep = ~over
    sub = dict((k, v[keep]) for k, v in col.items())
    nsub = int(keep.sum())

    taus = [1e-4, 3e-3]
    A = np.zeros(2, np.int64); B = np.zeros(2, np.int64)
    ia = ib = 0
    for f in frames:
        rot, t = poses[f]
        R = P.quat_to_matrix(rot)
        ga = E.prep(col, count, R, t, fx, fy, cx, cy, rw, rh)
        gb = E.prep(sub, nsub, R, t, fx, fy, cx, cy, rw, rh)
        va = visits(ga, taus); vb = visits(gb, taus)
        A += va; B += vb; ia += ga['total']; ib += gb['total']
        print('  frame %2d  inst %7d->%7d (%+6.2f%%)  visits@1e-4 %10d->%10d (%+6.2f%%)'
              '  visits@3e-3 %10d->%10d (%+6.2f%%)'
              % (f, ga['total'], gb['total'],
                 100.0 * (gb['total'] - ga['total']) / ga['total'],
                 va[0], vb[0], 100.0 * (vb[0] - va[0]) / va[0],
                 va[1], vb[1], 100.0 * (vb[1] - va[1]) / va[1]))
        sys.stdout.flush()
    print('')
    print('TOTAL instances %d -> %d (%+.2f%%)' % (ia, ib, 100.0 * (ib - ia) / ia))
    print('TOTAL visits @1e-4 %d -> %d (%+.2f%%)' % (A[0], B[0], 100.0 * (B[0] - A[0]) / A[0]))
    print('TOTAL visits @3e-3 %d -> %d (%+.2f%%)' % (A[1], B[1], 100.0 * (B[1] - A[1]) / A[1]))
    print('combined lever: visits %d (full,1e-4) -> %d (pruned,3e-3) = %+.2f%%'
          % (A[0], B[1], 100.0 * (B[1] - A[0]) / A[0]))


if __name__ == '__main__':
    main()
