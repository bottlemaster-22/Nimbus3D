"""Is the front of every ray a fog, and how much of the frame does it own?

The prune window never opened on this run (census ledger: "the prune window was
open in 0 of 39 pass(es)", pruneStartFraction 0.15 of a 30,000 budget = 4,500,
run ended at 4,000), so nothing removed a faint or oversized Gaussian all run.
This measures what is in front as a result.

Run: python -u lossq_fog.py [frame ...]
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import lossq_blur as B

TILE = 16


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    frames = dict(Dt.frames_with_images())
    want = [int(a) for a in sys.argv[1:]] or [0, 8]
    tx, ty = (rw + TILE - 1) // TILE, (rh + TILE - 1) // TILE

    for frame in want:
        rot, t = poses[frame]
        R = P.quat_to_matrix(rot)
        g = B.draw_state(col, count, R, t, fx, fy, cx, cy, 0.25 + 1.33333)
        rgb = B.colours(col, R, t, 2)
        ok = ((g['z'] > 0.05) & (g['z'] < 100) & (g['det'] > 1e-12)
              & (g['radius'] >= 0.5) & (g['alpha'] >= P.MIN_ALPHA))
        z = g['z']
        print('\n=== frame %d ===' % frame)
        print('  splats drawn %d of %d' % (ok.sum(), count))
        zz = z[ok]
        print('  drawn depth z: p1 %.3f p10 %.3f p50 %.3f p90 %.3f m' %
              tuple(np.percentile(zz, [1, 10, 50, 90])))
        for cut in (0.10, 0.20, 0.30, 0.50):
            n = (zz < cut).sum()
            print('    drawn nearer than %.2f m: %6d  (%.3f%% of drawn), '
                  'median screen radius %.1f px, median drawn alpha %.4f'
                  % (cut, n, 100 * n / ok.sum(),
                     np.median(g['radius'][ok][zz < cut]) if n else 0,
                     np.median(g['alpha'][ok][zz < cut]) if n else 0))

        idx = np.nonzero(ok)[0]
        mx, my = g['mx'], g['my']
        min_x = np.maximum(0, np.floor((mx - g['ex']) / TILE)).astype(np.int64)[idx]
        min_y = np.maximum(0, np.floor((my - g['ey']) / TILE)).astype(np.int64)[idx]
        max_x = np.minimum(tx, np.ceil((mx + g['ex']) / TILE)).astype(np.int64)[idx]
        max_y = np.minimum(ty, np.ceil((my + g['ey']) / TILE)).astype(np.int64)[idx]
        w = np.maximum(max_x - min_x, 0)
        h = np.maximum(max_y - min_y, 0)
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

        firsts = []
        Tat = {c: [] for c in (0.10, 0.20, 0.30, 0.50, 1.0)}
        ncontrib_near = []
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
            dx = gx[:, None] - mx[sel][None, :]
            dy = gy[:, None] - my[sel][None, :]
            power = (-0.5 * (g['cx'][sel][None, :] * dx * dx
                             + g['cz'][sel][None, :] * dy * dy)
                     - g['cy'][sel][None, :] * dx * dy)
            al = np.where(power <= 0, np.minimum(0.99, g['alpha'][sel][None, :]
                                                 * np.exp(np.clip(power, -60, 0))), 0.0)
            al = np.where(al >= P.MIN_ALPHA, al, 0.0)
            T = np.cumprod(1.0 - al, axis=1)
            live = al > 0
            has = live.any(1)
            f = np.where(has, np.argmax(live, axis=1), 0)
            firsts.append(np.where(has, z[sel][f], np.nan))
            zs = z[sel]
            for c in Tat:
                k = np.searchsorted(zs, c)
                Tat[c].append(T[:, k - 1] if k > 0 else np.ones(T.shape[0]))
            ncontrib_near.append((live & (zs[None, :] < 0.5)).sum(1))
        firsts = np.concatenate(firsts)
        print('  front-most contributor depth: p10 %.3f p50 %.3f p90 %.3f m'
              % tuple(np.nanpercentile(firsts, [10, 50, 90])))
        for c in sorted(Tat):
            Tc = np.concatenate(Tat[c])
            print('    transmittance still left after everything nearer than %.2f m: '
                  'p50 %.4f  -> that shell absorbs %.2f%% of the ray'
                  % (c, np.median(Tc), 100 * (1 - np.median(Tc))))
        nn = np.concatenate(ncontrib_near)
        print('  contributors nearer than 0.5 m per pixel: p50 %.0f p90 %.0f max %.0f'
              % (np.percentile(nn, 50), np.percentile(nn, 90), nn.max()))


if __name__ == '__main__':
    main()
