"""REFUTATION probes for the 'cacheline' findings.

(A) Which stats cache lines does trainer_preprocess ALREADY dirty every
    iteration?  TrainerShaders.metal:1062-1068 stores visibleFlag (off 24),
    atomic-adds denom (off 4) and atomic-max maxRadiusPxBits (off 8) for
    every splat with tilesTouched > 0.  If that set is a superset of the
    20,625 lines the backward touches, then NO compulsory fill/writeback is
    removed by merging: the line is dirty in the same iteration regardless.

(B) Is a 'falsely shared' stats line actually a COLLISION?  The backward
    walks the tile list in batches of 256 with a threadgroup_barrier between
    batches, and inside a batch every lane is on the SAME j.  Two splats that
    share a 64 B stats line are touched at DIFFERENT j.  Measure how far
    apart, and how often they even land in the same 256-entry batch.

Run: cd tools/offline && python -u refute_cl.py [n_frames] [tile_stride]
"""
import io, json, os, sys, time
import numpy as np
import project as P
from cacheline import geometry, instances, TILE

D = P.D
BATCH = 256


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    keys = sorted(poses.keys())
    step = max(1, len(keys) // n_frames)
    picked = keys[::step][:n_frames]
    print('splats %d  tiles %d  frames %s  stride %d' % (count, tx*ty, picked, stride))

    A_drawn, A_drawnlines, A_cont, A_contlines, A_subset = [], [], [], [], []
    gaps = []          # |position difference| in the depth-sorted tile list
    same_batch = 0
    diff_batch = 0
    pairs_same_batch_le1 = 0
    shared_pairs_total = 0

    for fi in picked:
        t0 = time.time()
        rot, t = poses[fi]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        z, mx, my, cxx, cyy, czz, opac = geometry(col, count, R, t, fx, fy, cx, cy)
        splat, tid, total = instances(ok, mx, my, ex, ey, tx, ty)
        order = np.lexsort((z[splat], tid))
        splat, tid = splat[order], tid[order]
        bounds = np.searchsorted(tid, np.arange(tx * ty + 1))

        px = (np.arange(TILE) + 0.5)
        contributed = np.zeros(count, dtype=bool)
        for tile in range(0, tx * ty, stride):
            s, e = bounds[tile], bounds[tile + 1]
            if e <= s:
                continue
            sel = splat[s:e]
            n = sel.size
            ty_, tx_ = divmod(tile, tx)
            gx = np.tile(tx_ * TILE + px, TILE)
            gy = np.repeat(ty_ * TILE + px, TILE)
            dx = gx[:, None] - mx[sel][None, :]
            dy = gy[:, None] - my[sel][None, :]
            power = -0.5*(cxx[sel][None,:]*dx*dx + czz[sel][None,:]*dy*dy) \
                    - cyy[sel][None,:]*dx*dy
            alpha = np.minimum(0.99, opac[sel][None,:]*np.exp(np.clip(power,-60,0)))
            aok = alpha >= P.MIN_ALPHA
            keepv = np.where(aok, alpha, 0.0)
            T = np.cumprod(1.0 - keepv, axis=1)
            brk = aok & (T < 1e-4)
            has = brk.any(axis=1)
            first = np.argmax(brk, axis=1)
            limit = np.where(has, first, n)
            live = aok & (np.arange(n)[None,:] < limit[:,None])
            any_live = live.any(axis=0)
            contributed[sel[any_live]] = True

            # --- (B) temporal separation of same-line splats --------------
            pos = np.nonzero(any_live)[0]          # position in the tile list
            idx = sel[pos]
            lines = idx // 2
            o = np.argsort(lines, kind='stable')
            L, pp = lines[o], pos[o]
            # runs of equal line
            newrun = np.concatenate([[True], L[1:] != L[:-1]])
            runid = np.cumsum(newrun) - 1
            cnt = np.bincount(runid)
            multi = np.nonzero(cnt > 1)[0]
            if multi.size:
                starts = np.nonzero(newrun)[0]
                for r in multi:                     # cnt is always 2 (32B/64B)
                    a, b = pp[starts[r]], pp[starts[r]+1]
                    g = abs(int(a) - int(b))
                    gaps.append(g)
                    shared_pairs_total += 1
                    if (a // BATCH) == (b // BATCH):
                        same_batch += 1
                        if g <= 1:
                            pairs_same_batch_le1 += 1
                    else:
                        diff_batch += 1

        d_idx = np.nonzero(ok)[0]
        c_idx = np.nonzero(contributed)[0]
        subset = bool(np.all(ok[c_idx]))
        A_drawn.append(d_idx.size)
        A_drawnlines.append(np.unique(d_idx // 2).size)
        A_cont.append(c_idx.size)
        A_contlines.append(np.unique(c_idx // 2).size)
        A_subset.append(subset)
        print('frame %4d  drawn %7d (lines %7d)  contributing %7d (lines %7d)'
              '  contributing subset of drawn: %s   [%.1fs]'
              % (fi, d_idx.size, A_drawnlines[-1], c_idx.size, A_contlines[-1],
                 subset, time.time()-t0))

    print()
    print('(A) STATS LINES ALREADY DIRTIED BY trainer_preprocess EVERY ITERATION')
    dm, dlm = np.mean(A_drawn), np.mean(A_drawnlines)
    cm, clm = np.mean(A_cont), np.mean(A_contlines)
    print('  drawn splats (tilesTouched>0)     mean %9.0f' % dm)
    print('  their distinct 64B stats lines    mean %9.0f  -> %.2f MB fill + %.2f MB writeback'
          % (dlm, dlm*64/1e6, dlm*64/1e6))
    print('  backward-touched stats lines      mean %9.0f  -> %.2f MB' % (clm, clm*64/1e6))
    print('  backward lines / preprocess lines      %9.3f' % (clm/dlm))
    print('  contributing always a subset of drawn: %s' % all(A_subset))
    print()
    print('(B) ARE SAME-LINE SPLATS EVER SIMULTANEOUS IN THE BACKWARD?')
    g = np.array(gaps)
    print('  shared-line splat pairs sampled   %9d' % shared_pairs_total)
    if g.size:
        print('  positions apart in the depth-sorted tile list:')
        print('    min %d  p10 %.0f  median %.0f  mean %.1f  p90 %.0f  max %d'
              % (g.min(), np.percentile(g,10), np.median(g), g.mean(),
                 np.percentile(g,90), g.max()))
        print('    gap == 1 (immediate neighbours in j)  %8d  (%.2f%%)'
              % (int((g==1).sum()), 100.0*(g==1).sum()/g.size))
        print('    gap <= 4                             %8d  (%.2f%%)'
              % (int((g<=4).sum()), 100.0*(g<=4).sum()/g.size))
        print('    gap <= 32 (one SIMD group of drift)   %8d  (%.2f%%)'
              % (int((g<=32).sum()), 100.0*(g<=32).sum()/g.size))
        print('  same 256-entry batch (no barrier between) %8d (%.2f%%)'
              % (same_batch, 100.0*same_batch/g.size))
        print('  DIFFERENT batch: separated by a threadgroup_barrier,')
        print('  so they CANNOT be in flight together      %8d (%.2f%%)'
              % (diff_batch, 100.0*diff_batch/g.size))
        print('  same batch AND adjacent j (gap<=1)        %8d (%.2f%%)'
              % (pairs_same_batch_le1, 100.0*pairs_same_batch_le1/g.size))


if __name__ == '__main__':
    main()
