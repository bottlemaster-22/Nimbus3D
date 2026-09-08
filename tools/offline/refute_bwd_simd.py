"""REFUTATION HARNESS for the backwardwork findings.

Two things bwdwork.py / bwdcheck.py do not do:

  1. SAMPLING VARIANCE.  bwdwork samples 60 of 1530 tiles at a fixed seed and
     scales by a ratio estimator.  Re-run at several sample sizes / seeds and
     see how far the headline populations move.

  2. SIMD MASKING.  bwdcheck multiplies the body instruction count by the
     number of CONTRIBUTING LANES (75.1 M).  On a 32-wide SIMD machine a lane
     that hits `continue` is MASKED, not skipped: the group still issues the
     instruction and the pipeline is still 32 lanes wide.  The real lane-slot
     cost of the body is (slots where >=1 lane is active) * 32.

Run:  python -u refute_bwd_simd.py [tiles] [frame] [seed_offset]
"""
import io
import json
import os
import sys

import numpy as np

import project as P
import raster as Rz

D = P.D
TILE = 16
AREA = 256
SIMD = 32
NSIMD = AREA // SIMD


def run_tile(g, order, tile_x, tile_y, minAlpha, acc):
    n = order.size
    px = tile_x * TILE + np.arange(TILE) + 0.5
    py = tile_y * TILE + np.arange(TILE) + 0.5
    gx, gy = np.meshgrid(px, py)
    gx, gy = gx.ravel(), gy.ravel()
    dx = gx[:, None] - g['mx'][order][None, :]
    dy = gy[:, None] - g['my'][order][None, :]
    cx = g['cx'][order][None, :]
    cy = g['cy'][order][None, :]
    cz = g['cz'][order][None, :]
    power = -0.5 * (cx * dx * dx + cz * dy * dy) - cy * dx * dy

    op = np.maximum(g['opacity'][order], 1e-8)
    cutoff = np.float64(np.float16(np.log(minAlpha / op) - 0.01))[None, :]
    pass_cut = power >= cutoff
    alpha = np.minimum(0.99, op[None, :] * np.exp(np.clip(power, -80, 0)))
    alpha_ok = pass_cut & (alpha >= minAlpha)

    keep = np.where(alpha_ok, alpha, 0.0)
    Tafter = np.cumprod(1.0 - keep, axis=1)
    stop = alpha_ok & (Tafter < 1e-4)
    has_stop = stop.any(axis=1)
    stop_pos = np.where(has_stop, np.argmax(stop, axis=1), n)
    colidx = np.arange(n)[None, :]
    used = alpha_ok & (colidx < stop_pos[:, None])
    any_used = used.any(axis=1)
    lastc = np.where(any_used, n - np.argmax(used[:, ::-1], axis=1), 0)

    fwd_eval = np.where(has_stop, stop_pos + 1, n)

    lane_group = np.arange(AREA) // SIMD
    deepest = np.zeros(AREA, dtype=np.int64)
    for gi in range(NSIMD):
        m = lane_group == gi
        deepest[m] = lastc[m].max()
    nb = np.minimum(n, np.ceil(deepest / AREA).astype(np.int64) * AREA)
    nb = np.where(deepest > 0, nb, 0)

    reached = colidx < lastc[:, None]
    reach_exp = pass_cut & reached

    acc['pairs_total'] += AREA * n
    acc['fwd_eval'] += int(fwd_eval.sum())
    acc['bwd_iter'] += int(nb.sum())
    acc['bwd_skip'] += int((nb - lastc).sum())
    acc['bwd_power'] += int(lastc.sum())
    acc['bwd_exp'] += int(reach_exp.sum())
    acc['bwd_full'] += int(used.sum())
    acc['pixels'] += AREA

    for gi in range(NSIMD):
        m = lane_group == gi
        nbg = int(nb[m][0])
        if nbg == 0:
            continue
        acc['slot_iter'] += nbg
        rch = reached[m][:, :nbg]
        rex = reach_exp[m][:, :nbg]
        usd = used[m][:, :nbg]
        acc['slot_pass_gi'] += int(rch.any(axis=0).sum())
        acc['slot_pass_exp'] += int(rex.any(axis=0).sum())
        acc['slot_full'] += int(usd.any(axis=0).sum())
        acc['lane_pass_gi'] += int(rch.sum())
        acc['lane_pass_exp'] += int(rex.sum())
        acc['lane_full'] += int(usd.sum())
        pj = usd.sum(axis=0)
        act = pj > 0
        acc['occ_hist'] += np.bincount(pj[act], minlength=SIMD + 1)[:SIMD + 1]


def main():
    n_tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seed_off = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    minAlpha = P.MIN_ALPHA

    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    g = Rz.geometry(col, count, R, t, fx, fy, cx, cy)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy,
                                          tx, ty)
    inst = int(tiles.sum())
    idx = np.nonzero(ok)[0]
    mx, my, z = g['mx'], g['my'], g['z']
    min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)

    acc = dict.fromkeys(
        ['pairs_total', 'fwd_eval', 'bwd_iter', 'bwd_skip', 'bwd_power',
         'bwd_exp', 'bwd_full', 'pixels', 'slot_iter', 'slot_pass_gi',
         'slot_pass_exp', 'slot_full', 'lane_pass_gi', 'lane_pass_exp',
         'lane_full'], 0)
    acc['occ_hist'] = np.zeros(SIMD + 1, dtype=np.int64)

    rng = np.random.RandomState(frame + seed_off * 1000)
    picks = rng.choice(tx * ty, size=min(n_tiles, tx * ty), replace=False)
    for p in picks:
        t_x, t_y = int(p % tx), int(p // tx)
        sel = idx[(min_x[idx] <= t_x) & (max_x[idx] > t_x)
                  & (min_y[idx] <= t_y) & (max_y[idx] > t_y)]
        if sel.size == 0:
            continue
        order = sel[np.argsort(z[sel])]
        run_tile(g, order, t_x, t_y, minAlpha, acc)

    scale = (AREA * inst) / acc['pairs_total']

    def S(v):
        return v * scale

    print('frame %d, %d tiles sampled of %d, seed offset %d'
          % (frame, n_tiles, tx * ty, seed_off))
    print('tile instances whole frame %d (census peak %d)'
          % (inst, sl['peakTileInstances']))
    print('')
    print('--- headline populations, per iteration ---')
    for k, name in [('fwd_eval', 'forward loop-body entries'),
                    ('bwd_iter', 'backward loop-body entries'),
                    ('bwd_skip', '  compare-only skips'),
                    ('bwd_power', '  reaching delta+power'),
                    ('bwd_exp', '  reaching exp()'),
                    ('bwd_full', '  FULL BODY')]:
        print('%-34s %16.0f' % (name, S(acc[k])))

    print('')
    print('--- SIMD MASKING: lane populations vs group-issue populations ---')
    print('%-34s %16s %16s %8s' % ('stage', 'lanes alive', 'slots x32',
                                   'waste'))
    for name, lk, sk in [
            ('pre-gate (globalIndex passes)', 'lane_pass_gi', 'slot_pass_gi'),
            ('reaching exp()', 'lane_pass_exp', 'slot_pass_exp'),
            ('full body (53 instr + guards)', 'lane_full', 'slot_full')]:
        lanes = S(acc[lk])
        slots = S(acc[sk]) * SIMD
        print('%-34s %16.0f %16.0f %7.2fx'
              % (name, lanes, slots, slots / max(lanes, 1.0)))
    print('%-34s %16s %16.0f' % ('iterated group slots x32', '-',
                                 S(acc['slot_iter']) * SIMD))
    h = acc['occ_hist']
    a = int(h.sum())
    print('')
    print('full-body occupancy: mean %.2f / 32 lanes, %.2f%% at 32 lanes'
          % (acc['lane_full'] / max(a, 1), 100.0 * h[32] / max(a, 1)))


if __name__ == '__main__':
    main()
