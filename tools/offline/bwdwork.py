"""WORK PER PAIR in trainer_rasterize_backward, measured not argued.

Simulates the backward inner loop exactly as written (batch gate on the
SIMD-group max of lastContributor, per-entry globalIndex test, pad0 cutoff,
alpha test) and counts every population the per-pair instruction budget is
multiplied by.

Run:  python -u bwdwork.py [nframes] [tiles_per_frame]
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
    gx, gy = gx.ravel(), gy.ravel()            # tid order: y*16+x
    dx = gx[:, None] - g['mx'][order][None, :]
    dy = gy[:, None] - g['my'][order][None, :]
    cx = g['cx'][order][None, :]
    cy = g['cy'][order][None, :]
    cz = g['cz'][order][None, :]
    power = -0.5 * (cx * dx * dx + cz * dy * dy) - cy * dx * dy

    # pad0: half(log(minAlpha / opacity) - 0.01), trainer_preprocess line 1057.
    op = np.maximum(g['opacity'][order], 1e-8)
    cutoff = np.log(minAlpha / op) - 0.01
    cutoff = np.float64(np.float16(cutoff))[None, :]
    pass_cut = power >= cutoff

    alpha = np.minimum(0.99, op[None, :] * np.exp(np.clip(power, -80, 0)))
    alpha_ok = pass_cut & (alpha >= minAlpha)

    keep = np.where(alpha_ok, alpha, 0.0)
    Tafter = np.cumprod(1.0 - keep, axis=1)
    # forward stops at the FIRST alpha_ok entry whose post-composite T < 1e-4
    stop = alpha_ok & (Tafter < 1e-4)
    has_stop = stop.any(axis=1)
    stop_pos = np.where(has_stop, np.argmax(stop, axis=1), n)   # 0-based
    colidx = np.arange(n)[None, :]
    used = alpha_ok & (colidx < stop_pos[:, None])
    any_used = used.any(axis=1)
    lastc = np.where(any_used, n - np.argmax(used[:, ::-1], axis=1), 0)  # 1-based

    # ---- forward: evaluates positions 1..stop_pos+1 (it breaks on the stop) --
    fwd_eval = np.where(has_stop, stop_pos + 1, n)

    # ---- backward -----------------------------------------------------------
    lane_group = np.arange(AREA) // SIMD
    deepest = np.zeros(AREA, dtype=np.int64)
    for gidx in range(NSIMD):
        m = lane_group == gidx
        deepest[m] = lastc[m].max()
    # batches entered by this lane's group: batchBase < deepest
    nb = np.minimum(n, np.ceil(deepest / AREA).astype(np.int64) * AREA)
    nb = np.where(deepest > 0, nb, 0)
    bwd_iter = nb                              # loop-body entries (incl. skips)
    bwd_skip = nb - lastc                      # compare-only, globalIndex > lastc
    bwd_power = lastc                          # entries reaching delta+power
    reached = colidx < lastc[:, None]
    bwd_cut_pass = (pass_cut & reached).sum(axis=1)          # reach exp()
    bwd_full = used.sum(axis=1)                              # full body

    acc['pairs_total'] += AREA * n
    acc['fwd_eval'] += int(fwd_eval.sum())
    acc['bwd_iter'] += int(bwd_iter.sum())
    acc['bwd_skip'] += int(bwd_skip.sum())
    acc['bwd_power'] += int(bwd_power.sum())
    acc['bwd_exp'] += int(bwd_cut_pass.sum())
    acc['bwd_full'] += int(bwd_full.sum())
    acc['pixels'] += AREA

    # ---- exact-zero deltas (would make a conic/mean2D atomic a no-op) -------
    acc['zero_dx'] += int(((dx == 0.0) & used).sum())
    acc['zero_dy'] += int(((dy == 0.0) & used).sum())

    # alpha clamped at 0.99 (the min() bites, gradient is wrong there)
    acc['clamped'] += int(((alpha >= 0.99) & used).sum())

    # ---- SIMD-group lane occupancy per (group, j) --------------------------
    # every lane of a group walks the SAME j, so all 32 lanes hit the SAME
    # splatIndex.  How many of them actually run the body?
    for gidx in range(NSIMD):
        m = lane_group == gidx
        sub = used[m]                      # (32, n)
        per_j = sub.sum(axis=0)            # contributing lanes at each j
        active = per_j > 0
        nbg = int(nb[m][0])                # slots this group actually iterates
        acc['grp_j_iter'] += nbg
        acc['grp_j_active'] += int(active[:nbg].sum())
        acc['grp_j_lanes'] += int(per_j.sum())
        acc['occ_hist'] += np.bincount(per_j[active], minlength=SIMD + 1)[:SIMD + 1]
        # batches this group enters that contribute nothing at all
        for bb in range(0, nbg, AREA):
            acc['grp_batches'] += 1
            if per_j[bb:bb + AREA].sum() == 0:
                acc['grp_batches_empty'] += 1

    # ---- quad (4 lane) occupancy, the cheap reduction ----------------------
    lane_quad = np.arange(AREA) // 4
    for qidx in range(AREA // 4):
        m = lane_quad == qidx
        per_j = used[m].sum(axis=0)
        active = per_j > 0
        acc['quad_active'] += int(active.sum())
        acc['quad_lanes'] += int(per_j.sum())


def main():
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_tiles = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx, ty = (rw + 15) // 16, (rh + 15) // 16
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
    poses = P.load_poses()
    minAlpha = P.MIN_ALPHA
    keys = sorted(poses.keys())
    if len(sys.argv) > 3:
        frames = [int(s) for s in sys.argv[3].split(',')]
    else:
        stride = max(1, len(keys) // nframes)
        frames = keys[::stride][:nframes]
    print('render %dx%d, %dx%d tiles, %d pixels, minAlpha %.6f'
          % (rw, rh, tx, ty, rw * rh, minAlpha))
    print('frames %s of %d poses' % (frames, len(keys)))

    acc = dict.fromkeys(
        ['pairs_total', 'fwd_eval', 'bwd_iter', 'bwd_skip', 'bwd_power',
         'bwd_exp', 'bwd_full', 'pixels', 'zero_dx', 'zero_dy', 'clamped',
         'grp_j_active', 'grp_j_lanes', 'grp_j_iter', 'grp_batches',
         'grp_batches_empty', 'quad_active', 'quad_lanes'], 0)
    acc['occ_hist'] = np.zeros(SIMD + 1, dtype=np.int64)
    tile_inst_frame = []

    for fr in frames:
        rot, t = poses[fr]
        R = P.quat_to_matrix(rot)
        g = Rz.geometry(col, count, R, t, fx, fy, cx, cy)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy,
                                              tx, ty)
        tile_inst_frame.append(int(tiles.sum()))
        idx = np.nonzero(ok)[0]
        mx, my, z = g['mx'], g['my'], g['z']
        min_x = np.maximum(0, np.floor((mx - ex) / TILE)).astype(np.int64)
        min_y = np.maximum(0, np.floor((my - ey) / TILE)).astype(np.int64)
        max_x = np.minimum(tx, np.ceil((mx + ex) / TILE)).astype(np.int64)
        max_y = np.minimum(ty, np.ceil((my + ey) / TILE)).astype(np.int64)
        rng = np.random.RandomState(int(fr))
        picks = rng.choice(tx * ty, size=min(n_tiles, tx * ty), replace=False)
        for p in picks:
            t_x, t_y = int(p % tx), int(p // tx)
            sel = idx[(min_x[idx] <= t_x) & (max_x[idx] > t_x)
                      & (min_y[idx] <= t_y) & (max_y[idx] > t_y)]
            if sel.size == 0:
                continue
            order = sel[np.argsort(z[sel])]
            run_tile(g, order, t_x, t_y, minAlpha, acc)
        print('  frame %s done, %d tile instances whole-frame' % (fr, tiles.sum()))

    tot = acc['pairs_total']
    mean_inst = float(np.mean(tile_inst_frame))
    scale = (AREA * mean_inst) / tot      # sample -> whole frame, ratio estimator
    print('\nSAMPLE: %d tiles, %d pixels, %d pairs'
          % (acc['pixels'] // AREA, acc['pixels'], tot))
    print('WHOLE FRAME: %.0f tile instances (census peak %d), %.0f pairs'
          % (mean_inst, sl['peakTileInstances'], AREA * mean_inst))
    print('\n%-36s %14s %8s %16s'
          % ('population', 'sampled', 'per px', 'per ITERATION'))
    rows = [
        ('all (pixel,gaussian) pairs in tiles', tot),
        ('forward: loop-body entries', acc['fwd_eval']),
        ('backward: loop-body entries', acc['bwd_iter']),
        ('  of which compare-only skips', acc['bwd_skip']),
        ('  reaching delta + power', acc['bwd_power']),
        ('  reaching exp()', acc['bwd_exp']),
        ('  FULL BODY (12 atomics)', acc['bwd_full']),
    ]
    for name, v in rows:
        print('%-36s %14d %8.1f %16.0f'
              % (name, v, v / acc['pixels'], v * scale))
    print('\nbackward iterations / forward iterations   %.3f'
          % (acc['bwd_iter'] / acc['fwd_eval']))
    print('backward full-body / forward iterations    %.3f'
          % (acc['bwd_full'] / acc['fwd_eval']))
    print('\nEXACT-ZERO GRADIENT COMPONENTS among full-body pairs')
    print('  delta.x == 0   %d  (%.4f%%)'
          % (acc['zero_dx'], 100 * acc['zero_dx'] / acc['bwd_full']))
    print('  delta.y == 0   %d  (%.4f%%)'
          % (acc['zero_dy'], 100 * acc['zero_dy'] / acc['bwd_full']))

    print('\nSIMD-GROUP LANE OCCUPANCY (all 32 lanes walk the same j, so the')
    print('same splatIndex: the 12 atomics are same-address across the group)')
    a = acc['grp_j_active']
    l = acc['grp_j_lanes']
    print('  (group, j) slots with >=1 contributing lane  %d' % a)
    print('  contributing lanes total                     %d' % l)
    print('  MEAN CONTRIBUTING LANES PER ACTIVE SLOT      %.2f / 32' % (l / a))
    h = acc['occ_hist']
    print('  occupancy histogram (lanes -> %% of active slots):')
    for k in range(1, SIMD + 1):
        if h[k]:
            print('    %2d lanes %10d  %5.2f%%' % (k, h[k], 100 * h[k] / a))
    print('\n  (group,j) slots ITERATED per iteration        %.0f'
          % (acc['grp_j_iter'] * scale))
    print('  (group,j) slots ACTIVE per iteration          %.0f  (%.1f%% of iterated)'
          % (a * scale, 100 * a / acc['grp_j_iter']))
    print('  device atomics/iteration now (12 per full-body pair): %.0f'
          % (12 * acc['bwd_full'] * scale))
    print('  device atomics/iteration with one simd_sum per slot : %.0f'
          % (12 * a * scale))
    print('  reduction factor                                    : %.2fx'
          % (acc['bwd_full'] / a))
    qa, ql = acc['quad_active'], acc['quad_lanes']
    print('\nQUAD (4-lane) occupancy, the cheaper reduction')
    print('  mean contributing lanes per active quad slot  %.2f / 4' % (ql / qa))
    print('  device atomics/iteration with a quad_sum      %.0f  (%.2fx fewer)'
          % (12 * qa * scale, ql / qa))
    print('\nBATCH-LEVEL WASTE (per SIMD group)')
    print('  batches entered            %.0f' % (acc['grp_batches'] * scale))
    print('  ... contributing nothing   %.0f  (%.1f%%)'
          % (acc['grp_batches_empty'] * scale,
             100 * acc['grp_batches_empty'] / acc['grp_batches']))
    print('\nALPHA CLAMPED AT 0.99 among full-body pairs  %d  (%.3f%%)'
          % (acc['clamped'], 100 * acc['clamped'] / acc['bwd_full']))


if __name__ == '__main__':
    main()
