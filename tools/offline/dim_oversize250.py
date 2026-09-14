"""DIMENSION: THE OVERSIZED PRUNE THAT NEVER FIRES, re-measured on BUILD 250.

Measures the exact quantity TrainerDensifier compares against
pruneMaxScreenRadiusPx (720): stats[i].maxRadiusPxBits, written by
trainer_preprocess as
    radius = 3 * sqrt(max eigenvalue of (Sigma2D + lowPass))    [PIXELS]
    atomic_fetch_max(&stats[gid].maxRadiusPxBits, as_type<uint>(radius))
only for splats that survive every early return, and reset to 0 after EVERY
densify pass, so on device it is a max over ONE 100-iteration interval.

Views used: the trainer's actual 120 selected keyframes (_kf.npy), split into
the 108 trained and the 12 held out (held_out_frames.json). The max over all
108 trained views is a STRICT UPPER BOUND on any single interval's value,
because one interval renders at most 100 of them. The 12 held-out views are
reported separately: evaluateHeldOut also calls gpu.preprocess on the same
stats buffer, so they reach maxRadiusPxBits too, every 400 iterations.

Both projections are measured on the same exported model: WITH the build-250
tangent clamp (what ran) and WITHOUT it (the pre-250 Jacobian), so the clamp's
effect on this statistic is isolated. No .npy is written.
"""
import io
import json
import os
import sys

import numpy as np

import project as P
from dim_oversize2 import world_covariance

CLAMP_FACTOR = 1.3


def per_view(mean, sigma, opacity, R, t, fx, fy, cx, cy, tx, ty, rw, rh, clamp):
    cam = mean @ R.T + t
    z = cam[:, 2]
    sigma_cam = np.einsum('ij,njk,lk->nil', R, sigma, R)
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    ux = cam[:, 0] * inv_z
    uy = cam[:, 1] * inv_z
    lim_x = CLAMP_FACTOR * (0.5 * rw / fx)
    lim_y = CLAMP_FACTOR * (0.5 * rh / fy)
    binds = (np.abs(ux) > lim_x) | (np.abs(uy) > lim_y)
    if clamp:
        txc = np.clip(ux, -lim_x, lim_x) * z
        tyc = np.clip(uy, -lim_y, lim_y) * z
    else:
        txc = cam[:, 0]
        tyc = cam[:, 1]
    j00 = fx * inv_z
    j11 = fy * inv_z
    j02 = -fx * txc * inv_z2
    j12 = -fy * tyc * inv_z2
    s00 = sigma_cam[:, 0, 0]; s01 = sigma_cam[:, 0, 1]; s02 = sigma_cam[:, 0, 2]
    s11 = sigma_cam[:, 1, 1]; s12 = sigma_cam[:, 1, 2]; s22 = sigma_cam[:, 2, 2]
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12
    det_before = np.maximum(sa * sc - sb * sb, 1e-12)
    low_pass = P.FILTER_2D_VARIANCE + P.FREQ_BLUR_VARIANCE
    sa = sa + low_pass
    sc = sc + low_pass
    det = sa * sc - sb * sb
    comp2d = np.sqrt(np.clip(det_before / np.maximum(det, 1e-12), 0, 1))
    mid = 0.5 * (sa + sc)
    disc = np.sqrt(np.maximum(mid * mid - det, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))
    alpha = opacity * comp2d
    level = np.clip(2.0 * np.log(np.maximum(alpha, 1e-30) / P.MIN_ALPHA), 0, 9)
    k = np.sqrt(level)
    ex = k * np.sqrt(np.maximum(sa, 1e-9)) * 1.001
    ey = k * np.sqrt(np.maximum(sc, 1e-9)) * 1.001
    mx = fx * ux + cx
    my = fy * uy + cy
    min_x = np.maximum(0, np.floor((mx - ex) / P.TILE_W)).astype(np.int64)
    min_y = np.maximum(0, np.floor((my - ey) / P.TILE_H)).astype(np.int64)
    max_x = np.minimum(tx, np.ceil((mx + ex) / P.TILE_W)).astype(np.int64)
    max_y = np.minimum(ty, np.ceil((my + ey) / P.TILE_H)).astype(np.int64)
    tiles = np.maximum(max_x - min_x, 0) * np.maximum(max_y - min_y, 0)
    ok = (z > 0.05) & (z < 100) & (det > 1e-12) & (radius >= 0.5)
    ok &= alpha >= P.MIN_ALPHA
    ok &= tiles > 0
    return ok, radius, np.where(ok, tiles, 0), binds & ok, z


def sweep(mean, sigma, opacity, poses, keys, intr, grid, clamp):
    fx, fy, cx, cy = intr
    tx, ty, rw, rh = grid
    n = mean.shape[0]
    mxr = np.zeros(n)
    seen = np.zeros(n, bool)
    inst = np.zeros(n, np.int64)
    nearest = np.full(n, np.inf)
    draws_r = []
    bind_big = 0
    big_draws = 0
    per_view_inst = []
    for f in keys:
        R = P.quat_to_matrix(poses[f][0])
        ok, radius, tiles, binds, z = per_view(
            mean, sigma, opacity, R, poses[f][1], fx, fy, cx, cy, tx, ty, rw, rh, clamp)
        mxr = np.where(ok, np.maximum(mxr, radius), mxr)
        nearest = np.where(ok, np.minimum(nearest, z), nearest)
        seen |= ok
        inst += tiles
        per_view_inst.append(int(tiles.sum()))
        draws_r.append(radius[ok].astype(np.float32))
        big = ok & (radius > 90)
        big_draws += int(big.sum())
        bind_big += int((big & binds).sum())
    return dict(mxr=mxr, seen=seen, inst=inst, nearest=nearest,
                draws=np.concatenate(draws_r), bind_big=bind_big,
                big_draws=big_draws, pvi=np.array(per_view_inst))


def report(label, r, count, largest_mm):
    d = r['mxr'][r['seen']]
    total = int(r['inst'].sum())
    print('')
    print('===== ' + label)
    print('  splats drawn in >=1 view: %d of %d   tile instances summed over views: %d'
          % (int(r['seen'].sum()), count, total))
    print('  tile instances per view: mean %d  max %d' % (int(r['pvi'].mean()), int(r['pvi'].max())))
    dr = r['draws']
    print('  PER-DRAW 3-sigma radius px over %d (splat,view) draws:' % dr.size)
    print('    p50 %.2f  p90 %.2f  p99 %.2f  p99.9 %.2f  max %.2f'
          % tuple(np.percentile(dr, [50, 90, 99, 99.9, 100])))
    print('  MAX-OVER-VIEWS radius px (the quantity stats.maxRadiusPxBits holds):')
    print('    p50 %.2f  p90 %.2f  p99 %.2f  p99.9 %.2f  max %.2f'
          % tuple(np.percentile(d, [50, 90, 99, 99.9, 100])))
    print('  (splat,view) draws with radius>90 px where the tangent clamp BINDS: %d of %d'
          % (r['bind_big'], r['big_draws']))
    print('  cutoff   splats>cut  pctOfDrawn  instShare%%  largestAxis_mm p50/max  nearest_z_m p50/min')
    for thr in (720, 540, 360, 270, 180, 120, 90):
        m = r['seen'] & (r['mxr'] > thr)
        nm = int(m.sum())
        if nm == 0:
            print('  %6d  %10d  %10.4f  %10.4f' % (thr, 0, 0.0, 0.0))
            continue
        print('  %6d  %10d  %10.4f  %10.4f   %8.2f %8.2f   %7.3f %7.3f'
              % (thr, nm, 100.0 * nm / max(int(r['seen'].sum()), 1),
                 100.0 * r['inst'][m].sum() / max(total, 1),
                 np.percentile(largest_mm[m], 50), largest_mm[m].max(),
                 np.percentile(r['nearest'][m], 50), r['nearest'][m].min()))
    sys.stdout.flush()


def keyframes(poses):
    """The trainer's 120-keyframe selection, reproduced (vis_count.py's
    replica) and ASSERTED against this run's held_out_frames.json. _kf.npy on
    disk is from an older run and is NOT this selection."""
    bundle = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))

    def qinv_act(q, v):
        x, y, z, w = q
        qv = np.array([-x, -y, -z])
        t = 2 * np.cross(qv, v)
        return v + w * t + np.cross(qv, t)
    frames = sorted(bundle['frames'], key=lambda f: f['index'])
    pool = [f for f in frames if f['qc']['weight'] > 0.05] or frames
    C, F = {}, {}
    for f in frames:
        q, t = poses[f['index']]
        C[f['index']] = -qinv_act(q, t)
        F[f['index']] = qinv_act(q, np.array([0., 0., 1.]))
    pl, prev = 0.0, None
    for f in pool:
        c = C[f['index']]
        if prev is not None:
            pl += np.linalg.norm(c - prev)
        prev = c
    spacing = pl / 120.0
    # MetalSplatTrainer.selectKeyframes, including the keyframeSharpnessLookahead
    # (5) added after vis_count.py was written: past the spacing gate, take the
    # highest qc.weight in pool[here ... here+5] (strictly greater wins).
    LOOKAHEAD = 5
    chosen, lc, lf = [], None, None
    for here, f in enumerate(pool):
        c, fw = C[f['index']], F[f['index']]
        if lc is not None:
            if (np.linalg.norm(lc - c) < spacing
                    and 1 - np.dot(fw / np.linalg.norm(fw), lf / np.linalg.norm(lf)) < 0.02):
                continue
        best = f
        limit = min(here + LOOKAHEAD, len(pool) - 1)
        for cand in pool[here:limit + 1]:
            if cand['qc']['weight'] > best['qc']['weight']:
                best = cand
        chosen.append(best['index'])
        lc, lf = C[best['index']], F[best['index']]
        if len(chosen) >= 120:
            break
    held = [chosen[i] for i in range(len(chosen)) if i % 10 == 5]
    trained = [chosen[i] for i in range(len(chosen)) if i % 10 != 5]
    truth = json.load(io.open(os.path.join(P.D, 'model', 'held_out_frames.json'), encoding='utf-8'))
    assert held == truth, ('keyframe repro mismatch', held, truth)
    return chosen, trained, held


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    tx = (rw + P.TILE_W - 1) // P.TILE_W
    ty = (rh + P.TILE_H - 1) // P.TILE_H
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    kf, trained, held = keyframes(poses)
    print('render %dx%d  tiles %dx%d = %d   fx %.3f fy %.3f cx %.2f cy %.2f'
          % (rw, rh, tx, ty, tx * ty, fx, fy, cx, cy))
    print('splats %d   poses %d   _kf.npy %d   held-out %d   held-out subset of _kf: %s'
          % (count, len(poses), len(kf), len(held), set(held) <= set(kf)))
    print('census keyframesSelected %d trained %d heldOut %d'
          % (census['keyframesSelected'], sl['keyframesTrained'], sl['keyframesHeldOut']))
    print('census peakTileInstances %d' % sl['peakTileInstances'])
    print('trained views used: %d (keyframe repro asserted against held_out_frames.json)' % len(trained))

    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    sigma, scale = world_covariance(col, count)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    largest_mm = 1000.0 * scale.max(axis=1)
    print('world scale: largest linear sigma  p50 %.2f mm  p99 %.2f mm  max %.2f mm'
          % tuple(np.percentile(largest_mm, [50, 99, 100])))
    sys.stdout.flush()

    intr = (fx, fy, cx, cy)
    grid = (tx, ty, rw, rh)
    rc = sweep(mean, sigma, opacity, poses, trained, intr, grid, True)
    report('WITH build-250 tangent clamp | %d TRAINED keyframes (upper bound per interval)'
           % len(trained), rc, count, largest_mm)
    ru = sweep(mean, sigma, opacity, poses, trained, intr, grid, False)
    report('WITHOUT clamp (pre-250 Jacobian) | same model, same %d views' % len(trained),
           ru, count, largest_mm)
    rh_ = sweep(mean, sigma, opacity, poses, held, intr, grid, True)
    report('WITH clamp | the 12 HELD-OUT views evaluateHeldOut also writes into stats',
           rh_, count, largest_mm)

    both = rc['seen'] | ru['seen']
    a, b = rc['mxr'][both], ru['mxr'][both]
    print('')
    print('===== CLAMP EFFECT on max-over-views radius, per splat')
    print('  splats changed by >1%%: %d of %d' % (int((np.abs(a - b) > 0.01 * np.maximum(b, 1e-9)).sum()), int(both.sum())))
    print('  splats >720 px: unclamped %d  clamped %d' % (int((b > 720).sum()), int((a > 720).sum())))
    print('  total tile instances: unclamped %d  clamped %d  (%.3f%% change)'
          % (int(ru['inst'].sum()), int(rc['inst'].sum()),
             100.0 * (rc['inst'].sum() - ru['inst'].sum()) / ru['inst'].sum()))

    # How many tile instances a splat can POSSIBLY carry is bounded by the grid,
    # whatever its radius: report the tile-instance concentration directly, so
    # a radius cutoff can be compared against the best possible cutoff.
    inst = rc['inst']
    total = inst.sum()
    order = np.argsort(-inst)
    cum = np.cumsum(inst[order])
    print('')
    print('===== TILE-INSTANCE CONCENTRATION (clamped, trained views)')
    for nn in (100, 1060, 3000, 10000, 30000):
        print('  top %6d splats by instances carry %.3f%%   (their max radius p50 %.1f px, min %.1f px)'
              % (nn, 100.0 * cum[nn - 1] / total,
                 np.percentile(rc['mxr'][order[:nn]], 50), rc['mxr'][order[:nn]].min()))


if __name__ == '__main__':
    main()
