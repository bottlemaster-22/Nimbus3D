"""EXACT replica of build 266 MetalSplatTrainer.selectKeyframes (look-ahead 5 on
qc.weight) + TrainerSlicePlanner.splitHeldOut, in float32 like the device.

Checks itself against model/held_out_frames.json and the census span, then
reports how close every gate decision came to flipping (so float noise can be
ruled in or out), the admission decomposition, and the rotation rate.

Importable: select(target, turn=0.02, spacing_scale=1.0, lookahead=5, ...).
"""
import io, json, os, sys
import numpy as np

D = r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840'
f32 = np.float32

_b = json.load(io.open(os.path.join(D, 'capture_bundle.json'), encoding='utf-8'))
_r = json.load(io.open(os.path.join(D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
REF = _r['refinedPoses']
FRAMES = sorted(_b['frames'], key=lambda f: f['index'])


def _inv_act(q, v):
    """simd_quatf(q).inverse.act(v), q = (x, y, z, w). Done in float32."""
    # q^-1 v q = conj(q) v q / |q|^2 = rotation by the unit quaternion conj(q)/|q|
    q = np.asarray(q, f32)
    q = (q / f32(np.linalg.norm(q))).astype(f32)
    u = np.array([-q[0], -q[1], -q[2]], f32); w = q[3]
    v = np.asarray(v, f32)
    t = (f32(2) * np.cross(u, v)).astype(f32)
    return (v + w * t + np.cross(u, t)).astype(f32)


def pose_of(f):
    p = REF.get(str(f['index'])) or f.get('refinedPose') or f['rawPose']
    t = np.asarray(p['translation'], f32)
    C = -_inv_act(p['rotation'], t)
    F = _inv_act(p['rotation'], np.array([0, 0, 1], f32))
    return C.astype(f32), F.astype(f32)


POOL = [f for f in FRAMES if f32(f['qc']['weight']) > f32(0.05)] or FRAMES
CC = np.array([pose_of(f)[0] for f in POOL], f32)
FF = np.array([pose_of(f)[1] for f in POOL], f32)
W = np.array([f['qc']['weight'] for f in POOL], f32)
IDX = np.array([f['index'] for f in POOL])
TS = np.array([f['timestampSeconds'] for f in POOL])
MB = np.array([f['qc']['motionBlurPixels'] for f in POOL], float)


def path_length():
    L = f32(0)
    for i in range(1, len(POOL)):
        L = f32(L + f32(np.linalg.norm((CC[i - 1] - CC[i]).astype(f32))))
    return L


PATH = path_length()


def _nz(v):
    return (v / f32(np.linalg.norm(v))).astype(f32)


def select(target, turn=0.02, spacing_scale=1.0, lookahead=5, spacing=None,
           keep_log=False, time_gate=None, key=None, nobreak=False,
           resume_after_best=False, monotonic=False):
    """Returns list of (pool position, reason, gate position). Mirrors the
    Swift control flow exactly (default args), including that the loop resumes
    AFTER `frame`, not after `best`, which is what re-picks frames.
    time_gate: optional seconds; a frame is also admitted if this long has
    passed since the last keyframe (a time/translation mix).
    key: per-pool-position score for the look-ahead (default qc.weight).
    resume_after_best: the proposed fix, the walk continues after `best`."""
    target = max(target, 8)
    if key is None:
        key = W
    if len(POOL) <= target and not nobreak:
        return [(i, 'all', i) for i in range(len(POOL))], []
    sp = f32(PATH / f32(target)) if spacing is None else f32(spacing)
    sp = f32(sp * f32(spacing_scale))
    turn = f32(turn)
    chosen, log = [], []
    lastC = lastF = None; lastT = None
    here = -1
    while True:
        here += 1
        if here >= len(POOL):
            break
        c, fw = CC[here], FF[here]
        why = 'first'
        if lastC is not None:
            moved = f32(np.linalg.norm((lastC - c).astype(f32)))
            turned = f32(f32(1) - f32(np.dot(_nz(lastF), _nz(fw))))
            timed = time_gate is not None and (TS[here] - lastT) >= time_gate
            if moved < sp and turned < turn and not timed:
                if keep_log:
                    log.append(('reject', here, float(moved - sp), float(turned - turn)))
                continue
            why = ('M' if moved >= sp else '') + ('R' if turned >= turn else '') + \
                  ('T' if (timed and moved < sp and turned < turn) else '')
            if keep_log:
                log.append(('accept', here, float(moved - sp), float(turned - turn)))
        # monotonic: the window never reaches back to or behind the last pick
        start = here
        if monotonic and chosen and chosen[-1][0] >= here:
            start = chosen[-1][0] + 1
            if start >= len(POOL):
                continue
        best = start
        if lookahead > 0:
            limit = min(here + lookahead, len(POOL) - 1)
            if limit > start:
                for cand in range(start, limit + 1):
                    if key[cand] > key[best]:
                        best = cand
        chosen.append((best, why, here))
        lastC, lastF, lastT = CC[best], FF[best], TS[best]
        if resume_after_best:
            here = best
        if len(chosen) >= target and not nobreak:
            break
    if not nobreak and len(chosen) < min(target, len(POOL)):
        step = max(len(POOL) // target, 1)
        return [(i, 'stride', i) for i in range(0, len(POOL), step)], log
    return chosen, log


def split_held_out(kf, fraction=0.10):
    if len(kf) < 20:
        return kf, []
    step = max(int(f32(1) / f32(fraction)), 2)
    tr, ho = [], []
    for i, k in enumerate(kf):
        (ho if i % step == step // 2 else tr).append(k)
    if len(tr) < len(kf) // 2:
        return kf, []
    return tr, ho


if __name__ == '__main__':
    print('frames %d  pool %d  path %.4f m  refined poses %d' % (len(FRAMES), len(POOL), PATH, len(REF)))
    ch, log = select(120, keep_log=True)
    kf = [int(IDX[b]) for b, _, _ in ch]
    tr, ho = split_held_out(kf)
    want = json.load(open(os.path.join(D, 'model', 'held_out_frames.json')))
    cen = json.load(open(os.path.join(D, 'model', 'train_census.json')))
    print('replica: n=%d span %d..%d | census: n=%d span %d..%d'
          % (len(kf), min(kf), max(kf), cen['keyframesSelected'], cen['keyframeFirstIndex'], cen['keyframeLastIndex']))
    print('held out replica :', ho)
    print('held out on disk :', want)
    print('EXACT MATCH held-out:', ho == want, '| span:', (min(kf), max(kf)) == (cen['keyframeFirstIndex'], cen['keyframeLastIndex']))
    print('duplicates in chosen:', len(kf) - len(set(kf)), ' non-monotonic steps:', sum(1 for a, b in zip(kf, kf[1:]) if b <= a))
    print('look-ahead moved the pick on %d of %d' % (sum(1 for b, _, h in ch if b != h), len(ch)))
    # margins: closest call to each threshold among decisions actually taken
    sp = float(PATH / f32(120))
    worst = min(log, key=lambda e: min(abs(e[2]), abs(e[3])))
    mm = min(abs(e[2]) for e in log); mt = min(abs(e[3]) for e in log)
    print('spacing %.5f m | closest move-gate margin %.2e m | closest turn-gate margin %.2e'
          ' (float32 eps at these magnitudes ~1e-8)' % (sp, mm, mt))
    from collections import Counter
    print('admission reasons:', dict(Counter(w for _, w, _ in ch)))
    # rotation rate along the walk, per 10 percent of the pool
    ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', FF[:-1] / np.linalg.norm(FF[:-1], axis=1)[:, None],
                                                    FF[1:] / np.linalg.norm(FF[1:], axis=1)[:, None]), -1, 1)))
    seg = np.linalg.norm(np.diff(CC.astype(float), axis=0), axis=1)
    dt = np.diff(TS)
    print('\nwalk decile | frames | turn deg | move m | deg/s | m/s | keyframes chosen here (R-only)')
    pos = np.array([b for b, _, _ in ch]); why = [w for _, w, _ in ch]
    n = len(POOL)
    for d in range(10):
        a, bb = d * n // 10, (d + 1) * n // 10
        sel = (pos >= a) & (pos < bb)
        ronly = sum(1 for p, w in zip(pos, why) if a <= p < bb and w == 'R')
        s = slice(a, max(bb - 1, a))
        print('  %3d-%3d    | %4d   | %7.1f  | %5.2f  | %5.1f | %.3f | %3d (%d)'
              % (IDX[a], IDX[bb - 1], bb - a, ang[s].sum(), seg[s].sum(), ang[s].sum() / max(dt[s].sum(), 1e-9),
                 seg[s].sum() / max(dt[s].sum(), 1e-9), sel.sum(), ronly))
    print('total turn %.0f deg over %.1f s; total path %.2f m' % (ang.sum(), TS[-1] - TS[0], seg.sum()))
