"""What the ICP cap buys, and what running the pairs on every core costs in
depth loads.  Candidate generation is the validated replica in
reg_candidates.py (411 anchors, 24,916 candidates, both match the census).

1. Depth loads of the CURRENT serial loop: candidates sorted by (a, b), an
   8-entry cache evicted first-in-first-out (a hit does not refresh it).
2. Depth loads of the proposed parallel form: one run per distinct frame A,
   A unprojected once per run, each B unprojected once per pair.
3. A submap-pair-diverse pick (round robin over submap pairs by rank) against
   the current top-N by score.

ICP convergence per candidate is NOT MEASURED here: the depth maps are not on
disk.  Everything below is about WHICH pairs get tried and what that costs.

Run: python -u reg_capsim.py
"""
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import reg_candidates as RC

POINTS_BYTES = 256 * 192 * (16 + 16 + 1 + 4)   # points, normals, valid, depth


def candidates():
    b, r = RC.load()
    frames = sorted(b['frames'], key=lambda f: f['timestampSeconds'])
    own = RC.owners(frames, r['submaps'])
    geo = {}
    for f in frames:
        R = RC.qmat(f['rawPose']['rotation']).astype(np.float32)
        t = np.asarray(f['rawPose']['translation'], np.float32)
        geo[f['index']] = (-R.T @ t, R.T @ np.array([0, 0, 1], np.float32))
    anchors, last_c, last_t = [], None, -1e300
    for f in frames:
        if f.get('depthPath') is None:
            continue
        c = geo[f['index']][0]
        moved = last_c is None or np.linalg.norm(c - last_c) >= RC.SPACING_M
        if anchors and not (moved or f['timestampSeconds'] - last_t >= RC.SPACING_S):
            continue
        anchors.append(f)
        last_c, last_t = c, f['timestampSeconds']
    out = []
    for i, a in enumerate(anchors):
        ca, fa = geo[a['index']]
        for bb in anchors[i + 1:]:
            if bb['timestampSeconds'] - a['timestampSeconds'] < RC.MIN_GAP_S:
                continue
            if own[bb['index']] == own[a['index']]:
                continue
            cb, fb = geo[bb['index']]
            d = float(np.linalg.norm(ca - cb))
            if d > RC.MAX_CENTRE_M:
                continue
            ua, ub = fa / np.linalg.norm(fa), fb / np.linalg.norm(fb)
            ang = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(ua, ub)), np.dot(ua, ub))))
            if ang > RC.MAX_ANGLE_DEG:
                continue
            s = a['qc']['weight'] * bb['qc']['weight'] * (1 - d / RC.MAX_CENTRE_M) * (1 - ang / RC.MAX_ANGLE_DEG)
            out.append((s, a['index'], bb['index'], own[a['index']], own[bb['index']], d, ang))
    out.sort(key=lambda c: -c[0])
    return out, len(r['submaps'])


def fifo_loads(pairs, limit=8):
    cache, loads = OrderedDict(), 0
    for a, b in pairs:
        for f in (a, b):
            if f in cache:
                continue
            loads += 1
            cache[f] = True
            if len(cache) > limit:
                cache.popitem(last=False)
    return loads


def round_robin(cands, cap):
    by = defaultdict(list)
    for c in cands:
        by[(c[3], c[4])].append(c)
    out, level = [], 0
    while len(out) < cap:
        layer = [lst[level] for lst in by.values() if len(lst) > level]
        if not layer:
            break
        layer.sort(key=lambda c: -c[0])
        out.extend(layer[:cap - len(out)])
        level += 1
    return out


def describe(name, pick, nsub):
    pairs = sorted((c[1], c[2]) for c in pick)
    runs = len(set(a for a, _ in pairs))
    frames = set(a for a, _ in pairs) | set(b for _, b in pairs)
    sp = Counter((c[3], c[4]) for c in pick)
    sub = Counter([c[3] for c in pick] + [c[4] for c in pick])
    per_sub = [sub.get(k, 0) for k in range(nsub)]
    print('%-26s n=%5d | submap pairs %3d | submaps with 0 pairs %d, min %d | '
          'dist p50 %.2f m angle p50 %.1f deg score p50 %.3f'
          % (name, len(pick), len(sp), sum(1 for x in per_sub if x == 0), min(per_sub),
             np.median([c[5] for c in pick]), np.median([c[6] for c in pick]),
             np.median([c[0] for c in pick])))
    print('%-26s   loads: serial FIFO-8 %5d | parallel one-run-per-A %5d (runs %d) | '
          'distinct frames %d = %.0f MB if all held' %
          ('', fifo_loads(pairs), runs + len(pairs), runs, len(frames),
           len(frames) * POINTS_BYTES / 2 ** 20))


def main():
    cands, nsub = candidates()
    print('candidates %d (census 24,916)\n' % len(cands))
    for cap in (600, 2000, 5000):
        describe('top-%d by score' % cap, cands[:cap], nsub)
        describe('round-robin-%d' % cap, round_robin(cands, cap), nsub)
        print()
    print('bytes per PrePassFramePoints: %d (%.2f MB)' % (POINTS_BYTES, POINTS_BYTES / 2 ** 20))


if __name__ == '__main__':
    main()
