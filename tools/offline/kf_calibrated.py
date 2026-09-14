"""Entries 40/47: does the keyframe prefix bias bite, and what is the fix that
is actually justified?

Compares, like-for-like (frustum visibility, same budget, same held-out split):
  A  shipped greedy          spacing = pathLength/target, break at target
  B  even stride             the naive "spread it out" fix
  C  calibrated greedy       same greedy, spacing bisected so the walk is
                             consumed to the END and still yields `target`
Also counts how often the `turned < 0.02` rotation clause admits a frame that
failed the movement gate.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import load_all

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)

b, r, frames = load_all()
pool = [f for f in frames if f['weight'] > 0.05] or frames
TARGET = 120

path = 0.0
prev = None
for fr in pool:
    if prev is not None:
        path += float(np.linalg.norm(fr['C'] - prev))
    prev = fr['C']


def greedy(spacing, target=TARGET, count_clause=False):
    chosen, lastC, lastF = [], None, None
    moved_only = turned_only = both = 0
    for i, fr in enumerate(pool):
        if lastC is not None:
            moved = float(np.linalg.norm(lastC - fr['C']))
            turned = 1.0 - float(np.dot(lastF / np.linalg.norm(lastF),
                                        fr['F'] / np.linalg.norm(fr['F'])))
            if moved < spacing and turned < 0.02:
                continue
            if moved >= spacing and turned >= 0.02:
                both += 1
            elif moved >= spacing:
                moved_only += 1
            else:
                turned_only += 1
        chosen.append(i)
        lastC, lastF = fr['C'], fr['F']
        if len(chosen) >= target:
            break
    if count_clause:
        return chosen, (moved_only, turned_only, both)
    return chosen


sp0 = path / TARGET
A_i, clause = greedy(sp0, count_clause=True)
print('shipped spacing %.4f m -> %d keyframes, last pool slot %d of %d'
      % (sp0, len(A_i), A_i[-1], len(pool) - 1))
print('  admitted by MOVEMENT only %d | by ROTATION only %d | by both %d'
      % clause)
print('  the rotation clause alone admitted %.1f%% of the chosen set'
      % (100.0 * clause[1] / len(A_i)))

# bisect spacing so the greedy runs to the end of the pool and still yields 120
lo, hi = sp0, sp0 * 20
for _ in range(60):
    mid = 0.5 * (lo + hi)
    n = len(greedy(mid, target=10 ** 9))
    if n > TARGET:
        lo = mid
    else:
        hi = mid
C_i = greedy(hi, target=10 ** 9)
print('calibrated spacing %.4f m -> %d keyframes, last pool slot %d of %d'
      % (hi, len(C_i), C_i[-1], len(pool) - 1))

st = max(len(pool) // TARGET, 1)
B_i = list(range(0, len(pool), st))[:TARGET]

sets = {'A shipped greedy': A_i, 'B even stride': B_i, 'C calibrated greedy': C_i}


def qi(q, v):
    x, y, z, w = q
    qv = np.array([-x, -y, -z]); t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


rng = np.random.default_rng(0)
samp = rng.choice(count, 15000, replace=False)
res = {}
for label, sel in sets.items():
    kfidx = [pool[i]['index'] for i in sel]
    trn = [k for i, k in enumerate(kfidx) if i % 10 != 5]
    V = np.zeros((len(trn), count), dtype=bool)
    for j, idx in enumerate(trn):
        R = P.quat_to_matrix(poses[idx][0]); t = poses[idx][1]
        ok, _, _, _, _ = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        V[j] = ok
    cams = np.array([-qi(poses[i][0], poses[i][1]) for i in trn])
    nv = V.sum(0)
    med, mx = [], []
    for i in samp:
        k = np.where(V[:, i])[0]
        if len(k) < 2:
            continue
        d = cams[k] - mean[i]; d /= np.linalg.norm(d, axis=1)[:, None]
        Aq = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        iu = np.triu_indices(len(k), 1)
        med.append(np.median(Aq[iu])); mx.append(Aq[iu].max())
    res[label] = (nv, np.median(med), np.median(mx), kfidx)
    print('%-22s span %3d..%3d | views/G %.2f | <=3 views %.2f%% | 0 views %.2f%% '
          '| pairwise angle med %.1f max %.1f'
          % (label, kfidx[0], kfidx[-1], nv.mean(), 100 * (nv <= 3).mean(),
             100 * (nv == 0).mean(), np.median(med), np.median(mx)))
