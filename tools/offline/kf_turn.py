"""Budget-neutral fix for the keyframe prefix: keep 120 frames, keep the greedy,
raise ONLY the rotation threshold until the greedy consumes the whole walk."""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import load_all

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15)//16, (rh + 15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
b, r, frames = load_all()
pool = [f for f in frames if f['weight'] > 0.05] or frames
TARGET = 120
path = 0.0; prev = None
for fr in pool:
    if prev is not None: path += float(np.linalg.norm(fr['C'] - prev))
    prev = fr['C']
SP = path / TARGET

def greedy(spacing, turn, target=10**9):
    chosen, lastC, lastF = [], None, None
    for i, fr in enumerate(pool):
        if lastC is not None:
            moved = float(np.linalg.norm(lastC - fr['C']))
            turned = 1.0 - float(np.dot(lastF/np.linalg.norm(lastF), fr['F']/np.linalg.norm(fr['F'])))
            if moved < spacing and turned < turn: continue
        chosen.append(i); lastC, lastF = fr['C'], fr['F']
        if len(chosen) >= target: break
    return chosen

# how many frames pass at each turn threshold, whole walk, no target break
for t in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    c = greedy(SP, t)
    print('turn %.3f (%.1f deg) -> %3d chosen, last pool slot %3d'
          % (t, np.degrees(np.arccos(1-t)), len(c), c[-1]))

lo, hi = 0.02, 1.0
for _ in range(60):
    mid = 0.5*(lo+hi)
    n = len(greedy(SP, mid))
    if n > TARGET: lo = mid
    else: hi = mid
D_i = greedy(SP, lo, target=TARGET)
print('\ncalibrated turn %.4f (%.2f deg) -> %d chosen, span pool slot %d..%d'
      % (lo, np.degrees(np.arccos(1-lo)), len(D_i), D_i[0], D_i[-1]))

A_i = greedy(SP, 0.02, target=TARGET)
st = max(len(pool)//TARGET, 1)
B_i = list(range(0, len(pool), st))[:TARGET]

def qi(q, v):
    x,y,z,w = q; qv = np.array([-x,-y,-z]); t = 2*np.cross(qv, v)
    return v + w*t + np.cross(qv, t)

rng = np.random.default_rng(0); samp = rng.choice(count, 15000, replace=False)
for label, sel in [('A shipped greedy', A_i), ('B even stride', B_i),
                   ('D turn-calibrated', D_i)]:
    kfidx = [pool[i]['index'] for i in sel]
    trn = [k for i, k in enumerate(kfidx) if i % 10 != 5]
    V = np.zeros((len(trn), count), dtype=bool)
    for j, idx in enumerate(trn):
        R = P.quat_to_matrix(poses[idx][0]); t = poses[idx][1]
        ok,_,_,_,_ = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        V[j] = ok
    cams = np.array([-qi(poses[i][0], poses[i][1]) for i in trn])
    nv = V.sum(0); med = []; mx = []
    for i in samp:
        k = np.where(V[:, i])[0]
        if len(k) < 2: continue
        d = cams[k]-mean[i]; d /= np.linalg.norm(d, axis=1)[:, None]
        Aq = np.degrees(np.arccos(np.clip(d@d.T, -1, 1))); iu = np.triu_indices(len(k), 1)
        med.append(np.median(Aq[iu])); mx.append(Aq[iu].max())
    print('%-20s n=%d span %3d..%3d | views/G %.2f | <=3 %.2f%% | 0 views %.2f%% | ang med %.1f max %.1f'
          % (label, len(kfidx), kfidx[0], kfidx[-1], nv.mean(), 100*(nv<=3).mean(),
             100*(nv==0).mean(), np.median(med), np.median(mx)))
