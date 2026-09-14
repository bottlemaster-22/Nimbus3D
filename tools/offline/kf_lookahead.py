"""Reimplements build-244 selectKeyframes exactly and measures what the
shipped `qc.weight` lookahead actually bought, against three alternatives."""
import io, json, os
import numpy as np
import project as P

D = P.D
b = json.load(io.open(os.path.join(D, 'capture_bundle.json'), encoding='utf-8'))
r = json.load(io.open(os.path.join(D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
ref = r['refinedPoses']

frames = sorted(b['frames'], key=lambda f: f['index'])
def pose(idx):
    p = ref.get(str(idx))
    if p is None: return None
    R = P.quat_to_matrix(np.asarray(p['rotation'], float))
    t = np.asarray(p['translation'], float)
    c = -R.T @ t
    fwd = -R[2, :]
    return c, fwd

pool = [f for f in frames if f['qc']['weight'] > 0.05] or frames
TARGET = 120
print('frames %d  pool %d  target %d' % (len(frames), len(pool), TARGET))

C = {}; F = {}
for f in pool:
    pr = pose(f['index'])
    if pr is None: continue
    C[f['index']], F[f['index']] = pr
pool = [f for f in pool if f['index'] in C]
print('pool with refined pose %d' % len(pool))

pathLength = 0.0
prev = None
for f in pool:
    c = C[f['index']]
    if prev is not None: pathLength += float(np.linalg.norm(c - prev))
    prev = c
spacing = pathLength / TARGET
print('path %.3f m  spacing %.4f m' % (pathLength, spacing))

def select(lookahead, key):
    """key(frame) -> higher is better; picked over the window."""
    chosen = []; lastC = None; lastF = None
    for i, f in enumerate(pool):
        c = C[f['index']]; fw = F[f['index']]
        if lastC is not None:
            moved = float(np.linalg.norm(lastC - c))
            turned = 1 - float(np.dot(lastF/np.linalg.norm(lastF), fw/np.linalg.norm(fw)))
            if moved < spacing and turned < 0.02: continue
        best = f
        if lookahead > 0:
            lim = min(i + lookahead, len(pool) - 1)
            if lim > i:
                for cand in pool[i:lim+1]:
                    if key(cand) > key(best): best = cand
        chosen.append(best)
        lastC = C[best['index']]; lastF = F[best['index']]
        if len(chosen) >= TARGET: break
    return chosen

def report(name, ch):
    mb = np.array([f['qc']['motionBlurPixels'] for f in ch])
    rend = mb * 720.0 / 1920.0
    lastpos = max(pool.index(f) for f in ch)
    print('%-34s n=%3d  native p50 %.2f p90 %.2f | render p50 %.3f p90 %.3f | >1px %3d/%d | last pool idx %3d (%.1f%% of walk)'
          % (name, len(ch), np.percentile(mb,50), np.percentile(mb,90),
             np.percentile(rend,50), np.percentile(rend,90),
             int((rend>1).sum()), len(ch), lastpos, 100.0*lastpos/(len(pool)-1)))

report('lookahead 0 (pre-fix)',            select(0, lambda f: 0))
report('lookahead 5 on qc.weight (SHIPPED)', select(5, lambda f: f['qc']['weight']))
report('lookahead 5 on -motionBlurPixels',   select(5, lambda f: -f['qc']['motionBlurPixels']))
report('lookahead 5 on qc.sharpness',        select(5, lambda f: f['qc']['sharpness']))
report('lookahead 3 on -motionBlurPixels',   select(3, lambda f: -f['qc']['motionBlurPixels']))
report('lookahead 8 on -motionBlurPixels',   select(8, lambda f: -f['qc']['motionBlurPixels']))

# how correlated are weight and blur at all?
w = np.array([f['qc']['weight'] for f in pool])
mb = np.array([f['qc']['motionBlurPixels'] for f in pool])
sh = np.array([f['qc']['sharpness'] for f in pool])
print()
print('pool: r(weight, -blur) = %+.3f   r(sharpness, -blur) = %+.3f   r(weight, sharpness) = %+.3f'
      % (np.corrcoef(w, -mb)[0,1], np.corrcoef(sh, -mb)[0,1], np.corrcoef(w, sh)[0,1]))
print('pool blur native p10 %.2f p50 %.2f p90 %.2f' % tuple(np.percentile(mb,[10,50,90])))

# --- proper coverage: cumulative path length reached, and spread of chosen ---
cum = [0.0]
for i in range(1, len(pool)):
    cum.append(cum[-1] + float(np.linalg.norm(C[pool[i]['index']] - C[pool[i-1]['index']])))
cum = np.array(cum)
idx_of = {f['index']: i for i, f in enumerate(pool)}
print()
print('%-34s %9s %9s %9s' % ('', 'walk m', '% of 39.7', 'chosen-span m'))
for name, ch in [
    ('lookahead 0 (pre-fix)', select(0, lambda f: 0)),
    ('lookahead 5 qc.weight (SHIPPED)', select(5, lambda f: f['qc']['weight'])),
    ('lookahead 5 -motionBlurPixels', select(5, lambda f: -f['qc']['motionBlurPixels'])),
    ('lookahead 3 -motionBlurPixels', select(3, lambda f: -f['qc']['motionBlurPixels'])),
]:
    ii = sorted(idx_of[f['index']] for f in ch)
    reach = cum[ii[-1]]
    span = sum(float(np.linalg.norm(C[pool[a]['index']] - C[pool[b]['index']]))
               for a, b in zip(ii, ii[1:]))
    print('%-34s %9.2f %8.1f%% %9.2f' % (name, reach, 100*reach/cum[-1], span))
