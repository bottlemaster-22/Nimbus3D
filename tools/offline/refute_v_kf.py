"""REFUTE the sharpness finding's headline number.

The finding quotes '-48% median blur' from a SHARPEST-IN-BUCKET set built over
120 even buckets across the WHOLE walk (last frame 867).  Its proposedChange is
a different thing: sharpest WITHIN the greedy loop's own acceptance window.
Measure the window the greedy loop actually leaves, then measure that variant.
"""
import io, json, os
import numpy as np
from cap_keyframes import load_all, select_keyframes, split_held_out

b, r, frames = load_all()
usable = [f for f in frames if f['weight'] > 0.05]
pool = usable if usable else frames
sel, info = select_keyframes(pool, 120)
print('pool %d  path %.3f m  spacing %.4f m  chosen %d  first %d last %d'
      % (len(pool), info['pathLength'], info['spacing'], len(sel),
         pool[sel[0]]['index'], pool[sel[-1]]['index']))

blur = np.array([f['qc'].get('motionBlurPixels', np.nan) for f in pool], float)
print('\nqc.motionBlurPixels present on %d/%d pool frames' % (np.isfinite(blur).sum(), len(pool)))
S = 720.0/1920.0

def rep(name, idxs):
    v = blur[np.asarray(idxs)]
    print('  %-42s n=%3d  native p50 %5.2f p90 %5.2f | render p50 %5.2f p90 %5.2f  >1px %d'
          % (name, len(idxs), np.percentile(v, 50), np.percentile(v, 90),
             S*np.percentile(v, 50), S*np.percentile(v, 90), int((S*v > 1).sum())))

print('\n=== blur of candidate keyframe sets ===')
rep('GREEDY (what shipped)', sel)
# how wide is the greedy loop's acceptance window?  i.e. how many pool frames
# were SKIPPED between one accepted keyframe and the next.
gaps = np.diff(np.asarray(sel))
print('\ngreedy acceptance gaps (pool frames between consecutive keyframes):')
print('  p10 %d p25 %d MEDIAN %d p75 %d p90 %d max %d   gap==1: %.1f%%'
      % (*np.percentile(gaps, [10, 25, 50, 75, 90]).astype(int), gaps.max(),
         100*np.mean(gaps == 1)))
print('  a window of size g offers g candidates to pick the sharpest from.')

# THE PROPOSED CHANGE: same greedy spacing rule, but once the rule admits a
# frame, take the sharpest frame in the run of frames since the last keyframe.
def greedy_sharpest_in_window(pool, target, spacing):
    chosen = []
    lastC = lastF = None
    window = []
    for i, fr in enumerate(pool):
        window.append(i)
        if lastC is not None:
            moved = float(np.linalg.norm(lastC - fr['C']))
            turned = 1.0 - float(np.dot(lastF/np.linalg.norm(lastF), fr['F']/np.linalg.norm(fr['F'])))
            if moved < spacing and turned < 0.02:
                continue
        k = min(window, key=lambda j: blur[j])
        chosen.append(k)
        lastC, lastF = pool[k]['C'], pool[k]['F']
        window = []
        if len(chosen) >= target:
            break
    return chosen

sw = greedy_sharpest_in_window(pool, 120, info['spacing'])
rep('PROPOSED: sharpest within greedy window', sw)

# the number the finding actually quotes: sharpest in each of 120 even buckets
edges = np.linspace(0, len(pool), 121).astype(int)
bucket = [int(np.arange(edges[i], edges[i+1])[np.argmin(blur[edges[i]:edges[i+1]])])
          for i in range(120)]
rep('QUOTED: sharpest in 120 even buckets', bucket)
rep('even stride over the whole walk', list(range(0, len(pool), max(1, len(pool)//120)))[:120])
print('\n  greedy last pool frame %d ; sharpest-in-window last %d ; bucket last %d'
      % (pool[sel[-1]]['index'], pool[sw[-1]]['index'], pool[bucket[-1]]['index']))

# --- the resampling floor: 1920 -> 720 is a 2.67x box, ~1 render px of blur
print('\n=== the 1920 -> 720 resample is itself a low-pass ===')
box = 1920.0/720.0
sig_box = box/np.sqrt(12.0)*S*np.sqrt(12.0)   # box width in render px = 1.0
print('  downsample box width = %.2f native px = %.2f render px' % (box, box*S))
for name, idxs in (('greedy', sel), ('sharpest-in-window', sw), ('bucket', bucket)):
    mb = S*np.median(blur[np.asarray(idxs)])
    tot = np.sqrt(mb**2 + 1.0**2)
    print('  %-20s motion %.2f render px -> total with resample %.2f render px'
          % (name, mb, tot))
