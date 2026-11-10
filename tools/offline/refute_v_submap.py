"""Does the pose-graph residual actually reach the trainer's supervision?

The pose graph corrects 19 SUBMAPS rigidly. Within a submap the relative
geometry is untouched VIO. So a splat whose views all sit inside one submap
cannot be hurt by any residual the graph left between submaps.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
r = json.load(io.open(os.path.join(P.D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
sub = r['submaps']
print('submaps %d' % len(sub))
tr = np.array([np.linalg.norm(s['correction']['translation']) for s in sub])
rot = np.array([2*np.degrees(np.arcsin(min(1.0, np.linalg.norm(s['correction']['rotation'][:3]))))
                for s in sub])
print('per-submap correction |t| cm: p10 %.2f MEDIAN %.2f p90 %.2f max %.2f'
      % tuple(list(100*np.percentile(tr, [10, 50, 90])) + [100*tr.max()]))
print('per-submap correction  rot deg: p10 %.3f MEDIAN %.3f p90 %.3f max %.3f'
      % tuple(list(np.percentile(rot, [10, 50, 90])) + [rot.max()]))
for s in sub:
    print('  submap %2d frames %4d-%4d  |t| %6.2f cm  rot %5.3f deg'
          % (s['index'], s['firstFrame'], s['lastFrame'], 100*np.linalg.norm(s['correction']['translation']), 
             2*np.degrees(np.arcsin(min(1.0, np.linalg.norm(s['correction']['rotation'][:3]))))))

# relative correction between CONSECUTIVE (overlapping) submaps: the
# discontinuity the graph actually leaves along the walk
rel = []
for a, b in zip(sub, sub[1:]):
    ta = np.asarray(a['correction']['translation']); tb = np.asarray(b['correction']['translation'])
    rel.append(np.linalg.norm(tb-ta))
rel = np.array(rel)
print('\nCONSECUTIVE-submap relative correction |dt| cm: p10 %.2f MEDIAN %.2f p90 %.2f max %.2f'
      % tuple(list(100*np.percentile(rel, [10, 50, 90])) + [100*rel.max()]))

# --- how many submaps does each splat's view set span?
b, rr, frames, pool, kf, info = keyframes()
kfidx = [f['index'] for f in kf]
# a frame belongs to every submap whose [firstFrame,lastFrame] contains it;
# assign it to the FIRST such (submaps overlap by design)
def submap_of(fi):
    for s in sub:
        if s['firstFrame'] <= fi <= s['lastFrame']:
            return s['index']
    return -1
kfsub = np.array([submap_of(fi) for fi in kfidx])
print('\n120 keyframes fall in %d distinct submaps: %s'
      % (len(set(kfsub.tolist())), sorted(set(kfsub.tolist()))))

viso = np.unpackbits(np.load(os.path.join(OUT, '_cap_viso.npy')), axis=1)[:, :len(kf)].astype(bool)
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
V = viso[samp]
nsub = np.array([len(set(kfsub[V[i]].tolist())) for i in range(V.shape[0])])
nv = V.sum(axis=1)
print('distinct submaps per sampled splat (occlusion-aware views):')
print('  p10 %d p25 %d MEDIAN %d p75 %d p90 %d   frac in ONE submap %.1f%%   frac in <=2 %.1f%%'
      % (*np.percentile(nsub[nv > 0], [10, 25, 50, 75, 90]).astype(int),
         100*np.mean(nsub[nv > 0] == 1), 100*np.mean(nsub[nv > 0] <= 2)))
np.save(os.path.join(OUT, '_v_nsub.npy'), nsub)
np.save(os.path.join(OUT, '_v_kfsub.npy'), kfsub)
