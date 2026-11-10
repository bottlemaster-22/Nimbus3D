"""Upper bound: the clamped max-over-views radius using EVERY refined pose,
not just a 120-view keyframe-sized sample. The trainer only ever renders the
120 selected keyframes, so this is a strict over-estimate of what
stats.maxRadiusPxBits can reach.
"""
import io, json, os, sys
import numpy as np
import project as P
from dim_oversize2 import per_view, world_covariance

census = json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D,'model','model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())
mean = np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)
sigma, scale = world_covariance(col, count)
opacity = 1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))
mxr = np.zeros(count); seen = np.zeros(count,bool); inst = np.zeros(count,np.int64)
for n,f in enumerate(keys):
    R = P.quat_to_matrix(poses[f][0])
    ok, radius, tiles = per_view(mean, sigma, opacity, R, poses[f][1], fx, fy, cx, cy, tx, ty, rw, rh, True)
    mxr = np.where(ok, np.maximum(mxr, radius), mxr); seen |= ok; inst += tiles
    if n % 100 == 0:
        print('  ...%d/%d  running max %.2f px' % (n, len(keys), mxr.max())); sys.stdout.flush()
d = mxr[seen]
print('ALL %d poses, clamped: drawn %d  max %.2f px  p99 %.2f  p99.9 %.2f'
      % (len(keys), int(seen.sum()), d.max(), np.percentile(d,99), np.percentile(d,99.9)))
tot = inst.sum()
for thr in (720,540,360,180,120,90):
    m = seen & (mxr > thr)
    print('  >%4d px: %7d splats (%.4f%%)  instShare %.4f%%'
          % (thr, int(m.sum()), 100.0*m.sum()/count, 100.0*inst[m].sum()/tot))
