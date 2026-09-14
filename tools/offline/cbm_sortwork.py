"""Measure the real per-iteration sort workload: tile-instance count over the
training keyframe set, the block counts each radix pass dispatches, and the
bytes each pass moves. Everything below is measured on the build-250 model,
not assumed.
"""
import io, json, os, numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()
ks = sorted(poses.keys())
print('splats %d  render %dx%d  tiles %dx%d=%d  poses %d' % (count, rw, rh, tx, ty, tx*ty, len(ks)))

sample = ks[::7]
tot = []
for idx in sample:
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    tot.append(int(tiles[ok].sum()))
a = np.array(tot, float)
print('instances over %d sampled views: mean %.0f  median %.0f  p90 %.0f  max %.0f'
      % (len(a), a.mean(), np.median(a), np.percentile(a,90), a.max()))
print('census peakTileInstances %d at splatCount %d'
      % (sl['peakTileInstances'], sl['splatCountAtPeakTileInstances']))

BLK = 256*4
def blocks(n): return (n + BLK - 1)//BLK
for n in (int(a.mean()), sl['peakTileInstances']):
    b = blocks(n); he = 16*b
    lvl = []
    m = he
    while True:
        bb = blocks(m); lvl.append((m,bb))
        if bb <= 1: break
        m = bb
    print('\nn=%d -> %d sort blocks, histogram entries %d, scan levels %s'
          % (n, b, he, lvl))
    # bytes per pass: histogram reads keys (4B); scatter reads key+value (8B)
    # and writes key+value (8B); scan touches histogram twice-ish.
    per_pass = n*4 + n*16 + he*4*3
    print('   bytes per radix pass ~ %.2f MB ; x8 passes = %.1f MB ; x7 = %.1f ; x6 = %.1f'
          % (per_pass/1e6, per_pass*8/1e6, per_pass*7/1e6, per_pass*6/1e6))
    print('   dispatches per pass = 2 (hist,scatter) + %d (scan) = %d ; x8 = %d'
          % (2*len(lvl)-1+0, 2+2*len(lvl)-1, 8*(2+2*len(lvl)-1)))
