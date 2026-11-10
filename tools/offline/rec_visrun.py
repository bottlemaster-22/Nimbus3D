"""How much is the visibility mask worth, and how much headroom does reordering have?

Entry 25 argued the sparse-Adam early return already recovers "between 16 and 68
percent of the four 48-byte streams" at an ASSUMED f of 0.25-0.5, and concluded
that Morton ordering is the only remaining lever. Measure f, measure the
correlation that is already there, and bound what reordering could add.
"""
import io, json, os
import numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
tx, ty = (RW+15)//16, (RH+15)//16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())[:16]

masks = []
for k in keys:
    rot, t = poses[k]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    masks.append(ok & (tiles > 0))
vm = np.stack(masks)
f = vm.mean()
print('splats %d, views %d, measured visible fraction f = %.4f' % (count, len(keys), f))
print()
print(' record  bytes/line  gaussians/line   P(line fully masked)   independent (1-f)^g   '
      'perfect-locality ceiling')
for rec, L in ((48, 64), (48, 128), (108, 128), (48, 256)):
    g = max(1, L//rec)
    per, ceil_ = [], []
    for r in range(vm.shape[0]):
        m = vm[r]
        n = (len(m)//g)*g
        blk = m[:n].reshape(-1, g)
        per.append((~blk.any(axis=1)).mean())
        # perfect locality: sort the mask so all visible are contiguous
        s = np.sort(m[:n])[::-1].reshape(-1, g)
        ceil_.append((~s.any(axis=1)).mean())
    print('%7d %11d %15d %22.4f %21.4f %24.4f'
          % (rec, L, g, float(np.mean(per)), (1-f)**g, float(np.mean(ceil_))))
print()
print('bytes actually skipped per iteration by the mask, four 48-byte streams '
      '(splat, grad, adamM, adamV) at %d splats:' % count)
for L in (64, 128):
    g = max(1, L//48)
    per = []
    for r in range(vm.shape[0]):
        m = vm[r]; n=(len(m)//g)*g
        per.append((~m[:n].reshape(-1,g).any(axis=1)).mean())
    p = float(np.mean(per))
    full = count*48*4*2/1e6       # read+write, four streams
    print('  %3d B lines: %.1f%% of lines skipped -> %.1f MB of %.1f MB'
          % (L, 100*p, full*p, full))
