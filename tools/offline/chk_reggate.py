"""How much of trainer_regularizer's work the sparse-Adam gate would remove.

Per-lane traffic saving = 1 - f.  Per-SIMD-group ALU/launch saving = fraction of
32-consecutive-gid groups with no visible splat at all.  Also counts the
lossAccum atomics removed.
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
print('splats %d  views %d  f = %.4f  (1-f = %.4f)' % (count, len(keys), f, 1-f))

for g in (8, 16, 32, 64, 256):
    per = []
    for r in range(vm.shape[0]):
        m = vm[r]; n = (len(m)//g)*g
        per.append((~m[:n].reshape(-1, g).any(axis=1)).mean())
    print('  groups of %4d consecutive gid: %.4f entirely invisible' % (g, float(np.mean(per))))

# Traffic model for trainer_regularizer, per splat, per iteration.
# reads: TrainerSplat 48 B, TrainerSplatStats 32 B; read+write TrainerSplatGrad 48+48.
per_splat = 48 + 32 + 48 + 48
total = count * per_splat
wasted = total * (1 - f)
print()
print('regulariser traffic %.1f MB/iter, wasted %.1f MB/iter' % (total/1e6, wasted/1e6))
print('over 4000 iters: %.1f GB wasted' % (wasted*4000/1e9))
for bw in (68e9, 100e9, 150e9):
    print('   at %5.0f GB/s -> %.2f s' % (bw/1e9, wasted*4000/bw))
print()
print('lossAccum atomics removed: %d/iter (disc term), %.2f G over the run'
      % (count*(1-f), count*(1-f)*4000/1e9))
