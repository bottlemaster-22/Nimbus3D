"""How the sort's element count moves with splat size -- the project's own
stated direction (19.6 mm -> Scaniverse's 3.39 mm).
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
ks = sorted(poses.keys())[::40]
print('views %d' % len(ks))
base = None
for mul in (1.0, 0.75, 0.5, 0.3, 0.173):
    c = dict(col)
    lg = np.float32(np.log(mul))
    for k in ('scale_0','scale_1','scale_2'):
        c[k] = col[k] + lg
    tot = []
    for idx in ks:
        rot, t = poses[idx]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(c, count, R, t, fx, fy, cx, cy, tx, ty)
        tot.append(int(tiles.sum()))
    a = np.array(tot, float)
    if base is None: base = a.mean()
    print('  size x%.3f (median largest axis ~%5.2f mm): mean instances %9.0f  = %.2fx  '
          'sort bytes/iter (8 passes, k+v, r+w) %6.1f MB'
          % (mul, 19.6*mul, a.mean(), a.mean()/base, a.mean()*8*2*8/1e6), flush=True)
