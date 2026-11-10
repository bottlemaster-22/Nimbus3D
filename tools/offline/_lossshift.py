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
names = list(col.keys())
ls = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
ls = np.clip(ls, -12, 3)
lam = np.exp(2*ls); p = lam/np.maximum(lam.sum(1, keepdims=True), 1e-20)
H = -(p*np.log(np.maximum(p, 1e-20))).sum(1); rank = np.exp(H)
disc = 0.001*0.5*(rank-2.0)**2
sc = np.exp(ls); over = np.maximum(sc-0.5, 0); hinge = 0.05*0.5*(over**2).sum(1)
poses = P.load_poses(); keys = sorted(poses.keys())[:16]
fr = []
for k in keys:
    rot, t = poses[k]; R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    m = ok & (tiles > 0)
    fr.append(((disc+hinge)[m].sum(), (disc+hinge).sum()))
fr = np.array(fr)
print('fields', names[:12])
print('prior loss per iteration, all splats  : %.3f' % fr[:,1].mean())
print('prior loss per iteration, visible only: %.3f  (ratio %.3f)' % (fr[:,0].mean(), (fr[:,0]/fr[:,1]).mean()))
print('loss census fields:', {k: v for k, v in census.items() if 'loss' in k.lower() and not isinstance(v, (list, dict))})
