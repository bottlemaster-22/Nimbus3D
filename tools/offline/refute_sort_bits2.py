"""Vectorised: per-tile instance counts via a 2D difference array, then the
per-pass radix digit occupancy on the REAL key encoding.
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

def per_tile(idx):
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    inv = 1.0/cam[:,2]
    mx = fx*cam[:,0]*inv + cx; my = fy*cam[:,1]*inv + cy
    a  = np.maximum(0, np.floor((mx-ex)/16)).astype(np.int64)
    b  = np.maximum(0, np.floor((my-ey)/16)).astype(np.int64)
    c  = np.minimum(tx, np.ceil((mx+ex)/16)).astype(np.int64)
    d  = np.minimum(ty, np.ceil((my+ey)/16)).astype(np.int64)
    s = ok & (tiles>0)
    a,b,c,d = a[s],b[s],c[s],d[s]
    diff = np.zeros((ty+1, tx+1), dtype=np.int64)
    np.add.at(diff, (b, a),  1)
    np.add.at(diff, (b, c), -1)
    np.add.at(diff, (d, a), -1)
    np.add.at(diff, (d, c),  1)
    g = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:ty,:tx]
    return g, int(tiles.sum())

print('tiles %dx%d = %d  (11 bits needed: 2^10=1024 < %d <= 2^11=2048)' % (tx,ty,tx*ty,tx*ty))
for idx in (4, 25, ks[len(ks)//2]):
    g, n = per_tile(idx)
    assert abs(g.sum()-n) < 2, (g.sum(), n)
    occ = g.ravel() > 0
    tid = np.arange(tx*ty)
    inst = g.ravel()
    print('\nframe %d: %d instances, %d/%d tiles occupied (max tile id used %d)'
          % (idx, n, occ.sum(), tx*ty, tid[occ].max()))
    for p in range(4, 8):
        dig = ((tid << 16) >> (4*p)) & 0xF
        vals = np.unique(dig[occ])
        share = np.array([inst[occ & (dig==v)].sum() for v in vals]) / n
        print('  pass %d (key bits %2d-%2d): %d distinct digits %s  instance share %s%s'
              % (p, 4*p, 4*p+3, len(vals), list(vals),
                 np.round(share,3).tolist(),
                 '   <-- TRUE NO-OP' if len(vals)==1 else ''))
    # how many instances carry a NON-ZERO pass-6 digit
    dig6 = ((tid << 16) >> 24) & 0xF
    nz = inst[occ & (dig6 != 0)].sum()
    print('  instances with pass-6 digit != 0: %d of %d = %.1f%%' % (nz, n, 100*nz/n))
