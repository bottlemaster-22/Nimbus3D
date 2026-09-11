"""The COUPLED change: (tile << 13) | depth13, 24 key bits, 6 passes, even.
Verifies the render is complete and measures the ordering cost of dropping
depth from 16 to 13 bits.
"""
import io, json, os, numpy as np
import project as P
D = P.D
census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
NT = tx*ty
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()
NEAR, FAR = 0.05, 100.0

def build(idx, shift, levels):
    rot, t = poses[idx]; R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:,2]; inv = 1.0/z
    mx = fx*cam[:,0]*inv + cx; my = fy*cam[:,1]*inv + cy
    a = np.maximum(0, np.floor((mx-ex)/16)).astype(np.int64)
    b = np.maximum(0, np.floor((my-ey)/16)).astype(np.int64)
    c = np.minimum(tx, np.ceil((mx+ex)/16)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my+ey)/16)).astype(np.int64)
    s = ok & (tiles>0)
    norm = np.clip((z-NEAR)/(FAR-NEAR), 0, 1)
    dk = (norm*float(levels)).astype(np.uint32)
    ai,bi,ci,di,dki = a[s],b[s],c[s],d[s],dk[s]
    gid = np.nonzero(s)[0]; zz = z[s]
    K=[];V=[];Z=[]
    for i in range(len(ai)):
        for yy in range(bi[i], di[i]):
            row = np.arange(ai[i], ci[i], dtype=np.uint32) + np.uint32(yy*tx)
            K.append((row << np.uint32(shift)) | np.uint32(dki[i]))
            V.append(np.full(len(row), gid[i], dtype=np.uint32))
            Z.append(np.full(len(row), zz[i]))
    return np.concatenate(K), np.concatenate(V), np.concatenate(Z)

def radix(k, v, passes):
    for p in range(passes):
        o = np.argsort((k >> np.uint32(4*p)) & np.uint32(0xF), kind='stable')
        k = k[o]; v = v[o]
    return k, v, passes % 2

def check(k, shift):
    n=len(k); tile=(k>>np.uint32(shift)).astype(np.int64)
    rng=np.zeros((NT,2),dtype=np.int64); rng[tile[0],0]=0
    ch=np.nonzero(tile[1:]!=tile[:-1])[0]+1
    for g,pv,cv in zip(ch, tile[ch-1], tile[ch]):
        rng[pv,1]=g; rng[cv,0]=g
    rng[tile[-1],1]=n
    return int(np.maximum(rng[:,1]-rng[:,0],0).sum()), n

for idx in (4,25):
    print('\n--- frame %d ---' % idx)
    k16,v16,z16 = build(idx, 16, 65535.0)
    k13,v13,z13 = build(idx, 13,  8191.0)
    for tag,(k,v,z,shift,passes) in {
        'today   (tile<<16 | d16, 8 passes)': (k16,v16,z16,16,8),
        'coupled (tile<<13 | d13, 6 passes)': (k13,v13,z13,13,6),
    }.items():
        ks,vs,par = radix(k,v,passes)
        vis,n = check(ks, shift)
        print('  %s: parity %d -> %s ; composites %d/%d = %.1f%%'
              % (tag, par, 'keysA' if par==0 else 'keysB', vis, n, 100*vis/n))
    # ordering cost: within-tile adjacent pairs that TIE on depth
    for tag,(k,z,shift,passes) in {'d16':(k16,z16,16,8),'d13':(k13,z13,13,6)}.items():
        ks,zs,_ = radix(k,z.astype(np.uint32)*0+np.arange(len(k),dtype=np.uint32), passes)
        # recover depth order correctness: compare sorted depth sequence within tiles
        order = np.lexsort((k & np.uint32((1<<shift)-1), (k>>np.uint32(shift))))
        zt = z[order]; tl = (k[order]>>np.uint32(shift))
        same = tl[1:]==tl[:-1]
        inv = same & (zt[1:] < zt[:-1] - 1e-9)     # strictly out of true depth order
        ties = same & (((k[order]&np.uint32((1<<shift)-1))[1:]) == ((k[order]&np.uint32((1<<shift)-1))[:-1]))
        print('     %s: %d/%d adjacent in-tile pairs tie on the quantised depth (%.2f%%), %d truly inverted (%.3f%%)'
              % (tag, ties.sum(), same.sum(), 100*ties.sum()/same.sum(), inv.sum(), 100*inv.sum()/same.sum()))
