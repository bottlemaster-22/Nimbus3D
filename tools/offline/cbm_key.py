import io, json, os, numpy as np
import project as P
D=P.D
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx,ty=(rw+15)//16,(rh+15)//16; NT=tx*ty
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses(); ks=sorted(poses.keys())

def inst(idx):
    rot,t=poses[idx]; R=P.quat_to_matrix(rot)
    ok,radius,tiles,ex,ey=P.project(col,count,R,t,fx,fy,cx,cy,tx,ty)
    mean=np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)
    cam=mean@R.T+t; z=cam[:,2]; iv=1.0/z
    mx=fx*cam[:,0]*iv+cx; my=fy*cam[:,1]*iv+cy
    a=np.maximum(0,np.floor((mx-ex)/16)).astype(np.int64)
    b=np.maximum(0,np.floor((my-ey)/16)).astype(np.int64)
    c=np.minimum(tx,np.ceil((mx+ex)/16)).astype(np.int64)
    d=np.minimum(ty,np.ceil((my+ey)/16)).astype(np.int64)
    s=ok&(tiles>0)
    T=[];Z=[]
    ai,bi,ci,di,zi=a[s],b[s],c[s],d[s],z[s]
    for i in range(len(ai)):
        for yy in range(bi[i],di[i]):
            r=np.arange(ai[i],ci[i],dtype=np.int64)+yy*tx
            T.append(r); Z.append(np.full(len(r),zi[i]))
    return np.concatenate(T), np.concatenate(Z)

def dkey(z, near, far, levels):
    return np.clip((z-near)/(far-near),0,1).astype(np.float32).__mul__(np.float32(levels)).astype(np.uint32)

def order_quality(T, Z, shift, levels, near, far, tag):
    dk = dkey(Z, near, far, levels)
    key = (T.astype(np.uint64)<<np.uint64(shift)) | dk.astype(np.uint64)
    top = int(key.max()).bit_length()
    passes = -(-top//4)
    o = np.lexsort((dk, T))
    zt=Z[o]; tl=T[o]; dq=dk[o]
    same = tl[1:]==tl[:-1]
    ties = same & (dq[1:]==dq[:-1])
    inv  = same & (zt[1:] < zt[:-1]-1e-9)
    print('  %-42s top bit %2d -> %d passes (%s) | quantum %6.3f mm | in-tile ties %5.2f%% | order inversions %5.2f%%'
          % (tag, top, passes, 'EVEN' if passes%2==0 else 'ODD',
             (far-near)/levels*1000, 100*ties.sum()/same.sum(), 100*inv.sum()/same.sum()))
    return passes

for idx in (4,25):
    T,Z = inst(idx)
    print('frame %d, %d instances, tiles %d (%d bits)' % (idx,len(T),NT,(NT-1).bit_length()))
    # per-pass digit occupancy on today's key
    dk = dkey(Z,0.05,100.0,65535.0)
    key = (T.astype(np.uint64)<<np.uint64(16)) | dk.astype(np.uint64)
    for p in range(8):
        u = np.unique((key>>np.uint64(4*p)) & np.uint64(0xF))
        print('    pass %d bits %2d-%2d: %2d distinct digits%s' % (p,4*p,4*p+3,len(u),'   <-- DEAD' if len(u)==1 else ''))
    order_quality(T,Z,16,65535.0,0.05,100.0,'today: tile<<16, d16 over 0.05-100 m')
    order_quality(T,Z,13, 8191.0,0.05,100.0,'naive 13-bit, SAME 0.05-100 m range')
    order_quality(T,Z,13, 8191.0,0.05, 12.0,'proposed: tile<<13, d13 over 0.05-12 m')
    order_quality(T,Z,13, 8191.0,0.05,  6.0,'tighter:  tile<<13, d13 over 0.05-6 m')
    print()
