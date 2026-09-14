"""Measure the effect of the INRIA tangent clamp on the EWA Jacobian, using the
trainer's EXACT alpha-threshold tile box so the baseline reproduces the census
peakTileInstances (1,058,989 at 299,352 splats)."""
import numpy as np, project as P

TILE=16; MIN_A=1/255.
def run(col, count, R, t, fx, fy, cx, cy, tx, ty, clamp):
    mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
    cam=mean@R.T+t; z=cam[:,2]
    scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
    q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True); x_,y_,z_,w_=q.T
    Rm=np.empty((count,3,3))
    Rm[:,0,0]=1-2*(y_*y_+z_*z_); Rm[:,0,1]=2*(x_*y_-w_*z_); Rm[:,0,2]=2*(x_*z_+w_*y_)
    Rm[:,1,0]=2*(x_*y_+w_*z_); Rm[:,1,1]=1-2*(x_*x_+z_*z_); Rm[:,1,2]=2*(y_*z_-w_*x_)
    Rm[:,2,0]=2*(x_*z_-w_*y_); Rm[:,2,1]=2*(y_*z_+w_*x_); Rm[:,2,2]=1-2*(x_*x_+y_*y_)
    M=Rm*scale[:,None,:]; sig=M@np.transpose(M,(0,2,1)); sc=R@sig@R.T
    iz=1/z; iz2=iz*iz
    ex_, ey_ = cam[:,0], cam[:,1]
    if clamp is not None:                      # INRIA computeCov2D tangent clamp
        lx=clamp*(cx/fx); ly=clamp*(cy/fy)
        ex_=np.clip(cam[:,0]*iz,-lx,lx)*z; ey_=np.clip(cam[:,1]*iz,-ly,ly)*z
    j00=fx*iz; j11=fy*iz; j02=-fx*ex_*iz2; j12=-fy*ey_*iz2
    s00,s01,s02=sc[:,0,0],sc[:,0,1],sc[:,0,2]; s11,s12,s22=sc[:,1,1],sc[:,1,2],sc[:,2,2]
    a0=j00*s00+j02*s02; a1=j00*s01+j02*s12; a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12; b2=j11*s12+j12*s22
    sa=a0*j00+a2*j02; sb=a1*j11+a2*j12; sc2=b1*j11+b2*j12
    db=np.maximum(sa*sc2-sb*sb,1e-12); sa=sa+0.25; sc2=sc2+0.25
    det=sa*sc2-sb*sb; comp=np.sqrt(np.clip(db/np.maximum(det,1e-12),0,1))
    mid=0.5*(sa+sc2); disc=np.sqrt(np.maximum(mid*mid-det,1e-9)); rad=3*np.sqrt(np.maximum(mid+disc,1e-9))
    op=1/(1+np.exp(-col['opacity'].astype(np.float64))); alpha=op*comp
    lvl=np.clip(2*np.log(alpha/MIN_A),0,9); k=np.sqrt(lvl)
    exs=k*np.sqrt(np.maximum(sa,1e-9))*1.001; eys=k*np.sqrt(np.maximum(sc2,1e-9))*1.001
    mx=fx*cam[:,0]*iz+cx; my=fy*cam[:,1]*iz+cy
    x0=np.maximum(0,np.floor((mx-exs)/TILE)); x1=np.minimum(tx,np.ceil((mx+exs)/TILE))
    y0=np.maximum(0,np.floor((my-eys)/TILE)); y1=np.minimum(ty,np.ceil((my+eys)/TILE))
    tiles=np.maximum(x1-x0,0)*np.maximum(y1-y0,0)
    ok=(z>0.05)&(z<100)&(det>1e-12)&(rad>=0.5)&(alpha>=MIN_A)&(tiles>0)
    return ok, np.where(ok,tiles,0), rad, np.abs(cam[:,0]*iz)/(cx/fx)

if __name__=='__main__':
    col,count=P.load_ply(P.os.path.join(P.D,'model','model.ply'))
    poses=P.load_poses(); fx,fy,cx,cy=P.load_intrinsics(720,540); tx,ty=45,34
    ks=sorted(poses)[::20]
    print('census peakTileInstances 1058989 at 299352 splats = 3.538 tiles/splat')
    for clamp in [None,1.3,1.15,1.0]:
        tot=[];vis=[]
        for k in ks:
            R=P.quat_to_matrix(poses[k][0]); t=poses[k][1]
            ok,ti,rad,off=run(col,count,R,t,fx,fy,cx,cy,tx,ty,clamp)
            tot.append(ti.sum()); vis.append(ok.sum())
        tot=np.array(tot); vis=np.array(vis)
        nm='none (current)' if clamp is None else '%.2f x half-FOV'%clamp
        print(' clamp %-16s : peak tileInstances %9d   mean %9d   mean visible %7d   tiles/visible %.3f'
              % (nm, tot.max(), tot.mean(), vis.mean(), tot.mean()/vis.mean()))
    # where do the tile instances come from
    R=P.quat_to_matrix(poses[ks[len(ks)//2]][0]); t=poses[ks[len(ks)//2]][1]
    ok,ti,rad,off=run(col,count,R,t,fx,fy,cx,cy,tx,ty,None)
    print()
    print(' one view, no clamp: total tileInstances %d' % ti.sum())
    for lo,hi in [(0,1),(1,1.3),(1.3,2),(2,5),(5,1e9)]:
        m=ok&(off>=lo)&(off<hi)
        print('   |x/z| / halfFOVtan in [%4.1f,%5.1f): %6d splats, %8d tile instances (%5.2f%%), mean %8.1f tiles each, median 3sigma radius %8.1f px'
              % (lo,hi,m.sum(),ti[m].sum(),100*ti[m].sum()/ti.sum(), ti[m].sum()/max(m.sum(),1), np.median(rad[m]) if m.sum() else 0))
