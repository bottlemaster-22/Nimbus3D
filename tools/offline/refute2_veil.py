import numpy as np, project as P, os
from sv_clamp_run import run
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses(); fx,fy,cx,cy=P.load_intrinsics(720,540); tx,ty=45,34
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
op=1/(1+np.exp(-col['opacity'].astype(np.float64)))
ks=sorted(poses)[::20]
allz=[];alld=[]
for k in ks[:12]:
    R=P.quat_to_matrix(poses[k][0]); t=poses[k][1]
    cam=mean@R.T+t; z=cam[:,2]
    ok,ti,rad,off=run(col,count,R,t,fx,fy,cx,cy,tx,ty,None)
    m=ok&(off>1.3)
    if m.sum()==0: continue
    zs=z[m]; zall=z[ok]
    # fraction of visible splats that are BEHIND each giant one
    frac=[(zall>zz).mean() for zz in zs]
    allz.append(zs); alld.append(np.array(frac))
    print('pose %s: giants %3d  z p50 %.3f m  (visible z p50 %.2f m)  median frac of visible splats behind them %.3f  their tiles p50 %.0f'
          %(k,m.sum(),np.median(zs),np.median(zall),np.median(frac),np.median(ti[m])))
z=np.concatenate(allz); d=np.concatenate(alld)
print('ALL: n=%d  z p10 %.3f p50 %.3f p90 %.3f   behind-fraction p50 %.3f'%(len(z),*np.percentile(z,[10,50,90]),np.median(d)))
