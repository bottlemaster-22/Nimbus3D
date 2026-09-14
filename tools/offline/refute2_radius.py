import numpy as np, project as P, os
from sv_clamp_run import run
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses(); fx,fy,cx,cy=P.load_intrinsics(720,540); tx,ty=45,34
ks=sorted(poses)[::20]
mx=np.zeros(count); giant=np.zeros(count,bool); ti=np.zeros(count)
for k in ks:
    R=P.quat_to_matrix(poses[k][0]); t=poses[k][1]
    ok,t_i,rad,off=run(col,count,R,t,fx,fy,cx,cy,tx,ty,None)
    r=np.where(ok,rad,0.0)
    mx=np.maximum(mx,r); ti+=t_i
    giant|=(ok&(off>1.3))
print('splats ever drawn in sample: %d'%(mx>0).sum())
print('ever beyond 1.3x halfFOV   : %d  (%.3f%% of model)'%(giant.sum(),100*giant.mean()))
print('their tile instances share : %.2f%%'%(100*ti[giant].sum()/ti.sum()))
for th in [720,1000,2000,3000,5000,10000]:
    sel=mx>th
    print(' maxRadius3sigma > %6d px : %6d splats (%.3f%%)  of which giants %6d  tileInst share %.2f%%  non-giant caught %d'
          %(th,sel.sum(),100*sel.mean(),(sel&giant).sum(),100*ti[sel].sum()/ti.sum(),(sel&~giant).sum()))
print('max radius percentiles over drawn: ',np.percentile(mx[mx>0],[50,90,99,99.9]).round(1))
print('giant max radius percentiles     : ',np.percentile(mx[giant],[1,10,50,90]).round(1))
