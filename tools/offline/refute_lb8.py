"""How common are the degenerate near-axis splats across frames, and what does
the depth structure look like once they are excluded?"""
import io,json,os,sys
import numpy as np
import project as P
census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8')); sl=census['slices'][0]
RW,RH=sl['renderWidth'],sl['renderHeight']; tx,ty=(RW+15)//16,(RH+15)//16
fx,fy,cx,cy=P.load_intrinsics(RW,RH)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
smax=scale.max(1); poses=P.load_poses()
print('frames with refined poses: %d'%len(poses))
print('%5s %8s %8s %8s %9s'%('frame','vis','z<0.5','z<0.2','maxproj_px'))
tot=[]
frames=sorted(poses.keys())[:600:20]
for F in frames:
    rot,t=poses[F]; R=P.quat_to_matrix(rot); camv=mean@R.T+t; z=camv[:,2]
    iz=1.0/np.where(z!=0,z,1); mx=fx*camv[:,0]*iz+cx; my=fy*camv[:,1]*iz+cy
    # cheap footprint proxy: 3-sigma of the isotropic upper bound
    rad=3.0*fx*smax*np.abs(iz)
    on=(z>0.05)&(z<100)&(mx+rad>0)&(mx-rad<RW)&(my+rad>0)&(my-rad<RH)
    n1=(on&(z<0.5)).sum(); n2=(on&(z<0.2)).sum()
    mp=rad[on&(z<0.5)].max() if n1 else 0
    tot.append((on.sum(),n1,n2))
    print('%5d %8d %8d %8d %9.0f'%(F,on.sum(),n1,n2,mp))
a=np.array(tot)
print('median over %d sampled frames: visible %d, z<0.5 %d (%.2f%%), z<0.2 %d'
      %(len(frames),np.median(a[:,0]),np.median(a[:,1]),100*np.median(a[:,1])/np.median(a[:,0]),np.median(a[:,2])))
