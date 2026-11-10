import json, io, os, numpy as np
import project as P
D=P.D
vis=np.load('_vis.npy'); allkf=np.load('_kf.npy')
poses=P.load_poses()
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
mean=np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)

def qinv_act(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v)
    return v+w*t+np.cross(qv,t)
cams=np.array([-qinv_act(poses[i][0],poses[i][1]) for i in allkf])
tr=np.array([i%10!=5 for i in range(len(allkf))])
CT=cams[tr]; VT=vis[tr]
print('=== CAMERA GEOMETRY OF THE 108 TRAINED VIEWS ===')
print('camera path bbox size (m):',np.round(CT.max(0)-CT.min(0),3))
d=np.linalg.norm(CT[:,None,:]-CT[None,:,:],axis=2)
iu=np.triu_indices(len(CT),1)
print('pairwise camera baseline (m): median %.3f  p90 %.3f  MAX %.3f'%(np.median(d[iu]),np.percentile(d[iu],90),d[iu].max()))
# PCA of camera centres
X=CT-CT.mean(0); ev=np.linalg.svd(X,compute_uv=False)**2/len(CT)
print('camera-centre PCA std (m): %.3f %.3f %.3f   (ratios %.2f %.2f)'%(
    np.sqrt(ev[0]),np.sqrt(ev[1]),np.sqrt(ev[2]),np.sqrt(ev[1]/ev[0]),np.sqrt(ev[2]/ev[0])))

rng=np.random.default_rng(0)
samp=rng.choice(count,40000,replace=False)
maxang=np.zeros(len(samp)); nviews=np.zeros(len(samp),int); meandepth=np.zeros(len(samp))
for j,s in enumerate(samp):
    ks=np.where(VT[:,s])[0]
    nviews[j]=len(ks)
    if len(ks)<2: continue
    dirs=CT[ks]-mean[s]
    r=np.linalg.norm(dirs,axis=1); meandepth[j]=r.mean()
    u=dirs/r[:,None]
    cosm=(u@u.T)
    maxang[j]=np.degrees(np.arccos(np.clip(cosm.min(),-1,1)))
ok=nviews>=2
print()
print('=== MAX TRIANGULATION ANGLE PER GAUSSIAN (40k sample, >=2 trained views) ===')
print('median %.2f deg   p10 %.2f   p25 %.2f   p75 %.2f   p90 %.2f   max %.2f'%(
    np.median(maxang[ok]),*[np.percentile(maxang[ok],q) for q in (10,25,75,90)],maxang[ok].max()))
for th in [1,2,3,5,10,15,20]:
    print('  max angle < %2d deg: %5.2f%%'%(th,100*(maxang[ok]<th).mean()))
print()
print('median distance from Gaussian to the cameras that see it: %.2f m'%np.median(meandepth[ok]))
print()
print('=== DEPTH UNCERTAINTY IMPLIED BY THAT PARALLAX ===')
# sigma_z ~ z^2 * sigma_px / (f * B)  ; equivalently sigma_z ~ z*sigma_px/(f*tan(angle))
rw,rh=720,540
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
ang=np.radians(np.maximum(maxang[ok],1e-6)); z=meandepth[ok]
sig_px=0.5
sz=z*sig_px/(fx*np.tan(ang))
print('f = %.1f px at 720x540, assume 0.5 px matching error'%fx)
print('implied depth sigma: median %.1f mm   p75 %.1f mm  p90 %.1f mm'%(1000*np.median(sz),1000*np.percentile(sz,75),1000*np.percentile(sz,90)))
print('LiDAR measured median sigma (prepass): 13.0 mm')
np.save('_maxang.npy',maxang); np.save('_samp.npy',samp); np.save('_nvsamp.npy',nviews)
