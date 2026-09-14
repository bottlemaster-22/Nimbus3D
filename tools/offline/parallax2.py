"""Triangulation angle using ACTUAL rendered contribution, not frustum overlap.
The QC card claims '~10 degrees of angular spread'; the frustum-only number was
60.7 deg, which counts cameras on the far side of the wall a Gaussian sits on."""
import numpy as np, os, json
import project as P
D=P.D
C=np.load('_contrib.npy')            # (120, 300000) sum of alpha*T per view
allkf=np.load('_kf.npy'); poses=P.load_poses()
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
def qi(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v); return v+w*t+np.cross(qv,t)
cams=np.array([-qi(poses[i][0],poses[i][1]) for i in allkf])
tr=np.array([i%10!=5 for i in range(len(allkf))])
CT=cams[tr]; CC=C[tr]
frus=np.load('_vis.npy')[tr]
print('=== HOW MUCH LIGHT EACH GAUSSIAN ACTUALLY PUTS ON SCREEN ===')
tot=CC.sum(0)
print('sum of alpha*T over all 108 trained views: median %.2f  p10 %.3f  p90 %.1f'%(
  np.median(tot),np.percentile(tot,10),np.percentile(tot,90)))
for TH in [0.5,1.0,2.0]:
    n=(CC>TH).sum(0)
    print('views where it contributes > %.1f pixel-units: mean %.2f  median %d  <=1 view: %.2f%%'%(
      TH,n.mean(),np.median(n),100*(n<=1).mean()))
print()
print('frustum-visible views per Gaussian (the earlier, inflated number): mean %.2f'%frus.sum(0).mean())
TH=1.0
VIS=CC>TH
rng=np.random.default_rng(0); samp=rng.choice(count,40000,replace=False)
for label,V in [('CONTRIBUTION (alpha*T>1.0)',VIS),('FRUSTUM ONLY',frus)]:
    ang=[];nv=[]
    for s in samp:
        k=np.where(V[:,s])[0]
        nv.append(len(k))
        if len(k)<2: ang.append(0.0); continue
        d=CT[k]-mean[s]; d/=np.linalg.norm(d,axis=1)[:,None]
        ang.append(np.degrees(np.arccos(np.clip((d@d.T).min(),-1,1))))
    ang=np.array(ang); nv=np.array(nv); ok=nv>=2
    print()
    print('--- %s ---'%label)
    print('Gaussians with >=2 such views: %.1f%%'%(100*ok.mean()))
    print('max triangulation angle: median %.1f deg  p10 %.1f  p25 %.1f  p75 %.1f'%(
      np.median(ang[ok]),np.percentile(ang[ok],10),np.percentile(ang[ok],25),np.percentile(ang[ok],75)))
    for t in [5,10,15,20,30]:
        print('   under %2d deg: %5.1f%%'%(t,100*(ang[ok]<t).mean()))
