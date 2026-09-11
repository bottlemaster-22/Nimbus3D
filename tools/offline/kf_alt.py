"""What the discarded half of the capture would add, measured the same way
for both selections (frustum visibility, so the comparison is like-for-like)."""
import json,io,os,numpy as np,time
import project as P
D=P.D
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx=(rw+15)//16; ty=(rh+15)//16
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses()
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
frames=sorted(bundle['frames'],key=lambda f:f['index'])
pool=[f['index'] for f in frames if f['qc']['weight']>0.05] or [f['index'] for f in frames]
shipped=[int(v) for v in np.load('_kf.npy')]
st=max(len(pool)//120,1)
alt=[pool[i] for i in range(0,len(pool),st)][:120]
def qi(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v); return v+w*t+np.cross(qv,t)
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
def measure(kfs,label):
    trn=[k for i,k in enumerate(kfs) if i%10!=5]
    V=np.zeros((len(trn),count),dtype=bool)
    for j,idx in enumerate(trn):
        R=P.quat_to_matrix(poses[idx][0]); t=poses[idx][1]
        ok,_,_,_,_=P.project(col,count,R,t,fx,fy,cx,cy,tx,ty)
        V[j]=ok
    cams=np.array([-qi(poses[i][0],poses[i][1]) for i in trn])
    nv=V.sum(0)
    rng=np.random.default_rng(0); s=rng.choice(count,15000,replace=False)
    med=[];mx=[]
    for i in s:
        k=np.where(V[:,i])[0]
        if len(k)<2: continue
        d=cams[k]-mean[i]; d/=np.linalg.norm(d,axis=1)[:,None]
        A=np.degrees(np.arccos(np.clip(d@d.T,-1,1))); iu=np.triu_indices(len(k),1)
        med.append(np.median(A[iu])); mx.append(A[iu].max())
    print('%-22s trained views %d | views/Gaussian mean %.2f  <=3 views %.2f%% | angle: median-pairwise %.1f deg, max %.1f deg'%(
      label,len(trn),nv.mean(),100*(nv<=3).mean(),np.median(med),np.median(mx)))
    return nv,np.array(med)
print('frustum visibility, identical method for both, so only the DIFFERENCE matters')
print()
a,ma=measure(shipped,'as shipped (greedy)')
b,mb=measure(alt,'even stride over all')
print()
print('even stride vs shipped:')
print('  views per Gaussian      : %.2f -> %.2f  (%+.0f%%)'%(a.mean(),b.mean(),100*(b.mean()/a.mean()-1)))
print('  Gaussians with <=3 views: %.2f%% -> %.2f%%'%(100*(a<=3).mean(),100*(b<=3).mean()))
print('  median pairwise angle   : %.1f -> %.1f deg'%(np.median(ma),np.median(mb)))
