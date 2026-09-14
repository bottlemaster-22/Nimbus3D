import json, io, os, numpy as np, time
import project as P

D = P.D
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]
rw,rh=sl['renderWidth'],sl['renderHeight']
tx=(rw+15)//16; ty=(rh+15)//16
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses()

# --- reproduce keyframe selection (validated against held_out_frames.json) ---
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
def qinv_act(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v)
    return v+w*t+np.cross(qv,t)
frames=sorted(bundle['frames'],key=lambda f:f['index'])
pool=[f for f in frames if f['qc']['weight']>0.05] or frames
C={};F={}
for f in frames:
    q,t=poses[f['index']]
    C[f['index']]=-qinv_act(q,t); F[f['index']]=qinv_act(q,np.array([0.,0.,1.]))
pl=0.0; prev=None
for f in pool:
    c=C[f['index']]
    if prev is not None: pl+=np.linalg.norm(c-prev)
    prev=c
spacing=pl/120.0
chosen=[]; lc=None; lf=None
for f in pool:
    c=C[f['index']]; fw=F[f['index']]
    if lc is not None:
        if np.linalg.norm(lc-c)<spacing and 1-np.dot(fw/np.linalg.norm(fw),lf/np.linalg.norm(lf))<0.02: continue
    chosen.append(f['index']); lc=c; lf=fw
    if len(chosen)>=120: break
held=[chosen[i] for i in range(len(chosen)) if i%10==5]
trained=[chosen[i] for i in range(len(chosen)) if i%10!=5]
assert held==json.load(io.open(os.path.join(D,'model','held_out_frames.json'))), "keyframe repro mismatch"
print('keyframes %d  trained %d  held %d  (repro validated vs held_out_frames.json)'%(len(chosen),len(trained),len(held)))

allkf=chosen
vis=np.zeros((len(allkf),count),dtype=bool)
depth=np.zeros((len(allkf),count),dtype=np.float32)
t0=time.time()
for k,idx in enumerate(allkf):
    rot,t=poses[idx]
    R=P.quat_to_matrix(rot)
    ok,radius,tiles,ex,ey=P.project(col,count,R,t,fx,fy,cx,cy,tx,ty)
    vis[k]=ok
    mean=np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)
    depth[k]=(mean@R.T+t)[:,2]
    if k%20==0: print('  view %d/%d  visible %d  (%.1fs)'%(k,len(allkf),ok.sum(),time.time()-t0))
np.save('_vis.npy',vis); np.save('_kf.npy',np.array(allkf))
tr_mask=np.array([i%10!=5 for i in range(len(allkf))])
nv=vis[tr_mask].sum(0)
print()
print('=== PER-GAUSSIAN TRAINED-VIEW COUNT (frustum visibility, 108 trained views) ===')
print('mean %.2f  median %d  p10 %d  p25 %d  p75 %d  p90 %d  max %d'%(
    nv.mean(),np.median(nv),np.percentile(nv,10),np.percentile(nv,25),np.percentile(nv,75),np.percentile(nv,90),nv.max()))
for th in [0,1,2,3,5,10]:
    print('  seen in <=%2d trained views: %7d  (%5.2f%%)'%(th,(nv<=th).sum(),100*(nv<=th).mean()))
print('total (splat,view) incidences: %d'%nv.sum())
