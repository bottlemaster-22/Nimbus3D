"""Local registration error between the render and the photograph.
A single global shift is the crudest model of pose error; a real 6-DoF pose
error produces a depth-dependent flow field. This block-matches 48x48 blocks
over +/-10 px and reports the residual displacement, which is the geometric
registration error of the model against that view."""
import json,io,os,numpy as np
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
poses=P.load_poses()
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
avail=[(f['index'],os.path.basename(f['imagePath'])) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
TRAINED={0,3,8,11,14}
B=48; Rr=10
print('block %dpx, search +/-%dpx, blocks kept only where the photo has real texture'%(B,Rr))
print()
print('frame  trn   blocks   median|d| px    p75    p90   |  median |d| in cm at 1.22 m')
res=[]
for idx,name in avail:
    img,_=RF.render(idx,poses,rw,rh,fx,fy,cx,cy)
    gt=RF.ground_truth(name,rw,rh)
    a=img.mean(2); b=gt.mean(2)
    a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9)
    ds=[]
    for y in range(Rr,rh-B-Rr,B):
        for x in range(Rr,rw-B-Rr,B):
            tgt=b[y:y+B,x:x+B]
            if tgt.std()<0.35: continue          # skip flat blocks: no signal
            best=-1e18;bd=(0,0)
            for dy in range(-Rr,Rr+1):
                for dx in range(-Rr,Rr+1):
                    v=a[y+dy:y+dy+B,x+dx:x+dx+B]
                    s=float((v*tgt).sum())
                    if s>best: best,bd=s,(dx,dy)
            ds.append(np.hypot(*bd))
    ds=np.array(ds)
    if len(ds)==0: continue
    res.append((idx,idx in TRAINED,len(ds),np.median(ds),np.percentile(ds,75),np.percentile(ds,90)))
    print('%5d  %3s  %6d   %8.2f   %6.2f %6.2f  |  %.2f cm'%(
      idx,'YES' if idx in TRAINED else 'no',len(ds),np.median(ds),np.percentile(ds,75),np.percentile(ds,90),
      100*np.median(ds)*1.22/fx))
tr=[r for r in res if r[1]]; un=[r for r in res if not r[1]]
print()
print('=== LOCAL REGISTRATION ERROR OF THE MODEL AGAINST EACH VIEW ===')
print('TRAINED (%d frames)      : median |d| = %.2f px  = %.2f cm at 1.22 m'%(
  len(tr),np.mean([r[3] for r in tr]),100*np.mean([r[3] for r in tr])*1.22/fx))
print('NEVER-TRAINED (%d frames): median |d| = %.2f px  = %.2f cm at 1.22 m'%(
  len(un),np.mean([r[3] for r in un]),100*np.mean([r[3] for r in un])*1.22/fx))
print()
print('prepass poseGraph finalResidualMedianCentimeters = 2.78 (converged: false)')
print('prepass qcCard driftCentimeters = 10.47')
print('model Gaussian largest axis at 1.22 m = 3.88 px (1 sigma)')
