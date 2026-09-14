import json,io,os,numpy as np,time
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
poses=P.load_poses()
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
avail=[(f['index'],os.path.basename(f['imagePath']),f['qc']) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
TRAINED={0,3,8,11,14}   # from the validated keyframe reproduction
print('render %dx%d  fx=%.2f  census trainedPSNR=%.2f heldOut=%.2f (fitted %.2f)'%(
  rw,rh,fx,sl['trainedPSNR'],sl['heldOutPSNR'],sl['heldOutPSNRExposureFitted']))
print()
print('frame  trained  qcW   blurPx  PSNRraw  PSNRfit   gain   bias   meanT')
rows=[]
for idx,name,qc in avail:
    t0=time.time()
    img,Tf=RF.render(idx,poses,rw,rh,fx,fy,cx,cy)
    gt=RF.ground_truth(name,rw,rh)
    g,b=RF.fit_exposure(img,gt)
    pr=RF.psnr(img,gt); pf=RF.psnr(np.clip(g*img+b,0,None),gt)
    rows.append((idx,idx in TRAINED,qc['weight'],qc['motionBlurPixels'],pr,pf,g,b,Tf.mean()))
    print('%5d  %7s  %.2f  %6.2f  %7.2f  %7.2f  %5.3f %6.3f  %.3f'%(
      idx,'YES' if idx in TRAINED else 'no',qc['weight'],qc['motionBlurPixels'],pr,pf,g,b,Tf.mean()),
      ' (%.0fs)'%(time.time()-t0))
np.save('_rows.npy',np.array([(r[0],r[1],r[2],r[3],r[4],r[5],r[8]) for r in rows]))
tr=[r for r in rows if r[1]]; un=[r for r in rows if not r[1]]
print()
print('=== TRAINED (%d frames) vs NEVER-TRAINED (%d frames), same 8 seconds of walk ==='%(len(tr),len(un)))
print('trained      PSNR raw %.2f  fitted %.2f'%(np.mean([r[4] for r in tr]),np.mean([r[5] for r in tr])))
print('never-trained PSNR raw %.2f  fitted %.2f'%(np.mean([r[4] for r in un]),np.mean([r[5] for r in un])))
print('GAP: raw %.2f dB   exposure-fitted %.2f dB'%(
  np.mean([r[4] for r in tr])-np.mean([r[4] for r in un]),
  np.mean([r[5] for r in tr])-np.mean([r[5] for r in un])))
