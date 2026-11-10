"""What gain/bias does each view actually want, and is the trainer's
0.9..1.1 / +-0.05 clamp binding? The held-out score that drives EARLY STOPPING
is computed with that clamp (evaluateHeldOut, fitExposure: true)."""
import json,io,os,numpy as np
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh); poses=P.load_poses()
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
avail=[(f['index'],os.path.basename(f['imagePath'])) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
TRAINED={0,3,8,11,14}
print('frame  trn   free gain  free bias | clamped PSNR  free PSNR   cost of the clamp')
tr=[];un=[]
for idx,name in avail:
    img,_=RF.render(idx,poses,rw,rh,fx,fy,cx,cy)
    gt=RF.ground_truth(name,rw,rh)
    x=img.ravel();y=gt.ravel();n=len(x)
    den=n*np.dot(x,x)-x.sum()**2
    g=(n*np.dot(x,y)-x.sum()*y.sum())/den; b=(y.sum()-g*x.sum())/n
    gc=np.clip(g,0.9,1.1); bc=np.clip(b,-0.05,0.05)
    pf=RF.psnr(g*img+b,gt); pc=RF.psnr(gc*img+bc,gt)
    (tr if idx in TRAINED else un).append((pc,pf,g,b))
    print('%5d  %3s   %8.3f  %+9.3f | %11.2f  %9.2f   %+.2f dB %s'%(
      idx,'YES' if idx in TRAINED else 'no',g,b,pc,pf,pf-pc,
      'CLAMP BINDING' if (g<0.9 or g>1.1 or abs(b)>0.05) else ''))
print()
for nm,rs in [('TRAINED',tr),('NEVER-TRAINED',un)]:
    print('%-14s mean gain %.3f  mean bias %+.3f | clamped %.2f  free %.2f  clamp costs %.2f dB'%(
      nm,np.mean([r[2] for r in rs]),np.mean([r[3] for r in rs]),
      np.mean([r[0] for r in rs]),np.mean([r[1] for r in rs]),
      np.mean([r[1]-r[0] for r in rs])))
print()
print('gap, clamped fit (what the trainer reports): %.2f dB'%(np.mean([r[0] for r in tr])-np.mean([r[0] for r in un])))
print('gap, free fit                              : %.2f dB'%(np.mean([r[1] for r in tr])-np.mean([r[1] for r in un])))
print()
print('census: heldOutPSNR 16.000 raw -> 16.335 with the CLAMPED fit (+0.335 dB)')
