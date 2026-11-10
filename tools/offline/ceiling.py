"""If the model were PERFECT but the view's pose were wrong by d pixels, what
PSNR could it score? Upper bound = PSNR(image, image shifted by d) on the real
frames of this capture."""
import json,io,os,numpy as np
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
names=[os.path.basename(f['imagePath']) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
imgs=[RF.ground_truth(n,rw,rh) for n in names]
print('render %dx%d, fx=%.2f px.  Median Gaussian-to-camera distance measured earlier: 1.22 m'%(rw,rh,fx))
print()
print(' shift px   cm at 1.22m    PSNR ceiling (mean over the 16 real frames)')
for d in [0,1,2,3,4,5,6,8,10]:
    ps=[]
    for im in imgs:
        a=im[20:-20,20:-20]; b=np.roll(im,d,axis=1)[20:-20,20:-20]
        ps.append(RF.psnr(a,b))
    cm=100*d*1.22/fx
    print('   %4d      %6.2f cm       %6.2f dB'%(d,cm,np.mean(ps)))
print()
print('census: heldOutPSNR 16.00 raw, 16.34 exposure-fitted; trainedPSNR 22.80')
print('prepass qcCard driftCentimeters = 10.47  ("surfaces seen twice will not line up")')
print('prepass poseGraph finalResidualMedianCentimeters = 2.78, converged = false')
for cm in [1.0,2.78,10.47]:
    print('   %5.2f cm of pose error = %.1f px at 1.22 m'%(cm,fx*cm/100/1.22))
