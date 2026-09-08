import numpy as np, os, io, json, sys
from PIL import Image
import project as P
INC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"
census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; RW,RH=sl['renderWidth'],sl['renderHeight']; N=RW*RH
bundle=json.load(io.open(os.path.join(P.D,'capture_bundle.json'),encoding='utf-8'))
held=json.load(io.open(os.path.join(P.D,'model','held_out_frames.json'),encoding='utf-8')) if os.path.exists(os.path.join(P.D,'model','held_out_frames.json')) else None
print('census trainedPSNR %.3f  heldOutPSNR %.3f'%(sl['trainedPSNR'],sl['heldOutPSNR']))
print('held_out_frames.json:', str(held)[:200])
for f in (4,8,12):
    z=np.load('_fwd_%d.npz'%f); rc,rd,rT=z['c'],z['d'],z['T']
    fr=[x for x in bundle['frames'] if x['index']==f][0]
    gt=np.asarray(Image.open(os.path.join(INC,fr['imagePath'].replace('/',os.sep))).convert('RGB').resize((RW,RH),Image.BOX),dtype=np.float64)/255.0
    gt=gt.reshape(N,3)
    mse=((rc-gt)**2).mean(); a=1.0-rT
    # least-squares exposure fit
    X=np.stack([rc.ravel(),np.ones(rc.size)],1); sol,_,_,_=np.linalg.lstsq(X,gt.ravel(),rcond=None)
    mse2=((sol[0]*rc+sol[1]-gt)**2).mean()
    print('frame %2d qc %.4f  PSNR %.2f dB  alpha p10 %.6f p50 %.6f p90 %.6f  |  best gain %.3f bias %+.3f -> %.2f dB'
          %(f,fr['qc']['weight'],10*np.log10(1/mse),*np.percentile(a,[10,50,90]),sol[0],sol[1],10*np.log10(1/mse2)))
