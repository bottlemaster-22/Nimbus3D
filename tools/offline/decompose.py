import json,io,os,numpy as np,time
from scipy.ndimage import shift as ndshift, gaussian_filter
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
poses=P.load_poses()
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
avail=[(f['index'],os.path.basename(f['imagePath']),f['qc']) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
TRAINED={0,3,8,11,14}

def fit_free(v,t):
    """UNclamped least-squares gain/bias, so the trainer's 0.9..1.1 clamp is not a confound."""
    x=v.ravel(); y=t.ravel(); n=len(x)
    den=n*np.dot(x,x)-x.sum()**2
    g=(n*np.dot(x,y)-x.sum()*y.sum())/den; b=(y.sum()-g*x.sum())/n
    return g,b
def best(v,t):
    g,b=fit_free(v,t); return RF.psnr(g*v+b,t)

print('Each frame: render, then (1) free exposure fit, (2) + best whole-image shift,')
print('(3) + best Gaussian blur of the render. Shift search +/-14 px, step 1.')
print()
print('frame  trn  blurPx   base   +shift  dx    dy   +blur  sigma | shift gain  blur gain')
rows=[]
for idx,name,qc in avail:
    img,Tf=RF.render(idx,poses,rw,rh,fx,fy,cx,cy)
    gt=RF.ground_truth(name,rw,rh)
    p0=best(img,gt)
    # integer shift search on a 3-channel image
    bestp=-1;bdx=0;bdy=0
    for dy in range(-14,15):
        for dx in range(-14,15):
            v=np.roll(np.roll(img,dy,axis=0),dx,axis=1)[20:-20,20:-20]
            t=gt[20:-20,20:-20]
            p=best(v,t)
            if p>bestp: bestp,bdx,bdy=p,dx,dy
    # crop-matched baseline so shift gain is measured on the same pixels
    p0c=best(img[20:-20,20:-20],gt[20:-20,20:-20])
    vs=np.roll(np.roll(img,bdy,axis=0),bdx,axis=1)[20:-20,20:-20]
    t=gt[20:-20,20:-20]
    bb=-1;bsig=0
    for sig in [0,0.5,1.0,1.5,2.0,3.0]:
        v=gaussian_filter(vs,(sig,sig,0)) if sig>0 else vs
        p=best(v,t)
        if p>bb: bb,bsig=p,sig
    rows.append((idx,idx in TRAINED,qc['motionBlurPixels'],p0c,bestp,bdx,bdy,bb,bsig))
    print('%5d  %3s  %6.2f  %6.2f  %6.2f %4d %4d  %6.2f  %4.1f | %+5.2f  %+5.2f'%(
      idx,'YES' if idx in TRAINED else 'no',qc['motionBlurPixels'],p0c,bestp,bdx,bdy,bb,bsig,bestp-p0c,bb-bestp))
np.save('_decomp.npy',np.array(rows,dtype=float))
tr=[r for r in rows if r[1]]; un=[r for r in rows if not r[1]]
def m(rs,i): return np.mean([r[i] for r in rs])
print()
print('=== DECOMPOSITION (means) ===')
print('                    TRAINED(5)   NEVER-TRAINED(11)   gap')
print('exposure-fitted     %7.2f      %7.2f        %5.2f dB'%(m(tr,3),m(un,3),m(tr,3)-m(un,3)))
print('+ best 2D shift     %7.2f      %7.2f        %5.2f dB'%(m(tr,4),m(un,4),m(tr,4)-m(un,4)))
print('+ best blur         %7.2f      %7.2f        %5.2f dB'%(m(tr,7),m(un,7),m(tr,7)-m(un,7)))
print()
print('shift magnitude |d| : trained %.2f px   never-trained %.2f px'%(
  np.mean([np.hypot(r[5],r[6]) for r in tr]),np.mean([np.hypot(r[5],r[6]) for r in un])))
print('dB recovered by shift: trained %+.2f   never-trained %+.2f'%(m(tr,4)-m(tr,3),m(un,4)-m(un,3)))
print('dB recovered by blur : trained %+.2f   never-trained %+.2f'%(m(tr,7)-m(tr,4),m(un,7)-m(un,4)))
print('best blur sigma      : trained %.2f px  never-trained %.2f px'%(m(tr,8),m(un,8)))
