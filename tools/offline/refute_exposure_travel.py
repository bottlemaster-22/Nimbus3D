"""How far can the learned per-frame exposure actually travel in one run?

Reimplements the EXACT update the trainer runs:
  TrainerShaders.metal trainer_loss_photometric  (L1 part, w = fw*(1-lambda)/N)
  TrainerShaders.metal trainer_ssim_stats/backward (SSIM part, w = fw*lambda/N)
  TrainerShaders.metal trainer_loss_finalize     (exposureGrad[0]=sum dot(pre,g),
                                                  exposureGrad[1]=sum (gx+gy+gz))
  MetalSplatTrainer.swift 2011-2019               (g -= lr*grad, clamp)
lr=0.002, lambdaSSIM=0.2, frameWeight=1, clamps 0.9..1.1 and -0.05..0.05.
28 updates per frame = 3000 iterations / 108 trained keyframes (build 182).
"""
import numpy as np, glob
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

LUMA=np.array([0.2126,0.7152,0.0722],dtype=np.float64)
LR=0.002; LAM=0.2; FW=1.0
C1=0.01**2; C2=0.03**2
# 11-tap sigma 1.5 separable blur, clamp to edge, matching trainer_blur_h/v
def blur(x):
    return gaussian_filter(x, sigma=1.5, mode='nearest', truncate=(5.0/1.5))

def ssim_grad_dLdX(X,Y,n):
    mux=blur(X); muy=blur(Y)
    sxx=np.maximum(blur(X*X)-mux*mux,0); syy=np.maximum(blur(Y*Y)-muy*muy,0)
    sxy=blur(X*Y)-mux*muy
    n1=2*mux*muy+C1; n2=2*sxy+C2; d1=mux*mux+muy*muy+C1; d2=sxx+syy+C2
    invD=1.0/np.maximum(d1*d2,1e-12)
    w=FW*LAM/n
    dS_dmux=(2*muy*n2*d1-2*mux*n1*n2)/np.maximum(d1*d1*d2,1e-12)
    dS_dsxy=2*n1*invD
    dS_dsxx=-n1*n2/np.maximum(d1*d2*d2,1e-12)
    Cc=-w*dS_dmux; A=-w*dS_dsxx; B=-w*dS_dsxy
    P0=blur(Cc-2*A*mux-B*muy); PA=blur(A); PB=blur(B)
    return P0+2*X*PA+Y*PB

def run(T,R,steps=28,use_ssim=True,label=""):
    H,W,_=T.shape; n=H*W
    g,b=1.0,0.0
    traj=[]
    for it in range(steps):
        rendered=g*R+b
        diff=rendered-T
        w=FW*(1-LAM)/n
        gpx=w*np.sign(diff)/3.0
        if use_ssim:
            X=(rendered*LUMA).sum(-1); Y=(T*LUMA).sum(-1)
            dLdX=ssim_grad_dLdX(X,Y,n)
            gpx=gpx+dLdX[...,None]*LUMA[None,None,:]
        grad_gain=float((R*gpx).sum())
        grad_bias=float(gpx.sum())
        g=min(max(g-LR*grad_gain,0.9),1.1)
        b=min(max(b-LR*grad_bias,-0.05),0.05)
        traj.append((g,b))
    return g,b,traj

D="C:/Users/Undea/Documents/LiKOVA/Scans/Incoming/scan_20260906_164840/images"
fs=sorted(glob.glob(D+"/*.jpg"))
imgs=[np.asarray(Image.open(f).convert("RGB").resize((720,540),Image.BILINEAR),dtype=np.float64)/255.0 for f in fs]
print("frames used:",len(imgs),"render grid 720x540 (the build-182 grid)")
print()
for sig,tag in [(8,"render = truth blurred sigma 8  (~19-20 dB, build-182-like)"),
                (4,"render = truth blurred sigma 4  (~23 dB)")]:
    print("---",tag)
    for use_ssim in (True,False):
        gs=[];bs=[]
        for T in imgs:
            R=gaussian_filter(T,(sig,sig,0),mode='nearest')
            g,b,_=run(T,R,28,use_ssim)
            gs.append(g);bs.append(b)
        print("   L1+SSIM" if use_ssim else "   L1 only",
              ": after 28 updates gain %.5f..%.5f (mean %.5f), bias %+.5f..%+.5f (mean %+.5f)"
              %(min(gs),max(gs),float(np.mean(gs)),min(bs),max(bs),float(np.mean(bs))))
        print("     -> %.1f%% of the gain clamp, %.1f%% of the bias clamp"
              %(100*max(abs(np.array(gs)-1))/0.1, 100*max(abs(np.array(bs)))/0.05))
print()
print("--- worst case: render is systematically DARK by 20%%, exposure has real work to do")
gs=[];bs=[]
for T in imgs:
    R=gaussian_filter(T,(8,8,0),mode='nearest')*0.8
    g,b,_=run(T,R,28,True); gs.append(g);bs.append(b)
print("   gain %.5f..%.5f mean %.5f ; bias %+.5f..%+.5f mean %+.5f"%(min(gs),max(gs),np.mean(gs),min(bs),max(bs),np.mean(bs)))
print()
print("--- how many updates would it take to REACH the clamp at this gradient scale?")
T=imgs[0]; R=gaussian_filter(T,(8,8,0),mode='nearest')
g,b,traj=run(T,R,28,True)
d=[abs(traj[i][0]-(1.0 if i==0 else traj[i-1][0])) for i in range(len(traj))]
print("   mean |dgain| per update %.6f -> %.0f updates to move 0.10"%(np.mean(d),0.10/max(np.mean(d),1e-12)))
db=[abs(traj[i][1]-(0.0 if i==0 else traj[i-1][1])) for i in range(len(traj))]
print("   mean |dbias| per update %.6f -> %.0f updates to move 0.05"%(np.mean(db),0.05/max(np.mean(db),1e-12)))
