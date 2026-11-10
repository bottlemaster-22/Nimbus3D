"""Finite-difference the SSIM+L1 dL/dC_final chain the way lossbal_back builds it,
against the loss the kernels accumulate. Small crop so FD is affordable."""
import numpy as np, os, io, json
from PIL import Image
import project as P
RW,RH=180,135; N=RW*RH
LUMA=np.array([0.2126,0.7152,0.0722]); LAM=0.2; C1,C2=1e-4,9e-4
K11=np.array([0.00102838,0.00759876,0.03600077,0.10936069,0.21300554,0.26601172,
              0.21300554,0.10936069,0.03600077,0.00759876,0.00102838])
rng=np.random.default_rng(1)
gt=rng.random((N,3)); ren=np.clip(gt+0.25*rng.standard_normal((N,3)),0,1)
qcw=0.8429; INV_N=1.0/N
def blur(pl):
    p=pl.reshape(-1,RH,RW)
    pad=np.pad(p,((0,0),(0,0),(5,5)),mode='edge'); o=np.zeros_like(p)
    for i,k in enumerate(K11): o+=k*pad[:,:,i:i+RW]
    pad=np.pad(o,((0,0),(5,5),(0,0)),mode='edge'); o2=np.zeros_like(p)
    for i,k in enumerate(K11): o2+=k*pad[:,i:i+RH,:]
    return o2.reshape(pl.shape)
def total_loss(r):
    l1=(np.abs(r-gt).sum(1)/3.0).mean()*qcw*(1-LAM)
    X=(r*LUMA).sum(1); Y=(gt*LUMA).sum(1)
    bl=blur(np.stack([X,Y,X*X,Y*Y,X*Y],0))
    mux,muy=bl[0],bl[1]
    sxx=np.maximum(bl[2]-mux*mux,0); syy=np.maximum(bl[3]-muy*muy,0); sxy=bl[4]-mux*muy
    n1=2*mux*muy+C1; n2=2*sxy+C2; d1=mux*mux+muy*muy+C1; d2=sxx+syy+C2
    ssim=n1*n2/np.maximum(d1*d2,1e-12)
    return l1 + qcw*LAM*(1-ssim.mean())
def analytic(r):
    gL1=qcw*(1-LAM)*INV_N*np.sign(r-gt)/3.0
    X=(r*LUMA).sum(1); Y=(gt*LUMA).sum(1)
    bl=blur(np.stack([X,Y,X*X,Y*Y,X*Y],0)); mux,muy=bl[0],bl[1]
    sxx=np.maximum(bl[2]-mux*mux,0); syy=np.maximum(bl[3]-muy*muy,0); sxy=bl[4]-mux*muy
    n1=2*mux*muy+C1; n2=2*sxy+C2; d1=mux*mux+muy*muy+C1; d2=sxx+syy+C2
    invD=1.0/np.maximum(d1*d2,1e-12)
    wss=qcw*LAM*INV_N
    dmux=(2*muy*n2*d1-2*mux*n1*n2)/np.maximum(d1*d1*d2,1e-12)
    dsxy=2*n1*invD; dsxx=-n1*n2/np.maximum(d1*d2*d2,1e-12)
    Cc=-wss*dmux; A=-wss*dsxx; B=-wss*dsxy
    bp=blur(np.stack([Cc-2*A*mux-B*muy,A,B],0))
    dLdX=bp[0]+2*X*bp[1]+Y*bp[2]
    return gL1 + dLdX[:,None]*LUMA[None,:]
g=analytic(ren)
idxs=rng.choice(N,60,replace=False); h=1e-6; errs=[]
for i in idxs:
    for c in range(3):
        rp=ren.copy(); rp[i,c]+=h; rm=ren.copy(); rm[i,c]-=h
        num=(total_loss(rp)-total_loss(rm))/(2*h)
        errs.append(abs(g[i,c]-num)/max(abs(num),1e-18))
errs=np.array(errs)
print('SSIM+L1 dL/dC_final vs finite difference of the accumulated loss')
print('  components %d   rel err  p50 %.3e  p90 %.3e  max %.3e'%(errs.size,*np.percentile(errs,[50,90]),errs.max()))
