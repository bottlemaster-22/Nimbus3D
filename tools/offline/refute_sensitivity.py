import numpy as np, glob
from PIL import Image
D="C:/Users/Undea/Documents/LiKOVA/Scans/Incoming/scan_20260906_164840/images"
fs=sorted(glob.glob(D+"/*.jpg"))
A=np.stack([np.asarray(Image.open(f).convert("RGB").resize((720,540),Image.BILINEAR),dtype=np.float32)/255.0 for f in fs])
def psnr(a,b):
    m=float(((a-b)**2).mean()); return 10*np.log10(1/max(m,1e-12)), m
print("== dB cost of a pure translational misalignment, measured on 16 real frames of this scan ==")
print("(720x540 render grid; the pose delta withheld from held-out frames is unclamped in total)")
for k in [0,1,2,3,4,6,8]:
    ds=[]
    for im in A:
        if k==0: ds.append(psnr(im,im)[0]); continue
        a=im[:, k:, :]; b=im[:, :-k, :]
        ds.append(psnr(a,b)[0])
    print("  shift %d px : mean PSNR against self %.2f dB"%(k,float(np.mean(ds))))

print("\n== how much a CLAMPED affine (g in .9-1.1, b in -.05..+.05) recovers ==")
print("   case A: residual is geometric (render = blurred/degraded truth), i.e. what a 19.6mm-splat model gives")
rng=np.random.default_rng(0)
def clamped_fit(x,y):
    g=float(np.cov(x.ravel(),y.ravel(),bias=True)[0,1]/max(np.var(x),1e-12)); b=float(y.mean()-g*x.mean())
    return min(max(g,0.9),1.1), min(max(b,-0.05),0.05)
from scipy.ndimage import gaussian_filter
for sig in [2,4,8,16]:
    gains=[]; base=[]
    for im in A:
        R=gaussian_filter(im,(sig,sig,0))
        d0,m0=psnr(R,im)
        g,b=clamped_fit(R,im)
        d1,m1=psnr(g*R+b,im)
        base.append(d0); gains.append(d1-d0)
    print("   blur sigma %2d -> render PSNR %.2f dB, clamped affine recovers %+.3f dB"%(sig,float(np.mean(base)),float(np.mean(gains))))
print("   case B: residual is genuinely photometric (render = truth scaled by a real exposure)")
for g0,b0 in [(1.10,0.05),(0.95,-0.02)]:
    gains=[]; base=[]
    for im in A:
        R=(im-b0)/g0
        d0,_=psnr(R,im); g,b=clamped_fit(R,im); d1,_=psnr(g*R+b,im)
        base.append(d0); gains.append(d1-d0)
    print("   true (g=%.2f b=%+.2f) -> identity PSNR %.2f dB, clamped affine recovers %+.2f dB"%(g0,b0,float(np.mean(base)),float(np.mean(gains))))
print("   case C: BOTH - blurred render AND a real exposure offset (the actual held-out situation)")
for sig in [4,8]:
    for g0,b0 in [(1.05,0.02),(1.10,0.05)]:
        gains=[]; base=[]
        for im in A:
            R=(gaussian_filter(im,(sig,sig,0))-b0)/g0
            d0,_=psnr(R,im); g,b=clamped_fit(R,im); d1,_=psnr(g*R+b,im)
            base.append(d0); gains.append(d1-d0)
        print("   sigma %2d, true (g=%.2f b=%+.2f) -> %.2f dB, clamped affine recovers %+.2f dB"%(sig,g0,b0,float(np.mean(base)),float(np.mean(gains))))
