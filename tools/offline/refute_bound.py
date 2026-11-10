import numpy as np, glob, os
from PIL import Image
D="C:/Users/Undea/Documents/LiKOVA/Scans/Incoming/scan_20260906_164840/images"
fs=sorted(glob.glob(D+"/*.jpg"))
print("images",len(fs))
imgs=[]
for f in fs:
    im=Image.open(f).convert("RGB").resize((720,540), Image.BILINEAR)
    imgs.append(np.asarray(im,dtype=np.float32)/255.0)
A=np.stack(imgs)   # N,H,W,3
print("stack",A.shape,"mean %.4f"%A.mean(),"p10 %.4f p50 %.4f p90 %.4f"%tuple(np.percentile(A,[10,50,90])))
print("frac of pixels in [0.15,0.6] (finding5's assumed band): %.3f"%(((A>=0.15)&(A<=0.6)).mean()))

# Finding 5's model: trained frame fitted (g,b) so that g*R+b = T  ->  R=(T-b)/g.
# Identity eval compares R to T. err = (T-b)/g - T.
print("\n-- MSE induced by evaluating at identity when the true fit was (g,b), on REAL pixels --")
for g,b in [(1.10,0.05),(0.90,-0.05),(1.10,-0.05),(0.90,0.05),(1.05,0.025),(1.02,0.01)]:
    R=(A-b)/g
    mse=float(((R-A)**2).mean())
    print("  g=%.2f b=%+.3f  MSE=%.6f  (=%.2f dB of headroom)"%(g,b,mse,10*np.log10(1/max(mse,1e-12))))
print("\n-- finding 5's own analytic band, recomputed on its assumed uniform truth in [0.15,0.6] --")
T=np.linspace(0.15,0.6,10001)
for g,b in [(1.10,0.05),(0.90,-0.05)]:
    R=(T-b)/g; print("  g=%.2f b=%+.3f  MSE=%.6f"%(g,b,float(((R-T)**2).mean())))

# What an UNCONSTRAINED affine fit can remove, given an actual residual:
# upper bound on recoverable MSE from affine (g,b) is  mean(e)^2 + cov(R,e)^2/var(R).
# Without a render we can at least measure the affine structure BETWEEN real frames.
print("\n-- best clamped affine fit between temporally adjacent real frames --")
lo_g,hi_g,lo_b,hi_b=0.9,1.1,-0.05,0.05
for i in range(len(A)-1):
    x=A[i].ravel(); y=A[i+1].ravel()
    m0=float(((x-y)**2).mean())
    # least squares y ~= g*x+b
    gg=float(np.cov(x,y,bias=True)[0,1]/max(np.var(x),1e-12)); bb=float(y.mean()-gg*x.mean())
    gc=min(max(gg,lo_g),hi_g); bc=min(max(bb,lo_b),hi_b)
    m1=float(((gc*x+bc-y)**2).mean())
    print("  %02d->%02d raw MSE %.5f (%.2f dB)  fitted g=%.4f b=%+.4f -> clamped g=%.4f b=%+.4f  MSE %.5f (%.2f dB)  gain %.2f dB"%(
        i,i+1,m0,10*np.log10(1/max(m0,1e-12)),gg,bb,gc,bc,m1,10*np.log10(1/max(m1,1e-12)),
        10*np.log10(1/max(m1,1e-12))-10*np.log10(1/max(m0,1e-12))))
