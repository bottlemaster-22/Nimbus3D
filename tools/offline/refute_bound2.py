import numpy as np, glob
from PIL import Image
D="C:/Users/Undea/Documents/LiKOVA/Scans/Incoming/scan_20260906_164840/images"
fs=sorted(glob.glob(D+"/*.jpg"))
A=np.stack([np.asarray(Image.open(f).convert("RGB").resize((720,540),Image.BILINEAR),dtype=np.float64)/255.0 for f in fs])
print("mean pixel value of this scan's real frames: %.4f"%A.mean())
def db(m): return 10*np.log10(1/max(m,1e-12))
print("\n== MSE cost of evaluating at identity when the trained fit was (g,b) ==")
print("   (err = (T-b)/g - T, finding 5's own model, evaluated on REAL pixels)")
cases=[("finding 5's clamp corner  ",1.10,0.05),
       ("finding 5's clamp corner  ",0.90,-0.05),
       ("MEASURED reachable in 28 updates (sigma-8 render)",1.00135,-0.00283),
       ("MEASURED reachable, render 20% dark (pathological)",1.00929,0.02368),
       ("HARD ceiling from the shader normalisation       ",1.0184,0.0448),
       ("HARD ceiling, opposite sign                      ",0.9816,-0.0448)]
for name,g,b in cases:
    R=(A-b)/g; m=float(((R-A)**2).mean())
    print("  %-50s g=%.5f b=%+.5f -> MSE %.7f"%(name,g,b,m))
print("\n== build 182 gap, and what each bound covers ==")
mt=10**(-20.654272/10); mh=10**(-13.69188/10); dm=mh-mt
print("  MSE(trained)=%.5f MSE(heldout)=%.5f  dMSE=%.5f  (NOTE: both are 10^(-mean_dB/10),")
print("   i.e. the GEOMETRIC mean of the 12 per-frame MSEs, not the arithmetic mean the bound is in)")
print("  MSE(trained)=%.5f MSE(heldout)=%.5f dMSE=%.5f"%(mt,mh,dm))
for name,g,b in cases:
    R=(A-b)/g; m=float(((R-A)**2).mean())
    print("  %-50s covers %6.2f%% of dMSE  -> %.2f dB recoverable"%(name,100*m/dm, db(max(mh-m,1e-9))-db(mh)))
