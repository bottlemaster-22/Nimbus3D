"""REFUTATION TEST 3: is the 16.8 dB baseline a MEASURE OF THE MODEL, or is it
already saturated by an error the prune cannot touch?

If the full-model render is 16.5 dB against the photo while the trainer's own
trainedPSNR is 22.80, then most of the residual comes from something other
than which splats exist, and a 0.33 dB delta on top of it is being read out of
a number dominated by that something.  Decomposes the residual by alpha
coverage and by a global gain/offset (exposure) fit.
"""
import io, json, os
import numpy as np
import project as P
import detail as Dt
import render as Rn

def psnr(a,b,m=None):
    d=(a-b)**2
    if m is not None: d=d[m]
    return 10*np.log10(1.0/max(d.mean(),1e-12))

census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses(); imgs=dict(Dt.frames_with_images())
print('render size %dx%d ; trainedPSNR %.2f ; heldOutPSNR %.2f ; heldOutExpFit %.2f'
      % (rw,rh,sl['trainedPSNR'],sl['heldOutPSNR'],sl['heldOutPSNRExposureFitted']))

rows=[]
for f in sorted(imgs)[:6]:
    _,rgb=Dt.luma_of(imgs[f],rw,rh)
    rot,t=poses[f]; R=P.quat_to_matrix(rot)
    img,al=Rn.render(col,count,R,t,fx,fy,cx,cy,rw,rh)
    covered=al>=0.90
    empty=al<0.10
    mse_all=((img-rgb)**2).mean()
    mse_cov=((img-rgb)**2)[covered].mean() if covered.any() else np.nan
    mse_emp=((img-rgb)**2)[empty].mean() if empty.any() else np.nan
    # share of total squared error contributed by the uncovered pixels
    tot=((img-rgb)**2).sum()
    share_emp=((img-rgb)**2)[empty].sum()/tot if empty.any() else 0.0
    # best global gain+offset per channel (an exposure fit the render lacks)
    fit=np.empty_like(img)
    for ch in range(3):
        x=img[:,:,ch].ravel(); y=rgb[:,:,ch].ravel()
        A=np.stack([x,np.ones_like(x)],1)
        g,o=np.linalg.lstsq(A,y,rcond=None)[0]
        fit[:,:,ch]=(g*img[:,:,ch]+o)
    rows.append((f, al.mean(), covered.mean(), empty.mean(),
                 10*np.log10(1/max(mse_all,1e-12)),
                 10*np.log10(1/max(mse_cov,1e-12)),
                 10*np.log10(1/max(mse_emp,1e-12)),
                 100*share_emp, psnr(np.clip(fit,0,1),rgb)))
    print('  frame %d done' % f)

print('\n%5s %7s %8s %8s %9s %11s %11s %10s %11s'
      % ('frame','meanA','cov>.9%','empty<.1%','PSNR all','PSNR cov','PSNR empty','SSE from','PSNR after'))
print('%5s %7s %8s %8s %9s %11s %11s %10s %11s'
      % ('','','','','(dB)','(dB)','(dB)','empty %','gain+off'))
for r in rows:
    print('%5d %7.3f %8.1f %8.1f %9.3f %11.3f %11.3f %10.1f %11.3f'
          % (r[0],r[1],100*r[2],100*r[3],r[4],r[5],r[6],r[7],r[8]))
a=np.array([[r[4],r[5],r[6],r[7],r[8]] for r in rows])
print('%5s %7s %8s %8s %9.3f %11.3f %11.3f %10.1f %11.3f'
      % ('MEAN','','','',a[:,0].mean(),a[:,1].mean(),a[:,2].mean(),a[:,3].mean(),a[:,4].mean()))
