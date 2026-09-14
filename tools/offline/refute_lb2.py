"""Independent shape stats on OUR model + target sweep of the disc prior."""
import numpy as np, os
import project as P
col,n = P.load_ply(os.path.join(P.D,'model','model.ply'))
ls = np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3)
s = np.exp(ls); ss = np.sort(s,1)
smin,smid,smax = ss[:,0],ss[:,1],ss[:,2]
lam=s*s; p=lam/np.maximum(lam.sum(1,keepdims=True),1e-20)
lp=np.log(np.maximum(p,1e-20)); H=-(p*lp).sum(1); rank=np.exp(H)
print('OURS n=%d'%n)
print('  largest mm  p10 %.3f p50 %.3f p90 %.3f   p90/p10 %.3f'%(*np.percentile(smax*1000,[10,50,90]),np.percentile(smax,90)/np.percentile(smax,10)))
print('  middle  mm  p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smid*1000,[10,50,90])))
print('  smallest mm p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smin*1000,[10,50,90])))
print('  aspect max/min p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smax/np.maximum(smin,1e-12),[10,50,90])))
print('  aspect max/mid p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smax/np.maximum(smid,1e-12),[10,50,90])))
print('  rank p10 %.4f p50 %.4f p90 %.4f mean %.4f  frac>2 %.4f'%(*np.percentile(rank,[10,50,90]),rank.mean(),(rank>2).mean()))

order=np.argsort(s,axis=1)
for tgt in (2.0,2.2,2.6,3.0):
    res=rank-tgt
    g=(-2.0*0.001*res*rank)[:,None]*p*(lp+H[:,None])   # CORRECTED factor -2
    g4=(-4.0*0.001*res*rank)[:,None]*p*(lp+H[:,None])  # as shipped
    gs=np.take_along_axis(g,order,axis=1)
    print('  target %.2f: med|g| corrected %.3e  as-shipped %.3e  med residual %+.4f  smallest-axis pushed UP frac %.4f'
          %(tgt,np.median(np.abs(g)),np.median(np.abs(g4)),np.median(res),(gs[:,0]<0).mean()))
