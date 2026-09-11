"""Independent Scaniverse shape/rank stats from the cached raw PLY columns."""
import numpy as np, os
S = r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
z = np.load(os.path.join(S,'sv.npz'))
print('sv.npz fields:', sorted(z.files))
n = z['x'].shape[0]; print('n =', n)
ls = np.stack([z['scale_0'],z['scale_1'],z['scale_2']],1).astype(np.float64)
print('raw scale_0 range %.4f .. %.4f  (log-space if negative)'%(ls.min(),ls.max()))
s = np.exp(np.clip(ls,-12,3)); ss=np.sort(s,1)
smin,smid,smax=ss[:,0],ss[:,1],ss[:,2]
lam=s*s; p=lam/np.maximum(lam.sum(1,keepdims=True),1e-20)
lp=np.log(np.maximum(p,1e-20)); H=-(p*lp).sum(1); rank=np.exp(H)
print('SCANIVERSE')
print('  largest mm  p10 %.3f p50 %.3f p90 %.3f  p90/p10 %.3f'%(*np.percentile(smax*1000,[10,50,90]),np.percentile(smax,90)/np.percentile(smax,10)))
print('  smallest mm p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smin*1000,[10,50,90])))
print('  aspect max/min p10 %.3f p50 %.3f p90 %.3f'%tuple(np.percentile(smax/np.maximum(smin,1e-12),[10,50,90])))
print('  rank p10 %.4f p50 %.4f p90 %.4f mean %.4f  frac<2 %.4f'%(*np.percentile(rank,[10,50,90]),rank.mean(),(rank<2).mean()))
g=(-4.0*0.001*(rank-2.0)*rank)[:,None]*p*(lp+H[:,None])
print('  our w=0.001 target=2 prior applied to THEIR splats: med|g| %.3e'%np.median(np.abs(g)))
# the size-vs-rank confound: is their high rank just small splats?
for lo,hi,lab in ((0,25,'smallest quartile by largest-axis'),(75,100,'largest quartile')):
    a,b_=np.percentile(smax,[lo,hi]); m=(smax>=a)&(smax<=b_)
    print('  %-34s rank p50 %.4f  largest-axis p50 %.3f mm'%(lab,np.median(rank[m]),np.median(smax[m])*1000))
