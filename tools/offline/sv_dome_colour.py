import sv_load, numpy as np
from scipy.spatial import cKDTree
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
c,_=sv_load.sv(); sent=np.load(SCR+r"\sv_sentinel.npy")
P=np.stack([c['x'],c['y'],c['z']],1).astype(np.float64)[sent]
dc=np.stack([c['f_dc_0'],c['f_dc_1'],c['f_dc_2']],1).astype(np.float64)[sent]
C0=0.28209479177387814
rgb=np.clip(0.5+C0*dc,0,1)
ctr=np.array([-1.26375,-0.24857,0.39607]); u=(P-ctr); u/=np.linalg.norm(u,axis=1,keepdims=True)
print('dome DC rgb: mean %s  std %s' % (np.round(rgb.mean(0),4), np.round(rgb.std(0),4)))
print('unique dc rows %d of %d' % (np.unique(dc,axis=0).shape[0], len(dc)))
t=cKDTree(u); d,i=t.query(u,k=7)
nb=rgb[i[:,1:]]                      # 6 neighbours
local=np.abs(nb-rgb[:,None,:]).mean(axis=(1,2))
# shuffled control
perm=np.random.default_rng(0).permutation(len(rgb))
shuf=np.abs(rgb[perm][i[:,1:]]-rgb[perm][:,None,:]).mean(axis=(1,2))
print('mean |neighbour - self| rgb: real %.5f   shuffled control %.5f   ratio %.3f'%(local.mean(), shuf.mean(), local.mean()/shuf.mean()))
print('  -> a value well below 1.0 means the dome colour is spatially SMOOTH i.e. trained, not noise')
# how much of the sphere is "sky-like" vs dark
print('luminance: p5 %.3f p50 %.3f p95 %.3f ; frac near mid-grey (0.48-0.52): %.2f%%'
      % (*np.percentile(rgb.mean(1),[5,50,95]), 100*np.mean(np.abs(rgb.mean(1)-0.5)<0.02)))
# upper vs lower hemisphere along each axis: which axis is "up"?
for ax,nm in enumerate('xyz'):
    hi=u[:,ax]>0.5; lo=u[:,ax]<-0.5
    print('  axis %s : +hemi lum %.3f  -hemi lum %.3f  delta %.3f' % (nm, rgb[hi].mean(), rgb[lo].mean(), rgb[hi].mean()-rgb[lo].mean()))
