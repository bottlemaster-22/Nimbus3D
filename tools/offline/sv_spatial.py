import sv_load, numpy as np
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
c,n = sv_load.sv()
sent = np.load(SCR+r"\sv_sentinel.npy")
P = np.stack([c['x'],c['y'],c['z']],1).astype(np.float64)
L = np.stack([c['scale_0'],c['scale_1'],c['scale_2']],1).astype(np.float64)
a = 1/(1+np.exp(-np.clip(c['opacity'].astype(np.float64),-60,60)))
CTR = np.array([-1.26375, -0.24857, 0.39607])   # dome centre = scene origin
F = ~sent
p = P[F]-CTR; smax = np.exp(L[F]).max(1); r = np.linalg.norm(p,axis=1)
print('foreground n=%d' % F.sum())
print('radius from scene origin: ', ' '.join('p%s %.3f'%(q,np.percentile(r,q)) for q in [1,25,50,75,90,95,99,99.9]), ' max %.2f'%r.max())
print()
print('  shell           count     %%       med largest mm   med alpha')
edges=[0,1,2,3,4,5,6,8,10,15,20,40,100,1e9]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(r>=lo)&(r<hi)
    if m.sum()==0: continue
    print('  %6.0f-%-8.0f %8d %6.2f%%   %10.3f    %7.4f' % (lo,hi,m.sum(),100*m.mean(), 1000*np.median(smax[m]), np.median(a[F][m])))
print()
core = r<6
print('core (r<6m) n=%d = %.2f%% of foreground' % (core.sum(), 100*core.mean()))
pc = p[core]
print('core box  x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f' % (pc[:,0].min(),pc[:,0].max(),pc[:,1].min(),pc[:,1].max(),pc[:,2].min(),pc[:,2].max()))
# voxel occupancy at several resolutions, core only
print()
print(' voxel   nOccupied   splats/occupied vox   p50   p90   p99   max      fill%% of bbox')
for v in [0.5,0.25,0.1,0.05,0.02,0.01]:
    key=np.floor(pc/v).astype(np.int64)
    key=key-key.min(0)
    dims=key.max(0)+1
    flat=(key[:,0].astype(np.int64)*dims[1]+key[:,1])*dims[2]+key[:,2]
    u,cnt=np.unique(flat,return_counts=True)
    print(' %5.2fm %10d %14.2f %9.0f %5.0f %5.0f %5.0f %13.3f%%'
          % (v,u.size,cnt.mean(),np.percentile(cnt,50),np.percentile(cnt,90),np.percentile(cnt,99),cnt.max(),100*u.size/np.prod(dims.astype(float))))
