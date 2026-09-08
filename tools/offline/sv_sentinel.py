import sv_load, numpy as np
c,n = sv_load.sv()
L = np.stack([c['scale_0'],c['scale_1'],c['scale_2']],1).astype(np.float64)
P = np.stack([c['x'],c['y'],c['z']],1).astype(np.float64)
op= c['opacity'].astype(np.float64)
a = 1/(1+np.exp(-np.clip(op,-60,60)))
sent = (L==2.0).all(1)
print('sentinel splats: %d  (%.4f%% of %d)' % (sent.sum(), 100*sent.mean(), n))
print('n = %d ; 427710-%d = %d' % (sent.sum(), sent.sum(), n-sent.sum()))
print()
print('--- sentinel population ---')
print(' alpha: min %.5f p50 %.5f max %.5f ; frac alpha<0.02: %.2f%%' % (a[sent].min(), np.median(a[sent]), a[sent].max(), 100*np.mean(a[sent]<0.02)))
print(' unique alpha values in sentinels: %d  ->' % np.unique(a[sent]).size, np.unique(a[sent])[:8])
print(' positions: x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f' % (P[sent,0].min(),P[sent,0].max(),P[sent,1].min(),P[sent,1].max(),P[sent,2].min(),P[sent,2].max()))
print(' unique positions among sentinels: %d' % np.unique(P[sent],axis=0).shape[0])
q = np.stack([c['rot_0'],c['rot_1'],c['rot_2'],c['rot_3']],1)[sent]
print(' unique rotations among sentinels: %d ; first rows:' % np.unique(q,axis=0).shape[0]); print(q[:3])
dc = np.stack([c['f_dc_0'],c['f_dc_1'],c['f_dc_2']],1)[sent]
print(' unique f_dc among sentinels: %d ; first rows:' % np.unique(dc,axis=0).shape[0]); print(dc[:3])
rest = np.stack([c['f_rest_%d'%i] for i in range(45)],1)[sent]
print(' sentinel SH-rest all zero? %s  (max abs %.4f)' % (np.all(rest==0), np.abs(rest).max()))
idx = np.flatnonzero(sent)
print(' index range %d..%d ; are they contiguous at the end? last %d indices all sentinel: %s'
      % (idx.min(), idx.max(), sent.sum(), np.all(sent[n-sent.sum():])))
print(' first 20 sentinel indices:', idx[:20])
print(' gaps between consecutive sentinel indices: unique %s' % np.unique(np.diff(idx))[:10])
print()
print('--- real population (sentinels removed) ---')
r = ~sent
S = np.sort(np.exp(L[r]),1)
for nm,v in [('largest mm',S[:,2]*1000),('mid mm',S[:,1]*1000),('smallest mm',S[:,0]*1000)]:
    p=np.percentile(v,[1,10,25,50,75,90,99,99.9]); print('  %-12s'%nm, ' '.join('%9.3f'%q for q in p), ' max %9.3f'%v.max())
print('  p90/p10 largest = %.2fx ; p99/p1 = %.2fx' % (np.percentile(S[:,2],90)/np.percentile(S[:,2],10), np.percentile(S[:,2],99)/np.percentile(S[:,2],1)))
print('  alpha: p1 %.4f p10 %.4f p50 %.4f p90 %.4f p99 %.4f ; frac>=0.99 %.3f%% ; frac<=0.05 %.3f%%'
      % (*np.percentile(a[r],[1,10,50,90,99]), 100*np.mean(a[r]>=0.99), 100*np.mean(a[r]<=0.05)))
print('  position box: x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f' % (P[r,0].min(),P[r,0].max(),P[r,1].min(),P[r,1].max(),P[r,2].min(),P[r,2].max()))
np.save(r'C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad\sv_sentinel.npy', sent)
