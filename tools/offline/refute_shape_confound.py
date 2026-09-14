"""REFUTATION TEST 7: finding 6 says the low-contribution tail is "smaller AND
blobbier ... optimiser byproduct rather than deliberate surface-fitting".
contrib is a SUM OVER PIXELS, so it is an area metric first.  Does the shape
effect survive controlling for size?  And do the finding's own "ALL" shape
percentages reconcile with the investigation brief's baseline?
"""
import os, numpy as np
import project as P, detail as Dt
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
contrib=np.load(os.path.join(Dt.SCRATCH,'contrib.npy'))
sc=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
s=np.sort(sc,1)[:,::-1]; mm=s[:,0]*1000
r21=s[:,1]/np.maximum(s[:,0],1e-12); r32=s[:,2]/np.maximum(s[:,1],1e-12)
needle=(r21<0.3)&(r32<0.3); disc=(r21>=0.3)&(r32<0.3); blob=r32>=0.3
print('brief baseline says: 16.2%% needle / 77.5%% disc / 6.3%% blob, med r21 0.76, med r32 0.18')
print('this PLY at contrib_profile.py thresholds: needle %.1f%% / disc %.1f%% / blob %.1f%%'
      % (100*needle.mean(),100*disc.mean(),100*blob.mean()))
print('med r21 %.3f  med r32 %.3f' % (np.median(r21),np.median(r32)))
print('-> the finding\'s own ALL row does NOT reconcile with the brief\'s baseline;')
print('   the two use different bucket thresholds on the same model.\n')

order=np.argsort(contrib); n=count
bot=np.zeros(n,bool); bot[order[:n//4]]=True
top=np.zeros(n,bool); top[order[3*n//4:]]=True
print('UNCONTROLLED (the finding\'s comparison):')
for nm,m in (('bottom 25%',bot),('top 25%',top)):
    print('  %-11s med axis %6.3f mm  blob %5.1f%%  disc %5.1f%%  needle %5.1f%%'
          % (nm,np.median(mm[m]),100*blob[m].mean(),100*disc[m].mean(),100*needle[m].mean()))

print('\nCONTROLLED for size -- same comparison INSIDE each size decile:')
edges=np.percentile(mm,np.arange(0,101,10))
print('%-16s %8s %10s %10s %10s' % ('size decile (mm)','n','blob bot%','blob top%','delta'))
tb=[];tt=[]
for i in range(10):
    lo,hi=edges[i],edges[i+1]
    band=(mm>=lo)&(mm<=hi if i==9 else mm<hi)
    q=contrib[band]; o=np.argsort(q); k=len(q)//4
    idx=np.nonzero(band)[0][o]
    b=blob[idx[:k]].mean()*100; t=blob[idx[-k:]].mean()*100
    tb.append(b); tt.append(t)
    print('%-16s %8d %9.1f%% %9.1f%% %9.1f'%('%.2f-%.2f'%(lo,hi),band.sum(),b,t,b-t))
print('%-16s %8s %9.1f%% %9.1f%% %9.1f'%('MEAN','',np.mean(tb),np.mean(tt),np.mean(tb)-np.mean(tt)))
print('\nuncontrolled blob gap bottom-minus-top: %.1f pts'
      % (100*blob[bot].mean()-100*blob[top].mean()))
print('size-controlled blob gap:                %.1f pts  (%.0f%% of it was size)'
      % (np.mean(tb)-np.mean(tt),
         100*(1-(np.mean(tb)-np.mean(tt))/(100*blob[bot].mean()-100*blob[top].mean()))))
