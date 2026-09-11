import sv_load, numpy as np
from scipy.spatial import cKDTree
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"

def prep(col, sent=None, flip=False):
    P = np.stack([col['x'],col['y'],col['z']],1).astype(np.float64)
    L = np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64)
    a = 1/(1+np.exp(-np.clip(col['opacity'].astype(np.float64),-60,60)))
    if sent is not None: P,L,a = P[~sent],L[~sent],a[~sent]
    return P,np.exp(L),a

def packing(tag,P,S,a,ctr,rmax):
    r=np.linalg.norm(P-ctr,axis=1); m=r<rmax
    P,S,a=P[m],S[m],a[m]
    t=cKDTree(P); d,_=t.query(P,k=9)
    nn1=d[:,1]; nn8=d[:,8]
    smax=S.max(1); smid=np.sort(S,1)[:,1]; smin=S.min(1)
    print('=== %s  (r<%.1fm, n=%d) ===' % (tag,rmax,m.sum()))
    print('  nearest-neighbour distance mm : p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(1000*np.percentile(nn1,[10,50,90])))
    print('  8th-neighbour distance mm     : p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(1000*np.percentile(nn8,[10,50,90])))
    print('  largest axis (sigma) mm       : p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(1000*np.percentile(smax,[10,50,90])))
    for nm,v in [('smax/nn1',smax/nn1),('2*smax/nn8*?',None),('smax/nn8',smax/nn8),('smin/nn1',smin/nn1)]:
        if v is None: continue
        print('  %-10s : p10 %6.3f p50 %6.3f p90 %6.3f  mean %6.3f' % (nm,*np.percentile(v,[10,50,90]),v.mean()))
    # how many neighbours fall inside one 1-sigma largest axis
    cnt=np.array([len(x) for x in t.query_ball_point(P, smax)])-1
    print('  neighbours within own 1-sigma largest axis: p50 %d p90 %d mean %.2f' % (np.percentile(cnt,50),np.percentile(cnt,90),cnt.mean()))
    cnt3=np.array([len(x) for x in t.query_ball_point(P, 3*smax)])-1
    print('  neighbours within own 3-sigma largest axis: p50 %d p90 %d mean %.2f' % (np.percentile(cnt3,50),np.percentile(cnt3,90),cnt3.mean()))
    # correlation between local density and size
    lr=np.log(nn8); ls=np.log(smax)
    print('  corr(log local spacing, log size) = %.4f ; slope %.3f' % (np.corrcoef(lr,ls)[0,1], np.polyfit(lr,ls,1)[0]))
    # size spread WITHIN a local neighbourhood (is size locally uniform or locally varied?)
    idx=t.query(P,k=17)[1]
    loc=np.log(smax)[idx]
    spread=np.exp(loc.max(1)-loc.min(1))
    print('  local size spread (max/min over 16 nearest): p50 %.2fx p90 %.2fx' % tuple(np.percentile(spread,[50,90])))
    print('  alpha: p10 %.3f p50 %.3f p90 %.3f' % tuple(np.percentile(a,[10,50,90])))
    return

c,_=sv_load.sv(); sent=np.load(SCR+r"\sv_sentinel.npy")
P,S,a = prep(c,sent)
packing('SCANIVERSE', P,S,a, np.array([-1.26375,-0.24857,0.39607]), 3.5)
print()
o,_=sv_load.ours()
Po = np.stack([o['x'],-o['y'],-o['z']],1).astype(np.float64)
So = np.exp(np.stack([o['scale_0'],o['scale_1'],o['scale_2']],1).astype(np.float64))
ao = 1/(1+np.exp(-o['opacity'].astype(np.float64)))
packing('OURS', Po,So,ao, Po.mean(0), 100.0)
