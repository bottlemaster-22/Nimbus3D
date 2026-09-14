import numpy as np, os, sys
import project as P
SV = r"C:\Users\Undea\Downloads\Four Marks.ply"
for label, path in (('LiKOVA  ', os.path.join(P.D,'model','model.ply')), ('Scaniverse', SV)):
    col, n = P.load_ply(path)
    logs = np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64)
    s = np.exp(np.clip(logs,-12,3))
    ss = np.sort(s,1)
    smin,smid,smax = ss[:,0],ss[:,1],ss[:,2]
    lam = s*s; p = lam/np.maximum(lam.sum(1,keepdims=True),1e-20)
    lp = np.log(np.maximum(p,1e-20)); H = -(p*lp).sum(1); rank = np.exp(H)
    print('\n=== %s : %d splats ===' % (label, n))
    print('  largest axis mm    p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(np.percentile(smax*1000,[10,50,90])))
    print('  smallest axis mm   p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(np.percentile(smin*1000,[10,50,90])))
    print('  aspect max/min     p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(np.percentile(smax/np.maximum(smin,1e-12),[10,50,90])))
    print('  aspect max/mid     p10 %7.3f p50 %7.3f p90 %7.3f' % tuple(np.percentile(smax/np.maximum(smid,1e-12),[10,50,90])))
    print('  effective rank     p10 %7.4f p50 %7.4f p90 %7.4f  mean %.4f' % (*np.percentile(rank,[10,50,90]), rank.mean()))
    print('  frac rank > 2.0    %.4f      frac rank < 2.0  %.4f' % ((rank>2).mean(), (rank<2).mean()))
    print('  p90/p10 largest    %.3f' % (np.percentile(smax,90)/np.percentile(smax,10)))
    # what would the disc prior do to Scaniverse's splats?
    res = rank-2.0; k = -4*0.001*res*rank
    g = k[:,None]*p*(lp+H[:,None])
    print('  |disc grad w=.001| p50 %.3e   (residual |rank-2| p50 %.4f)' % (np.median(np.abs(g)), np.median(np.abs(res))))
