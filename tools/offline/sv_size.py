import sv_load, numpy as np
np.set_printoptions(suppress=True)

def axes(col, flip):
    s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
    return np.exp(s)

def report(tag, col, n):
    S = axes(col, False)
    Ss = np.sort(S, axis=1)          # ascending: [smin, smid, smax]
    smin, smid, smax = Ss[:,0], Ss[:,1], Ss[:,2]
    logs = np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64)
    print('=== %s   n=%d ===' % (tag, n))
    for name, v in [('largest axis mm', smax*1000), ('mid axis mm', smid*1000), ('smallest axis mm', smin*1000)]:
        p = np.percentile(v, [1,10,25,50,75,90,99,99.9])
        print('  %-17s p1 %8.3f p10 %8.3f p25 %8.3f p50 %8.3f p75 %8.3f p90 %9.3f p99 %10.3f p99.9 %11.3f  max %11.3f'
              % (name, *p, v.max()))
    print('  p90/p10 of largest axis: %.2fx    p99/p1: %.2fx' %
          (np.percentile(smax,90)/np.percentile(smax,10), np.percentile(smax,99)/np.percentile(smax,1)))
    ar = smax/np.maximum(smin,1e-12); armid = smax/np.maximum(smid,1e-12)
    print('  aspect max/min   p10 %.2f p50 %.2f p90 %.2f p99 %.2f' % tuple(np.percentile(ar,[10,50,90,99])))
    print('  aspect max/mid   p10 %.2f p50 %.2f p90 %.2f p99 %.2f' % tuple(np.percentile(armid,[10,50,90,99])))
    print('  aspect mid/min   p10 %.2f p50 %.2f p90 %.2f p99 %.2f' % tuple(np.percentile(smid/np.maximum(smin,1e-12),[10,50,90,99])))
    # disc vs needle vs blob classification (on log axes)
    disc  = (armid < 2.0) & (smid/np.maximum(smin,1e-12) > 3.0)   # two big one small
    needle= (armid > 3.0)                                          # one big two small
    blob  = (ar < 2.0)
    print('  shape: blob(<2:1) %5.2f%%   disc(2 similar, 1 thin>3x) %5.2f%%   needle(max/mid>3) %5.2f%%'
          % (100*blob.mean(), 100*disc.mean(), 100*needle.mean()))
    # clamp / saturation
    for lv in [logs.max(), logs.min()]:
        print('  frac of AXES at log-scale %.4f : %.3f%%' % (lv, 100*np.mean(logs==lv)))
    print('  frac of SPLATS with any axis at max log %.4f : %.3f%%' % (logs.max(), 100*np.mean((logs==logs.max()).any(1))))
    print('  frac of SPLATS with ALL axes at max     : %.4f%%' % (100*np.mean((logs==logs.max()).all(1))))
    return Ss, logs

if __name__ == '__main__':
    c,n = sv_load.sv();  Ssv, Lsv = report('SCANIVERSE', c, n)
    print()
    o,m = sv_load.ours(); Sus, Lus = report('OURS (build 182)', o, m)
