"""The candidate settings, judged on the whole distribution and on its costs."""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))
scan = np.load('scan_largest_mm.npy')


def head(**kw):
    b = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
             opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    b.update(kw)
    return S.Cfg().clone(**b)


def show(name, cfg):
    s, t, h = S.run(cfg)
    lin = np.exp(s)
    big = 1000 * lin.max(axis=1)
    srt = np.sort(lin, axis=1)
    cover = (srt[:, 2] * srt[:, 1]).sum() / (150000 * 0.0073966 ** 2)
    q = np.percentile(big, [1, 10, 25, 50, 75, 90, 99])
    print('%-42s' % name)
    print('    p1 %7.3f p10 %7.3f p25 %7.3f p50 %7.3f p75 %7.3f p90 %8.3f p99 %9.3f'
          % tuple(q))
    print('    spread %7.3fx  sd %.4f  pop %7d  below 1 mm %5.2f%%  '
          'coverage %5.2fx  mean shrinks %.3f'
          % (q[5] / q[1], float(np.std(np.log10(big))), len(big),
             100.0 * (big < 1.0).mean(), cover, t['nshrink'].mean()))


print('SCANIVERSE (target)')
print('    p1 %7.3f p10 %7.3f p25 %7.3f p50 %7.3f p75 %7.3f p90 %8.3f p99 %9.3f'
      % tuple(np.percentile(scan, [1, 10, 25, 50, 75, 90, 99])))
print('    spread %7.3fx  sd %.4f  pop %7d  below 1 mm %5.2f%%'
      % (np.percentile(scan, 90) / np.percentile(scan, 10),
         float(np.std(np.log10(scan))), len(scan), 100.0 * (scan < 1.0).mean()))
print()
show('CURRENT HEAD', head())
show('splitShare 1.0 only', head(split_share=1.0))
show('splitShrink 2.0 only', head(split_shrink=2.0))
show('splitShare 1.0 + splitShrink 2.0',
     head(split_share=1.0, split_shrink=2.0))
show('splitShare 0.8 + splitShrink 2.2',
     head(split_share=0.8, split_shrink=2.2))
show('splitShare 0.6 + relocation unconditional',
     head(split_share=0.6, reloc_always=True))
