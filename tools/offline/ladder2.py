"""Exact per-slot shrink counts, with the calibrated optimiser term ON so the
split/clone quantile cut is well defined."""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))


def head(**kw):
    b = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
             opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    b.update(kw)
    return S.Cfg().clone(**b)


def show(name, cfg):
    s, t, h = S.run(cfg)
    n = t['nshrink']
    st = S.stats(s)
    print('%-46s pop %7d  split %7d clone %7d reloc %8d  spread %7.3fx'
          % (name, len(n), t['split'], t['clone'], t['reloc'], st['spread']))
    hist = np.bincount(n, minlength=9)
    print('     shrinks per splat: ' + ' '.join(
        '%d:%.1f%%' % (i, 100.0 * c / len(n))
        for i, c in enumerate(hist[:9]) if c))
    print('     never shrunk %.1f%%   mean %.3f   share reaching >=3 shrinks %.1f%%'
          % (100.0 * (n == 0).mean(), n.mean(), 100.0 * (n >= 3).mean()))


show('CURRENT HEAD', head())
show('splitShare 1.0', head(split_share=1.0))
show('relocation unconditional (frac 0.05)', head(reloc_always=True))
show('relocation unconditional (frac 0.10)', head(reloc_always=True, reloc_frac=0.10))
show('share 1.0 + reloc unconditional', head(split_share=1.0, reloc_always=True))
show('the four levers asked about, together',
     head(interval=50, reloc_frac=0.10, split_shrink=2.0, cap=400000))
