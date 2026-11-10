"""score_persistence is invented, and its sensitivity was measured at the WRONG
operating point.

Finding 7 sweeps score_persistence 0 -> 1 at the head baseline (splitShare 0.2)
gets 2.078x -> 2.232x, calls that a 7.4% band, and concludes 'nothing changes
in the ranking of levers'. But at splitShare 0.2 only a fifth of growth is a
split, so how concentrated the selection is barely matters. At splitShare 1.0 -
the setting being proposed - 154,608 splits have to land somewhere, and
persistence is exactly the parameter that decides whether they land on 150,000
distinct lineages or on 40,000. Re-measure the band AT the proposed setting.
"""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))
def head(**kw):
    b = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
             opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    b.update(kw); return S.Cfg().clone(**b)

TARGET = dict(p10=1.327, p50=3.388, p90=11.825)

def miss(st):
    return float(np.sqrt(np.mean([(np.log10(st[k]) - np.log10(TARGET[k])) ** 2
                                  for k in ('p10', 'p50', 'p90')])))

for label, extra in (('splitShare 0.2 (the baseline the band was measured at)', dict()),
                     ('splitShare 1.0 (the proposed setting)', dict(split_share=1.0)),
                     ('splitShare 1.0 + splitShrink 2.0 (the recommendation)',
                      dict(split_share=1.0, split_shrink=2.0))):
    print('\n--- %s' % label)
    rows = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfg = head(score_persistence=p, **extra)
        s, t, h = S.run(cfg)
        st = S.stats(s)
        n = t['nshrink']
        rows.append(st['spread'])
        print('   persistence %.2f  p10 %7.3f p50 %7.3f p90 %8.3f  spread %8.3fx  '
              'sd %.4f  never shrunk %5.1f%%  miss %.3f dec'
              % (p, st['p10'], st['p50'], st['p90'], st['spread'], st['sd'],
                 100.0 * (n == 0).mean(), miss(st)))
    lo, hi = min(rows), max(rows)
    mid = rows[2]
    print('   BAND: %.3fx to %.3fx around %.3fx  =  -%.1f%% / +%.1f%%'
          % (lo, hi, mid, 100 * (1 - lo / mid), 100 * (hi / mid - 1)))

# The rng seed is also unstated. How much is just noise?
print('\n--- rng seed sensitivity at the recommendation')
sp = []
for seed in range(7, 13):
    s, t, h = S.run(head(split_share=1.0, split_shrink=2.0, rng_seed=seed))
    st = S.stats(s); sp.append(st['spread'])
    print('   seed %d  spread %8.3fx  sd %.4f  miss %.3f dec' % (seed, st['spread'], st['sd'], miss(st)))
print('   spread across seeds: %.3f to %.3f (%.1f%% of the mean)'
      % (min(sp), max(sp), 100 * (max(sp) - min(sp)) / np.mean(sp)))
