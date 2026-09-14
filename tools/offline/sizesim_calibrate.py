"""Reproduce build 182 exactly, then solve for what the optimiser did.

Step 1: run the densification rules alone on build 182's schedule and check the
        three event counts against its census.
Step 2: the residual between that run's final distribution and the exported
        model's is the optimiser's. Solve for the per-pass drift and diffusion
        that closes it. Those two numbers are then carried, unchanged, into
        every lever sweep, so the sweep is comparing schedules and not
        comparing guesses about Adam.
"""
import json
import numpy as np
import sizesim2 as S


def event_check():
    p = json.load(open(S.CENSUS))['densifyPasses']
    m = dict(reloc=sum(r['relocated'] for r in p),
             split=sum(r['addedBySplit'] for r in p),
             clone=sum(r['addedByClone'] for r in p))
    cfg, prune, donors = S.build182()
    s, t, h = S.run(cfg, prune_series=prune, donor_series=donors)
    print('EVENT COUNTS, census against simulator (no optimiser term):')
    print('  %-12s %10s %10s %8s' % ('', 'census', 'sim', 'error'))
    for k in ('reloc', 'split', 'clone'):
        e = 100.0 * (t[k] - m[k]) / max(m[k], 1)
        print('  %-12s %10d %10d %7.1f%%' % (k, m[k], t[k], e))
    print('  relocation iterations sim %s   census [300, 400, 500]'
          % t['reloc_iters'])
    print('  final population sim %d   census 299209' % h[-1]['pop'])
    print('  densification-only final distribution:', fmt(S.stats(s)))
    print('  the exported model actually is:        ',
          'p10 %6.3f p50 %6.3f p90 %6.3f spread %6.3fx sd %.4f'
          % (S.MEASURED['p10'], S.MEASURED['p50'], S.MEASURED['p90'],
             S.MEASURED['spread'], S.MEASURED['sd']))
    return s, S.stats(s)


def fmt(d):
    return ('p10 %6.3f p50 %6.3f p90 %6.3f spread %6.3fx sd %.4f'
            % (d['p10'], d['p50'], d['p90'], d['spread'], d['sd']))


def calibrate(base):
    """Per-pass drift and diffusion, in decades, that carry the
    densification-only endpoint onto the measured one over 30 passes."""
    iters = 3000
    drift = (np.log10(S.MEASURED['p50']) - np.log10(base['p50'])) / iters
    resid_var = S.MEASURED['sd'] ** 2 - base['sd'] ** 2
    diffuse = np.sqrt(max(resid_var, 0) / iters)
    print('\nOPTIMISER TERM, solved from the residual:')
    print('  densification alone leaves the median at %.3f mm; the model is'
          ' %.3f mm' % (base['p50'], S.MEASURED['p50']))
    print('  -> drift    %+.7f decades/iteration  (x%.3f over 3000 iterations)'
          % (drift, 10 ** (drift * iters)))
    print('  densification alone leaves sd %.4f decades; the model is %.4f'
          % (base['sd'], S.MEASURED['sd']))
    print('  -> diffuse   %.7f decades/iteration  (%.4f decades over 3000)'
          % (diffuse, diffuse * np.sqrt(iters)))
    return drift, diffuse


def verify(drift, diffuse):
    cfg, prune, donors = S.build182(opt_drift=drift, opt_diffuse=diffuse)
    s, t, h = S.run(cfg, prune_series=prune, donor_series=donors)
    st = S.stats(s)
    print('\nBUILD 182 REPRODUCED with that optimiser term:')
    print('  sim    ', fmt(st))
    print('  actual  p10 %6.3f p50 %6.3f p90 %6.3f spread %6.3fx sd %.4f'
          % (S.MEASURED['p10'], S.MEASURED['p50'], S.MEASURED['p90'],
             S.MEASURED['spread'], S.MEASURED['sd']))
    print('  error   p10 %+.1f%% p50 %+.1f%% p90 %+.1f%% spread %+.1f%%'
          % (100 * (st['p10'] / S.MEASURED['p10'] - 1),
             100 * (st['p50'] / S.MEASURED['p50'] - 1),
             100 * (st['p90'] / S.MEASURED['p90'] - 1),
             100 * (st['spread'] / S.MEASURED['spread'] - 1)))
    return st


if __name__ == '__main__':
    s, base = event_check()
    drift, diffuse = calibrate(base)
    verify(drift, diffuse)
    json.dump(dict(opt_drift=drift, opt_diffuse=diffuse),
              open('optcal.json', 'w'))
    print('\nwrote optcal.json')
