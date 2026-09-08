"""The lever sweep. Current head, then one change at a time.

Every run uses the SAME optimiser drift and diffusion solved in
sizesim_calibrate.py against build 182's exported model, so any difference
between rows is the schedule and nothing else.
"""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))


def head(**kw):
    """Current head: seedTarget 150k (fillFraction 0.5), seedSpacingCompensation
    0, splitScreenRadiusPx 0, everything else at TrainerSupport defaults."""
    base = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
                opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    base.update(kw)
    return S.Cfg().clone(**base)


def first_pass_reaching(hist, spread):
    for h in hist:
        if h['spread'] >= spread:
            return h['it']
    return None


def row(name, cfg, note=''):
    s, t, h = S.run(cfg)
    st = S.stats(s)
    r3 = first_pass_reaching(h, 3.0)
    r89 = first_pass_reaching(h, 8.913)
    print('%-34s p10 %7.3f p50 %7.3f p90 %8.3f  spread %6.3fx  sd %.4f  '
          'pop %7d  split %7d clone %7d reloc %7d  3x@%s  8.9x@%s %s'
          % (name, st['p10'], st['p50'], st['p90'], st['spread'], st['sd'],
             st['n'], t['split'], t['clone'], t['reloc'],
             r3 if r3 else '  never', r89 if r89 else '  never', note))
    return dict(name=name, st=st, tally=t, hist=h, r3=r3, r89=r89)


def main():
    print('optimiser term carried into every row: drift %+.5f dec/pass, '
          'diffusion %.5f dec/pass  (calibrated on build 182)'
          % (CAL['opt_drift'], CAL['opt_diffuse']))
    print('TARGET (Scaniverse, same room): p10 1.327 p50 3.388 p90 11.825 '
          'spread 8.913x sd 0.6361\n')

    base = row('CURRENT HEAD (baseline)', head())
    print()
    print('--- THE FOUR LEVERS ASKED ABOUT, one at a time --------------------')
    row('densifyInterval 100 -> 50', head(interval=50))
    row('maxRelocationFraction .05 -> .10', head(reloc_frac=0.10))
    row('splitShrink 1.6 -> 2.0', head(split_shrink=2.0))
    row('splatCap 300k -> 400k', head(cap=400000))
    print()
    print('--- all four together ---------------------------------------------')
    row('all four', head(interval=50, reloc_frac=0.10, split_shrink=2.0,
                         cap=400000))
    print()
    print('--- levers the model says actually bind ---------------------------')
    row('splitShare 0.2 -> 1.0', head(split_share=1.0))
    row('splitScreenRadiusPx back on (b182)', head(screen_split_fraction=0.9918))
    row('seeds 150k -> 60k (more headroom)', head(seeds=60000))
    row('seeds 150k -> 30k', head(seeds=30000))
    row('densifyEnd 0.85 -> 1.0', head(densify_end=1.0))
    row('densifyStart 0.10 -> 0.02', head(densify_start=0.02))
    row('prune 0.00077 -> 0.01 of pop/pass', head(prune_frac=0.01))
    row('prune -> 0.05 of pop/pass', head(prune_frac=0.05))
    row('scorePersistence 0.5 -> 0.0', head(score_persistence=0.0))
    row('scorePersistence 0.5 -> 1.0', head(score_persistence=1.0))
    print()
    print('--- splitShareOfGrowth sweep, everything else at head defaults ----')
    for sh in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        row('  splitShare %.1f' % sh, head(split_share=sh))
    print()
    print('--- splitShrink sweep at splitShare 1.0 ---------------------------')
    for k in (1.4, 1.6, 2.0, 2.5, 3.0):
        row('  share 1.0, shrink %.1f' % k,
            head(split_share=1.0, split_shrink=k))
    print()
    print('--- interval sweep at splitShare 1.0 ------------------------------')
    for iv in (100, 50, 25):
        row('  share 1.0, interval %d' % iv,
            head(split_share=1.0, interval=iv))
    print()
    print('--- the combination that actually gets there ----------------------')
    row('seeds 30k + share 1.0 + prune .01',
        head(seeds=30000, split_share=1.0, prune_frac=0.01))
    row('  + interval 50', head(seeds=30000, split_share=1.0,
                                prune_frac=0.01, interval=50))
    row('  + shrink 2.0', head(seeds=30000, split_share=1.0,
                               prune_frac=0.01, interval=50,
                               split_shrink=2.0))
    row('  + start 0.02, end 1.0',
        head(seeds=30000, split_share=1.0, prune_frac=0.01, interval=50,
             split_shrink=2.0, densify_start=0.02, densify_end=1.0))
    print()
    print('--- baseline trace, so the plateau is visible ---------------------')
    for h in base['hist']:
        print('  it %4d pop %7d alw %6d rl %6d  p10 %7.3f p50 %7.3f p90 %7.3f'
              '  spread %6.3fx sd %.4f'
              % (h['it'], h['pop'], h['alw'], h['rl'], h['p10'], h['p50'],
                 h['p90'], h['spread'], h['sd']))


if __name__ == '__main__':
    main()
