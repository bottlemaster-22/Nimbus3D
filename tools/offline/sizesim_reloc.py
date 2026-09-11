"""Two questions the four-lever sweep cannot answer.

1. The relocation branch is the `else` of the growth branch, so it runs ONLY
   when growthAllowance is exactly zero. Under the current head that is never,
   because the prune reopens headroom every pass. What is relocation worth if
   it is made to run alongside growth instead of instead of it?

2. Which reachable setting actually lands on Scaniverse's distribution, judged
   on all three percentiles rather than on the p90/p10 ratio alone? A ratio can
   be hit by overshooting the small end into dust.
"""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))
TGT = dict(p10=1.327, p50=3.388, p90=11.825, spread=8.913)


def head(**kw):
    base = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
                opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    base.update(kw)
    return S.Cfg().clone(**base)


def coverage(s):
    """Sum of the disc face area (largest x middle) over the population,
    relative to 150,000 seeds at 7.397 x 7.397 mm. Splitting shrinks this and
    cloning does not, so it is the price of every row below."""
    lin = np.exp(s)
    srt = np.sort(lin, axis=1)
    area = (srt[:, 2] * srt[:, 1]).sum()
    return area / (150000 * (0.0073966 ** 2))


def miss(st):
    """RMS error in decades across the three percentiles."""
    e = [np.log10(st['p10'] / TGT['p10']), np.log10(st['p50'] / TGT['p50']),
         np.log10(st['p90'] / TGT['p90'])]
    return float(np.sqrt(np.mean(np.square(e))))


def row(name, cfg):
    s, t, h = S.run(cfg)
    st = S.stats(s)
    print('%-40s p10 %7.3f p50 %7.3f p90 %8.3f  spread %8.3fx  sd %.4f  '
          'reloc %7d split %7d clone %7d  cover %5.2fx  miss %.3f dec'
          % (name, st['p10'], st['p50'], st['p90'], st['spread'], st['sd'],
             t['reloc'], t['split'], t['clone'], coverage(s), miss(st)))
    return st


def main():
    print('TARGET  p10 %.3f p50 %.3f p90 %.3f  spread %.3fx  sd 0.6361'
          % (TGT['p10'], TGT['p50'], TGT['p90'], TGT['spread']))
    print('"cover" is total disc face area relative to the 150k seed set; '
          '1.00 means the model still covers the room as well as the seeds did.')
    print('"miss" is the RMS distance from the target in decades over '
          'p10/p50/p90; 0 is a perfect match.\n')

    print('--- relocation: as an else-branch (current) vs as its own pass ----')
    row('current head', head())
    row('reloc made unconditional, frac 0.05', head(reloc_always=True))
    row('reloc unconditional, frac 0.10', head(reloc_always=True, reloc_frac=0.10))
    row('reloc unconditional, frac 0.20', head(reloc_always=True, reloc_frac=0.20))
    row('reloc unconditional, donors unlimited',
        head(reloc_always=True, reloc_frac=0.05, donor_frac=1.0))
    row('reloc uncond .20 + donors unlimited',
        head(reloc_always=True, reloc_frac=0.20, donor_frac=1.0))
    print()
    print('--- searching for the setting that lands ON the target ------------')
    best = None
    for share in (0.6, 0.8, 1.0):
        for shrink in (1.6, 1.8, 2.0, 2.2):
            for ra in (False, True):
                cfg = head(split_share=share, split_shrink=shrink,
                           reloc_always=ra)
                s, t, h = S.run(cfg)
                st = S.stats(s)
                m = miss(st)
                tag = ('share %.1f shrink %.1f%s'
                       % (share, shrink, ' +reloc' if ra else ''))
                if best is None or m < best[0]:
                    best = (m, tag, st, coverage(s), t)
                print('  %-32s p10 %7.3f p50 %7.3f p90 %8.3f spread %7.3fx '
                      'cover %5.2fx  miss %.3f dec'
                      % (tag, st['p10'], st['p50'], st['p90'], st['spread'],
                         coverage(s), m))
    print('\nCLOSEST: %s  miss %.3f decades  spread %.3fx  coverage %.2fx'
          % (best[1], best[0], best[2]['spread'], best[3]))


if __name__ == '__main__':
    main()
