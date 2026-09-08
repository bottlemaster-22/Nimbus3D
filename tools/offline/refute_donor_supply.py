"""The relocation proposal rests on a donor pool that the census MEASURES and
contradicts.

sizesim2 models donor supply as `davail = int(pop * donor_frac)` with
donor_frac = 0.07354, taken from the share of the FINISHED model whose opacity
is below relocationDonorOpacity 0.05. That is a standing stock of ~22,000 that
is silently refilled in full every pass.

Two things in build 182's census say that is wrong.

1. WHICH TERM SUPPLIED THE DONORS. The donor test is
       opacity < relocationDonorOpacity || stats[i].visAccum <= 0
   and `relocationDonorsAvailable` equals `splatCount -
   candidatesAfterVisibilityFilter` EXACTLY at iterations 300 and 400, and to
   within 13 at 500. Every donor build 182 ever had came from visAccum <= 0.
   The opacity term supplied 13 donors in the whole run.

2. WHAT HAPPENS WHEN YOU DRAW ON IT. Three consecutive relocation passes take
   the pool 18,222 -> 4,833 -> 1,167, a 73.5% then 75.9% decay. Relocation
   CONSUMES donors: it writes a live, opacity-corrected copy of the target into
   the donor slot, so the slot stops being faint and stops being invisible.
   Passes 600-800, with relocation NOT running, show the pool standing at
   1,142 / 1,235 / 1,496. That is the replenishment rate, and it is ~1,100-1,500
   per pass against the simulator's 22,062.

This script re-runs the proposal with the measured supply.
"""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))
BASE = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
            opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])


def head(**kw):
    b = dict(BASE); b.update(kw)
    return S.Cfg().clone(**b)


def show(name, cfg, donor_series=None):
    s, t, h = S.run(cfg, donor_series=donor_series)
    st = S.stats(s)
    n = t['nshrink']
    print('%-52s p10 %7.3f p50 %7.3f p90 %8.3f  spread %8.3fx sd %.4f  '
          'reloc %7d  mean shrinks %.3f  never %.1f%%'
          % (name, st['p10'], st['p50'], st['p90'], st['spread'], st['sd'],
             t['reloc'], n.mean(), 100.0 * (n == 0).mean()))
    return st, t


print('=== what the census says the pool actually is ===')
c = json.load(open(S.CENSUS))['densifyPasses']
for r in c:
    if r['iteration'] <= 900:
        print('   it %4d  invisible %6d  donors %6d  relocated %6d'
              % (r['iteration'], r['splatCountBefore'] - r['candidatesAfterVisibilityFilter'],
                 r['relocationDonorsAvailable'], r['relocated']))

print('\n=== the finding, as published (donor_frac 0.07354 = 22,062/pass) ===')
show('head', head())
show('reloc unconditional, frac 0.05', head(reloc_always=True))
show('reloc unconditional, frac 0.10', head(reloc_always=True, reloc_frac=0.10))

print('\n=== the same thing with the MEASURED donor supply ===')
# Flat per-pass supply, swept across the range the census actually shows.
for supply in (1142, 1500, 2300, 5000, 22062):
    frac = supply / 300000.0
    show('reloc unconditional, supply %5d/pass (frac %.5f)' % (supply, frac),
         head(reloc_always=True, donor_frac=frac))

print('\n=== census-exact donor series, extended by the measured replenishment ===')
# Passes 300-500 get their real numbers; every later pass gets the standing
# pool the census shows when relocation is NOT drawing on it (1142/1235/1496),
# which is the most generous reading of the replenishment rate.
donors = {}
for r in c:
    it = r['iteration']
    inv = r['splatCountBefore'] - r['candidatesAfterVisibilityFilter']
    donors[it] = inv
print('   series:', {k: v for k, v in sorted(donors.items()) if k <= 1200}, '...')
show('reloc unconditional, census invisible-count series',
     head(reloc_always=True), donor_series=donors)
show('reloc unconditional + splitShare 1.0, census series',
     head(reloc_always=True, split_share=1.0), donor_series=donors)
show('splitShare 1.0 alone (no relocation change)', head(split_share=1.0))
