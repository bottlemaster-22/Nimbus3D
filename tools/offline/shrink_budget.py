"""The budget question, stripped of everything else.

A splat's size only ever moves by whole multiples of splitShrink under
densification. So: how many such steps does each splat need to take, to carry
a population that starts at ONE size onto Scaniverse's distribution, and how
many steps does the schedule actually issue?

Both sides are counts of the same unit, so they can be compared directly.
"""
import json
import numpy as np

SEED_MM = 7.397          # measured: every seed, 100 per cent of them
SHRINK = 1.6
PASSES_IN_WINDOW = 23    # iterations 300..2500 inclusive, every 100

scan = np.load('scan_largest_mm.npy')
ours = np.load('ours_largest_mm.npy')
CAL = json.load(open('optcal.json'))
DRIFT_TOTAL = 10 ** (CAL['opt_drift'] * 3000)   # what Adam does to the median


def steps_needed(target_mm, start_mm, shrink=SHRINK):
    return np.log(start_mm / target_mm) / np.log(shrink)


def main():
    print('SEED: every splat at %.3f mm (measured, zero spread).' % SEED_MM)
    print('Adam alone multiplies size by %.3f over the run (calibrated on '
          'build 182), so without densification the population lands at '
          '%.3f mm.' % (DRIFT_TOTAL, SEED_MM * DRIFT_TOTAL))
    start = SEED_MM * DRIFT_TOTAL

    # Quantile-match: the k-th smallest of ours must become the k-th smallest
    # of a Scaniverse-shaped distribution.
    q = np.linspace(0.0005, 0.9995, 2000)
    tgt = np.quantile(scan, q)
    need = steps_needed(tgt, start)
    need_pos = np.clip(need, 0, None)     # densification cannot grow anything
    print('\nSTEPS OF %.1fx EACH SPLAT NEEDS (three-axis shrink):' % SHRINK)
    for p in (10, 25, 50, 75, 90, 95, 99):
        print('  p%-3d target %9.3f mm  needs %6.2f steps'
              % (p, np.quantile(scan, p / 100.0),
                 steps_needed(np.quantile(scan, p / 100.0), start)))
    print('  population mean, clipped at zero: %.3f steps per splat' % need_pos.mean())
    print('  -> total three-axis shrink events required over 300,000 splats:'
          ' %.0f' % (need_pos.mean() * 300000))
    print('  (a ONE-axis relocation shrink moves the largest axis only every'
          ' other time on a disc, so it is worth about half a step:'
          ' %.0f relocations)' % (need_pos.mean() * 300000 * 2))

    print('\nWHAT THE SCHEDULE ISSUES, current head:')
    cap, growth_frac, seeds = 300000, 0.15, 150000
    alw = []
    pop = seeds
    for i in range(PASSES_IN_WINDOW):
        a = min(cap - pop, int(cap * growth_frac))
        # after the cap is full the allowance equals the previous pass's
        # deletions; build 182 measured 44..385, mean 232.
        if a <= 0:
            a = 232
        alw.append(a)
        pop = min(pop + a, cap)
    created = sum(alw)
    split_share = 0.20
    splits = created * split_share
    print('  growth allowance per pass: %s ... %s (sum %d)'
          % (alw[:4], alw[-3:], created))
    print('  at splitShareOfGrowth %.2f that is %.0f splits' % (split_share, splits))
    print('  each split shrinks TWO slots (the parent in place and the child),'
          ' so %.0f three-axis shrink events' % (2 * splits))
    print('  relocations: ZERO, because the relocation branch needs'
          ' growthAllowance == 0, and the prune reopens headroom every pass')
    supply = 2 * splits
    demand = need_pos.mean() * 300000
    print('\n  SUPPLY %.0f shrink events   DEMAND %.0f   -> short by %.1fx'
          % (supply, demand, demand / supply))

    print('\nAND THE SAME SUM IF EVERY LEVER IS PULLED:')
    for name, sh, ss, cp, iv in (
            ('splitShrink 2.0', 2.0, 0.20, 300000, 100),
            ('splitShare 1.0', 1.6, 1.00, 300000, 100),
            ('interval 50', 1.6, 0.20, 300000, 50),
            ('cap 400k', 1.6, 0.20, 400000, 100),
            ('all four', 2.0, 1.00, 400000, 50)):
        passes = int((0.85 - 0.10) * 3000 / iv) + 1
        pop, tot = seeds, 0
        for i in range(passes):
            a = min(cp - pop, int(cp * 0.15))
            if a <= 0:
                a = 232 * iv // 100
            tot += a
            pop = min(pop + a, cp)
        sp = tot * ss
        sup = 2 * sp
        nd = np.clip(steps_needed(tgt, start, sh), 0, None).mean() * cp
        print('  %-16s passes %3d  created %7d  splits %7d  supply %8.0f'
              '  demand %8.0f  short %5.1fx'
              % (name, passes, tot, sp, sup, nd, nd / sup))


if __name__ == '__main__':
    main()
