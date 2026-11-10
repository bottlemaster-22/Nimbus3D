"""Cross-check: does the hand instruction count reproduce the MEASURED
backward/forward time ratio, and what does one device atomic actually cost?

Populations from bwdwork.py (peak frame 4, 1,059,741 tile instances, which
matches the census peakTileInstances of 1,058,989 to 0.07 per cent).
Timings are the owner's measured 11.05 ms backward / 3.03 ms forward.

Run:  python -u bwdcheck.py
"""

FULL = 75089548        # full-body pairs (12 atomics each)
POWER = 174420305      # pairs reaching delta + power
SKIP = 46367104        # compare-only globalIndex > lastContributor
SLOTS_A = 3350713      # active (simdgroup, j) slots
SLOTS_I = 6899607      # iterated (simdgroup, j) slots
FWD_ITER = 175870241   # forward loop-body entries
PIXELS = 388800

B_PRE, B_BODY, B_GUARD, B_ATOM = 24, 53, 60, 12
F_PRE, F_BODY = 20, 12
T_FWD, T_BWD = 3.03e-3, 11.05e-3


def main():
    fwd = FWD_ITER * F_PRE + FULL * F_BODY
    bwd = SKIP * 3 + POWER * B_PRE + FULL * (B_BODY + B_GUARD + B_ATOM)
    alu = bwd - FULL * B_ATOM
    rate = fwd / T_FWD
    budget = T_BWD * rate
    k = (budget - alu) / (FULL * B_ATOM)

    print('forward   %.2f G instructions in %.2f ms -> %.3f x10^12 instr/s'
          % (fwd / 1e9, T_FWD * 1e3, rate / 1e12))
    print('backward  %.2f G instructions in %.2f ms -> %.3f x10^12 instr/s'
          % (bwd / 1e9, T_BWD * 1e3, bwd / T_BWD / 1e12))
    print('hand-count ratio %.2fx against a MEASURED %.2fx'
          % (bwd / fwd, T_BWD / T_FWD))
    print('\nsolving for the issue-slot cost of one device atomic_fetch_add,')
    print('assuming both kernels issue at the forward rate:')
    print('  slot budget %.2f G, non-atomic instructions %.2f G'
          % (budget / 1e9, alu / 1e9))
    print('  ONE DEVICE ATOMIC = %.2f ISSUE SLOTS' % k)

    print('\nbackward stream split (atomics weighted %.2f):' % k)
    tot = SKIP * 3 + POWER * B_PRE + FULL * (B_BODY + B_GUARD) + FULL * B_ATOM * k
    for name, v in [
        ('compare-only skips', SKIP * 3),
        ('pre-gate (delta, power, cutoff, exp, alpha)', POWER * B_PRE),
        ('body arithmetic', FULL * B_BODY),
        ('12 x trainer_finiteOrZero', FULL * B_GUARD),
        ('12 x device atomic', FULL * B_ATOM * k),
    ]:
        print('  %-44s %6.2f G  %5.1f%%' % (name, v / 1e9, 100 * v / tot))

    def ms(slots):
        return 1000.0 * slots / rate

    rows = [
        ('drop 12 finiteOrZero guards, 4 per-PIXEL root checks instead',
         FULL * 48, FULL * 60),
        ('12 atomics -> 12 simd_sum per active (group,j) slot',
         (FULL - SLOTS_A) * B_ATOM * k - (SLOTS_A * B_ATOM * 10 + SLOTS_I),
         (FULL - SLOTS_A) * B_ATOM * k - (SLOTS_A * B_ATOM * 10 + SLOTS_I)),
        ('absGrad2D: length() -> |x|+|y| (sqrt ~4 slots)', FULL * 6, FULL * 6),
        ('accumulate p,q; rebuild dL/dmean2D in preprocess_backward',
         FULL * 6, FULL * 6),
        ('per-lane inner-loop start index', SKIP * 3, SKIP * 5),
        ('stats into splatGrad2D.pad[]: one address, one cache line',
         FULL * 2, FULL * 2),
    ]
    rows.sort(key=lambda r: -r[2])
    print('\n%-58s %14s %13s' % ('candidate', 'slots/iter', 'ms of 11.05'))
    for n, lo, hi in rows:
        print('%-58s %5.2f-%5.2f G %5.2f-%5.2f'
              % (n[:58], lo / 1e9, hi / 1e9, ms(lo), ms(hi)))
    print('\nper-pixel root checks cost %d checks x 4 instr = %.4f G'
          % (PIXELS * 4, PIXELS * 4 * 4 / 1e9))


if __name__ == '__main__':
    main()
