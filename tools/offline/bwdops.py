"""Instruction ledger for trainer_rasterize_backward's inner loop.

The POPULATIONS are measured (bwdwork.py, real model + real poses).
The PER-PAIR INSTRUCTION COUNTS are a hand count of the MSL, given as a
[floor, likely] band, because there is no Metal toolchain on this machine and
so no disassembly.  Every entry says whether the compiler already removes it.

Run:  python -u bwdops.py
"""

# ---------------------------------------------------------------- populations
# bwdwork.py, 720x540, model.ply build 182, refined poses.
POP = {
    # frame 4: 1,059,741 tile instances, matches census peakTileInstances
    # (1,058,989) to 0.07 per cent, i.e. the busiest frame of the run.
    'peak': {
        'tile_instances':   1059741,
        'pairs_in_tiles':   271293696,
        'fwd_iter':         175870241,
        'bwd_iter':         220787408,
        'bwd_skip':          46367104,
        'bwd_power':        174420305,
        'bwd_exp':           75529323,
        'bwd_full':          75089548,
        'slots_iter':         6899607,
        'slots_active':       3350713,
        'lanes_per_slot':        22.41,
    },
    # frames 0/144/288/432/576/720, 40 tiles each: the typical frame
    'typ': {
        'tile_instances':    695416,
        'pairs_in_tiles':   178026368,
        'fwd_iter':         110663034,
        'bwd_iter':         150495255,
        'bwd_skip':          40763270,
        'bwd_power':        109731985,
        'bwd_exp':           49495268,
        'bwd_full':          49223395,
        'slots_iter':         4702977,
        'slots_active':       2100871,
        'lanes_per_slot':        23.43,
    },
}

# ------------------------------------------------- hand count of the MSL body
# (floor, likely) instructions per pair.  SFU ops counted once here and
# weighted separately below.
PREGATE = [
    ('globalIndex = batchBase + j + 1',            1, 2),
    ('if (globalIndex > lastContributor) continue', 2, 2),
    ('tgXY[j] load + delta = tgXY - pixelCenter',   3, 3),
    ('tgConicOpacity[j] load',                      1, 1),
    ('power (dx2, dy2, dxdy, 2 fma, 1 mul)',        6, 6),
    ('tgCutoff[j] load + f16->f32 + cmp + branch',  4, 4),
    ('gaussian = exp(power)  [1 SFU]',              2, 2),
    ('alpha = min(0.99, co.w * gaussian)',          2, 2),
    ('if (alpha < minAlpha) continue',              2, 2),
]

BODY = [
    ('T = T / max(1-alpha,1e-6)  [1 SFU rcp]',      4, 4),
    ('weight = alpha * T',                          1, 1),
    ('tgColorDepth[j] load (xyz and w, one load)',  1, 1),
    ('accumColor = lastAlpha*lastColor + oma*accum',7, 7),
    ('dLdAlpha += dot(color - accumColor, dLdC)',   6, 6),
    ('accumDepth (renderDepth is always 1)',        2, 2),
    ('dLdAlpha += (depth - accumDepth) * dLdD',     2, 2),
    ('dLdAlpha *= T',                               1, 1),
    ('renderDepth branch (compare hoisted by LICM)',1, 1),
    ('dLdAlpha += K * rcpOneMinusAlpha  [K hoisted]',2, 2),
    ('tgIndex[j] load',                             1, 1),
    ('&splatGrad2D[splatIndex]  (shl + add)',       2, 2),
    ('&stats[splatIndex]        (shl + add)',       2, 2),
    ('weight * dLdC.xyz',                           3, 3),
    ('dLdG = co.w * dLdAlpha',                      1, 1),
    ('gaussian * dLdAlpha',                         1, 1),
    ('gdx = -(cx*dx + cy*dy)',                      2, 2),
    ('gdy = -(cz*dy + cy*dx)',                      2, 2),
    ('dLdPower = dLdG * gaussian',                  1, 1),
    ('dLdMean2D = dLdPower * (gdx, gdy)',           2, 2),
    ('conic0/1/2 from live dx2,dy2,dxdy',           4, 4),
    ('length(dLdMean2D)  [1 SFU sqrt]',             3, 4),
    ('if (isUnknown > 0) branch',                   1, 1),
]

GUARD_EACH = (4, 5)     # trainer_finiteOrZero(v) == 0.0f
N_GUARDS = 12
N_ATOMICS = 12          # 11 unconditional + unknownAccum
SFU_WEIGHT = 4          # one special-function op costs ~4 ALU issue slots


def total(rows):
    return sum(r[1] for r in rows), sum(r[2] for r in rows)


def main():
    pf, pl = total(PREGATE)
    bf, bl = total(BODY)
    gf, gl = N_GUARDS * GUARD_EACH[0], N_GUARDS * GUARD_EACH[1]
    print('PER-PAIR HAND COUNT (floor, likely)')
    print('  pre-gate, per pair reaching delta+power     %3d  %3d' % (pf, pl))
    print('  full body arithmetic + threadgroup loads    %3d  %3d' % (bf, bl))
    print('  12 x trainer_finiteOrZero guard             %3d  %3d' % (gf, gl))
    print('  12 x atomic_fetch_add (device, relaxed)     %3d  %3d'
          % (N_ATOMICS, N_ATOMICS))
    print('  FULL-BODY TOTAL                             %3d  %3d'
          % (bf + gf + N_ATOMICS, bl + gl + N_ATOMICS))
    print('  guard as %% of the full-body instruction stream  %.0f%%'
          % (100.0 * gl / (bl + gl + N_ATOMICS)))

    for tag in ('peak', 'typ'):
        P = POP[tag]
        print('\n' + '=' * 74)
        print('%s FRAME  (%d tile instances, %d full-body pairs/iteration)'
              % (tag.upper(), P['tile_instances'], P['bwd_full']))
        print('=' * 74)
        stream = (P['bwd_skip'] * 3
                  + P['bwd_power'] * pl
                  + P['bwd_full'] * (bl + gl + N_ATOMICS))
        print('whole-kernel instruction stream, likely count   %.2f G'
              % (stream / 1e9))

        cands = []
        # 1. collapse the 12 finiteness guards to 2 root checks
        cands.append((
            '12 finiteOrZero guards -> 2 root checks (dLdAlpha, weight)',
            10 * GUARD_EACH[0], 10 * GUARD_EACH[1], P['bwd_full']))
        # 2. length() L2 -> L1 for absGrad2D
        l2 = 3 + SFU_WEIGHT          # 2 mul/fma + 1 add + 1 SFU sqrt
        cands.append((
            'absGrad2D: length() -> |x|+|y|  (SFU weighted %dx)' % SFU_WEIGHT,
            l2 - 1, l2 - 1, P['bwd_full']))
        # 3. accumulate p,q instead of dLdMean2D (needs 2 to have landed)
        cands.append((
            'accumulate p=dLdPower*dx, q=dLdPower*dy; rebuild dL/dmean2D '
            'in preprocess_backward (needs the row above)',
            6, 6, P['bwd_full']))
        # 4. per-lane inner-loop start index
        cands.append((
            'per-lane j start instead of the globalIndex compare',
            3, 5, P['bwd_skip']))
        # 5. fold stats into TrainerSplatGrad2DAtomic.pad[0..2]
        cands.append((
            'stats -> splatGrad2D.pad[0..2]: drop the 2nd address calc',
            2, 2, P['bwd_full']))

        print('\n%-62s %10s' % ('candidate', 'instr/iter'))
        rows = []
        for name, lo, hi, n in cands:
            rows.append((hi * n, lo * n, name, lo, hi, n))
        rows.sort(reverse=True)
        for tot_hi, tot_lo, name, lo, hi, n in rows:
            print('%-62s' % name[:62])
            if len(name) > 62:
                print('  ...%s' % name[62:])
            print('   %2d-%2d instr/pair x %11d pairs = %.3f-%.3f G  (%.1f%% of stream)'
                  % (lo, hi, n, tot_lo / 1e9, tot_hi / 1e9,
                     100.0 * tot_hi / stream))

        print('\nMEMORY-SIDE (not instruction count)')
        now = N_ATOMICS * P['bwd_full']
        red = N_ATOMICS * P['slots_active']
        print('  device atomics now                         %12d' % now)
        print('  with a simd_sum per active (group,j) slot  %12d  (%.1fx)'
              % (red, now / red))
        print('  device atomics REMOVED                     %12d' % (now - red))
        shuffles = N_ATOMICS * 10 * P['slots_active'] + P['slots_iter']
        print('  cost: 12 simd_sum (~10 shuffle+add each) on active slots')
        print('        + 1 simd_any ballot per iterated slot = %.3f G instr'
              % (shuffles / 1e9))
        print('  distinct 64B atomic target lines per pair: 2 -> 1')
        print('  cache lines saved per iteration            %12d'
              % P['bwd_full'])

    print('\nCOMPILER-CSEd ALREADY, WORTH ZERO')
    for s in [
        'max(1.0f - alpha, 1e-6f) written twice (same SSA, no store between)',
        'the two divides by it: fast-math arcp shares one reciprocal',
        'dLdG * dGdPower written three times',
        'dGdPower = gaussian (a register copy)',
        '-TFinal * dLdTTotal: loop-invariant, LICM hoists it',
        'tgColorDepth[j] read as .xyz then .w: one load',
        'delta.x*delta.x, delta.y*delta.y, delta.x*delta.y: live from power',
        'isUnknown > 0.0f: loop-invariant compare, hoisted (branch remains)',
    ]:
        print('  - %s' % s)

    print('\nTHREADGROUP MEMORY, EXACT (this gates how much latency hides)')
    fwd = 256 * (8 + 16 + 16 + 2)
    bwd = fwd + 256 * 4
    print('  forward  tgXY 2048 + tgConicOpacity 4096 + tgColorDepth 4096'
          ' + tgCutoff 512 = %d B' % fwd)
    print('  backward the same + tgIndex 1024                        = %d B' % bwd)
    print('  32768 / %d = %.2f -> %d resident threadgroups (forward)'
          % (fwd, 32768 / fwd, 32768 // fwd))
    print('  32768 / %d = %.2f -> %d resident threadgroups (backward)'
          % (bwd, 32768 / bwd, 32768 // bwd))
    repack = 256 * (8 + 12 + 2 + 6 + 4 + 2 + 4)
    print('  repacked (conic packed_float3 + opacity half + colour'
          ' packed_half3 + depth float) = %d B' % repack)
    print('  32768 / %d = %.2f -> %d resident threadgroups'
          % (repack, 32768 / repack, 32768 // repack))


if __name__ == '__main__':
    main()
