"""Donor supply as a function of relocationDonorOpacity, trainer convention,
plus the simulation re-run at the census-measured replenishment rate."""
import numpy as np, json
# NOTE: comp3d stored float32
exec(open('reloc_sim.py').read().split("print('=== BASELINE")[0])
comp = np.load('_reloc_comp3d.npy')
alpha_tr = np.clip((1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))/np.maximum(comp,1e-6),0,0.999999)

print('=== DONOR SUPPLY vs relocationDonorOpacity (trainer convention) ===')
print('INVARIANT: must stay above pruneOpacity 0.02. All rows below satisfy it.')
print('%8s %9s %8s %12s %10s' % ('donorOp', 'donors', '%pop', 'vs 5% cap', 'p50 size mm'))
s1 = np.sort(SIG0, axis=1)[:, 2]*1000
for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    m = alpha_tr < thr
    print('%8.2f %9d %7.2f%% %11s %10.3f'
          % (thr, m.sum(), 100*m.mean(),
             'BINDS' if m.sum() < LIMIT else 'cap binds', np.median(s1[m])))
print()
print('per-pass visAccum<=0 supply, measured from the census: at most 2,551')
print('(splatCountBefore - candidatesAfterVisibilityFilter, stable across all')
print('39 passes). splatsWithNonZeroScore == candidatesAfterVisibilityFilter')
print('EXACTLY in every pass, so the visibility filter itself removes nobody.')
print()
print('=== SIM AT THE CENSUS-MEASURED REPLENISHMENT (2,551/pass) ===')
print('%-40s %6s %7s %7s %8s %8s %7s %7s'
      % ('scenario', 'reloc', 'p10', 'p50', 'p90', 'spread', 's3/s2', 'cover'))
print('%-40s %6d %7.3f %7.3f %8.3f %8.2fx %7.3f %7.2f'
      % ('build 250 as shipped', 0, BASE_ST['p10'], BASE_ST['p50'],
         BASE_ST['p90'], BASE_ST['spread'], BASE_ST['r32'], 1.0))
for thr in [0.05, 0.10, 0.15]:
    DONOR0 = alpha_tr < thr
    globals()['DONOR0'] = DONOR0
    for passes, nm in [(5, 'refine only 5 passes'), (33, 'saturation on, 33 passes')]:
        sig, tot = run(passes, 2551, 'random', seed=7)
        st = stats(sig)
        print('%-40s %6d %7.3f %7.3f %8.3f %8.2fx %7.3f %7.2f'
              % ('donorOp %.2f, %s' % (thr, nm), tot, st['p10'], st['p50'],
                 st['p90'], st['spread'], st['r32'], st['cover']/COVER0))
print()
print('Scaniverse target                        %6s %7.3f %7.3f %8.3f %8.2fx %7.3f'
      % ('-', 1.327, 3.388, 11.825, 8.913, 0.78))
