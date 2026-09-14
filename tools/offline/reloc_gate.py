"""DIMENSION: resurrect relocation.

Q1  walk the 39 densify passes and show exactly why the else-if never fires
Q3  the donor pool, from the exported PLY, in BOTH opacity conventions
"""
import json, numpy as np

BASE = 'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840'
CAP = 300000
MAXGROW = 0.15
MAXRELOC = 0.05
PRUNE_OP = 0.02
DONOR_OP = 0.05

d = json.load(open(BASE + '/model/train_census.json', encoding='utf-8'))
P = d['densifyPasses']

print('=== Q1: THE GATE, PASS BY PASS ===')
print('growthAllowance = allowGrowth ? min(headroom, 0.15*cap=45000) : 0')
print('relocationLimit = (allowGrowth && growthAllowance==0) ? 0.05*count : 0')
print()
print('%3s %5s %6s %8s %7s %7s %7s %6s %6s %5s  %s'
      % ('#','iter','win','before','headrm','grAllw','added','prLow','carve','reloc','why relocation could not run'))
nz = 0
for i, p in enumerate(P):
    win = p['growthWindowOpen']
    hr = p['headroom']
    ga = p['growthAllowance']
    added = p['addedBySplit'] + p['addedByClone']
    if not win:
        why = 'allowGrowth false -> relocationLimit forced 0 by the `allowGrowth &&`'
    elif ga > 0:
        why = 'growthAllowance %d > 0 -> `if` branch taken, `else if` unreachable' % ga
    else:
        why = 'RELOCATION PATH REACHED'
        nz += 1
    print('%3d %5d %6s %8d %7d %7d %7d %6d %6d %5d  %s'
          % (i, p['iteration'], 'open' if win else 'shut', p['splatCountBefore'],
             hr, ga, added, p['prunedLowOpacity'], p['carvedFromEmptySpace'],
             p['relocated'], why))
print()
print('passes that reached the relocation branch : %d of %d' % (nz, len(P)))
print('total relocated                           : %d'
      % sum(p['relocated'] for p in P))
print('total relocationDonorsAvailable recorded  : %d  (the counter is only'
      % sum(p['relocationDonorsAvailable'] for p in P))
print('   incremented INSIDE the relocation branch, so a zero here is "never')
print('   measured", NOT "no donors existed". The census cannot answer Q3.')
print()

# the identity that pins the population
print('=== THE STEADY STATE, verified arithmetically ===')
print('For every pass from #6 on: added == headroom exactly, and')
print('headroom(next) == prunedLowOpacity + carved of this pass.')
bad = 0
for i in range(6, len(P) - 1):
    p, q = P[i], P[i + 1]
    added = p['addedBySplit'] + p['addedByClone']
    lost = p['prunedLowOpacity'] + p['carvedFromEmptySpace'] + p['prunedNonFinite'] + p['trimmedToCap']
    predicted_hr = lost if p['growthWindowOpen'] else q['headroom']
    ok_add = (added == p['growthAllowance'])
    ok_hr = (q['headroom'] == lost) if p['growthWindowOpen'] else True
    if not (ok_add and ok_hr):
        bad += 1
        print('  MISMATCH pass %d: added %d allow %d | next headroom %d lost %d'
              % (i, added, p['growthAllowance'], q['headroom'], lost))
print('  mismatches: %d' % bad)
print()

# ---------------------------------------------------------------- Q3 --------
print('=== Q3: THE DONOR POOL, from model.ply ===')
f = open(BASE + '/model/model.ply', 'rb')
h = b''
while not h.endswith(b'end_header\n'):
    h += f.read(1)
names, n = [], 0
for line in h.decode().split('\n'):
    q = line.split()
    if not q:
        continue
    if q[0] == 'element' and q[1] == 'vertex':
        n = int(q[2])
    elif q[0] == 'property' and q[1] == 'float':
        names.append(q[2])
arr = np.frombuffer(f.read(n * len(names) * 4), dtype='<f4').reshape(n, len(names))
col = {nm: arr[:, i] for i, nm in enumerate(names)}
logit = col['opacity'].astype(np.float64)
alpha = 1.0 / (1.0 + np.exp(-logit))
print('splats read: %d' % n)
print()
print('THE PLY OPACITY IS THE *FUSED* ONE. MetalSplatTrainer line 3153 calls')
print('SplatCloud.fuse3DFilter before export, and SplatMath.fusing3DFilter')
print('writes invSigmoid(sigmoid(trainerLogit) * comp3D). So')
print('   sigmoid(ply.opacity) == drawnOpacity  (what the PRUNE tests)')
print('   sigmoid(trainer opacityLogit) == drawnOpacity / comp3D  (what the')
print('   DONOR test uses) -- and comp3D <= 1, so the trainer-side opacity is')
print('   >= the PLY one. comp3D is per-splat and is NOT in the PLY, so the')
print('   trainer-side value is a LOWER BOUND from this file alone.')
print()
qs = [0.1, 1, 5, 10, 25, 50, 75, 90]
print('drawn-alpha percentiles: ' + '  '.join(
    'p%g %.4f' % (q, np.percentile(alpha, q)) for q in qs))
print()
below_prune = (alpha < PRUNE_OP).sum()
band = ((alpha >= PRUNE_OP) & (alpha < DONOR_OP)).sum()
below_donor = (alpha < DONOR_OP).sum()
print('drawn alpha < pruneOpacity 0.02          : %7d  (%.3f%%)' % (below_prune, 100*below_prune/n))
print('drawn alpha in [0.02, 0.05)  THE BAND    : %7d  (%.3f%%)' % (band, 100*band/n))
print('drawn alpha < relocationDonorOpacity 0.05: %7d  (%.3f%%)' % (below_donor, 100*below_donor/n))
print()
allowance = int(n * MAXRELOC)
print('maxRelocationFractionPerPass allowance   : %7d  (5%% of %d)' % (allowance, n))
print()
print('build 182 had 22,005 in the band (7.35%%) against a 5%% allowance.')
print()
print('visAccum: NOT recoverable from the PLY. The PLY carries only x y z,')
print('nx ny nz (written as zero), f_dc, f_rest, opacity, scale, rot')
print('(PLYCodec.swift lines 19-31). visAccum lives in TrainerSplatStats on')
print('the GPU and is never exported, and the census only sums it inside the')
print('dead branch. The OTHER half of the donor test is therefore unmeasured')
print('from these artefacts.')
