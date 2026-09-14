"""Q4: what relocation buys, simulated on the REAL exported model.

One relocation, exactly as TrainerDensifier writes it:
  target  <- splitGeometry(preserveCoverage: true): the argmax linear axis is
             divided by splitShrink 1.6, centre offset +-0.8*that axis
  donor   <- a COPY of that same shrunk geometry (its own geometry is gone)
  both    <- opacity 1 - sqrt(1 - o_target)
So one relocation shrinks one target axis AND deletes one donor-sized splat.
"""
import json, numpy as np

BASE = 'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840'
SHRINK = 1.6
RELOC_FRAC = 0.05

f = open(BASE + '/model/model.ply', 'rb'); h = b''
while not h.endswith(b'end_header\n'): h += f.read(1)
names, n = [], 0
for line in h.decode().split('\n'):
    q = line.split()
    if not q: continue
    if q[0] == 'element' and q[1] == 'vertex': n = int(q[2])
    elif q[0] == 'property' and q[1] == 'float': names.append(q[2])
arr = np.frombuffer(f.read(n*len(names)*4), dtype='<f4').reshape(n, len(names))
col = {nm: arr[:, i] for i, nm in enumerate(names)}
SIG0 = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                               axis=1).astype(np.float64), -12, 3))
DONOR0 = np.load('_reloc_donor_mask.npy')      # trainer-convention alpha<0.05
LIMIT = int(n * RELOC_FRAC)

def stats(sig):
    s = np.sort(sig, axis=1)
    lg = s[:, 2]*1000
    p10, p50, p90 = np.percentile(lg, [10, 50, 90])
    return dict(p10=p10, p50=p50, p90=p90, spread=p90/p10,
                aspect=np.median(s[:, 2]/np.maximum(s[:, 0], 1e-12)),
                r32=np.median(s[:, 0]/np.maximum(s[:, 1], 1e-12)),
                r21=np.median(s[:, 1]/np.maximum(s[:, 2], 1e-12)),
                cover=(s[:, 2]*s[:, 1]).sum())

BASE_ST = stats(SIG0)
COVER0 = BASE_ST['cover']

def run(passes, replen, policy, seed=0):
    rng = np.random.default_rng(seed)
    sig = SIG0.copy()
    donor = DONOR0.copy()
    pool = int(donor.sum())
    total = 0
    for _ in range(passes):
        avail = np.flatnonzero(donor)
        k = min(len(avail), LIMIT)
        if k <= 0:
            # replenish and carry on
            pass
        else:
            d = rng.choice(avail, size=k, replace=False)
            live = np.flatnonzero(~donor)
            if policy == 'largest':
                lg = np.sort(sig[live], axis=1)[:, 2]
                tgt = live[np.argsort(-lg)[:k]]
            else:
                tgt = rng.choice(live, size=k, replace=False)
            ax = np.argmax(sig[tgt], axis=1)
            sig[tgt, ax] /= SHRINK
            sig[d] = sig[tgt]           # donor slot becomes a copy
            donor[d] = False            # opacity corrected -> no longer faint
            total += k
        # replenishment: new splats drift into the faint band
        idle = np.flatnonzero(~donor)
        add = min(replen, len(idle))
        if add > 0:
            donor[rng.choice(idle, size=add, replace=False)] = True
    return sig, total

print('=== BASELINE, the exported model ===')
print('p10 %.3f  p50 %.3f  p90 %.3f  spread %.2fx  aspect %.1f:1  s3/s2 %.3f'
      % (BASE_ST['p10'], BASE_ST['p50'], BASE_ST['p90'], BASE_ST['spread'],
         BASE_ST['aspect'], BASE_ST['r32']))
print('Scaniverse, same room: p50 3.39  spread 8.9x  aspect 1.9:1  s3/s2 0.78')
print()
print('donor pool at export : %d (%.2f%%)   per-pass cap 5%% = %d'
      % (DONOR0.sum(), 100*DONOR0.mean(), LIMIT))
print('=> DONORS, not the allowance, are the binding limit: %d vs %d.'
      % (DONOR0.sum(), LIMIT))
print()
print('Replenishment is the one number no artefact here measures. Build 182\'s')
print('census (tools/offline/refute_donor_supply.py) measured the pool')
print('standing at 1,142 / 1,235 / 1,496 across passes where relocation did')
print('NOT run, so ~1,100-1,500 a pass is the only anchor there is. Rows below')
print('bracket it with 0 and 3,000.')
print()
hdr = ('%-34s %6s %7s %7s %8s %8s %7s %7s %7s'
       % ('scenario', 'reloc', 'p10', 'p50', 'p90', 'spread', 'asp', 's3/s2', 'cover'))
print(hdr); print('-'*len(hdr))
print('%-34s %6d %7.3f %7.3f %8.3f %8.2fx %7.1f %7.3f %7.2f'
      % ('build 250 as shipped (0 reloc)', 0, BASE_ST['p10'], BASE_ST['p50'],
         BASE_ST['p90'], BASE_ST['spread'], BASE_ST['aspect'], BASE_ST['r32'], 1.0))
for label, passes in [('refine phase only, 5 passes', 5),
                      ('from saturation, 33 passes', 33)]:
    for replen in [0, 1300, 3000]:
        for policy in ['random', 'largest']:
            sig, tot = run(passes, replen, policy)
            st = stats(sig)
            print('%-34s %6d %7.3f %7.3f %8.3f %8.2fx %7.1f %7.3f %7.2f'
                  % ('%s, refill %d, %s' % (label.split(',')[0], replen, policy[:4]),
                     tot, st['p10'], st['p50'], st['p90'], st['spread'],
                     st['aspect'], st['r32'], st['cover']/COVER0))

print()
print('=== HOW MANY RELOCATIONS WOULD IT TAKE TO REACH SCANIVERSE p50? ===')
print('Donors unbounded, targets uniform among the live, 5%% of the population')
print('per pass. This is the ceiling relocation could ever reach, not a plan.')
rng = np.random.default_rng(1)
sig = SIG0.copy()
tot = 0
print('%6s %9s %8s %8s %9s %8s %8s' % ('pass', 'reloc', 'p10', 'p50', 'p90', 'spread', 'cover'))
for p in range(1, 61):
    d = rng.choice(n, size=LIMIT, replace=False)
    live = np.setdiff1d(np.arange(n), d, assume_unique=False)
    tgt = rng.choice(live, size=LIMIT, replace=False)
    ax = np.argmax(sig[tgt], axis=1)
    sig[tgt, ax] /= SHRINK
    sig[d] = sig[tgt]
    tot += LIMIT
    if p in (1, 5, 10, 20, 33, 45, 60):
        st = stats(sig)
        print('%6d %9d %8.3f %8.3f %9.3f %8.2fx %8.2f'
              % (p, tot, st['p10'], st['p50'], st['p90'], st['spread'],
                 st['cover']/COVER0))
        if st['p50'] <= 3.39:
            print('  -> reached Scaniverse p50 after %d relocations' % tot)
            break
