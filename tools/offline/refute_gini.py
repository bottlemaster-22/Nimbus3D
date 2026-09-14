"""Finding 8 quotes a 5 cm-voxel Gini of 0.753 (theirs) vs 0.434 (ours) from a
'dedicated voxel run' that is not in any saved script. Recompute it, and then
test whether the gap survives a SYMMETRIC treatment -- the finding restricted
Scaniverse to a 3 m ball and left ours unrestricted, and Gini of voxel
occupancy is not scale-free: it moves with the volume included AND with the
population's own nearest-neighbour spacing, which differs 2.6x between the
two models."""
import numpy as np, sv_load
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"

def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    return float((2*np.arange(1, n+1) - n - 1) @ x / (n * x.sum()))

def report(tag, P, vox):
    q = np.floor(P/vox).astype(np.int64)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    order = np.sort(cnt)[::-1]
    tot = cnt.sum(); cum = np.cumsum(order)/tot
    def top(f):
        k = max(1, int(round(f*len(order))))
        return 100*cum[k-1]
    print('  %-34s vox %.2fm  n=%7d  occupied %6d  densest 1/5/10/25%% hold %5.1f/%5.1f/%5.1f/%5.1f%%  Gini %.3f'
          % (tag, vox, tot, len(cnt), top(.01), top(.05), top(.10), top(.25), gini(cnt)))

c,_ = sv_load.sv(); sent = np.load(SCR+r"\sv_sentinel.npy")
Psv = np.stack([c['x'],c['y'],c['z']],1).astype(np.float64)[~sent]
o,_ = sv_load.ours()
Pus = np.stack([o['x'],-o['y'],-o['z']],1).astype(np.float64)

ctr_sv = np.array([-1.26375,-0.24857,0.39607])
ctr_us = np.median(Pus,0)

print("A. THE FINDING'S OWN ASYMMETRIC SETUP (theirs r<3m, ours unrestricted)")
for v in (0.05, 0.02):
    report('SCANIVERSE r<3m', Psv[np.linalg.norm(Psv-ctr_sv,axis=1)<3.0], v)
    report('OURS unrestricted', Pus, v)

print()
print("B. SYMMETRIC: both clipped to the same radius about their own centre")
for r in (3.0, 5.0):
    for v in (0.05,):
        report('SCANIVERSE r<%.0fm'%r, Psv[np.linalg.norm(Psv-ctr_sv,axis=1)<r], v)
        report('OURS r<%.0fm'%r,       Pus[np.linalg.norm(Pus-ctr_us,axis=1)<r], v)

print()
print("C. THE CONFOUND: Gini of voxel occupancy is not independent of the")
print("   population's own spacing. Subsample Scaniverse down to OUR count and")
print("   to our nearest-neighbour spacing, and see how much of the gap is left.")
rng = np.random.default_rng(0)
m = np.linalg.norm(Psv-ctr_sv,axis=1)<3.0
Psv3 = Psv[m]
report('SCANIVERSE r<3m, full', Psv3, 0.05)
for n in (299209, 150000):
    idx = rng.choice(len(Psv3), size=min(n,len(Psv3)), replace=False)
    report('SCANIVERSE r<3m, random %d'%n, Psv3[idx], 0.05)
print()
print("D. voxel size sweep, symmetric r<3m, to show how much the number moves")
for v in (0.20, 0.10, 0.05, 0.02, 0.01):
    report('SCANIVERSE r<3m', Psv3, v)
    report('OURS r<3m',       Pus[np.linalg.norm(Pus-ctr_us,axis=1)<3.0], v)
