"""Box-counting on the REAL trained model, to test whether the 2.96x seed
overshoot needs a pose-smear explanation at all.

The pitch is set from an area measured on a 10 cm grid by PrePassSurveyor
(48 keyframes, ray stride 4).  The seeding then voxelises at 1.479 cm using
174 keyframes and EVERY sample.  If occupied-cell count does not scale as
h^-2 between those two scales, part or all of the overshoot is geometry and
sampling, not pose error.
"""
import json, math, os
import numpy as np
from project import load_ply, D

col, n = load_ply(os.path.join(D, 'model', 'model.ply'))
# RDF -> RUB (the trap): negate y and z on position.
P = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
print(f'model splats {n}')

cen = json.load(open(os.path.join(D, 'prepass', 'census.json')))
res = json.load(open(os.path.join(D, 'prepass', 'prepass_result.json')))
area = cen['seeding']['measuredSurfaceAreaSquareMeters']
h_seed = cen['seeding']['spacingMeters']
survey_cells = area / 0.10**2

def cells(h, pts):
    k = np.floor(pts / h).astype(np.int64)
    # pack to a single int64 key
    k -= k.min(axis=0)
    m = k.max(axis=0) + 1
    key = (k[:, 0] * m[1] + k[:, 1]) * m[2] + k[:, 2]
    return np.unique(key).size

print()
print(f'{"h (m)":>9} {"occupied cells":>15} {"pts/cell":>9}')
hs = [0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.035, 0.025, 0.0147930]
N = {}
for h in hs:
    N[h] = cells(h, P)
    print(f'{h:9.4f} {N[h]:15d} {n/N[h]:9.2f}')

print()
print('--- what the survey claims vs what the model occupies, both at 10 cm ---')
print(f'survey surfaceCellCount (implied from area/0.10^2) : {survey_cells:.0f}')
print(f'model occupied 10 cm cells                          : {N[0.10]}')
print(f'ratio model/survey                                  : {N[0.10]/survey_cells:.3f}')

print()
print('--- local slope d log N / d log(1/h), pairwise ---')
for a, b in zip(hs[:-1], hs[1:]):
    Dm = math.log(N[b]/N[a]) / math.log(a/b)
    print(f'  {a:7.4f} -> {b:7.4f} : D = {Dm:5.3f}  ({"saturating" if n/N[b] < 3 else ""})')

print()
print('--- extrapolate the 10cm survey area to the 1.479cm seeding grid ---')
r = 0.10 / h_seed
for Dm in (2.0, 2.1, 2.2, 2.3, 2.4):
    print(f'  if D={Dm:.1f}: cells at h_seed = {survey_cells * r**Dm:,.0f}  '
          f'(built = 888,951, target = 300,000)')
Dneed = math.log(888951/survey_cells)/math.log(r)
print(f'  D that would produce exactly 888,951 from the survey cells: {Dneed:.3f}')
