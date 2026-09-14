"""Decompose the 2.96x seed overshoot into causes that are MEASURABLE from the
data on disk, without invoking any pose-smear term.
"""
import json, math, os
import numpy as np
from scipy.spatial import cKDTree
from project import load_ply, D

cen = json.load(open(os.path.join(D, 'prepass', 'census.json')))
area   = cen['seeding']['measuredSurfaceAreaSquareMeters']
h_seed = cen['seeding']['spacingMeters']
built  = cen['seeding']['gaussiansBuilt']
target = cen['seeding']['targetSplatCount']
survey_cells = area / 0.10**2

col, n = load_ply(os.path.join(D, 'model', 'model.ply'))
P = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)

def cells(h, pts):
    k = np.floor(pts / h).astype(np.int64); k -= k.min(axis=0)
    m = k.max(axis=0) + 1
    return np.unique((k[:,0]*m[1] + k[:,1])*m[2] + k[:,2]).size

hs = np.array([0.20, 0.15, 0.10, 0.07, 0.05])
Ns = np.array([cells(h, P) for h in hs], dtype=float)
# least squares fit log N = c - D log h, only where pts/cell >= 6 (unsaturated)
A = np.vstack([np.ones_like(hs), -np.log(hs)]).T
c, Dfit = np.linalg.lstsq(A, np.log(Ns), rcond=None)[0]
print('box-counting fit over 0.20..0.05 m (pts/cell 165 -> 6.6, unsaturated)')
for h, N in zip(hs, Ns):
    print(f'   h={h:.3f}  N={N:7.0f}  fit={math.exp(c)*h**-Dfit:9.0f}')
print(f'   D = {Dfit:.3f}   (a flat 2D manifold would give exactly 2.000)')

r = 0.10 / h_seed
print()
print('=== decomposition of built/target = %.3f ===' % (built/target))
f_cov = cells(0.10, P) / survey_cells
f_dim = r**(Dfit - 2.0)
print(f'  A. survey UNDERCOUNTS surface at its own 10 cm scale')
print(f'     model occupies {cells(0.10,P)} cells at 10 cm; survey area implies {survey_cells:.0f}')
print(f'     factor = {f_cov:.3f}')
print(f'     cause: PrePassSurveyor.Settings maxKeyframes=48 (seeding uses 174),')
print(f'            raySubsampleStride=4 (1/16 of samples): 147,456 rays vs 8,538,104')
print(f'  B. the scene is not a flat 2D manifold: D={Dfit:.3f}, so cell count')
print(f'     grows as h^-{Dfit:.3f} from 10 cm to {h_seed*100:.3f} cm')
print(f'     factor = ({r:.3f})^({Dfit:.3f}-2) = {f_dim:.3f}')
print(f'  A x B = {f_cov*f_dim:.3f}   of the observed {built/target:.3f}')
print(f'  UNEXPLAINED REMAINDER = {(built/target)/(f_cov*f_dim):.3f}x')
print(f'  (the finding attributes ALL {built/target:.3f}x to a 43.8 mm pose smear)')
print()
print(f'  predicted seeds from A x B alone: {target*f_cov*f_dim:,.0f}  (actual {built:,})')

print()
print('=== local thickness of the trained model (PCA on 32-NN) ===')
rng = np.random.default_rng(0)
idx = rng.choice(len(P), 20000, replace=False)
tree = cKDTree(P)
d, nb = tree.query(P[idx], k=32, workers=-1)
Q = P[nb]                       # 20000 x 32 x 3
Q = Q - Q.mean(axis=1, keepdims=True)
cov = np.einsum('nki,nkj->nij', Q, Q) / 32.0
ev = np.linalg.eigvalsh(cov)    # ascending
thick = np.sqrt(np.maximum(ev[:, 0], 0))
inplane = np.sqrt(np.maximum(ev[:, 2], 0))
print(f'  median off-plane RMS   {np.median(thick)*1000:7.2f} mm')
print(f'  p90 off-plane RMS      {np.percentile(thick,90)*1000:7.2f} mm')
print(f'  median in-plane RMS    {np.median(inplane)*1000:7.2f} mm')
print(f'  a 43.8 mm uniform band would give off-plane RMS ~ {43.834/math.sqrt(12):.2f} mm')
print(f'  a 9.33 mm (measured) 1-sigma noise gives off-plane RMS ~ 9.33 mm')

SCAN = r'C:\Users\Undea\Downloads\Four Marks.ply'
if os.path.exists(SCAN):
    sc, sn = load_ply(SCAN)
    SP = np.stack([sc['x'], -sc['y'], -sc['z']], axis=1).astype(np.float64)
    sidx = rng.choice(len(SP), 20000, replace=False)
    st = cKDTree(SP)
    _, snb = st.query(SP[sidx], k=32, workers=-1)
    SQ = SP[snb]; SQ = SQ - SQ.mean(axis=1, keepdims=True)
    scov = np.einsum('nki,nkj->nij', SQ, SQ) / 32.0
    sev = np.linalg.eigvalsh(scov)
    print(f'  Scaniverse same room: median off-plane RMS '
          f'{np.median(np.sqrt(np.maximum(sev[:,0],0)))*1000:.2f} mm  (n={sn})')
