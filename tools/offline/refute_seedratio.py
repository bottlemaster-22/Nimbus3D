"""Finding 7 claims the SEEDS start at size/spacing = 0.5, because
radius = 0.5 * voxelSize. That equates "voxel size" with "distance to the
nearest other seed". They are not the same number: one representative sample
per occupied cell, placed anywhere inside the cell, gives nearest-neighbour
distances well BELOW the voxel size. This measures the real ratio.
"""
import numpy as np
from scipy.spatial import cKDTree

rs = np.random.RandomState(0)
out = []
for trial in range(8):
    # a random plane through a unit-cell lattice
    n = rs.normal(size=3); n /= np.linalg.norm(n)
    u = np.cross(n, [0, 0, 1.0]);  u /= np.linalg.norm(u)
    w = np.cross(n, u)
    # dense samples on a 60x60 patch of the plane (the depth map is dense)
    a = rs.uniform(-30, 30, 400000); b = rs.uniform(-30, 30, 400000)
    p = a[:, None] * u + b[:, None] * w + rs.uniform(0, 1) * n
    cell = np.floor(p).astype(np.int64)
    key = (cell[:, 0] * 1000003 + cell[:, 1]) * 1000003 + cell[:, 2]
    order = rs.permutation(p.shape[0])          # pick an arbitrary member per cell
    _, first = np.unique(key[order], return_index=True)
    rep = p[order[first]]
    # trim to the interior so the patch edge does not bias nn
    m = (np.abs(a[order[first]]) < 25) & (np.abs(b[order[first]]) < 25)
    tree = cKDTree(rep)
    d, _ = tree.query(rep[m], k=2)
    out.append(np.median(d[:, 1]))
nn = float(np.mean(out))
print('one seed per occupied cell, cell size v = 1.0')
print('  median nearest-neighbour distance = %.4f v  (over %d random plane orientations)'
      % (nn, len(out)))
print('  seed radius = 0.5 v, so size/spacing = %.2f' % (0.5 / nn))
V = 14.793017
print('\nAt this scan (voxelSize %.2f mm):' % V)
print('  seed radius            %.2f mm' % (0.5 * V))
print('  seed nearest neighbour %.2f mm' % (nn * V))
print('  seed size/spacing      %.2f   (finding 7 asserts 0.50; Scaniverse measures 0.74-0.76)'
      % (0.5 / nn))
