"""Does the QUALITY BAR allocate size by local detail? Registration-free.

Scaniverse's capture of the same room cannot be projected into our frames
without a registration nobody has done, so this asks the question INSIDE each
model, where no registration is needed:

  * is a splat's size tied to how far away its neighbours are (adaptive
    density), or is the cloud one density everywhere (uniform spacing)?
  * is a splat's size tied to how much the COLOUR varies among its
    neighbours, which is the only detail signal a splat cloud carries?

Ours is the control: it should show neither, because clone copies the parent.
"""
import numpy as np
from scipy.spatial import cKDTree
import sv_load as S

C0 = 0.28209479177387814


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def look(tag, col, n, sample=120000, k=9):
    xyz = np.stack([col['x'], col['y'], col['z']], 1).astype(np.float64)
    sc = np.exp(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1)
                .astype(np.float64))
    big = sc.max(axis=1)
    rgb = 0.5 + C0 * np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1).astype(np.float64)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    op = 1 / (1 + np.exp(-col['opacity'].astype(np.float64)))

    finite = np.isfinite(xyz).all(1) & np.isfinite(big)
    xyz, big, lum, op = xyz[finite], big[finite], lum[finite], op[finite]
    tree = cKDTree(xyz)
    rs = np.random.RandomState(0)
    idx = rs.choice(xyz.shape[0], size=min(sample, xyz.shape[0]), replace=False)
    d, j = tree.query(xyz[idx], k=k)
    nn = d[:, 1]                                   # nearest other splat, metres
    spread = lum[j[:, 1:]].std(axis=1)             # colour variation nearby

    print('\n=== %s: %d splats' % (tag, xyz.shape[0]))
    print('  largest axis mm      p10 %6.2f  p50 %6.2f  p90 %6.2f   p90/p10 %.1fx'
          % (1000 * np.percentile(big, 10), 1000 * np.percentile(big, 50),
             1000 * np.percentile(big, 90),
             np.percentile(big, 90) / max(np.percentile(big, 10), 1e-9)))
    print('  nearest neighbour mm p10 %6.2f  p50 %6.2f  p90 %6.2f   p90/p10 %.1fx'
          % (1000 * np.percentile(nn, 10), 1000 * np.percentile(nn, 50),
             1000 * np.percentile(nn, 90),
             np.percentile(nn, 90) / max(np.percentile(nn, 10), 1e-9)))
    print('  size / spacing       p10 %6.2f  p50 %6.2f  p90 %6.2f'
          % tuple(np.percentile(big[idx] / np.maximum(nn, 1e-9), q) for q in (10, 50, 90)))
    print('  Spearman(size, neighbour distance)  %+.3f' % spearman(big[idx], nn))
    print('  Spearman(size, neighbour colour sd) %+.3f' % spearman(big[idx], spread))
    print('  Spearman(neighbour distance, neighbour colour sd) %+.3f' % spearman(nn, spread))

    q = np.argsort(spread)
    e = np.linspace(0, q.size, 6).astype(int)
    print('  colour-variation quintile -> median size mm / median spacing mm')
    for a in range(5):
        s = q[e[a]:e[a + 1]]
        print('     %d  sd %.4f   size %6.2f   spacing %6.2f'
              % (a + 1, np.median(spread[s]), 1000 * np.median(big[idx][s]),
                 1000 * np.median(nn[s])))


col, n = S.sv()
look('SCANIVERSE  Four Marks.ply', col, n)
col, n = S.ours()
look('OURS        build 182 model.ply', col, n)
