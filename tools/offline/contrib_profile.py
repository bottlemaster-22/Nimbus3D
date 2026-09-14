"""What KIND of splat sits at the bottom of the contribution ranking?

Cross-tabulates contrib.npy (from contrib.py) against each splat's own
geometry: largest axis in mm, shape ratios s2/s1 and s3/s2, raw opacity, and
whether it carries the `densified` flag (created by a split/clone/relocation
rather than surviving from the original seed).

Run: python -u contrib_profile.py
"""
import io
import json
import os

import numpy as np

import project as P
import detail as Dt

SCRATCH = Dt.SCRATCH


def main():
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    contrib = np.load(os.path.join(SCRATCH, 'contrib.npy'))
    assert contrib.shape[0] == count

    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    s_sorted = np.sort(scale, axis=1)[:, ::-1]   # descending: s1 >= s2 >= s3
    largest_mm = s_sorted[:, 0] * 1000
    ratio21 = s_sorted[:, 1] / np.maximum(s_sorted[:, 0], 1e-12)
    ratio32 = s_sorted[:, 2] / np.maximum(s_sorted[:, 1], 1e-12)
    opacity = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))

    order = np.argsort(contrib)
    n = count

    bands = [('bottom 1%', 0, int(0.01 * n)),
             ('bottom 10%', 0, int(0.10 * n)),
             ('bottom 25%', 0, int(0.25 * n)),
             ('25-50%', int(0.25 * n), int(0.50 * n)),
             ('50-75%', int(0.50 * n), int(0.75 * n)),
             ('top 25%', int(0.75 * n), n),
             ('top 1%', int(0.99 * n), n),
             ('ALL', 0, n)]

    print('%-12s %8s %10s %10s %10s %10s %12s'
          % ('band', 'n', 'med.axis', 'med r21', 'med r32', 'med op', 'med contrib'))
    for name, a, b in bands:
        idx = order[a:b] if name != 'ALL' else np.arange(n)
        print('%-12s %8d %10.3f %10.3f %10.3f %10.3f %12.4f'
              % (name, len(idx), np.median(largest_mm[idx]), np.median(ratio21[idx]),
                 np.median(ratio32[idx]), np.median(opacity[idx]), np.median(contrib[idx])))

    print('\nshape buckets (needle: r21<0.3 and r32<0.3; disc: r21>=0.3 and r32<0.3; '
          'blob: r32>=0.3), bottom 25% vs top 25% by contribution:')
    needle = (ratio21 < 0.3) & (ratio32 < 0.3)
    disc = (ratio21 >= 0.3) & (ratio32 < 0.3)
    blob = ratio32 >= 0.3
    for name, a, b in [('bottom 25%', 0, int(0.25 * n)), ('top 25%', int(0.75 * n), n),
                        ('ALL', 0, n)]:
        idx = order[a:b] if name != 'ALL' else np.arange(n)
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        tot = m.sum()
        print('  %-12s needle %5.1f%%  disc %5.1f%%  blob %5.1f%%'
              % (name, 100 * (needle & m).sum() / tot, 100 * (disc & m).sum() / tot,
                 100 * (blob & m).sum() / tot))


if __name__ == '__main__':
    main()
