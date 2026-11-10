"""Measure the size distributions we are trying to explain.

Ours (build 182 export) against Scaniverse on the same room, plus the
per-axis breakdown, because "one size" is a claim about all three axes and
about the ratio between them, not only about the largest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project import load_ply
import numpy as np

OURS = r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840\model\model.ply'
SCAN = r'C:\Users\Undea\Downloads\Four Marks.ply'

def scales(path):
    col, n = load_ply(path)
    s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], axis=1).astype(np.float64)
    lin = np.exp(s)
    return s, lin, n

def report(name, lin):
    big = lin.max(axis=1) * 1000.0
    mid = np.sort(lin, axis=1)[:, 1] * 1000.0
    sml = lin.min(axis=1) * 1000.0
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f'--- {name}  n={len(lin)}')
    for label, v in (('largest', big), ('middle', mid), ('smallest', sml)):
        p = np.percentile(v, qs)
        print('  %-8s ' % label + ' '.join('p%-2d %8.3f' % (q, x) for q, x in zip(qs, p)))
    p10, p50, p90 = np.percentile(big, [10, 50, 90])
    print('  largest-axis spread p90/p10 = %.3fx   p90/p50 = %.3f  p50/p10 = %.3f'
          % (p90 / p10, p90 / p50, p50 / p10))
    lg = np.log(big)
    print('  log10 spread: sd(log largest) = %.4f  (decades p90-p10 = %.4f)'
          % (np.std(lg) / np.log(10), (np.log10(p90) - np.log10(p10))))
    aniso = big / np.maximum(sml, 1e-9)
    print('  anisotropy largest/smallest: p10 %.2f p50 %.2f p90 %.2f'
          % tuple(np.percentile(aniso, [10, 50, 90])))
    return big

if __name__ == '__main__':
    _, lo, _ = scales(OURS)
    _, ls, _ = scales(SCAN)
    a = report('OURS build 182', lo)
    b = report('SCANIVERSE', ls)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ours_largest_mm.npy'), a)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan_largest_mm.npy'), b)
    # What fraction of each population sits in each decade of size.
    print('\nsize decades (largest axis, mm), share of population')
    edges = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 1e9]
    lo_h = np.histogram(a, bins=edges)[0] / len(a)
    ls_h = np.histogram(b, bins=edges)[0] / len(b)
    for i in range(len(edges) - 1):
        print('  %7.2f - %-8.2f  ours %6.2f%%   scaniverse %6.2f%%'
              % (edges[i], edges[i + 1], 100 * lo_h[i], 100 * ls_h[i]))
