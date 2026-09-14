"""Refutation test for 'the spacing floor wins for 100% of seeds'.

seed_dist_measure.py used distance to the NEAREST of all 868 refined camera
centres. The pre-pass seeds from 174 selected keyframes, and rangeMeters[i] is
the depth-sample range from the keyframe that GENERATED the sample, which is
not in general the nearest camera. Nearest-of-868 is therefore a LOWER BOUND on
the true generating range, and a lower bound is exactly the wrong side to be on
when the question is 'does anything exceed 2.726 m'.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project import load_ply

BASE = r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840'
pp = json.load(open(os.path.join(BASE, 'prepass', 'prepass_result.json')))
cb = json.load(open(os.path.join(BASE, 'capture_bundle.json')))

from project import quat_to_matrix
poses = pp['refinedPoses']
keys = sorted(poses.keys(), key=lambda k: int(k))
C = []
for k in keys:
    p_ = poses[k]
    R = quat_to_matrix(p_['rotation'])
    t = np.asarray(p_['translation'], float)
    C.append(-R.T @ t)
C = np.array(C)
print('cameras', C.shape, 'span', np.ptp(C, axis=0))

col, n = load_ply(os.path.join(BASE, 'model', 'model.ply'))
P = np.stack([col['x'], col['y'], col['z']], axis=1).astype(np.float64)
P[:, 1] *= -1.0; P[:, 2] *= -1.0    # RDF -> RUB
print('splats', P.shape, 'span', np.ptp(P, axis=0))

fx_native = cb['intrinsics']['fx']
W = cb['intrinsics'].get('width', 1920)
dW = cb.get('settings', {}).get('depthWidth', 256)
fx_depth = fx_native * dW / W
spacing = json.load(open(os.path.join(BASE, 'prepass', 'census.json')))['seeding']['spacingMeters']
print('fx native %.3f  fx depth %.3f  spacing %.6f m' % (fx_native, fx_depth, spacing))

def crossover(fx):
    return spacing * fx

def nearest_dist(cams, pts, chunk=4000):
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        d = np.linalg.norm(pts[i:i+chunk, None, :] - cams[None, :, :], axis=2)
        out[i:i+chunk] = d.min(axis=1)
    return out

sub = P[::10]     # 29,921 splats, enough for percentiles
for label, cams in (('all 868', C), ('every 5th (174, mimics keyframe pick)', C[::5])):
    d = nearest_dist(cams, sub)
    print('\n-- nearest-camera range, %s' % label)
    print('   p50 %.3f p90 %.3f p99 %.3f p99.9 %.3f MAX %.3f m'
          % tuple(np.percentile(d, [50, 90, 99, 99.9]).tolist() + [d.max()]))
    for fx, nm in ((fx_depth, 'depth fx'), (fx_native, 'native fx')):
        x = crossover(fx)
        print('   crossover (%s, %.1f) = %.3f m -> share of splats beyond it: %.3f%%'
              % (nm, fx, x, 100.0 * (d > x).mean()))

# What the seed radius would have been, per splat, if range = nearest-of-174.
d174 = nearest_dist(C[::5], sub)
for fx, nm in ((fx_depth, 'depth fx'), (fx_native, 'native fx')):
    r = 0.5 * np.maximum(spacing, d174 / fx) * 1000.0
    p10, p50, p90 = np.percentile(r, [10, 50, 90])
    print('\nSEED radius under %s: p10 %.3f p50 %.3f p90 %.3f p99 %.3f mm  spread %.4fx  sd %.4f dec'
          % (nm, p10, p50, p90, np.percentile(r, 99), p90 / p10, np.std(np.log10(r))))
