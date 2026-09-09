"""Null test on my own parallax number.

A LiDAR ray does not return at grazing incidence and a disc seen edge-on
carries no photometric depth information, but my z-buffer visibility test
happily counts both. Gate visibility on the angle between the view ray and the
splat's own surface normal (the smallest axis of its covariance) and see what
happens to the triangulation angle. If it collapses toward the QC card's
10.28 deg, the QC card is right and my number was the inflated one.
"""
import io, json, os
import numpy as np
import project as P
from cap_keyframes import keyframes

OUT = os.path.dirname(os.path.abspath(__file__))
rays = np.load(os.path.join(OUT, '_cap_rays_occ.npy'))   # S x 120 x 3, splat->camera? NO: camera->splat
samp = np.load(os.path.join(OUT, '_cap_samp.npy'))
S, V, _ = rays.shape
seen = np.abs(rays).sum(axis=2) > 0

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                axis=1).astype(np.float64), -12, 3))[samp]
q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], axis=1).astype(np.float64)[samp]
q = q/np.linalg.norm(q, axis=1, keepdims=True)
x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
Rm = np.empty((S, 3, 3))
Rm[:, 0, 0] = 1-2*(y_*y_+z_*z_); Rm[:, 0, 1] = 2*(x_*y_-w_*z_); Rm[:, 0, 2] = 2*(x_*z_+w_*y_)
Rm[:, 1, 0] = 2*(x_*y_+w_*z_); Rm[:, 1, 1] = 1-2*(x_*x_+z_*z_); Rm[:, 1, 2] = 2*(y_*z_-w_*x_)
Rm[:, 2, 0] = 2*(x_*z_-w_*y_); Rm[:, 2, 1] = 2*(y_*z_+w_*x_); Rm[:, 2, 2] = 1-2*(x_*x_+y_*y_)
# surface normal = column of R for the SMALLEST scale axis
smallest = np.argmin(scale, axis=1)
N = Rm[np.arange(S), :, smallest]
N /= np.linalg.norm(N, axis=1, keepdims=True)

# incidence: |cos| between the view ray and the normal. rays are camera->splat
# unit vectors, so the ray hits the surface at cos = |dot(ray, normal)|.
cosinc = np.abs(np.einsum('svc,sc->sv', rays.astype(np.float64), N))
print('incidence cos over all seen (splat, view) pairs:')
cs = cosinc[seen]
for qn in (5, 10, 25, 50, 75, 90):
    print('  p%-3d cos %.3f  = %5.1f deg from normal' % (qn, np.percentile(cs, qn),
                                                          np.degrees(np.arccos(np.percentile(cs, qn)))))
print('  pairs at more than 60 deg off normal (cos<0.5): %.1f%%' % (100*(cs < 0.5).mean()))
print('  pairs at more than 75 deg off normal (cos<0.26): %.1f%%' % (100*(cs < 0.2588).mean()))


def stats(mask, label):
    nv = mask.sum(axis=1)
    mx = np.full(S, np.nan); md = np.full(S, np.nan)
    for i in range(S):
        m = mask[i]
        if m.sum() < 2:
            continue
        Rv = rays[i, m].astype(np.float64)
        a = np.degrees(np.arccos(np.clip(Rv @ Rv.T, -1, 1)))
        iu = np.triu_indices(len(Rv), 1)
        mx[i] = a[iu].max(); md[i] = np.median(a[iu])
    ok = ~np.isnan(mx)
    print('%-34s views p50 %4.1f | max-pair p25 %5.1f MED %5.1f p75 %5.1f | med-pair MED %5.1f '
          '| <2 views %5.2f%%'
          % (label, np.median(nv), *np.percentile(mx[ok], [25, 50, 75]),
             np.median(md[ok]), 100*(nv < 2).mean()))
    return mx, md, nv


print()
stats(seen, 'no incidence gate')
for c in (0.26, 0.50, 0.71, 0.87):
    stats(seen & (cosinc >= c), 'incidence within %2.0f deg of normal' % np.degrees(np.arccos(c)))
