"""HOW FAST does the error a splat cannot avoid fall when the splat shrinks?

This is the crux of the whole "our splats are too big" theory. If footprint
variance falls steeply with radius, shrinking pays. If it saturates - because
the content is broadband texture, or because the photograph's own noise floor
has been reached - shrinking buys nothing and the budget is better spent
elsewhere. Measured per gradient decile, on the real photographs.
"""
import os
import numpy as np
import detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
SIG, ladder, ngrad = Z['sigmas'], Z['var_ladder'], Z['ngrad']
z = Z['z']; FX = Dt.bundle()['intrinsics']['fx']
r_mm = SIG[None, :] * (z / FX * 1000.0)[:, None]

o = np.argsort(ngrad); e = np.linspace(0, o.size, 11).astype(int)
print('median footprint luma STD at each footprint radius, by gradient decile')
print('  the columns are full-res sigma in px; world mm varies with range')
print('  decile ' + ''.join('%8.1fpx' % s for s in SIG))
for k in range(10):
    s = o[e[k]:e[k + 1]]
    m = np.sqrt(np.maximum(np.median(ladder[s], axis=0), 0))
    print('    %2d   ' % (k + 1) + ''.join('%10.4f' % v for v in m))

print('\n  elasticity d log(std) / d log(radius), between 19.2 px and 2.4 px')
for k in range(10):
    s = o[e[k]:e[k + 1]]
    m = np.sqrt(np.maximum(np.median(ladder[s], axis=0), 1e-12))
    el = (np.log(m[5]) - np.log(m[2])) / (np.log(SIG[5]) - np.log(SIG[2]))
    print('    decile %2d   %.3f   (std %.4f -> %.4f for an 8x shrink)'
          % (k + 1, el, m[5], m[2]))

print('\n  the photograph\'s own floor: median footprint std at the smallest'
      '\n  scale sampled (%.1f px), which is JPEG noise, not scene detail: %.4f'
      % (SIG[0], np.sqrt(np.median(ladder[:, 0]))))
flat = ngrad < np.percentile(ngrad, 10)
print('  in the flattest decile that floor is %.4f' % np.sqrt(np.median(ladder[flat, 0])))
