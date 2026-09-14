"""THE DEPLOYMENT TEST: a seed is sized ONCE, from the keyframe that placed
it, but has to serve every other view.

So the number that matters is not "does this frame's gradient predict this
frame's footprint variance" (it must, they are the same photograph). It is:
does the gradient measured in the frame that PLACED the seed predict the
footprint variance in a DIFFERENT frame?

If it does not, the signal is a property of one photograph - a highlight, a
motion blur, an exposure - and sizing from it would be noise.
"""
import os
import numpy as np
import detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
frame, splat = Z['frame'], Z['splat']
ngrad, own_var = Z['ngrad'], np.sqrt(np.maximum(Z['var_ladder'][:, 0] * 0
                                                + Z['own_var'], 0))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


order = np.lexsort((frame, splat))
s, f, gr, st = splat[order], frame[order], ngrad[order], own_var[order]
start = np.searchsorted(s, np.unique(s))
counts = np.diff(np.concatenate([start, [s.size]]))
multi = np.nonzero(counts >= 2)[0]
print('%d splats seen in 2+ frames (of %d seen at all)'
      % (multi.size, counts.size))

# gradient from the FIRST frame that saw it, footprint std in the LAST
a = start[multi]
b = start[multi] + counts[multi] - 1
print('\n  Spearman(gradient in frame A, footprint std in frame A)  %+.3f'
      % spearman(gr[a], st[a]))
print('  Spearman(gradient in frame A, footprint std in frame B)  %+.3f'
      % spearman(gr[a], st[b]))
print('  Spearman(footprint std A,      footprint std B)          %+.3f  (ceiling)'
      % spearman(st[a], st[b]))
print('  frames A and B are the first and last of the up-to-16 that saw the splat')

print('\n  gradient decile IN FRAME A -> median footprint std IN FRAME B')
o = np.argsort(gr[a]); e = np.linspace(0, o.size, 11).astype(int)
for k in range(10):
    q = o[e[k]:e[k + 1]]
    print('    %2d   gradient %.4f   std in A %.4f   std in B %.4f'
          % (k + 1, np.median(gr[a][q]), np.median(st[a][q]), np.median(st[b][q])))

# how much does one splat's own gradient move between views?
g2 = np.zeros(counts.size); n2 = np.zeros(counts.size)
np.add.at(g2, np.searchsorted(np.unique(s), s), gr)
np.add.at(n2, np.searchsorted(np.unique(s), s), 1)
mean_g = g2 / n2
var_g = np.zeros(counts.size)
np.add.at(var_g, np.searchsorted(np.unique(s), s),
          (gr - mean_g[np.searchsorted(np.unique(s), s)]) ** 2)
within = var_g[multi] / np.maximum(counts[multi] - 1, 1)
between = mean_g[multi].var()
print('\n  variance of the gradient BETWEEN splats %.6f, WITHIN one splat across'
      '\n  views %.6f  ->  %.0f%% of the signal is a property of the SURFACE,'
      ' not the view'
      % (between, within.mean(), 100 * between / (between + within.mean())))
