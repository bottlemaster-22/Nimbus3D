"""Does the REAL model show the shrink ladder the simulator assumes?

The simulator's whole size model is: a delta-function seed, a multiplicative
1/1.6 ladder driven by densification events, and an independent additive
Gaussian random walk in log space from Adam. Build 182's own numbers make that
testable. It ran 6,836 growth splits (three-axis, so -log10(1.6) = -0.2041
decades on the largest axis for BOTH children = 13,672 slots) and 21,000
one-axis relocations (which move the largest axis by NOTHING on a disc whose
two big axes are equal, and only start moving it on the second hit).

So the predicted final population is a MIXTURE:
    ~95.4% at log10(seed) + drift + N(0, 0.1066)
    ~ 4.6% at log10(seed) + drift - 0.2041 + N(0, 0.1066)
The ladder step is 1.91 sigma of the fitted noise, so a 4.6% component sitting
1.9 sigma low is a visible left shoulder, not something the noise hides.

If the real histogram has no such structure and is instead smooth and
LEFT-skewed with a long low tail, the additive-noise-plus-ladder decomposition
is not what produced it, and the fitted 'Adam' term is absorbing a
size-dependent process rather than measuring one.
"""
import numpy as np, os
d = os.path.dirname(os.path.abspath(__file__))
big = np.load(os.path.join(d, 'ours_largest_mm.npy'))
L = np.log10(big)
print('n %d  mean %.4f  sd %.4f  skew %.4f  excess kurtosis %.4f'
      % (len(L), L.mean(), L.std(),
         ((L - L.mean()) ** 3).mean() / L.std() ** 3,
         ((L - L.mean()) ** 4).mean() / L.std() ** 4 - 3.0))

# Gaussianity of log-size: the simulator's random walk is exactly Gaussian.
from math import sqrt
q = np.percentile(L, [10, 25, 50, 75, 90])
print('log10 percentiles p10 %.4f p25 %.4f p50 %.4f p75 %.4f p90 %.4f' % tuple(q))
print('  (p90-p50)/(p50-p10) = %.3f   Gaussian would be 1.000' % ((q[4]-q[2])/(q[2]-q[0])))
print('  (p90-p10)/sd = %.3f          Gaussian would be 2.563' % ((q[4]-q[0])/L.std()))
print('  (p75-p25)/sd = %.3f          Gaussian would be 1.349' % ((q[3]-q[1])/L.std()))

# Look for the ladder shoulder at -0.2041 decades from the mode.
h, e = np.histogram(L, bins=220, range=(0.5, 1.8))
c = 0.5 * (e[1:] + e[:-1])
mode = c[h.argmax()]
print('\nmode at %.4f dec = %.3f mm ; ladder rungs would sit at:' % (mode, 10**mode))
for k in range(0, 4):
    x = mode - k * np.log10(1.6)
    lo, hi = x - 0.03, x + 0.03
    share = 100.0 * ((L > lo) & (L < hi)).mean()
    print('   rung %d  %.4f dec = %7.3f mm   share within +/-0.03 dec: %6.3f%%'
          % (k, x, 10 ** x, share))

print('\nhistogram of log10 largest (mm), 0.025-decade bins:')
h2, e2 = np.histogram(L, bins=np.arange(0.55, 1.75, 0.025))
for i in range(len(h2)):
    s = 100.0 * h2[i] / len(L)
    if s > 0.02:
        print('  %.3f dec %7.2f mm %6.2f%% %s'
              % (e2[i], 10 ** e2[i], s, '#' * int(s * 2)))
