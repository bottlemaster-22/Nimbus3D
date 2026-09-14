"""How small a splat can the trainer even SEE at its own render resolution?

Sizing seeds from image detail is pointless below the size one render pixel
subtends, because nothing below that changes the loss. The Mip-Splatting 2D
low-pass (FILTER_2D_VARIANCE = 0.25, i.e. 0.5 px of sigma) sets the real
floor a little above one pixel.
"""
import os
import numpy as np
import project as P, detail as Dt

Z = np.load(os.path.join(Dt.SCRATCH, 'seedsize.npz'))
b = Dt.bundle()
FX = b['intrinsics']['fx']; W = b['intrinsics']['width']
z, sig_px, ngrad, ladder = Z['z'], Z['sig_px'], Z['ngrad'], Z['var_ladder']
SIG = Z['sigmas']

for rw in (540, 720, 960, 1440, 1920):
    f = FX * rw / W
    mm = z / f * 1000.0                      # one render pixel, in world mm
    print('render %4d px wide: fx %6.1f   one pixel = %.2f mm at p10 range, '
          '%.2f mm median, %.2f mm at p90'
          % (rw, f, *[np.percentile(mm, q) for q in (10, 50, 90)]))

f720 = FX * 720.0 / W
floor_mm = z / f720 * 1000.0 * np.sqrt(P.FILTER_2D_VARIANCE + 1.0)
print('\nAt the 720x540 the trainer actually renders, the smallest splat whose'
      '\nfootprint still exceeds the 2D low-pass is %.2f mm (median), %.2f mm at p90 range.'
      % (np.median(floor_mm), np.percentile(floor_mm, 90)))

tau = 0.05
r_ladder = SIG[None, :] * (z / FX * 1000.0)[:, None]
req = np.where(ladder <= tau * tau, r_ladder, 0.0).max(axis=1)
req = np.where(req > 0, req, SIG[0] * z / FX * 1000.0)
print('\nRequired radius to hold footprint luma std under %.2f:' % tau)
for q in (5, 10, 25, 50, 75, 90):
    print('   p%-2d %7.2f mm' % (q, np.percentile(req, q)))
print('   fraction of splats whose requirement is BELOW the render floor: %.1f%%'
      % (100 * (req < floor_mm).mean()))
print('   fraction whose requirement is ABOVE the current %.1f mm radius: %.1f%%'
      % (np.median(sig_px * z / FX * 1000.0),
         100 * (req > (sig_px * z / FX * 1000.0)).mean()))
cur = sig_px * z / FX * 1000.0
clamped = np.clip(req, floor_mm, 40.0)
print('\nWith the render floor and a 40 mm cap applied, the DEPLOYABLE size range is')
for q in (10, 50, 90):
    print('   p%-2d %7.2f mm' % (q, np.percentile(clamped, q)))
print('   p90/p10 = %.1fx   (the field today is 1.8x, Scaniverse is 8.9x)'
      % (np.percentile(clamped, 90) / np.percentile(clamped, 10)))
