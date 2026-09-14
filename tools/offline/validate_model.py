"""Is "a splat's error is the image variance under its footprint" actually true?

The whole sizing argument rests on that model, so it gets checked against the
real render before anything is concluded from it. For every frame:

  actual     the luma error the numpy render of build 182's model makes
             against the photograph
  predicted  the local luma standard deviation of the photograph at the
             median footprint scale, which is what the model says the error
             has to be

If the two agree in magnitude AND correlate spatially, the model is a
description of this renderer, not a story about it.
"""
import os, glob
import numpy as np, cv2
import detail as Dt

SIG_RENDER = 15.68 * 720.0 / 1920.0     # median footprint sigma, render px

print('predicted-vs-actual, per frame, at footprint sigma %.2f render px'
      % SIG_RENDER)
print('frame   actual RMS   predicted RMS   spatial corr(|err|, local std)')
rows = []
for f in sorted(glob.glob(os.path.join(Dt.SCRATCH, 'render_*.npz'))):
    z = np.load(f)
    img, rgb = z['img'], z['rgb']
    lum_r = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    lum_p = z['luma']
    err = (lum_r - lum_p).astype(np.float32)
    k = int(2 * round(3 * SIG_RENDER) + 1)
    m = cv2.GaussianBlur(lum_p, (k, k), SIG_RENDER, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.GaussianBlur(lum_p * lum_p, (k, k), SIG_RENDER, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(m2 - m * m, 0))
    a, b = np.abs(err).ravel(), std.ravel()
    c = np.corrcoef(a, b)[0, 1]
    idx = int(os.path.basename(f).split('_')[1].split('.')[0])
    print('%5d   %10.4f   %13.4f   %.3f'
          % (idx, np.sqrt((err ** 2).mean()), np.sqrt((std ** 2).mean()), c))
    rows.append((np.sqrt((err ** 2).mean()), np.sqrt((std ** 2).mean()), c))
r = np.array(rows)
print('\nmedian actual RMS %.4f   median predicted RMS %.4f   median corr %.3f'
      % (np.median(r[:, 0]), np.median(r[:, 1]), np.median(r[:, 2])))
print('actual PSNR-equiv %.2f dB   predicted %.2f dB'
      % (20 * np.log10(1 / np.median(r[:, 0])), 20 * np.log10(1 / np.median(r[:, 1]))))
