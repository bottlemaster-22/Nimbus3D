"""Entry 43(3): measure the stride-4 Laplacian variance the capture sharpness
meter uses, on the 16 real photographs available, per 240x240 block, and report
the flat-wall vs textured ratio the header comment asserts as prose.

Exactly CaptureSharpnessMeter.sharpness: 4*c - up - down - left - right on a
stride-4 grid of the luma plane, variance of that value over the frame.
"""
import glob, io, json, os
import numpy as np
from PIL import Image

SRC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840\images"
STEP = 4

files = sorted(glob.glob(os.path.join(SRC, '*.jpg')))
print('photographs on disk: %d' % len(files))

def luma(path):
    im = Image.open(path).convert('L')      # BT.601 luma, same plane ARKit gives
    return np.asarray(im, dtype=np.int32), im.size

def lap_field(y):
    c = y[STEP:-STEP:1, STEP:-STEP:1]
    up = y[0:-2*STEP:1, STEP:-STEP:1]
    dn = y[2*STEP::1, STEP:-STEP:1]
    lf = y[STEP:-STEP:1, 0:-2*STEP:1]
    rt = y[STEP:-STEP:1, 2*STEP::1]
    return 4*c - up - dn - lf - rt

allvar = []
blockvars = []
for p in files:
    y, size = luma(p)
    lap = lap_field(y)
    # the meter's own sampling: stride 4 in both axes
    s = lap[::STEP, ::STEP].astype(np.float64)
    allvar.append((os.path.basename(p), size, float(s.var()), s.size))
    # per-block variance, 60x60 samples on the strided grid = 240x240 native px
    h, w = s.shape
    B = 60
    for by in range(0, h - B + 1, B):
        for bx in range(0, w - B + 1, B):
            blockvars.append(float(s[by:by+B, bx:bx+B].var()))

for n, size, v, c in allvar:
    print('  %-38s %sx%s  samples %7d  lapVar %10.1f' % (n, size[0], size[1], c, v))

bv = np.array(blockvars)
bv = bv[np.isfinite(bv)]
print('\nper-block (240x240 native px) Laplacian variance, %d blocks over %d frames'
      % (len(bv), len(files)))
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print('  p%-3d %12.1f' % (q, np.percentile(bv, q)))
print('  min %12.1f   max %12.1f' % (bv.min(), bv.max()))
print('\nRATIOS the header comment asserts as "an order of magnitude":')
print('  p90 / p10  = %8.1f x' % (np.percentile(bv, 90) / max(np.percentile(bv, 10), 1e-9)))
print('  p95 / p05  = %8.1f x' % (np.percentile(bv, 95) / max(np.percentile(bv, 5), 1e-9)))
print('  p99 / p01  = %8.1f x' % (np.percentile(bv, 99) / max(np.percentile(bv, 1), 1e-9)))
print('  max / min  = %8.1f x' % (bv.max() / max(bv.min(), 1e-9)))
fv = np.array([v for _, _, v, _ in allvar])
print('\nWHOLE-FRAME variance across the 16 frames: min %.1f max %.1f  ratio %.2fx'
      % (fv.min(), fv.max(), fv.max()/max(fv.min(), 1e-9)))
b = json.load(io.open(r"C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840\capture_bundle.json", encoding='utf-8'))
names = {os.path.basename(f['imagePath']): f for f in b['frames']}
print('\nreported qc.sharpness vs measured raw variance (same frames):')
for n, size, v, c in allvar:
    f = names.get(n)
    if f:
        print('  frame %3d  qc.sharpness %.4f  blur_px %.2f  measured lapVar %9.1f'
              % (f['index'], f['qc']['sharpness'], f['qc']['motionBlurPixels'], v))
