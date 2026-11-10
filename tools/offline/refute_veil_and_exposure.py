"""Three refutation tests in one render pass over the 16 frames that have images.

A. THE NEAR VEIL (finding "the camera can stand inside 4,736 Gaussians").
   Render each frame with the full model, then again with the near shell DELETED,
   and score both.  That measures the proposal's effect on the render instead of
   inferring it from 1.45% ray absorption.
   Two deletions:
     - per-view: every splat nearer than 0.50 m in THAT view (an upper bound on
       what any global prune could recover, because it is an oracle per frame);
     - global:   every splat whose MAX 3-sigma radius over 100 views exceeds
       180 px, which is exactly what pruneMaxScreenRadiusPx would delete.

B. THE EXPOSURE CLAMP (finding "14 of 16 frames want a gain or bias outside the
   box").  The capture bundle records exposureDurationSeconds and
   exposureOffsetEV per frame.  Report the UNCLAMPED least-squares optimum next
   to the TRUE exposure ratio and the frame's motion blur, and correlate.

C. WHICH FRAMES ARE THESE?  Held-out frame indices are in model/held_out_frames.json.
"""
import io, json, os
import numpy as np
import project as P
import detail as Dt
import lossq_blur as B

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
frames = dict(Dt.frames_with_images())
bundle = Dt.bundle()
byidx = {int(f['index']): f for f in bundle['frames']}
held = set(json.load(io.open(os.path.join(P.D, 'model', 'held_out_frames.json'))))

print('=== C. WHAT ARE THE 16 FRAMES WITH IMAGES? ===')
print('  held-out bundle frame indices: %s' % sorted(held))
print('  frames with images on disk   : %s' % sorted(frames))
print('  overlap                      : %s' % sorted(set(frames) & held))
print('  capture is %d frames / %.0f s; these 16 span %.1f s from the start'
      % (len(bundle['frames']), 276.3,
         byidx[15]['timestampSeconds'] - byidx[0]['timestampSeconds']))
print('  trained keyframes %d, held out %d, of %d bundle frames'
      % (sl['keyframesTrained'], sl['keyframesHeldOut'], len(bundle['frames'])))

# global 180 px mask, max over 100 spread views
keys = sorted(poses.keys())
use = keys[::max(len(keys)//100, 1)][:100]
mxr = np.zeros(count)
for f in use:
    rot, t = poses[f]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mxr = np.where(ok, np.maximum(mxr, radius), mxr)
big = mxr > 180.0
print('\n  global 180 px prune would remove %d of %d splats (%.2f%%)'
      % (big.sum(), count, 100*big.mean()))

def sub(colmap, keep):
    return {k: v[keep] for k, v in colmap.items()}

kept180 = sub(col, ~big)
n180 = int((~big).sum())

print('\n=== A/B. PER FRAME ===')
print('%5s %5s %7s %8s   %8s %8s %8s   %9s %9s %8s %8s' %
      ('frame', 'held', 'blurPx', 'expDur', 'PSNRfull', 'PSNRno<.5', 'PSNR180', 'gainRaw', 'biasRaw', 'clampd?', 'trueGain'))
gains, blurs, psnrs = [], [], []
for f in sorted(frames):
    rot, t = poses[f]
    R = P.quat_to_matrix(rot)
    img, Tf, ok, g, inst = B.render(col, count, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    _, gt = Dt.luma_of(frames[f], rw, rh)
    gt = gt.astype(np.float64)
    gain, bias = B.fit_exposure(img, gt)
    p_full = B.psnr(gain*img+bias, gt)

    # unclamped optimum
    x, y = img.ravel(), gt.ravel()
    n = float(x.size)
    den = n*(x*x).sum() - x.sum()**2
    graw = (n*(x*y).sum() - x.sum()*y.sum())/den
    braw = (y.sum() - graw*x.sum())/n

    # per-view oracle: drop everything nearer than 0.5 m in THIS view
    z = g['z']
    near = ok & (z < 0.5)
    k2 = ~near
    img2, _, _, _, _ = B.render(sub(col, k2), int(k2.sum()), R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    ga, ba = B.fit_exposure(img2, gt)
    p_near = B.psnr(ga*img2+ba, gt)

    img3, _, _, _, _ = B.render(kept180, n180, R, t, fx, fy, cx, cy, rw, rh, 0.25, 4)
    gb, bb = B.fit_exposure(img3, gt)
    p_180 = B.psnr(gb*img3+bb, gt)

    fr = byidx[f]
    clamped = 'YES' if (abs(gain-graw) > 1e-6 or abs(bias-braw) > 1e-6) else 'no'
    print('%5d %5s %7.3f %8.5f   %8.3f %8.3f %8.3f   %9.4f %9.4f %8s %8.4f'
          % (f, 'YES' if f in held else 'no', fr['qc']['motionBlurPixels'],
             fr['exposureDurationSeconds'], p_full, p_near, p_180,
             graw, braw, clamped,
             fr['exposureDurationSeconds']/byidx[0]['exposureDurationSeconds']))
    gains.append(graw); blurs.append(fr['qc']['motionBlurPixels']); psnrs.append(p_full)

gains, blurs, psnrs = map(np.array, (gains, blurs, psnrs))
print('\n=== B. IS THE "EXPOSURE" FIT ACTUALLY EXPOSURE? ===')
print('  TRUE per-frame exposure ratio across these 16 frames: max/min = %.4f'
      % (max(byidx[f]["exposureDurationSeconds"] for f in frames) /
         min(byidx[f]["exposureDurationSeconds"] for f in frames)))
print('  exposureOffsetEV on all 868 frames: %s' %
      sorted({f['exposureOffsetEV'] for f in bundle['frames']}))
print('  qc.exposureJumpEV max over all 868: %.4f' %
      max(f['qc']['exposureJumpEV'] for f in bundle['frames']))
print('  unclamped fitted gain: min %.4f max %.4f  (true spread is 1.0000)'
      % (gains.min(), gains.max()))
print('  corr(fitted gain, motionBlurPixels) = %+.3f' % np.corrcoef(gains, blurs)[0,1])
print('  corr(fitted gain, render PSNR)      = %+.3f' % np.corrcoef(gains, psnrs)[0,1])
