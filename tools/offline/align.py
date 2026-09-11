"""How well does a render of the model line up with the real photograph?

If the projection were in the wrong frame, or the poses did not belong to
these images, nothing sampled at a splat's projection would mean anything.
Phase correlation gives the residual shift in render pixels.
"""
import io, json, os
import numpy as np, cv2
import project as P, detail as Dt, render as Rd

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                           encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()

print('frame   PSNR dB   mean|err|   shift px (x, y)   correlation')
rows = []
for idx, path in Dt.frames_with_images():
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    img, alpha = Rd.render(col, count, R, t, fx, fy, cx, cy, rw, rh)
    luma_ref, rgb = Dt.luma_of(path, rw, rh)
    luma_img = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]).astype(np.float32)
    mse = ((img - rgb) ** 2).mean()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    (dx, dy), resp = cv2.phaseCorrelate(luma_ref.astype(np.float64),
                                        luma_img.astype(np.float64))
    print('%5d  %8.2f  %9.4f   %6.2f %6.2f      %.3f'
          % (idx, psnr, np.abs(img - rgb).mean(), dx, dy, resp))
    rows.append((idx, psnr, dx, dy, resp))
    np.savez(os.path.join(Dt.SCRATCH, 'render_%d.npz' % idx),
             img=img, rgb=rgb, luma=luma_ref, alpha=alpha)

a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print('\nmedian PSNR %.2f dB   median |shift| %.2f px   median response %.3f'
      % (np.median(a[:, 0]), np.median(np.hypot(a[:, 1], a[:, 2])),
         np.median(a[:, 3])))
