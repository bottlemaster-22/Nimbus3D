"""Does deleting the least-contributing share of the model cost PSNR?

Uses the per-splat accumulated-alpha weight from contrib.py (contrib.npy in
the scratchpad) to rank every splat, then re-renders with the FULL model and
with the bottom X% removed.

TWO measurements, because of a constraint discovered while writing this: none
of the 12 held-out frames (held_out_frames.json: 17, 61, 109, ... 460) have a
JPEG on this machine. Only frames 0-15 shipped with the scan
(tools/offline/detail.py's docstring says so, and Dt.frames_with_images()
confirms it: exactly 16 frames, 0..15, none of them held-out indices).

  (A) REAL PSNR vs photograph, on frames 0-15. These are NOT the held-out set
      (the smallest held-out index is 17), so this is a TRAINED-view fidelity
      check, not the held-out generalisation number. It answers "does the
      bottom X% still matter to reconstructing a view the trainer actually
      supervised on".

  (B) SELF-CONSISTENCY at the real 12 held-out camera poses: pruned model
      render vs FULL model render (synthetic vs synthetic, no photograph
      needed). This answers "how much does the picture at a held-out
      viewpoint change when the bottom X% is deleted", which is the
      question that actually matters for the cap, even without a ground
      truth photo to score it against.

Run: python -u prune_psnr.py
"""
import io
import json
import os

import numpy as np

import project as P
import detail as Dt
import render as Rn

SCRATCH = Dt.SCRATCH


def render_frame(col, count, frame, fx, fy, cx, cy, rw, rh, poses):
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    img, alpha = Rn.render(col, count, R, t, fx, fy, cx, cy, rw, rh)
    return img, alpha


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def subset(col, mask):
    return {k: v[mask] for k, v in col.items()}


def main():
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    held_out = sorted(json.load(io.open(
        os.path.join(P.D, 'model', 'held_out_frames.json'), encoding='utf-8')))
    photo_frames = sorted(dict(Dt.frames_with_images()).keys())
    print('frames with a real photo on disk:', photo_frames)
    print('held-out frames (no photo available for any of these):', held_out)
    assert not (set(photo_frames) & set(held_out)), \
        "held-out frames now have photos -- update this script's assumption"

    contrib = np.load(os.path.join(SCRATCH, 'contrib.npy'))
    assert contrib.shape[0] == count, (contrib.shape, count)
    order = np.argsort(contrib)          # ascending: least-contributing first

    fractions = [0.0, 0.25, 0.50, 0.75, 0.90]
    masks = {}
    for frac in fractions:
        n_drop = int(frac * count)
        keep = np.ones(count, dtype=bool)
        keep[order[:n_drop]] = False
        masks[frac] = keep
        print('fraction dropped %.2f -> %d splats kept (of %d)'
              % (frac, keep.sum(), count))

    # --- (A) real PSNR vs photo on the 16 available (trained-view) frames ---
    photo_results = {frac: [] for frac in fractions}
    for f in photo_frames:
        path = dict(Dt.frames_with_images())[f]
        _, rgb = Dt.luma_of(path, rw, rh)
        for frac in fractions:
            keep = masks[frac]
            sub = subset(col, keep)
            img, _ = render_frame(sub, keep.sum(), f, fx, fy, cx, cy, rw, rh, poses)
            p = psnr(img, rgb)
            photo_results[frac].append(p)
            print('  [A] frame %3d  drop %.2f  PSNR-vs-photo %.3f dB' % (f, frac, p))

    print('\n=== (A) TRAINED-VIEW PSNR vs real photo, mean over %d frames ==='
          % len(photo_frames))
    for frac in fractions:
        vals = photo_results[frac]
        print('  drop %5.1f%% (%6d splats kept): mean PSNR %6.3f dB   per-frame %s'
              % (frac * 100, masks[frac].sum(), float(np.mean(vals)),
                 ['%.2f' % v for v in vals]))

    # --- (B) self-consistency at the real held-out poses, pruned vs FULL ---
    full_col = col
    full_count = count
    selfcon_results = {frac: [] for frac in fractions if frac > 0}
    for f in held_out:
        img_full, _ = render_frame(full_col, full_count, f, fx, fy, cx, cy, rw, rh, poses)
        for frac in fractions:
            if frac == 0:
                continue
            keep = masks[frac]
            sub = subset(col, keep)
            img_pruned, _ = render_frame(sub, keep.sum(), f, fx, fy, cx, cy, rw, rh, poses)
            p = psnr(img_pruned, img_full)
            meanabs = np.abs(img_pruned - img_full).mean()
            selfcon_results[frac].append(p)
            print('  [B] held-out frame %3d  drop %.2f  PSNR-vs-FULL %.3f dB  mean|err| %.4f'
                  % (f, frac, p, meanabs))

    print('\n=== (B) HELD-OUT-POSE self-consistency, pruned vs FULL model, mean over %d frames ==='
          % len(held_out))
    for frac in fractions:
        if frac == 0:
            continue
        vals = selfcon_results[frac]
        print('  drop %5.1f%% (%6d splats kept): mean PSNR-vs-FULL %6.3f dB   per-frame %s'
              % (frac * 100, masks[frac].sum(), float(np.mean(vals)),
                 ['%.2f' % v for v in vals]))

    with open(os.path.join(SCRATCH, 'prune_psnr_results.json'), 'w') as fh:
        json.dump({'held_out': held_out,
                    'photo_frames': photo_frames,
                    'fractions': fractions,
                    'photo_results': {str(k): v for k, v in photo_results.items()},
                    'selfcon_results': {str(k): v for k, v in selfcon_results.items()},
                    'kept_counts': {str(k): int(masks[k].sum()) for k in fractions}},
                   fh, indent=2)


if __name__ == '__main__':
    main()
