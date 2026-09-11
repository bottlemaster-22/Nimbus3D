"""REFUTATION TEST 2: the missing CONTROLS for "drop the least-contributing
half costs only 0.35 dB".

prune_psnr.py reported the contrib-ranked cut only.  Without a control you
cannot tell whether the RANKING is smart or whether PSNR at this fidelity is
simply insensitive to which 150,000 splats you remove.  Four rankings, same
frames, same renderer:

    contrib  - the finding's ranking (ascending accumulated alpha*T)
    random   - seeded uniform random
    smallest - smallest largest-axis first   (the size-biased strawman)
    opacity  - lowest sigmoid(opacity) first

Also reports a DETAIL-SENSITIVE readout PSNR is blind to: gradient energy of
the render, and PSNR restricted to the top-20% gradient pixels of the photo.
"""
import io, json, os, sys
import numpy as np
import project as P
import detail as Dt
import render as Rn

SCRATCH = Dt.SCRATCH
FRAC = 0.50

def psnr(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        d = d[m]
    mse = d.mean()
    return 10 * np.log10(1.0 / max(mse, 1e-12))

def gradmag(g):
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    return gx + gy

def subset(col, mask):
    return {k: v[mask] for k, v in col.items()}

def main():
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    contrib = np.load(os.path.join(SCRATCH, 'contrib.npy'))
    held_out = sorted(json.load(io.open(os.path.join(P.D, 'model',
                                        'held_out_frames.json'), encoding='utf-8')))
    photo = sorted(dict(Dt.frames_with_images()).keys())
    imgpaths = dict(Dt.frames_with_images())

    scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']],
                                    axis=1).astype(np.float64), -12, 3))
    largest = np.sort(scale, axis=1)[:, -1]
    opac = 1.0 / (1.0 + np.exp(-col['opacity'].astype(np.float64)))
    rng = np.random.default_rng(20260909)

    ndrop = int(FRAC * count)
    rankings = {
        'contrib':  np.argsort(contrib),
        'random':   rng.permutation(count),
        'smallest': np.argsort(largest),
        'opacity':  np.argsort(opac),
    }
    masks = {'FULL': np.ones(count, bool)}
    for name, order in rankings.items():
        k = np.ones(count, bool); k[order[:ndrop]] = False
        masks[name] = k
        print('%-9s keeps %d, median largest axis %.2f mm'
              % (name, k.sum(), np.median(largest[k]) * 1000))
    # overlap of the cuts
    cset = ~masks['contrib']
    for name in ('random', 'smallest', 'opacity'):
        o = (cset & ~masks[name]).sum()
        print('  overlap of contrib-cut with %-9s cut: %6d (%.1f%%)'
              % (name, o, 100 * o / ndrop))

    pf = photo[:nframes]
    ho = held_out[:nframes]
    print('\nphoto frames %s\nheld-out frames %s' % (pf, ho))

    resA = {k: [] for k in masks}
    resA_hi = {k: [] for k in masks}
    grad = {k: [] for k in masks}
    for f in pf:
        _, rgb = Dt.luma_of(imgpaths[f], rw, rh)
        gph = gradmag(rgb.mean(axis=2))
        hi = gph >= np.percentile(gph, 80)
        him = np.zeros(rgb.shape[:2], bool); him[:-1, :-1] = hi
        for name, k in masks.items():
            rot, t = poses[f]; R = P.quat_to_matrix(rot)
            img, _ = Rn.render(subset(col, k), int(k.sum()), R, t, fx, fy, cx, cy, rw, rh)
            resA[name].append(psnr(img, rgb))
            resA_hi[name].append(psnr(img, rgb, him))
            grad[name].append(float(gradmag(img.mean(axis=2)).mean()))
        print('  frame %d done' % f)

    print('\n=== (A) PSNR vs real photo, mean of %d frames, drop %.0f%% ===' % (len(pf), FRAC*100))
    base = float(np.mean(resA['FULL']))
    bhi  = float(np.mean(resA_hi['FULL']))
    bg   = float(np.mean(grad['FULL']))
    print('%-9s %10s %10s %14s %10s %10s' % ('ranking', 'PSNR', 'delta', 'PSNR@hi-grad', 'delta', 'gradE'))
    for name in ('FULL', 'contrib', 'random', 'smallest', 'opacity'):
        m = float(np.mean(resA[name])); mh = float(np.mean(resA_hi[name]))
        print('%-9s %10.3f %10.3f %14.3f %10.3f %10.5f (%+.1f%%)'
              % (name, m, m - base, mh, mh - bhi, float(np.mean(grad[name])),
                 100 * (float(np.mean(grad[name])) - bg) / bg))

    print('\n=== (B) held-out-pose self-consistency vs FULL, mean of %d frames ===' % len(ho))
    resB = {k: [] for k in rankings}
    for f in ho:
        rot, t = poses[f]; R = P.quat_to_matrix(rot)
        full, _ = Rn.render(col, count, R, t, fx, fy, cx, cy, rw, rh)
        for name, k in masks.items():
            if name == 'FULL':
                continue
            img, _ = Rn.render(subset(col, k), int(k.sum()), R, t, fx, fy, cx, cy, rw, rh)
            resB[name].append(psnr(img, full))
        print('  held-out %d done' % f)
    for name in ('contrib', 'random', 'smallest', 'opacity'):
        print('  %-9s PSNR-vs-FULL %6.3f dB   %s'
              % (name, float(np.mean(resB[name])), ['%.1f' % v for v in resB[name]]))

if __name__ == '__main__':
    main()
