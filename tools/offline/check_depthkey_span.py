"""Which 12-bit depth span is safe for a 24-bit (6-pass) radix key?

Reuses build/paint/psnr from refute_depthbits.py.  Scores every key variant
against the EXACT float-order render (the quantity the key width actually
changes) and against the real photograph, and counts in-tile ties.

No .npy output.  Usage: python -u check_depthkey_span.py 0 4 8 12 15
"""
import io, json, os, sys
import numpy as np

import project as P
import detail as Dt
import refute_depthbits as R


def order_linear(g, bits, near, far):
    z = g['z'][g['splat']]
    norm = np.clip((z - near) / max(far - near, 1e-3), 0.0, 1.0)
    q = np.floor(norm * float((1 << bits) - 1)).astype(np.int64)
    return np.lexsort((np.arange(g['total']), q, g['tile_id'])), q


def order_log(g, bits, near, far):
    z = np.maximum(g['z'][g['splat']], near)
    norm = np.clip(np.log(z / near) / np.log(far / near), 0.0, 1.0)
    q = np.floor(norm * float((1 << bits) - 1)).astype(np.int64)
    return np.lexsort((np.arange(g['total']), q, g['tile_id'])), q


CASES = [
    ('16b lin 0.05-100 SHIPPED ', order_linear, 16, 0.05, 100.0),
    ('12b lin 0.05-10  proposal', order_linear, 12, 0.05, 10.0),
    ('12b lin 0.05-16          ', order_linear, 12, 0.05, 16.0),
    ('12b lin 0.05-24          ', order_linear, 12, 0.05, 24.0),
    ('12b log 0.05-100         ', order_log, 12, 0.05, 100.0),
    ('13b lin 0.05-16          ', order_linear, 13, 0.05, 16.0),
]


def main():
    frames = [int(x) for x in sys.argv[1:]] or [0, 4, 8, 12, 15]
    census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'),
                               encoding='utf-8'))
    sl = census['slices'][0]
    rw, rh = sl['renderWidth'], sl['renderHeight']
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    poses = P.load_poses()
    photos = dict(Dt.frames_with_images())
    agg = {c[0]: [] for c in CASES}
    for f in frames:
        rot, t = poses[f]
        Rm = P.quat_to_matrix(rot)
        g = R.build(col, count, Rm, t, fx, fy, cx, cy, rw, rh)
        zz = g['z'][g['splat']]
        exact = np.lexsort((np.arange(g['total']), zz, g['tile_id']))
        base = R.paint(g, exact, rw, rh)
        _, photo = Dt.luma_of(photos[f], rw, rh)
        pb = R.psnr(base, photo)
        print('')
        print('frame %d  instances %d  depth p50 %.2f p99 %.2f max %.2f m  >10m %.4f%%'
              % (f, g['total'], np.percentile(zz, 50), np.percentile(zz, 99), zz.max(),
                 100.0 * (zz > 10.0).mean()))
        print('  exact float order           vs photo %.4f' % pb)
        for name, fn, bits, near, far in CASES:
            o, q = fn(g, bits, near, far)
            img = R.paint(g, o, rw, rh)
            same = g['tile_id'][o][1:] == g['tile_id'][o][:-1]
            tied = same & (q[o][1:] == q[o][:-1])
            vp = R.psnr(img, photo)
            ve = R.psnr(img, base)
            agg[name].append((vp - pb, ve, 100.0 * tied.sum() / max(same.sum(), 1)))
            print('  %s vs photo %+.4f dB  vs exact %6.2f dB  maxpix %.4f  ties %5.2f%%'
                  % (name, vp - pb, ve, np.abs(img - base).max(), agg[name][-1][2]))
        sys.stdout.flush()
    print('\nMEAN over frames %s' % frames)
    for name, *_ in CASES:
        a = np.array(agg[name])
        print('  %s dPhoto %+.4f  (worst %+.4f)  vsExact %6.2f dB (worst %6.2f)  ties %5.2f%%'
              % (name, a[:, 0].mean(), a[:, 0].min(), a[:, 1].mean(), a[:, 1].min(),
                 a[:, 2].mean()))


if __name__ == '__main__':
    main()
