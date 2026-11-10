"""Renders a trained model (model.ply) offline so its SHAPE can be looked at.

Writes, per view: colour (base colour only), a surface-direction image (each
splat's flattest axis, the closest thing a splat has to a surface normal,
shown as colour), and depth. A clean reconstruction shows smooth flat walls in
the direction image; fuzz, layers, floaters and noise show up immediately.

    python render_view.py <scan folder> <out folder> [frame index ...]

Front-to-back alpha compositing in depth-sorted batches, the same 2D low-pass
(0.25 px^2) and opacity compensation as the device. View-dependent colour is
left out on purpose: this is for geometry.
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

import project as P

C0 = 0.28209479177387814
W, H = 640, 480
MAX_RADIUS_PX = 48


def prepare(col, count):
    mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
    logs = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
    scale = np.exp(np.clip(logs, -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    R = np.empty((count, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    M = R * scale[:, None, :]
    sigma = M @ np.transpose(M, (0, 2, 1))
    opacity = 1 / (1 + np.exp(-col['opacity'].astype(np.float64)))
    rgb = np.clip(0.5 + C0 * np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1), 0, 1)
    normal = R[np.arange(count), :, np.argmin(scale, 1)]
    keep = scale.max(1) < 1.0          # the far shell is background, not geometry
    return dict(mean=mean[keep], sigma=sigma[keep], opacity=opacity[keep],
                rgb=rgb[keep], normal=normal[keep], scale=scale[keep])


def render(m, Rc, t, fx, fy, cx, cy):
    cam = m['mean'] @ Rc.T + t
    z = cam[:, 2]
    vis = z > 0.1
    idx = np.nonzero(vis)[0]
    cam = cam[idx]; z = z[idx]
    S = Rc @ m['sigma'][idx] @ Rc.T
    iz = 1 / z
    j00, j11 = fx * iz, fy * iz
    j02, j12 = -fx * cam[:, 0] * iz * iz, -fy * cam[:, 1] * iz * iz
    a = j00 * j00 * S[:, 0, 0] + 2 * j00 * j02 * S[:, 0, 2] + j02 * j02 * S[:, 2, 2]
    b = j00 * j11 * S[:, 0, 1] + j00 * j12 * S[:, 0, 2] + j02 * j11 * S[:, 1, 2] + j02 * j12 * S[:, 2, 2]
    c = j11 * j11 * S[:, 1, 1] + 2 * j11 * j12 * S[:, 1, 2] + j12 * j12 * S[:, 2, 2]
    det0 = np.maximum(a * c - b * b, 1e-12)
    a += 0.25; c += 0.25
    det = a * c - b * b
    comp = np.sqrt(np.clip(det0 / np.maximum(det, 1e-12), 0, 1))
    alpha0 = m['opacity'][idx] * comp
    ia, ib, ic = c / det, -b / det, a / det
    mx = fx * cam[:, 0] * iz + cx
    my = fy * cam[:, 1] * iz + cy
    mid = 0.5 * (a + c)
    rad = np.minimum(3 * np.sqrt(mid + np.sqrt(np.maximum(mid * mid - det, 1e-9))), MAX_RADIUS_PX)
    on = (mx + rad >= 0) & (mx - rad < W) & (my + rad >= 0) & (my - rad < H) & (alpha0 >= 1 / 255)
    sel = np.nonzero(on)[0]
    order = sel[np.argsort(z[sel], kind='stable')]

    img = np.zeros((H * W, 3)); dep = np.zeros(H * W); nrm = np.zeros((H * W, 3))
    T = np.ones(H * W)
    rgb = m['rgb'][idx]; nw = m['normal'][idx] @ Rc.T
    nw *= np.where(np.sum(nw * cam, 1) > 0, -1.0, 1.0)[:, None]   # face the camera
    for start in range(0, len(order), 40000):
        o = order[start:start + 40000]
        x0 = np.clip(np.floor(mx[o] - rad[o]), 0, W - 1).astype(np.int64)
        x1 = np.clip(np.ceil(mx[o] + rad[o]), 0, W - 1).astype(np.int64)
        y0 = np.clip(np.floor(my[o] - rad[o]), 0, H - 1).astype(np.int64)
        y1 = np.clip(np.ceil(my[o] + rad[o]), 0, H - 1).astype(np.int64)
        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        n = nx * ny
        sp = np.repeat(np.arange(len(o)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        px = x0[sp] + k % nx[sp]
        py = y0[sp] + k // nx[sp]
        g = o[sp]
        dx = px + 0.5 - mx[g]; dy = py + 0.5 - my[g]
        power = -0.5 * (ia[g] * dx * dx + 2 * ib[g] * dx * dy + ic[g] * dy * dy)
        al = np.minimum(0.99, alpha0[g] * np.exp(np.minimum(power, 0)))
        ok = (power <= 0) & (al >= 1 / 255)
        px, py, g, al = px[ok], py[ok], g[ok], al[ok]
        pix = py * W + px
        s = np.argsort(pix, kind='stable')         # keeps depth order inside a pixel
        pix, g, al = pix[s], g[s], al[s]
        lt = np.log1p(-al)
        cs = np.cumsum(lt)
        first = np.r_[True, pix[1:] != pix[:-1]]
        seg = np.cumsum(first) - 1
        base = (cs - lt)[first]
        excl = cs - lt - base[seg]
        wgt = al * T[pix] * np.exp(excl)
        for ch in range(3):
            img[:, ch] += np.bincount(pix, wgt * rgb[g, ch], H * W)
            nrm[:, ch] += np.bincount(pix, wgt * nw[g, ch], H * W)
        dep += np.bincount(pix, wgt * z[g], H * W)
        upix = pix[first]
        tot = np.bincount(seg, lt)
        T[upix] *= np.exp(tot)
    acc = 1 - T
    return img.reshape(H, W, 3), dep.reshape(H, W), nrm.reshape(H, W, 3), acc.reshape(H, W)


def save(out, name, img, dep, nrm, acc):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out, name + '_colour.png'))
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-6)
    shown = (0.5 + 0.5 * n) * acc[..., None]
    Image.fromarray((np.clip(shown, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out, name + '_surface.png'))
    d = dep / np.maximum(acc, 1e-6)
    lo, hi = np.percentile(d[acc > 0.5], [2, 98]) if (acc > 0.5).any() else (0, 1)
    dd = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * (acc > 0.05)
    Image.fromarray((dd * 255).astype(np.uint8)).save(os.path.join(out, name + '_depth.png'))


def main():
    scan, out = sys.argv[1], sys.argv[2]
    frames = [int(a) for a in sys.argv[3:]]
    P.D = scan
    os.makedirs(out, exist_ok=True)
    col, count = P.load_ply(os.path.join(scan, 'model', 'model.ply'))
    m = prepare(col, count)
    poses = P.load_poses()
    fx, fy, cx, cy = P.load_intrinsics(W, H)
    keys = sorted(poses)
    if not frames:
        frames = [keys[len(keys) * i // 4] for i in range(4)]
    for f in frames:
        q, tr = poses[min(keys, key=lambda k: abs(k - f))]
        Rc = P.quat_to_matrix(q); t = np.asarray(tr, float)
        save(out, 'f%04d' % f, *render(m, Rc, t, fx, fy, cx, cy))
        # A NOVEL view: 35 cm to the right and 15 degrees to the left of
        # the recorded camera, where floaters and layered surfaces show.
        ang = np.radians(-15)
        Ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        save(out, 'f%04d_novel' % f, *render(m, Ry @ Rc, Ry @ t + np.array([-0.35, 0, 0]), fx, fy, cx, cy))
        print('rendered', f)


if __name__ == '__main__':
    main()
