"""MEASURE how much of the Scaniverse PLY's rendered colour comes from SH
degree 2 and 3, versus degree 0 (DC) and degree 1.

Not an estimate: loads every one of the 427,710 splats' real f_dc/f_rest
coefficients and evaluates the actual real-SH basis (the same basis the
INRIA/Scaniverse PLY convention and diff-gaussian-rasterization use) at a
Fibonacci-sphere set of view directions per splat, then measures the
per-splat, per-channel contribution of each band group as it would actually
appear in a rendered pixel.

No camera poses are used (the Scaniverse capture is not registered to our
prepass poses), so 'view direction' means uniform sampling over the full
sphere of possible viewing angles for that splat, which is the correct way to
ask "how much does this splat's appearance change with view angle" without
the confound of which directions any particular scan happened to visit.

Run with: cd tools/offline && python -u sh_bands.py
"""
import numpy as np
import time

PLY_PATH = r"C:\Users\Undea\Downloads\Four Marks.ply"
N_DIRS = 32          # Fibonacci-sphere view directions per splat
SEED = 0

# Real SH basis constants, degree 0..3 (Zhang / diff-gaussian-rasterization
# convention -- the standard the INRIA PLY f_rest layout is written for).
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = np.array([
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
])
C3 = np.array([
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
])


def load_ply(path):
    f = open(path, 'rb')
    header = b''
    while not header.endswith(b'end_header\n'):
        header += f.read(1)
    names, count = [], 0
    for line in header.decode().split('\n'):
        p = line.split()
        if not p:
            continue
        if p[0] == 'element' and p[1] == 'vertex':
            count = int(p[2])
        elif p[0] == 'property' and p[1] == 'float':
            names.append(p[2])
    data = np.frombuffer(f.read(count * len(names) * 4), dtype='<f4')
    data = data.reshape(count, len(names)).astype(np.float64)
    col = {n: data[:, i] for i, n in enumerate(names)}
    return col, count


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)  # (n, 3)


def eval_bands(dc, rest, dirs):
    """dc: (N,) one channel's DC coeff. rest: (N,15) one channel's f_rest,
    already in [deg1(3), deg2(5), deg3(7)] order. dirs: (D,3) unit vectors.

    Returns dc_val (N,) constant, and deg1/deg2/deg3 (N,D) contributions.
    """
    x = dirs[:, 0][None, :]
    y = dirs[:, 1][None, :]
    z = dirs[:, 2][None, :]

    dc_val = C0 * dc  # (N,), direction-independent

    r1 = rest[:, 0:3]
    deg1 = (-C1 * y * r1[:, 0:1] + C1 * z * r1[:, 1:2] - C1 * x * r1[:, 2:3])

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    r2 = rest[:, 3:8]
    deg2 = (C2[0] * xy * r2[:, 0:1] + C2[1] * yz * r2[:, 1:2]
            + C2[2] * (2 * zz - xx - yy) * r2[:, 2:3]
            + C2[3] * xz * r2[:, 3:4] + C2[4] * (xx - yy) * r2[:, 4:5])

    r3 = rest[:, 8:15]
    deg3 = (C3[0] * y * (3 * xx - yy) * r3[:, 0:1]
            + C3[1] * xy * z * r3[:, 1:2]
            + C3[2] * y * (4 * zz - xx - yy) * r3[:, 2:3]
            + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * r3[:, 3:4]
            + C3[4] * x * (4 * zz - xx - yy) * r3[:, 4:5]
            + C3[5] * z * (xx - yy) * r3[:, 5:6]
            + C3[6] * x * (xx - 3 * yy) * r3[:, 6:7])

    return dc_val, deg1, deg2, deg3


def main():
    t0 = time.time()
    col, count = load_ply(PLY_PATH)
    print(f"loaded {count} splats in {time.time()-t0:.1f}s")

    dirs = fibonacci_sphere(N_DIRS)

    opacity_raw = col['opacity']
    finite = np.isfinite(opacity_raw)
    sigmoid_op = np.zeros(count)
    sigmoid_op[finite] = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw[finite], -30, 30)))
    sigmoid_op[~finite] = 1.0  # +inf logits -> fully opaque
    print(f"opacity: {(~finite).sum()} non-finite logits (-> treated as opaque), "
          f"mean sigmoid(opacity) = {sigmoid_op.mean():.3f}")

    channels = ['dc_0', 'dc_1', 'dc_2']
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], axis=1)  # (N,3)
    rest = np.stack(
        [np.stack([col[f'f_rest_{c*15+k}'] for k in range(15)], axis=1)
         for c in range(3)],
        axis=1)  # (N, 3 channels, 15 coeffs)

    # Per-splat, per-channel accumulators across the D view directions.
    dc_mag_all = np.zeros((count, 3))
    deg1_ptp_all = np.zeros((count, 3))
    deg2_ptp_all = np.zeros((count, 3))
    deg3_ptp_all = np.zeros((count, 3))
    deg23_ptp_all = np.zeros((count, 3))
    total_ptp_all = np.zeros((count, 3))
    deg1_rms_all = np.zeros((count, 3))
    deg23_rms_all = np.zeros((count, 3))

    for c in range(3):
        dc_val, deg1, deg2, deg3 = eval_bands(dc[:, c], rest[:, c, :], dirs)
        dc_mag_all[:, c] = np.abs(dc_val)
        deg1_ptp_all[:, c] = deg1.max(axis=1) - deg1.min(axis=1)
        deg2_ptp_all[:, c] = deg2.max(axis=1) - deg2.min(axis=1)
        deg3_ptp_all[:, c] = deg3.max(axis=1) - deg3.min(axis=1)
        deg23 = deg2 + deg3
        deg23_ptp_all[:, c] = deg23.max(axis=1) - deg23.min(axis=1)
        total = deg1 + deg2 + deg3
        total_ptp_all[:, c] = total.max(axis=1) - total.min(axis=1)
        deg1_rms_all[:, c] = np.sqrt((deg1 ** 2).mean(axis=1))
        deg23_rms_all[:, c] = np.sqrt((deg23 ** 2).mean(axis=1))

    # Collapse channels: use the mean over R,G,B (matches how a viewer
    # perceives overall colour shift, not just one channel).
    dc_mag = dc_mag_all.mean(axis=1)
    deg1_ptp = deg1_ptp_all.mean(axis=1)
    deg2_ptp = deg2_ptp_all.mean(axis=1)
    deg3_ptp = deg3_ptp_all.mean(axis=1)
    deg23_ptp = deg23_ptp_all.mean(axis=1)
    total_ptp = total_ptp_all.mean(axis=1)
    deg1_rms = deg1_rms_all.mean(axis=1)
    deg23_rms = deg23_rms_all.mean(axis=1)

    eps = 1e-6

    print()
    print("=" * 70)
    print("UNWEIGHTED (every splat counted once, including near-invisible ones)")
    print("=" * 70)
    ratio_23_over_dc = deg23_ptp / (dc_mag + eps)
    ratio_1_over_dc = deg1_ptp / (dc_mag + eps)
    ratio_23_over_total = deg23_ptp / (total_ptp + eps)
    print(f"median deg1  view-dep range / |DC|      = {np.median(ratio_1_over_dc):.4f}")
    print(f"median deg2+3 view-dep range / |DC|      = {np.median(ratio_23_over_dc):.4f}")
    print(f"median deg2+3 range / (deg1+2+3 range)  = {np.median(ratio_23_over_total):.4f}")
    print(f"mean   deg2+3 range / (deg1+2+3 range)  = {np.mean(ratio_23_over_total):.4f}")
    print(f"p90    deg2+3 range / (deg1+2+3 range)  = {np.percentile(ratio_23_over_total,90):.4f}")
    ratio_rms = deg23_rms / (deg1_rms + eps)
    print(f"[robustness check, RMS not peak-to-peak] median deg2+3 RMS / deg1 RMS = "
          f"{np.median(ratio_rms):.4f}")
    print(f"[robustness check, RMS not peak-to-peak] mean   deg2+3 RMS / deg1 RMS = "
          f"{np.mean(ratio_rms):.4f}")

    print()
    print("=" * 70)
    print("OPACITY-WEIGHTED (weight = sigmoid(opacity), the splats that actually")
    print("contribute alpha*T to a rendered pixel count proportionally more)")
    print("=" * 70)
    w = sigmoid_op
    wsum = w.sum()

    def wmean(v):
        return float((v * w).sum() / wsum)

    print(f"opacity-weighted mean |DC|                = {wmean(dc_mag):.5f}")
    print(f"opacity-weighted mean deg1 range           = {wmean(deg1_ptp):.5f}")
    print(f"opacity-weighted mean deg2+3 range         = {wmean(deg23_ptp):.5f}")
    print(f"opacity-weighted mean deg2+3 range / |DC|  = {wmean(deg23_ptp)/ (wmean(dc_mag)+eps):.4f}")
    print(f"opacity-weighted mean deg1 range / |DC|    = {wmean(deg1_ptp)/ (wmean(dc_mag)+eps):.4f}")
    print(f"opacity-weighted mean deg2+3/(1+2+3) range = {wmean(deg23_ptp)/(wmean(total_ptp)+eps):.4f}")

    print()
    print("=" * 70)
    print("RESTRICTED to splats that actually matter for a render")
    print("(sigmoid(opacity) > 0.5, i.e. the splat is more opaque than not)")
    print("=" * 70)
    mask = sigmoid_op > 0.5
    print(f"{mask.sum()} / {count} splats ({100*mask.mean():.1f}%) pass this filter")
    print(f"median |DC|                 = {np.median(dc_mag[mask]):.5f}")
    print(f"median deg1 range           = {np.median(deg1_ptp[mask]):.5f}")
    print(f"median deg2+3 range         = {np.median(deg23_ptp[mask]):.5f}")
    print(f"median deg2+3 range / |DC|  = {np.median(deg23_ptp[mask]/(dc_mag[mask]+eps)):.4f}")
    print(f"median deg1 range / |DC|    = {np.median(deg1_ptp[mask]/(dc_mag[mask]+eps)):.4f}")
    print(f"median deg2+3/(1+2+3) range = {np.median(deg23_ptp[mask]/(total_ptp[mask]+eps)):.4f}")

    print()
    print("=" * 70)
    print("CLAMPED-COLOUR TEST: what does dropping degree 2+3 actually change")
    print("in the final [0,1]-clamped RGB colour, in 8-bit units?")
    print("(colour = clamp(dc_val + deg1[+deg2+deg3] + 0.5, 0, 1) per INRIA convention)")
    print("=" * 70)
    diffs_8bit = []
    for c in range(3):
        dc_val, deg1, deg2, deg3 = eval_bands(dc[:, c], rest[:, c, :], dirs)
        full = np.clip(dc_val[:, None] + deg1 + deg2 + deg3 + 0.5, 0, 1)
        deg1_only = np.clip(dc_val[:, None] + deg1 + 0.5, 0, 1)
        diffs_8bit.append(np.abs(full - deg1_only) * 255.0)
    diffs_8bit = np.mean(diffs_8bit, axis=0)  # (N, D) averaged over channel
    diffs_flat = diffs_8bit.reshape(-1)
    w_flat = np.repeat(sigmoid_op, N_DIRS)
    print(f"mean |delta| across all splats x directions           = {diffs_flat.mean():.3f} / 255")
    print(f"median |delta|                                        = {np.median(diffs_flat):.3f} / 255")
    print(f"opacity-weighted mean |delta|                          = "
          f"{(diffs_flat*w_flat).sum()/w_flat.sum():.3f} / 255")
    print(f"fraction of (splat,direction) samples with |delta|>1/255 = "
          f"{(diffs_flat > 1.0).mean()*100:.2f}%")
    print(f"fraction of (splat,direction) samples with |delta|>4/255 = "
          f"{(diffs_flat > 4.0).mean()*100:.2f}%")
    mask_rep = np.repeat(mask, N_DIRS)
    print(f"opaque-only (sigmoid>0.5) mean |delta|                 = {diffs_flat[mask_rep].mean():.3f} / 255")
    print(f"opaque-only fraction with |delta|>1/255                = "
          f"{(diffs_flat[mask_rep] > 1.0).mean()*100:.2f}%")

    print()
    print(f"total time {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
