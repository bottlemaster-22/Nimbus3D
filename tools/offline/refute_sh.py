"""REFUTATION harness for the sh_bands.py finding.

Independently reloads Four Marks.ply, reproduces the headline ratios, then
runs NULL CONTROLS the original did not:
  (a) per-band coefficient statistics (zero fraction, magnitude)
  (b) shuffle deg2+3 coefficients ACROSS splats -> destroys real per-splat
      view-dependent structure, preserves marginal distribution exactly.
      If the ratio survives the shuffle, the metric measures coefficient
      COUNT and the quantisation grid, not real view dependence.
  (c) count-matched comparison: deg1 (3 coeffs) vs a random 3 of the 12
      deg2+3 coeffs, removing the 4x count advantage.
"""
import numpy as np, time

PLY = r"C:\Users\Undea\Downloads\Four Marks.ply"
NDIR = 32
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = np.array([1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
               -1.0925484305920792, 0.5462742152960396])
C3 = np.array([-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
               0.3731763325901154, -0.4570457994644658, 1.445305721320277,
               -0.5900435899266435])

def load():
    f = open(PLY, 'rb'); h = b''
    while not h.endswith(b'end_header\n'): h += f.read(1)
    names, N = [], 0
    for line in h.decode().split('\n'):
        p = line.split()
        if not p: continue
        if p[0] == 'element' and p[1] == 'vertex': N = int(p[2])
        elif p[0] == 'property' and p[1] == 'float': names.append(p[2])
    d = np.frombuffer(f.read(N*len(names)*4), '<f4').reshape(N, len(names)).astype(np.float64)
    return {n: d[:, i] for i, n in enumerate(names)}, N, d

def fib(n):
    i = np.arange(n); phi = np.arccos(1 - 2*(i+0.5)/n)
    th = np.pi*(1+5**0.5)*i
    return np.stack([np.sin(phi)*np.cos(th), np.sin(phi)*np.sin(th), np.cos(phi)], 1)

def basis(dirs):
    """(D,15) basis values for deg1(3), deg2(5), deg3(7)."""
    x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
    xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
    B = np.stack([
        -C1*y, C1*z, -C1*x,
        C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy),
        C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy),
        C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
        C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy)], 1)
    return B  # (D,15)

def band_ptp_rms(rest, B, idx):
    """rest (N,3,15), B (D,15), idx list of coeff indices -> ptp/rms (N,) mean over RGB."""
    v = np.einsum('ncj,dj->ncd', rest[:, :, idx], B[:, idx])  # (N,3,D)
    return (v.max(2)-v.min(2)).mean(1), np.sqrt((v**2).mean(2)).mean(1)

def main():
    t = time.time()
    col, N, d = load()
    print(f"loaded {N} splats {time.time()-t:.1f}s")
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1)
    rest = np.stack([np.stack([col[f'f_rest_{c*15+k}'] for k in range(15)], 1)
                     for c in range(3)], 1)  # (N,3,15)
    print("rest shape", rest.shape)

    print("\n=== (a) PER-BAND COEFFICIENT STATISTICS ===")
    for name, idx in [("deg1", range(0,3)), ("deg2", range(3,8)), ("deg3", range(8,15))]:
        s = rest[:, :, list(idx)]
        print(f"{name}: n={s.size:>9}  zero={np.mean(s==0)*100:6.2f}%  "
              f"mean|c|={np.abs(s).mean():.5f}  max|c|={np.abs(s).max():.4f}  "
              f"rms={np.sqrt((s**2).mean()):.5f}")
    print(f"per-splat: splats with ALL 45 rest coeffs zero = "
          f"{np.mean(np.all(rest==0,axis=(1,2)))*100:.2f}%")

    B = basis(fib(NDIR))
    dcmag = np.abs(C0*dc).mean(1)
    eps = 1e-6
    I1, I23 = list(range(0,3)), list(range(3,15))

    p1, r1 = band_ptp_rms(rest, B, I1)
    p23, r23 = band_ptp_rms(rest, B, I23)
    pall, _ = band_ptp_rms(rest, B, list(range(15)))
    print("\n=== (b) REPRODUCE HEADLINE (real coefficients) ===")
    print(f"median deg1  ptp/|DC|          = {np.median(p1/(dcmag+eps)):.4f}   [claim 0.369]")
    print(f"median deg2+3 ptp/|DC|         = {np.median(p23/(dcmag+eps)):.4f}   [claim 0.411]")
    print(f"median deg2+3/(total) ptp      = {np.median(p23/(pall+eps)):.4f}   [claim 0.621]")
    print(f"median deg2+3 RMS / deg1 RMS   = {np.median(r23/(r1+eps)):.4f}   [claim 0.982]")

    print("\n=== (c) NULL CONTROL: shuffle deg2+3 coeffs ACROSS splats ===")
    print("    (destroys real per-splat structure, keeps the exact marginal")
    print("     distribution, quantisation grid and coefficient count)")
    rng = np.random.default_rng(0)
    sh = rest.copy()
    for c in range(3):
        for k in range(3, 15):
            sh[:, c, k] = rest[rng.permutation(N), c, k]
    p23s, r23s = band_ptp_rms(sh, B, I23)
    print(f"SHUFFLED median deg2+3 ptp/|DC|   = {np.median(p23s/(dcmag+eps)):.4f}"
          f"   (real {np.median(p23/(dcmag+eps)):.4f})")
    palls, _ = band_ptp_rms(sh, B, list(range(15)))
    print(f"SHUFFLED median deg2+3/(total)    = {np.median(p23s/(palls+eps)):.4f}"
          f"   (real {np.median(p23/(pall+eps)):.4f})")
    print(f"SHUFFLED median deg2+3RMS/deg1RMS = {np.median(r23s/(r1+eps)):.4f}"
          f"   (real {np.median(r23/(r1+eps)):.4f})")

    print("\n=== (d) COUNT-MATCHED: deg1 (3 coeffs) vs random 3 of deg2+3 ===")
    for trial in range(3):
        pick = sorted(rng.choice(np.arange(3,15), 3, replace=False).tolist())
        p3, r3 = band_ptp_rms(rest, B, pick)
        print(f"  coeffs {pick}: median ptp/|DC| = {np.median(p3/(dcmag+eps)):.4f}"
              f"  vs deg1 {np.median(p1/(dcmag+eps)):.4f}   "
              f"RMS ratio {np.median(r3/(r1+eps)):.4f}")

if __name__ == "__main__":
    main()
