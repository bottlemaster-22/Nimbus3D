"""Sparsity-preserving null for the deg2+3 magnitude statistic.

The verifier's null permuted EACH coefficient independently across splats.
That destroys per-splat zero-clustering, which is the dominant structure in a
quantised f_rest block. This script measures how much of the null's inflation
is that artifact alone.
"""
import numpy as np, sys, time

PLY = r"C:/Users/Undea/Downloads/Four Marks.ply"

def load(path):
    with open(path,'rb') as f:
        hdr=b''
        while b'end_header' not in hdr:
            hdr += f.readline()
        lines=hdr.decode('ascii',errors='replace').splitlines()
        n=0; props=[]
        for L in lines:
            if L.startswith('element vertex'): n=int(L.split()[-1])
            if L.startswith('property'):
                t=L.split()
                props.append((t[1],t[2]))
        fmt=[('f4' if p[0]=='float' else 'u1',p[1]) for p in props]
        dt=np.dtype([(nm, ty) for ty,nm in fmt])
        data=np.fromfile(f,dtype=dt,count=n)
    return data, [nm for _,nm in fmt]

t0=time.time()
d,names = load(PLY)
n=len(d)
dc = np.stack([d['f_dc_0'],d['f_dc_1'],d['f_dc_2']],1).astype(np.float64)
rest_names=[f'f_rest_{i}' for i in range(45)]
rest = np.stack([d[k] for k in rest_names],1).astype(np.float64)   # (n,45)
print(f"{n} splats loaded in {time.time()-t0:.1f}s")

# PLY layout: f_rest is [coef][channel] flattened as coef-major over 15 coefs x 3 ch
# standard 3DGS export: f_rest_{i} with i = c*15 + k  (channel-major)
# reshape to (n, 3, 15) then transpose -> (n,15,3)
R = rest.reshape(n,3,15).transpose(0,2,1)   # (n, 15 coefs, 3 ch)
deg1 = R[:,0:3,:]      # coefs 1..3
deg23 = R[:,3:15,:]    # coefs 4..15

C0=0.28209479177387814
C1=0.4886025119029199
C2=np.array([1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396])
C3=np.array([-0.5900435899266435,2.890611442640554,-0.4570457994644658,0.3731763325901154,
             -0.4570457994644658,1.445305721320277,-0.5900435899266435])

def basis(dirs):
    x,y,z=dirs[:,0],dirs[:,1],dirs[:,2]
    b=np.zeros((len(dirs),15))
    b[:,0]=-C1*y; b[:,1]=C1*z; b[:,2]=-C1*x
    xx,yy,zz,xy,yz,xz=x*x,y*y,z*z,x*y,y*z,x*z
    b[:,3]=C2[0]*xy; b[:,4]=C2[1]*yz; b[:,5]=C2[2]*(2*zz-xx-yy)
    b[:,6]=C2[3]*xz; b[:,7]=C2[4]*(xx-yy)
    b[:,8]=C3[0]*y*(3*xx-yy); b[:,9]=C3[1]*xy*z; b[:,10]=C3[2]*y*(4*zz-xx-yy)
    b[:,11]=C3[3]*z*(2*zz-3*xx-3*yy); b[:,12]=C3[4]*x*(4*zz-xx-yy)
    b[:,13]=C3[5]*z*(xx-yy); b[:,14]=C3[6]*x*(xx-3*yy)
    return b

# 32 Fibonacci directions
K=32
i=np.arange(K)+0.5
phi=np.arccos(1-2*i/K); th=np.pi*(1+5**0.5)*i
dirs=np.stack([np.cos(th)*np.sin(phi),np.sin(th)*np.sin(phi),np.cos(phi)],1)
B=basis(dirs)     # (32,15)

absdc = np.abs(dc).mean(1)          # |DC| proxy: per the finding, |colour-0.5| = |C0*dc|
absdc = np.maximum(C0*absdc, 1e-8)

def ptp_ratio(coefs, idx):
    # coefs (n, m, 3); idx = which basis columns
    v = np.einsum('kc,ncj->nkj', B[:,idx], coefs)   # (n,32,3)
    r = (v.max(1)-v.min(1)).mean(1)                  # mean over channels of ptp
    return r/absdc

d1 = ptp_ratio(deg1, np.arange(0,3))
d23= ptp_ratio(deg23, np.arange(3,15))
print(f"REAL   median deg1 ptp/|DC| = {np.median(d1):.4f}   deg2+3 = {np.median(d23):.4f}")

zeros_per_coef = (deg23.reshape(n,-1)==0).mean()
allzero_real = (np.abs(deg23).reshape(n,-1).max(1)==0).mean()
print(f"f_rest deg2+3: exact-zero fraction per coefficient = {zeros_per_coef:.4f}")
print(f"REAL   fraction of splats with deg2+3 IDENTICALLY zero = {allzero_real:.4f}")

rng=np.random.default_rng(0)
# NULL A - the verifier's: permute every coefficient independently
A = deg23.copy().reshape(n,-1)
for c in range(A.shape[1]):
    A[:,c] = A[rng.permutation(n),c]
A = A.reshape(n,12,3)
allzero_A = (np.abs(A).reshape(n,-1).max(1)==0).mean()
d23A = ptp_ratio(A, np.arange(3,15))
print(f"NULL A (per-coefficient shuffle, the verifier's): median = {np.median(d23A):.4f}"
      f"   all-zero splats = {allzero_A:.4f}")

# NULL B - permute whole per-splat deg2+3 blocks (preserves sparsity structure)
p = rng.permutation(n)
Bn = deg23[p]
d23B = ptp_ratio(Bn, np.arange(3,15))
print(f"NULL B (whole-block shuffle, sparsity preserved):  median = {np.median(d23B):.4f}"
      f"   all-zero splats = {(np.abs(Bn).reshape(n,-1).max(1)==0).mean():.4f}")

# NULL C - the honest one: randomise the SIGNS only. Preserves every magnitude,
# every zero, every splat's sparsity; destroys the fitted angular structure.
sgn = rng.choice([-1.0,1.0], size=deg23.shape)
Cn = deg23*sgn
d23C = ptp_ratio(Cn, np.arange(3,15))
print(f"NULL C (sign randomisation, all structure but angle): median = {np.median(d23C):.4f}")

# same three nulls applied to DEGREE 1, the band nobody disputes is real
A1 = deg1.copy().reshape(n,-1)
for c in range(A1.shape[1]): A1[:,c]=A1[rng.permutation(n),c]
d1A = ptp_ratio(A1.reshape(n,3,3), np.arange(0,3))
sgn1 = rng.choice([-1.0,1.0], size=deg1.shape)
d1C = ptp_ratio(deg1*sgn1, np.arange(0,3))
print(f"CONTROL deg1 under NULL A = {np.median(d1A):.4f}  under NULL C = {np.median(d1C):.4f}"
      f"   (real {np.median(d1):.4f})")
