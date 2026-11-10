import numpy as np
PLY=r"C:\Users\Undea\Downloads\Four Marks.ply"
C0=0.28209479177387814;C1=0.4886025119029199
C2=np.array([1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396])
C3=np.array([-0.5900435899266435,2.890611442640554,-0.4570457994644658,0.3731763325901154,-0.4570457994644658,1.445305721320277,-0.5900435899266435])
def load():
    f=open(PLY,'rb');h=b''
    while not h.endswith(b'end_header\n'):h+=f.read(1)
    names,N=[],0
    for l in h.decode().split('\n'):
        p=l.split()
        if not p:continue
        if p[0]=='element' and p[1]=='vertex':N=int(p[2])
        elif p[0]=='property' and p[1]=='float':names.append(p[2])
    d=np.frombuffer(f.read(N*len(names)*4),'<f4').reshape(N,len(names)).astype(np.float64)
    return {n:d[:,i] for i,n in enumerate(names)},N
def fib(n):
    i=np.arange(n);phi=np.arccos(1-2*(i+0.5)/n);th=np.pi*(1+5**0.5)*i
    return np.stack([np.sin(phi)*np.cos(th),np.sin(phi)*np.sin(th),np.cos(phi)],1)
def basis(D):
    x,y,z=D[:,0],D[:,1],D[:,2];xx,yy,zz,xy,yz,xz=x*x,y*y,z*z,x*y,y*z,x*z
    return np.stack([-C1*y,C1*z,-C1*x,C2[0]*xy,C2[1]*yz,C2[2]*(2*zz-xx-yy),C2[3]*xz,C2[4]*(xx-yy),
      C3[0]*y*(3*xx-yy),C3[1]*xy*z,C3[2]*y*(4*zz-xx-yy),C3[3]*z*(2*zz-3*xx-3*yy),
      C3[4]*x*(4*zz-xx-yy),C3[5]*z*(xx-yy),C3[6]*x*(xx-3*yy)],1)
col,N=load()
dc=np.stack([col['f_dc_0'],col['f_dc_1'],col['f_dc_2']],1)
rest=np.stack([np.stack([col[f'f_rest_{c*15+k}'] for k in range(15)],1) for c in range(3)],1)
op=1/(1+np.exp(-np.clip(col['opacity'],-30,30)))
D=fib(32);B=basis(D)

print("=== (i) WHICH CLAMP REPRODUCES THEIR median 3.92 / 66.5%% / 49.6%% ? ===")
base=(C0*dc)[:,:,None]+0.5
full_raw=base+np.einsum('ncj,dj->ncd',rest,B)
d1_raw=base+np.einsum('ncj,dj->ncd',rest[:,:,0:3],B[:,0:3])
for nm,fn in [("clip(0,1) both",lambda v:np.clip(v,0,1)),
              ("clamp>=0 only (trainer shader)",lambda v:np.maximum(v,0)),
              ("no clamp",lambda v:v)]:
    dd=np.abs(fn(full_raw)-fn(d1_raw))
    print(f"  {nm:32s} mean {dd.mean()*255:5.2f}  median {np.median(dd)*255:5.2f}"
          f"  >1/255 {np.mean(dd>1/255)*100:5.1f}%  >4/255 {np.mean(dd>4/255)*100:5.1f}%")
print("  [finding claims: mean 6.04  median 3.92  >1/255 66.5%  >4/255 49.6%]")

print("\n=== (j) THE |DC| DENOMINATOR IS |colour-0.5|, NOT THE COLOUR ===")
dcm=np.abs(C0*dc).mean(1)
print(f"  |C0*f_dc| mean over RGB: min {dcm.min():.5f}  p1 {np.percentile(dcm,1):.5f}"
      f"  median {np.median(dcm):.4f}")
for t in [0.02,0.05,0.10]:
    print(f"  splats with |DC| < {t}: {np.mean(dcm<t)*100:5.2f}%  (mid-grey splats; ratio blows up)")
eps=1e-6
def ptp(idx):
    v=np.einsum('ncj,dj->ncd',rest[:,:,idx],B[:,idx]);return (v.max(2)-v.min(2)).mean(1)
p1=ptp(list(range(0,3)));p23=ptp(list(range(3,15)))
lum=(C0*dc+0.5).mean(1)
print(f"  normalised by |DC|=|colour-0.5| : deg1 {np.median(p1/(dcm+eps)):.4f}  deg2+3 {np.median(p23/(dcm+eps)):.4f}")
print(f"  normalised by ACTUAL colour     : deg1 {np.median(p1/(np.abs(lum)+eps)):.4f}  deg2+3 {np.median(p23/(np.abs(lum)+eps)):.4f}")

print("\n=== (k) NULL CONTROL ON THE OPACITY>0.5 SUBSET (their 0.474 vs 0.381) ===")
m=op>0.5
print(f"  subset {m.sum()}/{N} = {m.mean()*100:.1f}%  [claim 228,890 / 53.5%]")
print(f"  REAL   : deg1 {np.median(p1[m]/(dcm[m]+eps)):.4f}   deg2+3 {np.median(p23[m]/(dcm[m]+eps)):.4f}")
rng=np.random.default_rng(2);sh=rest.copy()
for c in range(3):
    for k in range(3,15): sh[:,c,k]=rest[rng.permutation(N),c,k]
vs=np.einsum('ncj,dj->ncd',sh[:,:,3:15],B[:,3:15]);p23s=(vs.max(2)-vs.min(2)).mean(1)
print(f"  SHUFFLE: deg1 {np.median(p1[m]/(dcm[m]+eps)):.4f}   deg2+3 {np.median(p23s[m]/(dcm[m]+eps)):.4f}")

print("\n=== (l) PER-COEFFICIENT STRENGTH (count bias removed) ===")
for nm,sl in [("deg1",slice(0,3)),("deg2",slice(3,8)),("deg3",slice(8,15))]:
    s=rest[:,:,sl]
    print(f"  {nm}: {s.shape[2]:2d} coeffs  rms/coeff {np.sqrt((s**2).mean()):.5f}"
          f"  mean|c| {np.abs(s).mean():.5f}  zero {np.mean(s==0)*100:.1f}%")
