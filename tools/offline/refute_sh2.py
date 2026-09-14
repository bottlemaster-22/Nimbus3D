"""Part 2: is the deg2+3 content REAL, or the quantisation grid + 12-vs-3 count?
Plus the hemisphere restriction the original's own caveat asked for, using the
Gaussian's own disc normal (min-scale axis) instead of the full sphere."""
import numpy as np, time
PLY = r"C:\Users\Undea\Downloads\Four Marks.ply"
C0 = 0.28209479177387814; C1 = 0.4886025119029199
C2 = np.array([1.0925484305920792,-1.0925484305920792,0.31539156525252005,
               -1.0925484305920792,0.5462742152960396])
C3 = np.array([-0.5900435899266435,2.890611442640554,-0.4570457994644658,
               0.3731763325901154,-0.4570457994644658,1.445305721320277,
               -0.5900435899266435])
def load():
    f=open(PLY,'rb'); h=b''
    while not h.endswith(b'end_header\n'): h+=f.read(1)
    names,N=[],0
    for l in h.decode().split('\n'):
        p=l.split()
        if not p: continue
        if p[0]=='element' and p[1]=='vertex': N=int(p[2])
        elif p[0]=='property' and p[1]=='float': names.append(p[2])
    d=np.frombuffer(f.read(N*len(names)*4),'<f4').reshape(N,len(names)).astype(np.float64)
    return {n:d[:,i] for i,n in enumerate(names)}, N
def fib(n):
    i=np.arange(n); phi=np.arccos(1-2*(i+0.5)/n); th=np.pi*(1+5**0.5)*i
    return np.stack([np.sin(phi)*np.cos(th),np.sin(phi)*np.sin(th),np.cos(phi)],1)
def basis(D):
    x,y,z=D[:,0],D[:,1],D[:,2]; xx,yy,zz,xy,yz,xz=x*x,y*y,z*z,x*y,y*z,x*z
    return np.stack([-C1*y,C1*z,-C1*x,
        C2[0]*xy,C2[1]*yz,C2[2]*(2*zz-xx-yy),C2[3]*xz,C2[4]*(xx-yy),
        C3[0]*y*(3*xx-yy),C3[1]*xy*z,C3[2]*y*(4*zz-xx-yy),
        C3[3]*z*(2*zz-3*xx-3*yy),C3[4]*x*(4*zz-xx-yy),
        C3[5]*z*(xx-yy),C3[6]*x*(xx-3*yy)],1)

col,N=load()
dc=np.stack([col['f_dc_0'],col['f_dc_1'],col['f_dc_2']],1)
rest=np.stack([np.stack([col[f'f_rest_{c*15+k}'] for k in range(15)],1) for c in range(3)],1)
op=1/(1+np.exp(-np.clip(col['opacity'],-30,30)))

print("=== (e) HOW MANY SPLATS CARRY ANY HIGHER-BAND CONTENT AT ALL ===")
d1=rest[:,:,0:3]; d2=rest[:,:,3:8]; d3=rest[:,:,8:15]; d23=rest[:,:,3:15]
for nm,s in [("deg1",d1),("deg2",d2),("deg3",d3),("deg2+3",d23)]:
    any_nz=np.any(s!=0,axis=(1,2))
    print(f"  {nm:7s}: splats with ANY nonzero coeff = {any_nz.mean()*100:6.2f}%")
print(f"  splats where deg2+3 is ENTIRELY zero          = {np.all(d23==0,axis=(1,2)).mean()*100:6.2f}%")
print(f"  ...of those, deg1 nonzero                     = "
      f"{np.mean(np.any(d1!=0,axis=(1,2))[np.all(d23==0,axis=(1,2))])*100:6.2f}%")

print("\n=== (f) THE QUANTISATION GRID IS COARSER THAN THE EFFECT MEASURED ===")
Dfine=fib(4000); Bf=basis(Dfine)
peak=np.abs(Bf).max(0)
step=0.0625
print("  one quantisation step (0.0625) of each band, in 0-255 colour units:")
for nm,sl in [("deg1",slice(0,3)),("deg2",slice(3,8)),("deg3",slice(8,15))]:
    v=peak[sl]*step*255
    print(f"    {nm}: peak swing per single step = {v.min():.1f} .. {v.max():.1f} /255")
print("  => any coefficient the exporter rounds UP to one step lands a colour")
print("     swing of this size; anything smaller is rounded to exactly zero.")

print("\n=== (g) CLAMPED-COLOUR DELTA, WITH A SHUFFLE NULL ===")
Dv=fib(32); B=basis(Dv)
def colour(rest_):
    c=(C0*dc)[:,:,None]+0.5+np.einsum('ncj,dj->ncd',rest_,B)
    return np.clip(c,0,1)
full=colour(rest)
r1only=rest.copy(); r1only[:,:,3:]=0
d1c=colour(r1only)
delta=np.abs(full-d1c)
print(f"  REAL   : mean |delta| = {delta.mean()*255:.2f}/255  median {np.median(delta)*255:.2f}/255"
      f"  >1/255 {np.mean(delta>1/255)*100:.1f}%  >4/255 {np.mean(delta>4/255)*100:.1f}%")
rng=np.random.default_rng(1); sh=rest.copy()
for c in range(3):
    for k in range(3,15): sh[:,c,k]=rest[rng.permutation(N),c,k]
fs=colour(sh); ds=np.abs(fs-d1c)
print(f"  SHUFFLE: mean |delta| = {ds.mean()*255:.2f}/255  median {np.median(ds)*255:.2f}/255"
      f"  >1/255 {np.mean(ds>1/255)*100:.1f}%  >4/255 {np.mean(ds>4/255)*100:.1f}%")
print("  (shuffled deg2+3 carries ZERO real view-dependent information)")

print("\n=== (h) HEMISPHERE RESTRICTION (the caveat, measured) ===")
q=np.stack([col['rot_0'],col['rot_1'],col['rot_2'],col['rot_3']],1)
q=q/np.linalg.norm(q,axis=1,keepdims=True)
w,xq,yq,zq=q[:,0],q[:,1],q[:,2],q[:,3]
Rm=np.stack([1-2*(yq*yq+zq*zq),2*(xq*yq-w*zq),2*(xq*zq+w*yq),
             2*(xq*yq+w*zq),1-2*(xq*xq+zq*zq),2*(yq*zq-w*xq),
             2*(xq*zq-w*yq),2*(yq*zq+w*xq),1-2*(xq*xq+yq*yq)],1).reshape(N,3,3)
sc=np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1)
axis=np.argmin(sc,axis=1)
nrm=Rm[np.arange(N),:,axis]
nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)
dot=nrm@Dv.T                      # (N,D)
vis=np.abs(dot)>=0.0              # sanity
front=dot>0                       # the hemisphere the "front" face sees
print(f"  mean directions per splat in front hemisphere: {front.sum(1).mean():.1f} of {len(Dv)}")
eps=1e-6; dcm=np.abs(C0*dc).mean(1)
def ptp_masked(idx,mask):
    v=np.einsum('ncj,dj->ncd',rest[:,:,idx],B[:,idx])
    vm=np.where(mask[:,None,:],v,np.nan)
    return (np.nanmax(vm,2)-np.nanmin(vm,2)).mean(1)
I1=list(range(0,3)); I23=list(range(3,15))
allm=np.ones((N,len(Dv)),bool)
for nm,m in [("FULL SPHERE",allm),("FRONT HEMISPHERE",front)]:
    p1=ptp_masked(I1,m); p23=ptp_masked(I23,m)
    print(f"  {nm:17s}: median deg1 ptp/|DC| = {np.median(p1/(dcm+eps)):.4f}   "
          f"deg2+3 ptp/|DC| = {np.median(p23/(dcm+eps)):.4f}   "
          f"ratio23/(1+23) = {np.median(p23/(p1+p23+eps)):.4f}")
