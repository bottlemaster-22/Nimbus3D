"""Pose-free depth-complexity probe. Renders both models with an IDENTICAL
synthetic camera rig (same intrinsics, same distance-to-scene statistics) and
counts, per pixel, how many Gaussians are evaluated / clear alpha / composite.
"""
import sv_load, numpy as np, sys
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
W,H = 720,540
FX = 1381.9*W/1920.0; FY=FX; CX=W/2.0; CY=H/2.0
F2D = 0.25
MIN_A = 1.0/255.0

def build(col, keep, flipYZ):
    sy = -1.0 if flipYZ else 1.0
    P = np.stack([col['x'], sy*col['y'], sy*col['z']],1).astype(np.float64)[keep]
    L = np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64)[keep],-12,3)
    S = np.exp(L)
    if flipYZ:
        q = np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)[keep]
    else:
        q = np.stack([col['rot_1'], col['rot_2'], col['rot_3'],col['rot_0']],1).astype(np.float64)[keep]
    q = q/np.linalg.norm(q,axis=1,keepdims=True)
    x,y,z,w = q.T
    Rm=np.empty((len(q),3,3))
    Rm[:,0,0]=1-2*(y*y+z*z); Rm[:,0,1]=2*(x*y-w*z); Rm[:,0,2]=2*(x*z+w*y)
    Rm[:,1,0]=2*(x*y+w*z); Rm[:,1,1]=1-2*(x*x+z*z); Rm[:,1,2]=2*(y*z-w*x)
    Rm[:,2,0]=2*(x*z-w*y); Rm[:,2,1]=2*(y*z+w*x); Rm[:,2,2]=1-2*(x*x+y*y)
    M=Rm*S[:,None,:]
    sig = M@np.transpose(M,(0,2,1))
    op = 1/(1+np.exp(-np.clip(col['opacity'].astype(np.float64)[keep],-60,60)))
    return P,sig,op

def render(P,sig,op,R,t,npix):
    cam = P@R.T + t; z=cam[:,2]
    front = z>0.05
    cam=cam[front]; s=sig[front]; o=op[front]; z=z[front]
    sc = R@s@R.T
    iz=1/z; j00=FX*iz; j11=FY*iz; j02=-FX*cam[:,0]*iz*iz; j12=-FY*cam[:,1]*iz*iz
    s00,s01,s02=sc[:,0,0],sc[:,0,1],sc[:,0,2]; s11,s12,s22=sc[:,1,1],sc[:,1,2],sc[:,2,2]
    a0=j00*s00+j02*s02; a1=j00*s01+j02*s12; a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12; b2=j11*s12+j12*s22
    sa=a0*j00+a2*j02; sb=a1*j11+a2*j12; sc2=b1*j11+b2*j12
    detb=np.maximum(sa*sc2-sb*sb,1e-12)
    sa=sa+F2D; sc2=sc2+F2D
    det=sa*sc2-sb*sb
    comp=np.sqrt(np.clip(detb/np.maximum(det,1e-12),0,1))
    alpha=o*comp
    mid=0.5*(sa+sc2); disc=np.sqrt(np.maximum(mid*mid-det,1e-9))
    rad=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
    mx=FX*cam[:,0]*iz+CX; my=FY*cam[:,1]*iz+CY
    # frustum cull, applied identically to both models: a splat whose centre
    # lies outside the screen expanded by its own radius cannot cover a pixel,
    # and the EWA affine Jacobian is meaningless beyond ~2x the half-FOV, where
    # it produces radii of 1e9 px. Cap the radius at the screen diagonal.
    rad=np.minimum(rad, np.hypot(W,H))
    offax = np.maximum(np.abs(cam[:,0]*iz)/(CX/FX), np.abs(cam[:,1]*iz)/(CY/FY))
    ok=(det>1e-12)&(rad>=0.5)&(alpha>=MIN_A)&(z<400)&(offax<2.0)
    ok&=(mx>-rad)&(mx<W+rad)&(my>-rad)&(my<H+rad)
    mx,my,rad,alpha,z=mx[ok],my[ok],rad[ok],alpha[ok],z[ok]
    sa,sb,sc2,det=sa[ok],sb[ok],sc2[ok],det[ok]
    idet=1/det; ca=sc2*idet; cb=-sb*idet; cc=sa*idet   # conic
    # sample pixels
    rng=np.random.default_rng(1)
    px=rng.integers(0,W,npix)+0.5; py=rng.integers(0,H,npix)+0.5
    ev=np.zeros(npix); cl=np.zeros(npix); co=np.zeros(npix); Tend=np.ones(npix)
    order=np.argsort(z)
    mx,my,rad,alpha=mx[order],my[order],rad[order],alpha[order]
    ca,cb,cc=ca[order],cb[order],cc[order]
    for i in range(npix):
        dx=px[i]-mx; dy=py[i]-my
        hit=(np.abs(dx)<=rad)&(np.abs(dy)<=rad)     # bbox = what the tile bin delivers
        if not hit.any(): continue
        dx=dx[hit]; dy=dy[hit]
        pw=-0.5*(ca[hit]*dx*dx+cc[hit]*dy*dy)-cb[hit]*dx*dy
        ev[i]=hit.sum()
        av=np.minimum(0.99,alpha[hit]*np.exp(np.minimum(pw,0)))
        good=(pw<=0)&(av>=MIN_A)
        cl[i]=good.sum()
        av=np.where(good,av,0.0)
        T=np.cumprod(1-av)
        stop=np.argmax(T<1e-4) if (T<1e-4).any() else len(T)-1
        co[i]=(av[:stop+1]>0).sum(); Tend[i]=T[stop]
    return ev,cl,co,Tend

def rig(nviews, centre, radius, seed):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(nviews):
        p=rng.normal(size=3); p/=np.linalg.norm(p); p=centre+p*rng.uniform(0,radius)
        # random orientation
        q=rng.normal(size=4); q/=np.linalg.norm(q); x,y,z,w=q
        Rm=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        out.append((Rm, -Rm@p))
    return out

if __name__=='__main__':
    NV=int(sys.argv[1]) if len(sys.argv)>1 else 12
    NPIX=int(sys.argv[2]) if len(sys.argv)>2 else 400
    c,_=sv_load.sv(); sent=np.load(SCR+r"\sv_sentinel.npy")
    o,_=sv_load.ours()
    jobs=[]
    Pf,Sf,Of=build(c, ~sent, False); jobs.append(('SCANIVERSE fg only', Pf,Sf,Of, np.array([-1.26375,-0.24857,0.39607])))
    Pa,Sa,Oa=build(c, np.ones(len(c['x']),bool), False); jobs.append(('SCANIVERSE fg+dome', Pa,Sa,Oa, np.array([-1.26375,-0.24857,0.39607])))
    Po,So,Oo=build(o, np.ones(len(o['x']),bool), True); jobs.append(('OURS build182', Po,So,Oo, None))
    for tag,P,S,O,ctr in jobs:
        if ctr is None: ctr=np.median(P,axis=0)
        E=[];C=[];K=[];T=[]
        for R,t in rig(NV, ctr, 0.6, 7):
            e,cl,co,te=render(P,S,O,R,t,NPIX); E.append(e);C.append(cl);K.append(co);T.append(te)
        E=np.concatenate(E);C=np.concatenate(C);K=np.concatenate(K);T=np.concatenate(T)
        nz=E>0
        print('%-20s  n=%6d splats | per pixel (all px): eval %7.1f  clear %6.1f  composited %6.1f | frac px empty %5.1f%% | frac px reaching T<1e-4 %5.1f%%'
              % (tag,len(P),E.mean(),C.mean(),K.mean(),100*np.mean(~nz),100*np.mean(T<1e-4)))
        print('%-20s  covered px only : eval %7.1f  clear %6.1f  composited %6.1f  | median eval %6.0f'
              % ('',E[nz].mean(),C[nz].mean(),K[nz].mean(),np.median(E[nz])))
