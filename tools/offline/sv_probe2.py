"""Outward-looking room-scan rig, applied identically to both models.
Camera sits in a small ball around the scene origin and looks radially OUTWARD,
which is what a handheld room scan actually does and avoids putting splats at
z~0 in front of the camera (which makes the EWA Jacobian explode)."""
import sv_load, numpy as np, sv_probe as SP
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"

def outward_rig(nv, centre, ball, seed):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(nv):
        d=rng.normal(size=3); d/=np.linalg.norm(d)
        p=centre+d*rng.uniform(0.05,ball)
        f=d                                    # look outward
        up=np.array([0,1.0,0]); 
        if abs(f@up)>0.9: up=np.array([1.0,0,0])
        r=np.cross(up,f); r/=np.linalg.norm(r); u=np.cross(f,r)
        R=np.stack([r,u,f])                    # world->cam rows
        out.append((R,-R@p))
    return out

if __name__=='__main__':
    c,_=sv_load.sv(); sent=np.load(SCR+r"\sv_sentinel.npy"); o,_=sv_load.ours()
    Pf,Sf,Of=SP.build(c,~sent,False)
    Pa,Sa,Oa=SP.build(c,np.ones(len(c['x']),bool),False)
    Po,So,Oo=SP.build(o,np.ones(len(o['x']),bool),True)
    ctr_sv=np.array([-1.26375,-0.24857,0.39607]); ctr_us=np.median(Po,0)
    print('scene centres: SV %s   OURS %s' % (np.round(ctr_sv,3), np.round(ctr_us,3)))
    NV,NP=16,400
    for tag,P,S,O,ctr in [('SCANIVERSE fg',Pf,Sf,Of,ctr_sv),('SCANIVERSE fg+dome',Pa,Sa,Oa,ctr_sv),('OURS build182',Po,So,Oo,ctr_us)]:
        E=[];C=[];K=[];T=[]
        for R,t in outward_rig(NV,ctr,0.7,11):
            e,cl,co,te=SP.render(P,S,O,R,t,NP); E+=[e];C+=[cl];K+=[co];T+=[te]
        E=np.concatenate(E);C=np.concatenate(C);K=np.concatenate(K);T=np.concatenate(T); nz=E>0
        print('%-19s eval %7.1f | clearAlpha %7.1f | composited %6.1f | empty px %5.1f%% | T<1e-4 %5.1f%% | (covered px: comp %6.1f)'
              % (tag,E.mean(),C.mean(),K.mean(),100*np.mean(~nz),100*np.mean(T<1e-4),K[nz].mean()))
