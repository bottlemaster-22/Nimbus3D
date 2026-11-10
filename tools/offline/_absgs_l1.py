"""Does replacing length(dLdMean2D) with |x|+|y| change WHICH splats densify?

The AbsGS statistic is only ever used as a RANKING (TrainerDensifier scores
splats and takes the top ones). So the question is not whether L1 == L2, it is
whether the per-splat SUM changes the order. Measured on the real model, real
poses, real tile lists.

dLdMean2D = dLdPower * u  with u = -(C . delta) and dLdPower a scalar, so
L1/L2 per pair depends ONLY on the angle of u. The per-splat ratio is a
positive-weighted average of that.
"""
import io, json, os, sys
import numpy as np
import project as P, raster as Rz

D = P.D; TILE=16; AREA=256

def main():
    nframes = int(sys.argv[1]) if len(sys.argv)>1 else 3
    n_tiles = int(sys.argv[2]) if len(sys.argv)>2 else 80
    census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
    sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
    tx,ty=(rw+15)//16,(rh+15)//16
    fx,fy,cx0,cy0=P.load_intrinsics(rw,rh)
    col,count=P.load_ply(os.path.join(D,'model','model.ply'))
    poses=P.load_poses(); minAlpha=P.MIN_ALPHA
    keys=sorted(poses.keys()); stride=max(1,len(keys)//nframes); frames=keys[::stride][:nframes]
    print(f"render {rw}x{rh}, {count} splats, frames {frames}")

    sumL2=np.zeros(count); sumL1=np.zeros(count); npair=np.zeros(count)
    sumL2u=np.zeros(count); sumL1u=np.zeros(count)   # unweighted
    ratios=[]
    for fr in frames:
        rot,t=poses[fr]; R=P.quat_to_matrix(rot)
        g=Rz.geometry(col,count,R,t,fx,fy,cx0,cy0)
        ok,radius,tiles,ex,ey=P.project(col,count,R,t,fx,fy,cx0,cy0,tx,ty)
        idx=np.nonzero(ok)[0]; mx,my,z=g['mx'],g['my'],g['z']
        min_x=np.maximum(0,np.floor((mx-ex)/TILE)).astype(np.int64)
        min_y=np.maximum(0,np.floor((my-ey)/TILE)).astype(np.int64)
        max_x=np.minimum(tx,np.ceil((mx+ex)/TILE)).astype(np.int64)
        max_y=np.minimum(ty,np.ceil((my+ey)/TILE)).astype(np.int64)
        rng=np.random.RandomState(int(fr))
        picks=rng.choice(tx*ty,size=min(n_tiles,tx*ty),replace=False)
        for p in picks:
            t_x,t_y=int(p%tx),int(p//tx)
            sel=idx[(min_x[idx]<=t_x)&(max_x[idx]>t_x)&(min_y[idx]<=t_y)&(max_y[idx]>t_y)]
            if sel.size==0: continue
            order=sel[np.argsort(z[sel])]
            n=order.size
            px=t_x*TILE+np.arange(TILE)+0.5; py=t_y*TILE+np.arange(TILE)+0.5
            gx,gy=np.meshgrid(px,py); gx,gy=gx.ravel(),gy.ravel()
            dx=gx[:,None]-g['mx'][order][None,:]
            dy=gy[:,None]-g['my'][order][None,:]
            c_x=g['cx'][order][None,:]; c_y=g['cy'][order][None,:]; c_z=g['cz'][order][None,:]
            power=-0.5*(c_x*dx*dx+c_z*dy*dy)-c_y*dx*dy
            op=np.maximum(g['opacity'][order],1e-8)
            cutoff=np.float64(np.float16(np.log(minAlpha/op)-0.01))[None,:]
            alpha=np.minimum(0.99,op[None,:]*np.exp(np.clip(power,-80,0)))
            ok2=(power>=cutoff)&(alpha>=minAlpha)
            keep=np.where(ok2,alpha,0.0)
            Taf=np.cumprod(1.0-keep,axis=1)
            stop=ok2&(Taf<1e-4); has=stop.any(axis=1)
            spos=np.where(has,np.argmax(stop,axis=1),n)
            used=ok2&(np.arange(n)[None,:]<spos[:,None])
            # transmittance in FRONT of each entry
            Tfront=np.concatenate([np.ones((AREA,1)),Taf[:,:-1]],axis=1)
            w=np.where(used,alpha*Tfront,0.0)         # visibility weight alpha*T
            ux=-(c_x*dx+c_y*dy); uy=-(c_z*dy+c_y*dx)
            l2=np.sqrt(ux*ux+uy*uy); l1=np.abs(ux)+np.abs(uy)
            m=used
            si=order
            np.add.at(sumL2,si,(w*l2).sum(axis=0))
            np.add.at(sumL1,si,(w*l1).sum(axis=0))
            np.add.at(sumL2u,si,np.where(m,l2,0).sum(axis=0))
            np.add.at(sumL1u,si,np.where(m,l1,0).sum(axis=0))
            np.add.at(npair,si,m.sum(axis=0))
            r=(l1/np.maximum(l2,1e-30))[m]
            if r.size: ratios.append(r)
        print(f"  frame {fr} done")

    r=np.concatenate(ratios)
    print(f"\nPER-PAIR L1/L2 over {r.size:,} real contributing pairs")
    print(f"  min {r.min():.4f}  p05 {np.percentile(r,5):.4f}  median {np.median(r):.4f}"
          f"  p95 {np.percentile(r,95):.4f}  max {r.max():.4f}  mean {r.mean():.4f}")
    print(f"  (theoretical bounds 1.0000 .. 1.4142; isotropic-angle mean 4/pi = 1.2732)")

    sel=npair>=8
    R=sumL1[sel]/np.maximum(sumL2[sel],1e-30)
    print(f"\nPER-SPLAT ratio, alpha*T weighted, {sel.sum():,} splats with >=8 pairs")
    print(f"  min {R.min():.4f}  p01 {np.percentile(R,1):.4f}  p05 {np.percentile(R,5):.4f}"
          f"  median {np.median(R):.4f}  p95 {np.percentile(R,95):.4f}  p99 {np.percentile(R,99):.4f} max {R.max():.4f}")
    print(f"  spread p99/p01 = {np.percentile(R,99)/np.percentile(R,1):.4f}x"
          f"   (a uniform rescale of the threshold absorbs the MEDIAN, not the spread)")

    from scipy.stats import spearmanr, kendalltau
    a=sumL2[sel]; b=sumL1[sel]
    rho=spearmanr(a,b).statistic
    print(f"\nRANK AGREEMENT of the per-splat score  Spearman rho = {rho:.6f}")
    for frac in (0.05,0.10,0.15):
        k=int(len(a)*frac)
        ta=set(np.argsort(-a)[:k]); tb=set(np.argsort(-b)[:k])
        print(f"  top {frac*100:.0f}% by score: {len(ta&tb)}/{k} shared "
              f"= {len(ta&tb)/k*100:.2f}%   ({k-len(ta&tb)} splats swap in/out)")
    # unweighted, as a robustness check
    au=sumL2u[sel]; bu=sumL1u[sel]
    print(f"  unweighted control: Spearman rho = {spearmanr(au,bu).statistic:.6f}")

main()
