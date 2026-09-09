"""Contributing-pair cost of the unclamped EWA Jacobian, all tiles, real poses."""
import io,json,os,sys
import numpy as np, project as P
from refute2_showclamp import common
TILE=16
cen=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl=cen['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx,ty=(rw+15)//16,(rh+15)//16
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses()
frames=[int(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else [4]
for frame in frames:
    rot,t=poses[frame]; Rm=P.quat_to_matrix(rot)
    for name,clamp in [('unclamped',None),('clamped1.3',1.3)]:
        g,exq,eyq,radius,det,z,mx,my=common(col,count,Rm,t,fx,fy,cx,cy,clamp)
        minx=np.maximum(0,np.floor((mx-exq)/TILE)).astype(np.int64); maxx=np.minimum(tx,np.ceil((mx+exq)/TILE)).astype(np.int64)
        miny=np.maximum(0,np.floor((my-eyq)/TILE)).astype(np.int64); maxy=np.minimum(ty,np.ceil((my+eyq)/TILE)).astype(np.int64)
        tiles=np.maximum(maxx-minx,0)*np.maximum(maxy-miny,0)
        ok=(z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(g['opacity']>=P.MIN_ALPHA)&(tiles>0)
        idx=np.nonzero(ok)[0]
        ev=0;ap=0;con=0
        for tyi in range(ty):
            rows=idx[(miny[idx]<=tyi)&(maxy[idx]>tyi)]
            for txi in range(tx):
                sel=rows[(minx[rows]<=txi)&(maxx[rows]>txi)]
                if sel.size==0: continue
                order=sel[np.argsort(g['z'][sel])]
                px=txi*TILE+np.arange(TILE)+0.5; py=tyi*TILE+np.arange(TILE)+0.5
                gx,gy=np.meshgrid(px,py); gx=gx.ravel(); gy=gy.ravel()
                dx=gx[:,None]-g['mx'][order][None,:]; dy=gy[:,None]-g['my'][order][None,:]
                power=-0.5*(g['cx'][order][None,:]*dx*dx+g['cz'][order][None,:]*dy*dy)-g['cy'][order][None,:]*dx*dy
                alpha=np.minimum(0.99,g['opacity'][order][None,:]*np.exp(np.clip(power,-60,0)))
                aok=alpha>=P.MIN_ALPHA
                keep=np.where(aok,alpha,0.0)
                T=np.cumprod(1.0-keep,axis=1)
                before=np.concatenate([np.ones((T.shape[0],1)),T[:,:-1]],axis=1)
                live=before>=1e-4     # forward stops a pixel once T falls below 1e-4
                ev+=live.sum(); ap+=(aok&live).sum(); con+=(aok&live).sum()
        print('frame %3d %-11s tileInst %8d  evaluated %10d  contributing %10d'%(frame,name,tiles[ok].sum(),ev,ap))
