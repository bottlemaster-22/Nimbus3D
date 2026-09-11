"""Per-Gaussian, per-view CONTRIBUTION weight sum(alpha*T) over the full image.

The earlier parallax number used frustum visibility, which counts a Gaussian as
'seen' by a camera on the far side of the wall it sits on. That inflates the
triangulation angle. This uses what the rasteriser actually composites.
"""
import json,io,os,numpy as np,time
import project as P
D=P.D; TILE=16
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses(); allkf=np.load('_kf.npy')
tx=(rw+TILE-1)//TILE; ty=(rh+TILE-1)//TILE
mean=np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)
scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
q/=np.linalg.norm(q,axis=1,keepdims=True)
x_,y_,z_,w_=q[:,0],q[:,1],q[:,2],q[:,3]
Rm=np.empty((count,3,3))
Rm[:,0,0]=1-2*(y_*y_+z_*z_);Rm[:,0,1]=2*(x_*y_-w_*z_);Rm[:,0,2]=2*(x_*z_+w_*y_)
Rm[:,1,0]=2*(x_*y_+w_*z_);Rm[:,1,1]=1-2*(x_*x_+z_*z_);Rm[:,1,2]=2*(y_*z_-w_*x_)
Rm[:,2,0]=2*(x_*z_-w_*y_);Rm[:,2,1]=2*(y_*z_+w_*x_);Rm[:,2,2]=1-2*(x_*x_+y_*y_)
M=Rm*scale[:,None,:]; SIG=M@np.transpose(M,(0,2,1))
opac0=1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))
contrib=np.zeros((len(allkf),count),dtype=np.float32)
t0=time.time()
for k,idx in enumerate(allkf):
    R=P.quat_to_matrix(poses[idx][0]); t=poses[idx][1]
    cam=mean@R.T+t; z=cam[:,2]; iz=1.0/np.where(z!=0,z,1)
    mx=fx*cam[:,0]*iz+cx; my=fy*cam[:,1]*iz+cy
    sc3=R@SIG@R.T
    j00=fx*iz;j11=fy*iz;j02=-fx*cam[:,0]*iz*iz;j12=-fy*cam[:,1]*iz*iz
    s00,s01,s02=sc3[:,0,0],sc3[:,0,1],sc3[:,0,2]; s11,s12,s22=sc3[:,1,1],sc3[:,1,2],sc3[:,2,2]
    a0=j00*s00+j02*s02;a1=j00*s01+j02*s12;a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12;b2=j11*s12+j12*s22
    sa=a0*j00+a2*j02+P.FILTER_2D_VARIANCE; sb=a1*j11+a2*j12; scc=b1*j11+b2*j12+P.FILTER_2D_VARIANCE
    det=np.maximum(sa*scc-sb*sb,1e-12); idet=1.0/det
    detb=np.maximum((sa-P.FILTER_2D_VARIANCE)*(scc-P.FILTER_2D_VARIANCE)-sb*sb,1e-12)
    comp=np.sqrt(np.clip(detb/det,0,1)); op=opac0*comp
    cxc=scc*idet;cyc=-sb*idet;czc=sa*idet
    mid=0.5*(sa+scc);disc=np.sqrt(np.maximum(mid*mid-det,1e-9));rad=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
    lev=np.clip(2.0*np.log(op/max(P.MIN_ALPHA,1e-8)),0,9);kk=np.sqrt(lev)
    ex=kk*np.sqrt(np.maximum(sa,1e-9))*1.001; ey=kk*np.sqrt(np.maximum(scc,1e-9))*1.001
    mnx=np.maximum(0,np.floor((mx-ex)/TILE)).astype(np.int64); mny=np.maximum(0,np.floor((my-ey)/TILE)).astype(np.int64)
    mxx=np.minimum(tx,np.ceil((mx+ex)/TILE)).astype(np.int64); mxy=np.minimum(ty,np.ceil((my+ey)/TILE)).astype(np.int64)
    nt=np.maximum(mxx-mnx,0)*np.maximum(mxy-mny,0)
    ok=(z>0.05)&(z<100)&(det>1e-12)&(rad>=0.5)&(op>=P.MIN_ALPHA)&(nt>0)
    ii=np.where(ok)[0]
    rep=nt[ii]; sp=np.repeat(ii,rep)
    off=np.concatenate([[0],np.cumsum(rep)]); loc=np.arange(len(sp))-np.repeat(off[:-1],rep)
    wdt=(mxx-mnx)[ii]; wrep=np.repeat(wdt,rep)
    tcol=np.repeat(mnx[ii],rep)+(loc%wrep); trow=np.repeat(mny[ii],rep)+(loc//wrep)
    tid=trow*tx+tcol
    o=np.lexsort((z[sp],tid)); tid=tid[o]; sp=sp[o]
    bnd=np.searchsorted(tid,np.arange(tx*ty+1))
    px=np.arange(TILE)+0.5
    acc=np.zeros(count)
    for T in range(tx*ty):
        a,b=bnd[T],bnd[T+1]
        if a==b: continue
        s=sp[a:b]; t0x=(T%tx)*TILE; t0y=(T//tx)*TILE
        gx=(t0x+px)[None,:].repeat(TILE,0).ravel(); gy=(t0y+px)[:,None].repeat(TILE,1).ravel()
        dx=gx[:,None]-mx[s][None,:]; dy=gy[:,None]-my[s][None,:]
        pw=-0.5*(cxc[s][None,:]*dx*dx+czc[s][None,:]*dy*dy)-cyc[s][None,:]*dx*dy
        al=np.minimum(0.99,op[s][None,:]*np.exp(np.clip(pw,-60,0)))
        al=np.where(al>=P.MIN_ALPHA,al,0.0)
        Tb=np.concatenate([np.ones((al.shape[0],1)),np.cumprod(1-al,axis=1)[:,:-1]],axis=1)
        np.add.at(acc,s,(al*Tb).sum(0))
    contrib[k]=acc
    if k%10==0: print('  view %d/%d  (%.0fs)'%(k,len(allkf),time.time()-t0),flush=True)
np.save('_contrib.npy',contrib)
print('done %.0fs'%(time.time()-t0))
