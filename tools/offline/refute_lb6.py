"""Is 'expected sits 0.56 m behind the front surface' an artefact of the 5% rule?
Same composite, but report the whole depth-weight profile per pixel."""
import io, json, os, sys
import numpy as np
import project as P
census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8')); sl=census['slices'][0]
RW,RH=sl['renderWidth'],sl['renderHeight']; tx,ty=(RW+15)//16,(RH+15)//16
fx,fy,cx,cy=P.load_intrinsics(RW,RH)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
logs=np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3)
poses=P.load_poses(); FRAME=int(sys.argv[1]) if len(sys.argv)>1 else 4
rot,t=poses[FRAME]; R=P.quat_to_matrix(rot)
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
camv=mean@R.T+t; z=camv[:,2]; iz=1.0/np.where(z!=0,z,1)
mx=fx*camv[:,0]*iz+cx; my=fy*camv[:,1]*iz+cy
scale=np.exp(logs)
q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
q/=np.linalg.norm(q,axis=1,keepdims=True)
x_,y_,z_,w_=q[:,0],q[:,1],q[:,2],q[:,3]
Rm=np.empty((count,3,3))
Rm[:,0,0]=1-2*(y_*y_+z_*z_); Rm[:,0,1]=2*(x_*y_-w_*z_); Rm[:,0,2]=2*(x_*z_+w_*y_)
Rm[:,1,0]=2*(x_*y_+w_*z_); Rm[:,1,1]=1-2*(x_*x_+z_*z_); Rm[:,1,2]=2*(y_*z_-w_*x_)
Rm[:,2,0]=2*(x_*z_-w_*y_); Rm[:,2,1]=2*(y_*z_+w_*x_); Rm[:,2,2]=1-2*(x_*x_+y_*y_)
M=Rm*scale[:,None,:]; sigC=R@(M@np.transpose(M,(0,2,1)))@R.T
j00=fx*iz; j11=fy*iz; j02=-fx*camv[:,0]*iz*iz; j12=-fy*camv[:,1]*iz*iz
a0=j00*sigC[:,0,0]+j02*sigC[:,0,2]; a1=j00*sigC[:,0,1]+j02*sigC[:,1,2]; a2=j00*sigC[:,0,2]+j02*sigC[:,2,2]
b1=j11*sigC[:,1,1]+j12*sigC[:,1,2]; b2=j11*sigC[:,1,2]+j12*sigC[:,2,2]
sa=a0*j00+a2*j02+P.FILTER_2D_VARIANCE; sb=a1*j11+a2*j12; sc=b1*j11+b2*j12+P.FILTER_2D_VARIANCE
det=np.maximum(sa*sc-sb*sb,1e-12); cA,cB,cC=sc/det,-sb/det,sa/det
detb=np.maximum((sa-P.FILTER_2D_VARIANCE)*(sc-P.FILTER_2D_VARIANCE)-sb*sb,1e-12)
opac=(1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))*np.sqrt(np.clip(detb/det,0,1))
mid=0.5*(sa+sc); disc=np.sqrt(np.maximum(mid*mid-det,1e-9)); radius=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
kk=np.sqrt(np.clip(2.0*np.log(opac/max(P.MIN_ALPHA,1e-8)),0,9))
ex=kk*np.sqrt(np.maximum(sa,1e-9))*1.001; ey=kk*np.sqrt(np.maximum(sc,1e-9))*1.001
ok=(z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(opac>=P.MIN_ALPHA)
minx=np.maximum(0,np.floor((mx-ex)/16)).astype(np.int64); miny=np.maximum(0,np.floor((my-ey)/16)).astype(np.int64)
maxx=np.minimum(tx,np.ceil((mx+ex)/16)).astype(np.int64); maxy=np.minimum(ty,np.ceil((my+ey)/16)).astype(np.int64)
ok&=(maxx>minx)&(maxy>miny); idx=np.nonzero(ok)[0]
print('splat camera-depth z: p1 %.3f p10 %.3f p50 %.3f p90 %.3f (visible set)'%tuple(np.percentile(z[idx],[1,10,50,90])))
rs=np.random.RandomState(0); picks=rs.choice(tx*ty,120,replace=False)
Q=[0.01,0.05,0.10,0.25,0.50,0.90]
acc={q:[] for q in Q}; EX=[]; SD=[]; NC=[]
for pk in picks:
    ttx,tty=int(pk%tx),int(pk//tx)
    s2=idx[(minx[idx]<=ttx)&(maxx[idx]>ttx)&(miny[idx]<=tty)&(maxy[idx]>tty)]
    if s2.size==0: continue
    o=s2[np.argsort(z[s2],kind='stable')]
    gx=ttx*16+np.arange(16)+0.5; gy=tty*16+np.arange(16)+0.5
    GX,GY=np.meshgrid(gx,gy); GX=GX.ravel(); GY=GY.ravel()
    dx=GX[:,None]-mx[o][None,:]; dy=GY[:,None]-my[o][None,:]
    pw=-0.5*(cA[o][None,:]*dx*dx+cC[o][None,:]*dy*dy)-cB[o][None,:]*dx*dy
    al=np.minimum(0.99,opac[o][None,:]*np.exp(np.clip(pw,-60,0)))
    al=np.where(al>=P.MIN_ALPHA,al,0.0)
    Tc=np.cumprod(1.0-al,axis=1); done=Tc<1e-4
    fd=np.where(done.any(1),done.argmax(1),al.shape[1])
    live=np.arange(al.shape[1])[None,:]<=fd[:,None]; al=np.where(live,al,0.0)
    Tc=np.cumprod(1.0-al,axis=1); before=np.concatenate([np.ones((256,1)),Tc[:,:-1]],1)
    wg=al*before; ws=wg.sum(1); good=ws>0.05
    if not good.any(): continue
    m1=wg@z[o]; m2=wg@(z[o]**2)
    e=m1[good]/ws[good]; v=np.maximum(m2[good]/ws[good]-e*e,0)
    EX.append(e); SD.append(np.sqrt(v)); NC.append((al[good]>0).sum(1))
    cum=np.cumsum(wg,axis=1); zz=np.broadcast_to(z[o][None,:],al.shape)
    for qq in Q:
        f=(cum>=qq*ws[:,None]).argmax(1)
        acc[qq].append(zz[np.arange(256),f][good])
EX=np.concatenate(EX); SD=np.concatenate(SD); NC=np.concatenate(NC)
print('pixels %d   contributors/pixel p50 %.0f'%(EX.size,np.median(NC)))
print('expected depth m   p10 %.4f p50 %.4f p90 %.4f'%tuple(np.percentile(EX,[10,50,90])))
print('alpha-wtd STD m    p10 %.4f p50 %.4f p90 %.4f'%tuple(np.percentile(SD,[10,50,90])))
for qq in Q:
    d=np.concatenate(acc[qq])
    print('  depth at %4.0f%% cumulative weight: p10 %.4f p50 %.4f p90 %.4f   |  expected-minus-it p50 %+.4f'
          %(100*qq,*np.percentile(d,[10,50,90]),np.median(EX-d)))
