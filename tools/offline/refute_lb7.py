"""Are the near-camera splats real, and do they own the composite?
Renders a tile sample twice: full population, and with z<ZCUT removed."""
import io, json, os, sys
import numpy as np
from PIL import Image
import project as P
INC=r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"
LUMA=np.array([0.2126,0.7152,0.0722]); SH_C0,SH_C1=0.28209479177387814,0.48860251190291990
census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8')); sl=census['slices'][0]
RW,RH=sl['renderWidth'],sl['renderHeight']; tx,ty=(RW+15)//16,(RH+15)//16
fx,fy,cx,cy=P.load_intrinsics(RW,RH)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses(); F=int(sys.argv[1]); ZCUT=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
rot,t=poses[F]; R=P.quat_to_matrix(rot); cc=-R.T@t
mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
d=np.linalg.norm(mean-cc,axis=1)
print('frame %d camera centre %s'%(F,np.round(cc,3)))
print('  distance camera->splat: min %.4f   n<0.1m %d  n<0.2m %d  n<0.5m %d  of %d'
      %(d.min(),(d<0.1).sum(),(d<0.2).sum(),(d<0.5).sum(),count))
camv=mean@R.T+t; z=camv[:,2]; iz=1.0/np.where(z!=0,z,1)
mx=fx*camv[:,0]*iz+cx; my=fy*camv[:,1]*iz+cy
scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64); q/=np.linalg.norm(q,axis=1,keepdims=True)
x_,y_,z_,w_=q[:,0],q[:,1],q[:,2],q[:,3]
Rm=np.empty((count,3,3))
Rm[:,0,0]=1-2*(y_*y_+z_*z_); Rm[:,0,1]=2*(x_*y_-w_*z_); Rm[:,0,2]=2*(x_*z_+w_*y_)
Rm[:,1,0]=2*(x_*y_+w_*z_); Rm[:,1,1]=1-2*(x_*x_+z_*z_); Rm[:,1,2]=2*(y_*z_-w_*x_)
Rm[:,2,0]=2*(x_*z_-w_*y_); Rm[:,2,1]=2*(y_*z_+w_*x_); Rm[:,2,2]=1-2*(x_*x_+y_*y_)
M=Rm*scale[:,None,:]; sigC=R@(M@np.transpose(M,(0,2,1)))@R.T
j00=fx*iz; j11=fy*iz; j02=-fx*camv[:,0]*iz*iz; j12=-fy*camv[:,1]*iz*iz
a0=j00*sigC[:,0,0]+j02*sigC[:,0,2]; a1=j00*sigC[:,0,1]+j02*sigC[:,1,2]; a2=j00*sigC[:,0,2]+j02*sigC[:,2,2]
b1=j11*sigC[:,1,1]+j12*sigC[:,1,2]; b2=j11*sigC[:,1,2]+j12*sigC[:,2,2]
sa=a0*j00+a2*j02+P.FILTER_2D_VARIANCE; sb=a1*j11+a2*j12; scv=b1*j11+b2*j12+P.FILTER_2D_VARIANCE
det=np.maximum(sa*scv-sb*sb,1e-12); cA,cB,cC=scv/det,-sb/det,sa/det
detb=np.maximum((sa-P.FILTER_2D_VARIANCE)*(scv-P.FILTER_2D_VARIANCE)-sb*sb,1e-12)
opac=(1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))*np.sqrt(np.clip(detb/det,0,1))
mid=0.5*(sa+scv); disc=np.sqrt(np.maximum(mid*mid-det,1e-9)); radius=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
kk=np.sqrt(np.clip(2.0*np.log(opac/max(P.MIN_ALPHA,1e-8)),0,9))
ex=kk*np.sqrt(np.maximum(sa,1e-9))*1.001; ey=kk*np.sqrt(np.maximum(scv,1e-9))*1.001
base=(z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(opac>=P.MIN_ALPHA)
minx=np.maximum(0,np.floor((mx-ex)/16)).astype(np.int64); miny=np.maximum(0,np.floor((my-ey)/16)).astype(np.int64)
maxx=np.minimum(tx,np.ceil((mx+ex)/16)).astype(np.int64); maxy=np.minimum(ty,np.ceil((my+ey)/16)).astype(np.int64)
base&=(maxx>minx)&(maxy>miny)
dirv=mean-cc; dirv/=np.maximum(np.linalg.norm(dirv,axis=1,keepdims=True),1e-6)
dc=np.stack([col['f_dc_0'],col['f_dc_1'],col['f_dc_2']],1).astype(np.float64)
rest=np.stack([col['f_rest_%d'%i] for i in range(9)],1).astype(np.float64)
rgb=np.maximum(SH_C0*dc+(-SH_C1*dirv[:,1:2]*rest[:,0:3]+SH_C1*dirv[:,2:3]*rest[:,3:6]-SH_C1*dirv[:,0:1]*rest[:,6:9])+0.5,0.0)
bundle=json.load(io.open(os.path.join(P.D,'capture_bundle.json'),encoding='utf-8'))
fr=[f for f in bundle['frames'] if f['index']==F][0]
gtimg=np.asarray(Image.open(os.path.join(INC,fr['imagePath'].replace('/',os.sep))).convert('RGB').resize((RW,RH),Image.BOX),dtype=np.float64)/255.0
print('  visible-after-cull: %d ; of those z<%.2f: %d (%.2f%%)'%(base.sum(),ZCUT,(base&(z<ZCUT)).sum(),100*(base&(z<ZCUT)).mean()/base.mean()))
rs=np.random.RandomState(0); picks=rs.choice(tx*ty,150,replace=False)
for lab,keep in (('ALL splats',base),('z>=%.2f only'%ZCUT, base&(z>=ZCUT))):
    idx=np.nonzero(keep)[0]; EX=[];AL=[];ERR=[];NW=[]
    for pk in picks:
        ttx,tty=int(pk%tx),int(pk//tx)
        s2=idx[(minx[idx]<=ttx)&(maxx[idx]>ttx)&(miny[idx]<=tty)&(maxy[idx]>tty)]
        if s2.size==0: continue
        o=s2[np.argsort(z[s2],kind='stable')]
        gx=ttx*16+np.arange(16); gy=tty*16+np.arange(16)
        GX,GY=np.meshgrid(gx,gy); GX=GX.ravel(); GY=GY.ravel()
        ins=(GX<RW)&(GY<RH)
        dx=(GX+0.5)[:,None]-mx[o][None,:]; dy=(GY+0.5)[:,None]-my[o][None,:]
        pw=-0.5*(cA[o][None,:]*dx*dx+cC[o][None,:]*dy*dy)-cB[o][None,:]*dx*dy
        al=np.minimum(0.99,opac[o][None,:]*np.exp(np.clip(pw,-60,0)))
        al=np.where(al>=P.MIN_ALPHA,al,0.0)
        Tc=np.cumprod(1.0-al,axis=1); done=Tc<1e-4
        fd=np.where(done.any(1),done.argmax(1),al.shape[1])
        live=np.arange(al.shape[1])[None,:]<=fd[:,None]; al=np.where(live,al,0.0)
        Tc=np.cumprod(1.0-al,axis=1); before=np.concatenate([np.ones((256,1)),Tc[:,:-1]],1)
        wg=al*before; ws=wg.sum(1)
        cimg=wg@rgb[o]; dimg=wg@z[o]
        g=ins&(ws>1e-6)
        EX.append(dimg[g]/np.maximum(ws[g],1e-4)); AL.append(ws[g]); NW.append(ws[g])
        ERR.append(((cimg[g]-gtimg[GY[g],GX[g]])**2).mean(1))
    EX=np.concatenate(EX); AL=np.concatenate(AL); ERR=np.concatenate(ERR)
    print('  %-16s pixels %6d  alpha p10 %.5f p50 %.5f  expected depth p10 %.3f p50 %.3f p90 %.3f  PSNR %.2f dB'
          %(lab,EX.size,*np.percentile(AL,[10,50]),*np.percentile(EX,[10,50,90]),10*np.log10(1/ERR.mean())))
