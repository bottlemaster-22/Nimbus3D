"""Does the model render better WITH the training-time frequency blur?

cameraUniforms(): frequencyBlurVariance = 4.0 * (1 - fraction/0.2), fraction =
iteration/totalIterations. The run reached 4000 of a 30000 denominator, so
fraction = 0.1333 and the blur was still 1.333 px^2 on the LAST training step.
evaluateHeldOut() calls cameraUniforms(iteration: 1, totalIterations: 1), i.e.
fraction 1.0, so it scores with blur 0. The export has no blur either.
"""
import json,io,os,numpy as np
from PIL import Image
import project as P, render_full as RF
D=P.D; INC=RF.INC; TILE=16
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
poses=P.load_poses()
col,count=RF.model()

def render_blur(fi,fb):
    R=P.quat_to_matrix(poses[fi][0]); t=poses[fi][1]
    rgb=RF.sh_colour(col,count,-R.T@t)
    mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
    cam=mean@R.T+t; z=cam[:,2]; iz=1.0/np.where(z!=0,z,1)
    mx=fx*cam[:,0]*iz+cx; my=fy*cam[:,1]*iz+cy
    scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
    q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True); x_,y_,z_,w_=q[:,0],q[:,1],q[:,2],q[:,3]
    Rm=np.empty((count,3,3))
    Rm[:,0,0]=1-2*(y_*y_+z_*z_);Rm[:,0,1]=2*(x_*y_-w_*z_);Rm[:,0,2]=2*(x_*z_+w_*y_)
    Rm[:,1,0]=2*(x_*y_+w_*z_);Rm[:,1,1]=1-2*(x_*x_+z_*z_);Rm[:,1,2]=2*(y_*z_-w_*x_)
    Rm[:,2,0]=2*(x_*z_-w_*y_);Rm[:,2,1]=2*(y_*z_+w_*x_);Rm[:,2,2]=1-2*(x_*x_+y_*y_)
    M=Rm*scale[:,None,:]; sc3=R@(M@np.transpose(M,(0,2,1)))@R.T
    j00=fx*iz;j11=fy*iz;j02=-fx*cam[:,0]*iz*iz;j12=-fy*cam[:,1]*iz*iz
    s00,s01,s02=sc3[:,0,0],sc3[:,0,1],sc3[:,0,2];s11,s12,s22=sc3[:,1,1],sc3[:,1,2],sc3[:,2,2]
    a0=j00*s00+j02*s02;a1=j00*s01+j02*s12;a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12;b2=j11*s12+j12*s22
    raw_a=a0*j00+a2*j02; sb=a1*j11+a2*j12; raw_c=b1*j11+b2*j12
    lp=P.FILTER_2D_VARIANCE+fb                      # <-- the training low-pass
    sa=raw_a+lp; scc=raw_c+lp
    det=np.maximum(sa*scc-sb*sb,1e-12); idet=1.0/det
    detb=np.maximum(raw_a*raw_c-sb*sb,1e-12)
    op=(1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))*np.sqrt(np.clip(detb/det,0,1))
    cxc=scc*idet;cyc=-sb*idet;czc=sa*idet
    mid=0.5*(sa+scc);disc=np.sqrt(np.maximum(mid*mid-det,1e-9));rad=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
    lev=np.clip(2.0*np.log(op/max(P.MIN_ALPHA,1e-8)),0,9);kk=np.sqrt(lev)
    ex=kk*np.sqrt(np.maximum(sa,1e-9))*1.001;ey=kk*np.sqrt(np.maximum(scc,1e-9))*1.001
    tx=(rw+TILE-1)//TILE;ty=(rh+TILE-1)//TILE
    mnx=np.maximum(0,np.floor((mx-ex)/TILE)).astype(np.int64);mny=np.maximum(0,np.floor((my-ey)/TILE)).astype(np.int64)
    mxx=np.minimum(tx,np.ceil((mx+ex)/TILE)).astype(np.int64);mxy=np.minimum(ty,np.ceil((my+ey)/TILE)).astype(np.int64)
    nt=np.maximum(mxx-mnx,0)*np.maximum(mxy-mny,0)
    ok=(z>0.05)&(z<100)&(det>1e-12)&(rad>=0.5)&(op>=P.MIN_ALPHA)&(nt>0)
    ii=np.where(ok)[0];rep=nt[ii];sp=np.repeat(ii,rep)
    off=np.concatenate([[0],np.cumsum(rep)]);loc=np.arange(len(sp))-np.repeat(off[:-1],rep)
    wrep=np.repeat((mxx-mnx)[ii],rep)
    tid=(np.repeat(mny[ii],rep)+(loc//wrep))*tx+(np.repeat(mnx[ii],rep)+(loc%wrep))
    o=np.lexsort((z[sp],tid));tid=tid[o];sp=sp[o]
    bnd=np.searchsorted(tid,np.arange(tx*ty+1));px=np.arange(TILE)+0.5
    img=np.zeros((rh,rw,3))
    for T in range(tx*ty):
        a,b=bnd[T],bnd[T+1]
        if a==b: continue
        s=sp[a:b];t0x=(T%tx)*TILE;t0y=(T//tx)*TILE
        gx=(t0x+px)[None,:].repeat(TILE,0).ravel();gy=(t0y+px)[:,None].repeat(TILE,1).ravel()
        dx=gx[:,None]-mx[s][None,:];dy=gy[:,None]-my[s][None,:]
        pw=-0.5*(cxc[s][None,:]*dx*dx+czc[s][None,:]*dy*dy)-cyc[s][None,:]*dx*dy
        al=np.minimum(0.99,op[s][None,:]*np.exp(np.clip(pw,-60,0)))
        al=np.where(al>=P.MIN_ALPHA,al,0.0)
        Tb=np.concatenate([np.ones((al.shape[0],1)),np.cumprod(1-al,axis=1)[:,:-1]],axis=1)
        h=min(TILE,rh-t0y);wd=min(TILE,rw-t0x)
        img[t0y:t0y+h,t0x:t0x+wd]=((al*Tb)@rgb[s]).reshape(TILE,TILE,3)[:h,:wd]
    return img

def fit_free(v,t):
    x=v.ravel();y=t.ravel();n=len(x)
    den=n*np.dot(x,x)-x.sum()**2
    g=(n*np.dot(x,y)-x.sum()*y.sum())/den; return g,(y.sum()-g*x.sum())/n
def sc(v,t):
    g,b=fit_free(v,t); return RF.psnr(g*v+b,t)

bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
avail=[(f['index'],os.path.basename(f['imagePath'])) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
TRAINED={0,3,8,11,14}
BL=[0.0,0.667,1.333,2.0,2.667,4.0]
print('frequencyBlurVariance at iteration i of a 30000 denominator:')
for it in [2000,3000,4000,6000]:
    fr=it/30000.0; v=4.0*(1-fr/0.2) if fr<0.2 else 0.0
    print('   it %5d -> fraction %.4f -> blur %.3f px^2 (sigma %.2f px)'%(it,fr,v,np.sqrt(v) if v>0 else 0))
print()
print('PSNR (free exposure fit) vs the blur the render is done with:')
print('frame  trn  ' + '  '.join('%6.3f'%b for b in BL))
res={}
for idx,name in avail:
    gt=RF.ground_truth(name,rw,rh)
    row=[sc(render_blur(idx,b),gt) for b in BL]
    res[idx]=row
    print('%5d  %3s  '%(idx,'YES' if idx in TRAINED else 'no')+'  '.join('%6.2f'%v for v in row))
tr=[res[i] for i,_ in avail if i in TRAINED]; un=[res[i] for i,_ in avail if i not in TRAINED]
print()
print('MEAN trained      : '+'  '.join('%6.2f'%v for v in np.mean(tr,0)))
print('MEAN never-trained: '+'  '.join('%6.2f'%v for v in np.mean(un,0)))
print()
b=np.mean(un,0)
print('never-trained: blur 0 (what the eval and the export use) = %.2f dB'%b[0])
print('              blur 1.333 (the last training step)        = %.2f dB   delta %+.2f dB'%(b[2],b[2]-b[0]))
print('              best of the sweep                          = %.2f dB at %.3f px^2'%(b.max(),BL[int(np.argmax(b))]))
