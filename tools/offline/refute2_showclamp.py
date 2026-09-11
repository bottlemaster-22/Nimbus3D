"""Render a real frame with and without the INRIA tangent clamp and score both
against the photograph. Quality half of the EWA-clamp question."""
import io, json, os, sys
import numpy as np
import project as P, raster as R, showme as S

TILE=16
def common(col,count,Rm,t,fx,fy,cx,cy,clamp):
    mean=np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64)
    cam=mean@Rm.T+t; z=cam[:,2]; iz=1.0/np.where(z!=0,z,1); iz2=iz*iz
    scale=np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
    q=np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True); x_,y_,z_,w_=q.T
    M=np.empty((count,3,3))
    M[:,0,0]=1-2*(y_*y_+z_*z_); M[:,0,1]=2*(x_*y_-w_*z_); M[:,0,2]=2*(x_*z_+w_*y_)
    M[:,1,0]=2*(x_*y_+w_*z_); M[:,1,1]=1-2*(x_*x_+z_*z_); M[:,1,2]=2*(y_*z_-w_*x_)
    M[:,2,0]=2*(x_*z_-w_*y_); M[:,2,1]=2*(y_*z_+w_*x_); M[:,2,2]=1-2*(x_*x_+y_*y_)
    M=M*scale[:,None,:]
    sc3=Rm@(M@np.transpose(M,(0,2,1)))@Rm.T
    ex_,ey_=cam[:,0],cam[:,1]
    if clamp is not None:
        lx=clamp*(cx/fx); ly=clamp*(cy/fy)
        ex_=np.clip(cam[:,0]*iz,-lx,lx)*z; ey_=np.clip(cam[:,1]*iz,-ly,ly)*z
    j00=fx*iz; j11=fy*iz; j02=-fx*ex_*iz2; j12=-fy*ey_*iz2
    s00,s01,s02=sc3[:,0,0],sc3[:,0,1],sc3[:,0,2]; s11,s12,s22=sc3[:,1,1],sc3[:,1,2],sc3[:,2,2]
    a0=j00*s00+j02*s02; a1=j00*s01+j02*s12; a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12; b2=j11*s12+j12*s22
    sa=a0*j00+a2*j02; sb=a1*j11+a2*j12; scc=b1*j11+b2*j12
    detb=np.maximum(sa*scc-sb*sb,1e-12)
    lp=P.FILTER_2D_VARIANCE+P.FREQ_BLUR_VARIANCE
    sa=sa+lp; scc=scc+lp
    det=sa*scc-sb*sb
    comp=np.sqrt(np.clip(detb/np.maximum(det,1e-12),0,1))
    mid=0.5*(sa+scc); disc=np.sqrt(np.maximum(mid*mid-det,1e-9))
    radius=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
    op=1.0/(1.0+np.exp(-col['opacity'].astype(np.float64)))
    alpha=op*comp
    level=np.clip(2.0*np.log(alpha/max(P.MIN_ALPHA,1e-8)),0,9); k=np.sqrt(level)
    exq=k*np.sqrt(np.maximum(sa,1e-9))*1.001; eyq=k*np.sqrt(np.maximum(scc,1e-9))*1.001
    mx=fx*cam[:,0]*iz+cx; my=fy*cam[:,1]*iz+cy
    idet=1.0/np.maximum(det,1e-12)
    g={'z':z,'mx':mx,'my':my,'cx':scc*idet,'cy':-sb*idet,'cz':sa*idet,'opacity':alpha}
    return g,exq,eyq,radius,det,z,mx,my

def main():
    frame=int(sys.argv[1]) if len(sys.argv)>1 else 4
    cen=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
    sl=cen['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
    tx,ty=(rw+15)//16,(rh+15)//16
    fx,fy,cx,cy=P.load_intrinsics(rw,rh)
    col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
    poses=P.load_poses(); rot,t=poses[frame]; Rm=P.quat_to_matrix(rot)
    bundle=json.load(io.open(os.path.join(P.D,'capture_bundle.json'),encoding='utf-8'))
    path=[f['imagePath'] for f in bundle['frames'] if f['index']==frame][0]
    from PIL import Image
    src=os.path.join(r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840",path)
    truth=np.asarray(Image.open(src).convert('RGB').resize((rw,rh),Image.LANCZOS),dtype=np.float64)/255.
    for name,clamp in [('unclamped (current)',None),('clamped 1.3x',1.3)]:
        g,exq,eyq,radius,det,z,mx,my=common(col,count,Rm,t,fx,fy,cx,cy,clamp)
        minx=np.maximum(0,np.floor((mx-exq)/TILE)); maxx=np.minimum(tx,np.ceil((mx+exq)/TILE))
        miny=np.maximum(0,np.floor((my-eyq)/TILE)); maxy=np.minimum(ty,np.ceil((my+eyq)/TILE))
        tiles=np.maximum(maxx-minx,0)*np.maximum(maxy-miny,0)
        ok=(z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(g['opacity']>=P.MIN_ALPHA)&(tiles>0)
        img=S.render(col,count,g,ok,exq,eyq,g['mx'],g['my'],tx,ty,rw,rh,None)
        mse=float(np.mean((img-truth)**2)); psnr=10*np.log10(1/max(mse,1e-12))
        x,y=img.ravel(),truth.ravel(); n=x.size
        den=n*(x*x).sum()-x.sum()**2
        gain=(n*(x*y).sum()-x.sum()*y.sum())/den if den>1e-9 else 1.0
        bias=(y.sum()-gain*x.sum())/n
        fit=np.clip(gain*img+bias,0,1)
        psnrf=10*np.log10(1/max(float(np.mean((fit-truth)**2)),1e-12))
        print('frame %d  %-20s drawn %6d  tileInst %8d  raw PSNR %6.2f  fitted %6.2f  mean %.3f (photo %.3f)'
              %(frame,name,int(ok.sum()),int(tiles[ok].sum()),psnr,psnrf,img.mean(),truth.mean()))
if __name__=='__main__':
    main()
