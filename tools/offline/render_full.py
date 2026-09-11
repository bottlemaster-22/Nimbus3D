"""Full-frame offline render of model.ply, faithful to TrainerShaders.metal.

Validation gate: PSNR on the TRAINED keyframes must land near the census's
trainedPSNR of 22.80 dB. If it does not, nothing below is trustworthy.
"""
import io, json, os, sys, numpy as np
from PIL import Image
import project as P

D = P.D
INC = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840"
TILE = 16
SH_C0 = 0.28209479177387814
SH_C1 = 0.48860251190291990

_cache = {}
def model():
    if 'm' not in _cache:
        col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
        _cache['m'] = (col, count)
    return _cache['m']

def sh_colour(col, count, cam_center_rub):
    """dir = normalize(meanWorld - cameraCenter) in RUB, degree 1, +0.5, clamp>=0.
    PLY f_rest is CHANNEL-MAJOR (all R coeffs, then G, then B) - PLYCodec.swift:88."""
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    d = mean - cam_center_rub
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    dc = np.stack([col['f_dc_0'], col['f_dc_1'], col['f_dc_2']], 1).astype(np.float64)
    s1 = np.stack([col['f_rest_0'], col['f_rest_3'], col['f_rest_6']], 1).astype(np.float64)
    s2 = np.stack([col['f_rest_1'], col['f_rest_4'], col['f_rest_7']], 1).astype(np.float64)
    s3 = np.stack([col['f_rest_2'], col['f_rest_5'], col['f_rest_8']], 1).astype(np.float64)
    c = SH_C0 * dc
    c += (-SH_C1 * y)[:, None] * s1 + (SH_C1 * z)[:, None] * s2 + (-SH_C1 * x)[:, None] * s3
    return np.maximum(c + 0.5, 0.0)

def render(frame_idx, poses, rw, rh, fx, fy, cx, cy):
    col, count = model()
    R = P.quat_to_matrix(poses[frame_idx][0]); t = poses[frame_idx][1]
    cam_center = -R.T @ t
    rgb = sh_colour(col, count, cam_center)

    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(z != 0, z, 1)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy
    scale = np.exp(np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:,0,0]=1-2*(y_*y_+z_*z_); Rm[:,0,1]=2*(x_*y_-w_*z_); Rm[:,0,2]=2*(x_*z_+w_*y_)
    Rm[:,1,0]=2*(x_*y_+w_*z_); Rm[:,1,1]=1-2*(x_*x_+z_*z_); Rm[:,1,2]=2*(y_*z_-w_*x_)
    Rm[:,2,0]=2*(x_*z_-w_*y_); Rm[:,2,1]=2*(y_*z_+w_*x_); Rm[:,2,2]=1-2*(x_*x_+y_*y_)
    M = Rm * scale[:, None, :]
    sc3 = R @ (M @ np.transpose(M, (0,2,1))) @ R.T
    j00 = fx*inv_z; j11 = fy*inv_z
    j02 = -fx*cam[:,0]*inv_z*inv_z; j12 = -fy*cam[:,1]*inv_z*inv_z
    s00,s01,s02 = sc3[:,0,0],sc3[:,0,1],sc3[:,0,2]
    s11,s12,s22 = sc3[:,1,1],sc3[:,1,2],sc3[:,2,2]
    a0=j00*s00+j02*s02; a1=j00*s01+j02*s12; a2=j00*s02+j02*s22
    b1=j11*s11+j12*s12; b2=j11*s12+j12*s22
    sa=a0*j00+a2*j02+P.FILTER_2D_VARIANCE
    sb=a1*j11+a2*j12
    scc=b1*j11+b2*j12+P.FILTER_2D_VARIANCE
    det=np.maximum(sa*scc-sb*sb,1e-12); inv_det=1.0/det
    det_before=np.maximum((sa-P.FILTER_2D_VARIANCE)*(scc-P.FILTER_2D_VARIANCE)-sb*sb,1e-12)
    comp2d=np.sqrt(np.clip(det_before/det,0,1))
    opacity=(1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))*comp2d
    conic_x = scc*inv_det; conic_y = -sb*inv_det; conic_z = sa*inv_det
    mid=0.5*(sa+scc); disc=np.sqrt(np.maximum(mid*mid-det,1e-9))
    radius=3.0*np.sqrt(np.maximum(mid+disc,1e-9))
    level=np.clip(2.0*np.log(opacity/max(P.MIN_ALPHA,1e-8)),0,9)
    k=np.sqrt(level)
    ex=k*np.sqrt(np.maximum(sa,1e-9))*1.001; ey=k*np.sqrt(np.maximum(scc,1e-9))*1.001
    tx=(rw+TILE-1)//TILE; ty=(rh+TILE-1)//TILE
    min_x=np.maximum(0,np.floor((mx-ex)/TILE)).astype(np.int64)
    min_y=np.maximum(0,np.floor((my-ey)/TILE)).astype(np.int64)
    max_x=np.minimum(tx,np.ceil((mx+ex)/TILE)).astype(np.int64)
    max_y=np.minimum(ty,np.ceil((my+ey)/TILE)).astype(np.int64)
    nt=np.maximum(max_x-min_x,0)*np.maximum(max_y-min_y,0)
    ok=(z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(opacity>=P.MIN_ALPHA)&(nt>0)
    idx=np.where(ok)[0]
    # tile instance list
    tiles_list=[]; splat_list=[]
    for i in idx:
        for tyy in range(min_y[i],max_y[i]):
            for txx in range(min_x[i],max_x[i]):
                tiles_list.append(tyy*tx+txx); splat_list.append(i)
    tiles_arr=np.array(tiles_list,dtype=np.int64); splat_arr=np.array(splat_list,dtype=np.int64)
    order=np.lexsort((z[splat_arr],tiles_arr))
    tiles_arr=tiles_arr[order]; splat_arr=splat_arr[order]
    img=np.zeros((rh,rw,3)); Tfin=np.ones((rh,rw))
    bounds=np.searchsorted(tiles_arr,np.arange(tx*ty+1))
    px=np.arange(TILE)+0.5
    for tid in range(tx*ty):
        a,b=bounds[tid],bounds[tid+1]
        if a==b: continue
        s=splat_arr[a:b]
        t0x=(tid%tx)*TILE; t0y=(tid//tx)*TILE
        gx=(t0x+px)[None,:].repeat(TILE,0).ravel()
        gy=(t0y+px)[:,None].repeat(TILE,1).ravel()
        dx=gx[:,None]-mx[s][None,:]; dy=gy[:,None]-my[s][None,:]
        power=-0.5*(conic_x[s][None,:]*dx*dx+conic_z[s][None,:]*dy*dy)-conic_y[s][None,:]*dx*dy
        alpha=np.minimum(0.99,opacity[s][None,:]*np.exp(np.clip(power,-60,0)))
        alpha=np.where(alpha>=P.MIN_ALPHA,alpha,0.0)
        Tb=np.concatenate([np.ones((alpha.shape[0],1)),np.cumprod(1-alpha,axis=1)[:,:-1]],axis=1)
        w=alpha*Tb
        acc=w@rgb[s]
        Tl=np.prod(1-alpha,axis=1)
        h=min(TILE,rh-t0y); wd=min(TILE,rw-t0x)
        img[t0y:t0y+h,t0x:t0x+wd]=acc.reshape(TILE,TILE,3)[:h,:wd]
        Tfin[t0y:t0y+h,t0x:t0x+wd]=Tl.reshape(TILE,TILE)[:h,:wd]
    return img,Tfin

def ground_truth(name, rw, rh):
    im=Image.open(os.path.join(INC,'images',name)).convert('RGB').resize((rw,rh),Image.BILINEAR)
    return np.asarray(im,dtype=np.float64)/255.0

def psnr(a,b): 
    m=np.mean((a-b)**2)
    return 10*np.log10(1.0/m) if m>1e-12 else 99.0

def fit_exposure(v,t):
    """Closed-form least squares gain/bias, clamped exactly as the trainer does."""
    x=v.ravel(); y=t.ravel(); n=len(x)
    den=n*np.dot(x,x)-x.sum()**2
    if den<=1e-9: return 1.0,0.0
    g=(n*np.dot(x,y)-x.sum()*y.sum())/den
    bi=(y.sum()-g*x.sum())/n
    return float(np.clip(g,0.9,1.1)),float(np.clip(bi,-0.05,0.05))
