"""Render a real frame and run trainer_rasterize_backward's arithmetic on it,
splitting dL/dalpha and dL/dlogScale into the photometric and depth channels.

Nothing here is estimated: every constant comes from TrainerSupport.swift /
SmartCore.swift / TrainerGPULayouts.swift and every geometric factor comes from
compositing the actual 299,209-splat model through the actual refined pose.
"""
import io, json, os, sys, time
import numpy as np
import project as P

TILE = 16
LUMA = np.array([0.2126, 0.7152, 0.0722])
SH_C0, SH_C1 = 0.28209479177387814, 0.48860251190291990
K11 = np.array([0.00102838,0.00759876,0.03600077,0.10936069,0.21300554,
                0.26601172,0.21300554,0.10936069,0.03600077,0.00759876,0.00102838])
LAMBDA_SSIM = 0.2
C1, C2 = 1e-4, 9e-4
ALPHA_SUP_W = 0.05
FREE_W = 0.5
BIMODAL_W = 1.0
DEPTH_SCALE = 1.0            # TwoScaleTrustField.depthLossScale for i <= 0.35*total

frame = int(sys.argv[1]) if len(sys.argv) > 1 else 4

census = json.load(io.open(os.path.join(P.D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
N = RW*RH
SUPERVISED = sl['depthSamplesSupervisedTotal']/sl['depthSupervisionFramesMeasured']
DW, DH = 256, 192
print('render %dx%d = %d px ; supervised depth samples/frame (MEASURED) %.1f'%(RW,RH,N,SUPERVISED))
INV_N = 1.0/N
INV_S = 1.0/SUPERVISED
print('invN = %.6e   invSamples = %.6e   ratio invS/invN = %.3f'%(INV_N, INV_S, INV_S/INV_N))

tx, ty = (RW+15)//16, (RH+15)//16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(P.D,'model','model.ply'))
poses = P.load_poses()
rot, t = poses[frame]
R = P.quat_to_matrix(rot)
cam_center = -R.T @ t

mean = np.stack([col['x'], -col['y'], -col['z']],1).astype(np.float64)
camv = mean @ R.T + t
z = camv[:,2]
invz = 1.0/np.where(z!=0, z, 1)
mx = fx*camv[:,0]*invz + cx
my = fy*camv[:,1]*invz + cy

scale = np.exp(np.clip(np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64),-12,3))
q = np.stack([col['rot_1'],-col['rot_2'],-col['rot_3'],col['rot_0']],1).astype(np.float64)
q /= np.linalg.norm(q,axis=1,keepdims=True)
x_,y_,z_,w_ = q[:,0],q[:,1],q[:,2],q[:,3]
Rm = np.empty((count,3,3))
Rm[:,0,0]=1-2*(y_*y_+z_*z_); Rm[:,0,1]=2*(x_*y_-w_*z_); Rm[:,0,2]=2*(x_*z_+w_*y_)
Rm[:,1,0]=2*(x_*y_+w_*z_);   Rm[:,1,1]=1-2*(x_*x_+z_*z_); Rm[:,1,2]=2*(y_*z_-w_*x_)
Rm[:,2,0]=2*(x_*z_-w_*y_);   Rm[:,2,1]=2*(y_*z_+w_*x_);   Rm[:,2,2]=1-2*(x_*x_+y_*y_)
M = Rm*scale[:,None,:]
sigW = M @ np.transpose(M,(0,2,1))
sigC = R @ sigW @ R.T
j00=fx*invz; j11=fy*invz; j02=-fx*camv[:,0]*invz*invz; j12=-fy*camv[:,1]*invz*invz
s00,s01,s02 = sigC[:,0,0],sigC[:,0,1],sigC[:,0,2]
s11,s12,s22 = sigC[:,1,1],sigC[:,1,2],sigC[:,2,2]
a0=j00*s00+j02*s02; a1=j00*s01+j02*s12; a2=j00*s02+j02*s22
b1=j11*s11+j12*s12; b2=j11*s12+j12*s22
sa = a0*j00+a2*j02 + P.FILTER_2D_VARIANCE
sb = a1*j11+a2*j12
sc = b1*j11+b2*j12 + P.FILTER_2D_VARIANCE
det = np.maximum(sa*sc-sb*sb,1e-12); invdet=1.0/det
conic = np.stack([sc*invdet, -sb*invdet, sa*invdet],1)
det_before = np.maximum((sa-P.FILTER_2D_VARIANCE)*(sc-P.FILTER_2D_VARIANCE)-sb*sb,1e-12)
comp2d = np.sqrt(np.clip(det_before/det,0,1))
opac = (1.0/(1.0+np.exp(-col['opacity'].astype(np.float64))))*comp2d

# SH colour, degree 1 (shDegree 1 -> 4 coeffs; PLY carries f_dc + 9 f_rest = 4 coeffs)
dirv = mean - cam_center
dirv /= np.maximum(np.linalg.norm(dirv,axis=1,keepdims=True),1e-6)
dc = np.stack([col['f_dc_0'],col['f_dc_1'],col['f_dc_2']],1).astype(np.float64)
rest = np.stack([col['f_rest_%d'%i] for i in range(9)],1).astype(np.float64)
# PLY layout from PLYCodec: f_rest is coefficient-major (c1r,c1g,c1b, c2..., c3...)
s1 = rest[:,0:3]; s2 = rest[:,3:6]; s3 = rest[:,6:9]
rgb = SH_C0*dc + (-SH_C1*dirv[:,1:2]*s1 + SH_C1*dirv[:,2:3]*s2 - SH_C1*dirv[:,0:1]*s3) + 0.5
clamped = rgb < 0
rgb = np.maximum(rgb, 0.0)

mid = 0.5*(sa+sc); disc = np.sqrt(np.maximum(mid*mid-det,1e-9))
radius = 3.0*np.sqrt(np.maximum(mid+disc,1e-9))
level = np.clip(2.0*np.log(opac/max(P.MIN_ALPHA,1e-8)),0,9)
kk = np.sqrt(level)
ex = kk*np.sqrt(np.maximum(sa,1e-9))*1.001
ey = kk*np.sqrt(np.maximum(sc,1e-9))*1.001
ok = (z>0.05)&(z<100)&(det>1e-12)&(radius>=0.5)&(opac>=P.MIN_ALPHA)
minx=np.maximum(0,np.floor((mx-ex)/TILE)).astype(np.int64)
miny=np.maximum(0,np.floor((my-ey)/TILE)).astype(np.int64)
maxx=np.minimum(tx,np.ceil((mx+ex)/TILE)).astype(np.int64)
maxy=np.minimum(ty,np.ceil((my+ey)/TILE)).astype(np.int64)
ok &= (maxx>minx)&(maxy>miny)
idx = np.nonzero(ok)[0]
print('frame %d: %d splats drawn of %d'%(frame, idx.size, count))

# ---------------- pass 1: forward render -----------------------------------
render_c = np.zeros((N,3)); render_d = np.zeros(N); render_T = np.ones(N)
tile_lists = {}
t0=time.time()
for ti in range(tx*ty):
    ttx, tty = ti%tx, ti//tx
    sel = idx[(minx[idx]<=ttx)&(maxx[idx]>ttx)&(miny[idx]<=tty)&(maxy[idx]>tty)]
    if sel.size==0: continue
    order = sel[np.argsort(z[sel], kind='stable')]
    tile_lists[ti]=order
    px = ttx*TILE+np.arange(TILE)+0.5; py = tty*TILE+np.arange(TILE)+0.5
    gx,gy = np.meshgrid(px,py); gx=gx.ravel(); gy=gy.ravel()
    dx = gx[:,None]-mx[order][None,:]; dy = gy[:,None]-my[order][None,:]
    pw = -0.5*(conic[order,0][None,:]*dx*dx + conic[order,2][None,:]*dy*dy) - conic[order,1][None,:]*dx*dy
    al = np.minimum(0.99, opac[order][None,:]*np.exp(np.clip(pw,-60,0)))
    al = np.where(al>=P.MIN_ALPHA, al, 0.0)
    Tc = np.cumprod(1.0-al,axis=1)
    before = np.concatenate([np.ones((256,1)), Tc[:,:-1]],axis=1)
    done = Tc < 1e-4
    first_done = np.where(done.any(1), done.argmax(1), al.shape[1])
    live = np.arange(al.shape[1])[None,:] <= first_done[:,None]
    al = np.where(live, al, 0.0)
    Tc = np.cumprod(1.0-al,axis=1)
    before = np.concatenate([np.ones((256,1)), Tc[:,:-1]],axis=1)
    wgt = al*before
    gyi = (tty*TILE+np.arange(TILE)); gxi=(ttx*TILE+np.arange(TILE))
    GY,GX = np.meshgrid(gxi,gyi)  # note: meshgrid ordering matched to gx,gy above
    pix = (np.meshgrid(gxi,gyi)[1]*RW + np.meshgrid(gxi,gyi)[0]).ravel()
    inside = (np.meshgrid(gxi,gyi)[0].ravel()<RW)&(np.meshgrid(gxi,gyi)[1].ravel()<RH)
    cc = wgt @ rgb[order]
    dd = wgt @ z[order]
    render_c[pix[inside]] = cc[inside]
    render_d[pix[inside]] = dd[inside]
    render_T[pix[inside]] = Tc[inside,-1]
print('forward %.1fs, %d tiles'%(time.time()-t0, len(tile_lists)))
np.savez(os.path.join(os.path.dirname(__file__),'_fwd_%d.npz'%frame),
         c=render_c, d=render_d, T=render_T)
alpha_px = 1.0-render_T
print('rendered alpha: p10 %.4f p50 %.4f p90 %.4f'%tuple(np.percentile(alpha_px,[10,50,90])))
exp_d = render_d/np.maximum(alpha_px,1e-4)
m = alpha_px>0.05
print('expected depth (m) on alpha>0.05 (%.1f%% of px): p10 %.3f p50 %.3f p90 %.3f'
      %(100*m.mean(), *np.percentile(exp_d[m],[10,50,90])))
