import json, io, os, numpy as np, time
import project as P, raster as RS
D=P.D
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses()
allkf=np.load('_kf.npy'); vis=np.load('_vis.npy')
tr=np.array([i%10!=5 for i in range(len(allkf))])
peak=np.zeros(count); nvis=np.zeros(count,int)
t0=time.time()
for k,idx in enumerate(allkf):
    if not tr[k]: continue
    R=P.quat_to_matrix(poses[idx][0]); t=poses[idx][1]
    g=RS.geometry(col,count,R,t,fx,fy,cx,cy)
    a=np.where(vis[k], g['opacity'], 0.0)
    peak=np.maximum(peak,a); nvis+=vis[k]
print('computed in %.0fs'%(time.time()-t0))
np.save('_peakalpha.npy',peak)
MIN=1.0/255.0
print()
print('=== PEAK DRAWN ALPHA PER GAUSSIAN, best of the 108 trained views ===')
print('   (PLY is fuse3DFilter-fused, so sigmoid(ply opacity) already carries comp3D;')
print('    peak = sigmoid(opacity)*comp2D at the splat centre = exactly the rasteriser value)')
print('median %.4f   p10 %.5f  p25 %.4f  p75 %.4f  p90 %.4f  max %.4f'%(
  np.median(peak),np.percentile(peak,10),np.percentile(peak,25),np.percentile(peak,75),np.percentile(peak,90),peak.max()))
print()
print('BELOW the rasteriser 1/255 reject : %7d  (%5.2f%%)'%((peak<MIN).sum(),100*(peak<MIN).mean()))
print('BELOW pruneOpacity 0.02           : %7d  (%5.2f%%)'%((peak<0.02).sum(),100*(peak<0.02).mean()))
for th in [0.05,0.1,0.2,0.5]:
    print('BELOW %.2f                        : %7d  (%5.2f%%)'%(th,(peak<th).sum(),100*(peak<th).mean()))
print()
print('census says: prunedLowOpacity=0, prunedOversized=0, prune window open in 0 of 39 passes')
# raw stored opacity for comparison
raw=1/(1+np.exp(-col['opacity'].astype(np.float64)))
print()
print('STORED sigmoid(opacityLogit) (what a comp-blind prune would have tested):')
print('median %.4f   below 0.02: %d (%.2f%%)'%(np.median(raw),(raw<0.02).sum(),100*(raw<0.02).mean()))
