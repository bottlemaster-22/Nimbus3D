import io, json, os, numpy as np
import project as P
D = P.D
census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()
print('splats %d  render %dx%d  tiles %dx%d=%d  px=%d' % (count, rw, rh, tx, ty, tx*ty, rw*rh))
keys = sorted(poses)[:200:8]
dr=[]; ti=[]
for k in keys:
    rot,t = poses[k]; R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    dr.append(int(ok.sum())); ti.append(int(tiles.sum()))
dr=np.array(dr); ti=np.array(ti)
print('frames sampled %d' % len(keys))
print('drawn splats  mean %.0f (%.1f%%)  min %d  max %d' % (dr.mean(), 100*dr.mean()/count, dr.min(), dr.max()))
print('tile instances mean %.0f  min %d  max %d  peak-in-census %d' % (ti.mean(), ti.min(), ti.max(), sl['peakTileInstances']))
print('tiles per drawn splat mean %.2f' % (ti.mean()/dr.mean()))
