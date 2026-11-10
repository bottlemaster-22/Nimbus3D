"""How much of the 16-bit depth key is actually used, given nearPlane 0.05 and
farPlane 100 (MetalSplatTrainer.swift:2260-2261)."""
import io, json, os, numpy as np
import project as P
D=P.D
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx,ty=(rw+15)//16,(rh+15)//16
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(D,'model','model.ply'))
poses=P.load_poses(); ks=sorted(poses.keys())
NEAR,FAR=0.05,100.0
allz=[]
for idx in ks[::7]:
    rot,t=poses[idx]; R=P.quat_to_matrix(rot)
    ok,radius,tiles,ex,ey=P.project(col,count,R,t,fx,fy,cx,cy,tx,ty)
    mean=np.stack([col['x'],-col['y'],-col['z']],axis=1).astype(np.float64)
    z=(mean@R.T+t)[:,2]
    s=ok&(tiles>0)
    allz.append(np.repeat(z[s], tiles[s]))
z=np.concatenate(allz)
print('visible tile-instance depths over %d views, %d instances' % (len(ks[::7]), len(z)))
for q in (0,0.1,1,50,99,99.9,100):
    print('   p%-5s %8.3f m' % (q, np.percentile(z,q)))
span=FAR-NEAR
lo,hi=np.percentile(z,0.1),np.percentile(z,99.9)
print('\nkey levels used (16 bit, near %.2f far %.1f): %.0f of 65536 = %.1f%% -> effective %.1f bits'
      % (NEAR,FAR,(hi-lo)/span*65535,(hi-lo)/span*100, np.log2(max((hi-lo)/span*65535,1))))
print('   quantum today            : %.3f mm' % (span/65535*1000))
print('   quantum at 13 bits, same range : %.3f mm' % (span/8191*1000))
p999=np.percentile(z,99.9)
for nf in [(0.05,p999),(0.05,12.0),(0.10,10.0)]:
    n_,f_=nf
    print('   13 bits over near %.2f far %.2f -> quantum %.3f mm  (clipped instances %.3f%%)'
          % (n_,f_,(f_-n_)/8191*1000, 100*float((z>f_).mean())))
