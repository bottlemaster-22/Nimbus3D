"""Re-examine the EWA tangent-clamp finding against the BUILD 244 model+census."""
import numpy as np, project as P
from sv_clamp_run import run
import json,io,os

col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses(); fx,fy,cx,cy=P.load_intrinsics(720,540); tx,ty=45,34
cen=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
s=cen['slices'][0]
print('census peakTileInstances %d at %d splats (build %s)'%(s['peakTileInstances'],s['splatCountAtPeakTileInstances'],cen['appVersion']))
op=1/(1+np.exp(-col['opacity'].astype(np.float64)))
ks=sorted(poses)[::20]
for label,clamp in [('none',None),('1.30',1.3)]:
    tot=[];vis=[];big=[];bigti=[];bigop=[]
    for k in ks:
        R=P.quat_to_matrix(poses[k][0]); t=poses[k][1]
        ok,ti,rad,off=run(col,count,R,t,fx,fy,cx,cy,tx,ty,clamp)
        tot.append(ti.sum()); vis.append(ok.sum())
        m=ok&(off>1.3)
        big.append(m.sum()); bigti.append(ti[m].sum())
        if m.sum(): bigop.append(op[m])
    tot=np.array(tot,float);vis=np.array(vis,float)
    print('clamp %-5s peak %9d mean %9d  visible %7.0f  |x/z|>1.3 splats/view %6.1f  their tileInst/view %8.0f (%5.2f%%)'
          %(label,tot.max(),tot.mean(),vis.mean(),np.mean(big),np.mean(bigti),100*np.mean(bigti)/tot.mean()))
    if label=='none' and bigop:
        a=np.concatenate(bigop); print('   their opacity: n=%d  p10 %.4f p50 %.4f p90 %.4f  frac>0.0039 %.3f  frac>0.05 %.3f'%(len(a),*np.percentile(a,[10,50,90]),(a>1/255).mean(),(a>0.05).mean()))
print()
print('offline unclamped peak / census peak = %.3f'%(438057/ s['peakTileInstances']))
