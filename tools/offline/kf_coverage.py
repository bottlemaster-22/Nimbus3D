import json, numpy as np
D=r'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840'
bundle=json.load(open(D+'/capture_bundle.json'))
pre=json.load(open(D+'/prepass/prepass_result.json'))
rp=pre['refinedPoses']
def qinv_act(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v)
    return v + w*t + np.cross(qv,t)
frames=sorted(bundle['frames'],key=lambda f:f['index'])
C=[];F=[];T=[]
for f in frames:
    p=rp[str(f['index'])]
    q=np.array(p['rotation']); t=np.array(p['translation'])
    C.append(-qinv_act(q,t)); F.append(qinv_act(q,np.array([0.,0.,1.]))); T.append(f['timestampSeconds'])
C=np.array(C);F=np.array(F);T=np.array(T)
seg=np.linalg.norm(np.diff(C,axis=0),axis=1)
cum=np.concatenate([[0],np.cumsum(seg)])
CUT=486
print('=== WHAT THE TRAINER NEVER SEES ===')
print('total path length      : %.2f m'%cum[-1])
print('path in frames 0..485  : %.2f m  (%.1f%%)'%(cum[CUT-1],100*cum[CUT-1]/cum[-1]))
print('path in frames 486..867: %.2f m  (%.1f%%)'%(cum[-1]-cum[CUT-1],100*(cum[-1]-cum[CUT-1])/cum[-1]))
print('capture duration       : %.1f s total'%(T[-1]-T[0]))
print('  time 0..485          : %.1f s (%.1f%%)'%(T[CUT-1]-T[0],100*(T[CUT-1]-T[0])/(T[-1]-T[0])))
print('  time 486..867        : %.1f s (%.1f%%)'%(T[-1]-T[CUT-1],100*(T[-1]-T[CUT-1])/(T[-1]-T[0])))
print()
print('=== IS THE LATE HALF NEW SPACE OR A REVISIT? ===')
A=C[:CUT]; B=C[CUT:]
print('camera bbox  frames 0..485  : min',np.round(A.min(0),2),' max',np.round(A.max(0),2))
print('camera bbox  frames 486..867: min',np.round(B.min(0),2),' max',np.round(B.max(0),2))
# nearest distance from each late camera to any early camera
from scipy.spatial import cKDTree
tr=cKDTree(A)
d,_=tr.query(B)
print('late-camera distance to NEAREST early camera: median %.3f m  p90 %.3f  max %.3f'%(np.median(d),np.percentile(d,90),d.max()))
print('late cameras further than 1.0 m from every early camera: %d of %d (%.1f%%)'%((d>1.0).sum(),len(B),100*(d>1.0).mean()))
print('late cameras further than 0.5 m: %d (%.1f%%)'%((d>0.5).sum(),100*(d>0.5).mean()))
