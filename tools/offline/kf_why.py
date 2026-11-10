"""WHY does the greedy fill its quota in the first 56% of the walk?
selectKeyframes accepts a frame if it MOVED >= spacing OR TURNED >= 0.02
(1-cos, i.e. about 11.5 degrees). The turn clause fires far more often than
the spacing clause, so the quota is spent early and `break` ends the scan."""
import json,numpy as np
import project as P
D=P.D
bundle=json.load(open(D+'\capture_bundle.json')); pre=json.load(open(D+'\prepass\prepass_result.json'))
rp=pre['refinedPoses']
def qi(q,v):
    x,y,z,w=q; qv=np.array([-x,-y,-z]); t=2*np.cross(qv,v); return v+w*t+np.cross(qv,t)
frames=sorted(bundle['frames'],key=lambda f:f['index'])
C={};F={}
for f in frames:
    p=rp[str(f['index'])]; q=np.array(p['rotation']); t=np.array(p['translation'])
    C[f['index']]=-qi(q,t); F[f['index']]=qi(q,np.array([0.,0.,1.]))
pool=[f for f in frames if f['qc']['weight']>0.05] or frames
pl=0.0;prev=None
for f in pool:
    c=C[f['index']]
    if prev is not None: pl+=np.linalg.norm(c-prev)
    prev=c
spacing=pl/120.0
chosen=[];lc=None;lf=None;why=[]
for f in pool:
    c=C[f['index']];fw=F[f['index']]
    if lc is not None:
        mv=np.linalg.norm(lc-c); tn=1-np.dot(fw/np.linalg.norm(fw),lf/np.linalg.norm(lf))
        if mv<spacing and tn<0.02: continue
        why.append(('moved' if mv>=spacing else '')+('+turned' if tn>=0.02 else ''))
    else: why.append('first')
    chosen.append(f['index']);lc=c;lf=fw
    if len(chosen)>=120: break
from collections import Counter
print('spacing threshold = pathLength/target = %.4f m'%spacing)
print('turn threshold    = 1-cos(theta) >= 0.02  ->  theta >= %.1f degrees'%np.degrees(np.arccos(1-0.02)))
print()
print('WHY each of the 120 keyframes was accepted:', dict(Counter(why)))
n_turn_only=sum(1 for w in why if w=='+turned')
print('  accepted on the TURN clause alone (moved less than spacing): %d of 120 = %.0f%%'%(n_turn_only,100*n_turn_only/120))
print()
print('last chosen bundle frame: %d of %d  -> the loop `break`s after examining %.0f%% of the capture'%(
  chosen[-1],frames[-1]['index'],100*chosen[-1]/frames[-1]['index']))
seg=[np.linalg.norm(C[frames[i+1]['index']]-C[frames[i]['index']]) for i in range(len(frames)-1)]
cum=np.concatenate([[0],np.cumsum(seg)])
print('path consumed by those 120 keyframes: %.2f m of %.2f m (%.0f%%)'%(cum[chosen[-1]],cum[-1],100*cum[chosen[-1]]/cum[-1]))
print()
print('=== WHAT AN EVEN STRIDE OVER THE WHOLE POOL WOULD GIVE ===')
st=max(len(pool)//120,1)
alt=[pool[i]['index'] for i in range(0,len(pool),st)][:120]
print('stride %d -> %d frames, first %d last %d, spanning %.0f%% of the capture'%(
  st,len(alt),alt[0],alt[-1],100*(alt[-1]-alt[0])/frames[-1]['index']))
A=np.array([C[i] for i in chosen]); B=np.array([C[i] for i in alt])
def spread(X):
    X=X-X.mean(0); ev=np.linalg.svd(X,compute_uv=False)**2/len(X); return np.sqrt(ev)
print('camera-centre spread (PCA sigma, m):')
print('   as shipped   : %.3f %.3f %.3f   bbox %s'%(*spread(A),np.round(A.max(0)-A.min(0),2)))
print('   even stride  : %.3f %.3f %.3f   bbox %s'%(*spread(B),np.round(B.max(0)-B.min(0),2)))
d=lambda X: np.linalg.norm(X[:,None]-X[None],axis=2)[np.triu_indices(len(X),1)]
print('median pairwise camera baseline: as shipped %.3f m, even stride %.3f m'%(np.median(d(A)),np.median(d(B))))
