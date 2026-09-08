import sv_load, numpy as np
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
c,n = sv_load.sv()
sent = np.load(SCR+r"\sv_sentinel.npy")
P = np.stack([c['x'],c['y'],c['z']],1).astype(np.float64)
D = P[sent]
ctr = D.mean(0)
print('sentinel centroid: %s' % np.round(ctr,4))
r = np.linalg.norm(D-ctr,axis=1)
print('radius from centroid: min %.4f p1 %.4f p50 %.4f p99 %.4f max %.4f   std/mean %.6f'
      % (r.min(), np.percentile(r,1), np.median(r), np.percentile(r,99), r.max(), r.std()/r.mean()))
# least-squares sphere fit
A = np.hstack([2*D, np.ones((len(D),1))]); b=(D**2).sum(1)
sol,*_ = np.linalg.lstsq(A,b,rcond=None); C=sol[:3]; R=np.sqrt(sol[3]+ (C**2).sum())
rr=np.linalg.norm(D-C,axis=1)
print('LSQ sphere: centre %s  radius %.5f   residual rms %.6f m  max %.6f m' % (np.round(C,4), R, np.sqrt(((rr-R)**2).mean()), np.abs(rr-R).max()))
print('  -> residual as fraction of radius: rms %.3e' % (np.sqrt(((rr-R)**2).mean())/R))
print('  position quantisation step is 2.44e-4 m, so a perfect sphere would show rms ~7e-5')
# icosphere check: nearest-neighbour angular spacing uniformity
u = (D-C)/rr[:,None]
# valence: for a subdivided icosahedron, 12 vertices have 5 neighbours, rest 6
from scipy.spatial import cKDTree
t=cKDTree(u); dd,_ = t.query(u,k=7)
nn = dd[:,1]
print('nearest-neighbour angular chord: p1 %.5f p50 %.5f p99 %.5f  (spread %.3f%%)'
      % (np.percentile(nn,1), np.median(nn), np.percentile(nn,99), 100*(np.percentile(nn,99)/np.percentile(nn,1)-1)))
# count neighbours within 1.3x median chord
med=np.median(nn); cnt=np.array([len(x)-1 for x in t.query_ball_point(u, med*1.25)])
vals,c2=np.unique(cnt,return_counts=True)
print('neighbour valence histogram (within 1.25x median chord):', dict(zip(vals.tolist(),c2.tolist())))
print('  icosphere prediction: exactly 12 vertices of valence 5, %d of valence 6' % (len(D)-12))
print('  chord for icosphere subdiv 5 approx %.5f (10242 verts on unit sphere)' % (np.sqrt(4*np.pi/len(D)/(np.sqrt(3)/2))))
