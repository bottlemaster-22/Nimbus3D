import json, numpy as np
D=r'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840'
bundle=json.load(open(D+'/capture_bundle.json'))
pre=json.load(open(D+'/prepass/prepass_result.json'))
rp=pre['refinedPoses']

def qinv_act(q, v):
    # q = [x,y,z,w], world->cam rotation R. inverse acts: R^T v
    x,y,z,w=q
    # inverse of unit quaternion = conjugate
    xi,yi,zi,wi=-x,-y,-z,w
    # rotate v by qi:  v' = v + 2*qv x (qv x v + w*v)
    qv=np.array([xi,yi,zi]); t=2*np.cross(qv,v)
    return v + wi*t + np.cross(qv,t)

frames=sorted(bundle['frames'], key=lambda f:f['index'])
print('frames in bundle:', len(frames))
centers=[]; forwards=[]; idxs=[]; weights=[]; ts=[]
for f in frames:
    i=f['index']
    p = rp.get(str(i)) or f.get('refinedPose') or f['rawPose']
    q=np.array(p['rotation'],dtype=np.float64)
    t=np.array(p['translation'],dtype=np.float64)
    C = -qinv_act(q,t)
    F = qinv_act(q,np.array([0.,0.,1.]))
    centers.append(C); forwards.append(F); idxs.append(i); weights.append(f['qc']['weight']); ts.append(f['timestampSeconds'])
centers=np.array(centers); forwards=np.array(forwards); idxs=np.array(idxs); weights=np.array(weights); ts=np.array(ts)
print('refinedPoses present for', sum(1 for f in frames if str(f['index']) in rp), 'of', len(frames))

# pool = qc.weight > 0.05
mask = weights > 0.05
pool = np.where(mask)[0]
if len(pool)==0: pool=np.arange(len(frames))
print('pool size (qc.weight>0.05):', len(pool), 'of', len(frames))
print('qc weight stats: min %.4f p05 %.4f median %.4f max %.4f'%(weights.min(),np.percentile(weights,5),np.median(weights),weights.max()))
print('frames with weight<=0.05:', int((weights<=0.05).sum()))

target = max(120, 8)
pc = centers[pool]; pf = forwards[pool]
pathLength = np.sum(np.linalg.norm(np.diff(pc,axis=0),axis=1))
spacing = pathLength/target
print('path length (m): %.3f   spacing (m): %.4f'%(pathLength, spacing))

chosen=[]; lastC=None; lastF=None
skipped=0
for k in range(len(pool)):
    c=pc[k]; f=pf[k]
    if lastC is not None:
        moved=np.linalg.norm(lastC-c)
        turned=1-np.dot(f/np.linalg.norm(f), lastF/np.linalg.norm(lastF))
        if moved<spacing and turned<0.02:
            skipped+=1; continue
    chosen.append(k); lastC=c; lastF=f
    if len(chosen)>=target: break
chosen=np.array(chosen)
sel_frame_idx = idxs[pool[chosen]]
print()
print('CHOSEN keyframes:', len(chosen))
print('bundle frame indices of chosen: first', sel_frame_idx[:8].tolist())
print('  ... last', sel_frame_idx[-8:].tolist())
print('LAST chosen bundle frame index:', int(sel_frame_idx[-1]), 'of max', int(idxs.max()))
print('fraction of capture covered by keyframes: %.1f%%'%(100.0*(sel_frame_idx[-1]-sel_frame_idx[0])/(idxs.max()-idxs.min())))
print('loop broke early:', len(chosen)>=target, ' frames examined:', int(chosen[-1])+1, 'of', len(pool))

heldout_positions=[i for i in range(len(chosen)) if i%10==5]
print()
print('held-out keyframe positions (i%%10==5):', heldout_positions)
print('their bundle frame indices:', sel_frame_idx[heldout_positions].tolist())
print('ACTUAL held_out_frames.json:      ', json.load(open(D+'/model/held_out_frames.json')))
