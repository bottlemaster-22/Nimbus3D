"""REFUTATION TEST 6: contrib.py claims its 118-view sample is "deliberately a
SUPERSET of the ~120 keyframes actually used in training".  Check it.

The 12 held-out frames run 17..460 and are every 10th keyframe, so the 120
keyframes are spaced ~4 frames and span roughly 0..470 -- NOT 0..867.  The
stride-8 sample therefore (a) covers only about every other keyframe inside
that span and (b) spends ~42% of its views on frames 472..864 that the
trainer never saw.  Recompute contribution using ONLY in-span, non-held-out
views (the closest available proxy for the real supervision set) and see what
the "never drawn" and low-contribution counts do.
"""
import io, json, os, time
import numpy as np
import project as P
import detail as Dt
import render as Rn
import contrib as Cb

SCRATCH=Dt.SCRATCH; TILE=16
census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx,ty=(rw+TILE-1)//TILE,(rh+TILE-1)//TILE
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses()
held=sorted(json.load(io.open(os.path.join(P.D,'model','held_out_frames.json'),encoding='utf-8')))
sample=json.load(io.open(os.path.join(SCRATCH,'contrib_frames.json'),encoding='utf-8'))

print('held-out frames: %s' % held)
d=np.diff(held); print('held-out spacing: %s  -> implied keyframe spacing ~%.1f frames'
                       % (list(d), d.mean()/10))
print('implied keyframe span: 0 .. ~%d of 0..867' % (held[-1]+d.mean()/2))
inspan=[f for f in sample if f<=held[-1]+20]
outspan=[f for f in sample if f>held[-1]+20]
print('sample of %d views: %d inside the training span, %d OUTSIDE it (%.0f%%)'
      % (len(sample),len(inspan),len(outspan),100*len(outspan)/len(sample)))
train_like=[f for f in inspan if f not in held]
print('in-span, non-held-out views used here: %d' % len(train_like))

p=os.path.join(SCRATCH,'contrib_inspan.npy')
if os.path.exists(p):
    c=np.load(p); ed=np.load(os.path.join(SCRATCH,'ever_inspan.npy'))
else:
    c=np.zeros(count); ed=np.zeros(count,bool); t0=time.time()
    for i,f in enumerate(train_like):
        rot,t=poses[f]; R=P.quat_to_matrix(rot)
        Cb.accumulate(col,count,R,t,fx,fy,cx,cy,tx,ty,rw,rh,c,ed)
        if (i+1)%10==0: print('  %d/%d (%.0fs)'%(i+1,len(train_like),time.time()-t0))
    np.save(p,c); np.save(os.path.join(SCRATCH,'ever_inspan.npy'),ed)

full=np.load(os.path.join(SCRATCH,'contrib.npy'))
print('\n%-34s %12s %12s' % ('', '118-view', 'in-span %d'%len(train_like)))
for lbl,a,b in (('never drawn (no projection)', 300000-np.load(os.path.join(SCRATCH,'ever_drawn.npy')).sum(), count-ed.sum()),
                ('exactly zero weight', (full<=0).sum(), (c<=0).sum()),
                ('weight < 1.0', (full<1).sum(), (c<1).sum()),
                ('weight < 3.0', (full<3).sum(), (c<3).sum())):
    print('%-34s %7d (%4.2f%%) %7d (%4.2f%%)'%(lbl,a,100*a/count,b,100*b/count))

# re-rank on the in-span-only contribution and re-score at the held-out poses
def psnr(x,y): return 10*np.log10(1.0/max(((x-y)**2).mean(),1e-12))
def sub(cl,m): return {k:v[m] for k,v in cl.items()}
n=count; nd=n//2
m_full=np.ones(n,bool); m_full[np.argsort(full)[:nd]]=False
m_span=np.ones(n,bool); m_span[np.argsort(c)[:nd]]=False
print('\ncut disagreement between the two rankings: %.1f%% of the 150,000'
      % (100*(~m_full & m_span).sum()/nd))
res={'118-view ranking':[], 'in-span ranking':[]}
for f in held[:6]:
    rot,t=poses[f]; R=P.quat_to_matrix(rot)
    fu,_=Rn.render(col,count,R,t,fx,fy,cx,cy,rw,rh)
    for name,m in (('118-view ranking',m_full),('in-span ranking',m_span)):
        im,_=Rn.render(sub(col,m),int(m.sum()),R,t,fx,fy,cx,cy,rw,rh)
        res[name].append(psnr(im,fu))
    print('  held-out %d done'%f)
print('\n=== held-out-pose self-consistency vs FULL, drop 50%% ===')
for k,v in res.items(): print('  %-18s %6.3f dB'%(k,float(np.mean(v))))
