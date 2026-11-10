"""REFUTATION TESTS 4 and 5.

(4) LEAKAGE.  contrib.py's 118-view sample UNIONS IN the 12 held-out frames,
    then prune_psnr.py part (B) evaluates the resulting ranking AT THOSE SAME
    12 held-out poses.  The splats kept were chosen partly by how much they
    contribute to the very views used to score them.  Re-rank on a contrib
    that has the held-out frames' share subtracted out and re-score.

(5) EXPOSURE FLOOR.  The full-model render sits 1.7 dB below its own
    gain+offset fit, so part (A)'s residual is dominated by a photometric
    error the prune cannot touch, which COMPRESSES any geometry delta read
    off it.  Re-measure the drop-50%% delta after fitting gain+offset.
"""
import io, json, os, sys, time
import numpy as np
import project as P
import detail as Dt
import render as Rn
import contrib as Cb

SCRATCH = Dt.SCRATCH
TILE = 16

def psnr(a,b):
    return 10*np.log10(1.0/max(((a-b)**2).mean(),1e-12))

def expfit(img, ref):
    out=np.empty_like(img)
    for ch in range(3):
        x=img[:,:,ch].ravel(); y=ref[:,:,ch].ravel()
        A=np.stack([x,np.ones_like(x)],1)
        g,o=np.linalg.lstsq(A,y,rcond=None)[0]
        out[:,:,ch]=g*img[:,:,ch]+o
    return np.clip(out,0,1)

def subset(col,m): return {k:v[m] for k,v in col.items()}

census=json.load(io.open(os.path.join(P.D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
tx,ty=(rw+TILE-1)//TILE,(rh+TILE-1)//TILE
fx,fy,cx,cy=P.load_intrinsics(rw,rh)
col,count=P.load_ply(os.path.join(P.D,'model','model.ply'))
poses=P.load_poses()
held=sorted(json.load(io.open(os.path.join(P.D,'model','held_out_frames.json'),encoding='utf-8')))
contrib=np.load(os.path.join(SCRATCH,'contrib.npy'))

# ---- (4) contribution attributable to the 12 held-out views only ----
ho_path=os.path.join(SCRATCH,'contrib_heldout.npy')
if os.path.exists(ho_path):
    c_ho=np.load(ho_path)
else:
    c_ho=np.zeros(count); ed=np.zeros(count,bool); t0=time.time()
    for f in held:
        rot,t=poses[f]; R=P.quat_to_matrix(rot)
        Cb.accumulate(col,count,R,t,fx,fy,cx,cy,tx,ty,rw,rh,c_ho,ed)
        print('  held-out %d accumulated (%.0fs)'%(f,time.time()-t0))
    np.save(ho_path,c_ho)

c_tr = contrib - c_ho
print('\nheld-out share of total contribution: %.2f%%'
      % (100*c_ho.sum()/contrib.sum()))
print('negatives after subtraction (should be 0): %d' % (c_tr<-1e-9).sum())
c_tr=np.maximum(c_tr,0)

n=count; ndrop=n//2
ord_leaky=np.argsort(contrib); ord_clean=np.argsort(c_tr)
m_leaky=np.ones(n,bool); m_leaky[ord_leaky[:ndrop]]=False
m_clean=np.ones(n,bool); m_clean[ord_clean[:ndrop]]=False
print('cuts differ on %d splats (%.1f%% of the 150,000 cut)'
      % ((m_leaky!=m_clean).sum()//2*2, 100*(~m_leaky & m_clean).sum()/ndrop))

rng=np.random.default_rng(20260909)
m_rand=np.ones(n,bool); m_rand[rng.permutation(n)[:ndrop]]=False

print('\n=== (4) held-out-pose self-consistency vs FULL, drop 50%% ===')
res={'leaky-contrib':[], 'clean-contrib':[], 'random':[]}
for f in held[:6]:
    rot,t=poses[f]; R=P.quat_to_matrix(rot)
    full,_=Rn.render(col,count,R,t,fx,fy,cx,cy,rw,rh)
    for name,m in (('leaky-contrib',m_leaky),('clean-contrib',m_clean),('random',m_rand)):
        img,_=Rn.render(subset(col,m),int(m.sum()),R,t,fx,fy,cx,cy,rw,rh)
        res[name].append(psnr(img,full))
    print('  held-out %d done'%f)
for k,v in res.items():
    print('  %-14s %6.3f dB   %s'%(k,float(np.mean(v)),['%.1f'%x for x in v]))

print('\n=== (5) PSNR vs real photo, RAW vs after gain+offset fit, drop 50%% ===')
imgs=dict(Dt.frames_with_images()); pf=sorted(imgs)[:6]
raw={'FULL':[], 'contrib':[], 'random':[]}; fit={'FULL':[], 'contrib':[], 'random':[]}
for f in pf:
    _,rgb=Dt.luma_of(imgs[f],rw,rh)
    rot,t=poses[f]; R=P.quat_to_matrix(rot)
    for name,m in (('FULL',np.ones(n,bool)),('contrib',m_leaky),('random',m_rand)):
        img,_=Rn.render(subset(col,m),int(m.sum()),R,t,fx,fy,cx,cy,rw,rh)
        raw[name].append(psnr(img,rgb)); fit[name].append(psnr(expfit(img,rgb),rgb))
    print('  frame %d done'%f)
br=float(np.mean(raw['FULL'])); bf=float(np.mean(fit['FULL']))
print('%-9s %10s %9s %12s %9s'%('mask','PSNR raw','delta','PSNR expfit','delta'))
for k in ('FULL','contrib','random'):
    r=float(np.mean(raw[k])); q=float(np.mean(fit[k]))
    print('%-9s %10.3f %9.3f %12.3f %9.3f'%(k,r,r-br,q,q-bf))
