import json, numpy as np, os, glob
try:
    from PIL import Image
except Exception as e:
    print("no PIL:", e); raise SystemExit

B=r"C:/Users/Undea/Documents/LiKOVA/Scans/Incoming/scan_20260906_164840"
b=json.load(open(B+"/capture_bundle.json"))
fr=b['frames']
print("settings:", {k:v for k,v in b['settings'].items() if 'xpos' in k.lower() or 'racket' in k.lower()})
dur=np.array([f.get('exposureDurationSeconds',0) for f in fr])
ev =np.array([f.get('exposureOffsetEV',0) or 0 for f in fr])
iso=[f.get('iso') for f in fr]
print(f"frames {len(fr)}  duration min {dur.min():.6f} max {dur.max():.6f} distinct {len(set(dur.round(8)))}")
print(f"exposureOffsetEV nonzero on {int((ev!=0).sum())}/{len(fr)}   iso non-nil on {sum(1 for i in iso if i is not None)}/{len(fr)}")
jq=np.array([f['qc'].get('exposureJumpEV',0) for f in fr])
print(f"qc.exposureJumpEV nonzero on {int((jq!=0).sum())}/{len(fr)}  max {jq.max()}")

imgs=sorted(glob.glob(B+"/images/*.jpg"))
print(f"\n{len(imgs)} real JPEGs")
names={os.path.basename(f['imagePath']):f for f in fr}
rows=[]
for p in imgs:
    im=Image.open(p).convert('L').resize((256,192))
    a=np.asarray(im,dtype=np.float64)/255.0
    f=names[os.path.basename(p)]
    rows.append((f['index'], a.mean(), np.median(a), f['exposureDurationSeconds'], f['qc']['exposureJumpEV']))
rows.sort()
print(" idx    meanLuma  medLuma   durSec     qc.exposureJumpEV")
for r in rows: print(f"{r[0]:4d}  {r[1]:9.4f} {r[2]:8.4f}  {r[3]:.6f}   {r[4]}")
m=np.array([r[1] for r in rows]); d=np.array([r[3] for r in rows])
print(f"\nmean luma over the 16 frames: min {m.min():.4f} max {m.max():.4f} span {(m.max()/m.min()-1)*100:.1f}%")
print(f"durations over the same 16   : min {d.min():.6f} max {d.max():.6f}  distinct {len(set(d.round(8)))}")
adj=np.abs(np.diff(m))
print(f"adjacent-keyframe |dLuma|: median {np.median(adj):.4f} max {adj.max():.4f}"
      f"  as a multiplicative step: median {np.median(np.abs(np.diff(np.log2(m)))):.4f} stops")
# what a brightness-based exposureJump would report vs the shipped one
print(f"\nA luma-based jump metric would report a median of "
      f"{np.median(np.abs(np.diff(np.log2(m)))):.4f} stops and a max of "
      f"{np.abs(np.diff(np.log2(m))).max():.4f} stops on these 16 frames; "
      f"the shipped exposureJumpEV reports exactly 0.0 on all 868.")
