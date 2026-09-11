"""Does exposureDurationSeconds track image brightness in this capture?
Finding 4 asserts duration variation is 'the real auto-exposure drift' the QC
metric misses. Under free-running AE (settings.exposureLocked=false,
bracketEveryNFrames=0) duration moves to KEEP brightness constant, and the
bundle records no ISO for these frames (CaptureFrame.iso is Optional and nil
here), so duration alone cannot even determine exposure."""
import json,glob,os,numpy as np
from PIL import Image
ROOT="C:/Users/Undea/Documents/LiKOVA/Scans"
b=json.load(open(ROOT+"/diagnostics/scan_20260906_164840/capture_bundle.json"))
byname={os.path.basename(f["imagePath"]):f for f in b["frames"]}
print("frames carrying an 'iso' key in the bundle: %d / %d"%(sum(1 for f in b['frames'] if 'iso' in f),len(b['frames'])))
rows=[]
for p in sorted(glob.glob(ROOT+"/Incoming/scan_20260906_164840/images/*.jpg")):
    f=byname[os.path.basename(p)]
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.0
    rows.append((f["index"],f["exposureDurationSeconds"],float(a.mean()),float(np.median(a))))
print("\n idx   durationSec   meanLuma  medianLuma")
for r in sorted(rows): print("  %3d   %.6f    %.4f    %.4f"%r)
d=np.array([r[1] for r in rows]); m=np.array([r[2] for r in rows])
print("\n duration values present: %s"%sorted(set(d.tolist())))
if len(set(d.tolist()))>1:
    print(" Pearson r(duration, mean brightness) = %.3f over %d frames"%(float(np.corrcoef(d,m)[0,1]),len(d)))
    for v in sorted(set(d.tolist())):
        sel=m[d==v]; print("   duration %.6f : n=%d mean brightness %.4f (sd %.4f)"%(v,len(sel),sel.mean(),sel.std()))
