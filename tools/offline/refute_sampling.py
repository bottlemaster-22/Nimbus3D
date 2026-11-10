"""What finding 3 did NOT check: the held-out set's QC, and the fact that the
trained-view COMPARISON set is drawn from keyframe positions 0..99 while the
held-out set is positions 5..115 (MetalSplatTrainer.swift 1651-1657 vs
TrainerSlices.swift splitHeldOut, step=10, i%10==5)."""
import json, numpy as np
ROOT="C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840"
b=json.load(open(ROOT+"/capture_bundle.json")); fr=b["frames"]
ho=json.load(open(ROOT+"/model/held_out_frames.json"))
idx=np.array([f["index"] for f in fr])
sh=np.array([f["qc"]["sharpness"] for f in fr]); w=np.array([f["qc"]["weight"] for f in fr])
blur=np.array([f["qc"]["motionBlurPixels"] for f in fr])
print("all 868 frames: sharpness mean %.3f  weight mean %.3f"%(sh.mean(),w.mean()))
# trend along the walk
for a,bnd in [(0,120),(120,240),(240,360),(360,480),(480,600),(600,720),(720,868)]:
    s=(idx>=a)&(idx<bnd)
    print("  frames %3d-%3d : sharpness %.3f  weight %.3f  motionBlurPx %.2f"%(a,bnd,sh[s].mean(),w[s].mean(),blur[s].mean()))
hos=np.array([sh[i] for i in ho]); how=np.array([w[i] for i in ho])
print("\nHELD-OUT 12 (indices %s):"%ho)
print("  sharpness mean %.3f (min %.3f)  weight mean %.3f (min %.3f)"%(hos.mean(),hos.min(),how.mean(),how.min()))
# the population the held-out set was actually drawn from: the keyframe span
span=(idx>=ho[0])&(idx<=ho[-1]+20)
print("  same-span population (%d..%d, n=%d): sharpness %.3f  weight %.3f"%(ho[0],ho[-1]+20,span.sum(),sh[span].mean(),w[span].mean()))
print("  frames the keyframe list never reached (%d..867, n=%d): sharpness %.3f weight %.3f"
      %(ho[-1]+20,(idx>ho[-1]+20).sum(),sh[idx>ho[-1]+20].mean(),w[idx>ho[-1]+20].mean()))
# Where the trained COMPARISON set sits: keyframe positions 0,9,...,99 vs held-out 5,15,...,115.
# Held-out position p maps to a frame index; interpolate the mapping from the 12 known pairs.
pos=np.arange(5,120,10); f=np.polyfit(pos,ho,1)
print("\nkeyframe position -> frame index fit: idx = %.2f*pos + %.2f  (r2 %.4f)"
      %(f[0],f[1],np.corrcoef(pos,ho)[0,1]**2))
tp=np.arange(0,108,9)[:12]
tidx=np.clip(np.round(np.polyval(f,tp)).astype(int),0,867)
print("trained-comparison set is keyframe positions %s -> frame indices ~%s"%(tp.tolist(),tidx.tolist()))
print("  their sharpness mean %.3f  weight mean %.3f"%(sh[tidx].mean(),w[tidx].mean()))
print("  held-out    sharpness mean %.3f  weight mean %.3f"%(hos.mean(),how.mean()))
print("  mean frame index: trained-comparison %.0f, held-out %.0f  (held-out sits %.0f frames later in the walk)"
      %(tidx.mean(),np.mean(ho),np.mean(ho)-tidx.mean()))
