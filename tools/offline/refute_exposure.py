import json, statistics, collections
ROOT="C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840"
b=json.load(open(ROOT+"/capture_bundle.json"))
ho=set(json.load(open(ROOT+"/model/held_out_frames.json")))
fr=b["frames"]
print("frames",len(fr),"heldout",sorted(ho))
br=collections.Counter(f["bracket"] for f in fr)
off=collections.Counter(f["exposureOffsetEV"] for f in fr)
jump=collections.Counter(f["qc"]["exposureJumpEV"] for f in fr)
print("bracket",br); print("offsetEV",off); print("qc.exposureJumpEV",jump)
dur=[f["exposureDurationSeconds"] for f in fr]
print("dur mean %.6f std %.6f min %.6f max %.6f"%(statistics.mean(dur),statistics.pstdev(dur),min(dur),max(dur)))
print("dur sample stdev %.6f"%statistics.stdev(dur))
dc=collections.Counter(dur)
for k,v in sorted(dc.items()): print("  dur %.6f  n=%d  %.1f%%"%(k,v,100*v/len(fr)))
hod=[f["exposureDurationSeconds"] for f in fr if f["index"] in ho]
print("heldout n",len(hod))
print("heldout dur mean %.6f pstd %.6f sstd %.6f"%(statistics.mean(hod),statistics.pstdev(hod),statistics.stdev(hod)))
print("heldout dur counter",collections.Counter(hod))
# how many frames were even KEYFRAMES? held-out ids come from the 120 keyframes.
# Reconstruct plausible keyframe set: held-out are positions i%10==5 of the keyframe list.
