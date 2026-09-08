import json, statistics, collections, random
ROOT="C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840"
b=json.load(open(ROOT+"/capture_bundle.json"))
ho=sorted(json.load(open(ROOT+"/model/held_out_frames.json")))
fr=b["frames"]
dur={f["index"]:f["exposureDurationSeconds"] for f in fr}
idx=sorted(dur)
print("frame index range", idx[0], idx[-1], "n", len(idx))
print("held-out span", ho[0], ho[-1], "gaps", [ho[i+1]-ho[i] for i in range(len(ho)-1)])
# Frames beyond the last held-out frame:
print("frames with index > 460:", sum(1 for i in idx if i>460))
# duration distribution inside vs outside the held-out span
inside=[dur[i] for i in idx if 0<=i<=470]
outside=[dur[i] for i in idx if i>470]
for name,pop in (("frames 0-470",inside),("frames 471-867",outside),("all",list(dur.values()))):
    fl=sum(1 for d in pop if abs(d-0.01)<1e-9)
    print("%-14s n=%3d mean %.6f pstd %.6f  frac at floor %.3f"%(name,len(pop),statistics.mean(pop),statistics.pstdev(pop),fl/len(pop)))
# permutation test: is the held-out set's mean duration unusual vs same-size draws
# from the frames in the keyframe span?
hod=[dur[i] for i in ho]
obs=statistics.mean(hod)
random.seed(0)
pop=inside
cnt=0; N=200000
for _ in range(N):
    s=random.sample(pop,12)
    if statistics.mean(s)<=obs: cnt+=1
print("held-out mean %.6f ; P(random 12 from 0-470 has mean <= obs) = %.4f"%(obs,cnt/N))
# Same against ALL 868 (what finding 3 implicitly compared to)
cnt=0
for _ in range(N):
    s=random.sample(list(dur.values()),12)
    if statistics.mean(s)<=obs: cnt+=1
print("P(random 12 from all 868 has mean <= obs) = %.4f"%(cnt/N))
