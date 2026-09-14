"""Replays the trust build's DEPTH-CACHE traffic on a real scan, exactly as
TwoScaleTrustField.build (build 266) issues it, and counts disk loads.

Every cache miss is one depth16 read + half->float decode (and, for the slot's
own frame, one conf8 read). No timing: this counts WORK, which is what moves.

Model of the code:
  * poses, partners: trust_partners.py replica (coVisibilityTable, 4 partners).
  * SmartDepthCache: FIFO eviction (order.append on insert, removeFirst when
    over capacity; a hit does NOT refresh), capacity max(4, partners + 2) = 6.
  * computeSlot(slot): depth(own), confidence(own), then depth(p) per partner.
  * phase 1: slots 0..S-1 serial through `depthCache` (S = serial prefix).
  * phase 2: chunks of workerCount*8 slots; worker w takes slots
    base + 8w ... base + 8w + 7 through its OWN persistent cache.
  * rewriteConfidence: slot 0..867 in order through the phase-1 cache, depth
    and confidence both.
usage: python trust_cache_replay.py [serialSlots] [workerCount]
"""
import sys
import json
import numpy as np

S = int(sys.argv[1]) if len(sys.argv) > 1 else 2
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
D = "C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840"

bundle = json.load(open(f"{D}/capture_bundle.json"))
refined = json.load(open(f"{D}/prepass/prepass_result.json"))["refinedPoses"]


def rinv(q, v):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    qv = -np.array([x, y, z]) / n
    w = w / n
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


frames = sorted(bundle["frames"], key=lambda f: f["index"])
C, F = [], []
for f in frames:
    p = refined.get(str(f["index"])) or f.get("refinedPose") or f["rawPose"]
    C.append(-rinv(p["rotation"], np.array(p["translation"], float)))
    F.append(rinv(p["rotation"], np.array([0.0, 0.0, 1.0])))
C, F = np.array(C), np.array(F)
N = len(frames)
ideal = 0.05 + 0.4 * (2.5 - 0.05)
partners = []
for a in range(N):
    b = np.linalg.norm(C - C[a], axis=1)
    ok = (b >= 0.05) & (b <= 2.5) & (F @ F[a] >= 0.5)
    ok[a] = False
    cand = np.nonzero(ok)[0]
    o = np.argsort(np.abs(b[cand] - ideal).astype(np.float32), kind="stable")
    partners.append(list(cand[o][:4]))


class Fifo:
    def __init__(self, cap):
        self.cap, self.order, self.have, self.loads = cap, [], set(), 0

    def get(self, k):
        if k in self.have:
            return
        self.loads += 1
        self.have.add(k)
        self.order.append(k)
        while len(self.order) > self.cap:
            self.have.discard(self.order.pop(0))


CAP = 6
serial_d, serial_c = Fifo(CAP), Fifo(CAP)
work_d = [Fifo(CAP) for _ in range(W)]
work_c = [Fifo(CAP) for _ in range(W)]


def slot(s, d, c):
    d.get(s)
    c.get(s)
    for p in partners[s]:
        d.get(p)


for s in range(S):
    slot(s, serial_d, serial_c)
s = S
while s < N:
    chunk = min(W * 8, N - s)
    for w in range(W):
        for k in range(w * 8, min(w * 8 + 8, chunk)):
            slot(s + k, work_d[w], work_c[w])
    s += chunk
build_depth = serial_d.loads + sum(x.loads for x in work_d)
build_conf = serial_c.loads + sum(x.loads for x in work_c)
rw_d0, rw_c0 = serial_d.loads, serial_c.loads
for s in range(N):
    serial_d.get(s)
    serial_c.get(s)
rw_depth = serial_d.loads - rw_d0
rw_conf = serial_c.loads - rw_c0

uses = N * 5
print(f"slots {N}, serial prefix {S}, workers {W}")
print(f"computeSlot depth requests {uses}, depth16 loads {build_depth} "
      f"({100 * (1 - build_depth / uses):.1f} % hit), conf8 loads {build_conf}")
print(f"rewriteConfidence: depth16 loads {rw_depth}, conf8 loads {rw_conf}")
print(f"TOTAL depth16 loads {build_depth + rw_depth} against a floor of {N}")

# A worker that owned a WHOLE-SCAN cache would load each depth once. How much
# would a bigger per-worker FIFO buy, same slot assignment?
for cap in (6, 16, 32, 64, 128):
    ws = [Fifo(cap) for _ in range(W)]
    s = S
    while s < N:
        chunk = min(W * 8, N - s)
        for w in range(W):
            for k in range(w * 8, min(w * 8 + 8, chunk)):
                ws[w].get(s + k)
                for p in partners[s + k]:
                    ws[w].get(p)
        s += chunk
    print(f"  per-worker FIFO capacity {cap:3d}: phase-2 depth loads "
          f"{sum(x.loads for x in ws)}  ({cap * 196608 * W / 1e6:.0f} MB resident)")
