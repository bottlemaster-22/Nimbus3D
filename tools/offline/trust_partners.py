"""Replica of TwoScaleTrustField.coVisibilityTable on a real scan.

Answers: which frames verify which, how many slots have >= 2 partners (a slot
with fewer can never run a plane sweep, because a sample needs
residualCount >= 2 before the sweep is tried), and therefore the least number
of slots the serial prefix of the trust build can have.

Pose table: prePassPoses[str(index)] ?? frame.refinedPose ?? frame.rawPose.
Pose is world-to-camera: center = -R^-1 t, forward = R^-1 (0,0,1).
Quaternion JSON order is [x, y, z, w].

usage: python trust_partners.py <diagnostics dir>
"""
import json
import sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else \
    "C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics/scan_20260906_164840"

bundle = json.load(open(f"{D}/capture_bundle.json"))
result = json.load(open(f"{D}/prepass/prepass_result.json"))
refined = result.get("refinedPoses", {})


def rot_inv_apply(q, v):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    # inverse of unit quaternion = conjugate
    qv = np.array([-x, -y, -z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


frames = sorted(bundle["frames"], key=lambda f: f["index"])
centres, forwards, idx = [], [], []
for f in frames:
    p = refined.get(str(f["index"])) or f.get("refinedPose") or f["rawPose"]
    q = p["rotation"]
    t = np.array(p["translation"], dtype=np.float64)
    centres.append(-rot_inv_apply(q, t))
    forwards.append(rot_inv_apply(q, np.array([0.0, 0.0, 1.0])))
    idx.append(f["index"])
C = np.array(centres)
F = np.array(forwards)
N = len(idx)

MIN_B, MAX_B, MIN_DOT, MAXP = 0.05, 2.5, 0.5, 4
ideal = MIN_B + 0.4 * (MAX_B - MIN_B)
partners = {}
for a in range(N):
    base = np.linalg.norm(C - C[a], axis=1)
    dots = F @ F[a]
    ok = (base >= MIN_B) & (base <= MAX_B) & (dots >= MIN_DOT)
    ok[a] = False
    cand = np.nonzero(ok)[0]
    cost = np.abs(base[cand] - ideal).astype(np.float32)
    order = np.argsort(cost, kind="stable")  # Swift sort is not stable; ties are rare
    partners[idx[a]] = [idx[c] for c in cand[order][:MAXP]]

counts = np.array([len(partners[i]) for i in idx])
print(f"frames {N}; partner count histogram:",
      {k: int((counts == k).sum()) for k in range(MAXP + 1)})
print("first 12 slots:")
for i in idx[:12]:
    ps = partners[i]
    span = max((abs(p - i) for p in ps), default=0)
    print(f"  slot {i:3d}: partners {ps}  max |index gap| {span}")
first_eligible = [i for i in idx if len(partners[i]) >= 2][:5]
print("first slots with >= 2 partners:", first_eligible)
STRIDED = (256 // 2) * (192 // 2)
print(f"strided samples per slot at stride 2: {STRIDED}; budget 20000 needs "
      f">= {int(np.ceil(20000 / STRIDED))} sweeping slots")
gaps = [max((abs(p - i) for p in partners[i]), default=0) for i in idx]
print("median / p90 max-partner index gap:",
      int(np.median(gaps)), int(np.percentile(gaps, 90)))
