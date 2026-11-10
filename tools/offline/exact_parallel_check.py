"""Proves, on adversarial synthetic data, that three proposed parallel
restructures reproduce the serial code EXACTLY (same order, same bits).

1. SEEDING merge (PrePassInitialSplatBuilder.build): serial rule is
     slot = hash.indexOrInsert(key)
     if !inserted and w <= weight[slot]: skip
     else overwrite attrs; onEdge = inserted ? isEdge : (onEdge || isEdge)
   Proposed: each keyframe reduced on its own core to per-cell summaries
   (first-appearance order, frame max + its FIRST sample, the strictly
   increasing record chain with suffix-OR of edge flags), then merged serially
   per CELL in keyframe order.
2. TRUST bias accumulator: Welford float32 updates per cell. Proposed: shard
   cells by key; each shard walks ALL entries in slot order, adding only its
   own cells; union of shards sorted by key.
3. TRUST plane-sweep budget: first-come in slot/sample order. Proposed: eligible
   samples of a slot listed in order, sweeps computed speculatively in
   parallel, successes counted in order until the remaining budget is spent.
Run: python exact_parallel_check.py   (prints PASS/FAIL per design)
"""
import numpy as np

rng = np.random.default_rng(7)
f32 = np.float32

# ---------------------------------------------------------------- 1. seeding
def serial_seed(frames):
    index, key_of, w, attr, edge = {}, [], [], [], []
    for f, samples in enumerate(frames):
        for s, (k, wt, e) in enumerate(samples):
            if k not in index:
                index[k] = len(key_of)
                key_of.append(k); w.append(wt); attr.append((f, s)); edge.append(e)
            else:
                i = index[k]
                if wt <= w[i]:
                    continue
                w[i] = wt; attr[i] = (f, s); edge[i] = edge[i] or e
    return key_of, w, attr, edge


def reduce_frame(f, samples):
    # runs on its own core in the proposal; touches nothing shared
    local, cells = {}, []
    for s, (k, wt, e) in enumerate(samples):
        if k not in local:
            local[k] = len(cells)
            cells.append({"key": k, "max": wt, "who": (f, s), "chain": [(wt, e)]})
        else:
            c = cells[local[k]]
            if wt > c["max"]:          # same strict rule, frame-local
                c["max"] = wt; c["who"] = (f, s); c["chain"].append((wt, e))
    for c in cells:                     # suffix OR of edge flags along the chain
        acc, suf = False, []
        for wt, e in reversed(c["chain"]):
            acc = acc or e
            suf.append(acc)
        c["suffix"] = list(reversed(suf))
    return cells


def merged_seed(frames):
    summaries = [reduce_frame(f, s) for f, s in enumerate(frames)]  # parallel
    index, key_of, w, attr, edge = {}, [], [], [], []
    for cells in summaries:             # serial, keyframe order, per CELL
        for c in cells:
            k = c["key"]
            if k not in index:
                index[k] = len(key_of)
                key_of.append(k); w.append(c["max"]); attr.append(c["who"])
                edge.append(c["suffix"][0])
                continue
            i = index[k]
            if c["max"] <= w[i]:
                continue
            j = next(j for j, (wt, _) in enumerate(c["chain"]) if wt > w[i])
            edge[i] = edge[i] or c["suffix"][j]
            w[i] = c["max"]; attr[i] = c["who"]
    return key_of, w, attr, edge


ok = True
for trial in range(40):
    nf = int(rng.integers(3, 12))
    frames = []
    for _ in range(nf):
        n = int(rng.integers(50, 600))
        keys = rng.integers(0, 120, n)
        # few distinct weight levels -> many exact ties (first-wins must hold)
        wts = rng.choice(np.array([0, 0.1, 0.2, 0.2, 0.3, 0.35], dtype=f32), n)
        eds = rng.random(n) < 0.15
        frames.append(list(zip(keys.tolist(), wts.tolist(), eds.tolist())))
    if serial_seed(frames) != merged_seed(frames):
        ok = False
        break
print("1. seeding per-frame reduce + per-cell merge:", "PASS" if ok else "FAIL")

# -------------------------------------------------------- 2. bias accumulator
def welford(entries):
    cells = {}
    for k, r in entries:
        c = cells.setdefault(k, [0, f32(0), f32(0)])
        c[0] += 1
        d = f32(r) - c[1]
        c[1] = f32(c[1] + f32(d / f32(c[0])))
        c[2] = f32(c[2] + f32(d * f32(f32(r) - c[1])))
    return sorted((k, v[0], v[1].tobytes(), v[2].tobytes()) for k, v in cells.items())


entries = list(zip(rng.integers(0, 400, 60000).tolist(),
                   (rng.standard_normal(60000) * 0.03).astype(f32).tolist()))
serial = welford(entries)
S = 6
shards = []
for s in range(S):                       # each on its own core
    shards += welford([(k, r) for k, r in entries if (k * 0x9E3779B1) % S == s])
print("2. bias accumulator sharded by cell:",
      "PASS" if sorted(shards) == serial else "FAIL")

# ---------------------------------------------------- 3. sweep budget split
def serial_budget(slots, budget):
    used = []
    for slot in slots:
        mine = []
        for sample, success in slot:
            if budget > 0 and success:   # sweep attempted only while budget > 0
                budget -= 1
                mine.append(sample)
        used.append(mine)
    return used, budget


def two_pass_budget(slots, budget, batch=64):
    used = []
    for slot in slots:
        eligible = [(s, ok) for s, ok in slot]   # pass 1 (parallel rows): in order
        mine, i = [], 0
        while budget > 0 and i < len(eligible):
            chunk = eligible[i:i + max(batch, budget)]   # speculative, parallel
            for s, success in chunk:                     # commit in order
                if budget == 0:
                    break
                if success:
                    budget -= 1
                    mine.append(s)
            i += len(chunk)
        used.append(mine)
    return used, budget


ok = True
for trial in range(200):
    slots = [[(j, bool(rng.random() < 0.8)) for j in range(int(rng.integers(0, 900)))]
             for _ in range(int(rng.integers(1, 8)))]
    b = int(rng.integers(0, 3000))
    if serial_budget(slots, b) != two_pass_budget(slots, b):
        ok = False
        break
print("3. plane-sweep budget, speculative batches committed in order:",
      "PASS" if ok else "FAIL")
