"""Radix pass count and ping-pong parity, measured on the REAL keys.

Builds the exact (tile << 16) | depth16 keys trainer_duplicate_keys writes,
in the order it writes them (splat index ascending, ty outer, tx inner),
then runs the trainer's LSD 4-bit stable radix sort with the SAME A/B
ping-pong as TrainerGPU.radixSort, and checks:

  1. how many key bits the largest tile id needs at this render size
  2. that pass 7 (bits 28-31) has one digit value only -> identity pass
  3. 8 passes from A  == reference (what ships today, result in A)
  4. 7 passes from A  -> result lands in B; what A holds is NOT sorted
  5. 7 passes from B  -> result lands in A, bit-identical to (3)
  6. the indirect-dispatch argument values per frame, against their bounds

Also prints the per-frame instance count over every posed frame, which
is what the sort's cost scales with.
"""
import io, json, os, sys, numpy as np
import project as P

D = P.D
NEAR, FAR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05, \
            float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
ks = sorted(poses.keys())

tile_count = tx * ty
tile_bits = int(tile_count - 1).bit_length()
key_bits = 16 + tile_bits
passes_needed = -(-key_bits // 4)
print('render %dx%d  tiles %dx%d = %d  max tile id %d needs %d bits'
      % (rw, rh, tx, ty, tile_count, tile_count - 1, tile_bits))
print('key = tile<<16 | depth16 -> highest set bit %d -> %d key bits -> %d passes (odd=%s)'
      % (key_bits - 1, key_bits, passes_needed, passes_needed % 2 == 1))


def keys_for(idx):
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv = 1.0 / z
    mx = fx * cam[:, 0] * inv + cx
    my = fy * cam[:, 1] * inv + cy
    a = np.maximum(0, np.floor((mx - ex) / 16)).astype(np.int64)
    b = np.maximum(0, np.floor((my - ey) / 16)).astype(np.int64)
    c = np.minimum(tx, np.ceil((mx + ex) / 16)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my + ey) / 16)).astype(np.int64)
    s = ok & (tiles > 0)
    gid = np.nonzero(s)[0]
    a, b, c, d = a[s], b[s], c[s], d[s]
    w = np.maximum(c - a, 0)
    h = np.maximum(d - b, 0)
    n_per = w * h
    total = int(n_per.sum())
    rep = np.repeat(np.arange(len(gid)), n_per)
    start = np.repeat(np.cumsum(n_per) - n_per, n_per)
    local = np.arange(total) - start
    wr = w[rep]
    tile = (b[rep] + local // wr) * tx + (a[rep] + local % wr)
    norm = np.clip((z[gid] - NEAR) / max(FAR - NEAR, 1e-3), 0.0, 1.0)
    depth = (norm * 65535.0).astype(np.uint32)      # uint(norm * 65535.0f)
    keys = (tile.astype(np.uint32) << np.uint32(16)) | depth[rep]
    vals = gid[rep].astype(np.uint32)
    return keys, vals, int(tiles[s].sum())


def radix(keys, vals, passes, start_in_b):
    """TrainerGPU.radixSort: keysIn/keysOut swap every pass."""
    buf = [[None, None], [None, None]]           # buf[0]=A, buf[1]=B ; [keys, vals]
    src = 1 if start_in_b else 0
    buf[src] = [keys.copy(), vals.copy()]
    buf[1 - src] = [np.zeros_like(keys), np.zeros_like(vals)]
    for p in range(passes):
        k, v = buf[src]
        digit = (k >> np.uint32(4 * p)) & np.uint32(15)
        perm = np.argsort(digit, kind='stable')  # stable counting sort == scatter
        dst = 1 - src
        buf[dst] = [k[perm], v[perm]]
        src = dst
    return buf, src


def tile_ranges(keys, n_tiles):
    """trainer_tile_ranges on whatever buffer the consumer binds."""
    r = np.zeros((n_tiles, 2), dtype=np.int64)
    t = (keys >> np.uint32(16)).astype(np.int64)
    ok = t < n_tiles
    if len(t) == 0:
        return r
    brk = np.nonzero(np.diff(t) != 0)[0] + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(t)]])
    for s0, e0 in zip(starts, ends):
        if ok[s0]:
            r[t[s0]] = (s0, e0)
    return r


INST_CAP = max(300000 * 8, 1024)                 # instanceCapacity = capacity * 8
B_MAX = -(-INST_CAP // 1024)
print('instanceCapacity %d -> maxSortBlocks %d -> max histCount %d -> max histScanBlocks %d'
      % (INST_CAP, B_MAX, 16 * B_MAX, -(-(16 * B_MAX) // 1024)))

for idx in (ks[4], ks[25], ks[len(ks) // 2]):
    keys, vals, n_proj = keys_for(idx)
    n = len(keys)
    assert abs(n - n_proj) < 2, (n, n_proj)
    top = int(keys.max()) >> 28 if n else 0
    d7 = np.unique((keys >> np.uint32(28)) & np.uint32(15))
    d6 = np.unique((keys >> np.uint32(24)) & np.uint32(15))
    ref, where = radix(keys, vals, 8, False)
    assert where == 0
    rk, rv = ref[0]
    assert np.all(np.diff(rk.astype(np.int64)) >= 0), 'reference not sorted'
    naive, wn = radix(keys, vals, 7, False)
    ak = naive[0][0]                           # what a consumer bound to A reads
    bad_order = int(np.sum(np.diff((ak >> np.uint32(16)).astype(np.int64)) < 0))
    rr = tile_ranges(rk, tile_count)
    ra = tile_ranges(ak, tile_count)
    tiles_wrong = int(np.sum(np.any(rr != ra, axis=1)))
    fixed, wf = radix(keys, vals, 7, True)
    fk, fv = fixed[0]
    same = (wf == 0) and np.array_equal(fk, rk) and np.array_equal(fv, rv)
    blocks = max(-(-n // 1024), 1)
    hist = 16 * blocks
    print('\nframe %d: %d instances  pass-7 digits %s  pass-6 digits %d distinct'
          % (idx, n, d7.tolist(), len(d6)))
    print('  8 passes from A: result in %s, sorted' % 'AB'[where])
    print('  7 passes from A: result in %s; buffer A holds a 6-pass state with %d tile-order '
          'inversions, %d of %d tile ranges wrong'
          % ('AB'[wn], bad_order, tiles_wrong, tile_count))
    print('  7 passes from B: result in %s, bit-identical keys AND values to 8-pass: %s'
          % ('AB'[wf], same))
    print('  indirect args: sortBlocks %d  histCount %d  histScanBlocks %d  rangeGroups %d'
          % (blocks, hist, -(-hist // 1024), max(-(-n // 256), 1)))

# ---- instance count over every posed frame ----------------------------------
counts = []
for idx in ks:
    rot, t = poses[idx]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    counts.append(int(tiles[ok & (tiles > 0)].sum()))
c = np.array(counts, float)
print('\ninstances over %d posed frames: mean %.0f  median %.0f  p90 %.0f  max %.0f'
      % (len(c), c.mean(), np.median(c), np.percentile(c, 90), c.max()))
print('census peakTileInstances %d' % sl['peakTileInstances'])
print('sort blocks per pass at mean: %.0f  at max: %.0f' % (np.ceil(c.mean() / 1024), np.ceil(c.max() / 1024)))
