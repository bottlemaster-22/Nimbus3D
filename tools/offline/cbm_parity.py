"""What an ODD radix pass count actually does, measured on real keys.

Builds the real 32-bit keys for a real view, runs an LSD 4-bit radix sort for
P passes, and then runs trainer_tile_ranges over whatever landed in keysA
(the buffer tileRanges and rasterizeForward are hardcoded to read).
Reports how many instances the rasteriser would actually composite.
"""
import io, json, os, numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw+15)//16, (rh+15)//16
NT = tx*ty
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()

def build_keys(idx):
    rot, t = poses[idx]; R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:,2]; inv = 1.0/z
    mx = fx*cam[:,0]*inv + cx; my = fy*cam[:,1]*inv + cy
    a = np.maximum(0, np.floor((mx-ex)/16)).astype(np.int64)
    b = np.maximum(0, np.floor((my-ey)/16)).astype(np.int64)
    c = np.minimum(tx, np.ceil((mx+ex)/16)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my+ey)/16)).astype(np.int64)
    s = ok & (tiles>0)
    near, far = 0.05, 100.0
    span = max(far-near, 1e-3)
    norm = np.clip((z-near)/span, 0, 1)
    dk = (norm*65535.0).astype(np.uint32)
    keys=[]; vals=[]
    ai,bi,ci,di,dki = a[s],b[s],c[s],d[s],dk[s]
    gid = np.nonzero(s)[0]
    for i in range(len(ai)):
        for yy in range(bi[i], di[i]):
            row = np.arange(ai[i], ci[i], dtype=np.uint32) + np.uint32(yy*tx)
            keys.append((row << np.uint32(16)) | np.uint32(dki[i]))
            vals.append(np.full(len(row), gid[i], dtype=np.uint32))
    return np.concatenate(keys), np.concatenate(vals)

def radix(keys, vals, passes):
    """LSD 4-bit stable sort, returning (keys, vals, parity). parity==0 means
    the result is back in buffer A."""
    k = keys.copy(); v = vals.copy()
    for p in range(passes):
        dig = (k >> np.uint32(4*p)) & np.uint32(0xF)
        order = np.argsort(dig, kind='stable')
        k = k[order]; v = v[order]
    return k, v, passes % 2

def tile_ranges_and_render(k):
    """trainer_tile_ranges transcribed, then count instances the rasteriser
    would visit: sum over tiles of (end-start) for ranges the kernel wrote."""
    n = len(k)
    tile = (k >> np.uint32(16)).astype(np.int64)
    rng = np.zeros((NT,2), dtype=np.int64)
    # gid == 0
    rng[tile[0],0] = 0
    ch = np.nonzero(tile[1:] != tile[:-1])[0] + 1
    prev = tile[ch-1]; cur = tile[ch]
    # later writes win, exactly like the GPU's unordered stores would not, but
    # sequential semantics is the charitable reading
    for g, pv, cv in zip(ch, prev, cur):
        rng[pv,1] = g
        rng[cv,0] = g
    rng[tile[-1],1] = n
    visited = np.maximum(rng[:,1]-rng[:,0], 0).sum()
    # correctness: does every visited instance actually belong to its tile?
    correct = 0
    for tid in range(NT):
        s0,s1 = rng[tid]
        if s1>s0:
            correct += int((tile[s0:s1]==tid).sum())
    return visited, correct

for idx in (4, 25):
    keys, vals = build_keys(idx)
    n = len(keys)
    print('\nframe %d: %d instances, %d tiles' % (idx, n, NT))
    for passes in (8, 7, 6):
        k, v, par = radix(keys, vals, passes)
        landed = 'keysA (correct buffer)' if par==0 else 'keysB (NOT read by anyone)'
        # what the consumer actually sees:
        seen = k if par==0 else radix(keys, vals, passes-1)[0]
        vis, cor = tile_ranges_and_render(seen)
        print('  %d passes -> parity %d, result in %s' % (passes, par, landed))
        print('     rasteriser composites %d of %d instances (%.1f%%), of which %d (%.1f%%) are in the right tile'
              % (vis, n, 100*vis/n, cor, 100*cor/max(vis,1)))
