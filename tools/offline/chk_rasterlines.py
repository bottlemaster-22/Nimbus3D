"""Entry 17 check: does padding TrainerSplatRaster from stride 40 to 64 cut line fetches?

The recovery priced it as if every staged record were fetched in isolation
(1.5 lines at stride 40 on a 64 B line, 1.0 at stride 64). But a 256-lane
staging batch loads 256 records of splats that all overlap ONE tile, and the
export order is spatially correlated, so neighbouring records can share lines.
Count DISTINCT lines per staging batch on the real lists, both strides, both
line sizes. Also counts the two other places the stride changes traffic:
duplicate_keys (reads raster[gid] for every drawn gid, coalesced) and the
preprocess write (raster[gid] = r for every drawn gid).
"""
import io, json, os, sys
import numpy as np
import project as P

D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
RW, RH = sl['renderWidth'], sl['renderHeight']
tx, ty = (RW + 15)//16, (RH + 15)//16
fx, fy, cx, cy = P.load_intrinsics(RW, RH)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys())[:16]
BATCH = 256
REC = 40

acc = {}
def add(k, v):
    acc[k] = acc.get(k, 0) + v

mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
for k in keys:
    rot, t = poses[k]
    R = P.quat_to_matrix(rot)
    ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
    idx = np.nonzero(ok & (tiles > 0))[0]
    cam = mean @ R.T + t
    iz = 1.0/cam[:, 2]
    mx = fx*cam[:, 0]*iz + cx
    my = fy*cam[:, 1]*iz + cy
    a = np.maximum(0, np.floor((mx[idx]-ex[idx])/16)).astype(np.int64)
    b = np.minimum(tx, np.ceil((mx[idx]+ex[idx])/16)).astype(np.int64)
    c = np.maximum(0, np.floor((my[idx]-ey[idx])/16)).astype(np.int64)
    d = np.minimum(ty, np.ceil((my[idx]+ey[idx])/16)).astype(np.int64)
    w = np.maximum(b - a, 0); h = np.maximum(d - c, 0)
    n = w*h
    keep = n > 0
    idx, a, c, w, n = idx[keep], a[keep], c[keep], w[keep], n[keep]
    # expand to instances
    rep = np.repeat(np.arange(len(idx)), n)
    start = np.cumsum(n) - n
    local = np.arange(n.sum()) - np.repeat(start, n)
    tyy = c[rep] + local // w[rep]
    txx = a[rep] + local % w[rep]
    tile = tyy*tx + txx
    sid = idx[rep]
    depth = cam[sid, 2]
    order = np.lexsort((depth, tile))
    tile, sid = tile[order], sid[order]
    # rank within tile -> batch
    first = np.searchsorted(tile, tile, side='left')
    rank = np.arange(len(tile)) - first
    batch = rank // BATCH
    gkey = tile.astype(np.int64)*64 + batch
    add('instances', len(sid))
    for S in (40, 48, 64):
        for L in (64, 128):
            lo = (sid.astype(np.int64)*S)//L
            hi = (sid.astype(np.int64)*S + REC - 1)//L
            iso = int((hi - lo + 1).sum())            # isolated-fetch model
            k1 = np.unique(np.concatenate([gkey*(1 << 24) + lo, gkey*(1 << 24) + hi]))
            add(('gather_iso', S, L), iso)
            add(('gather_dist', S, L), len(k1))
            # duplicate_keys reads the first 16 B of every drawn record (coalesced by gid)
            lo16 = (idx.astype(np.int64)*S)//L
            hi16 = (idx.astype(np.int64)*S + 15)//L
            add(('dupkeys', S, L), len(np.unique(np.concatenate([lo16, hi16]))))
            # preprocess writes the whole 40 B for every drawn gid
            hiw = (idx.astype(np.int64)*S + REC - 1)//L
            add(('write', S, L), len(np.unique(np.concatenate([lo16, hiw]))))

V = len(keys)
print('%d views, %dx%d, %d instances/view mean' % (V, RW, RH, acc['instances']//V))
print()
print('PER ITERATION (one view), per rasteriser pass, lines fetched by the staging gather:')
for L in (64, 128):
    for S in (40, 48, 64):
        iso = acc[('gather_iso', S, L)]/V; dist = acc[('gather_dist', S, L)]/V
        print('  line %3d stride %2d: isolated model %8.0f lines (%5.1f MB) | distinct per batch %8.0f lines (%5.1f MB)'
              % (L, S, iso, iso*L/1e6, dist, dist*L/1e6))
print()
for L in (64, 128):
    g40 = acc[('gather_dist', 40, L)]/V*L; g64 = acc[('gather_dist', 64, L)]/V*L
    d40 = acc[('dupkeys', 40, L)]/V*L; d64 = acc[('dupkeys', 64, L)]/V*L
    w40 = acc[('write', 40, L)]/V*L; w64 = acc[('write', 64, L)]/V*L
    net = 2*(g40 - g64) + (d40 - d64) + (w40 - w64)
    print('line %3d: gather x2 passes %+.2f MB | dup_keys %+.2f MB | preprocess write %+.2f MB | NET saved by stride 64: %+.2f MB/iter'
          % (L, 2*(g40-g64)/1e6, (d40-d64)/1e6, (w40-w64)/1e6, net/1e6))
    print('          -> %.1f GB over 4000 iters' % (net*4000/1e9))
