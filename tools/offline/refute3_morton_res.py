"""Two measurements the 'rejected' pile deferred without pricing.

A. LiteGS Morton ordering (entry 28 item 1 called it "the only form of
   coherence that works here" and pushed it to another family, where nobody
   wrote it up).  The rasterisers gather raster[splatIndex], a 40-byte
   TrainerSplatRaster (TrainerShaders.metal:275-294), for every instance in a
   tile's depth-sorted list.  Metric: distinct 64-byte cache lines touched per
   cooperative batch of 256 instances, current array order vs Morton order of
   the 3D centre.

B. DashGaussian / progressive resolution (entry 26 called it "a legitimate
   separate proposal for a different family").  Metric: tile instances and
   (pixel, splat) pairs at 720x540 vs 360x270 on the SAME model, to test the
   claim that the 16x16 tile grid caps the saving well below the 4x the pixel
   count implies.
"""
import io, json, os, sys
import numpy as np
import project as P

D = P.D
TILE = 16
LINE = 64
STRIDE = 40          # TrainerSplatRaster
BATCH = 256          # TRAINER_TILE_AREA cooperative stage

census = json.load(io.open(os.path.join(D,'model','train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
col, count = P.load_ply(os.path.join(D,'model','model.ply'))
poses = P.load_poses()
keys = sorted(poses.keys(), key=lambda s: int(s))
sel = [keys[i] for i in np.linspace(0, len(keys)-1, 8).astype(int)]

# ---- Morton order of the centres -----------------------------------------
mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
lo, hi = mean.min(0), mean.max(0)
g = np.clip(((mean - lo) / np.maximum(hi - lo, 1e-9) * (2**21 - 1)).astype(np.uint64), 0, 2**21-1)
def spread(v):
    v = v.astype(np.uint64) & np.uint64(0x1fffff)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v
code = spread(g[:,0]) | (spread(g[:,1]) << np.uint64(1)) | (spread(g[:,2]) << np.uint64(2))
morton_rank = np.empty(count, np.int64)
morton_rank[np.argsort(code, kind='stable')] = np.arange(count)

def lines_per_batch(idx_in_order, depth):
    """idx sorted by depth; distinct 64B lines per 256-instance batch."""
    o = np.argsort(depth, kind='stable')
    a = idx_in_order[o]
    tot_lines, tot_inst = 0, 0
    for s in range(0, len(a), BATCH):
        b = a[s:s+BATCH]
        lo_l = (b.astype(np.int64)*STRIDE)//LINE
        hi_l = (b.astype(np.int64)*STRIDE + STRIDE - 1)//LINE
        # a 40B record straddles at most 2 lines
        tot_lines += len(np.unique(np.concatenate([lo_l, hi_l])))
        tot_inst  += len(b)
    return tot_lines, tot_inst

def run(scale, do_lines):
    rw = int(round(sl['renderWidth']*scale)); rh = int(round(sl['renderHeight']*scale))
    tx = (rw+TILE-1)//TILE; ty = (rh+TILE-1)//TILE
    fx, fy, cx, cy = P.load_intrinsics(rw, rh)
    tot_inst = tot_pairs = 0
    L_id = I_id = L_mo = I_mo = 0
    for k in sel:
        rot, t = poses[k]
        R = P.quat_to_matrix(rot)
        ok, radius, tiles, ex, ey = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)
        tot_inst += int(tiles.sum())
        tot_pairs += int(tiles.sum()) * TILE * TILE
        if not do_lines:
            continue
        # rebuild the per-tile lists for the locality metric
        camz = (np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64) @ R.T + t)[:,2]
        v = np.nonzero(ok)[0]
        # per-splat tile box, recomputed the same way project() does
        # (cheap: reuse tiles>0 and recompute the box)
        mean_c = np.stack([col['x'],-col['y'],-col['z']],1).astype(np.float64) @ R.T + t
        iz = 1.0/mean_c[:,2]
        mx = fx*mean_c[:,0]*iz + cx; my = fy*mean_c[:,1]*iz + cy
        minx = np.maximum(0, np.floor((mx-ex)/TILE)).astype(np.int64)
        miny = np.maximum(0, np.floor((my-ey)/TILE)).astype(np.int64)
        maxx = np.minimum(tx, np.ceil((mx+ex)/TILE)).astype(np.int64)
        maxy = np.minimum(ty, np.ceil((my+ey)/TILE)).astype(np.int64)
        buckets = {}
        for i in v:
            for tyy in range(miny[i], maxy[i]):
                row = buckets.setdefault(tyy, {})
                for txx in range(minx[i], maxx[i]):
                    row.setdefault(txx, []).append(i)
        for row in buckets.values():
            for lst in row.values():
                a = np.array(lst, np.int64)
                d = camz[a]
                l1, n1 = lines_per_batch(a, d)
                l2, n2 = lines_per_batch(morton_rank[a], d)
                L_id += l1; I_id += n1; L_mo += l2; I_mo += n2
    return rw, rh, tx, ty, tot_inst, tot_pairs, L_id, I_id, L_mo, I_mo

print('model %d splats, %d views' % (count, len(sel)))
for scale, dl in ((1.0, True), (0.5, False)):
    rw,rh,tx,ty,ti,tp,Li,Ii,Lm,Im = run(scale, dl)
    print('\n--- render %dx%d  tiles %dx%d = %d ---' % (rw,rh,tx,ty,tx*ty))
    print('  tile instances / view      %12.0f' % (ti/len(sel)))
    print('  (pixel,splat) pairs / view %12.0f' % (tp/len(sel)))
    if dl:
        print('  B. LOCALITY of raster[splatIndex], 256-instance batches:')
        print('    current order  %.4f cache lines per instance (%d lines / %d)' % (Li/Ii, Li, Ii))
        print('    Morton order   %.4f cache lines per instance (%d lines / %d)' % (Lm/Im, Lm, Im))
        print('    ideal (40B/64B, perfectly packed) 0.6250')
        print('    reduction      %.2f%%' % (100*(1-Lm/Li)))
