"""Reads the latest diagnostics census and prints where the run's time went.

Kept in the scratchpad so reading a new set of diagnostics is one command
rather than a fresh ad-hoc script every time.
"""
import json
import datetime as dt
import glob
import os

BASE = 'C:/Users/Undea/Documents/LiKOVA/Scans/diagnostics'

scans = sorted(glob.glob(BASE + '/*/model/train_census.json'),
               key=os.path.getmtime)
if not scans:
    raise SystemExit('no census found under ' + BASE)
path = scans[-1]
d = json.load(open(path, encoding='utf-8'))
m = json.load(open(path.replace('train_census.json', 'model.json'),
                   encoding='utf-8'))

started = dt.datetime.fromisoformat(d['startedAt'].replace('Z', '+00:00'))
ended = dt.datetime.fromisoformat(d['finishedAt'].replace('Z', '+00:00'))
wall = (ended - started).total_seconds()
iters = d['iterationsCompleted']

print('file       :', os.path.getmtime(path))
print('build      :', d.get('appVersion', 'ABSENT'))
print('iterations : %d/%d  %s  degraded %d'
      % (iters, d['iterationsRequested'], d['outcome'],
         len(d['budgetReductions'])))
print('TRAIN wall : %.0f s (%dm%02ds)   %.4f s/iter'
      % (wall, wall // 60, wall % 60, wall / iters))
print('  ^^ TRAINING ONLY. The pre-pass printed below is a SEPARATE stage with'
      ' its own census.')
print('     TOTAL PROCESSING = this + pre-pass. Nothing else prints the sum.')
print('splats     : %d   held-out PSNR %.2f'
      % (m['splatCount'], m['heldOutPSNR']))
sl = d.get('slices') or [{}]
tr = sl[0].get('trainedPSNR')
if tr is not None:
    print('trained-view PSNR %.2f   gap %+.2f dB  -> %s'
          % (tr, tr - m['heldOutPSNR'],
             'does not generalise' if tr - m['heldOutPSNR'] > 3.0
             else 'does not fit at all'))

# The pre-pass, which has its own census beside the trainer's.
pre_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'prepass', 'census.json')
try:
    pre = json.load(open(pre_path, encoding='utf-8'))
except Exception:
    pre = None
if pre:
    total = pre.get('durationSeconds', 0)
    print(chr(10) + 'pre-pass  : %.1f s   seeds %s'
          % (total, pre.get('seedsWritten')))
    st = pre.get('stages')
    if st:
        named = 0.0
        for k in ('timeOffset', 'revisits', 'poseGraph', 'carving', 'seeding'):
            v = st.get(k, 0)
            named += v
            print('  %-12s %6.2f s  %5.1f%%' % (k, v, 100 * v / max(total, 1e-9)))
        rest = total - named
        print('  %-12s %6.2f s  %5.1f%%  (bundle load + writes)'
              % ('unaccounted', rest, 100 * rest / max(total, 1e-9)))
    else:
        print('  no stage clocks in this census (build 144 or older)')

t = d.get('timings')
if not t:
    raise SystemExit('\nno timings in this census (build 100 or older)')

print('\n%-22s %9s %8s %11s' % ('', 'seconds', '% wall', 'ms/iter'))
ORDER = ['gpuBusy', 'gpuScan', 'gpuStep', 'gpuSort', 'gpuForward', 'gpuLosses',
         'earlyStopEval',
         'gpuBackward', 'gpuOptimiser', 'gpuOther', 'gpuWait',
         'supervision', 'supervisionPrefetched',
         'densify', 'previewSnapshot', 'filterSweep', 'upload']
for k in ORDER:
    if k not in t:
        continue
    print('%-22s %9.1f %7.1f%% %11.2f'
          % (k, t[k], 100 * t[k] / wall, 1000 * t[k] / iters))

th = d.get('thermals')
if th:
    names = ['nominal', 'fair', 'serious', 'critical']
    secs = th.get('secondsAtLevel', [])
    firsts = th.get('firstReachedAtIteration', [])
    print(chr(10) + 'thermals   : peak %s' % names[th.get('peak', 0)])
    for i, n in enumerate(names):
        if i < len(secs) and (secs[i] > 0.05 or (i < len(firsts) and firsts[i] >= 0)):
            at = firsts[i] if i < len(firsts) else -1
            print('  %-9s %6.1f s  %5.1f%%   first at iteration %s'
                  % (n, secs[i], 100 * secs[i] / max(wall, 1e-9),
                     at if at >= 0 else 'never'))

cb = t.get('commandBuffers', 0)
if cb:
    print('%-22s %9d %7s  %10.3f ms per buffer'
          % ('commandBuffers', cb, '', 1000 * t['gpuWait'] / cb))

# The critical path is supervision the loop actually waited for, plus the GPU
# wait, plus whatever is left. Prefetched work is deliberately NOT in it.
accounted = t.get('supervision', 0) + t.get('gpuWait', 0)
rest = wall - accounted
print('%-22s %9.1f %7.1f%% %11.2f'
      % ('unaccounted', rest, 100 * rest / wall, 1000 * rest / iters))

# Did the prefetch actually overlap?
pre = t.get('supervisionPrefetched', 0)
if pre:
    print('\nprefetch: worker built %.1f s off the critical path; the loop '
          'itself waited %.1f s for supervision.' % (pre, t['supervision']))
    print('ceiling if perfectly overlapped: max(worker, gpuBusy) + rest = '
          '%.1f s' % (max(pre, t['gpuBusy']) + rest))

# Densification health, which is where the quality questions live.
p = d.get('densifyPasses', [])
if p:
    print('\nsplit total %d   clone total %d'
          % (sum(x['addedBySplit'] for x in p),
             sum(x['addedByClone'] for x in p)))
    worst = max(p, key=lambda x: x['prunedLowOpacity'])
    print('largest opacity prune: %d at iteration %d (%d splats -> %d)'
          % (worst['prunedLowOpacity'], worst['iteration'],
             worst['splatCountBefore'], worst['splatCountAfter']))


# ---------------------------------------------------------------------------
# THE NUMBERS THAT MATTER FOR THE SIZE DISTRIBUTION, which is the whole gap.
#
# Added 2026-09-08 after the offline harness established that our Gaussians are
# all one size (p90/p10 spread 1.8x) against a good model's 8.9x, and that
# relocation, not the growth split, is the mechanism that can widen it.
# ---------------------------------------------------------------------------
import collections

passes = d.get('densifyPasses') or []
if passes:
    tot = collections.Counter()
    for e in passes:
        for k, v in e.items():
            if isinstance(v, (int, float)):
                tot[k] += v
    print()
    print('densification, summed over %d passes' % len(passes))
    print('  addedBySplit   %8d      addedByClone %8d'
          % (tot['addedBySplit'], tot['addedByClone']))
    created = tot['addedBySplit'] + tot['addedByClone']
    if created:
        print('  growth split share %5.1f%%   (reference is about 20%%)'
              % (100.0 * tot['addedBySplit'] / created))
    print('  relocated      %8d      donors available %8d'
          % (tot['relocated'], tot['relocationDonorsAvailable']))
    print("    ^ relocation IS a split using a dead splat's slot, and after")
    print('      the cap is reached it is the ONLY thing that can shrink a')
    print('      Gaussian. Build 182: 21,000 relocations, 835 donors a pass')
    print('      against a 15,000 allowance, so donors were the limit.')
    print('  prunedLowOpacity %6d    prunedOversized %6d   nonFinite %d'
          % (tot['prunedLowOpacity'], tot['prunedOversized'],
             tot['prunedNonFinite']))

# The size distribution, if the exported model is beside the census.
ply = path.replace('train_census.json', 'model.ply')
if os.path.exists(ply):
    try:
        import numpy as np
        f = open(ply, 'rb')
        h = b''
        while not h.endswith(b'end_header\n'):
            h += f.read(1)
        names, n = [], 0
        for line in h.decode().split('\n'):
            q = line.split()
            if not q:
                continue
            if q[0] == 'element' and q[1] == 'vertex':
                n = int(q[2])
            elif q[0] == 'property' and q[1] == 'float':
                names.append(q[2])
        arr = np.frombuffer(f.read(n * len(names) * 4), dtype='<f4')
        arr = arr.reshape(n, len(names))
        col = {nm: arr[:, i] for i, nm in enumerate(names)}
        sc = np.exp(np.clip(np.stack(
            [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
        ).astype(np.float64), -12, 3))
        lg = sc.max(axis=1) * 1000
        p10, p50, p90 = np.percentile(lg, [10, 50, 90])
        op = 1 / (1 + np.exp(-col['opacity'].astype(np.float64)))
        print()
        print('SIZE DISTRIBUTION, the whole quality gap')
        print('  largest axis mm : p10 %6.2f  p50 %6.2f  p90 %7.2f'
              % (p10, p50, p90))
        print('  spread p90/p10  : %6.2fx' % (p90 / max(p10, 1e-9)))
        print('  aspect p50      : %6.1f:1'
              % np.median(sc.max(axis=1) / np.maximum(sc.min(axis=1), 1e-12)))
        print('  opacity p50     : %6.3f' % np.median(op))
        print()
        print('  build 182 was p50 19.57 mm, spread 1.8x, aspect 7.6:1')
        print('  build 172 was p50 13.2 mm')
        print('  Scaniverse, same room: p50 3.39 mm, spread 8.9x, aspect 1.9:1')

        # ADAPTIVITY, the two numbers that separate "the seeder decided the
        # model" from "training decided the model". Both are cheap and both
        # measured Scaniverse as adaptive and us as not.
        try:
            from scipy.spatial import cKDTree      # noqa: F401
        except Exception:
            pass
        xyz = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
        cell = np.floor(xyz / 0.05).astype(np.int64)
        _, counts = np.unique(cell, axis=0, return_counts=True)
        cs = np.sort(counts)
        top10 = cs[int(len(cs) * 0.9):].sum() / max(cs.sum(), 1)
        n = len(cs)
        gini = (2 * np.arange(1, n + 1) - n - 1).dot(cs) / (n * cs.sum()) if n else 0.0
        # SHAPE, CLASSIFIED PROPERLY. `max/min` was the first version of this
        # and it is wrong: it cannot tell a NEEDLE (one long axis, two short)
        # from a DISC (two long, one short), and the disc prior in this trainer
        # deliberately produces discs. It reported 56 per cent "needles" on a
        # model that is 77.5 per cent discs and 16.2 per cent needles, and it
        # sent me looking for a bug in relocation that was not there.
        srt = np.sort(sc, axis=1)[:, ::-1]
        r21 = srt[:, 1] / np.maximum(srt[:, 0], 1e-12)
        r32 = srt[:, 2] / np.maximum(srt[:, 1], 1e-12)
        needle_m = r21 < 0.4
        disc_m = (~needle_m) & (r32 < 0.4)
        blob = float((~(needle_m | disc_m)).mean())
        needle = float(needle_m.mean())
        disc_frac = float(disc_m.mean())
        print()
        print('ADAPTIVITY, is the model multi-scale or one-size')
        print('  densest 10%% of 5cm voxels hold %5.1f%% of splats' % (100 * top10))
        print('  Gini of occupied-voxel density   %6.3f' % gini)
        print('  needles %5.1f%%   discs %5.1f%%   blobs %5.1f%%   (s2/s1 %.2f, s3/s2 %.2f)'
              % (100 * needle, 100 * disc_frac, 100 * blob,
                 float(np.median(r21)), float(np.median(r32))))
        print()
    
        print('  Scaniverse, same room, foreground only (dome excluded):')
        print('    needles  8.9%   discs  4.9%   blobs 86.3%   (s2/s1 0.73, s3/s2 0.78)')
        print('    densest 10% hold 61.2%, Gini 0.753')
        print('  ^ THE SHAPE GAP: their third axis is 78% of the second, ours is 18%.')
        print('    They are near-isotropic BLOBS; we are flat DISCS, which is what')
        print('    the effective-rank prior (discTargetRank 2) is asking for.')

    except Exception as exc:
        print('\n(model.ply present but not read: %s)' % exc)


# ---------------------------------------------------------------------------
# THE HELD-OUT CURVE. The only thing that says how many rounds this capture
# actually wants, as opposed to how many a paper used.
# ---------------------------------------------------------------------------
curve = d.get('heldOutCurve') or []
if curve:
    print()
    print('HELD-OUT CURVE, scored on frames the model never trains on')
    best = max(e['psnr'] for e in curve)
    for e in curve:
        mark = '  <-- best' if abs(e['psnr'] - best) < 1e-6 else ''
        settled = ''
        print('  iter %6d   PSNR %6.2f   splats %8d%s%s'
              % (e['iteration'], e['psnr'], e['splatCount'], settled, mark))
    sl0 = d['slices'][0]
    print()
    print('  stoppedEarly %s   best %.2f dB at iteration %s'
          % (sl0.get('stoppedEarly'), sl0.get('bestHeldOutPSNR') or float('nan'),
             sl0.get('bestHeldOutIteration')))
    if sl0.get('bestHeldOutIteration'):
        print('  ^ a fixed budget chosen from data would be about %d iterations'
              % sl0['bestHeldOutIteration'])
