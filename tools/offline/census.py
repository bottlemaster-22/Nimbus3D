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
_sl0 = d['slices'][0]
_ssim = _sl0.get('heldOutSSIM')
print('splats     : %d   held-out PSNR %.2f   SSIM %s'
      % (m['splatCount'], m['heldOutPSNR'],
         ('%.4f' % _ssim) if _ssim is not None else 'n/a'))
print('  ^^ SSIM is the STRUCTURAL score, 0 to 1, higher is better.')
print('     PSNR can RISE while the picture gets visibly worse: build 240')
print('     gained 3 dB over 226 and lost detail in cluttered regions.')
print('     When the two disagree, believe SSIM.')

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
        for k in ('timeOffset', 'revisits', 'poseGraph', 'carving',
                  'trust', 'edges', 'glass', 'seeding'):
            v = st.get(k, 0)
            named += v
            print('  %-12s %6.2f s  %5.1f%%' % (k, v, 100 * v / max(total, 1e-9)))
        rest = total - named
        print('  %-12s %6.2f s  %5.1f%%  (bundle load + writes)'
              % ('unaccounted', rest, 100 * rest / max(total, 1e-9)))
        tb = pre.get('trustBuild')
        if tb:
            print('  trust split : setup %.2f  serial %.2f (%d slots)  parallel %.2f'
                  ' (%d slots, %d cores)  confidence %.2f  write %.2f'
                  % (tb.get('setupSeconds', 0), tb.get('serialSeconds', 0),
                     tb.get('serialSlots', 0), tb.get('parallelSeconds', 0),
                     tb.get('parallelSlots', 0), tb.get('workerCount', 0),
                     tb.get('confidenceSeconds', 0), tb.get('writeSeconds', 0)))
            if 'sweepsRun' in tb:
                print('                of parallel, in-order apply %.2f   plane sweeps run %d'
                      % (tb.get('parallelApplySeconds', 0), tb.get('sweepsRun', 0)))
        sd = pre.get('seeding') or {}
        if 'secondsSampleLoop' in sd:
            print('  seed split  : before loop %.2f (edge warm-up %.2f)  sample loop %.2f'
                  '  shaping %.2f  write %.2f'
                  % (sd.get('secondsBeforeLoop', 0), sd.get('secondsEdgeWarmup', 0),
                     sd.get('secondsSampleLoop', 0), sd.get('secondsShaping', 0),
                     sd.get('secondsWrite', 0)))
            if 'secondsPrefetchWait' in sd:
                print('                of the loop, waiting on the prefetch %.2f'
                      % sd.get('secondsPrefetchWait', 0))
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
         'densify', 'previewSnapshot', 'filterSweep', 'upload', 'prologue',
         'smartLayer', 'smartLayerTrust', 'smartLayerEdges', 'smartLayerAuthority',
         'smartLayerBackground', 'smartLayerCarver', 'encodeStep']
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
# Timed CPU buckets on the critical path but outside gpuWait's own clock.
# earlyStopEval and densify CONTAIN some gpuWait (every finish() adds to it),
# so subtracting them whole over-subtracts: the figure below is a LOWER bound
# on what no clock covers. The wall is whole-second ISO dates, +/- 1 s.
cpu = sum(t.get(k, 0) for k in ('upload', 'previewSnapshot', 'densify',
                                'earlyStopEval', 'filterSweep', 'prologue',
                                'smartLayer', 'encodeStep'))
print('%-22s %9.1f %7.1f%% %11.2f   (lower bound, wall +/- 1 s)'
      % ('  of which untimed', rest - cpu, 100 * (rest - cpu) / wall,
         1000 * (rest - cpu) / iters))

# Did the prefetch actually overlap?
pre = t.get('supervisionPrefetched', 0)
if pre:
    print('\nprefetch: worker built %.1f s off the critical path; the loop '
          'itself waited %.1f s for supervision.' % (pre, t['supervision']))
    print('ceiling if perfectly overlapped: max(worker, gpuBusy) + rest = '
          '%.1f s' % (max(pre, t['gpuBusy']) + rest))
_sl = (d.get('slices') or [{}])[-1] if isinstance(d.get('slices'), list) else {}
if _sl.get('heldOutPSNRPoseAligned') is not None:
    print('HELD-OUT, CAMERAS ALIGNED (build 332+): PSNR %.2f  SSIM %.4f   (raw fitted %.2f / %.4f): the gap that was registration, not the model'
          % (_sl['heldOutPSNRPoseAligned'], _sl.get('heldOutSSIMPoseAligned') or 0,
             _sl.get('heldOutPSNRExposureFitted') or 0, _sl.get('heldOutSSIM') or 0))
if t.get('supervisionPreloaded'):
    print('preloaded %d frames in the background before their level began' % t['supervisionPreloaded'])
if t.get('poseCheckDescends', -1) >= 0:    print('POSE CHECK: photometric loss %.5f -> %.5f after one full-rate step  -> %s'
          % (t['poseCheckLossBefore'], t['poseCheckLossAfter'],
             'full-rate refinement ON' if t['poseCheckDescends'] else 'rate left inert'))
if t.get('coarseSteps'):    print('coarse phase: %d steps at a %d px long edge (the rest at full size)'
          % (t['coarseSteps'], t.get('coarseLongEdgePixels', 0)))
if t.get('snapshotsStaged'):    print('preview snapshots: %d copied inside a step and converted off the loop'
          % t['snapshotsStaged'])
if t.get('densifyGatherChecks'):    print('densify gather: %d pass(es) checked against the CPU path, %d words differed'
          % (t['densifyGatherChecks'], t.get('densifyGatherMismatches', 0)))
if t.get('mergedSteps'):    print('merged steps: %d of %d overlapped ran as one command buffer (%d in warm-up); %d ran short of instance slots'
          % (t['mergedSteps'], t.get('overlappedSteps', 0), t.get('warmupOverlappedSteps', 0),
             t.get('truncatedInstanceSteps', 0)))
if 'supervisionCacheHits' in t:
    print('frame cache: %d builds served without a decode, cache peak %.0f MB'
          % (t['supervisionCacheHits'], t.get('supervisionCacheMegabytes', 0)))
if t.get('opacityResets'):
    print('opacity resets: %d' % t['opacityResets'])
if t.get('memoryFootprintPeakMegabytes'):
    print('MEMORY: run footprint peaked at %.0f MB (ceiling %.0f MB)'
          % (t['memoryFootprintPeakMegabytes'],
             (d.get('budgetAsRun') or {}).get('memoryCeilingBytes', 0) / 1048576.0))

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

# Keyframe span: which part of the capture the model actually trained on.
_first, _last, _n = d.get('keyframeFirstIndex', -1), d.get('keyframeLastIndex', -1), d.get('framesInBundle', 0)
if _n and _last >= 0:
    print('\nKEYFRAMES  : frames %d to %d of %d  (%.0f%% of the capture; after the last keyframe: %d frames)'
          % (_first, _last, _n, 100.0 * (_last + 1) / _n, _n - 1 - _last))

# Camera deltas the trainer learned per trained frame (build 274+).
for s_ in (d.get('slices') or []):
    if s_.get('cameraDeltaFrames'):
        print('CAMERA DELTAS: %d frames  median %.3f deg / %.2f cm  max %.3f deg / %.2f cm'
              '  common %.3f deg / %.2f cm   (clamp ceiling ~0.9 deg / 6.3 cm)'
              % (s_['cameraDeltaFrames'], s_.get('cameraDeltaMedianDegrees', 0),
                 s_.get('cameraDeltaMedianCentimeters', 0), s_.get('cameraDeltaMaxDegrees', 0),
                 s_.get('cameraDeltaMaxCentimeters', 0), s_.get('cameraDeltaCommonDegrees', 0),
                 s_.get('cameraDeltaCommonCentimeters', 0)))

# Per-frame held-out scores (build 276+), and the pose-graph tear.
for s_ in (d.get('slices') or []):
    pf = s_.get('heldOutPerFrame')
    if pf:
        print('HELD-OUT PER FRAME: ' + '  '.join('%d:%.2f' % (e['frameIndex'], e['psnr']) for e in pf))
curve = [e for e in (d.get('heldOutCurve') or []) if 'psnrRaw' in e]
if curve:
    print('HELD-OUT CURVE raw vs fitted: ' + '  '.join(
        '%d:%.2f/%.2f' % (e['iteration'], e['psnrRaw'], e.get('psnrExposureFitted') or 0) for e in curve))
try:
    _pre_census = json.load(open(pre_path, encoding='utf-8'))
except Exception:
    _pre_census = None
if _pre_census:
    pg = _pre_census.get('poseGraph') or {}
    if 'maxConsecutiveTearCentimeters' in pg:
        print('POSE GRAPH TEAR: worst consecutive-frame %.1f cm / %.2f deg  (build 266: 23.9 cm / 3.1 deg)'
              % (pg['maxConsecutiveTearCentimeters'], pg.get('maxConsecutiveTearDegrees', 0)))

# Sampled GPU stage profile (build 286+): ms per iteration per stage.
_n = t.get('profiledSteps', 0)
if _n:
    parts = [('sort', 'gpuSort'), ('forward', 'gpuForward'), ('losses', 'gpuLosses'),
             ('backward', 'gpuBackward'), ('optimiser', 'gpuOptimiser')]
    tot = sum(t.get(k, 0) for _, k in parts)
    print('GPU STAGES (%d sampled steps): ' % _n + '  '.join(
        '%s %.2f ms (%.0f%%)' % (nm, 1000 * t.get(k, 0) / _n, 100 * t.get(k, 0) / max(tot, 1e-12))
        for nm, k in parts) + '  | sum %.2f ms' % (1000 * tot / _n))

# Backward calibration (build 288+): plain (A) against SIMD-summed (B).
if t.get('backwardCalibrationSteps', 0):
    _k = t['backwardCalibrationSteps']
    print('BACKWARD CALIBRATION: %d steps  A %.2f ms  B %.2f ms  (B/A %.3f)  worst rel diff %.2e  -> %s'
          % (_k, 1000 * t['backwardSecondsA'] / _k, 1000 * t['backwardSecondsB'] / _k,
             t['backwardSecondsB'] / max(t['backwardSecondsA'], 1e-12),
             t['backwardRelativeDifference'],
             'B (SIMD-summed) used' if t.get('backwardSimdSumChosen') else 'A kept'))

# Sort calibration (build 290+; 306+ adds the legacy reference L and the splat-order sort).
if t.get('sortCalibrationSteps', 0):
    _k = t['sortCalibrationSteps']
    if 'sortLegacySeconds' in t:
        print('SORT CALIBRATION: %d steps  legacy %.2f ms  legacy+SIMD %.2f ms  splat-order %.2f ms  +SIMD %.2f ms  '
              'mismatched: legacy+SIMD %d, splat-order %d, +SIMD %d  -> splat-order %s, SIMD scatter %s'
              % (_k, 1000 * t['sortLegacySeconds'] / _k, 1000 * t.get('sortLegacySimdSeconds', 0) / _k,
                 1000 * t['sortSecondsA'] / _k,
                 1000 * t['sortSecondsB'] / _k, t.get('sortLegacySimdMismatchSteps', 0),
                 t.get('sortSplatOrderMismatchSteps', 0),
                 t.get('sortMismatchSteps', 0),
                 'USED' if t.get('sortSplatOrderChosen') else 'not used',
                 'USED' if t.get('sortSimdScanChosen') else 'not used'))
    else:
        print('SORT CALIBRATION: %d steps  A %.2f ms  B %.2f ms  (B/A %.3f)  mismatched steps %d  -> %s'
              % (_k, 1000 * t['sortSecondsA'] / _k, 1000 * t['sortSecondsB'] / _k,
                 t['sortSecondsB'] / max(t['sortSecondsA'], 1e-12), t.get('sortMismatchSteps', 0),
                 'B (SIMD-prefix) used' if t.get('sortSimdScanChosen') else 'A kept'))

# Forward calibration (build 302+): one-pixel (A) against two-pixel (B).
if t.get('forwardCalibrationSteps', 0):
    _k = t['forwardCalibrationSteps']
    print('FORWARD CALIBRATION: %d steps  A %.2f ms  B %.2f ms  (B/A %.3f)  max diff %.2e  mismatched steps %d  -> %s'
          % (_k, 1000 * t['forwardSecondsA'] / _k, 1000 * t['forwardSecondsB'] / _k,
             t['forwardSecondsB'] / max(t['forwardSecondsA'], 1e-12), t.get('forwardMaxDifference', 0),
             t.get('forwardMismatchSteps', 0),
             'B (two-pixel) used' if t.get('forwardTwoPixelChosen') else 'A kept'))

# Two-pixel backward calibration (build 304+): chosen backward (A) against two-pixel (B).
if t.get('backwardTwoPixelCalibrationSteps', 0):
    _k = t['backwardTwoPixelCalibrationSteps']
    print('BACKWARD 2-PIXEL CALIBRATION: %d steps  A %.2f ms  B %.2f ms  (B/A %.3f)  worst rel diff %.2e  -> %s'
          % (_k, 1000 * t['backwardTwoPixelSecondsA'] / _k, 1000 * t['backwardTwoPixelSecondsB'] / _k,
             t['backwardTwoPixelSecondsB'] / max(t['backwardTwoPixelSecondsA'], 1e-12),
             t['backwardTwoPixelRelativeDifference'],
             'B (two-pixel) used' if t.get('backwardTwoPixelChosen') else 'A kept'))

# Blur calibration (build 318+): two-pass SSIM blur (A) against the fused one (B).
if t.get('blurCalibrationSteps', 0):
    _k = t['blurCalibrationSteps']
    print('BLUR CALIBRATION: %d steps  A %.2f ms  B %.2f ms  (B/A %.3f)  mismatched steps %d  -> %s'
          % (_k, 1000 * t['blurSecondsA'] / _k, 1000 * t['blurSecondsB'] / _k,
             t['blurSecondsB'] / max(t['blurSecondsA'], 1e-12), t.get('blurMismatchSteps', 0),
             'B (fused) used' if t.get('blurFusedChosen') else 'A kept'))
