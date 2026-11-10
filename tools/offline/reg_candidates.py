"""Replica of SubmapPoseRefiner.detectRevisits steps 1-2 (anchors, geometric
gate, top-N heap) plus a read of the pose graph's submap corrections.

Validation: the census for build 266 says anchorFrames 411 and
geometricCandidates 24,916. If this replica prints those, the cap analysis
below describes the pairs the device actually tried.

Run: python -u reg_candidates.py
"""
import io, json, os
from collections import Counter
import numpy as np
import project as P

MIN_GAP_S = 8.0
MAX_CENTRE_M = 1.5
MAX_ANGLE_DEG = 50.0
SPACING_M = 0.25
SPACING_S = 0.5
ROT_SIGMA_DEG, TR_SIGMA_M = 0.30, 0.015       # revisit noise model
SM_ROT_SIGMA_DEG, SM_TR_SIGMA_M = 1.0, 0.05   # smoothness noise model


def qmat(q):
    return P.quat_to_matrix(q)


def load():
    b = json.load(io.open(os.path.join(P.D, 'capture_bundle.json'), encoding='utf-8'))
    r = json.load(io.open(os.path.join(P.D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
    return b, r


def owners(frames, submaps):
    out = {}
    for f in frames:
        best, bd, found = submaps[0]['index'], 1e300, False
        for s in submaps:
            if not (s['firstFrame'] <= f['index'] <= s['lastFrame']):
                continue
            c = 0.5 * (s['startTimeSeconds'] + s['endTimeSeconds'])
            d = abs(f['timestampSeconds'] - c)
            if d < bd:
                bd, best, found = d, s['index'], True
        if not found:
            best = min(submaps, key=lambda s: abs(
                f['timestampSeconds'] - 0.5 * (s['startTimeSeconds'] + s['endTimeSeconds'])))['index']
        out[f['index']] = best
    return out


def se3_log_norms(R, t):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang, np.linalg.norm(t)


def main():
    b, r = load()
    frames = sorted(b['frames'], key=lambda f: f['timestampSeconds'])
    submaps = r['submaps']
    own = owners(frames, submaps)

    geo = {}
    for f in frames:
        R = qmat(f['rawPose']['rotation']).astype(np.float32)
        t = np.asarray(f['rawPose']['translation'], np.float32)
        geo[f['index']] = (-R.T @ t, R.T @ np.array([0, 0, 1], np.float32))

    anchors, last_c, last_t = [], None, -1e300
    for f in frames:
        if f.get('depthPath') is None:
            continue
        c = geo[f['index']][0]
        moved = last_c is None or np.linalg.norm(c - last_c) >= SPACING_M
        waited = f['timestampSeconds'] - last_t >= SPACING_S
        if anchors and not (moved or waited):
            continue
        anchors.append(f)
        last_c, last_t = c, f['timestampSeconds']
    print('anchors %d   (census anchorFrames 411)' % len(anchors))

    cands = []
    for i, a in enumerate(anchors):
        ca, fa = geo[a['index']]
        sa = own[a['index']]
        for j in range(i + 1, len(anchors)):
            bb = anchors[j]
            if bb['timestampSeconds'] - a['timestampSeconds'] < MIN_GAP_S:
                continue
            if own[bb['index']] == sa:
                continue
            cb, fb = geo[bb['index']]
            d = float(np.linalg.norm(ca - cb))
            if d > MAX_CENTRE_M:
                continue
            cr = np.linalg.norm(np.cross(fa / np.linalg.norm(fa), fb / np.linalg.norm(fb)))
            ang = float(np.degrees(np.arctan2(cr, np.dot(fa, fb) / np.linalg.norm(fa) / np.linalg.norm(fb))))
            if ang > MAX_ANGLE_DEG:
                continue
            q = a['qc']['weight'] * bb['qc']['weight']
            s = q * (1 - d / MAX_CENTRE_M) * (1 - ang / MAX_ANGLE_DEG)
            cands.append((s, a['index'], bb['index'], sa, own[bb['index']], d, ang,
                          bb['timestampSeconds'] - a['timestampSeconds']))
    print('geometric candidates %d   (census geometricCandidates 24,916)' % len(cands))

    cands.sort(key=lambda c: -c[0])
    allpairs = Counter((c[3], c[4]) for c in cands)
    print('distinct submap pairs among ALL candidates: %d' % len(allpairs))
    for cap in (600, 1000, 2000, 5000, len(cands)):
        top = cands[:cap]
        sp = Counter((c[3], c[4]) for c in top)
        fr = Counter([c[1] for c in top] + [c[2] for c in top])
        subs = Counter([c[3] for c in top] + [c[4] for c in top])
        per = np.array(sorted(sp.values(), reverse=True))
        print('\ntop %5d: cut score %.4f | submap pairs %3d | submaps touched %2d/19 | '
              'distinct frames %3d | pairs per submap-pair max %d median %.0f | '
              'top-3 submap pairs hold %.1f%%'
              % (cap, top[-1][0], len(sp), len(subs), len(fr), per.max(), np.median(per),
                 100 * per[:3].sum() / len(top)))
        print('   centre dist median %.2f m, view angle median %.1f deg, time gap median %.0f s'
              % (np.median([c[5] for c in top]), np.median([c[6] for c in top]),
                 np.median([c[7] for c in top])))
    top600 = cands[:600]
    print('\nper-submap involvement in the top 600 (submap: pairs touching it):')
    subs = Counter([c[3] for c in top600] + [c[4] for c in top600])
    print('  ' + '  '.join('%d:%d' % (k, subs.get(k, 0)) for k in range(len(submaps))))
    subs_all = Counter([c[3] for c in cands] + [c[4] for c in cands])
    print('per-submap involvement in ALL candidates:')
    print('  ' + '  '.join('%d:%d' % (k, subs_all.get(k, 0)) for k in range(len(submaps))))
    print('\ntop-600 submap pairs (a,b):count')
    print('  ' + '  '.join('%s:%d' % (k, v) for k, v in Counter((c[3], c[4]) for c in top600).most_common()))
    missing = [k for k in allpairs if k not in Counter((c[3], c[4]) for c in top600)]
    print('submap pairs with candidates but NONE in the top 600: %d  %s' % (len(missing), sorted(missing)))

    # ---------------- pose graph read-back
    print('\n=== SUBMAP CORRECTIONS (prepass_result.submaps[].correction) ===')
    corr = []
    for s in submaps:
        c = s['correction']
        R = qmat(c['rotation'])
        t = np.asarray(c['translation'], float)
        corr.append((R, t))
        ang, tr = se3_log_norms(R, t)
        print('  submap %2d frames %3d-%3d  |corr| %6.2f cm %5.2f deg'
              % (s['index'], s['firstFrame'], s['lastFrame'], 100 * tr, ang))
    print('\nadjacent smoothness residual  M_{k+1} M_k^-1  (in sigmas: 5 cm, 1 deg)')
    wn = []
    for k in range(len(corr) - 1):
        Ra, ta = corr[k]
        Rb, tb = corr[k + 1]
        Rd = Rb @ Ra.T
        td = tb - Rd @ ta
        ang, tr = se3_log_norms(Rd, td)
        w = np.hypot(ang / SM_ROT_SIGMA_DEG, tr / SM_TR_SIGMA_M)
        wn.append(w)
        print('  %2d->%2d  %6.2f cm %5.2f deg   whitened %.2f' % (k, k + 1, 100 * tr, ang, w))
    print('  median whitened smoothness residual %.2f (Huber delta at the end = 3*2.0 = 6.0)'
          % np.median(wn))

    # refined vs raw: per-frame shift and consistency with corrections
    ref = P.load_poses()
    shifts = []
    for f in frames:
        R0 = qmat(f['rawPose']['rotation']); t0 = np.asarray(f['rawPose']['translation'], float)
        R1, t1 = ref[f['index']]
        R1 = qmat(R1)
        c0 = -R0.T @ t0; c1 = -R1.T @ t1
        shifts.append(100 * np.linalg.norm(c1 - c0))
    shifts = np.array(shifts)
    print('\nrefined vs raw camera centre shift: median %.2f cm  p90 %.2f  max %.2f  (census 7.49 / 26.58 max)'
          % (np.median(shifts), np.percentile(shifts, 90), shifts.max()))

    # whitened revisit residual implied by the census medians
    pc = json.load(io.open(os.path.join(P.D, 'prepass', 'census.json'), encoding='utf-8'))['poseGraph']
    rt = pc['finalResidualMedianCentimeters'] / 100 / TR_SIGMA_M
    rr = pc['finalResidualMedianDegrees'] / ROT_SIGMA_DEG
    print('\ncensus final median revisit residual %.2f cm = %.2f sigma, %.2f deg = %.2f sigma'
          % (pc['finalResidualMedianCentimeters'], rt, pc['finalResidualMedianDegrees'], rr))
    print('  (medians of separate components; whitened norm of a median-like edge ~ %.2f sigma,'
          ' final Huber delta 2.0)' % np.hypot(rt, rr))

    cap = b.get('revisitPairs') or []
    if cap:
        print('\ncapture-time revisitPairs: %d; methods %s; confidence>0: %d'
              % (len(cap), Counter(p.get('method') for p in cap),
                 sum(1 for p in cap if p.get('confidence', 0) > 0)))


if __name__ == '__main__':
    main()
