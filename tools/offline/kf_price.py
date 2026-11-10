"""Prices keyframe counts and gates against the build 266 model, with the
CLAMPED projection (project.TANGENT_CLAMP). Visibility = trainer_preprocess
survives (frustum, alpha, tiles), no occlusion, same method as refutation [40].

For each config: span, entries / distinct, held-out leak, distinct-view count
per Gaussian (whole 299,976 model), share with <=3 and 0 views, median pairwise
view angle (15k-Gaussian sample, fixed seed), and visits at 4,000 iterations.
"""
import io, json, os, sys, time
import numpy as np
import project as P
import kf_exact as K

assert P.TANGENT_CLAMP
D = P.D
census = json.load(io.open(os.path.join(D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
tx, ty = (rw + 15) // 16, (rh + 15) // 16
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(D, 'model', 'model.ply'))
poses = P.load_poses()
mean = np.stack([col['x'], -col['y'], -col['z']], 1).astype(np.float64)
ITERS = 4000
SUBMAPS = 19

_vis = {}


def vis(idx):
    if idx not in _vis:
        R = P.quat_to_matrix(poses[idx][0]); t = poses[idx][1]
        ok = P.project(col, count, R, t, fx, fy, cx, cy, tx, ty)[0]
        _vis[idx] = np.packbits(ok)
    return np.unpackbits(_vis[idx])[:count].astype(bool)


def cam_center(idx):
    R = P.quat_to_matrix(poses[idx][0]); return -R.T @ poses[idx][1]


rng = np.random.default_rng(0)
SAMP = rng.choice(count, 15000, replace=False)


def price(name, ch):
    kf = [int(K.IDX[b]) for b, _, _ in ch]
    tr, ho = K.split_held_out(kf)
    dtr = sorted(set(tr))
    leak = len(set(ho) & set(tr))
    mult = {k: tr.count(k) for k in dtr}
    nv = np.zeros(count, np.int32)          # distinct views
    visits = np.zeros(count, np.float64)    # gradient visits at 4,000 iterations
    Vs = np.zeros((len(dtr), len(SAMP)), bool)
    for j, k in enumerate(dtr):
        v = vis(k)
        nv += v
        visits += v * (ITERS * mult[k] / len(tr))
        Vs[j] = v[SAMP]
    cams = np.array([cam_center(k) for k in dtr])
    med = []
    for s, g in enumerate(SAMP):
        kk = np.where(Vs[:, s])[0]
        if len(kk) < 2:
            continue
        d = cams[kk] - mean[g]; d /= np.linalg.norm(d, axis=1)[:, None]
        A = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        med.append(np.median(A[np.triu_indices(len(kk), 1)]))
    slices = SUBMAPS if len(kf) >= SUBMAPS * 12 else 1
    row = dict(name=name, n=len(kf), distinct=len(set(kf)), train=len(tr), dtrain=len(dtr),
               held=len(ho), leak=leak, first=min(kf), last=max(kf),
               vpg=float(nv.mean()), le3=float(100 * (nv <= 3).mean()), zero=float(100 * (nv == 0).mean()),
               ang=float(np.median(med)), vpk=ITERS / len(tr), vpd=ITERS / len(dtr),
               gvis=float(visits.mean()), gvis_p10=float(np.percentile(visits, 10)), slices=slices)
    print('%-34s n %3d (%3d dist) tr %3d (%3d dist) ho %2d leak %d | span %3d..%3d | views/G %5.2f '
          '<=3 %5.2f%% 0v %5.2f%% | ang med %5.1f | visits/kf %5.1f per distinct %5.1f | '
          'visits/G mean %6.0f p10 %5.0f | slices %d'
          % (name, row['n'], row['distinct'], row['train'], row['dtrain'], row['held'], leak,
             row['first'], row['last'], row['vpg'], row['le3'], row['zero'], row['ang'],
             row['vpk'], row['vpd'], row['gvis'], row['gvis_p10'], slices), flush=True)
    return row


def calibrate(N, param, lo, hi, **kw):
    """LARGEST param such that the un-broken walk still yields >= N entries,
    so the real walk (break at N) ends near the end of the capture and the
    even-stride fallback never fires."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        n = len(K.select(N, nobreak=True, **{param: mid}, **kw)[0])
        if n >= N:
            lo = mid
        else:
            hi = mid
    return lo


def fell_back(ch):
    return any(w == 'stride' for _, w, _ in ch)


if __name__ == '__main__':
    t0 = time.time()
    rows = []
    rows.append(price('266 shipped N=120', K.select(120)[0]))
    rows.append(price('264 replica (blur key)', K.select(120, key=-K.MB)[0]))
    rows.append(price('lookahead 0 N=120', K.select(120, lookahead=0)[0]))
    for N in [150, 180, 204, 240]:
        rows.append(price('shipped code N=%d' % N, K.select(N)[0]))
    for N in [120, 150, 204]:
        ch = K.select(N, resume_after_best=True)[0]
        rows.append(price('resume only N=%d%s' % (N, ' STRIDE-FALLBACK' if fell_back(ch) else ''), ch))
    sp120 = float(K.PATH / K.f32(120))
    rows.append(price('resume, gates of 120, whole walk',
                      K.select(120, spacing=sp120, nobreak=True, resume_after_best=True)[0]))
    # resume + spacing calibrated so N consumes the whole walk (turn stays 0.02)
    for N in [120, 150, 180, 204, 227]:
        s = calibrate(N, 'spacing_scale', 0.02, 20.0, resume_after_best=True)
        ch = K.select(N, spacing_scale=s, resume_after_best=True)[0]
        print('  calibrated spacing at N=%d: x%.4f of path/N = %.4f m' % (N, s, s * float(K.PATH) / N))
        rows.append(price('resume+spacing x%.3f N=%d%s' % (s, N, ' FB' if fell_back(ch) else ''), ch))
    # resume + rotation threshold calibrated (spacing stays path/N)
    for N in [120, 150, 180, 204]:
        t = calibrate(N, 'turn', 1e-4, 2.0, resume_after_best=True)
        ch = K.select(N, turn=t, resume_after_best=True)[0]
        print('  calibrated turn at N=%d: %.4f (%.1f deg)' % (N, t, np.degrees(np.arccos(max(-1, 1 - t)))))
        rows.append(price('resume+turn %.4f N=%d%s' % (t, N, ' FB' if fell_back(ch) else ''), ch))
    # shipped code (duplicates kept) + turn calibrated, at 120 only
    t = calibrate(120, 'turn', 1e-4, 2.0)
    ch = K.select(120, turn=t)[0]
    print('  shipped-code calibrated turn at 120: %.4f (%.1f deg)' % (t, np.degrees(np.arccos(1 - t))))
    rows.append(price('shipped+turn %.4f N=120' % t, ch))
    # time/translation mix: no rotation clause, admit on distance OR elapsed time
    for N in [120, 150, 180, 204]:
        tg = calibrate(N, 'time_gate', 0.01, 60.0, resume_after_best=True, turn=10.0)
        ch = K.select(N, turn=10.0, time_gate=tg, resume_after_best=True)[0]
        print('  calibrated time gate at N=%d: %.3f s' % (N, tg))
        rows.append(price('resume+dist|time %.2fs N=%d%s' % (tg, N, ' FB' if fell_back(ch) else ''), ch))
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), '_kf_price.json'), 'w'), indent=1)
    print('done in %.0f s, %d frames projected' % (time.time() - t0, len(_vis)))
