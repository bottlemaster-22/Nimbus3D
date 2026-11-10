"""REFUTATION TEST for "the F4 transition-width term is the unbounded one:
its gradient is 1.4/span with span floored at 0.1 mm, and at span 1e-4 it is
564,904x the whole photometric gradient".

The shader:
    const float span = max(s.mode1 - s.mode0, 1e-4f);
    const float t    = clamp((expected - s.mode0) / span, 0.0f, 1.0f);
    if (t > 0.0f && t < 1.0f) { dL_dExpected += tw * 4.0f * (1.0f - 2.0f*t) / span; }

The gradient is GATED on t strictly inside (0,1), i.e. on `expected` landing
strictly between the two modes.  That window IS span wide.  So the peak
coefficient grows as 1/span and the probability of ever paying it shrinks as
span.  Their product is span-INDEPENDENT:

    E[|g|] = integral over the window of p(x) * tw*4*|1-2t|/span dx
           = tw*4 * integral_0^1 p(mode0 + t*span) |1-2t| dt
          -> 2 * tw * p_bar        as span -> 0,          and
    E[ g ] = 0 exactly, because (1-2t) is antisymmetric about t = 1/2.

p_bar is the density of the rendered `expected` depth per metre, which IS
measurable from a real render.  This measures it.
"""
import io, json, os, sys
import numpy as np
import project as P
import detail as Dt
import lossq_ledger as L

census = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
sl = census['slices'][0]
rw, rh = sl['renderWidth'], sl['renderHeight']
N = rw*rh
SUP = sl['depthSamplesSupervisedTotal']/sl['depthSupervisionFramesMeasured']
invS = 1.0/SUP
fx, fy, cx, cy = P.load_intrinsics(rw, rh)
col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
poses = P.load_poses()
frames = dict(Dt.frames_with_images())
qc = {int(f['index']): float(f['qc']['weight']) for f in Dt.bundle()['frames']}
TRANS_W, DEPTH_SCALE = 0.35, 1.0

for frame in [int(a) for a in sys.argv[1:]] or [8]:
    rot, t = poses[frame]
    R = P.quat_to_matrix(rot)
    C, D, Tf, a0, c0, d0 = L.render_with_depth(col, count, R, t, fx, fy, cx, cy,
                                               rw, rh, 0.25+1.33333, 2)
    alpha = 1.0 - Tf
    expected = D/np.maximum(alpha, 1e-4)
    e = expected[(alpha > 0.01) & np.isfinite(expected)]
    print('\n=== frame %d: the rendered `expected` depth the F4 window has to catch ===' % frame)
    print('  expected depth (m): p1 %.4f p10 %.4f p50 %.4f p90 %.4f p99 %.4f'
          % tuple(np.percentile(e, [1, 10, 50, 90, 99])))
    # empirical density per metre, from a fine histogram, at the mode and at p90
    hist, edges = np.histogram(e, bins=2000, range=(0.0, float(np.percentile(e, 99.5))))
    dens = hist/e.size/(edges[1]-edges[0])
    print('  density of `expected` per metre: peak %.3f  p90-of-bins %.3f  mean over the '
          'occupied range %.3f' % (dens.max(), np.percentile(dens[dens>0], 90), dens[dens>0].mean()))
    pbar_peak = dens.max()

    # what the finding priced, and what the gate makes it worth
    ph = None
    # photometric dL/dalpha, same construction as lossq_ledger
    import lossq_ssim as S
    _, gt = Dt.luma_of(frames[frame], rw, rh); gt = gt.astype(np.float64)
    gain, bias = __import__('lossq_blur').fit_exposure(C, gt)
    rendered = gain*C + bias
    diff = rendered - gt
    g_l1 = qc[frame]*(1-S.LAMBDA_SSIM)/N*np.sign(diff)/3.0
    X = (rendered*S.LUMA).sum(2); Y = (gt*S.LUMA).sum(2)
    SS = S.ssim_terms(X, Y, qc[frame]*S.LAMBDA_SSIM/N)
    g_ss = SS['dLdX'][:, :, None]*S.LUMA[None, None, :]
    dLdC = gain*(g_l1+g_ss)
    behind = np.where((1-a0)[:, :, None] > 1e-6,
                      (C - a0[:, :, None]*c0)/np.maximum((1-a0)[:, :, None], 1e-6), 0.0)
    dBehind = np.where(1-a0 > 1e-6, (D - a0*d0)/np.maximum(1-a0, 1e-6), 0.0)
    live = a0 > 0
    ph_all = np.median(np.abs(((c0-behind)*dLdC).sum(2))[live])
    dep_unit = np.median(np.abs((d0-dBehind)/np.maximum(alpha, 1e-4))[live])

    print('\n  %-12s %14s %14s %16s %14s' %
          ('span (m)', 'peak coeff', 'peak vs L1+SSIM', 'P(t in (0,1))', 'E[|g|] vs L1+SSIM'))
    tw = DEPTH_SCALE*1.0*invS*TRANS_W       # s.weight 0.5 x edge boost 2 = 1.0
    for span in (0.30, 0.04, 0.01, 0.001, 1e-4):
        coeff = tw*4.0/span
        pfire = min(pbar_peak*span, 1.0)
        eabs = tw*4.0*0.5*pbar_peak         # = 2*tw*pbar, span-independent
        print('  %-12.5f %14.4e %14.1f %16.6f %14.2f'
              % (span, coeff, coeff*dep_unit/ph_all, pfire, eabs*dep_unit/ph_all))
    print('\n  E[|g|] = 2 * tw * p_bar = %.4e is INDEPENDENT of span; the signed'
          % (2*tw*pbar_peak))
    print('  expectation is exactly 0 because (1-2t) is antisymmetric on (0,1).')
    print('  For comparison the Huber at the same s.weight is %.4e (bounded by w).'
          % (DEPTH_SCALE*1.0*invS))
    print('  photometric |dL/dalpha| p50 = %.4e, depth lever p50 = %.4f' % (ph_all, dep_unit))
