"""The keyframe sets kf_price.py prices, regenerated without the projection
(no model load), for kf_tiles.py and for printing exact expected lists."""
import numpy as np
import kf_exact as K


def calibrate(N, param, lo, hi, **kw):
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        n = len(K.select(N, nobreak=True, **{param: mid}, **kw)[0])
        if n >= N:
            lo = mid
        else:
            hi = mid
    return lo


def sets():
    yield '266 shipped N=120', K.select(120)[0]
    yield '264 replica (blur key)', K.select(120, key=-K.MB)[0]
    yield 'lookahead 0 N=120', K.select(120, lookahead=0)[0]
    for N in [150, 180, 204, 240]:
        yield 'shipped code N=%d' % N, K.select(N)[0]
    for N in [120, 150, 204]:
        yield 'resume only N=%d' % N, K.select(N, resume_after_best=True)[0]
    sp120 = float(K.PATH / K.f32(120))
    yield 'resume, gates of 120, whole walk', K.select(120, spacing=sp120, nobreak=True, resume_after_best=True)[0]
    for N in [120, 150, 180, 204, 227]:
        s = calibrate(N, 'spacing_scale', 0.02, 20.0, resume_after_best=True)
        yield 'resume+spacing x%.3f N=%d' % (s, N), K.select(N, spacing_scale=s, resume_after_best=True)[0]
    for N in [120, 150, 180, 204]:
        t = calibrate(N, 'turn', 1e-4, 2.0, resume_after_best=True)
        yield 'resume+turn %.4f N=%d' % (t, N), K.select(N, turn=t, resume_after_best=True)[0]
    t = calibrate(120, 'turn', 1e-4, 2.0)
    yield 'shipped+turn %.4f N=120' % t, K.select(120, turn=t)[0]
    for N in [120, 150, 180, 204]:
        tg = calibrate(N, 'time_gate', 0.01, 60.0, resume_after_best=True, turn=10.0)
        yield 'resume+dist|time %.2fs N=%d' % (tg, N), K.select(N, turn=10.0, time_gate=tg, resume_after_best=True)[0]
