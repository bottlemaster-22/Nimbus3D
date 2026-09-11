"""Replica of build 276's keyframe selection with the FIXED held-out set.

Mirrors MetalSplatTrainer.selectKeyframes as of 276: look-ahead 0, pool
qc.weight > 0.05, held-out candidates index % stride == stride // 2 removed
from the walk, trainTarget = target - round(target * heldOutFraction),
spacing = walk path / trainTarget, and held-out = candidates inside the
[min, max] index of the chosen keyframes. Float32 like the device.

Run before reading a 276+ census: it prints the exact held_out_frames.json
and keyframe span the device should report (as long as the refined poses in
prepass_result.json are the ones the run used).
"""
import numpy as np
import kf_exact as K

f32 = np.float32


def select_fixed(target=120, stride=40, fraction=0.10, turn=0.02):
    target = max(target, 8)
    pool = K.POOL
    if len(pool) <= target:
        return [f['index'] for f in pool], None
    cand = {f['index'] for f in pool if stride > 1 and f['index'] % stride == stride // 2}
    walk = [f for f in pool if f['index'] not in cand] if cand else pool
    share = int(round(float(f32(target) * f32(fraction)))) if cand else 0
    train_target = max(target - share, 8)
    CC = np.array([K.pose_of(f)[0] for f in walk], f32)
    FF = np.array([K.pose_of(f)[1] for f in walk], f32)
    L = f32(0)
    for i in range(1, len(walk)):
        L = f32(L + f32(np.linalg.norm((CC[i - 1] - CC[i]).astype(f32))))
    sp = f32(L / f32(train_target)) if L > 0 else f32(0)
    turn = f32(turn)
    chosen = []
    lastC = lastF = None
    for i in range(len(walk)):
        if lastC is not None:
            moved = f32(np.linalg.norm((lastC - CC[i]).astype(f32)))
            turned = f32(f32(1) - f32(np.dot(K._nz(lastF), K._nz(FF[i]))))
            if moved < sp and turned < turn:
                continue
        chosen.append(i)
        lastC, lastF = CC[i], FF[i]
        if len(chosen) >= train_target:
            break
    if len(chosen) < min(train_target, len(walk)):
        step = max(len(walk) // train_target, 1)
        chosen = list(range(0, len(walk), step))
    kf = [walk[i]['index'] for i in chosen]
    if not cand:
        return kf, None
    lo, hi = min(kf), max(kf)
    held = [f['index'] for f in pool if f['index'] in cand and lo <= f['index'] <= hi]
    return kf, held or None


if __name__ == '__main__':
    kf, held = select_fixed()
    print('trained keyframes %d  distinct %d  span %d..%d' % (len(kf), len(set(kf)), min(kf), max(kf)))
    print('held out (fixed) %d : %s' % (len(held or []), held))
    print('overlap trained & held-out:', sorted(set(kf) & set(held or [])))
