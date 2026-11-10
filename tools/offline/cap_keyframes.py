"""Reconstruct the trainer's 120 selected keyframes from the bundle + prepass.

Mirrors MetalSplatTrainer.selectKeyframes and TrainerSlicePlanner.splitHeldOut.
Validated against model/held_out_frames.json, which the trainer wrote.
"""
import io, json, os
import numpy as np

D = r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840'


def quat_R(q):
    """q = (x,y,z,w) world->camera rotation. Returns 3x3 R."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def load_all():
    b = json.load(io.open(os.path.join(D, 'capture_bundle.json'), encoding='utf-8'))
    r = json.load(io.open(os.path.join(D, 'prepass', 'prepass_result.json'), encoding='utf-8'))
    refined = {int(k): v for k, v in r['refinedPoses'].items()}
    frames = sorted(b['frames'], key=lambda f: f['index'])
    out = []
    for f in frames:
        idx = f['index']
        p = refined.get(idx) or f.get('refinedPose') or f['rawPose']
        R = quat_R(p['rotation'])
        t = np.asarray(p['translation'], float)
        C = -R.T @ t                      # camera centre in world
        fwd = R.T @ np.array([0., 0., 1.])  # camera +Z forward, in world
        out.append(dict(index=idx, weight=f['qc']['weight'], C=C, F=fwd,
                        ts=f['timestampSeconds'], qc=f['qc']))
    return b, r, out


def select_keyframes(pool, target=120):
    """Exactly MetalSplatTrainer.selectKeyframes's greedy loop."""
    if len(pool) <= target:
        return list(range(len(pool))), None
    path = 0.0
    prev = None
    for fr in pool:
        if prev is not None:
            path += float(np.linalg.norm(fr['C'] - prev))
        prev = fr['C']
    spacing = path / target if path > 0 else 0.0
    chosen = []
    lastC = lastF = None
    for i, fr in enumerate(pool):
        if lastC is not None:
            moved = float(np.linalg.norm(lastC - fr['C']))
            turned = 1.0 - float(np.dot(lastF/np.linalg.norm(lastF),
                                        fr['F']/np.linalg.norm(fr['F'])))
            if moved < spacing and turned < 0.02:
                continue
        chosen.append(i)
        lastC, lastF = fr['C'], fr['F']
        if len(chosen) >= target:
            break
    return chosen, dict(pathLength=path, spacing=spacing)


def split_held_out(n, fraction=0.10):
    step = max(int(1.0/fraction), 2)
    held = [i for i in range(n) if i % step == step//2]
    train = [i for i in range(n) if i % step != step//2]
    return train, held


def keyframes():
    b, r, frames = load_all()
    usable = [f for f in frames if f['weight'] > 0.05]
    pool = usable if usable else frames
    sel, info = select_keyframes(pool, 120)
    kf = [pool[i] for i in sel]
    return b, r, frames, pool, kf, info


if __name__ == '__main__':
    b, r, frames, pool, kf, info = keyframes()
    print('frames in bundle      %d' % len(frames))
    print('usable (qc.weight>.05) %d' % len(pool))
    print('path length %.3f m   spacing %.4f m' % (info['pathLength'], info['spacing']))
    print('keyframes chosen      %d' % len(kf))
    idxs = [f['index'] for f in kf]
    print('first frame index %d   last frame index %d' % (idxs[0], idxs[-1]))
    tr, ho = split_held_out(len(kf))
    mine = [idxs[i] for i in ho]
    truth = json.load(io.open(os.path.join(D, 'model', 'held_out_frames.json'), encoding='utf-8'))
    print('held-out reconstructed %s' % mine)
    print('held-out on disk       %s' % truth)
    print('MATCH' if mine == sorted(truth) else '*** MISMATCH ***')
    print()
    print('keyframe frame indices:')
    print(idxs)
    ts = np.array([f['ts'] for f in kf])
    allts = np.array([f['ts'] for f in frames])
    print('capture spans %.1f s; keyframes span %.1f s (%.1f%% of the walk)'
          % (allts[-1]-allts[0], ts[-1]-ts[0], 100*(ts[-1]-ts[0])/(allts[-1]-allts[0])))
    print('frames after the last keyframe: %d of %d (%.1f%%)'
          % (len(frames)-1-frames.index(next(f for f in frames if f['index']==idxs[-1])),
             len(frames),
             100*(len(frames)-1-frames.index(next(f for f in frames if f['index']==idxs[-1])))/len(frames)))
