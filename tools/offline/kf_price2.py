"""Follow-up to kf_price.py: look-ahead OFF (tuning.keyframeSharpnessLookahead
= 0) at counts that stay under the 128-frame authority/edge caches, plus the
motion-blur price of dropping the look-ahead."""
import numpy as np
import kf_exact as K
import kf_price as KP


def blur(name, ch):
    kf = [int(K.IDX[b]) for b, _, _ in ch]
    tr, _ = K.split_held_out(kf)
    pos = {int(i): j for j, i in enumerate(K.IDX)}
    mb = np.array([K.MB[pos[k]] for k in tr]) * 720.0 / 1920.0
    w = np.array([K.W[pos[k]] for k in tr])
    print('   %-30s train entries render-blur p50 %.2f p90 %.2f px | >1px %3d/%d | qc.weight mean %.3f'
          % (name, np.percentile(mb, 50), np.percentile(mb, 90), int((mb > 1).sum()), len(tr), w.mean()))


if __name__ == '__main__':
    blur('266 shipped', K.select(120)[0])
    for N in [120, 130, 140, 150]:
        blur('lookahead 0 N=%d' % N, K.select(N, lookahead=0)[0])
    for N in [130, 140, 150]:
        KP.price('lookahead 0 N=%d' % N, K.select(N, lookahead=0)[0])
    print('done')
