"""Monotonic look-ahead: shipped selector, but the qc.weight window starts
after the last chosen frame, so no frame is picked twice and no held-out frame
also trains. Priced at 120-150 with blur, views, tiles and cache builds."""
import random
from collections import OrderedDict, Counter
import numpy as np
import kf_exact as K
import kf_price as KP
from kf_price2 import blur

T = np.load(r'C:\Users\Undea\Documents\TOMBLINE\Nimbus3D\tools\offline\_kf_tiles.npy')
warm = [f['index'] for f in K.FRAMES[::7] if f['qc']['weight'] > 0.05]


def builds(train, held, iters=4000, cap=128):
    c = OrderedDict(); n = 0
    def get(k):
        nonlocal n
        if k in c: return
        n += 1; c[k] = 1
        while len(c) > cap: c.popitem(last=False)
    for k in warm: get(k)
    order = list(range(len(train))); random.Random(0).shuffle(order)
    for it in range(iters):
        get(train[order[it % len(order)]])
        if it >= 700 and it % 400 == 0:      # held-out evals request their frames too
            for k in held: get(k)
    return n


base_t = None
for name, ch in [('266 shipped N=120', K.select(120)[0])] + \
        [('monotonic N=%d' % N, K.select(N, monotonic=True)[0]) for N in (120, 130, 140, 150)]:
    kf = [int(K.IDX[b]) for b, _, _ in ch]
    tr, ho = K.split_held_out(kf)
    t = np.mean([T[k] for k in tr]); base_t = base_t or t
    print('%s: reasons %s | fallback %s | authority builds %d | tiles/entry %+.1f%% | held-out %s'
          % (name, dict(Counter(w for _, w, _ in ch)), any(w == 'stride' for _, w, _ in ch),
             builds(tr, ho), 100 * (t / base_t - 1), ho))
    blur(name, ch)
    if name != '266 shipped N=120':
        KP.price(name, ch)
print('done')
