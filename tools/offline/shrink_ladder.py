"""How many splats ever get shrunk at all, and how many times.

With the optimiser term switched off every splat's largest axis sits on an
exact ladder of 1/splitShrink steps below the seed, so the rung it is on IS
the number of three-axis shrinks it has received. That histogram is the whole
mechanism in one table.
"""
import numpy as np
import sizesim2 as S


def ladder(name, cfg):
    s, t, h = S.run(cfg)
    big = np.exp(s.max(axis=1))
    rung = np.round(np.log(S.SEED_RADIUS_M * cfg.seed_multiplier / big)
                    / np.log(cfg.split_shrink)).astype(int)
    n = len(rung)
    print('%s   population %d   split %d  clone %d  reloc %d'
          % (name, n, t['split'], t['clone'], t['reloc']))
    lo, hi = rung.min(), min(rung.max(), 12)
    for r in range(lo, hi + 1):
        c = int((rung == r).sum())
        if c:
            print('    %2d shrinks: %7d  %6.2f%%   size %7.3f mm'
                  % (r, c, 100.0 * c / n,
                     1000 * S.SEED_RADIUS_M * cfg.seed_multiplier
                     / cfg.split_shrink ** r))
    print('    never shrunk: %6.2f%%   mean shrinks per splat: %.3f\n'
          % (100.0 * (rung <= 0).mean(), rung.mean()))


def head(**kw):
    base = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
                opt_drift=0.0, opt_diffuse=0.0)
    base.update(kw)
    return S.Cfg().clone(**base)


if __name__ == '__main__':
    ladder('CURRENT HEAD', head())
    ladder('splitShare 1.0', head(split_share=1.0))
    ladder('relocation made unconditional', head(reloc_always=True))
    ladder('share 1.0 + reloc unconditional + shrink 2.0',
           head(split_share=1.0, reloc_always=True, split_shrink=2.0))
