"""The size-distribution simulator, v2: calibrated, then swept.

Two things v1 was missing, both of which the census proves matter.

1. THE SCREEN-RADIUS SPLIT PATH. Build 182 ran splitScreenRadiusPx = 10, and
   TrainerSupport's own withdrawal note measures that criterion selecting
   99.18 per cent of the drawn population. The census recorded 6,836 splits
   against 64 clones, 99.07 per cent. Those two numbers agreeing to 0.1 per
   cent is what identifies which term did the selecting. Current head has it
   at 0, so the split share falls back to splitShareOfGrowth = 0.20.

2. THE OPTIMISER. logScale is a trained parameter and Adam moves it every
   iteration. Densification cannot be the whole story, and pretending it is
   would put every percentile on an exact 1.6 ladder. The term is CALIBRATED,
   not invented: build 182 seeded at 12.734 mm (7.397 mm times the 1.7213
   thinning compensation it shipped) and its exported model has median
   largest axis 19.567 mm and sd(log10) 0.1285 decades. The densification-only
   run gives one median and one sd; the difference is what Adam did, split
   into a per-pass drift and a per-pass diffusion.

Everything else is transcribed from TrainerDensifier.swift.
"""
import json
import numpy as np

LN = np.log
LOG10_16 = np.log10(1.6)

SEED_RADIUS_M = 0.0073966
MEDIAN_SIGMA_M = 0.009325768
TRUSTED_FRACTION = 421878 / 888951
EDGE_FRACTION = 69029 / 888951
SCENE_EXTENT_M = 7.6996

CENSUS = (r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics'
          r'\scan_20260906_164840\model\train_census.json')

# Measured on the exported model by sizedist_measure.py.
MEASURED = dict(p10=14.090, p50=19.567, p90=24.832, sd=0.12846, spread=1.7624)
TARGET = dict(p10=1.327, p50=3.388, p90=11.825, sd=0.63612, spread=8.913)


class Cfg(object):
    iterations = 3000
    interval = 100
    densify_start = 0.10
    densify_end = 0.85
    cap = 300000
    seeds = 150000
    seed_multiplier = 1.0          # seedSpacingCompensation; build 182 ran 1.7213
    growth_frac = 0.15
    reloc_frac = 0.05
    split_share = 0.20
    screen_split_fraction = 0.0    # splitScreenRadiusPx; 0 = off (current head)
    split_shrink = 1.6
    split_shrink_all_axes = True
    split_scale_fraction = 0.004
    donor_frac = 0.07354
    prune_frac = 0.00077
    prune_start = 0.15
    prune_end = 1.0
    score_persistence = 0.5
    # Make the relocation branch an `if` instead of the `else` of the growth
    # branch, so it runs on EVERY pass in the window rather than only on the
    # passes where headroom happens to be exactly zero.
    reloc_always = False
    # Calibrated optimiser term, PER ITERATION, in decades of log10 largest
    # axis. Set by calibrate(). Per iteration and not per pass on purpose:
    # Adam runs every iteration, so halving densifyIntervalIterations must not
    # silently double how much the optimiser is credited with doing.
    opt_drift = 0.0
    opt_diffuse = 0.0
    rng_seed = 7

    def clone(self, **kw):
        c = Cfg()
        for k in dir(self):
            if not k.startswith('_') and k != 'clone':
                setattr(c, k, getattr(self, k))
        for k, v in kw.items():
            if not hasattr(c, k):
                raise KeyError('no such setting: ' + k)
            setattr(c, k, v)
        return c


def make_seeds(cfg, rng):
    n = cfg.seeds
    r = SEED_RADIUS_M * cfg.seed_multiplier
    trusted = rng.random(n) < TRUSTED_FRACTION
    t_third = min(max(MEDIAN_SIGMA_M, 0.10 * r), 0.35 * r)
    d_third = min(max(MEDIAN_SIGMA_M * 2.5, 0.10 * r), 0.70 * r)
    third = np.where(trusted, t_third, d_third)
    s = np.empty((n, 3))
    s[:, 0] = LN(r)
    s[:, 1] = LN(r)
    s[:, 2] = LN(third)
    return s, rng.random(n) < EDGE_FRACTION


def _axis(sub, edge_sub):
    order = np.argsort(-sub, axis=1)
    return np.where(edge_sub, order[:, 1], order[:, 0])


def run(cfg, verbose=False, prune_series=None, donor_series=None):
    rng = np.random.default_rng(cfg.rng_seed)
    s, edge = make_seeds(cfg, rng)
    latent = rng.random(len(s))
    nshrink = np.zeros(len(s), np.int32)   # exact count of shrinks per slot
    pop = len(s)
    k = LN(cfg.split_shrink)
    thresh = SCENE_EXTENT_M * cfg.split_scale_fraction
    tally = dict(split=0, clone=0, reloc=0, pruned=0, passes=0,
                 reloc_iters=[], growth_iters=[])
    hist = []
    for it in range(cfg.interval, cfg.iterations + 1, cfg.interval):
        frac = it / float(cfg.iterations)
        grow_ok = cfg.densify_start <= frac <= cfg.densify_end
        prune_ok = cfg.prune_start <= frac <= cfg.prune_end
        tally['passes'] += 1
        headroom = max(cfg.cap - pop, 0)
        allowance = min(headroom, int(cfg.cap * cfg.growth_frac)) if grow_ok else 0
        if grow_ok and (cfg.reloc_always or allowance == 0):
            rlimit = int(pop * cfg.reloc_frac)
        else:
            rlimit = 0
        limit = max(allowance, rlimit)
        cand = None
        if limit > 0:
            score = (cfg.score_persistence * latent
                     + (1 - cfg.score_persistence) * rng.random(pop))
            tn = min(limit, pop)
            idx = np.argpartition(-score, tn - 1)[:tn]
            cand = idx[np.argsort(-score[idx])]

        if allowance > 0 and cand is not None:
            take = cand[:allowance]
            big = s[take].max(axis=1)
            cut = np.quantile(big, 1 - cfg.split_share) if cfg.split_share > 0 else np.inf
            is_split = (np.exp(big) > thresh) | (big >= cut)
            if cfg.screen_split_fraction > 0:
                is_split |= rng.random(len(take)) < cfg.screen_split_fraction
            sp, cl = take[is_split], take[~is_split]
            if cfg.split_shrink_all_axes:
                s[sp] -= k
                nshrink[sp] += 1
            else:
                s[sp, _axis(s[sp], edge[sp])] -= k
                nshrink[sp] += 1
            new = np.concatenate([s[sp].copy(), s[cl].copy()], axis=0)
            s = np.concatenate([s, new], axis=0)
            edge = np.concatenate([edge, edge[sp], edge[cl]])
            latent = np.concatenate([latent, latent[sp], latent[cl]])
            nshrink = np.concatenate([nshrink, nshrink[sp], nshrink[cl]])
            pop = len(s)
            tally['split'] += len(sp)
            tally['clone'] += len(cl)
            tally['growth_iters'].append(it)
        if (rlimit > 0 and cand is not None
                and (cfg.reloc_always or allowance == 0)):
            if donor_series is not None:
                davail = donor_series.get(it, 0)
            else:
                davail = int(pop * cfg.donor_frac)
            npair = min(davail, rlimit, len(cand))
            if npair > 0:
                target = cand[:npair]
                pool = rng.choice(pop, size=min(davail, pop), replace=False)
                donor = np.setdiff1d(pool, target)[:npair]
                npair = len(donor)
                target = target[:npair]
                s[target, _axis(s[target], edge[target])] -= k
                nshrink[target] += 1
                s[donor] = s[target]
                edge[donor] = edge[target]
                latent[donor] = latent[target]
                nshrink[donor] = nshrink[target]
                tally['reloc'] += npair
                tally['reloc_iters'].append(it)

        if prune_series is not None:
            nkill = min(prune_series.get(it, 0), pop - 1)
        else:
            nkill = int(pop * cfg.prune_frac) if prune_ok else 0
        if nkill > 0:
            kill = rng.choice(pop, size=nkill, replace=False)
            keep = np.ones(pop, bool)
            keep[kill] = False
            s, edge, latent = s[keep], edge[keep], latent[keep]
            nshrink = nshrink[keep]
            pop = len(s)
            tally['pruned'] += nkill

        # The optimiser, in decades of log10 on all three axes. Drift is
        # linear in the iterations since the last pass; diffusion is a random
        # walk, so it goes as the square root of them.
        if cfg.opt_drift or cfg.opt_diffuse:
            step = (cfg.opt_drift * cfg.interval
                    + cfg.opt_diffuse * np.sqrt(cfg.interval)
                    * rng.standard_normal((pop, 1)))
            s += step * np.log(10.0)

        big_mm = 1000 * np.exp(s.max(axis=1))
        p10, p50, p90 = np.percentile(big_mm, [10, 50, 90])
        hist.append(dict(it=it, pop=pop, alw=allowance, rl=rlimit,
                         p10=p10, p50=p50, p90=p90, spread=p90 / p10,
                         sd=float(np.std(np.log10(big_mm)))))
        if verbose:
            print('  it %4d pop %7d alw %6d rl %6d  p10 %7.3f p50 %7.3f '
                  'p90 %7.3f  spread %6.3fx sd %.4f'
                  % (it, pop, allowance, rlimit, p10, p50, p90,
                     p90 / p10, hist[-1]['sd']))
    tally['nshrink'] = nshrink
    return s, tally, hist


def stats(s):
    big = 1000 * np.exp(s.max(axis=1))
    p10, p50, p90 = np.percentile(big, [10, 50, 90])
    return dict(p10=p10, p50=p50, p90=p90, spread=p90 / p10,
                sd=float(np.std(np.log10(big))), n=len(big))


def build182(**kw):
    """Build 182 exactly: seeded AT the cap with the 1.7213 compensation, the
    screen-radius split criterion on, and its own measured prune and donor
    series."""
    c = Cfg().clone(seeds=300000, seed_multiplier=1.7213,
                    screen_split_fraction=0.9918, **kw)
    p = json.load(open(CENSUS))['densifyPasses']
    prune = {r['iteration']: (r['prunedLowOpacity'] + r['prunedOversized']
                              + r['prunedNonFinite'] + r['carvedFromEmptySpace']
                              + r['trimmedToCap']) for r in p}
    donors = {r['iteration']: r['relocationDonorsAvailable'] for r in p}
    return c, prune, donors
