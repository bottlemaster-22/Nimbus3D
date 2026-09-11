"""What the size distribution DOES under TrainerDensifier's rules, arithmetically.

Photometry is not modelled and does not need to be: the question is how many
times a Gaussian's scale vector gets multiplied by 1/splitShrink, and by which
mechanism, under a schedule whose every number is fixed by TrainerSupport and
MetalSplatTrainer. Everything the optimiser does to logScale is OUTSIDE this
model and is stated as such wherever it matters.

THE RULES, transcribed from TrainerDensifier.swift:

  growthAllowance   = allowGrowth ? min(cap - pop, cap * maxGrowthFractionPerPass) : 0
  relocationLimit   = (allowGrowth && growthAllowance == 0)
                          ? pop * maxRelocationFractionPerPass : 0

  GROWTH (growthAllowance > 0), over the top `growthAllowance` candidates:
    split  if largestLinear > sceneExtent*splitScaleFraction
           or logLargest >= splitScaleCut  (the (1-splitShareOfGrowth) quantile
                                            of THIS pass's candidate scales)
      -> parent slot AND one new slot both get scale - ln(splitShrink) on ALL
         THREE axes (splitShrinkAllAxes true, preserveCoverage false)
    else clone
      -> one new slot at the parent's EXACT scale

  RELOCATION (at the cap), over min(donors, candidates, relocationLimit) pairs:
    -> target slot gets scale - ln(splitShrink) on ONE axis
    -> donor slot is OVERWRITTEN with the target's post-shrink scale
       (preserveCoverage true, so one axis only)

  splitAxis = argmax(linear scale), or the SECOND largest when onEdgeCurve.

Seeds: measured. seed_dist_measure.py shows the pre-pass gives every seed the
same largest axis, 7.397 mm, for 100 percent of this scan's samples.
"""
import numpy as np

LN = np.log

# ---------------------------------------------------------------- the scan
SEED_RADIUS_M = 0.0073966          # measured: 0.5 * spacingMeters (floor wins 100%)
MEDIAN_SIGMA_M = 0.009325768       # prepass census seeding.medianSigmaMeters
TRUSTED_FRACTION = 421878 / 888951  # prepass census
EDGE_FRACTION = 69029 / 888951      # prepass census onEdgeCount
SCENE_EXTENT_M = 7.6996             # capture_bundle sceneBounds longest edge


class Cfg(object):
    iterations = 3000
    interval = 100
    densify_start = 0.10
    densify_end = 0.85
    cap = 300000
    seeds = 150000                 # seedTarget(cap, fillFraction 0.5)
    growth_frac = 0.15
    reloc_frac = 0.05
    split_share = 0.20
    split_shrink = 1.6
    split_shrink_all_axes = True
    split_scale_fraction = 0.004
    # Donor pool as a fraction of the population, per pass. MEASURED on the
    # finished model: 7.354% sit below relocationDonorOpacity 0.05. Treated as
    # a steady state; the replenishment rate is NOT measurable from the data on
    # disk and is swept separately.
    donor_frac = 0.07354
    # Splats removed per prune pass, as a fraction of population. MEASURED
    # from build 182's census: 44..385 prunedLowOpacity per pass on a ~300k
    # population, mean 232 -> 0.00077.
    prune_frac = 0.00077
    prune_start = 0.15
    prune_end = 1.0
    # How much of a Gaussian's densification score persists between passes.
    # 0 = a fresh random ranking every pass (shrinks spread out thinly),
    # 1 = the same Gaussians are picked every pass (shrinks concentrate).
    # NOT measurable from anything on disk; swept.
    score_persistence = 0.5
    seed_at_cap = False            # True reproduces build 182's seeding
    rng_seed = 7

    def clone(self, **kw):
        c = Cfg()
        for k in dir(self):
            if not k.startswith('_') and k != 'clone':
                setattr(c, k, getattr(self, k))
        for k, v in kw.items():
            if not hasattr(c, k):
                raise KeyError(k)
            setattr(c, k, v)
        return c


def make_seeds(cfg, rng):
    n = cfg.cap if cfg.seed_at_cap else cfg.seeds
    r = SEED_RADIUS_M
    trusted = rng.random(n) < TRUSTED_FRACTION
    # thirdScale = clamp(sigma*k, 0.10r, capFraction*r); with medianSigma
    # 9.33 mm against r 7.40 mm the UPPER clamp binds for both branches.
    t_third = min(max(MEDIAN_SIGMA_M * 1.0, 0.10 * r), 0.35 * r)
    d_third = min(max(MEDIAN_SIGMA_M * 2.5, 0.10 * r), 0.70 * r)
    third = np.where(trusted, t_third, d_third)
    s = np.empty((n, 3))
    s[:, 0] = LN(r)
    s[:, 1] = LN(r)
    s[:, 2] = LN(third)
    edge = rng.random(n) < EDGE_FRACTION
    return s, edge


def split_axis(sub, edge_sub):
    """argmax of the linear scale, or the second largest when on an edge."""
    order = np.argsort(-sub, axis=1)
    return np.where(edge_sub, order[:, 1], order[:, 0])


def run(cfg, verbose=False):
    rng = np.random.default_rng(cfg.rng_seed)
    s, edge = make_seeds(cfg, rng)
    latent = rng.random(len(s))           # the persistent half of the score
    pop = len(s)
    k = LN(cfg.split_shrink)
    split_thresh = SCENE_EXTENT_M * cfg.split_scale_fraction
    tally = dict(split=0, clone=0, reloc=0, pruned=0, passes=0,
                 growth_passes=0, reloc_passes=0, donor_bound=0, limit_bound=0)
    history = []

    for it in range(cfg.interval, cfg.iterations + 1, cfg.interval):
        frac = it / float(cfg.iterations)
        allow_growth = cfg.densify_start <= frac <= cfg.densify_end
        allow_prune = cfg.prune_start <= frac <= cfg.prune_end
        tally['passes'] += 1

        headroom = max(cfg.cap - pop, 0)
        allowance = min(headroom, int(cfg.cap * cfg.growth_frac)) if allow_growth else 0
        reloc_limit = int(pop * cfg.reloc_frac) if (allow_growth and allowance == 0) else 0

        cand = None
        limit = max(allowance, reloc_limit)
        if limit > 0:
            score = (cfg.score_persistence * latent
                     + (1 - cfg.score_persistence) * rng.random(pop))
            take_n = min(limit, pop)
            idx = np.argpartition(-score, take_n - 1)[:take_n]
            cand = idx[np.argsort(-score[idx])]

        if allowance > 0 and cand is not None:
            take = cand[:allowance]
            big = s[take].max(axis=1)
            if cfg.split_share > 0:
                cut = np.quantile(big, 1 - cfg.split_share)
            else:
                cut = np.inf
            is_split = (np.exp(big) > split_thresh) | (big >= cut)
            sp, cl = take[is_split], take[~is_split]
            if cfg.split_shrink_all_axes:
                s[sp] -= k
            else:
                ax = split_axis(s[sp], edge[sp])
                s[sp, ax] -= k
            child = s[sp].copy()
            new = np.concatenate([child, s[cl].copy()], axis=0)
            new_edge = np.concatenate([edge[sp], edge[cl]])
            new_lat = np.concatenate([latent[sp], latent[cl]])
            s = np.concatenate([s, new], axis=0)
            edge = np.concatenate([edge, new_edge])
            latent = np.concatenate([latent, new_lat])
            pop = len(s)
            tally['split'] += len(sp)
            tally['clone'] += len(cl)
            tally['growth_passes'] += 1

        elif reloc_limit > 0 and cand is not None:
            donors_avail = int(pop * cfg.donor_frac)
            npair = min(donors_avail, reloc_limit, len(cand))
            if donors_avail < reloc_limit:
                tally['donor_bound'] += 1
            else:
                tally['limit_bound'] += 1
            if npair > 0:
                target = cand[:npair]
                pool = rng.choice(pop, size=min(donors_avail, pop), replace=False)
                donor = np.setdiff1d(pool, target, assume_unique=False)
                donor = donor[:npair]
                npair = len(donor)
                target = target[:npair]
                ax = split_axis(s[target], edge[target])
                s[target, ax] -= k
                s[donor] = s[target]           # donor slot is OVERWRITTEN
                edge[donor] = edge[target]
                latent[donor] = latent[target]
                tally['reloc'] += npair
                tally['reloc_passes'] += 1

        if allow_prune and cfg.prune_frac > 0:
            nkill = int(pop * cfg.prune_frac)
            if nkill > 0:
                kill = rng.choice(pop, size=nkill, replace=False)
                keep = np.ones(pop, bool)
                keep[kill] = False
                s, edge, latent = s[keep], edge[keep], latent[keep]
                pop = len(s)
                tally['pruned'] += nkill

        big_mm = 1000 * np.exp(s.max(axis=1))
        p10, p50, p90 = np.percentile(big_mm, [10, 50, 90])
        history.append(dict(it=it, pop=pop, p10=p10, p50=p50, p90=p90,
                            alw=allowance, rl=reloc_limit,
                            spread=p90 / p10,
                            sd=float(np.std(np.log10(big_mm)))))
        if verbose:
            print('  it %4d pop %7d alw %6d rl %6d  p10 %7.3f p50 %7.3f p90 %7.3f'
                  '  spread %.3fx sd %.4f'
                  % (it, pop, allowance, reloc_limit, p10, p50, p90,
                     p90 / p10, history[-1]['sd']))
    return s, tally, history


def summarise(name, cfg):
    s, tally, hist = run(cfg)
    big = 1000 * np.exp(s.max(axis=1))
    p10, p50, p90 = np.percentile(big, [10, 50, 90])
    return dict(name=name, pop=hist[-1]['pop'], p10=p10, p50=p50, p90=p90,
                spread=p90 / p10, sd=float(np.std(np.log10(big))),
                split=tally['split'], clone=tally['clone'],
                reloc=tally['reloc'], tally=tally, hist=hist)
