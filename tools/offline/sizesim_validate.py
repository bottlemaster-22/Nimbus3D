"""Validate the size simulator against build 182's OWN densify census.

If the simulator's schedule arithmetic is right it must reproduce, without
being told them, the three counts the census recorded: 21,000 relocations,
6,836 growth splits, 64 clones - and, more to the point, WHEN each mechanism
switched off.
"""
import json
import numpy as np
import sizesim

CENSUS = (r'C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics'
          r'\scan_20260906_164840\model\train_census.json')


def measured():
    c = json.load(open(CENSUS))
    p = c['densifyPasses']
    return p


def main():
    p = measured()
    tot = dict(
        reloc=sum(r['relocated'] for r in p),
        split=sum(r['addedBySplit'] for r in p),
        clone=sum(r['addedByClone'] for r in p),
        prune=sum(r['prunedLowOpacity'] for r in p),
    )
    print('MEASURED build 182 census over 29 passes:')
    print('  relocations %d   growth splits %d   clones %d   pruned %d'
          % (tot['reloc'], tot['split'], tot['clone'], tot['prune']))
    reloc_passes = [r['iteration'] for r in p if r['relocated'] > 0]
    growth_passes = [r['iteration'] for r in p
                     if r['addedBySplit'] + r['addedByClone'] > 0]
    print('  relocation fired on iterations:', reloc_passes)
    print('  growth fired on iterations:', growth_passes[:4], '...', growth_passes[-2:])
    print('  headroom at every pass:', sorted(set(r['headroom'] for r in p))[:6], '...')
    print('  split share of growth actually observed: %.2f%%'
          % (100.0 * tot['split'] / max(tot['split'] + tot['clone'], 1)))

    # Drive the simulator with build 182's own per-pass prune counts, so the
    # only thing it is being asked to get right is the schedule arithmetic.
    prune_series = {r['iteration']: r['prunedLowOpacity'] for r in p}
    cfg = sizesim.Cfg().clone(seed_at_cap=True)
    s, tally, hist = _run_with_prune_series(cfg, prune_series)
    print('\nSIMULATED with build 182 seeding (300k = at the cap) and its own'
          ' measured per-pass prune counts:')
    print('  relocations %d   growth splits %d   clones %d'
          % (tally['reloc'], tally['split'], tally['clone']))
    print('  split share of growth: %.2f%%'
          % (100.0 * tally['split'] / max(tally['split'] + tally['clone'], 1)))
    print('  relocation fired on iterations:', tally['reloc_iters'])
    print('  growth fired on iterations:', tally['growth_iters'][:4], '...',
          tally['growth_iters'][-2:])


def _run_with_prune_series(cfg, prune_series):
    """sizesim.run, but the prune count each pass is the measured one."""
    rng = np.random.default_rng(cfg.rng_seed)
    s, edge = sizesim.make_seeds(cfg, rng)
    latent = rng.random(len(s))
    pop = len(s)
    k = np.log(cfg.split_shrink)
    split_thresh = sizesim.SCENE_EXTENT_M * cfg.split_scale_fraction
    tally = dict(split=0, clone=0, reloc=0, pruned=0,
                 reloc_iters=[], growth_iters=[])
    hist = []
    for it in range(cfg.interval, cfg.iterations + 1, cfg.interval):
        frac = it / float(cfg.iterations)
        allow_growth = cfg.densify_start <= frac <= cfg.densify_end
        headroom = max(cfg.cap - pop, 0)
        allowance = min(headroom, int(cfg.cap * cfg.growth_frac)) if allow_growth else 0
        reloc_limit = int(pop * cfg.reloc_frac) if (allow_growth and allowance == 0) else 0
        limit = max(allowance, reloc_limit)
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
            is_split = (np.exp(big) > split_thresh) | (big >= cut)
            sp, cl = take[is_split], take[~is_split]
            if cfg.split_shrink_all_axes:
                s[sp] -= k
            else:
                ax = sizesim.split_axis(s[sp], edge[sp])
                s[sp, ax] -= k
            new = np.concatenate([s[sp].copy(), s[cl].copy()], axis=0)
            s = np.concatenate([s, new], axis=0)
            edge = np.concatenate([edge, edge[sp], edge[cl]])
            latent = np.concatenate([latent, latent[sp], latent[cl]])
            pop = len(s)
            tally['split'] += len(sp)
            tally['clone'] += len(cl)
            tally['growth_iters'].append(it)
        elif reloc_limit > 0 and cand is not None:
            # Build 182 ran relocationDonorOpacity == pruneOpacity == 0.02, so
            # the faint band was empty by construction and every donor came
            # from `visAccum <= 0`. Use the census's own donor counts.
            donors = {300: 18222, 400: 4833, 500: 1167}.get(it, 0)
            npair = min(donors, reloc_limit, len(cand))
            if npair > 0:
                target = cand[:npair]
                pool = rng.choice(pop, size=min(donors, pop), replace=False)
                donor = np.setdiff1d(pool, target)[:npair]
                npair = len(donor)
                target = target[:npair]
                ax = sizesim.split_axis(s[target], edge[target])
                s[target, ax] -= k
                s[donor] = s[target]
                edge[donor] = edge[target]
                tally['reloc'] += npair
                tally['reloc_iters'].append(it)
        nkill = min(prune_series.get(it, 0), pop - 1)
        if nkill > 0:
            kill = rng.choice(pop, size=nkill, replace=False)
            keep = np.ones(pop, bool)
            keep[kill] = False
            s, edge, latent = s[keep], edge[keep], latent[keep]
            pop = len(s)
            tally['pruned'] += nkill
        hist.append((it, pop))
    return s, tally, hist


if __name__ == '__main__':
    main()
