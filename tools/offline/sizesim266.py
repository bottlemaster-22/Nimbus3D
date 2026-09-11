"""A54 re-run on BUILD 266's schedule, with relocation alive.

What changed since sizesim2/sizesim_reloc, all transcribed from the current
code and the 266 census:
  - 4,000 iterations, passes every 100, growth window iterations 300..3400
    (densifyStartMaxIterations 300, densifyEndFraction 0.85).
  - Relocation runs whenever growth is underfed (growthAllowance < 45,000),
    ALONGSIDE growth; targets are the ranked candidates immediately behind the
    growth ones; limit 5% of the population; donors = the census's measured
    relocationDonorsAvailable per pass (donor-limited every pass on 266).
    Geometry: preserveCoverage, so ONE-axis shrink by splitShrink (largest
    axis, second-largest for onEdge splats), donor becomes a copy.
    Donors are taken from the pre-growth population (growth children are
    appended after relocation in TrainerDensifier).
  - Growth split: the largest splitShareOfGrowth of the truncated list by
    max log-scale (integer per-mille cut, as the code) OR world size >
    sceneExtent*0.004; all-axes shrink. Clone copies the scale.
  - Prunes: the census's per-pass counts (low opacity + carved + ...).
  - Seeds: 149,999 at start (census), THREE lattices: edge radius 3.70 mm,
    fine 7.40, coarse 14.79 (spacing 14.795 mm, radius = spacing/2).
    Edge share 0.243 = onEdgeCount/seeds (census). Coarse share is NOT on
    disk; 0.13 is derived from the 46% coarse-sample share at 4x cell area,
    and it is swept.

THE OPTIMISER IS THE UNKNOWN. Densification alone sets the size ladder; Adam
then moves every log-scale. Two structures, each calibrated to 266's measured
p50 and sd so the as-run row matches by construction on those two numbers:
  rw        drift + diffusion (sizesim2's model): densification's effect persists.
  ou:tau    mean reversion toward an attractor with time constant tau
            iterations: densification's effect decays.
Unfitted checks per model: p10, p90, event counts, and the sign of the
relocation-off -> on change (device: 250 -> 256, p50 7.83 -> 7.48 mm, spread
4.75x -> 5.00x; confounded with other changes in that batch).

Run: cd tools/offline && python -u sizesim266.py
"""
import io, json, os, sys
import numpy as np
import project as P

CEN = json.load(io.open(os.path.join(P.D, 'model', 'train_census.json'), encoding='utf-8'))
PASSES = {r['iteration']: r for r in CEN['densifyPasses']}
DONORS = {it: r['relocationDonorsAvailable'] for it, r in PASSES.items()}
PRUNE = {it: (r['prunedLowOpacity'] + r['prunedOversized'] + r['prunedNonFinite']
              + r['carvedFromEmptySpace'] + r['trimmedToCap']) for it, r in PASSES.items()}
EVENTS = dict(split=sum(r['addedBySplit'] for r in PASSES.values()),
              clone=sum(r['addedByClone'] for r in PASSES.values()),
              reloc=sum(r['relocated'] for r in PASSES.values()))

ITERS, INTERVAL, CAP, WANTED, RELOC_FRAC = 4000, 100, 300000, 45000, 0.05
GROW_START, GROW_END = 300, 3400
SPLIT_THRESH = 7.6996 * 0.004
SEEDS0 = 149999
SPACING = 0.01479527
R_FINE = 0.5 * SPACING
R_EDGE, R_COARSE = R_FINE / 2, R_FINE * 2
SIGMA = 0.012759566
TRUSTED = 375502 / 831975
LN10 = np.log(10.0)


def measured():
    col, n = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
    s = np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64)
    return stats(s)


def stats(s):
    L = np.log10(1000 * np.exp(s.max(1)))
    p10, p50, p90 = 10 ** np.percentile(L, [10, 50, 90])
    return dict(p10=p10, p50=p50, p90=p90, spread=p90 / p10, sd=float(L.std()), L=L)


def make_seeds(rng, edge_share, coarse_share):
    u = rng.random(SEEDS0)
    edge = u < edge_share
    r = np.where(edge, R_EDGE, np.where(u < edge_share + coarse_share, R_COARSE, R_FINE))
    trusted = rng.random(SEEDS0) < TRUSTED
    third = np.where(trusted, np.clip(SIGMA, 0.10 * r, 0.35 * r),
                     np.clip(2.5 * SIGMA, 0.10 * r, 0.70 * r))
    return np.log(np.stack([r, r, third], 1)), edge


def reloc_axis(sub, edge_sub):
    order = np.argsort(-sub, axis=1)
    return np.where(edge_sub, order[:, 1], order[:, 0])


def run(share=0.8, shrink=1.6, pers=0.5, opt=('rw', 0.0, 0.0), edge_share=0.243,
        coarse_share=0.13, seed=7, reloc=True):
    rng = np.random.default_rng(seed)
    s, edge = make_seeds(rng, edge_share, coarse_share)
    area0 = (np.exp(s[:, 0]) * np.exp(s[:, 1])).sum()
    latent = rng.random(len(s))
    pop = len(s)
    k = np.log(shrink)
    tally = dict(split=0, clone=0, reloc=0)
    kind = opt[0]
    for it in range(INTERVAL, ITERS + 1, INTERVAL):
        grow_ok = GROW_START <= it <= GROW_END
        headroom = max(CAP - pop, 0)
        allowance = min(headroom, WANTED) if grow_ok else 0
        rlimit = int(pop * RELOC_FRAC) if (grow_ok and allowance < WANTED and reloc) else 0
        limit = allowance + rlimit
        cand = None
        if limit > 0:
            score = pers * latent + (1 - pers) * rng.random(pop)
            tn = min(limit, pop)
            idx = np.argpartition(-score, tn - 1)[:tn]
            cand = idx[np.argsort(-score[idx])]
        oldpop = pop
        if allowance > 0:
            take = cand[:allowance]
            big = s[take].max(1)
            if share > 0:
                srt = np.sort(big)
                per_mille = int(min(max(share, 0), 1) * 1000)
                keep_clone = len(take) * (1000 - per_mille) // 1000
                cut = srt[max(0, min(len(take) - 1, keep_clone))]
            else:
                cut = np.inf
            is_split = (np.exp(big) > SPLIT_THRESH) | (big >= cut)
            sp, cl = take[is_split], take[~is_split]
            s[sp] -= k
            s = np.concatenate([s, s[sp], s[cl]], 0)
            edge = np.concatenate([edge, edge[sp], edge[cl]])
            latent = np.concatenate([latent, latent[sp], latent[cl]])
            pop = len(s)
            tally['split'] += len(sp); tally['clone'] += len(cl)
        if rlimit > 0 and cand is not None:
            targets = cand[allowance:allowance + rlimit]
            npair = min(DONORS.get(it, 0), rlimit, len(targets))
            if npair > 0:
                claimed = np.zeros(oldpop, bool)
                claimed[cand[:allowance + rlimit]] = True
                pool = np.nonzero(~claimed)[0]
                donors = rng.choice(pool, size=min(npair, pool.size), replace=False)
                tg = targets[:donors.size]
                ax = reloc_axis(s[tg], edge[tg])
                s[tg, ax] -= k
                s[donors] = s[tg]; edge[donors] = edge[tg]; latent[donors] = latent[tg]
                tally['reloc'] += donors.size
        nkill = min(PRUNE.get(it, 0), pop - 1)
        if nkill > 0:
            keep = np.ones(pop, bool)
            keep[rng.choice(pop, size=nkill, replace=False)] = False
            s, edge, latent = s[keep], edge[keep], latent[keep]
            pop = len(s)
        noise = rng.standard_normal((pop, 1))
        if kind == 'rw':
            step = opt[1] * INTERVAL + opt[2] * np.sqrt(INTERVAL) * noise
        else:                                   # ('ou', tau, attractor log10 mm, sigma)
            tau, a, sg = opt[1], opt[2], opt[3]
            L = np.log10(1000 * np.exp(s.max(1, keepdims=True)))
            step = (1 - np.exp(-INTERVAL / tau)) * (a - L) + sg * np.sqrt(INTERVAL) * noise
        s = s + step * LN10
    st = stats(s)
    srt = np.sort(np.exp(s), 1)
    st['cover'] = float((srt[:, 2] * srt[:, 1]).sum() / area0)
    st.update(tally)
    return st


def calibrate(M, kind, tau=None, **kw):
    """Fit the optimiser's two free numbers to 266's measured p50 and sd."""
    Lm, sdm = np.log10(M['p50']), M['sd']
    if kind == 'rw':
        d, f = 0.0, 0.0
        for _ in range(6):
            st = run(opt=('rw', d, f), **kw)
            d += (Lm - np.log10(st['p50'])) / ITERS
            f = np.sqrt(max(f * f + (sdm ** 2 - st['sd'] ** 2) / ITERS, 0.0))
        return ('rw', d, f)
    a, sg = Lm, 0.002
    for _ in range(8):
        st = run(opt=('ou', tau, a, sg), **kw)
        a += (Lm - np.log10(st['p50']))
        sg = sg * np.sqrt(max(sdm ** 2, 1e-6) / max(st['sd'] ** 2, 1e-6)) if st['sd'] > 0 else sg
        sg = min(max(sg, 0.0), 0.05)
    return ('ou', tau, a, sg)


def row(tag, st, M=None):
    extra = ''
    if M is not None:
        extra = '  err p10 %+5.1f%% p90 %+5.1f%%' % (100 * (st['p10'] / M['p10'] - 1),
                                                  100 * (st['p90'] / M['p90'] - 1))
    print('  %-24s p10 %5.2f p50 %5.2f p90 %6.2f spread %5.2fx sd %.3f cover %.2f  '
          'split %6d clone %6d reloc %6d%s'
          % (tag, st['p10'], st['p50'], st['p90'], st['spread'], st['sd'], st['cover'],
             st['split'], st['clone'], st['reloc'], extra))


def main():
    M = measured()
    print('MEASURED 266 model: p10 %.2f p50 %.2f p90 %.2f spread %.2fx sd %.3f decades'
          % (M['p10'], M['p50'], M['p90'], M['spread'], M['sd']))
    print('CENSUS events: split %d clone %d reloc %d' % (EVENTS['split'], EVENTS['clone'], EVENTS['reloc']))
    print('Scaniverse:     p10 1.33 p50 3.39 p90 11.83 spread 8.91x sd 0.636\n')

    base = run(opt=('rw', 0, 0))
    print('DENSIFICATION ONLY (no optimiser), as-run 0.8/1.6, persistence 0.5:')
    row('densification only', base)
    print()

    LEVERS = [('as-run 0.8 / 1.6', 0.8, 1.6), ('A54 1.0 / 2.0', 1.0, 2.0),
              ('1.0 / 1.6', 1.0, 1.6), ('0.8 / 2.0', 0.8, 2.0)]
    summary = []
    for kind, tau in (('rw', None), ('ou', 4000.0), ('ou', 1000.0), ('ou', 300.0)):
        for pers in (0.0, 0.5, 1.0):
            opt = calibrate(M, kind, tau, pers=pers)
            name = ('rw' if kind == 'rw' else 'ou tau %d' % tau) + ', persistence %.1f' % pers
            print('OPTIMISER %s  -> %s' % (name, ', '.join('%.3g' % v for v in opt[1:])))
            res = {}
            for tag, sh, kk in LEVERS:
                st = run(share=sh, shrink=kk, pers=pers, opt=opt)
                res[tag] = st
                row(tag, st, M if tag.startswith('as-run') else None)
            off = run(pers=pers, opt=opt, reloc=False)
            row('relocation OFF', off)
            a, b = res['as-run 0.8 / 1.6'], res['A54 1.0 / 2.0']
            print('    reloc off->on: p50 %+.2f mm spread %+.2fx  (device 250->256: -0.35 mm, +0.25x)'
                  % (a['p50'] - off['p50'], a['spread'] - off['spread']))
            print('    A54 vs as-run: p50 %.2f -> %.2f mm (%+.0f%%), spread %.2fx -> %.2fx, cover %.2f -> %.2f'
                  % (a['p50'], b['p50'], 100 * (b['p50'] / a['p50'] - 1), a['spread'],
                     b['spread'], a['cover'], b['cover']))
            summary.append((name, a['p50'], b['p50'], a['spread'], b['spread'],
                            a['p10'] / M['p10'] - 1, a['p90'] / M['p90'] - 1,
                            a['p50'] - off['p50'], a['spread'] - off['spread']))
            print()
            sys.stdout.flush()

    print('SEED-SHARE SENSITIVITY (rw, persistence 0.5, recalibrated each time)')
    for es, cs in ((0.243, 0.05), (0.243, 0.25), (0.12, 0.13)):
        opt = calibrate(M, 'rw', None, edge_share=es, coarse_share=cs)
        a = run(opt=opt, edge_share=es, coarse_share=cs)
        b = run(share=1.0, shrink=2.0, opt=opt, edge_share=es, coarse_share=cs)
        print('  edge %.3f coarse %.2f: as-run p50 %.2f spread %.2fx -> A54 p50 %.2f spread %.2fx'
              % (es, cs, a['p50'], a['spread'], b['p50'], b['spread']))

    print('\nSUMMARY: A54 (1.0/2.0) against as-run (0.8/1.6), every optimiser model')
    print('  %-30s %8s %8s %8s %8s %9s %9s %11s' % ('model', 'p50 now', 'p50 A54', 'spr now',
                                                   'spr A54', 'p10 err', 'p90 err', 'reloc dp50'))
    for r in summary:
        print('  %-30s %8.2f %8.2f %7.2fx %7.2fx %+8.1f%% %+8.1f%% %+10.2f'
              % (r[0], r[1], r[2], r[3], r[4], 100 * r[5], 100 * r[6], r[7]))


if __name__ == '__main__':
    main()
