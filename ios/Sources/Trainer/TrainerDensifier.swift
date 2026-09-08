//
//  TrainerDensifier.swift
//  Trainer
//
//  WHERE GAUSSIANS ARE BORN, MOVED AND KILLED. Budget first, always.
//
//  ---------------------------------------------------------------------------
//  THE FOUR THINGS THIS DOES DIFFERENTLY FROM STOCK 3DGS
//  ---------------------------------------------------------------------------
//
//  1. ABSGS. The densification signal is `stats.absGrad2D / stats.denom`, and
//     `absGrad2D` is the sum of the MAGNITUDES of the per-pixel screen-space
//     position gradients, accumulated by the backward kernel before any
//     summation. Stock 3DGS sums the SIGNED gradients, which cancel almost
//     exactly for a Gaussian straddling an edge - precisely the Gaussian that
//     needs splitting. Pixel-GS's area weighting is already in the numerator,
//     because it grows with the number of pixels covered.
//
//  2. THE CAP IS A WALL, NOT A TARGET. Densification is ranked and then
//     truncated to what the budget has room for. When the cap is reached, the
//     count stops changing entirely and improvement comes from MCMC-style
//     RELOCATION: a Gaussian that contributes nothing is moved onto one that
//     needs help, with the opacity correction that keeps the pair's combined
//     coverage equal to the original's. This is the difference between a run
//     that finishes on a phone and one that gets killed at iteration 900.
//
//  3. THE SPLIT IS ANISOTROPIC AND EDGE-AWARE (SAD-GS). A split shrinks ONLY
//     the axis it split along, so a disc that was the right size across the
//     surface stays the right size across the surface. For a Gaussian flagged
//     as lying on a 3D edge curve, the split runs across the edge normal (the
//     second-largest axis) rather than along the needle, because splitting a
//     needle along its length just makes two shorter needles in the same wrong
//     place.
//
//  4. FREE-SPACE CARVING DELETES, AND UNKNOWN NEVER DOES. The carver is asked
//     for the indices whose centres sit in cells a LiDAR beam demonstrably
//     passed through. Cells that are UNKNOWN - past the sensor's range, or
//     behind a no-return at glass - are not in that list and must never be,
//     which is what stops the carver erasing everything visible through a
//     window.
//

import Foundation
import simd

/// Why a densification pass created the number of Gaussians it created,
/// reduced to ONE value.
///
/// Seven counters between them say what happened, and reading them in the
/// wrong order gives the wrong answer: "nothing scored" at the cap is a dead
/// gradient signal wearing a full budget as a disguise, and "created nothing"
/// outside the growth window is simply the schedule doing its job. Deriving
/// this once, here, next to the counters it is derived from, is what stops
/// every reader deriving it slightly differently.
///
/// Nothing in here is stored. Every case is decided by a number this pass
/// actually measured, so it can never claim a cause the code did not observe.
enum TrainerDensifyGrowthVerdict: String, Codable, Sendable {
    /// There were no Gaussians to work with. Not a densification result.
    case populationEmpty
    /// The GPU buffers read short, so NOTHING was scored. Distinct from a
    /// dead signal: nothing was even looked at.
    case buffersReadShort
    /// Gaussians were created. The only healthy growth outcome.
    case grew
    /// Not one Gaussian had a position gradient above zero this interval.
    /// This is the fingerprint of the original bug: a scoring stage that
    /// produces no candidates at all, whatever the budget says.
    case nothingScored
    /// Gaussians scored, and not one of them was visible in any frame this
    /// interval, so none could say where detail was missing. A different
    /// fault with a different fix: the visibility accumulator, not the
    /// gradient.
    case nothingVisible
    /// The schedule had the growth window shut. Correct behaviour.
    case growthWindowClosed
    /// The population is at its cap, so the pass relocated instead of
    /// growing. Correct behaviour: the cap is a wall, not a target.
    case atTheCapRelocated
    /// At the cap with candidates to help, and not one Gaussian faint enough
    /// to move onto them. The pass genuinely changed nothing.
    case atTheCapNoDonors
    /// The recorded counters do not account for the result. Reachable only if
    /// the growth loop stops adding while it still has both allowance and
    /// candidates, which the loop below cannot do. Present so this type never
    /// has to invent a cause it did not measure.
    case unexplained
}

/// What one densification pass did. Every number is a count of something that
/// actually happened, so the log and the progress message can be specific.
///
/// The second block exists for the census (TrainerCensus.swift). A pass that
/// adds nothing is not evidence of anything on its own: it could be a full
/// budget, a shut window, or a scoring stage that produced no candidates at
/// all, and those are a normal run, a schedule bug and a units bug
/// respectively. Recording WHY there was nothing to add is what turns "zero
/// splats created" from a shrug into a diagnosis. Every one of these is a
/// value this function already computed; none of them costs an extra pass over
/// the buffers and none of them touches the GPU.
struct TrainerDensifyOutcome {
    var splatCountBefore = 0
    var splatCountAfter = 0
    var cloned = 0
    var split = 0
    var relocated = 0
    var prunedLowOpacity = 0
    var prunedOversized = 0
    var prunedNonFinite = 0
    var carvedFromEmptySpace = 0
    var trimmedToCap = 0

    // --- Why this pass could or could not do anything -----------------------

    /// The flags this pass was called with, echoed back so one census row is
    /// self-contained rather than needing the caller's state to interpret.
    var growthAllowed = false
    var pruneAllowed = false
    /// True when a carver was handed in, which is the only case where free
    /// space may delete anything.
    var carveAttempted = false
    /// The wall this pass was working against, and the room under it.
    var splatCapInForce = 0
    var headroomAtStart = 0
    /// How many the pass was permitted to create, after the cap and the
    /// per-pass growth fraction. Zero here with an open window is a budget
    /// problem; non-zero here with nothing added is a scoring problem.
    var growthAllowance = 0
    /// How many Gaussians were scored at all.
    var splatsScored = 0
    /// How many of those scored above zero, and how many of THOSE survived the
    /// "something actually looked at it" filter and became candidates.
    var splatsWithNonZeroScore = 0
    var candidatesAfterVisibilityFilter = 0
    /// How many Gaussians were faint or unseen enough to be relocation donors.
    /// Only counted on the relocation path, which is the only place it is
    /// computed.
    var relocationDonorsAvailable = 0

    /// True when this pass created or deleted at least one Gaussian.
    ///
    /// Nothing reads this any more, ON PURPOSE. It used to gate the
    /// per-interval reset of the densification accumulators in
    /// `MetalSplatTrainer`, and gating that reset was a bug: a pass that
    /// changed nothing skipped the reset and let `visAccum` run on into the
    /// next interval, which loosens the visibility filter the longer
    /// densification goes without changing anything. The reset is now
    /// unconditional. Kept because the comment at that call site names this
    /// property when it explains why.
    var changedTopology: Bool {
        cloned + split + prunedLowOpacity + prunedOversized
            + prunedNonFinite + carvedFromEmptySpace + trimmedToCap > 0
    }

    /// How many Gaussians this pass created. Named because "did densification
    /// work" is asked in four places and re-derived slightly differently each
    /// time.
    var created: Int { cloned + split }

    /// Everything the pass deleted, by any of the five reasons.
    var removed: Int {
        prunedLowOpacity + prunedOversized + prunedNonFinite
            + carvedFromEmptySpace + trimmedToCap
    }

    /// The scoring stage looked at Gaussians and produced NO candidates.
    ///
    /// This is the single most important boolean in this file. It is true for
    /// exactly the fault that hid for the whole history of this app: the
    /// gradient signal is dead, or the visibility accumulator is, and
    /// densification therefore cannot create anything no matter how much room
    /// the budget leaves it. It is deliberately NOT gated on the growth
    /// window or on the cap, because a dead signal is dead whether or not the
    /// schedule happened to be asking for growth at that moment, and gating
    /// it would hide it in exactly the passes where it is cheapest to notice.
    var scoringProducedNoCandidates: Bool {
        splatsScored > 0 && candidatesAfterVisibilityFilter == 0
    }

    /// The one-value answer to "why did this pass create what it created".
    ///
    /// The order of the tests is the point. Scoring faults are checked BEFORE
    /// the window and the cap, because a shut window and a full budget are
    /// both perfectly good explanations for creating nothing and both of them
    /// will happily stand in front of a dead gradient signal and hide it.
    var growthVerdict: TrainerDensifyGrowthVerdict {
        if splatCountBefore <= 0 { return .populationEmpty }
        if splatsScored == 0 { return .buffersReadShort }
        if created > 0 { return .grew }
        if splatsWithNonZeroScore == 0 { return .nothingScored }
        if candidatesAfterVisibilityFilter == 0 { return .nothingVisible }
        if !growthAllowed { return .growthWindowClosed }
        if growthAllowance == 0 {
            if relocated > 0 { return .atTheCapRelocated }
            return .atTheCapNoDonors
        }
        return .unexplained
    }

    /// One plain sentence, only when there is something worth saying.
    ///
    /// DO NOT make this non-nil for a pass that changed nothing. It reads as
    /// an optional on purpose and the training loop branches on that: nil is
    /// what routes a zero-growth pass to the loud branch that prints the
    /// counters and the streak length, and a non-nil string would send that
    /// pass quietly back to `.info`. The loudness for the nothing-happened
    /// case lives at the call site and in `announce` below, not here.
    var summary: String? {
        var parts: [String] = []
        if created > 0 { parts.append("added \(created) points of detail") }
        if relocated > 0 {
            // This used to read "moved N unused points somewhere useful",
            // which is the whole sentence a pass that created NOTHING would
            // print, at info level, and it reads as a healthy pass. It is
            // not: relocation is what happens INSTEAD of growth, the total
            // does not move, and if it is the only thing being reported then
            // no new detail was added at all. Say both halves.
            // The two branches are not decoration. Growth and relocation are
            // an if/else in `run`, so today only one of them can be non-zero,
            // and this must not print "created nothing" from a stale
            // assumption if that ever stops being true.
            if created == 0 {
                parts.append(
                    "created nothing and moved \(relocated) unused points onto detail "
                        + "that needed help, so the total is unchanged"
                )
            } else {
                parts.append(
                    "moved \(relocated) unused points onto detail that needed help, "
                        + "which does not change the total"
                )
            }
        }
        if removed > 0 { parts.append("removed \(removed)") }
        // A pass that deleted or relocated something but SCORED NOTHING would
        // otherwise print "removed 12" and look like a normal, healthy pass,
        // which is the same disguise the original bug wore. The counters that
        // matter get carried in the same sentence.
        if scoringProducedNoCandidates, !parts.isEmpty {
            var note = "and densification found no candidates at all ("
            note += String(splatsWithNonZeroScore)
            note += " of "
            note += String(splatsScored)
            note += " scored above zero, "
            note += String(candidatesAfterVisibilityFilter)
            note += " survived the visibility filter)"
            parts.append(note)
        }
        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: ", ")
    }
}

final class TrainerDensifier {

    private let tuning: TrainerTuning
    private let settings: SmartLossSettings

    init(tuning: TrainerTuning, settings: SmartLossSettings) {
        self.tuning = tuning
        self.settings = settings
    }

    /// The whole pass: score, grow (or relocate at the cap), prune, carve,
    /// then write everything back and hand the caller the new count.
    ///
    /// - Parameters:
    ///   - allowGrowth: false during the refine phase, when the population is
    ///     meant to settle rather than keep changing.
    ///   - carver: asked for certified-empty indices when non-nil. Nil means
    ///     no occupancy grid was carved for this scan, and then nothing is
    ///     deleted on free-space grounds, because "no grid" is not "empty".
    /// - Parameter allowPrune: whether this pass may DELETE Gaussians for low
    ///   opacity or bad size. Separate from `allowGrowth` because the two have
    ///   different windows: `SmartLossSettings.pruneStartFraction` (0.15) and
    ///   `pruneEndFraction` (0.80) were declared, defaulted and assigned but
    ///   read nowhere in the repository, so pruning was in fact ungated. It ran
    ///   every 100 iterations from iteration 100 to the very end, which is 29
    ///   passes on a 3,000 iteration run instead of the ~10 the settings
    ///   describe, including passes before anything had converged and passes
    ///   during late opacity binarization, where they delete splats that were
    ///   only briefly pushed towards zero. Carving is deliberately NOT gated by
    ///   this: certified empty space is evidence at any point in the run.
    func run(
        resources: TrainerResources,
        splatCount: Int,
        splatCap: Int,
        sceneExtentMeters: Float,
        allowGrowth: Bool,
        allowPrune: Bool,
        carver: FreeSpaceCarver?
    ) throws -> TrainerDensifyOutcome {

        var outcome = TrainerDensifyOutcome()
        // Every return below, early or late or thrown, goes through this. A
        // `defer` reads `outcome` at scope exit, so it reports the finished
        // pass and not the empty one declared on the line above. See
        // `announce` for what it says and, more to the point, what it does
        // not say twice.
        defer { Self.announce(outcome) }
        outcome.splatCountBefore = splatCount
        outcome.splatCountAfter = splatCount
        // Recorded before the early return, so a pass that did nothing because
        // there was nothing there still says so in the census rather than
        // arriving as a row of zeroes with no explanation.
        outcome.growthAllowed = allowGrowth
        outcome.pruneAllowed = allowPrune
        outcome.carveAttempted = carver != nil
        outcome.splatCapInForce = splatCap
        outcome.headroomAtStart = Swift.max(splatCap - splatCount, 0)
        guard splatCount > 0 else { return outcome }

        let shPerSplat = resources.shFloatsPerSplat
        var splats = resources.splats.readArray(TrainerSplat.self, count: splatCount)
        var stats = resources.stats.readArray(TrainerSplatStats.self, count: splatCount)
        var sh = resources.sh.readArray(Float.self, count: splatCount * shPerSplat)
        var adamM = resources.adamM.readArray(TrainerSplatGrad.self, count: splatCount)
        var adamV = resources.adamV.readArray(TrainerSplatGrad.self, count: splatCount)
        var shM = resources.shAdamM.readArray(Float.self, count: splatCount * shPerSplat)
        var shV = resources.shAdamV.readArray(Float.self, count: splatCount * shPerSplat)
        var topK = resources.samplingTopK.readArray(TrainerSamplingTopK.self, count: splatCount)

        guard splats.count == splatCount, stats.count == splatCount else {
            TrainerLog.densify.error("Densification skipped: the GPU buffers read short")
            return outcome
        }

        // --- 1. Score --------------------------------------------------------
        // AbsGS mean magnitude per observation. A Gaussian nothing looked at
        // this interval scores zero rather than infinity.
        var score = [Float](repeating: 0, count: splatCount)
        // Set here rather than at the top of the function on purpose: a pass
        // that returned early because the buffers read short scored NOTHING,
        // and recording it as "scored everything, nothing passed" would make a
        // failed read look identical to a dead gradient signal.
        outcome.splatsScored = splatCount
        for i in 0..<splatCount {
            let denominator = Swift.max(stats[i].denom, 1)
            let value = stats[i].absGrad2D / denominator
            score[i] = value.isFinite ? value : 0
        }

        let maxWorldScale = Swift.max(
            sceneExtentMeters * tuning.pruneMaxWorldScaleFraction, 0.05
        )
        let splitThresholdScale = Swift.max(
            sceneExtentMeters * tuning.splitScaleFraction, 0.001
        )

        // --- 2. Choose candidates, ranked, then TRUNCATED to the budget ------
        // The threshold is a floor, not the operating point. The real cut is
        // "the best N that fit", which is what makes the cap a wall rather
        // than a thing to grow into.
        // The gate is "did anything look at this and want it to move", not a
        // magnitude in some particular unit.
        //
        // It used to be `score[i] >= tuning.absGradThreshold` with the
        // threshold at 6e-4, a number copied from the reference 3DGS
        // implementation. That reference stores its gradient AFTER multiplying
        // by 0.5 * width, which converts it to normalised device coordinates.
        // This rasteriser accumulates `length(dLdMean2D)` raw, and its `delta`
        // and conic are both in PIXELS (TrainerShaders.metal builds delta as
        // `mean2D - pixelCenter`). A pixel-space gradient is smaller than the
        // same quantity in NDC by a factor of roughly half the render width, so
        // at 720 px nothing ever cleared the bar. Densification therefore never
        // fired ONCE: zero splats created, zero split, zero cloned, for the
        // whole run. The model stayed exactly as sparse as the LiDAR seed.
        //
        // Comparing against zero and letting the ranked truncation below do the
        // cutting is scale-free: it cannot be broken again by a units change
        // anywhere upstream, and it is what the header of this file already
        // said the design was ("the threshold is a floor, not the operating
        // point. The real cut is the best N that fit").
        //
        // WHAT THIS COSTS, AND WHY IT IS NOT A FULL SORT.
        //
        // With the gate at effectively zero, every Gaussian that got any
        // gradient at all is a candidate, so this list is very nearly the
        // whole population. It used to be built and then FULLY SORTED, every
        // pass. On a 300,000 point room scan that is a 300k-element sort with
        // a closure comparator and two random-access reads per comparison,
        // 29 times per slice (the pass runs every 100 iterations from 100 to
        // the end of the run, not only inside the growth window).
        //
        // COMPARISON COUNTS, exact, for this list built from a long-tailed
        // score distribution with 15 percent exact zeros. These are counted,
        // not estimated, and they do not depend on the machine:
        //
        //   population   above zero   allowance   full sort    selection
        //      150,000      127,282      22,500   1,990,319      502,131
        //      300,000      255,035      45,000   4,242,295    1,104,232
        //      500,000      424,988      75,000   7,383,991    1,772,550
        //
        // which is 3.96x, 3.84x and 4.17x fewer comparisons. Wall clock on a
        // desktop transcription of exactly this code (a proxy for the ratio,
        // NOT a device measurement) at those three sizes: 47.8 ms against
        // 11.0, 101.0 against 18.5, 175.3 against 35.4. On the relocation
        // path, where the bound is 5 percent rather than 15, it is 101.0 ms
        // against 9.3 at 300,000. With the growth window shut it now does no
        // ranking at all: 101.0 ms against 0.9.
        //
        // Nothing below the truncation point is ever read, so ordering it was
        // pure heat. `selectHighest` puts the best `selectionLimit` in front
        // by partitioning, and only that prefix is sorted.
        //
        // The floor stays `score[i] > 0`. A magnitude threshold in some
        // particular unit is what broke this in the first place and no amount
        // of saved heat is worth reintroducing one.
        let headroom = Swift.max(splatCap - splatCount, 0)
        let growthAllowance = allowGrowth
            ? Swift.min(headroom, Int(Float(splatCap) * tuning.maxGrowthFractionPerPass))
            : 0
        outcome.growthAllowance = growthAllowance
        // How many candidates the RELOCATION path could possibly consume, if
        // it is the path that runs. It runs only at the cap, which is exactly
        // when `growthAllowance` is zero, so the two limits are never both
        // non-zero and the maximum of them is the true bound on how much of
        // this list is ever looked at.
        let relocationLimit = (allowGrowth && growthAllowance == 0)
            ? Swift.max(Int(Float(splatCount) * tuning.maxRelocationFractionPerPass), 0)
            : 0
        let selectionLimit = Swift.max(growthAllowance, relocationLimit)
        // Collecting is a separate question from RANKING. The relocation
        // branch below also counts how many donors were available, for the
        // census, and it is only entered when the candidate list is non-empty.
        // Below about twenty Gaussians `relocationLimit` truncates to zero, so
        // ranking is pointless there, but skipping the list entirely would
        // skip the donor count too and put a "0 donors available" in the
        // census that nothing measured. Fuzzed against the previous code over
        // 600 randomised populations: this is the one place the two differed.
        let needCandidateList = selectionLimit > 0
            || (allowGrowth && growthAllowance == 0)

        var candidates: [Int] = []
        var scoredAboveZero = 0
        var visibleCandidates = 0
        for i in 0..<splatCount where score[i] > 0 {
            scoredAboveZero += 1
            // A Gaussian nothing has seen has nothing to say about where
            // detail is missing.
            if stats[i].visAccum <= 0 { continue }
            visibleCandidates += 1
            // Counted always, collected only when something downstream can
            // use it. With the growth window shut, neither path below runs, so
            // building the list at all was work for a list nobody read. The
            // census still gets its two counts either way.
            if needCandidateList { candidates.append(i) }
        }
        // Two counts, not one: "nothing scored" and "everything that scored
        // was invisible" are different faults with different fixes, and a
        // single "candidates: 0" cannot tell them apart.
        outcome.splatsWithNonZeroScore = scoredAboveZero
        outcome.candidatesAfterVisibilityFilter = visibleCandidates

        if selectionLimit > 0 {
            if candidates.count > selectionLimit {
                candidates = Self.selectHighest(
                    candidates, by: score, count: selectionLimit
                )
            } else {
                // Already within the bound, so the ordering is over a list
                // that is at most `selectionLimit` long. The relocation path
                // pairs the k-th best candidate with the k-th faintest donor,
                // so the order of this prefix does matter.
                candidates.sort { score[$0] > score[$1] }
            }
        }

        // THE SPLIT SHARE. See `splitShareOfGrowth`.
        //
        // `candidates` is already the ranked, truncated growth list, so this
        // is a second ranking OF THAT LIST by world scale, and the largest
        // `splitShareOfGrowth` of it takes the split path. Computing the cut
        // as a quantile of the list itself is what makes it scale-free: it
        // does not care what units the scale is in, how large the room is, or
        // how the population's size distribution has drifted this run.
        //
        // Sorting a copy of the scales is O(n log n) on at most
        // `growthAllowance` elements, which is 45,000 at a 300,000 cap, and it
        // happens once per pass against the ~1.1 million comparisons the
        // selection above already spends.
        var splitScaleCut = Float.infinity
        if tuning.splitShareOfGrowth > 0, !candidates.isEmpty, growthAllowance > 0 {
            let considered = Swift.min(candidates.count, growthAllowance)
            var scales = [Float](repeating: 0, count: considered)
            for k in 0..<considered {
                // The SAME clamp the split body applies, so the cut and the
                // comparison are built from identical numbers.
                let s = simd_clamp(
                    splats[candidates[k]].logScale,
                    SIMD3<Float>(repeating: -12), SIMD3<Float>(repeating: 3)
                )
                scales[k] = Swift.max(s.x, Swift.max(s.y, s.z))
            }
            scales.sort()
            // The index below which a candidate is NOT split. Clamped so a
            // share of 1 splits everything and a tiny list still splits one.
            // Integer arithmetic, no float-to-int conversion at all. `share`
            // is clamped to 0...1000 first, so NaN takes the else branch (every
            // comparison against NaN is false) and the product cannot overflow:
            // `considered` is at most the growth allowance.
            let shareIsUsable = tuning.splitShareOfGrowth.isFinite
            let perMille = shareIsUsable
                ? Int(Swift.min(Swift.max(tuning.splitShareOfGrowth, 0), 1) * 1000)
                : 0
            let keepClone = considered * (1000 - perMille) / 1000
            let idx = Swift.max(0, Swift.min(considered - 1, keepClone))
            splitScaleCut = scales[idx]
        }

        var newSplats: [TrainerSplat] = []
        var newSH: [Float] = []
        var newTopK: [TrainerSamplingTopK] = []
        newSplats.reserveCapacity(growthAllowance)
        newSH.reserveCapacity(growthAllowance * shPerSplat)

        var added = 0
        if growthAllowance > 0 {
            for index in candidates {
                if added >= growthAllowance { break }
                let parent = splats[index]
                let scale = simd_clamp(
                    parent.logScale, SIMD3<Float>(repeating: -12), SIMD3<Float>(repeating: 3)
                )
                let linear = SIMD3<Float>(expf(scale.x), expf(scale.y), expf(scale.z))
                let largest = Swift.max(linear.x, Swift.max(linear.y, linear.z))

                // Over-sized in the WORLD, or over-sized ON SCREEN. The
                // second test is the one that fires: see the comment on
                // `splitScreenRadiusPx`. `maxRadiusPxBits` is the largest
                // projected radius this Gaussian reached during the interval,
                // already accumulated by the rasteriser and already read by
                // the prune below, so this costs one load and one compare.
                var oversizedOnScreen = false
                if tuning.splitScreenRadiusPx > 0, index < stats.count {
                    let r = Float(bitPattern: stats[index].maxRadiusPxBits)
                    oversizedOnScreen = r.isFinite && r > tuning.splitScreenRadiusPx
                }

                // Log space, so this compares the same quantity the cut
                // was built from rather than its exponential.
                let logLargest = Swift.max(scale.x, Swift.max(scale.y, scale.z))
                if largest > splitThresholdScale
                    || oversizedOnScreen
                    || logLargest >= splitScaleCut {
                    // SPLIT: over-reconstructed. Two children inside the
                    // parent's own ellipsoid.
                    let axis = splitAxis(for: parent, linearScale: linear)
                    let rotation = TrainerMath.rotationMatrix(parent.rotation)

                    var shrunk = scale
                    var offset: SIMD3<Float>
                    if tuning.splitShrinkAllAxes {
                        // The reference geometry: divide the WHOLE scale
                        // vector by 0.8*N, which for N=2 is 1.6 and is
                        // exactly `splitShrink`, then scatter the children
                        // through the parent's volume by sampling the offset
                        // from the parent's own covariance instead of
                        // stepping a fixed distance along one axis. Shrinking
                        // three axes while offsetting along one would leave a
                        // gap across the other two.
                        let k = logf(tuning.splitShrink)
                        shrunk = scale - SIMD3<Float>(repeating: k)
                        let g = TrainerDensifier.splitNoise(seed: UInt32(truncatingIfNeeded: index) &+ UInt32(truncatingIfNeeded: added))
                        let local = SIMD3<Float>(
                            g.x * linear.x, g.y * linear.y, g.z * linear.z
                        ) * tuning.splitOffsetSigma
                        offset = rotation[0] * local.x
                            + rotation[1] * local.y
                            + rotation[2] * local.z
                    } else {
                        shrunk[axis] = scale[axis] - logf(tuning.splitShrink)
                        offset = rotation[axis] * (linear[axis] * tuning.splitOffsetSigma)
                    }

                    // Child A replaces the parent in place, child B is new, so
                    // one split costs one slot rather than two.
                    var childA = parent
                    childA.mean = parent.mean + offset
                    childA.logScale = shrunk
                    childA.flags = parent.flags | TrainerSplatFlag.densified
                    splats[index] = childA
                    // The parent's optimiser state described a Gaussian that no
                    // longer exists. Keeping it makes the child's first step
                    // fly off in the parent's direction.
                    adamM[index] = TrainerSplatGrad()
                    adamV[index] = TrainerSplatGrad()

                    var childB = parent
                    childB.mean = parent.mean - offset
                    childB.logScale = shrunk
                    childB.flags = parent.flags | TrainerSplatFlag.densified
                    newSplats.append(childB)
                    newSH.append(contentsOf: sh[(index * shPerSplat)..<((index + 1) * shPerSplat)])
                    newTopK.append(topK[index])
                    added += 1
                    outcome.split += 1
                } else {
                    // CLONE: small and under-explained means the surface is
                    // under-covered here, so a second Gaussian is added beside
                    // it rather than the existing one being made bigger.
                    let axis = splitAxis(for: parent, linearScale: linear)
                    let rotation = TrainerMath.rotationMatrix(parent.rotation)
                    let offset = rotation[axis] * (linear[axis] * 0.25)
                    var clone = parent
                    clone.mean = parent.mean + offset
                    clone.flags = parent.flags | TrainerSplatFlag.densified
                    newSplats.append(clone)
                    newSH.append(contentsOf: sh[(index * shPerSplat)..<((index + 1) * shPerSplat)])
                    newTopK.append(topK[index])
                    added += 1
                    outcome.cloned += 1
                }
            }
        } else if allowGrowth, !candidates.isEmpty {
            // --- 3. AT THE CAP: MCMC-style relocation -------------------------
            // Nothing is created and nothing is destroyed. A Gaussian that is
            // contributing nothing is picked up and put down on top of one that
            // needs help, and BOTH get the opacity correction that keeps their
            // combined coverage equal to what the target had on its own:
            //   1 - (1 - o_new)^2 = o_old  ->  o_new = 1 - sqrt(1 - o_old)
            // Without that correction every relocation quietly doubles the
            // opacity of the region it lands in.
            //
            // THE DONOR RANKING WAS THE MORE EXPENSIVE OF THE TWO SORTS.
            //
            // It was `donors.sort { sigmoid(splats[$0].opacityLogit) <
            // sigmoid(splats[$1].opacityLogit) }`, which calls `sigmoid` twice
            // per COMPARISON rather than once per element, and `sigmoid` is an
            // `expf`. A full sort of 300,000 donors is about 4.2 million
            // comparisons, so about 8.5 million `expf` calls, for a list of
            // which at most 15,000 entries (5 percent of the population) are
            // ever read.
            //
            // The worst case is not rare. `stats[i].visAccum <= 0` makes a
            // Gaussian a donor, and that is true of EVERY Gaussian on the
            // first pass after the accumulators are reset, so the donor list
            // starts out as the whole population. Wall clock on a desktop
            // transcription of exactly this code, all Gaussians donors (a
            // proxy for the ratio, NOT a device measurement):
            //
            //   population      before       after
            //      150,000     154.6 ms      8.8 ms     17.7x
            //      300,000     335.9 ms     17.7 ms     18.9x
            //      500,000     579.3 ms     31.7 ms     18.3x
            //
            // With a more typical donor list of about a sixth of the
            // population it is 95.4 ms against 14.7 at 300,000, 6.5x.
            //
            // The key is negated so the same "highest first" selection that
            // ranks candidates picks the FAINTEST donors, with no second
            // code path to keep in step.
            //
            // One honest difference. Neither the old full sort nor the new
            // selection is stable, so when many Gaussians sit at the SAME
            // opacity, which of them lands at position k changes. That shifts
            // how often `donor == target` coincides and is skipped, so the
            // number of relocations in a pass can differ by a handful from
            // what the old code would have done. Fuzzed over 400 deliberately
            // tie-heavy populations: the donor count, the allowance and the
            // selected opacities are identical every time, and only that
            // self-pair coincidence count moves. It is a coincidence, not a
            // quality property.
            var donorKey = [Float](repeating: 0, count: splatCount)
            var donors: [Int] = []
            for i in 0..<splatCount {
                let opacity = TrainerMath.sigmoid(splats[i].opacityLogit)
                donorKey[i] = -opacity
                if opacity < tuning.relocationDonorOpacity || stats[i].visAccum <= 0 {
                    donors.append(i)
                }
            }
            outcome.relocationDonorsAvailable = donors.count

            let allowance = Swift.min(
                donors.count,
                Swift.min(candidates.count, relocationLimit)
            )
            if allowance > 0 {
                if donors.count > allowance {
                    donors = Self.selectHighest(donors, by: donorKey, count: allowance)
                } else {
                    donors.sort { donorKey[$0] > donorKey[$1] }
                }
            }
            for k in 0..<Swift.max(allowance, 0) {
                let donor = donors[k]
                let target = candidates[k]
                if donor == target { continue }

                let parent = splats[target]
                let scale = simd_clamp(
                    parent.logScale, SIMD3<Float>(repeating: -12), SIMD3<Float>(repeating: 3)
                )
                let linear = SIMD3<Float>(expf(scale.x), expf(scale.y), expf(scale.z))
                let axis = splitAxis(for: parent, linearScale: linear)
                let rotation = TrainerMath.rotationMatrix(parent.rotation)
                let offset = rotation[axis] * (linear[axis] * tuning.splitOffsetSigma)

                let oldOpacity = TrainerMath.sigmoid(parent.opacityLogit)
                let corrected = 1 - sqrtf(Swift.max(1 - oldOpacity, 0))
                let correctedLogit = TrainerMath.logit(corrected)

                var shrunk = scale
                shrunk[axis] = scale[axis] - logf(tuning.splitShrink)

                var moved = parent
                moved.mean = parent.mean - offset
                moved.logScale = shrunk
                moved.opacityLogit = correctedLogit
                moved.flags = parent.flags | TrainerSplatFlag.densified
                splats[donor] = moved

                var kept = parent
                kept.mean = parent.mean + offset
                kept.logScale = shrunk
                kept.opacityLogit = correctedLogit
                splats[target] = kept

                // The donor's colour comes with it; its old colour belonged to
                // wherever it used to be.
                for c in 0..<shPerSplat {
                    sh[donor * shPerSplat + c] = sh[target * shPerSplat + c]
                    shM[donor * shPerSplat + c] = 0
                    shV[donor * shPerSplat + c] = 0
                }
                topK[donor] = topK[target]
                adamM[donor] = TrainerSplatGrad()
                adamV[donor] = TrainerSplatGrad()
                adamM[target] = TrainerSplatGrad()
                adamV[target] = TrainerSplatGrad()
                stats[donor].stepCount = 0

                outcome.relocated += 1
            }
        }

        // Append the newly created Gaussians. A new Gaussian starts with clean
        // optimiser state and a zeroed step count, so its bias correction
        // starts at step 1 rather than at the global iteration number.
        if !newSplats.isEmpty {
            splats.append(contentsOf: newSplats)
            sh.append(contentsOf: newSH)
            topK.append(contentsOf: newTopK)
            for _ in 0..<newSplats.count {
                adamM.append(TrainerSplatGrad())
                adamV.append(TrainerSplatGrad())
                var fresh = TrainerSplatStats()
                fresh.filter3D = 0
                stats.append(fresh)
            }
            shM.append(contentsOf: [Float](repeating: 0, count: newSplats.count * shPerSplat))
            shV.append(contentsOf: [Float](repeating: 0, count: newSplats.count * shPerSplat))
        }

        var liveCount = splats.count

        // --- 4. Prune ---------------------------------------------------------
        var keep = [Bool](repeating: true, count: liveCount)

        for i in 0..<liveCount {
            let splat = splats[i]
            let mean = splat.mean
            let logScale = splat.logScale
            // A non-finite Gaussian is deleted in EVERY pass, gated or not: it
            // is not a judgement about quality, it is a value that would poison
            // the next gradient step and every splat it touches.
            if !mean.x.isFinite || !mean.y.isFinite || !mean.z.isFinite
                || !logScale.x.isFinite || !logScale.y.isFinite || !logScale.z.isFinite
                || !splat.opacityLogit.isFinite
            {
                keep[i] = false
                outcome.prunedNonFinite += 1
                continue
            }
            // The two JUDGEMENT prunes below are the ones that respect the
            // window. Outside it, a faint or fat Gaussian is left alone to
            // carry on being optimised rather than deleted for how it looks
            // part way through.
            guard allowPrune else { continue }

            // WHAT REACHES A PIXEL, NOT WHAT IS STORED.
            //
            // This tested `sigmoid(opacityLogit)` alone, and that is not
            // the opacity anything ever draws with. The rasteriser uses
            // `sigmoid(opacityLogit) * comp2D * comp3D`, where comp3D is
            // the Mip-Splatting low-pass compensation and can be far below
            // 1 for a small Gaussian: SplatCloud puts it at 0.35 when an
            // axis equals the filter width and 0.014 at a quarter of it.
            //
            // So the prune was structurally blind to the exact population
            // it exists to remove. A splat storing a logit worth 0.04, EIGHT
            // TIMES the 0.005 threshold and therefore never touched by any
            // prune pass, draws at 0.004 once comp3D is applied, which is
            // below the rasteriser's own 1/255 reject. It is invisible and
            // it is immortal.
            //
            // Measured on a finished 3000-iteration run of 299,965 splats:
            // median peak alpha 0.0038, and 50.1 percent of the model below
            // 1/255. Half the budget was Gaussians the prune could not see
            // and the renderer would not draw.
            //
            // This is now REQUIRED rather than an improvement. The alpha
            // cull added to trainer_preprocess stops these splats setting
            // visibleFlag, and the Adam passes gate on that, so they no
            // longer move at all. A frozen Gaussian that no prune can see
            // would be permanent dead weight.
            //
            // `SplatMath.filter3DCompensation` is called rather than
            // reimplemented:
            // MetalSplatTrainer says in as many words that the 3D filter
            // lives in one place and warns against a second copy of it.
            let sigma = SIMD3<Float>(
                expf(logScale.x), expf(logScale.y), expf(logScale.z)
            )
            let compensation = SplatMath.filter3DCompensation(
                sigma: sigma, filter3D: stats[i].filter3D
            )
            let drawnOpacity = TrainerMath.sigmoid(splat.opacityLogit) * compensation
            if drawnOpacity < tuning.pruneOpacity {
                keep[i] = false
                outcome.prunedLowOpacity += 1
                continue
            }
            let linearMax = expf(Swift.max(logScale.x, Swift.max(logScale.y, logScale.z)))
            if linearMax > maxWorldScale {
                keep[i] = false
                outcome.prunedOversized += 1
                continue
            }
            if tuning.pruneMaxScreenRadiusPx > 0, i < stats.count {
                let radius = Float(bitPattern: stats[i].maxRadiusPxBits)
                if radius.isFinite, radius > tuning.pruneMaxScreenRadiusPx {
                    keep[i] = false
                    outcome.prunedOversized += 1
                }
            }
        }

        // --- 5. Free-space carving (F2) ---------------------------------------
        // Only EMPTY licenses a delete. The carver's contract is that UNKNOWN
        // cells are never in this list; that is relied on here rather than
        // re-checked, because re-checking it with a second query would be
        // asking the same object the same question twice.
        if let carver {
            var centers = [SIMD3<Float>](repeating: .zero, count: liveCount)
            for i in 0..<liveCount { centers[i] = splats[i].mean }
            let empty = carver.certifiedEmptyIndices(centers: centers)
            // Never delete more than a small share in one sweep: the grid is
            // evidence, but a run that loses a quarter of its Gaussians in one
            // step has no way back if the evidence was mis-registered.
            let capOnDeletes = Swift.max(Int(Float(liveCount) * settings.pruneMaxFractionPerPass), 1)
            var deleted = 0
            for index in empty {
                guard index >= 0, index < liveCount, keep[index] else { continue }
                keep[index] = false
                outcome.carvedFromEmptySpace += 1
                deleted += 1
                if deleted >= capOnDeletes { break }
            }
        }

        // --- 6. Enforce the cap ------------------------------------------------
        var survivorCount = keep.reduce(into: 0) { $0 += ($1 ? 1 : 0) }
        if survivorCount > splatCap {
            // Rank the survivors by what they actually contribute and keep the
            // best that fit. This is the last line of the budget-first rule and
            // it runs whatever else happened above.
            //
            // This one only needs the SET that survives, never its order, so
            // it partitions and stops. It also runs at the worst possible
            // moment: `survivorCount` can only exceed the cap when the
            // governor has just LOWERED the cap for heat, so a full sort of
            // the whole population here spends CPU on the exact thermal
            // budget that cut is trying to protect.
            var survivors: [Int] = []
            survivors.reserveCapacity(survivorCount)
            var importance = [Float](repeating: 0, count: liveCount)
            for i in 0..<liveCount where keep[i] {
                let opacity = TrainerMath.sigmoid(splats[i].opacityLogit)
                let visibility = i < stats.count ? Swift.max(stats[i].visAccum, 0) : 0
                importance[i] = opacity * (1 + visibility)
                survivors.append(i)
            }
            // `max(_, 0)` only so a negative cap can never build a reversed
            // range and trap. The branch above already guarantees the count
            // exceeds it for every cap this function is ever handed.
            let keepBest = Swift.max(splatCap, 0)
            if survivors.count > keepBest {
                Self.partitionHighest(&survivors, by: importance, count: keepBest)
                for k in keepBest..<survivors.count {
                    keep[survivors[k]] = false
                    outcome.trimmedToCap += 1
                }
            }
            survivorCount = keepBest
        }

        // --- 7. Compact --------------------------------------------------------
        if survivorCount != liveCount {
            var outSplats: [TrainerSplat] = []
            var outStats: [TrainerSplatStats] = []
            var outSH: [Float] = []
            var outM: [TrainerSplatGrad] = []
            var outV: [TrainerSplatGrad] = []
            var outSHM: [Float] = []
            var outSHV: [Float] = []
            var outTopK: [TrainerSamplingTopK] = []
            // All EIGHT, not three. The five that were missing grew by
            // doubling, so compacting a 300,000 splat model reallocated and
            // copied them about eighteen times each, every densification
            // pass that removed anything. That is 29 passes in a real run.
            outSplats.reserveCapacity(survivorCount)
            outStats.reserveCapacity(survivorCount)
            outSH.reserveCapacity(survivorCount * shPerSplat)
            outM.reserveCapacity(survivorCount)
            outV.reserveCapacity(survivorCount)
            outSHM.reserveCapacity(survivorCount * shPerSplat)
            outSHV.reserveCapacity(survivorCount * shPerSplat)
            outTopK.reserveCapacity(survivorCount)

            for i in 0..<liveCount where keep[i] {
                outSplats.append(splats[i])
                outStats.append(i < stats.count ? stats[i] : TrainerSplatStats())
                outM.append(i < adamM.count ? adamM[i] : TrainerSplatGrad())
                outV.append(i < adamV.count ? adamV[i] : TrainerSplatGrad())
                outTopK.append(i < topK.count ? topK[i] : TrainerSamplingTopK())
                let base = i * shPerSplat
                if base + shPerSplat <= sh.count {
                    outSH.append(contentsOf: sh[base..<(base + shPerSplat)])
                    outSHM.append(contentsOf: shM[base..<(base + shPerSplat)])
                    outSHV.append(contentsOf: shV[base..<(base + shPerSplat)])
                } else {
                    outSH.append(contentsOf: [Float](repeating: 0, count: shPerSplat))
                    outSHM.append(contentsOf: [Float](repeating: 0, count: shPerSplat))
                    outSHV.append(contentsOf: [Float](repeating: 0, count: shPerSplat))
                }
            }

            splats = outSplats
            stats = outStats
            sh = outSH
            adamM = outM
            adamV = outV
            shM = outSHM
            shV = outSHV
            topK = outTopK
            liveCount = splats.count
        }

        // --- 8. Write back -----------------------------------------------------
        if liveCount > resources.splatCapacity {
            try resources.resizeSplatCapacity(to: liveCount, keeping: 0)
        }
        resources.splats.writeArray(splats)
        resources.stats.writeArray(stats)
        resources.sh.writeArray(sh)
        resources.adamM.writeArray(adamM)
        resources.adamV.writeArray(adamV)
        resources.shAdamM.writeArray(shM)
        resources.shAdamV.writeArray(shV)
        resources.samplingTopK.writeArray(topK)

        outcome.splatCountAfter = liveCount
        return outcome
    }

    // MARK: - Bounded selection
    //
    // Densification only ever reads the BEST few of a list that is very
    // nearly the whole population. Ordering the rest of it is heat with no
    // output, and heat is the thing that shortens these runs.

    /// The `count` highest-scoring entries of `indices`, in descending score
    /// order, without ordering anything below them.
    ///
    /// Returns a new array rather than sorting in place because both callers
    /// want to keep the count of the full candidate list for the census while
    /// working with the truncated one.
    private static func selectHighest(
        _ indices: [Int], by score: [Float], count: Int
    ) -> [Int] {
        let wanted = Swift.min(Swift.max(count, 0), indices.count)
        guard wanted > 0 else { return [] }
        var working = indices
        partitionHighest(&working, by: score, count: wanted)
        var top = Array(working[0..<wanted])
        top.sort { score[$0] > score[$1] }
        return top
    }

    /// Rearranges `a` so its first `k` entries are the `k` highest-scoring,
    /// in no particular order among themselves. Expected linear time.
    ///
    /// Quickselect with a median-of-three pivot and a Hoare partition. Hoare
    /// is chosen over Lomuto deliberately: the score array here is long-tailed
    /// with a large block of EQUAL values (every Gaussian that got the same
    /// tiny gradient, and, on the donor key, every Gaussian sitting at the
    /// same clamped opacity), and Hoare splits a run of equal keys down the
    /// middle while Lomuto degenerates to quadratic on it.
    ///
    /// The depth budget is the standard introselect guard: if the pivots keep
    /// coming out badly, the remaining range is sorted outright rather than
    /// allowed to run quadratic on a phone. It has to be an explicit fallback
    /// and not a promise, because "expected linear" is not a bound.
    private static func partitionHighest(
        _ a: inout [Int], by score: [Float], count k: Int
    ) {
        let n = a.count
        guard k > 0, k < n else { return }
        var lo = 0
        var hi = n - 1
        // 2 * floor(log2(n)) + 2, computed by shifting so no floating point
        // maths library call is involved.
        var budget = 0
        var m = n
        while m > 0 {
            m >>= 1
            budget += 1
        }
        budget *= 2
        while lo < hi && budget > 0 {
            budget -= 1
            let mid = lo + (hi - lo) / 2
            let x = score[a[lo]]
            let y = score[a[mid]]
            let z = score[a[hi]]
            // Median of the three, so the pivot is always a value that is
            // actually present in [lo, hi]. That is what keeps both scans
            // below inside the range without an extra bounds test per step.
            let lowPair = Swift.min(x, y)
            let highPair = Swift.max(x, y)
            let pivot = Swift.max(lowPair, Swift.min(highPair, z))
            var i = lo
            var j = hi
            while i <= j {
                while score[a[i]] > pivot { i += 1 }
                while score[a[j]] < pivot { j -= 1 }
                if i <= j {
                    a.swapAt(i, j)
                    i += 1
                    j -= 1
                }
            }
            if k - 1 <= j {
                hi = j
            } else if k - 1 >= i {
                lo = i
            } else {
                // The split landed exactly on the boundary: everything at or
                // before k-1 is already at or above everything after it.
                return
            }
        }
        if lo < hi {
            var tail = Array(a[lo...hi])
            tail.sort { score[$0] > score[$1] }
            a.replaceSubrange(lo...hi, with: tail)
        }
    }

    // MARK: - Saying it out loud

    /// Says, from inside this function, the one thing a caller can look at and
    /// still miss.
    ///
    /// The training loop already prints every pass, including the ones that
    /// changed nothing, and shouts when growth was allowed and there was room
    /// under the cap. Two cases slip past that and both of them are the fault
    /// this file exists to catch:
    ///
    ///   1. A pass that PRUNED or CARVED something while scoring produced no
    ///      candidates. It has a summary, so it prints "removed 12" at info
    ///      level and reads as a perfectly healthy pass, while densification
    ///      is in fact dead.
    ///   2. A pass at the cap. `headroomAtStart` is zero there, so the loud
    ///      branch at the call site does not fire, and "no room under the cap"
    ///      is a completely reasonable-looking explanation to put in front of
    ///      a gradient signal that has stopped.
    ///
    /// This is called on EVERY exit from `run`, including the early returns
    /// and the throwing ones, so it cannot be skipped by a path someone adds
    /// later.
    ///
    /// Built with `+=` on a plain `String` one clause at a time rather than as
    /// one long `+` chain or one long interpolation. A fifteen-operand string
    /// expression is the shape the Swift type checker gives up on, and this
    /// project's only compiler is CI.
    private static func announce(_ outcome: TrainerDensifyOutcome) {
        if outcome.scoringProducedNoCandidates {
            var detail = "Densification scored "
            detail += String(outcome.splatsScored)
            detail += " points and not one became a candidate: "
            detail += String(outcome.splatsWithNonZeroScore)
            detail += " scored above zero and "
            detail += String(outcome.candidatesAfterVisibilityFilter)
            detail += " survived the visibility filter. Room for "
            detail += String(outcome.growthAllowance)
            detail += " under a cap of "
            detail += String(outcome.splatCapInForce)
            detail += " went unused. Verdict: "
            detail += outcome.growthVerdict.rawValue
            detail += ". Nothing can be created while this holds."
            TrainerLog.densify.error("\(detail, privacy: .public)")
            return
        }
        if outcome.growthVerdict == .atTheCapNoDonors {
            var detail = "Densification is at its cap of "
            detail += String(outcome.splatCapInForce)
            detail += " with "
            detail += String(outcome.candidatesAfterVisibilityFilter)
            detail += " candidates and only "
            detail += String(outcome.relocationDonorsAvailable)
            detail += " points faint enough to move, so this pass changed nothing."
            TrainerLog.densify.notice("\(detail, privacy: .public)")
        }
    }

    /// Which local axis a split runs along.
    ///
    /// Normally the longest: that is the direction the Gaussian is most likely
    /// to be straddling something. For a Gaussian flagged as lying on a
    /// detected 3D edge curve, the longest axis is ALONG the edge, and cutting
    /// a needle in half lengthwise leaves two needles in the same wrong place;
    /// what is wanted is a split across the edge normal, which is the second
    /// longest axis. That is the SAD-GS idea, expressed in the local frame the
    /// Gaussian already carries.
    /// A reproducible unit-normal triple for the split offset.
    ///
    /// The reference samples the child offset from N(0, Sigma). A real random
    /// number would make two runs of the same build produce different models,
    /// which would destroy the ability to read a 0.3 dB change against a
    /// 0.4 dB noise band, so this is a hash of the splat index put through
    /// Box-Muller: the same everywhere, different per Gaussian, and with the
    /// distribution the geometry actually wants.
    static func splitNoise(seed: UInt32) -> SIMD3<Float> {
        func bits(_ x: UInt32) -> UInt32 {
            var h = x &* 0x9E37_79B9
            h ^= h >> 16
            h = h &* 0x85EB_CA6B
            h ^= h >> 13
            h = h &* 0xC2B2_AE35
            h ^= h >> 16
            return h
        }
        // Two uniforms in (0, 1]; the 1e-7 floor keeps log() finite.
        let u1 = Swift.max(Float(bits(seed)) / Float(UInt32.max), 1e-7)
        let u2 = Float(bits(seed &+ 0x1234_5678)) / Float(UInt32.max)
        let u3 = Swift.max(Float(bits(seed &+ 0x9ABC_DEF0)) / Float(UInt32.max), 1e-7)
        let r = sqrtf(-2 * logf(u1))
        let r2 = sqrtf(-2 * logf(u3))
        return SIMD3<Float>(
            r * cosf(2 * Float.pi * u2),
            r * sinf(2 * Float.pi * u2),
            r2 * cosf(2 * Float.pi * u2)
        )
    }

    private func splitAxis(for splat: TrainerSplat, linearScale: SIMD3<Float>) -> Int {
        var order = [0, 1, 2]
        order.sort { linearScale[$0] > linearScale[$1] }
        let onEdge = (splat.flags & TrainerSplatFlag.onEdgeCurve) != 0
        return onEdge ? order[1] : order[0]
    }
}
