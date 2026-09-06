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

    var changedTopology: Bool {
        cloned + split + prunedLowOpacity + prunedOversized
            + prunedNonFinite + carvedFromEmptySpace + trimmedToCap > 0
    }

    /// One plain sentence, only when there is something worth saying.
    var summary: String? {
        var parts: [String] = []
        if cloned + split > 0 { parts.append("added \(cloned + split) points of detail") }
        if relocated > 0 { parts.append("moved \(relocated) unused points somewhere useful") }
        let removed = prunedLowOpacity + prunedOversized + prunedNonFinite
            + carvedFromEmptySpace + trimmedToCap
        if removed > 0 { parts.append("removed \(removed)") }
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
        var candidates: [Int] = []
        var scoredAboveZero = 0
        for i in 0..<splatCount where score[i] > 0 {
            scoredAboveZero += 1
            // A Gaussian nothing has seen has nothing to say about where
            // detail is missing.
            if stats[i].visAccum <= 0 { continue }
            candidates.append(i)
        }
        candidates.sort { score[$0] > score[$1] }
        // Two counts, not one: "nothing scored" and "everything that scored
        // was invisible" are different faults with different fixes, and a
        // single "candidates: 0" cannot tell them apart.
        outcome.splatsWithNonZeroScore = scoredAboveZero
        outcome.candidatesAfterVisibilityFilter = candidates.count

        let headroom = Swift.max(splatCap - splatCount, 0)
        let growthAllowance = allowGrowth
            ? Swift.min(headroom, Int(Float(splatCap) * tuning.maxGrowthFractionPerPass))
            : 0
        outcome.growthAllowance = growthAllowance

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

                if largest > splitThresholdScale {
                    // SPLIT: over-sized and under-explained. Two children,
                    // offset along the split axis, shrunk ONLY along it.
                    let axis = splitAxis(for: parent, linearScale: linear)
                    let rotation = TrainerMath.rotationMatrix(parent.rotation)
                    let direction = rotation[axis]
                    let offset = direction * (linear[axis] * tuning.splitOffsetSigma)

                    var shrunk = scale
                    shrunk[axis] = scale[axis] - logf(tuning.splitShrink)

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
            var donors: [Int] = []
            for i in 0..<splatCount {
                let opacity = TrainerMath.sigmoid(splats[i].opacityLogit)
                if opacity < tuning.relocationDonorOpacity || stats[i].visAccum <= 0 {
                    donors.append(i)
                }
            }
            donors.sort { TrainerMath.sigmoid(splats[$0].opacityLogit)
                < TrainerMath.sigmoid(splats[$1].opacityLogit) }
            outcome.relocationDonorsAvailable = donors.count

            let allowance = Swift.min(
                donors.count,
                Swift.min(candidates.count, Int(Float(splatCount) * tuning.maxRelocationFractionPerPass))
            )
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
            if TrainerMath.sigmoid(splat.opacityLogit) < tuning.pruneOpacity {
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
            var ranked: [(index: Int, importance: Float)] = []
            ranked.reserveCapacity(survivorCount)
            for i in 0..<liveCount where keep[i] {
                let opacity = TrainerMath.sigmoid(splats[i].opacityLogit)
                let visibility = i < stats.count ? Swift.max(stats[i].visAccum, 0) : 0
                ranked.append((i, opacity * (1 + visibility)))
            }
            ranked.sort { $0.importance > $1.importance }
            for k in splatCap..<ranked.count {
                keep[ranked[k].index] = false
                outcome.trimmedToCap += 1
            }
            survivorCount = splatCap
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
            outSplats.reserveCapacity(survivorCount)
            outStats.reserveCapacity(survivorCount)
            outSH.reserveCapacity(survivorCount * shPerSplat)

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

    /// Which local axis a split runs along.
    ///
    /// Normally the longest: that is the direction the Gaussian is most likely
    /// to be straddling something. For a Gaussian flagged as lying on a
    /// detected 3D edge curve, the longest axis is ALONG the edge, and cutting
    /// a needle in half lengthwise leaves two needles in the same wrong place;
    /// what is wanted is a split across the edge normal, which is the second
    /// longest axis. That is the SAD-GS idea, expressed in the local frame the
    /// Gaussian already carries.
    private func splitAxis(for splat: TrainerSplat, linearScale: SIMD3<Float>) -> Int {
        var order = [0, 1, 2]
        order.sort { linearScale[$0] > linearScale[$1] }
        let onEdge = (splat.flags & TrainerSplatFlag.onEdgeCurve) != 0
        return onEdge ? order[1] : order[0]
    }
}
