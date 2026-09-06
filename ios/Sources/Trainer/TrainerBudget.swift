//
//  TrainerBudget.swift
//  Trainer
//
//  THE BUDGET GOVERNOR (F10). It may LOWER the budget. It may NEVER raise it.
//
//  `SplatTrainer` in Core/Contracts.swift states the rule outright: "The
//  trainer owns the right to LOWER `budget` as it measures real memory and
//  heat, and reports having done so in `TrainerProgress`. It may never raise
//  it." Everything below exists to make that literally true rather than
//  aspirationally true:
//
//   * `current` starts as a copy of the budget the caller handed in and is the
//     ONLY mutable budget in the trainer.
//   * Every field is written through `lower(...)`, which takes the minimum of
//     the proposed value and what is already there. A step-up is not rejected
//     with an error, it is arithmetically impossible.
//   * `ceiling` keeps the caller's original so a reviewer can see, at any
//     moment, both what was asked for and what is actually being run.
//
//  THE STEP-DOWN LADDER, in this order and no other:
//
//    1. SPLAT CAP. Costs the least visible quality per byte saved. Half a
//       million Gaussians and four hundred thousand look nearly identical from
//       the walked path; the difference is entirely in views the user cannot
//       reach anyway.
//    2. RESOLUTION. Cheaper still in memory but it costs sharpness the user
//       WILL see, so it comes second, and it steps through the spec's own
//       ladder (720, 600, 480, 384) rather than by an arbitrary factor.
//    3. ITERATIONS. Last, because it is the only one that shortens the run
//       rather than shrinking it, and a scan that finishes blurry is worth
//       more than a scan that does not finish.
//
//  NOTHING HERE ASSUMES A SCENE FITS. Every decision is taken against a
//  measurement: `os_proc_available_memory()` through `DeviceMemoryFacts`, and
//  `TrainerResources.residentBytes`, which is what Metal actually reserved.
//

import Foundation
import Metal

/// One budget reduction, with enough detail to say what happened in plain
/// language. Reported through `TrainerProgress.message`, which is the contract.
struct TrainerBudgetChange {
    enum Kind {
        case splatCap(from: Int, to: Int)
        case resolution(from: Int, to: Int)
        case iterations(from: Int, to: Int)
    }

    var kind: Kind
    var reason: Reason

    // --- WHEN it happened ----------------------------------------------------
    //
    // A reduction without a timestamp is half a fact. "The splat cap was cut
    // to 137,000" is not actionable; "the splat cap was cut to 137,000 at
    // iteration 100 while 182,000 Gaussians were alive" is a bug report,
    // because the second number says the cut deleted real geometry and left
    // zero headroom for the rest of the run. The governor stamps these from
    // whatever the loop last told it through `mark(...)`; they are -1 / 0 for
    // a reduction taken before any loop existed, which is the sizing pass in
    // `initialSplatCap` and is a genuinely different moment.
    var atSliceIndex: Int = -1
    var atIteration: Int = -1
    var liveSplatCount: Int = 0

    /// True when a splat-cap cut landed at or below the population that was
    /// already alive. That is the one-way ratchet from the first real scan:
    /// `applyBudgetChange` then trims real Gaussians to fit and rebuilds the
    /// buffer smaller, and `headroom = cap - count` is 0 afterwards, so
    /// densification is off for good.
    var landedAtOrBelowLivePopulation: Bool {
        switch kind {
        case .splatCap(_, let to):
            return liveSplatCount > 0 && to <= liveSplatCount
        case .resolution, .iterations:
            return false
        }
    }

    enum Reason {
        case thermal(ThermalLevel)
        case memory(residentBytes: UInt64, availableBytes: UInt64)
        case memoryCeiling(residentBytes: UInt64, ceilingBytes: UInt64)

        var plainCause: String {
            switch self {
            case .thermal:
                return "your phone is getting warm"
            case .memory, .memoryCeiling:
                return "this phone has less spare memory than this scan wanted"
            }
        }

        // --- For the census ---------------------------------------------------

        /// A short, stable phrase naming the cause. Separate from
        /// `plainCause`, which is written to be read out loud mid-sentence to
        /// the user and would not survive being reworded for tone.
        /// The heat phrase is `TrainerCensusBudgetReduction.heatReason` rather
        /// than a literal, because `model/census.json` decides whether a cap
        /// cut was thermal by matching this exact string. Two copies of the
        /// same phrase in two files is how a heat report silently becomes a
        /// zero when somebody rewords one of them.
        var censusReason: String {
            switch self {
            case .thermal: return TrainerCensusBudgetReduction.heatReason
            case .memory: return "memory share"
            case .memoryCeiling: return "memory ceiling"
            }
        }

        var thermalLevel: ThermalLevel? {
            if case .thermal(let level) = self { return level }
            return nil
        }

        var residentBytes: UInt64? {
            switch self {
            case .thermal: return nil
            case .memory(let resident, _): return resident
            case .memoryCeiling(let resident, _): return resident
            }
        }

        /// What `residentBytes` was measured against: the process's remaining
        /// allocation for `.memory`, the budget's own ceiling for
        /// `.memoryCeiling`.
        var comparedAgainstBytes: UInt64? {
            switch self {
            case .thermal: return nil
            case .memory(_, let available): return available
            case .memoryCeiling(_, let ceiling): return ceiling
            }
        }
    }

    // MARK: - The change itself, as three plain numbers

    /// "splatCap", "renderLongEdgePixels" or "iterations". The field name as
    /// it appears in `TrainingBudget`, so a reader can go straight to it.
    var changedField: String {
        switch kind {
        case .splatCap: return "splatCap"
        case .resolution: return "renderLongEdgePixels"
        case .iterations: return "iterations"
        }
    }

    var fromValue: Int {
        switch kind {
        case .splatCap(let from, _), .resolution(let from, _), .iterations(let from, _):
            return from
        }
    }

    var toValue: Int {
        switch kind {
        case .splatCap(_, let to), .resolution(_, let to), .iterations(_, let to):
            return to
        }
    }

    /// A sentence for a person, not a log line.
    var message: String {
        switch kind {
        case .splatCap(_, let to):
            return "Because \(reason.plainCause), this is now aiming for about "
                + "\(TrainerBudgetGovernor.round(to)) detail points instead."
        case .resolution(_, let to):
            return "Because \(reason.plainCause), the practice pictures are now "
                + "\(to) pixels across instead."
        case .iterations(_, let to):
            return "Because \(reason.plainCause), this will finish after "
                + "\(TrainerBudgetGovernor.round(to)) rounds instead of carrying on."
        }
    }
}

/// What the governor wants the loop to do right now.
enum TrainerThermalVerdict {
    case run
    /// Shed work but keep going.
    case degrade
    /// Stop and wait. The loop emits `.pausedThermal` and polls until this
    /// stops being the answer.
    case pause
    /// Stop for good and keep what has been built.
    case abort
}

final class TrainerBudgetGovernor {

    /// The budget the caller asked for. Never modified. Kept so the honest
    /// comparison "asked for X, running Y" is always available.
    let ceiling: TrainingBudget

    /// The budget actually in force.
    private(set) var current: TrainingBudget

    /// Every reduction made, in order. Written into the finished model's
    /// `budgetUsed` context and into the log.
    private(set) var changes: [TrainerBudgetChange] = []

    /// The spec's resolution ladder (F10: 480-720 px), plus one step below it
    /// for a phone that is genuinely struggling. Descending.
    static let resolutionLadder = [720, 600, 480, 384]

    private var lastThermalPoll = Date.distantPast
    private var cachedThermal: ThermalLevel = .nominal

    // MARK: - Where the run is right now
    //
    // Three integers the loop keeps up to date so that every reduction can
    // record WHEN it happened and what the population was at that moment. The
    // governor does not read them for any decision; they exist so the census
    // can say "cut to 137,000 while 182,000 were alive" instead of "cut to
    // 137,000", which is the difference between a fact and a bug report.
    //
    // -1 means "no loop has started yet", which is where `initialSplatCap`
    // takes its reduction from, and that is a real and distinct moment rather
    // than iteration zero of slice zero.
    private(set) var currentSliceIndex = -1
    private(set) var currentIteration = -1
    private(set) var currentSplatCount = 0

    /// Called once per iteration by the training loop. Three integer stores;
    /// it does no work and takes no lock.
    func mark(sliceIndex: Int, iteration: Int, splatCount: Int) {
        currentSliceIndex = sliceIndex
        currentIteration = iteration
        currentSplatCount = splatCount
    }

    /// How much of what the process may still allocate the trainer is willing
    /// to be holding. Above this, the budget comes down. Deliberately well
    /// under 1: the rasteriser's tile lists, the decoded frames and the OS's
    /// own headroom all live in the same pot.
    private let memoryUseFraction: Float = 0.6

    init(budget: TrainingBudget) {
        self.ceiling = budget
        self.current = budget
    }

    // MARK: - The one-way valve

    /// Applies a reduction. Takes the minimum every time, so a caller that
    /// asks for a larger value gets no change and no error: raising the budget
    /// is not a thing that can happen, rather than a thing that is checked for.
    @discardableResult
    private func lower(_ change: TrainerBudgetChange.Kind, reason: TrainerBudgetChange.Reason) -> TrainerBudgetChange? {
        switch change {
        case .splatCap(_, let proposed):
            let target = Swift.min(proposed, current.splatCap)
            guard target < current.splatCap, target > 0 else { return nil }
            var recorded = TrainerBudgetChange(
                kind: .splatCap(from: current.splatCap, to: target), reason: reason
            )
            stamp(&recorded)
            current.splatCap = target
            changes.append(recorded)
            TrainerLog.budget.notice(
                "Splat cap lowered to \(target) from \(recorded.fromValue)"
            )
            return recorded

        case .resolution(_, let proposed):
            let target = Swift.min(proposed, current.renderLongEdgePixels)
            guard target < current.renderLongEdgePixels, target > 0 else { return nil }
            var recorded = TrainerBudgetChange(
                kind: .resolution(from: current.renderLongEdgePixels, to: target), reason: reason
            )
            stamp(&recorded)
            current.renderLongEdgePixels = target
            changes.append(recorded)
            TrainerLog.budget.notice("Render long edge lowered to \(target)")
            return recorded

        case .iterations(_, let proposed):
            let target = Swift.min(proposed, current.iterations)
            guard target < current.iterations, target > 0 else { return nil }
            var recorded = TrainerBudgetChange(
                kind: .iterations(from: current.iterations, to: target), reason: reason
            )
            stamp(&recorded)
            current.iterations = target
            changes.append(recorded)
            TrainerLog.budget.notice("Iteration count lowered to \(target)")
            return recorded
        }
    }

    /// Writes the slice, iteration and live count onto a reduction before it
    /// is recorded, and says out loud when a splat-cap cut has landed at or
    /// below the population that already exists.
    ///
    /// That last line is the warning nobody got the first time. The ratchet
    /// that destroyed the owner's first scan cut the cap under the live count,
    /// deleted the difference and then left zero headroom, and the only thing
    /// in the log was a cheerful "Splat cap lowered to N".
    private func stamp(_ change: inout TrainerBudgetChange) {
        change.atSliceIndex = currentSliceIndex
        change.atIteration = currentIteration
        change.liveSplatCount = currentSplatCount
        let live = change.liveSplatCount
        let to = change.toValue
        if change.landedAtOrBelowLivePopulation {
            if to < live {
                TrainerLog.budget.error(
                    "Splat cap cut to \(to) while \(live) Gaussians are alive: \(live - to) real Gaussians will be deleted and densification will have no headroom"
                )
            } else {
                TrainerLog.budget.notice(
                    "Splat cap cut to \(to), exactly the live count: nothing is deleted, but densification has no headroom until something frees space"
                )
            }
        }
    }

    // MARK: - Thermal

    /// Polls no more often than the policy's sample interval. Thermal state is
    /// a coarse, slow signal and reading it every iteration is noise plus
    /// battery.
    func thermalVerdict(now: Date = Date()) -> (verdict: TrainerThermalVerdict, level: ThermalLevel) {
        let policy = current.thermalPolicy
        if now.timeIntervalSince(lastThermalPoll) >= policy.sampleIntervalSeconds {
            cachedThermal = ThermalLevel(ProcessInfo.processInfo.thermalState)
            lastThermalPoll = now
        }
        let level = cachedThermal
        if level >= policy.abortAt { return (.abort, level) }
        if level >= policy.pauseAt { return (.pause, level) }
        if level >= policy.degradeAt { return (.degrade, level) }
        return (.run, level)
    }

    /// The current cached level, without polling. For the progress tick, which
    /// runs far more often than the poll interval.
    var thermalLevel: ThermalLevel { cachedThermal }

    /// One step down the ladder because of heat. Returns what it changed, or
    /// nil when there is nothing left to give up, which is itself worth
    /// knowing: at that point the only remaining response is to pause.
    /// The gap a splat-cap cut must always leave above the live population, so
    /// densification survives the cut.
    ///
    /// `TrainerDensifier` computes `headroom = max(splatCap - splatCount, 0)`
    /// and adds nothing when that is zero, so a cap sitting on the population
    /// is indistinguishable from densification being switched off. An eighth of
    /// the original ceiling is enough for several growth passes at the
    /// densifier's own per-pass fraction, and it scales with the scene the way
    /// the ceiling does.
    static func minimumGrowthHeadroom(forCeiling ceilingCap: Int) -> Int {
        Swift.max(10_000, ceilingCap / 8)
    }

    func degradeForHeat(level: ThermalLevel, currentSplatCount: Int) -> TrainerBudgetChange? {
        let reason = TrainerBudgetChange.Reason.thermal(level)

        // 1. Splat cap. Cut the CEILING, and never below the living population.
        //
        // This used to read `min(current.splatCap, currentSplatCount) * 0.75`,
        // and that was a one-way ratchet that destroyed the scan. The live
        // count is always below the ceiling early in a run (the seeder aims for
        // splatCap/2), so the min() picked the LIVE count, and 0.75 of it
        // landed BELOW the population that already existed. `applyBudgetChange`
        // then trimmed real Gaussians to fit and rebuilt the GPU buffer at the
        // smaller size, so the geometry was physically deleted rather than
        // merely disallowed. Worse, `headroom = splatCap - splatCount` was then
        // exactly 0, which switched densification off for the rest of the run.
        // One warm phone in the first minute and the model could never grow
        // again, which is what "a three minute scan that looks like nothing"
        // actually was.
        //
        // Cutting the ceiling gives back exactly the memory a thermal cut is
        // for, because `resizeSplatCapacity` allocates what the cap says AFTER
        // the cut. An unused ceiling costs nothing to hold, so lowering one the
        // run has not reached is a real response, not a fake one.
        //
        // The floor is scene-derived rather than a flat 20,000. `ceiling`
        // already scales with scene extent (150k for one object, 300k for a
        // room, 500k for a floor), so a quarter of it keeps a surface a
        // surface. A model may get smaller under heat; it must never get so
        // small it stops describing the room.
        // AND NEVER ONTO IT EITHER. Flooring at `currentSplatCount` was only
        // half the fix, and the half that was missing is the one that matters.
        // Landing the cap exactly ON the live population stops the deletion but
        // still leaves `headroom = splatCap - splatCount` at zero, which
        // switches densification off just as completely, and because a change
        // was returned the ladder stopped here and rungs 2 and 3 never ran, so
        // the phone shed no actual work either. A cut that frees nothing and
        // disables growth is worse than no cut at all.
        //
        // So the cap may come down, but never nearer the live population than
        // the room densification needs to keep working. When there is no room
        // left to give, this rung returns nil ON PURPOSE and the ladder moves
        // on to resolution and then iterations, which shed real work.
        let splatFloor = Swift.max(20_000, ceiling.splatCap / 4)
        let capTarget = Swift.max(
            Int(Float(current.splatCap) * 0.75),
            currentSplatCount + Self.minimumGrowthHeadroom(forCeiling: ceiling.splatCap),
            splatFloor
        )
        if let change = lower(.splatCap(from: current.splatCap, to: capTarget), reason: reason) {
            return change
        }

        // 2. Resolution, one rung.
        if let next = Self.nextResolutionDown(from: current.renderLongEdgePixels),
           let change = lower(.resolution(from: current.renderLongEdgePixels, to: next), reason: reason)
        {
            return change
        }

        // 3. Iterations, last.
        let iterationTarget = Swift.max(Int(Float(current.iterations) * 0.8), 200)
        return lower(.iterations(from: current.iterations, to: iterationTarget), reason: reason)
    }

    // MARK: - Memory

    /// What the trainer is actually holding, and what the process may still
    /// allocate. Both measured; neither derived from `physicalMemory`.
    struct MemoryReading {
        var residentBytes: UInt64
        var availableBytes: UInt64
        var availableIsEstimated: Bool
        var ceilingBytes: UInt64

        var overCeiling: Bool { residentBytes > ceilingBytes && ceilingBytes > 0 }
        /// True when what is held is a large share of what is left to give.
        func overShare(_ fraction: Float) -> Bool {
            guard availableBytes > 0 else { return false }
            return Float(residentBytes) > Float(availableBytes) * fraction
        }
    }

    func measureMemory(resources: TrainerResources?) -> MemoryReading {
        let facts = DeviceMemoryFacts.probe()
        return MemoryReading(
            residentBytes: resources?.residentBytes ?? 0,
            availableBytes: facts.availableBytes,
            availableIsEstimated: facts.availableIsEstimated,
            ceilingBytes: current.memoryCeilingBytes
        )
    }

    /// Decides whether memory pressure requires a step down, and takes one
    /// step if it does. The step is sized from the measurement rather than
    /// being a fixed fraction: if the run is holding 1.4x what it should, the
    /// cap comes down by roughly that factor, so one step is usually enough.
    func degradeForMemory(
        reading: MemoryReading,
        currentSplatCount: Int,
        shCoefficientCount: Int,
        pixelCount: Int
    ) -> TrainerBudgetChange? {

        let overCeiling = reading.overCeiling
        let overShare = reading.overShare(memoryUseFraction)
        guard overCeiling || overShare else { return nil }

        let reason: TrainerBudgetChange.Reason = overCeiling
            ? .memoryCeiling(residentBytes: reading.residentBytes, ceilingBytes: reading.ceilingBytes)
            : .memory(residentBytes: reading.residentBytes, availableBytes: reading.availableBytes)

        // What is a safe number of bytes to be holding?
        let safeBytes: UInt64
        if overCeiling {
            safeBytes = reading.ceilingBytes
        } else {
            safeBytes = UInt64(Float(reading.availableBytes) * memoryUseFraction * 0.9)
        }

        // Per-Gaussian and per-pixel costs, measured from the layouts.
        let perSplat = UInt64(TrainerResources.bytesPerSplat(shCoefficientCount: shCoefficientCount))
        let perPixel = UInt64(TrainerResources.bytesPerPixel())
        let pixelBytes = perPixel * UInt64(Swift.max(pixelCount, 1))

        // 1. Splat cap, sized to fit what is left after the pixel buffers.
        if safeBytes > pixelBytes, perSplat > 0 {
            let affordable = Int((safeBytes - pixelBytes) / perSplat)

            // MEMORY IS NOT HEAT, and this rung is deliberately not the same
            // as `degradeForHeat`'s.
            //
            // Heat can be answered by shedding work elsewhere, so that rung
            // refuses to cut onto the live population and lets resolution and
            // iterations do the shedding. Memory cannot: the bytes are already
            // held, and if the phone genuinely will not hold this model then
            // trimming real Gaussians is the honest answer rather than a bug.
            //
            // Two things this used to get wrong. It took `currentSplatCount`
            // and then explicitly discarded it, so it could not tell whether a
            // cut was a harmless ceiling trim or the deletion of live geometry,
            // and the census had nothing to report. And `current.splatCap - 1`
            // manufactured a one-splat "reduction" whenever the phone could
            // actually afford MORE than the current cap, which freed nothing,
            // returned a change, and stopped the ladder before the rungs that
            // would have freed something real.
            let capTarget = Swift.max(Swift.min(affordable, current.splatCap), 20_000)

            if capTarget <= currentSplatCount, currentSplatCount > 0 {
                // This one really will delete geometry. Say so plainly here as
                // well as in the census, because a silent trim is how the
                // original ratchet hid for every run this app has ever done.
                TrainerLog.budget.notice(
                    """
                    Memory cut lands at \(capTarget) with \(currentSplatCount) \
                    Gaussians alive: real geometry will be trimmed, and \
                    densification has no room left until the cap rises.
                    """
                )
            }

            if capTarget < current.splatCap,
               let change = lower(.splatCap(from: current.splatCap, to: capTarget), reason: reason)
            {
                return change
            }
        } else {
            // The pixel buffers alone do not fit. That is a resolution problem
            // and no amount of splat cutting fixes it.
            if let next = Self.nextResolutionDown(from: current.renderLongEdgePixels),
               let change = lower(
                   .resolution(from: current.renderLongEdgePixels, to: next), reason: reason
               )
            {
                return change
            }
        }

        // 2. Resolution.
        if let next = Self.nextResolutionDown(from: current.renderLongEdgePixels),
           let change = lower(.resolution(from: current.renderLongEdgePixels, to: next), reason: reason)
        {
            return change
        }

        // 3. Iterations. A shorter run does not save a single byte of the
        // splat buffers, so this is a last resort that only helps by ending
        // the pressure sooner.
        // `currentSplatCount` is genuinely not needed by THIS rung (shortening
        // a run frees no buffer bytes), but it is used by rung 1 above, which
        // is the point: it used to be discarded here for the whole function,
        // which is why a memory cut could not tell a ceiling trim from the
        // deletion of live geometry.
        let iterationTarget = Swift.max(Int(Float(current.iterations) * 0.8), 200)
        return lower(.iterations(from: current.iterations, to: iterationTarget), reason: reason)
    }

    // MARK: - Sizing

    /// The splat cap that actually fits, given a measured memory reading and
    /// the real per-Gaussian cost for this SH degree. Called ONCE before any
    /// allocation, so the first allocation is already inside the truth rather
    /// than being cut back after the fact.
    ///
    /// This is where the owner's "never assume a scene fits" lands: the number
    /// returned is the smaller of what the caller asked for and what the
    /// device measured, every time.
    func initialSplatCap(shCoefficientCount: Int, pixelCount: Int) -> Int {
        let facts = DeviceMemoryFacts.probe()
        let budgetCeiling = current.memoryCeilingBytes > 0
            ? Swift.min(current.memoryCeilingBytes, facts.availableBytes)
            : facts.availableBytes
        let usable = UInt64(Float(budgetCeiling) * memoryUseFraction)

        let perPixel = UInt64(TrainerResources.bytesPerPixel())
        let pixelBytes = perPixel * UInt64(Swift.max(pixelCount, 1))
        guard usable > pixelBytes else { return Swift.min(current.splatCap, 20_000) }

        let perSplat = UInt64(
            Swift.max(TrainerResources.bytesPerSplat(shCoefficientCount: shCoefficientCount), 1)
        )
        let affordable = Int((usable - pixelBytes) / perSplat)
        let capped = Swift.max(Swift.min(current.splatCap, affordable), 5_000)

        if capped < current.splatCap {
            let reason = TrainerBudgetChange.Reason.memory(
                residentBytes: 0, availableBytes: facts.availableBytes
            )
            lower(.splatCap(from: current.splatCap, to: capped), reason: reason)
        }
        return current.splatCap
    }

    /// Render size for a long edge, from the capture's aspect ratio. Never
    /// larger than the source frame: upsampling a photo to train on it adds
    /// no information and costs the square of the factor in memory.
    static func renderSize(
        forLongEdge longEdge: Int,
        intrinsics: CameraIntrinsics
    ) -> TrainerRenderSize {
        let sourceLong = Swift.max(intrinsics.width, intrinsics.height)
        let sourceShort = Swift.min(intrinsics.width, intrinsics.height)
        guard sourceLong > 0, sourceShort > 0 else {
            return TrainerRenderSize(width: longEdge, height: longEdge)
        }
        let effectiveLong = Swift.min(longEdge, sourceLong)
        let shortEdge = Swift.max(
            Int((Float(effectiveLong) * Float(sourceShort) / Float(sourceLong)).rounded()), 1
        )
        return intrinsics.width >= intrinsics.height
            ? TrainerRenderSize(width: effectiveLong, height: shortEdge)
            : TrainerRenderSize(width: shortEdge, height: effectiveLong)
    }

    static func nextResolutionDown(from current: Int) -> Int? {
        for step in resolutionLadder where step < current { return step }
        return nil
    }

    /// Rounds a count to something a person would say out loud.
    static func round(_ value: Int) -> String {
        if value >= 100_000 { return "\((value / 10_000) * 10)000" }
        if value >= 10_000 { return "\((value / 1_000) * 1)000" }
        return "\(value)"
    }
}


