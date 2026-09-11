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
//  measurement, and the measurement is of the PROCESS: two readings of
//  `os_proc_available_memory()` through `DeviceMemoryFacts`, one taken before
//  this run had allocated anything and one taken now, with the fall between
//  them as what the run has consumed. `TrainerResources.residentBytes` is
//  still read, but only to size the headroom a cut needs in order to be able
//  to perform itself. It is the trainer's own Metal buffers and nothing else:
//  on the desk scan that was about 37 MB inside a 3.26 GB app footprint, so
//  no threshold is compared against it any more.
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
        /// The process is close enough to its allocation limit that the next
        /// large allocation is the one that gets it killed. Kept separate from
        /// `.memory` because it is a different question with a different
        /// answer: `.memory` asks whether this run is holding too big a share
        /// of what is left, and this asks whether there is still enough left
        /// to survive one buffer rebuild. The second is the one that decides
        /// whether the scan finishes at all, so the census has to be able to
        /// tell them apart afterwards.
        case memoryHeadroom(residentBytes: UInt64, remainingBytes: UInt64)

        var plainCause: String {
            switch self {
            case .thermal:
                return "your phone is getting warm"
            case .memory, .memoryCeiling, .memoryHeadroom:
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
            case .memoryHeadroom: return "memory headroom"
            }
        }

        var thermalLevel: ThermalLevel? {
            if case .thermal(let level) = self { return level }
            return nil
        }

        /// For the three memory reasons, what the RUN had consumed
        /// process-wide at the moment the reduction was taken, not the
        /// trainer's own Metal buffers. The name is kept as it is because it
        /// is a field name in `model/train_census.json`; the meaning changed
        /// when `TrainerBudgetGovernor.MemoryReading` started measuring the
        /// process instead of the buffer list. Zero for the sizing pass in
        /// `initialSplatCap`, which runs before there is a footprint to
        /// report.
        var residentBytes: UInt64? {
            switch self {
            case .thermal: return nil
            case .memory(let resident, _): return resident
            case .memoryCeiling(let resident, _): return resident
            case .memoryHeadroom(let resident, _): return resident
            }
        }

        /// What `residentBytes` was measured against: the process's remaining
        /// allocation for `.memory`, the budget's own ceiling for
        /// `.memoryCeiling`, the headroom floor for `.memoryHeadroom`.
        var comparedAgainstBytes: UInt64? {
            switch self {
            case .thermal: return nil
            case .memory(_, let available): return available
            case .memoryCeiling(_, let ceiling): return ceiling
            case .memoryHeadroom(_, let remaining): return remaining
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
    /// Consecutive POLLS, not iterations, at or above `degradeAt`.
    private var consecutiveWarmPolls = 0

    /// How many consecutive warm polls it takes before the first thermal
    /// cut. At the default five-second sample interval this is fifteen
    /// seconds of sustained warmth.
    ///
    /// WHY THIS EXISTS. `degradeAt` defaults to `.fair`, and `.fair` is
    /// iOS's LOWEST non-nominal thermal state. A phone running a Metal
    /// trainer flat out sits at `.fair` as a matter of course; it is what
    /// "this device is doing work" looks like, not a warning. Apple's own
    /// guidance treats `.serious` as the point to shed load.
    ///
    /// So the first poll of any real run found `.fair`, cut immediately,
    /// and because `lower(...)` is one-way and global the run never got it
    /// back. The owner, on a build where the model quality had just been
    /// fixed: "It still decreases everything almost instantly at normal
    /// temperatures and normal temperature climbs." That is this.
    ///
    /// Raising `degradeAt` to `.serious` was the obvious alternative and it
    /// does not work: `thermalVerdict` tests pause before degrade and
    /// `pauseAt` is already `.serious`, so degrade would become unreachable
    /// and the ladder would lose its middle rung entirely. With four
    /// thermal states and three thresholds there is no room to shift the
    /// ladder up. Hysteresis is the change that fits.
    private static let warmPollsBeforeDegrading = 3

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

    /// How much of what the process may still allocate this run is willing to
    /// take: `degradeForMemory` steps the budget down above this share, and
    /// `initialSplatCap` sizes the very first allocation from the same
    /// fraction. Deliberately well under 1: the rasteriser's tile lists, the
    /// decoded frames and the OS's own headroom all live in the same pot.
    // 0.75 (build 346; was 0.6). `availableBytes` is what iOS says this
    // process may still allocate, so 0.6 of it tripped at 1.11 GB of run
    // footprint on a phone reporting 1.85 GB free, 70 MB above build 338's
    // whole run; the floor gate below is the real jetsam guard.
    private let memoryUseFraction: Float = 0.8

    /// The least unallocated headroom this governor will let the process run
    /// on before it starts shedding work, whatever else is true.
    ///
    /// iOS gives no warning before jetsam. `os_proc_available_memory()` counts
    /// down towards zero and the process is killed at zero, so the only safe
    /// way to use it is to keep a margin large enough to survive the biggest
    /// thing the run can still do in one go, and that includes the cut itself.
    /// A cut is not free: `applyBudgetChange` trims the live population first,
    /// and `MetalSplatTrainer.trimSplats` reads the splats, statistics and SH
    /// coefficients out into Swift arrays and builds a second set of them
    /// before writing them back, after which
    /// `TrainerResources.resizeSplatCapacity` rebuilds every per-Gaussian
    /// buffer. A quarter of a gigabyte is the floor under that, and
    /// `headroomFloor(trainerBufferBytes:)` raises it whenever the trainer is
    /// holding more than that.
    ///
    /// It is a judgement, not a measurement. The first device run writes the
    /// footprint and what it was compared against into
    /// `model/train_census.json` for every reduction, which is what a real
    /// number would have to be tuned from.
    static let minimumHeadroomBytes: UInt64 = 256 * 1024 * 1024

    /// The headroom this run has to keep free, sized from what the trainer is
    /// holding right now: a cut that cannot afford to perform itself kills the
    /// app instead of saving it.
    ///
    /// Deliberately conservative rather than exact, and it is worth writing
    /// down which way. `resizeSplatCapacity` used to construct a complete
    /// second `TrainerResources` and push eight buffers out through host
    /// arrays on top of that, so the peak really was about two of everything;
    /// it now replaces one buffer at a time and releases the ten transient
    /// buffers before it allocates anything, so its peak over the steady state
    /// is close to a single replacement buffer. What did NOT change is
    /// `trimSplats`, whose host arrays still scale with the live population.
    /// The trainer's own allocation is the only figure available at this
    /// moment that scales with either of those, so the floor is tied to it.
    /// Erring high costs a slightly earlier cut; erring low costs the scan.
    static func headroomFloor(trainerBufferBytes: UInt64) -> UInt64 {
        Swift.max(minimumHeadroomBytes, trainerBufferBytes)
    }

    /// What `os_proc_available_memory()` said before this run had allocated
    /// anything of its own.
    ///
    /// This is what lets the governor see memory it could not see before. The
    /// governor is constructed at the top of `MetalSplatTrainer.run(...)`,
    /// before `prepare()` makes a Metal device, before the SMART sidecars are
    /// read, before a keyframe is decoded and before the first slice cloud
    /// exists. So the gap between this number and a fresh reading is what the
    /// TRAINING has added to the process since: the SMART sidecars, decoded
    /// image caches, seed arrays, the retained slice clouds (`parts` and
    /// `mergedSoFar` in MetalSplatTrainer), the merge temporaries, and the
    /// trainer's own buffers. Only the last of those appears in
    /// `TrainerResources.residentBytes`.
    ///
    /// What it does NOT include, and the comment says so because getting this
    /// wrong would make the number look like an absolute footprint: anything
    /// allocated before `run(...)` was entered. The pre-pass has already
    /// finished by then (its result arrives as a parameter), so whatever it
    /// kept is inside the baseline rather than counted against this run.
    ///
    /// A difference of two readings from the same source, rather than an
    /// absolute footprint, on purpose. `os_proc_available_memory()` is the one
    /// number iOS gives that already accounts for this process's real limit,
    /// which may or may not include the increased-memory-limit entitlement,
    /// and an absolute footprint would have to be compared against a limit
    /// nothing here knows.
    let baselineAvailableBytes: UInt64

    init(budget: TrainingBudget) {
        self.ceiling = budget
        self.current = budget
        self.baselineAvailableBytes = DeviceMemoryFacts.probe().availableBytes
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
        // The counter advances on a POLL, never on a call. This function
        // runs every iteration and polls every few seconds, so counting
        // calls would reach any threshold within a millisecond and the
        // hysteresis below would be decorative.
        if now.timeIntervalSince(lastThermalPoll) >= policy.sampleIntervalSeconds {
            cachedThermal = ThermalLevel(ProcessInfo.processInfo.thermalState)
            lastThermalPoll = now
            if cachedThermal >= policy.degradeAt {
                consecutiveWarmPolls += 1
            } else {
                consecutiveWarmPolls = 0
            }
        }
        let level = cachedThermal

        // Pausing and aborting stay IMMEDIATE. Those levels are iOS saying
        // the device is in trouble now, and waiting three polls to believe
        // it would be the wrong kind of patience.
        if level >= policy.abortAt { return (.abort, level) }
        if level >= policy.pauseAt { return (.pause, level) }

        // Degrading waits for the warmth to persist. See
        // `warmPollsBeforeDegrading`: a single `.fair` reading is the normal
        // condition of a phone doing this work, and treating it as a signal
        // cut every run to pieces in its first seconds.
        if level >= policy.degradeAt,
           consecutiveWarmPolls >= Self.warmPollsBeforeDegrading {
            return (.degrade, level)
        }
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

    /// What this run has consumed, what the process may still allocate, and
    /// what the trainer's own allocation accounts for. All measured; none of
    /// it derived from `physicalMemory`.
    ///
    /// WHY THIS READING HAS THIS SHAPE. It used to carry one number called
    /// `residentBytes`, and that number was `TrainerResources.residentBytes`:
    /// the sum of `allocatedSize` over the trainer's own Metal buffers, and
    /// nothing else. It was compared against
    /// `TrainingBudget.memoryCeilingBytes`, which `TrainingBudget.recommended`
    /// sets to `availableMemoryBytes / 2` and `ProcessingBudgetPlanner.plan`
    /// re-clamps to the same thing, so something over a gigabyte on a phone
    /// with a normal allowance.
    ///
    /// TWO DEVICE RUNS SAY HOW FAR OUT THAT WAS. A room scan reported "766 MB
    /// of memory in use" while iOS put the app at 2.04 GB, an undercount of
    /// about 2.7x. A 94-shot scan of a single desk reached 3.02 GB of Metal
    /// inside a 3.26 GB app footprint with 279 MB of headroom left, and
    /// finished with 55,041 Gaussians, whose buffers come to about 37 MB. So
    /// the number the governor was watching was roughly a hundredth of the
    /// number that was about to get the process killed, and the ceiling it was
    /// being compared against was half of what the phone had to give.
    /// `overCeiling` was unreachable by arithmetic, and `overShare(0.6)`
    /// needed the trainer's buffers ALONE to pass 0.6 of what remained, which
    /// meant the process was already past saving before the governor noticed
    /// anything. A governor that can only fire once the app is about to be
    /// killed is not a governor.
    ///
    /// It was watching the one pool that did not need watching. The trainer's
    /// buffers are allocated once per capacity change and are exactly the size
    /// the splat cap says they are. Everything that actually varies was
    /// invisible to it: the SMART sidecars, decoded image caches, seed arrays,
    /// the retained slice clouds and the merge temporaries.
    struct MemoryReading {
        /// `TrainerResources.residentBytes`: every Metal buffer the trainer
        /// holds for the slice being trained. Still measured, because it sets
        /// the headroom floor below, but no longer what any threshold is
        /// compared against.
        ///
        /// It is NOT `splatCap * bytesPerSplat + pixelCount * bytesPerPixel`,
        /// and nothing may treat it as though it were. `keysA`, `keysB`,
        /// `valuesA` and `valuesB` are sized from `instanceCapacity`, which
        /// `MetalSplatTrainer` sets to `capacity * 8`, so they add 128 bytes
        /// per unit of splat capacity on top of the per-Gaussian cost, the
        /// radix histograms add more, and `growInstanceCapacity` can raise all
        /// of it mid-run. Bytes in and Gaussians out do not round-trip, which
        /// is why the reduction below is sized in Gaussians rather than by
        /// dividing this number by anything.
        var trainerBufferBytes: UInt64
        /// How many Gaussians the trainer has actually allocated room for
        /// right now.
        ///
        /// This is NOT `current.splatCap`. `TrainerSlices` gives each slice
        /// `splatCap * (that slice's share of the frames)`, and
        /// `MetalSplatTrainer` allocates the slice's share, so on a
        /// four-slice scan this is about a quarter of the cap. The reduction
        /// below needs it because a cap cut ABOVE what is allocated makes
        /// `applyBudgetChange` call `resizeSplatCapacity` upwards, which
        /// allocates in answer to memory pressure. Zero when there are no
        /// resources yet, which the caller treats as "no clamp available".
        var allocatedSplatCapacity: Int
        /// `os_proc_available_memory()`, read fresh for this reading.
        var availableBytes: UInt64
        /// True when `availableBytes` is `physicalMemory / 4` because
        /// `os_proc_available_memory()` returned 0, which it does outside a
        /// normal app process. That fallback is a CONSTANT, so baseline and
        /// current are then equal, `footprintBytes` is 0, and the two
        /// footprint gates below are off. That is the right way round: a
        /// governor with no measurement degrades nothing rather than degrading
        /// on a guess. The headroom gate still reads it, which is harmless: a
        /// quarter of physical RAM is comfortably above the floor on every
        /// device that meets this app's memory requirement.
        var availableIsEstimated: Bool
        var ceilingBytes: UInt64
        /// `TrainerBudgetGovernor.baselineAvailableBytes`, carried in so the
        /// reading is a self-contained fact rather than something that has to
        /// be read against governor state to mean anything.
        var baselineAvailableBytes: UInt64
        /// The headroom this run has to keep free. See
        /// `TrainerBudgetGovernor.headroomFloor(trainerBufferBytes:)`.
        var headroomFloorBytes: UInt64

        /// How many bytes this training run has added to the process since the
        /// governor was built, taken from the fall in what the process may
        /// still allocate. THIS is the footprint figure, and it counts the
        /// pools the old `residentBytes` could not see.
        ///
        /// Saturating rather than wrapping: a later reading can come back
        /// higher than the baseline when something else in the process lets
        /// go, and UInt64 subtraction below zero is a trap, not a negative
        /// number.
        var footprintBytes: UInt64 {
            baselineAvailableBytes > availableBytes
                ? baselineAvailableBytes - availableBytes
                : 0
        }

        var overCeiling: Bool { footprintBytes > ceilingBytes && ceilingBytes > 0 }

        /// True when what this run has consumed is a large share of what is
        /// left to give.
        func overShare(_ fraction: Float) -> Bool {
            guard availableBytes > 0 else { return false }
            return Float(footprintBytes) > Float(availableBytes) * fraction
        }

        /// True when there is no longer enough unallocated headroom to survive
        /// the largest allocation the run can still make. This is the gate the
        /// old reading could not even ask, and it is deliberately indifferent
        /// to who consumed the memory: another part of the app holding half a
        /// gigabyte kills the process just as dead as the trainer holding it.
        var belowHeadroomFloor: Bool { availableBytes < headroomFloorBytes }
    }

    func measureMemory(resources: TrainerResources?) -> MemoryReading {
        let facts = DeviceMemoryFacts.probe()
        let trainerBuffers = resources?.residentBytes ?? 0
        return MemoryReading(
            trainerBufferBytes: trainerBuffers,
            allocatedSplatCapacity: resources?.splatCapacity ?? 0,
            availableBytes: facts.availableBytes,
            availableIsEstimated: facts.availableIsEstimated,
            ceilingBytes: current.memoryCeilingBytes,
            baselineAvailableBytes: baselineAvailableBytes,
            headroomFloorBytes: Self.headroomFloor(trainerBufferBytes: trainerBuffers)
        )
    }

    /// Decides whether memory pressure requires a step down, and takes one
    /// step if it does. The step is sized from the measurement rather than
    /// being a fixed fraction: the overage that tripped the gate is converted
    /// into a number of Gaussians and that many come off the cap, so a small
    /// crossing costs a small cut and a large one costs a large cut. That
    /// matters because the loop polls this every 50 iterations, which usually
    /// catches a crossing while the overage is still small.
    func degradeForMemory(
        reading: MemoryReading,
        currentSplatCount: Int,
        shCoefficientCount: Int,
        pixelCount: Int
    ) -> TrainerBudgetChange? {

        // THREE GATES, and the third is the one that matters most.
        //
        // `overCeiling` and `overShare` now read `reading.footprintBytes`,
        // which is what this run has actually consumed process-wide, so they
        // are reachable at last; `MemoryReading` above carries the arithmetic
        // that made them unreachable before. `belowHeadroomFloor` is new and
        // is independent of both: it fires on how little is LEFT, regardless
        // of who is holding it, because what kills the app is the next
        // allocation failing, not this run's share of the blame for it.
        let belowFloor = reading.belowHeadroomFloor
        let overCeiling = reading.overCeiling
        let overShare = reading.overShare(memoryUseFraction)
        guard belowFloor || overCeiling || overShare else { return nil }

        let footprint = reading.footprintBytes

        // Which fact gets recorded, most urgent first. The census keeps these
        // apart ("memory headroom" against "memory ceiling") because they mean
        // different things for the next run: a headroom cut says this phone
        // cannot hold this scan at all, a ceiling cut says the budget asked
        // for more than it planned for.
        let reason: TrainerBudgetChange.Reason
        if belowFloor {
            reason = .memoryHeadroom(
                residentBytes: footprint, remainingBytes: reading.availableBytes
            )
        } else if overCeiling {
            reason = .memoryCeiling(
                residentBytes: footprint, ceilingBytes: reading.ceilingBytes
            )
        } else {
            reason = .memory(
                residentBytes: footprint, availableBytes: reading.availableBytes
            )
        }

        // HOW MANY BYTES HAVE TO GO BACK, taken as the worst of whichever
        // gates tripped. An overage is the only quantity here that converts
        // into a number of Gaussians without having to assume that the
        // trainer's buffers are made of nothing but Gaussians and pixels,
        // which `MemoryReading.trainerBufferBytes` explains they are not.
        var deficitBytes: UInt64 = 0
        // Kept apart from the running maximum on purpose. `belowFloor` is
        // the one gate allowed to shed without a step limit, so it must be
        // sized by ITS OWN overage and not by whichever gate happened to be
        // worse. See the exemption below.
        var floorDeficitBytes: UInt64 = 0
        if overCeiling, footprint > reading.ceilingBytes {
            deficitBytes = Swift.max(deficitBytes, footprint - reading.ceilingBytes)
        }
        if overShare {
            let share = UInt64(Float(reading.availableBytes) * memoryUseFraction * 0.9)
            if footprint > share {
                deficitBytes = Swift.max(deficitBytes, footprint - share)
            }
        }
        if belowFloor {
            floorDeficitBytes = reading.headroomFloorBytes - reading.availableBytes
            deficitBytes = Swift.max(
                deficitBytes, floorDeficitBytes
            )
        }

        // The per-Gaussian cost, measured from the layouts. The per-PIXEL
        // cost is deliberately not consulted here any more: rung 1 now cuts a
        // number of Gaussians rather than sizing a byte budget that the pixel
        // buffers would have to be subtracted from, and the pixel buffers are
        // rung 2's business. `pixelCount` is still a parameter because
        // `initialSplatCap` and the caller both speak in those terms.
        let perSplat = UInt64(TrainerResources.bytesPerSplat(shCoefficientCount: shCoefficientCount))

        // 1. Splat cap, cut by the overage expressed in Gaussians.
        //
        // TWO THINGS THIS MUST NOT DO, both of which fall out of trying to
        // express the cut as a byte target for the trainer's buffers and then
        // dividing that target by `bytesPerSplat`.
        //
        // It must not read the target off `reading.trainerBufferBytes`. That
        // is what THIS SLICE allocated, and `TrainerSlices` hands each slice
        // `splatCap * (its share of the frames)`, so on a four-slice scan it
        // is about a quarter of the cap. `lower(...)` is global and one-way,
        // so a cap read off a quarter-sized allocation would pin the cap at a
        // quarter for every remaining slice on the first trip of any gate,
        // whatever the overage actually was.
        //
        // And it must not divide that number by `bytesPerSplat`, because the
        // two do not describe the same buffers: the tile sort keys alone are
        // `instanceCapacity * 16` bytes, `instanceCapacity` is `capacity * 8`,
        // and none of that is in `bytesPerSplat`. Dividing anyway reports the
        // trainer as affording about a quarter MORE Gaussians than it has room
        // for, so a small overage produced no cap cut at all and the ladder
        // fell through to the resolution rung, which is the wrong order: this
        // file's header puts the splat cap first precisely because it costs
        // the least visible quality per byte saved.
        //
        // So: shed a number of Gaussians sized from the overage, taken off
        // whichever is smaller of the cap and what is actually allocated. The
        // second half of that is not tidiness. `applyBudgetChange` answers a
        // cap change with `resizeSplatCapacity(to:keeping:)`, which rebuilds
        // the per-Gaussian buffers at whatever capacity it is handed in EITHER
        // direction, so a "cut" to a number above the slice's allocation would
        // allocate more memory in answer to memory pressure.
        if perSplat > 0 {
            let cutFrom = reading.allocatedSplatCapacity > 0
                ? Swift.min(current.splatCap, reading.allocatedSplatCapacity)
                : current.splatCap
            let splatsWanted = Int(deficitBytes / perSplat)

            // HOW DEEP ONE STEP MAY GO, and why the overage alone cannot say.
            //
            // Cutting Gaussians hands back the trainer's buffers and nothing
            // else, and those are a small part of the footprint the gates now
            // measure: 37 MB of the desk scan's 3.26 GB. A deficit taken
            // across the whole process therefore converts into more Gaussians
            // than the trainer has, routinely. Left unbounded that is the same
            // one-way ratchet this file already carries two scars from,
            // arrived at from the other side: at the moment `overShare`
            // crosses, the deficit is the 10% band between the gate and its
            // target, which is 6% of what the process may still allocate, and
            // on a 2.5 GB allowance that is 100 MB, or about 180,000
            // Gaussians at SH degree 1. `affordable` would be 0, `capTarget`
            // would land on its 20,000 floor on the first memory check of the
            // run, and `lower(...)` is global and one-way, so every remaining
            // slice would be built at 20,000 too.
            //
            // So a share or ceiling crossing sheds at most a quarter of what
            // is allocated in one step. It is the same fraction the thermal
            // rung uses, and the loop re-measures every 50 iterations, so a
            // pressure that is really there takes another quarter shortly
            // afterwards while a single crossing costs a step rather than the
            // scan. `belowFloor` is deliberately exempt: that gate says the
            // next allocation may be the one that gets the process killed, and
            // there is no time to converge on an answer.
            //
            // BUT THE EXEMPTION IS SIZED FROM THE FLOOR'S OWN OVERAGE, not
            // from `deficitBytes`, and that distinction is the whole point.
            // `deficitBytes` is the maximum across every gate that tripped,
            // and `belowFloor` almost never trips alone: by the time headroom
            // is short, `overShare` is usually short too, and its deficit is
            // process-scale. Feeding the exempt path that maximum reinstated
            // the exact ratchet the step limit above was added to remove,
            // just through the one door left open. The floor's own overage is
            // the distance back to a safe headroom and nothing more, so the
            // cut it asks for is the cut that gate actually needs.
            let stepLimit = Swift.max(cutFrom / 4, 1)
            let floorSplatsWanted = Int(floorDeficitBytes / perSplat)
            let splatsToShed = belowFloor
                ? Swift.max(floorSplatsWanted, Swift.min(splatsWanted, stepLimit))
                : Swift.min(splatsWanted, stepLimit)
            let affordable = cutFrom > splatsToShed ? cutFrom - splatsToShed : 0

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
            var capTarget = Swift.max(Swift.min(affordable, current.splatCap), 20_000)
            // BUILD 346: THE SOFT GATES STOP GROWTH; THEY DO NOT DELETE THE
            // MODEL. Build 344 crossed the share gate at iteration 200 with
            // 200,000 points alive and was cut every 50 iterations down to
            // 20,000: the run finished with a tenth of its model, which is
            // worse than any memory figure the gate was protecting. The share
            // and ceiling gates are this file's own budget, so once the frame
            // caches are gone (the loop drops them before asking) they may
            // freeze the cap at the live population and no lower. Only the
            // headroom floor, which measures what the OS has left, may cut
            // into live points.
            if !belowFloor, currentSplatCount > 0 {
                let alive = Swift.min(currentSplatCount, current.splatCap)
                if capTarget < alive { capTarget = alive }
                if capTarget >= current.splatCap { return nil }
            }

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
            // `bytesPerSplat` came back zero, which can only mean the GPU
            // layouts changed underneath this. Nothing can be sized from a
            // zero divisor, so go straight to the rung that needs no
            // per-Gaussian cost at all.
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


