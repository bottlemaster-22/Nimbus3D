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
            let recorded = TrainerBudgetChange(
                kind: .splatCap(from: current.splatCap, to: target), reason: reason
            )
            current.splatCap = target
            changes.append(recorded)
            TrainerLog.budget.notice(
                "Splat cap lowered to \(target) from \(recorded.kindFromValue)"
            )
            return recorded

        case .resolution(_, let proposed):
            let target = Swift.min(proposed, current.renderLongEdgePixels)
            guard target < current.renderLongEdgePixels, target > 0 else { return nil }
            let recorded = TrainerBudgetChange(
                kind: .resolution(from: current.renderLongEdgePixels, to: target), reason: reason
            )
            current.renderLongEdgePixels = target
            changes.append(recorded)
            TrainerLog.budget.notice("Render long edge lowered to \(target)")
            return recorded

        case .iterations(_, let proposed):
            let target = Swift.min(proposed, current.iterations)
            guard target < current.iterations, target > 0 else { return nil }
            let recorded = TrainerBudgetChange(
                kind: .iterations(from: current.iterations, to: target), reason: reason
            )
            current.iterations = target
            changes.append(recorded)
            TrainerLog.budget.notice("Iteration count lowered to \(target)")
            return recorded
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
    func degradeForHeat(level: ThermalLevel, currentSplatCount: Int) -> TrainerBudgetChange? {
        let reason = TrainerBudgetChange.Reason.thermal(level)

        // 1. Splat cap. Cut towards what is actually live rather than towards
        // an abstract fraction: cutting a cap the run is nowhere near saves
        // nothing and reads as a fake response.
        let capTarget = Swift.max(
            Int(Float(Swift.min(current.splatCap, Swift.max(currentSplatCount, 1))) * 0.75),
            20_000
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
            let capTarget = Swift.max(Swift.min(affordable, current.splatCap - 1), 20_000)
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
        _ = currentSplatCount
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

private extension TrainerBudgetChange {
    /// The "from" side of whichever case this is, for a log line.
    var kindFromValue: Int {
        switch kind {
        case .splatCap(let from, _): return from
        case .resolution(let from, _): return from
        case .iterations(let from, _): return from
        }
    }
}
