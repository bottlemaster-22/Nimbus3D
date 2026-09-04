//
//  ScanProcessingCoordinator.swift
//  Pipeline
//
//  THE MISSING MIDDLE: the thing that takes a recorded scan and drives it to a
//  finished 3D model.
//
//  Capture writes a scan. `PrePassService` checks it over. `SplatTrainer`
//  builds the model. Both of those were registered in `NimbusApp.swift` and
//  neither had a caller, so a recorded scan stopped dead at "Recorded. Not
//  checked over yet." This file is the caller.
//
//  ---------------------------------------------------------------------------
//  WHAT IT DOES, IN ORDER
//  ---------------------------------------------------------------------------
//   1. Reads what is already on disk (`ScanLibraryReader.readDetail`), because
//      the check-over is twenty minutes of work and redoing it by accident is
//      the rudest thing this app could do. If `prepass/prepass_result.json` is
//      there and readable, it is OFFERED for reuse rather than silently reused
//      or silently redone; the screen asks.
//   2. `prePass.run(bundle:at:)`, consuming both the result stream and, when
//      the concrete `PrePassPipeline` is what is registered, its stage stream,
//      so the screen can say "Finding the places you walked back over" instead
//      of drawing a bar.
//   3. Sizes a `TrainingBudget` from the device tier, the scan's own measured
//      extent and the memory this process may actually allocate
//      (`ProcessingBudgetPlanner`).
//   4. `trainer.train(bundle:prePass:at:budget:)`, consuming `TrainerProgress`.
//
//  ---------------------------------------------------------------------------
//  HONESTY RULES THIS FILE IS RESPONSIBLE FOR
//  ---------------------------------------------------------------------------
//  * The stage and the sentence come from the service. Nothing here invents a
//    percentage. `TrainerProgress.fractionComplete` is optional on purpose and
//    a nil is passed through as nil.
//  * The trainer is allowed to LOWER the budget and is required to report it.
//    When it does, that sentence is pulled out of the progress stream and kept
//    where the user can read it, and the finished model's `budgetUsed` is
//    compared against what was asked for so the run ends with a plain statement
//    of what actually got built. Silently shipping a smaller model is exactly
//    the dishonesty this project exists to avoid.
//  * Cancel calls `trainer.cancel()` AND tears down the consuming task, so
//    nothing is left running behind a screen that says it stopped.
//  * A failure leaves the scan describing itself truthfully: the two index
//    files the library reads are written by their own modules as the last act
//    of a successful run, and anything unreadable is cleared away
//    (`ProcessingArtifacts.discardUnreadableIndexFiles`) so a half-finished run
//    can never look complete.
//
//  ---------------------------------------------------------------------------
//  WHY THIS IS A SINGLETON
//  ---------------------------------------------------------------------------
//  There is one GPU and one registered trainer, so there is one job at a time
//  whether or not this object is shared. Sharing it means the work survives the
//  user navigating away from the screen - a twenty minute job that dies because
//  someone looked at their library would be its own kind of lie - and it means
//  the pre-pass's single lifetime-scoped progress stream is consumed exactly
//  once, which is what that stream's shape requires.
//

import Foundation
import SwiftUI

// MARK: - Notices

/// One thing worth telling the user about a run, kept after it happened.
///
/// These are not log lines. Each one is a sentence shown on screen, because the
/// interesting parts of a training run (the phone got warm, the model got
/// smaller, an old check-over was reused) all happen while nobody is watching.
struct ProcessingNotice: Identifiable, Equatable {
    enum Kind: String {
        case reused
        case budgetLowered
        case paused
        case outcome
        case warning
    }

    let id = UUID()
    var kind: Kind
    var text: String
    var at: Date = Date()

    var iconName: String {
        switch kind {
        case .reused: return "arrow.uturn.backward.circle"
        case .budgetLowered: return "arrow.down.right.circle"
        case .paused: return "pause.circle"
        case .outcome: return "checkmark.circle"
        case .warning: return "exclamationmark.triangle"
        }
    }
}

// MARK: - The coordinator

@MainActor
final class ScanProcessingCoordinator: ObservableObject {

    static let shared = ScanProcessingCoordinator()

    /// Which part of the job is running. No associated values on purpose: it is
    /// `Equatable` so a SwiftUI `onChange` can watch it, and the payloads live
    /// in their own published properties where a view can bind to them
    /// individually without redrawing on every tick.
    enum Phase: String {
        case idle
        case reading
        case checkingOver
        case buildingModel
        case done
        case failed
        case cancelled
    }

    /// What the user asked for.
    enum Intent: String {
        /// Just the check-over.
        case checkOver
        /// Just the model, from a check-over that already exists.
        case buildModel
        /// Both, without stopping in the middle.
        case everything
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var intent: Intent = .everything
    /// Which scan this coordinator is (or was last) working on. A screen shows
    /// live state only when this matches its own scan.
    @Published private(set) var activeScanID: ScanID?

    /// The latest tick from each stage. Both stay on screen after the run so a
    /// finished job still says how it ended.
    @Published private(set) var prePassTick: PrePassProgress?
    @Published private(set) var trainerTick: TrainerProgress?

    @Published private(set) var notices: [ProcessingNotice] = []
    @Published private(set) var plan: ProcessingBudgetPlan?
    @Published private(set) var prePassResult: PrePassResult?
    @Published private(set) var finishedModel: SplatModel?

    /// Plain-language failure text, and what to try, when `phase == .failed`.
    @Published private(set) var problem: String?
    @Published private(set) var problemHint: String?

    private var runTask: Task<Void, Never>?
    private var prePassObserver: Task<Void, Never>?
    /// Stage the trainer was last in, so a pause is announced once rather than
    /// on every tick.
    private var lastAnnouncedStage: TrainerStage?

    private init() {}

    // MARK: Queries

    var isRunning: Bool {
        phase == .reading || phase == .checkingOver || phase == .buildingModel
    }

    func isWorking(on scanID: ScanID) -> Bool {
        isRunning && activeScanID == scanID
    }

    /// True when this coordinator's published state describes `scanID`. A
    /// screen for any other scan shows nothing live, rather than another scan's
    /// progress under its own name.
    func isShowing(_ scanID: ScanID) -> Bool {
        activeScanID == scanID
    }

    /// The tier the compatibility probe worked out, read back from the one
    /// place the app already keeps it: `RootView.runCompatibilityCheck()` sets
    /// it on the registered `PrePassPipeline`. Reading it back rather than
    /// storing a second copy is the whole point - two copies of a device tier
    /// is two chances to disagree.
    var deviceTier: DeviceTier? {
        (NimbusServices.shared.prePass as? PrePassPipeline)?.deviceTier
    }

    // MARK: Starting

    /// Starts a run. Does nothing if one is already going: there is one GPU.
    func start(_ intent: Intent, for summary: ScanSummary, reuseExistingPrePass: Bool) {
        guard !isRunning else { return }

        self.intent = intent
        activeScanID = summary.scanID
        phase = .reading
        prePassTick = nil
        trainerTick = nil
        notices = []
        plan = nil
        prePassResult = nil
        finishedModel = nil
        problem = nil
        problemHint = nil
        lastAnnouncedStage = nil

        observePrePassStages()

        runTask = Task { [weak self] in
            await self?.run(intent: intent, summary: summary, reuse: reuseExistingPrePass)
        }
    }

    /// Stops everything, for real: the trainer is told to cancel (it frees its
    /// GPU buffers on the thread that owns them), and the task consuming the
    /// streams is torn down, which terminates the pre-pass stream and cancels
    /// the work behind it.
    func cancel() {
        guard isRunning else { return }
        runTask?.cancel()
        runTask = nil
        // Resolved inside the task rather than captured: the registry is
        // main-actor isolated and this task inherits that, so nothing
        // non-Sendable crosses a boundary.
        Task { await NimbusServices.shared.trainer?.cancel() }
        phase = .cancelled
        problem = nil
        problemHint = nil
    }

    /// Clears a finished run's state so the screen goes back to its buttons.
    /// Never called while something is running.
    func dismissOutcome() {
        guard !isRunning else { return }
        phase = .idle
        problem = nil
        problemHint = nil
    }

    // MARK: The run

    private func run(intent: Intent, summary: ScanSummary, reuse: Bool) async {
        let ref = summary.ref
        let paths = summary.paths

        let detail = await Task.detached(priority: .userInitiated) {
            ScanLibraryReader.readDetail(summary)
        }.value

        guard let bundle = detail.bundle else {
            fail(NimbusError.malformedData(
                "this scan has no readable index (capture_bundle.json) in it"
            ), at: paths)
            return
        }
        guard bundle.formatVersion == CaptureBundle.currentFormatVersion else {
            fail(NimbusError.malformedData(
                "this scan was recorded in a format this version of the app does not know"
            ), at: paths)
            return
        }
        guard !bundle.frames.isEmpty else {
            fail(NimbusError.malformedData("this scan has no photos in it"), at: paths)
            return
        }

        var prePass = detail.prePass
        prePassResult = prePass

        // --- 1. The check-over.
        if intent == .checkOver || intent == .everything {
            if reuse, let existing = prePass {
                note(
                    .reused,
                    "Used the check-over that was already done on "
                    + "\(ProcessingFormat.date(existing.completedAt)). "
                    + "Nothing was thrown away and nothing was redone."
                )
            } else {
                do {
                    prePass = try await runPrePass(bundle: bundle, at: ref)
                    prePassResult = prePass
                } catch {
                    finishWithError(error, at: paths)
                    return
                }
            }
        }

        if Task.isCancelled { finishCancelled(); return }

        if intent == .checkOver {
            succeed()
            note(.outcome, "This scan is checked over and ready to build.")
            return
        }

        // --- 2. The budget.
        guard let ready = prePass else {
            fail(NimbusError.trainingFailed(
                "this scan has not been checked over yet, and the check-over is where the "
                + "starting points and the camera positions come from"
            ), at: paths)
            return
        }

        let budgetPlan = ProcessingBudgetPlanner.plan(
            bundle: bundle,
            prePass: ready,
            tier: deviceTier
        )
        plan = budgetPlan
        if let warning = budgetPlan.sizeWarning {
            note(.warning, warning)
        }

        // --- 3. The model.
        do {
            let model = try await runTraining(
                bundle: bundle,
                prePass: ready,
                at: ref,
                budget: budgetPlan.budget
            )
            finishedModel = model
            succeed()
            note(.outcome, outcomeSentence(for: model, asked: budgetPlan.budget))
        } catch {
            finishWithError(error, at: paths)
        }
    }

    // MARK: Pre-pass

    private func runPrePass(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> PrePassResult {
        guard let service = NimbusServices.shared.prePass else {
            throw NimbusError.moduleNotInstalled(module: "scan checking")
        }

        phase = .checkingOver

        // The stream yields a partial result as soon as the quick card exists
        // and keeps yielding as the heavier stages land, so the last one is the
        // finished index. Every one of them is published, because the quality
        // card on the first is worth reading two minutes before the rest.
        var last: PrePassResult?
        for try await partial in service.run(bundle: bundle, at: ref) {
            last = partial
            prePassResult = partial
        }
        try Task.checkCancellation()

        guard let result = last else {
            throw NimbusError.prePassFailed(
                "the check finished without producing anything to save."
            )
        }
        try ProcessingArtifacts.ensurePrePassResultWritten(result, at: ref)
        return result
    }

    /// Subscribes to the pre-pass's stage stream, once, for the life of this
    /// object.
    ///
    /// `PrePassPipeline.progress` is created in its initialiser and lives as
    /// long as the pipeline does, with one intended consumer. Dropping an
    /// iterator on an `AsyncStream` finishes it for good, so this is started
    /// once and never cancelled: a per-run subscription would silently lose
    /// every stage name from the second run onwards.
    private func observePrePassStages() {
        guard prePassObserver == nil,
              let pipeline = NimbusServices.shared.prePass as? PrePassPipeline
        else { return }

        let stream = pipeline.progress
        prePassObserver = Task { [weak self] in
            for await tick in stream {
                guard let self else { return }
                guard self.phase == .checkingOver else { continue }
                self.prePassTick = tick
            }
        }
    }

    // MARK: Training

    private func runTraining(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        budget: TrainingBudget
    ) async throws -> SplatModel {
        guard let trainer = NimbusServices.shared.trainer else {
            throw NimbusError.moduleNotInstalled(module: "3D model building")
        }

        phase = .buildingModel

        for try await tick in trainer.train(
            bundle: bundle, prePass: prePass, at: ref, budget: budget
        ) {
            trainerTick = tick
            recordNotices(from: tick)
        }
        try Task.checkCancellation()

        guard let model = await trainer.finishedModel() else {
            throw NimbusError.trainingFailed(
                "it stopped before there was a model to save."
            )
        }
        try ProcessingArtifacts.ensureModelWritten(model, at: ref)
        return model
    }

    // MARK: Notices

    /// Pulls the things worth saying out of a progress tick.
    ///
    /// HONEST LIMITATION, and it is filed in `INTEGRATION_REQUESTS.md`:
    /// `TrainerProgress` has no flag for "I lowered the budget". The contract
    /// says the trainer must report a reduction and the only channel it has is
    /// `message`, which `TrainerBudgetChange.message` always writes as a
    /// sentence beginning "Because ...". Matching that prefix is a workaround,
    /// not a contract. It can only ever miss a notice or catch a harmless one,
    /// never invent a reduction: the sentence shown is the trainer's own, and
    /// the run ends with a comparison of `SplatModel.budgetUsed` against what
    /// was asked for, which IS contractual and is the number of record.
    private func recordNotices(from tick: TrainerProgress) {
        if tick.message.hasPrefix("Because "), !notices.contains(where: { $0.text == tick.message }) {
            note(.budgetLowered, tick.message)
        }

        guard tick.stage != lastAnnouncedStage else { return }
        lastAnnouncedStage = tick.stage

        switch tick.stage {
        case .pausedThermal:
            note(
                .paused,
                "Your phone got warm, so this paused itself. It carries on by itself "
                + "once it cools down, and nothing that was already built is lost."
            )
        case .pausedMemory:
            note(
                .paused,
                "This phone ran short of memory, so the model is being made smaller and "
                + "picked up from where it got to."
            )
        default:
            break
        }
    }

    /// What actually got built, compared with what was asked for. This is the
    /// contractual version of the "did it get smaller?" question:
    /// `SplatModel.budgetUsed` is defined as the budget the run really ran
    /// under, after any live degradation.
    private func outcomeSentence(for model: SplatModel, asked: TrainingBudget) -> String {
        let built = "Your 3D model is ready: "
            + "\(ProcessingFormat.count(model.splatCount)) detail points from "
            + "\(ProcessingFormat.count(model.iterationsCompleted)) rounds."

        guard let used = model.budgetUsed else { return built }

        var reductions: [String] = []
        if used.splatCap < asked.splatCap {
            reductions.append(
                "aimed for \(ProcessingFormat.count(used.splatCap)) detail points instead "
                + "of \(ProcessingFormat.count(asked.splatCap))"
            )
        }
        if used.renderLongEdgePixels < asked.renderLongEdgePixels {
            reductions.append(
                "practised on \(used.renderLongEdgePixels) pixel pictures instead of "
                + "\(asked.renderLongEdgePixels)"
            )
        }
        if used.iterations < asked.iterations {
            reductions.append(
                "stopped after \(ProcessingFormat.count(used.iterations)) rounds instead "
                + "of \(ProcessingFormat.count(asked.iterations))"
            )
        }

        guard !reductions.isEmpty else { return built }
        return built + " Your phone made this one smaller than planned as it went: it "
            + list(reductions) + ". That is why it finished cleanly instead of stopping."
    }

    private func list(_ parts: [String]) -> String {
        switch parts.count {
        case 0: return ""
        case 1: return parts[0]
        case 2: return "\(parts[0]) and \(parts[1])"
        default:
            return parts.dropLast().joined(separator: ", ") + " and \(parts[parts.count - 1])"
        }
    }

    private func note(_ kind: ProcessingNotice.Kind, _ text: String) {
        notices.append(ProcessingNotice(kind: kind, text: text))
    }

    // MARK: Finishing

    private func finishWithError(_ error: Error, at paths: ViewerScanPaths) {
        if ProcessingProblem.isCancellation(error) {
            finishCancelled()
            return
        }
        fail(error, at: paths)
    }

    private func succeed() {
        phase = .done
        problem = nil
        problemHint = nil
        runTask = nil
    }

    private func finishCancelled() {
        phase = .cancelled
        problem = nil
        problemHint = nil
        runTask = nil
    }

    private func fail(_ error: Error, at paths: ViewerScanPaths) {
        ProcessingLog.coordinator.error(
            "Processing failed: \(String(describing: error), privacy: .public)"
        )
        phase = .failed
        problem = ProcessingProblem.plainText(for: error)
        problemHint = ProcessingProblem.whatToTry(for: error)
        runTask = nil

        // A run that died must not leave the library claiming a stage finished.
        // Both index files are written atomically by their own modules as the
        // last act of a successful run, so in practice there is nothing to
        // clear; this catches the case where there is.
        for text in ProcessingArtifacts.discardUnreadableIndexFiles(at: paths) {
            note(.warning, text)
        }
    }
}
