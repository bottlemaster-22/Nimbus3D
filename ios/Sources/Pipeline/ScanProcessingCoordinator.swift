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
//  * Stopping is COOPERATIVE, and this file says so rather than implying
//    otherwise. `cancel()` asks: it cancels the task consuming the streams
//    (which terminates them and cancels the work behind them) and it calls
//    `trainer.cancel()`. Neither stops on the spot. The trainer looks at the
//    flag between GPU steps; the check-over looks between its own stages. So
//    the screen says "Stopping" and keeps saying it, and only says "Stopped"
//    once the run has genuinely returned. `runTraining` waits for
//    `MetalSplatTrainer.waitUntilIdle()` before it lets the run end, so the
//    "Stopped" on screen means the GPU is quiet, not that a message was sent.
//  * No second run can start in the gap. `canStartNewRun` is false until the
//    run task has returned AND the trainer reports itself idle, and the
//    trainer refuses a second `train()` outright while one is in flight. That
//    matters because there is exactly one registered trainer with one set of
//    GPU buffers behind it.
//  * The ONE thing not waited for is the tail of a cancelled check-over.
//    `PrePassService` has no "am I finished unwinding?" to ask, so a stop
//    during the check-over is honestly cooperative but cannot be waited on.
//    Filed in INTEGRATION_REQUESTS.md. Nothing else in the app can start a
//    second check-over in the meantime, because the same guard applies.
//  * The live preview is a picture of the model as it is being built, taken
//    from `trainer.snapshot()` at most every few seconds and drawn from ONE
//    fixed camera. It never blocks the training loop: a refresh that is late,
//    that fails, or that arrives while the previous one is still uploading is
//    dropped, and the last good frame stays on screen.
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
import UIKit

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

    /// True from the moment Stop is pressed until the run has genuinely
    /// finished putting itself down. The screen shows "Stopping" for exactly
    /// this long, because saying "Stopped" while the GPU is still busy is the
    /// lie this coordinator is not allowed to tell.
    @Published private(set) var isStopping = false

    /// The live picture of the model as it is being built. Made when training
    /// starts, taken down when the run ends. Nil the rest of the time, so the
    /// preview renderer's Metal pipelines are only ever built for a run that
    /// is actually happening.
    @Published private(set) var preview: TrainingPreviewController?

    /// The run task, kept until the task has ACTUALLY RETURNED rather than
    /// nilled the moment a stop is requested. That distinction is the whole
    /// fix: `Task.cancel()` sets a flag, and the work behind it stops at its
    /// next check point, which for the trainer is the next GPU step boundary.
    private var runTask: Task<Void, Never>?
    /// Bumped by every `start`. A task that is still unwinding compares its
    /// own number against this before it writes any state, so a late finish
    /// can never clobber a newer run. `start` already refuses while a task is
    /// unwinding, so this is a second lock on the same door: cheap, and the
    /// failure it prevents (a screen saying "stopped" over a live run) is not.
    private var runGeneration = 0
    private var prePassObserver: Task<Void, Never>?
    /// Stage the trainer was last in, so a pause is announced once rather than
    /// on every tick.
    private var lastAnnouncedStage: TrainerStage?

    private init() {}

    // MARK: Queries

    var isRunning: Bool {
        phase == .reading || phase == .checkingOver || phase == .buildingModel
    }

    /// Whether a new run may be started right now.
    ///
    /// Deliberately NOT `!isRunning`. Stopping is cooperative, so there is a
    /// window in which the phase has moved on and the previous task has not
    /// returned yet. Starting inside that window would put two runs on one
    /// GPU, one set of trainer buffers and one screen.
    var canStartNewRun: Bool {
        guard runTask == nil, !isStopping else { return false }
        if let metal = NimbusServices.shared.trainer as? MetalSplatTrainer, metal.isTraining {
            return false
        }
        return true
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

    /// Starts a run.
    ///
    /// Does nothing while another run is going, AND does nothing while the
    /// previous one is still putting itself down. There is one GPU, one
    /// registered trainer and one set of buffers behind it, and a stop that
    /// has been asked for is not a stop that has happened.
    func start(_ intent: Intent, for summary: ScanSummary, reuseExistingPrePass: Bool) {
        guard canStartNewRun else { return }

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
        isStopping = false
        lastAnnouncedStage = nil
        releasePreview()

        observePrePassStages()

        // Hold the screen awake for the whole run.
        //
        // Capture already did this (ARCaptureService sets it around the
        // session) but processing never did, so the display would dim and
        // lock partway through a twenty minute job. That is not a cosmetic
        // problem: this is foreground work with a live Metal trainer and a
        // preview attached to it, and iOS is free to suspend the app once
        // the screen locks, which loses the run.
        //
        // Cleared in `retireRun`, which is the one place that runs after
        // the task has actually returned, so a cancelled or failed run
        // releases it exactly like a finished one.
        UIApplication.shared.isIdleTimerDisabled = true

        runGeneration += 1
        let generation = runGeneration
        runTask = Task { [weak self] in
            await self?.run(
                intent: intent,
                summary: summary,
                reuse: reuseExistingPrePass,
                generation: generation
            )
            self?.retireRun(generation)
        }
    }

    /// ASKS everything to stop, and says so on screen until it has.
    ///
    /// Both halves of the ask are real: the trainer is told to cancel (it
    /// frees its GPU buffers on the thread that owns them) and the task
    /// consuming the streams is cancelled, which terminates them and cancels
    /// the work behind them. Neither is instant. Swift cancellation is a flag
    /// that cooperating code checks, and the trainer checks it between GPU
    /// steps, so this method returning means "asked", not "stopped".
    ///
    /// `runTask` is therefore KEPT, `isStopping` goes true, and the run's own
    /// terminal state is set later, by the task, once it has really finished.
    /// Until then `canStartNewRun` is false, so the Stop-then-start-again tap
    /// that used to put two runs on one GPU now simply does not take.
    func cancel() {
        guard let task = runTask, !isStopping else { return }
        isStopping = true
        task.cancel()
        // Resolved inside the task rather than captured: the registry is
        // main-actor isolated and this task inherits that, so nothing
        // non-Sendable crosses a boundary.
        //
        // The generation check is not decoration. This hop is asynchronous, so
        // in principle it could land after THIS run has finished and a later
        // one has begun, and `trainer.cancel()` would then stop the wrong run.
        // Refusing a stale cancel is one comparison.
        let generation = runGeneration
        Task { [weak self] in
            guard let self, self.runGeneration == generation else { return }
            await NimbusServices.shared.trainer?.cancel()
        }
        problem = nil
        problemHint = nil
    }

    /// Clears a finished run's state so the screen goes back to its buttons.
    /// Never called while something is running or still stopping.
    func dismissOutcome() {
        guard canStartNewRun else { return }
        phase = .idle
        problem = nil
        problemHint = nil
        releasePreview()
    }

    /// Called by the run task, and only by the run task, once it has actually
    /// returned. This is the moment a new run becomes possible.
    private func retireRun(_ generation: Int) {
        guard generation == runGeneration else { return }
        runTask = nil
        isStopping = false
        // Let the screen sleep again. Paired with the disable in `start`,
        // and here rather than at the end of `run` because this is the
        // point the run has genuinely returned, cancelled ones included.
        UIApplication.shared.isIdleTimerDisabled = false
        releasePreview()
    }

    // MARK: The run

    private func run(
        intent: Intent,
        summary: ScanSummary,
        reuse: Bool,
        generation: Int
    ) async {
        let ref = summary.ref
        let paths = summary.paths

        let detail = await Task.detached(priority: .userInitiated) {
            ScanLibraryReader.readDetail(summary)
        }.value

        guard let bundle = detail.bundle else {
            fail(NimbusError.malformedData(
                "this scan has no readable index (capture_bundle.json) in it"
            ), at: paths, generation: generation)
            return
        }
        guard bundle.formatVersion == CaptureBundle.currentFormatVersion else {
            fail(NimbusError.malformedData(
                "this scan was recorded in a format this version of the app does not know"
            ), at: paths, generation: generation)
            return
        }
        guard !bundle.frames.isEmpty else {
            fail(
                NimbusError.malformedData("this scan has no photos in it"),
                at: paths,
                generation: generation
            )
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
                    finishWithError(error, at: paths, generation: generation)
                    return
                }
            }
        }

        if Task.isCancelled { finishCancelled(generation); return }

        if intent == .checkOver {
            succeed(generation)
            note(.outcome, "This scan is checked over and ready to build.")
            return
        }

        // --- 2. The budget.
        guard let ready = prePass else {
            fail(NimbusError.trainingFailed(
                "this scan has not been checked over yet, and the check-over is where the "
                + "starting points and the camera positions come from"
            ), at: paths, generation: generation)
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
            succeed(generation)
            note(.outcome, outcomeSentence(for: model, asked: budgetPlan.budget))
        } catch {
            finishWithError(error, at: paths, generation: generation)
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

        // A run that was stopped a moment ago can still be unwinding on the
        // GPU, because cancellation is cooperative and the trainer only looks
        // between steps. `start` refuses inside that window, so in practice
        // this returns straight away; it is here so that the ONE trainer
        // instance can never have two loops in it even if a future caller
        // reaches this method by some other route.
        await waitForTrainerToStop(trainer)

        makePreview(for: bundle)

        do {
            for try await tick in trainer.train(
                bundle: bundle, prePass: prePass, at: ref, budget: budget
            ) {
                trainerTick = tick
                recordNotices(from: tick)
                // Fire and forget on purpose. This returns immediately, does
                // its own throttling, and drops a refresh rather than let the
                // preview hold the training loop up for a single tick.
                if tick.previewAvailable {
                    preview?.offerRefresh()
                }
            }
        } catch {
            // The stream ends the instant the task is cancelled; the loop
            // behind it stops at the next step boundary. Waiting here is what
            // makes the "Stopped" the screen shows afterwards true.
            await waitForTrainerToStop(trainer)
            throw error
        }
        await waitForTrainerToStop(trainer)
        try Task.checkCancellation()

        guard let model = await trainer.finishedModel() else {
            throw NimbusError.trainingFailed(
                "it stopped before there was a model to save."
            )
        }
        // WHOSE MODEL IS THIS? `finishedModel()` is the last model the trainer
        // holds, and the trainer is one shared instance for the life of the
        // app. Writing it into this scan's folder without checking would, on
        // any path that ends a run without building anything, file scan A's
        // model under scan B and leave the library saying scan B is "Ready to
        // look at." The trainer clears its own `latestModel` at the top of
        // every run, so both ends of this are covered.
        guard model.scanID == bundle.scanID else {
            ProcessingLog.coordinator.error(
                "Refusing a model for \(model.scanID, privacy: .public) while building \(bundle.scanID, privacy: .public)"
            )
            throw NimbusError.trainingFailed(
                "the model that came back belongs to a different scan, so it was not saved "
                + "here. Nothing of yours was overwritten. Please build this one again."
            )
        }
        try ProcessingArtifacts.ensureModelWritten(model, at: ref)
        return model
    }

    /// Waits for the trainer to be genuinely idle, when it is one this app can
    /// ask. `SplatTrainer` has no "have you stopped yet?" in the contract, so
    /// this is a downcast to the concrete trainer the app registers, and a
    /// trainer that is something else is simply not waited for rather than
    /// waited for wrongly. Filed in INTEGRATION_REQUESTS.md.
    private func waitForTrainerToStop(_ trainer: any SplatTrainer) async {
        guard let metal = trainer as? MetalSplatTrainer else { return }
        await metal.waitUntilIdle()
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
    //
    // None of these clears `runTask`. That is `retireRun`'s job, and it only
    // happens once the task has genuinely returned, which is what keeps a
    // second run from starting on top of one that is still unwinding.
    //
    // Every one of them takes the run's generation and writes nothing if it is
    // stale, so a task finishing late can never repaint a newer run's screen.

    private func finishWithError(_ error: Error, at paths: ViewerScanPaths, generation: Int) {
        if ProcessingProblem.isCancellation(error) {
            finishCancelled(generation)
            return
        }
        fail(error, at: paths, generation: generation)
    }

    private func succeed(_ generation: Int) {
        guard generation == runGeneration else { return }
        phase = .done
        problem = nil
        problemHint = nil
    }

    private func finishCancelled(_ generation: Int) {
        guard generation == runGeneration else { return }
        phase = .cancelled
        problem = nil
        problemHint = nil
    }

    private func fail(_ error: Error, at paths: ViewerScanPaths, generation: Int) {
        guard generation == runGeneration else { return }
        ProcessingLog.coordinator.error(
            "Processing failed: \(String(describing: error), privacy: .public)"
        )
        phase = .failed
        problem = ProcessingProblem.plainText(for: error)
        problemHint = ProcessingProblem.whatToTry(for: error)

        // A run that died must not leave the library claiming a stage finished.
        // Both index files are written atomically by their own modules as the
        // last act of a successful run, so in practice there is nothing to
        // clear; this catches the case where there is.
        for text in ProcessingArtifacts.discardUnreadableIndexFiles(at: paths) {
            note(.warning, text)
        }
    }

    // MARK: The live preview

    /// Builds the preview for a run that is about to start training.
    ///
    /// One camera intrinsics is handed over so the preview is not stretched:
    /// only the PIXEL ASPECT is taken from it, the field of view comes from
    /// the framing.
    private func makePreview(for bundle: CaptureBundle) {
        releasePreview()
        preview = TrainingPreviewController(sourceIntrinsics: bundle.intrinsics)
    }

    /// Takes the preview down and gives its cloud back. A finished model is
    /// opened in the review screen, which loads the real file from disk, so
    /// there is nothing to gain by holding a whole cloud in memory behind a
    /// screen that is no longer showing it.
    private func releasePreview() {
        preview?.tearDown()
        preview = nil
    }
}
