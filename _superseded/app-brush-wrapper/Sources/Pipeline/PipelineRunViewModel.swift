//
//  PipelineRunViewModel.swift
//  Nimbus3D - Pipeline module
//
//  MainActor view model that drives one PipelineOrchestrator run and publishes
//  ordered, per-stage state for the Process screen. It owns:
//    - the list of capture bundles to choose from (CaptureBundleStore),
//    - the selected bundle and run options,
//    - live per-stage status + overall fraction,
//    - the run/cancel lifecycle.
//
//  Concurrency: the orchestrator emits PipelineEvents from its actor executor
//  (a background thread). Events are funnelled through an AsyncStream and
//  consumed IN ORDER on the MainActor, so SwiftUI state is only ever mutated
//  here. Nothing in this file fakes a stage result; it merely reflects what the
//  orchestrator (and the registered dependencies behind it) actually report.
//

import Foundation
import Observation

@MainActor
@Observable
public final class PipelineRunViewModel {

    /// Fixed display order of the pipeline stages on the Process screen.
    public static let orderedStages: [PipelineStage] = [
        .capture,
        .splatTraining,
        .meshExtraction,
        .hdriAssembly,
        .delighting,
        .materialClassification,
        .textureBuild,
        .export,
    ]

    /// Overall run lifecycle for the Process screen's header/footer.
    public enum RunPhase: Equatable {
        case idle
        case running
        case finished(ExportedAsset)
        case failed(String)
        case cancelled
    }

    // MARK: Published state

    /// Capture bundles available to process, newest first.
    public private(set) var bundles: [CaptureBundle] = []
    /// The bundle the user chose to process.
    public var selectedBundleID: UUID?
    /// Per-stage status, keyed by stage. Missing key == .pending.
    public private(set) var stageStatus: [PipelineStage: PipelineStageStatus] = [:]
    /// Weighted 0...1 progress across the whole pipeline.
    public private(set) var overallFraction: Double = 0
    public private(set) var phase: RunPhase = .idle
    /// The most recent successful export, if any (also handed to the Library tab).
    public private(set) var lastExport: ExportedAsset?

    public var isRunning: Bool { if case .running = phase { return true }; return false }

    public var selectedBundle: CaptureBundle? {
        guard let id = selectedBundleID else { return nil }
        return bundles.first { $0.id == id }
    }

    // MARK: Private run state

    /// Sendable terminal outcome funnelled back from the detached run task.
    /// (A raw `Result<ExportedAsset, any Error>` is not Sendable because
    /// `any Error` isn't, which strict concurrency rejects as a Task result.)
    private enum RunOutcome: Sendable {
        case success(ExportedAsset)
        case failure(message: String, wasCancelled: Bool)
    }

    private var orchestrator: PipelineOrchestrator?
    private var runTask: Task<Void, Never>?

    public init() {}

    // MARK: Bundle loading

    /// Reloads capture bundles from disk and preserves/repairs the selection.
    public func refreshBundles() {
        bundles = CaptureBundleStore.loadAll()
        if let id = selectedBundleID, bundles.contains(where: { $0.id == id }) {
            return
        }
        selectedBundleID = bundles.first?.id
    }

    /// Selects a bundle and resets the stage board to pending (unless a run is
    /// in flight, which is left untouched).
    public func select(_ bundle: CaptureBundle) {
        guard !isRunning else { return }
        selectedBundleID = bundle.id
        resetBoard()
    }

    // MARK: Status accessors for the view

    public func status(for stage: PipelineStage) -> PipelineStageStatus {
        stageStatus[stage] ?? .pending
    }

    private func resetBoard() {
        stageStatus = [:]
        for stage in Self.orderedStages { stageStatus[stage] = .pending }
        overallFraction = 0
        phase = .idle
    }

    // MARK: Run lifecycle

    /// Starts a pipeline run for the selected bundle. No-op if one is running or
    /// nothing is selected. Reads registered dependencies from PipelineServices;
    /// unwired stages surface as .notImplemented and (for optional stages) skip.
    public func run(options: PipelineRunOptions = PipelineRunOptions()) {
        guard !isRunning, let bundle = selectedBundle else { return }

        resetBoard()
        phase = .running

        let orchestrator = PipelineOrchestrator(dependencies: PipelineServices.shared)
        self.orchestrator = orchestrator

        runTask = Task { [weak self] in
            let (stream, continuation) = AsyncStream<PipelineEvent>.makeStream()

            // Run the orchestrator concurrently with event consumption. The
            // closure only yields into the stream, so it stays @Sendable and
            // never touches MainActor state directly.
            let work = Task.detached { () -> RunOutcome in
                do {
                    let asset = try await orchestrator.run(bundle: bundle, options: options) { event in
                        continuation.yield(event)
                    }
                    continuation.finish()
                    return .success(asset)
                } catch is CancellationError {
                    continuation.finish()
                    return .failure(message: "Cancelled", wasCancelled: true)
                } catch NimbusError.cancelled {
                    continuation.finish()
                    return .failure(message: "Cancelled", wasCancelled: true)
                } catch {
                    continuation.finish()
                    return .failure(message: error.localizedDescription, wasCancelled: false)
                }
            }

            // Consume events in order on the MainActor.
            for await event in stream {
                self?.apply(event)
            }

            let result = await work.value
            self?.finish(with: result)
        }
    }

    /// Requests cooperative cancellation. The run ends with .cancelled at the
    /// next stage boundary; the trainer is asked to stop immediately.
    public func cancel() {
        guard isRunning else { return }
        let orchestrator = self.orchestrator
        Task { await orchestrator?.cancel() }
        runTask?.cancel()
    }

    // MARK: Event application (MainActor)

    private func apply(_ event: PipelineEvent) {
        switch event {
        case let .stageStatus(stage, status):
            stageStatus[stage] = status
        case let .stageProgress(_, overall):
            overallFraction = max(overallFraction, overall)
        }
    }

    private func finish(with outcome: RunOutcome) {
        runTask = nil
        orchestrator = nil
        switch outcome {
        case let .success(asset):
            lastExport = asset
            overallFraction = 1
            phase = .finished(asset)
        case let .failure(message, wasCancelled):
            phase = wasCancelled ? .cancelled : .failed(message)
        }
    }
}
