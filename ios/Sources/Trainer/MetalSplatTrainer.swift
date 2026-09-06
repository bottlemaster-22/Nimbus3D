//
//  MetalSplatTrainer.swift
//  Trainer
//
//  CORE'S `SplatTrainer`, IMPLEMENTED. The on-device 3D Gaussian splatting
//  trainer: Metal, no Rust, no cloud, no third-party trainer wrapped.
//
//  ---------------------------------------------------------------------------
//  ONE ITERATION, IN ORDER
//  ---------------------------------------------------------------------------
//    pick a keyframe (round robin over a shuffled keyframe list)
//    upload its ground truth, its background, its depth samples
//    command buffer A   clear, preprocess, exclusive-scan the tile counts
//    read back the tile instance count            <- the one unavoidable sync
//    command buffer B   duplicate keys, radix sort by (tile, depth), tile
//                       ranges, rasterise forward, photometric + SSIM + depth
//                       losses, finalise, rasterise backward, preprocess
//                       backward, regularise, Adam on the splats, Adam on SH
//    read back the loss, the exposure gradient and the camera gradient
//    periodically: densify or relocate, carve free space, re-size the
//                  Mip-Splatting 3D filter, take a preview snapshot
//
//  There are two GPU synchronisations per iteration and they are both real:
//  the sort cannot be sized without knowing how many (Gaussian, tile) pairs
//  the frame produced, and the per-frame exposure cannot be stepped without
//  its gradient. Neither is hidden behind an estimate.
//
//  ---------------------------------------------------------------------------
//  WHAT IS NOT HERE, STATED RATHER THAN IMPLIED
//  ---------------------------------------------------------------------------
//  * `TrainingBudget.useHalfPrecision` is recorded as FALSE in the budget this
//    run reports, whatever was asked for. Every GPU struct in
//    TrainerGPULayouts.swift is fp32 by deliberate design (that file explains
//    why: `Float16` does not exist on x86_64 and would not build in an Intel
//    simulator), so claiming fp16 storage would be a lie about the file next
//    to this one.
//  * `SplatModel.observedDirectionsPath` is left nil. The honesty mask's
//    backing store is built by `Sources/Viewer`'s `ObservedDirectionBuilder`
//    from the capture's own depth maps, which needs no trained model and is
//    that module's to own; writing a second copy here would be two sources of
//    truth for the same file.
//

import Foundation
import Metal
import simd

public final class MetalSplatTrainer: SplatTrainer, @unchecked Sendable {

    // MARK: - Configuration

    private let tuning: TrainerTuning
    private let settings: SmartLossSettings

    // MARK: - GPU

    private var device: MTLDevice?
    private var queue: MTLCommandQueue?
    private var library: MTLLibrary?
    private var pipelines: TrainerPipelines?
    private var resources: TrainerResources?

    // MARK: - Shared state
    //
    // Touched from the training task and from `snapshot()` / `cancel()` /
    // `finishedModel()`, which the caller may invoke from anywhere. One lock,
    // held only for the assignment, never across GPU work.

    private let lock = NSLock()
    private var cancelRequested = false
    private var latestSnapshot: SplatCloud?
    private var latestModel: SplatModel?
    private var activeTask: Task<Void, Never>?
    /// True from the moment `train` accepts a run until that run's task has
    /// genuinely returned, cancelled runs included. This is the difference
    /// between "we asked it to stop" and "it has stopped": see the guard at
    /// the top of `train`.
    private var runInFlight = false

    // MARK: - Init

    /// The initializer the app uses. `TrainerTuning` is an internal knob bag,
    /// so it cannot appear in a public signature or as a public default
    /// argument: this stays public and takes the internal designated init's
    /// defaults, and anything inside the module that wants to override the
    /// tuning uses that one directly.
    public convenience init() {
        self.init(tuning: TrainerTuning(), settings: .default)
    }

    init(
        tuning: TrainerTuning = TrainerTuning(),
        settings: SmartLossSettings = .default
    ) {
        self.tuning = tuning
        self.settings = settings
    }

    // MARK: - SplatTrainer

    public func train(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        budget: TrainingBudget
    ) -> AsyncThrowingStream<TrainerProgress, Error> {

        // ONE RUN AT A TIME, AND THIS IS WHAT ENFORCES IT.
        //
        // Cancellation in Swift is cooperative, and this trainer only looks at
        // the flag between GPU steps, so a run that was told to stop a moment
        // ago can still be unwinding when a second `train` arrives. Resetting
        // `cancelRequested` blind would UN-CANCEL that first loop, and the two
        // of them would then share one set of GPU buffers, one `pipelines`,
        // one `exposureRecords` and one `heldOutFrameIndices`, because the app
        // registers a single trainer for the life of the process
        // (NimbusApp.swift). Refusing is the only honest answer.
        lock.lock()
        if runInFlight {
            lock.unlock()
            TrainerLog.general.error("A second train() arrived while one was still running")
            // Only the error goes out, deliberately: no progress tick. A tick
            // is built from `resources`, which belongs to the loop that is
            // still running, and reading it from here would be a race for the
            // sake of a sentence the thrown error already carries.
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: TrainerError.alreadyRunning.asNimbusError)
            }
        }
        runInFlight = true
        cancelRequested = false
        latestSnapshot = nil
        // Cleared, not merely overwritten on success. `finishedModel()` hands
        // back whatever is in here, and this was only ever assigned, so a
        // model left over from the PREVIOUS scan could be picked up by a run
        // that ended without building one and written into THIS scan's folder.
        latestModel = nil
        // Per-run, not per-instance. The app registers one trainer for the
        // life of the process, so leaving these full would write the previous
        // scan's exposures and held-out list into the next scan's model folder.
        exposureRecords.removeAll()
        heldOutFrameIndices.removeAll()
        lock.unlock()

        return AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                // Every way out of the block below ends here, so `runInFlight`
                // cannot be left stuck on by a throw.
                defer { self.markRunFinished() }
                do {
                    try await self.run(
                        bundle: bundle,
                        prePass: prePass,
                        at: ref,
                        budget: budget,
                        emit: { progress in continuation.yield(progress) }
                    )
                    continuation.finish()
                } catch let error as TrainerError {
                    continuation.yield(
                        self.progressTick(
                            stage: .failed,
                            iteration: 0,
                            total: budget.iterations,
                            splatCount: 0,
                            loss: nil,
                            thermal: ThermalLevel(ProcessInfo.processInfo.thermalState),
                            message: error.errorDescription ?? "The 3D model could not be built.",
                            previewAvailable: false
                        )
                    )
                    self.releaseGPU()
                    continuation.finish(throwing: error.asNimbusError)
                } catch is CancellationError {
                    // A terminal stage goes out before the stream finishes, so
                    // a UI watching only the stage sees "cancelled" rather than
                    // the stream simply stopping mid-sentence.
                    continuation.yield(
                        self.progressTick(
                            stage: .cancelled,
                            iteration: 0,
                            total: budget.iterations,
                            splatCount: 0,
                            loss: nil,
                            thermal: ThermalLevel(ProcessInfo.processInfo.thermalState),
                            message: "Stopped. Nothing was left running.",
                            previewAvailable: false
                        )
                    )
                    self.releaseGPU()
                    continuation.finish(throwing: NimbusError.cancelled)
                } catch {
                    self.releaseGPU()
                    continuation.finish(throwing: error)
                }
            }
            self.lock.lock()
            self.activeTask = task
            self.lock.unlock()
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func finishedModel() async -> SplatModel? {
        lock.lock()
        defer { lock.unlock() }
        return latestModel
    }

    /// A snapshot of the current field, for the live preview.
    ///
    /// Cheap by construction: the training loop copies the splat buffers into
    /// a `SplatCloud` every `snapshotIntervalIterations` between command
    /// buffers, and this hands back that copy. It does not stall the GPU and
    /// it does not read a buffer mid-flight, which is what "cheap enough to
    /// call between iterations" has to mean.
    ///
    /// Cheap is not free: it retains a whole cloud's worth of arrays. The
    /// caller is `ScanProcessingCoordinator`'s live preview, which asks at
    /// most once every few seconds and never while a previous one is still
    /// being uploaded.
    public func snapshot() async throws -> SplatCloud {
        lock.lock()
        let cloud = latestSnapshot
        lock.unlock()
        guard let cloud else {
            throw NimbusError.trainingFailed("there is nothing to preview yet")
        }
        return cloud
    }

    public func cancel() async {
        lock.lock()
        cancelRequested = true
        let task = activeTask
        lock.unlock()
        task?.cancel()
        // The loop frees the GPU buffers itself on the way out, so that the
        // release happens on the thread that owns them rather than racing the
        // command buffer that may still be in flight.
        //
        // This returning does NOT mean the run has stopped. It has been ASKED
        // to stop, and it notices between GPU steps. `waitUntilIdle()` is how
        // a caller finds out that it really has.
    }

    /// True while a run is on the GPU, or still unwinding after a `cancel()`.
    ///
    /// A caller about to start a second run must treat this as "no": there is
    /// one set of GPU buffers on this object and one `latestModel`.
    public var isTraining: Bool {
        lock.lock()
        defer { lock.unlock() }
        return runInFlight
    }

    /// Waits until a run that was told to stop has genuinely stopped.
    ///
    /// Cancellation is cooperative, so `cancel()` returning and the GPU going
    /// quiet are two different moments. This is the second one. It returns
    /// straight away when nothing is running, and it does not itself cancel
    /// anything: a caller that wants the run to end must call `cancel()`
    /// first, or this waits for the run to finish normally.
    public func waitUntilIdle() async {
        lock.lock()
        let task = activeTask
        lock.unlock()
        guard let task else { return }
        _ = await task.value
    }

    /// Marks the end of a run, however it ended. Paired with the `runInFlight`
    /// guard at the top of `train`.
    ///
    /// `activeTask` is deliberately left pointing at the finished task rather
    /// than cleared here: `train` assigns it a moment after the task is
    /// created, and clearing from inside the task could race that assignment
    /// and leave a live run unreachable by `cancel()`. Awaiting or cancelling
    /// an already-finished task is free.
    private func markRunFinished() {
        lock.lock()
        runInFlight = false
        lock.unlock()
    }

    // MARK: - Preparation

    /// Builds the device, the queue, the library and every pipeline, and
    /// checks the Swift and Metal struct layouts against each other BEFORE a
    /// single buffer is allocated. A layout mismatch here is the difference
    /// between a trained scan and plausible garbage, so it stops the run with
    /// a sentence naming the offending struct.
    private func prepare() throws {
        if let existing = pipelines, device != nil, queue != nil, library != nil {
            _ = existing
            return
        }

        if let problem = TrainerGPULayouts.verify() {
            throw TrainerError.layoutMismatch(problem)
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TrainerError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw TrainerError.noMetalDevice
        }
        queue.label = "trainer.queue"
        guard let library = device.makeDefaultLibrary() else {
            throw TrainerError.noShaderLibrary
        }

        self.device = device
        self.queue = queue
        self.library = library
        self.pipelines = try TrainerPipelines(device: device, library: library)

        // Logger messages are built as ONE interpolated literal. `os.Logger`
        // takes an `OSLogMessage`, not a `String`, so a `+` between two pieces
        // does not compile: there is nothing to concatenate.
        let kernelCount = TrainerKernel.all.count
        TrainerLog.gpu.info(
            "Trainer ready on \(device.name, privacy: .public) with \(kernelCount) kernels"
        )
    }

    private func releaseGPU() {
        resources = nil
        pipelines = nil
        library = nil
        queue = nil
        device = nil
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelRequested
    }

    private func checkCancellation() throws {
        if isCancelled || Task.isCancelled { throw CancellationError() }
    }

    // MARK: - The run

    // swiftlint:disable:next function_body_length cyclomatic_complexity
    private func run(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        budget: TrainingBudget,
        emit: @escaping @Sendable (TrainerProgress) -> Void
    ) async throws {

        let governor = TrainerBudgetGovernor(budget: budget)

        // --- THE CENSUS -------------------------------------------------------
        //
        // Where this run's geometry goes, counted as it happens and written
        // once at the end to `model/train_census.json`. See TrainerCensus.swift
        // for why: four separate faults each destroyed most of the owner's
        // first scan and not one of them logged, warned or failed.
        //
        // The write is in a `defer` on purpose. A run that THREW is the run you
        // most want this for, and a run that was cancelled or stopped by heat
        // is the second. Putting the write on the success path only would give
        // us a census for exactly the runs that did not need one.
        var census = TrainerCensus(scanID: bundle.scanID, requested: budget)
        defer {
            if census.outcome == TrainerCensus.unfinishedOutcome,
               isCancelled || Task.isCancelled
            {
                census.outcome = "cancelled"
            }
            // Read at the last possible moment, so a budget the governor
            // lowered on the way out is still the one that gets recorded.
            census.budgetAsRun = TrainerCensusBudget(governor.current)
            census.budgetReductions = governor.changes.map { TrainerCensusBudgetReduction($0) }
            TrainerCensusWriter.write(census, at: ref)
        }

        emit(
            progressTick(
                stage: .preparing,
                iteration: 0,
                total: governor.current.iterations,
                splatCount: 0,
                loss: nil,
                thermal: governor.thermalLevel,
                message: "Getting the graphics ready.",
                previewAvailable: false
            )
        )

        try prepare()
        try checkCancellation()

        // --- The SMART layer -------------------------------------------------
        // Every one of these is optional. A scan with no pre-pass sidecars
        // still trains, as plain 3DGS with LiDAR seeding, which is honest
        // degradation rather than a hard failure. What is NOT done is
        // pretending a missing sidecar is a neutral one.
        let smart = try await loadSmartLayer(bundle: bundle, prePass: prePass, at: ref)

        // --- Keyframes --------------------------------------------------------
        let keyframes = selectKeyframes(bundle: bundle, prePass: prePass, budget: governor.current)
        guard !keyframes.isEmpty else { throw TrainerError.noKeyframes }

        let slices = TrainerSlicePlanner.plan(
            bundle: bundle,
            prePass: prePass,
            keyframes: keyframes,
            budget: governor.current,
            heldOutFraction: tuning.heldOutFraction
        )
        guard !slices.isEmpty else { throw TrainerError.noKeyframes }

        TrainerLog.general.info(
            "Training \(slices.count) time slice(s) from \(keyframes.count) keyframes"
        )
        census.keyframesSelected = keyframes.count
        census.sliceCount = slices.count

        // --- Train each slice --------------------------------------------------
        var parts: [(slice: TrainerSlice, cloud: SplatCloud)] = []
        var mergedSoFar: [SplatCloud] = []
        var totalIterationsRun = 0
        var lastPSNR: Float?

        for slice in slices {
            try checkCancellation()
            let cloud = try await trainSlice(
                slice: slice,
                sliceCount: slices.count,
                bundle: bundle,
                prePass: prePass,
                at: ref,
                governor: governor,
                smart: smart,
                completedParts: mergedSoFar,
                emit: emit,
                iterationsRunSoFar: &totalIterationsRun,
                heldOutPSNR: &lastPSNR,
                census: &census
            )
            parts.append((slice, cloud))
            mergedSoFar.append(cloud)
        }

        try checkCancellation()

        // --- Merge, one owner per region ---------------------------------------
        emit(
            progressTick(
                stage: .finalizing,
                iteration: totalIterationsRun,
                total: governor.current.iterations,
                splatCount: parts.reduce(0) { $0 + $1.cloud.count },
                loss: nil,
                thermal: governor.thermalLevel,
                message: slices.count > 1
                    ? "Joining the parts of your scan together."
                    : "Tidying up the finished model.",
                previewAvailable: true
            )
        )

        let merged = try TrainerSliceMerger.merge(
            parts: parts, slices: slices, splatCap: governor.current.splatCap
        )
        if slices.count > 1 {
            TrainerLog.general.info("\(merged.report.summary, privacy: .public)")
        }

        // The merge is the last place a Gaussian can quietly disappear, and on
        // a multi-slice run it can delete a lot of them for a reason that is
        // entirely correct (one owner per region) or entirely wrong (the
        // regions do not match the geometry). The census cannot tell those
        // apart, but it can put the number where somebody will see it.
        let splatsHandedToMerge = parts.map { $0.cloud.count }
        census.merge = TrainerCensusMerge(
            splatsInPerSlice: splatsHandedToMerge,
            splatsIn: splatsHandedToMerge.reduce(0, +),
            kept: merged.report.kept,
            droppedToAnotherOwner: merged.report.droppedToOtherOwners,
            trimmedToCap: merged.report.trimmedToCap,
            splatsOut: merged.cloud.count
        )
        census.finalSplatCount = merged.cloud.count

        lock.lock()
        latestSnapshot = merged.cloud
        lock.unlock()

        // --- Write everything out -----------------------------------------------
        let model = try await writeModel(
            cloud: merged.cloud,
            bundle: bundle,
            at: ref,
            governor: governor,
            iterationsCompleted: totalIterationsRun,
            heldOutPSNR: lastPSNR,
            background: smart.background
        )

        lock.lock()
        latestModel = model
        lock.unlock()

        releaseGPU()

        // Set here rather than in the `defer`: reaching this line is the only
        // thing that makes "completed" true.
        // The same 95 per cent tolerance the census's own alert uses: slice
        // iteration budgets are integer shares of the whole and round down, so
        // a full run legitimately lands a few iterations short.
        census.outcome = totalIterationsRun * 100 < governor.current.iterations * 95
            ? "stopped early"
            : "completed"

        emit(
            progressTick(
                stage: .done,
                iteration: totalIterationsRun,
                total: governor.current.iterations,
                splatCount: merged.cloud.count,
                loss: nil,
                thermal: governor.thermalLevel,
                message: doneMessage(cloud: merged.cloud, psnr: lastPSNR, governor: governor),
                previewAvailable: true
            )
        )
    }

    // MARK: - One slice

    // swiftlint:disable:next function_body_length cyclomatic_complexity function_parameter_count
    private func trainSlice(
        slice: TrainerSlice,
        sliceCount: Int,
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        governor: TrainerBudgetGovernor,
        smart: SmartLayer,
        completedParts: [SplatCloud],
        emit: @escaping @Sendable (TrainerProgress) -> Void,
        iterationsRunSoFar: inout Int,
        heldOutPSNR: inout Float?,
        census: inout TrainerCensus
    ) async throws -> SplatCloud {

        guard let device, let queue, let pipelines else { throw TrainerError.noMetalDevice }
        guard !slice.keyframes.isEmpty else { return SplatCloud.empty(shDegree: .zero) }

        let sliceLabel = slice.label(of: sliceCount)
        let shDegree = governor.current.shDegree
        let shCoefficientCount = 1 + shDegree.restCoefficientCount

        // --- Render size, from the first decodable frame -----------------------
        let supervision = TrainerSupervisionBuilder(
            bundle: bundle,
            prePass: prePass,
            at: ref,
            longEdgePixels: governor.current.renderLongEdgePixels,
            settings: settings,
            tuning: tuning,
            trust: smart.trust,
            authority: smart.authority,
            edges: smart.edges,
            background: smart.background
        )

        var renderSize = TrainerBudgetGovernor.renderSize(
            forLongEdge: governor.current.renderLongEdgePixels,
            intrinsics: bundle.intrinsics
        )

        // --- Census: open this slice's row now ---------------------------------
        // Opened before anything can go wrong and filled in as the slice runs,
        // so a slice that throws half way through still leaves behind
        // everything it had managed to measure.
        census.slices.append(
            TrainerCensusSlice(
                index: slice.index,
                label: sliceLabel,
                keyframesTrained: slice.keyframes.count,
                keyframesHeldOut: slice.heldOutKeyframes.count,
                splatCapAsked: slice.splatCapShare,
                renderWidth: renderSize.width,
                renderHeight: renderSize.height,
                iterationsRequested: slice.iterationBudget
            )
        )
        let censusRow = census.slices.count - 1

        emit(
            progressTick(
                stage: .initializing,
                iteration: iterationsRunSoFar,
                total: governor.current.iterations,
                splatCount: 0,
                loss: nil,
                thermal: governor.thermalLevel,
                message: sliceLabel.isEmpty
                    ? "Working out where to start from."
                    : "Working out where to start from (\(sliceLabel)).",
                previewAvailable: false
            )
        )

        // --- What actually fits, MEASURED, before anything is built --------------
        // The measurement comes first so the seeder thins to the real cap in
        // one pass. Seeding to the asked-for cap and truncating afterwards
        // would throw away a carefully spread set and keep an arbitrary
        // prefix of it, which is exactly the uneven start the seeder's own
        // thinning exists to avoid.
        let measuredCap = governor.initialSplatCap(
            shCoefficientCount: shCoefficientCount,
            pixelCount: renderSize.pixelCount
        )
        renderSize = TrainerBudgetGovernor.renderSize(
            forLongEdge: governor.current.renderLongEdgePixels,
            intrinsics: bundle.intrinsics
        )
        let effectiveCap = Swift.max(
            Swift.min(Swift.min(slice.splatCapShare, measuredCap), governor.current.splatCap),
            5_000
        )

        // --- Seed ----------------------------------------------------------------
        var sliceBudget = governor.current
        sliceBudget.splatCap = effectiveCap
        let seedResult = try TrainerInitializer.seed(
            bundle: bundle,
            prePass: prePass,
            at: ref,
            keyframes: slice.keyframes,
            budget: sliceBudget,
            trust: smart.trust,
            authority: smart.authority,
            edges: smart.edges,
            settings: settings
        )
        TrainerLog.general.info("\(seedResult.summary, privacy: .public)")

        // What the seeder actually produced, including the disc-versus-blob
        // split, which is the seeder's trust gate made countable.
        census.slices[censusRow].seedSource = seedResult.source
        census.slices[censusRow].seedFramesUsed = seedResult.framesUsed
        census.slices[censusRow].seedSamplesConsidered = seedResult.samplesConsidered
        census.slices[censusRow].seedSamplesRejected = seedResult.samplesRejected
        census.slices[censusRow].seedsPinnedAsDiscs = seedResult.seedsPinnedAsDiscs
        census.slices[censusRow].seedsStretchedAlongRay = seedResult.seedsStretchedAlongRay
        census.slices[censusRow].seedsBuilt = seedResult.seeds.count
        census.slices[censusRow].seedMedianSpacingMillimetres =
            Int((seedResult.medianSpacingMeters * 1000).rounded())
        census.slices[censusRow].splatCapMeasuredAffordable = measuredCap
        census.slices[censusRow].splatCapEffective = effectiveCap
        census.slices[censusRow].renderWidth = renderSize.width
        census.slices[censusRow].renderHeight = renderSize.height

        try checkCancellation()

        // --- Allocate ---------------------------------------------------------------
        // The seeder already thinned to `effectiveCap`, so this is both the
        // room the seeds need and the room densification is allowed to grow
        // into. The `max` is a floor against a pathological measurement, not a
        // fudge: a capacity below the seed count would silently drop starting
        // points at upload.
        let capacity = Swift.max(effectiveCap, seedResult.seeds.count)

        let depthSampleCapacity = Swift.max(
            bundle.settings.depthWidth * bundle.settings.depthHeight, 1
        )
        let resources = try TrainerResources(
            device: device,
            splatCapacity: capacity,
            renderSize: renderSize,
            shCoefficientCount: shCoefficientCount,
            depthSampleCapacity: depthSampleCapacity,
            instanceCapacity: Swift.max(capacity * 8, TrainerGPUConstants.scanBlockElements)
        )
        self.resources = resources
        var gpu = TrainerGPU(pipelines: pipelines, resources: resources)

        var splatCount = TrainerInitializer.upload(seedResult.seeds, into: resources)
        // Recorded BEFORE the guard: "the seeder built 180,000 and 0 reached
        // the GPU" is a different fault from "the seeder built 0", and the
        // throw below cannot tell them apart on its own.
        census.slices[censusRow].seedsUploaded = splatCount
        // The flag is what tells `model/census.json` that the zero above is a
        // measurement. A slice row is opened before seeding, so without it a
        // run that died in allocation would report "the build started with 0
        // points" as a confident fact about a number nobody took.
        census.slices[censusRow].seedsUploadedCounted = true
        guard splatCount > 0 else {
            throw TrainerError.nothingToTrain("no starting points survived the memory budget")
        }

        // --- Per-run state --------------------------------------------------------
        let densifier = TrainerDensifier(tuning: tuning, settings: settings)
        let sceneExtent = Swift.max(
            bundle.sceneBounds?.longestEdgeMeters ?? slice.bounds.longestEdgeMeters, 0.5
        )
        var exposures: [FrameID: SIMD2<Float>] = [:]      // (gain, bias)
        var cameraDeltas: [FrameID: Pose] = [:]
        var lossEMA: Float?
        var order = Array(slice.keyframes.indices)
        var orderCursor = 0
        // A fixed shuffle rather than a fresh one each epoch: reproducible, and
        // an epoch still visits every keyframe exactly once, which is what
        // matters for coverage.
        order.shuffle()

        let totalIterations = Swift.max(slice.iterationBudget, 1)
        // This slice's share of the whole run. When the governor lowers the
        // global iteration count mid-run, the slice shrinks with it rather
        // than one slice keeping its full length and the last one getting
        // nothing.
        let sliceFraction = Float(slice.iterationBudget)
            / Float(Swift.max(governor.ceiling.iterations, 1))
        var iteration = 0
        var lastEmit = Date.distantPast
        // How many polls in a row the phone has been too hot to work. A phone
        // that never cools has to end the run rather than sit in a loop
        // reporting "paused" forever, so this is a real limit and not a
        // formality: at the default 5 second poll it is ten minutes of waiting.
        var consecutivePauses = 0
        let maximumConsecutivePauses = 120

        // --- Census: the gates, copied from the point of use --------------------
        // Written from here, immediately above the loop that reads them, rather
        // than from wherever the structs were built. A setting that is declared
        // and never read looks exactly like a setting that works, and the only
        // way to tell them apart on paper is to put the value next to the
        // behaviour it is supposed to be producing.
        if census.gates == nil {
            census.gates = TrainerCensusGates(
                densifyStartFraction: tuning.densifyStartFraction,
                densifyEndFraction: tuning.densifyEndFraction,
                densifyIntervalIterations: tuning.densifyIntervalIterations,
                pruneStartFraction: settings.pruneStartFraction,
                pruneEndFraction: settings.pruneEndFraction,
                carveIntervalIterations: tuning.carveIntervalIterations,
                // Zero, because zero is what `TrainerDensifier` compares
                // against. See `TrainerCensusGates.densifyScoreFloor`:
                // `tuning.absGradThreshold` is a leftover the densifier no
                // longer reads, and printing it here would be a dead setting
                // dressed up as a live one.
                densifyScoreFloor: 0,
                pruneOpacity: tuning.pruneOpacity,
                pruneMaxWorldScaleFraction: tuning.pruneMaxWorldScaleFraction,
                pruneMaxScreenRadiusPx: tuning.pruneMaxScreenRadiusPx,
                maxGrowthFractionPerPass: tuning.maxGrowthFractionPerPass,
                maxRelocationFractionPerPass: tuning.maxRelocationFractionPerPass,
                pruneMaxFractionPerPass: settings.pruneMaxFractionPerPass,
                warmupFraction: tuning.warmupFraction,
                binarizeLastFraction: tuning.binarizeLastFraction,
                minimumAuthorityForDepth: settings.minimumAuthorityForDepth
            )
        }

        iterationLoop: while true {
            let effectiveTotal = Swift.max(
                Swift.min(
                    totalIterations,
                    Int(Float(governor.current.iterations) * sliceFraction)
                ),
                1
            )
            if iteration >= effectiveTotal { break iterationLoop }
            try checkCancellation()

            // Three integer stores. The governor never reads these to make a
            // decision; they are there so that every budget reduction it makes
            // below can record WHEN it happened and how many Gaussians were
            // alive at the time, which is the difference between "the cap was
            // cut" and "the cap was cut below the population and deleted it".
            governor.mark(
                sliceIndex: slice.index,
                iteration: iterationsRunSoFar + iteration,
                splatCount: splatCount
            )

            // --- Heat and memory, measured -------------------------------------
            let thermal = governor.thermalVerdict()
            switch thermal.verdict {
            case .abort:
                TrainerLog.budget.error("Thermal abort at iteration \(iteration)")
                census.slices[censusRow].stopReason =
                    "the phone got too hot to keep going (thermal abort)"
                emit(
                    progressTick(
                        stage: .pausedThermal,
                        iteration: iterationsRunSoFar + iteration,
                        total: governor.current.iterations,
                        splatCount: splatCount,
                        loss: lossEMA,
                        thermal: thermal.level,
                        message: "Your phone got too warm to keep going. What is finished has "
                            + "been kept.",
                        previewAvailable: true
                    )
                )
                break iterationLoop

            case .pause:
                consecutivePauses += 1
                if consecutivePauses > maximumConsecutivePauses {
                    TrainerLog.budget.error(
                        "Still too hot after \(consecutivePauses) checks; stopping and keeping what is built"
                    )
                    census.slices[censusRow].stopReason =
                        "the phone never cooled down (paused \(consecutivePauses) times in a row)"
                    emit(
                        progressTick(
                            stage: .pausedThermal,
                            iteration: iterationsRunSoFar + iteration,
                            total: governor.current.iterations,
                            splatCount: splatCount,
                            loss: lossEMA,
                            thermal: thermal.level,
                            message: "Your phone did not cool down enough to carry on. What is "
                                + "finished has been kept.",
                            previewAvailable: true
                        )
                    )
                    break iterationLoop
                }
                // Said once when the pause begins, and then every tenth check,
                // rather than every few seconds forever.
                if consecutivePauses == 1 || consecutivePauses % 10 == 0 {
                    emit(
                        progressTick(
                            stage: .pausedThermal,
                            iteration: iterationsRunSoFar + iteration,
                            total: governor.current.iterations,
                            splatCount: splatCount,
                            loss: lossEMA,
                            thermal: thermal.level,
                            message: "Paused while your phone cools down. This carries on by itself.",
                            previewAvailable: true
                        )
                    )
                }
                try await Task.sleep(
                    nanoseconds: UInt64(
                        Swift.max(governor.current.thermalPolicy.sampleIntervalSeconds, 1) * 1_000_000_000
                    )
                )
                continue

            case .degrade:
                consecutivePauses = 0
                if iteration > 0, iteration % 100 == 0,
                   let change = governor.degradeForHeat(
                       level: thermal.level, currentSplatCount: splatCount
                   )
                {
                    try applyBudgetChange(
                        change,
                        resources: resources,
                        gpu: &gpu,
                        splatCount: &splatCount,
                        renderSize: &renderSize,
                        supervision: supervision,
                        bundle: bundle,
                        governor: governor
                    )
                    emit(
                        progressTick(
                            stage: stage(for: iteration, of: effectiveTotal),
                            iteration: iterationsRunSoFar + iteration,
                            total: governor.current.iterations,
                            splatCount: splatCount,
                            loss: lossEMA,
                            thermal: thermal.level,
                            message: change.message,
                            previewAvailable: true
                        )
                    )
                }

            case .run:
                consecutivePauses = 0
            }

            if iteration % 50 == 0 {
                let reading = governor.measureMemory(resources: resources)
                if let change = governor.degradeForMemory(
                    reading: reading,
                    currentSplatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    pixelCount: renderSize.pixelCount
                ) {
                    emit(
                        progressTick(
                            stage: .pausedMemory,
                            iteration: iterationsRunSoFar + iteration,
                            total: governor.current.iterations,
                            splatCount: splatCount,
                            loss: lossEMA,
                            thermal: thermal.level,
                            message: change.message,
                            previewAvailable: true
                        )
                    )
                    try applyBudgetChange(
                        change,
                        resources: resources,
                        gpu: &gpu,
                        splatCount: &splatCount,
                        renderSize: &renderSize,
                        supervision: supervision,
                        bundle: bundle,
                        governor: governor
                    )
                }
            }

            // --- Pick a keyframe -------------------------------------------------
            if orderCursor >= order.count { orderCursor = 0 }
            let frame = slice.keyframes[order[orderCursor]]
            orderCursor += 1

            guard let frameSupervision = supervision.build(
                frame: frame,
                iteration: iteration,
                // The run that will actually happen, for the same reason
                // `progressFraction` uses it: this drives the depth-loss decay
                // and the SH degree schedule, and keying those to a length the
                // run will never reach means their tails never execute.
                totalIterations: effectiveTotal
            ) else {
                // A frame whose photo would not decode. Counted rather than
                // skipped in silence: a run where most iterations land here is
                // a run that trained on almost nothing, and the wall clock
                // looks identical either way.
                census.slices[censusRow].iterationsSkippedNoSupervision += 1
                iteration += 1
                continue
            }

            // The builder fixes the render grid from the first decodable photo.
            // If that is not the grid the buffers were made at, the buffers are
            // rebuilt once rather than every frame being resampled.
            if frameSupervision.renderSize != renderSize {
                renderSize = frameSupervision.renderSize
                try resources.resizeRenderSize(to: renderSize)
                gpu = TrainerGPU(pipelines: pipelines, resources: resources)
            }

            let exposure = exposures[frame.index] ?? SIMD2<Float>(1, 0)

            let step = try runIteration(
                gpu: gpu,
                resources: resources,
                queue: queue,
                supervision: frameSupervision,
                cameraDelta: cameraDeltas[frame.index],
                exposure: exposure,
                splatCount: splatCount,
                iteration: iteration,
                totalIterations: totalIterations,
                sceneExtent: sceneExtent,
                shCoefficientCount: shCoefficientCount,
                trust: smart.trust,
                lossEMA: &lossEMA,
                exposures: &exposures,
                cameraDeltas: &cameraDeltas,
                background: smart.background,
                frame: frame
            )
            switch step {
            case .stepped:
                break
            case .skippedNothingToRender:
                census.slices[censusRow].iterationsSkippedNothingToRender += 1
            case .grewTileBufferAndRetried:
                census.slices[censusRow].iterationsSkippedGrowingTileBuffer += 1
            }

            // --- Periodic work -----------------------------------------------------
            //
            // Measured against the run that is ACTUALLY GOING TO HAPPEN, not
            // against the one originally planned.
            //
            // This divided by `totalIterations` (the planned slice budget) while
            // the loop exits at `effectiveTotal` (what heat and memory have left
            // of it). So a run cut from 3,000 to 1,500 did not do half the work:
            // it stopped at progressFraction 0.5 and everything scheduled past
            // that point NEVER RAN AT ALL. Late opacity binarization (the last
            // 20 percent), the final spherical-harmonic degree, the tail of the
            // learning-rate decay and the end of the prune window were simply
            // deleted from the run, silently, while the app still reported
            // success. Compressing the schedule instead means a shortened run
            // is a faster version of the same run rather than a truncated one.
            let progressFraction = Float(iteration) / Float(Swift.max(effectiveTotal, 1))

            if iteration > 0,
               iteration % Swift.max(tuning.filter3DIntervalIterations, 1) == 0
            {
                try updateFilter3D(
                    gpu: gpu,
                    resources: resources,
                    queue: queue,
                    keyframes: slice.keyframes,
                    supervision: supervision,
                    cameraDeltas: cameraDeltas,
                    splatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    renderSize: renderSize
                )
            }

            let inDensifyWindow = progressFraction >= tuning.densifyStartFraction
                && progressFraction <= tuning.densifyEndFraction
            // `SmartLossSettings.pruneStartFraction` and `pruneEndFraction`
            // existed, were defaulted and were assigned, and were then read
            // nowhere at all, so pruning ran on the densify interval from the
            // first pass to the last. Reading them here is what makes the two
            // settings mean something.
            let inPruneWindow = progressFraction >= settings.pruneStartFraction
                && progressFraction <= settings.pruneEndFraction
            if iteration > 0,
               iteration % Swift.max(tuning.densifyIntervalIterations, 1) == 0
            {
                let carveDue = iteration % Swift.max(tuning.carveIntervalIterations, 1) == 0
                let outcome = try densifier.run(
                    resources: resources,
                    splatCount: splatCount,
                    // The wall is whichever is lowest right now: what this
                    // slice was allotted, what the device measured it could
                    // hold, and whatever the governor has since cut it to.
                    splatCap: Swift.min(
                        Swift.min(effectiveCap, governor.current.splatCap),
                        resources.splatCapacity
                    ),
                    sceneExtentMeters: sceneExtent,
                    allowGrowth: inDensifyWindow,
                    allowPrune: inPruneWindow,
                    carver: carveDue ? smart.carver : nil
                )
                splatCount = outcome.splatCountAfter

                // One census row per pass. This is the whole ledger of where
                // the geometry went: what the windows said, what room there
                // was, what scored, what was created, and what each prune
                // reason and the carve removed. Appending a small struct every
                // hundred iterations is the entire cost.
                census.densifyPasses.append(
                    TrainerCensusDensifyPass(
                        sliceIndex: slice.index,
                        iteration: iteration,
                        progressFraction: progressFraction,
                        carverAvailable: smart.carver != nil,
                        outcome: outcome
                    )
                )

                if outcome.changedTopology || outcome.relocated > 0 {
                    try resetDensifyStats(
                        gpu: gpu, resources: resources, queue: queue, splatCount: splatCount
                    )
                }
                if let summary = outcome.summary {
                    TrainerLog.densify.info(
                        "Iteration \(iteration): \(summary, privacy: .public)"
                    )
                }
            }

            if iteration % Swift.max(tuning.snapshotIntervalIterations, 1) == 0 {
                let cloud = readCloud(
                    resources: resources, count: splatCount, shDegree: shDegree
                )
                lock.lock()
                latestSnapshot = mergePreview(completedParts: completedParts, current: cloud)
                lock.unlock()
            }

            // --- Progress ------------------------------------------------------------
            let now = Date()
            if now.timeIntervalSince(lastEmit) > 0.5 || iteration == effectiveTotal - 1 {
                lastEmit = now
                emit(
                    progressTick(
                        stage: stage(for: iteration, of: totalIterations),
                        iteration: iterationsRunSoFar + iteration,
                        total: governor.current.iterations,
                        splatCount: splatCount,
                        loss: lossEMA,
                        thermal: governor.thermalLevel,
                        message: stageMessage(
                            for: iteration, of: totalIterations, sliceLabel: sliceLabel
                        ),
                        previewAvailable: iteration >= tuning.snapshotIntervalIterations
                    )
                )
            }

            iteration += 1
        }

        iterationsRunSoFar += iteration

        census.iterationsCompleted += iteration
        census.slices[censusRow].iterationsCompleted = iteration
        census.slices[censusRow].splatCountAtEndOfTraining = splatCount
        // The size the buffers were actually at when the slice ended, which is
        // not the size it started at if the governor stepped the resolution
        // down mid-run.
        census.slices[censusRow].renderWidth = renderSize.width
        census.slices[censusRow].renderHeight = renderSize.height
        if census.slices[censusRow].stopReason == TrainerCensus.unfinishedOutcome {
            census.slices[censusRow].stopReason = "ran out its iterations"
        }

        // The learned per-frame exposures belong to the whole run, not to this
        // slice: `model/exposure.bin` is keyed by frame index and a frame in
        // an overlap was trained by two slices. Last writer wins, which is the
        // slice that saw it most recently.
        lock.lock()
        for (frameIndex, value) in exposures { exposureRecords[frameIndex] = value }
        lock.unlock()

        // --- Held-out evaluation -------------------------------------------------
        if !slice.heldOutKeyframes.isEmpty {
            // Written out at the end as `model/held_out_frames.json`. The
            // review screen's photo-versus-scan slider is only honest if it
            // compares against photos the model never saw, and without this
            // list it has to guess (every 20th frame) and say so on screen.
            lock.lock()
            for frame in slice.heldOutKeyframes { heldOutFrameIndices.insert(frame.index) }
            lock.unlock()

            let psnr = try evaluateHeldOut(
                gpu: gpu,
                resources: resources,
                queue: queue,
                frames: slice.heldOutKeyframes,
                supervision: supervision,
                cameraDeltas: cameraDeltas,
                exposures: exposures,
                splatCount: splatCount,
                shCoefficientCount: shCoefficientCount,
                renderSize: renderSize
            )
            census.slices[censusRow].heldOutPSNR = psnr
            if let psnr {
                heldOutPSNR = heldOutPSNR.map { Swift.min($0, psnr) } ?? psnr
                let formatted = String(format: "%.2f", psnr)
                TrainerLog.general.info(
                    "Held-out PSNR for this part: \(formatted, privacy: .public) dB"
                )
            }
        }

        let cloud = readCloud(resources: resources, count: splatCount, shDegree: shDegree)

        // `readCloud` drops a non-finite Gaussian rather than exporting a NaN,
        // and it does that silently. The difference between what went in and
        // what came out is the count of those, and it should always be zero.
        census.slices[censusRow].droppedNonFiniteOnReadback =
            Swift.max(splatCount - cloud.count, 0)
        census.slices[censusRow].splatsHandedToMerge = cloud.count

        // The next slice allocates its own buffers, so this one's go now
        // rather than at the end of the whole run.
        self.resources = nil

        return cloud
    }

    // MARK: - One iteration

    /// What one call to `runIteration` actually did.
    ///
    /// Two of these three cases used to be a bare `return` in the middle of the
    /// function, which is a whole iteration doing no work and saying nothing.
    /// A run where most iterations end that way is a run that trained on almost
    /// nothing, and from the outside it looks exactly like a slow one, so the
    /// loop counts them.
    private enum StepResult {
        /// A full forward, backward and Adam step happened.
        case stepped
        /// There was nothing to render: no pixels or no Gaussians.
        case skippedNothingToRender
        /// The tile-instance buffer was too small, so it was grown and this
        /// frame was abandoned to be retried on the next iteration.
        case grewTileBufferAndRetried
    }

    // swiftlint:disable:next function_body_length function_parameter_count
    private func runIteration(
        gpu: TrainerGPU,
        resources: TrainerResources,
        queue: MTLCommandQueue,
        supervision: TrainerFrameSupervision,
        cameraDelta: Pose?,
        exposure: SIMD2<Float>,
        splatCount: Int,
        iteration: Int,
        totalIterations: Int,
        sceneExtent: Float,
        shCoefficientCount: Int,
        trust: TwoScaleTrustField?,
        lossEMA: inout Float?,
        exposures: inout [FrameID: SIMD2<Float>],
        cameraDeltas: inout [FrameID: Pose],
        background: DirectionalBackgroundModel?,
        frame: CaptureFrame
    ) throws -> StepResult {

        let size = resources.renderSize
        let pixelCount = size.pixelCount
        guard pixelCount > 0, splatCount > 0 else { return .skippedNothingToRender }

        // --- Upload this frame's supervision -------------------------------------
        resources.gtColor.writeArray(supervision.groundTruth)
        if supervision.hasBackground, !supervision.background.isEmpty {
            resources.bgColor.writeArray(supervision.background)
        }
        let sampleCount = Swift.min(
            supervision.depthSamples.count, resources.depthSampleCapacity
        )
        if sampleCount > 0 {
            resources.depthSamples.writeArray(Array(supervision.depthSamples.prefix(sampleCount)))
        }

        // --- Uniforms --------------------------------------------------------------
        var camera = cameraUniforms(
            supervision: supervision,
            cameraDelta: cameraDelta,
            size: size,
            splatCount: splatCount,
            shCoefficientCount: shCoefficientCount,
            iteration: iteration,
            totalIterations: totalIterations
        )

        var loss = TrainerLossUniforms()
        loss.pixelCount = UInt32(pixelCount)
        loss.width = UInt32(size.width)
        loss.height = UInt32(size.height)
        loss.lambdaSSIM = tuning.lambdaSSIM
        loss.frameWeight = supervision.qcWeight
        loss.exposureGain = exposure.x
        loss.exposureBias = exposure.y
        loss.depthScale = trust?.depthLossScale(iteration: iteration, of: totalIterations)
            ?? TwoScaleTrustField.depthLossScale(
                iteration: iteration, of: totalIterations, floor: settings.depthScheduleFloor
            )
        loss.depthSampleCount = UInt32(sampleCount)
        loss.bimodalWeight = settings.bimodalWeight
        loss.transitionWidthWeight = settings.transitionWidthWeight
        loss.freeSpaceWeight = settings.freeSpaceLowerBoundWeight
        loss.alphaSupervisionWeight = tuning.alphaSupervisionWeight
        loss.hasBackground = supervision.hasBackground ? 1 : 0

        // --- Command buffer A: preprocess and size the sort -------------------------
        guard let bufferA = queue.makeCommandBuffer(),
              let encoderA = bufferA.makeComputeCommandEncoder()
        else { throw TrainerError.noMetalDevice }
        encoderA.label = "trainer.preprocess"
        gpu.clearPerIteration(encoderA, splatCount: splatCount)
        gpu.preprocess(encoderA, camera: &camera, splatCount: splatCount)
        gpu.exclusiveScan(
            encoderA,
            input: resources.tilesTouched,
            output: resources.offsets,
            count: splatCount
        )
        encoderA.endEncoding()
        bufferA.commit()
        bufferA.waitUntilCompleted()

        // The one unavoidable readback: how many (Gaussian, tile) pairs this
        // frame produced. The exclusive scan means the total is the last
        // offset plus the last count.
        let lastOffset = resources.offsets.readElement(UInt32.self, at: splatCount - 1) ?? 0
        let lastTouched = resources.tilesTouched.readElement(UInt32.self, at: splatCount - 1) ?? 0
        var instanceCount = Int(lastOffset) + Int(lastTouched)

        if instanceCount > resources.instanceCapacity {
            // `trainer_duplicate_keys` stops writing at the cap rather than
            // overrunning, so an under-estimate loses splats instead of
            // corrupting memory. Losing splats is still wrong, so the buffer
            // grows and this frame is retried on the next iteration rather
            // than being rendered short.
            TrainerLog.gpu.notice(
                "Tile instances needed \(instanceCount), had \(resources.instanceCapacity); growing"
            )
            try resources.growInstanceCapacity(to: instanceCount + instanceCount / 4)
            return .grewTileBufferAndRetried
        }
        instanceCount = Swift.max(instanceCount, 0)

        // --- Command buffer B: everything else ---------------------------------------
        guard let bufferB = queue.makeCommandBuffer(),
              let encoderB = bufferB.makeComputeCommandEncoder()
        else { throw TrainerError.noMetalDevice }
        encoderB.label = "trainer.step"

        gpu.duplicateKeys(encoderB, camera: &camera, splatCount: splatCount)
        gpu.radixSort(encoderB, count: instanceCount)
        gpu.tileRanges(encoderB, instanceCount: instanceCount)
        gpu.rasterizeForward(encoderB, camera: &camera)

        gpu.lossPhotometric(encoderB, loss: &loss)
        gpu.ssim(encoderB, loss: &loss)
        gpu.lossFinalize(encoderB, loss: &loss)
        gpu.lossDepth(encoderB, loss: &loss, sampleCount: sampleCount)

        gpu.rasterizeBackward(encoderB, camera: &camera, loss: &loss)
        gpu.preprocessBackward(encoderB, camera: &camera, splatCount: splatCount)

        var reg = regularizerUniforms(
            splatCount: splatCount, iteration: iteration, totalIterations: totalIterations
        )
        gpu.regularizer(encoderB, reg: &reg)

        var adam = adamUniforms(
            splatCount: splatCount,
            shCoefficientCount: shCoefficientCount,
            iteration: iteration,
            totalIterations: totalIterations,
            sceneExtent: sceneExtent
        )
        gpu.adamSplat(encoderB, adam: &adam)
        gpu.adamSH(encoderB, adam: &adam)

        encoderB.endEncoding()
        bufferB.commit()
        bufferB.waitUntilCompleted()

        // --- Readbacks ------------------------------------------------------------------
        let lossValue = resources.lossAccum.readElement(Float.self, at: 0) ?? 0
        if lossValue.isFinite {
            lossEMA = lossEMA.map { $0 * 0.98 + lossValue * 0.02 } ?? lossValue
        }

        // Per-frame exposure (F5, F8). Learned, and kept on a very short lead:
        // exposure is a nuisance parameter, and a loose one lets the model
        // explain geometry error as brightness error.
        let exposureGradient = resources.exposureGrad.readArray(Float.self, count: 2)
        if exposureGradient.count == 2,
           exposureGradient[0].isFinite, exposureGradient[1].isFinite
        {
            var gain = exposure.x - tuning.exposureLearningRate * exposureGradient[0]
            var bias = exposure.y - tuning.exposureLearningRate * exposureGradient[1]
            gain = TrainerMath.clamp(
                gain, tuning.exposureGainRange.lowerBound, tuning.exposureGainRange.upperBound
            )
            bias = TrainerMath.clamp(
                bias, tuning.exposureBiasRange.lowerBound, tuning.exposureBiasRange.upperBound
            )
            exposures[frame.index] = SIMD2<Float>(gain, bias)
        }

        // Per-camera pose delta (F1). FROZEN during warm-up, exactly as
        // TrainerStage.warmup says: everything else is free first, and the
        // cameras only start moving once the geometry has somewhere to be.
        let warmupEnd = Int(Float(totalIterations) * tuning.warmupFraction)
        if iteration > warmupEnd {
            let gradient = resources.cameraGrad.readArray(Float.self, count: 6)
            if gradient.count == 6, gradient.allSatisfy({ $0.isFinite }) {
                cameraDeltas[frame.index] = updatedCameraDelta(
                    current: cameraDeltas[frame.index], gradient: gradient
                )
            }
        }

        // The background field is refined jointly during warm-up and frozen
        // after it, which is what stops it absorbing foreground error.
        if let background, iteration <= warmupEnd {
            accumulateBackgroundGradient(
                background: background,
                resources: resources,
                supervision: supervision
            )
            if iteration % 20 == 0 {
                background.applyAccumulatedGradient(learningRate: 0.25)
            }
        } else if let background, iteration == warmupEnd + 1 {
            background.freeze()
            TrainerLog.general.info("Background field frozen at iteration \(iteration)")
        }

        return .stepped
    }

    // MARK: - Uniform construction

    private func cameraUniforms(
        supervision: TrainerFrameSupervision,
        cameraDelta: Pose?,
        size: TrainerRenderSize,
        splatCount: Int,
        shCoefficientCount: Int,
        iteration: Int,
        totalIterations: Int
    ) -> TrainerCameraUniforms {
        var camera = TrainerCameraUniforms()

        // The learned delta is applied on the LEFT, in camera space, exactly
        // as `trainer_preprocess_backward` assumes when it accumulates its
        // se(3) gradient. The shader never composes poses; it gets one matrix.
        let base = supervision.pose.matrix
        camera.viewMatrix = cameraDelta.map { $0.matrix * base } ?? base

        let k = supervision.intrinsics
        camera.fx = k.fx
        camera.fy = k.fy
        camera.cx = k.cx
        camera.cy = k.cy
        camera.imageWidth = UInt32(size.width)
        camera.imageHeight = UInt32(size.height)
        camera.tileCountX = UInt32(size.tileCountX)
        camera.tileCountY = UInt32(size.tileCountY)
        camera.nearPlane = 0.05
        camera.farPlane = 100
        camera.splatCount = UInt32(splatCount)
        camera.shCoeffCount = UInt32(shCoefficientCount)

        // The camera centre the SH view direction is measured from has to be
        // the centre of the ACTUAL view matrix, delta included, or the colour
        // gradient pushes against a direction the forward never used.
        let effective = cameraDelta.map { delta -> Pose in
            let combined = delta.matrix * base
            return Pose(
                rotation: Quaternion(
                    simd_quatf(
                        simd_float3x3(
                            simd_make_float3(combined.columns.0),
                            simd_make_float3(combined.columns.1),
                            simd_make_float3(combined.columns.2)
                        )
                    ).normalized
                ),
                translation: Vector3(simd_make_float3(combined.columns.3))
            )
        } ?? supervision.pose
        let centre = effective.center.simd
        camera.cameraCenterX = centre.x
        camera.cameraCenterY = centre.y
        camera.cameraCenterZ = centre.z

        camera.filter2DVariance = 0.25
        camera.minAlpha = 1.0 / 255.0
        camera.renderDepth = 1

        // Coarse to fine: DC only at first, then the view-dependent terms.
        let fraction = Float(iteration) / Float(Swift.max(totalIterations, 1))
        let shGate = TrainerMath.clamp(fraction / Swift.max(tuning.shFullyEnabledFraction, 1e-3), 0, 1)
        let activeCoefficients = 1 + Int(Float(shCoefficientCount - 1) * shGate)
        camera.activeSHCoeffCount = UInt32(Swift.max(Swift.min(activeCoefficients, shCoefficientCount), 1))

        // And low frequencies first, in screen space. This decays to exactly
        // zero, not to a small number, so late training is not permanently
        // blurred by a leftover epsilon.
        if fraction < tuning.frequencyBlurEndFraction, tuning.frequencyBlurEndFraction > 0 {
            let t = fraction / tuning.frequencyBlurEndFraction
            camera.frequencyBlurVariance = tuning.frequencyBlurStartVariance * (1 - t)
        } else {
            camera.frequencyBlurVariance = 0
        }

        return camera
    }

    private func regularizerUniforms(
        splatCount: Int,
        iteration: Int,
        totalIterations: Int
    ) -> TrainerRegUniforms {
        var reg = TrainerRegUniforms()
        reg.count = UInt32(splatCount)
        reg.discWeight = settings.discPriorWeight
        reg.discTargetRank = settings.discTargetEffectiveRank
        reg.edgeTargetRank = settings.edgeTargetEffectiveRank

        // Late opacity binarization (F4), over the last stretch only, and
        // switched off per Gaussian by the kernel wherever what it saw was
        // mostly UNKNOWN. Forcing a decision on glass is inventing an answer.
        let fraction = Float(iteration) / Float(Swift.max(totalIterations, 1))
        let binarizeStart = 1 - tuning.binarizeLastFraction
        if fraction >= binarizeStart, tuning.binarizeLastFraction > 0 {
            let ramp = (fraction - binarizeStart) / Swift.max(tuning.binarizeLastFraction, 1e-3)
            reg.binarizeWeight = tuning.binarizeWeight * TrainerMath.clamp(ramp, 0, 1)
        } else {
            reg.binarizeWeight = 0
        }
        reg.binarizeUnknownCutoff = 0.5
        reg.maxScaleMeters = 0.5
        reg.maxScaleWeight = 0.05
        return reg
    }

    private func adamUniforms(
        splatCount: Int,
        shCoefficientCount: Int,
        iteration: Int,
        totalIterations: Int,
        sceneExtent: Float
    ) -> TrainerAdamUniforms {
        var adam = TrainerAdamUniforms()
        adam.count = UInt32(splatCount)
        adam.beta1 = 0.9
        adam.beta2 = 0.999
        adam.epsilon = 1e-15

        let t = Float(iteration) / Float(Swift.max(totalIterations, 1))
        adam.lrMean = TrainerMath.expLerp(
            tuning.positionLRInitialScaled * sceneExtent,
            tuning.positionLRFinalScaled * sceneExtent,
            t: t
        )
        adam.lrScale = tuning.scaleLR
        adam.lrRotation = tuning.rotationLR
        adam.lrOpacity = tuning.opacityLR
        adam.lrSHDC = tuning.shDCLR
        adam.lrSHRest = tuning.shDCLR / Swift.max(tuning.shRestLRDivisor, 1)
        adam.sparse = 1                 // visibility-masked sparse Adam
        adam.shCoeffCount = UInt32(shCoefficientCount)
        adam.pinnedPositionLRScale = 0.1

        // Scale bounds, in log space. The lower bound is ten micrometres,
        // which is far below anything a phone can resolve and exists only to
        // stop a degenerate Gaussian reaching zero and producing a singular
        // covariance. The upper is a two-metre Gaussian, which is already
        // absurd for a room.
        adam.minLogScale = -11.5
        adam.maxLogScale = Swift.min(0.7, logf(Swift.max(sceneExtent * 0.5, 0.05)))
        adam.maxOpacityLogit = 12
        return adam
    }

    // MARK: - Budget changes applied to live buffers

    private func applyBudgetChange(
        _ change: TrainerBudgetChange,
        resources: TrainerResources,
        gpu: inout TrainerGPU,
        splatCount: inout Int,
        renderSize: inout TrainerRenderSize,
        supervision: TrainerSupervisionBuilder,
        bundle: CaptureBundle,
        governor: TrainerBudgetGovernor
    ) throws {
        guard let pipelines else { return }
        switch change.kind {
        case .splatCap(_, let to):
            if splatCount > to {
                // Keep the most useful, drop the rest. This is the same
                // importance ranking the densifier uses, so a Gaussian that
                // survives a heat cut is the one that would have survived a
                // budget cut, and the two never disagree.
                splatCount = try trimSplats(resources: resources, from: splatCount, to: to)
            }
            try resources.resizeSplatCapacity(to: to, keeping: splatCount)
            gpu = TrainerGPU(pipelines: pipelines, resources: resources)

        case .resolution(_, let to):
            // The supervision builder decodes the photos and fixes the pixel
            // grid every frame is measured against, so it has to be told too.
            // Resizing only the buffers would last exactly one iteration: the
            // next frame would arrive at the old size and the training loop
            // would grow them straight back, reallocating both ways each time
            // and leaving the phone doing the very work the governor just cut.
            supervision.lowerLongEdge(to: to)
            let size = TrainerBudgetGovernor.renderSize(
                forLongEdge: to, intrinsics: bundle.intrinsics
            )
            try resources.resizeRenderSize(to: size)
            renderSize = size
            gpu = TrainerGPU(pipelines: pipelines, resources: resources)

        case .iterations:
            // Nothing to reallocate; the loop reads `governor.current.iterations`
            // on every pass and simply stops sooner.
            _ = governor
        }
    }

    /// Drops the least useful Gaussians down to a new cap, in place.
    private func trimSplats(
        resources: TrainerResources,
        from count: Int,
        to target: Int
    ) throws -> Int {
        guard target > 0, count > target else { return count }
        let shPerSplat = resources.shFloatsPerSplat
        let splats = resources.splats.readArray(TrainerSplat.self, count: count)
        let stats = resources.stats.readArray(TrainerSplatStats.self, count: count)
        let sh = resources.sh.readArray(Float.self, count: count * shPerSplat)

        var ranked: [(index: Int, importance: Float)] = []
        ranked.reserveCapacity(count)
        for i in 0..<count {
            let opacity = TrainerMath.sigmoid(splats[i].opacityLogit)
            let visibility = i < stats.count ? Swift.max(stats[i].visAccum, 0) : 0
            ranked.append((i, opacity * (1 + visibility)))
        }
        ranked.sort { $0.importance > $1.importance }
        let keep = ranked.prefix(target).map(\.index).sorted()

        var outSplats: [TrainerSplat] = []
        var outStats: [TrainerSplatStats] = []
        var outSH: [Float] = []
        outSplats.reserveCapacity(target)
        for i in keep {
            outSplats.append(splats[i])
            outStats.append(i < stats.count ? stats[i] : TrainerSplatStats())
            let base = i * shPerSplat
            if base + shPerSplat <= sh.count {
                outSH.append(contentsOf: sh[base..<(base + shPerSplat)])
            } else {
                outSH.append(contentsOf: [Float](repeating: 0, count: shPerSplat))
            }
        }
        resources.splats.writeArray(outSplats)
        resources.stats.writeArray(outStats)
        resources.sh.writeArray(outSH)
        // The optimiser state referred to Gaussians at other indices, so it is
        // cleared rather than carried across a reindex.
        resources.adamM.zeroAll()
        resources.adamV.zeroAll()
        resources.shAdamM.zeroAll()
        resources.shAdamV.zeroAll()
        return outSplats.count
    }

    private func resetDensifyStats(
        gpu: TrainerGPU,
        resources: TrainerResources,
        queue: MTLCommandQueue,
        splatCount: Int
    ) throws {
        guard let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder()
        else { throw TrainerError.noMetalDevice }
        encoder.label = "trainer.resetDensifyStats"
        gpu.resetDensifyStats(encoder, count: splatCount)
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
    }

    // MARK: - Mip-Splatting 3D filter sweep

    /// Re-measures every Gaussian's observed sampling rate across the whole
    /// keyframe set and re-sizes its 3D low-pass filter from a HIGH PERCENTILE
    /// of those rates rather than from the maximum, which is what stops one
    /// accidental close-up shrinking the filter for a Gaussian the rest of the
    /// capture only saw from three metres away.
    private func updateFilter3D(
        gpu: TrainerGPU,
        resources: TrainerResources,
        queue: MTLCommandQueue,
        keyframes: [CaptureFrame],
        supervision: TrainerSupervisionBuilder,
        cameraDeltas: [FrameID: Pose],
        splatCount: Int,
        shCoefficientCount: Int,
        renderSize: TrainerRenderSize
    ) throws {
        guard splatCount > 0, !keyframes.isEmpty else { return }
        guard let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder()
        else { throw TrainerError.noMetalDevice }
        encoder.label = "trainer.filter3d"

        // The top-K list is rebuilt from scratch: it is a running maximum, and
        // a Gaussian that moved since the last sweep would otherwise keep a
        // rate it earned somewhere else.
        gpu.fillFloat(
            encoder,
            buffer: resources.samplingTopK,
            count: splatCount * 4,
            value: 0
        )

        let intrinsics = supervision.renderIntrinsics
            ?? CameraIntrinsics(
                width: renderSize.width, height: renderSize.height,
                fx: Float(renderSize.width), fy: Float(renderSize.width),
                cx: Float(renderSize.width) / 2, cy: Float(renderSize.height) / 2
            )

        // Every keyframe would be exact and slow; a stride of at most 64
        // cameras is a high percentile of the same distribution for a fraction
        // of the cost.
        let stride = Swift.max(keyframes.count / 64, 1)
        var index = 0
        while index < keyframes.count {
            let frame = keyframes[index]
            let base = supervision.pose(for: frame).matrix
            var camera = TrainerCameraUniforms()
            camera.viewMatrix = cameraDeltas[frame.index].map { $0.matrix * base } ?? base
            camera.fx = intrinsics.fx
            camera.fy = intrinsics.fy
            camera.cx = intrinsics.cx
            camera.cy = intrinsics.cy
            camera.imageWidth = UInt32(renderSize.width)
            camera.imageHeight = UInt32(renderSize.height)
            camera.tileCountX = UInt32(renderSize.tileCountX)
            camera.tileCountY = UInt32(renderSize.tileCountY)
            camera.nearPlane = 0.05
            camera.farPlane = 100
            camera.splatCount = UInt32(splatCount)
            camera.shCoeffCount = UInt32(shCoefficientCount)
            gpu.samplingRateUpdate(encoder, camera: &camera, splatCount: splatCount)
            index += stride
        }

        gpu.filter3DFinalize(
            encoder,
            splatCount: splatCount,
            filterScale: tuning.filter3DScale,
            fallback: tuning.filter3DFallbackMeters
        )
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
    }

    // MARK: - Camera delta

    /// One damped, hard-clamped se(3) step on this camera's correction.
    ///
    /// Left perturbation, matching what `trainer_preprocess_backward`
    /// accumulates: `p_cam -> (I + omega^) p_cam + nu`. The step is clamped in
    /// both rotation and translation before it is composed, so a single bad
    /// frame cannot throw a camera across the room no matter how large its
    /// gradient was.
    private func updatedCameraDelta(current: Pose?, gradient: [Float]) -> Pose {
        let omega = SIMD3<Float>(gradient[0], gradient[1], gradient[2]) * -tuning.cameraRotationLR
        let nu = SIMD3<Float>(gradient[3], gradient[4], gradient[5]) * -tuning.cameraTranslationLR

        let omegaLength = simd_length(omega)
        let clampedOmega = omegaLength > tuning.cameraMaxRotationStepRadians
            ? omega * (tuning.cameraMaxRotationStepRadians / omegaLength)
            : omega
        let nuLength = simd_length(nu)
        let clampedNu = nuLength > tuning.cameraMaxTranslationStepMeters
            ? nu * (tuning.cameraMaxTranslationStepMeters / nuLength)
            : nu

        let angle = simd_length(clampedOmega)
        let step: simd_quatf = angle > 1e-9
            ? simd_quatf(angle: angle, axis: clampedOmega / angle)
            : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        let existing = current ?? Pose.identity
        let composedRotation = (step * existing.rotation.simd).normalized
        let composedTranslation = step.act(existing.translation.simd) + clampedNu

        return Pose(
            rotation: Quaternion(composedRotation),
            translation: Vector3(composedTranslation)
        )
    }

    // MARK: - Background refinement

    /// Scatters the photometric residual of the pixels the Gaussians did NOT
    /// explain into the direction-only far field.
    ///
    /// The weight carries `1 - accumulatedAlpha`, because a pixel the
    /// Gaussians already cover opaquely says nothing about what is behind
    /// them. Sampled on a stride: the field is a 32x32x6 cubemap, and every
    /// pixel of every frame would be a hundred samples per texel per iteration.
    private func accumulateBackgroundGradient(
        background: DirectionalBackgroundModel,
        resources: TrainerResources,
        supervision: TrainerFrameSupervision
    ) {
        let size = resources.renderSize
        let pixelCount = size.pixelCount
        guard pixelCount > 0 else { return }

        let rotationInverse = supervision.pose.rotation.simd.inverse
        let k = supervision.intrinsics

        // Roughly two thousand samples per iteration, not every pixel. The
        // field is a 32x32x6 cubemap: at 720p, every pixel would be several
        // hundred samples per texel per iteration, each one taking the
        // background model's lock, and the extra samples buy nothing because
        // the texel is an average either way. The stride is forced odd so the
        // sampled set is not a grid aligned to the image width.
        var stride = Swift.max(pixelCount / 2_000, 1)
        if stride % 2 == 0 { stride += 1 }

        // Read the two buffers in place. They are shared-storage, the GPU is
        // idle at this point in the iteration, and copying 4 MB of gradient
        // into a Swift array to walk it once would cost more than the walk.
        _ = resources.gradFinal.withElements(Float.self, count: pixelCount * 3) { gradFinal in
            _ = resources.renderTFinal.withElements(Float.self, count: pixelCount) { tFinal in
                var index = 0
                while index < pixelCount {
                    let transmittance = tFinal[index]
                    // Nothing to learn where the Gaussians already own the pixel.
                    if transmittance > 0.05 {
                        let x = index % size.width
                        let y = index / size.width
                        let ray = SmartCamera.ray(
                            SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5), k
                        )
                        let worldRay = rotationInverse.act(ray)
                        let gradient = SIMD3<Float>(
                            gradFinal[index * 3 + 0],
                            gradFinal[index * 3 + 1],
                            gradFinal[index * 3 + 2]
                        )
                        background.accumulateGradient(
                            direction: Vector3(worldRay),
                            dLossDRadiance: gradient * transmittance,
                            weight: transmittance
                        )
                    }
                    index += stride
                }
            }
        }
    }

    // MARK: - Held-out evaluation

    /// Renders the held-out frames forward only and reports PSNR.
    ///
    /// Reported whatever it says. A bad number here is the most useful thing
    /// this trainer can tell anybody, and hiding it would make every other
    /// quality claim in the app unfalsifiable.
    private func evaluateHeldOut(
        gpu: TrainerGPU,
        resources: TrainerResources,
        queue: MTLCommandQueue,
        frames: [CaptureFrame],
        supervision: TrainerSupervisionBuilder,
        cameraDeltas: [FrameID: Pose],
        exposures: [FrameID: SIMD2<Float>],
        splatCount: Int,
        shCoefficientCount: Int,
        renderSize: TrainerRenderSize
    ) throws -> Float? {

        guard splatCount > 0, !frames.isEmpty else { return nil }
        var totalMSE: Double = 0
        var evaluated = 0

        for frame in frames.prefix(24) {
            guard let frameSupervision = supervision.build(
                frame: frame, iteration: 0, totalIterations: 1
            ) else { continue }
            guard frameSupervision.renderSize == renderSize else { continue }

            resources.gtColor.writeArray(frameSupervision.groundTruth)
            if frameSupervision.hasBackground, !frameSupervision.background.isEmpty {
                resources.bgColor.writeArray(frameSupervision.background)
            }

            var camera = cameraUniforms(
                supervision: frameSupervision,
                cameraDelta: cameraDeltas[frame.index],
                size: renderSize,
                splatCount: splatCount,
                shCoefficientCount: shCoefficientCount,
                iteration: 1,
                totalIterations: 1
            )

            guard let bufferA = queue.makeCommandBuffer(),
                  let encoderA = bufferA.makeComputeCommandEncoder()
            else { return nil }
            encoderA.label = "trainer.eval.preprocess"
            gpu.fillUInt(encoderA, buffer: resources.tilesTouched, count: splatCount, value: 0)
            gpu.preprocess(encoderA, camera: &camera, splatCount: splatCount)
            gpu.exclusiveScan(
                encoderA,
                input: resources.tilesTouched,
                output: resources.offsets,
                count: splatCount
            )
            encoderA.endEncoding()
            bufferA.commit()
            bufferA.waitUntilCompleted()

            let lastOffset = resources.offsets.readElement(UInt32.self, at: splatCount - 1) ?? 0
            let lastTouched = resources.tilesTouched.readElement(UInt32.self, at: splatCount - 1) ?? 0
            let instanceCount = Int(lastOffset) + Int(lastTouched)
            guard instanceCount > 0, instanceCount <= resources.instanceCapacity else { continue }

            guard let bufferB = queue.makeCommandBuffer(),
                  let encoderB = bufferB.makeComputeCommandEncoder()
            else { return nil }
            encoderB.label = "trainer.eval.render"
            gpu.duplicateKeys(encoderB, camera: &camera, splatCount: splatCount)
            gpu.radixSort(encoderB, count: instanceCount)
            gpu.tileRanges(encoderB, instanceCount: instanceCount)
            gpu.rasterizeForward(encoderB, camera: &camera)
            encoderB.endEncoding()
            bufferB.commit()
            bufferB.waitUntilCompleted()

            // Composite and exposure are applied here rather than by a kernel,
            // because the evaluation must not touch the gradient buffers.
            let pixelCount = renderSize.pixelCount
            let rendered = resources.renderColor.readArray(Float.self, count: pixelCount * 3)
            let transmittance = resources.renderTFinal.readArray(Float.self, count: pixelCount)
            guard rendered.count == pixelCount * 3, transmittance.count == pixelCount else { continue }

            let exposure = exposures[frame.index] ?? SIMD2<Float>(1, 0)
            var sum: Double = 0
            for i in 0..<pixelCount {
                for c in 0..<3 {
                    var value = rendered[i * 3 + c]
                    if frameSupervision.hasBackground, !frameSupervision.background.isEmpty {
                        value += transmittance[i] * frameSupervision.background[i * 3 + c]
                    }
                    value = exposure.x * value + exposure.y
                    let truth = frameSupervision.groundTruth[i * 3 + c]
                    let diff = Double(value - truth)
                    sum += diff * diff
                }
            }
            totalMSE += sum / Double(pixelCount * 3)
            evaluated += 1
        }

        guard evaluated > 0 else { return nil }
        let mse = totalMSE / Double(evaluated)
        guard mse > 1e-12 else { return 99 }
        return Float(10 * log10(1.0 / mse))
    }

    // MARK: - Reading the field back

    private func readCloud(
        resources: TrainerResources,
        count: Int,
        shDegree: SHDegree
    ) -> SplatCloud {
        guard count > 0 else { return SplatCloud.empty(shDegree: shDegree) }
        let splats = resources.splats.readArray(TrainerSplat.self, count: count)
        let shPerSplat = resources.shFloatsPerSplat
        let sh = resources.sh.readArray(Float.self, count: count * shPerSplat)
        let restCount = shDegree.restCoefficientCount

        var positions: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var logScales: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var colorDC: [SIMD3<Float>] = []
        var shRest: [[SIMD3<Float>]] = []
        positions.reserveCapacity(splats.count)

        for (i, splat) in splats.enumerated() {
            let mean = splat.mean
            let logScale = splat.logScale
            // A non-finite Gaussian is dropped rather than exported: it would
            // become a NaN in a .ply that every downstream viewer then has to
            // survive.
            guard mean.x.isFinite, mean.y.isFinite, mean.z.isFinite,
                  logScale.x.isFinite, logScale.y.isFinite, logScale.z.isFinite,
                  splat.opacityLogit.isFinite
            else { continue }

            positions.append(mean)
            rotations.append(splat.rotation)
            logScales.append(logScale)
            opacities.append(splat.opacityLogit)

            let base = i * shPerSplat
            if base + 3 <= sh.count {
                colorDC.append(SIMD3<Float>(sh[base], sh[base + 1], sh[base + 2]))
            } else {
                colorDC.append(.zero)
            }

            if restCount > 0 {
                var rest: [SIMD3<Float>] = []
                rest.reserveCapacity(restCount)
                for c in 0..<restCount {
                    let offset = base + 3 * (c + 1)
                    if offset + 2 < sh.count {
                        rest.append(SIMD3<Float>(sh[offset], sh[offset + 1], sh[offset + 2]))
                    } else {
                        rest.append(.zero)
                    }
                }
                shRest.append(rest)
            }
        }

        do {
            return try SplatCloud(
                shDegree: shDegree,
                positions: positions,
                rotations: rotations,
                logScales: logScales,
                opacityLogits: opacities,
                colorDC: colorDC,
                shRest: restCount > 0 ? shRest : []
            )
        } catch {
            TrainerLog.general.error(
                "Could not assemble a preview cloud: \(error.localizedDescription, privacy: .public)"
            )
            return SplatCloud.empty(shDegree: shDegree)
        }
    }

    /// The preview during a multi-slice run shows the parts already finished
    /// plus the one being worked on, so the user watches their house fill in
    /// rather than watching one room appear and vanish.
    private func mergePreview(completedParts: [SplatCloud], current: SplatCloud) -> SplatCloud {
        guard !completedParts.isEmpty else { return current }
        var positions = current.positions
        var rotations = current.rotations
        var logScales = current.logScales
        var opacities = current.opacityLogits
        var colorDC = current.colorDC
        var shRest = current.shRest

        for part in completedParts where part.shDegree == current.shDegree {
            positions.append(contentsOf: part.positions)
            rotations.append(contentsOf: part.rotations)
            logScales.append(contentsOf: part.logScales)
            opacities.append(contentsOf: part.opacityLogits)
            colorDC.append(contentsOf: part.colorDC)
            if current.shDegree != .zero { shRest.append(contentsOf: part.shRest) }
        }

        return (try? SplatCloud(
            shDegree: current.shDegree,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacities,
            colorDC: colorDC,
            shRest: current.shDegree == .zero ? [] : shRest
        )) ?? current
    }

    // MARK: - Writing the result

    private func writeModel(
        cloud: SplatCloud,
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        governor: TrainerBudgetGovernor,
        iterationsCompleted: Int,
        heldOutPSNR: Float?,
        background: DirectionalBackgroundModel?
    ) async throws -> SplatModel {

        // `PLYCodec.write` refuses an empty cloud, and rightly so. Reaching
        // here with nothing means every Gaussian was pruned or carved away,
        // which is a real outcome worth naming plainly rather than surfacing
        // as an export error the user cannot act on.
        guard cloud.count > 0 else {
            throw TrainerError.nothingToTrain(
                "every starting point was ruled out by the scan's own depth measurements, "
                    + "so there was nothing left to build"
            )
        }

        let folder = BrandConfig.Folder.model
        let plyRelative = "\(folder)/model.ply"
        let plyURL = ref.url(forRelativePath: plyRelative)
        try FileManager.default.createDirectory(
            at: plyURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try PLYCodec.write(cloud, to: plyURL)

        var backgroundPath: String?
        if let background {
            backgroundPath = try? await background.write(to: ref)
        }

        // Per-frame exposure, in the exact record layout docs/DATA_FORMAT.md
        // section 8 fixes: (UInt32 frameIndex, Float32 gain, Float32 bias).
        var exposurePath: String?
        if !exposureRecords.isEmpty {
            var data = Data()
            data.reserveCapacity(exposureRecords.count * 12)
            for (frameIndex, value) in exposureRecords.sorted(by: { $0.key < $1.key }) {
                SmartBinary.append(UInt32(frameIndex), to: &data)
                SmartBinary.append(value.x, to: &data)
                SmartBinary.append(value.y, to: &data)
            }
            let relative = "\(folder)/exposure.bin"
            try? SmartBinary.write(data, to: ref.url(forRelativePath: relative))
            exposurePath = relative
        }

        var recordedBudget = governor.current
        // Honest record: the GPU structs in TrainerGPULayouts.swift are fp32
        // by design, so this run did not use half precision whatever was asked.
        recordedBudget.useHalfPrecision = false
        recordedBudget.target = .onDevice

        let model = SplatModel(
            modelID: UUID().uuidString,
            scanID: bundle.scanID,
            createdAt: Date(),
            source: .onDevice,
            plyPath: plyRelative,
            spzPath: nil,
            splatCount: cloud.count,
            shDegree: cloud.shDegree,
            bounds: bounds(of: cloud),
            iterationsCompleted: iterationsCompleted,
            budgetUsed: recordedBudget,
            backgroundModelPath: backgroundPath,
            exposurePath: exposurePath,
            observedDirectionsPath: nil,
            heldOutPSNR: heldOutPSNR
        )

        // Which photos this run kept out of training, for the review screen's
        // photo-versus-scan slider. A bare array of frame indices, which is one
        // of the two shapes `HeldOutFrameSelector.readSidecar` accepts. Written
        // only when there actually were held-out frames: an empty file would
        // read as "the trainer recorded that it held nothing out", which is a
        // different and less honest claim than "nothing was written down".
        if !heldOutFrameIndices.isEmpty {
            let indices = heldOutFrameIndices.sorted()
            if let json = try? ContractsJSON.encoder().encode(indices) {
                try? SmartBinary.write(
                    json,
                    to: ref.url(forRelativePath: "\(folder)/held_out_frames.json")
                )
            }
        }

        let json = try ContractsJSON.encoder().encode(model)
        try SmartBinary.write(json, to: ref.url(forRelativePath: "\(folder)/model.json"))

        return model
    }

    /// Collected across slices so `writeModel` can serialise them in one pass.
    private var exposureRecords: [FrameID: SIMD2<Float>] = [:]

    /// Every frame this run kept out of training, across all slices. Written
    /// as `model/held_out_frames.json` so the review screen can prove which
    /// photos the model never saw instead of estimating.
    private var heldOutFrameIndices: Set<FrameID> = []

    private func bounds(of cloud: SplatCloud) -> BoundingBox {
        guard !cloud.positions.isEmpty else {
            return BoundingBox(min: .zero, max: .zero)
        }
        var lo = cloud.positions[0]
        var hi = cloud.positions[0]
        for position in cloud.positions {
            lo = simd_min(lo, position)
            hi = simd_max(hi, position)
        }
        return BoundingBox(min: Vector3(lo), max: Vector3(hi))
    }

    // MARK: - The SMART layer

    /// Every SMART component the trainer will use, loaded once. Each is
    /// optional and each failure is logged with what was lost, never silently
    /// swallowed into a neutral default.
    private struct SmartLayer {
        var trust: TwoScaleTrustField?
        var authority: SmartAuthorityMap?
        var edges: NativeDepthEdgeClassifier?
        var background: DirectionalBackgroundModel?
        var carver: FreeSpaceCarver?
    }

    private func loadSmartLayer(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef
    ) async throws -> SmartLayer {

        var layer = SmartLayer()

        if let refs = prePass.trust {
            let field = TwoScaleTrustField(settings: settings)
            field.prepare(bundle: bundle)
            do {
                try await field.load(refs, at: ref)
                layer.trust = field
            } catch {
                let why = error.localizedDescription
                TrainerLog.general.error(
                    "The trust fields could not be read (\(why, privacy: .public)); depth is supervised at a flat weight instead"
                )
            }
        }

        if let refs = prePass.edges {
            let classifier = NativeDepthEdgeClassifier(settings: settings)
            classifier.load(refs, bundle: bundle, at: ref)
            if classifier.isLoaded {
                layer.edges = classifier
            } else {
                TrainerLog.general.error(
                    "No edge maps were found; depth edges are not sharpened this run"
                )
            }
        }

        let authorityMap = SmartAuthorityMap(settings: settings)
        authorityMap.prepare(
            bundle: bundle,
            at: ref,
            prePassPoses: prePass.refinedPoses,
            glassRegions: prePass.glassRegions,
            trust: layer.trust
        )
        if authorityMap.isPrepared { layer.authority = authorityMap }

        let background = DirectionalBackgroundModel(settings: settings)
        background.prepare(authorityMap)
        do {
            try await background.warmUp(bundle: bundle, at: ref, iterations: 120)
            layer.background = background
        } catch {
            let why = error.localizedDescription
            TrainerLog.general.error(
                "The far field could not be fitted (\(why, privacy: .public)); distant surfaces are left to the Gaussians"
            )
        }

        if let grid = prePass.occupancy {
            let carver = VoxelFreeSpaceCarver()
            do {
                try await carver.load(grid, at: ref)
                layer.carver = carver
            } catch {
                let why = error.localizedDescription
                TrainerLog.general.error(
                    "The free-space map could not be read (\(why, privacy: .public)); nothing is deleted on free-space grounds this run"
                )
            }
        }

        return layer
    }

    // MARK: - Keyframe selection

    /// Which frames are actually trained on.
    ///
    /// A 4000-frame house walk does not need 4000 supervision views: adjacent
    /// frames are nearly the same picture, and the budget says how many the
    /// device can afford. Selection is greedy on movement, which keeps the
    /// views spread over the walk rather than clustered wherever the user
    /// stood still longest.
    private func selectKeyframes(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        budget: TrainingBudget
    ) -> [CaptureFrame] {
        let frames = bundle.frames.sorted { $0.index < $1.index }
        guard !frames.isEmpty else { return [] }

        // A frame with no usable pixels is not a supervision view. This is a
        // quality floor, not a discard: the frame stays in the capture and in
        // every other stage.
        let usable = frames.filter { $0.qc.weight > 0.05 }
        let pool = usable.isEmpty ? frames : usable

        let target = Swift.max(budget.keyframeCount, 8)
        if pool.count <= target { return pool }

        // Greedy spacing on the refined poses.
        var chosen: [CaptureFrame] = []
        var lastCenter: SIMD3<Float>?
        var lastForward: SIMD3<Float>?

        // Spacing that would land on roughly the target count if the walk were
        // uniform. Measured from the actual path length, not assumed.
        var pathLength: Float = 0
        var previous: SIMD3<Float>?
        for frame in pool {
            let pose = prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
            let c = pose.center.simd
            if let previous { pathLength += simd_distance(previous, c) }
            previous = c
        }
        let spacing = pathLength > 0 ? pathLength / Float(target) : 0

        for frame in pool {
            let pose = prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
            let center = pose.center.simd
            let forward = pose.forward.simd
            if let lastCenter, let lastForward {
                let moved = simd_distance(lastCenter, center)
                let turned = 1 - simd_dot(simd_normalize(lastForward), simd_normalize(forward))
                // Either enough movement or enough rotation. A user who turns
                // on the spot in a doorway is producing genuinely new views.
                if moved < spacing && turned < 0.02 { continue }
            }
            chosen.append(frame)
            lastCenter = center
            lastForward = forward
            if chosen.count >= target { break }
        }

        if chosen.count < Swift.min(target, pool.count) {
            // The greedy pass was too strict for this walk (a scan taken from
            // one spot, for instance). Fall back to an even stride, which is
            // still a spread rather than a prefix.
            let step = Swift.max(pool.count / target, 1)
            chosen = Swift.stride(from: 0, to: pool.count, by: step).map { pool[$0] }
        }
        return chosen
    }

    // MARK: - Progress

    private func stage(for iteration: Int, of total: Int) -> TrainerStage {
        let fraction = Float(iteration) / Float(Swift.max(total, 1))
        if fraction < tuning.warmupFraction { return .warmup }
        if fraction <= tuning.densifyEndFraction { return .densifying }
        if fraction >= 1 - tuning.binarizeLastFraction { return .binarizing }
        return .refining
    }

    private func stageMessage(for iteration: Int, of total: Int, sliceLabel: String) -> String {
        let suffix = sliceLabel.isEmpty ? "" : " (\(sliceLabel))"
        switch stage(for: iteration, of: total) {
        case .warmup: return "Laying out the shape of the room\(suffix)."
        case .densifying: return "Adding detail where it is missing\(suffix)."
        case .refining: return "Sharpening edges\(suffix)."
        case .binarizing: return "Deciding what is solid and what is not\(suffix)."
        default: return "Building your model\(suffix)."
        }
    }

    private func doneMessage(
        cloud: SplatCloud,
        psnr: Float?,
        governor: TrainerBudgetGovernor
    ) -> String {
        var sentence = "Your scan is ready, built from "
            + "\(TrainerBudgetGovernor.round(cloud.count)) points of detail."
        if !governor.changes.isEmpty {
            sentence += " It was made a little smaller along the way because "
                + (governor.changes.last?.reason.plainCause ?? "the phone needed the room") + "."
        }
        if let psnr {
            sentence += " On the photos it was not trained with, it scores "
                + String(format: "%.1f", psnr) + " decibels."
        }
        return sentence
    }

    private func progressTick(
        stage: TrainerStage,
        iteration: Int,
        total: Int,
        splatCount: Int,
        loss: Float?,
        thermal: ThermalLevel,
        message: String,
        previewAvailable: Bool
    ) -> TrainerProgress {
        // `fractionComplete` is nil, not zero, whenever there is no honest
        // number to give: Contracts.swift says the UI draws an indeterminate
        // spinner for nil, and a bar stuck at 0% is a lie the user can see.
        let fraction: Double?
        switch stage {
        case .preparing, .initializing, .failed, .cancelled:
            fraction = nil
        default:
            fraction = total > 0
                ? Swift.min(Swift.max(Double(iteration) / Double(total), 0), 1)
                : nil
        }

        return TrainerProgress(
            stage: stage,
            iteration: iteration,
            totalIterations: total,
            splatCount: splatCount,
            lossEMA: loss,
            fractionComplete: fraction,
            thermalLevel: thermal,
            residentBytes: resources?.residentBytes ?? 0,
            message: message,
            previewAvailable: previewAvailable
        )
    }
}
