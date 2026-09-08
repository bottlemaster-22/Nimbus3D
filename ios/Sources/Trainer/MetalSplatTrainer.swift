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

    /// Where the run's time actually went. Written by `finish` and by the
    /// supervision call in the training loop, copied into the census when
    /// the run is sealed. Single-threaded: everything that touches it runs
    /// on the training thread.
    var timings = TrainerTimings()

    // MARK: - Configuration

    private let tuning: TrainerTuning
    private let settings: SmartLossSettings

    /// How many densification passes in a row may add nothing, while growth was
    /// permitted and there was room under the cap, before the trainer says so
    /// on screen and in the census.
    ///
    /// Named once because two places read it: the live progress message inside
    /// the slice loop, and the run's final outcome string. A threshold written
    /// twice is a threshold that will disagree with itself. At the default
    /// 100-iteration densify interval this is a thousand iterations of a stage
    /// that is meant to be adding geometry adding none.
    private static let zeroGrowthPassesBeforeSaying = 10

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
    /// Largest (Gaussian, tile) pair count any iteration of the CURRENT slice
    /// produced, and the live population at that moment. Folded into the
    /// census when the slice finishes, and reset there so the next slice
    /// measures itself rather than inheriting.
    private var peakTileInstances = 0
    private var splatCountAtPeakTileInstances = 0
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
            // The clocks, copied in at the last moment so a run that ends
            // any way at all still reports where its time went.
            census.timings = timings
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
        // Times round the loop, and times a gradient step actually ran. These
        // are NOT the same number and the run is only allowed to call itself
        // completed on the strength of the second one.
        var totalGradientSteps = 0
        // The worst run of consecutive densification passes that were allowed
        // to add geometry and added none.
        var worstZeroGrowthStreak = 0
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
                gradientStepsRunSoFar: &totalGradientSteps,
                zeroGrowthStreakWorstSoFar: &worstZeroGrowthStreak,
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

        // HOW MUCH OF WHAT THE TRAINER LOOKED AT WAS WINDOW.
        //
        // `SmartAuthorityMap` counts, per frame, the ones that were at least
        // half confirmed glass. A confirmed pane multiplies depth authority by
        // zero, so those frames hand the trainer almost no geometry. Both
        // numbers are read here, at the end, when the count has finished
        // rising. Left nil when there was no authority map: a run that
        // measured nothing must not report a zero it did not measure.
        if let authority = smart.authority {
            census.authorityFramesBuilt = authority.builtFrameCount
            census.glassDominatedFrames = authority.glassDominatedFrameCount
        }

        // WAS THE MIDDLE DISTANCE MEASURED, OR ROUTED AROUND.
        //
        // The monocular depth model is a stub on the phone, so the 4.5 to 30 m
        // band falls back to parallax plus the far field. The background model
        // has always known which of the two it did and nothing carried it
        // anywhere, so no scan on disk records it. Left nil when the far field
        // could not be fitted at all, which is a third case and not a false.
        if let background = smart.background {
            census.midRegimeIsReal = background.isMidRegimeReal
            census.midRegimeProvenance = background.midRegimeProvenance
        }

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
        //
        // AND REACHING THIS LINE IS NOT ENOUGH ON ITS OWN.
        //
        // `totalIterationsRun` counts times round the loop. A run that could
        // not decode a single photo, or whose buffers were never big enough,
        // advanced that counter on every one of those iterations and arrived
        // here with a full count and an empty model, and said "completed". The
        // work actually done is `totalGradientSteps`, and the outcome is now
        // judged on it.
        //
        // The same 95 per cent tolerance the census's own alert uses: slice
        // iteration budgets are integer shares of the whole and round down, so
        // a full run legitimately lands a few iterations short.
        let ranItsBudget = totalIterationsRun * 100 >= governor.current.iterations * 95
        // Half. Below this the run trained on less than it skipped, and no
        // amount of wall clock makes that a finished scan.
        let didRealWork = totalGradientSteps * 2 >= totalIterationsRun
        let skippedIterations = Swift.max(totalIterationsRun - totalGradientSteps, 0)

        if !didRealWork {
            census.outcome = "ran \(totalIterationsRun) iterations but only "
                + "\(totalGradientSteps) of them took a real training step"
        } else if !ranItsBudget {
            census.outcome = "stopped early"
        } else if worstZeroGrowthStreak >= Self.zeroGrowthPassesBeforeSaying {
            census.outcome = "completed, but densification added nothing for "
                + "\(worstZeroGrowthStreak) consecutive passes that were allowed to add something"
        } else {
            census.outcome = "completed"
        }

        TrainerLog.general.info(
            """
            Run finished: \(totalIterationsRun) iterations, \(totalGradientSteps) real gradient \
            steps, \(skippedIterations) that did nothing. Worst run of densification passes \
            that were allowed to add geometry and added none: \(worstZeroGrowthStreak).
            """
        )

        emit(
            progressTick(
                stage: .done,
                iteration: totalIterationsRun,
                total: governor.current.iterations,
                splatCount: merged.cloud.count,
                loss: nil,
                thermal: governor.thermalLevel,
                message: doneMessage(
                    cloud: merged.cloud,
                    psnr: lastPSNR,
                    governor: governor,
                    iterationsRun: totalIterationsRun,
                    gradientSteps: totalGradientSteps,
                    worstZeroGrowthStreak: worstZeroGrowthStreak
                ),
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
        /// Times round the loop is not work done. This is the run-level total of
        /// iterations that actually ran a forward, a backward and an Adam step,
        /// and it is what decides whether the run may call itself completed.
        gradientStepsRunSoFar: inout Int,
        /// The worst run of consecutive densification passes, anywhere in the
        /// run, that were allowed to add geometry and added none.
        zeroGrowthStreakWorstSoFar: inout Int,
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
        // Builds the NEXT frame while the GPU works on this one. See
        // TrainerSupervisionPrefetch: supervision and gpuWait measured 24.70
        // and 24.69 ms per iteration on the owner's phone, one after the
        // other, with the other device idle each time.
        let prefetch = TrainerSupervisionPrefetch(builder: supervision)
        defer {
            prefetch.drain()
            // Accumulated across slices: what the worker built off the
            // critical path, which `timings.supervision` no longer sees.
            timings.supervisionPrefetched += prefetch.workerSeconds
        }


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
        // Clamped before the conversion, not trusted to be in range. A
        // Float-to-Int conversion outside Int's range TRAPS, in release as
        // well as in debug, and this value arrives from another file. It is
        // in range today; making it unreachable by construction costs one
        // clamp and removes a crash on the line that logs how the run started.
        //
        // nil, not 0, when it could not be measured. `TrainerSeedResult` uses
        // 0 for "no nearest neighbour was found", and 0 mm stored in a census
        // reads as "the starting points were on top of each other", which is
        // the opposite of what happened.
        let spacingMetres = seedResult.medianSpacingMeters
        if spacingMetres.isFinite, spacingMetres > 0 {
            let millimetres = TrainerMath.clamp(spacingMetres * 1000, 0, 1_000_000)
            census.slices[censusRow].seedMedianSpacingMillimetres = Int(millimetres.rounded())
        } else {
            census.slices[censusRow].seedMedianSpacingMillimetres = nil
        }
        // The trust line this slice drew, and the distribution it drew it on.
        // Copied across rather than re-derived: `TrainerInitializer` is the
        // only place that knows whether a gate was consulted at all, and
        // `wasMeasured == false` is what stops the census reporting "the gate
        // rejected everything" about a scan that had no gate.
        if let cut = seedResult.trustCut {
            census.slices[censusRow].seedTrustWasMeasured = cut.wasMeasured
            census.slices[censusRow].seedTrustCut = cut.cut
            census.slices[censusRow].seedTrustFloor = cut.floor
            census.slices[censusRow].seedTrustQuantile = cut.quantile
            census.slices[censusRow].seedTrustP05 = cut.p05
            census.slices[censusRow].seedTrustMedian = cut.median
            census.slices[censusRow].seedTrustP95 = cut.p95
            census.slices[censusRow].seedTrustCellsConsidered = cut.cellsConsidered
        }
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
        // TIMES ROUND THE LOOP IS NOT WORK DONE, AND THE TWO WERE THE SAME
        // NUMBER.
        //
        // `iteration` counts revolutions. Three of those revolutions do no
        // optimisation at all: a keyframe whose photo will not decode, a frame
        // with no pixels or no Gaussians, and a frame abandoned to grow the
        // tile buffer. All three still advance `iteration`, so a run that
        // decoded not one photo burned its whole budget, wrote
        // `iterationsCompleted == iterationsRequested`, and left through the
        // normal success path saying "completed".
        //
        // This is the number that says otherwise: incremented ONLY on
        // `.stepped`, which is the only return that has run a forward pass, a
        // backward pass and an Adam step. Everything below that judges whether
        // the run did any work judges it on this, not on `iteration`.
        var gradientSteps = 0
        // Densification passes that added nothing, in a row, and the worst such
        // run in this slice. `TrainerDensifyOutcome.summary` is nil when a pass
        // did nothing at all, and the log line was inside `if let summary`, so
        // a densification stage that had stopped producing anything was
        // COMPLETELY SILENT. That is the exact shape of the fault that made the
        // owner's first scan look like nothing.
        var zeroGrowthStreak = 0
        var longestZeroGrowthStreak = 0
        var densifyPassesRun = 0
        var densifyPassesThatAddedNothing = 0
        let zeroGrowthStreakToReport = Self.zeroGrowthPassesBeforeSaying
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
                // Literal FIRST in both clamps. `Swift.max(x, 1)` is
                // `1 >= x ? 1 : x`, so a NaN interval came straight back out
                // and `UInt64(NaN * 1e9)` is a trapping conversion. Same
                // result for every sane interval; the upper bound keeps a
                // corrupt policy from parking a paused run for a century.
                let pollSeconds = Swift.min(60, Swift.max(1, governor.current.thermalPolicy.sampleIntervalSeconds))
                try await Task.sleep(nanoseconds: UInt64(pollSeconds * 1_000_000_000))
                continue

            case .degrade:
                consecutivePauses = 0
                if iteration > 0, iteration % 100 == 0,
                   let change = governor.degradeForHeat(
                       level: thermal.level, currentSplatCount: splatCount
                   )
                {
                    // The governor can swap the builder's image cache out
                    // from under a worker. Nothing may be in flight.
                    prefetch.drain()
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
                    // The governor can swap the builder's image cache out
                    // from under a worker. Nothing may be in flight.
                    prefetch.drain()
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

            let supervisionFrom = CFAbsoluteTimeGetCurrent()
            let builtSupervision: TrainerFrameSupervision?
            switch prefetch.take(
                frame: frame, iteration: iteration, totalIterations: effectiveTotal
            ) {
            case .hit(let ready):
                // Built during the previous iteration's GPU wait. This is the
                // whole point, and on this branch the loop pays nothing for it.
                builtSupervision = ready
            case .miss:
                builtSupervision = supervision.build(
                    frame: frame,
                    iteration: iteration,
                    // The run that will actually happen, for the same reason
                    // `progressFraction` uses it: this drives the depth-loss decay
                    // and the SH degree schedule, and keying those to a length the
                    // run will never reach means their tails never execute.
                    totalIterations: effectiveTotal
                )
            }
            // Timed whether or not it succeeded: a frame that fails to decode
            // still spent the time trying, and hiding that would flatter the
            // number.
            timings.supervision += CFAbsoluteTimeGetCurrent() - supervisionFrom
            guard let frameSupervision = builtSupervision else {
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

            // START THE NEXT FRAME NOW, so it is built during the GPU wait
            // that `runIteration` is about to sit in rather than after it.
            // `orderCursor` has already moved on, so this is genuinely the
            // frame the next iteration will ask for, and `iteration + 1` is
            // the number it will ask with; both are checked on the way out, so
            // a wrong guess costs the work and nothing else.
            if !order.isEmpty {
                let nextFrame = slice.keyframes[order[orderCursor % order.count]]
                prefetch.start(
                    frame: nextFrame,
                    iteration: iteration + 1,
                    totalIterations: effectiveTotal
                )
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
                // EVERY schedule inside `runIteration` keys off this one
                // number: the spherical-harmonic degree ramp and the
                // frequency-blur decay in `cameraUniforms`, the depth-loss
                // decay, the late opacity binarization in
                // `regularizerUniforms`, the position learning-rate decay in
                // `adamUniforms`, and the warm-up end that unfreezes the
                // cameras and freezes the background. Handing it the PLANNED
                // slice length while the loop exits at `effectiveTotal` is the
                // same fault `progressFraction` had: a run cut from 3,000 to
                // 1,500 would stop at fraction 0.5 and the entire back half of
                // every one of those schedules would never execute. This is
                // the fix applied to all of them at once, because they all
                // read the same argument.
                totalIterations: effectiveTotal,
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
                // The ONLY place this is incremented, and the only return that
                // ran a gradient step.
                gradientSteps += 1
                // Recorded on `.stepped` alone, because that is the only
                // return that actually handed these samples to the loss.
                // Counting them on a skipped iteration would inflate the
                // supervised fraction with frames the laser never voted on.
                census.slices[censusRow].depthSamplesPerFrameTotal +=
                    frameSupervision.depthSamples.count
                census.slices[censusRow].depthSamplesSupervisedTotal +=
                    frameSupervision.supervisedSampleCount
                census.slices[censusRow].depthSupervisionFramesMeasured += 1
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
                let sweepFrom = CFAbsoluteTimeGetCurrent()
                defer { timings.filterSweep += CFAbsoluteTimeGetCurrent() - sweepFrom }
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
                let densifyFrom = CFAbsoluteTimeGetCurrent()
                defer { timings.densify += CFAbsoluteTimeGetCurrent() - densifyFrom }
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

                // UNCONDITIONAL, and this used to be
                // `if outcome.changedTopology || outcome.relocated > 0`.
                //
                // The kernel's own comment says it runs after every pass, and
                // the accumulators are meant to average over ONE interval. A
                // pass that changed nothing was skipping the reset, so
                // `absGrad2D`, `denom`, `visAccum` and `unknownAccum` kept
                // running into the next interval. `absGrad2D / denom` is a
                // mean and largely survives that; `visAccum` does not. It is
                // only ever compared against zero, so never resetting it makes
                // the "something actually looked at this" filter steadily more
                // permissive the longer densification goes without changing
                // anything. A stage that is producing nothing must not also be
                // quietly loosening the gate that decides whether it has
                // candidates to work with.
                try resetDensifyStats(
                    gpu: gpu, resources: resources, queue: queue, splatCount: splatCount
                )

                // --- Did this pass add anything, and how long has that been
                //     true --------------------------------------------------
                //
                // The streak is counted ONLY over passes where growth was
                // permitted AND there was room under the cap, which is the
                // same pair of conditions `TrainerCensus`'s
                // `passesWithGrowthWindowOpenAndHeadroom` uses. Adding nothing
                // outside the densify window, or with a full budget, is
                // correct behaviour; counting those would put a false alarm in
                // front of a user who cannot check it.
                densifyPassesRun += 1
                // `created` is `TrainerDensifyOutcome`'s own name for
                // `cloned + split`, and relocation is deliberately not in it:
                // moving a Gaussian is not adding one and the total does not
                // change, which is exactly the disguise a stalled
                // densification stage wears.
                let addedThisPass = outcome.created
                let couldHaveAdded = outcome.growthAllowed && outcome.headroomAtStart > 0
                if couldHaveAdded {
                    if addedThisPass > 0 {
                        zeroGrowthStreak = 0
                    } else {
                        zeroGrowthStreak += 1
                        densifyPassesThatAddedNothing += 1
                        longestZeroGrowthStreak = Swift.max(
                            longestZeroGrowthStreak, zeroGrowthStreak
                        )
                    }
                }

                // EVERY PASS LOGS. NO EXCEPTIONS.
                //
                // This was `if let summary = outcome.summary`, and `summary` is
                // nil precisely when a pass changed nothing, so the passes most
                // worth knowing about were the only ones that never printed.
                if let summary = outcome.summary {
                    TrainerLog.densify.info(
                        "Iteration \(iteration): \(summary, privacy: .public)"
                    )
                } else {
                    TrainerLog.densify.info(
                        """
                        Iteration \(iteration): densification changed nothing. Growth \
                        allowed: \(outcome.growthAllowed), room under the cap: \
                        \(outcome.headroomAtStart).
                        """
                    )
                }

                // AND THE STREAK GETS ITS OWN LINE, at error level, separate
                // from the summary above.
                //
                // Deliberately not folded into the `else` branch. A pass that
                // pruned twelve Gaussians and created none has a perfectly
                // cheerful non-nil summary ("removed 12") and would take the
                // quiet path, and a run of those is exactly a stalled
                // densification stage looking like a working one. The test is
                // on what was CREATED and on how long that has been true,
                // nothing else. Every number here is one
                // `TrainerDensifyOutcome` already carries; none of them costs
                // a pass over a buffer or touches the GPU.
                if couldHaveAdded, addedThisPass == 0,
                   zeroGrowthStreak >= zeroGrowthStreakToReport
                {
                    TrainerLog.densify.error(
                        """
                        Iteration \(iteration): densification has added NOTHING for \
                        \(zeroGrowthStreak) passes in a row while it was allowed to and had \
                        room. Verdict: \(outcome.growthVerdict.rawValue). Room for \
                        \(outcome.headroomAtStart), allowance \
                        \(outcome.growthAllowance), scored \(outcome.splatsScored), \
                        \(outcome.splatsWithNonZeroScore) above zero, \
                        \(outcome.candidatesAfterVisibilityFilter) candidates after the \
                        visibility filter.
                        """
                    )
                }

                // The SCREEN is told by the progress tick below, which runs at
                // most twice a second and rebuilds its sentence from
                // `zeroGrowthStreak` every time. That is deliberately not a
                // one-shot announcement: the warning stays up for as long as
                // the streak lasts and disappears by itself the moment a pass
                // adds something, because `zeroGrowthStreak` goes back to zero.
            }

            if iteration % Swift.max(tuning.snapshotIntervalIterations, 1) == 0 {
                let snapshotFrom = CFAbsoluteTimeGetCurrent()
                let cloud = readCloud(
                    resources: resources, count: splatCount, shDegree: shDegree
                )
                lock.lock()
                latestSnapshot = mergePreview(completedParts: completedParts, current: cloud)
                lock.unlock()
                timings.previewSnapshot += CFAbsoluteTimeGetCurrent() - snapshotFrom
            }

            // --- Progress ------------------------------------------------------------
            let now = Date()
            if now.timeIntervalSince(lastEmit) > 0.5 || iteration == effectiveTotal - 1 {
                lastEmit = now
                // `effectiveTotal`, not `totalIterations`. The stage label and
                // the sentence under it are the user's only view of where the
                // run is, and keying them to the planned length on a run that
                // was shortened means the screen sits on "Adding detail where
                // it is missing" and never reaches "Deciding what is solid",
                // while the loop underneath has already binarized and stopped.
                // The stage shown and the stage running have to be the same
                // stage.
                //
                // The sentence itself is normally the stage label; but once
                // densification has gone `zeroGrowthStreakToReport` passes in a
                // row without adding a single point, that is the more important
                // thing to say, and it is said WHILE THE RUN IS STILL GOING
                // rather than discovered afterwards in a file. That silence is
                // the exact failure that produced a scan looking like nothing.
                var tickMessage = stageMessage(
                    for: iteration, of: effectiveTotal, sliceLabel: sliceLabel
                )
                if zeroGrowthStreak >= zeroGrowthStreakToReport {
                    let suffix: String = sliceLabel.isEmpty ? "" : " (" + sliceLabel + ")"
                    tickMessage = "Still working, but no new detail has been added for "
                        + String(zeroGrowthStreak) + " passes in a row" + suffix + "."
                }
                emit(
                    progressTick(
                        stage: stage(for: iteration, of: effectiveTotal),
                        iteration: iterationsRunSoFar + iteration,
                        total: governor.current.iterations,
                        splatCount: splatCount,
                        loss: lossEMA,
                        thermal: governor.thermalLevel,
                        message: tickMessage,
                        previewAvailable: iteration >= tuning.snapshotIntervalIterations,
                        // Both measured, both whole-run to match `iteration`
                        // above. `tickMessage` already says the second of
                        // these in prose; a number lets the screen show it as
                        // a state rather than only print it.
                        gradientStepsCompleted: gradientStepsRunSoFar + gradientSteps,
                        consecutiveZeroGrowthPasses: zeroGrowthStreak
                    )
                )
            }

            iteration += 1
        }

        iterationsRunSoFar += iteration

        census.iterationsCompleted += iteration
        census.slices[censusRow].iterationsCompleted = iteration
        // Measured, not derived. It should equal `iterationsCompleted` minus
        // the three skip counters; storing both sides is what makes a skip
        // path that stops being counted visible instead of invisible.
        census.slices[censusRow].iterationsWithGradientStep = gradientSteps
        census.slices[censusRow].splatCountAtEndOfTraining = splatCount
        census.slices[censusRow].peakTileInstances = peakTileInstances
        census.slices[censusRow].splatCountAtPeakTileInstances =
            splatCountAtPeakTileInstances
        // Reset so the next slice measures itself rather than inheriting.
        peakTileInstances = 0
        splatCountAtPeakTileInstances = 0
        // The size the buffers were actually at when the slice ended, which is
        // not the size it started at if the governor stepped the resolution
        // down mid-run.
        census.slices[censusRow].renderWidth = renderSize.width
        census.slices[censusRow].renderHeight = renderSize.height
        if census.slices[censusRow].stopReason == TrainerCensus.unfinishedOutcome {
            census.slices[censusRow].stopReason = "ran out its iterations"
        }

        // --- What this slice ACTUALLY did, said out loud -------------------------
        //
        // Three numbers, all measured, none inferred:
        //   `iteration`      times round the loop
        //   `gradientSteps`  times an optimisation step actually ran
        //   the difference   iterations that burned budget and did nothing
        //
        // The census records the three skip reasons separately
        // (`iterationsSkippedNoSupervision`, `...NothingToRender`,
        // `...GrowingTileBuffer`) and sums them in `iterationsSkippedTotal`, so
        // `gradientSteps` is `iterationsCompleted` minus that sum and the two
        // sides cannot disagree. This line is what puts it in the log while the
        // run is fresh rather than leaving it to be worked out from a file.
        let skippedInSlice = iteration - gradientSteps
        TrainerLog.general.info(
            """
            Slice finished: \(iteration) iterations, \(gradientSteps) of them took a real \
            gradient step, \(skippedInSlice) did nothing. Densification ran \
            \(densifyPassesRun) pass(es); \(densifyPassesThatAddedNothing) of the passes that \
            were allowed to add something added nothing, worst run \
            \(longestZeroGrowthStreak) in a row.
            """
        )

        // The slice's own outcome sentence, and this is the point of the whole
        // change: a slice that went round its loop the full number of times but
        // optimised on almost none of them used to say "ran out its iterations",
        // which reads as a finished slice.
        //
        // APPENDED, NOT OVERWRITTEN. A slice stopped by heat already wrote why
        // it stopped, and that reason is the more important one; this is added
        // to it rather than in place of it, so neither fact is lost.
        if iteration > 0, gradientSteps * 2 < iteration {
            census.slices[censusRow].stopReason +=
                " (only \(gradientSteps) of \(iteration) iterations took a real training step; "
                + "the other \(skippedInSlice) did nothing)"
        } else if longestZeroGrowthStreak >= zeroGrowthStreakToReport {
            census.slices[censusRow].stopReason +=
                " (densification added nothing for \(longestZeroGrowthStreak) consecutive "
                + "passes that were allowed to add something)"
        }

        gradientStepsRunSoFar += gradientSteps
        zeroGrowthStreakWorstSoFar = Swift.max(
            zeroGrowthStreakWorstSoFar, longestZeroGrowthStreak
        )

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

            // The same measurement on frames the model DID see. Sampled
            // evenly across the shuffle rather than taking the first few, and
            // limited to the SAME COUNT as the held-out set so the two numbers
            // cost the same and mean the same thing. Six extra renders once,
            // at the end, against three thousand iterations.
            let wanted = slice.heldOutKeyframes.count
            if wanted > 0, slice.keyframes.count >= wanted {
                let stride = Swift.max(1, slice.keyframes.count / wanted)
                let sampled = Swift.stride(
                    from: 0, to: slice.keyframes.count, by: stride
                ).prefix(wanted).map { slice.keyframes[$0] }
                let trained = try evaluateHeldOut(
                    gpu: gpu,
                    resources: resources,
                    queue: queue,
                    frames: Array(sampled),
                    supervision: supervision,
                    cameraDeltas: cameraDeltas,
                    exposures: exposures,
                    splatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    renderSize: renderSize
                )
                census.slices[censusRow].trainedPSNR = trained
                if let trained, let psnr {
                    let shown = String(format: "%.2f", trained)
                    let gap = String(format: "%.2f", trained - psnr)
                    TrainerLog.general.info(
                        "Trained-view PSNR \(shown, privacy: .public) dB, \(gap, privacy: .public) dB above held-out"
                    )
                }
            }

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
        let uploadFrom = CFAbsoluteTimeGetCurrent()
        resources.gtColor.writeArray(supervision.groundTruth)
        if supervision.hasBackground, !supervision.background.isEmpty {
            resources.bgColor.writeArray(supervision.background)
        }
        let sampleCount = Swift.min(
            supervision.depthSamples.count, resources.depthSampleCapacity
        )
        if sampleCount > 0 {
            // NOT `Array(...prefix(sampleCount))`. That allocated and copied
            // 1.57 MB every iteration to produce exactly what writeArray
            // would have written anyway: it clamps to `length / stride`,
            // which IS `sampleCount`, since sampleCount is already
            // min(count, depthSampleCapacity).
            resources.depthSamples.writeArray(supervision.depthSamples)
        }
        timings.upload += CFAbsoluteTimeGetCurrent() - uploadFrom

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
        // The divisor that turns the five geometry terms in `trainer_loss_depth`
        // into per-sample MEANS, so they sit on the same scale as the two
        // photometric terms instead of ~10^4 above them.
        //
        // `supervisedSampleCount` counts the WHOLE sample array. `sampleCount`
        // above is the prefix that fitted in the GPU buffer, and the two are
        // the same number on every normal frame. When capacity truncates, the
        // count is retaken over exactly the prefix that was uploaded: dividing
        // by samples the GPU never saw would quietly weaken the geometry terms
        // on precisely the densest frames.
        let supervisedCount: Int
        if sampleCount == supervision.depthSamples.count {
            supervisedCount = supervision.supervisedSampleCount
        } else {
            supervisedCount = supervision.depthSamples.prefix(sampleCount)
                .reduce(into: 0) { $0 += ($1.weight > 0 ? 1 : 0) }
        }
        loss.depthSupervisedCount = UInt32(supervisedCount)
        loss.bimodalWeight = settings.bimodalWeight
        loss.transitionWidthWeight = settings.transitionWidthWeight
        loss.freeSpaceWeight = settings.freeSpaceLowerBoundWeight
        loss.alphaSupervisionWeight = tuning.alphaSupervisionWeight
        loss.hasBackground = supervision.hasBackground ? 1 : 0

        // --- Command buffer A: preprocess and size the sort -------------------------
        //
        // POOLED, and this is the biggest single accumulator left in the app.
        // `makeCommandBuffer()` and `makeComputeCommandEncoder()` are
        // Objective-C methods that hand back AUTORELEASED objects, and a
        // command buffer keeps a reference to every resource it was encoded
        // against until it is deallocated. This function runs once per training
        // iteration, and `TrainingBudget` sets 2,000 or 3,000 of those for the
        // WHOLE run: `governor.ceiling.iterations` is that number and each slice
        // takes `sliceFraction` of it, so the slices divide the budget rather
        // than each spending it. The loop that calls this never suspends, so
        // with no pool of its own every command buffer and encoder of the
        // entire run stays alive until the run ends, each one pinning the
        // trainer's Metal buffers. Two per iteration here and two in buffer B
        // below is eight to twelve thousand live objects on a full run.
        //
        // The pool closes after `Self.finish`, which is where the wait lives, so
        // the GPU work has finished and its results are already in `resources`;
        // nothing below reads `bufferA` or `encoderA` again. A fault makes
        // `finish` throw, and the pool drains on the way out just as it does on
        // the normal path. `camera` is a local of this function captured by a
        // non-escaping closure, so passing it inout in here is the same store it
        // was before.
        try autoreleasepool { () throws -> Void in
            guard let bufferA = queue.makeCommandBuffer(),
                  let blitA = bufferA.makeBlitCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            // The zeroing is a DMA now rather than nine compute dispatches, so
            // it needs a blit encoder, and an encoder has to be ended before
            // the next one begins. Encoders run in the order they were created
            // within a command buffer, so every fill is complete before
            // anything below reads what it cleared.
            blitA.label = "trainer.clear"
            gpu.clearPerIteration(blitA, splatCount: splatCount)
            blitA.endEncoding()

            guard let encoderA = bufferA.makeComputeCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            encoderA.label = "trainer.preprocess"
            gpu.resetVisibilityForIteration(encoderA, splatCount: splatCount)
            gpu.preprocess(encoderA, camera: &camera, splatCount: splatCount)
            gpu.exclusiveScan(
                encoderA,
                input: resources.tilesTouched,
                output: resources.offsets,
                count: splatCount
            )
            encoderA.endEncoding()
            bufferA.commit()
            try finish(bufferA, "the tile scan")
        }

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

        // Recorded rather than discarded. See the census fields for why:
        // this is the only place the app ever learns how many tiles a
        // Gaussian really touches, and the sort buffers are sized on an
        // assumption about it that has never been checked.
        //
        // Held on the instance because `runIteration` takes no census: it is
        // folded into the slice row where the slice is finalised.
        if instanceCount > peakTileInstances {
            peakTileInstances = instanceCount
            splatCountAtPeakTileInstances = splatCount
        }

        // --- Command buffer B: everything else ---------------------------------------
        //
        // Pooled for the same reason as buffer A above, and this is the
        // expensive half: the encoder it drains carries the whole forward and
        // backward pass and both Adam steps, so it references nearly every
        // buffer in `TrainerResources`. `reg` and `adam` are declared inside
        // the pool because nothing outside it reads them; `camera`, `loss`,
        // `instanceCount` and `sampleCount` are locals of this function
        // captured by a non-escaping closure.
        //
        // The pool closes after `Self.finish`, so every readback below is
        // reading finished results out of `resources` rather than anything the
        // pool owned.
        // BACK TO ONE COMMAND BUFFER. It was split five ways to find out
        // which kernel owned the GPU, and it did: backward raster 13.18 ms,
        // forward 4.63, sort 2.19, losses 1.99, optimiser 1.23. That answer is
        // recorded and the split has stopped earning its keep, which was four
        // extra commit-and-waits at about 0.24 ms of pure round trip each,
        // roughly 1 ms per iteration and 12,000 extra command buffers a run.
        //
        // The per-stage census fields stay. Nothing writes them now, so they
        // read zero, and a zero in TrainerTimings means zero rather than "not
        // instrumented" only for the fields that ARE measured; these are the
        // exception and this comment is the note saying so. Re-splitting is
        // uncommenting one function.
        try autoreleasepool { () throws -> Void in
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
            try finish(bufferB, "the tile sort")
        }

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
        } else if let background, iteration > warmupEnd, !background.isFrozen {
            // Was `iteration == warmupEnd + 1`. `warmupEnd` is now derived from
            // `effectiveTotal`, which MOVES when the governor shortens the run,
            // so an exact equality can be stepped straight over: one budget cut
            // that pushes `warmupEnd` below the current iteration and the
            // background would train for the whole run, absorbing foreground
            // error forever, with nothing logged. A range test plus the model's
            // own flag cannot be missed, and `freeze()` still logs exactly once
            // because it checks that flag itself.
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
        try finish(buffer, "a reset pass")
    }

    /// Wait for a batch of GPU work AND ask whether it actually worked.
    ///
    /// `waitUntilCompleted()` returns when the GPU is finished, not when
    /// it has succeeded. A command buffer that faulted comes back from it
    /// exactly like one that did not, with `status == .error` and a
    /// populated `error`, and every call site in this file used to ignore
    /// both and read the output buffers regardless.
    ///
    /// That mattered more than a missing check usually does, because a GPU
    /// fault on iOS can take the whole process down without leaving a
    /// crash report anywhere a person can find one. The owner hit a repeat
    /// crash during processing with no app-named report and no JetsamEvent
    /// at the time, which is what that looks like from the outside.
    ///
    /// Throwing here cannot stop the driver killing the process. What it
    /// does is turn every fault the process SURVIVES into a named stage
    /// and a message, instead of a wrong model built from whatever was
    /// left in the buffers.
    ///
    /// It is also where the run is timed. EVERY command buffer in this
    /// file goes through here, so one pair of clocks in this function
    /// measures the whole GPU side of the run: how long the CPU spent
    /// blocked, and separately what Metal says the GPU spent executing.
    /// Those two being far apart is itself the finding.
    func finish(_ buffer: MTLCommandBuffer, _ stage: String) throws {
        let blockedFrom = CFAbsoluteTimeGetCurrent()
        buffer.waitUntilCompleted()
        timings.gpuWait += CFAbsoluteTimeGetCurrent() - blockedFrom
        timings.commandBuffers += 1
        // GPUStartTime and GPUEndTime are populated once the buffer has
        // completed and are zero if the device did not report them, which
        // is why this is guarded rather than trusted.
        let executing = buffer.gpuEndTime - buffer.gpuStartTime
        if executing.isFinite, executing > 0 {
            timings.gpuBusy += executing
            // Bucketed by the caller's own stage name, which every command
            // buffer already carries, so this costs a string compare per
            // buffer and no Metal objects at all.
            switch stage {
            case "the tile scan": timings.gpuScan += executing
            case "the tile sort": timings.gpuSort += executing
            case "the forward raster": timings.gpuForward += executing
            case "the losses": timings.gpuLosses += executing
            case "the backward raster": timings.gpuBackward += executing
            case "the optimiser": timings.gpuOptimiser += executing
            default: timings.gpuOther += executing
            }
        }
        if let error = buffer.error {
            TrainerLog.gpu.error(
                """
                GPU work for \(stage, privacy: .public) failed: \
                \(error.localizedDescription, privacy: .public)
                """
            )
            throw TrainerError.gpuFailed(
                stage: stage, detail: error.localizedDescription
            )
        }
        guard buffer.status == .completed else {
            let detail = "the command buffer ended in state "
                + "\(buffer.status.rawValue) rather than completed"
            TrainerLog.gpu.error(
                "GPU work for \(stage, privacy: .public) \(detail, privacy: .public)"
            )
            throw TrainerError.gpuFailed(stage: stage, detail: detail)
        }
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

        // ONE COMMAND BUFFER PER CAMERA, not one for the whole sweep.
        //
        // This used to encode the clear, up to 64 per-camera passes over
        // every Gaussian, and the finalise into a SINGLE command buffer and
        // wait on it once. At the 204,000 Gaussians a room reaches by the
        // time this first runs, that is around thirteen million kernel
        // invocations in one submission, and every one of them does atomic
        // work on the shared top-K list.
        //
        // A single submission that long is what the GPU driver watchdog
        // exists to catch. When it fires, the driver resets the GPU and
        // takes the process with it, and iOS writes no crash report a
        // person can find in Settings.
        //
        // That matches the observed failure exactly. A screen recording of
        // a full run shows the app vanish at round 500 of 3000 with memory
        // flat at about 2.05 GB and 1.5 GB still available, so it is not
        // memory, and there was no app-named crash report and no
        // JetsamEvent at that time. Iteration 500 is the FIRST iteration
        // this sweep ever runs, because filter3DIntervalIterations is 500.
        //
        // Splitting the work does not make it less work. It makes each
        // submission short enough that the watchdog has nothing to catch,
        // and it gives the driver a boundary to schedule other work at,
        // including the preview.
        func submit(_ label: String, _ body: (MTLComputeCommandEncoder) -> Void) throws {
            guard let buffer = queue.makeCommandBuffer(),
                  let encoder = buffer.makeComputeCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            encoder.label = label
            body(encoder)
            encoder.endEncoding()
            buffer.commit()
            try finish(buffer, label)
        }

        // The top-K list is rebuilt from scratch: it is a running maximum, and
        // a Gaussian that moved since the last sweep would otherwise keep a
        // rate it earned somewhere else.
        try submit("the 3D filter clear") { encoder in
            gpu.fillFloat(
                encoder,
                buffer: resources.samplingTopK,
                count: splatCount * 4,
                value: 0
            )
        }

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
            // Cancellation is honoured between cameras now that each one
            // is its own submission. Before, a stop during the sweep had
            // to wait for all 64 passes to finish first.
            if Task.isCancelled { throw NimbusError.cancelled }
            try submit("the 3D filter sweep") { encoder in
                gpu.samplingRateUpdate(
                    encoder, camera: &camera, splatCount: splatCount
                )
            }
            index += stride
        }

        try submit("the 3D filter finalise") { encoder in
            gpu.filter3DFinalize(
                encoder,
                splatCount: splatCount,
                filterScale: tuning.filter3DScale,
                fallback: tuning.filter3DFallbackMeters
            )
        }
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
            try finish(bufferA, "the filter sweep")

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
            try finish(bufferB, "the filter finalise")

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

        // THE 3D LOW-PASS FILTER HAS TO LEAVE WITH THE MODEL, AND IT CAN ONLY
        // LEAVE FOLDED IN.
        //
        // `trainer_preprocess` never renders the Gaussians stored in `splats`.
        // It renders each one widened to `Sigma + filter3D^2 * I` and dimmed by
        // `sqrt(det(Sigma) / det(Sigma + filter3D^2 * I))`. Every opacity the
        // optimiser fitted was fitted against that dimming and every scale
        // against that widening, so a cloud read off the GPU without them is a
        // DIFFERENT MODEL from the one that was trained: sharper and more
        // solid, which is the direction that makes a good run look like noise.
        //
        // `filter3D` exists in exactly one place, `TrainerSplatStats.filter3D`,
        // and no splat file format has a field for it, so it is folded into the
        // stored log-scale and opacity here, once, by the single
        // implementation in Export (`SplatCloud.fuse3DFilter`). Do not
        // re-derive the formula: one copy is the whole point of it living
        // there.
        let stats = resources.stats.readArray(TrainerSplatStats.self, count: count)
        // `readArray` returns an EMPTY array when the buffer is shorter than
        // asked for, so this is a real test and not a formality. Indexing a
        // short array with `i` below would be a crash, and pairing a filter
        // with the wrong Gaussian would be worse: it would look like a
        // slightly wrong model rather than like a bug.
        let haveFilters = stats.count == splats.count

        var positions: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var logScales: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var colorDC: [SIMD3<Float>] = []
        var shRest: [[SIMD3<Float>]] = []
        // Appended inside the same loop and after the same `continue`, so it
        // stays index-for-index with `logScales` and `opacities`. A filter
        // list built by indexing the ORIGINAL array with `i` afterwards would
        // drift by one for every dropped Gaussian.
        var filters: [Float] = []
        positions.reserveCapacity(splats.count)
        filters.reserveCapacity(splats.count)

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
            if haveFilters { filters.append(stats[i].filter3D) }

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

        var cloud: SplatCloud
        do {
            cloud = try SplatCloud(
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

        // The fuse is deliberately NOT inside the `do` above. A cloud that was
        // assembled and could not be fused is still a cloud, and throwing it
        // away would turn a cosmetic loss into an empty model. It is also not
        // allowed to pass in silence, so both failure paths set the flag to
        // `false`: the value that means "the producer KNEW there was a filter
        // and did not apply it", which the viewer already warns on.
        guard haveFilters, filters.count == cloud.count else {
            cloud.filter3DFused = false
            let complaint = "Read \(cloud.count) points back but \(filters.count) filter "
                + "widths, so the 3D low-pass filter could NOT be folded in. Every point "
                + "will draw sharper and more solid than it was trained."
            TrainerLog.general.error("\(complaint, privacy: .public)")
            return cloud
        }
        do {
            let changed = try cloud.fuse3DFilter(filters)
            TrainerLog.general.notice(
                "Folded the 3D low-pass filter into \(changed) of \(cloud.count) points."
            )
        } catch {
            cloud.filter3DFused = false
            // `String(describing:)`, not `localizedDescription`: `ExportError`
            // is `CustomStringConvertible` and not `LocalizedError`, so
            // `localizedDescription` would print Foundation's generic
            // "operation could not be completed" instead of the reason.
            let complaint = "The 3D low-pass filter could not be folded in: "
                + String(describing: error)
            TrainerLog.general.error("\(complaint, privacy: .public)")
        }
        return cloud
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

        guard var merged = try? SplatCloud(
            shDegree: current.shDegree,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacities,
            colorDC: colorDC,
            shRest: current.shDegree == .zero ? [] : shRest
        ) else { return current }
        // A fresh cloud starts at `nil`, so without this the merged preview
        // would report "cannot say" about a fact every one of its parts knew.
        // Only the parts whose splats were actually taken above: a part at a
        // different SH degree contributed nothing and must not vote.
        let contributing = completedParts.filter { $0.shDegree == current.shDegree }
        merged.filter3DFused = SplatCloud.mergedFilter3DFused(
            [current.filter3DFused] + contributing.map { $0.filter3DFused }
        )
        return merged
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

    /// The one sentence the user reads at the end.
    ///
    /// `iterationsRun` and `gradientSteps` are separate arguments on purpose.
    /// This used to be handed only the finished cloud, so a run that went round
    /// its loop three thousand times and optimised on none of them produced the
    /// identical cheerful sentence as a run that worked. If the second number
    /// is far below the first, the sentence says so, in plain words, on the
    /// screen the owner actually looks at.
    private func doneMessage(
        cloud: SplatCloud,
        psnr: Float?,
        governor: TrainerBudgetGovernor,
        iterationsRun: Int,
        gradientSteps: Int,
        worstZeroGrowthStreak: Int
    ) -> String {
        var sentence = "Your scan is ready, built from "
            + "\(TrainerBudgetGovernor.round(cloud.count)) points of detail."
        if iterationsRun > 0, gradientSteps * 2 < iterationsRun {
            sentence += " Be warned: only " + String(gradientSteps) + " of "
                + String(iterationsRun) + " training rounds actually did any work, so this "
                + "model has had much less training than it looks like."
        } else if worstZeroGrowthStreak >= Self.zeroGrowthPassesBeforeSaying {
            sentence += " Be warned: it stopped adding new detail for "
                + String(worstZeroGrowthStreak) + " rounds in a row part way through, so it "
                + "may be thinner than it should be."
        }
        if !governor.changes.isEmpty {
            sentence += " It was made a little smaller along the way because "
                + (governor.changes.last?.reason.plainCause ?? "the phone needed the room") + "."
        }
        if let psnr {
            sentence += " On the photos it was not trained with, it scores "
                + String(format: "%.1f", psnr) + " decibels."
            // Said in full, every time, and deliberately not shortened to
            // "measured on held-out frames". That phrase is exactly the one
            // that lets a wrong reading survive: it sounds like a score for
            // the preview, and it is not measured on the preview.
            sentence += " That score compares the model against photos it never trained on, "
                + "at the size it was trained at. It is not a score for how the preview looks "
                + "on screen."
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
        previewAvailable: Bool,
        // Defaulted to nil, which means NOBODY COUNTED, not zero. Only the
        // in-loop tick can honestly supply these; the stage-change ticks
        // around it fire before, between and after the loop, and passing a
        // stale count from one of those would be a measurement presented at
        // the wrong moment.
        gradientStepsCompleted: Int? = nil,
        consecutiveZeroGrowthPasses: Int? = nil
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
            previewAvailable: previewAvailable,
            gradientStepsCompleted: gradientStepsCompleted,
            consecutiveZeroGrowthPasses: consecutiveZeroGrowthPasses
        )
    }
}
