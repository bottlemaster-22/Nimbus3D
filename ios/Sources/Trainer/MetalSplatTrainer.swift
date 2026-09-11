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

/// SplitMix64: a fixed, portable generator, so the keyframe visiting order is
/// the same on every run of a build.
struct TrainerSplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public final class MetalSplatTrainer: SplatTrainer, @unchecked Sendable {

    /// Where the run's time actually went. Written by `finish` and by the
    /// supervision call in the training loop, copied into the census when
    /// the run is sealed. Single-threaded: everything that touches it runs
    /// on the training thread.
    var timings = TrainerTimings()

    /// Running mean of the per-slice held-out PSNR. `heldOutPSNR` used to be
    /// `min` across slices, which reported the WORST part of the scene as the
    /// model's score; a number that only moves when the weakest slice moves
    /// cannot measure a change made everywhere. Reset per run, exactly like
    /// `timings`, because a stale accumulator here would silently average two
    /// different models together.
    /// SSIM from the most recent `evaluateHeldOut`. A second return value
    /// would mean touching every call site for a diagnostic, so it rides here;
    /// every caller reads it immediately after the call it belongs to.
    private var lastHeldOutSSIM: Float?
    /// Set by evaluateHeldOut(alsoScoreExposureFitted: true); nil otherwise.
    private var lastHeldOutPSNRExposureFitted: Float?
    /// (frame index, raw PSNR) for every frame the most recent evaluateHeldOut
    /// scored, in the order it scored them. Read immediately after the call.
    private var lastHeldOutPerFrame: [TrainerHeldOutFrameScore] = []
    /// Set by the backward calibration (build 288): use the SIMD-summed
    /// backward rasteriser for the rest of the run.
    private var backwardSimdSumChosen = false
    /// Set by the sort calibration (build 290): use the SIMD-prefix radix
    /// scatter for the rest of the run.
    private var sortSimdScanChosen = false
    /// Set by the forward calibration (build 302): use the two-pixel forward
    /// rasteriser for the rest of the run.
    private var forwardTwoPixelChosen = false
    /// Set by the two-pixel backward calibration (build 304).
    private var backwardTwoPixelChosen = false
    /// Set by the blur calibration (build 318): use the fused SSIM blur.
    private var blurFusedChosen = false
    /// Build 320: the densifier applies its decision on the GPU; cleared
    /// for the run if the first pass's check found a difference.
    private var densifyGatherUsable = true
    /// Build 326: the cubemap-too-large complaint is said once per trainer.
    private var loggedCubemapTruncation = false
    /// Build 328: the active render size over the full one (0.5 in the
    /// coarse phase, 1 otherwise). cameraUniforms scales the coarse-to-fine
    /// blur by its square so the blur in the photograph is unchanged.
    private var resolutionScale: Float = 1
    /// Build 330: the camera learning rates are multiplied by this. 0.1
    /// (build 292's inert rate) until the pose check has passed, then 1.
    private var poseRefinementScale: Float = 0.1
    private var poseCheckDone = false
    /// Set by the sort calibration (build 306): use the splat-order tile sort.
    private var splatOrderChosen = false

    /// A training step whose command buffer B was committed and left running
    /// (build 292): the CPU went on to the next iteration's supervision and
    /// buffer A instead of idling the GPU through that work. Completed by
    /// `drainPendingStep`.
    private struct PendingStep {
        let buffer: MTLCommandBuffer
        let slot: Int
        let frame: CaptureFrame
        let exposure: SIMD2<Float>
        let iteration: Int
        let totalIterations: Int
        let splatCount: Int
        /// Build 316: one command buffer with the sort sized on the GPU; its
        /// staging slot also carries the instance count (needed, used).
        let merged: Bool
        /// Build 322: this step's command buffer copied that many Gaussians
        /// (splats, sh, stats) into snapshotStaging for the preview; 0 = none.
        let snapshotCount: Int
        /// Build 324: a warm-up step, whose buffer copied gradFinal and
        /// renderTFinal into warmupStaging for the far field's update. The
        /// supervision (pose, intrinsics) and the model it updates ride along.
        let warmup: Bool
        let supervision: TrainerFrameSupervision?
        let background: DirectionalBackgroundModel?
    }
    private var pendingStep: PendingStep?
    /// Build 316: a merged step ran short of instance slots and needs this
    /// many; the buffers grow before the next merged step is encoded.
    private var pendingInstanceGrowth: Int?
    /// Build 322: a preview snapshot is wanted; the next merged step copies
    /// the model into the staging inside its own command buffer and the copy
    /// is converted off the loop, instead of draining the GPU to read the
    /// live buffers. All four under `lock`: the conversion runs elsewhere.
    private var snapshotRequested = false
    private var snapshotParts: [SplatCloud] = []
    private var snapshotDegree: SHDegree = .zero
    private var snapshotConverting = false

    /// Held-out (and trained-view) supervision built by evaluateHeldOut, kept
    /// for the run (build 300). ~1.2 MB a frame (bytes since 314), 11 or 12 frames.
    private var evalSupervisionCache: [FrameID: TrainerFrameSupervision] = [:]

    /// One frame's render, read back by evaluateHeldOut for scoring.
    private struct HeldOutRender {
        let frame: CaptureFrame
        let rendered: [Float]
        let transmittance: [Float]
        let background: [Float]?
        /// The supervision bytes as they are (build 318); scoreHeldOut turns
        /// each into `Float(byte) / 255`, the decoder's float, where it reads it.
        let groundTruth: [UInt8]
        let exposure: SIMD2<Float>
    }

    /// One frame's scores. `ssim` is nil when no 8x8 block qualified, which is
    /// when the loop used to add nothing to the SSIM total.
    private struct HeldOutScore {
        let psnr: Double
        let psnrFitted: Double
        let ssim: Double?
    }
    private var heldOutPSNRSum: Double = 0
    private var heldOutPSNRCount: Int = 0

    /// How hot it got and when. A class property for the same reason
    /// `timings` is: the loop that sees the thermal level lives in
    /// `trainSlice`, and the census is sealed in `run`.
    var thermals = TrainerCensus.Thermals()
    private var thermalMark = Date()
    private var thermalLevel = 0

    /// Closes the interval at the current level and opens one at `level`.
    /// Called only when the level actually changes, from the one place in the
    /// loop that already reads it, so it adds no polling.
    func markThermal(_ level: ThermalLevel, iteration: Int) {
        let now = Date()
        thermals.secondsAtLevel[thermalLevel] += now.timeIntervalSince(thermalMark)
        thermalMark = now
        thermalLevel = level.rawValue
        if thermals.firstReachedAtIteration[thermalLevel] < 0 {
            thermals.firstReachedAtIteration[thermalLevel] = iteration
        }
        thermals.peak = Swift.max(thermals.peak, thermalLevel)
    }

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
        // ZEROED PER RUN, beside the census it will be copied into.
        //
        // `timings` is a stored property on this class, so a trainer instance
        // that ran twice handed the second run a census carrying BOTH runs'
        // totals against only the second run's startedAt and finishedAt. It
        // surfaced as gpuWait 115.3 s inside a 68 s run, which is impossible
        // on its face: that clock is main-thread blocking time and cannot
        // exceed the wall clock. Every per-run field was still correct, so the
        // conclusions drawn from splat counts and wall clock survive; only the
        // timings table was wrong, and only on a second run.
        timings = TrainerTimings()
        backwardSimdSumChosen = false
        sortSimdScanChosen = false
        forwardTwoPixelChosen = false
        backwardTwoPixelChosen = false
        blurFusedChosen = false
        densifyGatherUsable = true
        poseRefinementScale = tuning.poseRefinementCheck ? 0.1 : 1
        poseCheckDone = !tuning.poseRefinementCheck
        splatOrderChosen = false
        // A step left running by a run that threw cannot belong to this one.
        // Its buffer finishes on its own; its read-backs are not wanted.
        pendingStep = nil
        pendingInstanceGrowth = nil
        lock.lock()
        snapshotRequested = false
        snapshotConverting = false
        lock.unlock()
        evalSupervisionCache.removeAll()
        heldOutPSNRSum = 0
        heldOutPSNRCount = 0
        thermals = TrainerCensus.Thermals()
        thermalMark = Date()
        thermalLevel = ThermalLevel(ProcessInfo.processInfo.thermalState).rawValue
        thermals.firstReachedAtIteration[thermalLevel] = 0
        thermals.peak = thermalLevel
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
            // Close the open interval so the seconds add up to the run.
            thermals.secondsAtLevel[thermalLevel] +=
                Date().timeIntervalSince(thermalMark)
            census.thermals = thermals
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
        let smartClock = CFAbsoluteTimeGetCurrent()
        let smart = try await loadSmartLayer(bundle: bundle, prePass: prePass, at: ref)
        timings.smartLayer += CFAbsoluteTimeGetCurrent() - smartClock

        // --- Keyframes --------------------------------------------------------
        let selection = selectKeyframes(bundle: bundle, prePass: prePass, budget: governor.current)
        let keyframes = selection.keyframes
        guard !keyframes.isEmpty else { throw TrainerError.noKeyframes }

        let slices = TrainerSlicePlanner.plan(
            bundle: bundle,
            prePass: prePass,
            keyframes: keyframes,
            fixedHeldOut: selection.fixedHeldOut,
            budget: governor.current,
            heldOutFraction: tuning.heldOutFraction
        )
        guard !slices.isEmpty else { throw TrainerError.noKeyframes }

        TrainerLog.general.info(
            "Training \(slices.count) time slice(s) from \(keyframes.count) keyframes"
        )
        census.keyframesSelected = keyframes.count
        census.keyframeFirstIndex = keyframes.map { Int($0.index) }.min() ?? -1
        census.keyframeLastIndex = keyframes.map { Int($0.index) }.max() ?? -1
        census.framesInBundle = bundle.frames.count
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

        // WARM THIS SLICE'S EDGE MAPS ON EVERY CORE, AND FINISH BEFORE THE LOOP.
        // Build 256 made edge maps lazy, which took about 4 s out of the
        // pre-pass, but the trainer's own maps were then built one at a time
        // on the supervision prefetch worker during the first cycle, and the
        // loop waited on it: supervision 1.5 s to 4.0 s. Build 258 warmed them
        // in the BACKGROUND, which only got that to 3.5 s: the warm-up raced
        // the prefetch worker for the same cores and the same frames. This
        // blocks instead, about 120 maps across every core, so the first
        // cycle finds every map built.
        if let edges = smart.edges {
            let warm = (slice.keyframes + slice.heldOutKeyframes).map(\.index)
            DispatchQueue.concurrentPerform(iterations: warm.count) { i in
                _ = edges.map(for: warm[i])
            }
        }

        let sliceLabel = slice.label(of: sliceCount)
        let shDegree = governor.current.shDegree
        let shCoefficientCount = 1 + shDegree.restCoefficientCount

        // Build 344: ONE depth cache shared by the level builders, holding
        // every training and held-out frame. Three builders each held 128
        // frames of the same native depth (about 79 MB for 108 frames), and
        // at 200 frames a 128-entry cache would have missed on every visit.
        let sharedDepthCache = SmartDepthCache(
            capacity: slice.keyframes.count + slice.heldOutKeyframes.count + 8,
            sampleCount: Swift.max(bundle.settings.depthWidth, 1) * Swift.max(bundle.settings.depthHeight, 1)
        )
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
            background: smart.background,
            depthCache: sharedDepthCache
        )
        // Builds the NEXT frame while the GPU works on this one. See
        // TrainerSupervisionPrefetch: supervision and gpuWait measured 24.70
        // and 24.69 ms per iteration on the owner's phone, one after the
        // other, with the other device idle each time.
        let prefetch = TrainerSupervisionPrefetch(builder: supervision)
        // BUILD 328/332: ONE BUILDER PER RESOLUTION LEVEL, each decoding at
        // its own long edge with its own prefetch worker and frame cache. The
        // loop takes each iteration's frame from the level that iteration
        // belongs to, and the render buffers follow the frame's size as they
        // always have. Each level's frames are decoded in the background
        // before its phase starts (TrainerSupervisionPreload).
        let fullLongEdge = governor.current.renderLongEdgePixels
        // Integer arithmetic on clamped per-milles, as the densify schedule
        // does: a NaN entry takes the else branch and is dropped.
        var levelScalesPerMille: [Int] = []
        var levelFractionsPerMille: [Int] = []
        for (fraction, scale) in zip(tuning.coarseResolutionFractions, tuning.coarseResolutionScales) {
            guard fraction.isFinite, scale.isFinite else { continue }
            let f = Int(Swift.min(Swift.max(fraction, 0), 1) * 1000)
            let sc = Int(Swift.min(Swift.max(scale, 0), 1) * 1000)
            guard f > 0, sc > 0, sc < 1000, fullLongEdge * sc / 1000 >= 64 else { continue }
            // Levels must be in order; a fraction that does not grow ends the list.
            if let last = levelFractionsPerMille.last, f <= last { break }
            levelFractionsPerMille.append(f)
            levelScalesPerMille.append(sc)
        }
        let coarsePhaseOn = !levelFractionsPerMille.isEmpty
        let coarseSupervisions: [TrainerSupervisionBuilder] = levelScalesPerMille.map { sc in
            TrainerSupervisionBuilder(
                bundle: bundle,
                prePass: prePass,
                at: ref,
                longEdgePixels: fullLongEdge * sc / 1000,
                settings: settings,
                tuning: tuning,
                trust: smart.trust,
                authority: smart.authority,
                edges: smart.edges,
                background: smart.background,
                depthCache: sharedDepthCache
            )
        }
        let coarsePrefetches = coarseSupervisions.map { TrainerSupervisionPrefetch(builder: $0) }
        if let first = levelScalesPerMille.first {
            timings.coarseLongEdgePixels = fullLongEdge * first / 1000
        }
        // One preload per level, plus one for the full builder; level 0's
        // starts now (it overlaps the seed load below), each later one when
        // the level before it becomes active.
        let preloads: [TrainerSupervisionPreload] = (coarseSupervisions + [supervision]).map {
            TrainerSupervisionPreload(builder: $0, frames: slice.keyframes)
        }
        // The full builder fixes its grid and intrinsics from its first
        // decode. The 3D-filter sweep reads those intrinsics at iteration 500,
        // inside the levels, before the full builder would otherwise have
        // decoded anything, so it decodes one frame now (no samples, not
        // cached: about 15 ms).
        if coarsePhaseOn, let first = slice.keyframes.first {
            _ = supervision.build(frame: first, iteration: 0, totalIterations: 1, includeDepthSamples: false)
        }
        if coarsePhaseOn, tuning.supervisionPreload { preloads.first?.start() }
        defer {
            prefetch.drain()
            coarsePrefetches.forEach { $0.drain() }
            preloads.forEach { $0.cancelAndJoin() }
            // Accumulated across slices: what the worker built off the
            // critical path, which `timings.supervision` no longer sees.
            timings.supervisionPrefetched += prefetch.workerSeconds
                + coarsePrefetches.reduce(0) { $0 + $1.workerSeconds }
            timings.supervisionCacheHits += supervision.frameCacheHits
                + coarseSupervisions.reduce(0) { $0 + $1.frameCacheHits }
            timings.supervisionPreloaded += preloads.reduce(0) { $0 + $1.built }
            timings.supervisionCacheMegabytes = Swift.max(
                timings.supervisionCacheMegabytes, Double(supervision.frameCacheBytes) / 1_048_576
            )
            resolutionScale = 1
        }


        var renderSize = TrainerBudgetGovernor.renderSize(
            forLongEdge: governor.current.renderLongEdgePixels,
            intrinsics: bundle.intrinsics
        )
        // The full-size grid, for the 3D-filter sweep during the coarse phase
        // (its sampling rates are a property of the full-size camera).
        var fullRenderSize = renderSize
        // Levels whose phase has ended have had their caches dropped up to
        // this index; preloads started up to this index.
        var levelsDropped = 0
        var preloadsStarted = 1

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
        // TIMED: see TrainerTimings.prologue.
        let seedClock = CFAbsoluteTimeGetCurrent()
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
        timings.prologue += CFAbsoluteTimeGetCurrent() - seedClock
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
        // A fixed shuffle rather than a fresh one each epoch: an epoch still
        // visits every keyframe exactly once, which is what matters for
        // coverage. SEEDED since build 276: `shuffle()` alone drew from the
        // system generator, so the order (and the model) differed on every
        // run and the comment's "reproducible" was false. Two runs of one
        // build now train the same sequence, which takes that share out of
        // the run-to-run noise every A/B is read against.
        var orderRNG = TrainerSplitMix64(seed: 0x4C69_4B4F_5641)
        order.shuffle(using: &orderRNG)

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

        // EARLY STOPPING STATE. See `earlyStopEvalIntervalIterations`.
        //
        // `bestHeldOut` is the best score any evaluation has seen, and
        // `sinceBest` counts consecutive evaluations that failed to beat it.
        // The whole curve goes into the census either way, because knowing
        // WHERE the model stopped improving is worth as much as stopping
        // there: it is the only honest way to choose a fixed budget later.
        var bestHeldOut: Float = -.infinity
        var bestHeldOutIteration = 0
        /// THE BEST MODEL, KEPT. Build 234 measured 17.126 dB at iteration
        /// 2,000, carried on to 4,000, and exported the 16.335 dB model. The
        /// run knew which one was better and threw it away: 0.791 dB, measured
        /// on the same frames with the same exposure fit, discarded for
        /// nothing. Reading the cloud costs about the same as one preview
        /// snapshot, which this loop already does every 200 iterations, and it
        /// only happens when the score actually improves.
        var bestCloud: SplatCloud?
        var sinceBest = 0
        var stoppedEarly = false
        // Rounded UP to a whole number of densify intervals, because the
        // evaluation has to sit immediately before a densify pass so that
        // pass's stats reset wipes what the extra preprocess wrote.
        let evalEvery: Int = {
            let want = tuning.earlyStopEvalIntervalIterations
            guard want > 0 else { return 0 }
            let step = Swift.max(tuning.densifyIntervalIterations, 1)
            return Swift.max(step, ((want + step - 1) / step) * step)
        }()

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

            // Build 328/332: the levels are shares of the run that will
            // actually happen (`effectiveTotal`, like every schedule). Level i
            // covers iterations below fraction i; `levelCount` is full size.
            let levelCount = coarseSupervisions.count
            func levelFor(_ at: Int) -> Int {
                for (i, perMille) in levelFractionsPerMille.enumerated()
                where at < effectiveTotal * perMille / 1000 { return i }
                return levelCount
            }
            func supervisionFor(_ at: Int) -> (TrainerSupervisionBuilder, TrainerSupervisionPrefetch) {
                let level = levelFor(at)
                if level < levelCount { return (coarseSupervisions[level], coarsePrefetches[level]) }
                return (supervision, prefetch)
            }
            let activeLevel = levelFor(iteration)
            let (activeSupervision, activePrefetch) = supervisionFor(iteration)
            let activeIsCoarse = activeLevel < levelCount
            // A level that has ended is never needed again: its worker is
            // joined and its frames released. And the level after the active
            // one starts decoding its frames now, while this level trains.
            while levelsDropped < activeLevel {
                coarsePrefetches[levelsDropped].drain()
                // Hits are summed once, by the slice's defer, over every
                // builder; `dropFrameCache` keeps the counter for it.
                coarseSupervisions[levelsDropped].dropFrameCache(disable: true)
                levelsDropped += 1
            }
            while tuning.supervisionPreload, preloadsStarted <= activeLevel + 1, preloadsStarted < preloads.count {
                preloads[preloadsStarted].start()
                preloadsStarted += 1
            }

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
                if thermal.level.rawValue != thermalLevel {
                    markThermal(thermal.level, iteration: iteration)
                }
                if iteration > 0, iteration % 100 == 0,
                   let change = governor.degradeForHeat(
                       level: thermal.level, currentSplatCount: splatCount
                   )
                {
                    // The governor can swap the builder's image cache out
                    // from under a worker. Nothing may be in flight.
                    prefetch.drain()
                    coarsePrefetches.forEach { $0.drain() }
                    preloads.forEach { $0.cancelAndJoin() }
                    try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                    try applyBudgetChange(
                        change,
                        resources: resources,
                        gpu: &gpu,
                        splatCount: &splatCount,
                        renderSize: &renderSize,
                        supervision: supervision,
                        coarseSupervisions: Array(zip(coarseSupervisions, levelScalesPerMille)),
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
                var reading = governor.measureMemory(resources: resources)
                timings.memoryFootprintPeakMegabytes = Swift.max(
                    timings.memoryFootprintPeakMegabytes, Double(reading.footprintBytes) / 1_048_576
                )
                var change = governor.degradeForMemory(
                    reading: reading,
                    currentSplatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    pixelCount: renderSize.pixelCount
                )
                // BUILD 318: THE FRAME CACHE GOES FIRST. The governor measures
                // this run's footprint as everything the process allocated
                // since the run began, which includes the per-run frame cache
                // (build 310, up to 420 MB), so the cache could have been paid
                // for in Gaussians. It is dropped, and stays off, before any
                // cut is considered; the reading is then retaken.
                if change != nil, supervision.frameCacheBytes > 0
                    || coarseSupervisions.contains(where: { $0.frameCacheBytes > 0 })
                {
                    // BUILD 336: EVERY builder's cache, and no worker may be
                    // inside any builder while its cache is emptied. The full
                    // builder's background preload (started when the last
                    // level begins, iteration 1,350 on 4,500) was writing the
                    // very dictionary this cleared from the loop thread: a
                    // Swift dictionary mutated from two threads is the crash
                    // build 334 hit just past that boundary, the first poll
                    // at which two whole caches were live at once.
                    prefetch.drain()
                    coarsePrefetches.forEach { $0.drain() }
                    preloads.forEach { $0.cancelAndJoin() }
                    supervision.dropFrameCache(disable: true)
                    coarseSupervisions.forEach { $0.dropFrameCache(disable: true) }
                    TrainerLog.budget.notice(
                        "Frame caches released for memory before any cut was considered"
                    )
                    reading = governor.measureMemory(resources: resources)
                    change = governor.degradeForMemory(
                        reading: reading,
                        currentSplatCount: splatCount,
                        shCoefficientCount: shCoefficientCount,
                        pixelCount: renderSize.pixelCount
                    )
                }
                if let change {
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
                    coarsePrefetches.forEach { $0.drain() }
                    preloads.forEach { $0.cancelAndJoin() }
                    try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                    try applyBudgetChange(
                        change,
                        resources: resources,
                        gpu: &gpu,
                        splatCount: &splatCount,
                        renderSize: &renderSize,
                        supervision: supervision,
                        coarseSupervisions: Array(zip(coarseSupervisions, levelScalesPerMille)),
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
            // The first iteration of the run: nothing has been prefetched, and
            // level 0's preload must hand the builder over before the loop
            // builds on it.
            if iteration == 0, activeLevel < levelCount { preloads[activeLevel].join() }
            switch activePrefetch.take(
                frame: frame, iteration: iteration, totalIterations: effectiveTotal
            ) {
            case .hit(let ready):
                // Built during the previous iteration's GPU wait. This is the
                // whole point, and on this branch the loop pays nothing for it.
                builtSupervision = ready
            case .miss:
                builtSupervision = activeSupervision.build(
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

            // START THE NEXT FRAME NOW, so it is built during the GPU wait
            // that `runIteration` is about to sit in rather than after it.
            // `orderCursor` has already moved on, so this is genuinely the
            // frame the next iteration will ask for, and `iteration + 1` is
            // the number it will ask with; both are checked on the way out, so
            // a wrong guess costs the work and nothing else.
            //
            // ABOVE THE SKIP GUARD, not below it. It used to sit after the
            // `guard let frameSupervision`, so a frame whose photo would not
            // decode took the `continue` straight past this and left no worker
            // running at all. The next iteration then missed its key and built
            // the whole frame inline on the main thread: about 38 ms of
            // supervision charged to one skipped frame, on top of the wasted
            // build. Nothing between here and the old position touches the
            // builder, and `iteration += 1` happens on the skip path too, so
            // the guess stays correct.
            // Build 324: a warm-up step that is about to apply the far field's
            // accumulated update completes BEFORE the next frame's supervision
            // is built, so that frame's copy of the field is the updated one,
            // exactly as when the step ran synchronously.
            if let running = pendingStep, running.warmup, running.iteration % 20 == 0 {
                try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
            }
            if !order.isEmpty {
                let nextFrame = slice.keyframes[order[orderCursor % order.count]]
                // The pair the NEXT iteration belongs to (build 328): the last
                // iteration of a level starts the next level's first frame.
                // That level's preload must be done with the builder first.
                let nextLevel = levelFor(iteration + 1)
                if nextLevel != activeLevel, nextLevel < preloads.count {
                    preloads[nextLevel].join()
                }
                supervisionFor(iteration + 1).1.start(
                    frame: nextFrame,
                    iteration: iteration + 1,
                    totalIterations: effectiveTotal
                )
            }

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
                try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                renderSize = frameSupervision.renderSize
                try resources.resizeRenderSize(to: renderSize)
                gpu = TrainerGPU(pipelines: pipelines, resources: resources)
                if !activeIsCoarse { fullRenderSize = renderSize }
            }
            resolutionScale = activeIsCoarse
                ? Float(levelScalesPerMille[activeLevel]) / 1000 : 1
            if activeIsCoarse { timings.coarseSteps += 1 }


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
                try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                let sweepFrom = CFAbsoluteTimeGetCurrent()
                defer { timings.filterSweep += CFAbsoluteTimeGetCurrent() - sweepFrom }
                // The full-size builder and grid whatever phase this is (build
                // 328): a sampling rate is a property of the camera the model
                // will be viewed through, and the sweep touches no pixel buffer.
                try updateFilter3D(
                    gpu: gpu,
                    resources: resources,
                    queue: queue,
                    keyframes: slice.keyframes,
                    supervision: supervision,
                    cameraDeltas: cameraDeltas,
                    splatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    renderSize: fullRenderSize
                )
            }

            // --- IS THIS RUN STILL GETTING BETTER? -----------------------------
            //
            // Immediately before a densify pass. The reset after that pass
            // does NOT protect the score, because the pass reads stats first.
            // What keeps held-out frames out of the AbsGS score is the
            // snapshot and restore of the stats buffer inside evaluateHeldOut.
            //
            // Scored WITH the per-frame exposure fit, because a held-out frame
            // has no fitted exposure of its own and the raw number therefore
            // moves when the capture's auto-exposure drifted. The stopping
            // decision must not turn on that.
            if evalEvery > 0,
               iteration > 0,
               iteration >= tuning.earlyStopMinIterations,
               iteration % evalEvery == 0,
               !slice.heldOutKeyframes.isEmpty
            {
                try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                let evalFrom = CFAbsoluteTimeGetCurrent()
                // Raw AND fitted from the same renders (alsoScoreExposureFitted
                // computes the fitted MSE with the same expression fitExposure
                // did, so `score` is bit-identical to before). Selection still
                // runs on the fitted score; the raw one is recorded beside it so
                // switching selection to raw (a device A/B) can be priced from
                // one run. See REFUTATION_LEDGER EXP-LS.
                // Build 334: the full-size builder's background preload (started
                // when the last level began) may still be inside the builder;
                // the evaluation builds on it from this thread, so wait first.
                preloads.last?.join()
                let rawScore = try evaluateHeldOut(
                    gpu: gpu,
                    resources: resources,
                    queue: queue,
                    frames: slice.heldOutKeyframes,
                    supervision: supervision,
                    cameraDeltas: cameraDeltas,
                    exposures: exposures,
                    splatCount: splatCount,
                    shCoefficientCount: shCoefficientCount,
                    renderSize: renderSize,
                    alsoScoreExposureFitted: true
                )
                timings.earlyStopEval += CFAbsoluteTimeGetCurrent() - evalFrom
                let fittedScore = lastHeldOutPSNRExposureFitted
                let score = fittedScore
                if let score, score.isFinite {
                    census.heldOutCurve.append(
                        TrainerHeldOutSample(
                            iteration: iterationsRunSoFar + iteration,
                            psnr: score,
                            splatCount: splatCount,
                            psnrRaw: rawScore,
                            psnrExposureFitted: fittedScore
                        )
                    )
                    // A DIP WHILE THE MODEL IS STILL BEING BUILT IS NOT
                    // OVERFITTING, and counting it as one is what stopped the
                    // first run of this feature at 5,500 iterations, two
                    // thousand after densification had finally been allowed to
                    // start. Its curve is the evidence: PSNR went 15.14, 14.71,
                    // 14.68 while the population sat at 148,000, then 16.19 the
                    // moment growth reached the cap. Every one of those
                    // "declines" was a model that had not been built yet.
                    //
                    // So patience only accrues once the population has settled.
                    // Growth changing the splat count by more than a per cent
                    // between two evaluations means the thing being scored is
                    // not the thing that will be shipped.
                    let previous = census.heldOutCurve.dropLast().last?.splatCount
                    let settled = previous.map { prior in
                        abs(splatCount - prior) <= Swift.max(prior / 100, 1)
                    } ?? false
                    if score > bestHeldOut + tuning.earlyStopMinImprovementDB {
                        bestHeldOut = score
                        bestHeldOutIteration = iteration
                        sinceBest = 0
                        bestCloud = readCloud(
                            resources: resources, count: splatCount, shDegree: shDegree
                        )
                    } else if settled {
                        sinceBest += 1
                        if sinceBest >= Swift.max(tuning.earlyStopPatienceEvals, 1) {
                            stoppedEarly = true
                            let best = String(format: "%.2f", bestHeldOut)
                            let now = String(format: "%.2f", score)
                            let note = "Stopping at iteration \(iteration): held-out PSNR peaked at \(best) dB near iteration \(bestHeldOutIteration), is \(now) dB now, and has not improved in \(sinceBest) checks."
                            TrainerLog.general.info("\(note, privacy: .public)")
                            break iterationLoop
                        }
                    }
                }
            }

            // The fraction OR the absolute ceiling, whichever comes first.
            // See `densifyStartMaxIterations`: a fraction that is right at
            // 3,000 iterations delays growth to iteration 3,000 at 30,000, and
            // the held-out curve showed the run optimising a half-size
            // population until it did.
            // Integer arithmetic, so there is no float-to-int conversion to
            // trap on. The fraction is clamped to 0...1 first, and NaN takes
            // the else branch because every comparison against NaN is false,
            // so the product can never exceed `effectiveTotal`.
            let startPerMille: Int = tuning.densifyStartFraction.isFinite
                ? Int(Swift.min(Swift.max(tuning.densifyStartFraction, 0), 1) * 1000)
                : 100
            let fractionStart = effectiveTotal * startPerMille / 1000
            let densifyStartsAt = tuning.densifyStartMaxIterations > 0
                ? Swift.min(fractionStart, tuning.densifyStartMaxIterations)
                : fractionStart
            let inDensifyWindow = iteration >= densifyStartsAt
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
                try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
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
                    carver: carveDue ? smart.carver : nil,
                    // Build 320: the bulk arrays stay on the GPU. The first
                    // pass of the run also runs the CPU path and compares; a
                    // difference switches the run back to the CPU path.
                    gather: densifyGatherUsable
                        ? TrainerDensifyGather(gpu: gpu, queue: queue) : nil,
                    checkGather: densifyPassesRun == 0
                )
                splatCount = outcome.splatCountAfter
                if let mismatches = outcome.gatherMismatches {
                    timings.densifyGatherChecks += 1
                    timings.densifyGatherMismatches += mismatches
                    if mismatches > 0 { densifyGatherUsable = false }
                }

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
                if let running = pendingStep, running.merged {
                    // Build 322: the next merged step copies the model in its
                    // own command buffer and the copy is converted off the
                    // loop (startSnapshotConversion). Nothing waits here.
                    lock.lock()
                    snapshotRequested = true
                    snapshotParts = completedParts
                    snapshotDegree = shDegree
                    lock.unlock()
                } else {
                    try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)
                    let snapshotFrom = CFAbsoluteTimeGetCurrent()
                    let cloud = readCloud(
                        resources: resources, count: splatCount, shDegree: shDegree
                    )
                    lock.lock()
                    latestSnapshot = mergePreview(completedParts: completedParts, current: cloud)
                    lock.unlock()
                    timings.previewSnapshot += CFAbsoluteTimeGetCurrent() - snapshotFrom
                }
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
        // Nothing after the loop may see a step still running (build 292).
        try drainPendingStep(resources: resources, lossEMA: &lossEMA, exposures: &exposures, cameraDeltas: &cameraDeltas)

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

            preloads.last?.join()
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
                renderSize: renderSize,
                alsoScoreExposureFitted: true
            )
            census.slices[censusRow].heldOutPSNR = psnr
            census.slices[censusRow].heldOutSSIM = lastHeldOutSSIM
            census.slices[censusRow].heldOutPerFrame = lastHeldOutPerFrame
            census.slices[censusRow].stoppedEarly = stoppedEarly
            census.slices[censusRow].bestHeldOutPSNR =
                bestHeldOut.isFinite ? bestHeldOut : nil
            census.slices[censusRow].bestHeldOutIteration =
                bestHeldOut.isFinite ? bestHeldOutIteration : nil

            // The same frames, scored after a two-scalar photometric alignment
            // fitted to each one. Reported BESIDE the raw number, never
            // instead of it: the raw number is what the model actually
            // produces, and this one says how much of the gap to the trained
            // views was ever about geometry.
            // Scored inside the call above from the SAME renders; a second
            // evaluation re-rendered twelve identical frames for two scalars.
            census.slices[censusRow].heldOutPSNRExposureFitted = lastHeldOutPSNRExposureFitted

            // BUILD 332: THE SAME FRAMES WITH THEIR CAMERAS ALIGNED. The
            // trained frames' poses are refined during the run; the held-out
            // frames keep the pre-pass pose, so part of the raw gap is
            // registration and not the model. Each held-out camera takes a
            // few clamped gradient steps against the finished model (the
            // model does not move: the only thing written is the camera's
            // own correction), and the frames are scored again. Reported
            // beside the raw and exposure-fitted numbers, never instead.
            do {
                let aligned = try alignHeldOutPoses(
                    gpu: gpu, resources: resources, queue: queue,
                    frames: slice.heldOutKeyframes, supervision: supervision,
                    cameraDeltas: cameraDeltas, splatCount: splatCount,
                    shCoefficientCount: shCoefficientCount, renderSize: renderSize
                )
                let alignedScore = try evaluateHeldOut(
                    gpu: gpu, resources: resources, queue: queue,
                    frames: slice.heldOutKeyframes, supervision: supervision,
                    cameraDeltas: aligned, exposures: exposures,
                    splatCount: splatCount, shCoefficientCount: shCoefficientCount,
                    renderSize: renderSize, alsoScoreExposureFitted: true
                )
                // Recorded only when something was scored (build 336): a
                // second pass that skipped every frame would otherwise report
                // the raw numbers as the aligned ones.
                if alignedScore != nil {
                    census.slices[censusRow].heldOutPSNRPoseAligned = lastHeldOutPSNRExposureFitted
                    census.slices[censusRow].heldOutSSIMPoseAligned = lastHeldOutSSIM
                }
            } catch {
                TrainerLog.general.error(
                    "The pose-aligned held-out score could not be taken: \(error.localizedDescription, privacy: .public)"
                )
            }

            // Camera deltas, read-only: size per frame and the common
            // (world-frame mean) component. view' = D * view, so the centre
            // moves by -R^T R_D^T t_D; the turn is R^T times the delta's axis
            // (overall sign irrelevant to the magnitude of the mean).
            let deltaFrames = Dictionary(
                slice.keyframes.map { ($0.index, $0) }, uniquingKeysWith: { first, _ in first }
            )
            var deltaDegrees: [Float] = []
            var deltaCentimetres: [Float] = []
            var worldShift = SIMD3<Float>.zero
            var worldTurn = SIMD3<Float>.zero
            for (index, delta) in cameraDeltas {
                guard let frame = deltaFrames[index] else { continue }
                let dq = delta.rotation.simd.normalized
                let halfAngle: Float = acos(Swift.min(Swift.abs(dq.real), 1))
                deltaDegrees.append(2 * halfAngle * 180 / Float.pi)
                deltaCentimetres.append(simd_length(delta.translation.simd) * 100)
                let base = supervision.pose(for: frame).rotation.simd
                let shift: SIMD3<Float> = base.inverse.act(dq.inverse.act(delta.translation.simd))
                worldShift -= shift
                let axis: SIMD3<Float> = dq.real < 0 ? -dq.imag : dq.imag
                worldTurn += base.inverse.act(axis * 2)
            }
            if !deltaDegrees.isEmpty {
                deltaDegrees.sort()
                deltaCentimetres.sort()
                let count = Float(deltaDegrees.count)
                let row = censusRow
                census.slices[row].cameraDeltaFrames = deltaDegrees.count
                census.slices[row].cameraDeltaMedianDegrees = deltaDegrees[deltaDegrees.count / 2]
                census.slices[row].cameraDeltaMaxDegrees = deltaDegrees[deltaDegrees.count - 1]
                census.slices[row].cameraDeltaMedianCentimeters =
                    deltaCentimetres[deltaCentimetres.count / 2]
                census.slices[row].cameraDeltaMaxCentimeters =
                    deltaCentimetres[deltaCentimetres.count - 1]
                census.slices[row].cameraDeltaCommonCentimeters = simd_length(worldShift / count) * 100
                census.slices[row].cameraDeltaCommonDegrees =
                    simd_length(worldTurn / count) * 180 / Float.pi
            }

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
                // WAS `min`, which reported the WORST slice as the model's
                // PSNR. A number that tracks the weakest part of the scene
                // moves when that part moves and is flat otherwise, which is
                // not a measurement of the change being tested. Mean across
                // slices is what every 3DGS paper reports.
                heldOutPSNRSum += Double(psnr)
                heldOutPSNRCount += 1
                heldOutPSNR = Float(heldOutPSNRSum / Double(heldOutPSNRCount))
                let formatted = String(format: "%.2f", psnr)
                TrainerLog.general.info(
                    "Held-out PSNR for this part: \(formatted, privacy: .public) dB"
                )
            }
        }

        // THE BEST MODEL, NOT THE LAST ONE. If any mid-run evaluation scored
        // better than where the run ended, that is the model to ship: it is
        // the same measurement, on the same held-out frames, with the same
        // exposure fit, and the run has already paid to find out.
        //
        // `readCloud` on the final state still runs when there is no better
        // checkpoint, which is every run where the score improved to the end
        // or where early stopping is switched off.
        // ...AND ONLY WHEN IT ACTUALLY BEAT WHERE THE RUN ENDED. Build 250
        // shipped its iteration-3,600 checkpoint (20.392 dB) over an end state
        // that scored 20.444 dB on the same frames with the same exposure fit,
        // because nothing compared the two. Both scores are exposure-fitted
        // held-out PSNR from evaluateHeldOut. The end state is also what
        // exposure.bin, the refined poses and heldOutSSIM describe, so
        // shipping it keeps the bundle consistent.
        let endScore = census.slices[censusRow].heldOutPSNRExposureFitted
        var shipBest = bestCloud != nil
        if shipBest, let endScore, endScore.isFinite, endScore >= bestHeldOut {
            shipBest = false
        }
        let cloud = (shipBest ? bestCloud : nil) ?? readCloud(
            resources: resources, count: splatCount, shDegree: shDegree
        )
        if shipBest {
            let note = String(
                format: "Exporting the model from iteration %d, which scored %.2f dB, rather than the one this run ended on.",
                bestHeldOutIteration, bestHeldOut
            )
            TrainerLog.general.info("\(note, privacy: .public)")
        }

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
    /// True on the iterations whose step is split into timed stages or runs a
    /// calibration (builds 286 to 290). Those wait on every buffer they commit
    /// and are never overlapped.
    private func stepIsSplit(_ iteration: Int, gpu: TrainerGPU) -> Bool {
        let every = tuning.stageProfileEvery
        if every > 0, iteration % every == every / 2 { return true }
        let backwardStart = tuning.backwardCalibrationStart
        if gpu.pipelines.rasterizeBackwardSimdSum != nil,
           tuning.backwardCalibrationSteps > 0,
           iteration >= backwardStart,
           iteration < backwardStart + tuning.backwardCalibrationSteps {
            return true
        }
        let sortStart = tuning.sortCalibrationStart
        if tuning.sortCalibrationSteps > 0,
           iteration >= sortStart,
           iteration < sortStart + tuning.sortCalibrationSteps {
            return true
        }
        let forwardStart = tuning.forwardCalibrationStart
        if gpu.pipelines.rasterizeForward2 != nil,
           tuning.forwardCalibrationSteps > 0,
           iteration >= forwardStart,
           iteration < forwardStart + tuning.forwardCalibrationSteps {
            return true
        }
        let backward2Start = tuning.backwardTwoPixelCalibrationStart
        if gpu.pipelines.rasterizeBackward2 != nil,
           tuning.backwardTwoPixelCalibrationSteps > 0,
           iteration >= backward2Start,
           iteration < backward2Start + tuning.backwardTwoPixelCalibrationSteps {
            return true
        }
        let blurStart = tuning.blurCalibrationStart
        if gpu.pipelines.blurHV != nil,
           tuning.blurCalibrationSteps > 0,
           iteration >= blurStart,
           iteration < blurStart + tuning.blurCalibrationSteps {
            return true
        }
        return false
    }

    /// Completes the overlapped step, if there is one: waits for its buffer B,
    /// then applies its read-backs from the staging slot, with the same
    /// arithmetic in the same order as runIteration's synchronous path applies
    /// them from the live buffers. KEEP THE TWO IN STEP.
    ///
    /// MUST run before anything reads or reshapes GPU state between
    /// iterations: densify, the held-out evaluation, the 3D-filter sweep, the
    /// preview snapshot, a budget change, a render-size change, and the end of
    /// the slice. Each of those calls it first, and runIteration calls it
    /// before any step it will not overlap.
    private func drainPendingStep(
        resources: TrainerResources,
        lossEMA: inout Float?,
        exposures: inout [FrameID: SIMD2<Float>],
        cameraDeltas: inout [FrameID: Pose]
    ) throws {
        guard let pending = pendingStep else { return }
        pendingStep = nil
        try completeStep(
            pending, resources: resources, lossEMA: &lossEMA,
            exposures: &exposures, cameraDeltas: &cameraDeltas
        )
    }

    /// Waits for one left-running step and applies its read-backs. Split out
    /// of drainPendingStep (build 316) so a merged iteration can complete the
    /// PREVIOUS step after it has committed its own.
    private func completeStep(
        _ pending: PendingStep,
        resources: TrainerResources,
        lossEMA: inout Float?,
        exposures: inout [FrameID: SIMD2<Float>],
        cameraDeltas: inout [FrameID: Pose]
    ) throws {
        try finish(pending.buffer, "the training step")
        let staged = resources.readbackStaging.readArray(Float.self, count: 32)
        guard staged.count == 32 else { return }
        let base = pending.slot * 16

        let lossValue = staged[base]
        if lossValue.isFinite {
            lossEMA = lossEMA.map { $0 * 0.98 + lossValue * 0.02 } ?? lossValue
        }

        let gainGradient = staged[base + 4]
        let biasGradient = staged[base + 5]
        if gainGradient.isFinite, biasGradient.isFinite {
            var gain = pending.exposure.x - tuning.exposureLearningRate * gainGradient
            var bias = pending.exposure.y - tuning.exposureLearningRate * biasGradient
            gain = TrainerMath.clamp(
                gain, tuning.exposureGainRange.lowerBound, tuning.exposureGainRange.upperBound
            )
            bias = TrainerMath.clamp(
                bias, tuning.exposureBiasRange.lowerBound, tuning.exposureBiasRange.upperBound
            )
            exposures[pending.frame.index] = SIMD2<Float>(gain, bias)
        }

        let warmupEnd = Int(Float(pending.totalIterations) * tuning.warmupFraction)
        if pending.iteration > warmupEnd {
            let gradient = Array(staged[(base + 8)..<(base + 14)])
            if gradient.allSatisfy({ $0.isFinite }) {
                cameraDeltas[pending.frame.index] = updatedCameraDelta(
                    current: cameraDeltas[pending.frame.index], gradient: gradient
                )
            }
        }

        // Build 324: the far field's update, from the step's staged copy of
        // gradFinal and renderTFinal, the arithmetic runIteration applies
        // from the live buffers, applied every twentieth iteration as there.
        if pending.warmup, let background = pending.background,
           let supervision = pending.supervision, pending.iteration <= warmupEnd {
            let size = resources.renderSize
            let slotBase = pending.slot * TrainerResources.warmupStagingSlotBytes(pixelCount: size.pixelCount)
            accumulateBackgroundGradient(
                background: background,
                gradFinal: resources.warmupStaging, gradOffset: slotBase,
                tFinal: resources.warmupStaging, tOffset: slotBase + size.pixelCount * 12,
                size: size, supervision: supervision
            )
            if pending.iteration % 20 == 0 {
                background.applyAccumulatedGradient(learningRate: 0.25)
            }
        }

        // Build 316: the instance count the setup kernel found, one step late.
        // `needed` is what the frame produced, `used` what the sort was given;
        // they differ only when the buffers were too small, and then this step
        // trained with some splats missing tiles. Counted, and the buffers grow
        // before the next merged step (runMergedIteration).
        if pending.merged {
            let words = resources.readbackStaging.readArray(UInt32.self, count: 32)
            if words.count == 32 {
                let needed = Int(words[base + 14])
                let used = Int(words[base + 15])
                if needed > peakTileInstances {
                    peakTileInstances = needed
                    splatCountAtPeakTileInstances = pending.splatCount
                }
                if needed > used {
                    timings.truncatedInstanceSteps += 1
                    pendingInstanceGrowth = Swift.max(pendingInstanceGrowth ?? 0, needed)
                }
            }
        }
        if pending.snapshotCount > 0 {
            timings.snapshotsStaged += 1
            startSnapshotConversion(resources: resources, count: pending.snapshotCount)
        }
    }

    /// Build 322: the staged copy becomes the preview on a background thread.
    /// The three arrays are read out of the staging HERE, on the loop thread,
    /// so a later step's copy can never overtake the read; only the
    /// conversion runs elsewhere. A conversion still running when the next
    /// snapshot lands makes that one wait for the following interval.
    private func startSnapshotConversion(resources: TrainerResources, count: Int) {
        lock.lock()
        let busy = snapshotConverting
        let parts = snapshotParts
        let degree = snapshotDegree
        if !busy { snapshotConverting = true }
        lock.unlock()
        guard !busy else { return }
        let shPerSplat = resources.shFloatsPerSplat
        let staging = resources.snapshotStaging
        let splatBytes = count * MemoryLayout<TrainerSplat>.stride
        let shBytes = count * shPerSplat * MemoryLayout<Float>.stride
        let splats = staging.readArray(TrainerSplat.self, count: count, byteOffset: 0)
        let sh = staging.readArray(Float.self, count: count * shPerSplat, byteOffset: splatBytes)
        let stats = staging.readArray(
            TrainerSplatStats.self, count: count, byteOffset: splatBytes + shBytes
        )
        DispatchQueue.global(qos: .utility).async { [self] in
            let cloud = Self.buildCloud(
                splats: splats, sh: sh, shPerSplat: shPerSplat, stats: stats, shDegree: degree
            )
            let merged = mergePreview(completedParts: parts, current: cloud)
            lock.lock()
            latestSnapshot = merged
            snapshotConverting = false
            lock.unlock()
        }
    }

    /// This frame's supervision into the GPU's input slot, and the camera and
    /// loss uniforms for the step. Shared by the two-buffer and the merged
    /// paths (build 316); the arithmetic is the one runIteration carried.
    private func stageInputs(
        supervision: TrainerFrameSupervision,
        resources: TrainerResources,
        size: TrainerRenderSize,
        cameraDelta: Pose?,
        exposure: SIMD2<Float>,
        splatCount: Int,
        iteration: Int,
        totalIterations: Int,
        shCoefficientCount: Int,
        trust: TwoScaleTrustField?
    ) -> (camera: TrainerCameraUniforms, loss: TrainerLossUniforms, sampleCount: Int) {
        let pixelCount = size.pixelCount
        // --- Upload this frame's supervision -------------------------------------
        let uploadFrom = CFAbsoluteTimeGetCurrent()
        resources.gtColorIn.writeArray(supervision.groundTruthBytes)
        // The far field is 72 KB of cubemap now, not a 4.67 MB rasterised
        // image. `trainer_background` turns it into bgColor on the GPU in
        // command buffer A below.
        if supervision.hasBackground, !supervision.backgroundTexels.isEmpty {
            // writeArray clamps to the buffer, so a cubemap larger than the
            // buffer (a face size above TrainerGPUConstants.backgroundFaceSize)
            // would upload a torn map without a word. Said out loud once.
            let written = resources.bgCubemapIn.writeArray(supervision.backgroundTexels)
            if written < supervision.backgroundTexels.count, !loggedCubemapTruncation {
                loggedCubemapTruncation = true
                TrainerLog.gpu.error(
                    "The far-field cubemap has \(supervision.backgroundTexels.count) floats but the GPU buffer holds \(written); the far field is truncated"
                )
            }
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
            resources.depthSamplesIn.writeArray(supervision.depthSamples)
        }
        timings.upload += CFAbsoluteTimeGetCurrent() - uploadFrom

        // --- Uniforms --------------------------------------------------------------
        let camera = cameraUniforms(
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

        return (camera, loss, sampleCount)
    }

    /// BUILD 316: ONE COMMAND BUFFER PER STEP, WITH THE SORT SIZED ON THE GPU.
    ///
    /// The step was two command buffers with the CPU between them: A (clear,
    /// far field, preprocess, tile scan), a wait, a read of two integers to
    /// size the sort, then B (everything else). That wait and B's encode sat
    /// between the two on every iteration with the GPU idle, and the CPU could
    /// never be more than half a step ahead.
    ///
    /// Now `trainer_sort_setup` reads the two integers on the GPU and writes
    /// every count, uniform block and indirect dispatch argument the sort
    /// needs into `sortArgs`; the sort's dispatches take their threadgroup
    /// counts from that buffer and their uniforms from it too. The step is one
    /// command buffer, and this iteration commits its step BEFORE completing
    /// the previous one, so the GPU always has the next step queued.
    ///
    /// The one thing the CPU did with the count, growing the instance buffers
    /// and retrying the frame, is lagged: the kernel clamps the sort to the
    /// capacity (trainer_duplicate_keys already stops there), the count comes
    /// back with the step's other read-backs, and a step that needed more is
    /// counted (`truncatedInstanceSteps`) and grows the buffers before the
    /// step after next: the next step was already encoded when this one's
    /// count came back, so an overflow costs TWO clamped steps, and the
    /// census counts both. The buffers hold eight instances per splat and a
    /// room peaks near four, so this is recorded rather than expected.
    ///
    /// Only for steps that would have been overlapped anyway (after warm-up,
    /// background frozen, not a profiled or calibration step): those read
    /// nothing off the GPU between A and B except that count.
    private func runMergedIteration(
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
        frame: CaptureFrame,
        /// Build 324: non-nil for a warm-up step, whose gradFinal and
        /// renderTFinal are copied into warmupStaging for this model's update.
        background: DirectionalBackgroundModel? = nil
    ) throws -> StepResult {
        let size = resources.renderSize

        // Growth asked for by a completed step. The step still running keeps
        // the buffers it was encoded with (a command buffer holds its
        // resources); only steps encoded from here on see the new ones.
        if let needed = pendingInstanceGrowth {
            pendingInstanceGrowth = nil
            if needed > resources.instanceCapacity {
                TrainerLog.gpu.notice(
                    "Tile instances needed \(needed), had \(resources.instanceCapacity); growing"
                )
                try resources.growInstanceCapacity(to: needed + needed / 4)
            }
        }
        // The splat-order sort ranks the splats in the instance buffers.
        if resources.instanceCapacity < splatCount {
            try resources.growInstanceCapacity(to: splatCount + splatCount / 4)
        }

        let previous = pendingStep
        pendingStep = nil
        resources.inputSlot = previous.map { 1 - $0.slot } ?? 0
        let slot = resources.inputSlot

        let inputs = stageInputs(
            supervision: supervision, resources: resources, size: size,
            cameraDelta: cameraDelta, exposure: exposure, splatCount: splatCount,
            iteration: iteration, totalIterations: totalIterations,
            shCoefficientCount: shCoefficientCount, trust: trust
        )
        var camera = inputs.camera
        var loss = inputs.loss
        let sampleCount = inputs.sampleCount

        // Build 322: a requested preview snapshot rides in this step's buffer,
        // copied BEFORE the step's Adam touches anything: the state after the
        // previous step, which is what the synchronous snapshot read.
        var snapshotCount = 0
        lock.lock()
        let snapshotWanted = snapshotRequested && !snapshotConverting
        lock.unlock()
        let snapshotBytes = TrainerResources.snapshotStagingBytes(
            splats: splatCount, shFloats: splatCount * resources.shFloatsPerSplat
        )
        if snapshotWanted, resources.snapshotStaging.length >= snapshotBytes {
            lock.lock()
            snapshotRequested = false
            lock.unlock()
            snapshotCount = splatCount
        }

        let encodeFrom = CFAbsoluteTimeGetCurrent()
        // Pooled for the reason runIteration's buffers are; the committed
        // buffer itself is kept, as build 292 kept buffer B.
        let committed: MTLCommandBuffer = try autoreleasepool { () throws -> MTLCommandBuffer in
            guard let buffer = queue.makeCommandBuffer(),
                  let blit = buffer.makeBlitCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            if snapshotCount > 0 {
                let splatBytes = snapshotCount * MemoryLayout<TrainerSplat>.stride
                let shBytes = snapshotCount * resources.shFloatsPerSplat * MemoryLayout<Float>.stride
                let statsBytes = snapshotCount * MemoryLayout<TrainerSplatStats>.stride
                blit.copy(from: resources.splats, sourceOffset: 0,
                          to: resources.snapshotStaging, destinationOffset: 0, size: splatBytes)
                blit.copy(from: resources.sh, sourceOffset: 0,
                          to: resources.snapshotStaging, destinationOffset: splatBytes, size: shBytes)
                blit.copy(from: resources.stats, sourceOffset: 0,
                          to: resources.snapshotStaging, destinationOffset: splatBytes + shBytes,
                          size: statsBytes)
            }
            blit.label = "trainer.clear"
            gpu.clearPerIteration(blit, splatCount: splatCount)
            blit.endEncoding()

            guard let front = buffer.makeComputeCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            front.label = "trainer.preprocess"
            if supervision.hasBackground, supervision.backgroundFaceSize > 0 {
                let q = supervision.pose.rotation.simd.inverse
                var bg = TrainerBackgroundUniforms(
                    rotationInverse: SIMD4<Float>(q.imag.x, q.imag.y, q.imag.z, q.real),
                    fx: supervision.intrinsics.fx,
                    fy: supervision.intrinsics.fy,
                    cx: supervision.intrinsics.cx,
                    cy: supervision.intrinsics.cy,
                    width: UInt32(size.width),
                    height: UInt32(size.height),
                    faceSize: UInt32(supervision.backgroundFaceSize),
                    pad: 0
                )
                gpu.background(front, uniforms: &bg)
            }
            gpu.preprocess(front, camera: &camera, splatCount: splatCount)
            if splatOrderChosen {
                gpu.orderSplats(
                    front, camera: &camera, splatCount: splatCount,
                    simdScan: sortSimdScanChosen
                )
            } else {
                gpu.exclusiveScan(
                    front,
                    input: resources.tilesTouched,
                    output: resources.offsets,
                    count: splatCount
                )
            }
            gpu.sortSetup(front, camera: &camera, ordered: splatOrderChosen)
            // Its own encoder ends here, so the arguments the setup kernel
            // wrote are complete before the first dispatch that reads them.
            front.endEncoding()

            guard let step = buffer.makeComputeCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            step.label = "trainer.step"
            gpu.duplicateKeys(
                step, camera: &camera, splatCount: splatCount, ordered: splatOrderChosen
            )
            gpu.radixSortIndirect(step, simdScan: sortSimdScanChosen, tileOnly: splatOrderChosen)
            gpu.tileRangesIndirect(step)
            gpu.rasterizeForward(step, camera: &camera, twoPixels: forwardTwoPixelChosen)

            gpu.lossPhotometric(step, loss: &loss)
            let partials = gpu.ssim(step, loss: &loss, fused: blurFusedChosen)
            gpu.lossFinalize(step, loss: &loss, blurredPartials: partials)
            gpu.lossDepth(step, loss: &loss, sampleCount: sampleCount)

            gpu.rasterizeBackward(
                step, camera: &camera, loss: &loss, simdSum: backwardSimdSumChosen,
                twoPixels: backwardTwoPixelChosen
            )
            gpu.preprocessBackward(step, camera: &camera, splatCount: splatCount)

            var reg = regularizerUniforms(
                splatCount: splatCount, iteration: iteration, totalIterations: totalIterations
            )
            gpu.regularizer(step, reg: &reg)
            var adam = adamUniforms(
                splatCount: splatCount,
                shCoefficientCount: shCoefficientCount,
                iteration: iteration,
                totalIterations: totalIterations,
                sceneExtent: sceneExtent
            )
            gpu.adamSplat(step, adam: &adam)
            gpu.adamSH(step, adam: &adam)
            step.endEncoding()

            // The read-backs into this step's staging slot, after everything
            // that writes them: loss, exposure and camera gradients as in
            // build 292, then the instance count (needed, used).
            guard let copy = buffer.makeBlitCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            let base = slot * 64
            copy.copy(from: resources.lossAccum, sourceOffset: 0,
                      to: resources.readbackStaging, destinationOffset: base, size: 4)
            copy.copy(from: resources.exposureGrad, sourceOffset: 0,
                      to: resources.readbackStaging, destinationOffset: base + 16, size: 8)
            copy.copy(from: resources.cameraGrad, sourceOffset: 0,
                      to: resources.readbackStaging, destinationOffset: base + 32, size: 24)
            copy.copy(from: resources.sortArgs, sourceOffset: 0,
                      to: resources.readbackStaging, destinationOffset: base + 56, size: 8)
            if background != nil {
                // Build 324: the far field's gradient planes, for completeStep.
                let px = size.pixelCount
                let warmBase = slot * TrainerResources.warmupStagingSlotBytes(pixelCount: px)
                copy.copy(from: resources.gradFinal, sourceOffset: 0,
                          to: resources.warmupStaging, destinationOffset: warmBase, size: px * 12)
                copy.copy(from: resources.renderTFinal, sourceOffset: 0,
                          to: resources.warmupStaging, destinationOffset: warmBase + px * 12,
                          size: px * 4)
            }
            copy.endEncoding()
            buffer.commit()
            return buffer
        }
        timings.encodeStep += CFAbsoluteTimeGetCurrent() - encodeFrom
        pendingStep = PendingStep(
            buffer: committed, slot: slot, frame: frame,
            exposure: exposure, iteration: iteration,
            totalIterations: totalIterations, splatCount: splatCount, merged: true,
            snapshotCount: snapshotCount,
            warmup: background != nil,
            supervision: background != nil ? supervision : nil,
            background: background
        )
        timings.overlappedSteps += 1
        timings.mergedSteps += 1
        if background != nil { timings.warmupOverlappedSteps += 1 }

        // The previous step is queued in front of this one; complete it now.
        // Its read-backs land one iteration late, as build 292's did, and its
        // input slot is the one the NEXT step takes.
        if let previous {
            try completeStep(
                previous, resources: resources, lossEMA: &lossEMA,
                exposures: &exposures, cameraDeltas: &cameraDeltas
            )
        }
        return .stepped
    }

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

        // OVERLAPPED ITERATIONS (build 292). After warm-up, with the
        // background frozen, a normal step's buffer B is committed and LEFT
        // RUNNING: the next iteration's CPU work (supervision, upload,
        // uniforms) and its buffer A go ahead while B executes, and B is
        // completed just after that A is committed (see drainPendingStep).
        // What makes it safe:
        //  - the model parameters are only touched by the GPU, in queue order,
        //    so A(n+1) sees B(n)'s Adam step exactly as before;
        //  - the three buffers the CPU writes inputs into are doubled and the
        //    next step takes the slot the running step is NOT using;
        //  - B copies its read-backs (loss, exposure and camera gradients) into
        //    a staging slot, because A(n+1) clears the originals;
        //  - exposure and camera updates are per frame and applied before that
        //    frame is next visited, and the loss EMA only feeds progress, so
        //    applying them one iteration later changes nothing that trains;
        //  - everything that reads GPU state between iterations drains first.
        // Warm-up reads the background gradient off the GPU every iteration,
        // and the split and calibration steps wait on every buffer, so none of
        // those are overlapped.
        let overlapWarmupEnd = Int(Float(totalIterations) * tuning.warmupFraction)
        let backgroundStillLearning = background.map { !$0.isFrozen } ?? false
        // Build 324: warm-up steps are overlapped too, as merged steps that
        // stage the far field's gradient for completeStep. The one iteration
        // that freezes the field (the first past warm-up) still runs
        // synchronously below, after draining the last warm-up step, so the
        // field's last update lands before the freeze as it always did.
        let warmupOverlap = tuning.overlapWarmup && tuning.mergedCommandBuffer
            && iteration <= overlapWarmupEnd
        let deferCompletion = tuning.overlapIterations
            && !stepIsSplit(iteration, gpu: gpu)
            && ((iteration > overlapWarmupEnd && !backgroundStillLearning) || warmupOverlap)
        if deferCompletion, tuning.mergedCommandBuffer {
            return try runMergedIteration(
                gpu: gpu, resources: resources, queue: queue, supervision: supervision,
                cameraDelta: cameraDelta, exposure: exposure, splatCount: splatCount,
                iteration: iteration, totalIterations: totalIterations,
                sceneExtent: sceneExtent, shCoefficientCount: shCoefficientCount,
                trust: trust, lossEMA: &lossEMA, exposures: &exposures,
                cameraDeltas: &cameraDeltas, frame: frame,
                background: warmupOverlap ? background : nil
            )
        }
        if !deferCompletion {
            try drainPendingStep(
                resources: resources, lossEMA: &lossEMA,
                exposures: &exposures, cameraDeltas: &cameraDeltas
            )
        }
        resources.inputSlot = pendingStep.map { 1 - $0.slot } ?? 0

        let inputs = stageInputs(
            supervision: supervision, resources: resources, size: size,
            cameraDelta: cameraDelta, exposure: exposure, splatCount: splatCount,
            iteration: iteration, totalIterations: totalIterations,
            shCoefficientCount: shCoefficientCount, trust: trust
        )
        var camera = inputs.camera
        var loss = inputs.loss
        let sampleCount = inputs.sampleCount

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
        // Build 306: the splat-order sort ranks the splats in the instance
        // sort's buffers, so those must hold at least one entry per splat.
        if resources.instanceCapacity < splatCount {
            try resources.growInstanceCapacity(to: splatCount + splatCount / 4)
        }
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
            // No visibility reset: nothing reads visibleFlag any more. The
            // per-step "drawn" predicate is tilesTouched, which
            // trainer_preprocess rewrites for every splat each iteration.

            // The far field, rasterised here instead of on the CPU. Runs
            // before anything reads bgColor, which is the loss in buffer B.
            if supervision.hasBackground, supervision.backgroundFaceSize > 0 {
                let q = supervision.pose.rotation.simd.inverse
                var bg = TrainerBackgroundUniforms(
                    rotationInverse: SIMD4<Float>(q.imag.x, q.imag.y, q.imag.z, q.real),
                    fx: supervision.intrinsics.fx,
                    fy: supervision.intrinsics.fy,
                    cx: supervision.intrinsics.cx,
                    cy: supervision.intrinsics.cy,
                    width: UInt32(size.width),
                    height: UInt32(size.height),
                    faceSize: UInt32(supervision.backgroundFaceSize),
                    pad: 0
                )
                gpu.background(encoderA, uniforms: &bg)
            }
            gpu.preprocess(encoderA, camera: &camera, splatCount: splatCount)
            if splatOrderChosen {
                gpu.orderSplats(
                    encoderA, camera: &camera, splatCount: splatCount,
                    simdScan: sortSimdScanChosen
                )
            } else {
                gpu.exclusiveScan(
                    encoderA,
                    input: resources.tilesTouched,
                    output: resources.offsets,
                    count: splatCount
                )
            }
            encoderA.endEncoding()
            bufferA.commit()
            // The previous step's buffer B ran while this iteration's CPU work
            // happened; this A is queued behind it. Complete it now, before
            // waiting on A. No-op when nothing is pending.
            try drainPendingStep(
                resources: resources, lossEMA: &lossEMA,
                exposures: &exposures, cameraDeltas: &cameraDeltas
            )
            try finish(bufferA, "the tile scan")
        }

        let encodeStepFrom = CFAbsoluteTimeGetCurrent()
        // The one unavoidable readback: how many (Gaussian, tile) pairs this
        // frame produced. The exclusive scan means the total is the last
        // offset plus the last count.
        let lastOffset = resources.offsets.readElement(UInt32.self, at: splatCount - 1) ?? 0
        let lastTouched = (splatOrderChosen ? resources.keysA : resources.tilesTouched).readElement(UInt32.self, at: splatCount - 1) ?? 0
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
            // SAMPLED STAGE PROFILE (build 286). The same kernels in the same
            // order, split into five command buffers so `finish` can time each
            // stage on the GPU. Buffers on one queue run in order and see each
            // other's writes, so the step computes exactly what the single
            // buffer below does; only ~1 ms of round trips is added, on 16
            // iterations a run. This is the breakdown every speed decision
            // from here on is read against.
            let profileEvery = tuning.stageProfileEvery
            let profiling = profileEvery > 0 && iteration % profileEvery == profileEvery / 2
            // BACKWARD CALIBRATION (build 288). On these iterations both backward
            // rasterisers run on the same inputs: A (the plain loop) into
            // splatGrad2D, read back, splatGrad2D cleared, then B (SIMD-summed),
            // read back. Their gradients are compared and both are timed. The
            // step then continues from B's gradients only if they agreed on this
            // iteration; otherwise A's are written back first, so a wrong B can
            // never train a single step. After the last calibration iteration B
            // is kept for the rest of the run only if it agreed every time and
            // was at least 3 % faster in total.
            let calibrationStart = tuning.backwardCalibrationStart
            let calibrating = gpu.pipelines.rasterizeBackwardSimdSum != nil
                && tuning.backwardCalibrationSteps > 0
                && iteration >= calibrationStart
                && iteration < calibrationStart + tuning.backwardCalibrationSteps
            // SORT CALIBRATION (build 290), the same pattern: the tile keys are
            // generated once, sorted by the plain scatter (A), restored, sorted
            // by the SIMD-prefix scatter (B), and both results must be IDENTICAL
            // (it is an integer ranking, so any difference is a bug). A mismatch
            // restores A's order before the step goes on. B is kept only if
            // every step matched and it was at least 3 % faster.
            let sortStart = tuning.sortCalibrationStart
            let sortCalibrating = tuning.sortCalibrationSteps > 0
                && iteration >= sortStart
                && iteration < sortStart + tuning.sortCalibrationSteps
            // FORWARD CALIBRATION (build 302): the frame is rendered by the
            // one-pixel kernel (A), read back, rendered by the two-pixel kernel
            // (B), read back and compared; a disagreement re-renders with A so
            // the step goes on from A's image. B is kept only if every step
            // agreed and it was at least 3 % faster.
            let forwardStart = tuning.forwardCalibrationStart
            let forwardCalibrating = gpu.pipelines.rasterizeForward2 != nil
                && tuning.forwardCalibrationSteps > 0
                && iteration >= forwardStart
                && iteration < forwardStart + tuning.forwardCalibrationSteps
            // TWO-PIXEL BACKWARD CALIBRATION (build 304): the same as the
            // backward calibration, with A the backward that one chose and B
            // the two-pixel kernel.
            let backward2Start = tuning.backwardTwoPixelCalibrationStart
            let backward2Calibrating = gpu.pipelines.rasterizeBackward2 != nil
                && tuning.backwardTwoPixelCalibrationSteps > 0
                && iteration >= backward2Start
                && iteration < backward2Start + tuning.backwardTwoPixelCalibrationSteps
            // BLUR CALIBRATION (build 318): the SSIM stage is run with the
            // two-pass blur (A), its blurred partials read back, then with the
            // fused blur (B) and compared BIT FOR BIT. A mismatch re-runs A so
            // the step goes on from A's planes. B is kept only if every step
            // matched and it was at least 3 % faster.
            let blurStart = tuning.blurCalibrationStart
            let blurCalibrating = gpu.pipelines.blurHV != nil
                && tuning.blurCalibrationSteps > 0
                && iteration >= blurStart
                && iteration < blurStart + tuning.blurCalibrationSteps
            if profiling || calibrating || sortCalibrating || forwardCalibrating
                || backward2Calibrating || blurCalibrating {
                // Calibration iterations are timed under their own label, so
                // they never skew the stage profile.
                func tag(_ label: String) -> String {
                    (calibrating || sortCalibrating || forwardCalibrating || backward2Calibrating
                     || blurCalibrating)
                        ? "the backward calibration" : label
                }
                @discardableResult
                func stage(
                    _ label: String, _ encode: (MTLComputeCommandEncoder) -> Void
                ) throws -> Double {
                    guard let buffer = queue.makeCommandBuffer(),
                          let encoder = buffer.makeComputeCommandEncoder()
                    else { throw TrainerError.noMetalDevice }
                    encoder.label = label
                    encode(encoder)
                    encoder.endEncoding()
                    buffer.commit()
                    try finish(buffer, label)
                    let executing = buffer.gpuEndTime - buffer.gpuStartTime
                    return executing.isFinite && executing > 0 ? executing : 0
                }
                if sortCalibrating {
                    // Build 306: three sorts of this frame's instances.
                    //   L  the legacy sort: offsets over splat index, 24-bit
                    //      keys, six passes, plain scatter. The reference.
                    //   S  the splat-order sort, plain scatter: splats ranked
                    //      by depth, instances emitted in that order, three
                    //      passes on the tile bits.
                    //   P  S with the SIMD-prefix scatter, where it exists.
                    // S and P must reproduce L exactly: every key, and the
                    // splat of every real entry. Padding entries all carry the
                    // same key and no tile range reaches them, so only their
                    // splat numbers may differ. A mismatch on the last sort run
                    // writes L's result back before the step goes on.
                    let n = instanceCount
                    let hasSimdScatter = gpu.pipelines.radixScatterSimdScan != nil
                    let secondsL = try stage(tag("the tile sort")) { e in
                        gpu.exclusiveScan(
                            e, input: resources.tilesTouched, output: resources.offsets,
                            count: splatCount
                        )
                        gpu.duplicateKeys(e, camera: &camera, splatCount: splatCount, ordered: false)
                        gpu.radixSort(e, count: n, simdScan: false, tileOnly: false)
                    }
                    let keysL = resources.keysA.readArray(UInt32.self, count: n)
                    let valuesL = resources.valuesA.readArray(UInt32.self, count: n)
                    let realCount = keysL.firstIndex { ($0 >> 12) == 0xFFF } ?? keysL.count
                    func matchesLegacy() -> Bool {
                        let keys = resources.keysA.readArray(UInt32.self, count: n)
                        let values = resources.valuesA.readArray(UInt32.self, count: n)
                        guard keysL.count == n, valuesL.count == n,
                              keys == keysL, values.count == n
                        else { return false }
                        return values[0..<realCount] == valuesL[0..<realCount]
                    }
                    let secondsS = try stage(tag("the tile sort")) { e in
                        gpu.orderSplats(e, camera: &camera, splatCount: splatCount, simdScan: false)
                        gpu.duplicateKeys(e, camera: &camera, splatCount: splatCount, ordered: true)
                        gpu.radixSort(e, count: n, simdScan: false, tileOnly: true)
                    }
                    let matchS = matchesLegacy()
                    var matchP = false
                    var secondsP = 0.0
                    var matchQ = false
                    var secondsQ = 0.0
                    if hasSimdScatter {
                        secondsP = try stage(tag("the tile sort")) { e in
                            gpu.orderSplats(
                                e, camera: &camera, splatCount: splatCount, simdScan: true
                            )
                            gpu.duplicateKeys(
                                e, camera: &camera, splatCount: splatCount, ordered: true
                            )
                            gpu.radixSort(e, count: n, simdScan: true, tileOnly: true)
                        }
                        matchP = matchesLegacy()
                        // Q: the legacy passes with the SIMD scatter (build 326,
                        // from review). This is the combination the run uses
                        // when the splat order is NOT chosen, and until now it
                        // was the one combination the window never checked.
                        secondsQ = try stage(tag("the tile sort")) { e in
                            gpu.exclusiveScan(
                                e, input: resources.tilesTouched, output: resources.offsets,
                                count: splatCount
                            )
                            gpu.duplicateKeys(e, camera: &camera, splatCount: splatCount, ordered: false)
                            gpu.radixSort(e, count: n, simdScan: true, tileOnly: false)
                        }
                        matchQ = matchesLegacy()
                    }
                    let lastMatched = hasSimdScatter ? matchQ : matchS
                    if !lastMatched, keysL.count == n, valuesL.count == n {
                        _ = resources.keysA.writeArray(keysL)
                        _ = resources.valuesA.writeArray(valuesL)
                    }
                    timings.sortCalibrationSteps += 1
                    timings.sortLegacySeconds += secondsL
                    timings.sortSecondsA += secondsS
                    timings.sortSecondsB += secondsP
                    timings.sortLegacySimdSeconds += secondsQ
                    if !matchS { timings.sortSplatOrderMismatchSteps += 1 }
                    if hasSimdScatter, !matchP { timings.sortMismatchSteps += 1 }
                    if hasSimdScatter, !matchQ { timings.sortLegacySimdMismatchSteps += 1 }
                    if iteration == sortStart + tuning.sortCalibrationSteps - 1 {
                        splatOrderChosen = tuning.splatOrderSort
                            && timings.sortSplatOrderMismatchSteps == 0
                            && timings.sortSecondsA < timings.sortLegacySeconds * 0.97
                        timings.sortSplatOrderChosen = splatOrderChosen ? 1 : 0
                        // The SIMD scatter is judged on the path it will run
                        // on: P against S when the splat order is chosen, Q
                        // against L when it is not.
                        if splatOrderChosen {
                            sortSimdScanChosen = hasSimdScatter
                                && timings.sortMismatchSteps == 0
                                && timings.sortSecondsB < timings.sortSecondsA * 0.97
                        } else {
                            sortSimdScanChosen = hasSimdScatter
                                && timings.sortLegacySimdMismatchSteps == 0
                                && timings.sortLegacySimdSeconds < timings.sortLegacySeconds * 0.97
                        }
                        timings.sortSimdScanChosen = sortSimdScanChosen ? 1 : 0
                    }
                    try stage(tag("the tile sort")) { e in
                        gpu.tileRanges(e, instanceCount: n)
                    }
                } else {
                    try stage(tag("the tile sort")) { e in
                        gpu.duplicateKeys(
                            e, camera: &camera, splatCount: splatCount, ordered: splatOrderChosen
                        )
                        gpu.radixSort(
                            e, count: instanceCount, simdScan: sortSimdScanChosen,
                            tileOnly: splatOrderChosen
                        )
                        gpu.tileRanges(e, instanceCount: instanceCount)
                    }
                }
                if forwardCalibrating {
                    let pixels = resources.renderSize.pixelCount
                    let secondsA = try stage(tag("the forward raster")) { e in
                        gpu.rasterizeForward(e, camera: &camera, twoPixels: false)
                    }
                    let colorA = resources.renderColor.readArray(Float.self, count: pixels * 3)
                    let alphaA = resources.renderAlpha.readArray(Float.self, count: pixels)
                    let depthA = resources.renderDepth.readArray(Float.self, count: pixels)
                    let transA = resources.renderTFinal.readArray(Float.self, count: pixels)
                    let countA = resources.renderNContrib.readArray(UInt32.self, count: pixels)
                    let secondsB = try stage(tag("the forward raster")) { e in
                        gpu.rasterizeForward(e, camera: &camera, twoPixels: true)
                    }
                    let colorB = resources.renderColor.readArray(Float.self, count: pixels * 3)
                    let alphaB = resources.renderAlpha.readArray(Float.self, count: pixels)
                    let depthB = resources.renderDepth.readArray(Float.self, count: pixels)
                    let transB = resources.renderTFinal.readArray(Float.self, count: pixels)
                    let countB = resources.renderNContrib.readArray(UInt32.self, count: pixels)
                    var maxDifference: Float = 0
                    func compare(_ a: [Float], _ b: [Float]) {
                        guard a.count == b.count, !a.isEmpty else {
                            maxDifference = .infinity
                            return
                        }
                        for i in 0..<a.count {
                            let d = abs(a[i] - b[i])
                            if d.isNaN {
                                maxDifference = .infinity
                            } else if d > maxDifference {
                                maxDifference = d
                            }
                        }
                    }
                    compare(colorA, colorB)
                    compare(alphaA, alphaB)
                    compare(depthA, depthB)
                    compare(transA, transB)
                    var countMismatch = 0
                    if countA.count == pixels, countB.count == pixels {
                        for i in 0..<pixels where countA[i] != countB[i] { countMismatch += 1 }
                    } else {
                        countMismatch = pixels
                    }
                    let agreed = maxDifference <= 1e-4 && countMismatch * 10_000 <= pixels
                    if !agreed {
                        // The step goes on from A's image.
                        try stage(tag("the forward raster")) { e in
                            gpu.rasterizeForward(e, camera: &camera, twoPixels: false)
                        }
                        timings.forwardMismatchSteps += 1
                    }
                    timings.forwardCalibrationSteps += 1
                    timings.forwardSecondsA += secondsA
                    timings.forwardSecondsB += secondsB
                    timings.forwardMaxDifference = Swift.max(
                        timings.forwardMaxDifference,
                        maxDifference.isFinite ? Double(maxDifference) : 1e9
                    )
                    if iteration == forwardStart + tuning.forwardCalibrationSteps - 1 {
                        forwardTwoPixelChosen = timings.forwardMismatchSteps == 0
                            && timings.forwardSecondsB < timings.forwardSecondsA * 0.97
                        timings.forwardTwoPixelChosen = forwardTwoPixelChosen ? 1 : 0
                    }
                } else {
                    try stage(tag("the forward raster")) { e in
                        gpu.rasterizeForward(e, camera: &camera, twoPixels: forwardTwoPixelChosen)
                    }
                }
                if blurCalibrating {
                    let pixels = resources.renderSize.pixelCount
                    try stage(tag("the losses")) { e in
                        gpu.lossPhotometric(e, loss: &loss)
                    }
                    // ssimStats adds the SSIM loss into lossAccum, so the
                    // total is put back between runs and the step's loss
                    // read-back sees one contribution.
                    let lossBefore = resources.lossAccum.readElement(Float.self, at: 0) ?? 0
                    let secondsA = try stage(tag("the losses")) { e in
                        gpu.ssim(e, loss: &loss, fused: false)
                    }
                    let planesA = resources.ssimTmp.readArray(Float.self, count: pixels * 3)
                    _ = resources.lossAccum.writeArray([lossBefore])
                    let secondsB = try stage(tag("the losses")) { e in
                        gpu.ssim(e, loss: &loss, fused: true)
                    }
                    let planesB = resources.ssimMid.readArray(Float.self, count: pixels * 3)
                    // Bit for bit: the bit patterns, so a NaN in the same
                    // place counts as a match and -0 against +0 does not.
                    let identical = planesA.count == pixels * 3 && planesB.count == pixels * 3
                        && zip(planesA, planesB).allSatisfy { $0.bitPattern == $1.bitPattern }
                    var partials: MTLBuffer = resources.ssimMid
                    if !identical {
                        _ = resources.lossAccum.writeArray([lossBefore])
                        try stage(tag("the losses")) { e in
                            partials = gpu.ssim(e, loss: &loss, fused: false)
                        }
                        timings.blurMismatchSteps += 1
                    }
                    timings.blurCalibrationSteps += 1
                    timings.blurSecondsA += secondsA
                    timings.blurSecondsB += secondsB
                    if iteration == blurStart + tuning.blurCalibrationSteps - 1 {
                        blurFusedChosen = timings.blurMismatchSteps == 0
                            && timings.blurSecondsB < timings.blurSecondsA * 0.97
                        timings.blurFusedChosen = blurFusedChosen ? 1 : 0
                    }
                    let chosenPartials = partials
                    try stage(tag("the losses")) { e in
                        gpu.lossFinalize(e, loss: &loss, blurredPartials: chosenPartials)
                        gpu.lossDepth(e, loss: &loss, sampleCount: sampleCount)
                    }
                } else {
                    try stage(tag("the losses")) { e in
                        gpu.lossPhotometric(e, loss: &loss)
                        let partials = gpu.ssim(e, loss: &loss, fused: blurFusedChosen)
                        gpu.lossFinalize(e, loss: &loss, blurredPartials: partials)
                        gpu.lossDepth(e, loss: &loss, sampleCount: sampleCount)
                    }
                }
                if calibrating || backward2Calibrating {
                    let floats = splatCount * 16
                    // Window 1: plain (A) against SIMD-summed (B). Window 2: the
                    // backward window 1 chose (A) against the two-pixel one (B).
                    let twoPixelWindow = !calibrating
                    let referenceSimdSum = twoPixelWindow && backwardSimdSumChosen
                    let secondsA = try stage(tag("the backward raster")) { e in
                        gpu.rasterizeBackward(
                            e, camera: &camera, loss: &loss, simdSum: referenceSimdSum
                        )
                    }
                    let gradientsA = resources.splatGrad2D.readArray(Float.self, count: floats)
                    guard let clearBuffer = queue.makeCommandBuffer(),
                          let clear = clearBuffer.makeBlitCommandEncoder()
                    else { throw TrainerError.noMetalDevice }
                    clear.fill(buffer: resources.splatGrad2D, range: 0..<(floats * 4), value: 0)
                    clear.endEncoding()
                    clearBuffer.commit()
                    try finish(clearBuffer, "the backward calibration")
                    let secondsB = try stage(tag("the backward raster")) { e in
                        gpu.rasterizeBackward(
                            e, camera: &camera, loss: &loss, simdSum: true,
                            twoPixels: twoPixelWindow
                        )
                    }
                    let gradientsB = resources.splatGrad2D.readArray(Float.self, count: floats)
                    var difference = 0.0
                    var magnitude = 0.0
                    let compared = Swift.min(gradientsA.count, gradientsB.count)
                    for i in 0..<compared {
                        let x = Double(gradientsA[i])
                        let y = Double(gradientsB[i])
                        if x.isFinite, y.isFinite {
                            difference += abs(x - y)
                            magnitude += abs(x)
                        } else if x.isFinite != y.isFinite {
                            difference = .infinity
                        }
                    }
                    var relative = magnitude > 0 ? difference / magnitude : (difference > 0 ? 1 : 0)
                    if !relative.isFinite || compared < floats { relative = 1e9 }
                    relative = Swift.min(relative, 1e9)
                    let agreedNow = relative < 1e-3
                    if !agreedNow, gradientsA.count == floats {
                        _ = resources.splatGrad2D.writeArray(gradientsA)
                    }
                    if twoPixelWindow {
                        timings.backwardTwoPixelCalibrationSteps += 1
                        timings.backwardTwoPixelSecondsA += secondsA
                        timings.backwardTwoPixelSecondsB += secondsB
                        timings.backwardTwoPixelRelativeDifference = Swift.max(
                            timings.backwardTwoPixelRelativeDifference, relative
                        )
                        let last = backward2Start + tuning.backwardTwoPixelCalibrationSteps - 1
                        if iteration == last {
                            let agrees = timings.backwardTwoPixelRelativeDifference < 1e-3
                            let faster = timings.backwardTwoPixelSecondsB
                                < timings.backwardTwoPixelSecondsA * 0.97
                            backwardTwoPixelChosen = agrees && faster
                            timings.backwardTwoPixelChosen = backwardTwoPixelChosen ? 1 : 0
                        }
                    } else {
                        timings.backwardCalibrationSteps += 1
                        timings.backwardSecondsA += secondsA
                        timings.backwardSecondsB += secondsB
                        timings.backwardRelativeDifference = Swift.max(
                            timings.backwardRelativeDifference, relative
                        )
                        if iteration == calibrationStart + tuning.backwardCalibrationSteps - 1 {
                            let agrees = timings.backwardRelativeDifference < 1e-3
                            let faster = timings.backwardSecondsB < timings.backwardSecondsA * 0.97
                            backwardSimdSumChosen = agrees && faster
                            timings.backwardSimdSumChosen = backwardSimdSumChosen ? 1 : 0
                        }
                    }
                    try stage(tag("the backward raster")) { e in
                        gpu.preprocessBackward(e, camera: &camera, splatCount: splatCount)
                    }
                } else {
                    try stage(tag("the backward raster")) { e in
                        gpu.rasterizeBackward(
                            e, camera: &camera, loss: &loss, simdSum: backwardSimdSumChosen,
                            twoPixels: backwardTwoPixelChosen
                        )
                        gpu.preprocessBackward(e, camera: &camera, splatCount: splatCount)
                    }
                }
                try stage(tag("the optimiser")) { e in
                    var reg = regularizerUniforms(
                        splatCount: splatCount, iteration: iteration,
                        totalIterations: totalIterations
                    )
                    gpu.regularizer(e, reg: &reg)
                    var adam = adamUniforms(
                        splatCount: splatCount,
                        shCoefficientCount: shCoefficientCount,
                        iteration: iteration,
                        totalIterations: totalIterations,
                        sceneExtent: sceneExtent
                    )
                    gpu.adamSplat(e, adam: &adam)
                    gpu.adamSH(e, adam: &adam)
                }
                // THE POSE CHECK (build 330), once per run, on the first
                // profiled step past warm-up and the coarse phase. The step
                // above left this frame's camera gradient in cameraGrad and
                // its Adam update in the model. The frame is rendered twice
                // more on that model, at its current correction and at the
                // correction plus one FULL-rate gradient step, and the two
                // photometric losses are compared. Descent turns the full
                // rate on for the rest of the run; anything else leaves the
                // rate where build 292 had it. The check's own dispatches
                // touch nothing the step reads back: lossAccum is put back,
                // the per-interval stats are put back (preprocess counts an
                // observation), and gradFinal, the render buffers and the
                // tile lists are rebuilt by the next step before anything
                // reads them.
                let anyCalibration = calibrating || sortCalibrating || forwardCalibrating
                    || backward2Calibrating || blurCalibrating
                let checkPerMille = tuning.warmupFraction.isFinite
                    ? Int(Swift.min(Swift.max(tuning.warmupFraction, 0), 1) * 1000)
                    : 1000
                let checkAfter = totalIterations * checkPerMille / 1000
                if profiling, !anyCalibration, !poseCheckDone, iteration > checkAfter {
                    poseCheckDone = true
                    let gradient = resources.cameraGrad.readArray(Float.self, count: 6)
                    let lossStep = resources.lossAccum.readElement(Float.self, at: 0) ?? 0
                    let statsBefore = resources.stats.readArray(TrainerSplatStats.self, count: splatCount)
                    if gradient.count == 6, gradient.allSatisfy({ $0.isFinite }),
                       gradient.contains(where: { $0 != 0 }) {
                        func photometricLoss(at delta: Pose?) throws -> Float {
                            var probe = cameraUniforms(
                                supervision: supervision,
                                cameraDelta: delta,
                                size: size,
                                splatCount: splatCount,
                                shCoefficientCount: shCoefficientCount,
                                iteration: iteration,
                                totalIterations: totalIterations
                            )
                            _ = resources.lossAccum.writeArray([Float(0)])
                            try stage("the pose check") { e in
                                gpu.preprocess(e, camera: &probe, splatCount: splatCount)
                                if splatOrderChosen {
                                    gpu.orderSplats(
                                        e, camera: &probe, splatCount: splatCount,
                                        simdScan: sortSimdScanChosen
                                    )
                                } else {
                                    gpu.exclusiveScan(
                                        e, input: resources.tilesTouched,
                                        output: resources.offsets, count: splatCount
                                    )
                                }
                                gpu.sortSetup(e, camera: &probe, ordered: splatOrderChosen)
                            }
                            try stage("the pose check") { e in
                                gpu.duplicateKeys(
                                    e, camera: &probe, splatCount: splatCount, ordered: splatOrderChosen
                                )
                                gpu.radixSortIndirect(
                                    e, simdScan: sortSimdScanChosen, tileOnly: splatOrderChosen
                                )
                                gpu.tileRangesIndirect(e)
                                gpu.rasterizeForward(e, camera: &probe, twoPixels: forwardTwoPixelChosen)
                                gpu.lossPhotometric(e, loss: &loss)
                                gpu.ssim(e, loss: &loss, fused: blurFusedChosen)
                            }
                            return resources.lossAccum.readElement(Float.self, at: 0) ?? .nan
                        }
                        let before = try photometricLoss(at: cameraDelta)
                        let stepped = updatedCameraDelta(
                            current: cameraDelta, gradient: gradient, rateScale: 1
                        )
                        let after = try photometricLoss(at: stepped)
                        let descends = before.isFinite && after.isFinite && after < before
                        timings.poseCheckLossBefore = Double(before.isFinite ? before : 1e9)
                        timings.poseCheckLossAfter = Double(after.isFinite ? after : 1e9)
                        timings.poseCheckDescends = descends ? 1 : 0
                        if descends { poseRefinementScale = 1 }
                        TrainerLog.general.info(
                            "Pose check at iteration \(iteration): loss \(before) -> \(after) after one step; full-rate refinement \(descends ? "on" : "off")"
                        )
                    }
                    _ = resources.lossAccum.writeArray([lossStep])
                    if statsBefore.count == splatCount { _ = resources.stats.writeArray(statsBefore) }
                }
                timings.encodeStep += CFAbsoluteTimeGetCurrent() - encodeStepFrom
                if profiling && !calibrating { timings.profiledSteps += 1 }
                return
            }

            guard let bufferB = queue.makeCommandBuffer(),
                  let encoderB = bufferB.makeComputeCommandEncoder()
            else { throw TrainerError.noMetalDevice }
            encoderB.label = "trainer.step"

            gpu.duplicateKeys(
                encoderB, camera: &camera, splatCount: splatCount, ordered: splatOrderChosen
            )
            gpu.radixSort(
                encoderB, count: instanceCount, simdScan: sortSimdScanChosen,
                tileOnly: splatOrderChosen
            )
            gpu.tileRanges(encoderB, instanceCount: instanceCount)
            gpu.rasterizeForward(encoderB, camera: &camera, twoPixels: forwardTwoPixelChosen)

            gpu.lossPhotometric(encoderB, loss: &loss)
            let partials = gpu.ssim(encoderB, loss: &loss, fused: blurFusedChosen)
            gpu.lossFinalize(encoderB, loss: &loss, blurredPartials: partials)
            gpu.lossDepth(encoderB, loss: &loss, sampleCount: sampleCount)

            gpu.rasterizeBackward(
                encoderB, camera: &camera, loss: &loss, simdSum: backwardSimdSumChosen,
                twoPixels: backwardTwoPixelChosen
            )
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
            // Snapshot the read-backs into this step's staging slot, in the same
            // buffer, after everything that writes them. Only for a step that
            // will be left running; a synchronous step reads the originals.
            var staged = false
            if deferCompletion, let blit = bufferB.makeBlitCommandEncoder() {
                let base = resources.inputSlot * 64
                blit.copy(from: resources.lossAccum, sourceOffset: 0,
                          to: resources.readbackStaging, destinationOffset: base, size: 4)
                blit.copy(from: resources.exposureGrad, sourceOffset: 0,
                          to: resources.readbackStaging, destinationOffset: base + 16, size: 8)
                blit.copy(from: resources.cameraGrad, sourceOffset: 0,
                          to: resources.readbackStaging, destinationOffset: base + 32, size: 24)
                blit.endEncoding()
                staged = true
            }
            bufferB.commit()
            timings.encodeStep += CFAbsoluteTimeGetCurrent() - encodeStepFrom
            if staged {
                pendingStep = PendingStep(
                    buffer: bufferB, slot: resources.inputSlot, frame: frame,
                    exposure: exposure, iteration: iteration,
                    totalIterations: totalIterations, splatCount: splatCount, merged: false,
                    snapshotCount: 0, warmup: false, supervision: nil, background: nil
                )
                timings.overlappedSteps += 1
                return
            }
            // NOT "the tile sort". This one command buffer holds the
            // sort, the forward raster, the losses, the backward raster and
            // the optimiser, so labelling it as the sort credited all five
            // to `gpuSort` and left the other four buckets reading exactly
            // 0.00 - which is what a build-172 census showed, and which made
            // every remaining speed decision unreadable.
            try finish(bufferB, "the training step")
        }

        // A step left running (build 292) applies its read-backs in drainPendingStep.
        if pendingStep != nil { return .stepped }

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
                gradFinal: resources.gradFinal, gradOffset: 0,
                tFinal: resources.renderTFinal, tOffset: 0,
                size: resources.renderSize, supervision: supervision
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
        camera.activeSHCoeffCount = UInt32(activeSHCoefficients(
            iteration: iteration, totalIterations: totalIterations,
            shCoefficientCount: shCoefficientCount
        ))

        // And low frequencies first, in screen space. This decays to exactly
        // zero, not to a small number, so late training is not permanently
        // blurred by a leftover epsilon.
        if fraction < tuning.frequencyBlurEndFraction, tuning.frequencyBlurEndFraction > 0 {
            let t = fraction / tuning.frequencyBlurEndFraction
            // Scaled by the square of the render scale (build 328), so the
            // coarse phase blurs by the same amount of the photograph.
            camera.frequencyBlurVariance = tuning.frequencyBlurStartVariance * (1 - t)
                * resolutionScale * resolutionScale
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
        reg.sparse = 1                  // MUST match adam.sparse in adamUniforms
        return reg
    }

    /// The spherical-harmonic ramp: DC only at first, all coefficients from
    /// `shFullyEnabledFraction` of the run. Read by cameraUniforms (what the
    /// rasteriser evaluates) and adamUniforms (what the optimiser walks,
    /// build 332), so the two can never disagree.
    private func activeSHCoefficients(
        iteration: Int, totalIterations: Int, shCoefficientCount: Int
    ) -> Int {
        let fraction = Float(iteration) / Float(Swift.max(totalIterations, 1))
        let shGate = TrainerMath.clamp(fraction / Swift.max(tuning.shFullyEnabledFraction, 1e-3), 0, 1)
        let activeCoefficients = 1 + Int(Float(shCoefficientCount - 1) * shGate)
        return Swift.max(Swift.min(activeCoefficients, shCoefficientCount), 1)
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
        adam.activeSHCoeffCount = UInt32(activeSHCoefficients(
            iteration: iteration, totalIterations: totalIterations,
            shCoefficientCount: shCoefficientCount
        ))
        adam.beta1 = 0.9
        adam.beta2 = 0.999
        adam.epsilon = 1e-15

        let t = Float(iteration) / Float(Swift.max(totalIterations, 1))
        adam.lrMean = TrainerMath.expLerp(
            tuning.positionLRInitialScaled * sceneExtent,
            tuning.positionLRFinalScaled * sceneExtent,
            t: t
        )
        // Position already decayed above. These four did not, because the
        // reference does not decay them either - and the reference runs
        // 30,000 iterations against our 3,000, so a rate that should have
        // annealed by the end is still near its starting value when we stop.
        // See `lateLRFraction`. Setting it to 1 restores the flat behaviour.
        let lrDecay = TrainerMath.expLerp(
            1, Swift.max(tuning.lateLRFraction, 1e-4), t: t
        )
        let shDC = tuning.shDCLR * Swift.max(tuning.shDCLRMultiplier, 0)
        adam.lrScale = tuning.scaleLR * lrDecay
        adam.lrRotation = tuning.rotationLR * lrDecay
        adam.lrOpacity = tuning.opacityLR * lrDecay
        adam.lrSHDC = shDC * lrDecay
        adam.lrSHRest = shDC * lrDecay / Swift.max(tuning.shRestLRDivisor, 1)
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
        coarseSupervisions: [(TrainerSupervisionBuilder, Int)] = [],
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
            // Each level by its own clamped per-mille, as the slice sized them.
            for (builder, perMille) in coarseSupervisions {
                builder.lowerLongEdge(to: to * perMille / 1000)
            }
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
            case "the training step": timings.gpuStep += executing
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
    private func updatedCameraDelta(
        current: Pose?, gradient: [Float], rateScale: Float? = nil
    ) -> Pose {
        let scale = rateScale ?? poseRefinementScale
        let omega = SIMD3<Float>(gradient[0], gradient[1], gradient[2])
            * -(tuning.cameraRotationLR * scale)
        let nu = SIMD3<Float>(gradient[3], gradient[4], gradient[5])
            * -(tuning.cameraTranslationLR * scale)

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
    /// them. Sampled on a stride: the field is a 64x64x6 cubemap, and every
    /// pixel of every frame would be a hundred samples per texel per iteration.
    /// `gradFinal` and `tFinal` are the live buffers (the synchronous path) or
    /// a warm-up step's staged copies of them at the given offsets (build 324).
    private func accumulateBackgroundGradient(
        background: DirectionalBackgroundModel,
        gradFinal: MTLBuffer,
        gradOffset: Int,
        tFinal: MTLBuffer,
        tOffset: Int,
        size: TrainerRenderSize,
        supervision: TrainerFrameSupervision
    ) {
        let pixelCount = size.pixelCount
        guard pixelCount > 0 else { return }

        let rotationInverse = supervision.pose.rotation.simd.inverse
        let k = supervision.intrinsics

        // Roughly two thousand samples per iteration, not every pixel. The
        // field is a 64x64x6 cubemap (4,000 samples a step since build 326,
        // twice build 292's 2,000 for four times the texels): at 720p, every pixel would be several
        // hundred samples per texel per iteration, each one taking the
        // background model's lock, and the extra samples buy nothing because
        // the texel is an average either way. The stride is forced odd so the
        // sampled set is not a grid aligned to the image width.
        var stride = Swift.max(pixelCount / 4_000, 1)
        if stride % 2 == 0 { stride += 1 }

        // Read the two buffers in place. They are shared-storage, the GPU is
        // idle at this point in the iteration, and copying 4 MB of gradient
        // into a Swift array to walk it once would cost more than the walk.
        _ = gradFinal.withElements(Float.self, count: pixelCount * 3, byteOffset: gradOffset) { gradFinal in
            _ = tFinal.withElements(Float.self, count: pixelCount, byteOffset: tOffset) { tFinal in
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
        renderSize: TrainerRenderSize,
        /// Fit a per-frame gain and bias to THIS frame's own render before
        /// scoring it. See the use site: a held-out frame otherwise gets
        /// identity exposure while a trained frame gets a fitted one, which
        /// puts part of the reported train/test gap in the protocol rather
        /// than in the model.
        fitExposure: Bool = false,
        /// Also score the SAME renders after the per-frame exposure fit and
        /// leave it in `lastHeldOutPSNRExposureFitted`.
        alsoScoreExposureFitted: Bool = false
    ) throws -> Float? {

        lastHeldOutPSNRExposureFitted = nil
        lastHeldOutSSIM = nil
        lastHeldOutPerFrame = []
        guard splatCount > 0, !frames.isEmpty else { return nil }
        // HELD-OUT VIEWS MUST NOT REACH A DENSIFY PASS. The eval's preprocess
        // writes denom, visibleFlag and maxRadiusPxBits into the live stats
        // buffer, and the loop runs eval, THEN densifier.run (which reads
        // them), THEN resetDensifyStats. So every early-stop evaluation was
        // adding up to 12 observations with no gradient to the AbsGS
        // denominators of the pass right after it. Restored on every exit.
        // Shared storage, and every eval buffer is waited on, so the GPU is
        // idle when this runs.
        let statsBeforeEval = resources.stats.readArray(TrainerSplatStats.self, count: splatCount)
        defer { resources.stats.writeArray(statsBeforeEval) }
        // PER-IMAGE PSNR, averaged. This used to sum the MSE across frames and
        // convert once at the end, which by Jensen's inequality is ALWAYS the
        // lower number: one dark or badly-posed frame with a large MSE drags
        // the mean far more than it drags the mean of the logs. Every 3DGS
        // paper averages per-image PSNR, so the old number was not comparable
        // with any published figure either.
        var totalPSNR: Double = 0
        // STRUCTURE, WHICH PSNR CANNOT SEE. The owner compared two builds on a
        // real photograph and found the one with the HIGHER PSNR visibly worse
        // in cluttered regions: better colour and brightness, visible
        // artefacting where the other was smooth. Both readings were correct,
        // because mean squared error is dominated by large flat areas being
        // approximately the right brightness and barely notices whether an
        // edge is an edge.
        //
        // Global SSIM on luma is the cheap standard answer. It compares local
        // means, variances and covariance rather than per-pixel difference, so
        // a smeared edge costs it and a slight overall brightness shift does
        // not. Reported BESIDE PSNR, never instead of it: the two disagree
        // exactly when something interesting has happened, which is the point.
        var totalSSIM: Double = 0
        var totalPSNRFitted: Double = 0
        var evaluated = 0

        var collected: [HeldOutRender] = []
        // SCORED ON EVERY CORE (build 300). The GPU renders the frames (the
        // loop below); the per-frame metrics, three Double passes over ~1.2 M
        // values each, run in parallel over what has been collected and are
        // combined in frame order. Each frame's arithmetic is the loop's,
        // unchanged, so every score is the same number it was.
        let scoreTuning = tuning
        let scorePixelCount = renderSize.pixelCount
        let bw = renderSize.width, bh = renderSize.height
        let scoreChunk = Swift.max(ProcessInfo.processInfo.activeProcessorCount, 1)
        func scoreCollected() {
            let renders = collected
            collected.removeAll(keepingCapacity: true)
            guard !renders.isEmpty else { return }
            var scores = [HeldOutScore?](repeating: nil, count: renders.count)
            scores.withUnsafeMutableBufferPointer { out in
                DispatchQueue.concurrentPerform(iterations: renders.count) { k in
                    out[k] = Self.scoreHeldOut(
                        renders[k], pixelCount: scorePixelCount, width: bw, height: bh,
                        fitExposure: fitExposure, alsoScoreExposureFitted: alsoScoreExposureFitted,
                        tuning: scoreTuning
                    )
                }
            }
            for (k, score) in scores.enumerated() {
                guard let score else { continue }
                if let ssim = score.ssim { totalSSIM += ssim }
                totalPSNR += score.psnr
                lastHeldOutPerFrame.append(
                    TrainerHeldOutFrameScore(frameIndex: Int(renders[k].frame.index), psnr: Float(score.psnr))
                )
                if alsoScoreExposureFitted { totalPSNRFitted += score.psnrFitted }
                evaluated += 1
            }
        }
        // BUILD 322: THE FRAMES RENDER BACK TO BACK. Each frame is one command
        // buffer with the sort sized on the GPU (build 316's setup kernel and
        // indirect dispatches), ending in a copy of its outputs into one of
        // two staging slots; the next frame is committed BEFORE this one is
        // waited for, so its render overlaps this one's read-back and score.
        // It was two command buffers and two waits per frame with the GPU
        // idle through every read-back: 1.6 s a run on build 292.
        let px = renderSize.pixelCount
        let slotBytes = TrainerResources.evalStagingSlotBytes(pixelCount: px)
        struct EvalPending {
            let buffer: MTLCommandBuffer
            let slot: Int
            let frame: CaptureFrame
            let supervision: TrainerFrameSupervision
            let hasBackground: Bool
        }
        var pending: EvalPending?
        // Runs before the stats restore above (later defers run first): a
        // frame still rendering when this returns early must finish before
        // the live stats are written over.
        defer { pending?.buffer.waitUntilCompleted() }
        var slot = 0
        func collect(_ done: EvalPending) throws {
            try finish(done.buffer, "the held-out render")
            let base = done.slot * slotBytes
            // The same rule the two-buffer path applied from its CPU count: a
            // frame that produced no instances, or more than the buffers
            // hold, is not scored.
            let counts = resources.evalStaging.readArray(UInt32.self, count: 2, byteOffset: base)
            guard counts.count == 2, counts[0] > 0,
                  counts[0] <= UInt32(resources.instanceCapacity)
            else { return }
            let rendered = resources.evalStaging.readArray(
                Float.self, count: px * 3, byteOffset: base + 16
            )
            let transmittance = resources.evalStaging.readArray(
                Float.self, count: px, byteOffset: base + 16 + px * 12
            )
            guard rendered.count == px * 3, transmittance.count == px else { return }
            let heldOutBackground: [Float]? = done.hasBackground
                ? resources.evalStaging.readArray(Float.self, count: px * 3, byteOffset: base + 16 + px * 16)
                : nil
            if let heldOutBackground, heldOutBackground.count != px * 3 { return }

            // A HELD-OUT FRAME HAS NO FITTED EXPOSURE, AND THAT IS NOT A
            // PROPERTY OF THE MODEL.
            //
            // `exposures` and `cameraDeltas` are only ever written for frames
            // drawn from `slice.keyframes`, never from `slice.heldOutKeyframes`.
            // So a trained view is scored with a per-frame gain and bias fitted
            // to it, and a held-out view is scored at gain 1 bias 0. Part of the
            // reported train/test gap is therefore the protocol, not
            // generalisation, and it moves whenever the capture's auto-exposure
            // drifts rather than when the model changes.
            //
            // The fix that does not leak training signal: solve the closed-form
            // least-squares (gain, bias) between this render and its own ground
            // truth, clamp it to the SAME range the trainer allows, and report
            // the corrected number ALONGSIDE the raw one. Two scalars fitted to
            // a frame the model never trained on is a photometric alignment, not
            // a fit of the geometry; the raw number is still reported so nothing
            // is hidden.
            collected.append(HeldOutRender(
                frame: done.frame,
                rendered: rendered,
                transmittance: transmittance,
                background: heldOutBackground,
                groundTruth: done.supervision.groundTruthBytes,
                exposure: exposures[done.frame.index] ?? SIMD2<Float>(1, 0)
            ))
            // Scored a core's worth at a time (build 318): holding every
            // frame's read-backs until the end was up to 24 x 15 MB alive at
            // once, on top of peak training memory, right where the governor
            // polls. Chunks keep the peak at a few frames; the order the
            // scores are combined in is unchanged.
            if collected.count >= scoreChunk { scoreCollected() }
        }

        for frame in frames.prefix(24) {
            // CACHED for the run (build 300): the eval visits the same frames
            // ~9 times, and every build re-decoded the photo (about half the
            // eval's time). Nothing in a held-out frame's supervision changes
            // once the background is frozen (evals start after warm-up); a
            // render-size change invalidates the entry.
            let frameSupervision: TrainerFrameSupervision
            if let cached = evalSupervisionCache[frame.index], cached.renderSize == renderSize {
                frameSupervision = cached
            } else {
                guard let built = supervision.build(
                    frame: frame, iteration: 0, totalIterations: 1, includeDepthSamples: false
                ) else { continue }
                evalSupervisionCache[frame.index] = built
                frameSupervision = built
            }
            guard frameSupervision.renderSize == renderSize else { continue }

            // No ground-truth upload: the held-out render runs no loss kernel,
            // and the score is taken on the CPU from the bytes. The far field
            // comes from the SAME trainer_background kernel the training path
            // uses. The cubemap goes into the input slot the frame still
            // rendering is NOT reading (the two are double-buffered).
            let useGPUBackground = frameSupervision.hasBackground
                && frameSupervision.backgroundFaceSize > 0
                && !frameSupervision.backgroundTexels.isEmpty
            resources.inputSlot = slot
            if useGPUBackground {
                resources.bgCubemapIn.writeArray(frameSupervision.backgroundTexels)
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

            if resources.instanceCapacity < splatCount {
                try resources.growInstanceCapacity(to: splatCount + splatCount / 4)
            }
            guard let buffer = queue.makeCommandBuffer(),
                  let front = buffer.makeComputeCommandEncoder()
            else { return nil }
            front.label = "trainer.eval.preprocess"
            if useGPUBackground {
                let q = frameSupervision.pose.rotation.simd.inverse
                var bg = TrainerBackgroundUniforms(
                    rotationInverse: SIMD4<Float>(q.imag.x, q.imag.y, q.imag.z, q.real),
                    fx: frameSupervision.intrinsics.fx,
                    fy: frameSupervision.intrinsics.fy,
                    cx: frameSupervision.intrinsics.cx,
                    cy: frameSupervision.intrinsics.cy,
                    width: UInt32(renderSize.width),
                    height: UInt32(renderSize.height),
                    faceSize: UInt32(frameSupervision.backgroundFaceSize),
                    pad: 0
                )
                gpu.background(front, uniforms: &bg)
            }
            gpu.fillUInt(front, buffer: resources.tilesTouched, count: splatCount, value: 0)
            gpu.preprocess(front, camera: &camera, splatCount: splatCount)
            if splatOrderChosen {
                gpu.orderSplats(
                    front, camera: &camera, splatCount: splatCount,
                    simdScan: sortSimdScanChosen
                )
            } else {
                gpu.exclusiveScan(
                    front,
                    input: resources.tilesTouched,
                    output: resources.offsets,
                    count: splatCount
                )
            }
            gpu.sortSetup(front, camera: &camera, ordered: splatOrderChosen)
            front.endEncoding()

            guard let render = buffer.makeComputeCommandEncoder() else { return nil }
            render.label = "trainer.eval.render"
            gpu.duplicateKeys(
                render, camera: &camera, splatCount: splatCount, ordered: splatOrderChosen
            )
            gpu.radixSortIndirect(render, simdScan: sortSimdScanChosen, tileOnly: splatOrderChosen)
            gpu.tileRangesIndirect(render)
            gpu.rasterizeForward(render, camera: &camera, twoPixels: forwardTwoPixelChosen)
            render.endEncoding()

            guard let copy = buffer.makeBlitCommandEncoder() else { return nil }
            let base = slot * slotBytes
            copy.copy(from: resources.sortArgs, sourceOffset: 0,
                      to: resources.evalStaging, destinationOffset: base, size: 8)
            copy.copy(from: resources.renderColor, sourceOffset: 0,
                      to: resources.evalStaging, destinationOffset: base + 16, size: px * 12)
            copy.copy(from: resources.renderTFinal, sourceOffset: 0,
                      to: resources.evalStaging, destinationOffset: base + 16 + px * 12, size: px * 4)
            if useGPUBackground {
                copy.copy(from: resources.bgColor, sourceOffset: 0,
                          to: resources.evalStaging, destinationOffset: base + 16 + px * 16, size: px * 12)
            }
            copy.endEncoding()
            buffer.commit()

            let next = EvalPending(
                buffer: buffer, slot: slot, frame: frame,
                supervision: frameSupervision, hasBackground: useGPUBackground
            )
            if let previous = pending {
                pending = nil
                try collect(previous)
            }
            pending = next
            slot = 1 - slot
        }
        if let last = pending {
            pending = nil
            try collect(last)
        }
        scoreCollected()

        guard evaluated > 0 else { return nil }
        lastHeldOutSSIM = Float(totalSSIM / Double(evaluated))
        if alsoScoreExposureFitted {
            lastHeldOutPSNRExposureFitted = Float(totalPSNRFitted / Double(evaluated))
        }
        return Float(totalPSNR / Double(evaluated))
    }

    // MARK: - Test-time pose alignment (build 332)

    /// A few clamped gradient steps on each held-out frame's camera against
    /// the finished model, exposure fitted first, the correction with the
    /// lowest photometric loss kept. Returns the training deltas with the
    /// held-out frames' entries replaced. Nothing of the model is written:
    /// the backward runs only for the camera gradient, the per-interval
    /// stats it folds are put back, and every other buffer it touches is
    /// per-step scratch.
    private func alignHeldOutPoses(
        gpu: TrainerGPU,
        resources: TrainerResources,
        queue: MTLCommandQueue,
        frames: [CaptureFrame],
        supervision: TrainerSupervisionBuilder,
        cameraDeltas: [FrameID: Pose],
        splatCount: Int,
        shCoefficientCount: Int,
        renderSize: TrainerRenderSize
    ) throws -> [FrameID: Pose] {
        var aligned = cameraDeltas
        guard splatCount > 0, !frames.isEmpty else { return aligned }
        let px = renderSize.pixelCount
        guard px > 0 else { return aligned }
        let statsBefore = resources.stats.readArray(TrainerSplatStats.self, count: splatCount)
        defer { if statsBefore.count == splatCount { _ = resources.stats.writeArray(statsBefore) } }
        let steps = 8
        // Steps saturate the per-step clamps (2 mm, 0.5 mrad) while the
        // gradient is large and shrink as it settles; eight of them reach
        // 1.6 cm, above the pose graph's own residual.
        let rateScale: Float = 100

        for frame in frames.prefix(24) {
            let frameSupervision: TrainerFrameSupervision
            if let cached = evalSupervisionCache[frame.index], cached.renderSize == renderSize {
                frameSupervision = cached
            } else {
                guard let built = supervision.build(
                    frame: frame, iteration: 0, totalIterations: 1, includeDepthSamples: false
                ) else { continue }
                evalSupervisionCache[frame.index] = built
                frameSupervision = built
            }
            guard frameSupervision.renderSize == renderSize else { continue }
            let useGPUBackground = frameSupervision.hasBackground
                && frameSupervision.backgroundFaceSize > 0
                && !frameSupervision.backgroundTexels.isEmpty
            resources.inputSlot = 0
            // The photo itself (build 336). The loss kernels read slot 0's
            // ground truth, which the training loop last filled with a
            // TRAINING frame; without this upload every loss and camera
            // gradient below was measured against the wrong picture.
            guard frameSupervision.groundTruthBytes.count == px * 3 else { continue }
            _ = resources.gtColorIn.writeArray(frameSupervision.groundTruthBytes)
            if useGPUBackground {
                _ = resources.bgCubemapIn.writeArray(frameSupervision.backgroundTexels)
            }
            if resources.instanceCapacity < splatCount {
                try resources.growInstanceCapacity(to: splatCount + splatCount / 4)
            }

            var loss = TrainerLossUniforms()
            loss.pixelCount = UInt32(px)
            loss.width = UInt32(renderSize.width)
            loss.height = UInt32(renderSize.height)
            loss.lambdaSSIM = tuning.lambdaSSIM
            loss.frameWeight = 1
            loss.exposureGain = 1
            loss.exposureBias = 0
            loss.depthScale = 0
            loss.depthSampleCount = 0
            loss.depthSupervisedCount = 0
            loss.hasBackground = useGPUBackground ? 1 : 0

            /// One render of the frame at `delta`. With `backward`, the
            /// camera gradient is left in cameraGrad. Returns the loss.
            func render(at delta: Pose?, backward: Bool) throws -> Float {
                var camera = cameraUniforms(
                    supervision: frameSupervision, cameraDelta: delta, size: renderSize,
                    splatCount: splatCount, shCoefficientCount: shCoefficientCount,
                    iteration: 1, totalIterations: 1
                )
                guard let buffer = queue.makeCommandBuffer(),
                      let blit = buffer.makeBlitCommandEncoder()
                else { throw TrainerError.noMetalDevice }
                blit.label = "trainer.align.clear"
                gpu.clearPerIteration(blit, splatCount: splatCount)
                blit.endEncoding()
                guard let front = buffer.makeComputeCommandEncoder()
                else { throw TrainerError.noMetalDevice }
                front.label = "trainer.align.preprocess"
                if useGPUBackground {
                    let q = frameSupervision.pose.rotation.simd.inverse
                    var bg = TrainerBackgroundUniforms(
                        rotationInverse: SIMD4<Float>(q.imag.x, q.imag.y, q.imag.z, q.real),
                        fx: frameSupervision.intrinsics.fx,
                        fy: frameSupervision.intrinsics.fy,
                        cx: frameSupervision.intrinsics.cx,
                        cy: frameSupervision.intrinsics.cy,
                        width: UInt32(renderSize.width),
                        height: UInt32(renderSize.height),
                        faceSize: UInt32(frameSupervision.backgroundFaceSize),
                        pad: 0
                    )
                    gpu.background(front, uniforms: &bg)
                }
                gpu.preprocess(front, camera: &camera, splatCount: splatCount)
                if splatOrderChosen {
                    gpu.orderSplats(
                        front, camera: &camera, splatCount: splatCount, simdScan: sortSimdScanChosen
                    )
                } else {
                    gpu.exclusiveScan(
                        front, input: resources.tilesTouched, output: resources.offsets,
                        count: splatCount
                    )
                }
                gpu.sortSetup(front, camera: &camera, ordered: splatOrderChosen)
                front.endEncoding()
                guard let step = buffer.makeComputeCommandEncoder()
                else { throw TrainerError.noMetalDevice }
                step.label = "trainer.align.render"
                gpu.duplicateKeys(step, camera: &camera, splatCount: splatCount, ordered: splatOrderChosen)
                gpu.radixSortIndirect(step, simdScan: sortSimdScanChosen, tileOnly: splatOrderChosen)
                gpu.tileRangesIndirect(step)
                gpu.rasterizeForward(step, camera: &camera, twoPixels: forwardTwoPixelChosen)
                gpu.lossPhotometric(step, loss: &loss)
                let partials = gpu.ssim(step, loss: &loss, fused: blurFusedChosen)
                if backward {
                    gpu.lossFinalize(step, loss: &loss, blurredPartials: partials)
                    gpu.rasterizeBackward(
                        step, camera: &camera, loss: &loss, simdSum: backwardSimdSumChosen,
                        twoPixels: backwardTwoPixelChosen
                    )
                    gpu.preprocessBackward(step, camera: &camera, splatCount: splatCount)
                }
                step.endEncoding()
                buffer.commit()
                try finish(buffer, "the pose alignment")
                return resources.lossAccum.readElement(Float.self, at: 0) ?? .nan
            }

            // Exposure first, closed form, from a render at the starting
            // correction: the same two-scalar fit the score applies, so the
            // pose is aligned against the brightness it will be scored at.
            var delta = aligned[frame.index]
            _ = try render(at: delta, backward: false)
            let rendered = resources.renderColor.readArray(Float.self, count: px * 3)
            let transmittance = resources.renderTFinal.readArray(Float.self, count: px)
            let background: [Float]? = useGPUBackground
                ? resources.bgColor.readArray(Float.self, count: px * 3) : nil
            if rendered.count == px * 3, transmittance.count == px,
               background == nil || background?.count == px * 3 {
                let truth = frameSupervision.groundTruthBytes
                var sx: Double = 0, sy: Double = 0, sxx: Double = 0, sxy: Double = 0, n: Double = 0
                for i in 0..<px {
                    for c in 0..<3 {
                        var v = Double(rendered[i * 3 + c])
                        if let background { v += Double(transmittance[i] * background[i * 3 + c]) }
                        let t = Double(Float(truth[i * 3 + c]) / 255)
                        sx += v; sy += t; sxx += v * v; sxy += v * t; n += 1
                    }
                }
                let denom = n * sxx - sx * sx
                if denom > 1e-9 {
                    let gain = (n * sxy - sx * sy) / denom
                    let bias = (sy - gain * sx) / n
                    if gain.isFinite, bias.isFinite {
                        loss.exposureGain = TrainerMath.clamp(
                            Float(gain), tuning.exposureGainRange.lowerBound, tuning.exposureGainRange.upperBound
                        )
                        loss.exposureBias = TrainerMath.clamp(
                            Float(bias), tuning.exposureBiasRange.lowerBound, tuning.exposureBiasRange.upperBound
                        )
                    }
                }
            }

            var best = delta
            var bestLoss = Float.infinity
            for k in 0...steps {
                let value = try render(at: delta, backward: k < steps)
                if value.isFinite, value < bestLoss {
                    bestLoss = value
                    best = delta
                }
                guard k < steps else { break }
                let gradient = resources.cameraGrad.readArray(Float.self, count: 6)
                guard gradient.count == 6, gradient.allSatisfy({ $0.isFinite }) else { break }
                delta = updatedCameraDelta(current: delta, gradient: gradient, rateScale: rateScale)
            }
            if let best { aligned[frame.index] = best }
        }
        return aligned
    }

    // MARK: - Reading the field back

    /// One held-out frame's PSNR, exposure-fitted PSNR and block SSIM, exactly
    /// as evaluateHeldOut's loop computed them before build 300 (the same
    /// expressions, loops and Double accumulations in the same order), moved
    /// here so the frames can be scored on every core.
    private static func scoreHeldOut(
        _ frameRender: HeldOutRender,
        pixelCount: Int,
        width bw: Int,
        height bh: Int,
        fitExposure: Bool,
        alsoScoreExposureFitted: Bool,
        tuning: TrainerTuning
    ) -> HeldOutScore {
        let rendered = frameRender.rendered
        let transmittance = frameRender.transmittance
        let heldOutBackground = frameRender.background
        let groundTruth = frameRender.groundTruth

        var exposure = frameRender.exposure
        var fittedExposure = exposure
        if fitExposure || alsoScoreExposureFitted {
            var sx: Double = 0, sy: Double = 0, sxx: Double = 0
            var sxy: Double = 0, n: Double = 0
            for i in 0..<pixelCount {
                for c in 0..<3 {
                    var v = Double(rendered[i * 3 + c])
                    if let heldOutBackground {
                        v += Double(transmittance[i] * heldOutBackground[i * 3 + c])
                    }
                    let t = Double(Float(groundTruth[i * 3 + c]) / 255)
                    sx += v; sy += t; sxx += v * v; sxy += v * t; n += 1
                }
            }
            let denom = n * sxx - sx * sx
            if denom > 1e-9 {
                let gain = (n * sxy - sx * sy) / denom
                let bias = (sy - gain * sx) / n
                if gain.isFinite, bias.isFinite {
                    fittedExposure = SIMD2<Float>(
                        Swift.min(Swift.max(Float(gain), tuning.exposureGainRange.lowerBound),
                                  tuning.exposureGainRange.upperBound),
                        Swift.min(Swift.max(Float(bias), tuning.exposureBiasRange.lowerBound),
                                  tuning.exposureBiasRange.upperBound)
                    )
                }
            }
        }
        if fitExposure { exposure = fittedExposure }
        var sum: Double = 0
        var sumFitted: Double = 0
        for i in 0..<pixelCount {
            for c in 0..<3 {
                var composited = rendered[i * 3 + c]
                if let heldOutBackground {
                    composited += transmittance[i] * heldOutBackground[i * 3 + c]
                }
                let truth = Float(groundTruth[i * 3 + c]) / 255
                let value = exposure.x * composited + exposure.y
                let diff = Double(value - truth)
                sum += diff * diff
                if alsoScoreExposureFitted {
                    let fittedValue = fittedExposure.x * composited + fittedExposure.y
                    let fittedDiff = Double(fittedValue - truth)
                    sumFitted += fittedDiff * fittedDiff
                }
            }
        }
        // Luma SSIM over 8x8 blocks, as before.
        var ssimSum = 0.0
        var ssimBlocks = 0
        let c1 = 0.01 * 0.01, c2 = 0.03 * 0.03
        var by = 0
        while by + 8 <= bh {
            var bx = 0
            while bx + 8 <= bw {
                var mr = 0.0, mt = 0.0
                var vr = 0.0, vt = 0.0, cov = 0.0
                for dy in 0..<8 {
                    for dx in 0..<8 {
                        let i = (by + dy) * bw + (bx + dx)
                        var rv = rendered[i * 3]
                        var gv = rendered[i * 3 + 1]
                        var bv = rendered[i * 3 + 2]
                        if let heldOutBackground {
                            let t = transmittance[i]
                            rv += t * heldOutBackground[i * 3]
                            gv += t * heldOutBackground[i * 3 + 1]
                            bv += t * heldOutBackground[i * 3 + 2]
                        }
                        let lr = Double(exposure.x * (0.299 * rv + 0.587 * gv + 0.114 * bv) + exposure.y)
                        let lt = Double(
                            0.299 * (Float(groundTruth[i * 3]) / 255)
                                + 0.587 * (Float(groundTruth[i * 3 + 1]) / 255)
                                + 0.114 * (Float(groundTruth[i * 3 + 2]) / 255)
                        )
                        mr += lr; mt += lt
                        vr += lr * lr; vt += lt * lt; cov += lr * lt
                    }
                }
                let n = 64.0
                mr /= n; mt /= n
                vr = Swift.max(vr / n - mr * mr, 0)
                vt = Swift.max(vt / n - mt * mt, 0)
                cov = cov / n - mr * mt
                let num = (2 * mr * mt + c1) * (2 * cov + c2)
                let den = (mr * mr + mt * mt + c1) * (vr + vt + c2)
                if den > 1e-12 { ssimSum += num / den; ssimBlocks += 1 }
                bx += 8
            }
            by += 8
        }

        let frameMSE = sum / Double(pixelCount * 3)
        let framePSNR = frameMSE > 1e-12 ? 10 * log10(1.0 / frameMSE) : 99
        var fittedPSNR: Double = 0
        if alsoScoreExposureFitted {
            let fittedMSE = sumFitted / Double(pixelCount * 3)
            fittedPSNR = fittedMSE > 1e-12 ? 10 * log10(1.0 / fittedMSE) : 99
        }
        return HeldOutScore(
            psnr: framePSNR,
            psnrFitted: fittedPSNR,
            ssim: ssimBlocks > 0 ? ssimSum / Double(ssimBlocks) : nil
        )
    }

    private func readCloud(
        resources: TrainerResources,
        count: Int,
        shDegree: SHDegree
    ) -> SplatCloud {
        guard count > 0 else { return SplatCloud.empty(shDegree: shDegree) }
        let shPerSplat = resources.shFloatsPerSplat
        return Self.buildCloud(
            splats: resources.splats.readArray(TrainerSplat.self, count: count),
            sh: resources.sh.readArray(Float.self, count: count * shPerSplat),
            shPerSplat: shPerSplat,
            stats: resources.stats.readArray(TrainerSplatStats.self, count: count),
            shDegree: shDegree
        )
    }

    /// The cloud from three arrays read off the GPU: the live buffers, or the
    /// build 322 snapshot staging. Static, because the preview conversion
    /// runs it on a background thread.
    private static func buildCloud(
        splats: [TrainerSplat],
        sh: [Float],
        shPerSplat: Int,
        stats: [TrainerSplatStats],
        shDegree: SHDegree
    ) -> SplatCloud {
        guard !splats.isEmpty else { return SplatCloud.empty(shDegree: shDegree) }
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

        // BUILD 314: THE INDEPENDENT LOADS RUN AT THE SAME TIME. The trust
        // fields, the edge maps and the free-space map are three separate
        // files with nothing in common, so they load together; the authority
        // map needs the trust fields and the far field needs the authority
        // map, so those two follow. Each load is timed on its own for the
        // census (its duration, overlapped or not), beside the total.
        let settings = self.settings
        async let trustLoad = Self.loadTrust(prePass.trust, bundle: bundle, at: ref, settings: settings)
        async let carverLoad = Self.loadCarver(prePass.occupancy, at: ref)

        let edgesClock = CFAbsoluteTimeGetCurrent()
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
        timings.smartLayerEdges += CFAbsoluteTimeGetCurrent() - edgesClock

        let trust = await trustLoad
        layer.trust = trust.value.field
        timings.smartLayerTrust += trust.value.seconds

        let authorityClock = CFAbsoluteTimeGetCurrent()
        let authorityMap = SmartAuthorityMap(settings: settings)
        authorityMap.prepare(
            bundle: bundle,
            at: ref,
            prePassPoses: prePass.refinedPoses,
            glassRegions: prePass.glassRegions,
            trust: layer.trust
        )
        if authorityMap.isPrepared { layer.authority = authorityMap }
        timings.smartLayerAuthority += CFAbsoluteTimeGetCurrent() - authorityClock

        let backgroundClock = CFAbsoluteTimeGetCurrent()
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
        timings.smartLayerBackground += CFAbsoluteTimeGetCurrent() - backgroundClock

        let carver = await carverLoad
        layer.carver = carver.value.carver
        timings.smartLayerCarver += carver.value.seconds

        return layer
    }

    /// A value handed back from an `async let` child. The SMART objects are
    /// plain classes; nothing touches one until its load has returned.
    private struct SmartLoad<T>: @unchecked Sendable {
        let value: T
    }

    private static func loadTrust(
        _ refs: TrustFieldRefs?,
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        settings: SmartLossSettings
    ) async -> SmartLoad<(field: TwoScaleTrustField?, seconds: Double)> {
        guard let refs else { return SmartLoad(value: (nil, 0)) }
        let clock = CFAbsoluteTimeGetCurrent()
        let field = TwoScaleTrustField(settings: settings)
        field.prepare(bundle: bundle)
        do {
            try await field.load(refs, at: ref)
            return SmartLoad(value: (field, CFAbsoluteTimeGetCurrent() - clock))
        } catch {
            let why = error.localizedDescription
            TrainerLog.general.error(
                "The trust fields could not be read (\(why, privacy: .public)); depth is supervised at a flat weight instead"
            )
            return SmartLoad(value: (nil, CFAbsoluteTimeGetCurrent() - clock))
        }
    }

    private static func loadCarver(
        _ grid: OccupancyGridRef?,
        at ref: CaptureBundleRef
    ) async -> SmartLoad<(carver: VoxelFreeSpaceCarver?, seconds: Double)> {
        guard let grid else { return SmartLoad(value: (nil, 0)) }
        let clock = CFAbsoluteTimeGetCurrent()
        let carver = VoxelFreeSpaceCarver()
        do {
            try await carver.load(grid, at: ref)
            return SmartLoad(value: (carver, CFAbsoluteTimeGetCurrent() - clock))
        } catch {
            let why = error.localizedDescription
            TrainerLog.general.error(
                "The free-space map could not be read (\(why, privacy: .public)); nothing is deleted on free-space grounds this run"
            )
            return SmartLoad(value: (nil, CFAbsoluteTimeGetCurrent() - clock))
        }
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
    ) -> (keyframes: [CaptureFrame], fixedHeldOut: [CaptureFrame]?) {
        let frames = bundle.frames.sorted { $0.index < $1.index }
        guard !frames.isEmpty else { return ([], nil) }

        // A frame with no usable pixels is not a supervision view. This is a
        // quality floor, not a discard: the frame stays in the capture and in
        // every other stage.
        let usable = frames.filter { $0.qc.weight > 0.05 }
        let pool = usable.isEmpty ? frames : usable

        let target = Swift.max(budget.keyframeCount, 8)
        if pool.count <= target { return (pool, nil) }

        // A FIXED HELD-OUT SET (build 276). The held-out frames used to be
        // every tenth entry of the chosen list, so any change to the walk (a
        // look-ahead, a gate, or a pose change upstream, since the walk runs
        // on refined poses) swapped the whole test set and no two builds
        // could be compared. 274 against 266 was exactly that: best held-out
        // 17.84 against a leaked 20.42 on a different set of frames. Now the
        // candidates are fixed by frame index alone (index % stride ==
        // stride / 2), they are removed from the walk so they can never
        // train, and the held-out set is the candidates inside the trained
        // span. The walk picks `target` minus the held-out share, so train
        // plus held-out stays at the budget and under the 128-frame caches.
        let heldOutStride = tuning.heldOutFrameStride
        let heldOutCandidates: Set<FrameID> = heldOutStride > 1
            ? Set(pool.filter { Int($0.index) % heldOutStride == heldOutStride / 2 }.map(\.index))
            : []
        let walk = heldOutCandidates.isEmpty
            ? pool : pool.filter { !heldOutCandidates.contains($0.index) }
        // Guarded: a tuning value is not trusted to be finite.
        let rawShare = (Float(target) * tuning.heldOutFraction).rounded()
        let heldOutShare = heldOutCandidates.isEmpty || !rawShare.isFinite
            ? 0 : Int(Swift.min(Float(target), Swift.max(0, rawShare)))
        let trainTarget = Swift.max(target - heldOutShare, 8)

        // Greedy spacing on the refined poses.
        var chosen: [CaptureFrame] = []
        var lastCenter: SIMD3<Float>?
        var lastForward: SIMD3<Float>?

        // Spacing that would land on roughly the target count if the walk were
        // uniform. Measured from the actual path length, not assumed.
        var pathLength: Float = 0
        var previous: SIMD3<Float>?
        for frame in walk {
            let pose = prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
            let c = pose.center.simd
            if let previous { pathLength += simd_distance(previous, c) }
            previous = c
        }
        let spacing = pathLength > 0 ? pathLength / Float(trainTarget) : 0

        for frame in walk {
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
            // SHARPNESS DECIDES WHICH FRAME, NOW THAT SPACING HAS DECIDED
            // WHERE. This selector read `qc.weight > 0.05` as a floor and then
            // never looked at sharpness again: measured on the owner's capture,
            // 93 of the 120 chosen keyframes carry more than one render pixel
            // of motion blur, native p50 3.84 px and p90 6.80, render p50 1.44
            // and p90 2.55. Every one of those is a blurred photograph the
            // model is asked to reproduce exactly.
            //
            // The fix is cheap because the frames are already sorted by index
            // and the walk is dense: rather than taking the FIRST frame that
            // clears the spacing gate, look a short way ahead and take the
            // sharpest of the candidates that also clear it. Spacing is
            // unchanged, because every candidate in the window is past the
            // same gate; only the choice within it changes.
            var best = frame
            if tuning.keyframeSharpnessLookahead > 0,
               let here = walk.firstIndex(where: { $0.index == frame.index })
            {
                let limit = Swift.min(
                    here + tuning.keyframeSharpnessLookahead, walk.count - 1
                )
                if limit > here {
                    // By qc.weight, NOT by motion blur. Build 264 compared
                    // motionBlurPixels here, and best held-out PSNR fell
                    // 20.47 to 19.73 and SSIM 0.6668 to 0.6414. The least
                    // blurred candidate tends to sit further ahead in the
                    // window, so lastCenter jumps further, the walk reaches
                    // further into the capture with the same 120 keyframes
                    // (last held-out frame 421 became 515), and every region
                    // gets fewer views. Spreading the same 120 frames was
                    // measured as a loss offline too: 13.9 views per splat
                    // fell to 11.6 to 12.1. Sharper supervision has to come
                    // with MORE keyframes, not a thinner spread.
                    for candidate in walk[here...limit]
                    where candidate.qc.weight > best.qc.weight {
                        best = candidate
                    }
                }
            }
            chosen.append(best)
            let bestPose = prePass.refinedPose(for: best.index)
                ?? best.refinedPose ?? best.rawPose
            lastCenter = bestPose.center.simd
            lastForward = bestPose.forward.simd
            if chosen.count >= trainTarget { break }
        }

        if chosen.count < Swift.min(trainTarget, walk.count) {
            // The greedy pass was too strict for this walk (a scan taken from
            // one spot, for instance). Fall back to an even stride, which is
            // still a spread rather than a prefix.
            let step = Swift.max(walk.count / trainTarget, 1)
            chosen = Swift.stride(from: 0, to: walk.count, by: step).map { walk[$0] }
        }

        guard !heldOutCandidates.isEmpty,
              let low = chosen.map(\.index).min(), let high = chosen.map(\.index).max()
        else { return (chosen, nil) }
        let heldOut = pool.filter {
            heldOutCandidates.contains($0.index) && $0.index >= low && $0.index <= high
        }
        return (chosen, heldOut.isEmpty ? nil : heldOut)
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
