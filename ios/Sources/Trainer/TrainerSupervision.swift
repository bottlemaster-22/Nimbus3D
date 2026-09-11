//
//  TrainerSupervision.swift
//  Trainer
//
//  WHAT ONE KEYFRAME SUPERVISES, BUILT ON THE CPU ONCE PER USE.
//
//  The GPU depth-loss kernel is a flat loop over an array with no branching on
//  module state. That is a deliberate split, stated at the top of
//  `TrainerDepthSample` in TrainerGPULayouts.swift: everything that needs a
//  protocol call (`TrustField.weight`, `BackgroundModel.authority`,
//  `EdgeClassifier.map`) happens here, once, and lands in a plain array.
//
//  THE RULES THIS FILE ENFORCES (F3, F5, F6), each one a thing a generic 3DGS
//  trainer gets wrong:
//
//   1. DEPTH IS SUPERVISED ONLY AT THE NATIVE SAMPLES. 256x192 real returns,
//      not the 1920-wide map ARKit will hand you. Everything between them is
//      the interpolator's opinion, and training against an interpolator
//      teaches the model to reproduce the interpolator.
//
//   2. THE DILATED EDGE BAND IS ZEROED, NOT DOWN-WEIGHTED. Inside the band the
//      upsampled depth is WRONG rather than noisy, and averaging a wrong value
//      in at low weight still moves the surface.
//
//   3. THE AUTHORITY MAP DECIDES WHO OWNS A PIXEL. Where authority is at or
//      below the floor - past LiDAR range, at glass, at a saturated window, or
//      with no parallax - the pixel gets NO depth supervision and is left to
//      the background model. Pulling those pixels to a wrong distance is the
//      exact failure the authority map exists to prevent.
//
//   4. UNKNOWN IS MARKED EVEN WHEN IT CARRIES NO WEIGHT, because the unknown
//      mask is what switches late opacity binarization off for the Gaussians
//      that land there.
//
//   5. FREE SPACE IS EVIDENCE. A return at 2.4 m certifies the 2.35 m in front
//      of it as empty, which becomes a hinge: the rendered surface may not be
//      nearer than that, whatever the photometry would prefer. A no-return
//      sample borrows the bound from its valid neighbours, which is how the
//      space in front of a window is recovered without inventing anything
//      beyond the glass.
//

import Foundation
import simd

/// Everything one training step needs about one keyframe.
struct TrainerFrameSupervision {
    var frame: FrameID
    /// Ground-truth RGB at the render resolution, THREE BYTES per pixel
    /// (build 314). The loss kernel turns each byte into `Float(byte) / 255`
    /// through a table Swift computed, so it is the decoder's float exactly.
    var groundTruthBytes: [UInt8]
    /// The background CUBEMAP, flattened to three floats per texel, face-major
    /// then row-major. Empty when no background model is available, and
    /// `hasBackground` is then false rather than a black image being passed
    /// off as a far field.
    ///
    /// WAS the rasterised image, three floats per PIXEL. Rasterising it cost 6
    /// to 10 ms of Swift per iteration on the prefetch worker and then 4.67 MB
    /// of memcpy on the main thread; `trainer_background` does it on the GPU
    /// from these 72 KB instead.
    var backgroundTexels: [Float]
    var backgroundFaceSize: Int
    var hasBackground: Bool
    var depthSamples: [TrainerDepthSample]
    var renderSize: TrainerRenderSize
    /// The intrinsics rescaled to `renderSize`.
    var intrinsics: CameraIntrinsics
    var pose: Pose
    /// The frame's own QC weight, 0...1.
    var qcWeight: Float
    /// Mean per-pixel authority over this frame, for the log.
    var meanAuthority: Float
    /// How many native samples actually carry depth weight. Zero is a real and
    /// reportable answer for a frame shot entirely through a window.
    var supervisedSampleCount: Int
}

/// Builds `TrainerFrameSupervision`, and caches the decoded images so a
/// keyframe visited twice in a row is not decoded twice.
/// Builds ONE frame of supervision ahead, on a background queue, so the CPU
/// work for frame N+1 happens while the GPU is busy with frame N.
///
/// WHY THIS EXISTS, measured on the owner's phone at build 102 over a 3000
/// iteration run:
///
///     GPU actually executing      69.8 s   23.3 ms per iteration
///     CPU blocked waiting for it  74.1 s   24.7 ms per iteration
///     CPU building supervision    74.1 s   24.7 ms per iteration
///
/// Those last two are the same number, and they happen one after the other.
/// The loop builds a frame for 25 ms with the GPU idle, submits it, then stops
/// dead for 25 ms with the CPU idle. Neither device is ever busy at the same
/// time as the other, and the frame the CPU will need next has been known
/// since `order` was shuffled once at the start of the run.
///
/// Overlapping them is worth up to `min(24.7, 23.3)` ms of a 62.3 ms
/// iteration without changing a single line of arithmetic. It is also why four
/// separate attempts at making the GPU faster measured nothing: the GPU is 37%
/// of the run.
///
/// -----------------------------------------------------------------------
/// WHAT MAKES IT SAFE
/// -----------------------------------------------------------------------
/// `TrainerSupervisionBuilder` is not thread safe and is not made thread safe
/// here. Instead EXACTLY ONE THREAD TOUCHES IT AT A TIME, by construction:
///
///   * `start` is called only after the main thread has finished with the
///     builder for this iteration, and it waits for any previous worker first.
///   * `take` waits for the worker before returning, so the main thread never
///     reads the builder while the worker is inside it.
///   * `drain` does the same and throws the result away. It MUST be called
///     before anything that mutates the builder, which today means
///     `lowerLongEdge` by way of `applyBudgetChange`. The budget governor
///     lowers the render grid when the phone is hot or short of memory, and
///     that swaps the image cache outright; a worker running through the old
///     one at that moment is the one genuine hazard in this design.
///
/// There is no lock because there is no concurrent access, and
/// `DispatchWorkItem.wait()` is the ordering barrier that makes the handoff of
/// `built` well defined.
///
/// -----------------------------------------------------------------------
/// WHY IT IS KEYED, AND WHY A MISS IS FINE
/// -----------------------------------------------------------------------
/// A prefetch is a guess about which frame the next iteration will want, at
/// which iteration number, against which total. Every one of those can change:
/// the loop skips an iteration when a photo will not decode, and the governor
/// can shorten the run underneath it. So the result is keyed on all three and
/// handed over only on an exact match. A miss costs nothing but the work
/// already thrown away, and the caller simply builds the frame itself, exactly
/// as it did before this class existed. Correctness never depends on the guess
/// being right.
final class TrainerSupervisionPrefetch: @unchecked Sendable {

    /// A prefetched frame, or an admission that the guess was wrong.
    enum Outcome {
        /// The guess matched. The payload is what `build` returned, including
        /// `nil` for a photo that would not decode: that is a real answer and
        /// re-deriving it would just fail again more slowly.
        case hit(TrainerFrameSupervision?)
        /// No usable prefetch. Build it on this thread.
        case miss
    }

    /// What a prefetched frame is only valid for.
    private struct Key: Equatable {
        var frameIndex: FrameID
        var iteration: Int
        var totalIterations: Int
    }

    private let builder: TrainerSupervisionBuilder
    private let queue = DispatchQueue(
        label: "\(BrandConfig.bundleIdentifier).trainer.supervision-prefetch",
        qos: .userInitiated
    )
    private var work: DispatchWorkItem?
    private var key: Key?
    private var built: TrainerFrameSupervision?
    private var didBuild = false

    /// Seconds the WORKER spent building, which is the cost that moved off the
    /// critical path rather than disappeared. Reported next to the main
    /// thread's own supervision time so the two together show the overlap
    /// actually happening.
    private(set) var workerSeconds: Double = 0

    init(builder: TrainerSupervisionBuilder) {
        self.builder = builder
    }

    /// Starts building `frame` in the background. Waits for any previous
    /// worker first, so only one is ever in flight.
    func start(frame: CaptureFrame, iteration: Int, totalIterations: Int) {
        drain()
        let item = DispatchWorkItem { [self] in
            let from = CFAbsoluteTimeGetCurrent()
            built = builder.build(
                frame: frame, iteration: iteration, totalIterations: totalIterations
            )
            didBuild = true
            workerSeconds += CFAbsoluteTimeGetCurrent() - from
        }
        key = Key(
            frameIndex: frame.index, iteration: iteration,
            totalIterations: totalIterations
        )
        work = item
        queue.async(execute: item)
    }

    /// The prefetched frame if it is the one being asked for. Blocks until the
    /// worker is finished either way, so the builder is free afterwards.
    func take(
        frame: CaptureFrame, iteration: Int, totalIterations: Int
    ) -> Outcome {
        let wanted = Key(
            frameIndex: frame.index, iteration: iteration,
            totalIterations: totalIterations
        )
        work?.wait()
        work = nil
        let matched = (key == wanted) && didBuild
        let value = built
        key = nil
        built = nil
        didBuild = false
        return matched ? .hit(value) : .miss
    }

    /// Waits for anything in flight and discards it. Call before touching the
    /// builder from this thread.
    func drain() {
        work?.wait()
        work = nil
        key = nil
        built = nil
        didBuild = false
    }
}

/// BUILD 332: fills a builder's per-run frame cache in the background, one
/// frame at a time, before the phase that will use the builder begins. A
/// builder is not thread safe, so exactly one thread touches it at a time:
/// this one until `join()` returns, the phase's prefetch worker after. The
/// loop joins before the first `start` on that builder's prefetch.
///
/// Why: build 330 waited 1.1 s for supervision, almost all of it the first
/// visit to each frame of each level (a decode and a 49,152-sample pass),
/// while the CPU sat idle through the previous level's steps.
final class TrainerSupervisionPreload: @unchecked Sendable {
    private let builder: TrainerSupervisionBuilder
    private let frames: [CaptureFrame]
    private let queue = DispatchQueue(
        label: "\(BrandConfig.bundleIdentifier).trainer.supervision-preload",
        qos: .utility
    )
    private var work: DispatchWorkItem?
    private let lock = NSLock()
    private var cancelled = false
    private(set) var built = 0

    init(builder: TrainerSupervisionBuilder, frames: [CaptureFrame]) {
        self.builder = builder
        self.frames = frames
    }

    /// Starts building. Idempotent.
    func start() {
        lock.lock()
        defer { lock.unlock() }
        // Nothing to preload into when the cache is off (the memory governor
        // turns it off for the rest of the run).
        guard work == nil, builder.frameCacheEnabled else { return }
        // The governor cancels every preload, started or not, before a
        // budget change; one that had not started yet is still wanted
        // (build 336).
        cancelled = false
        let item = DispatchWorkItem { [self] in
            for frame in frames {
                lock.lock()
                let stop = cancelled
                lock.unlock()
                if stop { break }
                // The iteration is not part of what is cached.
                _ = builder.build(frame: frame, iteration: 0, totalIterations: 1)
                lock.lock()
                built += 1
                lock.unlock()
            }
        }
        work = item
        queue.async(execute: item)
    }

    /// Waits for the background pass to finish, then the builder is free.
    /// Nothing to wait for if `start` was never called.
    func join() {
        lock.lock()
        let item = work
        lock.unlock()
        item?.wait()
    }

    /// Stops after the frame in progress and waits for it.
    func cancelAndJoin() {
        lock.lock()
        cancelled = true
        lock.unlock()
        join()
    }
}

final class TrainerSupervisionBuilder {

    private let bundle: CaptureBundle
    private let prePass: PrePassResult
    private let ref: CaptureBundleRef
    private let settings: SmartLossSettings
    private let tuning: TrainerTuning

    private let trust: TwoScaleTrustField?
    private let authority: SmartAuthorityMap?
    private let edges: NativeDepthEdgeClassifier?
    private let background: DirectionalBackgroundModel?

    private let depthCache: SmartDepthCache

    private let depthWidth: Int
    private let depthHeight: Int
    private let nativeIntrinsics: CameraIntrinsics
    private let lidarMaxRange: Float

    /// Fixed once the first frame is decoded, so every frame in a run shares a
    /// pixel grid and the render buffers never have to be reshaped mid-loop.
    private(set) var renderSize: TrainerRenderSize?
    private(set) var renderIntrinsics: CameraIntrinsics?

    /// Set once a photo has come back a different shape from the camera that
    /// describes it, so that failure is reported once instead of once per
    /// iteration.
    private var loggedShapeMismatch = false

    /// The long edge the photos are decoded at. Not fixed for the life of the
    /// run: the budget governor can lower the render resolution mid-run to cool
    /// the phone or free memory, and the supervision grid has to follow it down
    /// (see `lowerLongEdge(to:)`).
    private var requestedLongEdge: Int

    // BUILD 310: A TRAINING FRAME IS DECODED AND SAMPLED ONCE PER RUN.
    //
    // Nothing a training frame's supervision is built from changes during a
    // run except the background cubemap: the photo, the pose, the intrinsics,
    // the trust slices, the authority map, the edge map and the QC weight are
    // all fixed once the pre-pass has finished, and `depthSamples` does not
    // read the iteration it is handed. Yet every step rebuilt it from scratch:
    // a JPEG decode, a float copy and a 49,152-sample loop, 7.7 ms of the
    // prefetch worker per step on build 292 (30.7 s a run), with the loop
    // still waiting 2.9 s for it, and every frame rebuilt about 37 times.
    //
    // So the first build of each frame keeps its photo bytes (3 per pixel,
    // 1.2 MB; since build 314 the bytes ARE the supervision) and its depth
    // samples, and later builds hand those back: a cached build is identical
    // to a fresh one. The background is still read fresh every time.
    //
    // Sized from the memory this process may still allocate, so the cache
    // can never be what pushes the budget governor into shedding splats.
    private struct CachedFrame {
        let rgbBytes: [UInt8]
        let samples: [TrainerDepthSample]
        let supervised: Int
    }
    private var frameCache: [FrameID: CachedFrame] = [:]
    private var frameCacheLimitBytes = 0
    /// Whether a build can still be kept: false once the governor has dropped
    /// the cache for memory.
    var frameCacheEnabled: Bool { frameCacheLimitBytes > 0 }
    private(set) var frameCacheBytes = 0
    /// Builds served from the cache, for the census.
    private(set) var frameCacheHits = 0

    init(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        longEdgePixels: Int,
        settings: SmartLossSettings,
        tuning: TrainerTuning,
        trust: TwoScaleTrustField?,
        authority: SmartAuthorityMap?,
        edges: NativeDepthEdgeClassifier?,
        background: DirectionalBackgroundModel?,
        /// Build 344: one depth cache for every builder of a run (the
        /// native depth maps are the same at every render size), sized to
        /// the frame set instead of 128 a builder.
        depthCache: SmartDepthCache? = nil
    ) {
        self.bundle = bundle
        self.prePass = prePass
        self.ref = ref
        self.settings = settings
        self.tuning = tuning
        self.trust = trust
        self.authority = authority
        self.edges = edges
        self.background = background
        self.requestedLongEdge = Swift.max(longEdgePixels, 64)

        depthWidth = Swift.max(bundle.settings.depthWidth, 1)
        depthHeight = Swift.max(bundle.settings.depthHeight, 1)
        lidarMaxRange = bundle.settings.lidarMaxRangeMeters
        nativeIntrinsics = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: depthWidth, depthHeight: depthHeight
        )

        // NO IMAGE CACHE ANY MORE (build 314): photos are decoded straight to
        // bytes by SmartImageLoader.loadRGB8 and kept in `frameCache` for the
        // run. (The 3-entry SmartImage cache that was here held 7.78 MB
        // floats a frame and hit 0 per cent over a 108-frame round robin.)
        //
        // THE DEPTH CACHE covers the cycle: a frame's samples are
        // depthWidth * depthHeight floats, about 196 KB at 256x192, so 128
        // frames is roughly 25 MB.
        self.depthCache = depthCache
            ?? SmartDepthCache(capacity: 128, sampleCount: depthWidth * depthHeight)

        // A third of whatever is free above 1.5 GB, at most 420 MB (108
        // training frames take about 300 MB). Nothing when the reading is a
        // stand-in rather than a measurement.
        let memory = DeviceMemoryFacts.probe()
        if !memory.availableIsEstimated {
            let reserve: UInt64 = 1_536 * 1_048_576
            let spare = memory.availableBytes > reserve ? memory.availableBytes - reserve : 0
            // 520 MB (build 342): 200 training frames at full size are about
            // 470 MB, and 338 peaked at 1,041 MB of a 1,463 MB ceiling.
            frameCacheLimitBytes = Int(Swift.min(spare / 3, 520 * 1_048_576))
        }
    }

    /// Adopts a smaller supervision grid part way through a run.
    ///
    /// The budget governor lowers the render resolution when the phone is hot
    /// or short of memory, and it resizes the GPU buffers itself. It cannot
    /// resize THIS, and this is where the pixel grid is actually decided: the
    /// grid is fixed from the first photo decoded and every later frame is
    /// measured against it. Without this call the next frame arrives at the old
    /// size, the training loop grows the buffers straight back to match it, and
    /// the cut that was made to cool the phone is undone a moment later, over
    /// and over, with a full buffer reallocation each time.
    ///
    /// Only downward, and only when it really is smaller: the governor's
    /// resolution ladder is one way, and re-decoding a run's photos larger part
    /// way through would spend memory at exactly the moment there is none.
    ///
    /// The cached decodes are dropped with the old grid, because they are at
    /// the old size. Returns whether anything changed, for the caller's log.
    @discardableResult
    func lowerLongEdge(to pixels: Int) -> Bool {
        let target = Swift.max(pixels, 64)
        guard target < requestedLongEdge else { return false }
        requestedLongEdge = target
        // Cleared, not recomputed: the next decodable frame fixes the new grid
        // and rescales the intrinsics to it, by the same path the first frame
        // of the run took.
        renderSize = nil
        renderIntrinsics = nil
        // At the old grid, like the decodes above.
        dropFrameCache(disable: false)
        return true
    }

    /// Lets go of every cached frame. `disable` also stops the cache refilling
    /// for the rest of the run: the memory governor calls it that way, since a
    /// cache that refilled after being dropped for memory would be dropped
    /// again at the next poll, and a decode storm every fifty iterations is
    /// worse than no cache (build 318).
    func dropFrameCache(disable: Bool) {
        frameCache.removeAll()
        frameCacheBytes = 0
        if disable { frameCacheLimitBytes = 0 }
    }

    /// The pose the trainer should render this frame from: the pre-pass's
    /// refined pose when there is one, otherwise the frame's own.
    func pose(for frame: CaptureFrame) -> Pose {
        prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
    }

    /// Decodes one keyframe and builds everything the GPU needs from it.
    /// Returns nil when the photo could not be read, which is logged and
    /// skipped rather than substituted with grey.
    func build(
        frame: CaptureFrame,
        iteration: Int,
        totalIterations: Int,
        includeDepthSamples: Bool = true
    ) -> TrainerFrameSupervision? {

        // Build 310: a training frame already built this run. See CachedFrame.
        if includeDepthSamples,
           let fixedSize = renderSize, let k = renderIntrinsics,
           let cached = frameCache[frame.index],
           cached.rgbBytes.count == fixedSize.pixelCount * 3 {
            frameCacheHits += 1
            let far = backgroundCubemap()
            return TrainerFrameSupervision(
                frame: frame.index,
                groundTruthBytes: cached.rgbBytes,
                backgroundTexels: far.texels,
                backgroundFaceSize: far.faceSize,
                hasBackground: far.present,
                depthSamples: cached.samples,
                renderSize: fixedSize,
                intrinsics: k,
                pose: pose(for: frame),
                qcWeight: TrainerMath.clamp(frame.qc.weight, 0, 1),
                meanAuthority: authority?.map(for: frame.index)?.meanAuthority ?? 0,
                supervisedSampleCount: cached.supervised
            )
        }

        guard let image = SmartImageLoader.loadRGB8(
            url: ref.url(forRelativePath: frame.imagePath), longEdge: requestedLongEdge
        ) else {
            TrainerLog.general.error(
                "Photo for frame \(frame.index) could not be read; that frame is skipped"
            )
            return nil
        }

        let size = TrainerRenderSize(width: image.width, height: image.height)
        if renderSize == nil {
            // The photo has to be the same SHAPE as the intrinsics that
            // describe it. `scaled(toWidth:height:)` scales fx and fy by
            // width and height separately, so a photo that came back turned a
            // quarter turn (a landscape sensor frame decoded as portrait,
            // which is what an EXIF orientation tag would do) is not caught by
            // that scale: it is absorbed into a camera that is wrong in both
            // axes, and the run trains against it without ever complaining.
            //
            // Capture writes its JPEGs with no orientation tag precisely so
            // that pixels, intrinsics and poses stay in one frame, so this
            // should never fire. If it ever does, the cause is upstream and
            // saying so is far better than quietly fitting a wrong camera.
            let sourceWidth = bundle.intrinsics.width
            let sourceHeight = bundle.intrinsics.height
            let sourceAspect = Float(sourceWidth) / Float(Swift.max(sourceHeight, 1))
            let decodedAspect = Float(size.width) / Float(Swift.max(size.height, 1))
            guard sourceAspect > 0,
                  abs(decodedAspect - sourceAspect) <= sourceAspect * 0.02
            else {
                // Said once. The grid stays unfixed while this is true, so
                // every frame of the run comes back here, and a line per
                // iteration would bury the one line that matters.
                if !loggedShapeMismatch {
                    loggedShapeMismatch = true
                    let frameIndex = frame.index
                    TrainerLog.general.error(
                        """
                        Photo for frame \(frameIndex) decoded \(size.width)x\(size.height), \
                        which is not the shape of the capture's \
                        \(sourceWidth)x\(sourceHeight) camera; those frames are skipped
                        """
                    )
                }
                return nil
            }
            renderSize = size
            renderIntrinsics = bundle.intrinsics.scaled(
                toWidth: image.width, height: image.height
            )
        }
        guard let fixedSize = renderSize, let k = renderIntrinsics else { return nil }
        // Every frame of a capture comes from the same camera at the same
        // resolution, so this should never fire. If it does, the frame is
        // skipped rather than sampled into the wrong grid.
        guard size == fixedSize else {
            TrainerLog.general.error(
                "Frame \(frame.index) decoded at \(size.width)x\(size.height), not \(fixedSize.width)x\(fixedSize.height); skipped"
            )
            return nil
        }

        let pixelCount = fixedSize.pixelCount
        guard image.rgb.count == pixelCount * 3 else { return nil }
        // The decoder's bytes, as they are. See TrainerFrameSupervision.
        let groundTruthBytes = image.rgb

        let framePose = pose(for: frame)

        let far = backgroundCubemap()
        let backgroundTexels = far.texels
        let backgroundFaceSize = far.faceSize
        let hasBackground = far.present

        // The held-out evaluation reads the photo, background, pose and
        // intrinsics and never a depth sample, so it asks for none.
        let samples = includeDepthSamples
            ? depthSamples(
                frame: frame,
                pose: framePose,
                renderIntrinsics: k,
                size: fixedSize,
                iteration: iteration,
                totalIterations: totalIterations
            )
            : []

        let supervised = samples.reduce(into: 0) { $0 += ($1.weight > 0 ? 1 : 0) }
        if includeDepthSamples {
            rememberFrame(frame.index, bytes: groundTruthBytes, samples: samples, supervised: supervised)
        }
        let meanAuthority = authority?.map(for: frame.index)?.meanAuthority ?? 0

        return TrainerFrameSupervision(
            frame: frame.index,
            groundTruthBytes: groundTruthBytes,
            backgroundTexels: backgroundTexels,
            backgroundFaceSize: backgroundFaceSize,
            hasBackground: hasBackground,
            depthSamples: samples,
            renderSize: fixedSize,
            intrinsics: k,
            pose: framePose,
            qcWeight: TrainerMath.clamp(frame.qc.weight, 0, 1),
            meanAuthority: meanAuthority,
            supervisedSampleCount: supervised
        )
    }

    /// The background cubemap, flattened, read fresh: the one input that
    /// changes during a run. The per-pixel rasterisation this used to do is
    /// `trainer_background` now.
    private func backgroundCubemap() -> (texels: [Float], faceSize: Int, present: Bool) {
        guard let background else { return ([], 0, false) }
        let map = background.cubemapSnapshot
        let texels = [Float](
            unsafeUninitializedCapacity: map.texels.count * 3
        ) { buffer, initialized in
            for i in 0..<map.texels.count {
                let t = map.texels[i]
                buffer[i * 3 + 0] = t.x
                buffer[i * 3 + 1] = t.y
                buffer[i * 3 + 2] = t.z
            }
            initialized = map.texels.count * 3
        }
        return (texels, map.faceSize, true)
    }

    /// Keeps a freshly built training frame for the rest of the run, if it
    /// fits.
    private func rememberFrame(
        _ id: FrameID, bytes: [UInt8], samples: [TrainerDepthSample], supervised: Int
    ) {
        guard frameCacheLimitBytes > 0, frameCache[id] == nil else { return }
        let cost = bytes.count + samples.count * MemoryLayout<TrainerDepthSample>.stride
        guard frameCacheBytes + cost <= frameCacheLimitBytes else { return }
        frameCache[id] = CachedFrame(rgbBytes: bytes, samples: samples, supervised: supervised)
        frameCacheBytes += cost
    }

    // MARK: - Depth supervision

    // swiftlint:disable:next function_body_length - one pass over the native
    // samples with every F3/F5/F6 rule visible in order; splitting it would
    // hide the ordering, which is the part that matters.
    private func depthSamples(
        frame: CaptureFrame,
        pose: Pose,
        renderIntrinsics k: CameraIntrinsics,
        size: TrainerRenderSize,
        iteration: Int,
        totalIterations: Int
    ) -> [TrainerDepthSample] {

        guard frame.depthPath != nil,
              let depth = depthCache.depth(for: frame, at: ref)
        else { return [] }

        let edgeMap = edges?.map(for: frame.index) ?? []
        // Hoisted out of the per-sample loop below, alongside edgeMap which
        // was already being fetched once.
        //
        // `SmartAuthorityMap.regime(frame:sampleIndex:)` and its sibling are
        // each `map(for: frame)?...`, and `map(for:)` takes a lock and does a
        // dictionary lookup before it can answer. Calling them per sample
        // meant two locked lookups for every one of up to 49,152 depth
        // samples in a frame, roughly 98,000 per iteration, all returning the
        // same object. `SmartFrameAuthority` is a Sendable struct whose own
        // accessors take no lock, so one lookup serves the whole frame.
        let frameAuthority = authority?.map(for: frame.index)
        // The two fallbacks are NOT the same value, and collapsing them into
        // one optional would have changed behaviour silently. The map-level
        // accessors returned `.far` and 0 when the MAP existed but had no
        // entry for this frame, and the call site's own `?? .near` / `?? 0.5`
        // applied only when there was no authority map at all. Two distinct
        // cases, two distinct answers, both preserved here.
        let noEntryRegime: SmartDepthRegime = (authority == nil) ? .near : .far
        let noEntryAuthority: Float = (authority == nil) ? 0.5 : 0
        let affine = trust?.depthAffine(frame: frame.index) ?? .identity
        // The frame's trust slices, fetched ONCE. The per-sample calls below
        // went through TwoScaleTrustField.weight and .sigmaMeters, each of
        // which takes the field's lock and then the reader's lock to reach
        // the same cached slice: about 5 lock round trips and a dozen
        // refcount operations per sample, 49,152 samples a frame, on the
        // prefetch worker that the loop waited 4.2 s for on build 264.
        // SmartFrameTrust computes from the same slices with the same
        // defaults for out-of-range and non-finite samples, so every weight
        // and sigma is identical. The two "no trust" cases stay distinct: no
        // trust field at all means 0.5, a field without its noise reader 0.
        let frameTrust = trust?.frameTrust(frame: frame.index)
        let qcWeight = TrainerMath.clamp(frame.qc.weight, 0, 1)
        let modeRadius = Swift.max(settings.modeWindowRadius, 1)
        let freeSpaceMargin = settings.freeSpaceBoundMarginMeters

        var samples: [TrainerDepthSample] = []
        samples.reserveCapacity(depthWidth * depthHeight / 2)

        for v in 0..<depthHeight {
            for u in 0..<depthWidth {
                let index = v * depthWidth + u

                // Where does this native sample land on the render grid? The
                // depth sensor is registered to the colour camera by ARKit, so
                // this is a pure rescale, not a reprojection.
                let px = Int((Float(u) + 0.5) * Float(size.width) / Float(depthWidth))
                let py = Int((Float(v) + 0.5) * Float(size.height) / Float(depthHeight))
                guard px >= 0, px < size.width, py >= 0, py < size.height else { continue }
                let pixelIndex = py * size.width + px

                let rawEdge: EdgeClass = index < edgeMap.count ? edgeMap[index] : .none
                let regime = frameAuthority?.regime(sampleIndex: index)
                    ?? noEntryRegime
                let authorityValue = frameAuthority?.value(sampleIndex: index)
                    ?? noEntryAuthority

                var sample = TrainerDepthSample()
                sample.pixelIndex = UInt32(pixelIndex)
                sample.edgeClass = UInt32(rawEdge.rawValue)

                let raw = depth[index]
                let hasReturn = raw > 0.05 && raw < lidarMaxRange
                let z = hasReturn ? affine.apply(raw) : 0

                // RULE 3 and RULE 4. Anything the authority map does not back
                // is UNKNOWN: it is marked so binarization is switched off for
                // whatever Gaussians cover it, and it carries no weight, so the
                // far field keeps ownership of the pixel.
                let belowAuthorityFloor = authorityValue <= settings.minimumAuthorityForDepth
                if !hasReturn || regime != .near || belowAuthorityFloor || rawEdge == .unknown {
                    sample.edgeClass = UInt32(EdgeClass.unknown.rawValue)
                    sample.weight = 0
                    sample.depth = 0
                    // RULE 5 for a no-return: borrow a bound from the valid
                    // neighbours, which certifies only the space in FRONT of
                    // the nearest known surface around it and nothing beyond.
                    if !hasReturn {
                        sample.freeSpaceBound = borrowedFreeSpaceBound(
                            depth: depth, u: u, v: v, margin: freeSpaceMargin, affine: affine
                        )
                    }
                    samples.append(sample)
                    continue
                }

                sample.depth = z

                // RULE 2. Zero inside the dilation band, and keep the class so
                // the GPU kernel takes the same branch.
                if rawEdge == .band {
                    sample.weight = 0
                    sample.freeSpaceBound = Swift.max(z - freeSpaceMargin, 0)
                    samples.append(sample)
                    continue
                }

                // The trust weight is inverse-variance and soft by
                // construction; the authority is the per-pixel right to be
                // believed; the QC weight is how good the photo was. All three
                // multiply, which is what the depth-sample contract asks for.
                let trustWeight: Float = trust == nil
                    ? 0.5
                    : (frameTrust?.weight(sampleIndex: index) ?? 0)
                var weight = trustWeight * authorityValue * qcWeight
                if rawEdge == .geometric {
                    // The sharpen half of the WHERE/WHAT split: a real depth
                    // step is the most informative sample in the frame.
                    weight *= tuning.geometricEdgeBoost
                }
                sample.weight = TrainerMath.clamp(weight, 0, 4)

                // Huber transition grows with range: 2 cm at 40 cm and 2 cm at
                // 5 m are not the same event.
                let sigma = frameTrust?.sigmaMeters(sampleIndex: index)
                sample.huberDelta = Swift.max(
                    sigma ?? (settings.huberDeltaMeters * Swift.max(z / 2, 1)),
                    0.004
                )

                // RULE 5 for a valid return.
                sample.freeSpaceBound = Swift.max(z - freeSpaceMargin, 0)

                // F4 bimodal supervision, and only at a real depth step. Away
                // from one, both modes are the sample itself, which makes the
                // bimodal term vanish exactly rather than approximately.
                if rawEdge == .geometric {
                    let modes = localModes(
                        depth: depth, u: u, v: v, radius: modeRadius, affine: affine
                    )
                    sample.mode0 = modes.near
                    sample.mode1 = modes.far
                } else {
                    sample.mode0 = z
                    sample.mode1 = z
                }

                samples.append(sample)
            }
        }
        return samples
    }

    /// The two local depth modes around a geometric edge: the mean of the
    /// nearer half and the mean of the farther half of the window's valid
    /// returns, split at the midpoint of its range. A two-cluster split rather
    /// than a histogram because the window is 25 samples and a histogram of 25
    /// values is noise with extra steps.
    private func localModes(
        depth: [Float],
        u: Int,
        v: Int,
        radius: Int,
        affine: SmartDepthAffine
    ) -> (near: Float, far: Float) {
        var values: [Float] = []
        values.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
        for dv in -radius...radius {
            for du in -radius...radius {
                let x = u + du, y = v + dv
                guard x >= 0, x < depthWidth, y >= 0, y < depthHeight else { continue }
                let raw = depth[y * depthWidth + x]
                guard raw > 0.05, raw < lidarMaxRange else { continue }
                values.append(affine.apply(raw))
            }
        }
        guard values.count >= 4 else {
            let centre = depth[v * depthWidth + u]
            let z = centre > 0 ? affine.apply(centre) : 0
            return (z, z)
        }
        let lo = values.min() ?? 0
        let hi = values.max() ?? 0
        let split = (lo + hi) * 0.5
        var nearSum: Float = 0, nearCount: Float = 0
        var farSum: Float = 0, farCount: Float = 0
        for value in values {
            if value <= split { nearSum += value; nearCount += 1 }
            else { farSum += value; farCount += 1 }
        }
        guard nearCount > 0, farCount > 0 else { return (lo, hi) }
        let near = nearSum / nearCount
        let far = farSum / farCount
        // The kernel only applies the bimodal term when mode1 > mode0, so a
        // window that turned out to be flat disables it by itself.
        return far > near ? (near, far) : (near, near)
    }

    /// The free-space bound for a sample with no return: the NEAREST valid
    /// return in the neighbourhood, less a margin. Nearest and not farthest,
    /// because the beam demonstrably reached at least as far as the closest
    /// thing around it and claiming more than that is inventing geometry.
    /// Returns 0, which disables the hinge, when nothing valid is nearby.
    private func borrowedFreeSpaceBound(
        depth: [Float],
        u: Int,
        v: Int,
        margin: Float,
        affine: SmartDepthAffine
    ) -> Float {
        let radius = 3
        var nearest = Float.greatestFiniteMagnitude
        for dv in -radius...radius {
            for du in -radius...radius {
                let x = u + du, y = v + dv
                guard x >= 0, x < depthWidth, y >= 0, y < depthHeight else { continue }
                let raw = depth[y * depthWidth + x]
                guard raw > 0.05, raw < lidarMaxRange else { continue }
                nearest = Swift.min(nearest, affine.apply(raw))
            }
        }
        guard nearest.isFinite else { return 0 }
        return Swift.max(nearest - margin, 0)
    }

}
