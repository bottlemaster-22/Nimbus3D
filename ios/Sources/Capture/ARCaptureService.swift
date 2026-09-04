//
//  ARCaptureService.swift
//  Capture
//
//  THE ARKIT SESSION. Everything else in this directory is a part; this is the
//  machine those parts are bolted into, and it is what `CaptureService` in
//  Core/Contracts.swift promises the rest of the app.
//
//  ---------------------------------------------------------------------------
//  WHICH CODE RUNS WHERE, AND WHY IT IS SPLIT THAT WAY
//  ---------------------------------------------------------------------------
//  There are exactly two serial contexts and nothing is shared outside them.
//
//  THE MAIN ACTOR is where ARKit delivers frames (the session's delegate queue
//  is left at its default, which is the main queue). An `ARFrame` is only
//  valid for the length of that callback and holding one starves ARKit's own
//  frame pool, so everything that has to touch the frame happens here and
//  nothing here keeps a reference to it afterwards. That is: the rate gate,
//  the native depth copy, the sharpness measurement, the QC record, window
//  mode, the keyframe decision, and the exposure bracket state machine. The
//  cost is roughly two milliseconds fifteen times a second.
//
//  THE WRITE QUEUE is a private serial queue and owns everything expensive:
//  the JPEG encode (10-15 ms, which is most of a 60 Hz delivery slot and can
//  never be on the delegate thread), the depth and confidence sidecars, the
//  append to `sensor_data/frames.jsonl`, the point cloud, and the coverage
//  field.
//
//  AT MOST ONE KEYFRAME IS EVER IN FLIGHT. That is deliberate and it is doing
//  three jobs at once: it bounds memory (a copied 1920x1440 frame is about
//  four megabytes, and an unbounded queue of them is how a capture app gets
//  killed), it means the writer can push back when the phone is busy, and it
//  is what lets `CaptureKeyframeSelector.didAcceptKeyframe` be called only
//  after the write actually succeeded, which is what that method's own
//  documentation asks for.
//
//  ---------------------------------------------------------------------------
//  WHAT THIS FILE REFUSES TO DO
//  ---------------------------------------------------------------------------
//   * It never reads `smoothedSceneDepth`. `frameSemantics` asks for
//     `.sceneDepth` only. The smoothed map is temporally filtered and
//     upsampled, which invents edges that were never measured, and F3's whole
//     argument is that depth supervision happens at the ~49k real samples per
//     frame and nowhere else.
//   * It never resets tracking to recover from an interruption. Resetting
//     throws away the map and orphans every frame already on disk, so the
//     session is allowed to relocalise instead and the user is told, in words,
//     to walk back to where they were.
//   * It never writes a number it did not measure. The camera-to-IMU offset is
//     nil when the sweep had no clear peak; revisit residuals are zero because
//     no alignment has run and the method field says so; depth dimensions come
//     from the map that actually arrived.
//

import ARKit
import AVFoundation
import Foundation
import QuartzCore
import UIKit
import simd

// =============================================================================
//  MARK: - Values that cross between the two contexts
//
//  Declared at file scope, not nested inside the service, because a type
//  nested in a `@MainActor` class can pick up that isolation and these have to
//  be readable from the write queue.
// =============================================================================

/// One accepted keyframe, complete, on its way to the disk.
struct CapturePendingKeyframe: Sendable {
    let index: FrameID
    let timestampSeconds: Double
    let wallClock: Date
    let image: CapturePixelBuffer
    let depth: CaptureDepthFrame?
    let pose: Pose
    let intrinsics: CameraIntrinsics
    let exposureDurationSeconds: Double
    let exposureOffsetEV: Float
    let iso: Float?
    let angularVelocity: SIMD3<Float>
    let qc: FrameQC
    let bracket: ExposureBracket
    /// Whether the coverage field is due an update from this frame's depth.
    let updatesCoverage: Bool
}

/// A frame that is not being written but whose depth still teaches the
/// coverage field something. Carries no pixels, so it costs a fraction of a
/// keyframe to hand over.
struct CapturePendingCoverage: Sendable {
    let depth: CaptureDepthFrame
    let pose: Pose
    let intrinsics: CameraIntrinsics
    let sharpness: Float
}

/// The frames written so far, in capture order.
///
/// Appended from the write queue, read once at the end of the session from
/// whichever context is assembling the bundle, and read for its count by the
/// HUD. One lock, held for a push or a copy.
final class CaptureRecordedFrames: @unchecked Sendable {

    private let lock = NSLock()
    private var frames: [CaptureFrame] = []

    func append(_ frame: CaptureFrame) {
        lock.lock()
        frames.append(frame)
        lock.unlock()
    }

    func snapshot() -> [CaptureFrame] {
        lock.lock()
        defer { lock.unlock() }
        return frames
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return frames.count
    }

    func removeAll() {
        lock.lock()
        frames.removeAll(keepingCapacity: false)
        lock.unlock()
    }
}

/// Feeds one frame's native depth samples into the coverage field.
///
/// A free function rather than a method so the write queue can call it without
/// touching the main-actor service at all.
func captureAccumulateCoverage(
    depth: CaptureDepthFrame,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    sharpness: Float,
    coverageField: CaptureCoverageField,
    meshStore: CaptureMeshStore
) {
    let cameraToWorld = pose.matrix.inverse
    let center = pose.center.simd
    let step = Swift.max(1, CaptureTuning.coverageDepthStride)

    var y = 0
    while y < depth.height {
        var x = 0
        while x < depth.width {
            guard
                let cameraPoint = depth.unprojectToCameraSpace(
                    x: x,
                    y: y,
                    intrinsics: intrinsics
                )
            else {
                x += step
                continue
            }
            let world = cameraToWorld * SIMD4<Float>(cameraPoint, 1)
            let point = SIMD3<Float>(world.x, world.y, world.z)
            coverageField.observe(
                worldPosition: point,
                cameraCenter: center,
                sharpness: sharpness,
                surfaceClass: meshStore.surfaceClass(at: point)
            )
            x += step
        }
        y += step
    }
}

// =============================================================================
//  MARK: - The service
// =============================================================================

/// Runs the ARKit session and writes one capture bundle.
///
/// `@MainActor` because `CaptureService` is: it owns an `ARSession`, drives
/// the HUD, and is fed by delegate callbacks that arrive on the main queue.
@MainActor
public final class ARCaptureService: NSObject, CaptureService {

    /// The instance the app shell registers and the capture screen talks to.
    ///
    /// A single shared session is not a style choice: two `ARSession`s
    /// competing for the same camera is a documented way to get neither.
    public static let shared = ARCaptureService()

    // MARK: - Parts the capture screen also needs

    /// The three coverage channels, live. The HUD's Metal renderer samples
    /// this directly.
    let coverageField = CaptureCoverageField()

    /// ARKit's classified scene mesh, copied out of ARKit and safe to read.
    let meshStore = CaptureMeshStore()

    /// Sound, speech and haptics. Owned here rather than by the screen so
    /// guidance keeps working while the user is looking at the room.
    let guidance = CaptureGuidanceEngine()

    /// Exposure locks and the dark bracket.
    let exposureController = CaptureExposureController()

    /// Called on the main actor for every delivered frame, before anything
    /// else touches it, so the HUD's renderer can pull its textures out while
    /// the frame is alive.
    ///
    /// The closure MUST NOT retain the frame past its own return: ARKit's
    /// frame pool is small and a held frame is one the tracker cannot reuse.
    var liveFrameObserver: (@MainActor (ARFrame) -> Void)?

    // MARK: - Parts nobody else touches

    private let session = ARSession()
    private let gyro = CaptureGyroSampler()
    private let anchorRecorder = CaptureAnchorRecorder()
    private let sharpnessMeter = CaptureSharpnessMeter()
    private let qcEvaluator = CaptureQCEvaluator()
    private let keyframeSelector = CaptureKeyframeSelector()
    private let windowMode = CaptureWindowMode()
    private let pointCloud = CapturePointCloudAccumulator()
    private let recorded = CaptureRecordedFrames()
    private lazy var timeOffsetCalibrator = CaptureTimeOffsetCalibrator(gyro: gyro)

    private let writeQueue = DispatchQueue(
        label: "\(BrandConfig.bundleIdentifier).capture.write",
        qos: .userInitiated
    )

    // MARK: - Session state

    private enum Phase: Equatable {
        case idle
        case recording
        /// A guard tripped (heat, storage, a session error). Frames are no
        /// longer accepted; what was recorded is still on disk and waiting to
        /// be finished.
        case halted
        case finishing
        case finished
    }

    private var phase: Phase = .idle

    private var folder: CaptureScanFolder?
    private var writer: CaptureFrameWriter?
    private var displayName: String = ""

    private var sessionStartTimestamp: TimeInterval?
    private var cachedIntrinsics: CameraIntrinsics?
    private var nextFrameIndex: FrameID = 0
    private var isKeyframeInFlight = false
    private var isCoverageInFlight = false
    private var didStartExposureController = false

    private var lastTrackingQuality: TrackingQuality = .notAvailable
    private var lastQC: FrameQC?
    private var lastDepthSize: (width: Int, height: Int)?
    private var lastCameraTransform: simd_float4x4?
    private var lastCoverageDispatch: TimeInterval = 0
    private var bracketRequestedAt: TimeInterval?
    private var lastBracketCadence = CaptureTuning.bracketEveryNKeyframes

    private var frameCount = 0
    private var bytesWritten: Int64 = 0
    private var lastCoverageRecompute: CFTimeInterval = 0
    private var lastDiskCheck: CFTimeInterval = 0
    private var liveTickTask: Task<Void, Never>?

    /// Why the capture stopped itself, in one plain sentence, or nil while all
    /// is well. The capture screen watches this and offers to finish.
    public private(set) var stopReason: String?

    /// Whether frames are being recorded right now.
    public var isRecording: Bool { phase == .recording }

    /// Whether there is a scan open that has not been finished or thrown away.
    public var hasOpenScan: Bool { phase == .recording || phase == .halted }

    // MARK: - Live state stream

    private var liveStateStream: AsyncStream<CaptureLiveState>?
    private var liveStateContinuation: AsyncStream<CaptureLiveState>.Continuation?

    /// Live HUD state, at `CaptureTuning.liveStateHz`.
    ///
    /// The stream is created on first access and finished when the session
    /// stops, exactly as the contract describes. Asking again after that
    /// starts a fresh one for the next capture, so a screen that holds on to
    /// the service across two scans does not have to know anything about the
    /// lifetime.
    ///
    /// Buffering is newest-one: a HUD that falls behind wants the current
    /// state, never a backlog of stale ones.
    public var liveState: AsyncStream<CaptureLiveState> {
        if let liveStateStream { return liveStateStream }
        let (stream, continuation) = AsyncStream.makeStream(
            of: CaptureLiveState.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        liveStateStream = stream
        liveStateContinuation = continuation
        return stream
    }

    // MARK: - Start

    /// Creates the scan folder, opens the live log, and starts ARKit.
    ///
    /// Returns as soon as there is a folder to point at, before a single frame
    /// has landed, so the screen has somewhere real to show immediately.
    public func start(displayName: String) async throws -> CaptureBundleRef {
        guard phase == .idle || phase == .finished else {
            throw NimbusError.captureFailed("A scan is already running.")
        }

        let configuration = try Self.makeConfiguration()
        try await Self.requireCameraPermission()

        guard CaptureScanFolder.hasRoomToContinue() else {
            throw NimbusError.captureFailed(
                Self.outOfSpaceMessage(freeBytes: CaptureScanFolder.availableDiskBytes())
            )
        }

        let folder = try CaptureScanFolder.create()
        let writer = CaptureFrameWriter(folder: folder)
        do {
            try writer.open()
        } catch {
            folder.deleteEverything()
            throw NimbusError.captureFailed(
                "The scan's notes file could not be created, so nothing would "
                    + "have been recorded. Nothing was saved."
            )
        }

        self.folder = folder
        self.writer = writer
        self.displayName = Self.resolvedDisplayName(displayName, startedAt: folder.startedAt)

        resetForNewSession()

        session.delegate = self
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        gyro.start()
        guidance.start()

        UIApplication.shared.isIdleTimerDisabled = true
        phase = .recording
        startLiveTick()

        CaptureLog.session.notice(
            "Capture started: \(folder.scanID, privacy: .public)"
        )
        return CaptureBundleRef(scanID: folder.scanID, rootURL: folder.root)
    }

    private func resetForNewSession() {
        coverageField.reset()
        meshStore.reset()
        anchorRecorder.reset()
        sharpnessMeter.reset()
        qcEvaluator.reset()
        keyframeSelector.reset()
        windowMode.reset()
        pointCloud.reset()
        recorded.removeAll()
        timeOffsetCalibrator.reset()
        guidance.reset()

        sessionStartTimestamp = nil
        cachedIntrinsics = nil
        nextFrameIndex = 0
        isKeyframeInFlight = false
        isCoverageInFlight = false
        didStartExposureController = false
        lastTrackingQuality = .notAvailable
        lastQC = nil
        lastDepthSize = nil
        lastCameraTransform = nil
        lastCoverageDispatch = 0
        bracketRequestedAt = nil
        lastBracketCadence = CaptureTuning.bracketEveryNKeyframes
        frameCount = 0
        bytesWritten = 0
        lastCoverageRecompute = 0
        lastDiskCheck = CACurrentMediaTime()
        stopReason = nil
    }

    // MARK: - Finish

    /// Stops the session, re-reads the anchors, writes everything, and returns
    /// the index.
    public func finish() async throws -> CaptureBundle {
        guard phase == .recording || phase == .halted else {
            throw NimbusError.captureFailed("There is no scan running to finish.")
        }
        guard let folder, let writer else {
            throw NimbusError.captureFailed("There is no scan running to finish.")
        }
        phase = .finishing
        liveTickTask?.cancel()
        liveTickTask = nil

        // THE ANCHOR RE-READ HAPPENS BEFORE THE PAUSE. After `pause()` there is
        // no current frame and no anchors left to read, and this list is half
        // of the free drift measurement (F8).
        let finalAnchors = anchorRecorder.rereadAnchors(from: session)
        let sessionAnchors = anchorRecorder.sessionAnchors

        session.pause()
        session.delegate = nil
        gyro.stop()
        guidance.stop()
        UIApplication.shared.isIdleTimerDisabled = false

        let timeOffset = timeOffsetCalibrator.result()
        if timeOffset == nil, let reason = timeOffsetCalibrator.rejectionReason {
            CaptureLog.session.notice("\(reason, privacy: .public)")
        }

        let snapshots = meshStore.allSnapshots
        let settings = currentSettings()
        let meshBounds = meshStore.bounds()
        let name = displayName
        let intrinsics = cachedIntrinsics

        guard let intrinsics else {
            // Not one frame arrived. There is nothing to index and an empty
            // folder is litter, so it goes.
            writer.close()
            folder.deleteEverything()
            phase = .finished
            finishLiveState()
            throw NimbusError.captureFailed(
                "The camera never sent a picture, so there was nothing to save."
            )
        }

        let frames = recorded.snapshot()
        let pointCloud = self.pointCloud

        do {
            let bundle: CaptureBundle = try await withCheckedThrowingContinuation {
                continuation in
                // Queued behind every pending write, because the queue is
                // serial: by the time this runs, every file the bundle is
                // about to describe is already on disk.
                writeQueue.async {
                    do {
                        let revisits = CaptureRevisitDetector.detect(frames: frames)
                        let bundle = try CaptureBundleWriter.write(
                            folder: folder,
                            displayName: name,
                            intrinsics: intrinsics,
                            settings: settings,
                            frames: frames,
                            anchorsDuringSession: sessionAnchors,
                            anchorsAtEndOfSession: finalAnchors,
                            meshSnapshots: snapshots,
                            pointCloud: pointCloud,
                            sceneBoundsFallback: meshBounds,
                            revisitPairs: revisits,
                            cameraToIMUTimeOffsetSeconds: timeOffset
                        )
                        writer.close()
                        continuation.resume(returning: bundle)
                    } catch {
                        writer.close()
                        continuation.resume(throwing: error)
                    }
                }
            }
            phase = .finished
            finishLiveState()
            return bundle
        } catch {
            // The frames, the depth and the live log are all still on disk;
            // only the index failed. The session is over either way, so the
            // service goes back to a state a new scan can start from rather
            // than wedging on the failure.
            phase = .finished
            finishLiveState()
            throw error
        }
    }

    /// Stops and deletes everything written so far.
    ///
    /// Only a scan that is still open can be thrown away. Once `finish` has
    /// written the bundle the scan belongs to the library, and deleting it is
    /// the library's business, not a stray call to this method.
    public func cancel() async {
        guard phase == .recording || phase == .halted else { return }
        phase = .finishing
        liveTickTask?.cancel()
        liveTickTask = nil

        session.pause()
        session.delegate = nil
        gyro.stop()
        guidance.stop()
        UIApplication.shared.isIdleTimerDisabled = false

        let writer = self.writer
        let folder = self.folder
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            // Behind the pending writes again, so nothing is still creating
            // files in a folder that is being deleted.
            writeQueue.async {
                writer?.close()
                folder?.deleteEverything()
                continuation.resume()
            }
        }

        self.writer = nil
        self.folder = nil
        phase = .idle
        finishLiveState()
        CaptureLog.session.notice("Capture cancelled and its folder deleted.")
    }

    private func finishLiveState() {
        liveStateContinuation?.finish()
        liveStateContinuation = nil
        liveStateStream = nil
    }

    // MARK: - The per-frame pipeline (main actor)

    fileprivate func handle(frame: ARFrame) {
        guard phase == .recording else { return }

        let timestamp = frame.timestamp
        let camera = frame.camera
        let quality = TrackingQuality(camera.trackingState)

        if sessionStartTimestamp == nil { sessionStartTimestamp = timestamp }
        if !didStartExposureController {
            // ARKit only publishes the capture device once it owns it, which
            // is not true until a frame has actually been delivered.
            didStartExposureController = true
            exposureController.sessionDidStart()
        }

        lastTrackingQuality = quality
        lastCameraTransform = camera.transform
        exposureController.observeTracking(quality)
        anchorRecorder.noteFrameIndex(nextFrameIndex)

        // The HUD gets the frame first and while it is still alive.
        liveFrameObserver?(frame)

        updateBracketState(timestamp: timestamp, quality: quality)

        guard keyframeSelector.shouldEvaluate(at: timestamp) else { return }

        let intrinsics = resolvedIntrinsics(from: camera)
        let pose = Pose.fromARKitCameraTransform(camera.transform)
        let exposureDuration = camera.exposureDuration

        // Fed before the in-flight gate below, because it costs two quaternion
        // multiplies and the clock measurement is only as good as the number
        // of samples it gets.
        timeOffsetCalibrator.observe(timestamp: timestamp, rotation: pose.rotation)

        // Nothing to gain from measuring a frame that cannot be written yet:
        // the writer is still busy with the last one.
        guard !isKeyframeInFlight else { return }

        // The gyro is asked for the exposure MIDPOINT, not for "now". ARKit
        // stamps a frame at capture time and does not say which edge of the
        // shutter that is, so half the exposure is the best available guess
        // and the leftover is exactly what the camera-to-IMU sweep measures.
        let angularVelocity = gyro.angularVelocity(at: timestamp + exposureDuration / 2)

        // NATIVE depth. Not `smoothedSceneDepth`, not the upsampled map.
        let depth = frame.sceneDepth.flatMap { CaptureDepthFrame(depthData: $0) }
        if let depth { lastDepthSize = (depth.width, depth.height) }

        let sharpness = sharpnessMeter.sharpness(of: frame.capturedImage)
        let bracket = exposureController.bracketClass(forFrameAt: timestamp)

        let qc = qcEvaluator.evaluate(
            angularVelocity: angularVelocity,
            exposureDurationSeconds: exposureDuration,
            exposureOffsetEV: camera.exposureOffset,
            sharpness: sharpness,
            depthValidFraction: depth?.validFraction ?? 0,
            trackingQuality: quality,
            isBracketed: bracket == .darker
        )
        lastQC = qc

        if let depth {
            _ = windowMode.update(
                depth: depth,
                pose: pose,
                intrinsics: intrinsics,
                meshStore: meshStore,
                now: timestamp
            )
        }

        let thermal = ThermalLevel(ProcessInfo.processInfo.thermalState)
        let decision = keyframeSelector.decide(
            pose: pose,
            timestamp: timestamp,
            qc: qc,
            isBracketRequested: bracket == .darker,
            thermalLevel: thermal
        )

        let coverageIsDue = isCoverageDue(at: timestamp)

        if decision != nil {
            acceptKeyframe(
                frame: frame,
                timestamp: timestamp,
                pose: pose,
                intrinsics: intrinsics,
                depth: depth,
                exposureDuration: exposureDuration,
                exposureOffsetEV: camera.exposureOffset,
                angularVelocity: angularVelocity,
                qc: qc,
                bracket: bracket,
                updatesCoverage: coverageIsDue
            )
        } else if coverageIsDue, let depth, !isCoverageInFlight {
            dispatchCoverage(
                depth: depth,
                pose: pose,
                intrinsics: intrinsics,
                sharpness: sharpness,
                timestamp: timestamp
            )
        }

        requestBracketIfDue(timestamp: timestamp, qc: qc, quality: quality)
    }

    private func acceptKeyframe(
        frame: ARFrame,
        timestamp: TimeInterval,
        pose: Pose,
        intrinsics: CameraIntrinsics,
        depth: CaptureDepthFrame?,
        exposureDuration: Double,
        exposureOffsetEV: Float,
        angularVelocity: SIMD3<Float>,
        qc: FrameQC,
        bracket: ExposureBracket,
        updatesCoverage: Bool
    ) {
        guard
            let writer,
            let folder,
            let sessionStartTimestamp,
            let image = CapturePixelBuffer(copying: frame.capturedImage)
        else {
            // A frame whose pixels could not be copied is simply not written.
            // It is never recorded with a path to a file that does not exist.
            CaptureLog.writer.error(
                "A camera frame could not be copied and was skipped."
            )
            return
        }

        let index = nextFrameIndex
        nextFrameIndex &+= 1
        isKeyframeInFlight = true
        lastCoverageDispatch = updatesCoverage ? timestamp : lastCoverageDispatch

        let payload = CapturePendingKeyframe(
            index: index,
            timestampSeconds: timestamp,
            wallClock: folder.wallClock(
                forFrameTimestamp: timestamp,
                sessionStartTimestamp: sessionStartTimestamp
            ),
            image: image,
            depth: depth,
            pose: pose,
            intrinsics: intrinsics,
            exposureDurationSeconds: exposureDuration,
            exposureOffsetEV: exposureOffsetEV,
            iso: exposureController.currentISO,
            angularVelocity: angularVelocity,
            qc: qc,
            bracket: bracket,
            updatesCoverage: updatesCoverage
        )

        let coverageField = self.coverageField
        let meshStore = self.meshStore
        let pointCloud = self.pointCloud
        let recorded = self.recorded

        writeQueue.async { [weak self] in
            // The failure crosses back as text rather than as an `Error`: an
            // error existential is not `Sendable`, and the only thing the main
            // actor does with it is put it in the log and in a sentence.
            var failureText: String?
            var bytes: Int64 = 0
            do {
                let stamp = writer.uniqueStamp(forWallClock: payload.wallClock)
                let written = try writer.writeFiles(
                    stamp: stamp,
                    image: payload.image,
                    depth: payload.depth
                )
                let record = CaptureFrame(
                    index: payload.index,
                    timestampSeconds: payload.timestampSeconds,
                    imagePath: written.imagePath,
                    depthPath: written.depthPath,
                    confidencePath: written.confidencePath,
                    rawPose: payload.pose,
                    exposureDurationSeconds: payload.exposureDurationSeconds,
                    exposureOffsetEV: payload.exposureOffsetEV,
                    iso: payload.iso,
                    angularVelocity: Vector3(payload.angularVelocity),
                    qc: payload.qc,
                    bracket: payload.bracket
                )
                // The log line goes last, once the files it names are real.
                writer.appendLogLine(for: record)
                recorded.append(record)

                if let depth = payload.depth {
                    pointCloud.add(
                        depth: depth,
                        pose: payload.pose,
                        intrinsics: payload.intrinsics,
                        image: payload.image
                    )
                    if payload.updatesCoverage {
                        captureAccumulateCoverage(
                            depth: depth,
                            pose: payload.pose,
                            intrinsics: payload.intrinsics,
                            sharpness: payload.qc.sharpness,
                            coverageField: coverageField,
                            meshStore: meshStore
                        )
                    }
                }
                bytes = writer.bytesWritten
            } catch {
                failureText = error.localizedDescription
            }

            let writtenBytes = bytes
            let problem = failureText
            Task { @MainActor in
                self?.keyframeDidFinish(
                    payload: payload,
                    bytesWritten: writtenBytes,
                    failureText: problem
                )
            }
        }
    }

    private func keyframeDidFinish(
        payload: CapturePendingKeyframe,
        bytesWritten: Int64,
        failureText: String?
    ) {
        isKeyframeInFlight = false

        if let failureText {
            // The pixels never landed, so the selector's reference pose is
            // deliberately left where it was: the next frame is still a
            // candidate for the keyframe this one failed to be.
            CaptureLog.writer.error(
                "Keyframe write failed: \(failureText, privacy: .public)"
            )
            if CaptureScanFolder.hasRoomToContinue() {
                halt(
                    reason:
                        "A shot could not be saved to this iPhone, so the scan "
                        + "stopped rather than carrying on with holes in it. "
                        + "Everything up to that point is saved."
                )
            } else {
                halt(
                    reason: Self.outOfSpaceMessage(
                        freeBytes: CaptureScanFolder.availableDiskBytes()
                    )
                )
            }
            return
        }

        keyframeSelector.didAcceptKeyframe(
            pose: payload.pose,
            timestamp: payload.timestampSeconds
        )
        exposureController.keyframeWasWritten()
        frameCount += 1
        self.bytesWritten = bytesWritten

        if payload.bracket == .darker {
            // One dark frame is the whole point of a bracket. Restore
            // immediately so the next frame is exposed like its neighbours.
            exposureController.endBracket(at: payload.timestampSeconds)
            bracketRequestedAt = nil
        }
    }

    private func dispatchCoverage(
        depth: CaptureDepthFrame,
        pose: Pose,
        intrinsics: CameraIntrinsics,
        sharpness: Float,
        timestamp: TimeInterval
    ) {
        isCoverageInFlight = true
        lastCoverageDispatch = timestamp

        let payload = CapturePendingCoverage(
            depth: depth,
            pose: pose,
            intrinsics: intrinsics,
            sharpness: sharpness
        )
        let coverageField = self.coverageField
        let meshStore = self.meshStore

        writeQueue.async { [weak self] in
            captureAccumulateCoverage(
                depth: payload.depth,
                pose: payload.pose,
                intrinsics: payload.intrinsics,
                sharpness: payload.sharpness,
                coverageField: coverageField,
                meshStore: meshStore
            )
            Task { @MainActor in
                self?.isCoverageInFlight = false
            }
        }
    }

    /// Coverage is fed from any evaluated frame, not just written ones, so the
    /// paint on the wall keeps up with a user who is standing still and
    /// panning at a wall they have already photographed.
    private func isCoverageDue(at timestamp: TimeInterval) -> Bool {
        timestamp - lastCoverageDispatch >= 1.0 / CaptureTuning.coverageUpdateHz
    }

    private func resolvedIntrinsics(from camera: ARCamera) -> CameraIntrinsics {
        if let cachedIntrinsics { return cachedIntrinsics }
        let matrix = camera.intrinsics
        let value = CameraIntrinsics(
            width: Int(camera.imageResolution.width),
            height: Int(camera.imageResolution.height),
            fx: matrix.columns.0.x,
            fy: matrix.columns.1.y,
            cx: matrix.columns.2.x,
            cy: matrix.columns.2.y
        )
        cachedIntrinsics = value
        return value
    }

    // MARK: - Exposure bracketing (F5)

    /// Keeps the bracket state machine honest, once per delivered frame.
    private func updateBracketState(timestamp: TimeInterval, quality: TrackingQuality) {
        let isDark = exposureController.bracketClass(forFrameAt: timestamp) == .darker

        // BACK OUT IMMEDIATELY IF TRACKING SLIPS. A dark frame during a
        // tracking wobble is the one way this feature can damage a scan, and a
        // scan that tracks is worth more than a scan with good window pixels.
        if isDark, quality != .normal {
            exposureController.endBracket(at: timestamp)
            bracketRequestedAt = nil
            CaptureLog.exposure.notice(
                "Tracking slipped during a dark frame, so the exposure went straight back."
            )
            return
        }

        // A request that never arrived would otherwise block every future one.
        if let requestedAt = bracketRequestedAt,
            !isDark,
            timestamp - requestedAt >= CaptureTuning.bracketApplyTimeoutSeconds
        {
            exposureController.abandonPendingBracket(at: timestamp)
            bracketRequestedAt = nil
        }
    }

    private func requestBracketIfDue(
        timestamp: TimeInterval,
        qc: FrameQC,
        quality: TrackingQuality
    ) {
        guard bracketRequestedAt == nil else { return }
        let cadence = windowMode.bracketEveryNKeyframes
        lastBracketCadence = cadence
        guard
            exposureController.shouldBracketNextKeyframe(
                everyN: cadence,
                angularSpeed: qc.angularSpeedRadPerSec,
                trackingQuality: quality,
                now: timestamp
            )
        else { return }
        exposureController.beginBracket()
        bracketRequestedAt = timestamp
    }

    // MARK: - Live HUD

    private func startLiveTick() {
        liveTickTask?.cancel()
        let interval = UInt64(1_000_000_000 / Swift.max(1, CaptureTuning.liveStateHz))
        liveTickTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: interval)
                guard !Task.isCancelled else { return }
                self?.publishLiveState()
            }
        }
    }

    private func publishLiveState() {
        guard phase == .recording || phase == .halted, let folder else { return }
        let now = CACurrentMediaTime()

        if phase == .recording {
            scheduleCoverageRecompute(now: now)
            checkThermalAndStorage(now: now)
        }

        let coverage = coverageField.fraction
        let blur = lastQC?.motionBlurPixels ?? 0
        let thermal = ThermalLevel(ProcessInfo.processInfo.thermalState)

        if phase == .recording {
            guidance.update(
                blurPixels: blur,
                trackingQuality: lastTrackingQuality,
                worstChannel: coverageField.worstChannel,
                windowHint: windowMode.guidanceHint,
                coverageFraction: coverage,
                isDone: coverageField.isDone,
                now: now
            )
        }

        liveStateContinuation?.yield(
            CaptureLiveState(
                frameCount: frameCount,
                elapsedSeconds: Date().timeIntervalSince(folder.startedAt),
                coverageFraction: coverage,
                motionBlurPixels: blur,
                trackingQuality: lastTrackingQuality,
                thermalLevel: thermal,
                bytesWritten: bytesWritten,
                guidanceHint: stopReason ?? guidance.currentHint
            )
        )
    }

    /// The coverage percentage walks every voxel, so it runs on the write
    /// queue on its own slow schedule rather than on the HUD's tick.
    private func scheduleCoverageRecompute(now: CFTimeInterval) {
        guard now - lastCoverageRecompute
            >= 1.0 / CaptureTuning.coverageFractionRecomputeHz
        else { return }
        lastCoverageRecompute = now
        let field = coverageField
        writeQueue.async { field.recomputeFraction() }
    }

    // MARK: - Guards

    /// Heat and storage, both checked honestly and both ending the same way:
    /// stop, keep what is recorded, and say why in a sentence.
    private func checkThermalAndStorage(now: CFTimeInterval) {
        let thermal = ThermalLevel(ProcessInfo.processInfo.thermalState)
        if thermal >= CaptureTuning.thermalStopAt {
            halt(
                reason:
                    "Your iPhone got too warm to keep scanning safely. "
                    + "Everything recorded so far is saved. Let it cool down "
                    + "for a few minutes and carry on with a new scan."
            )
            return
        }

        guard now - lastDiskCheck >= CaptureTuning.diskCheckIntervalSeconds else { return }
        lastDiskCheck = now
        guard !CaptureScanFolder.hasRoomToContinue() else { return }
        halt(reason: Self.outOfSpaceMessage(freeBytes: CaptureScanFolder.availableDiskBytes()))
    }

    /// Stops recording without throwing anything away. The scan is left open
    /// so the screen can finish it and the user still gets their files.
    private func halt(reason: String) {
        guard phase == .recording else { return }
        phase = .halted
        stopReason = reason
        CaptureLog.session.notice("Capture halted: \(reason, privacy: .public)")
        publishLiveState()
    }

    private static func outOfSpaceMessage(freeBytes: Int64?) -> String {
        let needed = CaptureTuning.minFreeDiskBytes / (1024 * 1024)
        if let freeBytes {
            let freeMB = freeBytes / (1024 * 1024)
            return "This iPhone is nearly full: about \(freeMB) MB free, and a "
                + "scan needs at least \(needed) MB of room to finish safely. "
                + "Everything recorded so far is saved. Delete something and "
                + "try again."
        }
        return "This iPhone is nearly full, so the scan stopped before it ran "
            + "out of room. Everything recorded so far is saved."
    }

    // MARK: - Things the capture screen can ask for

    var isSoundEnabled: Bool {
        get { guidance.isSoundEnabled }
        set { guidance.isSoundEnabled = newValue }
    }

    var isHapticsEnabled: Bool {
        get { guidance.isHapticsEnabled }
        set { guidance.isHapticsEnabled = newValue }
    }

    var isExposureLocked: Bool { exposureController.isExposureLocked }
    var isWhiteBalanceLocked: Bool { exposureController.isWhiteBalanceLocked }
    var isBracketingAvailable: Bool { exposureController.isBracketingAvailable }

    var isWindowModeActive: Bool { windowMode.isActive }
    var windowShouldStandBack: Bool { windowMode.shouldStandBack }
    var windowCentreDistanceMeters: Float? { windowMode.centreDistanceMeters }

    var coverageFraction: Float { coverageField.fraction }
    var coverageIsDone: Bool { coverageField.isDone }
    var countedCoverageVoxels: Int { coverageField.countedVoxelCount }
    var currentScanID: ScanID? { folder?.scanID }

    /// Holds exposure (and white balance with it) for the rest of the session.
    ///
    /// - Returns: false when this iPhone will not hand the camera's exposure
    ///   over, which is reported to the user rather than swallowed.
    @discardableResult
    func lockExposureAndWhiteBalance() -> Bool {
        let exposure = exposureController.setExposureLocked(true)
        let whiteBalance = exposureController.setWhiteBalanceLocked(true)
        return exposure && whiteBalance
    }

    @discardableResult
    func unlockExposureAndWhiteBalance() -> Bool {
        let exposure = exposureController.setExposureLocked(false)
        let whiteBalance = exposureController.setWhiteBalanceLocked(false)
        return exposure && whiteBalance
    }

    func setBracketingEnabled(_ enabled: Bool) {
        exposureController.setBracketingEnabledByUser(enabled)
    }

    /// Drops an anchor on the window the user is pointing at.
    ///
    /// This is the honest version of "mark the region": ARKit anchors are
    /// already written to `anchors/anchors_session.json` and re-read into
    /// `anchors_final.json`, and `CaptureAnchorRecorder` classifies an anchor
    /// with this name as a window. So the mark lands in the format that
    /// already exists, survives the end-of-session re-read, and gives the
    /// pre-pass's glass detector (F5) a place to look. Nothing here writes
    /// `SurfaceClass.glass`: that class means "detected, LiDAR-silent and
    /// image-bright and planar", and detecting it is the pre-pass's job.
    ///
    /// - Returns: false when the middle of the screen has no measurable
    ///   distance at all, which is what happens when the glass fills the frame.
    ///   The screen asks the user to get a bit of the wall in shot and tap
    ///   again, which is both true and useful.
    @discardableResult
    func markWindowInFront() -> Bool {
        guard phase == .recording else { return false }
        guard let transform = lastCameraTransform else { return false }
        guard let distance = windowMode.centreDistanceMeters else { return false }

        // ARKit's camera looks down its own -Z.
        let forward = -simd_make_float3(transform.columns.2)
        let origin = simd_make_float3(transform.columns.3)
        let position = origin + forward * distance

        var anchorTransform = transform
        anchorTransform.columns.3 = SIMD4<Float>(position, 1)
        session.add(
            anchor: ARAnchor(
                name: CaptureAnchorRecorder.userMarkedWindowAnchorName,
                transform: anchorTransform
            )
        )
        CaptureLog.session.notice("The user marked a window by hand.")
        return true
    }

    // MARK: - Settings actually used

    private func currentSettings() -> CaptureSettings {
        CaptureSettings(
            bracketEveryNFrames: exposureController.settingsCadence(
                requested: lastBracketCadence
            ),
            bracketStops: CaptureTuning.bracketStops,
            exposureLocked: exposureController.isExposureLocked,
            whiteBalanceLocked: exposureController.isWhiteBalanceLocked,
            // Measured from the map that actually arrived, never assumed to be
            // 256x192. Zero means no depth map was ever delivered, which is a
            // fact a reader needs rather than a default it would trust.
            depthWidth: lastDepthSize?.width ?? 0,
            depthHeight: lastDepthSize?.height ?? 0,
            lidarMaxRangeMeters: CaptureTuning.lidarMaxRangeMeters
        )
    }

    private static func resolvedDisplayName(_ name: String, startedAt: Date) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty else { return trimmed }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return "Scan on \(formatter.string(from: startedAt))"
    }

    // MARK: - Configuration and permission

    /// Builds the world-tracking configuration, or explains what this iPhone
    /// cannot do.
    static func makeConfiguration() throws -> ARWorldTrackingConfiguration {
        guard ARWorldTrackingConfiguration.isSupported else {
            throw NimbusError.deviceIncompatible(
                "This iPhone cannot run the camera tracking this app is built "
                    + "on, so there is no way for it to record a scan."
            )
        }
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            throw NimbusError.deviceIncompatible(
                "This iPhone does not have the laser scanner (LiDAR) that "
                    + "measures how far away things are. Without it a scan "
                    + "would be guesswork, so the app will not pretend."
            )
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity

        // `.sceneDepth` ONLY. `.smoothedSceneDepth` is temporally filtered and
        // upsampled: it invents edges nothing measured, and F3 depends on the
        // native samples being the only ones we supervise against.
        configuration.frameSemantics = [.sceneDepth]

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            configuration.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
            // Built as a String first on purpose: `Logger.notice` takes an
            // `OSLogMessage`, and a `+` of two string literals is a `String`,
            // which does not convert. This is a compile error, not a style
            // preference.
            let note = "This iPhone meshes the room but will not label what the "
                + "surfaces are, so window mode cannot switch itself on."
            CaptureLog.session.notice("\(note, privacy: .public)")
        }

        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .none
        configuration.isLightEstimationEnabled = false
        // No microphone: there is deliberately no NSMicrophoneUsageDescription
        // in the Info.plist, and asking for audio here would make iOS demand
        // one.
        configuration.providesAudioData = false

        if let format = preferredVideoFormat() {
            configuration.videoFormat = format
        }
        return configuration
    }

    /// The capture format closest to the 1920 px wide stream the data format
    /// and the blur meter's 0.0426 deg/px pitch are both written against.
    ///
    /// Not simply "the biggest": a 4K stream would quadruple the storage and
    /// the heat for detail the trainer downsamples away anyway, and it would
    /// silently move the pixel pitch the amber and red blur thresholds are
    /// calibrated to.
    static func preferredVideoFormat() -> ARConfiguration.VideoFormat? {
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        guard !formats.isEmpty else { return nil }
        return formats.min { first, second in
            formatScore(first) < formatScore(second)
        }
    }

    private static func formatScore(_ format: ARConfiguration.VideoFormat) -> Double {
        let width = Double(format.imageResolution.width)
        // A frame rate above 60 buys nothing here: keyframes are chosen on
        // motion, and the extra frames are heat.
        let frameRatePenalty = format.framesPerSecond > 60 ? 4_000.0 : 0
        return abs(width - 1920) + frameRatePenalty
    }

    /// Asks for the camera once, plainly, and turns a refusal into a sentence
    /// rather than a black screen.
    static func requireCameraPermission() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            guard granted else {
                throw NimbusError.captureFailed(Self.cameraDeniedMessage)
            }
        case .denied, .restricted:
            throw NimbusError.captureFailed(Self.cameraDeniedMessage)
        @unknown default:
            throw NimbusError.captureFailed(Self.cameraDeniedMessage)
        }
    }

    /// `nonisolated` because the ARKit failure callback below is, and it needs
    /// this sentence before it hops to the main actor.
    nonisolated static let cameraDeniedMessage =
        "This app needs the camera to record a scan, and it is currently "
        + "switched off for \(BrandConfig.displayName). You can turn it back "
        + "on in Settings, under Privacy and Security, then Camera."
}

// =============================================================================
//  MARK: - ARKit callbacks
//
//  Every method here is `nonisolated` because `ARSessionDelegate` is not
//  isolated to any actor, and each one immediately asserts the isolation that
//  is actually in force. The session's `delegateQueue` is deliberately left at
//  its default, which is the main queue, so that assertion holds; do not set a
//  delegate queue without changing every method below.
// =============================================================================

extension ARCaptureService: ARSessionDelegate {

    public nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        MainActor.assumeIsolated {
            self.handle(frame: frame)
        }
    }

    public nonisolated func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        MainActor.assumeIsolated {
            self.anchorRecorder.anchorsWereAdded(anchors)
            for anchor in anchors {
                if let mesh = anchor as? ARMeshAnchor {
                    self.meshStore.update(from: mesh)
                }
            }
        }
    }

    public nonisolated func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        MainActor.assumeIsolated {
            // The copy has to happen here, on the thread ARKit handed the
            // anchor over on and while it is still alive: `ARMeshAnchor`'s
            // geometry is backed by Metal buffers ARKit owns and rewrites.
            for anchor in anchors {
                if let mesh = anchor as? ARMeshAnchor {
                    self.meshStore.update(from: mesh)
                }
            }
        }
    }

    public nonisolated func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        MainActor.assumeIsolated {
            self.anchorRecorder.anchorsWereRemoved(anchors)
            for anchor in anchors {
                self.meshStore.remove(identifier: anchor.identifier)
            }
        }
    }

    public nonisolated func session(
        _ session: ARSession,
        cameraDidChangeTrackingState camera: ARCamera
    ) {
        let quality = TrackingQuality(camera.trackingState)
        MainActor.assumeIsolated {
            self.lastTrackingQuality = quality
            self.exposureController.observeTracking(quality)
        }
    }

    public nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        let message = Self.plainMessage(for: error)
        MainActor.assumeIsolated {
            CaptureLog.session.error(
                "ARKit session failed: \(error.localizedDescription, privacy: .public)"
            )
            self.halt(reason: message)
        }
    }

    public nonisolated func sessionWasInterrupted(_ session: ARSession) {
        MainActor.assumeIsolated {
            CaptureLog.session.notice("The session was interrupted.")
            self.guidance.reset()
        }
    }

    public nonisolated func sessionInterruptionEnded(_ session: ARSession) {
        MainActor.assumeIsolated {
            CaptureLog.session.notice(
                "The interruption ended; the session is relocalising."
            )
        }
    }

    /// Yes, always.
    ///
    /// The alternative is resetting tracking, which starts a brand new world
    /// origin and would leave every frame already written pointing at a map
    /// that no longer exists. Relocalising can fail, and when it does the user
    /// is told to walk back to where they were, which is a thing a person can
    /// actually do.
    public nonisolated func sessionShouldAttemptRelocalization(_ session: ARSession) -> Bool {
        true
    }

    /// ARKit's errors, in words a person can act on.
    nonisolated static func plainMessage(for error: Error) -> String {
        guard let arError = error as? ARError else {
            return "The scan stopped because of a problem with the camera. "
                + "Everything recorded so far is saved."
        }
        switch arError.code {
        case .cameraUnauthorized:
            return cameraDeniedMessage
        case .sensorUnavailable, .sensorFailed:
            return "The camera or the motion sensors stopped responding, so "
                + "the scan stopped. Everything recorded so far is saved. "
                + "Closing and reopening the app usually clears this."
        case .worldTrackingFailed:
            return "The phone lost track of where it was and could not find "
                + "its way back. Everything recorded so far is saved."
        case .unsupportedConfiguration, .invalidConfiguration:
            return "This iPhone will not run the kind of scan this app needs."
        default:
            return "The scan stopped because of a problem with the camera. "
                + "Everything recorded so far is saved."
        }
    }
}
