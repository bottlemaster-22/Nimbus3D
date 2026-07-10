//
//  ARCaptureService.swift — the real CaptureService implementation.
//  Owned by the Capture agent. Implements the CaptureService protocol from
//  Sources/Core/Contracts.swift using ARKit world tracking + AVFoundation-backed
//  frame capture, with optional LiDAR depth on Pro devices.
//
//  What is REAL here:
//    - ARKit world-tracking session, gravity aligned, highest available video format.
//    - Sparse posed RGB frame capture: each saved frame carries its camera-to-world pose
//      and pinhole intrinsics, written to disk as HEIC (JPEG fallback).
//    - LiDAR sceneDepth + confidence captured as a per-frame depth prior on Pro devices.
//    - Live guidance events (tracking quality, frame count, angular coverage, warnings).
//    - Assembling and persisting a CaptureBundle (manifest.json + depth sidecar).
//
//  Honest limitation (see MODULE_STATUS.md): the HDRI "brackets" are ARKit auto-exposed
//  environment frames tagged with the sensor's reported exposure offset. They are NOT
//  true multi-EV bracketed exposures, because ARKit owns exposure while its session runs
//  and will not let a concurrent AVCapturePhotoOutput drive an exposure bracket. The HDRI
//  module can still project them, but a real HDR merge needs a dedicated bracket pass.
//

import Foundation
import ARKit
import AVFoundation
import CoreVideo
import simd

final class ARCaptureService: NSObject, CaptureService, ARSessionDelegate {

    /// Which kind of frames the running session is currently collecting.
    enum Phase: Sendable, Hashable { case object, environment }

    /// The AR session is created and owned here and shared with the preview view.
    let session = ARSession()

    // MARK: - CaptureService protocol surface

    private(set) var isCapturing: Bool = false
    let events: AsyncStream<CaptureEvent>
    private let eventContinuation: AsyncStream<CaptureEvent>.Continuation

    // MARK: - Tuning constants

    /// Minimum camera translation (meters) since the last saved object frame.
    private let minTranslation: Float = 0.03
    /// Minimum orbit angle (radians) around the subject since the last saved object frame.
    private let minOrbitAngle: Float = CaptureMath.degreesToRadians(4)
    /// Cap on environment frames collected for HDRI assembly.
    private let maxBrackets = 24
    /// Minimum look-direction change (radians) between environment frames.
    private let minBracketAngle: Float = CaptureMath.degreesToRadians(12)
    /// Minimum time (seconds) between environment frames.
    private let minBracketInterval: TimeInterval = 0.4

    // MARK: - Session state (mutated only on captureQueue)

    private let captureQueue = DispatchQueue(label: "com.nimbus3d.capture.session")
    private var active = false
    private var options = CaptureOptions()
    private var writer: CaptureBundleWriter?
    private var rootDirectory: URL?
    private var coverage = CoverageTracker()
    private var savedFrames: [CapturedFrame] = []
    private var hdriBrackets: [ExposureBracketFrame] = []
    private var frameIndex = 0
    private var sessionStartTime: TimeInterval?
    private var hasLiDAR = false
    private var currentPhase: Phase = .object

    private var lastSavedPosition: SIMD3<Float>?
    private var lastSavedCenterVector: SIMD3<Float>?
    private var lastBracketForward: SIMD3<Float>?
    private var lastBracketTime: TimeInterval?

    private var lastEmittedCoverage: Float = -1
    private var lastEmittedTracking: TrackingQuality = .notAvailable

    // MARK: - Init

    override init() {
        var continuation: AsyncStream<CaptureEvent>.Continuation!
        self.events = AsyncStream<CaptureEvent>(bufferingPolicy: .bufferingNewest(32)) { continuation = $0 }
        self.eventContinuation = continuation
        super.init()
        session.delegate = self
        session.delegateQueue = captureQueue
    }

    /// Re-attach ourselves as the session delegate. The preview view calls this after it
    /// binds `session` so a view rebuild can never orphan our frame callbacks.
    func reassertDelegate() {
        session.delegate = self
        session.delegateQueue = captureQueue
    }

    /// Switch between collecting object frames and environment (HDRI) frames mid-session.
    func setPhase(_ phase: Phase) {
        captureQueue.async { self.currentPhase = phase }
    }

    // MARK: - CaptureService: start

    func startCapture(options: CaptureOptions) async throws {
        guard CaptureDeviceCapabilities.isWorldTrackingSupported else {
            throw NimbusError.captureFailed("ARKit world tracking is not supported on this device.")
        }

        let root = try Self.resolveRootDirectory(options.outputDirectory)
        let writer: CaptureBundleWriter
        do {
            writer = try CaptureBundleWriter(rootDirectory: root)
        } catch {
            throw NimbusError.captureFailed("Could not create the capture directory: \(error.localizedDescription)")
        }

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            captureQueue.async {
                self.options = options
                self.writer = writer
                self.rootDirectory = root
                self.coverage.reset()
                self.savedFrames.removeAll()
                self.hdriBrackets.removeAll()
                self.frameIndex = 0
                self.sessionStartTime = nil
                self.hasLiDAR = options.captureLiDARDepth && CaptureDeviceCapabilities.supportsLiDARDepth
                self.currentPhase = .object
                self.lastSavedPosition = nil
                self.lastSavedCenterVector = nil
                self.lastBracketForward = nil
                self.lastBracketTime = nil
                self.lastEmittedCoverage = -1
                self.lastEmittedTracking = .notAvailable
                self.active = true
                continuation.resume()
            }
        }

        let config = CaptureDeviceCapabilities.makeConfiguration(options: options)
        await MainActor.run {
            self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
            self.isCapturing = true
        }
    }

    // MARK: - CaptureService: finish

    func finishCapture() async throws -> CaptureBundle {
        await MainActor.run {
            self.session.pause()
            self.isCapturing = false
        }

        return try await withCheckedThrowingContinuation { continuation in
            // Runs after every previously enqueued frame encode on this serial queue,
            // so all saved frames are flushed to disk before we assemble the manifest.
            captureQueue.async {
                self.active = false
                guard let root = self.rootDirectory, let writer = self.writer else {
                    continuation.resume(throwing: NimbusError.captureFailed("Capture was never started."))
                    return
                }
                guard !self.savedFrames.isEmpty else {
                    continuation.resume(throwing: NimbusError.captureFailed(
                        "No frames were captured. Move slowly around the object in good lighting and keep tracking steady."))
                    return
                }

                let bundle = CaptureBundle(
                    rootDirectory: root,
                    frames: self.savedFrames,
                    hdriBrackets: self.hdriBrackets,
                    hasLiDARDepth: self.hasLiDAR && self.savedFrames.contains { $0.depthMapURL != nil },
                    sceneBoundingRadius: self.coverage.estimatedRadius
                )

                do {
                    try writer.writeManifest(bundle)
                    continuation.resume(returning: bundle)
                } catch {
                    continuation.resume(throwing: NimbusError.captureFailed(
                        "Failed to write the capture manifest: \(error.localizedDescription)"))
                }
            }
        }
    }

    // MARK: - CaptureService: cancel

    func cancelCapture() async {
        await MainActor.run {
            self.session.pause()
            self.isCapturing = false
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            captureQueue.async {
                self.active = false
                if let root = self.rootDirectory {
                    try? FileManager.default.removeItem(at: root)
                }
                self.savedFrames.removeAll()
                self.hdriBrackets.removeAll()
                self.writer = nil
                self.rootDirectory = nil
                continuation.resume()
            }
        }
    }

    // MARK: - ARSessionDelegate (all callbacks run on captureQueue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard active else { return }
        if sessionStartTime == nil { sessionStartTime = frame.timestamp }

        emitTrackingIfChanged(frame.camera.trackingState)

        switch currentPhase {
        case .object: handleObjectFrame(frame)
        case .environment: handleEnvironmentFrame(frame)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        eventContinuation.yield(.warning("AR session error: \(error.localizedDescription)"))
    }

    func sessionWasInterrupted(_ session: ARSession) {
        eventContinuation.yield(.warning("Capture interrupted. Return to the object to resume tracking."))
    }

    // MARK: - Object frame handling

    private func handleObjectFrame(_ frame: ARFrame) {
        guard case .normal = frame.camera.trackingState else { return }

        let transform = frame.camera.transform
        let camPos = transform.columns.3.xyz
        var centerDepth: Float?
        if hasLiDAR, let depthMap = frame.sceneDepth?.depthMap {
            centerDepth = FrameWriter.centerDepth(of: depthMap)
        }
        coverage.registerFrame(cameraTransform: transform, centerDepth: centerDepth)

        if savedFrames.count < options.targetFrameCount,
           shouldSaveObjectFrame(camPos: camPos, center: coverage.subjectCenter) {
            do {
                try saveObjectFrame(frame)
            } catch {
                eventContinuation.yield(.warning("Could not save frame: \(error.localizedDescription)"))
            }
        }

        emitCoverageIfChanged(coverage.coverage)
    }

    private func shouldSaveObjectFrame(camPos: SIMD3<Float>, center: SIMD3<Float>?) -> Bool {
        guard let lastPosition = lastSavedPosition else { return true }
        let translated = simd_distance(camPos, lastPosition) >= minTranslation

        var orbited = false
        if let center, let lastVector = lastSavedCenterVector {
            let currentVector = simd_normalize(camPos - center)
            orbited = CaptureMath.angleBetween(currentVector, lastVector) >= minOrbitAngle
        }
        return translated || orbited
    }

    private func saveObjectFrame(_ frame: ARFrame) throws {
        guard let writer else { return }
        let index = frameIndex
        let camera = frame.camera

        let imageURL = try writer.writeColorImage(frame.capturedImage, index: index)

        var depthURL: URL?
        var confidenceURL: URL?
        if hasLiDAR, let sceneDepth = frame.sceneDepth {
            depthURL = try writer.writeDepth(sceneDepth.depthMap, index: index)
            if let confidence = sceneDepth.confidenceMap {
                confidenceURL = try writer.writeConfidence(confidence, index: index)
            }
        }

        let captured = CapturedFrame(
            index: index,
            imageURL: imageURL,
            depthMapURL: depthURL,
            depthConfidenceURL: confidenceURL,
            pose: CameraPose(matrix: camera.transform),
            intrinsics: Self.intrinsics(camera),
            timestamp: frame.timestamp - (sessionStartTime ?? frame.timestamp),
            exposureDuration: camera.exposureDuration,
            // TODO(nimbus): ARKit does not vend sensor ISO through ARCamera. A real value
            // needs a concurrent AVCaptureDevice.iso observation or EXIF parsing of a
            // separately captured still. Left at 0 rather than faking a plausible number.
            iso: 0
        )

        savedFrames.append(captured)
        frameIndex += 1

        let position = camera.transform.columns.3.xyz
        lastSavedPosition = position
        if let center = coverage.subjectCenter {
            lastSavedCenterVector = simd_normalize(position - center)
        }

        eventContinuation.yield(.frameCaptured(index: index, target: options.targetFrameCount))
    }

    // MARK: - Environment (HDRI) frame handling

    private func handleEnvironmentFrame(_ frame: ARFrame) {
        guard case .normal = frame.camera.trackingState else { return }
        guard hdriBrackets.count < maxBrackets else { return }

        let camera = frame.camera
        let forward = -simd_normalize(camera.transform.columns.2.xyz)
        let now = frame.timestamp

        var rotatedEnough = true
        if let last = lastBracketForward {
            rotatedEnough = CaptureMath.angleBetween(forward, last) >= minBracketAngle
        }
        var intervalOK = true
        if let last = lastBracketTime {
            intervalOK = (now - last) >= minBracketInterval
        }
        guard rotatedEnough, intervalOK else { return }

        do {
            guard let writer else { return }
            let index = hdriBrackets.count
            let url = try writer.writeBracketImage(frame.capturedImage, index: index)
            let bracket = ExposureBracketFrame(
                imageURL: url,
                // Real value reported by the sensor. Because ARKit auto-exposes, these are
                // environment frames at the metered exposure, not a controlled EV sweep.
                exposureBias: Float(camera.exposureOffset),
                exposureDuration: camera.exposureDuration,
                iso: 0, // TODO(nimbus): same ISO limitation as object frames.
                pose: CameraPose(matrix: camera.transform),
                intrinsics: Self.intrinsics(camera)
            )
            hdriBrackets.append(bracket)
            lastBracketForward = forward
            lastBracketTime = now
        } catch {
            eventContinuation.yield(.warning("Could not save environment frame: \(error.localizedDescription)"))
        }
    }

    // MARK: - Event emission (deduplicated)

    private func emitTrackingIfChanged(_ state: ARCamera.TrackingState) {
        let quality = Self.trackingQuality(state)
        guard quality != lastEmittedTracking else { return }
        lastEmittedTracking = quality
        eventContinuation.yield(.trackingStateChanged(quality))
    }

    private func emitCoverageIfChanged(_ value: Float) {
        guard abs(value - lastEmittedCoverage) > 0.0001 else { return }
        lastEmittedCoverage = value
        eventContinuation.yield(.coverageUpdated(value))
    }

    // MARK: - Helpers

    private static func resolveRootDirectory(_ provided: URL?) throws -> URL {
        let base: URL
        if let provided {
            base = provided
        } else {
            let documents = try FileManager.default.url(
                for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            base = documents
                .appendingPathComponent("Captures", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
        }
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private static func intrinsics(_ camera: ARCamera) -> CameraIntrinsics {
        // ARKit intrinsics are a column-major 3x3: col0 = (fx,0,0), col1 = (0,fy,0), col2 = (cx,cy,1).
        let k = camera.intrinsics
        let resolution = camera.imageResolution
        return CameraIntrinsics(
            focalLength: SIMD2<Float>(k.columns.0.x, k.columns.1.y),
            principalPoint: SIMD2<Float>(k.columns.2.x, k.columns.2.y),
            imageWidth: Int(resolution.width),
            imageHeight: Int(resolution.height)
        )
    }

    private static func trackingQuality(_ state: ARCamera.TrackingState) -> TrackingQuality {
        switch state {
        case .notAvailable: return .notAvailable
        case .limited: return .limited
        case .normal: return .normal
        }
    }
}
