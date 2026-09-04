//
//  ScanReviewModel.swift
//  Viewer
//
//  WHAT THE REVIEW SCREEN KNOWS AND OWNS.
//
//  One renderer, one camera controller, one scan. The screen above it is
//  deliberately dumb: every decision that could be got wrong quietly (which
//  pose the camera is at, whether the honesty mask has anything to say,
//  whether a photo in the A/B slider was actually held out of training) is
//  made here, once, and exposed as a value the screen can only display.
//
//  Three modes share ONE Metal surface rather than one each:
//
//    walkThrough  the F9 fly-through, pinned to where the user really walked
//    freeLook     orbit around the scan, for poking at a specific corner
//    compare      the camera parked on one held-out photo's exact pose, with
//                 that photo wiped across the render
//
//  Sharing the surface is not a shortcut. An `MTKView` per mode would mean
//  three attach/detach lifecycles racing over one renderer, and a second copy
//  of a half-gigabyte splat cloud in GPU memory for the comparison.
//

import Foundation
import SwiftUI
import UIKit
import simd

@MainActor
final class ScanReviewModel: ObservableObject {

    // MARK: - Modes

    enum Mode: String, CaseIterable, Identifiable {
        case walkThrough
        case freeLook
        case compare

        var id: String { rawValue }

        var title: String {
            switch self {
            case .walkThrough: return "Walk-through"
            case .freeLook: return "Look around"
            case .compare: return "Photo vs scan"
            }
        }
    }

    // MARK: - Owned objects

    let renderer = MetalSplatRenderer()
    let camera = ViewerCameraController()

    // MARK: - The scan

    let summary: ScanSummary
    @Published private(set) var detail: ScanDetail?

    /// Non-nil while something is loading. Plain sentence, shown as-is.
    @Published private(set) var loadingMessage: String?

    /// Non-nil when the model could not be shown at all. Plain sentence.
    @Published private(set) var problem: String?

    // MARK: - Mode

    @Published var mode: Mode = .walkThrough {
        didSet {
            guard mode != oldValue else { return }
            applyMode()
        }
    }

    // MARK: - Overlays

    @Published var honestyMaskEnabled = true {
        didSet { renderer.setHonestyMaskEnabled(honestyMaskEnabled) }
    }

    @Published var artifactHeatmapEnabled = false {
        didSet { renderer.setArtifactHeatmapEnabled(artifactHeatmapEnabled) }
    }

    // MARK: - Fly-through

    @Published private(set) var flyThroughPath: PreviewCameraPath?
    /// Why there is no fly-through, when there is not one. Plain sentence.
    @Published private(set) var flyThroughProblem: String?
    /// Measured, not promised: the furthest the smoothed camera ever gets from
    /// the walked track, in metres.
    @Published private(set) var worstDeviationMeters: Float?
    @Published private(set) var isPlaying = true

    // MARK: - Honesty record

    @Published private(set) var isBuildingHonestyRecord = false
    /// What happened while working out which directions were observed. Shown
    /// verbatim, including when the answer is "this could not be worked out".
    @Published private(set) var honestyRecordNote: String?

    // MARK: - Photo versus scan

    @Published private(set) var heldOut: HeldOutFrameSet?
    @Published private(set) var compareIndex = 0
    @Published private(set) var comparePhoto: UIImage?
    @Published private(set) var comparePhotoProblem: String?
    /// Wipe position, 0 = all photo, 1 = all scan.
    @Published var compareWipe: Double = 0.5

    // MARK: - Private

    private var hasOpened = false
    private var honestyTask: Task<Void, Never>?
    private var honestyCancelFlag: ViewerCancellationFlag?
    private var photoTask: Task<Void, Never>?

    init(summary: ScanSummary) {
        self.summary = summary
    }

    // MARK: - Opening

    func open() async {
        guard !hasOpened else { return }
        hasOpened = true

        renderer.setHonestyMaskEnabled(honestyMaskEnabled)
        renderer.setArtifactHeatmapEnabled(artifactHeatmapEnabled)

        loadingMessage = "Opening this scan..."
        let target = summary
        let loaded = await Task.detached(priority: .userInitiated) {
            ScanLibraryReader.readDetail(target)
        }.value
        detail = loaded
        camera.sourceIntrinsics = loaded.bundle?.intrinsics

        if let bundle = loaded.bundle {
            heldOut = HeldOutFrameSelector.resolve(bundle: bundle, paths: loaded.paths)
        }

        await loadModelIfPresent(loaded)
        await buildFlyThrough(loaded)

        loadingMessage = nil
        applyMode()
        startHonestyRecordBuildIfNeeded()
    }

    private func loadModelIfPresent(_ loaded: ScanDetail) async {
        if let startupProblem = renderer.startupProblem {
            problem = startupProblem
            return
        }
        guard let model = loaded.model else {
            problem = nil
            return
        }
        loadingMessage = "Loading the 3D model..."
        do {
            try await renderer.load(model, at: loaded.ref)
        } catch {
            problem = error.localizedDescription
            ViewerLog.review.error(
                "load failed: \(error.localizedDescription, privacy: .public)"
            )
        }
    }

    private func buildFlyThrough(_ loaded: ScanDetail) async {
        guard let bundle = loaded.bundle else {
            flyThroughProblem =
                "This scan has no record of the walk, so there is nothing to replay."
            return
        }
        loadingMessage = "Working out the path you walked..."
        let prePass = loaded.prePass

        // Deliberately returns a plain pair rather than the builder's own
        // `Outcome`: that enum is not declared Sendable, and a value crossing
        // an actor boundary should not depend on a type somebody may later add
        // a non-Sendable payload to.
        let built = await Task.detached(priority: .userInitiated) {
            () -> (path: PreviewCameraPath?, problem: String?, worst: Float) in
            switch PreviewCameraPathBuilder.build(bundle: bundle, prePass: prePass) {
            case .path(let path):
                return (path, nil, PreviewPathSampler.worstDeviation(path))
            case .unavailable(let reason):
                return (nil, reason, 0)
            }
        }.value

        flyThroughPath = built.path
        flyThroughProblem = built.problem
        worstDeviationMeters = built.path == nil ? nil : built.worst
    }

    // MARK: - Modes

    private func applyMode() {
        switch mode {
        case .walkThrough:
            if let path = flyThroughPath {
                camera.adopt(path: path)
                camera.setPlaying(isPlaying)
            } else {
                frameFreeLook()
            }
            comparePhoto = nil

        case .freeLook:
            frameFreeLook()
            comparePhoto = nil

        case .compare:
            camera.setPlaying(false)
            showCompareFrame(at: compareIndex)
        }
    }

    private func frameFreeLook() {
        let bounds = renderer.contentBounds
            ?? detail?.model?.bounds
            ?? detail?.bundle?.sceneBounds
        if let bounds {
            camera.frame(bounds: bounds, fovDegrees: 70)
        } else {
            camera.setMode(.freeOrbit)
        }
    }

    func setPlaying(_ playing: Bool) {
        isPlaying = playing
        camera.setPlaying(playing)
    }

    func seek(toProgress progress: Double) {
        camera.seek(toProgress: progress)
    }

    // MARK: - Photo versus scan

    /// Parks the camera on one held-out photo's exact pose and loads that
    /// photo.
    ///
    /// The field of view is set from the CAPTURE camera here, not widened to
    /// the fly-through's 105 degrees. Widening would put more of the room in
    /// the render than is in the photo, and the two halves of the wipe would
    /// no longer line up, which would make the comparison look worse than the
    /// scan really is.
    func showCompareFrame(at index: Int) {
        guard let heldOut, !heldOut.frames.isEmpty else {
            comparePhoto = nil
            comparePhotoProblem = nil
            return
        }
        let clamped = Swift.min(Swift.max(index, 0), heldOut.frames.count - 1)
        compareIndex = clamped
        let frame = heldOut.frames[clamped]

        let pose = detail?.prePass?.refinedPose(for: frame.index)
            ?? frame.refinedPose
            ?? frame.rawPose

        let fov = detail?.bundle?.intrinsics.horizontalFOVDegrees ?? 65
        let stillPath = PreviewCameraPath(
            keyframes: [
                PreviewCameraPath.Keyframe(
                    pose: pose,
                    timeSeconds: 0,
                    sourceFrame: frame.index
                )
            ],
            horizontalFOVDegrees: fov,
            maxDeviationMeters: 0.20,
            honestyMaskEnabled: honestyMaskEnabled
        )
        camera.adopt(path: stillPath)
        camera.setPlaying(false)

        loadComparePhoto(frame)
    }

    func stepCompare(by delta: Int) {
        showCompareFrame(at: compareIndex + delta)
    }

    private func loadComparePhoto(_ frame: CaptureFrame) {
        photoTask?.cancel()
        comparePhoto = nil
        comparePhotoProblem = nil

        guard let paths = detail?.paths else { return }
        let url = paths.url(frame.imagePath)
        let relative = frame.imagePath

        photoTask = Task { [weak self] in
            let data = await Task.detached(priority: .userInitiated) { () -> Data? in
                try? Data(contentsOf: url, options: [.mappedIfSafe])
            }.value
            guard !Task.isCancelled else { return }
            guard let data, let image = UIImage(data: data) else {
                self?.comparePhotoProblem =
                    "The photo for this moment of the walk (\(relative)) could not be opened."
                return
            }
            self?.comparePhoto = image
        }
    }

    /// Aspect ratio the comparison should be shown at: the photo's, so the
    /// render and the photo frame the same rectangle of the world.
    var compareAspectRatio: CGFloat {
        guard let intrinsics = detail?.bundle?.intrinsics,
              intrinsics.width > 0, intrinsics.height > 0
        else { return 4.0 / 3.0 }
        return CGFloat(intrinsics.width) / CGFloat(intrinsics.height)
    }

    // MARK: - Honesty record

    /// Works out which directions each part of the scene was really looked at
    /// from, when the model did not come with that record already.
    ///
    /// This reads every usable depth map on disk, so it runs off the main
    /// actor and is cancelled the moment the screen goes away. The result is
    /// written to `model/observed_directions.bin` so it only ever happens once
    /// per scan.
    func startHonestyRecordBuildIfNeeded() {
        guard honestyTask == nil,
              !isBuildingHonestyRecord,
              !renderer.hasObservationField,
              renderer.startupProblem == nil,
              let loaded = detail,
              let bundle = loaded.bundle
        else {
            if !renderer.hasObservationField, honestyRecordNote == nil {
                honestyRecordNote = renderer.observationFieldProblem
            }
            return
        }

        isBuildingHonestyRecord = true
        honestyRecordNote =
            "Working out which parts of this scan you really looked at, and from where."

        let paths = loaded.paths
        let prePass = loaded.prePass
        // `Task.detached` does not inherit cancellation, so cancelling the
        // wrapper below would leave the file-reading work running for another
        // minute after the user left. This flag is what actually stops it.
        let cancelled = ViewerCancellationFlag()
        honestyCancelFlag = cancelled

        honestyTask = Task { [weak self] in
            let outcome = await Task.detached(priority: .utility) {
                () -> (field: ObservedDirectionField?, note: String) in
                do {
                    let report = try ObservedDirectionBuilder.build(
                        bundle: bundle,
                        prePass: prePass,
                        paths: paths,
                        isCancelled: { cancelled.isCancelled }
                    )
                    guard report.field.cellCount > 0 else {
                        return (
                            nil,
                            "This scan has no usable depth measurements, so there is no "
                                + "way to tell which parts of it were really looked at."
                        )
                    }
                    // Keep it, so this only ever happens once for this scan.
                    var saveNote = ""
                    do {
                        try FileManager.default.createDirectory(
                            at: paths.modelDirectory,
                            withIntermediateDirectories: true
                        )
                        try report.field.write(to: paths.observedDirectionsBin)
                    } catch {
                        saveNote = " It could not be saved for next time (\(error.localizedDescription))."
                    }
                    return (report.field, report.summary + saveNote)
                } catch {
                    return (nil, error.localizedDescription)
                }
            }.value

            guard !Task.isCancelled, let self else { return }
            if let field = outcome.field {
                self.renderer.adopt(field)
            }
            self.honestyRecordNote = outcome.note
            self.isBuildingHonestyRecord = false
            self.honestyTask = nil
        }
    }

    /// True when the hatching on screen means something. False means nothing
    /// is hatched because nothing is KNOWN, which is a different statement and
    /// the screen says so.
    var honestyMaskHasEvidence: Bool { renderer.hasObservationField }

    // MARK: - Closing

    func close() {
        honestyCancelFlag?.cancel()
        honestyCancelFlag = nil
        honestyTask?.cancel()
        honestyTask = nil
        photoTask?.cancel()
        photoTask = nil
        renderer.unload()
    }
}

// MARK: - Cancellation across a detached task

/// A cancellation flag a detached task can poll.
///
/// `Task.detached` deliberately does not inherit cancellation from the task
/// that created it, and the honesty-record build reads hundreds of files, so
/// "the user left the screen" has to reach it somehow. This is that somehow:
/// one bool behind one lock, which is the whole of it.
final class ViewerCancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return flag
    }

    func cancel() {
        lock.lock()
        flag = true
        lock.unlock()
    }
}
