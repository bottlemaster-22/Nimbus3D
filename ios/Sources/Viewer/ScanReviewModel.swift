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

    // MARK: - Where the geometry went

    /// The splat census for this scan: one sentence naming the step that lost
    /// the most, with the full ladder behind it.
    ///
    /// Published twice on purpose. The first version is up as soon as the
    /// scan's JSON has been read, so the sentence is on screen while the model
    /// is still loading; the second replaces it once the splat file itself has
    /// been counted, which is what turns "480,000 points in the file" into
    /// "480,000 points in the file and 3,000 of them can draw".
    @Published private(set) var census: ScanCensus = .unread

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

    // MARK: - Which way up the phone was

    /// Quarter turns clockwise this scan's pictures and poses need before they
    /// are the right way up on screen.
    ///
    /// ARKit hands out poses, intrinsics and pixels in the sensor's own
    /// LANDSCAPE frame however the phone is held, and capture writes all three
    /// to disk unchanged, which is right: they agree with each other, so the
    /// COLMAP files, the trainer and every export are correct and the model
    /// itself stands up straight. The one thing missing from the record is
    /// which way up the phone was. A capture now records it, in
    /// `CaptureSettings.imageQuarterTurnsClockwiseToUpright`, and that is what
    /// is used where a scan has it; a scan written before that field existed
    /// (which is every scan already on the phone) has it measured back out of
    /// its own poses instead, by `ViewerPoseMath.uprightQuarterTurns(of:)`.
    ///
    /// DISPLAY only, either way. Nothing on disk is touched, and nothing
    /// exported moves.
    private(set) var displayQuarterTurns = 0

    /// The capture camera as it is being SHOWN: the recorded intrinsics turned
    /// by `displayQuarterTurns`, so width, height, fx, fy and the principal
    /// point all describe the picture the user is actually looking at. The
    /// recorded ones stay untouched on `detail`.
    private(set) var displayIntrinsics: CameraIntrinsics?

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
        census = ScanCensus.make(loaded.summary.censusInputs)
        displayQuarterTurns = Self.uprightQuarterTurns(of: loaded.bundle)
        displayIntrinsics = loaded.bundle?.intrinsics.rotatedForDisplay(
            quarterTurnsClockwise: displayQuarterTurns
        )
        // The renderer takes only the PIXEL aspect from these, so handing it
        // the turned copy is what stops a non-square-pixel capture being
        // stretched the wrong way once the preview camera is rolled upright.
        camera.sourceIntrinsics = displayIntrinsics

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

        // Whether the load succeeded or failed, the renderer will have counted
        // the file if it got far enough to parse it. A model that failed to
        // load is exactly when the count matters most, so this is outside the
        // catch rather than inside the success path.
        refreshCensusFromLoadedModel(loaded)
    }

    /// Rebuilds the census with the measurement of the splat file itself, once
    /// the renderer has one.
    ///
    /// Nothing is invented when the renderer has nothing: the census then keeps
    /// its "the model file has not been opened on this screen" wording, which
    /// is the truth, rather than a zero.
    private func refreshCensusFromLoadedModel(_ loaded: ScanDetail) {
        let measurement = renderer.loadedCloudMeasurement
        census = ScanCensus.make(loaded.summary.censusInputs, drawable: measurement)
        if let measurement {
            // Built as a String first: an os.Logger message is one literal, and
            // five of them joined with + is not one.
            let note = "census: \(measurement.total) in file, "
                + "\(measurement.drawable) can draw, "
                + "\(measurement.belowAlphaCutoff) too faint, "
                + "\(measurement.nonFinite) broken, "
                + "\(measurement.needles) streaks"
            ViewerLog.review.notice("\(note, privacy: .public)")
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
        let quarterTurns = displayQuarterTurns

        // Deliberately returns a plain pair rather than the builder's own
        // `Outcome`: that enum is not declared Sendable, and a value crossing
        // an actor boundary should not depend on a type somebody may later add
        // a non-Sendable payload to.
        let built = await Task.detached(priority: .userInitiated) {
            () -> (path: PreviewCameraPath?, problem: String?, worst: Float) in
            var options = PreviewCameraPathBuilder.Options()
            options.uprightQuarterTurns = quarterTurns
            switch PreviewCameraPathBuilder.build(
                bundle: bundle,
                prePass: prePass,
                options: options
            ) {
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

        // Read from the TURNED camera, because the render is rolled upright
        // and so is the photo beside it. On a portrait scan the capture's
        // picture runs up the panel, so it is the turned camera's horizontal
        // field of view that has to span the panel's width. Framing it with
        // the sensor's own would leave the two halves of the wipe out of step,
        // which would make the scan look worse than it really is.
        let fov = displayIntrinsics?.horizontalFOVDegrees ?? 65
        let stillPath = PreviewCameraPath(
            keyframes: [
                PreviewCameraPath.Keyframe(
                    pose: pose.rolledForDisplay(
                        quarterTurnsClockwise: displayQuarterTurns
                    ),
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
            guard let self else { return }
            self.comparePhoto = ViewerPhoto.upright(
                image,
                quarterTurnsClockwise: self.displayQuarterTurns
            )
        }
    }

    /// Aspect ratio the comparison should be shown at: the photo's as it is
    /// SHOWN, so the render and the photo frame the same rectangle of the
    /// world.
    ///
    /// The turned camera's, because the recorded size is the sensor's, which
    /// is landscape whatever way the phone was held. Turning the picture
    /// upright turns the frame around it too, or the photo would sit
    /// letterboxed inside a panel the wrong shape and would no longer line up
    /// with the render.
    var compareAspectRatio: CGFloat {
        guard let intrinsics = displayIntrinsics,
              intrinsics.width > 0, intrinsics.height > 0
        else { return 4.0 / 3.0 }
        return CGFloat(intrinsics.width) / CGFloat(intrinsics.height)
    }

    /// Which way up the phone was held for this scan: what the capture
    /// recorded, or, for a scan written before captures recorded it, what its
    /// own poses say.
    ///
    /// When a scan has both, the recorded number wins (it is exact, and it is
    /// right even for a scan of nothing but a ceiling) but the measured one is
    /// still worked out and a disagreement is logged. The two are derived
    /// completely differently, one from the interface orientation at session
    /// start and one from where gravity ended up in the pictures, so if they
    /// ever disagree on a scan with a horizon in it, something is wrong and
    /// this is the line that says so instead of leaving someone to guess from
    /// a sideways preview.
    private static func uprightQuarterTurns(of bundle: CaptureBundle?) -> Int {
        guard let bundle else { return 0 }
        let measured = ViewerPoseMath.uprightQuarterTurns(
            of: bundle.frames.lazy.map { $0.refinedPose ?? $0.rawPose }
        )
        guard let recorded = bundle.settings.imageQuarterTurnsClockwiseToUpright
        else { return measured }
        let turns = ((recorded % 4) + 4) % 4
        if turns != measured {
            // Built as a String first: an os.Logger message is one literal,
            // and two of them joined with + is not one.
            let note = "capture recorded \(turns) quarter turns to upright, "
                + "the poses measure \(measured); using the recorded one"
            ViewerLog.review.notice("\(note, privacy: .public)")
        }
        return turns
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
