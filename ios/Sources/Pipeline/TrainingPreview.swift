//
//  TrainingPreview.swift
//  Pipeline
//
//  THE LIVE PICTURE OF A MODEL BEING BUILT (F9).
//
//  Three good pieces existed and none of them was wired to the others:
//  `SplatTrainer.snapshot()` handed back the current splat field and nobody
//  asked for it, `SplatRenderer.load(_ cloud:)` drew an in-memory cloud and
//  nobody called it, and `TrainerProgress.previewAvailable` was set on every
//  tick and read by nobody. So a twenty minute build showed a spinner and some
//  numbers. This file is the wire between them.
//
//  ---------------------------------------------------------------------------
//  THE RULES IT WORKS UNDER
//  ---------------------------------------------------------------------------
//  * TRAINING COMES FIRST, ALWAYS. `offerRefresh` returns immediately and
//    never awaits anything on the caller's thread, so a tick handler in the
//    training stream cannot be held up by it. A refresh that arrives while the
//    previous one is still uploading is DROPPED, not queued.
//  * A FIXED CAMERA. The point of watching is to see the scene sharpen. A
//    viewpoint that moves between refreshes hides exactly that, so the camera
//    is framed ONCE, on the first frame that arrives, and never touched again.
//    Gestures are off for the same reason: this is a progress picture, not the
//    review screen, and the review screen is one tap away when it is done.
//  * A FAILED REFRESH IS NOT A FAILED BUILD. `snapshot()` throws while there
//    is nothing to preview yet, and can throw again later. Every one of those
//    is a note in the log and the last good frame stays on screen.
//  * IT MUST NOT COST THE MODEL ANYTHING. This is the sharp one. The trainer
//    lowers its own budget when free memory runs short
//    (`TrainerBudgetGovernor.degradeForMemory` weighs what it holds against
//    `os_proc_available_memory()`), so a preview that quietly allocated a few
//    hundred megabytes beside it would make the model the user is watching
//    SMALLER, and the app would then honestly report the smaller model without
//    anyone knowing the preview caused it. So the frame that gets drawn is a
//    deliberately LIGHT copy: capped in point count and stripped of its
//    higher-order colour, which is most of the bytes. `TrainingPreviewCloud`
//    below does that, and the screen says it is a lighter version rather than
//    letting the user believe they are looking at every point.
//  * THE REDRAW IS A SHORT BURST, NOT A CONSTANT 60 Hz. The renderer reveals a
//    freshly loaded cloud over several frames (`residencyStep` in
//    MetalSplatRenderer), so a single redraw would show a fraction of the
//    splats. `isSettling` therefore runs the surface for a moment after each
//    load and then parks it. Between bursts the preview costs nothing, which
//    is the point: the GPU belongs to the trainer.
//

import Foundation
import SwiftUI

/// Owns the renderer, the camera and the throttle for the live preview of a
/// training run. One per run; `ScanProcessingCoordinator` makes it when
/// training starts and takes it down when the run ends.
@MainActor
final class TrainingPreviewController: ObservableObject {

    // MARK: - Owned objects

    /// The renderer this preview draws with. Its own, not the one
    /// `NimbusApp` registers.
    ///
    /// The registered instance is shared app state, and this preview turns
    /// overlays off and unloads the cloud when it is done; doing that to an
    /// object another module owns would be a change made behind that module's
    /// back. A private renderer costs one pipeline build per training run,
    /// which is milliseconds at the start of a job measured in minutes, and it
    /// is the same choice `ScanReviewModel` makes for the same reason.
    let renderer = MetalSplatRenderer()

    /// Fixed for the whole run. Framed once, then left alone.
    let camera = ViewerCameraController()

    // MARK: - Published state

    /// How many frames have been loaded successfully. Zero means nothing has
    /// been drawn yet, and the screen says so plainly rather than showing a
    /// black rectangle that looks like an empty scan.
    @Published private(set) var frameCount = 0

    /// True during the short redraw burst after a load. The surface animates
    /// while this is true and is parked when it is false.
    @Published private(set) var isSettling = false

    /// Detail points the model actually had when this frame was taken.
    @Published private(set) var modelSplatCount = 0

    /// Detail points this picture is drawing. Lower than `modelSplatCount`
    /// whenever the frame had to be lightened, and the screen says so: a
    /// preview that let the user believe they were seeing every point would be
    /// the small lie this project exists to avoid.
    @Published private(set) var drawnSplatCount = 0

    /// Why the preview could not be drawn, when the reason is worth reading.
    /// Never set for "there is nothing to preview yet": that is a wait, not a
    /// problem.
    @Published private(set) var note: String?

    /// Non-nil when the renderer could not start at all. Shown verbatim.
    var startupProblem: String? { renderer.startupProblem }

    // MARK: - Throttle

    /// Wall clock between refreshes. The trainer takes a snapshot every 50
    /// iterations and ticks at most twice a second, so this is the number that
    /// decides how often the preview costs anything, and it is deliberately
    /// slow: the user is watching a model form over minutes, not frames.
    private let minimumInterval: TimeInterval = 2.5

    /// How long the surface animates after a load, so the renderer's staged
    /// reveal finishes. The renderer brings in 60k splats a frame and a preview
    /// frame is capped at `TrainingPreviewCloud.splatCap`, so five frames is
    /// enough and this is many times that: the margin is for a phone that is
    /// busy training and not actually hitting 60 Hz.
    private let settleSeconds: TimeInterval = 0.7

    private var lastRefreshStartedAt: Date?
    private var refreshTask: Task<Void, Never>?
    private var settleTask: Task<Void, Never>?
    private var hasFramedCamera = false
    private var isTornDown = false

    // MARK: - Init

    init(sourceIntrinsics: CameraIntrinsics?) {
        camera.sourceIntrinsics = sourceIntrinsics
        // A progress picture, not a review. The honesty mask needs the record
        // built from the finished capture and there is none yet mid-run, so
        // asking for it here would draw nothing and mean nothing; the review
        // screen is where it belongs and where it is on by default.
        renderer.setHonestyMaskEnabled(false)
        renderer.setArtifactHeatmapEnabled(false)
    }

    // MARK: - Refreshing

    /// Offered on every tick that says a preview is worth showing. Takes at
    /// most one snapshot per `minimumInterval` and never more than one at a
    /// time.
    ///
    /// Returns immediately, always. The work happens in its own task so that
    /// the training stream's tick handler is never waiting on the GPU, on a
    /// buffer copy, or on this.
    func offerRefresh() {
        guard !isTornDown, startupProblem == nil else { return }
        // One in flight is enough. A second would only queue work behind the
        // first and show an older field later than the newer one.
        guard refreshTask == nil else { return }
        if let last = lastRefreshStartedAt,
           Date().timeIntervalSince(last) < minimumInterval {
            return
        }
        lastRefreshStartedAt = Date()

        refreshTask = Task { [weak self] in
            guard let self else { return }
            await self.refresh()
            self.refreshTask = nil
        }
    }

    private func refresh() async {
        // The trainer is resolved HERE rather than passed in and captured, for
        // the same reason `ScanProcessingCoordinator.cancel` resolves it inside
        // its own task: `any SplatTrainer` is not Sendable, and reaching the
        // registry from inside the task means nothing non-Sendable is carried
        // across a boundary.
        guard let trainer = NimbusServices.shared.trainer else { return }

        let cloud: SplatCloud
        do {
            cloud = try await trainer.snapshot()
        } catch {
            // Early in a run there is genuinely nothing to show yet, and that
            // is what this throw means most of the time. Either way the next
            // tick asks again and the last good frame stays up.
            ProcessingLog.preview.debug(
                "No snapshot this time: \(error.localizedDescription, privacy: .public)"
            )
            return
        }

        guard !isTornDown else { return }
        guard cloud.count > 0 else { return }

        // Lightened OFF the main thread, and before anything is allocated on
        // the GPU. See `TrainingPreviewCloud`: this is the step that keeps the
        // preview from taking memory the trainer would otherwise have had.
        let drawable: SplatCloud
        do {
            drawable = try await Task.detached(priority: .utility) {
                try TrainingPreviewCloud.lightened(cloud)
            }.value
        } catch {
            ProcessingLog.preview.error(
                "Preview frame could not be prepared: \(error.localizedDescription, privacy: .public)"
            )
            return
        }

        guard !isTornDown, drawable.count > 0 else { return }

        do {
            try await renderer.load(drawable)
        } catch {
            // This one IS worth a sentence: the field existed and could not be
            // put on the GPU. Training is untouched by it.
            note = error.localizedDescription
            ProcessingLog.preview.error(
                "Preview upload failed: \(error.localizedDescription, privacy: .public)"
            )
            return
        }

        guard !isTornDown else { return }

        frameCamera()
        modelSplatCount = cloud.count
        drawnSplatCount = drawable.count
        frameCount += 1
        note = nil
        beginSettling()
    }

    /// Frames the scene once and then leaves the camera exactly where it is
    /// for the rest of the run. Re-framing on every refresh would move the
    /// viewpoint as the model grows, which is the jumping camera this preview
    /// exists to avoid.
    private func frameCamera() {
        guard !hasFramedCamera, let bounds = renderer.contentBounds else { return }
        hasFramedCamera = true
        camera.frame(bounds: bounds, fovDegrees: 70)
    }

    /// Runs the surface for `settleSeconds` so the renderer's staged reveal of
    /// a new cloud finishes, then parks it again.
    private func beginSettling() {
        isSettling = true
        settleTask?.cancel()
        let nanoseconds = UInt64(settleSeconds * 1_000_000_000)
        settleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: nanoseconds)
            guard !Task.isCancelled else { return }
            self?.isSettling = false
        }
    }

    // MARK: - Taking it down

    /// Gives the cloud back and stops everything this object started. Safe to
    /// call twice.
    func tearDown() {
        isTornDown = true
        refreshTask?.cancel()
        refreshTask = nil
        settleTask?.cancel()
        settleTask = nil
        isSettling = false
        renderer.unload()
    }
}

// MARK: - The light copy that actually gets drawn

/// Makes a training snapshot cheap enough to draw beside the run that produced
/// it.
///
/// WHY THIS EXISTS AT ALL, because it is not obvious and it is not optional.
/// The trainer measures free memory as it goes and lowers its own budget when
/// it is short (`TrainerBudgetGovernor.degradeForMemory` compares what it
/// holds against `os_proc_available_memory()`). A full snapshot is not small:
/// the Viewer's upload alone is 64 bytes a splat plus, at SH degree 3, another
/// 180 bytes a splat of higher-order colour, and the same again as Metal
/// buffers. Handing that straight to the renderer every few seconds could push
/// the trainer into making the model smaller. The user would then be shown a
/// smaller model, described accurately, with no way to know the preview had
/// caused it. That is the exact shape of dishonesty this project is built to
/// avoid, so the preview gives things up instead.
enum TrainingPreviewCloud {

    /// The most points a preview frame may draw. 300k at 64 bytes is about
    /// 19 MB in the upload and the same again on the GPU, which is small
    /// beside any training budget this app plans, and it is far more points
    /// than a 220 point tall picture can resolve anyway.
    static let splatCap = 300_000

    /// What comes off, and what it costs:
    ///
    /// * The higher-order spherical harmonics. They are how the colour of a
    ///   surface shifts as you walk around it, and this preview has a camera
    ///   that never moves, so nothing is lost that could have been seen. They
    ///   are also most of the bytes. The base colour of every point stays.
    /// * Points over the cap, dropped by taking a regular stride through the
    ///   field rather than the first N of it: the trainer holds its splats in
    ///   an order that is strongly clustered in space, so the first N would be
    ///   one corner of the room and a stride is the whole room, thinly.
    ///
    /// Returns the cloud unchanged when there is nothing worth taking off.
    static func lightened(_ cloud: SplatCloud) throws -> SplatCloud {
        let count = cloud.count
        guard count > 0 else { return cloud }
        if count <= splatCap && cloud.shDegree == .zero { return cloud }

        let step = Swift.max(1, (count + splatCap - 1) / splatCap)
        let kept = (count + step - 1) / step

        var positions: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var logScales: [SIMD3<Float>] = []
        var opacityLogits: [Float] = []
        var colorDC: [SIMD3<Float>] = []
        positions.reserveCapacity(kept)
        rotations.reserveCapacity(kept)
        logScales.reserveCapacity(kept)
        opacityLogits.reserveCapacity(kept)
        colorDC.reserveCapacity(kept)

        var index = 0
        while index < count {
            positions.append(cloud.positions[index])
            rotations.append(cloud.rotations[index])
            logScales.append(cloud.logScales[index])
            opacityLogits.append(cloud.opacityLogits[index])
            colorDC.append(cloud.colorDC[index])
            index += step
        }

        return try SplatCloud(
            shDegree: .zero,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacityLogits,
            colorDC: colorDC,
            shRest: []
        )
    }
}
