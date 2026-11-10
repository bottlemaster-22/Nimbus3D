//
//  CaptureScreen.swift
//  Capture
//
//  THE CAPTURE HUD (F9). What a person actually holds while they walk around
//  their living room.
//
//  ---------------------------------------------------------------------------
//  THE ONE IDEA THIS SCREEN IS BUILT ON
//  ---------------------------------------------------------------------------
//  The user is not looking at it. They are looking THROUGH the phone at the
//  wall, walking backwards around a sofa, holding it over their head to get
//  the ceiling. So sound and vibration are the guidance channel and the screen
//  is the redundant copy: every sentence here is also spoken by
//  `CaptureGuidanceEngine`, and the screen is what you check when you stop and
//  look down.
//
//  That is why the layout is what it is. One large sentence, one large button,
//  three numbers, and the room itself painted with what still needs doing. No
//  dashboards, no graphs, no jargon that needs a legend to be understood while
//  walking.
//
//  ---------------------------------------------------------------------------
//  WHAT IS ON IT AND WHY EACH THING EARNS ITS SPACE
//  ---------------------------------------------------------------------------
//  THE PAINTED ROOM     ARKit's mesh, coloured by whichever of the three
//                       coverage channels is furthest behind. Violet means
//                       walk around it, blue means get closer, amber means
//                       give it a steadier look, green means done, teal means
//                       glass (which has no fix and is not held against you). The picker
//                       lets someone ask for one channel on its own.
//  THE BLUR METER       Pixels of smear, live: the turn rate from the gyro
//                       times the shutter time, divided by the wide camera's
//                       0.0426 degrees per pixel. The amber and red marks come
//                       from `CaptureTuning`, which explains where they sit and
//                       why (the trainer only ever sees these photographs at
//                       720 px on the long edge, so capture pixels are not
//                       trainer pixels). It is the only number on screen that
//                       reacts to what your hands are doing this instant.
//  COVERAGE             One percentage, and the line where it counts as done.
//                       This is the finish criterion, not a decoration.
//  THE WINDOW CARD      When ARKit says the middle of the screen is a window,
//                       four things change and the user is told all four in
//                       one card with one button.
//  THE QUALITY CARD     After the scan, from poses and LiDAR only, in seconds.
//
//  Nothing here computes quality itself. The QC card comes from the pre-pass's
//  `quickQCCard`, and when that module is not in the build the screen says so
//  plainly and falls back to the handful of things the recording itself can
//  honestly state about itself.
//

import ARKit
import Combine
import Metal
import MetalKit
import SwiftUI
import UIKit

// =============================================================================
//  MARK: - Screen
// =============================================================================

/// The capture tab. Registered with `NimbusUI.shared.captureScreen`.
///
/// `@MainActor` on the whole view, not just on `body`: every helper property
/// below reads a main-actor model, and isolating the type is the one spelling
/// of that which is correct rather than merely tolerated.
@MainActor
public struct CaptureScreen: View {

    @StateObject private var model = CaptureScreenModel()

    public init() {}

    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.ignoresSafeArea()

                if model.stage.showsCamera {
                    cameraLayer
                }

                // Build 452: focus pins, and the double-tap that places one.
                if model.stage == .recording {
                    // Full screen, like the camera layer, so a tap and a pin
                    // use the same coordinates as the picture under them.
                    FocusPinsOverlay(store: model.focusPins)
                        .allowsHitTesting(false)
                        .ignoresSafeArea()
                    if model.focusMode {
                        Color.clear
                            .contentShape(Rectangle())
                            .ignoresSafeArea()
                            .onTapGesture(count: 2) { location in
                                model.placeFocus(at: location)
                            }
                    }
                }

                switch model.stage {
                case .ready, .starting:
                    CaptureStartPanel(model: model)
                case .recording, .finishing:
                    recordingOverlay
                case .report:
                    if let report = model.report {
                        CaptureReportPanel(report: report) {
                            // Finishing a capture used to end here, leaving the
                            // user on this tab with the check-over and
                            // model-building flow one tab away and nothing
                            // pointing at it. Hand them straight to the scan
                            // they just recorded. This starts no work: the user
                            // still chooses whether to check it over, and when.
                            let scanID = report.ref.scanID
                            model.dismissReport()
                            AppNavigation.shared.showSavedScan(scanID)
                        }
                    }
                case .failed:
                    CaptureProblemPanel(
                        message: model.errorText ?? "Something went wrong.",
                        onDismiss: { model.dismissError() }
                    )
                }
            }
            // Build 452: the FULL screen, safe areas included. The camera
            // layer ignores the safe area, so the size it is mapped with has
            // to as well; the inset size stretched the picture slightly and
            // would have put focus pins off their targets.
            .onChange(of: geometry.size) { _, _ in
                model.viewportSize = Self.fullScreenSize(geometry)
            }
            .onAppear { model.viewportSize = Self.fullScreenSize(geometry) }
        }
        .task { await model.prepare() }
        .onAppear {
            // `UIDevice.orientationDidChangeNotification` is only posted while
            // something has asked for orientation generation. Nothing else in
            // the app asks, so without this the subscription below never fires
            // and the live camera keeps whatever orientation it read once, at
            // `prepare()`, however the phone is turned afterwards.
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            model.refreshOrientation()
        }
        .onDisappear {
            UIDevice.current.endGeneratingDeviceOrientationNotifications()
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIDevice.orientationDidChangeNotification
            )
        ) { _ in
            model.refreshOrientation()
        }
    }

    private static func fullScreenSize(_ geometry: GeometryProxy) -> CGSize {
        let insets = geometry.safeAreaInsets
        return CGSize(
            width: geometry.size.width + insets.leading + insets.trailing,
            height: geometry.size.height + insets.top + insets.bottom
        )
    }

    @ViewBuilder
    private var cameraLayer: some View {
        if let renderer = model.renderer {
            CaptureCoverageMetalView(renderer: renderer)
                .ignoresSafeArea()
        } else if let note = model.rendererUnavailableNote {
            // No Metal overlay. The scan still records perfectly well, and
            // saying so beats a black rectangle nobody can interpret.
            VStack(spacing: 12) {
                Image(systemName: "square.dashed")
                    .font(.system(size: 44))
                    .foregroundStyle(.white.opacity(0.5))
                Text(note)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.75))
                    .padding(.horizontal, 32)
            }
        }
    }

    private var recordingOverlay: some View {
        VStack(spacing: 0) {
            CaptureTopBar(model: model)

            // Build 452: the focus menu, top right.
            HStack {
                Spacer()
                CaptureFocusMenu(model: model)
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            if model.focusMode {
                Text("Double-tap something detailed to give it extra attention.")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.black)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(Color.yellow))
                    .padding(.top, 8)
            }

            if let reason = model.stopReason {
                CaptureBanner(text: reason, tone: .problem)
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
            }

            if model.showsWindowCard {
                CaptureWindowCard(model: model)
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
            }

            Spacer(minLength: 0)

            CaptureBottomPanel(model: model)
        }
    }
}

// =============================================================================
//  MARK: - Model
// =============================================================================

/// Everything the capture screen knows, and the only place it talks to
/// `ARCaptureService`.
@MainActor
final class CaptureScreenModel: ObservableObject {

    enum Stage: Equatable {
        case ready
        case starting
        case recording
        case finishing
        case report
        case failed

        /// Whether the live camera belongs behind this stage.
        var showsCamera: Bool {
            self == .recording || self == .finishing || self == .starting
        }
    }

    @Published var stage: Stage = .ready
    @Published var live: CaptureLiveState?
    @Published var report: CaptureReport?
    @Published var errorText: String?

    @Published var scanName: String = ""
    @Published private(set) var channel: CaptureCoverageChannel?
    @Published var overlayStrength: Double = 0.75 {
        didSet { renderer?.overlayOpacity = Float(overlayStrength) }
    }
    @Published var soundOn = true {
        didSet { service?.isSoundEnabled = soundOn }
    }
    @Published var hapticsOn = true {
        didSet { service?.isHapticsEnabled = hapticsOn }
    }

    /// The user's own switch for the darker shots (F5).
    ///
    /// Defaults on, which is what `CaptureExposureController` defaults to, and
    /// is re-read from the service in `prepare()` so the two never disagree.
    /// Turning it back on only undoes the user's own switch: it cannot
    /// override the tracking guard, which is one-way on purpose.
    @Published var darkShotsOn = true {
        didSet { service?.setBracketingEnabled(darkShotsOn) }
    }

    @Published var isWindowModeActive = false
    @Published var windowCardDismissed = false
    @Published var windowActionNote: String?
    @Published var exposureIsLocked = false
    @Published var stopReason: String?
    @Published private(set) var rendererUnavailableNote: String?

    /// nil until Metal and the shaders are both available.
    private(set) var renderer: CaptureCoverageRenderer?

    /// The size of the camera layer, in points, which is what
    /// `ARFrame.displayTransform` expects.
    var viewportSize: CGSize = .zero

    private var service: ARCaptureService?
    private var interfaceOrientation: UIInterfaceOrientation = .portrait
    /// Build 452: focus points.
    @Published var focusMode = false
    @Published private(set) var focusCount = 0
    let focusPins = FocusPinStore()

    func placeFocus(at location: CGPoint) {
        guard let service,
              service.addFocusPoint(
                viewPoint: location, viewportSize: viewportSize, orientation: interfaceOrientation
              ) != nil
        else {
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
            return
        }
        focusCount = service.focusPoints.count
        focusMode = false
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    func removeLastFocus() {
        service?.removeLastFocusPoint()
        focusCount = service?.focusPoints.count ?? 0
    }
    private var liveTask: Task<Void, Never>?
    private var bundleRef: CaptureBundleRef?
    private var bytesAtFinish: Int64 = 0
    private var coverageAtFinish: Float = 0

    var showsWindowCard: Bool { isWindowModeActive && !windowCardDismissed }

    // MARK: - Setup

    func prepare() async {
        guard service == nil else { return }
        let resolved = (NimbusServices.shared.capture as? ARCaptureService)
            ?? ARCaptureService.shared
        service = resolved
        soundOn = resolved.isSoundEnabled
        hapticsOn = resolved.isHapticsEnabled
        darkShotsOn = !resolved.isBracketingTurnedOffByUser
        refreshOrientation()
        makeRenderer(for: resolved)
    }

    private func makeRenderer(for service: ARCaptureService) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            rendererUnavailableNote =
                "This iPhone will not let the app draw the coloured overlay, "
                + "so you get the plain camera instead. The scan itself "
                + "records exactly the same."
            return
        }
        guard
            let renderer = CaptureCoverageRenderer(
                device: device,
                coverageField: service.coverageField,
                meshStore: service.meshStore
            )
        else {
            rendererUnavailableNote =
                "The coloured overlay could not start on this iPhone, so you "
                + "get the plain camera instead. The scan itself records "
                + "exactly the same."
            return
        }
        renderer.overlayOpacity = Float(overlayStrength)
        self.renderer = renderer

        service.liveFrameObserver = { [weak self] frame in
            guard let self, let renderer = self.renderer else { return }
            guard self.viewportSize.width > 1, self.viewportSize.height > 1 else {
                return
            }
            renderer.ingest(
                frame: frame,
                viewportSize: self.viewportSize,
                orientation: self.interfaceOrientation
            )
            if let service = self.service, !service.focusPoints.isEmpty {
                self.focusPins.update(
                    frame: frame, points: service.focusPoints,
                    viewportSize: self.viewportSize, orientation: self.interfaceOrientation
                )
            } else if !self.focusPins.points.isEmpty {
                self.focusPins.points = []
            }
        }
    }

    /// Re-reads which way up the interface is, for the live camera layer.
    ///
    /// Two things this deliberately does NOT do. It does not take the first
    /// scene it finds: an app with a second scene (an external display, a
    /// Stage Manager window) can hand back one that is not the one being
    /// looked at, so the foreground-active scene is preferred. And it never
    /// lets `.unknown` through: `ARFrame.displayTransform(for:viewportSize:)`
    /// and `ARCamera.viewMatrix(for:)` both take this value, and what they do
    /// with `.unknown` is not documented, so an unusable answer becomes
    /// portrait (which is how this is held) rather than a guess passed on to
    /// ARKit.
    ///
    /// This is presentation only. Nothing that reaches the disk reads it: the
    /// intrinsics, the poses and the JPEG pixels are all in ARKit's own
    /// landscape sensor frame and they agree with each other there.
    func refreshOrientation() {
        let scenes = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
        let scene = scenes.first { $0.activationState == .foregroundActive }
            ?? scenes.first
        let reported = scene?.interfaceOrientation ?? .portrait
        interfaceOrientation = reported == .unknown ? .portrait : reported
    }

    // MARK: - Running a scan

    func startScanning() async {
        guard let service, stage == .ready || stage == .failed else { return }
        stage = .starting
        errorText = nil
        stopReason = nil
        windowCardDismissed = false
        windowActionNote = nil
        renderer?.reset()

        do {
            bundleRef = try await service.start(displayName: scanName)
            stage = .recording
            listenForLiveState()
        } catch {
            stage = .failed
            errorText = Self.plainText(for: error)
        }
    }

    /// Follows the live stream until the session ends it.
    ///
    /// The loop is never cancelled by hand: `finish()` and `cancel()` both
    /// finish the stream, which ends the loop on its own. Cancelling a task
    /// that is in the middle of writing a bundle is how a scan gets half
    /// written.
    private func listenForLiveState() {
        guard let service else { return }
        liveTask?.cancel()
        liveTask = Task { [weak self] in
            for await state in service.liveState {
                guard let self else { return }
                self.live = state
                self.isWindowModeActive = service.isWindowModeActive
                self.exposureIsLocked = service.isExposureLocked
                if !self.isWindowModeActive { self.windowCardDismissed = false }

                if let reason = service.stopReason, self.stopReason == nil {
                    // The session stopped itself for heat or for storage. It
                    // has kept everything; finishing writes the index so the
                    // user still gets the scan they walked for.
                    self.stopReason = reason
                    break
                }
            }
            guard let self, self.stopReason != nil, self.stage == .recording
            else { return }
            await self.finishScanning()
        }
    }

    func finishScanning() async {
        guard let service, stage == .recording else { return }
        stage = .finishing
        bytesAtFinish = live?.bytesWritten ?? 0
        coverageAtFinish = live?.coverageFraction ?? 0

        do {
            let bundle = try await service.finish()
            guard let ref = bundleRef ?? Self.fallbackRef(scanID: bundle.scanID) else {
                throw NimbusError.captureFailed(
                    "The scan was written but its folder could not be found again."
                )
            }
            report = CaptureReport(
                bundle: bundle,
                ref: ref,
                summary: CaptureSessionSummary.make(
                    bundle: bundle,
                    coverageFraction: coverageAtFinish,
                    bytesWritten: bytesAtFinish
                ),
                card: nil,
                cardNote: nil,
                stopReason: stopReason
            )
            stage = .report

            // The quality card is the pre-pass's job and is contracted to
            // arrive in under three seconds. It is fetched with the report
            // already on screen, so nobody is ever left watching a spinner.
            let outcome = await loadQualityCard(for: bundle, at: ref)
            report?.card = outcome.card
            report?.cardNote = outcome.note
        } catch {
            stage = .failed
            errorText = Self.plainText(for: error)
        }
    }

    private func loadQualityCard(
        for bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async -> (card: QCCard?, note: String?) {
        guard let prePass = NimbusServices.shared.prePass else {
            return (
                nil,
                "The full check-over of a scan comes from a part of the app "
                    + "that is not in this build yet. Everything below is "
                    + "measured from the recording itself."
            )
        }
        do {
            let card = try await prePass.quickQCCard(bundle: bundle, at: ref)
            return (card, nil)
        } catch {
            return (
                nil,
                "The scan saved fine, but the check-over could not finish: "
                    + Self.plainText(for: error)
            )
        }
    }

    func throwAwayScan() async {
        guard let service else { return }
        await service.cancel()
        live = nil
        stopReason = nil
        bundleRef = nil
        stage = .ready
    }

    func dismissReport() {
        report = nil
        live = nil
        bundleRef = nil
        stopReason = nil
        scanName = ""
        stage = .ready
    }

    func dismissError() {
        errorText = nil
        stage = .ready
    }

    // MARK: - Window mode actions

    func lockExposureForWindow() {
        guard let service else { return }
        if service.lockExposureAndWhiteBalance() {
            exposureIsLocked = true
            windowActionNote =
                "Brightness is held steady now, so the window stops making "
                + "the rest of the room flicker."
        } else {
            windowActionNote =
                "This iPhone will not let the app hold the brightness steady. "
                + "The scan carries on as normal."
        }
    }

    /// Hands the brightness back to the camera.
    ///
    /// Without this the hold is a one-way door. A user taps "Hold the
    /// brightness" at a bright window, walks on into a dark hallway, and every
    /// remaining frame of the scan is exposed for the window: the hallway
    /// comes out black, and there is no control anywhere in the app that gives
    /// the exposure back. `exposureIsLocked` is re-read from the service
    /// rather than assumed, because unlocking the white balance can fail on
    /// its own while the exposure unlock succeeded.
    func releaseExposureLock() {
        guard let service else { return }
        let released = service.unlockExposureAndWhiteBalance()
        exposureIsLocked = service.isExposureLocked
        if released {
            windowActionNote =
                "Brightness is back to adjusting itself, so the next room "
                + "will not come out dark."
        } else {
            windowActionNote =
                "This iPhone would not hand the brightness back. Finish this "
                + "scan and start another one to clear it."
        }
    }

    func markWindow() {
        guard let service else { return }
        if service.markWindowInFront() {
            windowActionNote =
                "Marked. The window is noted in the scan, so the rest of the "
                + "app knows not to trust the laser there."
        } else {
            windowActionNote =
                "Point a bit of the wall around the window into the middle of "
                + "the screen and tap again, so there is something to measure."
        }
    }

    func dismissWindowCard() {
        windowCardDismissed = true
    }

    // MARK: - Derived values for the view

    var coverageFraction: Float { live?.coverageFraction ?? 0 }
    var blurPixels: Float { live?.motionBlurPixels ?? 0 }
    var frameCount: Int { live?.frameCount ?? 0 }
    var elapsedSeconds: Double { live?.elapsedSeconds ?? 0 }
    var bytesWritten: Int64 { live?.bytesWritten ?? 0 }
    var trackingQuality: TrackingQuality { live?.trackingQuality ?? .notAvailable }
    var thermalLevel: ThermalLevel { live?.thermalLevel ?? .nominal }
    var hint: String? { live?.guidanceHint }
    var isCoverageDone: Bool {
        coverageFraction >= CaptureTuning.coverageDoneFraction
    }
    var shouldStandBackFromWindow: Bool { service?.windowShouldStandBack ?? false }
    var isBracketingAvailable: Bool { service?.isBracketingAvailable ?? false }
    var countedCoverageVoxels: Int { service?.countedCoverageVoxels ?? 0 }

    func applyChannel(_ channel: CaptureCoverageChannel?) {
        self.channel = channel
        renderer?.isolatedChannel = channel
    }

    private static func fallbackRef(scanID: ScanID) -> CaptureBundleRef? {
        guard let root = try? BrandConfig.scansDirectory() else { return nil }
        return CaptureBundleRef(
            scanID: scanID,
            rootURL: root.appendingPathComponent(scanID, isDirectory: true)
        )
    }

    static func plainText(for error: Error) -> String {
        if let nimbus = error as? NimbusError {
            return nimbus.errorDescription ?? "Something went wrong."
        }
        return error.localizedDescription
    }
}

// =============================================================================
//  MARK: - The Metal layer
// =============================================================================

/// Hosts the `MTKView` the coverage renderer draws into.
///
/// The renderer is built once by the model and handed in here, rather than
/// being created in `makeUIView`: SwiftUI rebuilds a representable whenever it
/// likes, and rebuilding a Metal pipeline at that rate is a stutter with no
/// cause anyone would ever find.
@MainActor
struct CaptureCoverageMetalView: UIViewRepresentable {

    let renderer: CaptureCoverageRenderer

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        renderer.configure(for: view)
        view.delegate = renderer
        view.isUserInteractionEnabled = false
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // Nothing to push: the renderer pulls what it needs from the ARKit
        // frame and from the coverage field.
    }
}

#Preview("Capture") {
    CaptureScreen()
}

// =============================================================================
//  MARK: - Focus points (build 452)
// =============================================================================

/// Where the focus pins are on screen this frame. Separate from the screen
/// model so a 60 Hz update redraws only the pins.
@MainActor
final class FocusPinStore: ObservableObject {
    @Published var points: [CGPoint] = []

    func update(
        frame: ARFrame, points world: [FocusPoint],
        viewportSize: CGSize, orientation: UIInterfaceOrientation
    ) {
        let view = frame.camera.viewMatrix(for: orientation)
        var visible: [CGPoint] = []
        for point in world {
            let p = SIMD3<Float>(point.position.x, point.position.y, point.position.z)
            // ARKit's view space looks down -Z: in front means z < 0.
            let inView = view * SIMD4<Float>(p, 1)
            guard inView.z < -0.05 else { continue }
            let screen = frame.camera.projectPoint(p, orientation: orientation, viewportSize: viewportSize)
            guard screen.x.isFinite, screen.y.isFinite else { continue }
            visible.append(screen)
        }
        if visible != points { points = visible }
    }
}

struct FocusPinsOverlay: View {
    @ObservedObject var store: FocusPinStore

    var body: some View {
        ZStack {
            ForEach(Array(store.points.enumerated()), id: \.offset) { _, point in
                ZStack {
                    Circle()
                        .stroke(Color.white, lineWidth: 2)
                        .frame(width: 34, height: 34)
                    Circle()
                        .fill(Color.yellow)
                        .frame(width: 10, height: 10)
                }
                .shadow(color: .black.opacity(0.5), radius: 3)
                .position(point)
            }
        }
    }
}

struct CaptureFocusMenu: View {
    @ObservedObject var model: CaptureScreenModel

    var body: some View {
        Menu {
            Button {
                model.focusMode.toggle()
            } label: {
                Label(
                    model.focusMode ? "Cancel focus point" : "Add focus point",
                    systemImage: "scope"
                )
            }
            if model.focusCount > 0 {
                Button(role: .destructive) {
                    model.removeLastFocus()
                } label: {
                    Label("Remove last focus point", systemImage: "arrow.uturn.backward")
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "scope")
                if model.focusCount > 0 { Text("\(model.focusCount)") }
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(model.focusMode ? Color.black : Color.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule().fill(model.focusMode ? Color.yellow : Color.black.opacity(0.45))
            )
        }
    }
}
