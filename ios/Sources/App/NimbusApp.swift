//
//  NimbusApp.swift
//  App
//
//  The app shell: @main, the first-run compatibility gate, and the tab bar.
//
//  The tab bar's selection lives in `AppNavigation` near the bottom of this
//  file, so a screen that finishes its job (a capture that has just been
//  saved) can hand the user on to the tab that continues it instead of
//  leaving them where they were.
//
//  This file is deliberately thin. It owns no scanning, no training, no
//  rendering and no networking - it decides which screen is on top and gets
//  out of the way. Every screen it shows comes from a module through
//  `NimbusUI` (Core/Contracts.swift). All of them are registered in the
//  integration block below; if one ever is not, the tab says so in plain words
//  rather than showing a fake screen that looks finished and does nothing.
//
//  The product is never named here. Every string the user reads that contains
//  the app's name pulls it from `BrandConfig`, which reads it from the one
//  brand block in ios/project.yml.
//

import SwiftUI
import os

// =============================================================================
//  MARK: - INTEGRATION BLOCK
//
//  THE ONE PLACE MODULES ARE WIRED IN. Every module of this app now exists on
//  disk and every line below is live; nothing here is commented out any more.
//  This list is the honest, at-a-glance answer to "what is actually built?",
//  and the one place to look when a screen is missing at runtime.
//
//  THE PRE-PASS AND THE TRAINER ARE NOW REACHABLE. `Sources/Pipeline` owns the
//  screen that starts them: the scan library pushes `ScanProcessingScreen` for
//  any scan without a model, and that screen calls `PrePassService.run` and
//  then `SplatTrainer.train` through this registry. It needs no registration of
//  its own, which is why there is no Pipeline line below.
//
//  SAID PLAINLY: reachable is not the same as proven. Neither the pre-pass nor
//  the trainer has ever been executed on a phone - there is no macOS on the
//  machine this was written on, so none of it has been compiled either. The
//  first real test is a device.
// =============================================================================

@MainActor
enum NimbusBootstrap {

    static func registerAvailableModules() {
        let services = NimbusServices.shared
        let ui = NimbusUI.shared

        // --- Onboarding (Sources/Onboarding).
        // The module's own front door, so a rename inside Onboarding is not an
        // edit to this file. It sets exactly two things:
        //     services.deviceCompatibility = DeviceCompatibilityProbe()
        //     ui.onboardingFlow            = OnboardingFlowView(report:onFinish:)
        OnboardingModule.register()

        // --- Capture (Sources/Capture).
        // `.shared`, never a fresh instance: this object owns an `ARSession`,
        // and two sessions competing for the same camera give you neither.
        services.capture = ARCaptureService.shared
        ui.captureScreen = { AnyView(CaptureScreen()) }

        // --- PrePass (Sources/PrePass).
        // The no-training pass: submaps, loop closure, carving, trust fields,
        // the quality card. `RootView` hands it the device tier once the
        // compatibility check has one, so its suggested budget is sized for
        // this phone rather than for a guess.
        services.prePass = PrePassPipeline()

        // --- Trainer (Sources/Trainer).
        services.trainer = MetalSplatTrainer()

        // --- Smart (Sources/Smart). Nothing to register, by design.
        // The trust field, the background model and the edge classifier are
        // collaborators that PrePass and Trainer construct and own. They are
        // deliberately not app-wide singletons, so there is nothing here.

        // --- Viewer (Sources/Viewer).
        //
        // The review screen builds its own renderer, one per open scan, so it
        // never touches this instance. Registering one anyway is deliberate
        // and costs a few milliseconds at launch: it is what makes
        // `NimbusServices.installedModules` able to say truthfully that Viewer
        // is in this build, and it compiles the preview shaders early, so a
        // shader that did not make it into the build shows up in the log at
        // launch rather than as a black rectangle the first time somebody
        // opens a scan.
        services.renderer = MetalSplatRenderer()
        ui.libraryScreen = { AnyView(ScanLibraryScreen()) }

        // --- Export (Sources/Export).
        services.exporter = ExportService()

        // --- Booster (Sources/Booster). Always optional, never required.
        services.booster = BoosterClient.shared
        ui.boosterScreen = { AnyView(BoosterTabView()) }

        // THE HONEST ANSWER TO "WHAT IS ACTUALLY BUILT?", SAID OUT LOUD.
        //
        // `NimbusServices.installedModules` was written for exactly this and
        // nothing read it, so the list above was only ever checkable by
        // reading this file. It is derived from the registrations that just
        // ran, not from a hand-kept list, so it cannot claim a module that
        // failed to register.
        // Built here rather than held as a static: this runs once, at launch.
        let log = Logger(
            subsystem: BrandConfig.loggingSubsystem, category: "App"
        )
        let installed = services.installedModules
        log.info(
            """
            \(BrandConfig.displayName, privacy: .public) \
            \(BrandConfig.versionString, privacy: .public) started. \
            Modules in this build: \(installed.joined(separator: ", "), privacy: .public)
            """
        )
        let expected = ["Onboarding", "Capture", "PrePass", "Trainer", "Viewer", "Export", "Booster"]
        let missing = expected.filter { !installed.contains($0) }
        if !missing.isEmpty {
            log.error(
                """
                These modules did NOT register and their screens will say so: \
                \(missing.joined(separator: ", "), privacy: .public)
                """
            )
        }
    }
}

// -----------------------------------------------------------------------------
//  Conformances that let already-built modules satisfy the Core protocols
//  without editing their files. Both are one-liners; see CONTRACTS.md.
// -----------------------------------------------------------------------------

extension BoosterClient: BoosterService {
    /// Core's app-level view of a job. `BoosterJobRecord` (the on-disk form
    /// Booster persists) has exactly these fields, so this is a rename, not a
    /// translation.
    public func jobs() -> [BoosterJob] {
        resultsHistory().map { record in
            BoosterJob(
                jobID: record.jobID,
                scanID: record.scanID,
                boosterID: record.boosterID,
                boosterName: record.boosterName,
                createdAt: record.createdAt,
                stage: record.stage,
                message: record.message,
                fractionComplete: nil,
                resultDirectory: record.resultDirectory
            )
        }
    }
}

extension ExportService: SplatExporting {
    public func exportAsset(
        _ cloud: SplatCloud,
        scanID: ScanID,
        format: ExportFormat
    ) async throws -> ExportedAsset {
        let url = try export(cloud, scanID: scanID, format: format)
        return ExportedAsset(
            url: url,
            fileExtension: format.rawValue,
            byteCount: Self.byteCount(of: url),
            createdAt: Date(),
            scanID: scanID,
            splatCount: cloud.count
        )
    }

    public func packageCaptureBundle(scanID: ScanID) async throws -> ExportedAsset {
        let summary = try packageForBooster(scanID: scanID)
        return ExportedAsset(
            url: summary.zipURL,
            fileExtension: "zip",
            byteCount: Self.byteCount(of: summary.zipURL),
            createdAt: Date(),
            scanID: scanID,
            splatCount: nil
        )
    }

    public func importSplatCloud(
        from url: URL
    ) async throws -> (cloud: SplatCloud, warning: String?) {
        switch url.pathExtension.lowercased() {
        case "ply":
            return (try importPLY(from: url), nil)
        case "spz":
            return try importSPZ(from: url)
        default:
            throw ExportError.unsupportedFormat(url.lastPathComponent)
        }
    }

    private static func byteCount(of url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }
}

// =============================================================================
//  MARK: - App entry point
// =============================================================================

@main
struct NimbusApp: App {

    init() {
        BrandConfig.assertConsistent()
        NimbusBootstrap.registerAvailableModules()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

// =============================================================================
//  MARK: - Root flow
// =============================================================================

/// Decides between the first-run gate and the main tabs.
///
/// The gate is not skippable on an incompatible device. That is the point: the
/// core of this app is a laser this phone either has or does not have, and
/// letting someone scan anyway would waste their time and then blame them for
/// it.
struct RootView: View {

    private enum Phase {
        case checking
        case onboarding(DeviceCapabilityReport)
        case blocked(DeviceCapabilityReport)
        case ready(DeviceCapabilityReport?)
    }

    @State private var phase: Phase = .checking
    @AppStorage(BrandConfig.defaultsPrefix + "onboardingCompleted")
    private var onboardingCompleted = false

    var body: some View {
        Group {
            switch phase {
            case .checking:
                CheckingDeviceView()

            case .onboarding(let report):
                if let flow = NimbusUI.shared.onboardingFlow {
                    flow(report) {
                        onboardingCompleted = true
                        phase = .ready(report)
                    }
                } else {
                    // Onboarding has not landed. Show the shell's own minimal
                    // gate rather than pretending the check never happened.
                    MinimalCompatibilitySummaryView(report: report) {
                        onboardingCompleted = true
                        phase = .ready(report)
                    }
                }

            case .blocked(let report):
                if let flow = NimbusUI.shared.onboardingFlow {
                    // Onboarding's own incompatible screen says more than the
                    // shell's can: it has the measurements behind the verdict,
                    // and it offers a re-check. If that re-check ever comes
                    // back compatible, the flow itself calls the completion
                    // and the app opens normally.
                    flow(report) {
                        onboardingCompleted = true
                        phase = .ready(report)
                    }
                } else {
                    IncompatibleDeviceView(report: report)
                }

            case .ready(let report):
                MainTabView(report: report)
            }
        }
        .task {
            await runCompatibilityCheck()
        }
    }

    private func runCompatibilityCheck() async {
        guard case .checking = phase else { return }

        guard let service = NimbusServices.shared.deviceCompatibility else {
            // No Onboarding module in this build. Do not invent a verdict:
            // go straight to the tabs, where every screen says what it is.
            phase = .ready(nil)
            return
        }

        let report = await service.evaluate()

        // The pre-pass sizes its suggested training budget from the device
        // tier. It has a weaker fallback (whatever this process can allocate
        // right now), but the real verdict is better and we have it here.
        //
        // This is also the app's ONE copy of the tier: `Sources/Pipeline`'s
        // processing coordinator reads it back from this same object rather
        // than keeping a second one, because two copies of a device tier is two
        // chances to disagree about what this phone can do.
        (NimbusServices.shared.prePass as? PrePassPipeline)?.deviceTier = report.tier

        switch report.tier {
        case .incompatible:
            phase = .blocked(report)
        case .full, .limited:
            phase = onboardingCompleted ? .ready(report) : .onboarding(report)
        }
    }
}

private struct CheckingDeviceView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
            Text("Checking what this iPhone can do...")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// =============================================================================
//  MARK: - Compatibility screens
//
//  Sources/Onboarding owns the finished versions of these. What is here is the
//  minimum that is honest: it shows the real report, in plain language, with
//  no invented reassurance.
// =============================================================================

/// The screen an incompatible device gets. Warm, specific, and never generic:
/// the reason names the actual missing thing on THIS phone.
struct IncompatibleDeviceView: View {
    let report: DeviceCapabilityReport

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("Sorry - \(BrandConfig.displayName) cannot run on this iPhone.")
                    .font(.title2.weight(.semibold))

                Text("We are trying our best to support as many devices as possible.")
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 12) {
                    Text("Why is my device incompatible?")
                        .font(.title.weight(.bold))

                    Text(report.incompatibleReason ?? Self.fallbackReason)
                        .font(.body)
                }

                if !report.features.isEmpty {
                    Divider()
                    FeatureListView(features: report.features)
                }

                Text("Device: \(report.deviceModel) - iOS \(report.systemVersion)")
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// Used only if a compatibility service returned `.incompatible` without
    /// filling in a reason - a bug, but the user should still get a sentence
    /// that is true rather than a blank space.
    private static let fallbackReason =
        "This app measures distance with a laser scanner (LiDAR) built into "
        + "some iPhone models. This iPhone does not have one, and there is no "
        + "way to make up that measurement from the camera alone."
}

/// The shell's stand-in for the first-run flow.
///
/// `Sources/Onboarding` ships the real one and registers it, so in a normal
/// build this view is never reached. It stays because the shell must still be
/// correct if that registration is ever removed: it shows the real report, in
/// plain language, and gets out of the way.
struct MinimalCompatibilitySummaryView: View {
    let report: DeviceCapabilityReport
    let onContinue: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("What this iPhone can do")
                .font(.title2.weight(.semibold))

            if report.tier == .limited {
                Text(
                    "This iPhone can record scans, but building the finished 3D "
                    + "model on the phone itself will be slow or may not fit in "
                    + "memory. You can send big scans to a computer on your "
                    + "Wi-Fi instead."
                )
                .foregroundStyle(.secondary)
            }

            FeatureListView(features: report.features)

            Spacer()

            Button("Continue", action: onContinue)
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
        }
        .padding(24)
    }
}

private struct FeatureListView: View {
    let features: [FeatureAvailability]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(features) { feature in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Image(
                        systemName: feature.isAvailable
                            ? "checkmark.circle.fill" : "xmark.circle.fill"
                    )
                    .foregroundStyle(feature.isAvailable ? Color.green : Color.secondary)

                    VStack(alignment: .leading, spacing: 3) {
                        Text(feature.title)
                        if let detail = feature.detail {
                            Text(detail)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
//  MARK: - Main tabs
// =============================================================================

/// WHICH TAB IS ON SCREEN, AND WHICH SCAN THE LIBRARY SHOULD LEAD WITH.
///
/// This exists because finishing a capture used to be a dead end: the report
/// panel dismissed itself and left the user on the Capture tab, with the whole
/// check-over and model-building flow sitting one tab away with nothing
/// pointing at it. A tab bar with no selection binding cannot be driven from
/// code, so it gets one, and the binding lives here rather than in `Capture`
/// so no module has to reach into another module's view.
///
/// It is a plain shared object rather than a `NimbusUI` entry because
/// `NimbusUI` (Core/Contracts.swift) publishes screens, not state, and this is
/// state. Every module in this app is compiled into the one target, so
/// `Capture` can call this directly.
///
/// `@MainActor` because it drives SwiftUI and nothing else.
@MainActor
final class AppNavigation: ObservableObject {

    static let shared = AppNavigation()

    enum Tab: Hashable {
        case capture
        case scans
        case booster
    }

    @Published var selectedTab: Tab = .capture

    /// The scan a screen asked the library to lead with, usually the one just
    /// recorded. The library clears it once it has acted on it; nothing here
    /// clears it, because this object cannot know when that has happened.
    @Published var scanToLeadWith: ScanID?

    private init() {}

    /// WHAT THE CAPTURE FLOW CALLS ONCE A SCAN IS SAFELY ON DISK.
    ///
    /// Switches to the Scans tab and asks the library to put that scan
    /// forward. It does not start any work: the user still chooses whether to
    /// check the scan over, and when.
    func showSavedScan(_ scanID: ScanID) {
        scanToLeadWith = scanID
        selectedTab = .scans
    }

    /// Switches to the Scans tab without singling out a scan.
    func showScans() {
        selectedTab = .scans
    }
}

@MainActor
struct MainTabView: View {
    let report: DeviceCapabilityReport?

    /// Shared, not owned: `Capture` writes to this same object when a scan is
    /// saved, and that write is what moves the user to their new scan.
    @ObservedObject private var navigation = AppNavigation.shared

    var body: some View {
        TabView(selection: $navigation.selectedTab) {
            captureTab
                .tabItem { Label("Scan", systemImage: "camera.viewfinder") }
                .tag(AppNavigation.Tab.capture)

            libraryTab
                .tabItem { Label("Scans", systemImage: "square.stack.3d.up") }
                .tag(AppNavigation.Tab.scans)

            boosterTab
                .tabItem { Label("Booster", systemImage: "bolt.horizontal.circle") }
                .tag(AppNavigation.Tab.booster)
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            if report?.tier == .limited {
                LimitedTierBanner()
            }
        }
    }

    @ViewBuilder
    private var captureTab: some View {
        if let screen = NimbusUI.shared.captureScreen {
            screen()
        } else {
            ScreenUnavailableView(part: "the scanning screen")
        }
    }

    @ViewBuilder
    private var libraryTab: some View {
        if let screen = NimbusUI.shared.libraryScreen {
            screen()
        } else {
            ScreenUnavailableView(part: "your scan library")
        }
    }

    @ViewBuilder
    private var boosterTab: some View {
        if let screen = NimbusUI.shared.boosterScreen {
            screen()
        } else {
            ScreenUnavailableView(part: "the computer helper")
        }
    }
}

/// Shown above the tabs on a LiDAR phone that is too old or too small to
/// train comfortably. It states the limit once, plainly, and does not nag.
private struct LimitedTierBanner: View {
    var body: some View {
        Text("Big scans on this iPhone may need a computer to finish.")
            .font(.footnote)
            .foregroundStyle(.secondary)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity)
            .background(Color.secondary.opacity(0.12))
    }
}

// =============================================================================
//  MARK: - When a screen is missing
//
//  Every module of this app is built and registered in the integration block
//  at the top of this file, so in a working build nothing below is ever shown.
//  It exists because a registry entry that is nil at runtime is a real fault,
//  and a black rectangle is the worst possible way to report one.
// =============================================================================

/// Shown in a tab whose screen did not register. Names what is missing, says
/// what is safe, and stops. No fake progress bar, no "coming soon".
@MainActor
struct ScreenUnavailableView: View {
    /// Plain-language name of the missing part, e.g. "the scanning screen".
    let part: String

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.secondary)

            Text("\(BrandConfig.displayName) could not open \(part).")
                .font(.headline)
                .multilineTextAlignment(.center)

            Text(
                "This is a fault in the app, not in anything you have recorded. "
                + "Your scans are still on this phone. Closing the app and "
                + "opening it again is worth a try."
            )
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)

            // The registry's own answer to "what is in this build?", so the
            // one screen a person reaches when something did not register can
            // also say what did. Read out over a phone call, this is the
            // difference between "it is broken" and one named missing module.
            Text("Parts of the app that did load: \(installedModules)")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var installedModules: String {
        NimbusServices.shared.installedModules.joined(separator: ", ")
    }
}

// =============================================================================
//  MARK: - Previews
// =============================================================================

#Preview("Tabs") {
    MainTabView(report: nil)
}

#Preview("Incompatible") {
    IncompatibleDeviceView(
        report: DeviceCapabilityReport(
            tier: .incompatible,
            hasLiDAR: false,
            deviceModel: "iPhone14,7",
            chipName: "A15 Bionic",
            totalMemoryBytes: 4 * 1024 * 1024 * 1024,
            availableMemoryBytes: 2 * 1024 * 1024 * 1024,
            systemVersion: "17.5",
            metalGPUFamily: "Apple8",
            lowPowerModeEnabled: false,
            sustainedPerformanceClass: 2,
            features: [
                FeatureAvailability(
                    key: "capture",
                    title: "Record a scan",
                    isAvailable: false,
                    detail: "Needs the laser scanner this iPhone does not have."
                )
            ],
            incompatibleReason:
                "This app measures distance with a laser (LiDAR) that this "
                + "iPhone does not have. Without it there is no way to know how "
                + "far away anything in the room is, and the scan would be "
                + "guesswork."
        )
    )
}
