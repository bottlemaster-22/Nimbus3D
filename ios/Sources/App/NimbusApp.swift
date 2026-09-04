//
//  NimbusApp.swift
//  App
//
//  The app shell: @main, the first-run compatibility gate, and the tab bar.
//
//  This file is deliberately thin. It owns no scanning, no training, no
//  rendering and no networking - it decides which screen is on top and gets
//  out of the way. Every screen it shows comes from a module through
//  `NimbusUI` (Core/Contracts.swift); anything not registered renders as an
//  honest placeholder that names the module which will fill it, rather than a
//  fake screen that looks finished and does nothing.
//
//  The product is never named here. Every string the user reads that contains
//  the app's name pulls it from `BrandConfig`, which reads it from the one
//  brand block in ios/project.yml.
//

import SwiftUI

// =============================================================================
//  MARK: - INTEGRATION BLOCK
//
//  THE ONE PLACE MODULES ARE WIRED IN. When a module lands, uncomment its
//  lines here; nothing else in the app changes. A line stays commented out
//  only while its module does not exist, so this list doubles as an honest,
//  at-a-glance answer to "what is actually built?".
// =============================================================================

@MainActor
enum NimbusBootstrap {

    static func registerAvailableModules() {
        let services = NimbusServices.shared
        let ui = NimbusUI.shared

        // --- Booster (Sources/Booster) - BUILT.
        services.booster = BoosterClient.shared
        ui.boosterScreen = { AnyView(BoosterTabView()) }

        // --- Export (Sources/Export) - BUILT.
        services.exporter = ExportService()

        // --- Onboarding (Sources/Onboarding) - not built yet.
        // services.deviceCompatibility = DeviceCompatibilityProbe()
        // ui.onboardingFlow = { report, done in
        //     AnyView(OnboardingFlowView(report: report, onFinish: done))
        // }

        // --- Capture (Sources/Capture) - not built yet.
        // services.capture = ARCaptureService()
        // ui.captureScreen = { AnyView(CaptureScreen()) }

        // --- PrePass (Sources/PrePass) - not built yet.
        // services.prePass = PrePassPipeline()

        // --- Trainer (Sources/Trainer) - not built yet.
        // services.trainer = MetalSplatTrainer()

        // --- Viewer (Sources/Viewer) - not built yet.
        // services.renderer = MetalSplatRenderer()
        // ui.libraryScreen = { AnyView(ScanLibraryScreen()) }
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
                IncompatibleDeviceView(report: report)

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

/// The shell's stand-in for the first-run flow: shows what this phone can and
/// cannot do, and gets out of the way.
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

            ModuleNotBuiltNote(module: "Onboarding")

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

struct MainTabView: View {
    let report: DeviceCapabilityReport?

    var body: some View {
        TabView {
            captureTab
                .tabItem { Label("Scan", systemImage: "camera.viewfinder") }

            libraryTab
                .tabItem { Label("Scans", systemImage: "square.stack.3d.up") }

            boosterTab
                .tabItem { Label("Booster", systemImage: "bolt.horizontal.circle") }
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
            ModulePlaceholderView(
                module: "Capture",
                title: "Scan a space",
                summary:
                    "Points the camera and the laser scanner at a room and "
                    + "records it, while telling you where to walk next.",
                willInclude: [
                    "Live coverage painted onto the room in three colours",
                    "A blur warning that reacts to how fast you turn",
                    "Spoken and buzzing guidance so you can watch the room, not the screen",
                    "A quality report a few seconds after you stop",
                ]
            )
        }
    }

    @ViewBuilder
    private var libraryTab: some View {
        if let screen = NimbusUI.shared.libraryScreen {
            screen()
        } else {
            ModulePlaceholderView(
                module: "Viewer",
                title: "Your scans",
                summary:
                    "Every scan on this phone: look around it, compare it "
                    + "against the real photos, and export it.",
                willInclude: [
                    "A fly-through that stays where you actually walked",
                    "Hatching over anything the scan never really saw",
                    "A photo-versus-scan slider",
                    "Export to .ply, .spz and .glb",
                ]
            )
        }
    }

    @ViewBuilder
    private var boosterTab: some View {
        if let screen = NimbusUI.shared.boosterScreen {
            screen()
        } else {
            ModulePlaceholderView(
                module: "Booster",
                title: "Use a computer",
                summary:
                    "Hands a big scan to a computer on the same Wi-Fi to "
                    + "finish, then brings the result back. Always optional.",
                willInclude: []
            )
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
//  MARK: - Placeholders
// =============================================================================

/// A screen for a module that is not in this build yet.
///
/// It says so plainly. It does not show a fake progress bar, a disabled
/// button, or a "coming soon" splash pretending to be a product - it names the
/// module, describes what will be there, and stops.
struct ModulePlaceholderView: View {
    let module: String
    let title: String
    let summary: String
    let willInclude: [String]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(summary)
                        .foregroundStyle(.secondary)

                    if !willInclude.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("What goes here")
                                .font(.headline)
                            ForEach(willInclude, id: \.self) { line in
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text("-")
                                    Text(line)
                                }
                                .font(.subheadline)
                            }
                        }
                    }

                    ModuleNotBuiltNote(module: module)

                    Spacer(minLength: 0)
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .navigationTitle(title)
        }
    }
}

/// The honest one-liner. Kept in one place so every placeholder says it the
/// same way and it is trivial to grep for what is still missing.
struct ModuleNotBuiltNote: View {
    let module: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "hammer")
            Text("Not built yet. This screen comes from the \(module) module.")
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
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
