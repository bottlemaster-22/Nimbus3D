//
//  ScanExportScreen.swift
//  Viewer
//
//  GETTING A SCAN OUT OF THE PHONE.
//
//  Two different jobs live here and they are kept visibly separate, because
//  confusing them is how people lose work:
//
//    Share the 3D model     one file (.ply, .spz or .glb) that other software
//                           can open. Small. Not enough to rebuild the scan.
//
//    Keep the whole scan    a zip of the photos, the laser measurements and
//                           the camera track. Large. This is the one that can
//                           be trained again later, on this phone or on a
//                           computer.
//
//  Sending a scan to a computer on the Wi-Fi goes through the Booster module's
//  own client, which is real and already does chunked, resumable transfers.
//  Nothing here reimplements any part of it: this screen calls
//  `BoosterService.sendScan` and shows the Booster's own progress section.
//

import SwiftUI

// MARK: - Screen

struct ScanExportScreen: View {

    let summary: ScanSummary
    let detail: ScanDetail?

    @StateObject private var model: ScanExportModel
    @ObservedObject private var boosterClient = BoosterClient.shared
    @ObservedObject private var discovery = BoosterClient.shared.discovery

    init(summary: ScanSummary, detail: ScanDetail?) {
        self.summary = summary
        self.detail = detail
        _model = StateObject(
            wrappedValue: ScanExportModel(summary: summary, detail: detail)
        )
    }

    var body: some View {
        List {
            modelSection
            wholeScanSection
            boosterSection
            if !model.produced.isEmpty {
                readySection
            }
            #if DEBUG
            selfTestSection
            #endif
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Share this scan")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            model.loadFilesAlreadyOnDisk()
            #if DEBUG
            model.runExportSelfTestOnce()
            #endif
        }
        .onAppear { discovery.start() }
        .onDisappear { discovery.stop() }
        .alert(
            "That did not work",
            isPresented: Binding(
                get: { model.problem != nil },
                set: { if !$0 { model.clearProblem() } }
            )
        ) {
            Button("OK", role: .cancel) { model.clearProblem() }
        } message: {
            Text(model.problem ?? "")
        }
    }

    // MARK: The 3D model

    @ViewBuilder
    private var modelSection: some View {
        Section {
            if detail?.model == nil {
                Text(
                    "There is no 3D model for this scan yet, so there is nothing to "
                    + "share in a 3D file. The photos and measurements below are still "
                    + "here, and are what a model gets built from."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            } else {
                ForEach(ScanExportModel.offeredFormats, id: \.self) { format in
                    Button {
                        model.export(format)
                    } label: {
                        HStack(alignment: .firstTextBaseline) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(ScanExportModel.title(for: format))
                                Text(ScanExportModel.blurb(for: format))
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if model.working == .splatFile(format) {
                                ProgressView()
                            }
                        }
                    }
                    .disabled(model.working != nil)
                }
            }
        } header: {
            Text("Share the 3D model")
        } footer: {
            Text(
                "One file that other 3D software can open. It holds the finished "
                + "shapes and colours, not the original photos."
            )
        }
    }

    // MARK: The whole scan

    private var wholeScanSection: some View {
        Section {
            Button {
                model.packageWholeScan()
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Keep the whole scan as one file")
                        Text("About \(ViewerFormat.bytes(summary.byteCount)) before zipping.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if model.working == .wholeScan {
                        ProgressView()
                    }
                }
            }
            .disabled(model.working != nil)
        } header: {
            Text("Keep the whole scan")
        } footer: {
            Text(
                "The photos, the laser measurements and the camera path, zipped into "
                + "one file. This is the only version that can be built into a 3D model "
                + "again later, so it is the one worth backing up."
            )
        }
    }

    // MARK: Booster

    @ViewBuilder
    private var boosterSection: some View {
        if NimbusServices.shared.booster == nil {
            Section {
                Text(
                    "Sending scans to a computer is not part of this build yet."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            } header: {
                Text("Finish this on a computer")
            }
        } else {
            Section {
                if discovery.devices.isEmpty {
                    HStack(spacing: 10) {
                        if discovery.isSearching { ProgressView() }
                        Text(
                            discovery.isSearching
                                ? "Looking for a computer on your Wi-Fi..."
                                : "No computers found on your Wi-Fi yet."
                        )
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    }
                } else {
                    ForEach(discovery.devices) { device in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(device.name)
                                Text(status(for: device))
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Send") {
                                model.sendToBooster(device)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(
                                !device.isPaired || !device.isReachable || boosterClient.isBusy
                            )
                        }
                    }
                }
            } header: {
                Text("Finish this on a computer")
            } footer: {
                Text(
                    "Hands this scan to a computer on the same Wi-Fi to do the heavy "
                    + "work, then brings the finished model back to this phone. Nothing "
                    + "leaves your own network. Pair a computer on the Booster tab "
                    + "first."
                )
            }

            BoosterProgressSection(client: boosterClient)
        }
    }

    private func status(for device: BoosterDevice) -> String {
        switch (device.isPaired, device.isReachable) {
        case (true, true): return "Paired and ready"
        case (true, false): return "Paired, but not on your Wi-Fi right now"
        case (false, true): return "Found, but not paired yet. Pair it on the Booster tab."
        case (false, false): return "Not available"
        }
    }

    // MARK: The module's own round-trip self-check (DEBUG builds only)

    #if DEBUG
    /// `ExportSelfTest.runAll()` writes a small model out in every format,
    /// reads it back and checks it survived. It existed with no caller, so
    /// the PLY, SPZ and GLB round trips had never actually been run on a
    /// device. It runs once when this screen opens in a DEBUG build, and the
    /// button runs it again. Release builds do not compile any of this.
    private var selfTestSection: some View {
        Section {
            ForEach(model.selfTestReport.indices, id: \.self) { index in
                selfTestLine(model.selfTestReport[index])
            }
            Button("Run the file checks again") {
                model.runExportSelfTest()
            }
            .disabled(model.selfTestRunning)
        } header: {
            Text("Developer: file writers")
        } footer: {
            Text(
                "Writes a small test model to a scratch file in each format, reads it "
                + "back and checks nothing was lost. Only in a debug build."
            )
        }
    }

    private func selfTestLine(_ line: String) -> some View {
        Text(line)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(line.hasPrefix("FAIL") ? Color.orange : Color.secondary)
    }
    #endif

    // MARK: Files ready to send

    private var readySection: some View {
        Section {
            ForEach(model.produced) { asset in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(asset.url.lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(ViewerFormat.bytes(asset.byteCount))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    ShareLink(item: asset.url) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
        } header: {
            Text("Ready to send")
        } footer: {
            Text(
                "Every file this scan has already produced, saved on this iPhone under "
                + "the scan's own folder. Tap the share button to send one somewhere "
                + "else."
            )
        }
    }
}

// MARK: - Model

/// What the export folder is allowed to offer for sharing. A top-level
/// constant rather than a static on the model below, because the directory
/// listing that reads it runs off the main actor and `ScanExportModel` is
/// `@MainActor`.
private let shareableExportExtensions: Set<String> = ["ply", "spz", "glb", "zip"]

@MainActor
final class ScanExportModel: ObservableObject {

    enum Job: Equatable {
        case splatFile(ExportFormat)
        case wholeScan
    }

    @Published private(set) var working: Job?
    @Published private(set) var produced: [ExportedAsset] = []
    @Published private(set) var problem: String?

    #if DEBUG
    @Published private(set) var selfTestReport: [String] = []
    @Published private(set) var selfTestRunning = false
    private var didRunSelfTest = false
    #endif

    static let offeredFormats: [ExportFormat] = [.ply, .spz, .glb]

    private let summary: ScanSummary
    private let detail: ScanDetail?
    private var didLoadFilesOnDisk = false

    init(summary: ScanSummary, detail: ScanDetail?) {
        self.summary = summary
        self.detail = detail
    }

    func clearProblem() { problem = nil }

    // MARK: Copy

    static func title(for format: ExportFormat) -> String {
        switch format {
        case .ply: return "PLY file"
        case .spz: return "SPZ file (much smaller)"
        case .glb: return "GLB file"
        }
    }

    static func blurb(for format: ExportFormat) -> String {
        switch format {
        case .ply:
            return "The full-detail version. Opens in most 3D splat viewers and in Blender."
        case .spz:
            return "Roughly ten times smaller, with a little detail traded away. "
                + "Good for sending to someone."
        case .glb:
            return "The standard 3D file format. Viewers that do not understand splats "
                + "still show a coloured point cloud."
        }
    }

    // MARK: Actions

    func export(_ format: ExportFormat) {
        guard working == nil else { return }
        guard let model = detail?.model else {
            problem = "This scan has no 3D model to share yet."
            return
        }
        guard let exporter = MetalSplatRenderer.exporter() else {
            problem = ViewerError.exporterUnavailable.localizedDescription
            return
        }

        let paths = summary.paths
        // Prefer the lossless .ply on disk as the source. Re-encoding a .spz
        // into a .ply would hand the user a full-size file that has already
        // been through the small format's rounding, which looks like detail
        // that is not there.
        let sourceRelative: String?
        if let ply = model.plyPath, FileManager.default.fileExists(atPath: paths.url(ply).path) {
            sourceRelative = ply
        } else if let spz = model.spzPath,
                  FileManager.default.fileExists(atPath: paths.url(spz).path) {
            sourceRelative = spz
        } else {
            sourceRelative = nil
        }

        guard let sourceRelative else {
            problem = ViewerError.modelHasNoSplatFile(summary.scanID).localizedDescription
            return
        }
        let sourceURL = paths.url(sourceRelative)
        let scanID = summary.scanID

        working = .splatFile(format)
        Task { [weak self] in
            do {
                let imported = try await exporter.importSplatCloud(from: sourceURL)
                if let warning = imported.warning {
                    ViewerLog.review.notice("import warning: \(warning, privacy: .public)")
                }
                let asset = try await exporter.exportAsset(
                    imported.cloud,
                    scanID: scanID,
                    format: format
                )
                self?.record(asset)
            } catch {
                self?.problem = error.localizedDescription
            }
            self?.working = nil
        }
    }

    func packageWholeScan() {
        guard working == nil else { return }
        guard let exporter = MetalSplatRenderer.exporter() else {
            problem = ViewerError.exporterUnavailable.localizedDescription
            return
        }
        let scanID = summary.scanID
        working = .wholeScan
        Task { [weak self] in
            do {
                let asset = try await exporter.packageCaptureBundle(scanID: scanID)
                self?.record(asset)
            } catch {
                self?.problem = error.localizedDescription
            }
            self?.working = nil
        }
    }

    /// Hands the scan folder to the Booster module.
    ///
    /// This calls Core's `BoosterService` and nothing else: the discovery
    /// list, the pairing, the chunked upload and the download back are all the
    /// Booster's own, already-built code.
    func sendToBooster(_ device: BoosterDevice) {
        guard let booster = NimbusServices.shared.booster else {
            problem = "Sending scans to a computer is not part of this build yet."
            return
        }
        booster.sendScan(
            scanID: summary.scanID,
            scanDirectory: summary.rootURL,
            to: device
        )
    }

    private func record(_ asset: ExportedAsset) {
        produced.removeAll { $0.url == asset.url }
        produced.insert(asset, at: 0)
    }

    // MARK: Files already on disk

    /// Lists the files this scan exported in an EARLIER run of the app.
    ///
    /// Without this, `produced` only ever held what the current session made,
    /// so a file exported yesterday sat in the scan's export folder with no
    /// row and no share button anywhere in the app. The share sheet itself is
    /// SwiftUI's `ShareLink` in `readySection`.
    func loadFilesAlreadyOnDisk() {
        guard !didLoadFilesOnDisk else { return }
        didLoadFilesOnDisk = true
        let root = summary.rootURL
        let scanID = summary.scanID
        Task { [weak self] in
            let found = await Task.detached(priority: .utility) { () -> [ExportedAsset] in
                ScanExportModel.filesOnDisk(scanRoot: root, scanID: scanID)
            }.value
            self?.merge(found)
        }
    }

    /// Off the main actor: a directory listing plus one `resourceValues` call
    /// per file, both of which touch the disk.
    nonisolated static func filesOnDisk(scanRoot: URL, scanID: ScanID) -> [ExportedAsset] {
        let directory = scanRoot.appendingPathComponent(
            BrandConfig.Folder.exports,
            isDirectory: true
        )
        let keys: [URLResourceKey] = [
            .fileSizeKey, .contentModificationDateKey, .isRegularFileKey
        ]
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            // No export folder yet is the normal case for a scan nobody has
            // exported, not a problem worth an alert.
            return []
        }

        var assets: [ExportedAsset] = []
        for url in entries {
            let fileExtension = url.pathExtension.lowercased()
            guard shareableExportExtensions.contains(fileExtension) else { continue }
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile == true else { continue }
            assets.append(
                ExportedAsset(
                    url: url,
                    fileExtension: fileExtension,
                    byteCount: Int64(values?.fileSize ?? 0),
                    createdAt: values?.contentModificationDate
                        ?? Date(timeIntervalSince1970: 0),
                    scanID: scanID,
                    splatCount: nil
                )
            )
        }
        assets.sort { $0.createdAt > $1.createdAt }
        return assets
    }

    /// Adds anything found on disk that this session has not already listed.
    /// A file made in THIS session wins, because that record knows its splat
    /// count and a directory listing does not.
    private func merge(_ found: [ExportedAsset]) {
        var combined = produced
        for asset in found where !combined.contains(where: { $0.url == asset.url }) {
            combined.append(asset)
        }
        produced = combined
    }

    // MARK: The Export module's own round-trip checks (DEBUG only)

    #if DEBUG
    /// Runs the checks once per screen. `ExportSelfTest.runAll()` had no
    /// caller anywhere in the app, so its PLY / SPZ / GLB / zip round trips
    /// had never been run on a device.
    func runExportSelfTestOnce() {
        guard !didRunSelfTest else { return }
        didRunSelfTest = true
        runExportSelfTest()
    }

    func runExportSelfTest() {
        guard !selfTestRunning else { return }
        selfTestRunning = true
        selfTestReport = ["Checking the file writers..."]
        Task { [weak self] in
            let lines = await Task.detached(priority: .utility) { () -> [String] in
                ExportSelfTest.runAll()
            }.value
            for line in lines where line.hasPrefix("FAIL") {
                ViewerLog.review.error("export self-test: \(line, privacy: .public)")
            }
            self?.selfTestReport = lines
            self?.selfTestRunning = false
        }
    }
    #endif
}
