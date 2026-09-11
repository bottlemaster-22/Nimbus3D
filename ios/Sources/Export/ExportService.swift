//
//  ExportService.swift
//  Export
//
//  The module's public facade: writes a SplatCloud to .ply / .spz / .glb,
//  packages a scan directory for the Booster, imports .ply/.spz back into a
//  SplatCloud, and hands back a share sheet for a finished export.
//
//  Written before Core/Contracts.swift existed (see SplatCloud.swift's
//  header), so `ExportServicing` below is this module's own proposed shape
//  for that eventual shared contract - a plain protocol so Viewer/App could
//  depend on an abstraction without pulling in whatever Core ended up
//  defining.
//
//  UPDATE: Core/Contracts.swift now exists and defines `SplatExporting`
//  (CONTRACTS.md section 6.2) as the shared-contract name - `ExportService`
//  was already this module's concrete class name and `ExportServicing` was
//  already this module's own protocol name, so neither could be reused
//  without a rename inside shipped code. `ExportService` conforms to
//  `SplatExporting` via a short extension in `Sources/App/NimbusApp.swift`;
//  this file's `ExportServicing` stays as the module-local protocol and the
//  two coexist. TODO(nimbus): optional tidying - delete `ExportServicing`
//  and conform `ExportService` directly to `SplatExporting`, updating the
//  three method names below (`export`->`exportAsset`, add `scanID: ScanID`/
//  `async`) to match; not required, see CONTRACTS.md 6.2.
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif

public enum ExportFormat: String, Sendable {
    case ply
    case spz
    case glb

    var fileExtension: String { rawValue }
}

public protocol ExportServicing {
    /// Writes `cloud` to a new file named `<scanID>.<format>` under this
    /// scan's `export/` folder (`BrandConfig.Folder.exports`) and returns
    /// its URL.
    func export(_ cloud: SplatCloud, scanID: String, format: ExportFormat) throws -> URL

    /// Zips the given scan's capture bundle for the LAN Booster. See
    /// BoosterBundle.swift for exactly what is and is not included.
    func packageForBooster(scanID: String) throws -> BoosterBundle.Summary

    /// Reads a `.ply` file (this app's own export, or any compatible
    /// Gaussian-splat PLY) back into a SplatCloud.
    func importPLY(from url: URL) throws -> SplatCloud

    /// Reads a `.spz` file (versions 1-3; see SPZCodec.swift for the
    /// version-4 scope note) back into a SplatCloud, plus a non-fatal
    /// warning string when the file carried data this codec skipped.
    func importSPZ(from url: URL) throws -> (cloud: SplatCloud, warning: String?)
}

public final class ExportService: ExportServicing {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    // MARK: - Export

    public func export(_ cloud: SplatCloud, scanID: String, format: ExportFormat) throws -> URL {
        let exportDir = try exportsDirectory(scanID: scanID)
        let url = exportDir.appendingPathComponent("\(scanID).\(format.fileExtension)")

        switch format {
        case .ply: try PLYCodec.write(cloud, to: url)
        case .spz: try SPZCodec.write(cloud, to: url)
        case .glb: try GLTFExporter.writeGLB(cloud, to: url)
        }
        return url
    }

    private func exportsDirectory(scanID: String) throws -> URL {
        let scanDir = try scanDirectory(scanID: scanID)
        let exportDir = scanDir.appendingPathComponent(BrandConfig.Folder.exports, isDirectory: true)
        do {
            try fileManager.createDirectory(at: exportDir, withIntermediateDirectories: true)
        } catch {
            throw ExportError.ioFailure("creating export folder: \(error.localizedDescription)")
        }
        return exportDir
    }

    private func scanDirectory(scanID: String) throws -> URL {
        let scans = try BrandConfig.scansDirectory(fileManager: fileManager)
        return scans.appendingPathComponent(scanID, isDirectory: true)
    }

    // MARK: - Booster packaging

    public func packageForBooster(scanID: String) throws -> BoosterBundle.Summary {
        let scanDir = try scanDirectory(scanID: scanID)
        let exportDir = try exportsDirectory(scanID: scanID)
        let destination = exportDir.appendingPathComponent("\(scanID)-capture-bundle.zip")
        return try BoosterBundle.packageForBooster(scanDirectory: scanDir, to: destination)
    }

    // MARK: - Import-back

    public func importPLY(from url: URL) throws -> SplatCloud {
        try PLYCodec.read(from: url)
    }

    public func importSPZ(from url: URL) throws -> (cloud: SplatCloud, warning: String?) {
        let result = try SPZCodec.read(from: url)
        return (result.cloud, result.warning)
    }
}

// MARK: - Share sheet

#if canImport(UIKit)
public enum ExportShareSheet {
    /// Builds a share sheet (`UIActivityViewController`) for one or more
    /// exported files. This module does not own a presentation context (no
    /// view controller, no window), so it hands the controller back for the
    /// caller (App/Viewer) to present - and, on iPad, the caller must also
    /// set `popoverPresentationController.sourceView`/`sourceRect` before
    /// presenting, or UIKit will assert.
    public static func makeController(for fileURLs: [URL]) -> UIActivityViewController {
        UIActivityViewController(activityItems: fileURLs, applicationActivities: nil)
    }
}
#endif
