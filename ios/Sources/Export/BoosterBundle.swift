//
//  BoosterBundle.swift
//  Export
//
//  Packages one scan's on-disk capture bundle into a single .zip for the
//  optional PC Booster: "the Booster capture-bundle packaging (zip of the
//  DATA_FORMAT layout)" per this module's mandate.
//
//  RECONCILED against docs/DATA_FORMAT.md section 1 and docs/BOOSTER_PROTOCOL.md
//  section 13, both now on disk. Section 13 settles the one open question this
//  file used to flag: this zip is deliberately NOT the Booster's wire
//  transport (that is the per-file chunked resumable manifest upload in
//  Sources/Booster, which can resume, doesn't double peak disk, and verifies
//  before unpacking) - this is the separate "hand a single file to AirDrop,
//  back up, or carry to a PC with no Wi-Fi" job. `SplatExporting`
//  .packageCaptureBundle (Core/Contracts.swift) is the contract name for it;
//  ExportService keeps the method named `packageForBooster` internally since
//  that is what `Sources/App/NimbusApp.swift`'s bridging extension already
//  calls (see CONTRACTS.md section 6.3 - the rename is cosmetic and the
//  Core-facing name is already correct).
//
//  What gets zipped, per DATA_FORMAT.md section 1's directory layout:
//  `capture_bundle.json` (the index - without it a receiving tool has no
//  join key for any sidecar) plus the `images/`, `sensor_data/`, `sparse/`,
//  `anchors/`, `mesh/`, and `prepass/` folders (each optional; skipped
//  silently when absent, e.g. `prepass/` before the pre-pass has run) under
//  the given scan directory - i.e. everything a PC trainer needs to run
//  F1-F6 from scratch on this scan. `BrandConfig.Folder.model`, `.exports`
//  and `.cache` are deliberately excluded: `model/` is this phone's own
//  (possibly partial, possibly absent) on-device training output, not an
//  input to a fresh Booster job; `export/` is where THIS module writes
//  finished .ply/.spz/.glb files, not scan input; `cache/` is explicitly
//  scratch space, safe to delete at any time per BrandConfig's own doc
//  comment.
//

import Foundation

public enum BoosterBundle {

    /// Sections copied into the bundle, in a fixed, deterministic order, so
    /// two zips of the same untouched scan are otherwise byte-identical
    /// (mtimes aside - see ZipWriter's DOS-timestamp handling).
    private static let includedSections = [
        BrandConfig.Folder.images,
        BrandConfig.Folder.sensorData,
        BrandConfig.Folder.sparse,
        BrandConfig.Folder.anchors,
        BrandConfig.Folder.mesh,
        BrandConfig.Folder.prePass,
    ]

    /// Root-level files (not inside any of `includedSections`) that
    /// DATA_FORMAT.md section 1 lists at the scan root. `capture_bundle.json`
    /// is the index every sidecar joins against by filename; a bundle without
    /// it is not openable by anything that does not already know this app's
    /// internal layout, so it is not optional the way a folder section is -
    /// its absence is still tolerated (a capture in progress may not have
    /// written it yet, see `sensor_data/frames.jsonl`'s crash-safety note),
    /// but it is never silently left out when present.
    private static let includedRootFiles = [
        "capture_bundle.json"
    ]

    /// Zips `scanDirectory` (e.g. `Documents/<brand>/Scans/<scan-id>`) into
    /// a new .zip at `destinationURL`, returning a small summary useful for
    /// a "sending N files, M MB to your PC" progress UI.
    @discardableResult
    static func packageForBooster(
        scanDirectory: URL,
        to destinationURL: URL
    ) throws -> Summary {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: scanDirectory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ExportError.sourceDirectoryMissing(scanDirectory.path)
        }

        let writer = try ZipWriter(creatingAt: destinationURL)

        var fileCount = 0
        var totalBytes: UInt64 = 0
        var includedSectionsFound: [String] = []

        for rootFile in includedRootFiles {
            let fileURL = scanDirectory.appendingPathComponent(rootFile, isDirectory: false)
            guard fm.fileExists(atPath: fileURL.path) else {
                continue  // not written yet, e.g. mid-capture - see doc comment above
            }
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey])
            try writer.addEntry(archivePath: rootFile, fileURL: fileURL)
            fileCount += 1
            totalBytes += UInt64(values?.fileSize ?? 0)
        }

        for section in includedSections {
            let sectionURL = scanDirectory.appendingPathComponent(section, isDirectory: true)
            var sectionIsDir: ObjCBool = false
            guard fm.fileExists(atPath: sectionURL.path, isDirectory: &sectionIsDir), sectionIsDir.boolValue else {
                continue  // optional section (e.g. prepass/ before the pre-pass has run) - fine to skip
            }
            includedSectionsFound.append(section)

            guard let enumerator = fm.enumerator(
                at: sectionURL,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }
            for case let fileURL as URL in enumerator {
                let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard values?.isRegularFile == true else { continue }
                let archivePath = try relativeArchivePath(of: fileURL, under: scanDirectory)
                try writer.addEntry(archivePath: archivePath, fileURL: fileURL)
                fileCount += 1
                totalBytes += UInt64(values?.fileSize ?? 0)
            }
        }

        guard fileCount > 0 else {
            throw ExportError.sourceDirectoryMissing(
                "no capture data found under \(scanDirectory.lastPathComponent) "
                    + "(expected one of: \(includedSections.joined(separator: ", ")))"
            )
        }

        let manifest = Manifest(
            bundleFormatVersion: 1,
            productName: BrandConfig.productName,
            productVersion: BrandConfig.versionString,
            scanID: scanDirectory.lastPathComponent,
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            sections: includedSectionsFound,
            fileCount: fileCount,
            totalBytes: totalBytes
        )
        try writer.addEntry(archivePath: "manifest.json", data: manifest.jsonData())

        try writer.finish()

        return Summary(
            zipURL: destinationURL,
            fileCount: fileCount,
            totalUncompressedBytes: totalBytes,
            sections: includedSectionsFound
        )
    }

    public struct Summary {
        public let zipURL: URL
        public let fileCount: Int
        public let totalUncompressedBytes: UInt64
        public let sections: [String]
    }

    private struct Manifest {
        let bundleFormatVersion: Int
        let productName: String
        let productVersion: String
        let scanID: String
        let generatedAt: String
        let sections: [String]
        let fileCount: Int
        let totalBytes: UInt64

        func jsonData() -> Data {
            let json: JSONValue = .object([
                ("bundleFormatVersion", .int(bundleFormatVersion)),
                ("productName", .string(productName)),
                ("productVersion", .string(productVersion)),
                ("scanID", .string(scanID)),
                ("generatedAt", .string(generatedAt)),
                ("sections", .array(sections.map { .string($0) })),
                ("fileCount", .int(fileCount)),
                ("totalBytes", .int(Int(totalBytes))),
            ])
            return Data(json.serialize().utf8)
        }
    }

    /// Builds the forward-slash archive path for `fileURL`, relative to
    /// `root`. ZIP entry names are always forward-slash regardless of host
    /// platform, per the PKWARE APPNOTE.
    private static func relativeArchivePath(of fileURL: URL, under root: URL) throws -> String {
        let filePath = fileURL.standardizedFileURL.path
        let rootPath = root.standardizedFileURL.path
        guard filePath.hasPrefix(rootPath) else {
            throw ExportError.ioFailure("\(fileURL.lastPathComponent) is not under \(root.lastPathComponent)")
        }
        var relative = String(filePath.dropFirst(rootPath.count))
        if relative.hasPrefix("/") { relative.removeFirst() }
        return relative.replacingOccurrences(of: "\\", with: "/")
    }
}
