//
//  BoosterScanLister.swift
//  Booster
//
//  Lists scans available to send to a Booster, by reading the folders under
//  BrandConfig.scansDirectory() directly. Sources/Viewer (not yet built) will
//  own the real scan-library screen; per CONTRACTS.md section 6.5, once it
//  exists the scan picker in the Booster tab should switch to it instead of
//  this folder scan. Until then this is real, working code: it lists exactly
//  the folders the rest of the app already writes scans into, per
//  BrandConfig.Folder.scans.
//

import Foundation

public struct BoosterScanSummary: Identifiable, Hashable, Sendable {
    public var id: String { scanID }
    public var scanID: String
    public var directory: URL
    public var createdAt: Date?
    public var fileCount: Int
    public var totalByteCount: Int64
}

enum BoosterScanLister {

    static func listAvailableScans() -> [BoosterScanSummary] {
        guard let scansDirectory = try? BrandConfig.scansDirectory() else { return [] }
        let fileManager = FileManager.default
        guard
            let entries = try? fileManager.contentsOfDirectory(
                at: scansDirectory,
                includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
                options: [.skipsHiddenFiles]
            )
        else { return [] }

        return entries.compactMap { url -> BoosterScanSummary? in
            guard
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .creationDateKey]),
                values.isDirectory == true
            else { return nil }

            var fileCount = 0
            var totalBytes: Int64 = 0
            if let enumerator = fileManager.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            ) {
                for case let fileURL as URL in enumerator {
                    guard
                        let fileValues = try? fileURL.resourceValues(
                            forKeys: [.isRegularFileKey, .fileSizeKey]
                        ),
                        fileValues.isRegularFile == true
                    else { continue }
                    fileCount += 1
                    totalBytes += Int64(fileValues.fileSize ?? 0)
                }
            }

            guard fileCount > 0 else { return nil }

            return BoosterScanSummary(
                scanID: url.lastPathComponent,
                directory: url,
                createdAt: values.creationDate,
                fileCount: fileCount,
                totalByteCount: totalBytes
            )
        }
        .sorted { ($0.createdAt ?? .distantPast) > ($1.createdAt ?? .distantPast) }
    }
}
