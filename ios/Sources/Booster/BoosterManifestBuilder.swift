//
//  BoosterManifestBuilder.swift
//  Booster
//
//  Walks a capture bundle folder on disk (Documents/<brand>/Scans/<scanID>,
//  per BrandConfig.Folder - images/, sensor_data/, sparse/, anchors/, mesh/)
//  and produces the BoosterManifest that drives chunked upload, resume and
//  final integrity checking.
//
//  Operates on a plain folder URL rather than Core's `CaptureBundle`/
//  `CaptureBundleRef` on purpose: a manifest is a generic "hash every file
//  under this root" operation and has no need of the pipeline's richer
//  index, so any folder layout works. It also means this same code can
//  manifest a result bundle coming back from the Booster (see
//  BoosterDownloadManager's checksum verify, which reuses `sha256(ofFileAt:)`
//  below) without depending on the capture-only type.
//

import CryptoKit
import Foundation

enum BoosterManifestBuilder {

    /// Streams each file in 1 MiB windows so building a manifest for a large
    /// house scan never has to hold a whole file in memory at once.
    private static let hashWindow = 1_048_576

    static func buildManifest(scanID: String, scanDirectory: URL) throws -> BoosterManifest {
        let fileManager = FileManager.default
        guard
            let enumerator = fileManager.enumerator(
                at: scanDirectory,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            )
        else {
            throw BoosterError.scanFolderUnreadable(
                "This scan's folder could not be opened."
            )
        }

        var files: [BoosterManifestFile] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true else { continue }
            let byteCount = Int64(values.fileSize ?? 0)
            let filePath = relativePath(of: url, under: scanDirectory)
            let fileHash = try sha256(ofFileAt: url)
            files.append(
                BoosterManifestFile(
                    relativePath: filePath,
                    byteCount: byteCount,
                    sha256: fileHash
                )
            )
        }

        guard !files.isEmpty else {
            throw BoosterError.scanFolderUnreadable(
                "This scan does not have any files to send yet."
            )
        }

        // Deterministic order so upload progress and resume offsets line up
        // the same way every time this manifest is rebuilt.
        files.sort { $0.relativePath < $1.relativePath }
        return BoosterManifest(scanID: scanID, files: files)
    }

    private static func relativePath(of url: URL, under root: URL) -> String {
        let rootComponents = root.standardizedFileURL.pathComponents
        let fileComponents = url.standardizedFileURL.pathComponents
        let relative = fileComponents.dropFirst(rootComponents.count)
        return relative.joined(separator: "/")
    }

    /// Streaming SHA-256 of a single file. Exposed (not `private`) so
    /// BoosterDownloadManager can verify one just-downloaded result file
    /// without re-walking and re-hashing an entire directory.
    static func sha256(ofFileAt url: URL) throws -> String {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            throw BoosterError.scanFolderUnreadable(
                "\"\(url.lastPathComponent)\" could not be read."
            )
        }
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: hashWindow) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
