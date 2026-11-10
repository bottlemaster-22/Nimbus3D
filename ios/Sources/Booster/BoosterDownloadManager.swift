//
//  BoosterDownloadManager.swift
//  Booster
//
//  Downloads a finished job's result bundle into the phone's library, under
//  Documents/<brand>/Scans/<scanID>/model (BrandConfig.Folder.model) -
//  wherever the trained model and its exports belong for every other module
//  to find. Uses standard HTTP Range requests, chunked, so a dropped Wi-Fi
//  connection mid-download resumes from the last complete chunk rather than
//  starting the whole (potentially large) result over.
//

import Foundation

public struct BoosterDownloadProgress: Sendable {
    public var bytesReceived: Int64
    public var totalBytes: Int64
    public var currentFileName: String
    public var fraction: Double {
        totalBytes > 0 ? Double(bytesReceived) / Double(totalBytes) : 0
    }
}

actor BoosterDownloadManager {

    private let address: BoosterEndpointAddress

    init(address: BoosterEndpointAddress) {
        self.address = address
    }

    /// Downloads every file in the job's result manifest into
    /// `destinationDirectory`, creating it if needed. Returns the manifest
    /// that was actually fetched, so the caller can hand it to
    /// `BoosterJobStore` for the results list.
    func downloadResult(
        jobID: String,
        into destinationDirectory: URL,
        onProgress: @escaping @Sendable (BoosterDownloadProgress) -> Void
    ) async throws -> BoosterManifest {
        let manifest: BoosterManifest = try await BoosterHTTP.get(
            address,
            path: BoosterAPI.resultManifest(jobID)
        )

        try FileManager.default.createDirectory(
            at: destinationDirectory,
            withIntermediateDirectories: true
        )

        let totalBytes = manifest.totalByteCount
        var bytesReceivedSoFar: Int64 = 0

        for file in manifest.files {
            try Task.checkCancellation()
            // Sendable closures cannot capture a mutable local var, so
            // snapshot the running total as a `let` before handing the
            // closure off.
            let baseBytesReceived = bytesReceivedSoFar
            let received = try await downloadFile(
                jobID: jobID,
                file: file,
                into: destinationDirectory
            ) { fileBytesReceived in
                onProgress(
                    BoosterDownloadProgress(
                        bytesReceived: baseBytesReceived + fileBytesReceived,
                        totalBytes: totalBytes,
                        currentFileName: file.relativePath
                    )
                )
            }
            bytesReceivedSoFar += received
        }

        return manifest
    }

    // MARK: - Private

    private func downloadFile(
        jobID: String,
        file: BoosterManifestFile,
        into destinationDirectory: URL,
        onFileProgress: @escaping @Sendable (Int64) -> Void
    ) async throws -> Int64 {
        let destinationURL = destinationDirectory.appendingPathComponent(file.relativePath)
        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let fileManager = FileManager.default
        var startOffset: Int64 = 0
        if let existing = try? fileManager.attributesOfItem(atPath: destinationURL.path),
            let existingSize = existing[.size] as? Int64,
            existingSize <= file.byteCount
        {
            // A previous attempt got partway through this exact file; resume
            // rather than re-downloading bytes we already have.
            startOffset = existingSize
        } else {
            fileManager.createFile(atPath: destinationURL.path, contents: nil)
        }

        guard let handle = FileHandle(forWritingAtPath: destinationURL.path) else {
            throw BoosterError.resultDownloadFailed(
                "\"\(file.relativePath)\" could not be saved to your phone."
            )
        }
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(startOffset))

        let path = BoosterAPI.resultFile(jobID, relativePath: file.relativePath)
        var offset = startOffset

        while offset < file.byteCount {
            try Task.checkCancellation()
            let remaining = Int(min(Int64(BoosterAPI.chunkSize), file.byteCount - offset))
            let (data, isLast) = try await BoosterHTTP.getRange(
                address,
                path: path,
                offset: offset,
                length: remaining
            )
            if data.isEmpty { break }
            try handle.write(contentsOf: data)
            offset += Int64(data.count)
            onFileProgress(offset)
            if isLast { break }
        }

        guard offset >= file.byteCount else {
            throw BoosterError.resultDownloadFailed(
                "\"\(file.relativePath)\" stopped partway through downloading."
            )
        }

        try handle.close()
        try Self.verifyChecksum(at: destinationURL, expected: file.sha256, fileName: file.relativePath)
        return offset - startOffset
    }

    private static func verifyChecksum(
        at url: URL,
        expected: String,
        fileName: String
    ) throws {
        // Re-uses the same streaming hasher BoosterManifestBuilder uses for
        // outgoing files, so both directions are held to the same integrity
        // standard.
        let actual = try BoosterManifestBuilder.sha256(ofFileAt: url)
        guard actual.caseInsensitiveCompare(expected) == .orderedSame else {
            throw BoosterError.checksumMismatch(fileName: fileName)
        }
    }
}
