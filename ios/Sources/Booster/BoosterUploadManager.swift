//
//  BoosterUploadManager.swift
//  Booster
//
//  Chunked, resumable upload of a capture bundle to a paired Booster.
//
//  Resume model: sequential per-file offsets, not a chunk bitmap. For each
//  file the client uploads 1 MiB windows strictly in order; the server
//  always replies with the authoritative total bytes received for that
//  file, and the client trusts that number over its own bookkeeping. If the
//  app is killed mid-upload, relaunching and calling `upload(...)` again
//  with the same scanID re-creates/resumes the same job (BoosterCreateJobBody
//  is keyed by scanID server-side) and the server's receivedByteOffsets tell
//  the client exactly where to pick back up - no local resume-state file
//  needed at all, which also means resume survives a full app reinstall as
//  long as the Booster still has the job.
//

import CryptoKit
import Foundation

public struct BoosterUploadProgress: Sendable {
    public var bytesSent: Int64
    public var totalBytes: Int64
    public var currentFileName: String
    public var fraction: Double {
        totalBytes > 0 ? Double(bytesSent) / Double(totalBytes) : 0
    }
}

actor BoosterUploadManager {

    private let address: BoosterEndpointAddress

    init(address: BoosterEndpointAddress) {
        self.address = address
    }

    /// Builds the manifest (hashing every file - can be a real amount of CPU
    /// and I/O for a multi-gigabyte whole-house scan) then creates or resumes
    /// the job and uploads every byte not already on the server. Returns the
    /// jobID for the caller to hand to `BoosterJobMonitor` and, eventually,
    /// `finalize`.
    ///
    /// Deliberately takes `scanID`/`scanDirectory` rather than a pre-built
    /// `BoosterManifest`: this type is an `actor`, so hashing happens on its
    /// own executor, off the `@MainActor` caller (`BoosterClient`) that would
    /// otherwise freeze the UI for the seconds-to-minutes a large scan's
    /// checksums take.
    func upload(
        scanID: String,
        scanDirectory: URL,
        onProgress: @escaping @Sendable (BoosterUploadProgress) -> Void
    ) async throws -> String {
        let manifest = try BoosterManifestBuilder.buildManifest(
            scanID: scanID,
            scanDirectory: scanDirectory
        )
        let createBody = BoosterCreateJobBody(
            manifest: manifest,
            appVersion: BrandConfig.versionString
        )
        let created: BoosterCreateJobResponse = try await BoosterHTTP.post(
            address,
            path: BoosterAPI.createJob(),
            body: createBody
        )

        let totalBytes = manifest.totalByteCount
        var bytesSentSoFar: Int64 = created.receivedByteOffsets.values.reduce(0, +)
        onProgress(
            BoosterUploadProgress(
                bytesSent: bytesSentSoFar,
                totalBytes: totalBytes,
                currentFileName: manifest.files.first?.relativePath ?? ""
            )
        )

        for file in manifest.files {
            try Task.checkCancellation()
            let alreadyReceived = created.receivedByteOffsets[file.relativePath] ?? 0
            // Sendable closures cannot capture a mutable local var, so snapshot
            // the running total as a `let` before handing the closure off.
            let baseBytesSent = bytesSentSoFar - alreadyReceived
            let sentForThisFile = try await uploadFile(
                jobID: created.jobID,
                file: file,
                scanDirectory: scanDirectory,
                startingAt: alreadyReceived
            ) { fileBytesSent in
                onProgress(
                    BoosterUploadProgress(
                        bytesSent: baseBytesSent + fileBytesSent,
                        totalBytes: totalBytes,
                        currentFileName: file.relativePath
                    )
                )
            }
            bytesSentSoFar += (sentForThisFile - alreadyReceived)
        }

        return created.jobID
    }

    func finalize(jobID: String) async throws {
        let response: BoosterFinalizeResponse = try await BoosterHTTP.post(
            address,
            path: BoosterAPI.finalizeJob(jobID)
        )
        guard response.accepted else {
            throw BoosterError.serverRejected(
                response.reason ?? "The Booster could not accept this scan."
            )
        }
    }

    func cancel(jobID: String) async throws {
        try await BoosterHTTP.delete(address, path: BoosterAPI.cancelJob(jobID))
    }

    // MARK: - Private

    /// Uploads one file starting at `startingAt` (which may already equal the
    /// file's full size, in which case this is a no-op - the common case on
    /// a resumed job for files that finished before the interruption).
    /// Returns the server's final "bytes received" total for this file.
    private func uploadFile(
        jobID: String,
        file: BoosterManifestFile,
        scanDirectory: URL,
        startingAt: Int64,
        onFileProgress: @escaping @Sendable (Int64) -> Void
    ) async throws -> Int64 {
        guard startingAt < file.byteCount else {
            onFileProgress(file.byteCount)
            return file.byteCount
        }

        let fileURL = scanDirectory.appendingPathComponent(file.relativePath)
        guard let handle = FileHandle(forReadingAtPath: fileURL.path) else {
            throw BoosterError.scanFolderUnreadable(
                "\"\(file.relativePath)\" is missing from this scan's folder."
            )
        }
        defer { try? handle.close() }

        var offset = startingAt
        try handle.seek(toOffset: UInt64(offset))
        let path = BoosterAPI.jobFile(jobID, relativePath: file.relativePath)

        while offset < file.byteCount {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: BoosterAPI.chunkSize) ?? Data()
            if chunk.isEmpty { break }

            let chunkHash = Self.sha256Hex(chunk)
            // Sendable closures cannot capture a mutable local var, so pass
            // this iteration's offset in as a `let` snapshot.
            let chunkOffset = offset
            let response = try await withRetry {
                try await BoosterHTTP.putChunk(
                    address,
                    path: path,
                    offset: chunkOffset,
                    sha256: chunkHash,
                    data: chunk
                )
            }

            if response.receivedByteOffset != offset + Int64(chunk.count) {
                // The server disagrees with our bookkeeping (a dropped
                // response, a partial write on its side, etc). Trust it and
                // re-seek rather than silently drifting out of sync.
                offset = response.receivedByteOffset
                try handle.seek(toOffset: UInt64(offset))
            } else {
                offset += Int64(chunk.count)
            }
            onFileProgress(offset)
        }

        return offset
    }

    private func withRetry<T: Sendable>(
        attempts: Int = 3,
        _ body: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        var lastError: Error?
        for attempt in 0..<attempts {
            do {
                return try await body()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
                if attempt < attempts - 1 {
                    let delay = UInt64(pow(2.0, Double(attempt)) * 500_000_000)
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }
        throw lastError ?? BoosterError.connectionFailed("Upload failed after several tries.")
    }

    private static func sha256Hex(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
