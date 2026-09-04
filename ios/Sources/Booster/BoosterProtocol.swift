//
//  BoosterProtocol.swift
//  Booster
//
//  THE WIRE PROTOCOL BETWEEN THIS PHONE CLIENT AND THE PC BOOSTER SERVER.
//
//  ================================ RECONCILED ================================
//  This module was originally built with docs/BOOSTER_PROTOCOL.md, CONTRACTS.md
//  and Core/Contracts.swift absent from disk, so this file defined the wire
//  protocol from the client side only. docs/BOOSTER_PROTOCOL.md has since been
//  written by reverse-specifying from this exact file, byte for byte - so the
//  two now agree by construction. docs/BOOSTER_PROTOCOL.md is the normative
//  spec going forward; this file must track it if it ever changes. The Python
//  server (booster/) still has to be checked against it independently -
//  nothing here has been run against a real server. See CONTRACTS.md section 6
//  and the "booster-client" row in MODULE_STATUS.md.
//  =============================================================================
//
//  Transport: plain HTTP (not HTTPS - this is LAN-only; self-signed TLS here
//  would be a false sense of security, not a real one) for control, pairing,
//  and chunked upload/download, plus one WebSocket for live progress.
//
//  All requests are namespaced under /v1.
//

import Foundation

/// Endpoint paths, versioning and tunables for talking to a PC Booster.
/// Byte-identical to what `booster/` (the Python server) must implement.
public enum BoosterAPI {

    /// Bumped only on a breaking wire-format change. Sent as the
    /// X-Nimbus-Api-Version header on every request so a mismatched pair
    /// fails with a clear "please update" message instead of a garbled parse.
    public static let apiVersion = "1"

    public static let pathPrefix = "/v1"

    /// Upload/download chunk size. 1 MiB balances progress granularity
    /// against HTTP request overhead on a home LAN.
    public static let chunkSize: Int = 1_048_576

    public static let requestTimeout: TimeInterval = 20
    public static let uploadChunkTimeout: TimeInterval = 60
    public static let downloadChunkTimeout: TimeInterval = 60

    /// WebSocket reconnect backoff for the live-progress stream.
    public static let progressReconnectDelays: [TimeInterval] = [1, 2, 5, 10, 20]

    // MARK: Paths

    public static func info() -> String { "\(pathPrefix)/info" }

    public static func pairRequest() -> String { "\(pathPrefix)/pair/requests" }
    public static func pairConfirm(requestID: String) -> String {
        "\(pathPrefix)/pair/requests/\(requestID)/confirm"
    }
    public static func pairStatus(requestID: String) -> String {
        "\(pathPrefix)/pair/requests/\(requestID)"
    }

    public static func createJob() -> String { "\(pathPrefix)/jobs" }
    public static func job(_ jobID: String) -> String { "\(pathPrefix)/jobs/\(jobID)" }
    public static func jobFile(_ jobID: String, relativePath: String) -> String {
        let encoded = relativePath
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? relativePath
        return "\(pathPrefix)/jobs/\(jobID)/files/\(encoded)"
    }
    public static func finalizeJob(_ jobID: String) -> String {
        "\(pathPrefix)/jobs/\(jobID)/finalize"
    }
    public static func cancelJob(_ jobID: String) -> String { job(jobID) }
    public static func jobStream(_ jobID: String) -> String { "\(pathPrefix)/jobs/\(jobID)/stream" }
    public static func resultManifest(_ jobID: String) -> String {
        "\(pathPrefix)/jobs/\(jobID)/result/manifest"
    }
    public static func resultFile(_ jobID: String, relativePath: String) -> String {
        let encoded = relativePath
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? relativePath
        return "\(pathPrefix)/jobs/\(jobID)/result/files/\(encoded)"
    }
}

// MARK: - Manifests

/// A flat description of every file in a capture bundle (or a result bundle),
/// used both to drive chunked upload/download and to let the server tell the
/// client "this file already has its first N bytes" for resume.
public struct BoosterManifest: Codable, Equatable, Sendable {
    public var scanID: String
    public var files: [BoosterManifestFile]
    /// Total bytes across every file, for a single top-level progress bar.
    public var totalByteCount: Int64 { files.reduce(0) { $0 + $1.byteCount } }

    public init(scanID: String, files: [BoosterManifestFile]) {
        self.scanID = scanID
        self.files = files
    }
}

public struct BoosterManifestFile: Codable, Equatable, Sendable {
    /// POSIX-style, forward-slash-separated path relative to the scan folder
    /// root, e.g. images/frame_20260903_141205_512.jpg.
    public var relativePath: String
    public var byteCount: Int64
    /// Lowercase hex SHA-256 of the whole file, used both for resume
    /// verification (per chunk) and the final integrity check (whole file).
    public var sha256: String

    public init(relativePath: String, byteCount: Int64, sha256: String) {
        self.relativePath = relativePath
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

// MARK: - Booster identity / health

public struct BoosterInfo: Codable, Equatable, Sendable {
    public var boosterID: String
    public var boosterName: String
    public var apiVersion: String
    public var freeDiskBytes: Int64
    public var gpuName: String?
    public var queueDepth: Int
}

// MARK: - Pairing

public struct BoosterPairRequestBody: Codable, Sendable {
    public var deviceID: String
    public var deviceName: String
    public var appVersion: String
}

public struct BoosterPairRequestResponse: Codable, Sendable {
    public var requestID: String
    /// True once the PC has shown a confirmation code and is waiting for the
    /// phone to type it back.
    public var awaitingCode: Bool
}

public struct BoosterPairConfirmBody: Codable, Sendable {
    /// The digits the user read off the PC's Booster window and typed into
    /// the phone. Never transmitted anywhere except this one LAN request.
    public var code: String
}

public enum BoosterPairStatusValue: String, Codable, Sendable {
    case pending
    case approved
    case denied
    case expired
}

public struct BoosterPairStatusResponse: Codable, Sendable {
    public var status: BoosterPairStatusValue
    /// Present only once status == .approved. A long-lived opaque token the
    /// phone stores in the Keychain and sends as "Authorization: Bearer" on
    /// every later request to this Booster.
    public var token: String?
    public var boosterID: String?
    public var boosterName: String?
}

// MARK: - Jobs

public struct BoosterCreateJobBody: Codable, Sendable {
    public var manifest: BoosterManifest
    public var appVersion: String
}

public struct BoosterCreateJobResponse: Codable, Sendable {
    public var jobID: String
    public var chunkSize: Int
    /// Bytes already on disk for each relative path, keyed by
    /// BoosterManifestFile.relativePath - 0 for a brand-new job, non-zero
    /// when this call is resuming a job the client asked for by scan ID.
    public var receivedByteOffsets: [String: Int64]
}

public struct BoosterChunkUploadResponse: Codable, Sendable {
    /// Total bytes the server now holds for this file after applying the
    /// chunk. If it does not equal offset + chunk.count, the client must
    /// discard its own assumption and resume from this value instead - the
    /// server is always the source of truth, never the client's own count.
    public var receivedByteOffset: Int64
}

public struct BoosterFinalizeResponse: Codable, Sendable {
    public var accepted: Bool
    /// Populated only when accepted == false, e.g. a checksum mismatch or a
    /// missing file - plain language, shown directly in the UI.
    public var reason: String?
}

/// Coarse machine-readable stage, always paired with a human-readable
/// message the UI shows verbatim (plain language, no jargon).
public enum BoosterJobStage: String, Codable, Equatable, Sendable {
    case queued
    case receiving
    case verifying
    case training
    case exporting
    case ready
    case failed
    case cancelled
}

public struct BoosterProgressEvent: Codable, Sendable {
    public var stage: BoosterJobStage
    /// 0...1, or nil when the stage has no meaningful percentage yet.
    public var fractionComplete: Double?
    public var message: String
    public var timestamp: Date
}

public struct BoosterJobStatusResponse: Codable, Sendable {
    public var jobID: String
    public var stage: BoosterJobStage
    public var fractionComplete: Double?
    public var message: String
    public var resultManifest: BoosterManifest?
}
