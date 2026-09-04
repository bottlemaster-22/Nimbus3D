//
//  DeviceReportStore.swift
//  Onboarding
//
//  Persisting the compatibility report so the first-run screens do not have to
//  probe the hardware every time a view redraws.
//
//  Where it goes and why:
//
//    Application Support/<brand>/device_report.json
//
//  Not Documents. Documents is the user's folder: it is where scans live, it
//  is what the Files app shows, and it is what iTunes-style file sharing
//  exposes. A cache of what the phone told us about itself is not the user's
//  document, and putting it there would make it look like one. Application
//  Support is the documented place for app-managed data that should be backed
//  up but never presented as a file.
//
//  What this is NOT:
//
//    * It is not the source of truth. `DeviceCompatibilityProbe` is, and it
//      re-probes on every launch because the probe is cheap. The stored copy
//      exists so a screen can be drawn before, or without, a fresh probe.
//    * It is not a decision cache. Nothing reads a tier from here and acts on
//      it. A stored report from a previous iOS version could be wrong, which
//      is exactly why the schema version and the app version travel with it
//      and why a mismatch throws the file away rather than trusting it.
//

import Foundation

#if canImport(os)
import os
#endif

// MARK: - What gets written

/// One saved compatibility check, plus enough context to know whether it is
/// still worth believing.
public struct StoredDeviceReport: Codable, Sendable {

    /// Bumped whenever the shape of `DeviceCompatibilityFindings` changes in a
    /// way that a decoder cannot absorb. `load()` returns nil for anything
    /// that is not the current version: a stale device report is worth
    /// nothing, so migrating one would be effort spent on a file we are happy
    /// to regenerate in a millisecond.
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int

    /// The check itself.
    public var findings: DeviceCompatibilityFindings

    /// When it was written. Distinct from `findings.checkedAt`, which is when
    /// the probe ran. They are normally the same instant; they differ if a
    /// caller ever saves an older set of findings.
    public var savedAt: Date

    /// `BrandConfig.versionString` at save time, e.g. `0.1.0 (1)`.
    public var appVersion: String

    /// The iOS version at save time. A device that was incompatible on iOS 17
    /// is still incompatible on iOS 18 (the scanner does not grow back), but
    /// an OS upgrade is a good reason not to reuse a cached answer about what
    /// the OS is willing to offer this app.
    public var systemVersion: String

    /// - Parameter systemVersion: nil means "read it from `ProcessInfo` now",
    ///   which is what every caller wants. It is a parameter at all so a test
    ///   can write a record that claims to be from another iOS version.
    ///
    ///   It is resolved in the body rather than as a default argument because
    ///   the short-version helper is internal and this initialiser is public:
    ///   a public default argument may only reference public API.
    public init(
        schemaVersion: Int = StoredDeviceReport.currentSchemaVersion,
        findings: DeviceCompatibilityFindings,
        savedAt: Date = Date(),
        appVersion: String = BrandConfig.versionString,
        systemVersion: String? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.findings = findings
        self.savedAt = savedAt
        self.appVersion = appVersion
        self.systemVersion = systemVersion
            ?? ProcessInfo.processInfo.operatingSystemVersionString(short: true)
    }

    /// True when this report was written by a different build of the app, or
    /// under a different iOS version, or is simply old.
    ///
    /// Stale does not mean wrong and it does not mean discard: the screens
    /// still draw a stale report rather than showing nothing, and the probe
    /// replaces it moments later. It means "say when this was measured".
    public var isStale: Bool {
        if appVersion != BrandConfig.versionString { return true }
        let currentSystem = ProcessInfo.processInfo
            .operatingSystemVersionString(short: true)
        if systemVersion != currentSystem { return true }
        return Date().timeIntervalSince(savedAt) > StoredDeviceReport.staleAfter
    }

    /// Thirty days. Nothing about a phone's hardware changes in thirty days,
    /// so this is not a correctness bound. It is the point past which the UI
    /// should say "measured on 4 September" instead of quietly implying "now".
    static let staleAfter: TimeInterval = 30 * 24 * 60 * 60
}

// MARK: - The store

/// Reads and writes the one `StoredDeviceReport` on disk.
///
/// `@unchecked Sendable` is deliberate and narrow: the only mutable state is
/// `memo`, and every read and write of it is inside `lock`. File access is
/// serialised by the same lock, so two threads cannot half-write the file.
public final class DeviceReportStore: @unchecked Sendable {

    /// The one instance. `DeviceCompatibilityProbe` writes through it on every
    /// check and the onboarding screens read through it.
    public static let shared = DeviceReportStore()

    private let fileManager: FileManager
    private let lock = NSLock()
    private var memo: StoredDeviceReport?

    /// True once we have tried to read the file, so a missing file is not
    /// re-read from disk on every screen redraw.
    private var didAttemptLoad = false

    #if canImport(os)
    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Onboarding"
    )
    #endif

    /// - Parameter fileManager: injectable so a test can point the store at a
    ///   scratch container instead of the real one.
    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    // MARK: Location

    /// `Application Support/<brand>/device_report.json`.
    ///
    /// nil when the container could not be resolved, which on iOS means
    /// something is very wrong. Every caller treats nil as "cannot persist"
    /// and carries on: a device check that cannot be saved is still a valid
    /// device check.
    public var fileURL: URL? {
        guard
            let support = try? fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            )
        else { return nil }
        return support
            .appendingPathComponent(BrandConfig.documentsFolderName, isDirectory: true)
            .appendingPathComponent(Self.fileName, isDirectory: false)
    }

    static let fileName = "device_report.json"

    // MARK: Save

    /// Writes the findings out, replacing whatever was there.
    ///
    /// Deliberately cannot throw. It is called from the middle of
    /// `DeviceCompatibilityProbe.evaluateDetailed()`, and a device check that
    /// produced a correct verdict must not be turned into a failure because a
    /// cache file could not be written. A failure is logged and dropped.
    public func save(_ findings: DeviceCompatibilityFindings) {
        let record = StoredDeviceReport(findings: findings)

        lock.lock()
        memo = record
        didAttemptLoad = true
        lock.unlock()

        guard let url = fileURL else {
            #if canImport(os)
            log.error("Device report not saved: no Application Support directory.")
            #endif
            return
        }

        do {
            try fileManager.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try ContractsJSON.encoder().encode(record)
            // `.atomic` so a crash mid-write leaves the previous file intact
            // rather than a truncated one that fails to decode.
            try data.write(to: url, options: [.atomic])
        } catch {
            #if canImport(os)
            log.error(
                "Device report not saved: \(error.localizedDescription, privacy: .public)"
            )
            #endif
        }
    }

    // MARK: Load

    /// The last saved report, or nil if there is none this build can trust.
    ///
    /// Returns nil, rather than a partly-decoded record, for a file written by
    /// a different schema version or one that will not decode. Both cases are
    /// cheap to recover from: the caller re-probes.
    public func load() -> StoredDeviceReport? {
        lock.lock()
        if let cached = memo {
            lock.unlock()
            return cached
        }
        if didAttemptLoad {
            lock.unlock()
            return nil
        }
        lock.unlock()

        let decoded = readFromDisk()

        lock.lock()
        memo = decoded
        didAttemptLoad = true
        lock.unlock()

        return decoded
    }

    private func readFromDisk() -> StoredDeviceReport? {
        guard let url = fileURL else { return nil }
        guard fileManager.fileExists(atPath: url.path) else { return nil }

        do {
            let data = try Data(contentsOf: url)
            let record = try ContractsJSON.decoder()
                .decode(StoredDeviceReport.self, from: data)
            guard record.schemaVersion == StoredDeviceReport.currentSchemaVersion
            else {
                #if canImport(os)
                log.info(
                    """
                    Ignoring device report written by schema version \
                    \(record.schemaVersion, privacy: .public); this build reads \
                    \(StoredDeviceReport.currentSchemaVersion, privacy: .public).
                    """
                )
                #endif
                return nil
            }
            return record
        } catch {
            #if canImport(os)
            log.info(
                """
                Stored device report could not be read, re-probing: \
                \(error.localizedDescription, privacy: .public)
                """
            )
            #endif
            return nil
        }
    }

    // MARK: Clear

    /// Deletes the stored report and forgets the in-memory copy.
    ///
    /// Used by a "reset first run" developer action, and it is the right thing
    /// to call if the on-disk format is ever suspected of being the cause of a
    /// wrong screen.
    @discardableResult
    public func clear() -> Bool {
        lock.lock()
        memo = nil
        didAttemptLoad = false
        lock.unlock()

        guard let url = fileURL else { return false }
        guard fileManager.fileExists(atPath: url.path) else { return true }
        do {
            try fileManager.removeItem(at: url)
            return true
        } catch {
            #if canImport(os)
            log.error(
                "Device report not cleared: \(error.localizedDescription, privacy: .public)"
            )
            #endif
            return false
        }
    }
}
