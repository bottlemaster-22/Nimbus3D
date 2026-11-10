//
//  CrashReports.swift
//
//  iOS's own crash and hang reports for this app, kept where the diagnostics
//  sender can find them.
//
//  WHY: a run the phone kills leaves nothing behind but the train census
//  written so far, and the owner cannot be asked to dig crash logs out of
//  Settings. MetricKit hands the app its own crash, hang and exception
//  reports (with call stacks) on the next launch. They are written as JSON
//  to Documents/<brand>/crash_reports and go out with every diagnostics send,
//  so the next failure explains itself.
//

import Foundation
import MetricKit
import os

enum CrashReportStore {
    /// `Documents/<brand>/crash_reports`, created when missing.
    static func directory() -> URL? {
        guard let scans = try? BrandConfig.scansDirectory() else { return nil }
        let url = scans.deletingLastPathComponent()
            .appendingPathComponent("crash_reports", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// The saved reports, newest first, at most `limit`.
    static func files(limit: Int = 8) -> [URL] {
        guard let directory = directory(),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.contentModificationDateKey]
              )
        else { return [] }
        let dated = entries.filter { $0.pathExtension == "json" }.map { url -> (URL, Date) in
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            return (url, date)
        }
        return dated.sorted { $0.1 > $1.1 }.prefix(limit).map { $0.0 }
    }

    /// Named by a hash of the report, so the same report delivered again
    /// (iOS re-offers past payloads on every launch) is written once.
    static func save(_ data: Data, kind: String) {
        guard let directory = directory(), !data.isEmpty else { return }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in data {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        let url = directory.appendingPathComponent("\(kind)-\(String(hash, radix: 16)).json")
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try? data.write(to: url)
    }
}

final class CrashReportCollector: NSObject, MXMetricManagerSubscriber {
    static let shared = CrashReportCollector()
    private let log = Logger(subsystem: BrandConfig.loggingSubsystem, category: "CrashReports")

    /// Called once at launch. Also saves anything iOS delivered before this
    /// launch registered.
    func start() {
        let manager = MXMetricManager.shared
        manager.add(self)
        for payload in manager.pastDiagnosticPayloads {
            CrashReportStore.save(payload.jsonRepresentation(), kind: "diagnostic")
        }
    }

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for payload in payloads {
            CrashReportStore.save(payload.jsonRepresentation(), kind: "diagnostic")
        }
        log.notice("Saved \(payloads.count, privacy: .public) diagnostic payload(s)")
    }

    /// Daily metrics carry the exit counts, including memory-limit kills
    /// (which are not crashes and never appear above).
    func didReceive(_ payloads: [MXMetricPayload]) {
        for payload in payloads {
            CrashReportStore.save(payload.jsonRepresentation(), kind: "metrics")
        }
    }
}
