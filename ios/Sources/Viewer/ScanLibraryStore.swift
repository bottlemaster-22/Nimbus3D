//
//  ScanLibraryStore.swift
//  Viewer
//
//  THE LIBRARY'S MODEL: WHAT IS ON THIS PHONE.
//
//  Scans are found by listing folders under `Documents/<brand>/Scans`, not by
//  reading an index file. That is deliberate and it is what CONTRACTS.md 6.5
//  settled: folders ARE the format, an index would be a second source of truth
//  that can go stale, and a scan copied onto the phone by hand has to show up.
//
//  Everything here is defensive about partial scans, because partial scans are
//  the normal case: a capture that was interrupted has frames and no pre-pass,
//  a scan sent to the Booster has a pre-pass and no model until it comes back,
//  and a scan whose training was cancelled has all three but a model with two
//  hundred iterations in it. Each of those renders as itself, with the honest
//  next step, rather than as an error.
//

import Foundation
import SwiftUI

// MARK: - Summary

/// One row of the library. Cheap: reading it touches `capture_bundle.json`,
/// `prepass/prepass_result.json` and `model/model.json` and nothing else - no
/// image, no depth map, no point cloud.
struct ScanSummary: Identifiable, Sendable {
    var scanID: ScanID
    var id: ScanID { scanID }

    var rootURL: URL
    var displayName: String
    var createdAt: Date

    var frameCount: Int
    var durationSeconds: Double?
    var sceneBounds: BoundingBox?

    var hasPrePass: Bool
    var hasModel: Bool
    /// True when `model/model.json` is there and readable but the `.ply` or
    /// `.spz` it points at is not. Kept apart from `problem` because the scan
    /// itself is fine: the photos and measurements are all still there, and
    /// building the model again fixes it.
    var modelFileMissing: Bool
    var modelSource: ModelSource?
    var splatCount: Int?
    var iterationsCompleted: Int?
    var heldOutPSNR: Float?
    var hasObservationRecord: Bool

    var coverageFraction: Float?
    var worstFindingSeverity: QCFinding.Severity?
    var qcFindingCount: Int

    var byteCount: Int64
    /// Relative path of a representative frame, used as the row's thumbnail.
    var thumbnailRelativePath: String?

    /// Set when the folder is there but something in it could not be read.
    /// Shown on the row rather than swallowed, so a scan that is quietly
    /// broken is visibly broken.
    var problem: String?

    var paths: ViewerScanPaths { ViewerScanPaths(scanID: scanID, root: rootURL) }
    var ref: CaptureBundleRef { CaptureBundleRef(scanID: scanID, rootURL: rootURL) }

    /// What this scan is waiting for, in the user's words. One sentence, and
    /// never a lie: if the trainer is not in this build, it says so, and if the
    /// phone is working on this scan right now, it says that instead of naming
    /// a step the user cannot start.
    ///
    /// `@MainActor` because it reads `NimbusServices.shared` and the processing
    /// coordinator, both of which are.
    @MainActor
    var nextStep: String {
        // A scan that could not be read says what is wrong with it, in its own
        // words, rather than a generic line the user cannot act on. The
        // sentences written into `problem` are already plain language.
        if let problem { return problem }
        if ScanProcessingCoordinator.shared.isWorking(on: scanID) {
            return "Working on this one now."
        }
        if hasModel { return "Ready to look at." }
        if modelFileMissing {
            return "The 3D file for this scan is missing. Building it again replaces it."
        }
        if hasPrePass {
            return NimbusServices.shared.trainer == nil
                ? "Checked over. Building the 3D model is not part of this build yet."
                : "Checked over. Ready to build the 3D model."
        }
        if frameCount > 0 {
            // Both steps are in this build and both are one tap away on the
            // processing screen, so this says so. Undersell is a lie too: the
            // old wording ("Not checked over yet") described the scan and left
            // out that the phone can do it right now.
            return NimbusServices.shared.prePass == nil
                ? "Recorded. Checking it over is not part of this build yet."
                : "Recorded. Ready to check over."
        }
        return "This folder has no frames in it."
    }

    /// The words for the button that starts `nextStep`, kept in the same file
    /// as the sentence so the two can never drift apart. Nil when there is
    /// nothing for the user to start: the scan is unreadable, empty, or the
    /// phone is already working on it.
    ///
    /// The button says the same thing the row's sentence promises, so a user
    /// who read "Ready to check over." taps "Check this scan over" and not
    /// some generic "Open".
    @MainActor
    var primaryActionTitle: String? {
        if problem != nil { return nil }
        if ScanProcessingCoordinator.shared.isWorking(on: scanID) { return nil }
        if hasModel { return "Look at this scan" }
        if modelFileMissing {
            return NimbusServices.shared.trainer == nil ? nil : "Build the 3D model again"
        }
        if hasPrePass {
            return NimbusServices.shared.trainer == nil ? nil : "Build the 3D model"
        }
        if frameCount > 0 {
            return NimbusServices.shared.prePass == nil ? nil : "Check this scan over"
        }
        return nil
    }

    /// True when this scan is waiting on the user to start something. The
    /// library uses it to put the freshly recorded scan forward as the thing
    /// to do next instead of leaving it as one more identical row.
    @MainActor
    var isWaitingOnYou: Bool {
        primaryActionTitle != nil && !hasModel
    }
}

// MARK: - Detail

/// A scan with its heavy indices loaded. Held only while a review screen is
/// on screen.
struct ScanDetail: Sendable {
    var summary: ScanSummary
    var bundle: CaptureBundle?
    var prePass: PrePassResult?
    var model: SplatModel?

    var paths: ViewerScanPaths { summary.paths }
    var ref: CaptureBundleRef { summary.ref }
}

// MARK: - Reading

/// The file reads behind the library. Free functions on purpose: they touch
/// no state, run off the main actor, and are the same code the review screen
/// uses when it wants one scan in detail.
enum ScanLibraryReader {

    static func readSummary(at root: URL) -> ScanSummary {
        let scanID = root.lastPathComponent
        let paths = ViewerScanPaths(scanID: scanID, root: root)
        let decoder = ContractsJSON.decoder()

        var summary = ScanSummary(
            scanID: scanID,
            rootURL: root,
            displayName: scanID,
            createdAt: folderDate(root) ?? Date(timeIntervalSince1970: 0),
            frameCount: 0,
            durationSeconds: nil,
            sceneBounds: nil,
            hasPrePass: false,
            hasModel: false,
            modelFileMissing: false,
            modelSource: nil,
            splatCount: nil,
            iterationsCompleted: nil,
            heldOutPSNR: nil,
            hasObservationRecord: false,
            coverageFraction: nil,
            worstFindingSeverity: nil,
            qcFindingCount: 0,
            byteCount: directorySize(root),
            thumbnailRelativePath: nil,
            problem: nil
        )

        // capture_bundle.json
        if let data = try? Data(contentsOf: paths.captureBundleJSON) {
            do {
                let bundle = try decoder.decode(CaptureBundle.self, from: data)
                guard bundle.formatVersion == CaptureBundle.currentFormatVersion else {
                    summary.problem =
                        "This scan is in format version \(bundle.formatVersion); "
                        + "this version of the app understands version "
                        + "\(CaptureBundle.currentFormatVersion)."
                    return summary
                }
                summary.displayName = bundle.displayName.isEmpty ? scanID : bundle.displayName
                summary.createdAt = bundle.createdAt
                summary.frameCount = bundle.frames.count
                summary.sceneBounds = bundle.sceneBounds
                if let first = bundle.frames.first, let last = bundle.frames.last {
                    summary.durationSeconds = Swift.max(
                        0,
                        last.timestampSeconds - first.timestampSeconds
                    )
                }
                summary.thumbnailRelativePath = representativeFrame(of: bundle)?.imagePath
            } catch {
                summary.problem = "The scan index could not be read: \(error.localizedDescription)"
            }
        } else {
            summary.problem = "This folder has no scan index (capture_bundle.json) in it."
        }

        // prepass/prepass_result.json - the file the pre-pass writes last, and
        // the only evidence that the check-over finished. A result in a format
        // version this build does not know reads as "not checked over" rather
        // than being decoded on a guess (docs/DATA_FORMAT.md section 9).
        if let data = try? Data(contentsOf: paths.prePassResultJSON),
           let prePass = try? decoder.decode(PrePassResult.self, from: data),
           prePass.formatVersion == PrePassResult.currentFormatVersion {
            summary.hasPrePass = true
            summary.coverageFraction = prePass.qcCard.coverageFraction
            summary.qcFindingCount = prePass.qcCard.findings.count
            summary.worstFindingSeverity = worstSeverity(prePass.qcCard.findings)
        }

        // model/model.json - written last by whoever built the model, so it
        // existing means the build finished. The splat file it points at is
        // checked too: a `model.json` with no `.ply` or `.spz` beside it is the
        // one way a scan can say "Ready to look at." and then fail to open.
        if let data = try? Data(contentsOf: paths.modelJSON),
           let model = try? decoder.decode(SplatModel.self, from: data) {
            let splatFiles = [model.plyPath, model.spzPath].compactMap { $0 }
            let hasSplatFile = splatFiles.contains {
                FileManager.default.fileExists(atPath: paths.url($0).path)
            }
            summary.hasModel = hasSplatFile
            summary.modelFileMissing = !hasSplatFile
            summary.modelSource = model.source
            // Only quoted when there is a file to back them up. Printing
            // "480k splats" for a model whose splat file has gone would be a
            // number about something that is not there.
            if hasSplatFile {
                summary.splatCount = model.splatCount
                summary.iterationsCompleted = model.iterationsCompleted
                summary.heldOutPSNR = model.heldOutPSNR
            }
            let observed = model.observedDirectionsPath.map { paths.url($0) }
                ?? paths.observedDirectionsBin
            summary.hasObservationRecord = FileManager.default.fileExists(atPath: observed.path)
        }

        return summary
    }

    /// The same reads as `readSummary`, plus the heavy indices the review and
    /// processing screens need.
    ///
    /// THE FORMAT-VERSION GATE IS THE SAME ONE `readSummary` APPLIES, and it
    /// has to be: this is the function `ScanProcessingCoordinator.run` asks
    /// before deciding an existing pre-pass can be reused. If this reader
    /// accepted a `prepass_result.json` written by a version it does not
    /// understand, the same folder would read as "not checked over" in the
    /// library row and as "reusable check-over" in the pipeline, and the
    /// trainer would be fed a file nobody checked. `docs/DATA_FORMAT.md`
    /// section 9 is explicit: a reader seeing a version it does not know must
    /// refuse and say so, not guess.
    ///
    /// The refusal is said out loud rather than swallowed: it goes into
    /// `summary.problem`, which the row and the screens already display, so a
    /// scan this build cannot read looks unreadable instead of looking empty.
    static func readDetail(_ summary: ScanSummary) -> ScanDetail {
        let decoder = ContractsJSON.decoder()
        let paths = summary.paths
        var summary = summary
        var refusals: [String] = []

        var bundle: CaptureBundle?
        if let data = try? Data(contentsOf: paths.captureBundleJSON),
           let decoded = try? decoder.decode(CaptureBundle.self, from: data) {
            if decoded.formatVersion == CaptureBundle.currentFormatVersion {
                bundle = decoded
            } else {
                refusals.append(
                    "The scan's own index is in format version "
                    + "\(decoded.formatVersion), and this version of the app "
                    + "understands version \(CaptureBundle.currentFormatVersion)."
                )
            }
        }

        var prePass: PrePassResult?
        if let data = try? Data(contentsOf: paths.prePassResultJSON),
           let decoded = try? decoder.decode(PrePassResult.self, from: data) {
            if decoded.formatVersion == PrePassResult.currentFormatVersion {
                prePass = decoded
            } else {
                refusals.append(
                    "The check-over saved with this scan is in format version "
                    + "\(decoded.formatVersion), and this version of the app "
                    + "understands version \(PrePassResult.currentFormatVersion)."
                )
            }
        }

        var model: SplatModel?
        if let data = try? Data(contentsOf: paths.modelJSON) {
            model = try? decoder.decode(SplatModel.self, from: data)
        }

        if prePass == nil { summary.hasPrePass = false }

        if !refusals.isEmpty {
            let sentence = refusals.joined(separator: " ")
                + " Nothing has been lost: the file was written by a different "
                + "version of the app, and this one will not guess at it."
            summary.problem = summary.problem.map { $0 + " " + sentence } ?? sentence
        }

        return ScanDetail(summary: summary, bundle: bundle, prePass: prePass, model: model)
    }

    /// The sharpest, best-exposed frame near the middle of the walk. A better
    /// thumbnail than "frame 0", which on nearly every capture is the ceiling
    /// or the floor while the user was still getting ready.
    static func representativeFrame(of bundle: CaptureBundle) -> CaptureFrame? {
        guard !bundle.frames.isEmpty else { return nil }
        let middle = bundle.frames.count / 2
        let window = 60
        let lower = Swift.max(0, middle - window)
        let upper = Swift.min(bundle.frames.count, middle + window)
        let candidates = bundle.frames[lower..<upper].filter { $0.bracket != .darker }
        let pool = candidates.isEmpty ? Array(bundle.frames[lower..<upper]) : candidates
        return pool.max { $0.qc.weight < $1.qc.weight } ?? bundle.frames[middle]
    }

    static func worstSeverity(_ findings: [QCFinding]) -> QCFinding.Severity? {
        if findings.contains(where: { $0.severity == .problem }) { return .problem }
        if findings.contains(where: { $0.severity == .warning }) { return .warning }
        return findings.isEmpty ? nil : .good
    }

    private static func folderDate(_ url: URL) -> Date? {
        let values = try? url.resourceValues(forKeys: [.creationDateKey])
        return values?.creationDate
    }

    /// Recursive byte count. Used for the "this scan is 2.1 GB" line, which is
    /// the number the user needs before deciding what to delete.
    static func directorySize(_ url: URL) -> Int64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let child as URL in enumerator {
            let values = try? child.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            total += Int64(values?.fileSize ?? 0)
        }
        return total
    }
}

// MARK: - Store

/// The library screen's view model.
@MainActor
final class ScanLibraryStore: ObservableObject {

    @Published private(set) var scans: [ScanSummary] = []
    @Published private(set) var isLoading = false
    /// Non-nil when the Scans folder itself could not be read. Different from
    /// an empty library, and shown differently.
    @Published private(set) var problem: String?

    /// Total bytes across every scan, for the storage line.
    var totalBytes: Int64 { scans.reduce(0) { $0 + $1.byteCount } }

    /// The scan the library should put forward as the thing to do next: the
    /// newest one that is waiting on the user to start something. Nil when
    /// every scan is finished, unreadable, empty, or already being worked on.
    ///
    /// `scans` is sorted newest first, so this is the scan that was just
    /// recorded, which is the whole point: coming back from the Capture tab,
    /// the user should not have to work out which row is theirs.
    var nextUpScanID: ScanID? {
        scans.first { $0.isWaitingOnYou }?.scanID
    }

    /// The same scan as a summary, for a screen that wants to draw its name
    /// and its button without searching the list again.
    var nextUpScan: ScanSummary? {
        scans.first { $0.isWaitingOnYou }
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }

        let outcome = await Task.detached(priority: .userInitiated) {
            () -> (scans: [ScanSummary], problem: String?) in
            do {
                let directories = try ViewerScanPaths.allScanDirectories()
                let summaries = directories.map { ScanLibraryReader.readSummary(at: $0) }
                return (summaries, nil)
            } catch {
                return ([], error.localizedDescription)
            }
        }.value

        // Newest first. Folder names are `scan_YYYYMMDD_HHMMSS`, so a
        // date sort and a name sort agree; the date is used because a scan
        // copied in by hand may not follow the naming rule.
        scans = outcome.scans.sorted { $0.createdAt > $1.createdAt }
        problem = outcome.problem
    }

    func detail(for summary: ScanSummary) async -> ScanDetail {
        await Task.detached(priority: .userInitiated) {
            ScanLibraryReader.readDetail(summary)
        }.value
    }

    /// Moves a scan to the trash. Deliberately `trashItem`, not
    /// `removeItem`: on iOS this puts the folder somewhere recoverable rather
    /// than destroying a capture the user can never take again.
    func delete(_ summary: ScanSummary) async -> String? {
        let url = summary.rootURL
        let outcome = await Task.detached(priority: .userInitiated) { () -> String? in
            do {
                try FileManager.default.trashItem(at: url, resultingItemURL: nil)
                return nil
            } catch {
                return error.localizedDescription
            }
        }.value
        if outcome == nil {
            scans.removeAll { $0.scanID == summary.scanID }
        }
        return outcome
    }

    func rename(_ summary: ScanSummary, to newName: String) async -> String? {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "A scan needs a name." }
        let paths = summary.paths

        let outcome = await Task.detached(priority: .userInitiated) { () -> String? in
            do {
                let data = try Data(contentsOf: paths.captureBundleJSON)
                var bundle = try ContractsJSON.decoder().decode(CaptureBundle.self, from: data)
                bundle.displayName = trimmed
                let encoded = try ContractsJSON.encoder().encode(bundle)
                try encoded.write(to: paths.captureBundleJSON, options: [.atomic])
                return nil
            } catch {
                return error.localizedDescription
            }
        }.value

        if outcome == nil, let index = scans.firstIndex(where: { $0.scanID == summary.scanID }) {
            scans[index].displayName = trimmed
        }
        return outcome
    }
}
