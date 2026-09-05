//
//  ProcessingSupport.swift
//  Pipeline
//
//  THE SMALL PIECES THE PROCESSING FLOW STANDS ON: what is already on disk,
//  what a training budget for THIS scan on THIS phone should be, and how a
//  thrown error becomes a sentence a person can act on.
//
//  Nothing here touches SwiftUI, ARKit or Metal, so every function below can be
//  called from a detached task without an actor hop.
//
//  ---------------------------------------------------------------------------
//  WHO WRITES WHAT
//  ---------------------------------------------------------------------------
//  `docs/DATA_FORMAT.md` fixes two files as the evidence that a stage finished:
//
//      prepass/prepass_result.json     the pre-pass finished
//      model/model.json                a model was built
//
//  Both are written by the module that produced them - `PrePassPipeline` writes
//  the first (atomically, see `PrePassBinary.write`), `MetalSplatTrainer` the
//  second - and both are the SAME paths `ScanLibraryStore` reads to decide what
//  a scan is. This file's `ensure...` helpers exist for the one case the
//  contract does not cover: a service that returns a result without having
//  written it. They write only when the file is missing or unreadable, so they
//  can never overwrite a richer file with a poorer one.
//

import Foundation
import os
import simd

// MARK: - Logging

enum ProcessingLog {
    static let coordinator = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "pipeline.coordinator"
    )
    static let budget = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "pipeline.budget"
    )
    /// The live training preview. Everything it says is a note, never a
    /// failure: a preview that cannot be refreshed leaves the last good frame
    /// on screen and the training run carries on untouched.
    static let preview = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "pipeline.preview"
    )
}

// MARK: - Turning an error into a sentence

/// Plain-language error text for a screen a non-technical person is looking at.
///
/// Every module already writes its own good sentences (`NimbusError`,
/// `TrainerError`, `ViewerError`, `BoosterError` are all `LocalizedError` with
/// real prose in them), so the job here is to find that sentence rather than to
/// invent a new one, and to add a line about what to do only when the error
/// itself does not already say.
enum ProcessingProblem {

    static func plainText(for error: Error) -> String {
        if isCancellation(error) {
            return "Stopped. Nothing that was already finished was thrown away."
        }
        if let nimbus = error as? NimbusError, let text = nimbus.errorDescription {
            return text
        }
        if let localized = error as? LocalizedError, let text = localized.errorDescription {
            return text
        }
        return "This did not finish: \(error.localizedDescription)"
    }

    /// What to try next, when the error type tells us something specific. nil
    /// when there is nothing honest to suggest, which is better than padding
    /// the screen with advice that does not apply.
    static func whatToTry(for error: Error) -> String? {
        if isCancellation(error) { return nil }
        guard let nimbus = error as? NimbusError else {
            return "Trying again with the phone cool and plugged in is worth a go. "
                + "Everything this scan recorded is still on the phone."
        }
        switch nimbus {
        case .outOfMemory:
            return "Closing your other apps frees some of that memory up. If it still "
                + "will not fit, a computer on your Wi-Fi can build this one instead."
        case .thermalAbort:
            return "Give the phone twenty minutes somewhere cool, then start it again. "
                + "It picks up from what was already finished."
        case .moduleNotInstalled:
            return "This is a gap in the app itself, not in your scan. Your recording "
                + "is safe where it is."
        case .prePassFailed, .trainingFailed, .captureFailed, .malformedData:
            return "Trying again with the phone cool and plugged in is worth a go. "
                + "Everything this scan recorded is still on the phone."
        case .deviceIncompatible, .scanNotFound, .cancelled:
            return nil
        }
    }

    static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let nimbus = error as? NimbusError, case .cancelled = nimbus { return true }
        return false
    }
}

// MARK: - What is already on disk

/// Reads and, only where nobody else did, writes the two files that say how far
/// a scan has got. Every path comes from `ViewerScanPaths`, which is the same
/// place `ScanLibraryStore` reads, so the library and this flow can never
/// disagree about where a result lives.
enum ProcessingArtifacts {

    /// The finished pre-pass for this scan, or nil.
    ///
    /// A result whose `formatVersion` this build does not know reads as "not
    /// checked over" rather than being decoded on a guess, exactly as
    /// `docs/DATA_FORMAT.md` section 9 requires.
    static func readPrePassResult(at paths: ViewerScanPaths) -> PrePassResult? {
        guard let data = try? Data(contentsOf: paths.prePassResultJSON) else { return nil }
        guard let result = try? ContractsJSON.decoder().decode(PrePassResult.self, from: data),
              result.formatVersion == PrePassResult.currentFormatVersion
        else { return nil }
        return result
    }

    static func readModel(at paths: ViewerScanPaths) -> SplatModel? {
        guard let data = try? Data(contentsOf: paths.modelJSON) else { return nil }
        return try? ContractsJSON.decoder().decode(SplatModel.self, from: data)
    }

    /// Whether the splat file a `model.json` points at is actually there.
    ///
    /// A `model.json` with no `.ply` or `.spz` beside it is the one way a scan
    /// can look finished in the library and then fail to open, so the library
    /// checks this before it says "Ready to look at."
    static func splatFileExists(for model: SplatModel, at paths: ViewerScanPaths) -> Bool {
        let fileManager = FileManager.default
        for relative in [model.plyPath, model.spzPath].compactMap({ $0 }) {
            if fileManager.fileExists(atPath: paths.url(relative).path) { return true }
        }
        return false
    }

    /// Writes `prepass/prepass_result.json` only if it is missing or cannot be
    /// read back. `PrePassPipeline` writes it itself as its last act, so in the
    /// normal case this reads the file, finds it good, and does nothing.
    static func ensurePrePassResultWritten(
        _ result: PrePassResult,
        at ref: CaptureBundleRef
    ) throws {
        let paths = ViewerScanPaths(ref: ref)
        if readPrePassResult(at: paths) != nil { return }
        do {
            let data = try ContractsJSON.encoder().encode(result)
            try write(data, to: paths.prePassResultJSON)
        } catch {
            throw NimbusError.prePassFailed(
                "the results of the check could not be saved: \(error.localizedDescription)"
            )
        }
    }

    /// Writes `model/model.json` only if it is missing or cannot be read back.
    /// `MetalSplatTrainer` writes it itself; this is the safety net.
    static func ensureModelWritten(_ model: SplatModel, at ref: CaptureBundleRef) throws {
        let paths = ViewerScanPaths(ref: ref)
        if readModel(at: paths) != nil { return }
        do {
            let data = try ContractsJSON.encoder().encode(model)
            try write(data, to: paths.modelJSON)
        } catch {
            throw NimbusError.trainingFailed(
                "the finished model could not be saved: \(error.localizedDescription)"
            )
        }
    }

    /// Removes an index file that exists but cannot be read, so a run that died
    /// half way cannot leave the library claiming a stage finished when it did
    /// not. Only ever deletes a file that is already unreadable, and returns the
    /// sentence to show for it.
    ///
    /// Nothing else in `prepass/` or `model/` is touched: those files are real
    /// work, the next run overwrites them, and the index is what the library
    /// reads.
    @discardableResult
    static func discardUnreadableIndexFiles(at paths: ViewerScanPaths) -> [String] {
        var notes: [String] = []

        if removeIfNotEvenJSON(paths.prePassResultJSON) {
            notes.append(
                "A half-written record of the check-over was cleared away, so this scan "
                + "says it still needs checking over rather than pretending it is done."
            )
        }

        if removeIfNotEvenJSON(paths.modelJSON) {
            notes.append(
                "A half-written record of the 3D model was cleared away, so this scan "
                + "does not claim to have a model it cannot open."
            )
        }

        return notes
    }

    /// Deletes `url` only when it is there and is not valid JSON at all, which
    /// means a writer died part way through it.
    ///
    /// The test is deliberately "is this even JSON", not "can this build decode
    /// it": a file from a future format version decodes as nothing here and is
    /// still somebody's real work, and deleting it would be the app destroying
    /// data it merely did not understand.
    private static func removeIfNotEvenJSON(_ url: URL) -> Bool {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: url.path) else { return false }
        guard let data = try? Data(contentsOf: url) else { return false }
        if (try? JSONSerialization.jsonObject(with: data)) != nil { return false }
        do {
            try fileManager.removeItem(at: url)
        } catch {
            return false
        }
        ProcessingLog.coordinator.notice(
            "Removed unparseable \(url.lastPathComponent, privacy: .public)"
        )
        return true
    }

    private static func write(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: [.atomic])
    }
}

// MARK: - The budget

/// A training budget for one scan on one phone, with everything that went into
/// it kept so the screen can say WHY it is that size.
///
/// Nothing here assumes a scene fits. Every number is either measured
/// (`os_proc_available_memory()` through `DeviceMemoryFacts`, the frame count,
/// the scan's own extent) or comes from the compatibility probe, and the
/// trainer is still free to lower all of it once it sees the real thing.
struct ProcessingBudgetPlan: Sendable {
    var budget: TrainingBudget

    var tier: DeviceTier
    /// False when no compatibility report reached this app run and the tier had
    /// to be worked out from free memory instead. Said on screen, not hidden.
    var tierWasMeasured: Bool

    var sceneExtentMeters: Float
    /// Plain-language description of where the extent came from.
    var extentSource: String

    var frameCount: Int
    var availableMemoryBytes: UInt64
    var memoryWasMeasured: Bool

    /// Roughly what the splat field alone will occupy at the cap, at the same
    /// 200 bytes per splat `TrainingBudget.recommended` reasons with.
    var estimatedSplatBytes: UInt64

    /// Present when this scan is big for this phone. Never a refusal: the
    /// trainer shrinks itself as it goes, and the user is told what to expect.
    var sizeWarning: String?

    /// The "why it is this size" lines, in the order they should be read.
    var lines: [String]
}

enum ProcessingBudgetPlanner {

    /// Bytes per resident splat, asked of the trainer's own GPU layouts so no
    /// two parts of the app can quietly disagree about how big a splat is.
    ///
    /// This was a hardcoded 200 that mirrored an equally wrong 200 in
    /// `TrainingBudget.recommended`. Because both sides used the same wrong
    /// number, the "this will be tight" warning below could never fire: the
    /// budget's own cap was derived by dividing available memory by 200, so
    /// multiplying it back by 200 could never exceed the memory it came from.
    /// Two agreeing wrong numbers looked exactly like a working check.
    static func bytesPerSplat(at shDegree: SHDegree) -> UInt64 {
        UInt64(Swift.max(1, TrainerResources.bytesPerSplat(
            shCoefficientCount: 1 + shDegree.restCoefficientCount
        )))
    }

    /// Sizes a budget from the device tier, the scan's own measured extent, its
    /// frame count and the memory this process may actually allocate.
    ///
    /// - Parameters:
    ///   - tier: from the compatibility probe, by way of the pre-pass pipeline
    ///     that `RootView` already sets it on. nil when this app run never got
    ///     a report, which is reported rather than papered over.
    static func plan(
        bundle: CaptureBundle,
        prePass: PrePassResult?,
        tier: DeviceTier?,
        memory: DeviceMemoryFacts = DeviceMemoryFacts.probe()
    ) -> ProcessingBudgetPlan {

        let extent = sceneExtent(bundle: bundle)
        let frameCount = bundle.frames.count

        var resolvedTier = tier
        let tierWasMeasured = tier != nil
        if resolvedTier == nil {
            // The same rule the pre-pass uses for the same situation: 1.5 GB is
            // roughly where a 300k-splat field plus its optimiser state and the
            // frame cache stop fitting.
            resolvedTier = memory.availableBytes >= 1_500_000_000 ? .full : .limited
        }
        let deviceTier = resolvedTier ?? .limited

        var budget = TrainingBudget.recommended(
            for: deviceTier,
            sceneExtentMeters: extent.meters,
            availableMemoryBytes: memory.availableBytes
        )

        // Never ask for more supervision views than there are photos. Only ever
        // downwards: every adjustment in this function is a `min`, so a plan can
        // come out smaller than `recommended` and never larger.
        budget.keyframeCount = Swift.min(budget.keyframeCount, Swift.max(frameCount, 1))

        // The pre-pass suggested a size for this scene too. Take the smaller of
        // the two, field by field: two honest estimates disagreeing is a reason
        // to be careful, not a reason to pick the bigger one.
        var usedPrePassSuggestion = false
        if let suggested = prePass?.suggestedBudget {
            usedPrePassSuggestion = true
            budget.splatCap = Swift.min(budget.splatCap, suggested.splatCap)
            budget.iterations = Swift.min(budget.iterations, suggested.iterations)
            budget.renderLongEdgePixels = Swift.min(
                budget.renderLongEdgePixels, suggested.renderLongEdgePixels
            )
            budget.keyframeCount = Swift.min(budget.keyframeCount, suggested.keyframeCount)
            budget.memoryCeilingBytes = Swift.min(
                budget.memoryCeilingBytes, suggested.memoryCeilingBytes
            )
            budget.shDegree = SHDegree(
                rawValue: Swift.min(budget.shDegree.rawValue, suggested.shDegree.rawValue)
            ) ?? budget.shDegree
        }

        // Whatever else was decided, the ceiling never goes above half of what
        // this process may still allocate.
        budget.memoryCeilingBytes = Swift.min(
            budget.memoryCeilingBytes, memory.availableBytes / 2
        )
        budget.target = .onDevice

        let estimatedBytes = UInt64(Swift.max(0, budget.splatCap))
            * bytesPerSplat(at: budget.shDegree)

        var lines: [String] = []
        lines.append(
            "Worked out from the \(frameCount) photos in this scan and the "
            + "\(ProcessingFormat.meters(extent.meters)) it covers, "
            + "\(extent.source)."
        )
        lines.append(
            "This phone has \(ProcessingFormat.bytes(memory.availableBytes)) of memory "
            + "free for this app right now"
            + (memory.availableIsEstimated
                ? ", which is an estimate rather than a measurement on this device."
                : ".")
        )
        lines.append(
            "Aiming for up to \(ProcessingFormat.count(budget.splatCap)) detail points "
            + "over \(ProcessingFormat.count(budget.iterations)) rounds, practising on "
            + "\(budget.keyframeCount) of your photos at \(budget.renderLongEdgePixels) "
            + "pixels across."
        )
        if !tierWasMeasured {
            lines.append(
                "This phone was not checked over at startup this time, so the size was "
                + "worked out from the memory it has free rather than from what it is."
            )
        }
        if usedPrePassSuggestion {
            lines.append(
                "The check-over suggested a size for this scan as well; the smaller of "
                + "the two was taken."
            )
        }
        lines.append(
            "Your phone may still make this smaller as it goes, if it gets warm or runs "
            + "short of memory. It will say so when it does."
        )

        let warning = sizeWarning(
            tier: deviceTier,
            extentMeters: extent.meters,
            frameCount: frameCount,
            estimatedSplatBytes: estimatedBytes,
            availableMemoryBytes: memory.availableBytes
        )

        let summary = "Budget for \(bundle.scanID): tier=\(deviceTier.rawValue) "
            + "measuredTier=\(tierWasMeasured) extent=\(extent.meters)m frames=\(frameCount) "
            + "cap=\(budget.splatCap) iters=\(budget.iterations) "
            + "render=\(budget.renderLongEdgePixels) keyframes=\(budget.keyframeCount) "
            + "availableBytes=\(memory.availableBytes)"
        ProcessingLog.budget.info("\(summary, privacy: .public)")

        return ProcessingBudgetPlan(
            budget: budget,
            tier: deviceTier,
            tierWasMeasured: tierWasMeasured,
            sceneExtentMeters: extent.meters,
            extentSource: extent.source,
            frameCount: frameCount,
            availableMemoryBytes: memory.availableBytes,
            memoryWasMeasured: !memory.availableIsEstimated,
            estimatedSplatBytes: estimatedBytes,
            sizeWarning: warning,
            lines: lines
        )
    }

    // MARK: Extent

    /// How big this scene actually is, measured rather than assumed.
    ///
    /// First choice is the LiDAR bounds the capture recorded. Second is the
    /// spread of the camera positions, which is a real measurement of the walk
    /// even when the point cloud's bounds were never written. Only if there are
    /// no poses at all does a stand-in appear, and it is labelled as one.
    static func sceneExtent(bundle: CaptureBundle) -> (meters: Float, source: String) {
        if let bounds = bundle.sceneBounds {
            let edge = bounds.longestEdgeMeters
            if edge.isFinite, edge > 0.25 {
                return (edge, "measured from the laser readings")
            }
        }

        var lo = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        var counted = 0
        for frame in bundle.frames {
            let centre = (frame.refinedPose ?? frame.rawPose).center.simd
            guard centre.x.isFinite, centre.y.isFinite, centre.z.isFinite else { continue }
            lo = simd_min(lo, centre)
            hi = simd_max(hi, centre)
            counted += 1
        }
        if counted > 1 {
            let size = hi - lo
            let edge = Swift.max(size.x, Swift.max(size.y, size.z))
            if edge.isFinite, edge > 0.25 {
                // A walk covers less ground than the room it is in, so this is
                // a floor rather than an equal substitute. Said plainly.
                return (edge, "measured from where you walked, because this scan has no "
                        + "recorded laser bounds")
            }
        }

        return (
            5,
            "taken as a room-sized 5 metres, because this scan recorded neither laser "
            + "bounds nor enough camera positions to measure it"
        )
    }

    // MARK: Warning

    private static func sizeWarning(
        tier: DeviceTier,
        extentMeters: Float,
        frameCount: Int,
        estimatedSplatBytes: UInt64,
        availableMemoryBytes: UInt64
    ) -> String? {
        let tight = estimatedSplatBytes > availableMemoryBytes / 2
        let big = extentMeters > 12 || frameCount > 2_500
        let weakPhone = tier == .limited && (extentMeters > 8 || frameCount > 1_200)

        guard tight || big || weakPhone else { return nil }

        var sentence = "This is a big scan for a phone to build on its own. "
        if tight {
            sentence += "The detail points alone want about "
                + "\(ProcessingFormat.bytes(estimatedSplatBytes)) and this phone has "
                + "\(ProcessingFormat.bytes(availableMemoryBytes)) free. "
        }
        sentence += "Your phone will try, and it will build a smaller model rather than "
            + "give up. If you would rather have the full-size one, a computer on your "
            + "Wi-Fi can build it instead."
        return sentence
    }
}

// MARK: - Formatting

/// Numbers as a person reads them. Deliberately separate from Viewer's
/// `ViewerFormat`, which owns the review screens' wording; this one only has
/// what the processing screen needs and adds a rounded count that reads as
/// "300 thousand" rather than "300000".
enum ProcessingFormat {

    static func bytes(_ count: UInt64) -> String {
        let clamped = count > UInt64(Int64.max) ? Int64.max : Int64(count)
        return bytes(clamped)
    }

    static func bytes(_ count: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        formatter.allowedUnits = [.useMB, .useGB]
        return formatter.string(fromByteCount: Swift.max(0, count))
    }

    static func count(_ value: Int) -> String {
        if value >= 1_000_000 {
            return String(format: "%.1f million", Double(value) / 1_000_000)
        }
        if value >= 10_000 {
            return "\(Int((Double(value) / 1_000).rounded())) thousand"
        }
        return "\(value)"
    }

    static func meters(_ value: Float) -> String {
        guard value.isFinite else { return "an unknown distance" }
        return value < 10
            ? String(format: "%.1f metres", value)
            : "\(Int(value.rounded())) metres"
    }

    static func date(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}
