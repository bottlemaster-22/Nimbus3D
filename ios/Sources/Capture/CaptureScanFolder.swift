//
//  CaptureScanFolder.swift
//  Capture
//
//  The scan's home on disk: its id, its directory tree, its per-frame
//  filename stamps, and the free-space check that decides whether starting is
//  even honest.
//
//  Every name in here is fixed by docs/DATA_FORMAT.md section 1 and comes
//  from `BrandConfig.Folder`, never from a literal.
//

import Foundation

/// Creates and describes one scan's folder tree.
///
/// Not an actor: it is created once on the main actor before capture starts,
/// then read (never mutated) from the frame pipeline's queue. Every property is
/// a `let`.
public struct CaptureScanFolder: Sendable {

    public let scanID: ScanID
    public let root: URL

    /// Wall-clock instant capture started, paired with the ARKit frame
    /// timestamp observed at the same moment. Together they convert any
    /// `ARFrame.timestamp` (mach-continuous seconds since boot) into a wall
    /// clock time, which is what the `frame_YYYYMMDD_HHMMSS_mmm` stamp needs.
    public let startedAt: Date

    // MARK: - Sub-directories

    public var imagesDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.images, isDirectory: true)
    }
    public var sensorDataDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.sensorData, isDirectory: true)
    }
    public var depthDirectory: URL {
        sensorDataDirectory.appendingPathComponent("depth", isDirectory: true)
    }
    public var confidenceDirectory: URL {
        sensorDataDirectory.appendingPathComponent("confidence", isDirectory: true)
    }
    public var framesLogURL: URL {
        sensorDataDirectory.appendingPathComponent("frames.jsonl")
    }
    /// `<scan>/sparse/0`, the COLMAP model directory.
    ///
    /// Built from `BrandConfig.Folder.sparseModel` rather than from
    /// `Folder.sparse` plus a hand-written `"0"`. The brand block exists so
    /// there is ONE place a folder name is written down, and two hand-written
    /// zeroes in this file were where that stopped being true.
    ///
    /// Appended one component at a time on purpose: the constant contains a
    /// separator, and this makes no assumption about how
    /// `appendingPathComponent` treats one.
    public var sparseModelDirectory: URL {
        var url = root
        for component in BrandConfig.Folder.sparseModel.split(separator: "/") {
            url = url.appendingPathComponent(String(component), isDirectory: true)
        }
        return url
    }
    public var anchorsDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.anchors, isDirectory: true)
    }
    public var meshDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.mesh, isDirectory: true)
    }
    public var cacheDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.cache, isDirectory: true)
    }
    public var bundleURL: URL {
        root.appendingPathComponent("capture_bundle.json")
    }

    // MARK: - Relative paths (what goes inside the JSON)

    /// docs/DATA_FORMAT.md: every path recorded in a scan is POSIX and
    /// relative to the scan root, so the folder can be zipped and opened
    /// anywhere.
    public static func imageRelativePath(stamp: String) -> String {
        "\(BrandConfig.Folder.images)/\(stamp).jpg"
    }
    public static func depthRelativePath(stamp: String) -> String {
        "\(BrandConfig.Folder.sensorData)/depth/\(stamp).depth16"
    }
    public static func confidenceRelativePath(stamp: String) -> String {
        "\(BrandConfig.Folder.sensorData)/confidence/\(stamp).conf8"
    }
    public static let pointCloudRelativePath =
        "\(BrandConfig.Folder.sparseModel)/points3D.txt"

    // MARK: - Creation

    /// Makes a fresh scan folder with every sub-directory the writers expect.
    ///
    /// - Parameter now: injectable clock, so the id is testable.
    public static func create(
        now: Date = Date(),
        fileManager: FileManager = .default
    ) throws -> CaptureScanFolder {
        let scansRoot = try BrandConfig.scansDirectory(fileManager: fileManager)
        let base = scanIDString(for: now)

        // `scan_YYYYMMDD_HHMMSS` is unique per second. Two captures started
        // inside the same second is absurd but cheap to survive, and silently
        // writing into an existing scan's folder would not be.
        var candidate = base
        var suffix = 2
        while fileManager.fileExists(
            atPath: scansRoot.appendingPathComponent(candidate).path
        ) {
            candidate = "\(base)_\(suffix)"
            suffix += 1
        }

        let root = scansRoot.appendingPathComponent(candidate, isDirectory: true)
        let folder = CaptureScanFolder(scanID: candidate, root: root, startedAt: now)

        for directory in [
            folder.root,
            folder.imagesDirectory,
            folder.sensorDataDirectory,
            folder.depthDirectory,
            folder.confidenceDirectory,
            folder.sparseModelDirectory,
            folder.anchorsDirectory,
            folder.meshDirectory,
            folder.cacheDirectory,
        ] {
            try fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
        }

        // Scans are the user's data and can be large; they must not be backed
        // up to iCloud behind the user's back. Best-effort: a failure here is
        // not a reason to refuse to scan.
        var mutableRoot = folder.root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutableRoot.setResourceValues(values)

        return folder
    }

    public func deleteEverything(fileManager: FileManager = .default) {
        do {
            try fileManager.removeItem(at: root)
        } catch {
            let message = "Could not delete cancelled scan \(scanID): "
                    + "\(error.localizedDescription)"
            CaptureLog.writer.error("\(message, privacy: .public)")
        }
    }

    // MARK: - Identifiers and stamps

    /// `scan_YYYYMMDD_HHMMSS`, local device time. docs/DATA_FORMAT.md 1.
    public static func scanIDString(for date: Date) -> String {
        "scan_" + Self.stampFormatter.string(from: date)
    }

    /// `frame_YYYYMMDD_HHMMSS_mmm`, local device time, derived from the
    /// frame's own capture timestamp rather than from when it hit the disk.
    ///
    /// - Parameter wallClock: the frame's capture instant in wall time.
    /// - Parameter collisionIndex: 0 for the first frame in a millisecond,
    ///   1 for the second (`_512b`), 2 for the third (`_512c`). Two frames can
    ///   land in the same millisecond at 60 fps plus a bracket; the
    ///   authoritative ordering is always `CaptureFrame.index`, never the name.
    public static func frameStamp(
        wallClock: Date,
        collisionIndex: Int = 0
    ) -> String {
        let seconds = Self.stampFormatter.string(from: wallClock)
        let millis = Int(
            (wallClock.timeIntervalSince1970
                - wallClock.timeIntervalSince1970.rounded(.down)) * 1000
        )
        let clamped = Swift.max(0, Swift.min(999, millis))
        var stamp = String(format: "frame_%@_%03d", seconds, clamped)
        if collisionIndex > 0 {
            // 1 -> "b", 2 -> "c". Past "z" (26 frames in one millisecond,
            // which cannot happen) it wraps to a digit rather than crashing.
            let scalarValue = UnicodeScalar(97 + (collisionIndex % 26))
            let letter = scalarValue.map { String(Character($0)) } ?? "\(collisionIndex)"
            stamp += letter
        }
        return stamp
    }

    /// Converts an `ARFrame.timestamp` (seconds on the device's
    /// mach-continuous clock) to wall time, using the pair captured when the
    /// session started.
    public func wallClock(
        forFrameTimestamp timestamp: TimeInterval,
        sessionStartTimestamp: TimeInterval
    ) -> Date {
        startedAt.addingTimeInterval(timestamp - sessionStartTimestamp)
    }

    private static let stampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()

    // MARK: - Disk

    /// Bytes actually available for this app to write, or `nil` if the system
    /// declines to say.
    public static func availableDiskBytes(fileManager: FileManager = .default) -> Int64? {
        guard
            let documents = try? fileManager.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            ),
            let values = try? documents.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]
            ),
            let capacity = values.volumeAvailableCapacityForImportantUsage
        else { return nil }
        return capacity
    }

    /// True when there is enough room left that continuing is honest.
    public static func hasRoomToContinue(fileManager: FileManager = .default) -> Bool {
        guard let available = availableDiskBytes(fileManager: fileManager) else {
            // The system would not answer. Refusing to scan on that basis
            // would be worse than trying and stopping cleanly if it fails.
            return true
        }
        return available > CaptureTuning.minFreeDiskBytes
    }
}
