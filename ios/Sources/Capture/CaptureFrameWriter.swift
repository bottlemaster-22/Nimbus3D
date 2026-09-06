//
//  CaptureFrameWriter.swift
//  Capture
//
//  Everything one keyframe puts on disk: the JPEG, the native depth sidecar,
//  the confidence sidecar, and one appended line of `sensor_data/frames.jsonl`.
//
//  THE JSONL IS THE CRASH SURVIVOR. docs/DATA_FORMAT.md section 3 makes this
//  explicit: `capture_bundle.json` is written once, at the end, and if the app
//  is killed before that happens the append-only log is the only record that
//  the capture ever existed. So the order here is deliberate - pixels, depth,
//  confidence, THEN the log line - and the log line is flushed to the file
//  descriptor as it is written. A log entry always describes files that are
//  already on disk, never files that were about to be.
//
//  Owned by, and only touched from, the capture pipeline's serial queue.
//

import Foundation

/// Writes one keyframe's files and its live log line.
///
/// `@unchecked Sendable` because it is handed to the capture pipeline's SERIAL
/// queue and touched from nowhere else. The queue is the synchronisation; there
/// is no lock inside, and adding a second consumer would be a bug this
/// annotation cannot catch for you.
final class CaptureFrameWriter: @unchecked Sendable {

    private let folder: CaptureScanFolder
    private let fileManager: FileManager
    private let encoder: JSONEncoder

    private var logHandle: FileHandle?

    /// Bytes this writer has put on disk, for the HUD's storage readout.
    private(set) var bytesWritten: Int64 = 0

    /// Frame stamps already used, so two frames landing in the same
    /// millisecond get `_512b` / `_512c` rather than silently overwriting each
    /// other. docs/DATA_FORMAT.md section 1.
    private var stampUseCount: [String: Int] = [:]

    init(folder: CaptureScanFolder, fileManager: FileManager = .default) {
        self.folder = folder
        self.fileManager = fileManager
        // Not pretty-printed: `frames.jsonl` is one object per line by
        // definition, and a pretty-printed object spans lines.
        self.encoder = ContractsJSON.encoder(prettyPrinted: false)
    }

    // MARK: - Lifecycle

    /// Creates (or truncates) `sensor_data/frames.jsonl` and holds it open.
    func open() throws {
        let url = folder.framesLogURL
        if !fileManager.fileExists(atPath: url.path) {
            fileManager.createFile(atPath: url.path, contents: nil)
        }
        logHandle = try FileHandle(forWritingTo: url)
        try logHandle?.truncate(atOffset: 0)
    }

    func close() {
        try? logHandle?.synchronize()
        try? logHandle?.close()
        logHandle = nil
    }

    // MARK: - Stamps

    /// A stamp unique within this session, derived from the frame's own
    /// capture instant.
    func uniqueStamp(forWallClock wallClock: Date) -> String {
        let base = CaptureScanFolder.frameStamp(wallClock: wallClock)
        let used = stampUseCount[base, default: 0]
        stampUseCount[base] = used + 1
        guard used > 0 else { return base }
        return CaptureScanFolder.frameStamp(
            wallClock: wallClock,
            collisionIndex: used
        )
    }

    // MARK: - Writing

    /// What a successful write produced, so the caller can build the
    /// `CaptureFrame` record without re-deriving any of it.
    struct Written {
        var imagePath: String
        var depthPath: String?
        var confidencePath: String?
    }

    /// Writes the pixels and the sensor sidecars for one keyframe.
    ///
    /// - Parameter depth: nil when ARKit delivered no depth for this frame. The
    ///   record is then written with `depthPath == nil`, which a reader can
    ///   tell apart from "the laser saw nothing" - two completely different
    ///   facts, and collapsing them would be a lie the whole pre-pass would
    ///   then build on.
    /// - Throws: whatever `Data.write` throws. A throwing write must abort the
    ///   frame, not be swallowed: a log line pointing at a file that is not
    ///   there is worse than a missing frame.
    func writeFiles(
        stamp: String,
        image: CapturePixelBuffer,
        depth: CaptureDepthFrame?
    ) throws -> Written {
        guard let jpeg = image.jpegData() else {
            throw NimbusError.captureFailed("A camera frame could not be encoded.")
        }

        // Not .atomic, and deliberately so. An atomic write is a write to
        // a temp file followed by a rename, which buys crash safety this
        // path already has: frames.jsonl is appended only AFTER this
        // function returns, so a torn file is never named by any log line
        // and never reaches the bundle. The cost was two extra directory
        // and inode dirtyings per file and three renames per keyframe, at
        // five keyframes a second, against a phone already over its daily
        // write budget.
        let imageURL = folder.imagesDirectory.appendingPathComponent("\(stamp).jpg")
        try jpeg.write(to: imageURL)
        bytesWritten += Int64(jpeg.count)

        var depthPath: String?
        var confidencePath: String?

        if let depth {
            let depthBytes = depth.depthSidecarBytes
            let depthURL = folder.depthDirectory
                .appendingPathComponent("\(stamp).depth16")
            try depthBytes.write(to: depthURL)
            bytesWritten += Int64(depthBytes.count)
            depthPath = CaptureScanFolder.depthRelativePath(stamp: stamp)

            let confidenceBytes = depth.confidenceSidecarBytes
            let confidenceURL = folder.confidenceDirectory
                .appendingPathComponent("\(stamp).conf8")
            try confidenceBytes.write(to: confidenceURL)
            bytesWritten += Int64(confidenceBytes.count)
            confidencePath = CaptureScanFolder.confidenceRelativePath(stamp: stamp)
        }

        return Written(
            imagePath: CaptureScanFolder.imageRelativePath(stamp: stamp),
            depthPath: depthPath,
            confidencePath: confidencePath
        )
    }

    /// Appends one `CaptureFrame` to `sensor_data/frames.jsonl`.
    ///
    /// A failure here is logged and swallowed on purpose, and it is the one
    /// place in this file where that is right: the frame's pixels and depth are
    /// already safely on disk, and refusing to continue the capture because a
    /// log line would not append would throw away a scan over its index.
    func appendLogLine(for frame: CaptureFrame) {
        guard let logHandle else { return }
        do {
            var line = try encoder.encode(frame)
            line.append(0x0A)  // '\n', never "\r\n" - DATA_FORMAT section 5.
            try logHandle.write(contentsOf: line)
            bytesWritten += Int64(line.count)
        } catch {
            let message = "Could not append to frames.jsonl: "
                    + "\(error.localizedDescription). The "
                    + "frame's files are on disk; only the live index line is "
                    + "missing."
            CaptureLog.writer.error("\(message, privacy: .public)")
        }
    }
}
