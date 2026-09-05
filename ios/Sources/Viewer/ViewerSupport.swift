//
//  ViewerSupport.swift
//  Viewer
//
//  SMALL SHARED PIECES THE REST OF THE VIEWER STANDS ON.
//
//  Nothing in here touches Metal or SwiftUI, so it is safe to call from any
//  actor and cheap to reason about: paths, errors, logging, the Morton /
//  direction-bin encodings that the honesty mask shares with
//  `SplatRenderShaders.metal`, and a couple of numeric helpers.
//
//  Conventions are Core's, unchanged. World is right-handed, Y up, metres.
//  A `Pose` is world -> camera with +X right, +Y DOWN, +Z FORWARD. Nothing in
//  this module negates an axis; where a camera basis is built it is built from
//  those rules directly (see `ViewerPoseMath.lookAt`).
//

import Foundation
import os
import simd

// MARK: - Logging

enum ViewerLog {
    static let renderer = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "viewer.renderer"
    )
    static let library = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "viewer.library"
    )
    static let review = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "viewer.review"
    )
}

// MARK: - Errors

/// What can go wrong inside the Viewer. Every message is a plain sentence,
/// because every one of them can end up in front of a non-technical user in a
/// "couldn't open this scan" box.
///
/// `NimbusError` is what crosses a module boundary; this is the richer local
/// type, exactly as `ExportError` and `BoosterError` are for their modules.
enum ViewerError: LocalizedError {
    case metalUnavailable(String)
    case shaderLibraryMissing(String)
    case pipelineFailed(String)
    case modelHasNoSplatFile(ScanID)
    case fileMissing(String)
    case malformedFile(String)
    case exporterUnavailable
    case emptyModel

    var errorDescription: String? {
        switch self {
        case .metalUnavailable(let detail):
            return "This device's graphics system could not start the preview. \(detail)"
        case .shaderLibraryMissing(let detail):
            return "The preview's graphics code is missing from this build. \(detail)"
        case .pipelineFailed(let detail):
            return "The preview could not be set up. \(detail)"
        case .modelHasNoSplatFile(let scanID):
            return "The scan \(scanID) has a result file recorded but no .ply or .spz next to it."
        case .fileMissing(let path):
            return "A file this scan needs is missing: \(path)"
        case .malformedFile(let detail):
            return "A file this scan needs could not be read: \(detail)"
        case .exporterUnavailable:
            return "The part of the app that reads splat files is not available in this build."
        case .emptyModel:
            return "This scan's result has no splats in it."
        }
    }

    /// The form that crosses a module boundary. `NimbusError` has no
    /// "rendering failed" case and inventing one would mean editing Core, so
    /// each case maps to the existing case that is actually true of it: a GPU
    /// this app cannot draw on IS a device-compatibility fact, and a missing
    /// shader library IS a module that did not make it into the build.
    var asNimbusError: NimbusError {
        switch self {
        case .metalUnavailable:
            return .deviceIncompatible(errorDescription ?? "No usable GPU.")
        case .shaderLibraryMissing, .pipelineFailed:
            return .moduleNotInstalled(module: "Viewer")
        case .modelHasNoSplatFile(let scanID):
            return .scanNotFound(scanID)
        case .fileMissing(let path):
            return .malformedData(path)
        case .malformedFile(let detail):
            return .malformedData(detail)
        case .exporterUnavailable:
            return .moduleNotInstalled(module: "Export")
        case .emptyModel:
            return .malformedData("the result file has no splats in it")
        }
    }
}

// MARK: - Numeric helpers

enum ViewerMath {
    @inline(__always)
    static func clamp(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
        Swift.min(Swift.max(v, lo), hi)
    }

    @inline(__always)
    static func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Double {
        Swift.min(Swift.max(v, lo), hi)
    }

    /// `normalize` that returns `fallback` instead of NaN for a zero vector.
    @inline(__always)
    static func safeNormalize(
        _ v: SIMD3<Float>,
        fallback: SIMD3<Float> = SIMD3<Float>(0, 0, 1)
    ) -> SIMD3<Float> {
        let l = simd_length(v)
        guard l > 1e-9, l.isFinite else { return fallback }
        return v / l
    }

    /// Next power of two at or above `n`, minimum 1. The GPU bitonic sort
    /// needs a power-of-two array length.
    static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        return 1 << (Int.bitWidth - (n - 1).leadingZeroBitCount)
    }

    /// Angle between two directions, degrees. Both are normalised first.
    static func angleDegrees(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let na = safeNormalize(a)
        let nb = safeNormalize(b)
        return acos(clamp(simd_dot(na, nb), -1, 1)) * 180 / .pi
    }
}

// MARK: - Camera basis

/// Builds this app's camera convention from an eye and a target.
///
/// A free function rather than an extension on `Pose`, because `Pose` is
/// Core's type and Core's contract surface; the Viewer does not add members to
/// it. The construction below is the +X right / +Y DOWN / +Z forward rule
/// applied literally - there is no sign flip hidden anywhere in it.
enum ViewerPoseMath {

    /// World-space camera axes for a camera at `eye` looking at `target`.
    ///
    /// - Returns: `(right, down, forward)`, orthonormal, right-handed in the
    ///   sense `right x down == forward` (which is what "+X right, +Y down,
    ///   +Z forward" means for an image-coordinates camera).
    static func basis(
        eye: SIMD3<Float>,
        target: SIMD3<Float>,
        worldUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    ) -> (right: SIMD3<Float>, down: SIMD3<Float>, forward: SIMD3<Float>) {
        let forward = ViewerMath.safeNormalize(target - eye)
        var right = simd_cross(forward, worldUp)
        if simd_length(right) < 1e-5 {
            // Looking straight up or straight down: any horizontal right
            // vector will do, and picking one deterministically beats NaN.
            right = simd_cross(forward, SIMD3<Float>(0, 0, 1))
            if simd_length(right) < 1e-5 {
                right = SIMD3<Float>(1, 0, 0)
            }
        }
        right = ViewerMath.safeNormalize(right, fallback: SIMD3<Float>(1, 0, 0))
        let down = simd_cross(forward, right)
        return (right, down, forward)
    }

    /// A world -> camera `Pose` for a camera at `eye` looking at `target`.
    static func lookAt(
        eye: SIMD3<Float>,
        target: SIMD3<Float>,
        worldUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    ) -> Pose {
        let b = basis(eye: eye, target: target, worldUp: worldUp)
        // Rows of R are the camera axes in world coordinates; simd is
        // column-major, so the columns below are the transposed rows.
        let r = simd_float3x3(
            SIMD3<Float>(b.right.x, b.down.x, b.forward.x),
            SIMD3<Float>(b.right.y, b.down.y, b.forward.y),
            SIMD3<Float>(b.right.z, b.down.z, b.forward.z)
        )
        let q = simd_quatf(r).normalized
        let t = -(q.act(eye))
        return Pose(rotation: Quaternion(q), translation: Vector3(t))
    }

    /// The camera's world-space "up" (which is minus its +Y axis, because +Y
    /// is down). Used by the fly-through smoother to keep the horizon level.
    static func worldUp(of pose: Pose) -> SIMD3<Float> {
        -pose.rotation.simd.inverse.act(SIMD3<Float>(0, 1, 0))
    }
}

// MARK: - Morton codes

/// The 3 x 21-bit Morton encoding shared with `SplatRenderShaders.metal`
/// (`viewer_morton_part` / `viewer_morton_key`) and used as the key of both
/// `model/observed_directions.bin` and `prepass/trust_bias.bin`.
///
/// `PrePassMorton` and `SmartMorton` exist for the same job in their own
/// modules; this one is the Viewer's because a duplicate top-level type name
/// anywhere in this single-target project is a hard link error, so the three
/// cannot be merged without one module reaching into another's directory.
enum ViewerMorton {

    /// Spreads 21 bits so they occupy every third bit.
    @inline(__always)
    static func encodePart(_ value: UInt32) -> UInt64 {
        var v = UInt64(value & 0x1F_FFFF)
        v = (v | (v << 32)) & 0x001F_0000_0000_FFFF
        v = (v | (v << 16)) & 0x001F_0000_FF00_00FF
        v = (v | (v << 8)) & 0x100F_00F0_0F00_F00F
        v = (v | (v << 4)) & 0x10C3_0C30_C30C_30C3
        v = (v | (v << 2)) & 0x1249_2492_4924_9249
        return v
    }

    /// The exact inverse of `encodePart`.
    @inline(__always)
    static func decodePart(_ key: UInt64) -> UInt32 {
        var v = key & 0x1249_2492_4924_9249
        v = (v ^ (v >> 2)) & 0x10C3_0C30_C30C_30C3
        v = (v ^ (v >> 4)) & 0x100F_00F0_0F00_F00F
        v = (v ^ (v >> 8)) & 0x001F_0000_FF00_00FF
        v = (v ^ (v >> 16)) & 0x001F_0000_0000_FFFF
        v = (v ^ (v >> 32)) & 0x0000_0000_001F_FFFF
        return UInt32(truncatingIfNeeded: v)
    }

    @inline(__always)
    static func key(_ cell: SIMD3<UInt32>) -> UInt64 {
        encodePart(cell.x) | (encodePart(cell.y) << 1) | (encodePart(cell.z) << 2)
    }

    @inline(__always)
    static func cell(_ key: UInt64) -> SIMD3<UInt32> {
        SIMD3<UInt32>(
            decodePart(key),
            decodePart(key >> 1),
            decodePart(key >> 2)
        )
    }

    /// Largest coordinate a 21-bit Morton part can hold.
    static let maxCoordinate: UInt32 = 0x1F_FFFF
}

// MARK: - Binary reading

/// Bounds-checked little-endian reads out of a `Data`, so a truncated or
/// corrupt sidecar produces a named error instead of a crash.
///
/// Every binary file in `docs/DATA_FORMAT.md` is little-endian and both ends
/// of this project are little-endian machines, so these are plain loads with
/// no byte swapping - the type names just make the assumption visible.
struct ViewerByteReader {
    private let data: Data
    private(set) var offset: Int

    init(_ data: Data) {
        self.data = data
        self.offset = 0
    }

    var remaining: Int { data.count - offset }

    mutating func skip(_ count: Int) throws {
        guard count >= 0, remaining >= count else {
            throw ViewerError.malformedFile("ran off the end after \(offset) bytes")
        }
        offset += count
    }

    mutating func readBytes(_ count: Int) throws -> Data {
        guard count >= 0, remaining >= count else {
            throw ViewerError.malformedFile(
                "wanted \(count) bytes at offset \(offset), only \(remaining) left"
            )
        }
        let start = data.startIndex + offset
        let slice = data[start..<(start + count)]
        offset += count
        return Data(slice)
    }

    mutating func readUInt32() throws -> UInt32 {
        let bytes = try readBytes(4)
        return bytes.withUnsafeBytes { raw in
            UInt32(littleEndian: raw.loadUnaligned(as: UInt32.self))
        }
    }

    mutating func readUInt64() throws -> UInt64 {
        let bytes = try readBytes(8)
        return bytes.withUnsafeBytes { raw in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
    }

    mutating func readUInt16() throws -> UInt16 {
        let bytes = try readBytes(2)
        return bytes.withUnsafeBytes { raw in
            UInt16(littleEndian: raw.loadUnaligned(as: UInt16.self))
        }
    }

    mutating func readFloat() throws -> Float {
        Float(bitPattern: try readUInt32())
    }
}

extension Data {
    /// Appends `value` little-endian. Used by the honesty-mask writer.
    mutating func appendLittle<T: FixedWidthInteger>(_ value: T) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }

    mutating func appendLittle(_ value: Float) {
        appendLittle(value.bitPattern)
    }
}

// MARK: - Scan paths

/// Where things live inside one scan folder.
///
/// Every path recorded inside a scan's JSON is relative to the scan root, so
/// this is the one place the Viewer turns a relative path into an absolute
/// URL. The folder names come from `BrandConfig.Folder`, which does not change
/// when the product is renamed.
struct ViewerScanPaths: Sendable {
    let scanID: ScanID
    let root: URL

    init(scanID: ScanID, root: URL) {
        self.scanID = scanID
        self.root = root
    }

    init(ref: CaptureBundleRef) {
        self.init(scanID: ref.scanID, root: ref.rootURL)
    }

    var ref: CaptureBundleRef { CaptureBundleRef(scanID: scanID, rootURL: root) }

    func url(_ relativePath: String) -> URL {
        relativePath
            .split(separator: "/")
            .reduce(root) { $0.appendingPathComponent(String($1)) }
    }

    var captureBundleJSON: URL { root.appendingPathComponent("capture_bundle.json") }

    var prePassResultJSON: URL {
        url("\(BrandConfig.Folder.prePass)/prepass_result.json")
    }

    var modelJSON: URL {
        url("\(BrandConfig.Folder.model)/model.json")
    }

    var modelDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.model, isDirectory: true)
    }

    var cacheDirectory: URL {
        root.appendingPathComponent(BrandConfig.Folder.cache, isDirectory: true)
    }

    /// Default location of the honesty mask's backing store, per
    /// `docs/DATA_FORMAT.md` section 8.
    var observedDirectionsBin: URL {
        modelDirectory.appendingPathComponent("observed_directions.bin")
    }

    /// Every scan folder under `Documents/<brand>/Scans`, newest name last.
    /// Folders are the format, so listing them is the correct way to find
    /// scans - there is no index file to fall out of date.
    static func allScanDirectories(
        fileManager: FileManager = .default
    ) throws -> [URL] {
        let scansRoot = try BrandConfig.scansDirectory(fileManager: fileManager)
        let contents = try fileManager.contentsOfDirectory(
            at: scansRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        return contents
            .filter { url in
                // The Booster's Incoming/Completed/Failed folders are the PC
                // side of the format and never appear on the phone, but a
                // hand-copied folder could, so they are skipped by name.
                let name = url.lastPathComponent
                if name == BrandConfig.Folder.Booster.incoming
                    || name == BrandConfig.Folder.Booster.completed
                    || name == BrandConfig.Folder.Booster.failed {
                    return false
                }
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
                return values?.isDirectory == true
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}

// MARK: - Formatting

/// Number and byte formatting for the review screens. One place, so "1.2 GB"
/// and "480 k splats" read the same everywhere.
enum ViewerFormat {

    static func bytes(_ count: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        return formatter.string(fromByteCount: Swift.max(0, count))
    }

    static func splatCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM splats", Double(count) / 1_000_000)
        }
        if count >= 1_000 {
            return String(format: "%.0fk splats", Double(count) / 1_000)
        }
        return "\(count) splats"
    }

    static func meters(_ value: Float, decimals: Int = 2) -> String {
        String(format: "%.\(decimals)f m", value)
    }

    static func centimeters(_ value: Float) -> String {
        value < 10
            ? String(format: "%.1f cm", value)
            : String(format: "%.0f cm", value)
    }

    static func percent(_ fraction: Float) -> String {
        "\(Int((ViewerMath.clamp(fraction, 0, 1) * 100).rounded()))%"
    }

    static func degrees(_ value: Float) -> String {
        String(format: "%.0f\u{00B0}", value)
    }

    static func date(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    static func duration(_ seconds: Double) -> String {
        let total = Int(Swift.max(0, seconds).rounded())
        let minutes = total / 60
        let remainder = total % 60
        if minutes == 0 { return "\(remainder)s" }
        return "\(minutes)m \(remainder)s"
    }
}
