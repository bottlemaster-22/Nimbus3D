//
//  CaptureBundleWriter.swift — on-disk layout for a CaptureBundle.
//  Owned by the Capture agent.
//
//  This owns the bundle DIRECTORY layout and manifest. Per-file encoding (HEIC/JPEG,
//  packed Float32 depth, UInt8 confidence) is delegated to FrameWriter so there is one
//  encoder in the module, not two.
//
//  Directory layout produced under `rootDirectory`:
//    rootDirectory/
//      manifest.json                 encoded CaptureBundle (source of truth)
//      frames/
//        frame_0000.heic             color image (JPEG fallback if HEIC encode fails)
//        depth_0000_256x192.f32      row-major little-endian Float32 depth, meters (LiDAR only)
//        confidence_0000_256x192.u8  row-major UInt8 ARConfidenceLevel 0/1/2 (LiDAR only)
//      hdri/
//        bracket_0000.heic           environment frames for HDRI assembly
//
//  Depth/confidence dimensions are encoded in the file name (see FrameWriter), so no
//  separate sidecar is needed to reshape the raw blobs.
//
//  NOTE ON PORTABILITY: iOS app-container paths are NOT stable across launches, so the
//  URLs stored in manifest.json are only valid for the session that wrote them. Reload a
//  bundle with `CaptureBundleReader.load(from:)`, which rebuilds every file URL under the
//  directory you hand it (using the fixed frames/ and hdri/ layout above).
//
//  NOTE (integrator): this reader is named `CaptureBundleReader`, not
//  `CaptureBundleStore`, because the Pipeline module already exposes a public
//  `CaptureBundleStore`. Every file compiles into one Swift module, so the two
//  names would collide. The Process/Library screens use Pipeline's store; this
//  Capture-owned reader is the low-level per-bundle loader.
//

import Foundation
import CoreVideo

final class CaptureBundleWriter {

    let rootDirectory: URL
    let framesDirectory: URL
    let hdriDirectory: URL

    private let frameWriter = FrameWriter()

    init(rootDirectory: URL) throws {
        self.rootDirectory = rootDirectory
        self.framesDirectory = rootDirectory.appendingPathComponent("frames", isDirectory: true)
        self.hdriDirectory = rootDirectory.appendingPathComponent("hdri", isDirectory: true)
        try FileManager.default.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Color images

    func writeColorImage(_ pixelBuffer: CVPixelBuffer, index: Int) throws -> URL {
        let base = framesDirectory.appendingPathComponent(String(format: "frame_%04d", index))
        return try frameWriter.writeColor(pixelBuffer, toBaseURL: base)
    }

    func writeBracketImage(_ pixelBuffer: CVPixelBuffer, index: Int) throws -> URL {
        try FileManager.default.createDirectory(at: hdriDirectory, withIntermediateDirectories: true)
        let base = hdriDirectory.appendingPathComponent(String(format: "bracket_%04d", index))
        return try frameWriter.writeColor(pixelBuffer, toBaseURL: base)
    }

    // MARK: - Depth

    func writeDepth(_ depthMap: CVPixelBuffer, index: Int) throws -> URL {
        let base = framesDirectory.appendingPathComponent(String(format: "depth_%04d", index))
        return try frameWriter.writeDepthFloat32(depthMap, toBaseURL: base)
    }

    func writeConfidence(_ confidenceMap: CVPixelBuffer, index: Int) throws -> URL {
        let base = framesDirectory.appendingPathComponent(String(format: "confidence_%04d", index))
        return try frameWriter.writeConfidenceUInt8(confidenceMap, toBaseURL: base)
    }

    // MARK: - Manifest

    func writeManifest(_ bundle: CaptureBundle) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(bundle)
        try data.write(to: rootDirectory.appendingPathComponent("manifest.json"))
    }
}

// MARK: - Reloading a bundle in a later session

/// Loads a previously written CaptureBundle and rebases every file URL under the given
/// directory (app-container paths are not stable across launches, so stored URLs are stale).
/// File names (including the WxH suffix on depth/confidence) are preserved via lastPathComponent.
enum CaptureBundleReader {

    static func load(from directory: URL) throws -> CaptureBundle {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let raw = try decoder.decode(CaptureBundle.self, from: data)
        return rebase(raw, to: directory)
    }

    /// Rewrites all URLs in `bundle` to point under `directory` using the fixed layout.
    static func rebase(_ bundle: CaptureBundle, to directory: URL) -> CaptureBundle {
        let frames = directory.appendingPathComponent("frames", isDirectory: true)
        let hdri = directory.appendingPathComponent("hdri", isDirectory: true)

        let rebasedFrames = bundle.frames.map { frame -> CapturedFrame in
            var f = frame
            f.imageURL = frames.appendingPathComponent(frame.imageURL.lastPathComponent)
            f.depthMapURL = frame.depthMapURL.map { frames.appendingPathComponent($0.lastPathComponent) }
            f.depthConfidenceURL = frame.depthConfidenceURL.map { frames.appendingPathComponent($0.lastPathComponent) }
            return f
        }

        let rebasedBrackets = bundle.hdriBrackets.map { bracket -> ExposureBracketFrame in
            var b = bracket
            b.imageURL = hdri.appendingPathComponent(bracket.imageURL.lastPathComponent)
            return b
        }

        var result = bundle
        result.rootDirectory = directory
        result.frames = rebasedFrames
        result.hdriBrackets = rebasedBrackets
        return result
    }
}
