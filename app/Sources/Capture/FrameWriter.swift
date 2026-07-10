//
//  FrameWriter.swift
//  Nimbus3D (Capture module)
//
//  Disk serialization for capture artifacts: HEIC/JPEG color frames,
//  raw Float32 LiDAR depth maps, and UInt8 depth-confidence maps.
//
//  Depth and confidence files carry no header. Because CapturedFrame has no
//  dimension fields for depth, dimensions are encoded in the file name:
//    depth_00042_256x192.f32   (little-endian Float32, row-major, meters)
//    confidence_00042_256x192.u8 (ARConfidenceLevel raw values 0/1/2, row-major)
//

import CoreImage
import CoreVideo
import Foundation
import ImageIO

final class FrameWriter {

    /// CIContext is thread-safe; a single shared instance avoids per-frame setup cost.
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private static let qualityOption =
        CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)

    // MARK: Color

    /// Encodes the camera pixel buffer to HEIC when the hardware encoder is
    /// available, otherwise JPEG. Returns the URL actually written
    /// (`base` + ".heic" or ".jpg").
    ///
    /// The image is written in raw sensor orientation (landscape). The pose and
    /// intrinsics recorded alongside it refer to exactly this pixel layout, so
    /// downstream consumers must NOT rotate the image before training.
    func writeColor(_ pixelBuffer: CVPixelBuffer,
                    toBaseURL base: URL,
                    compressionQuality: CGFloat = 0.85) throws -> URL {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let options: [CIImageRepresentationOption: Any] = [Self.qualityOption: compressionQuality]

        if let heic = context.heifRepresentation(of: image,
                                                 format: .RGBA8,
                                                 colorSpace: colorSpace,
                                                 options: options) {
            let url = base.appendingPathExtension("heic")
            try heic.write(to: url, options: .atomic)
            return url
        }

        // Some GPU/simulator configurations have no HEIC encoder; fall back to JPEG.
        guard let jpeg = context.jpegRepresentation(of: image,
                                                    colorSpace: colorSpace,
                                                    options: options) else {
            throw NimbusError.captureFailed("Could not encode the camera image.")
        }
        let url = base.appendingPathExtension("jpg")
        try jpeg.write(to: url, options: .atomic)
        return url
    }

    // MARK: Depth

    /// Writes a kCVPixelFormatType_DepthFloat32 map as tightly packed,
    /// row-major, little-endian Float32 (meters). Returns the written URL,
    /// named `<base>_<width>x<height>.f32`.
    func writeDepthFloat32(_ depthMap: CVPixelBuffer, toBaseURL base: URL) throws -> URL {
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
            throw NimbusError.captureFailed("Unexpected depth pixel format.")
        }
        let (data, width, height) = try Self.packedBytes(of: depthMap, bytesPerPixel: 4)
        let url = Self.dimensionedURL(base: base, width: width, height: height, ext: "f32")
        try data.write(to: url, options: .atomic)
        return url
    }

    /// Writes a kCVPixelFormatType_OneComponent8 ARKit depth-confidence map
    /// (ARConfidenceLevel raw values: 0 low, 1 medium, 2 high), one byte per
    /// pixel, row-major. Returns the written URL, named `<base>_<width>x<height>.u8`.
    func writeConfidenceUInt8(_ confidenceMap: CVPixelBuffer, toBaseURL base: URL) throws -> URL {
        guard CVPixelBufferGetPixelFormatType(confidenceMap) == kCVPixelFormatType_OneComponent8 else {
            throw NimbusError.captureFailed("Unexpected depth-confidence pixel format.")
        }
        let (data, width, height) = try Self.packedBytes(of: confidenceMap, bytesPerPixel: 1)
        let url = Self.dimensionedURL(base: base, width: width, height: height, ext: "u8")
        try data.write(to: url, options: .atomic)
        return url
    }

    // MARK: Helpers

    /// Copies the buffer contents row by row so padding bytes (bytesPerRow
    /// larger than width * bytesPerPixel) never reach disk.
    private static func packedBytes(of buffer: CVPixelBuffer,
                                    bytesPerPixel: Int) throws -> (Data, Int, Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw NimbusError.captureFailed("Could not access pixel buffer memory.")
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = width * bytesPerPixel

        var data = Data(capacity: rowBytes * height)
        for row in 0..<height {
            let rowStart = baseAddress.advanced(by: row * bytesPerRow)
            data.append(UnsafeBufferPointer(start: rowStart.assumingMemoryBound(to: UInt8.self),
                                            count: rowBytes))
        }
        return (data, width, height)
    }

    private static func dimensionedURL(base: URL, width: Int, height: Int, ext: String) -> URL {
        base.deletingLastPathComponent()
            .appendingPathComponent("\(base.lastPathComponent)_\(width)x\(height)")
            .appendingPathExtension(ext)
    }

    /// Reads the depth value (meters) at the center pixel of a DepthFloat32
    /// buffer. Used to estimate the camera-to-subject distance for coverage
    /// guidance. Returns nil for invalid or non-finite samples.
    static func centerDepth(of depthMap: CVPixelBuffer) -> Float? {
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
            return nil
        }
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)

        let pointer = baseAddress
            .advanced(by: (height / 2) * bytesPerRow + (width / 2) * MemoryLayout<Float32>.size)
        let value = pointer.assumingMemoryBound(to: Float32.self).pointee
        guard value.isFinite, value > 0 else { return nil }
        return value
    }
}
