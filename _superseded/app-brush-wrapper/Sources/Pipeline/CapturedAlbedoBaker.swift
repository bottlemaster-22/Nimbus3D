//
//  CapturedAlbedoBaker.swift
//  Nimbus3D - Pipeline module
//
//  PLACEHOLDER albedo source. The shared contract has no stage that projects
//  captured frames onto the mesh UV layout, but the Delighter contract requires
//  a lit albedo texture as input. Until a real baker exists, this produces a
//  flat texture filled with the mean color of the captured frames. That is
//  real, working code, and it is honestly labelled as a placeholder: the
//  output is a single averaged color, NOT a projected surface texture.
//
//  TODO(nimbus): a real implementation needs UV-space projection baking:
//  rasterize the UV-unwrapped mesh into texture space on the GPU (Metal
//  compute or render pass), and for each texel sample the best-facing,
//  unoccluded captured frames using CapturedFrame.pose + intrinsics with a
//  depth-based visibility test, then blend by view angle. That capability
//  belongs with the Mesh/Materials modules and should be added to the shared
//  contract as its own stage protocol.
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum CapturedAlbedoBaker {

    /// Writes `albedo_captured_placeholder.png` (resolution x resolution, flat
    /// mean capture color) into `outputDirectory` and returns its URL.
    static func bakePlaceholderAlbedo(from bundle: CaptureBundle,
                                      resolution: Int,
                                      outputDirectory: URL) throws -> URL {
        guard !bundle.frames.isEmpty else {
            throw NimbusError.textureBuildFailed("Capture bundle contains no frames")
        }

        // Sample up to 6 frames spread across the capture.
        let sampleCount = min(6, bundle.frames.count)
        let step = max(1, bundle.frames.count / sampleCount)
        var sum = SIMD3<Double>(repeating: 0)
        var sampled = 0
        for i in stride(from: 0, to: bundle.frames.count, by: step) {
            if let mean = meanColor(ofImageAt: bundle.frames[i].imageURL) {
                sum += mean
                sampled += 1
            }
        }
        guard sampled > 0 else {
            throw NimbusError.textureBuildFailed("Could not decode any capture frames for albedo estimation")
        }
        let mean = sum / Double(sampled)

        // Fill a square texture with the mean color.
        let size = min(max(resolution, 64), 8192)
        guard let context = CGContext(data: nil,
                                      width: size,
                                      height: size,
                                      bitsPerComponent: 8,
                                      bytesPerRow: size * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw NimbusError.textureBuildFailed("Could not create bitmap context for placeholder albedo")
        }
        context.setFillColor(red: CGFloat(mean.x),
                             green: CGFloat(mean.y),
                             blue: CGFloat(mean.z),
                             alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        guard let image = context.makeImage() else {
            throw NimbusError.textureBuildFailed("Could not render placeholder albedo image")
        }

        let url = outputDirectory.appendingPathComponent("albedo_captured_placeholder.png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL,
                                                                UTType.png.identifier as CFString,
                                                                1,
                                                                nil) else {
            throw NimbusError.textureBuildFailed("Could not create PNG destination at \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NimbusError.textureBuildFailed("Could not write placeholder albedo PNG")
        }
        return url
    }

    /// Mean linear-ish RGB (0...1) of the image, via a 32px thumbnail drawn
    /// into a 1x1 bitmap so CoreGraphics does the averaging.
    private static func meanColor(ofImageAt url: URL) -> SIMD3<Double>? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let thumbOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 32,
        ]
        guard let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions as CFDictionary) else {
            return nil
        }
        var pixel = [UInt8](repeating: 0, count: 4)
        guard let context = CGContext(data: &pixel,
                                      width: 1,
                                      height: 1,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        context.interpolationQuality = .medium
        context.draw(thumb, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        return SIMD3<Double>(Double(pixel[0]) / 255.0,
                             Double(pixel[1]) / 255.0,
                             Double(pixel[2]) / 255.0)
    }
}
