//
//  MaterialImageIO.swift
//  Nimbus3D (Materials module)
//
//  CPU image helpers used by the Materials module: load/save, resampling,
//  mean-color measurement, tinting in linear space, and the normal-from-height
//  baseline (Sobel). All real code, no external dependencies beyond ImageIO,
//  CoreGraphics, and UniformTypeIdentifiers.
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import simd

enum MaterialImageIO {

    // MARK: - Load / save

    static func loadCGImage(at url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NimbusError.textureBuildFailed("Could not read image at \(url.lastPathComponent)")
        }
        return image
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL,
                                                                UTType.png.identifier as CFString,
                                                                1, nil) else {
            throw NimbusError.textureBuildFailed("Could not create PNG destination at \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NimbusError.textureBuildFailed("Could not write PNG at \(url.lastPathComponent)")
        }
    }

    // MARK: - Pixel access

    /// Draws `image` into a square RGBA8 buffer of edge length `size` and returns the bytes.
    static func rgba8Pixels(of image: CGImage, size: Int) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        try pixels.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: size,
                                          height: size,
                                          bitsPerComponent: 8,
                                          bytesPerRow: size * 4,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo) else {
                throw NimbusError.textureBuildFailed("Could not create RGBA8 context")
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        return pixels
    }

    /// Builds a CGImage from a square RGBA8 buffer.
    static func cgImage(fromRGBA8 pixels: [UInt8], size: Int) throws -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        var mutable = pixels
        let image: CGImage? = mutable.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: size,
                                          height: size,
                                          bitsPerComponent: 8,
                                          bytesPerRow: size * 4,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo) else { return nil }
            return context.makeImage()
        }
        guard let image else {
            throw NimbusError.textureBuildFailed("Could not create image from RGBA8 buffer")
        }
        return image
    }

    /// Draws `image` into a square 8-bit grayscale buffer of edge length `size`.
    static func gray8Pixels(of image: CGImage, size: Int) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size)
        try pixels.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: size,
                                          height: size,
                                          bitsPerComponent: 8,
                                          bytesPerRow: size,
                                          space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                throw NimbusError.textureBuildFailed("Could not create grayscale context")
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        return pixels
    }

    /// Builds a CGImage from a square 8-bit grayscale buffer.
    static func cgImage(fromGray8 pixels: [UInt8], size: Int) throws -> CGImage {
        var mutable = pixels
        let image: CGImage? = mutable.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: size,
                                          height: size,
                                          bitsPerComponent: 8,
                                          bytesPerRow: size,
                                          space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
            return context.makeImage()
        }
        guard let image else {
            throw NimbusError.textureBuildFailed("Could not create image from grayscale buffer")
        }
        return image
    }

    // MARK: - Resampling

    /// Resamples `image` to a square `size` x `size` RGBA image.
    static func resampled(_ image: CGImage, to size: Int) throws -> CGImage {
        let pixels = try rgba8Pixels(of: image, size: size)
        return try cgImage(fromRGBA8: pixels, size: size)
    }

    /// Resamples `image` to a square grayscale image (for height/roughness/AO maps).
    static func resampledGray(_ image: CGImage, to size: Int) throws -> CGImage {
        let pixels = try gray8Pixels(of: image, size: size)
        return try cgImage(fromGray8: pixels, size: size)
    }

    /// Constant-value grayscale map (e.g. flat roughness or zero metallic).
    static func constantGray(value: UInt8, size: Int) throws -> CGImage {
        let pixels = [UInt8](repeating: value, count: size * size)
        return try cgImage(fromGray8: pixels, size: size)
    }

    // MARK: - Color math

    @inline(__always)
    static func srgbToLinear(_ v: Double) -> Double {
        v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
    }

    @inline(__always)
    static func linearToSrgb(_ v: Double) -> Double {
        v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055
    }

    /// Mean color of the image in linear RGB. Downsamples to 32x32 first, which is
    /// plenty for a global tint estimate and keeps this O(1) regardless of input size.
    static func meanLinearRGB(of image: CGImage) throws -> SIMD3<Double> {
        let size = 32
        let pixels = try rgba8Pixels(of: image, size: size)
        var sum = SIMD3<Double>(repeating: 0)
        var count = 0.0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let a = Double(pixels[i + 3]) / 255.0
            guard a > 0.5 else { continue }  // skip transparent texels (outside UV islands)
            sum.x += srgbToLinear(Double(pixels[i]) / 255.0)
            sum.y += srgbToLinear(Double(pixels[i + 1]) / 255.0)
            sum.z += srgbToLinear(Double(pixels[i + 2]) / 255.0)
            count += 1
        }
        guard count > 0 else {
            throw NimbusError.textureBuildFailed("Albedo texture is fully transparent; cannot measure mean color")
        }
        return sum / count
    }

    /// Multiplies every pixel by `gain` in linear space and converts back to sRGB.
    /// Used to tint a library albedo toward the captured surface color.
    static func tinted(_ image: CGImage, gain: SIMD3<Double>, size: Int) throws -> CGImage {
        var pixels = try rgba8Pixels(of: image, size: size)
        // Precompute a 256-entry lookup per channel: sRGB byte -> tinted sRGB byte.
        var lut = [[UInt8]](repeating: [UInt8](repeating: 0, count: 256), count: 3)
        let gains = [gain.x, gain.y, gain.z]
        for c in 0..<3 {
            for v in 0..<256 {
                let linear = srgbToLinear(Double(v) / 255.0) * gains[c]
                let srgb = linearToSrgb(min(max(linear, 0), 1))
                lut[c][v] = UInt8((srgb * 255.0).rounded())
            }
        }
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i]     = lut[0][Int(pixels[i])]
            pixels[i + 1] = lut[1][Int(pixels[i + 1])]
            pixels[i + 2] = lut[2][Int(pixels[i + 2])]
        }
        return try cgImage(fromRGBA8: pixels, size: size)
    }

    // MARK: - Normal-from-height baseline

    /// Generates a tangent-space normal map (OpenGL convention, +Y up) from a
    /// grayscale height buffer using a Sobel filter. This is the honest baseline:
    /// it derives relief from whatever height signal it is given (a real library
    /// displacement map, or luminance when nothing better exists). It does NOT
    /// recover true captured surface relief.
    static func normalMap(fromHeightPixels height: [UInt8], size: Int, strength: Float, wrap: Bool) throws -> CGImage {
        precondition(height.count == size * size, "height buffer size mismatch")
        var out = [UInt8](repeating: 255, count: size * size * 4)

        @inline(__always)
        func sample(_ x: Int, _ y: Int) -> Float {
            var sx = x, sy = y
            if wrap {
                sx = (x + size) % size
                sy = (y + size) % size
            } else {
                sx = min(max(x, 0), size - 1)
                sy = min(max(y, 0), size - 1)
            }
            return Float(height[sy * size + sx]) / 255.0
        }

        for y in 0..<size {
            for x in 0..<size {
                // Sobel gradients
                let tl = sample(x - 1, y - 1), t = sample(x, y - 1), tr = sample(x + 1, y - 1)
                let l  = sample(x - 1, y),                            r  = sample(x + 1, y)
                let bl = sample(x - 1, y + 1), b = sample(x, y + 1), br = sample(x + 1, y + 1)
                let dx = (tr + 2 * r + br) - (tl + 2 * l + bl)
                let dy = (bl + 2 * b + br) - (tl + 2 * t + tr)
                var n = SIMD3<Float>(-dx * strength, -dy * strength, 1)
                n = simd_normalize(n)
                let i = (y * size + x) * 4
                out[i]     = UInt8(((n.x * 0.5 + 0.5) * 255).rounded())
                out[i + 1] = UInt8(((n.y * 0.5 + 0.5) * 255).rounded())
                out[i + 2] = UInt8(((n.z * 0.5 + 0.5) * 255).rounded())
                out[i + 3] = 255
            }
        }
        return try cgImage(fromRGBA8: out, size: size)
    }
}
