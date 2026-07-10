//
//  TexturePacker.swift — Export module
//
//  Real, deterministic on-device image work with Core Graphics / ImageIO:
//   - read a material map's raw bytes + MIME (so PNG/JPEG pass through untouched),
//   - pack separate roughness / metallic / AO grayscale maps into a single glTF
//     "ORM" texture (R = occlusion, G = roughness, B = metallic), which is exactly
//     what glTF's `metallicRoughnessTexture` (G/B) + `occlusionTexture` (R) expect.
//
//  No network, no ML, no faking: this is standard raster compositing.
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum TexturePacker {

    struct EncodedImage {
        var bytes: Data
        var mime: String        // "image/png" | "image/jpeg"
        var fileExtension: String
    }

    /// Returns a material map ready to embed/reference in glTF. PNG and JPEG pass
    /// through byte-for-byte (glTF's only allowed image formats); anything else is
    /// transcoded to PNG so the output is always spec-valid.
    static func encodedImage(at url: URL) -> EncodedImage? {
        let ext = url.pathExtension.lowercased()
        if ext == "png", let data = try? Data(contentsOf: url) {
            return EncodedImage(bytes: data, mime: "image/png", fileExtension: "png")
        }
        if ext == "jpg" || ext == "jpeg", let data = try? Data(contentsOf: url) {
            return EncodedImage(bytes: data, mime: "image/jpeg", fileExtension: "jpg")
        }
        // Unknown/HEIC/etc: transcode to PNG.
        guard let cg = loadCGImage(url), let png = pngData(cg) else { return nil }
        return EncodedImage(bytes: png, mime: "image/png", fileExtension: "png")
    }

    /// Packs an ORM texture. `nil` inputs get sensible constant channels
    /// (occlusion = 255 = none, roughness = 255 = fully rough, metallic = 0 = dielectric).
    /// Returns nil only if every input is nil (caller should then fall back to scalar factors).
    static func packORM(roughness: URL?,
                        metallic: URL?,
                        ao: URL?,
                        resolution: Int) -> EncodedImage? {
        guard roughness != nil || metallic != nil || ao != nil else { return nil }
        let size = max(1, resolution)

        let roughGray = roughness.flatMap { loadGray($0, size: size) }
        let metalGray = metallic.flatMap { loadGray($0, size: size) }
        let aoGray    = ao.flatMap { loadGray($0, size: size) }

        let pixelCount = size * size
        var rgbx = [UInt8](repeating: 0, count: pixelCount * 4)
        for i in 0..<pixelCount {
            let o = i * 4
            rgbx[o + 0] = aoGray?[i]    ?? 255   // R = ambient occlusion
            rgbx[o + 1] = roughGray?[i] ?? 255   // G = roughness
            rgbx[o + 2] = metalGray?[i] ?? 0     // B = metallic
            rgbx[o + 3] = 255                    // unused (noneSkipLast)
        }
        guard let cg = makeRGBXImage(rgbx, size: size), let png = pngData(cg) else { return nil }
        return EncodedImage(bytes: png, mime: "image/png", fileExtension: "png")
    }

    // MARK: - Core Graphics helpers

    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// Decodes any image to a `size`x`size` single-channel (8-bit gray) buffer,
    /// resampling as needed. Data maps (roughness/metallic/AO) are treated as raw
    /// linear values, which is correct for glTF's ORM channels.
    private static func loadGray(_ url: URL, size: Int) -> [UInt8]? {
        guard let img = loadCGImage(url) else { return nil }
        var buffer = [UInt8](repeating: 0, count: size * size)
        let gray = CGColorSpaceCreateDeviceGray()
        let ok: Bool = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress,
                                      width: size,
                                      height: size,
                                      bitsPerComponent: 8,
                                      bytesPerRow: size,
                                      space: gray,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        return ok ? buffer : nil
    }

    private static func makeRGBXImage(_ pixels: [UInt8], size: Int) -> CGImage? {
        var data = pixels
        let rgb = CGColorSpaceCreateDeviceRGB()
        return data.withUnsafeMutableBytes { raw -> CGImage? in
            guard let ctx = CGContext(data: raw.baseAddress,
                                      width: size,
                                      height: size,
                                      bitsPerComponent: 8,
                                      bytesPerRow: size * 4,
                                      space: rgb,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
            return ctx.makeImage()
        }
    }

    private static func pngData(_ image: CGImage) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out,
                                                          UTType.png.identifier as CFString,
                                                          1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
