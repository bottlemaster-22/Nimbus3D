//
//  RadianceImage.swift — HDRI module.
//
//  Decodes captured LDR bracket frames and performs a Debevec-style weighted
//  radiance merge into a single linear (scene-radiance) image per camera view.
//
//  Honesty note on the camera response function (CRF):
//  Debevec & Malik (1997) recover an arbitrary per-device CRF by solving a large
//  least-squares system. iOS HEIC/JPEG frames are, by contract, encoded in the
//  sRGB transfer function, so instead of recovering an unknown CRF we invert the
//  KNOWN sRGB response (a mathematically exact, verifiable operation) and apply
//  Debevec's log-domain weighted-average merge equation on top of it. This is a
//  deliberate, valid simplification — NOT a stub. If a device is ever found whose
//  frames use a non-sRGB response, swap `srgbLUT` for a recovered CRF.
//  TODO(nimbus): optional full Debevec CRF recovery (Accelerate LAPACK sgelsd_)
//  for cameras that expose RAW/non-sRGB bracket frames.
//

import Foundation
import CoreGraphics
import ImageIO
import simd

/// A single camera view merged to linear radiance, with the geometry needed to
/// project it into an equirectangular panorama.
struct RadianceImage {
    let width: Int
    let height: Int
    /// Linear radiance, top-origin, row-major (`pixels[y*width + x]`).
    var pixels: [SIMD3<Float>]
    /// Camera-to-world transform (ARKit world space).
    let pose: simd_float4x4
    /// Focal length in pixels, scaled to the decoded image dimensions.
    let focalLength: SIMD2<Float>
    /// Principal point in pixels, scaled to the decoded image dimensions.
    let principalPoint: SIMD2<Float>
}

enum RadianceMerge {

    /// Precomputed exact sRGB electro-optical transfer (sRGB 8-bit -> linear).
    static let srgbLUT: [Float] = (0..<256).map { i in
        let c = Float(i) / 255.0
        return c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }

    /// Debevec triangle ("hat") weighting: de-weights under/over-exposed samples,
    /// peaks at mid-gray. Range 0...1.
    @inline(__always)
    static func hat(_ z: UInt8) -> Float {
        let f = Float(z) / 255.0
        return f <= 0.5 ? f * 2.0 : (1.0 - f) * 2.0
    }

    /// Relative exposure of a bracket frame in consistent units, so that
    /// `linear / exposure` yields comparable scene radiance across frames.
    ///
    /// Physically, exposure ∝ shutter_time × ISO_gain. When ISO is unavailable
    /// (the Capture module documents that ARKit does not vend per-frame ISO, so it
    /// is left 0), we fall back to shutter time and, if present, the recorded EV
    /// bias as a gain proxy.
    /// TODO(nimbus): once Capture can vend true per-frame ISO, drop the EV-bias
    /// proxy branch — it can leave brightness seams between differently-gained views.
    @inline(__always)
    static func relativeExposure(_ f: ExposureBracketFrame) -> Float {
        var t = Float(max(f.exposureDuration, 1e-6))
        if f.iso > 0 {
            t *= f.iso / 100.0
        } else if f.exposureBias != 0 {
            t *= powf(2.0, f.exposureBias)
        }
        return max(t, 1e-9)
    }

    /// Merge one pose-group of bracket frames into a single linear radiance image.
    /// A group of one degrades gracefully to `linear / exposure` (a correct
    /// exposure-normalized linearization, no averaging).
    static func merge(_ frames: [ExposureBracketFrame]) throws -> RadianceImage {
        precondition(!frames.isEmpty, "merge requires at least one frame")

        // Decode every frame to top-origin sRGB-8 RGBA.
        var decoded: [(w: Int, h: Int, buf: [UInt8], logT: Float)] = []
        decoded.reserveCapacity(frames.count)
        for f in frames {
            let (w, h, buf) = try loadRGBA8(f.imageURL)
            decoded.append((w, h, buf, logf(relativeExposure(f))))
        }

        let w = decoded[0].w
        let h = decoded[0].h
        for d in decoded where d.w != w || d.h != h {
            throw NimbusError.hdriAssemblyFailed(
                "Bracket frames in a pose group have mismatched dimensions.")
        }

        let lut = srgbLUT
        var pixels = [SIMD3<Float>](repeating: .zero, count: w * h)

        pixels.withUnsafeMutableBufferPointer { pp in
            DispatchQueue.concurrentPerform(iterations: h) { y in
                for x in 0..<w {
                    let pi = y * w + x
                    let bo = pi * 4
                    var lnNum = SIMD3<Float>.zero
                    var wsum = SIMD3<Float>.zero

                    for d in decoded {
                        let logT = d.logT
                        // Per channel: Debevec log-domain merge
                        //   lnE += w(z) * (ln(response(z)) - ln(t))
                        let r = d.buf[bo], g = d.buf[bo + 1], b = d.buf[bo + 2]

                        let wr = hat(r) + 1e-3
                        lnNum.x += wr * (logf(max(lut[Int(r)], 1e-6)) - logT); wsum.x += wr

                        let wg = hat(g) + 1e-3
                        lnNum.y += wg * (logf(max(lut[Int(g)], 1e-6)) - logT); wsum.y += wg

                        let wb = hat(b) + 1e-3
                        lnNum.z += wb * (logf(max(lut[Int(b)], 1e-6)) - logT); wsum.z += wb
                    }

                    pp[pi] = SIMD3<Float>(
                        wsum.x > 0 ? expf(lnNum.x / wsum.x) : 0,
                        wsum.y > 0 ? expf(lnNum.y / wsum.y) : 0,
                        wsum.z > 0 ? expf(lnNum.z / wsum.z) : 0)
                }
            }
        }

        // Scale intrinsics from their captured reference size to the decoded size.
        let ref = frames[0].intrinsics
        let sx = Float(w) / Float(max(ref.imageWidth, 1))
        let sy = Float(h) / Float(max(ref.imageHeight, 1))
        let focal = SIMD2<Float>(ref.focalLength.x * sx, ref.focalLength.y * sy)
        let principal = SIMD2<Float>(ref.principalPoint.x * sx, ref.principalPoint.y * sy)

        return RadianceImage(width: w, height: h, pixels: pixels,
                             pose: frames[0].pose.matrix,
                             focalLength: focal, principalPoint: principal)
    }

    /// Decode an image file to a top-origin, row-major RGBA8 buffer in the sRGB
    /// color space (values kept sRGB-encoded; the caller linearizes via `srgbLUT`).
    private static func loadRGBA8(_ url: URL) throws -> (Int, Int, [UInt8]) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NimbusError.hdriAssemblyFailed(
                "Cannot decode bracket image: \(url.lastPathComponent)")
        }
        let w = img.width
        let h = img.height
        guard w > 0, h > 0 else {
            throw NimbusError.hdriAssemblyFailed(
                "Bracket image has zero size: \(url.lastPathComponent)")
        }
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw NimbusError.hdriAssemblyFailed("Cannot create sRGB color space.")
        }
        // RGBX, 8bpc. CGContext allocates and owns the backing store.
        let bmp = CGImageAlphaInfo.noneSkipLast.rawValue
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: bmp),
              let base = drawAndBase(ctx, img, w, h) else {
            throw NimbusError.hdriAssemblyFailed(
                "Cannot rasterize bracket image: \(url.lastPathComponent)")
        }

        // CGBitmapContext memory is bottom-origin; flip rows to top-origin so
        // pixel[y=0] is the top of the image (matches the pinhole v axis).
        let rowBytes = w * 4
        var out = [UInt8](repeating: 0, count: w * h * 4)
        let srcPtr = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<h {
                let srcRow = (h - 1 - y) * rowBytes
                let dstRow = y * rowBytes
                memcpy(dst.baseAddress! + dstRow, srcPtr + srcRow, rowBytes)
            }
        }
        return (w, h, out)
    }

    private static func drawAndBase(_ ctx: CGContext, _ img: CGImage,
                                    _ w: Int, _ h: Int) -> UnsafeMutableRawPointer? {
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)))
        return ctx.data
    }
}
