//
//  EquirectProjector.swift — HDRI module.
//
//  Gather-based projection of posed pinhole radiance images into a single
//  equirectangular (2:1) panorama, with feathered multi-view blending.
//
//  For every output panorama pixel we form its world-space view ray, transform
//  that ray into each source camera, project it through the pinhole intrinsics,
//  and accumulate a distance-to-edge-feathered radiance sample. Gather (rather
//  than scatter) guarantees every covered output pixel is filled exactly once
//  with no seams from source-pixel gaps.
//
//  Conventions (ARKit): camera-to-world `pose`; camera local axes are +X right,
//  +Y up, and the camera looks down local -Z. A world ray is visible when its
//  camera-space z is negative.
//

import Foundation
import simd

enum EquirectProjector {

    struct Result {
        var pixels: [SIMD3<Float>]   // linear radiance, top-origin row-major
        var width: Int
        var height: Int
        /// Fraction of output pixels that received at least one source sample.
        var coverage: Float
    }

    private struct Cam {
        let rt: simd_float3x3        // world-to-camera rotation (pose rotation transposed)
        let fx: Float, fy: Float
        let cx: Float, cy: Float
        let w: Float, h: Float
        let imageIndex: Int
    }

    /// Equirectangular direction for output pixel center (x, y). +Y is up;
    /// phi wraps around the horizon, theta spans +pi/2 (top) to -pi/2 (bottom).
    @inline(__always)
    static func direction(x: Int, y: Int, width: Int, height: Int) -> SIMD3<Float> {
        let u = (Float(x) + 0.5) / Float(width)
        let v = (Float(y) + 0.5) / Float(height)
        let phi = u * 2.0 * .pi - .pi
        let theta = .pi / 2.0 - v * .pi
        let ct = cosf(theta)
        return SIMD3<Float>(ct * sinf(phi), sinf(theta), ct * cosf(phi))
    }

    static func project(images: [RadianceImage],
                        outputWidth: Int,
                        onRowBatch: (Int) -> Void) -> Result {
        let W = max(outputWidth, 2)
        let H = W / 2

        let cams: [Cam] = images.enumerated().map { (idx, img) in
            let m = img.pose
            // Upper-left 3x3 rotation, then transpose to get world->camera.
            let r = simd_float3x3(
                SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            return Cam(rt: r.transpose,
                       fx: img.focalLength.x, fy: img.focalLength.y,
                       cx: img.principalPoint.x, cy: img.principalPoint.y,
                       w: Float(img.width), h: Float(img.height),
                       imageIndex: idx)
        }

        var rgb = [SIMD3<Float>](repeating: .zero, count: W * H)
        var wsum = [Float](repeating: 0, count: W * H)

        rgb.withUnsafeMutableBufferPointer { rgbp in
            wsum.withUnsafeMutableBufferPointer { wp in
                DispatchQueue.concurrentPerform(iterations: H) { y in
                    for x in 0..<W {
                        let dir = direction(x: x, y: y, width: W, height: H)
                        var acc = SIMD3<Float>.zero
                        var wa: Float = 0

                        for cam in cams {
                            let c = cam.rt * dir       // ray in camera space
                            let cz = -c.z              // forward depth (camera looks down -Z)
                            if cz <= 1e-6 { continue } // behind camera

                            let px = cam.fx * (c.x / cz) + cam.cx
                            let py = cam.fy * (-c.y / cz) + cam.cy
                            if px < 0 || px >= cam.w || py < 0 || py >= cam.h { continue }

                            // Feather: fade to zero at the frame border, weight
                            // central/on-axis samples more (cz = cos angle-to-axis).
                            let nx = abs(px - cam.cx) / (cam.w * 0.5)
                            let ny = abs(py - cam.cy) / (cam.h * 0.5)
                            let edge = max(nx, ny)
                            let feather = max(1.0 - edge, 0.0)
                            let fw = feather * feather * cz
                            if fw <= 0 { continue }

                            let s = bilinear(images[cam.imageIndex], px, py)
                            acc += s * fw
                            wa += fw
                        }

                        let idx = y * W + x
                        rgbp[idx] = acc
                        wp[idx] = wa
                    }
                    onRowBatch(y)
                }
            }
        }

        // Normalize covered pixels; fill uncovered with the covered mean so the
        // EXR is usable for image-based lighting without black gaps.
        // TODO(nimbus): replace flat mean-fill with proper hole-filling
        // (push/pull pyramid or Poisson inpainting) for gap-tolerant IBL.
        var coveredCount = 0
        var meanAccum = SIMD3<Double>.zero
        for i in 0..<(W * H) where wsum[i] > 0 {
            let v = rgb[i] / wsum[i]
            rgb[i] = v
            meanAccum += SIMD3<Double>(Double(v.x), Double(v.y), Double(v.z))
            coveredCount += 1
        }
        if coveredCount > 0 {
            let mean = SIMD3<Float>(Float(meanAccum.x / Double(coveredCount)),
                                    Float(meanAccum.y / Double(coveredCount)),
                                    Float(meanAccum.z / Double(coveredCount)))
            for i in 0..<(W * H) where wsum[i] <= 0 {
                rgb[i] = mean
            }
        }

        return Result(pixels: rgb, width: W, height: H,
                      coverage: Float(coveredCount) / Float(W * H))
    }

    /// Clamped bilinear sample of a radiance image at continuous pixel (px, py).
    @inline(__always)
    private static func bilinear(_ img: RadianceImage,
                                 _ px: Float, _ py: Float) -> SIMD3<Float> {
        let x = min(max(px - 0.5, 0), Float(img.width - 1))
        let y = min(max(py - 0.5, 0), Float(img.height - 1))
        let x0 = Int(x), y0 = Int(y)
        let x1 = min(x0 + 1, img.width - 1)
        let y1 = min(y0 + 1, img.height - 1)
        let fx = x - Float(x0)
        let fy = y - Float(y0)
        let row0 = y0 * img.width
        let row1 = y1 * img.width
        let p00 = img.pixels[row0 + x0]
        let p10 = img.pixels[row0 + x1]
        let p01 = img.pixels[row1 + x0]
        let p11 = img.pixels[row1 + x1]
        let a = p00 * (1 - fx) + p10 * fx
        let b = p01 * (1 - fx) + p11 * fx
        return a * (1 - fy) + b * fy
    }
}
