//
//  CaptureDepthFrame.swift
//  Capture
//
//  ONE COPY of the native LiDAR depth map, made while the ARFrame is alive,
//  used by everything downstream.
//
//  The rule this file exists to enforce: this is `ARFrame.sceneDepth.depthMap`
//  at its true 256x192 resolution. It is NEVER `smoothedSceneDepth`, and it is
//  never the map upsampled to RGB resolution. docs/DATA_FORMAT.md section 5
//  and F3 both hang off that: depth supervision happens only at these ~49k
//  real samples per frame, because everything between them is interpolation,
//  and training against interpolation teaches the model to reproduce the
//  interpolator.
//
//  `ARWorldTrackingConfiguration.frameSemantics` therefore requests
//  `.sceneDepth` and deliberately does NOT request `.smoothedSceneDepth`.
//  See `ARCaptureService.makeConfiguration()`.
//

import ARKit
import CoreVideo
import Foundation
import simd

/// A native-resolution depth map plus its confidence, copied out of ARKit into
/// plain arrays so the rest of the pipeline can use it after the `ARFrame` has
/// been released.
struct CaptureDepthFrame: Sendable {

    let width: Int
    let height: Int

    /// Row-major, top-left origin, millimetres. `0` is a no-return, exactly as
    /// docs/DATA_FORMAT.md section 5 defines it: not "distance zero" and not
    /// "empty", but "the beam came back with nothing".
    let depthMillimetres: [UInt16]

    /// Row-major, same dimensions. `ARConfidenceLevel` raw values:
    /// 0 low, 1 medium, 2 high. Stored, not believed - the pre-pass remaps it
    /// through observed revisit residuals (F6).
    let confidence: [UInt8]

    /// Fraction of samples with a usable return, 0...1. Feeds `FrameQC`.
    let validFraction: Float

    /// Copies `ARDepthData` out of ARKit.
    ///
    /// - Returns: nil when the buffers cannot be locked or are not the format
    ///   ARKit documents. Returning nil rather than a zero-filled map matters:
    ///   a frame with no depth is written with `depthPath == nil`, and a
    ///   downstream reader can tell "no LiDAR here" from "LiDAR saw nothing",
    ///   which are completely different facts.
    init?(depthData: ARDepthData) {
        let depthMap = depthData.depthMap
        let mapWidth = CVPixelBufferGetWidth(depthMap)
        let mapHeight = CVPixelBufferGetHeight(depthMap)
        guard mapWidth > 0, mapHeight > 0 else { return nil }
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32
        else {
            CaptureLog.session.error(
                "sceneDepth.depthMap was not DepthFloat32; refusing to guess "
                    + "its layout."
            )
            return nil
        }
        guard CVPixelBufferLockBaseAddress(depthMap, .readOnly) == kCVReturnSuccess
        else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let depthRowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        let sampleCount = mapWidth * mapHeight
        let maxRange = CaptureTuning.lidarMaxRangeMeters

        var millimetres = [UInt16](repeating: CaptureTuning.depthNoReturn, count: sampleCount)
        var validCount = 0

        millimetres.withUnsafeMutableBufferPointer { out in
            for y in 0..<mapHeight {
                let row = (depthBase + y * depthRowBytes)
                    .assumingMemoryBound(to: Float32.self)
                for x in 0..<mapWidth {
                    let metres = row[x]
                    // A NaN, a non-positive value, or anything past the
                    // sensor's honest range is written as a no-return. Past
                    // the range the reading is not a long measurement, it is
                    // noise, and F2 needs "beyond range" to stay UNKNOWN
                    // rather than becoming a confident far surface.
                    guard metres.isFinite, metres > 0.01, metres <= maxRange else {
                        continue
                    }
                    let mm = metres * 1000
                    guard mm < Float(UInt16.max) else { continue }
                    out[y * mapWidth + x] = UInt16(mm)
                    validCount += 1
                }
            }
        }

        var confidenceBytes = [UInt8](repeating: 0, count: sampleCount)
        if let confidenceMap = depthData.confidenceMap,
            CVPixelBufferGetWidth(confidenceMap) == mapWidth,
            CVPixelBufferGetHeight(confidenceMap) == mapHeight,
            CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) == kCVReturnSuccess
        {
            defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
            if let confidenceBase = CVPixelBufferGetBaseAddress(confidenceMap) {
                let rowBytes = CVPixelBufferGetBytesPerRow(confidenceMap)
                confidenceBytes.withUnsafeMutableBufferPointer { out in
                    for y in 0..<mapHeight {
                        let row = (confidenceBase + y * rowBytes)
                            .assumingMemoryBound(to: UInt8.self)
                        for x in 0..<mapWidth {
                            out[y * mapWidth + x] = row[x]
                        }
                    }
                }
            }
        }

        self.width = mapWidth
        self.height = mapHeight
        self.depthMillimetres = millimetres
        self.confidence = confidenceBytes
        self.validFraction = sampleCount > 0
            ? Float(validCount) / Float(sampleCount)
            : 0
    }

    // MARK: - Bytes for the sidecars

    /// `sensor_data/depth/frame_*.depth16`: `UInt16` millimetres,
    /// little-endian, row-major, no header, no padding.
    ///
    /// Both ends of this project are little-endian (arm64 iPhone, x86-64 or
    /// arm64 PC), so this is a straight memory copy; the explicit
    /// `littleEndian` conversion is there so the code stays correct if that
    /// ever stops being true, and costs nothing when it is.
    var depthSidecarBytes: Data {
        var littleEndian = depthMillimetres.map { $0.littleEndian }
        return littleEndian.withUnsafeMutableBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }

    /// `sensor_data/confidence/frame_*.conf8`: one `UInt8` per sample.
    var confidenceSidecarBytes: Data {
        confidence.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    // MARK: - Unprojection

    /// Unprojects one native depth sample into camera space, in the app's
    /// camera convention (+X right, +Y down, +Z forward).
    ///
    /// `intrinsics` must be the RGB-resolution intrinsics; they are scaled to
    /// the depth map's resolution here, because the depth map is a lower
    /// resolution view through the same lens. That scaling is the ONLY
    /// relationship between the two rasters this module assumes, and it is the
    /// one ARKit documents.
    ///
    /// - Returns: nil for a no-return sample.
    func unprojectToCameraSpace(
        x: Int,
        y: Int,
        intrinsics: CameraIntrinsics
    ) -> SIMD3<Float>? {
        let index = y * width + x
        guard index >= 0, index < depthMillimetres.count else { return nil }
        let mm = depthMillimetres[index]
        guard mm != CaptureTuning.depthNoReturn else { return nil }
        let z = Float(mm) / 1000

        let scaled = intrinsics.scaled(toWidth: width, height: height)
        let u = Float(x) + 0.5
        let v = Float(y) + 0.5
        return SIMD3<Float>(
            (u - scaled.cx) / scaled.fx * z,
            (v - scaled.cy) / scaled.fy * z,
            z
        )
    }

    /// Surface normal at a native sample, estimated from its right and down
    /// neighbours in camera space. Used for the incidence-angle term of the
    /// point cloud's expected-error column.
    ///
    /// - Returns: nil when any of the three samples is a no-return, which is
    ///   correct: a normal estimated across a depth discontinuity is a
    ///   fabrication, and this is exactly the band F3 tells us to distrust.
    func cameraSpaceNormal(
        x: Int,
        y: Int,
        intrinsics: CameraIntrinsics
    ) -> SIMD3<Float>? {
        guard x + 1 < width, y + 1 < height else { return nil }
        guard
            let center = unprojectToCameraSpace(x: x, y: y, intrinsics: intrinsics),
            let right = unprojectToCameraSpace(x: x + 1, y: y, intrinsics: intrinsics),
            let down = unprojectToCameraSpace(x: x, y: y + 1, intrinsics: intrinsics)
        else { return nil }
        let normal = simd_cross(right - center, down - center)
        let length = simd_length(normal)
        guard length > 1e-8 else { return nil }
        // Point it back towards the camera: the camera is at the origin in
        // camera space, so the outward-facing choice is the one with -Z.
        let unit = normal / length
        return unit.z > 0 ? -unit : unit
    }
}
