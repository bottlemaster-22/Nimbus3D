//
//  CapturePointCloudAccumulator.swift
//  Capture
//
//  The LiDAR cloud that gets baked into `sparse/0/points3D.txt`.
//
//  Why accumulate at all rather than dump every sample: a four-minute walk at
//  6 keyframes a second is ~1,400 frames, and each one carries ~49k native
//  depth samples. That is 70 million points, of which the overwhelming
//  majority are the same square centimetre of wall measured eighty times.
//  docs/DATA_FORMAT.md section 4 fixes the answer: voxel-downsample at 1 cm
//  before writing.
//
//  Two things are averaged inside a voxel and one is not:
//
//    position   averaged. The mean of N independent measurements of a static
//               surface is a better estimate than any one of them.
//    colour     averaged. Same argument, and it makes the cloud look like the
//               room rather than like the last frame that hit it.
//    error      NOT averaged - the MINIMUM is kept. The `ERROR` column is the
//               expected metric error of the estimate, and if one frame saw
//               this patch head-on from 80 cm then that is how well the patch
//               is known, however many grazing 4 m looks came later.
//
//  Memory is bounded, never assumed (the product's adaptive rule). Past the
//  cap the voxel size doubles and every existing voxel is re-binned, which
//  loses resolution and says so, rather than growing until the phone is killed
//  mid-walk.
//

import Foundation
import simd

/// Accumulates LiDAR returns into a 1 cm voxel grid.
///
/// Thread-safe: fed from the capture pipeline's queue, drained once at the end
/// of the session from whichever queue is writing the COLMAP model.
final class CapturePointCloudAccumulator: @unchecked Sendable {

    /// What one occupied voxel knows.
    private struct Bin {
        var positionSum: SIMD3<Double> = .zero
        var colorSum: SIMD3<Double> = .zero
        var count: UInt32 = 0
        /// Best (lowest) expected metric error any observation achieved.
        var bestError: Float = .greatestFiniteMagnitude
    }

    /// One point as written to `points3D.txt`.
    struct Point {
        var position: SIMD3<Float>
        var color: SIMD3<UInt8>
        /// Expected metric error, METRES. docs/DATA_FORMAT.md section 4
        /// documents the deliberate reinterpretation of COLMAP's `ERROR`
        /// column, which would otherwise be reprojection pixels.
        var errorMeters: Float
    }

    private let lock = NSLock()
    private var bins: [Int64: Bin] = [:]
    private var voxelSize: Float = CaptureTuning.pointCloudVoxelSizeMeters

    /// True once the accumulator has had to coarsen to stay inside its cap.
    /// Reported, never hidden: the cloud a downstream stage receives is then
    /// genuinely lower resolution than the format's nominal 1 cm.
    private(set) var didDegrade = false

    /// The voxel size actually in force, which is the nominal 1 cm unless the
    /// cap forced a coarsening.
    var effectiveVoxelSizeMeters: Float {
        lock.lock()
        defer { lock.unlock() }
        return voxelSize
    }

    var pointCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return bins.count
    }

    // MARK: - Feeding

    /// Adds every usable native depth sample of one frame.
    ///
    /// - Parameters:
    ///   - depth: the native 256x192 map, never the upsampled one.
    ///   - pose: this frame's world -> camera pose.
    ///   - intrinsics: RGB-resolution intrinsics; scaled internally.
    ///   - image: the frame's pixels, for colour. Pass nil to accumulate
    ///     geometry without colour, which writes mid-grey rather than black -
    ///     black reads as "measured and dark" to a viewer, and it was not.
    ///   - stride: subsampling stride through the depth map.
    func add(
        depth: CaptureDepthFrame,
        pose: Pose,
        intrinsics: CameraIntrinsics,
        image: CapturePixelBuffer?,
        stride: Int = CaptureTuning.pointCloudDepthStride
    ) {
        let cameraToWorld = pose.matrix.inverse
        let step = Swift.max(1, stride)

        var sampled: [(SIMD3<Float>, SIMD3<UInt8>, Float)] = []
        sampled.reserveCapacity((depth.width / step) * (depth.height / step))

        let hasColor = image?.beginSampling() ?? false
        defer { if hasColor { image?.endSampling() } }

        var y = 0
        while y < depth.height {
            var x = 0
            while x < depth.width {
                guard
                    let cameraPoint = depth.unprojectToCameraSpace(
                        x: x,
                        y: y,
                        intrinsics: intrinsics
                    )
                else {
                    x += step
                    continue
                }

                let range = simd_length(cameraPoint)
                guard range > 0.05 else {
                    x += step
                    continue
                }

                // Incidence: 1 head-on, towards 0 at a grazing angle. The
                // normal is nil across a depth discontinuity, and a normal
                // estimated across one is a fabrication - so those samples are
                // treated as maximally grazing rather than as perfect.
                let rayDirection = cameraPoint / range
                var incidenceCosine: Float = 0.35
                if let normal = depth.cameraSpaceNormal(
                    x: x,
                    y: y,
                    intrinsics: intrinsics
                ) {
                    incidenceCosine = abs(simd_dot(normal, rayDirection))
                }

                let confidence = depth.confidence[y * depth.width + x]
                let error = CaptureTuning.expectedDepthErrorMeters(
                    rangeMeters: range,
                    incidenceCosine: incidenceCosine,
                    confidence: confidence
                )

                let world = cameraToWorld * SIMD4<Float>(cameraPoint, 1)
                let worldPoint = SIMD3<Float>(world.x, world.y, world.z)

                var color = SIMD3<UInt8>(128, 128, 128)
                if hasColor, let image {
                    color = image.rgb(
                        atNormalizedX: (Float(x) + 0.5) / Float(depth.width),
                        y: (Float(y) + 0.5) / Float(depth.height)
                    )
                }

                sampled.append((worldPoint, color, error))
                x += step
            }
            y += step
        }

        guard !sampled.isEmpty else { return }

        lock.lock()
        for (position, color, error) in sampled {
            insertLocked(position: position, color: color, error: error)
        }
        enforceCapLocked()
        lock.unlock()
    }

    // MARK: - Draining

    /// Every accumulated point, ready for `points3D.txt`.
    ///
    /// Sorted by voxel key so two runs over the same input produce the same
    /// file, which is what makes a diff between two scans meaningful.
    func makePoints() -> [Point] {
        lock.lock()
        defer { lock.unlock() }
        return bins.keys.sorted().compactMap { key in
            guard let bin = bins[key], bin.count > 0 else { return nil }
            let n = Double(bin.count)
            let position = bin.positionSum / n
            let color = bin.colorSum / n
            return Point(
                position: SIMD3<Float>(
                    Float(position.x),
                    Float(position.y),
                    Float(position.z)
                ),
                color: SIMD3<UInt8>(
                    UInt8(Swift.max(0, Swift.min(255, color.x))),
                    UInt8(Swift.max(0, Swift.min(255, color.y))),
                    UInt8(Swift.max(0, Swift.min(255, color.z)))
                ),
                errorMeters: bin.bestError.isFinite ? bin.bestError : 0.05
            )
        }
    }

    /// World-space extent of everything measured, for `CaptureBundle.sceneBounds`
    /// and the trainer's budget. nil when nothing was measured.
    func bounds() -> BoundingBox? {
        lock.lock()
        defer { lock.unlock() }
        guard !bins.isEmpty else { return nil }
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for bin in bins.values where bin.count > 0 {
            let n = Double(bin.count)
            let p = bin.positionSum / n
            let point = SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z))
            minimum = simd_min(minimum, point)
            maximum = simd_max(maximum, point)
        }
        return BoundingBox(min: Vector3(minimum), max: Vector3(maximum))
    }

    func reset() {
        lock.lock()
        bins.removeAll(keepingCapacity: false)
        voxelSize = CaptureTuning.pointCloudVoxelSizeMeters
        didDegrade = false
        lock.unlock()
    }

    // MARK: - Private

    private func insertLocked(
        position: SIMD3<Float>,
        color: SIMD3<UInt8>,
        error: Float
    ) {
        let key = Self.key(for: position, voxelSize: voxelSize)
        var bin = bins[key] ?? Bin()
        bin.positionSum += SIMD3<Double>(
            Double(position.x),
            Double(position.y),
            Double(position.z)
        )
        bin.colorSum += SIMD3<Double>(
            Double(color.x),
            Double(color.y),
            Double(color.z)
        )
        bin.count &+= 1
        bin.bestError = Swift.min(bin.bestError, error)
        bins[key] = bin
    }

    /// Coarsens rather than grows past the cap. Halving the resolution divides
    /// the voxel count by up to eight, so one pass is normally enough; the loop
    /// is there because a pathological scan could need two.
    private func enforceCapLocked() {
        while bins.count > CaptureTuning.pointCloudMaxPoints {
            let previous = voxelSize
            voxelSize *= 2
            didDegrade = true
            var rebinned: [Int64: Bin] = [:]
            rebinned.reserveCapacity(bins.count / 4)
            for bin in bins.values where bin.count > 0 {
                let n = Double(bin.count)
                let mean = bin.positionSum / n
                let point = SIMD3<Float>(
                    Float(mean.x),
                    Float(mean.y),
                    Float(mean.z)
                )
                let key = Self.key(for: point, voxelSize: voxelSize)
                var merged = rebinned[key] ?? Bin()
                merged.positionSum += bin.positionSum
                merged.colorSum += bin.colorSum
                merged.count &+= bin.count
                merged.bestError = Swift.min(merged.bestError, bin.bestError)
                rebinned[key] = merged
            }
            bins = rebinned
            let message = "Point cloud hit its cap; voxel size coarsened from "
                    + "\(String(format: "%.3f", previous)) m to "
                    + "\(String(format: "%.3f", self.voxelSize)) m. The "
                    + "written cloud is lower resolution than 1 cm and the "
                    + "bundle records that."
            CaptureLog.writer.notice("\(message, privacy: .public)")
        }
    }

    /// Three 21-bit signed voxel coordinates packed into an `Int64`. At 1 cm
    /// that covers +-10 km, which is not a limit any scan will meet.
    @inline(__always)
    private static func key(for position: SIMD3<Float>, voxelSize: Float) -> Int64 {
        let size = Swift.max(voxelSize, 0.0001)
        let ix = Int64((position.x / size).rounded(.down))
        let iy = Int64((position.y / size).rounded(.down))
        let iz = Int64((position.z / size).rounded(.down))
        let mask: Int64 = 0x1F_FFFF
        return ((ix & mask) << 42) | ((iy & mask) << 21) | (iz & mask)
    }
}
