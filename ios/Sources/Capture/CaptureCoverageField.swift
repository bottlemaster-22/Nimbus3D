//
//  CaptureCoverageField.swift
//  Capture
//
//  THE THREE COVERAGE CHANNELS (F9), as a sparse world-space voxel field.
//
//  Coverage lives in the world, not on the mesh. That is the one structural
//  decision in this file and it is worth stating plainly: ARKit's scene mesh
//  is rebuilt constantly - vertices appear, move and are renumbered between
//  frames - so a coverage value stored per mesh vertex would evaporate every
//  time the mesh refined. Storing it per 10 cm world voxel and RE-SAMPLING it
//  at each vertex when the mesh is redrawn means the user's progress survives
//  the mesh churning underneath it, which is the difference between a HUD that
//  feels solid and one that flickers back to red for no reason.
//
//  The three channels, each with its own fix hint:
//
//    angles     how many distinct directions this patch has been seen from
//               -> "walk around it"
//    distance   how close to the band where the LiDAR and camera agree best
//               -> "walk towards it"
//    sharpness  the best sharpness any frame that saw it managed
//               -> "slow down"
//
//  They are kept separate rather than averaged into one score because the
//  three have three different remedies, and a single number cannot tell the
//  user which one to do.
//

import Foundation
import simd

/// One 10 cm patch of the world and what has been seen of it.
struct CaptureCoverageVoxel {
    /// One bit per direction bucket the patch has been seen from.
    var directionMask: UInt32 = 0
    /// Best (highest) distance-band score any observation achieved, 0...1.
    var bestDistanceScore: Float = 0
    /// Best sharpness any frame that saw this patch had, 0...1.
    var bestSharpness: Float = 0
    /// Times this voxel has been hit by a depth sample. Saturating.
    var observations: UInt16 = 0
    /// The ARKit mesh class most recently associated with the patch. A window
    /// cannot be walked around, so its angles channel is not held against the
    /// user.
    var surfaceClass: SurfaceClass = .none

    /// Distinct directions seen, 0...`coverageDirectionBuckets`.
    var directionCount: Int {
        Swift.min(
            directionMask.nonzeroBitCount,
            CaptureTuning.coverageDirectionBuckets
        )
    }

    /// The three channels, each 0...1.
    ///
    /// ALL THREE ARE NORMALISED AGAINST THEIR OWN TARGET, so that "1.0" means
    /// the same thing on all three: this patch has had enough of that. Angles
    /// divide by `coverageDirectionsForFull`, distance is already a 0...1 band
    /// score, and sharpness divides by `coverageSharpnessTarget` - the
    /// sharpness at which a patch is as good as it is going to get.
    ///
    /// That last division is not cosmetic. `bestSharpness` is normalised
    /// against the SESSION RUNNING MAXIMUM (see `CaptureSharpnessMeter`), so
    /// feeding it in raw made the sharpness channel ask every patch to have
    /// been seen by a frame within `coverageChannelDoneThreshold` of the
    /// sharpest frame in the entire scan. That is a relative test against a
    /// moving reference, not the absolute "sharp enough" the tuning constant
    /// describes, and it is much harsher: a patch measured at exactly the
    /// documented target of 0.55 scored 0.55, below the 0.7 done threshold,
    /// and could never be counted as covered no matter how long the user
    /// stood there.
    var channels: SIMD3<Float> {
        let angles = Swift.min(
            1,
            Float(directionCount) / Float(CaptureTuning.coverageDirectionsForFull)
        )
        let sharpnessTarget = Swift.max(CaptureTuning.coverageSharpnessTarget, 0.0001)
        let sharpness = Swift.min(1, Swift.max(0, bestSharpness / sharpnessTarget))
        return SIMD3<Float>(
            surfaceClass.isOpticallyUnreliable ? Swift.max(angles, 0.8) : angles,
            bestDistanceScore,
            sharpness
        )
    }

    /// True once every channel is at or above the done threshold.
    var isSatisfied: Bool {
        let c = channels
        let t = CaptureTuning.coverageChannelDoneThreshold
        return c.x >= t && c.y >= t && c.z >= t
    }
}

/// Which channel is the user's problem right now.
enum CaptureCoverageChannel: Int, CaseIterable {
    case angles = 0
    case distance = 1
    case sharpness = 2

    /// Plain-language, imperative, no jargon. This string is both spoken and
    /// shown, so it has to survive being read out loud.
    var fixHint: String {
        switch self {
        case .angles: return "Walk around it a bit more"
        case .distance: return "Move a little closer"
        case .sharpness: return "Give this bit a steadier look"
        }
    }
}

/// The sparse coverage field.
///
/// Thread-safe. Written from the capture pipeline's queue, read in bulk by the
/// Metal renderer when it rebuilds a mesh chunk's coverage attribute. Bulk
/// reads take the lock once for the whole batch rather than once per vertex.
final class CaptureCoverageField: @unchecked Sendable {

    private let lock = NSLock()
    private var voxels: [Int64: CaptureCoverageVoxel] = [:]

    /// Cached so the HUD can read it every frame without walking the map.
    private var cachedFraction: Float = 0
    private var cachedWorstChannel: CaptureCoverageChannel?
    private var cachedCountedVoxels: Int = 0

    // MARK: - Keys

    /// Voxel key from a world position. Three 21-bit signed coordinates packed
    /// into an `Int64`, which covers +-104 km at 10 cm voxels - far past any
    /// plausible scan, and a plain integer hash rather than a Morton curve
    /// because nothing here needs spatial locality of the key.
    @inline(__always)
    static func key(for position: SIMD3<Float>) -> Int64 {
        let size = CaptureTuning.coverageVoxelSizeMeters
        let ix = voxelIndex(position.x, size)
        let iy = voxelIndex(position.y, size)
        let iz = voxelIndex(position.z, size)
        let mask: Int64 = 0x1F_FFFF  // 21 bits
        return ((ix & mask) << 42) | ((iy & mask) << 21) | (iz & mask)
    }

    /// One axis, converted WITHOUT the possibility of trapping.
    ///
    /// `Int64(someFloat)` is a trapping conversion: it kills the process on
    /// NaN, on infinity, and on any finite value outside `Int64`. This used to
    /// be three bare `Int64(...)` calls, and it is reached roughly half a
    /// million times a second during a capture (12,288 depth samples a pass at
    /// 8 Hz for coverage, plus the point cloud, plus window mode), so ONE bad
    /// frame anywhere in a scan was a hard crash.
    ///
    /// A bad frame is not hypothetical. `Pose.fromARKitCameraTransform` inverts
    /// the camera transform with no determinant check, so a singular transform
    /// from ARKit (which can happen while tracking is unavailable) produces NaN
    /// in every world position derived from that frame.
    ///
    /// There IS a NaN guard in `observe`, and it never got the chance to run:
    /// `ARCaptureService` passes `meshStore.surfaceClass(at: point)` as an
    /// ARGUMENT to `observe`, and Swift evaluates arguments first, so the
    /// trapping conversion happened one stack frame before the guard.
    ///
    /// Dropping the sample is the right answer rather than clamping it to a
    /// real voxel: a position we cannot trust must not be allowed to mark a
    /// real part of the room as covered. Key 0 is a sentinel bucket that
    /// `observe` rejects on distance anyway.
    @inline(__always)
    static func voxelIndex(_ value: Float, _ size: Float) -> Int64 {
        let scaled = (value / Swift.max(size, 0.0001)).rounded(.down)
        guard scaled.isFinite else { return 0 }
        // The key only keeps 21 signed bits per axis, so clamping here costs
        // nothing that the mask was not already discarding, and it closes the
        // "finite but astronomically large" case that traps just as hard.
        let limit: Float = 1_048_575  // 2^20 - 1
        return Int64(Swift.max(-limit, Swift.min(limit, scaled)))
    }

    /// Elevation bands the sphere is cut into.
    static let elevationBins: Int = 4

    /// Azimuth bins: whatever is left of `coverageDirectionBuckets` once the
    /// elevation bands have their share.
    ///
    /// Derived rather than written out as an `8` so that
    /// `CaptureTuning.coverageDirectionBuckets` is the single authority on the
    /// size of `directionMask`, instead of a constant that describes two
    /// literals living in another file and has no way of knowing when they
    /// stop agreeing with it.
    static let azimuthBins: Int = Swift.max(
        1,
        CaptureTuning.coverageDirectionBuckets / CaptureCoverageField.elevationBins
    )

    /// The highest bucket index a `UInt32` mask can hold, whatever the tuning
    /// constant is set to. 32 bits is a hard ceiling, not a preference.
    static let highestBucketIndex: Int =
        Swift.min(CaptureTuning.coverageDirectionBuckets, 32) - 1

    /// Bucket index for a viewing direction: `azimuthBins` azimuth by
    /// `elevationBins` elevation.
    ///
    /// `direction` points from the surface towards the camera and must be a
    /// unit vector. 32 buckets over the sphere is roughly 20 degrees of
    /// angular resolution, which is about the granularity at which "walk
    /// around it a bit more" is useful advice rather than pedantry.
    @inline(__always)
    static func directionBucket(_ direction: SIMD3<Float>) -> Int {
        let azimuthCount = CaptureCoverageField.azimuthBins
        let elevationCount = CaptureCoverageField.elevationBins
        let azimuth = atan2(direction.z, direction.x)  // -pi ... pi
        let azimuthTurn: Float = (azimuth + .pi) / (2 * .pi)
        let azimuthBin = Int((azimuthTurn * Float(azimuthCount)).rounded(.down))
            .clampedToRange(0...(azimuthCount - 1))
        let elevation = Swift.max(-1, Swift.min(1, direction.y))  // -1 ... 1
        let elevationTurn: Float = (elevation + 1) / 2
        let elevationBin = Int((elevationTurn * Float(elevationCount)).rounded(.down))
            .clampedToRange(0...(elevationCount - 1))
        let bucket = elevationBin * azimuthCount + azimuthBin
        return Swift.min(bucket, CaptureCoverageField.highestBucketIndex)
    }

    // MARK: - Update

    /// One observation of one patch.
    ///
    /// - Parameters:
    ///   - worldPosition: the LiDAR sample's position in world space.
    ///   - cameraCenter: where the camera was.
    ///   - sharpness: the observing frame's normalised sharpness, 0...1.
    ///   - surfaceClass: ARKit's class for the patch, when known.
    func observe(
        worldPosition: SIMD3<Float>,
        cameraCenter: SIMD3<Float>,
        sharpness: Float,
        surfaceClass: SurfaceClass
    ) {
        let toCamera = cameraCenter - worldPosition
        let distance = simd_length(toCamera)
        guard distance > 0.05,
            distance <= CaptureTuning.coverageMaxUsefulDistanceMeters
        else { return }
        let direction = toCamera / distance

        let key = Self.key(for: worldPosition)
        let bucket = Self.directionBucket(direction)
        let distanceScore = Self.distanceScore(distance)

        lock.lock()
        var voxel = voxels[key] ?? CaptureCoverageVoxel()
        voxel.directionMask |= (1 << UInt32(bucket))
        voxel.bestDistanceScore = Swift.max(voxel.bestDistanceScore, distanceScore)
        voxel.bestSharpness = Swift.max(voxel.bestSharpness, sharpness)
        if voxel.observations < UInt16.max { voxel.observations += 1 }
        if surfaceClass != .none { voxel.surfaceClass = surfaceClass }
        voxels[key] = voxel
        lock.unlock()
    }

    /// How well a viewing distance sits in the band where the LiDAR footprint
    /// and the camera's detail agree. 1 inside the band, tapering to 0 at the
    /// near/mid boundary and at very close range.
    @inline(__always)
    static func distanceScore(_ distance: Float) -> Float {
        let near = CaptureTuning.coverageIdealDistanceMinMeters
        let far = CaptureTuning.coverageIdealDistanceMaxMeters
        let limit = CaptureTuning.coverageMaxUsefulDistanceMeters
        if distance >= near && distance <= far { return 1 }
        if distance < near {
            // Too close: the depth map's 256x192 footprint is now coarser than
            // the detail in front of it.
            return Swift.max(0, distance / Swift.max(near, 0.0001))
        }
        return Swift.max(0, 1 - (distance - far) / Swift.max(limit - far, 0.0001))
    }

    // MARK: - Read

    /// The three channels at a world position, for the mesh shader's per-vertex
    /// attribute. `nil` for a patch nothing has been recorded about, which the
    /// renderer draws as "not seen yet" rather than as "seen and bad".
    func channels(at worldPosition: SIMD3<Float>) -> SIMD3<Float>? {
        lock.lock()
        defer { lock.unlock() }
        return voxels[Self.key(for: worldPosition)]?.channels
    }

    /// Bulk sample, one lock acquisition for a whole mesh chunk.
    ///
    /// - Parameter positions: world positions, one per mesh vertex.
    /// - Returns: `(channels, seen)` per position, in the same order.
    func sampleBatch(
        positions: UnsafeBufferPointer<SIMD3<Float>>,
        into output: UnsafeMutableBufferPointer<SIMD4<Float>>
    ) {
        lock.lock()
        defer { lock.unlock() }
        let count = Swift.min(positions.count, output.count)
        for index in 0..<count {
            if let voxel = voxels[Self.key(for: positions[index])] {
                let channels = voxel.channels
                output[index] = SIMD4<Float>(channels.x, channels.y, channels.z, 1)
            } else {
                output[index] = SIMD4<Float>(0, 0, 0, 0)
            }
        }
    }

    /// The "done" number: the fraction of observed patches that satisfy all
    /// three channels.
    ///
    /// Patches with fewer than `coverageMinObservationsToCount` hits are
    /// excluded from BOTH the numerator and the denominator. A single grazing
    /// return off a distant surface should not create a permanent red patch
    /// the user has no way to clear.
    func recomputeFraction() {
        lock.lock()
        var counted = 0
        var satisfied = 0
        var deficits = SIMD3<Float>(repeating: 0)
        let threshold = CaptureTuning.coverageChannelDoneThreshold
        for voxel in voxels.values
        where Int(voxel.observations) >= CaptureTuning.coverageMinObservationsToCount {
            counted += 1
            let channels = voxel.channels
            if voxel.isSatisfied {
                satisfied += 1
            } else {
                deficits += simd_max(
                    SIMD3<Float>(repeating: threshold) - channels,
                    SIMD3<Float>(repeating: 0)
                )
            }
        }
        cachedCountedVoxels = counted
        cachedFraction = counted > 0 ? Float(satisfied) / Float(counted) : 0
        cachedWorstChannel = Self.worstChannel(of: deficits)
        lock.unlock()
    }

    /// 0...1, the "done" number the capture screen shows and the done
    /// criterion compares against `CaptureTuning.coverageDoneFraction`.
    var fraction: Float {
        lock.lock()
        defer { lock.unlock() }
        return cachedFraction
    }

    /// The channel with the largest total deficit across unsatisfied patches:
    /// the single most useful thing to tell the user to do right now.
    var worstChannel: CaptureCoverageChannel? {
        lock.lock()
        defer { lock.unlock() }
        return cachedWorstChannel
    }

    /// Patches currently counting towards the percentage. Shown in the HUD's
    /// detail view so a user who wonders why 84% is not moving can see that
    /// the denominator is still growing as they walk into a new room.
    var countedVoxelCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedCountedVoxels
    }

    var isDone: Bool { fraction >= CaptureTuning.coverageDoneFraction }

    func reset() {
        lock.lock()
        voxels.removeAll(keepingCapacity: false)
        cachedFraction = 0
        cachedWorstChannel = nil
        cachedCountedVoxels = 0
        lock.unlock()
    }

    private static func worstChannel(of deficits: SIMD3<Float>) -> CaptureCoverageChannel? {
        let maximum = Swift.max(deficits.x, Swift.max(deficits.y, deficits.z))
        guard maximum > 0 else { return nil }
        if deficits.x == maximum { return .angles }
        if deficits.y == maximum { return .distance }
        return .sharpness
    }
}

// MARK: - Small helper

extension Int {
    /// Clamp without pulling in a whole numerics dependency.
    @inline(__always)
    func clampedToRange(_ range: ClosedRange<Int>) -> Int {
        Swift.min(range.upperBound, Swift.max(range.lowerBound, self))
    }
}
