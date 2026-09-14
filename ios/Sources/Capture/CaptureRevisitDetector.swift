//
//  CaptureRevisitDetector.swift
//  Capture
//
//  REVISIT CANDIDATES, FOUND AT THE END OF THE SESSION FROM POSES ALONE (F1).
//
//  A revisit is two keyframes that look at the same surface from nearly the
//  same place at two different times. They are the constraints that let the
//  pre-pass's submap pose graph pull drift out of a walk, and they are the
//  reason a scan that ends where it started is worth more than one that does
//  not.
//
//  WHAT THIS FILE DOES AND, MORE IMPORTANTLY, WHAT IT DOES NOT.
//
//  It finds CANDIDATES: pairs whose camera centres are within 60 cm, whose
//  view directions agree within 30 degrees, and which are at least twenty
//  seconds apart in time. That is exactly `RevisitMethod.poseProximity`, which
//  Core defines as "poses are close and view directions agree; no depth
//  alignment run yet".
//
//  It does NOT measure how far apart the two views really are. Doing that
//  needs point-to-plane ICP on the two frames' native depth maps, which is the
//  pre-pass's job (`PoseRefiner.detectRevisits`) and is far too heavy to run
//  on the phone while the user is still walking. So every pair this file emits
//  carries `translationResidualMeters` and `rotationResidualDegrees` of ZERO,
//  and that zero means "not measured yet", not "no drift". Nothing downstream
//  may read a residual from a `poseProximity` pair as a measurement; the
//  method field is what says so.
//
//  Why emit them at all, then: finding the candidates is the expensive search
//  (every frame against every other frame), the poses needed to do it are
//  already in memory at the end of a capture, and handing the pre-pass a
//  shortlist of a few hundred pairs instead of a few million saves it the one
//  part of the job that scales quadratically. A bundle that never reaches the
//  pre-pass still tells a reader where the walk crossed itself.
//
//  The search is a spatial hash rather than a double loop: a four-thousand
//  frame house walk is sixteen million pose comparisons the naive way, and a
//  grid at the proximity radius turns that into a handful of buckets per
//  frame.
//

import Foundation
import simd

/// Finds pose-proximity revisit candidates in a finished capture.
enum CaptureRevisitDetector {

    /// One frame reduced to what the search needs.
    private struct Entry {
        var index: FrameID
        var timestamp: Double
        var center: SIMD3<Float>
        var forward: SIMD3<Float>
        var rotation: Quaternion
        var translation: SIMD3<Float>
    }

    /// Searches `frames` for pairs that revisit the same viewpoint.
    ///
    /// - Parameter frames: the capture's frames, in capture order.
    /// - Returns: candidate pairs, ordered by the later frame, at most
    ///   `CaptureTuning.revisitMaxPairs` of them.
    static func detect(frames: [CaptureFrame]) -> [RevisitPair] {
        guard frames.count > 2 else { return [] }

        let radius = CaptureTuning.revisitMaxCameraDistanceMeters
        let maxAngle = CaptureTuning.revisitMaxViewAngleDegrees
        let minGap = CaptureTuning.revisitMinTimeGapSeconds
        let minSpacing = CaptureTuning.revisitMinPairSpacingSeconds
        let cellSize = Swift.max(radius, 0.05)

        var entries: [Entry] = []
        entries.reserveCapacity(frames.count)
        for frame in frames {
            let pose = frame.rawPose
            entries.append(
                Entry(
                    index: frame.index,
                    timestamp: frame.timestampSeconds,
                    center: pose.center.simd,
                    forward: pose.forward.simd,
                    rotation: pose.rotation,
                    translation: pose.translation.simd
                )
            )
        }

        // Bucket of entry offsets per grid cell, filled as the scan advances so
        // a frame only ever sees frames that came before it.
        var grid: [Int64: [Int]] = [:]
        var pairs: [RevisitPair] = []
        var lastEmittedTime = -Double.greatestFiniteMagnitude

        for (offset, entry) in entries.enumerated() {
            defer { grid[cellKey(entry.center, cellSize: cellSize), default: []].append(offset) }

            guard pairs.count < CaptureTuning.revisitMaxPairs else { continue }
            guard entry.timestamp - lastEmittedTime >= minSpacing else { continue }

            var bestOffset: Int?
            var bestScore: Float = 0
            var supporters = 0

            for neighbour in neighbourKeys(entry.center, cellSize: cellSize) {
                guard let bucket = grid[neighbour] else { continue }
                for candidateOffset in bucket {
                    let candidate = entries[candidateOffset]
                    guard entry.timestamp - candidate.timestamp >= minGap else { continue }

                    let distance = simd_distance(entry.center, candidate.center)
                    guard distance <= radius else { continue }

                    let cosine = Swift.max(
                        -1,
                        Swift.min(1, simd_dot(entry.forward, candidate.forward))
                    )
                    let angle = acos(cosine) * 180 / .pi
                    guard angle <= maxAngle else { continue }

                    supporters += 1
                    // Both terms are 1 for a perfect match and fall to 0 at the
                    // limits, so the product is a confidence that says "how
                    // much like the same viewpoint is this" and nothing more.
                    let proximity = 1 - distance / Swift.max(radius, 0.0001)
                    let agreement = 1 - angle / Swift.max(maxAngle, 0.0001)
                    let score = proximity * agreement
                    if score > bestScore {
                        bestScore = score
                        bestOffset = candidateOffset
                    }
                }
            }

            guard let bestOffset, bestScore > 0 else { continue }
            let partner = entries[bestOffset]

            pairs.append(
                RevisitPair(
                    frameA: partner.index,
                    frameB: entry.index,
                    method: .poseProximity,
                    measuredRelativePose: relativePose(from: partner, to: entry),
                    // ZERO MEANS NOT MEASURED. See the note at the top of this
                    // file: nothing has been aligned, so there is no residual
                    // to report, and inventing one would put a drift number in
                    // front of the user that no instrument produced.
                    translationResidualMeters: 0,
                    rotationResidualDegrees: 0,
                    inlierCount: supporters,
                    confidence: Swift.max(0, Swift.min(1, bestScore))
                )
            )
            lastEmittedTime = entry.timestamp
        }

        CaptureLog.session.notice(
            "Found \(pairs.count, privacy: .public) revisit candidates by pose proximity."
        )
        return pairs
    }

    // MARK: - Geometry

    /// The rigid transform that takes a point from A's camera frame into B's.
    ///
    /// Both poses are world -> camera, so with `X_camA = R_a X + t_a` and
    /// `X_camB = R_b X + t_b`:
    ///
    ///     X_camB = (R_b R_a^-1) X_camA + (t_b - R_b R_a^-1 t_a)
    ///
    /// which is the constraint the pose graph consumes. Derived from the raw
    /// VIO poses, because at capture time that is the only measurement there
    /// is; the pre-pass replaces it with an ICP result.
    private static func relativePose(from a: Entry, to b: Entry) -> Pose {
        let relative = simd_normalize(b.rotation.simd * a.rotation.simd.inverse)
        let translation = b.translation - relative.act(a.translation)
        return Pose(
            rotation: Quaternion(relative),
            translation: Vector3(translation)
        )
    }

    // MARK: - Spatial hash

    /// Non-trapping, sharing the one implementation in `CaptureCoverageField`.
    ///
    /// This is the THIRD copy of the same three lines, and all three used a
    /// bare `Int64(someFloat)`, which kills the process on NaN, on infinity and
    /// on any out-of-range finite value. Revisit detection runs on camera
    /// positions, which come from the same ARKit poses that could be NaN when
    /// tracking was unavailable, so this had exactly the same trigger as the
    /// coverage-field crash and simply needed the detector to run on the bad
    /// frame first.
    @inline(__always)
    private static func cellKey(_ position: SIMD3<Float>, cellSize: Float) -> Int64 {
        key(
            x: CaptureCoverageField.voxelIndex(position.x, cellSize),
            y: CaptureCoverageField.voxelIndex(position.y, cellSize),
            z: CaptureCoverageField.voxelIndex(position.z, cellSize)
        )
    }

    @inline(__always)
    private static func key(x: Int64, y: Int64, z: Int64) -> Int64 {
        // 21 bits per axis, the same packing the coverage field uses. At the
        // proximity radius that covers hundreds of kilometres, which no walk
        // is going to reach.
        let mask: Int64 = 0x1F_FFFF
        return ((x & mask) << 42) | ((y & mask) << 21) | (z & mask)
    }

    /// The 27 cells that can hold a point within one cell edge of `position`.
    private static func neighbourKeys(
        _ position: SIMD3<Float>,
        cellSize: Float
    ) -> [Int64] {
        let cx = CaptureCoverageField.voxelIndex(position.x, cellSize)
        let cy = CaptureCoverageField.voxelIndex(position.y, cellSize)
        let cz = CaptureCoverageField.voxelIndex(position.z, cellSize)
        var keys: [Int64] = []
        keys.reserveCapacity(27)
        for dx in -1...1 {
            for dy in -1...1 {
                for dz in -1...1 {
                    keys.append(
                        key(x: cx + Int64(dx), y: cy + Int64(dy), z: cz + Int64(dz))
                    )
                }
            }
        }
        return keys
    }
}
