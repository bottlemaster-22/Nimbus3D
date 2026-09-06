//
//  PrePassICP.swift
//  PrePass
//
//  POINT-TO-PLANE ICP ON THE NATIVE DEPTH MAPS (F1, the revisit measurement).
//
//  Why point-to-plane and not point-to-point: a LiDAR sweep of a room is
//  mostly flat walls and floor. Point-to-point ICP on a flat wall is free to
//  slide the two clouds around inside the plane and will happily report
//  convergence on an alignment that is wrong by 20 cm laterally. Point-to-
//  plane costs only the component of the error ALONG the surface normal,
//  which is the only component the geometry actually constrains, and
//  converges in a handful of iterations instead of dozens.
//
//  Why projective data association and not a KD-tree: both clouds come from
//  the same sensor on the same 256x192 grid. Transforming a point from A into
//  B's camera frame and projecting it onto B's grid IS the nearest-neighbour
//  query, in O(1), with no tree to build. The cost is that association is only
//  valid where the two views overlap - which is exactly the condition a
//  revisit already satisfies.
//
//  Everything here is Float. ICP runs on ~49k samples per frame over hundreds
//  of candidate pairs; the alignment it produces is good to millimetres and
//  Float carries seven digits. The pose GRAPH that consumes these
//  measurements is Double, because that is where hundreds of them accumulate.
//

import Foundation
import simd

// MARK: - One frame, unprojected

/// A native depth frame turned into camera-space points with normals, ready
/// for ICP. Built once per frame and reused across every candidate pair that
/// frame takes part in - a revisit-rich scan matches some frames a dozen
/// times, and unprojecting 49k samples twelve times over is pure waste.
struct PrePassFramePoints: Sendable {
    let width: Int
    let height: Int
    /// Camera-space point per native pixel. Meaningless where `valid` is false.
    let points: [SIMD3<Float>]
    /// Unit surface normal in camera space, from the local depth gradient.
    let normals: [SIMD3<Float>]
    /// True where the sample had a return, was inside range, and had enough
    /// valid neighbours for a normal.
    let valid: [Bool]
    /// Z depth in metres, 0 where invalid. Kept separately because the
    /// projective association looks depth up by pixel and does not need the
    /// full point.
    let depth: [Float]
    let validCount: Int

    /// Builds points and normals from a depth frame.
    ///
    /// - Parameter maxRangeMeters: samples beyond this are dropped. Not
    ///   because they are wrong, but because their error grows fast enough
    ///   (F6's physics prior) that including them in an alignment biases it.
    /// - Parameter minConfidence: ARKit confidence floor, as a RANKING - see
    ///   F6. `0` accepts everything, which is the right default for ICP
    ///   because the alignment's own inlier test is a better filter than
    ///   ARKit's miscalibrated flag.
    static func build(
        depthFrame: PrePassDepthFrame,
        geometry: PrePassDepthGeometry,
        maxRangeMeters: Float,
        minConfidence: UInt8 = 0
    ) -> PrePassFramePoints {
        let w = geometry.width
        let h = geometry.height
        let n = w * h

        var points = [SIMD3<Float>](repeating: .zero, count: n)
        var depth = [Float](repeating: 0, count: n)
        var valid = [Bool](repeating: false, count: n)

        for i in 0..<n {
            guard depthFrame.hasReturn(at: i) else { continue }
            let z = depthFrame.depthMeters(at: i)
            guard z > 0.05, z <= maxRangeMeters else { continue }
            guard depthFrame.confidenceLevel(at: i) >= minConfidence else { continue }
            depth[i] = z
            points[i] = geometry.cameraPoint(index: i, depthMeters: z)
            valid[i] = true
        }

        // Normals from the 4-neighbourhood, central differences in camera
        // space. A neighbour whose depth jumps by more than `depthStep` is on
        // the other side of a discontinuity, and taking a cross product across
        // one produces a normal pointing at nothing. Those samples get no
        // normal and drop out of the alignment - which is correct: a depth
        // edge is precisely where the native map is least trustworthy (F3).
        //
        // The neighbour test READS `depthValid`, a frozen snapshot of what the
        // unprojection above decided, and WRITES its answer into a separate
        // `hasNormal`. Those must not be the same array. In raster order
        // `valid[i - 1]` and `valid[i - w]` would already have been overwritten
        // by this very loop, so a single dead sample killed the whole rest of
        // its row, and that row then killed the same columns in every row
        // below it. The surviving set collapsed into a left-anchored staircase:
        // on a 256x192 frame with 20% of samples missing, 2 points survived
        // instead of ~15,100, which is two orders of magnitude below the 300
        // `PrePassICP.Settings.minInliers` needs before it will even start.
        let depthValid = valid
        var hasNormal = [Bool](repeating: false, count: n)
        var normals = [SIMD3<Float>](repeating: .zero, count: n)
        var validCount = 0
        for y in 1..<Swift.max(h - 1, 1) {
            let row = y * w
            for x in 1..<Swift.max(w - 1, 1) {
                let i = row + x
                guard depthValid[i] else { continue }
                let z = depth[i]
                // Scale the allowed neighbour jump with range: at 4 m the
                // sample spacing itself is ~2 cm, so a fixed 2 cm threshold
                // would reject an ordinary slanted floor.
                let depthStep = Swift.max(0.02, 0.04 * z)

                let left = i - 1, right = i + 1
                let up = i - w, down = i + w
                // `depthValid`, not `valid`. This is the whole point of the
                // snapshot taken above, and reading the live array here was the
                // bug the comment describes: `valid[left]` and `valid[up]` have
                // already been rewritten by this same loop, so one dead sample
                // cascaded along its row and then down every row beneath it.
                guard depthValid[left], depthValid[right],
                      depthValid[up], depthValid[down],
                      abs(depth[left] - z) < depthStep,
                      abs(depth[right] - z) < depthStep,
                      abs(depth[up] - z) < depthStep,
                      abs(depth[down] - z) < depthStep
                else {
                    continue
                }

                let dx = points[right] - points[left]
                let dy = points[down] - points[up]
                let cross = simd_cross(dx, dy)
                let length = simd_length(cross)
                guard length > 1e-9 else { continue }
                var normal = cross / length
                // Orient towards the camera: the camera sits at the origin of
                // this frame, so a surface we can see has n . (-p) > 0.
                if simd_dot(normal, points[i]) > 0 { normal = -normal }
                normals[i] = normal
                hasNormal[i] = true
                validCount += 1
            }
        }

        // A sample is usable only if it got a normal, so that IS the validity
        // this function returns. Writing it as a separate array and swapping it
        // in at the end (rather than mutating `valid` inside the loop) is what
        // makes the aliasing bug above structurally impossible rather than
        // merely fixed: there is no longer a live array to read by mistake.
        //
        // Border samples fall out for free, because the loops above run
        // 1..<h-1 and 1..<w-1 and can never set `hasNormal` on an edge.
        valid = hasNormal

        return PrePassFramePoints(
            width: w,
            height: h,
            points: points,
            normals: normals,
            valid: valid,
            depth: depth,
            validCount: validCount
        )
    }

    /// A stride-subsampled list of valid sample indices, for the source side
    /// of an alignment. 49k correspondences per iteration is far more than
    /// point-to-plane needs; ~4k gives the same answer for an eighth of the
    /// work, and the pre-pass budget is measured in seconds.
    func sourceIndices(targetCount: Int) -> [Int] {
        var indices: [Int] = []
        guard validCount > 0 else { return indices }
        let stride = Swift.max(1, validCount / Swift.max(targetCount, 1))
        indices.reserveCapacity(Swift.min(validCount, targetCount) + 8)
        var seen = 0
        for i in 0..<valid.count where valid[i] {
            if seen % stride == 0 { indices.append(i) }
            seen += 1
        }
        return indices
    }

    /// Mean camera-space point of the valid samples: the "what is this frame
    /// looking at" centroid the revisit gate uses.
    var centroid: SIMD3<Float> {
        var sum = SIMD3<Float>.zero
        var count = 0
        for i in 0..<valid.count where valid[i] {
            sum += points[i]
            count += 1
        }
        guard count > 0 else { return .zero }
        return sum / Float(count)
    }
}

// MARK: - The alignment itself

/// What an ICP run produced. `converged == false` means the caller must NOT
/// use `relativePose`: it is the last iterate, not an answer.
struct PrePassICPResult {
    /// `X_camB = relativePose.act(X_camA)`. The same convention as
    /// `PrePassRigid.relative(from:to:)`, and the convention
    /// `RevisitPair.measuredRelativePose` is written in.
    var relativePose: Pose
    var inlierCount: Int
    /// RMS point-to-plane residual over the inliers, metres.
    var rmsMeters: Float
    /// Fraction of source samples that found an inlier correspondence.
    var inlierFraction: Float
    var converged: Bool
}

enum PrePassICP {

    /// Tunables, gathered so the numbers are visible in one place rather than
    /// scattered through the loop as literals.
    struct Settings {
        var iterations = 20
        /// Correspondence rejection distance at the FIRST iteration, metres.
        /// Annealed down to `finalRejectionMeters` over the run, which is what
        /// lets ICP start from a 20 cm VIO error and still finish tight.
        var initialRejectionMeters: Float = 0.30
        var finalRejectionMeters: Float = 0.04
        /// Correspondences whose normals disagree by more than this are not
        /// the same surface, whatever the distance says.
        var maxNormalAngleDegrees: Float = 45
        /// Huber threshold on the point-to-plane residual, metres.
        var huberDeltaMeters: Double = 0.03
        var sourceSampleCount = 4_000
        /// Below this many inliers the result is noise, not an alignment.
        var minInliers = 300
        /// Below this fraction of the sampled source, the two frames are not
        /// really looking at the same thing.
        var minInlierFraction: Float = 0.25
        /// Converged when the update's translation and rotation both fall
        /// under these in one step.
        var translationToleranceMeters: Double = 1e-4
        var rotationToleranceDegrees: Double = 5e-3

        init() {}
    }

    /// Aligns `source` (frame A) onto `target` (frame B).
    ///
    /// - Parameter initial: starting guess for `X_camB = T.act(X_camA)`,
    ///   normally the relative pose the raw VIO track implies.
    static func align(
        source: PrePassFramePoints,
        target: PrePassFramePoints,
        geometry: PrePassDepthGeometry,
        initial: Pose,
        settings: Settings = Settings()
    ) -> PrePassICPResult {
        let sourceIndices = source.sourceIndices(targetCount: settings.sourceSampleCount)
        guard sourceIndices.count >= settings.minInliers, target.validCount >= settings.minInliers else {
            return PrePassICPResult(
                relativePose: initial,
                inlierCount: 0,
                rmsMeters: .greatestFiniteMagnitude,
                inlierFraction: 0,
                converged: false
            )
        }

        var transform = PrePassSE3(initial)
        let cosNormalLimit = Foundation.cos(Double(settings.maxNormalAngleDegrees) * .pi / 180)

        var bestInliers = 0
        var bestRMS = Float.greatestFiniteMagnitude
        var converged = false

        for iteration in 0..<settings.iterations {
            // Anneal the rejection distance geometrically. Geometric rather
            // than linear because the first iteration has to cover a VIO-sized
            // error and the last has to resolve a millimetre one; a linear
            // ramp spends most of its iterations at distances that are already
            // far too loose to sharpen anything.
            let t = Double(iteration) / Double(Swift.max(settings.iterations - 1, 1))
            let ratio = Double(settings.finalRejectionMeters / settings.initialRejectionMeters)
            let rejection = Float(Double(settings.initialRejectionMeters) * Foundation.pow(ratio, t))

            var h = [Double](repeating: 0, count: 36)
            var g = [Double](repeating: 0, count: 6)
            var inliers = 0
            var squaredSum: Double = 0

            for index in sourceIndices {
                let pointA = source.points[index]
                let normalA = source.normals[index]

                let pointInB = SIMD3<Float>(transform.act(SIMD3<Double>(pointA)))
                guard let pixel = geometry.project(cameraPoint: pointInB) else { continue }
                let x = Int(pixel.x), y = Int(pixel.y)
                let targetIndex = y * target.width + x
                guard targetIndex >= 0, targetIndex < target.valid.count,
                      target.valid[targetIndex] else { continue }

                let pointB = target.points[targetIndex]
                let normalB = target.normals[targetIndex]

                let difference = pointInB - pointB
                let distance = simd_length(difference)
                guard distance < rejection else { continue }

                // Same surface? Compare the source normal, rotated into B.
                let normalAInB = SIMD3<Float>(transform.rotate(SIMD3<Double>(normalA)))
                guard Double(simd_dot(normalAInB, normalB)) > cosNormalLimit else { continue }

                // Point-to-plane residual: only the component along B's normal.
                let residual = Double(simd_dot(difference, normalB))

                // Jacobian of the residual with respect to a LEFT increment
                // exp(xi) applied to `transform`, xi = (omega, v):
                //   d/dxi  n . (exp(xi) T p - q)  =  [ (T p) x n , n ]
                let p = SIMD3<Double>(pointInB)
                let n = SIMD3<Double>(normalB)
                let rotationPart = simd_cross(p, n)
                let jacobian = [
                    rotationPart.x, rotationPart.y, rotationPart.z,
                    n.x, n.y, n.z
                ]

                let weight = PrePassStats.huberWeight(
                    residual: residual, delta: settings.huberDeltaMeters
                )

                for r in 0..<6 {
                    let jr = jacobian[r] * weight
                    g[r] += jr * residual
                    for c in r..<6 {
                        h[r * 6 + c] += jr * jacobian[c]
                    }
                }

                inliers += 1
                squaredSum += residual * residual
            }

            // Mirror the upper triangle.
            for r in 0..<6 {
                for c in 0..<r { h[r * 6 + c] = h[c * 6 + r] }
            }

            let fraction = Float(inliers) / Float(sourceIndices.count)
            guard inliers >= settings.minInliers else { break }

            bestInliers = inliers
            bestRMS = Float((squaredSum / Double(inliers)).squareRoot())

            guard let step = PrePassDenseSolver.solveDamped(h: h, g: g, n: 6, lambda: 1e-6) else {
                break
            }

            let omega = SIMD3<Double>(step[0], step[1], step[2])
            let v = SIMD3<Double>(step[3], step[4], step[5])
            // A single ICP step should never be metres or radians. When the
            // normal equations are that badly conditioned the honest move is
            // to stop, not to leap.
            guard simd_length(omega) < 0.5, simd_length(v) < 1.0 else { break }

            let increment = PrePassSE3.exp(omega: omega, v: v)
            // `a.then(b)` is "apply a, then b", i.e. the matrix product b * a,
            // so this is the LEFT update  T <- exp(xi) * T  that the Jacobian
            // above was derived for.
            transform = transform.then(increment)

            if simd_length(v) < settings.translationToleranceMeters,
               increment.rotationAngleDegrees < settings.rotationToleranceDegrees {
                converged = fraction >= settings.minInlierFraction
                break
            }
            converged = fraction >= settings.minInlierFraction
        }

        let fraction = Float(bestInliers) / Float(Swift.max(sourceIndices.count, 1))
        let accepted = converged
            && bestInliers >= settings.minInliers
            && fraction >= settings.minInlierFraction
            && bestRMS.isFinite

        return PrePassICPResult(
            relativePose: transform.pose,
            inlierCount: bestInliers,
            rmsMeters: bestRMS,
            inlierFraction: fraction,
            converged: accepted
        )
    }
}
