//
//  PrePassMath.swift
//  PrePass
//
//  THE NUMERIC KERNEL THE WHOLE PRE-PASS SITS ON.
//
//  Nothing in here knows what a scan is. It is Lie-group algebra for SE(3),
//  Morton (Z-order) keys for the sparse voxel grids, a small dense symmetric
//  solver for the pose graph's normal equations, a Jacobi eigen-decomposition
//  for 3x3 covariance matrices (PCA normals), and the handful of robust
//  statistics the QC card needs.
//
//  Everything here is deliberately DOUBLE precision. The pose graph
//  accumulates over hundreds of edges and the whole point of F1 is recovering
//  sub-pixel accuracy: Float has ~7 decimal digits, and a 20 m house scan
//  measured to 1 mm needs 5 of them before the optimiser has done anything.
//  Conversion to `Pose` (Float) happens once, at the boundary.
//
//  NAMING: every top-level type in this module is prefixed `PrePass` because
//  the app is one Xcode target with no namespacing (see CONTRACTS.md section
//  2) and a duplicate top-level name is a hard compile error. `SE3`, `Morton`
//  and `Plane` are exactly the names another module would reach for.
//

import Foundation
import simd

// MARK: - SE(3)

/// Rigid transforms on the Lie group SE(3), in double precision.
///
/// The convention throughout: a transform `T` maps a point `p` to `T.act(p)`,
/// and composition `a.then(b)` means "apply `a`, then apply `b`", i.e. the
/// matrix product `b.matrix * a.matrix`. Increments are applied on the LEFT
/// (`T <- exp(xi) * T`), which is the convention the Jacobians below assume.
struct PrePassSE3 {
    /// Rotation, as a unit quaternion.
    var rotation: simd_quatd
    /// Translation, metres.
    var translation: SIMD3<Double>

    static let identity = PrePassSE3(
        rotation: simd_quatd(ix: 0, iy: 0, iz: 0, r: 1),
        translation: .zero
    )

    init(rotation: simd_quatd, translation: SIMD3<Double>) {
        self.rotation = rotation
        self.translation = translation
    }

    /// Bridges a contract `Pose` (world -> camera, Float) into double precision.
    init(_ pose: Pose) {
        let q = pose.rotation.simd
        self.rotation = simd_quatd(
            ix: Double(q.vector.x),
            iy: Double(q.vector.y),
            iz: Double(q.vector.z),
            r: Double(q.vector.w)
        ).normalized
        let t = pose.translation
        self.translation = SIMD3<Double>(Double(t.x), Double(t.y), Double(t.z))
    }

    /// Back to the contract type. Normalises on the way out so a long chain of
    /// compositions cannot hand the rest of the app a non-unit quaternion.
    var pose: Pose {
        let q = rotation.normalized
        return Pose(
            rotation: Quaternion(
                x: Float(q.vector.x),
                y: Float(q.vector.y),
                z: Float(q.vector.z),
                w: Float(q.vector.w)
            ),
            translation: Vector3(
                Float(translation.x),
                Float(translation.y),
                Float(translation.z)
            )
        )
    }

    @inline(__always)
    func act(_ p: SIMD3<Double>) -> SIMD3<Double> {
        rotation.act(p) + translation
    }

    /// Rotation only - for directions and normals, which must not translate.
    @inline(__always)
    func rotate(_ v: SIMD3<Double>) -> SIMD3<Double> {
        rotation.act(v)
    }

    var inverse: PrePassSE3 {
        let ri = rotation.inverse
        return PrePassSE3(rotation: ri, translation: -ri.act(translation))
    }

    /// `self` then `other`: `other.act(self.act(p))`.
    func then(_ other: PrePassSE3) -> PrePassSE3 {
        PrePassSE3(
            rotation: (other.rotation * rotation).normalized,
            translation: other.rotation.act(translation) + other.translation
        )
    }

    /// Matrix product `self * other` in the usual left-to-right reading:
    /// applies `other` first. Same thing as `other.then(self)`, spelled the
    /// way the pose-graph residual derivation reads on paper.
    static func * (lhs: PrePassSE3, rhs: PrePassSE3) -> PrePassSE3 {
        rhs.then(lhs)
    }

    var matrix: simd_double4x4 {
        var m = simd_double4x4(rotation.normalized)
        m.columns.3 = SIMD4<Double>(translation, 1)
        return m
    }

    /// Exponential map from a 6-vector `xi = (omega, v)`: `omega` is the
    /// rotation vector (axis * angle, radians), `v` the translation part of
    /// the twist. Uses the closed form, falling back to the Taylor expansion
    /// near `theta = 0` where the closed form divides by zero.
    static func exp(omega: SIMD3<Double>, v: SIMD3<Double>) -> PrePassSE3 {
        let theta2 = simd_length_squared(omega)
        let theta = theta2.squareRoot()

        let skew = PrePassSE3.skew(omega)
        let rotation: simd_quatd
        // V is the left Jacobian of SO(3); t = V * v.
        let vMatrix: simd_double3x3

        if theta < 1e-8 {
            // Taylor: R ~ I + [w], V ~ I + [w]/2.
            rotation = simd_quatd(ix: omega.x / 2, iy: omega.y / 2, iz: omega.z / 2, r: 1).normalized
            vMatrix = matrix_identity_double3x3 + skew * 0.5
        } else {
            let axis = omega / theta
            rotation = simd_quatd(angle: theta, axis: axis)
            let a = (1 - Foundation.cos(theta)) / theta2
            let b = (theta - Foundation.sin(theta)) / (theta2 * theta)
            vMatrix = matrix_identity_double3x3 + skew * a + (skew * skew) * b
        }

        return PrePassSE3(rotation: rotation, translation: vMatrix * v)
    }

    /// Convenience: `xi[0..2]` rotation, `xi[3..5]` translation.
    static func exp(_ xi: [Double]) -> PrePassSE3 {
        precondition(xi.count == 6, "an se(3) twist has exactly 6 components")
        return exp(
            omega: SIMD3<Double>(xi[0], xi[1], xi[2]),
            v: SIMD3<Double>(xi[3], xi[4], xi[5])
        )
    }

    /// Logarithm map. Returns `(omega, v)` such that `exp(omega, v) == self`.
    /// The 6-vector this produces is the residual the pose graph minimises.
    func log() -> (omega: SIMD3<Double>, v: SIMD3<Double>) {
        let q = rotation.normalized
        // Angle from the quaternion directly - more stable near 0 and near pi
        // than going through the trace of R.
        let vecNorm = simd_length(q.vector4Imaginary)
        let w = q.vector.w
        var theta: Double
        var axis: SIMD3<Double>

        if vecNorm < 1e-12 {
            theta = 0
            axis = SIMD3<Double>(0, 0, 0)
        } else {
            // atan2 keeps full precision across the whole range, unlike acos(w).
            theta = 2 * Foundation.atan2(vecNorm, Swift.abs(w))
            axis = q.vector4Imaginary / vecNorm
            if w < 0 { axis = -axis }   // shortest arc
        }

        let omega = axis * theta

        var vInverse: simd_double3x3
        if theta < 1e-8 {
            let skew = PrePassSE3.skew(omega)
            vInverse = matrix_identity_double3x3 - skew * 0.5
        } else {
            let skew = PrePassSE3.skew(omega)
            let halfTheta = theta / 2
            // 1/theta^2 - (1 + cos t) / (2 t sin t), the standard V^-1 coefficient,
            // written as (1 - (t/2) cot(t/2)) / t^2 which is numerically calmer.
            let cot = Foundation.cos(halfTheta) / Foundation.sin(halfTheta)
            let coefficient = (1 - halfTheta * cot) / (theta * theta)
            vInverse = matrix_identity_double3x3 - skew * 0.5 + (skew * skew) * coefficient
        }

        return (omega, vInverse * translation)
    }

    /// Log as a flat 6-vector, rotation first.
    func logVector() -> [Double] {
        let (omega, v) = log()
        return [omega.x, omega.y, omega.z, v.x, v.y, v.z]
    }

    /// Rotation angle in degrees, for residual reporting.
    var rotationAngleDegrees: Double {
        let q = rotation.normalized
        let vecNorm = simd_length(q.vector4Imaginary)
        guard vecNorm > 1e-12 else { return 0 }
        let theta = 2 * Foundation.atan2(vecNorm, Swift.abs(q.vector.w))
        return theta * 180 / .pi
    }

    @inline(__always)
    static func skew(_ v: SIMD3<Double>) -> simd_double3x3 {
        // simd columns, not rows: column 0 is (0, v.z, -v.y).
        simd_double3x3(
            SIMD3<Double>(0, v.z, -v.y),
            SIMD3<Double>(-v.z, 0, v.x),
            SIMD3<Double>(v.y, -v.x, 0)
        )
    }
}

private extension simd_quatd {
    /// The imaginary part as a 3-vector. `simd_quatd.imag` exists but spelling
    /// it out keeps this file readable next to the float paths.
    var vector4Imaginary: SIMD3<Double> {
        SIMD3<Double>(vector.x, vector.y, vector.z)
    }
}

// MARK: - Float-side rigid helpers
//
// The per-sample geometry (unprojection, carving, normals) stays in Float:
// there are tens of millions of samples and Double would double the memory
// traffic for accuracy nobody can measure at 5 cm voxels.

enum PrePassRigid {
    /// World point from a camera-frame point under a world -> camera pose:
    /// `X_world = R^T (X_cam - t)`.
    @inline(__always)
    static func worldPoint(cameraPoint: SIMD3<Float>, pose: Pose) -> SIMD3<Float> {
        pose.rotation.simd.inverse.act(cameraPoint - pose.translation.simd)
    }

    /// Camera-frame point from a world point: `X_cam = R X_world + t`.
    @inline(__always)
    static func cameraPoint(worldPoint: SIMD3<Float>, pose: Pose) -> SIMD3<Float> {
        pose.rotation.simd.act(worldPoint) + pose.translation.simd
    }

    /// World-space direction from a camera-frame direction (rotation only).
    @inline(__always)
    static func worldDirection(cameraDirection: SIMD3<Float>, pose: Pose) -> SIMD3<Float> {
        pose.rotation.simd.inverse.act(cameraDirection)
    }

    @inline(__always)
    static func center(_ pose: Pose) -> SIMD3<Float> {
        pose.center.simd
    }

    /// Composition of two contract poses: apply `first`, then `second`.
    static func compose(_ first: Pose, then second: Pose) -> Pose {
        let a = PrePassSE3(first)
        let b = PrePassSE3(second)
        return a.then(b).pose
    }

    static func inverse(_ pose: Pose) -> Pose {
        PrePassSE3(pose).inverse.pose
    }

    /// `B relative to A`: the transform taking a point in A's camera frame to
    /// B's camera frame, given both poses are world -> camera.
    /// `T_ab = T_b * T_a^-1`.
    static func relative(from a: Pose, to b: Pose) -> Pose {
        let ta = PrePassSE3(a)
        let tb = PrePassSE3(b)
        return (tb * ta.inverse).pose
    }
}

// MARK: - Morton keys

/// Z-order curve keys for the sparse voxel grids (`occupancy.bin`,
/// `trust_bias.bin`). 21 bits per axis packed into a `UInt64`, which is
/// 2,097,152 cells per axis - 104 km at 5 cm voxels, so the ceiling is not a
/// practical limit, but it IS enforced rather than silently wrapping.
enum PrePassMorton {
    /// Largest coordinate representable, per axis.
    static let axisLimit: UInt32 = (1 << 21) - 1

    /// Spreads the low 21 bits of `value` out so each occupies every third
    /// bit. The classic magic-number bit twiddle; each step doubles the gap.
    @inline(__always)
    static func spread(_ value: UInt32) -> UInt64 {
        var x = UInt64(value & axisLimit)
        x = (x | (x << 32)) & 0x1F00000000FFFF
        x = (x | (x << 16)) & 0x1F0000FF0000FF
        x = (x | (x << 8)) & 0x100F00F00F00F00F
        x = (x | (x << 4)) & 0x10C30C30C30C30C3
        x = (x | (x << 2)) & 0x1249249249249249
        return x
    }

    /// Inverse of `spread`: collects every third bit back down.
    @inline(__always)
    static func compact(_ value: UInt64) -> UInt32 {
        var x = value & 0x1249249249249249
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3
        x = (x | (x >> 4)) & 0x100F00F00F00F00F
        x = (x | (x >> 8)) & 0x1F0000FF0000FF
        x = (x | (x >> 16)) & 0x1F00000000FFFF
        x = (x | (x >> 32)) & 0x1FFFFF
        return UInt32(truncatingIfNeeded: x)
    }

    @inline(__always)
    static func key(x: UInt32, y: UInt32, z: UInt32) -> UInt64 {
        spread(x) | (spread(y) << 1) | (spread(z) << 2)
    }

    @inline(__always)
    static func decode(_ key: UInt64) -> (x: UInt32, y: UInt32, z: UInt32) {
        (compact(key), compact(key >> 1), compact(key >> 2))
    }

    /// Signed integer cell coordinates -> key, with the grid's own origin
    /// already subtracted by the caller. Returns nil when a coordinate is
    /// outside the representable range, which the caller must treat as
    /// "outside the grid", never as cell zero.
    @inline(__always)
    static func key(cell: SIMD3<Int32>) -> UInt64? {
        guard cell.x >= 0, cell.y >= 0, cell.z >= 0,
              cell.x <= Int32(axisLimit), cell.y <= Int32(axisLimit), cell.z <= Int32(axisLimit)
        else { return nil }
        return key(x: UInt32(cell.x), y: UInt32(cell.y), z: UInt32(cell.z))
    }
}

// MARK: - Dense symmetric solver

/// Cholesky (LL^T) solve for the small dense normal equations the pose graph
/// and ICP produce. `n` is 6 for ICP and `6 * submapCount` for the pose graph,
/// so a few hundred at the very worst; a dense factorisation is the right
/// call and a sparse one would be more code for no gain at this size.
enum PrePassDenseSolver {

    /// Solves `(H + lambda * diag(H)) x = -g` in place-free form.
    ///
    /// - Parameters:
    ///   - h: row-major `n x n` symmetric positive semi-definite matrix.
    ///   - g: length-`n` gradient.
    ///   - lambda: Levenberg damping, scaling the diagonal.
    /// - Returns: the step `x`, or nil when the damped matrix is still not
    ///   positive definite - which the caller answers by raising `lambda`,
    ///   never by pretending a step happened.
    static func solveDamped(h: [Double], g: [Double], n: Int, lambda: Double) -> [Double]? {
        precondition(h.count == n * n && g.count == n, "solver dimensions disagree")
        guard n > 0 else { return [] }

        var a = h
        for i in 0..<n {
            let d = a[i * n + i]
            // Damp relative to the diagonal, with an absolute floor so a
            // completely unconstrained parameter still gets a finite step.
            a[i * n + i] = d + lambda * Swift.max(d, 1e-9)
        }

        guard let l = cholesky(a, n: n) else { return nil }

        // Forward substitution: L y = -g
        var y = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = -g[i]
            for k in 0..<i {
                sum -= l[i * n + k] * y[k]
            }
            let diagonal = l[i * n + i]
            guard diagonal > 0, diagonal.isFinite else { return nil }
            y[i] = sum / diagonal
        }

        // Back substitution: L^T x = y
        var x = [Double](repeating: 0, count: n)
        var i = n - 1
        while i >= 0 {
            var sum = y[i]
            var k = i + 1
            while k < n {
                sum -= l[k * n + i] * x[k]
                k += 1
            }
            x[i] = sum / l[i * n + i]
            i -= 1
        }

        for value in x where !value.isFinite { return nil }
        return x
    }

    /// Lower-triangular Cholesky factor, or nil if `a` is not positive
    /// definite. Row-major, upper triangle of the result left as zeros.
    private static func cholesky(_ a: [Double], n: Int) -> [Double]? {
        var l = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                var sum = a[i * n + j]
                for k in 0..<j {
                    sum -= l[i * n + k] * l[j * n + k]
                }
                if i == j {
                    guard sum > 0, sum.isFinite else { return nil }
                    l[i * n + j] = sum.squareRoot()
                } else {
                    let diagonal = l[j * n + j]
                    guard diagonal > 0 else { return nil }
                    l[i * n + j] = sum / diagonal
                }
            }
        }
        return l
    }
}

// MARK: - 3x3 symmetric eigen-decomposition

/// Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix, used for PCA
/// normals and for the linear/planar/spherical shape test that decides which
/// initial Gaussians are exempt from the disc prior (F4).
///
/// Jacobi rather than the analytic closed form on purpose: the closed form
/// loses most of its digits on the near-degenerate covariances you get from a
/// flat wall patch, which is exactly the case that matters here.
enum PrePassEigen {
    /// - Returns: eigenvalues sorted DESCENDING and the matching eigenvectors
    ///   as the columns of `vectors`.
    static func symmetric3x3(_ input: simd_float3x3) -> (values: SIMD3<Float>, vectors: simd_float3x3) {
        // Work in double: covariance entries of a 2 cm neighbourhood are ~1e-4
        // and their differences are what the decomposition is about.
        var a = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for c in 0..<3 {
            for r in 0..<3 {
                a[r][c] = Double(input[c][r])
            }
        }
        // Force exact symmetry; a covariance built by accumulation can be off
        // by an ulp and Jacobi assumes symmetry.
        for r in 0..<3 {
            for c in (r + 1)..<3 {
                let m = 0.5 * (a[r][c] + a[c][r])
                a[r][c] = m
                a[c][r] = m
            }
        }

        var v = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for i in 0..<3 { v[i][i] = 1 }

        for _ in 0..<24 {
            // Largest off-diagonal magnitude.
            var p = 0, q = 1
            var maxOff = Swift.abs(a[0][1])
            if Swift.abs(a[0][2]) > maxOff { maxOff = Swift.abs(a[0][2]); p = 0; q = 2 }
            if Swift.abs(a[1][2]) > maxOff { maxOff = Swift.abs(a[1][2]); p = 1; q = 2 }
            if maxOff < 1e-18 { break }

            let apq = a[p][q]
            let app = a[p][p]
            let aqq = a[q][q]
            let theta = (aqq - app) / (2 * apq)
            let t: Double
            if theta >= 0 {
                t = 1 / (theta + (1 + theta * theta).squareRoot())
            } else {
                t = -1 / (-theta + (1 + theta * theta).squareRoot())
            }
            let c = 1 / (1 + t * t).squareRoot()
            let s = t * c

            // Rotate A.
            for k in 0..<3 where k != p && k != q {
                let akp = a[k][p]
                let akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]
                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]
            }
            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = 0
            a[q][p] = 0

            // Accumulate the eigenvectors.
            for k in 0..<3 {
                let vkp = v[k][p]
                let vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }

        var pairs: [(value: Double, vector: SIMD3<Double>)] = (0..<3).map { i in
            (a[i][i], SIMD3<Double>(v[0][i], v[1][i], v[2][i]))
        }
        pairs.sort { $0.value > $1.value }

        let values = SIMD3<Float>(
            Float(pairs[0].value),
            Float(pairs[1].value),
            Float(pairs[2].value)
        )
        let vectors = simd_float3x3(
            SIMD3<Float>(Float(pairs[0].vector.x), Float(pairs[0].vector.y), Float(pairs[0].vector.z)),
            SIMD3<Float>(Float(pairs[1].vector.x), Float(pairs[1].vector.y), Float(pairs[1].vector.z)),
            SIMD3<Float>(Float(pairs[2].vector.x), Float(pairs[2].vector.y), Float(pairs[2].vector.z))
        )
        return (values, vectors)
    }
}

// MARK: - Robust statistics

enum PrePassStats {
    /// Median of an unsorted array. Returns 0 for empty input, which every
    /// call site here treats as "no data" and guards separately.
    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        var sorted = values
        sorted.sort()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return 0.5 * (sorted[mid - 1] + sorted[mid])
    }

    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        var sorted = values
        sorted.sort()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return 0.5 * (sorted[mid - 1] + sorted[mid])
    }

    /// `p` in 0...1. Nearest-rank on a sorted copy.
    static func percentile(_ values: [Float], _ p: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        var sorted = values
        sorted.sort()
        // ARGUMENT ORDER IS LOAD-BEARING, and this was written the wrong way
        // round. `Swift.max(p, 0)` is `0 >= p ? 0 : p`, and NaN compares false
        // against everything, so a NaN `p` came straight back out of both
        // calls and landed on the trapping `Int(...)` below. With the literal
        // written FIRST the clamp absorbs it and a NaN quantile reads as the
        // smallest value instead of killing the process. Identical result for
        // every finite input.
        let clamped = Swift.min(1, Swift.max(0, p))
        let index = Int((clamped * Float(sorted.count - 1)).rounded())
        return sorted[Swift.min(Swift.max(index, 0), sorted.count - 1)]
    }

    /// Median absolute deviation, scaled by 1.4826 so it estimates the same
    /// thing as a standard deviation for Gaussian data. This is what makes
    /// "4 sigma" an honest outlier threshold on data with real outliers in it.
    static func medianAbsoluteDeviation(_ values: [Float], median m: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        let deviations = values.map { Swift.abs($0 - m) }
        return 1.4826 * median(deviations)
    }

    /// Huber weight for a residual: 1 inside `delta`, `delta / |r|` outside.
    /// Multiplying a squared residual by this gives the Huber cost's own
    /// gradient, which is what the Gauss-Newton normal equations want.
    @inline(__always)
    static func huberWeight(residual: Double, delta: Double) -> Double {
        let absolute = Swift.abs(residual)
        guard absolute > delta, delta > 0 else { return 1 }
        return delta / absolute
    }

    /// Sub-sample-accurate minimum of three consecutive cost samples, by
    /// fitting a parabola. Returns the offset from the centre sample in
    /// units of one step, clamped to +-0.5 so a flat or noisy triple cannot
    /// throw the estimate into the neighbouring bin.
    static func parabolicMinimumOffset(previous: Double, centre: Double, next: Double) -> Double {
        let denominator = previous - 2 * centre + next
        guard Swift.abs(denominator) > 1e-12 else { return 0 }
        let offset = 0.5 * (previous - next) / denominator
        guard offset.isFinite else { return 0 }
        return Swift.min(Swift.max(offset, -0.5), 0.5)
    }
}

// MARK: - Planes

/// A world-space plane, `dot(normal, X) + offset == 0`, with the residual
/// spread of the points it was fitted to so a caller can tell a real wall from
/// four points that happened to be nearly coplanar.
struct PrePassPlane {
    var normal: SIMD3<Float>
    var offset: Float
    /// RMS point-to-plane distance of the fit, metres.
    var rmsMeters: Float
    var pointCount: Int

    @inline(__always)
    func signedDistance(to point: SIMD3<Float>) -> Float {
        simd_dot(normal, point) + offset
    }

    /// Total least squares plane through a point set: the eigenvector of the
    /// covariance with the smallest eigenvalue is the normal. Returns nil for
    /// fewer than three points or a degenerate (collinear) set.
    static func fit(_ points: [SIMD3<Float>]) -> PrePassPlane? {
        guard points.count >= 3 else { return nil }

        var centroid = SIMD3<Float>.zero
        for p in points { centroid += p }
        centroid /= Float(points.count)

        var xx: Float = 0, xy: Float = 0, xz: Float = 0
        var yy: Float = 0, yz: Float = 0, zz: Float = 0
        for p in points {
            let d = p - centroid
            xx += d.x * d.x; xy += d.x * d.y; xz += d.x * d.z
            yy += d.y * d.y; yz += d.y * d.z; zz += d.z * d.z
        }
        let inverseCount = 1 / Float(points.count)
        let covariance = simd_float3x3(
            SIMD3<Float>(xx, xy, xz) * inverseCount,
            SIMD3<Float>(xy, yy, yz) * inverseCount,
            SIMD3<Float>(xz, yz, zz) * inverseCount
        )

        let (values, vectors) = PrePassEigen.symmetric3x3(covariance)
        // Degenerate: the two largest eigenvalues must be meaningfully above
        // the smallest, or this is a line or a point, not a plane.
        guard values[0] > 1e-12 else { return nil }
        let normal = simd_normalize(vectors.columns.2)
        guard simd_length_squared(normal).isFinite else { return nil }

        let offset = -simd_dot(normal, centroid)
        var sumSquared: Float = 0
        for p in points {
            let d = simd_dot(normal, p) + offset
            sumSquared += d * d
        }
        let rms = (sumSquared * inverseCount).squareRoot()

        return PrePassPlane(
            normal: normal,
            offset: offset,
            rmsMeters: rms.isFinite ? rms : .greatestFiniteMagnitude,
            pointCount: points.count
        )
    }
}

// MARK: - Angles

enum PrePassAngle {
    /// Angle between two vectors in degrees, robust for near-parallel and
    /// near-antiparallel inputs (where `acos(dot)` loses half its digits).
    @inline(__always)
    static func degreesBetween(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let la = simd_length(a), lb = simd_length(b)
        guard la > 1e-12, lb > 1e-12 else { return 0 }
        let ua = a / la, ub = b / lb
        // atan2(|u x v|, u.v) is stable across the full 0..180 range.
        let cross = simd_length(simd_cross(ua, ub))
        let dot = simd_dot(ua, ub)
        return Foundation.atan2(cross, dot) * 180 / .pi
    }
}
