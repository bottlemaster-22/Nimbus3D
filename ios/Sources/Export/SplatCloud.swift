//
//  SplatCloud.swift
//  Export
//
//  THE MODULE'S OWN MODEL OF A GAUSSIAN SPLAT CLOUD.
//
//  Written before ios/Sources/Core/Contracts.swift existed, when Export could
//  not sit idle waiting for a contract that had not landed yet, so this file
//  defined the minimal, self-contained splat representation Export needs.
//
//  UPDATE: Core/Contracts.swift and CONTRACTS.md now exist. Per CONTRACTS.md
//  section 6.1, `SplatCloud`, `SHDegree`, `ExportError` and `ExportFormat` are
//  PROMOTED to contract types in place rather than re-declared in Core (a
//  duplicate top-level type name in this single-target project is a hard
//  compile error), so this file stays the source of truth exactly as-is.
//  Core adds one conformance this file must never also declare:
//  `extension SHDegree: Codable {}` lives in Contracts.swift - see that file's
//  comment above it before touching SHDegree here.
//
//      Optional, no-risk later tidying (CONTRACTS.md 6.1): move this file
//      from Sources/Export to Sources/Core unchanged - a file move with no
//      content or call-site change, since there is no `import` to update in
//      a single target. Do it when this module is not mid-flight, or never;
//      it costs nothing to leave as-is.
//
//  Storage convention (deliberate, documented so every writer/reader agrees):
//
//    - Coordinate frame: RUB (Right, Up, Back) - X right, Y up, Z toward the
//      viewer / away from where the camera initially looked. This is BOTH
//      ARKit's world-space convention (world Y vertical, per docs/DATA_FORMAT
//      as described in the product spec) AND the default/native frame of the
//      SPZ format AND glTF's coordinate convention. So SplatCloud values are
//      stored exactly as the trainer produces them, with zero coordinate
//      conversion needed for SPZ or GLB export. Only PLY (RDF: Right, Down,
//      Front - the INRIA reference-implementation convention) needs a flip,
//      applied inside PLYCodec.
//    - Rotation: unit quaternion in (x, y, z, w) order - "the usual glTF
//      order" per KHR_gaussian_splatting, and the order SPZ's GaussianCloud
//      uses internally. Not required to arrive pre-normalized; every writer
//      normalizes on the way out.
//    - Scale: stored on a LOG scale (exp(logScale) = the world-space standard
//      deviation along that axis). This is the INRIA/SPZ convention. glTF's
//      KHR_gaussian_splatting:SCALE wants LINEAR values, so GLTFExporter
//      exponentiates on the way out.
//    - Opacity: stored as a pre-sigmoid LOGIT. PLY and SPZ both want this
//      treated with a sigmoid at write/read time; glTF wants the activated
//      linear value. Each writer applies sigmoid itself; SplatCloud never
//      stores the activated value.
//    - Scale and opacity are DRAW-READY, not raw optimiser parameters.
//      The trainer fits every Gaussian through a Mip-Splatting 3D low-pass
//      filter of per-Gaussian width `filter3D` metres: it renders with the
//      covariance widened to `Sigma + filter3D^2 * I`, and with the opacity
//      multiplied by `sqrt(det(Sigma) / det(Sigma + filter3D^2 * I))` so the
//      widening does not also brighten the Gaussian. `filter3D` lives only in
//      the trainer's per-Gaussian stats buffer; no splat file format on earth
//      has a field for it. So it is FUSED into `logScales` and
//      `opacityLogits` once, by `fuse3DFilter(_:)` below, before a cloud
//      leaves the trainer. Every reader after that - this app's viewer, the
//      Blender add-on, SuperSplat, anything - draws the model the trainer
//      actually fitted with no extra field and no extra code.
//      `filter3DFused` records whether that has happened.
//    - Color DC / spherical harmonics: stored as raw, unnormalized SH
//      coefficients exactly as the trainer holds them (the INRIA convention:
//      display color = 0.5 + 0.282095 * dc, evaluated by each downstream
//      consumer, never baked into storage). `shRest[i]` has
//      `SplatCloud.restCoefficientCount(forDegree:)` entries, each an (r,g,b)
//      triple, ordered by ascending (degree, order) exactly as SPZ and
//      KHR_gaussian_splatting both number them: degree 1 gives 3 coefficients
//      (m = -1, 0, +1), degree 2 gives the next 5, degree 3 the next 7.
//

import Foundation
import simd

/// Degree of spherical harmonics carried by a `SplatCloud`. Bounded at 3
/// because that is the ceiling of the ratified `KHR_gaussian_splatting` glTF
/// extension; SPZ's own format allows degree 4 but nothing in this product
/// spec (F10: "SH degree 1 default, 0-2 configurable") asks for it, so degree
/// 4 is out of scope rather than silently mishandled.
public enum SHDegree: Int, Sendable, CaseIterable {
    case zero = 0
    case one = 1
    case two = 2
    case three = 3

    /// Number of *rest* (non-DC) coefficients for this degree, i.e. how many
    /// entries `SplatCloud.shRest[i]` must have. The DC term is stored
    /// separately in `colorDC` and is never counted here.
    public var restCoefficientCount: Int {
        switch self {
        case .zero: return 0
        case .one: return 3
        case .two: return 8
        case .three: return 15
        }
    }
}

/// Errors raised across the Export module's writers, readers and packaging
/// helpers. One error type for the whole module keeps call sites simple; the
/// `reason` string is always plain-language enough to show a non-technical
/// user in a "couldn't export" alert.
public enum ExportError: Error, CustomStringConvertible, Sendable {
    case inconsistentAttributeCounts(String)
    case unsupportedSHDegree(Int)
    case emptyCloud
    case malformedFile(String)
    case unsupportedFormat(String)
    case ioFailure(String)
    case archiveLimitExceeded(String)
    case sourceDirectoryMissing(String)
    /// `fuse3DFilter(_:)` was called on a cloud that had already been fused.
    /// Fusing twice would widen and dim every Gaussian a second time, which
    /// is exactly the kind of quiet, plausible-looking damage this project
    /// keeps finding, so it is an error rather than a shrug.
    case alreadyFused(String)

    public var description: String {
        switch self {
        case .inconsistentAttributeCounts(let s): return "Splat data is inconsistent: \(s)"
        case .unsupportedSHDegree(let d): return "Unsupported spherical-harmonics degree: \(d)"
        case .emptyCloud: return "There are no splats to export."
        case .malformedFile(let s): return "The file could not be read: \(s)"
        case .unsupportedFormat(let s): return "Unsupported file format: \(s)"
        case .ioFailure(let s): return "A file could not be written or read: \(s)"
        case .archiveLimitExceeded(let s): return "The scan is too large to package: \(s)"
        case .sourceDirectoryMissing(let s): return "Expected scan data is missing: \(s)"
        case .alreadyFused(let s):
            return "This model has already been prepared for viewing: \(s)"
        }
    }
}

/// A cloud of 3D Gaussian splats, in the storage convention documented above.
/// This is what `ExportService` writes to `.ply` / `.spz` / `.glb` and what
/// import-back produces from a file on disk.
public struct SplatCloud: Sendable {
    public var shDegree: SHDegree

    /// World-space (RUB) splat centers, in meters.
    public var positions: [SIMD3<Float>]
    /// Unit-ish quaternions, (x, y, z, w). Normalized by every writer; not
    /// required to be normalized on input.
    public var rotations: [SIMD4<Float>]
    /// Per-axis log-scale. `exp(logScales[i])` is the world-space standard
    /// deviation along the corresponding (rotated) axis, in meters.
    public var logScales: [SIMD3<Float>]
    /// Pre-sigmoid opacity logit. `sigmoid(opacityLogits[i])` is the alpha
    /// used at render time, in [0, 1].
    public var opacityLogits: [Float]
    /// Raw (unnormalized) degree-0 SH coefficient, per point, as (r, g, b).
    public var colorDC: [SIMD3<Float>]
    /// Raw higher-order SH coefficients. `shRest[i].count ==
    /// shDegree.restCoefficientCount` for every `i`, or `shRest` is empty
    /// when `shDegree == .zero`.
    public var shRest: [[SIMD3<Float>]]

    /// Whether the Mip-Splatting 3D low-pass filter has already been folded
    /// into `logScales` and `opacityLogits` by `fuse3DFilter(_:)`.
    ///
    ///  * `true`  - fused. This cloud draws the way the trainer fitted it.
    ///  * `false` - NOT fused, and whoever built it KNEW the filter widths.
    ///              Every Gaussian in it will draw sharper and more opaque
    ///              than the trainer's own render.
    ///  * `nil`   - unknown, which is what a cloud parsed out of a `.ply`,
    ///              `.spz` or `.glb` gets. None of those formats carries a
    ///              marker, and answering `true` here would be a guess
    ///              dressed up as a fact.
    ///
    /// Deliberately NOT a parameter of `init`: adding one would change every
    /// construction site in the app for a value only the trainer can supply.
    ///
    /// No `= nil` on purpose. An Optional stored property defaults to nil by
    /// language rule, so the explicit initialiser below never has to assign it
    /// and every existing construction site still compiles. Writing `= nil`
    /// here would instead trip SwiftLint's `redundant_optional_initialization`.
    public var filter3DFused: Bool?

    public var count: Int { positions.count }

    public init(
        shDegree: SHDegree,
        positions: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        logScales: [SIMD3<Float>],
        opacityLogits: [Float],
        colorDC: [SIMD3<Float>],
        shRest: [[SIMD3<Float>]]
    ) throws {
        let n = positions.count
        guard rotations.count == n,
              logScales.count == n,
              opacityLogits.count == n,
              colorDC.count == n,
              shRest.count == n || (shDegree == .zero && shRest.isEmpty)
        else {
            throw ExportError.inconsistentAttributeCounts(
                "positions=\(n) rotations=\(rotations.count) logScales=\(logScales.count) "
                    + "opacityLogits=\(opacityLogits.count) colorDC=\(colorDC.count) "
                    + "shRest=\(shRest.count)"
            )
        }
        if shDegree != .zero {
            let expected = shDegree.restCoefficientCount
            for (i, coeffs) in shRest.enumerated() where coeffs.count != expected {
                throw ExportError.inconsistentAttributeCounts(
                    "point \(i) has \(coeffs.count) SH rest coefficients, expected \(expected) "
                        + "for degree \(shDegree.rawValue)"
                )
            }
        }
        self.shDegree = shDegree
        self.positions = positions
        self.rotations = rotations
        self.logScales = logScales
        self.opacityLogits = opacityLogits
        self.colorDC = colorDC
        self.shRest = shRest
    }

    /// Folds the trainer's Mip-Splatting 3D low-pass filter into this cloud's
    /// stored scale and opacity, once and for good.
    ///
    /// WHY THIS EXISTS. The trainer does not render the Gaussians it stores.
    /// It renders each one widened to `Sigma + f^2 * I` and dimmed by
    /// `sqrt(det(Sigma) / det(Sigma + f^2 * I))`, where `f` is that
    /// Gaussian's own `filter3D` in metres. Every opacity it fitted was fitted
    /// against that dimming. `f` lives only in the trainer's GPU stats buffer,
    /// and `.ply`, `.spz` and `.glb` have nowhere to put it, so a cloud that
    /// leaves the trainer without this call is a DIFFERENT MODEL from the one
    /// that was trained: sharper, and too opaque, in the direction that makes
    /// a good run look like noise.
    ///
    /// The fuse is exact for rendering and one-way for training. After it,
    /// `exp(logScales[i])` is the widened standard deviation and
    /// `sigmoid(opacityLogits[i])` is the dimmed alpha, so an ordinary
    /// unmodified viewer draws the trainer's render. The raw parameters
    /// cannot be recovered, which is fine here because nothing in this app
    /// resumes optimisation from a `SplatCloud`; the trainer seeds from
    /// `PrePassResult.initialSplats`, never from one of these.
    ///
    /// A Gaussian whose filter is zero, negative or non-finite is left exactly
    /// as it is, so a producer that has no filter to report ("no compensation")
    /// is a no-op and never a crash.
    ///
    /// - Parameter filter3D: one filter standard deviation in world metres per
    ///   splat, in this cloud's own index order.
    /// - Returns: how many splats the fuse actually changed. Zero from a
    ///   non-empty cloud means every filter was zero, which is worth saying
    ///   out loud rather than reporting as a successful fuse.
    @discardableResult
    public mutating func fuse3DFilter(_ filter3D: [Float]) throws -> Int {
        if filter3DFused == true {
            throw ExportError.alreadyFused(
                "the 3D low-pass filter is already folded into its sizes and opacities, "
                    + "and folding it in twice would blur and dim every splat a second time"
            )
        }
        guard filter3D.count == count else {
            throw ExportError.inconsistentAttributeCounts(
                "filter3D=\(filter3D.count) but the cloud holds \(count) splats"
            )
        }

        var changed = 0
        for index in 0..<count {
            let width = filter3D[index]
            guard width.isFinite, width > 0 else { continue }
            let before = (logScales[index], opacityLogits[index])
            let after = SplatMath.fusing3DFilter(
                logScale: before.0,
                opacityLogit: before.1,
                filter3D: width
            )
            if after.logScale != before.0 || after.opacityLogit != before.1 {
                changed += 1
            }
            logScales[index] = after.logScale
            opacityLogits[index] = after.opacityLogit
        }

        filter3DFused = true
        return changed
    }

    /// The honest `filter3DFused` for a cloud assembled out of several others.
    ///
    /// A merge builds a fresh `SplatCloud` from concatenated arrays, and a
    /// fresh cloud starts at `nil`, so without this every merged preview and
    /// every multi-part model would report "cannot say" about a fact its own
    /// parts knew. The rule is the cautious one: `true` only when EVERY part
    /// says true, `false` as soon as one part says false (one unfused part
    /// makes the whole model unfaithful), and `nil` only when the parts
    /// genuinely disagree in the "unknown" direction.
    ///
    /// An empty list is `nil`, not `true`: nothing was merged, so nothing is
    /// known.
    static func mergedFilter3DFused(_ parts: [Bool?]) -> Bool? {
        guard !parts.isEmpty else { return nil }
        if parts.contains(where: { $0 == false }) { return false }
        if parts.contains(where: { $0 == nil }) { return nil }
        return true
    }

    /// An empty cloud at a given SH degree - useful as a builder starting
    /// point or for round-trip tests.
    public static func empty(shDegree: SHDegree = .zero) -> SplatCloud {
        // swiftlint:disable:next force_try - all-empty arrays always satisfy the length check.
        try! SplatCloud(
            shDegree: shDegree,
            positions: [], rotations: [], logScales: [],
            opacityLogits: [], colorDC: [], shRest: []
        )
    }
}

// MARK: - Shared numeric helpers

/// Math shared by every wire-format codec in this module. Kept in one place
/// so the sigmoid/SH conventions can never drift between PLY, SPZ and GLB.
enum SplatMath {
    /// Scale applied to the raw degree-0 SH coefficient to turn it into a
    /// color. Matches the INRIA 3D Gaussian Splatting reference and the SPZ
    /// format exactly (`Y_0,0 = 0.5 * sqrt(1/pi) ~= 0.282095`).
    static let shDCToColor: Float = 0.282095_017

    @inline(__always)
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    /// Inverse sigmoid (logit), clamped away from 0/1 so a fully-opaque or
    /// fully-transparent splat never produces +-infinity. This is a
    /// deliberate robustness improvement over naive `log(x/(1-x))`: the
    /// reference SPZ decoder produces `inf` at the byte-255 boundary, which
    /// is a landmine for any downstream trainer that resumes from an
    /// imported cloud.
    @inline(__always)
    static func invSigmoid(_ x: Float) -> Float {
        let clamped = min(max(x, 1e-6), 1 - 1e-6)
        return logf(clamped / (1 - clamped))
    }

    /// Rounds and clamps to [0, 255]. Guards NaN/infinity explicitly first:
    /// `UInt8(_: Float)` traps on a non-finite input, and every quantized
    /// byte stream in this module (SPZ's alpha/color/scale/SH bytes) funnels
    /// through here, so one NaN splat from an upstream trainer bug must not
    /// be able to crash export.
    @inline(__always)
    static func clampToUInt8(_ x: Float) -> UInt8 {
        guard x.isFinite else { return x > 0 ? 255 : 0 }
        return UInt8(max(0, min(255, x.rounded())))
    }

    /// Replaces a non-finite value with 0, for callers about to feed a Float
    /// into a trapping fixed-width integer conversion (`Int32(_:)`, etc.).
    @inline(__always)
    static func finiteOrZero(_ x: Float) -> Float {
        x.isFinite ? x : 0
    }

    // MARK: - Mip-Splatting 3D low-pass filter

    /// The window a log-scale is clamped to before exponentiating. These are
    /// the SAME two numbers `trainer_preprocess` uses
    /// (`exp(clamp(logScale, -12, 3))`), written here so the fuse below
    /// reproduces the trainer's own sigma rather than a slightly different
    /// one. exp(-12) is 6.1 micrometres; exp(3) is 20 metres.
    static let logScaleClampLow: Float = -12
    static let logScaleClampHigh: Float = 3

    /// Mip-Splatting's 3D opacity compensation for one Gaussian:
    /// `sqrt(det(Sigma) / det(Sigma + f^2 * I))`.
    ///
    /// In the Gaussian's own frame Sigma is diagonal, so the determinant ratio
    /// collapses to a product of `s / sqrt(s^2 + f^2)` over the three axes and
    /// no 3x3 determinant is needed. The result is always in (0, 1]: the
    /// filter can only ever spread a Gaussian out, and this is the factor that
    /// takes the added brightness straight back off.
    ///
    /// Worth knowing how sharply it bites, because it is not a small
    /// correction on the splats that matter. With all three axes at `s`:
    /// `s = 5f` gives 0.94, `s = 2f` gives 0.72, `s = f` gives 0.35,
    /// `s = f/4` gives 0.014. A Gaussian much finer than its own filter is
    /// meant to nearly vanish, and does.
    ///
    /// - Parameter sigma: LINEAR per-axis standard deviations, world metres.
    /// - Parameter filter3D: filter standard deviation, world metres.
    @inline(__always)
    static func filter3DCompensation(sigma: SIMD3<Float>, filter3D: Float) -> Float {
        guard filter3D.isFinite, filter3D > 0 else { return 1 }
        let fSquared: Float = filter3D * filter3D
        var product: Float = 1
        for axis in 0..<3 {
            let s: Float = sigma[axis]
            guard s.isFinite, s > 0 else { return 1 }
            let widened: Float = sqrtf(s * s + fSquared)
            guard widened.isFinite, widened > 0 else { return 1 }
            product *= s / widened
        }
        return product
    }

    /// One Gaussian's half of `SplatCloud.fuse3DFilter(_:)`: the widened
    /// log-scale and the compensated opacity logit, in the storage convention
    /// this file documents.
    ///
    /// Returns its input untouched for any filter or parameter it cannot use,
    /// so "no filter" always means "no compensation" and never a NaN.
    static func fusing3DFilter(
        logScale: SIMD3<Float>,
        opacityLogit: Float,
        filter3D: Float
    ) -> (logScale: SIMD3<Float>, opacityLogit: Float) {
        let unchanged = (logScale: logScale, opacityLogit: opacityLogit)
        guard filter3D.isFinite, filter3D > 0, opacityLogit.isFinite else { return unchanged }

        var sigma = SIMD3<Float>(repeating: 0)
        for axis in 0..<3 {
            let raw: Float = logScale[axis]
            guard raw.isFinite else { return unchanged }
            let clamped: Float = min(max(raw, logScaleClampLow), logScaleClampHigh)
            sigma[axis] = expf(clamped)
        }

        let compensation: Float = filter3DCompensation(sigma: sigma, filter3D: filter3D)
        guard compensation.isFinite, compensation > 0, compensation < 1 else { return unchanged }

        let fSquared: Float = filter3D * filter3D
        var widenedLogScale = SIMD3<Float>(repeating: 0)
        for axis in 0..<3 {
            let s: Float = sigma[axis]
            let widened: Float = sqrtf(s * s + fSquared)
            guard widened.isFinite, widened > 0 else { return unchanged }
            widenedLogScale[axis] = logf(widened)
        }

        let alpha: Float = sigmoid(opacityLogit) * compensation
        let fusedLogit: Float = invSigmoid(alpha)
        guard fusedLogit.isFinite else { return unchanged }

        return (logScale: widenedLogScale, opacityLogit: fusedLogit)
    }
}
