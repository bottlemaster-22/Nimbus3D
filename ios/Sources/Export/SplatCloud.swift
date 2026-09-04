//
//  SplatCloud.swift
//  Export
//
//  THE MODULE'S OWN MODEL OF A GAUSSIAN SPLAT CLOUD.
//
//  There is no Core/Contracts.swift on disk yet (the architect pass that was
//  meant to produce ios/Sources/Core/Contracts.swift and CONTRACTS.md has not
//  run). Export cannot sit idle waiting for a contract that does not exist, so
//  this file defines the minimal, self-contained splat representation Export
//  needs and documents it as a migration target:
//
//      TODO(nimbus): once Core/Contracts.swift exists, replace SplatCloud with
//      whatever shared splat type Trainer/Viewer/PrePass agree on, and make
//      this file a thin typealias + conversion shim instead of the source of
//      truth. Keep the wire-format math (PLYCodec/SPZCodec/GLTFExporter) as-is
//      - only the in-memory container should change.
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
//    - Scale: stored on a LOG scale (raw trainer parameter, exp(logScale) =
//      the actual world-space standard deviation along that axis). This is
//      the INRIA/SPZ convention. glTF's KHR_gaussian_splatting:SCALE wants
//      LINEAR values, so GLTFExporter exponentiates on the way out.
//    - Opacity: stored as a pre-sigmoid LOGIT (raw trainer parameter). PLY
//      and SPZ both want this treated with a sigmoid at write/read time;
//      glTF wants the activated linear value. Each writer applies sigmoid
//      itself; SplatCloud never stores the activated value.
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
}
