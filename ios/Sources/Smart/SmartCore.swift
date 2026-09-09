//
//  SmartCore.swift
//  Smart
//
//  Shared plumbing for the SMART layer: settings, small math, the binary
//  sidecar readers and writers, a minimal image loader, and the physics noise
//  model. Nothing in this file is a stub.
//
//  NAMING: the app is ONE Xcode target with no namespacing (CONTRACTS.md §2),
//  so every type this module adds is prefixed `Smart` except the three
//  concrete names CONTRACTS.md §5 reserves for it:
//  `NativeDepthEdgeClassifier`, `TwoScaleTrustField`, `DirectionalBackgroundModel`.
//

import Foundation
import simd
import os

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

// MARK: - Logging

enum SmartLog {
    static let general = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Smart")
    static let trust = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Smart.Trust")
    static let edges = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Smart.Edges")
    static let background = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Smart.Background")
    static let loss = Logger(subsystem: BrandConfig.loggingSubsystem, category: "Smart.Loss")
}

// MARK: - Settings

/// Every tunable the SMART layer has, in one struct, with the spec's numbers
/// as defaults. Passed in rather than read from globals so the Booster (which
/// has a desktop GPU and no thermal ceiling) can run the same code at a
/// different operating point.
public struct SmartLossSettings: Codable, Hashable, Sendable {

    // --- F3: native depth supervision ---------------------------------------

    /// Depth step, as a fraction of range, that makes a native-resolution edge
    /// "geometric". Relative because a 4 cm step at 0.5 m is a real edge and
    /// at 5 m it is noise.
    public var geometricEdgeRelativeStep: Float
    /// Absolute floor on that step so a near-field surface does not shatter
    /// into edges.
    public var geometricEdgeMinStepMeters: Float
    /// Normalised image-gradient magnitude above which a pixel is a strong
    /// image edge. 0...1 on the native-resolution luma.
    public var textureEdgeGradient: Float
    /// Dilation around a geometric edge, in NATIVE depth pixels. One native
    /// pixel is ~7.5 RGB pixels at 1920 wide, so 1 here is the ~8 px band the
    /// spec asks for (CONTRACTS.md §7).
    public var edgeBandRadiusNativePixels: Int

    // --- F4: bimodal edge supervision ---------------------------------------

    /// Weight on the "be one of the two modes, not the average of them" term.
    public var bimodalWeight: Float
    /// Weight on the transition-width penalty: `4t(1-t)` over the normalised
    /// position between the two modes, which is maximal exactly halfway.
    public var transitionWidthWeight: Float
    /// Half-width, in native pixels, of the window the two local depth modes
    /// are estimated from.
    public var modeWindowRadius: Int
    /// Extra weight on a GEOMETRIC edge sample. This is the "sharpen" half of
    /// the WHERE/WHAT split: a real depth step is the most informative sample
    /// in the frame and is worth more than a sample in the middle of a wall.
    public var geometricEdgeSharpenBoost: Float
    /// Weight on the "flatten" half: a penalty on the LAPLACIAN of rendered
    /// depth at TEXTURE edges. A 3DGS optimiser left alone invents a ridge at
    /// a hard colour edge, because a ridge is a cheap way to explain one. This
    /// makes it not cheap.
    public var textureFlattenWeight: Float
    /// Authority below which a sample gets no depth supervision at all and is
    /// routed to the background instead. Deliberately small: the authority
    /// ramps are already soft, and this is only the floor where the product
    /// has decayed to numerical noise.
    public var minimumAuthorityForDepth: Float

    // --- F6: trust ----------------------------------------------------------

    /// Reference depth sigma, metres. `weight = 1 / (1 + (sigma/ref)^2)`, so a
    /// sample at exactly this noise level gets weight 0.5. Soft everywhere,
    /// never a step: hard per-region switches make visible seams.
    public var noiseReferenceMeters: Float
    /// Huber transition at 1 m range, metres. Grows with depth.
    public var huberDeltaMeters: Float
    /// Floor the depth-loss schedule decays to. Not zero: depth stays a weak
    /// prior to the end so late photometric fitting cannot walk a wall away.
    public var depthScheduleFloor: Float
    /// Hard clamp on the per-frame learned sensor scale, as a deviation from 1.
    public var maxDepthScaleDeviation: Float
    /// Hard clamp on the per-frame learned sensor shift, metres.
    public var maxDepthShiftMeters: Float

    // --- F5: authority ------------------------------------------------------

    /// Below this, LiDAR is authoritative.
    public var nearRangeMeters: Float
    /// Above this, LiDAR has no authority at all. Cosine ramp in between.
    public var farRangeMeters: Float
    /// Mid-regime upper bound; beyond it only the direction-only far field can
    /// say anything.
    public var midRangeMeters: Float
    /// A patch never seen from more than this angle apart is unknowable from
    /// this capture and gets routed to infinity.
    public var parallaxMinDegrees: Float
    /// Angle at which the parallax gate is fully open.
    public var parallaxFullDegrees: Float
    /// Normalised luma at or above which a pixel counts as saturated.
    public var saturationLuma: Float

    // --- F2: free space -----------------------------------------------------

    /// Weight on the no-return / free-space lower-bound hinge.
    public var freeSpaceLowerBoundWeight: Float
    /// Safety margin subtracted from a certified free-space bound before it is
    /// used as a constraint, metres.
    public var freeSpaceBoundMarginMeters: Float
    /// Fraction of the run before free-space pruning is allowed to start.
    public var pruneStartFraction: Float
    /// Fraction of the run after which pruning stops (deleting during final
    /// convergence just leaves holes).
    public var pruneEndFraction: Float
    /// Iterations between pruning passes.
    public var pruneIntervalIterations: Int
    /// Never delete more than this fraction of the field in one pass.
    public var pruneMaxFractionPerPass: Float

    // --- F4: shape ----------------------------------------------------------

    /// Weight on the effective-rank / disc prior.
    public var discPriorWeight: Float
    /// Target effective rank for a surface splat. 2 is a disc.
    public var discTargetEffectiveRank: Float
    /// Target effective rank for a splat sitting on a detected 3D edge curve.
    /// 1 is a needle, which is what an edge actually looks like.
    public var edgeTargetEffectiveRank: Float

    // --- Build cost ---------------------------------------------------------

    /// Co-visible partner frames each frame is compared against when the trust
    /// field is built. More is better and slower.
    public var trustPartnerFrames: Int
    /// Stride over the native grid during the trust build. 2 means one sample
    /// in four is used, which is ~12k per frame instead of ~49k.
    public var trustSampleStride: Int
    /// Total budget for ZNCC plane-sweep verification across the whole scan.
    /// 0 disables it.
    public var planeSweepSampleBudget: Int
    /// Plane-sweep search half-range, metres.
    public var planeSweepRangeMeters: Float
    /// Plane-sweep steps across that range.
    public var planeSweepSteps: Int
    /// ZNCC patch half-width, native pixels.
    public var planeSweepPatchRadius: Int

    public init(
        geometricEdgeRelativeStep: Float = 0.03,
        geometricEdgeMinStepMeters: Float = 0.04,
        textureEdgeGradient: Float = 0.18,
        edgeBandRadiusNativePixels: Int = 1,
        bimodalWeight: Float = 1.0,
        transitionWidthWeight: Float = 0.35,
        modeWindowRadius: Int = 2,
        geometricEdgeSharpenBoost: Float = 2.0,
        textureFlattenWeight: Float = 0.5,
        minimumAuthorityForDepth: Float = 0.05,
        noiseReferenceMeters: Float = 0.02,
        huberDeltaMeters: Float = 0.02,
        depthScheduleFloor: Float = 0.05,
        maxDepthScaleDeviation: Float = 0.03,
        maxDepthShiftMeters: Float = 0.03,
        nearRangeMeters: Float = 4.5,
        farRangeMeters: Float = 5.5,
        midRangeMeters: Float = 30.0,
        parallaxMinDegrees: Float = 0.5,
        parallaxFullDegrees: Float = 2.0,
        saturationLuma: Float = 0.97,
        freeSpaceLowerBoundWeight: Float = 0.5,
        freeSpaceBoundMarginMeters: Float = 0.05,
        pruneStartFraction: Float = 0.15,
        pruneEndFraction: Float = 1.0,
        pruneIntervalIterations: Int = 200,
        pruneMaxFractionPerPass: Float = 0.02,
        // WAS 0.01. In trainer_regularizer the disc term produces a scale
        // gradient of order 0.024 per Gaussian per iteration, applied
        // unconditionally to every visible one. The photometric contribution
        // to the SAME accumulator, arriving through dL/dconic from the 50-150
        // pixels a splat covers, is of order 2.5e-5 to 7.5e-5. That is a
        // ratio between 300 and 1000, so Gaussian SHAPE is being set by the
        // prior and the photographs are only allowed to nudge it.
        //
        // 0.001 keeps the prior meaningful, it still exceeds the photometric
        // term, while leaving the images room to make a Gaussian the
        // anisotropic sliver a fine texture edge needs. That is exactly what
        // a 13.2 mm median splat is failing to become.
        //
        // AND A WARNING ABOUT HOW TO REASON ABOUT THIS CONSTANT AT ALL.
        //
        // The prior's dL/dlogScale still exceeds the photometric one by a
        // median 442x to 1138x even at 0.001, and it is the larger term on 95
        // to 98 per cent of scale components. That sounds decisive and is not:
        // trainer_adam_splat uses mhat / (sqrt(vhat) + epsilon) with epsilon
        // 1e-15, which is SCALE-INVARIANT. A term a thousand times larger does
        // not produce a step a thousand times larger; it produces a step of
        // the learning rate, in whatever direction it points.
        //
        // So the magnitude ratio is the wrong statistic. The decisive one is
        // that the prior flips the SIGN of the combined gradient on about 48
        // per cent of components, which is what actually stops the photographs
        // setting shape. Getting the images to win on magnitude would need
        // roughly 1e-6, three orders below this, not one.
        //
        // Do not make that change yet. Our splats are 88.7 per cent flat discs
        // against the reference model's 56.4 per cent near-isotropic blobs,
        // but with growth headroom at zero the prior is currently the ONLY
        // thing setting shape, so the two are confounded. Re-measure the
        // aspect ratio once densification has a real budget again.
        //
        // THE COST: surface-normal quality and mesh extraction quality both
        // lean on this prior, and both matter for the export format. If the
        // meshes come out worse, this is the constant that did it.
        // BACK UP TO 0.005, BECAUSE 30,000 ITERATIONS MEASURED WHAT 3,000
        // COULD NOT. At 3,000 the weakening looked free; at 30,000 the share
        // of the population that are NEEDLES (largest axis over eight times
        // the smallest) went from 23.4 per cent to 44.4, the median aspect
        // from 5.8:1 to 7.6:1, and p90 splat size from 14.02 mm to 26.09 mm.
        // A long thin Gaussian stretched to fit a handful of viewpoints is the
        // classic 3DGS overfitting artifact, and this prior is the only thing
        // in the trainer that opposes it.
        //
        // Note the effective change is smaller than the number looks. The
        // gradient used to be exactly twice the derivative of the loss it
        // reported, so the original 0.01 behaved like 0.02 and 0.001 behaved
        // like 0.001 once that was fixed. 0.005 sits a quarter of the way back
        // to the original behaviour, deliberately: the reference model's
        // median aspect is 1.9:1 against our 7.6:1, so shape does need to be
        // freer than it was, just not this free over ten times the iterations.
        // 0.001, WHICH IS WHAT BUILD 226 RAN AND 226 IS THE ONE THAT LOOKS
        // BEST SO FAR.
        //
        // The sequence, all measured against the owner's own eye on the same
        // photograph rather than against PSNR, which was blind to all of it:
        //
        //   226   weight 0.001, edge target 1.0   best structure so far
        //   240   weight 0.005, edge target 1.0   better colour, artefacting
        //                                          in cluttered regions
        //   244   weight 0.002, edge target 3.0   worse than both
        //
        // 240 says the weight matters and 0.005 is too much. 244 says the edge
        // target was not the problem. So this goes back to exactly what 226
        // ran, and everything else 240 and 244 gained - the schedule fix that
        // finally trains the spherical harmonics, sharper keyframe selection,
        // a converged pose graph, exporting the best model rather than the
        // last - is kept.
        discPriorWeight: Float = 0.001,
        discTargetEffectiveRank: Float = 2.0,
        // BACK TO 1.0. I CHANGED THIS AND IT MADE THE PICTURE WORSE.
        //
        // Build 244 flipped this from 1.0 (a needle) to 3.0 (a sphere) on the
        // theory that the edge flag fires on clutter and clutter should not be
        // slivers. The owner compared the result against builds 226 and 240 on
        // a real photograph and called it worse than both.
        //
        // AND 226 VERSUS 244 IS A CLEAN CONTROLLED COMPARISON, which is why
        // this is a revert rather than another guess. Both ran with the disc
        // prior gradient already corrected and a similar weight; the one thing
        // that changed between them is this constant. 226 looked better.
        //
        // The mechanism is obvious in hindsight. This target applies to splats
        // flagged as lying ON A 3D EDGE CURVE, and an edge genuinely IS a
        // one-dimensional structure: a table lip, a cable, a chair leg, a
        // skirting board. Asking those to become spheres blurs every crease in
        // the room. The shape census made it visible and I did not read it
        // properly: needles went from 16.2 per cent of the population to
        // exactly 0.0. A real room does not contain zero elongated things.
        //
        // The artefacting in 240 was the prior being too STRONG, not this
        // target being wrong, and the weight below is where that is fixed.
        edgeTargetEffectiveRank: Float = 1.0,
        trustPartnerFrames: Int = 4,
        trustSampleStride: Int = 2,
        planeSweepSampleBudget: Int = 20_000,
        planeSweepRangeMeters: Float = 0.15,
        planeSweepSteps: Int = 32,
        planeSweepPatchRadius: Int = 3
    ) {
        self.geometricEdgeRelativeStep = geometricEdgeRelativeStep
        self.geometricEdgeMinStepMeters = geometricEdgeMinStepMeters
        self.textureEdgeGradient = textureEdgeGradient
        self.edgeBandRadiusNativePixels = edgeBandRadiusNativePixels
        self.bimodalWeight = bimodalWeight
        self.transitionWidthWeight = transitionWidthWeight
        self.modeWindowRadius = modeWindowRadius
        self.geometricEdgeSharpenBoost = geometricEdgeSharpenBoost
        self.textureFlattenWeight = textureFlattenWeight
        self.minimumAuthorityForDepth = minimumAuthorityForDepth
        self.noiseReferenceMeters = noiseReferenceMeters
        self.huberDeltaMeters = huberDeltaMeters
        self.depthScheduleFloor = depthScheduleFloor
        self.maxDepthScaleDeviation = maxDepthScaleDeviation
        self.maxDepthShiftMeters = maxDepthShiftMeters
        self.nearRangeMeters = nearRangeMeters
        self.farRangeMeters = farRangeMeters
        self.midRangeMeters = midRangeMeters
        self.parallaxMinDegrees = parallaxMinDegrees
        self.parallaxFullDegrees = parallaxFullDegrees
        self.saturationLuma = saturationLuma
        self.freeSpaceLowerBoundWeight = freeSpaceLowerBoundWeight
        self.freeSpaceBoundMarginMeters = freeSpaceBoundMarginMeters
        self.pruneStartFraction = pruneStartFraction
        self.pruneEndFraction = pruneEndFraction
        self.pruneIntervalIterations = pruneIntervalIterations
        self.pruneMaxFractionPerPass = pruneMaxFractionPerPass
        self.discPriorWeight = discPriorWeight
        self.discTargetEffectiveRank = discTargetEffectiveRank
        self.edgeTargetEffectiveRank = edgeTargetEffectiveRank
        self.trustPartnerFrames = trustPartnerFrames
        self.trustSampleStride = trustSampleStride
        self.planeSweepSampleBudget = planeSweepSampleBudget
        self.planeSweepRangeMeters = planeSweepRangeMeters
        self.planeSweepSteps = planeSweepSteps
        self.planeSweepPatchRadius = planeSweepPatchRadius
    }

    public static let `default` = SmartLossSettings()

    /// Cheaper build for a phone that is already warm, or for a house-sized
    /// scan where the trust build would otherwise dominate the pre-pass.
    ///
    /// Selected by `trustBuildCost(requested:frameCount:thermalLevel:)`, which
    /// `TwoScaleTrustField.build` calls at the top of every trust build. It is
    /// not something a screen offers: the two things that decide it are both
    /// measurements, and asking a person how warm their phone is would be a
    /// worse answer than reading it.
    public static let economy = SmartLossSettings(
        trustPartnerFrames: 2,
        trustSampleStride: 3,
        planeSweepSampleBudget: 0
    )

    /// Frames above which a walk is treated as house-sized.
    ///
    /// The same 1200 the processing screen uses to decide a scan is big for a
    /// phone, so the two do not disagree about what "big" means.
    public static let houseSizedFrameCount = 1_200

    /// Which cost preset a trust build should actually run at.
    ///
    /// WHY THIS EXISTS: `economy` was written, documented, and then selected
    /// by nothing, so every trust build ran at full cost no matter how warm
    /// the phone was or how long the walk had been. This is the one place the
    /// choice is made, and it is made from two measurements rather than a
    /// preference: the live thermal state, and the number of frames in the
    /// capture.
    ///
    /// Only the three BUILD-COST knobs are taken from `economy`. Everything
    /// else in `requested` is kept, because the rest of this struct is
    /// calibration (edge bands, ramps, thresholds) and quietly swapping
    /// calibration for a preset's copy of it is how two runs stop being
    /// comparable.
    ///
    /// - Parameters:
    ///   - requested: what the caller asked for.
    ///   - frameCount: frames in the capture being built.
    ///   - thermalLevel: how warm the phone is right now.
    /// - Returns: the settings to build with, and, when it downshifted, the
    ///   plain-language reason it did. `nil` means full cost ran.
    public static func trustBuildCost(
        requested: SmartLossSettings,
        frameCount: Int,
        thermalLevel: ThermalLevel
    ) -> (settings: SmartLossSettings, downshiftReason: String?) {
        var reasons: [String] = []
        if thermalLevel >= .serious {
            reasons.append("the phone is already warm")
        }
        if frameCount > houseSizedFrameCount {
            reasons.append("this scan is \(frameCount) frames")
        }
        guard !reasons.isEmpty else { return (requested, nil) }

        var cheaper = requested
        cheaper.trustPartnerFrames = economy.trustPartnerFrames
        cheaper.trustSampleStride = economy.trustSampleStride
        cheaper.planeSweepSampleBudget = economy.planeSweepSampleBudget
        return (cheaper, reasons.joined(separator: " and "))
    }
}

// MARK: - Errors

/// Errors raised inside the SMART layer. Crosses the module boundary as a
/// `NimbusError` via `asNimbusError`.
public enum SmartError: LocalizedError, Sendable {
    case notPrepared(String)
    case missingSidecar(String)
    case malformedSidecar(path: String, expectedBytes: Int, actualBytes: Int)
    case frameOutOfRange(FrameID)
    case imageUnreadable(String)

    public var errorDescription: String? {
        switch self {
        case .notPrepared(let what):
            return "\(what) was used before it was prepared."
        case .missingSidecar(let path):
            return "A file this scan needs is missing: \(path)"
        case .malformedSidecar(let path, let expected, let actual):
            return "\(path) is \(actual) bytes; \(expected) were expected."
        case .frameOutOfRange(let frame):
            return "Frame \(frame) is not in this capture."
        case .imageUnreadable(let path):
            return "The photo \(path) could not be read."
        }
    }

    /// What a caller outside this module sees.
    public var asNimbusError: NimbusError {
        .malformedData(errorDescription ?? "Smart layer error")
    }
}

// MARK: - Small math

enum SmartMath {

    /// Smooth 0 -> 1 ramp. `edge0 == edge1` degrades to a step rather than a
    /// division by zero.
    @inline(__always)
    static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        if edge1 <= edge0 { return x < edge0 ? 0 : 1 }
        let t = simd_clamp((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)
    }

    /// Smooth 1 -> 0 ramp, the complement of `smoothstep`.
    @inline(__always)
    static func smoothdrop(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        1 - smoothstep(edge0, edge1, x)
    }

    @inline(__always)
    static func clamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
        simd_clamp(x, lo, hi)
    }

    /// Huber loss and its derivative in one pass.
    /// Quadratic within `delta`, linear outside, C1 at the join.
    @inline(__always)
    static func huber(_ r: Float, delta: Float) -> (value: Float, grad: Float) {
        let d = Swift.max(delta, 1e-6)
        let a = abs(r)
        if a <= d {
            return (0.5 * r * r / d, r / d)
        }
        return (a - 0.5 * d, r < 0 ? -1 : 1)
    }

    /// Finite and inside a sane metric range. Every sidecar read goes through
    /// this: one NaN in a depth map must not poison a whole frame's loss.
    @inline(__always)
    static func isUsableDepth(_ z: Float) -> Bool {
        z.isFinite && z > 0.01 && z < 1000
    }

    /// Median of a small array. Copies, so keep the arrays small.
    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        var v = values
        v.sort()
        let n = v.count
        return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])
    }

    /// Percentile 0...1 of a small array, nearest rank.
    static func percentile(_ values: [Float], _ p: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        var v = values
        v.sort()
        let idx = Int((Float(v.count - 1) * simd_clamp(p, 0, 1)).rounded())
        return v[idx]
    }

    /// The integer pixel a continuous image coordinate falls in, or nil when
    /// that coordinate is not one we are allowed to trust.
    ///
    /// `Int(someFloat)` is a TRAPPING conversion: it kills the process, in
    /// release as well as debug, on NaN, on infinity, and on any finite value
    /// outside `Int`'s range. A projected pixel is exactly the value that can
    /// be all three, because it is a pose and a division away from the sensor,
    /// and a pose read back off disk is not promised to be sane. Every
    /// projected pixel in this module goes through here so that one bad frame
    /// drops one sample instead of the app.
    @inline(__always)
    static func pixelIndex(
        _ p: SIMD2<Float>,
        width: Int,
        height: Int
    ) -> (x: Int, y: Int)? {
        guard p.x.isFinite, p.y.isFinite else { return nil }
        let fx = p.x.rounded(.down)
        let fy = p.y.rounded(.down)
        guard fx >= 0, fy >= 0, fx < Float(width), fy < Float(height) else { return nil }
        return (Int(fx), Int(fy))
    }
}

// MARK: - Morton keys

/// Z-order keys for the sparse voxel fields.
///
/// CONVENTION, and it matters: `TrustFieldRefs` carries a voxel size but no
/// origin, so a bias-field key is only meaningful against a fixed origin
/// convention. This is it: world coordinates are offset by
/// `SmartMorton.biasMeters` before quantisation, so the world origin lands on
/// cell `biasMeters / voxelSize` on every axis and negative coordinates are
/// representable. 21 bits per axis at 0.25 m covers +/- 262 km, several orders
/// of magnitude more than any scan.
///
/// `OccupancyGridRef` DOES carry an explicit origin and is written by
/// `Sources/PrePass`; this convention applies only to the trust bias field,
/// which this module writes and reads.
enum SmartMorton {

    /// Fixed world-origin offset in metres, so negatives quantise cleanly.
    static let biasMeters: Float = 1024

    static let maxCoordinate: UInt32 = (1 << 21) - 1

    @inline(__always)
    static func cell(_ p: SIMD3<Float>, voxelSize: Float) -> SIMD3<UInt32> {
        let v = Swift.max(voxelSize, 1e-4)
        let shifted = (p + SIMD3<Float>(repeating: biasMeters)) / v
        func q(_ f: Float) -> UInt32 {
            guard f.isFinite else { return 0 }
            let i = f.rounded(.down)
            if i <= 0 { return 0 }
            if i >= Float(maxCoordinate) { return maxCoordinate }
            return UInt32(i)
        }
        return SIMD3<UInt32>(q(shifted.x), q(shifted.y), q(shifted.z))
    }

    /// Centre of a cell back in world space, for debugging and for the
    /// floorplan trust layer.
    @inline(__always)
    static func center(_ c: SIMD3<UInt32>, voxelSize: Float) -> SIMD3<Float> {
        SIMD3<Float>(Float(c.x) + 0.5, Float(c.y) + 0.5, Float(c.z) + 0.5) * voxelSize
            - SIMD3<Float>(repeating: biasMeters)
    }

    @inline(__always)
    static func part1By2(_ n: UInt32) -> UInt64 {
        var x = UInt64(n) & 0x1F_FFFF                       // 21 bits
        x = (x | (x << 32)) & 0x001F_0000_0000_FFFF
        x = (x | (x << 16)) & 0x001F_0000_FF00_00FF
        x = (x | (x << 8))  & 0x100F_00F0_0F00_F00F
        x = (x | (x << 4))  & 0x10C3_0C30_C30C_30C3
        x = (x | (x << 2))  & 0x1249_2492_4924_9249
        return x
    }

    @inline(__always)
    static func encode(_ c: SIMD3<UInt32>) -> UInt64 {
        part1By2(c.x) | (part1By2(c.y) << 1) | (part1By2(c.z) << 2)
    }

    @inline(__always)
    static func key(_ p: SIMD3<Float>, voxelSize: Float) -> UInt64 {
        encode(cell(p, voxelSize: voxelSize))
    }
}

// MARK: - Camera helpers

/// Projection and unprojection in the one convention this project uses:
/// `X_cam = R * X_world + t`, `+Z` forward, `u = fx * Xc/Zc + cx`.
/// No axis is negated anywhere in this file (CONTRACTS.md §4).
enum SmartCamera {

    @inline(__always)
    static func worldToCamera(_ pose: Pose, _ world: SIMD3<Float>) -> SIMD3<Float> {
        pose.rotation.simd.act(world) + pose.translation.simd
    }

    @inline(__always)
    static func cameraToWorld(_ pose: Pose, _ camera: SIMD3<Float>) -> SIMD3<Float> {
        pose.rotation.simd.inverse.act(camera - pose.translation.simd)
    }

    /// Projects a camera-space point to pixels. Returns nil behind the camera.
    @inline(__always)
    static func project(_ camera: SIMD3<Float>, _ k: CameraIntrinsics) -> SIMD2<Float>? {
        guard camera.z > 1e-4 else { return nil }
        return SIMD2<Float>(
            k.fx * (camera.x / camera.z) + k.cx,
            k.fy * (camera.y / camera.z) + k.cy
        )
    }

    /// Unit ray through a pixel, in camera space.
    @inline(__always)
    static func ray(_ pixel: SIMD2<Float>, _ k: CameraIntrinsics) -> SIMD3<Float> {
        simd_normalize(SIMD3<Float>((pixel.x - k.cx) / k.fx, (pixel.y - k.cy) / k.fy, 1))
    }

    /// Camera-space point for a pixel at a given *depth along +Z* (not along
    /// the ray). ARKit's `sceneDepth` is a Z-style depth, so this is the
    /// correct back-projection for it.
    @inline(__always)
    static func unproject(_ pixel: SIMD2<Float>, depthZ: Float, _ k: CameraIntrinsics) -> SIMD3<Float> {
        SIMD3<Float>(
            (pixel.x - k.cx) / k.fx * depthZ,
            (pixel.y - k.cy) / k.fy * depthZ,
            depthZ
        )
    }

    /// Intrinsics for the native depth grid, derived from the RGB intrinsics
    /// by pure rescaling. The depth sensor and the wide camera are registered
    /// by ARKit, which is what makes this legitimate: `sceneDepth` is already
    /// delivered in the colour camera's frame.
    static func nativeIntrinsics(
        _ k: CameraIntrinsics,
        depthWidth: Int,
        depthHeight: Int
    ) -> CameraIntrinsics {
        k.scaled(toWidth: depthWidth, height: depthHeight)
    }

    /// Centre of native depth pixel `(u, v)` expressed in RGB pixel
    /// coordinates.
    @inline(__always)
    static func nativePixelInImage(
        u: Int,
        v: Int,
        depthWidth: Int,
        depthHeight: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> SIMD2<Float> {
        let sx = Float(imageWidth) / Float(Swift.max(depthWidth, 1))
        let sy = Float(imageHeight) / Float(Swift.max(depthHeight, 1))
        return SIMD2<Float>((Float(u) + 0.5) * sx, (Float(v) + 0.5) * sy)
    }
}

// MARK: - Physics noise prior

/// What a LiDAR return's error is *expected* to be, before any measurement.
///
/// Two terms, both real physics, combined in quadrature:
///
///  * **Range.** A time-of-flight return's variance grows with distance
///    because the returned photon count falls off as `1/z^2` and the depth
///    estimate is a timing estimate on that signal. Modelled as
///    `sigma0 + k * z^2`.
///  * **Incidence.** A beam hitting a surface at a grazing angle spreads its
///    footprint over `1/cos(theta)` more surface, so the return mixes depths
///    across that footprint. Modelled as `spot * z * tan(theta)`, clamped at
///    80 degrees because beyond that the model stops meaning anything.
///
/// HONEST NOTE: the coefficients below are defaults chosen to put a 1 m return
/// at ~6 mm and a 5 m return at ~4 cm, which is the published order of
/// magnitude for this class of sensor. They have NOT been calibrated against
/// measured ground truth on an actual iPhone. `TwoScaleTrustField` does not
/// rely on them being right: they are the prior, and the observed cross-frame
/// residuals are the likelihood that corrects them.
public struct SmartDepthNoiseModel: Codable, Hashable, Sendable {
    /// Constant term, metres.
    public var sigma0: Float
    /// Quadratic range term, metres per metre squared.
    public var rangeCoefficient: Float
    /// Beam footprint at 1 m, metres. Multiplies `z * tan(incidence)`.
    public var spotMeters: Float
    /// Incidence angle beyond which the model is clamped, degrees.
    public var maxIncidenceDegrees: Float

    public init(
        sigma0: Float = 0.004,
        rangeCoefficient: Float = 0.0015,
        spotMeters: Float = 0.004,
        maxIncidenceDegrees: Float = 80
    ) {
        self.sigma0 = sigma0
        self.rangeCoefficient = rangeCoefficient
        self.spotMeters = spotMeters
        self.maxIncidenceDegrees = maxIncidenceDegrees
    }

    public static let `default` = SmartDepthNoiseModel()

    /// Expected 1-sigma depth error, metres.
    /// `incidenceCosine` is `|dot(surfaceNormal, viewDirection)|`; pass 1 when
    /// no normal is known, which yields the range term alone.
    public func sigma(rangeMeters z: Float, incidenceCosine: Float) -> Float {
        let zz = Swift.max(z, 0.05)
        let rangeTerm = sigma0 + rangeCoefficient * zz * zz

        let minCos = cos(maxIncidenceDegrees * .pi / 180)
        let c = Swift.max(abs(incidenceCosine), minCos)
        let tanTheta = sqrt(Swift.max(0, 1 - c * c)) / c
        let incidenceTerm = spotMeters * zz * tanTheta

        return sqrt(rangeTerm * rangeTerm + incidenceTerm * incidenceTerm)
    }
}

// MARK: - Binary sidecar IO

/// Readers and writers for the headerless little-endian sidecars in
/// `docs/DATA_FORMAT.md` sections 5 and 7.
enum SmartBinary {

    // --- Reads --------------------------------------------------------------

    /// Whole-file map. `.mappedIfSafe` matters: a house scan's noise field is
    /// hundreds of megabytes and must never be resident all at once.
    static func map(_ url: URL) throws -> Data {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SmartError.missingSidecar(url.lastPathComponent)
        }
        return try Data(contentsOf: url, options: .mappedIfSafe)
    }

    /// A frame's native depth map, converted from UInt16 millimetres to metres.
    /// A stored `0` is a NO RETURN and comes back as `0`, which callers must
    /// treat as "nothing came back", never as "zero metres".
    static func readDepth16(_ url: URL, count: Int) throws -> [Float] {
        let data = try map(url)
        let expected = count * 2
        guard data.count >= expected else {
            throw SmartError.malformedSidecar(
                path: url.lastPathComponent, expectedBytes: expected, actualBytes: data.count
            )
        }
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let stored = raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                let mm = UInt16(littleEndian: stored)
                out[i] = mm == 0 ? 0 : Float(mm) * 0.001
            }
        }
        return out
    }

    /// A frame's raw ARKit confidence, 0 low / 1 medium / 2 high.
    static func readConfidence8(_ url: URL, count: Int) throws -> [UInt8] {
        let data = try map(url)
        guard data.count >= count else {
            throw SmartError.malformedSidecar(
                path: url.lastPathComponent, expectedBytes: count, actualBytes: data.count
            )
        }
        return [UInt8](data.prefix(count))
    }

    /// A run of little-endian `Float32`.
    static func readFloats(_ data: Data, offset: Int, count: Int) -> [Float] {
        guard count > 0, offset >= 0, data.count >= offset + count * 4 else { return [] }
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: offset + i * 4, as: UInt32.self)
                out[i] = Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
        return out
    }

    // --- Writes -------------------------------------------------------------

    static func append(_ value: UInt16, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    static func append(_ value: UInt32, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    static func append(_ value: UInt64, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    static func append(_ value: Float, to data: inout Data) {
        // Guarded here rather than at every call site: one diverged value must
        // not put a NaN into a file another process reads back as truth.
        let safe = value.isFinite ? value : 0
        withUnsafeBytes(of: safe.bitPattern.littleEndian) { data.append(contentsOf: $0) }
    }

    static func write(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }
}

/// Random access to a per-native-sample `Float32` field that is frame-major on
/// disk (`trust_noise.bin`, `confidence_recal.bin`).
///
/// Holds the file mapped and decodes one frame's slice at a time into a small
/// cache, because `TrustField.weight(frame:sampleIndex:)` is synchronous and
/// O(1) and a per-call `withUnsafeBytes` on a 700 MB map is neither.
final class SmartSampleFieldReader {
    private let data: Data
    private let samplesPerFrame: Int
    /// Internal rather than private: `TwoScaleTrustField.frameTrust` builds a
    /// per-frame snapshot and has to reproduce `value(frame:sampleIndex:)`
    /// exactly, fallback included.
    let defaultValue: Float

    private var cachedFrame: FrameID?
    private var cachedSlice: [Float] = []
    private let lock = NSLock()

    let frameCount: Int

    init(url: URL, samplesPerFrame: Int, defaultValue: Float) throws {
        self.data = try SmartBinary.map(url)
        self.samplesPerFrame = Swift.max(samplesPerFrame, 1)
        self.defaultValue = defaultValue
        self.frameCount = data.count / (self.samplesPerFrame * 4)
    }

    /// One frame's whole slice, or an all-default slice when the frame is not
    /// in the file. Never throws: a missing frame is a known state, not an
    /// error, and the caller's fallback is the physics prior.
    func slice(frame: FrameID) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        if cachedFrame == frame { return cachedSlice }
        let index = Int(frame)
        guard index >= 0, index < frameCount else {
            cachedFrame = frame
            cachedSlice = [Float](repeating: defaultValue, count: samplesPerFrame)
            return cachedSlice
        }
        let values = SmartBinary.readFloats(
            data, offset: index * samplesPerFrame * 4, count: samplesPerFrame
        )
        cachedFrame = frame
        cachedSlice = values.isEmpty
            ? [Float](repeating: defaultValue, count: samplesPerFrame)
            : values
        return cachedSlice
    }

    func value(frame: FrameID, sampleIndex: Int) -> Float {
        let s = slice(frame: frame)
        guard sampleIndex >= 0, sampleIndex < s.count else { return defaultValue }
        let v = s[sampleIndex]
        return v.isFinite ? v : defaultValue
    }
}

// MARK: - Image loading

/// A decoded frame at (roughly) native-depth resolution: small, and cheap
/// enough to hold a handful of.
public struct SmartImage: Sendable {
    public let width: Int
    public let height: Int
    /// Rec. 709 luma of the encoded (sRGB) pixel, 0...1, row major. Used for
    /// edge gradients, saturation and ZNCC.
    public let luma: [Float]
    /// sRGB colour 0...1, row major. Used to fit the background field.
    public let rgb: [SIMD3<Float>]

    public init(width: Int, height: Int, luma: [Float], rgb: [SIMD3<Float>]) {
        self.width = width
        self.height = height
        self.luma = luma
        self.rgb = rgb
    }

    @inline(__always)
    public func lumaAt(_ x: Int, _ y: Int) -> Float {
        guard x >= 0, x < width, y >= 0, y < height else { return 0 }
        return luma[y * width + x]
    }

    @inline(__always)
    public func rgbAt(_ x: Int, _ y: Int) -> SIMD3<Float> {
        guard x >= 0, x < width, y >= 0, y < height else { return .zero }
        return rgb[y * width + x]
    }

    /// Bilinear luma sample at a continuous pixel coordinate.
    public func lumaBilinear(_ p: SIMD2<Float>) -> Float {
        let x = SmartMath.clamp(p.x - 0.5, 0, Float(Swift.max(width - 1, 0)))
        let y = SmartMath.clamp(p.y - 0.5, 0, Float(Swift.max(height - 1, 0)))
        let x0 = Int(x), y0 = Int(y)
        let x1 = Swift.min(x0 + 1, width - 1), y1 = Swift.min(y0 + 1, height - 1)
        let fx = x - Float(x0), fy = y - Float(y0)
        let a = lumaAt(x0, y0) * (1 - fx) + lumaAt(x1, y0) * fx
        let b = lumaAt(x0, y1) * (1 - fx) + lumaAt(x1, y1) * fx
        return a * (1 - fy) + b * fy
    }

    /// Nearest-neighbour colour at a continuous pixel coordinate.
    ///
    /// Black for a coordinate that is off the image, not a number, or
    /// infinite. `rgbAt` already range-checks, but the `Int()` conversions
    /// that reach it are trapping, so the check has to happen BEFORE the
    /// conversion, not after. `lumaBilinear` above clamps for the same reason.
    public func rgbNearest(_ p: SIMD2<Float>) -> SIMD3<Float> {
        guard let pixel = SmartMath.pixelIndex(p, width: width, height: height)
        else { return .zero }
        return rgbAt(pixel.x, pixel.y)
    }
}

/// Decodes a capture JPEG straight down to the size we want.
///
/// `CGImageSourceCreateThumbnailAtIndex` decodes at reduced scale rather than
/// decoding 12 megapixels and throwing 99% away, which is the difference
/// between this being usable on a phone and not.
enum SmartImageLoader {

    // Everything ImageIO and CoreGraphics allocate in here is autoreleased,
    // and this runs once per frame in loops that walk thousands of frames
    // without ever suspending. A Swift concurrency job drains its
    // autorelease pool when the job ENDS, not between iterations, so with
    // no explicit pool every decode in the whole pre-pass stays resident
    // until the stage finishes.
    //
    // Not theoretical. The phone reported the process growing 1814 MB in
    // 158 seconds under a stack in ImageIO and CoreGraphics, and the crash
    // lands in the second half of processing rather than at any particular
    // frame, which is what running out of headroom looks like rather than
    // hitting bad data. There was not one autoreleasepool in this codebase.
    //
    // Closing the pool around the return value is safe: each of these
    // returns a Swift value type owning its own byte storage, so no
    // CoreFoundation object outlives the pool.
    static func load(url: URL, longEdge: Int) -> SmartImage? {
        #if canImport(CoreGraphics)
        return autoreleasepool { () -> SmartImage? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                SmartLog.general.error(
                    "Could not open image \(url.lastPathComponent, privacy: .public)"
                )
                return nil
            }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: longEdge
            ]
            guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            else { return nil }
            return decode(cg)
        }

        #else
        _ = url
        _ = longEdge
        return nil
        #endif
    }

    #if canImport(CoreGraphics)
    static func decode(_ cg: CGImage) -> SmartImage? {
        let w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.noneSkipLast.rawValue
        let drew: Bool = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let base = buffer.baseAddress,
                let ctx = CGContext(
                    data: base,
                    width: w,
                    height: h,
                    bitsPerComponent: 8,
                    bytesPerRow: w * 4,
                    space: space,
                    bitmapInfo: info
                )
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drew else { return nil }

        // Both uninitialised on purpose. The loop assigns every element of
        // both before anything reads either, so the `repeating:` form's
        // zeroing was 7.78 MB of memset per decode that the very next
        // instruction overwrote.
        let count = w * h
        let luma = [Float](unsafeUninitializedCapacity: count) { buffer, n in
            n = count
            for i in 0..<count {
                let r = Float(pixels[i * 4 + 0]) / 255
                let g = Float(pixels[i * 4 + 1]) / 255
                let b = Float(pixels[i * 4 + 2]) / 255
                // Kept in sRGB space deliberately: saturation is a property of
                // the encoded pixel, not of scene radiance.
                buffer[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b
            }
        }
        let rgb = [SIMD3<Float>](
            unsafeUninitializedCapacity: count
        ) { buffer, n in
            n = count
            for i in 0..<count {
                buffer[i] = SIMD3<Float>(
                    Float(pixels[i * 4 + 0]) / 255,
                    Float(pixels[i * 4 + 1]) / 255,
                    Float(pixels[i * 4 + 2]) / 255
                )
            }
        }
        return SmartImage(width: w, height: h, luma: luma, rgb: rgb)
    }
    #endif
}

/// Small LRU so an inner loop over co-visible frames does not re-decode the
/// same three JPEGs a thousand times.
final class SmartImageCache {
    private var order: [FrameID] = []
    private var storage: [FrameID: SmartImage] = [:]
    private let capacity: Int
    private let longEdge: Int
    private let lock = NSLock()

    init(capacity: Int = 8, longEdge: Int = 256) {
        self.capacity = Swift.max(1, capacity)
        self.longEdge = Swift.max(16, longEdge)
    }

    func image(for frame: CaptureFrame, at ref: CaptureBundleRef) -> SmartImage? {
        lock.lock()
        if let hit = storage[frame.index] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        let url = ref.url(forRelativePath: frame.imagePath)
        guard let loaded = SmartImageLoader.load(url: url, longEdge: longEdge) else { return nil }

        lock.lock()
        storage[frame.index] = loaded
        order.append(frame.index)
        while order.count > capacity {
            let evicted = order.removeFirst()
            storage.removeValue(forKey: evicted)
        }
        lock.unlock()
        return loaded
    }

    func removeAll() {
        lock.lock()
        storage.removeAll()
        order.removeAll()
        lock.unlock()
    }
}
