//
//  SmartAuthorityMap.swift
//  Smart
//
//  F5, first half: the per-pixel AUTHORITY map, the glass mask, the parallax
//  gate, and the honest stub for on-device mid-regime monocular depth.
//
//  ---------------------------------------------------------------------------
//  WHAT "AUTHORITY" MEANS
//  ---------------------------------------------------------------------------
//  Authority is one number per native depth sample, 0 to 1, answering exactly
//  one question: HOW MUCH MAY THIS PIXEL'S DEPTH BE BELIEVED?
//
//  It is not the same thing as trust (F6). Trust asks "how noisy is this
//  measurement". Authority asks "is a measurement the right kind of evidence
//  here at all". A LiDAR return off a window pane can be beautifully
//  repeatable across ten frames - high trust - and still be worthless, because
//  what it repeatably measures is the pane, not the tree behind it.
//
//  Six inputs, all multiplied, none of them a hard switch:
//
//    1. VALIDITY      a return came back and is inside the sensor's range
//    2. CONFIDENCE    the RECALIBRATED ARKit confidence (F6), not the raw one
//    3. RANGE RAMP    a smooth 4.5 -> 5.5 m fade, because the sensor does not
//                     stop working at a cliff edge and neither should we
//    4. GLASS / SKY   detected panes, ARKit window anchors, and open apertures
//    5. SATURATION    a blown-out pixel has no usable colour to fit either
//    6. PARALLAX      a patch never viewed from more than ~0.5 degrees apart
//                     is geometrically unknowable from this capture, no matter
//                     how confident the sensor was
//
//  Multiplied, not switched. Every one of these is a soft ramp because a hard
//  per-region threshold draws a visible seam across the middle of a surface,
//  and a seam is a worse artefact than the slight blur a ramp costs.
//
//  Authority 0 does not mean "delete". It means "this pixel's depth is not
//  evidence: route it to the background model and let photometry decide".
//

import Foundation
import simd

// MARK: - Regimes

/// The three regimes of F5. Which one a pixel is in is a consequence of its
/// range and its authority, never a user setting.
public enum SmartDepthRegime: String, Codable, Sendable {
    /// 0 to ~4.5 m. LiDAR is authoritative and the depth loss is at full
    /// strength.
    case near
    /// ~4.5 to ~30 m. LiDAR has nothing useful to say. A monocular metric
    /// depth model scale-anchored to the LiDAR-valid pixels of the SAME frame
    /// would; on-device there is no such model, so this regime falls back to
    /// parallax routing (see `SmartMonocularDepthStub`).
    case mid
    /// Beyond ~30 m, or any aperture with no parallax at all. Only the
    /// direction-only background field can say anything, and what it says is
    /// colour, not geometry.
    case far
}

// MARK: - Mid-regime monocular depth

/// What a monocular metric-depth model returns for one frame, already anchored
/// to that frame's own LiDAR-valid pixels.
public struct SmartMidRegimeEstimate: Sendable {
    /// Metric depth per native sample, metres. Same ordering as the depth
    /// sidecar.
    public var depthMeters: [Float]
    /// Per-sample 0...1 confidence from the model, or all-ones when the model
    /// does not produce one.
    public var confidence: [Float]
    /// The scale and shift that anchored the model's output to this frame's
    /// LiDAR. Reported so it can be inspected: a scale far from 1 means the
    /// anchoring had almost nothing to hold on to.
    public var anchor: SmartDepthAffine
    /// How many LiDAR-valid samples the anchor was fitted on.
    public var anchorSampleCount: Int
    /// Human-readable origin of these numbers, shown in the QC card. Never
    /// blank, so a stubbed run is legible in a log a year later.
    public var provenance: String

    public init(
        depthMeters: [Float],
        confidence: [Float],
        anchor: SmartDepthAffine,
        anchorSampleCount: Int,
        provenance: String
    ) {
        self.depthMeters = depthMeters
        self.confidence = confidence
        self.anchor = anchor
        self.anchorSampleCount = anchorSampleCount
        self.provenance = provenance
    }
}

/// Supplies mid-regime (4.5-30 m) metric depth for one frame.
///
/// Two real implementations are expected to exist over this protocol's life:
/// the PC Booster's Prompt Depth Anything (CVPR 2025, Apache-2.0), which runs
/// for real; and, if Apple or the authors ever ship a Core ML export, an
/// on-device one. Until then the on-device side is `SmartMonocularDepthStub`
/// and says so.
public protocol SmartMidRegimeDepthProvider: AnyObject {
    /// False when no model is bundled. Callers MUST check this rather than
    /// treating a nil estimate as a transient failure.
    var isAvailable: Bool { get }
    /// One line naming what actually produced the numbers.
    var provenance: String { get }

    /// Metric depth for one frame, anchored to that frame's LiDAR-valid
    /// samples. Returns nil when unavailable, which is a defined state, not an
    /// error.
    func estimate(
        frame: FrameID,
        image: SmartImage,
        nativeDepth: [Float],
        width: Int,
        height: Int
    ) -> SmartMidRegimeEstimate?
}

/// The honest on-device stub for F5's mid regime.
///
/// TODO(nimbus): to make this real on the phone we need, specifically:
///   1. A Core ML package of Prompt Depth Anything (or an equivalent metric
///      monocular depth model), converted at a fixed input resolution -
///      256x192 matching the native depth grid is enough, and is roughly 40x
///      cheaper than the 518x518 the reference implementation uses. No such
///      export is published as of this writing, and converting it needs a
///      macOS machine with coremltools plus the original PyTorch weights.
///   2. That `.mlpackage` added to `ios/project.yml`'s resources, and a
///      `MLModel` load guarded by `availableComputeDevices` so it falls back
///      to this stub on a device where the ANE cannot host it.
///   3. A `VNCoreMLRequest` (or a direct `MLModel.prediction`) wrapped so it
///      runs off the training queue, since a per-frame inference in the
///      training loop would stall the Metal encoder.
///   4. A thermal guard: this is a per-keyframe cost on top of an already
///      thermally gated trainer, so it must be skippable mid-run.
///
/// Until all four exist, this returns nil and the mid regime is handled by
/// `SmartParallaxField` routing plus the far-field background model. That is a
/// genuine degradation, not a hidden one: `provenance` says so, the QC card
/// shows it, and nothing anywhere pretends a depth was estimated.
public final class SmartMonocularDepthStub: SmartMidRegimeDepthProvider {

    public init() {}

    public var isAvailable: Bool { false }

    public var provenance: String {
        "No on-device monocular depth model is bundled; "
            + "mid-range geometry falls back to parallax routing."
    }

    public func estimate(
        frame: FrameID,
        image: SmartImage,
        nativeDepth: [Float],
        width: Int,
        height: Int
    ) -> SmartMidRegimeEstimate? {
        nil
    }
}

/// Anchors a RELATIVE monocular depth map to the LiDAR-valid pixels of the
/// same frame (F5).
///
/// This is real, finished code with no model behind it yet. It is separated
/// from the provider on purpose: the moment either a Core ML export appears or
/// the Booster runs the PyTorch model, the anchoring is already written,
/// already correct, and identical on both ends of the product.
///
/// Fitted in DISPARITY (`1/z`), not depth. Monocular models are trained with a
/// scale-and-shift-invariant loss in disparity space, so that is where their
/// output is actually affine to the truth; fitting in depth instead
/// systematically over-weights the far tail and drags near geometry with it.
public enum SmartMonocularScaleAnchor {

    public struct Fit: Sendable {
        /// `disparity_metric = scale * disparity_model + shift`
        public var scale: Float
        public var shift: Float
        public var sampleCount: Int
        /// Median absolute residual of the fit, in metres of depth.
        public var medianResidualMeters: Float
    }

    /// Least-squares fit over the samples where LiDAR is valid AND inside its
    /// authoritative range. Returns nil when there are too few anchors, which
    /// is the case that matters: a frame looking entirely out of a window has
    /// nothing to anchor to, and inventing a scale there is exactly the
    /// failure this whole design is built to avoid.
    public static func fit(
        modelRelativeDepth: [Float],
        lidarDepth: [Float],
        lidarMaxRangeMeters: Float,
        minimumSamples: Int = 64
    ) -> Fit? {
        let n = Swift.min(modelRelativeDepth.count, lidarDepth.count)
        guard n > 0 else { return nil }

        var sx: Double = 0, sy: Double = 0, sxx: Double = 0, sxy: Double = 0
        var count = 0
        for i in 0..<n {
            let z = lidarDepth[i]
            guard z > 0.1, z <= lidarMaxRangeMeters, SmartMath.isUsableDepth(z) else { continue }
            let m = modelRelativeDepth[i]
            guard m.isFinite, m > 1e-4 else { continue }
            let x = Double(1 / m)                     // model disparity
            let y = Double(1 / z)                     // metric disparity
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            count += 1
        }
        guard count >= minimumSamples else { return nil }

        let dn = Double(count)
        let denominator = dn * sxx - sx * sx
        guard abs(denominator) > 1e-9 else { return nil }
        let scale = Float((dn * sxy - sx * sy) / denominator)
        let shift = Float((sy - Double(scale) * sx) / dn)
        guard scale.isFinite, shift.isFinite, scale > 0 else { return nil }

        var residuals: [Float] = []
        residuals.reserveCapacity(count)
        for i in 0..<n {
            let z = lidarDepth[i]
            guard z > 0.1, z <= lidarMaxRangeMeters, SmartMath.isUsableDepth(z) else { continue }
            let m = modelRelativeDepth[i]
            guard m.isFinite, m > 1e-4 else { continue }
            let disparity = scale * (1 / m) + shift
            guard disparity > 1e-4 else { continue }
            residuals.append(abs(1 / disparity - z))
        }

        return Fit(
            scale: scale,
            shift: shift,
            sampleCount: count,
            medianResidualMeters: SmartMath.median(residuals)
        )
    }

    /// Applies a fit, producing metric depth. Samples the fit cannot express
    /// (non-positive disparity, i.e. "further than infinity") come back as 0,
    /// which every caller in this module reads as "no depth", never as
    /// "zero metres".
    public static func apply(_ fit: Fit, to modelRelativeDepth: [Float]) -> [Float] {
        modelRelativeDepth.map { m in
            guard m.isFinite, m > 1e-4 else { return 0 }
            let disparity = fit.scale * (1 / m) + fit.shift
            guard disparity > 1e-4 else { return 0 }
            let z = 1 / disparity
            return z.isFinite && z > 0 ? z : 0
        }
    }
}

// MARK: - Parallax field

/// Per-frame, coarse-grid maximum viewing-angle spread (F5).
///
/// A patch that was only ever seen from within half a degree of one direction
/// is not "poorly reconstructed"; it is UNKNOWABLE from this capture. Depth
/// there is a guess dressed as a measurement, and the honest response is to
/// route it to infinity and let the user see the honesty hatching rather than
/// to synthesise plausible geometry.
///
/// Coarse on purpose: the parallax available to a patch is a property of where
/// the photographer stood, which does not vary meaningfully between adjacent
/// native samples. A 16x12 grid over the native map is ~1.3k blocks per frame
/// instead of 49k, and loses nothing.
public struct SmartParallaxField: Sendable {
    public let blocksX: Int
    public let blocksY: Int
    public let width: Int
    public let height: Int
    /// Max angular spread in DEGREES per block, row major. `0` means no
    /// co-visible frame observed the block at all.
    public let degreesPerBlock: [Float]

    public init(
        blocksX: Int,
        blocksY: Int,
        width: Int,
        height: Int,
        degreesPerBlock: [Float]
    ) {
        self.blocksX = blocksX
        self.blocksY = blocksY
        self.width = width
        self.height = height
        self.degreesPerBlock = degreesPerBlock
    }

    /// Angular spread at a native sample, degrees.
    public func degrees(sampleIndex: Int) -> Float {
        guard width > 0, height > 0, !degreesPerBlock.isEmpty else { return 0 }
        let u = sampleIndex % width
        let v = sampleIndex / width
        let bx = Swift.min(blocksX - 1, u * blocksX / width)
        let by = Swift.min(blocksY - 1, v * blocksY / height)
        let index = by * blocksX + bx
        guard index >= 0, index < degreesPerBlock.count else { return 0 }
        return degreesPerBlock[index]
    }

    /// Builds the field for one frame from its depth map and the optical
    /// centres of its co-visible partners.
    ///
    /// The angle measured is the one that matters: the angle subtended AT THE
    /// SURFACE POINT between this camera and each partner camera. The angle
    /// between the two cameras' look directions is a different and much more
    /// forgiving number, and using it is how a system convinces itself it has
    /// parallax it does not have.
    public static func build(
        depth: [Float],
        width: Int,
        height: Int,
        blocksX: Int = 16,
        blocksY: Int = 12,
        pose: Pose,
        partnerCenters: [SIMD3<Float>],
        nativeIntrinsics k: CameraIntrinsics
    ) -> SmartParallaxField {
        let bx = Swift.max(1, blocksX)
        let by = Swift.max(1, blocksY)
        var out = [Float](repeating: 0, count: bx * by)
        guard width > 0, height > 0, depth.count >= width * height, !partnerCenters.isEmpty else {
            return SmartParallaxField(
                blocksX: bx, blocksY: by, width: width, height: height, degreesPerBlock: out
            )
        }

        let cameraCenter = pose.center.simd

        for blockY in 0..<by {
            let y0 = blockY * height / by
            let y1 = Swift.max(y0 + 1, (blockY + 1) * height / by)
            for blockX in 0..<bx {
                let x0 = blockX * width / bx
                let x1 = Swift.max(x0 + 1, (blockX + 1) * width / bx)

                // Median depth of the block, so one no-return or one spike
                // does not decide where the block's representative point is.
                var depths: [Float] = []
                for v in y0..<Swift.min(y1, height) {
                    for u in x0..<Swift.min(x1, width) {
                        let z = depth[v * width + u]
                        if z > 0, SmartMath.isUsableDepth(z) { depths.append(z) }
                    }
                }
                guard depths.count >= 4 else { continue }
                let z = SmartMath.median(depths)

                let centrePixel = SIMD2<Float>(
                    Float(x0 + x1) * 0.5, Float(y0 + y1) * 0.5
                )
                let cameraPoint = SmartCamera.unproject(centrePixel, depthZ: z, k)
                let world = SmartCamera.cameraToWorld(pose, cameraPoint)

                let toThis = world - cameraCenter
                let toThisLength = simd_length(toThis)
                guard toThisLength > 1e-4 else { continue }
                let a = toThis / toThisLength

                var maxDegrees: Float = 0
                for partner in partnerCenters {
                    let toPartner = world - partner
                    let length = simd_length(toPartner)
                    guard length > 1e-4 else { continue }
                    let b = toPartner / length
                    let cosine = SmartMath.clamp(simd_dot(a, b), -1, 1)
                    maxDegrees = Swift.max(maxDegrees, acos(cosine) * 180 / .pi)
                }
                out[blockY * bx + blockX] = maxDegrees
            }
        }

        return SmartParallaxField(
            blocksX: bx, blocksY: by, width: width, height: height, degreesPerBlock: out
        )
    }
}

// MARK: - Glass

/// Per-sample glass verdict (F5).
///
/// Deliberately three-valued. "Suspected" and "confirmed" behave differently:
/// a confirmed pane (ARKit agreed, or the pre-pass fitted an actual plane)
/// zeroes authority outright, while a suspicion merely tapers it, because the
/// image-brightness heuristic alone also fires on a white wall in sunlight.
public enum SmartGlassVerdict: UInt8, Codable, Sendable {
    case none = 0
    case suspected = 1
    case confirmed = 2

    /// How much authority survives this verdict.
    var authorityMultiplier: Float {
        switch self {
        case .none: return 1
        case .suspected: return 0.35
        case .confirmed: return 0
        }
    }
}

/// Turns the pre-pass's fitted `[GlassRegion]` plus ARKit's window anchors
/// into a per-sample verdict for one frame.
///
/// DIVISION OF LABOUR, worth stating because both halves are easy to confuse:
/// detecting a pane is `Sources/PrePass`'s job (it owns `[GlassRegion]` and
/// has the mesh and the planarity test). This type does NOT re-detect glass.
/// It projects the regions that were already found into a frame, and adds the
/// one refinement that only makes sense per pixel: a sample that returned
/// nothing while its pixel is bright is behaving exactly like glass, and a
/// sample that returned nothing in a dark corner is behaving exactly like a
/// black surface. Those need different treatment and only the image can tell
/// them apart.
public struct SmartGlassMask: Sendable {
    public let verdicts: [SmartGlassVerdict]
    public let width: Int
    public let height: Int

    public init(verdicts: [SmartGlassVerdict], width: Int, height: Int) {
        self.verdicts = verdicts
        self.width = width
        self.height = height
    }

    public func verdict(sampleIndex: Int) -> SmartGlassVerdict {
        guard sampleIndex >= 0, sampleIndex < verdicts.count else { return .none }
        return verdicts[sampleIndex]
    }

    public var confirmedFraction: Float {
        guard !verdicts.isEmpty else { return 0 }
        var n = 0
        for verdict in verdicts where verdict == .confirmed { n += 1 }
        return Float(n) / Float(verdicts.count)
    }

    /// Builds the mask for one frame.
    ///
    /// - Parameters:
    ///   - regions: fitted panes from the pre-pass, world space.
    ///   - windowAnchors: ARKit anchors classified `.window` or `.door`,
    ///     used as independent corroboration.
    ///   - brightLuma: normalised luma above which a no-return pixel is
    ///     considered "bright enough that something should have come back".
    public static func build(
        depth: [Float],
        luma: [Float]?,
        width: Int,
        height: Int,
        pose: Pose,
        nativeIntrinsics k: CameraIntrinsics,
        regions: [GlassRegion],
        windowAnchors: [AnchorRecord],
        lidarMaxRangeMeters: Float,
        brightLuma: Float = 0.55
    ) -> SmartGlassMask {
        let n = width * height
        var verdicts = [SmartGlassVerdict](repeating: .none, count: n)
        guard n > 0, depth.count >= n else {
            return SmartGlassMask(verdicts: verdicts, width: width, height: height)
        }

        let cameraCenter = pose.center.simd

        // Anchor centres, for corroborating a region that ARKit also saw.
        var anchorCenters: [SIMD3<Float>] = []
        for anchor in windowAnchors
        where anchor.classification == .window || anchor.classification == .door {
            let m = anchor.matrix
            anchorCenters.append(simd_make_float3(m.columns.3))
        }

        for v in 0..<height {
            for u in 0..<width {
                let i = v * width + u
                let z = depth[i]
                let hasReturn = z > 0 && SmartMath.isUsableDepth(z) && z <= lidarMaxRangeMeters

                let pixel = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)
                let rayCamera = SmartCamera.ray(pixel, k)
                let rayWorld = pose.rotation.simd.inverse.act(rayCamera)

                // --- Does this ray cross a detected pane? -----------------
                var hitConfirmedPane = false
                for region in regions where region.confidence >= 0.35 {
                    let normal = region.planeNormal.simd
                    let denominator = simd_dot(normal, rayWorld)
                    guard abs(denominator) > 1e-4 else { continue }
                    // dot(n, C + t*d) + offset = 0
                    let t = -(simd_dot(normal, cameraCenter) + region.planeOffset) / denominator
                    guard t > 0.1, t < 60 else { continue }
                    let hit = cameraCenter + rayWorld * t
                    guard Self.contains(region.bounds, hit, slack: 0.15) else { continue }

                    // A return well in FRONT of the pane is a real object
                    // between us and the window, not the window. Do not blank
                    // it just because the ray happens to continue into glass.
                    if hasReturn, z < t - 0.15 { continue }

                    let corroborated = region.confirmedByARKit
                        || anchorCenters.contains { simd_length($0 - hit) < 1.0 }
                    hitConfirmedPane = true
                    verdicts[i] = (corroborated || region.confidence >= 0.7)
                        ? .confirmed
                        : .suspected
                    break
                }
                if hitConfirmedPane { continue }

                // --- Per-pixel refinement: silent but bright. -------------
                // A LiDAR-silent pixel that is BRIGHT is behaving like glass
                // or like sky. A LiDAR-silent pixel that is DARK is behaving
                // like a black sofa, which is a completely different problem
                // and must not be blanketed with the same verdict.
                if !hasReturn, let luma, i < luma.count, luma[i] >= brightLuma {
                    verdicts[i] = .suspected
                }
            }
        }

        return SmartGlassMask(verdicts: verdicts, width: width, height: height)
    }

    static func contains(_ box: BoundingBox, _ p: SIMD3<Float>, slack: Float) -> Bool {
        p.x >= box.min.x - slack && p.x <= box.max.x + slack
            && p.y >= box.min.y - slack && p.y <= box.max.y + slack
            && p.z >= box.min.z - slack && p.z <= box.max.z + slack
    }
}

// MARK: - The authority map

/// One frame's authority, plus the intermediate signals that produced it.
///
/// The parts are kept, not just the product, because "authority is 0.1 here"
/// is unactionable and "authority is 0.1 here because there was no parallax"
/// tells the user to walk sideways.
public struct SmartFrameAuthority: Sendable {
    public let frame: FrameID
    public let width: Int
    public let height: Int
    /// The product, 0...1, per native sample.
    public let authority: [Float]
    /// Which of the three F5 regimes each sample fell into.
    public let regimes: [SmartDepthRegime]
    public let glass: SmartGlassMask
    public let parallax: SmartParallaxField
    /// Mean authority over the frame, for the QC card.
    public let meanAuthority: Float

    public init(
        frame: FrameID,
        width: Int,
        height: Int,
        authority: [Float],
        regimes: [SmartDepthRegime],
        glass: SmartGlassMask,
        parallax: SmartParallaxField,
        meanAuthority: Float
    ) {
        self.frame = frame
        self.width = width
        self.height = height
        self.authority = authority
        self.regimes = regimes
        self.glass = glass
        self.parallax = parallax
        self.meanAuthority = meanAuthority
    }

    public func value(sampleIndex: Int) -> Float {
        guard sampleIndex >= 0, sampleIndex < authority.count else { return 0 }
        return authority[sampleIndex]
    }

    public func regime(sampleIndex: Int) -> SmartDepthRegime {
        guard sampleIndex >= 0, sampleIndex < regimes.count else { return .far }
        return regimes[sampleIndex]
    }
}

/// Builds and caches per-frame authority.
///
/// Computed lazily with a small LRU rather than precomputed for the whole
/// scan: 500 frames x 49152 samples x 4 bytes is 98 MB, and the trainer only
/// ever looks at the handful of frames in the current batch.
public final class SmartAuthorityMap {

    private let settings: SmartLossSettings
    private let lock = NSLock()

    private var bundle: CaptureBundle?
    private var bundleRef: CaptureBundleRef?
    private var poses: [FrameID: Pose] = [:]
    private var partnerCenters: [FrameID: [SIMD3<Float>]] = [:]
    private var framesByIndex: [FrameID: CaptureFrame] = [:]
    private var glassRegions: [GlassRegion] = []
    private var windowAnchors: [AnchorRecord] = []
    private var nativeK: CameraIntrinsics?
    private var width: Int = 0
    private var height: Int = 0
    private var lidarMaxRange: Float = 5

    // Held strongly. There is no cycle to break - the trust field knows
    // nothing about this type - and a weak reference here would silently drop
    // the recalibrated-confidence term the moment the caller stopped holding
    // it, which is a quality regression with no error message.
    private var trust: TwoScaleTrustField?
    private var imageCache: SmartImageCache?
    private var depthCache: SmartDepthCache?

    private var cacheOrder: [FrameID] = []
    private var cache: [FrameID: SmartFrameAuthority] = [:]
    private let cacheCapacity: Int

    public init(settings: SmartLossSettings = .default, cacheCapacity: Int = 6) {
        self.settings = settings
        self.cacheCapacity = Swift.max(1, cacheCapacity)
    }

    /// Wires everything the authority map needs. Cheap: nothing is read off
    /// disk here except what the co-visibility table needs, which is poses.
    ///
    /// `trust` is held weakly and may be nil. Without it the recalibrated
    /// confidence term drops out of the product and every other term still
    /// applies, which is a real degradation and is logged, not hidden.
    public func prepare(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        prePassPoses: [String: Pose],
        glassRegions: [GlassRegion],
        trust: TwoScaleTrustField?
    ) {
        let w = bundle.settings.depthWidth
        let h = bundle.settings.depthHeight
        let poseTable = TwoScaleTrustField.poseTable(bundle: bundle, prePassPoses: prePassPoses)
        let partners = TwoScaleTrustField.coVisibilityTable(
            bundle: bundle,
            poses: poseTable,
            maxPartners: Swift.max(4, settings.trustPartnerFrames),
            minBaselineMeters: 0.02,
            maxBaselineMeters: 8.0,
            minDirectionDot: 0.2
        )

        var centers: [FrameID: [SIMD3<Float>]] = [:]
        for (frame, ids) in partners {
            centers[frame] = ids.compactMap { poseTable[$0]?.center.simd }
        }

        var byIndex: [FrameID: CaptureFrame] = [:]
        byIndex.reserveCapacity(bundle.frames.count)
        for frame in bundle.frames { byIndex[frame.index] = frame }

        // End-of-session anchors are preferred: ARKit silently moves anchors
        // when it relocalises, and the re-read is the more accurate placement.
        let anchors = bundle.anchorsAtEndOfSession.isEmpty
            ? bundle.anchorsDuringSession
            : bundle.anchorsAtEndOfSession

        lock.lock()
        self.bundle = bundle
        bundleRef = ref
        poses = poseTable
        partnerCenters = centers
        framesByIndex = byIndex
        self.glassRegions = glassRegions
        windowAnchors = anchors.filter {
            $0.classification == .window || $0.classification == .door
        }
        nativeK = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: w, depthHeight: h
        )
        width = w
        height = h
        lidarMaxRange = bundle.settings.lidarMaxRangeMeters
        self.trust = trust
        imageCache = SmartImageCache(capacity: 4, longEdge: Swift.max(w, h))
        depthCache = SmartDepthCache(capacity: 4, sampleCount: Swift.max(1, w * h))
        cache.removeAll()
        cacheOrder.removeAll()
        let windowAnchorCount = windowAnchors.count
        lock.unlock()

        if trust == nil {
            SmartLog.background.notice(
                """
                Authority map prepared WITHOUT a trust field: the recalibrated-confidence \
                term is absent and authority will be systematically optimistic on noisy \
                samples. Run the trust build first for the real number.
                """
            )
        }
        SmartLog.background.info(
            """
            Authority map prepared: \(bundle.frames.count) frames, \
            \(glassRegions.count) glass regions, \(windowAnchorCount) window anchors
            """
        )
    }

    public var isPrepared: Bool {
        lock.lock()
        defer { lock.unlock() }
        return bundle != nil
    }

    /// Authority for one native sample, 0...1. Synchronous and O(1) after the
    /// frame's map has been built once.
    public func authority(frame: FrameID, sampleIndex: Int) -> Float {
        map(for: frame)?.value(sampleIndex: sampleIndex) ?? 0
    }

    public func regime(frame: FrameID, sampleIndex: Int) -> SmartDepthRegime {
        map(for: frame)?.regime(sampleIndex: sampleIndex) ?? .far
    }

    /// The whole frame, built on first request and cached.
    public func map(for frame: FrameID) -> SmartFrameAuthority? {
        lock.lock()
        if let hit = cache[frame] {
            lock.unlock()
            return hit
        }
        let captureFrame = framesByIndex[frame]
        let pose = poses[frame]
        let k = nativeK
        let w = width
        let h = height
        let ref = bundleRef
        let centers = partnerCenters[frame] ?? []
        let regions = glassRegions
        let anchors = windowAnchors
        let maxRange = lidarMaxRange
        let images = imageCache
        let depths = depthCache
        let trustField = trust
        lock.unlock()

        guard
            let captureFrame,
            let pose,
            let k,
            let ref,
            let depths,
            w > 0,
            h > 0,
            let depth = depths.depth(for: captureFrame, at: ref)
        else { return nil }

        let image = images?.image(for: captureFrame, at: ref)
        let luma = image.map { Self.resampleLuma($0, width: w, height: h) }

        let glass = SmartGlassMask.build(
            depth: depth,
            luma: luma,
            width: w,
            height: h,
            pose: pose,
            nativeIntrinsics: k,
            regions: regions,
            windowAnchors: anchors,
            lidarMaxRangeMeters: maxRange
        )

        let parallax = SmartParallaxField.build(
            depth: depth,
            width: w,
            height: h,
            pose: pose,
            partnerCenters: centers,
            nativeIntrinsics: k
        )

        let n = w * h
        var authority = [Float](repeating: 0, count: n)
        var regimes = [SmartDepthRegime](repeating: .far, count: n)
        var sum: Float = 0

        for i in 0..<n {
            let z = depth[i]
            let hasReturn = z > 0 && SmartMath.isUsableDepth(z)

            // 3. Range ramp first, because it also picks the regime.
            let rangeAuthority = hasReturn
                ? SmartMath.smoothdrop(settings.nearRangeMeters, settings.farRangeMeters, z)
                : 0
            if !hasReturn {
                regimes[i] = .far
            } else if z <= settings.nearRangeMeters {
                regimes[i] = .near
            } else if z <= settings.midRangeMeters {
                regimes[i] = .mid
            } else {
                regimes[i] = .far
            }

            guard hasReturn, rangeAuthority > 0 else { continue }

            // 2. Recalibrated confidence, via the trust field's soft weight.
            let confidenceAuthority = trustField.map {
                0.25 + 0.75 * SmartMath.clamp($0.weight(frame: frame, sampleIndex: i), 0, 1)
            } ?? 1

            // 4. Glass and sky.
            let glassAuthority = glass.verdict(sampleIndex: i).authorityMultiplier

            // 5. Saturation. A blown-out pixel has no usable colour, and a
            // depth sample sitting under one cannot be photometrically
            // checked by anything.
            let saturationAuthority: Float = {
                guard let luma, i < luma.count else { return 1 }
                return SmartMath.smoothdrop(
                    settings.saturationLuma - 0.05, settings.saturationLuma, luma[i]
                )
            }()

            // 6. Parallax gate.
            let parallaxAuthority = SmartMath.smoothstep(
                settings.parallaxMinDegrees,
                settings.parallaxFullDegrees,
                parallax.degrees(sampleIndex: i)
            )

            let a = rangeAuthority
                * confidenceAuthority
                * glassAuthority
                * saturationAuthority
                * parallaxAuthority
            authority[i] = SmartMath.clamp(a, 0, 1)
            sum += authority[i]
        }

        let built = SmartFrameAuthority(
            frame: frame,
            width: w,
            height: h,
            authority: authority,
            regimes: regimes,
            glass: glass,
            parallax: parallax,
            meanAuthority: n > 0 ? sum / Float(n) : 0
        )

        lock.lock()
        cache[frame] = built
        cacheOrder.append(frame)
        while cacheOrder.count > cacheCapacity {
            cache.removeValue(forKey: cacheOrder.removeFirst())
        }
        lock.unlock()
        return built
    }

    /// Drops every cached frame. Called when the trainer moves to a new block
    /// in whole-house mode (F7), where the resident frame set changes wholesale.
    public func flush() {
        lock.lock()
        cache.removeAll()
        cacheOrder.removeAll()
        lock.unlock()
    }

    /// Box-filters a decoded image down to the native depth grid.
    ///
    /// A box filter rather than a point sample, for the same reason the edge
    /// classifier uses one: the question is "what is happening in the region
    /// this native sample covers", and point-sampling a 1920-wide image at 256
    /// columns aliases that question into noise.
    static func resampleLuma(_ image: SmartImage, width: Int, height: Int) -> [Float] {
        if image.width == width && image.height == height { return image.luma }
        var out = [Float](repeating: 0, count: width * height)
        guard image.width > 0, image.height > 0, width > 0, height > 0 else { return out }
        for y in 0..<height {
            let y0 = y * image.height / height
            let y1 = Swift.max(y0 + 1, (y + 1) * image.height / height)
            for x in 0..<width {
                let x0 = x * image.width / width
                let x1 = Swift.max(x0 + 1, (x + 1) * image.width / width)
                var sum: Float = 0
                var count = 0
                for iy in y0..<Swift.min(y1, image.height) {
                    for ix in x0..<Swift.min(x1, image.width) {
                        sum += image.luma[iy * image.width + ix]
                        count += 1
                    }
                }
                out[y * width + x] = count > 0 ? sum / Float(count) : 0
            }
        }
        return out
    }
}
