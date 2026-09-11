//
//  PrePassInitialSplats.swift
//  PrePass
//
//  THE SET OF GAUSSIANS THE TRAINER STARTS FROM, ALREADY SHAPED BY TRUST.
//
//  A generic 3DGS run starts from a sparse cloud of triangulated points and
//  spends its first thousand iterations discovering the shape of the room. We
//  already know the shape of the room: the laser measured it. So the initial
//  set is not a scattering of seeds, it is a real surface, and each Gaussian
//  is shaped by how much that particular measurement deserves to be believed
//  (F6):
//
//   * A TRUSTED sample becomes a thin, opaque DISC lying in the surface, with
//     its position pinned. It is already right; the optimiser's job is to
//     colour it, not to move it.
//   * A DOUBTFUL sample becomes a translucent ELLIPSOID stretched along the
//     viewing ray. That is the honest shape of the uncertainty: the sensor
//     knows which direction the surface is in and is unsure how far, so the
//     Gaussian is free to slide along exactly that axis and is held in the
//     other two.
//
//  That difference is the whole point. Initialising everything as an isotropic
//  blob throws away the direction of the uncertainty, and the optimiser then
//  has to rediscover it from photometry alone, which is what produces the
//  soft, hedging cloud that generic pipelines make out of a hard wall.
//
//  ---------------------------------------------------------------------------
//  DENSITY IS DERIVED, NOT PICKED
//
//  The downsampling spacing comes from the surface area the survey actually
//  measured and the splat budget the device actually has:
//
//      spacing = sqrt(measured surface area / target splat count)
//
//  clamped below by the sensor's own sample spacing (going finer than the
//  laser sampled is inventing detail) and above by 12 cm (coarser than that
//  and a chair leg has no splats on it at all).
//
//  ---------------------------------------------------------------------------
//  NORMALS, HONESTLY
//
//  Per-splat normals ARE estimated, from the local depth gradient on the
//  native map, and they ARE delivered: the disc's rotation puts its thin axis
//  along the normal, which is the form the trainer actually consumes. They are
//  NOT written into the PLY's `nx, ny, nz` slots, because
//  `Sources/Export/PLYCodec.swift` writes those as zeros and this module does
//  not edit another module's file. `InitialSplatSetRef.hasNormals` therefore
//  reports `false`, which is the truth about the file, and an integration
//  request is filed for the one-line change that would make it `true`.
//

import Foundation
import simd

// MARK: - Reading one keyframe ahead

/// Everything a keyframe needs decoded before its samples can be walked.
struct PrePassSeedInputs {
    var depthFrame: PrePassDepthFrame?
    var points: PrePassFramePoints?
    var image: PrePassColorImage?
    /// True when the sidecar existed but would not open, which is data loss
    /// and is counted differently from a frame that never recorded depth.
    var depthUnreadable = false
}

/// Decodes ONE keyframe ahead, on a background queue, so the loading for
/// frame N+1 happens while frame N's samples are being walked.
///
/// WHY: seeding measured 16.65 s of a 22.0 s pre-pass, 76% of it, across 174
/// keyframes. That is 96 ms a frame, and each frame begins by reading a depth
/// sidecar, unprojecting 49,152 samples into points and normals, and decoding
/// a 1920x1440 JPEG, all before a single sample is examined. None of that
/// needs the previous frame's answer.
///
/// The same shape as `TrainerSupervisionPrefetch`, and safe for the same
/// reason: exactly one thread touches the work at a time. `start` waits for
/// any previous worker, `take` waits before handing back. Everything captured
/// here is read-only for the whole pass.
///
/// The edge map is deliberately NOT prefetched. `edgeMapFor` arrives as a
/// non-escaping closure and cannot cross to a worker thread, and it is a
/// cached lookup rather than a decode, so it stays where it is.
final class PrePassSeedPrefetch: @unchecked Sendable {

    private let bundle: CaptureBundle
    private let ref: CaptureBundleRef
    private let geometry: PrePassDepthGeometry
    private let maxRange: Float
    private let width: Int
    private let height: Int

    private let queue = DispatchQueue(
        label: "likova.prepass.seed-prefetch", qos: .userInitiated
    )
    private var work: DispatchWorkItem?
    private var key: FrameID?
    private var built = PrePassSeedInputs()

    init(
        bundle: CaptureBundle, ref: CaptureBundleRef,
        geometry: PrePassDepthGeometry, maxRange: Float,
        width: Int, height: Int
    ) {
        self.bundle = bundle
        self.ref = ref
        self.geometry = geometry
        self.maxRange = maxRange
        self.width = width
        self.height = height
    }

    /// Decodes `frame` on this thread. The worker calls it, and so does the
    /// caller on a miss, so there is one definition of the work.
    func load(_ frame: CaptureFrame) -> PrePassSeedInputs {
        var out = PrePassSeedInputs()
        do {
            out.depthFrame = try PrePassDepthFrame.load(
                frame: frame, settings: bundle.settings, at: ref
            )
        } catch {
            out.depthUnreadable = true
            return out
        }
        guard let depth = out.depthFrame else { return out }
        out.points = PrePassFramePoints.build(
            depthFrame: depth, geometry: geometry, maxRangeMeters: maxRange
        )
        out.image = PrePassImageLoader.loadColor(
            url: ref.url(forRelativePath: frame.imagePath),
            width: width, height: height
        )
        return out
    }

    func start(_ frame: CaptureFrame) {
        drain()
        let item = DispatchWorkItem { [self] in
            built = load(frame)
        }
        key = frame.index
        work = item
        queue.async(execute: item)
    }

    /// The prefetched frame if it is the one being asked for, otherwise nil
    /// and the caller loads it itself. Blocks until the worker is done either
    /// way, so nothing is in flight afterwards.
    func take(_ frame: CaptureFrame) -> PrePassSeedInputs? {
        work?.wait()
        work = nil
        let matched = key == frame.index
        let value = built
        key = nil
        built = PrePassSeedInputs()
        return matched ? value : nil
    }

    func drain() {
        work?.wait()
        work = nil
        key = nil
        built = PrePassSeedInputs()
    }
}

// MARK: - Result

struct PrePassInitialSplatOutput {
    var ref: InitialSplatSetRef
    /// World positions of the same points, in PLY vertex order, so the caller
    /// can write `points3D.txt` from the identical cloud rather than a second,
    /// subtly different one.
    var positions: [SIMD3<Float>]
    var colors: [SIMD3<Float>]
    /// Expected metric error per point, metres. This is what
    /// `points3D.txt`'s ERROR column carries (docs/DATA_FORMAT.md section 4).
    var expectedErrorMeters: [Float]
    var bounds: BoundingBox?
    var trustedCount: Int
    var doubtfulCount: Int
    var edgeCount: Int
    var spacingMeters: Float
}

// MARK: - Builder

enum PrePassInitialSplatBuilder {

    /// Bit meanings of `init_splats.flags`, quoted from
    /// `docs/DATA_FORMAT.md` section 7 so the writer and the document sit next
    /// to each other.
    enum Flag {
        /// Trusted sample: thin, opaque, held in place.
        static let pinned: UInt8 = 1 << 0
        /// Doubtful sample: translucent, free to slide along the viewing ray.
        static let elongated: UInt8 = 1 << 1
        /// On a detected 3D edge curve, so exempt from the disc prior (F4).
        static let onEdgeCurve: UInt8 = 1 << 2
    }

    struct Settings {
        /// Frames opened. Every keyframe costs a depth read plus a JPEG
        /// decode at native resolution.
        var maxKeyframes = 160
        /// Never finer than the laser itself sampled.
        var minSpacingMeters: Float = 0.010
        var maxSpacingMeters: Float = 0.120
        /// COARSEN FLAT INTERIORS: give a sample with a good surface normal
        /// and no edge flag a voxel this many times larger, and therefore a
        /// seed this many times bigger. 1 disables it.
        ///
        /// THIS IS THE VERSION THE MEASUREMENTS SUPPORT, and the radius-only
        /// a radius-only version is not. Setting a seed's RADIUS from local
        /// detail while every seed still sits on one uniform lattice was
        /// measured to be worth essentially nothing at the 300,000 cap: the
        /// best monotone rule over 16 detail buckets reached 16.90 dB against
        /// the as-built 16.97.
        ///
        /// What the reference model actually does is couple size to SPACING.
        /// Regressing log splat size on log local neighbour spacing gives
        /// Scaniverse a slope of 0.94 with correlation 0.79; ours are 0.18 and
        /// 0.17, which is no relationship at all. Their median splat is 0.75x
        /// its nearest-neighbour distance so splats touch but barely overlap,
        /// ours is 1.66x so each swallows several neighbours, and 69.3
        /// neighbours fall inside one of our splats against 15.2 of theirs.
        ///
        /// A second lattice makes size follow spacing BY CONSTRUCTION, because
        /// the radius is already cut from the cell size. Flat interiors get
        /// one seed where they used to get four, which is also why the seed
        /// count and the seeding stage's cost both fall: 888,951 seeds for a
        /// room is four times what the trainer keeps, and the shaping loop,
        /// the PLY write and the thinning all scale with it.
        var flatInteriorCoarsening: Float = 2

        /// Give a sample flagged as a geometric EDGE a voxel this many times
        /// smaller. 1 disables it. Edges are where a splat too big to fit
        /// produces the blur that shows, and they are only 7.8 per cent of
        /// samples, so this is cheap: measured at 244,318 seeds against
        /// today's 248,057 when paired with the coarsening above.
        var edgeRefinement: Float = 2

        /// ABSOLUTE FLOOR for trust. Deliberately low, because the real
        /// decision is made RELATIVE to the scan (see `trustedQuantile`).
        ///
        /// This was 0.5, which sounds reasonable and was not. `TrustField`
        /// computes `w = 1/(1 + (sigma/0.02)^2) * (0.25 + 0.75 * confidence)`,
        /// so clearing 0.5 needs a measured sigma under about 2 cm even at
        /// perfect confidence. That sigma comes from cross-frame residuals, so
        /// it absorbs POSE error as well as sensor noise, and this pipeline's
        /// own header says as much: "a 2 cm pose error reads as 2 cm of sensor
        /// noise everywhere". On a first handheld scan by someone whose hands
        /// shake, 2 to 4 cm of residual is entirely ordinary, so the gate
        /// failed for essentially every sample at once and every seed became a
        /// stretched translucent blob instead of a solid disc. A threshold that
        /// rejects 100 percent of real input is a badly placed threshold, not a
        /// high standard.
        var trustedWeight: Float = 0.20
        /// The REAL decision: a sample is trusted if it is in the better half
        /// of THIS scan and clears the floor above.
        ///
        /// Relative rather than absolute, so a scan is judged against what it
        /// actually achieved rather than against a fixed bar that depends on
        /// how steady the hands were. It also preserves the ordering the trust
        /// field genuinely measured, which is the useful part of it: the good
        /// half of a shaky scan is still meaningfully better than the bad half.
        var trustedQuantile: Float = 0.5
        /// Opacity a trusted splat starts at. High, because the laser says
        /// something solid is there; not 1, because the optimiser still has to
        /// be able to fade a wrong one out.
        var trustedOpacity: Float = 0.85
        /// And a doubtful one. Low enough that it contributes without
        /// committing.
        var doubtfulOpacity: Float = 0.25
        /// A trusted disc's thickness, as a multiple of its own measurement
        /// sigma. 1 sigma: the disc is exactly as thick as the sensor is
        /// unsure, which is the honest thickness.
        var discThicknessSigmas: Float = 1.0
        /// Floor on that thickness as a fraction of the disc radius, so a
        /// splat never degenerates into a zero-volume sliver the rasteriser
        /// cannot integrate.
        var minThicknessFraction: Float = 0.10

        /// The most of its own radius a TRUSTED seed may be thick.
        ///
        /// There was a floor here and no ceiling, and the floor was doing
        /// nothing while the missing ceiling was doing real damage. The
        /// thickness came out as `sigma * discThicknessSigmas`, and at the
        /// 1.29 m stand-off this app actually measures, one sigma of LiDAR
        /// noise is about the same size as the radius a seed gets from its
        /// own sample spacing. So the "disc" was as thick as it was wide: a
        /// blob, not a surface element.
        ///
        /// That is not merely a soft-looking seed. `TrainerDensifier`
        /// clones along `splitAxis`, which returns the LONGEST local axis,
        /// and for a seed thicker than it is wide the longest axis is the
        /// SURFACE NORMAL. So every clone was displaced into the wall
        /// instead of across it, and densification spent the whole growth
        /// budget building depth into flat surfaces. A room reconstructed
        /// out of blobs extruded along their normals is what the owner
        /// described as only somewhat looking like his room.
        ///
        /// 0.35 keeps a trusted seat genuinely oblate, so the longest axis
        /// is always in the surface and the existing clone logic becomes
        /// correct by construction rather than by a second patch.
        ///
        /// The doubtful branch below is deliberately NOT capped. Those
        /// seeds are elongated along the view ray on purpose, because the
        /// thing that is uncertain about them is range.
        var maxThicknessFraction: Float = 0.35
        /// A doubtful ellipsoid's half-length along the ray, in sigmas.
        var rayLengthSigmas: Float = 2.5

        /// The most of its own radius a DOUBTFUL seed may be long along
        /// the view ray. Must stay below 1.
        ///
        /// This branch was deliberately left uncapped when the trusted one
        /// was clamped, on the reasoning that a doubtful seed SHOULD be
        /// stretched along the ray because range is what is uncertain about
        /// it. That reasoning was wrong, and a screen recording of the
        /// finished model shows why: the whole reconstruction was needles
        /// radiating from a point, which is what a field of ray-aligned
        /// prolate Gaussians looks like from anywhere but the camera that
        /// made them.
        ///
        /// The reason is the disc prior in `trainer_regularizer`, which
        /// drives the covariance towards an effective rank of 2. For an
        /// (r, r, t) Gaussian with k = t/r, that target has TWO stable
        /// solutions, not one:
        ///
        ///     k = 0     -> p = (.5, .5, 0),      rank 2   a disc
        ///     k = 1     -> p = (1/3, 1/3, 1/3),  rank 3   the ridge
        ///     k = 2.61  -> p = (.11, .11, .77),  rank 2   a NEEDLE
        ///
        /// So k = 1 is a watershed, not a midpoint. A seed starting above
        /// it is not merely left elongated, it is actively driven further
        /// from a disc every iteration until it settles at 2.61. The old
        /// floor of `radius` put EVERY doubtful seed at k >= 1, and
        /// `sigma * rayLengthSigmas` routinely put it well past. The prior
        /// meant to prevent needles was manufacturing them.
        ///
        /// 0.9 keeps these the thickest seeds in the field, so they still
        /// carry more range uncertainty than a trusted one at 0.35, while
        /// sitting on the disc side of the watershed so the prior resolves
        /// them instead of inflating them. The honest expression of "we do
        /// not know the range here" is the lower opacity these already
        /// carry, which training can raise as evidence arrives, not a shape
        /// the prior will lock in.
        /// 0.7 rather than 0.9 for MARGIN. k = 1 is an unstable
        /// separatrix: a seed a little below it descends towards a disc,
        /// a seed a little above it climbs towards the needle, and the
        /// photometric gradient is perfectly capable of pushing one across
        /// a boundary it is sitting against. 0.9 gives a rank of 2.99
        /// against a ridge of 3.00, which is no margin at all. 0.7 gives
        /// 2.83 and still leaves these the thickest seeds in the field.
        var maxRayLengthFraction: Float = 0.7
        /// Hard ceiling regardless of what the budget asks for: the initial
        /// set is a starting point, and densification exists to grow it.
        var absoluteMaxSplats = 400_000
        /// The reference sigma the trust weight is built around, metres.
        ///
        /// Not used to compute anything: used to REPORT what the trust line
        /// means in metres. `TwoScaleTrustField.weight` is
        /// `1 / (1 + (sigma / reference)^2)` times a confidence factor, so a
        /// cut can be turned back into "the sigma a sample has to beat", which
        /// is the form a person can compare against a real measurement.
        ///
        /// Kept in step with `SmartLossSettings.default.noiseReferenceMeters`,
        /// which is what the pipeline builds its trust field with. It is a
        /// number for the census only, so if the two ever drift apart the
        /// reported sigma is wrong and nothing else is.
        var trustNoiseReferenceMeters: Float = SmartLossSettings.default.noiseReferenceMeters

        init() {}
    }

    /// Builds and writes `prepass/init_splats.ply` and `init_splats.flags`.
    ///
    /// - Parameters:
    ///   - poseFor: refined pose per frame.
    ///   - trustWeight: `TrustField.weight(frame:sampleIndex:)`, or a closure
    ///     returning a constant when the trust field did not build. A constant
    ///     is honest here: it means "no opinion", and every splat then gets
    ///     the doubtful shape, which is the conservative direction.
    ///   - sigmaFor: measured 1-sigma for a sample, metres, when the trust
    ///     field has one. `nil` falls back to the physics prior.
    ///   - edgeMapFor: `EdgeClassifier.map(for:)`, or a closure returning an
    ///     empty array when edges were not classified.
    ///   - surfaceAreaSquareMeters: from the survey, used to derive spacing.
    ///   - census: filled in as the build goes, INCLUDING on the two paths
    ///     that throw. It is `inout` rather than part of the return value for
    ///     exactly that reason: the run where a stage produces nothing is the
    ///     run whose numbers matter most, and a thrown error would take a
    ///     returned value with it. The caller passes a plain local variable,
    ///     whose storage is written directly, so everything counted before the
    ///     throw is still there afterwards.
    static func build(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        poseFor: (CaptureFrame) -> Pose,
        /// ONE snapshot per frame, not two closure calls per sample.
        ///
        /// This was `trustWeight: (FrameID, Int) -> Float` and
        /// `sigmaFor: (FrameID, Int) -> Float?`, both of which reached into
        /// TwoScaleTrustField and took four to six locks and three or four
        /// whole-array retain/release cycles to read two floats. Seeding runs
        /// them for 8,538,104 samples, so roughly 34 million lock
        /// acquisitions, every one of which for a given frame read the same
        /// two arrays.
        trustFrameFor: (FrameID) -> SmartFrameTrust?,
        edgeMapFor: (FrameID) -> [EdgeClass],
        surfaceAreaSquareMeters: Float,
        targetSplatCount: Int,
        noiseModel: SmartDepthNoiseModel = .default,
        settings: Settings = Settings(),
        census: inout PrePassCensus.Seeding
    ) throws -> PrePassInitialSplatOutput {

        census.attempted = true
        let seedingStarted = Date()
        census.trustedFloor = settings.trustedWeight
        census.trustedQuantile = settings.trustedQuantile
        census.targetSplatCount = targetSplatCount
        census.measuredSurfaceAreaSquareMeters = surfaceAreaSquareMeters

        let frames = bundle.frames
            .filter { $0.depthPath != nil && $0.qc.trackingQuality != .notAvailable }
            .sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard !frames.isEmpty else {
            throw NimbusError.prePassFailed(
                "there are no frames with depth to build a starting point set from"
            )
        }

        let target = Swift.max(
            1_000, Swift.min(targetSplatCount, settings.absoluteMaxSplats)
        )

        // Spacing from measured area and the real budget. `max(area, small)`
        // rather than a guard so a tiny object scan still gets the minimum
        // spacing instead of a division by zero.
        let area = Swift.max(surfaceAreaSquareMeters, 0.01)
        let requestedSpacing = (area / Float(target)).squareRoot()
        let spacing = Swift.min(
            Swift.max(requestedSpacing, settings.minSpacingMeters), settings.maxSpacingMeters
        )
        census.spacingRequestedMeters = requestedSpacing
        census.spacingMeters = spacing
        // A clamp here silently changes the density of the whole scan away
        // from what the budget asked for, in either direction.
        census.spacingWasClamped = abs(spacing - requestedSpacing) > 1e-6

        let geometry = PrePassDepthGeometry(
            rgbIntrinsics: bundle.intrinsics, settings: bundle.settings
        )
        let width = geometry.width
        let height = geometry.height
        let maxRange = Swift.max(bundle.settings.lidarMaxRangeMeters, 0.5)

        let step = Swift.max(1, frames.count / Swift.max(settings.maxKeyframes, 1))
        var keyframes: [CaptureFrame] = []
        var index = 0
        while index < frames.count {
            keyframes.append(frames[index])
            index += step
        }

        // WARM THIS PASS'S EDGE MAPS ON EVERY CORE. Since build 256 an edge
        // map is built the first time anyone asks for it, and the loop below
        // asks one keyframe at a time, so the seeder built its 174 maps
        // serially inside its own critical path. Built concurrently here
        // first, the loop then reads each back instead (from the cache, or
        // from the file it was written to). Same maps, same order of use.
        let warmupStarted = Date()
        DispatchQueue.concurrentPerform(iterations: keyframes.count) { k in
            _ = edgeMapFor(keyframes[k].index)
        }
        census.secondsEdgeWarmup = Date().timeIntervalSince(warmupStarted)

        // Grid origin from the camera path grown by the sensor's reach, the
        // same envelope the carver and the survey use.
        var pathMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        for frame in frames { pathMin = simd_min(pathMin, poseFor(frame).center.simd) }
        let gridOrigin = pathMin - SIMD3<Float>(repeating: maxRange + 1)
        let voxelFrame = PrePassVoxelFrame(origin: gridOrigin, voxelSize: spacing)
        // THE SECOND LATTICE. See `flatInteriorCoarsening`.
        //
        // Both lattices share one hash, which is safe because a Morton key
        // uses 21 bits per axis and `spread(z) << 2` tops out at bit 62, so
        // bit 63 is free and tags which lattice a key came from. Two cells
        // from different lattices can never collide, whatever they contain.
        // THREE LEVELS, and the shares are measured, not guessed. Simulated by
        // re-voxelising the owner's real geometry at each option:
        //
        //   config                          seeds  slope   p10   p50   p90  spread
        //   today, one lattice             248057  0.000  7.40  7.40  7.40   1.00x
        //   coarsen 80% of the scan        170836  0.186  7.40 14.79 14.79   2.00x
        //   refine edges only              253815  0.043  7.40  7.40  7.40   1.00x
        //   refine edges + coarsen 46%     224515  0.218  3.70  7.40 14.79   4.00x
        //
        // The second row is what a first attempt shipped, and it is the wrong
        // trade: coarsening the MAJORITY moves the median by definition, and
        // the median splat is the thing this whole effort is trying to shrink
        // (ours 19.57 mm against the reference's 3.39 mm). It also took
        // smax/nn1, the packing ratio, from 0.54 to 0.92 when the reference is
        // 0.75 and lower is tighter.
        //
        // The fourth row is strictly better than today on every measured
        // number: same median, same packing, 9 per cent fewer seeds, and the
        // size spread goes from 1.00x to 4.00x. Refining edges ALONE does
        // almost nothing, because edges are 7.8 per cent of samples (census
        // `onEdgeCount` 69,029 of 888,951) so they never reach the p10 line.
        // The spread has to come from coarsening a MINORITY of genuinely flat,
        // well-measured surface.
        let edgeFrame = PrePassVoxelFrame(
            origin: gridOrigin, voxelSize: spacing / Swift.max(settings.edgeRefinement, 1)
        )
        let coarseSpacing = spacing * Swift.max(settings.flatInteriorCoarsening, 1)
        let coarseFrame = PrePassVoxelFrame(origin: gridOrigin, voxelSize: coarseSpacing)
        let edgeSpacing = spacing / Swift.max(settings.edgeRefinement, 1)
        // Two tag bits, in the key's top nibble. A Morton key spreads each axis
        // over every third bit, so with every cell coordinate below 2^20 the
        // highest bit any coordinate can reach is `spread(z) << 2` at bit 59,
        // leaving 60 through 63 free. The guard below enforces that bound, and
        // at the finest 7.4 mm cell 2^20 cells is a 7.7 km grid, so no real
        // scan can approach it.
        let cellLimit: Int32 = 1 << 20

        // One representative sample per cell: the best-trusted one, NOT an
        // average. Averaging across a depth edge invents a surface halfway
        // between the near thing and the far thing, which is the single most
        // recognisable artefact in a badly initialised splat field.
        // SIZED FOR THE ANSWER, not for a guess. This run produces 888,951
        // seeds; 1 << 16 meant the hash rehashed every one of them four times
        // over as it doubled, and every array below reallocated and copied
        // about twenty times on its way up. `targetSplatCount` is the number
        // the caller actually wants, and the reserve costs nothing when the
        // real count comes in lower.
        let expectedSeeds = Swift.max(targetSplatCount, 1 << 16)
        var hash = PrePassVoxelHash(expectedCount: expectedSeeds)
        var position: [SIMD3<Float>] = []
        var normal: [SIMD3<Float>] = []
        var viewDirection: [SIMD3<Float>] = []
        var color: [SIMD3<Float>] = []
        var weight: [Float] = []
        var sigma: [Float] = []
        /// Whether this cell's sigma is a MEASUREMENT or the physics
        /// prediction. Kept per cell rather than as a single flag because a
        /// scan can be half and half, and a median taken over a mixture of the
        /// two would be neither.
        var sigmaMeasured: [Bool] = []
        var rangeMeters: [Float] = []
        var hasNormal: [Bool] = []
        var onEdge: [Bool] = []
        /// The size of the cell this seed came from. The radius is cut from
        /// it, which is what makes splat size follow splat SPACING rather
        /// than being one number for the whole scan.
        var cellSize: [Float] = []

        position.reserveCapacity(expectedSeeds)
        normal.reserveCapacity(expectedSeeds)
        viewDirection.reserveCapacity(expectedSeeds)
        color.reserveCapacity(expectedSeeds)
        weight.reserveCapacity(expectedSeeds)
        sigma.reserveCapacity(expectedSeeds)
        sigmaMeasured.reserveCapacity(expectedSeeds)
        rangeMeters.reserveCapacity(expectedSeeds)
        hasNormal.reserveCapacity(expectedSeeds)
        onEdge.reserveCapacity(expectedSeeds)
        cellSize.reserveCapacity(expectedSeeds)

        // The funnel, counted. Plain Ints on the stack: the inner loop runs
        // about eight million times on a room scan and does one add per
        // survivor, which is not measurable next to the unprojection it sits
        // beside.
        var keyframesDepthLoaded = 0
        var keyframesDepthMissing = 0
        var keyframesImageMissing = 0
        var samplesInspected = 0
        var samplesWithReturn = 0
        var samplesInRange = 0
        var samplesInsideGrid = 0
        var samplesWithZeroTrust = 0
        var confidenceLow = 0
        var confidenceMedium = 0
        var confidenceHigh = 0
        census.keyframesSelected = keyframes.count

        // Reads one keyframe ahead. See PrePassSeedPrefetch: the depth
        // sidecar, the point-and-normal build and the JPEG decode are the bulk
        // of a 96 ms frame and none of them needs the previous frame's answer.
        let prefetch = PrePassSeedPrefetch(
            bundle: bundle, ref: ref, geometry: geometry,
            maxRange: maxRange, width: width, height: height
        )
        defer { prefetch.drain() }
        if let first = keyframes.first { prefetch.start(first) }

        let loopStarted = Date()
        census.secondsBeforeLoop = loopStarted.timeIntervalSince(seedingStarted)
        for (keyframeIndex, frame) in keyframes.enumerated() {
            try Task.checkCancellation()

            let inputs = prefetch.take(frame) ?? prefetch.load(frame)

            // STARTED BEFORE THIS FRAME'S SAMPLE LOOP, which is the whole
            // point: the next frame decodes while these 49,152 samples are
            // walked. Above the skip guards below, so a frame with no depth
            // still leaves a worker running for the one after it.
            if keyframeIndex + 1 < keyframes.count {
                prefetch.start(keyframes[keyframeIndex + 1])
            }

            if inputs.depthUnreadable {
                // A corrupt or unreadable sidecar. Skipped in silence before
                // the census, so a scan that read no depth at all and a scan
                // whose depth was all out of range produced the same nothing.
                keyframesDepthMissing += 1
                continue
            }
            guard let depthFrame = inputs.depthFrame, let points = inputs.points
            else {
                keyframesDepthMissing += 1
                continue
            }
            keyframesDepthLoaded += 1

            let image = inputs.image
            // No picture means every splat from this frame comes out mid grey.
            // Worth knowing before someone spends a day wondering why the
            // model is the colour of concrete.
            if image == nil { keyframesImageMissing += 1 }
            let edges = edgeMapFor(frame.index)
            let pose = poseFor(frame)
            let cameraCentre = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse
            // Taken ONCE for the frame. See the parameter's note.
            let frameTrust = trustFrameFor(frame.index)

            for sampleIndex in 0..<(width * height) {
                samplesInspected += 1
                guard depthFrame.hasReturn(at: sampleIndex) else { continue }
                samplesWithReturn += 1
                let z = depthFrame.depthMeters(at: sampleIndex)
                let range = geometry.range(index: sampleIndex, depthMeters: z)
                guard range > 0.2, range <= maxRange else { continue }
                samplesInRange += 1
                // Recorded, not filtered on. A histogram that is entirely
                // medium means the confidence sidecar was missing, because
                // that is what a missing file reads back as.
                switch depthFrame.confidenceLevel(at: sampleIndex) {
                case 0: confidenceLow += 1
                case 1: confidenceMedium += 1
                default: confidenceHigh += 1
                }

                let direction = rotationInverse.act(geometry.rayDirections[sampleIndex])
                let world = cameraCentre + direction * range
                // WHICH LATTICE. A sample with a usable surface normal and no
                // edge flag is a flat interior: a wall does not need a
                // millimetre Gaussian and one seed covers what four used to.
                // Everything else keeps the fine lattice, because an edge is
                // exactly where a splat too big to fit produces the blur that
                // shows.
                //
                // Both predicates are plain array reads, hoisted above the key
                // so the routing decision costs nothing beyond the two loads
                // it already had to do further down.
                let sampleIsEdge = sampleIndex < edges.count
                    && edges[sampleIndex] == .geometric
                let sampleHasNormal = points.valid[sampleIndex]
                // High depth confidence, a usable normal and no edge flag is
                // "flat, well-measured interior", and it selects about 46 per
                // cent of samples on this scan (census: 4,878,493 high of
                // 8,552,448 inspected, 773,856 of 888,951 seeds with a normal,
                // 69,029 on an edge). That is the share the sweep above found.
                let highConfidence = depthFrame.confidenceLevel(at: sampleIndex) >= 2
                let frame: PrePassVoxelFrame
                let cellSpacing: Float
                let levelTag: UInt64
                if sampleIsEdge && settings.edgeRefinement > 1 {
                    frame = edgeFrame; cellSpacing = edgeSpacing; levelTag = 1
                } else if settings.flatInteriorCoarsening > 1
                            && sampleHasNormal && !sampleIsEdge && highConfidence {
                    frame = coarseFrame; cellSpacing = coarseSpacing; levelTag = 2
                } else {
                    frame = voxelFrame; cellSpacing = spacing; levelTag = 0
                }
                let cell = frame.cell(world)
                guard cell.x >= 0, cell.y >= 0, cell.z >= 0,
                      cell.x < cellLimit, cell.y < cellLimit, cell.z < cellLimit,
                      let morton = PrePassMorton.key(cell: cell)
                else { continue }
                let key = morton | (levelTag << 60)
                samplesInsideGrid += 1

                let sampleWeight = clamp01(frameTrust?.weight(sampleIndex: sampleIndex) ?? 0)
                if sampleWeight <= 0 { samplesWithZeroTrust += 1 }

                let slot = hash.indexOrInsert(key)
                if !slot.inserted, sampleWeight <= weight[slot.index] { continue }

                let normalValid = points.valid[sampleIndex]
                let worldNormal = normalValid
                    ? rotationInverse.act(points.normals[sampleIndex])
                    : -direction
                // Incidence cosine for the physics prior: how square-on the
                // beam hit. A grazing hit is a long thin footprint and a much
                // worse depth estimate.
                let incidence = normalValid
                    ? abs(simd_dot(
                        points.normals[sampleIndex], geometry.rayDirections[sampleIndex]
                    ))
                    : 1
                // The physics prior is the fallback, not the answer: where the
                // trust field measured a real sigma from cross-frame
                // residuals, the measurement wins.
                var sampleSigma = noiseModel.sigma(
                    rangeMeters: range, incidenceCosine: incidence
                )
                var sampleSigmaMeasured = false
                if let measured = frameTrust?.sigmaMeters(sampleIndex: sampleIndex),
                   measured.isFinite, measured > 0 {
                    sampleSigma = measured
                    sampleSigmaMeasured = true
                }

                let rgb = image?.rgb(
                    x: sampleIndex % width, y: sampleIndex / width
                ) ?? SIMD3<Float>(repeating: 0.5)

                let isEdge = sampleIndex < edges.count && edges[sampleIndex] == .geometric

                if slot.inserted {
                    position.append(world)
                    normal.append(worldNormal)
                    viewDirection.append(direction)
                    color.append(rgb)
                    weight.append(sampleWeight)
                    sigma.append(sampleSigma)
                    sigmaMeasured.append(sampleSigmaMeasured)
                    rangeMeters.append(range)
                    hasNormal.append(normalValid)
                    onEdge.append(isEdge)
                    cellSize.append(cellSpacing)
                } else {
                    let s = slot.index
                    position[s] = world
                    normal[s] = worldNormal
                    viewDirection[s] = direction
                    color[s] = rgb
                    weight[s] = sampleWeight
                    sigma[s] = sampleSigma
                    sigmaMeasured[s] = sampleSigmaMeasured
                    rangeMeters[s] = range
                    hasNormal[s] = normalValid
                    // An edge seen in any frame stays an edge: a curve that is
                    // a silhouette from one side is still a real 3D edge.
                    onEdge[s] = onEdge[s] || isEdge
                    cellSize[s] = cellSpacing
                }
            }
        }

        census.secondsSampleLoop = Date().timeIntervalSince(loopStarted)
        let count = position.count

        // Recorded BEFORE the guard below, so the run that produced nothing
        // still reports exactly how much it started with and where the last
        // survivor was lost. That run is the whole reason this exists.
        census.keyframesWithDepthLoaded = keyframesDepthLoaded
        census.keyframesDepthMissing = keyframesDepthMissing
        census.keyframesImageMissing = keyframesImageMissing
        census.samplesInspected = samplesInspected
        census.samplesWithReturn = samplesWithReturn
        census.samplesInRange = samplesInRange
        census.samplesInsideGrid = samplesInsideGrid
        census.samplesWithZeroTrustWeight = samplesWithZeroTrust
        census.samplesConfidenceLow = confidenceLow
        census.samplesConfidenceMedium = confidenceMedium
        census.samplesConfidenceHigh = confidenceHigh
        census.gaussiansBuilt = count

        guard count > 0 else {
            throw NimbusError.prePassFailed(
                "no laser measurements survived to build a starting point set from"
            )
        }

        // --- Shape each Gaussian.
        let shapingStarted = Date()
        var positions = [SIMD3<Float>](repeating: .zero, count: count)
        var rotations = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 1), count: count)
        var logScales = [SIMD3<Float>](repeating: .zero, count: count)
        var opacityLogits = [Float](repeating: 0, count: count)
        var colorDC = [SIMD3<Float>](repeating: .zero, count: count)
        var flags = [UInt8](repeating: 0, count: count)
        var expectedError = [Float](repeating: 0, count: count)

        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var trustedCount = 0
        var doubtfulCount = 0
        var edgeCount = 0
        var normalCount = 0

        let trustedLogit = SplatMath.invSigmoid(settings.trustedOpacity)
        let doubtfulLogit = SplatMath.invSigmoid(settings.doubtfulOpacity)
        let nativeFX = Swift.max(geometry.intrinsics.fx, 1)

        // Where the trusted/doubtful line actually falls for THIS scan: the
        // better half of the samples, but never below the absolute floor.
        //
        // Computed once over all samples rather than per sample, so it is a
        // property of the scan and not of the order the loop happens to visit.
        // A steady, well lit scan puts the quantile well above the floor and
        // the floor does nothing. A shaky first scan puts it below, and the
        // floor stops the model calling genuinely bad geometry solid.
        // The same sorted array serves the cut and the census, so recording
        // the distribution costs one extra read of numbers already in hand.
        // The cut and the distribution have to be read together: a cut is only
        // wrong RELATIVE to the data, which is the lesson of the 0.5 gate that
        // rejected every sample on the first real scan.
        let sortedWeights: [Float] = weight.prefix(count).filter { $0.isFinite }.sorted()
        let trustCut: Float = {
            guard !sortedWeights.isEmpty else { return settings.trustedWeight }
            let q = Swift.max(0, Swift.min(1, settings.trustedQuantile))
            let index = Swift.min(
                sortedWeights.count - 1, Int(Float(sortedWeights.count - 1) * q)
            )
            return Swift.max(sortedWeights[index], settings.trustedWeight)
        }()
        census.trustCut = trustCut
        if !sortedWeights.isEmpty {
            func quantile(_ q: Float) -> Float {
                let index = Swift.min(
                    sortedWeights.count - 1,
                    Swift.max(0, Int(Float(sortedWeights.count - 1) * q))
                )
                return sortedWeights[index]
            }
            census.weightMinimum = sortedWeights[0]
            census.weightP05 = quantile(0.05)
            census.weightMedian = quantile(0.5)
            census.weightP95 = quantile(0.95)
            census.weightMaximum = sortedWeights[sortedWeights.count - 1]
        }

        // --- The trust line, expressed in metres.
        //
        // The cut is on a WEIGHT, and a weight means nothing to a reader. The
        // trust field builds it as
        //
        //     weight = 1 / (1 + (sigma / reference)^2) * (0.25 + 0.75 * conf)
        //
        // so at perfect confidence the cut inverts to
        //
        //     sigma = reference * sqrt(1 / cut - 1)
        //
        // which is "how tight a measurement has to be to be believed". Sanity
        // check on the arithmetic: the old cut of 0.5 at the 0.02 m reference
        // gives 0.02 * sqrt(1) = 2 cm, which is exactly the gate that rejected
        // every sample on the first real scan. Printed next to the median
        // sigma below, that fault is one line instead of one day.
        let reference = Swift.max(settings.trustNoiseReferenceMeters, 1e-4)
        census.trustNoiseReferenceMeters = reference
        if trustCut > 0, trustCut < 1 {
            let equivalent = reference * (1 / trustCut - 1).squareRoot()
            if equivalent.isFinite { census.trustGateEquivalentSigmaMeters = equivalent }
        }

        // The median of the MEASURED sigmas only. Mixing in the physics prior
        // would produce a number that is neither a measurement nor a
        // prediction, and it is reported to the rest of the app under the name
        // "measured".
        var measuredSigmas: [Float] = []
        measuredSigmas.reserveCapacity(count)
        for i in 0..<count where sigmaMeasured[i] && sigma[i].isFinite {
            measuredSigmas.append(sigma[i])
        }
        census.gaussiansWithMeasuredSigma = measuredSigmas.count
        if !measuredSigmas.isEmpty {
            census.medianSigmaMeters = PrePassStats.median(measuredSigmas)
            census.sigmaIsMeasured = true
        }

        for i in 0..<count {
            let p = position[i]
            positions[i] = p
            minimum = simd_min(minimum, p)
            maximum = simd_max(maximum, p)

            // Radius: wide enough to cover the gap to its neighbours. Two
            // gaps exist and the larger one wins - the downsampling spacing,
            // and the laser's own sample spacing at this range, which is the
            // range divided by the native focal length in pixels.
            let sensorSpacing = rangeMeters[i] / nativeFX

            // EVERY SEED IN THIS SCAN WAS THE SAME SIZE. 7.397 mm at every
            // percentile from p1 to p99, for 100 per cent of the population,
            // measured on the real pre-pass output. `spacing` is one global
            // number and `sensorSpacing` only exceeds it beyond 2.726 m, while
            // the p99 distance from any point to its nearest camera in this
            // scan is 1.32 m. So the max() always picked the same constant and
            // the seed set had a size spread of 1.0000x and a sd(log10) of
            // exactly 0.0000 decades.
            //
            // That is the root of the quality gap. The reference capture of
            // the same room has a size spread of 8.9x and its splat size
            // tracks local neighbour spacing with a regression slope of 0.94
            // and correlation 0.79; ours are 0.18 and 0.17, which is no
            // relationship at all. And nothing downstream can repair it:
            // cloning copies the parent's scale exactly, and splitting only
            // ever divides by 1.6, so densification was being asked to
            // manufacture 0.64 decades of spread out of zero.
            //
            // So the seeder produces a size for each sample from what it
            // already knows about that sample, instead of one number for the
            // whole scan:
            //
            // The fix is the LATTICE, not the radius. See
            // `flatInteriorCoarsening`: setting a radius from local detail
            // while every seed still sits on one uniform grid was measured to
            // be worth essentially nothing, because size has to follow
            // SPACING, and on a uniform grid the spacing is a constant.
            //
            // THE CELL THIS SEED ACTUALLY OCCUPIES, not one global number.
            // `cellSize[i]` is `spacing` on the fine lattice and
            // `spacing * flatInteriorCoarsening` on the coarse one, so a
            // seed's size follows its own local spacing by construction. That
            // is the property the reference model has and we did not: their
            // log-size on log-spacing slope is 0.94, ours was 0.18.
            let localSpacing = cellSize.isEmpty ? spacing : cellSize[i]
            let radius = 0.5 * Swift.max(localSpacing, sensorSpacing)

            // Two gates, not one. A point can clear the trust cut and still be
            // doubtful because it never got a surface direction, and if the
            // normals fail everywhere the trusted count is zero no matter
            // where the cut sits. Counted separately for exactly that reason.
            if hasNormal[i] { normalCount += 1 }
            let trusted = weight[i] >= trustCut && hasNormal[i]
            let axis: SIMD3<Float>
            let thirdScale: Float
            if trusted {
                axis = normal[i]
                // Clamped from BOTH ends. See `maxThicknessFraction`: the
                // ceiling is what keeps this a disc rather than a blob, and
                // what keeps the surface normal from being the longest axis
                // and so the axis densification clones along.
                thirdScale = Swift.min(
                    Swift.max(
                        sigma[i] * settings.discThicknessSigmas,
                        radius * settings.minThicknessFraction
                    ),
                    radius * settings.maxThicknessFraction
                )
                opacityLogits[i] = trustedLogit
                flags[i] |= Flag.pinned
                trustedCount += 1
            } else {
                axis = viewDirection[i]
                // Below the k = 1 watershed. See `maxRayLengthFraction`:
                // above it the disc prior drives this seed to a needle
                // rather than towards a surface.
                thirdScale = Swift.min(
                    Swift.max(
                        sigma[i] * settings.rayLengthSigmas,
                        radius * settings.minThicknessFraction
                    ),
                    radius * settings.maxRayLengthFraction
                )
                opacityLogits[i] = doubtfulLogit
                flags[i] |= Flag.elongated
                doubtfulCount += 1
            }
            if onEdge[i] {
                flags[i] |= Flag.onEdgeCurve
                edgeCount += 1
            }

            rotations[i] = orientation(thirdAxis: axis)
            logScales[i] = SIMD3<Float>(
                safeLog(radius), safeLog(radius), safeLog(thirdScale)
            )
            // Display colour = 0.5 + shDCToColor * dc, so the raw degree-0
            // coefficient is the inverse of that. `SplatMath` is
            // Sources/Export's, and using it rather than repeating 0.282095
            // here is what stops the two ever drifting apart.
            colorDC[i] = (color[i] - SIMD3<Float>(repeating: 0.5)) / SplatMath.shDCToColor
            expectedError[i] = sigma[i]
        }

        // Recorded here rather than after the file is written, so a failure in
        // the PLY writer does not also lose the answer to "how many of them
        // were trusted", which is the question that matters most.
        census.gaussiansWithNormal = normalCount
        census.trustedCount = trustedCount
        census.doubtfulCount = doubtfulCount
        census.onEdgeCount = edgeCount

        var cloud = try SplatCloud(
            shDegree: .zero,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacityLogits,
            colorDC: colorDC,
            shRest: []
        )
        // `false`, not `nil`. This set has never been trained, so there is no
        // Mip-Splatting 3D filter to fold in and this producer KNOWS it. `nil`
        // is the value for a cloud read back from a file that cannot say,
        // which is a different fact, and the viewer warns on one and not the
        // other.
        //
        // SAID PLAINLY: nothing reads this today. The cloud is local, it
        // leaves this function only as a `.ply`, and no splat file format has
        // anywhere to put the marker. It is set because the value is a fact
        // about the object and the next person to hand this cloud to a
        // renderer should not have to work it out again, not because a
        // mechanism depends on it.
        cloud.filter3DFused = false

        let writeStarted = Date()
        census.secondsShaping = writeStarted.timeIntervalSince(shapingStarted)
        let plyURL = ref.url(forRelativePath: PrePassPaths.initialSplats)
        try FileManager.default.createDirectory(
            at: plyURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        do {
            try PLYCodec.write(cloud, to: plyURL)
        } catch {
            throw NimbusError.prePassFailed(
                "the starting point set could not be written: \(error)"
            )
        }
        try PrePassBinary.write(
            Data(flags), to: ref.url(forRelativePath: PrePassPaths.initialSplatFlags)
        )

        census.splatsWritten = count
        census.secondsWrite = Date().timeIntervalSince(writeStarted)

        return PrePassInitialSplatOutput(
            ref: InitialSplatSetRef(
                path: PrePassPaths.initialSplats,
                splatCount: count,
                flagsPath: PrePassPaths.initialSplatFlags,
                // The nx/ny/nz slots hold zeros: see the file header. The
                // normal is carried by the splat's rotation instead.
                hasNormals: false
            ),
            positions: positions,
            colors: color,
            expectedErrorMeters: expectedError,
            bounds: BoundingBox(min: Vector3(minimum), max: Vector3(maximum)),
            trustedCount: trustedCount,
            doubtfulCount: doubtfulCount,
            edgeCount: edgeCount,
            spacingMeters: spacing
        )
    }

    // MARK: Internals

    /// A rotation whose LOCAL Z axis is `thirdAxis`, so the third entry of
    /// `logScales` is the extent along that direction.
    ///
    /// The basis is built by crossing against whichever world axis is least
    /// parallel to `thirdAxis`; crossing against a fixed axis produces a
    /// zero-length vector exactly when the surface faces that way, which for a
    /// floor or a ceiling is most of the scan.
    static func orientation(thirdAxis: SIMD3<Float>) -> SIMD4<Float> {
        let lengthSquared = simd_length_squared(thirdAxis)
        guard lengthSquared.isFinite, lengthSquared > 1e-12 else {
            return SIMD4<Float>(0, 0, 0, 1)
        }
        let n = thirdAxis / lengthSquared.squareRoot()

        let helper: SIMD3<Float>
        if abs(n.x) < 0.9 {
            helper = SIMD3<Float>(1, 0, 0)
        } else {
            helper = SIMD3<Float>(0, 1, 0)
        }
        var t1 = simd_cross(helper, n)
        let t1Length = simd_length(t1)
        guard t1Length > 1e-9 else { return SIMD4<Float>(0, 0, 0, 1) }
        t1 /= t1Length
        let t2 = simd_cross(n, t1)

        // Columns (t1, t2, n). det = dot(cross(t1, t2), n) = dot(n, n) = 1, so
        // this is a proper rotation and never a reflection.
        let basis = simd_float3x3(t1, t2, n)
        let q = simd_quatf(basis).normalized
        return SIMD4<Float>(q.vector.x, q.vector.y, q.vector.z, q.vector.w)
    }

    @inline(__always)
    private static func safeLog(_ value: Float) -> Float {
        // 1 mm floor. A log of zero is -infinity, and one of those in a PLY
        // is a whole scan the trainer cannot load.
        Foundation.log(Swift.max(value, 0.001))
    }

    @inline(__always)
    private static func clamp01(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return Swift.min(Swift.max(value, 0), 1)
    }
}
