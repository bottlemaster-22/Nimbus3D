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
        /// A doubtful ellipsoid's half-length along the ray, in sigmas.
        var rayLengthSigmas: Float = 2.5
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
        trustWeight: (FrameID, Int) -> Float,
        sigmaFor: (FrameID, Int) -> Float?,
        edgeMapFor: (FrameID) -> [EdgeClass],
        surfaceAreaSquareMeters: Float,
        targetSplatCount: Int,
        noiseModel: SmartDepthNoiseModel = .default,
        settings: Settings = Settings(),
        census: inout PrePassCensus.Seeding
    ) throws -> PrePassInitialSplatOutput {

        census.attempted = true
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

        // Grid origin from the camera path grown by the sensor's reach, the
        // same envelope the carver and the survey use.
        var pathMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        for frame in frames { pathMin = simd_min(pathMin, poseFor(frame).center.simd) }
        let voxelFrame = PrePassVoxelFrame(
            origin: pathMin - SIMD3<Float>(repeating: maxRange + 1),
            voxelSize: spacing
        )

        // One representative sample per cell: the best-trusted one, NOT an
        // average. Averaging across a depth edge invents a surface halfway
        // between the near thing and the far thing, which is the single most
        // recognisable artefact in a badly initialised splat field.
        var hash = PrePassVoxelHash(expectedCount: 1 << 16)
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

        for frame in keyframes {
            try Task.checkCancellation()

            let depthFrame: PrePassDepthFrame?
            do {
                depthFrame = try PrePassDepthFrame.load(
                    frame: frame, settings: bundle.settings, at: ref
                )
            } catch {
                // A corrupt or unreadable sidecar. Skipped in silence before
                // the census, so a scan that read no depth at all and a scan
                // whose depth was all out of range produced the same nothing.
                keyframesDepthMissing += 1
                continue
            }
            guard let depthFrame else {
                keyframesDepthMissing += 1
                continue
            }
            keyframesDepthLoaded += 1

            // Normals come from the same unprojection ICP uses, so the surface
            // orientation a splat gets and the orientation a loop closure was
            // measured against are the same estimate.
            let points = PrePassFramePoints.build(
                depthFrame: depthFrame,
                geometry: geometry,
                maxRangeMeters: maxRange
            )

            let image = PrePassImageLoader.loadColor(
                url: ref.url(forRelativePath: frame.imagePath),
                width: width, height: height
            )
            // No picture means every splat from this frame comes out mid grey.
            // Worth knowing before someone spends a day wondering why the
            // model is the colour of concrete.
            if image == nil { keyframesImageMissing += 1 }
            let edges = edgeMapFor(frame.index)
            let pose = poseFor(frame)
            let cameraCentre = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse

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
                guard let key = voxelFrame.key(world) else { continue }
                samplesInsideGrid += 1

                let sampleWeight = clamp01(trustWeight(frame.index, sampleIndex))
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
                if let measured = sigmaFor(frame.index, sampleIndex),
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
                }
            }
        }

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
            let radius = 0.5 * Swift.max(spacing, sensorSpacing)

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
                thirdScale = Swift.max(
                    sigma[i] * settings.discThicknessSigmas,
                    radius * settings.minThicknessFraction
                )
                opacityLogits[i] = trustedLogit
                flags[i] |= Flag.pinned
                trustedCount += 1
            } else {
                axis = viewDirection[i]
                thirdScale = Swift.max(
                    sigma[i] * settings.rayLengthSigmas, radius
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
