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
        /// Trust weight at or above which a sample is treated as trusted.
        /// `TrustField.weight` is an inverse-variance weight in 0...1, so 0.5
        /// is "the measurement is at least as good as the field's reference
        /// noise", which is what `SmartLossSettings.noiseReferenceMeters`
        /// (2 cm) defines.
        var trustedWeight: Float = 0.5
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
        settings: Settings = Settings()
    ) throws -> PrePassInitialSplatOutput {

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
        var spacing = (area / Float(target)).squareRoot()
        spacing = Swift.min(
            Swift.max(spacing, settings.minSpacingMeters), settings.maxSpacingMeters
        )

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
        var rangeMeters: [Float] = []
        var hasNormal: [Bool] = []
        var onEdge: [Bool] = []

        for frame in keyframes {
            try Task.checkCancellation()

            let depthFrame: PrePassDepthFrame?
            do {
                depthFrame = try PrePassDepthFrame.load(
                    frame: frame, settings: bundle.settings, at: ref
                )
            } catch {
                continue
            }
            guard let depthFrame else { continue }

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
            let edges = edgeMapFor(frame.index)
            let pose = poseFor(frame)
            let cameraCentre = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse

            for sampleIndex in 0..<(width * height) {
                guard depthFrame.hasReturn(at: sampleIndex) else { continue }
                let z = depthFrame.depthMeters(at: sampleIndex)
                let range = geometry.range(index: sampleIndex, depthMeters: z)
                guard range > 0.2, range <= maxRange else { continue }

                let direction = rotationInverse.act(geometry.rayDirections[sampleIndex])
                let world = cameraCentre + direction * range
                guard let key = voxelFrame.key(world) else { continue }

                let sampleWeight = clamp01(trustWeight(frame.index, sampleIndex))

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
                if let measured = sigmaFor(frame.index, sampleIndex),
                   measured.isFinite, measured > 0 {
                    sampleSigma = measured
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
                    rangeMeters[s] = range
                    hasNormal[s] = normalValid
                    // An edge seen in any frame stays an edge: a curve that is
                    // a silhouette from one side is still a real 3D edge.
                    onEdge[s] = onEdge[s] || isEdge
                }
            }
        }

        let count = position.count
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

        let trustedLogit = SplatMath.invSigmoid(settings.trustedOpacity)
        let doubtfulLogit = SplatMath.invSigmoid(settings.doubtfulOpacity)
        let nativeFX = Swift.max(geometry.intrinsics.fx, 1)

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

            let trusted = weight[i] >= settings.trustedWeight && hasNormal[i]
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

        let cloud = try SplatCloud(
            shDegree: .zero,
            positions: positions,
            rotations: rotations,
            logScales: logScales,
            opacityLogits: opacityLogits,
            colorDC: colorDC,
            shRest: []
        )

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
