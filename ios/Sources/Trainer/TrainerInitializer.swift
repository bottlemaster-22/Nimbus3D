//
//  TrainerInitializer.swift
//  Trainer
//
//  WHERE THE GAUSSIANS COME FROM. This is not random init and it is not a
//  uniform point cloud: it is the pre-pass's own evidence, shaped by how much
//  that evidence is worth (F6).
//
//  Two sources, in order of preference:
//
//   1. `PrePassResult.initialSplats` - the set the pre-pass already built and
//      wrote to `prepass/init_splats.ply` with a `.flags` sidecar. If it is
//      there it is used, because the pre-pass had the trust fields, the edge
//      curves and the glass regions in memory when it built it. The flags are
//      honoured exactly: bit 0 pinned, bit 1 elongated along the ray, bit 2 on
//      a 3D edge curve.
//
//   2. Otherwise, seeded here from the native LiDAR depth maps of the
//      keyframes, which is the same evidence one step earlier.
//
//  THE SHAPE OF A SEED, and why:
//
//   * A TRUSTED sample (high trust weight, high authority, a near-regime
//     return) becomes a THIN, OPAQUE, PINNED disc: two axes at the local point
//     spacing, the third an order of magnitude thinner, opacity near 1, and
//     the `positionPinned` flag so Adam moves it at a tenth speed. Not frozen:
//     a trusted prior can still be wrong, and the photometry has to be able to
//     say so.
//
//   * A DOUBTFUL sample (low trust, low authority, mid regime, or a wide edge
//     band) becomes a TRANSLUCENT ELLIPSOID ELONGATED ALONG THE VIEWING RAY:
//     the direction the measurement is uncertain in is exactly the ray, so
//     that is the direction the Gaussian is allowed to be vague in and to
//     slide along. Opacity starts low so it can be deleted cheaply if the
//     photometry never finds a use for it.
//
//   * SCALE COMES FROM MEASURED LOCAL SPACING, never a constant. A constant
//     scale is the single most common way an on-device 3DGS run wastes its
//     budget: too big and every surface is a blur that densification then has
//     to undo, too small and there are holes the optimiser fills with floaters.
//     The spacing here is the distance to the nearest other seed, measured on
//     the voxel grid the seeds were deduplicated on.
//
//   * DC COLOUR IS THE OBSERVED PIXEL, converted back through the INRIA SH
//     convention (`colour = 0.5 + 0.282095 * dc`), so iteration 0 already
//     looks like the room rather than grey mud.
//
//  HOW MANY SEEDS, and why it is NOT the cap:
//
//    A seeder that fills the splat cap leaves densification nothing to do.
//    The densifier's growth allowance is `max(splatCap - splatCount, 0)`, so
//    starting at the cap makes that zero on every growth pass of the run and
//    the model can never add a single Gaussian, no matter how well the
//    densification scores are working. Both sources therefore thin to
//    `seedTarget(forSplatCap:)`, which is at most three quarters of the cap
//    and normally half of it. See that function for the arithmetic.
//

import Foundation
import simd

/// One seeded Gaussian, before it is written into the GPU buffer.
struct TrainerSeed {
    var position: SIMD3<Float>
    var rotation: SIMD4<Float>     // (x, y, z, w)
    var logScale: SIMD3<Float>
    var opacityLogit: Float
    var colorDC: SIMD3<Float>      // raw SH degree-0 coefficient
    var flags: UInt32
}

/// What a seeding run produced, plus the honest count of what it skipped.
struct TrainerSeedResult {
    var seeds: [TrainerSeed]
    var source: String
    var framesUsed: Int
    var samplesConsidered: Int
    var samplesRejected: Int
    var medianSpacingMeters: Float

    /// How many seeds were laid as SOLID DISCS across the surface because the
    /// depth sample was trusted, and how many were STRETCHED ALONG THE VIEWING
    /// RAY because it was not.
    ///
    /// This split is the seeder's trust gate made countable. The owner's first
    /// scan had a trust threshold that required a measured sigma under about
    /// two centimetres, while that sigma also absorbs pose error and two to
    /// four centimetres is ordinary on a handheld walk, so EVERY sample failed
    /// and every seed came out a stretched blob. Nothing said so. These two
    /// numbers say so.
    ///
    /// Computed from the flags rather than counted during construction, so
    /// they are measured AFTER the thinning to the cap and there is only one
    /// place that can be wrong. One pass over the seeds, once per slice.
    var seedsPinnedAsDiscs: Int {
        seeds.reduce(0) { $0 + (($1.flags & TrainerSplatFlag.positionPinned) != 0 ? 1 : 0) }
    }

    var seedsStretchedAlongRay: Int {
        seeds.reduce(0) { $0 + (($1.flags & TrainerSplatFlag.elongatedAlongRay) != 0 ? 1 : 0) }
    }

    /// One plain sentence for the log and the progress message.
    var summary: String {
        if seeds.isEmpty {
            return "No starting points could be built from this scan."
        }
        return "Started from \(seeds.count) points taken from \(framesUsed) photos, "
            + "spaced about \(Int((medianSpacingMeters * 1000).rounded())) mm apart."
    }
}

enum TrainerInitializer {

    /// The exact `TrainerSeedResult.source` value the depth-map seeding path
    /// reports. Named once because the census matches on it: on this path
    /// `samplesRejected` is a real quality rejection and a rate near 100 per
    /// cent is a bug, while on the pre-pass path the same field counts the
    /// spatial thinning down to the cap and is MEANT to be large. Comparing
    /// against a loose string literal in two files is how that distinction
    /// silently stops working.
    static let depthSeedSourceName = "native depth maps"

    /// How many seeds a slice is allowed to START with, given the splat cap
    /// that slice has to live inside for the whole run.
    ///
    /// THIS IS NOT THE CAP, AND THAT IS THE ENTIRE POINT. `SplatDensifier`
    /// works out what it may add as
    ///
    ///     headroom = max(splatCap - splatCount, 0)
    ///
    /// so a seed set that already fills the cap makes the headroom zero on
    /// pass one and keeps it zero for the rest of the run. Densification then
    /// scores correctly, logs correctly, and adds nothing, which is exactly
    /// the failure the thermal-cap ratchet produced from the other direction.
    /// Growth being switched off is not a smaller version of growth: it is no
    /// growth.
    ///
    /// The rule, in order:
    ///
    ///  * Never more than THREE QUARTERS of the cap, so there is always real
    ///    room to grow into even when the cap is tiny (`effectiveCap` has a
    ///    5,000 floor in `MetalSplatTrainer`, and at 5,000 a "20,000 minimum"
    ///    would silently mean "no thinning at all").
    ///  * Otherwise HALF the cap, which is the split the depth-map path was
    ///    already sizing its voxels for, so densification gets as much room as
    ///    the seeds took.
    ///  * With a 20,000 floor underneath, because a very small starting set is
    ///    a poor start regardless of what the cap says, and that floor is
    ///    itself clamped by the three-quarter ceiling above.
    ///
    /// Worked through for the caps this app actually produces:
    /// 150,000 (one object, full tier) seeds 75,000 and leaves 75,000;
    /// 300,000 (a room) seeds 150,000 and leaves 150,000;
    /// 40,000 (limited tier) seeds 20,000 and leaves 20,000;
    /// 5,000 (the emergency floor) seeds 3,750 and leaves 1,250.
    static func seedTarget(forSplatCap cap: Int) -> Int {
        guard cap > 0 else { return 0 }
        let ceilingWithHeadroom = (cap * 3) / 4
        return Swift.max(cap / 2, Swift.min(20_000, ceilingWithHeadroom))
    }

    // MARK: - Entry point

    /// Builds the starting Gaussian set. Prefers the pre-pass's own set, falls
    /// back to seeding from the depth maps, and throws only when neither
    /// produced a single point, because a trainer with nothing to train is a
    /// failure worth naming rather than an empty result worth pretending about.
    static func seed(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        keyframes: [CaptureFrame],
        budget: TrainingBudget,
        trust: TwoScaleTrustField?,
        authority: SmartAuthorityMap?,
        edges: NativeDepthEdgeClassifier?,
        settings: SmartLossSettings
    ) throws -> TrainerSeedResult {

        if let refs = prePass.initialSplats,
           let loaded = loadPrePassSet(refs, at: ref, budget: budget)
        {
            TrainerLog.general.info(
                "Seeded from the pre-pass set: \(loaded.seeds.count) points"
            )
            return loaded
        }

        let seeded = seedFromDepth(
            bundle: bundle,
            prePass: prePass,
            at: ref,
            keyframes: keyframes,
            budget: budget,
            trust: trust,
            authority: authority,
            edges: edges,
            settings: settings
        )
        guard !seeded.seeds.isEmpty else {
            throw TrainerError.nothingToTrain(
                "none of the photos in this scan came with depth the trainer could use"
            )
        }
        return seeded
    }

    // MARK: - Source 1: the pre-pass's own set

    private static func loadPrePassSet(
        _ refs: InitialSplatSetRef,
        at ref: CaptureBundleRef,
        budget: TrainingBudget
    ) -> TrainerSeedResult? {
        let url = ref.url(forRelativePath: refs.path)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let cloud = try? PLYCodec.read(from: url), cloud.count > 0 else {
            TrainerLog.general.error(
                "The pre-pass starting points at \(refs.path, privacy: .public) could not be read; seeding from the depth maps instead"
            )
            return nil
        }

        var flags = [UInt8](repeating: 0, count: cloud.count)
        if let flagsPath = refs.flagsPath {
            let flagsURL = ref.url(forRelativePath: flagsPath)
            if let data = try? Data(contentsOf: flagsURL, options: .mappedIfSafe),
               data.count >= cloud.count
            {
                flags = [UInt8](data.prefix(cloud.count))
            }
        }

        var seeds: [TrainerSeed] = []
        seeds.reserveCapacity(cloud.count)
        for i in 0..<cloud.count {
            seeds.append(
                TrainerSeed(
                    position: cloud.positions[i],
                    rotation: cloud.rotations[i],
                    logScale: cloud.logScales[i],
                    opacityLogit: cloud.opacityLogits[i],
                    colorDC: cloud.colorDC[i],
                    flags: UInt32(flags[i])
                )
            )
        }

        // `seedTarget`, NOT `budget.splatCap`. Thinning to the cap itself fills
        // it exactly, which makes `headroom = max(splatCap - splatCount, 0)`
        // zero on the first growth pass and every pass after it, so
        // densification scores correctly, logs correctly and adds nothing. This
        // is the PREFERRED seeding path, so that was the state of every real
        // run: growth was impossible even after the gradient-threshold fix that
        // was supposed to switch densification back on.
        seeds = thin(seeds, to: seedTarget(forSplatCap: budget.splatCap))
        let spacing = medianNearestSpacing(of: seeds)
        return TrainerSeedResult(
            seeds: seeds,
            source: refs.path,
            framesUsed: 0,
            samplesConsidered: cloud.count,
            samplesRejected: cloud.count - seeds.count,
            medianSpacingMeters: spacing
        )
    }

    // MARK: - Source 2: the native depth maps

    // swiftlint:disable:next function_body_length - one pass over the samples,
    // and splitting it would mean handing eight parallel arrays between
    // functions, which is harder to check than the loop is to read.
    private static func seedFromDepth(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        keyframes: [CaptureFrame],
        budget: TrainingBudget,
        trust: TwoScaleTrustField?,
        authority: SmartAuthorityMap?,
        edges: NativeDepthEdgeClassifier?,
        settings: SmartLossSettings
    ) -> TrainerSeedResult {

        let depthWidth = bundle.settings.depthWidth
        let depthHeight = bundle.settings.depthHeight
        let sampleCount = depthWidth * depthHeight
        guard sampleCount > 0, !keyframes.isEmpty else {
            return TrainerSeedResult(
                seeds: [], source: depthSeedSourceName, framesUsed: 0,
                samplesConsidered: 0, samplesRejected: 0, medianSpacingMeters: 0
            )
        }

        let nativeK = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: depthWidth, depthHeight: depthHeight
        )
        let lidarMaxRange = bundle.settings.lidarMaxRangeMeters

        // Voxel size: aim to fill roughly half the budget with one seed per
        // voxel, from the scene's measured extent. Never a fixed 1 cm.
        // The measured extent when the capture recorded one, otherwise the
        // median camera-to-surface distance from the QC card, which is a
        // measurement too. Never a constant.
        let extent = Swift.max(
            bundle.sceneBounds?.longestEdgeMeters
                ?? (prePass.qcCard.medianCameraToSurfaceMeters * 3),
            0.5
        )
        let targetSeeds = Swift.max(budget.splatCap / 2, 20_000)
        // A room is a shell, not a solid, so the seeds live on a surface:
        // count grows with the SQUARE of extent over voxel size, not the cube.
        let voxel = Swift.max(
            0.005,
            Swift.min(0.10, extent / Swift.max(sqrtf(Float(targetSeeds)), 1))
        )

        let imageCache = SmartImageCache(capacity: 3, longEdge: Swift.max(depthWidth * 2, 256))
        let depthCache = SmartDepthCache(capacity: 3, sampleCount: sampleCount)

        /// One accumulating voxel. Averaging colour and position inside a
        /// voxel is legitimate (they are repeat measurements of one surface
        /// patch); averaging TRUST across a voxel is not, so the best trust in
        /// the voxel is kept rather than the mean of them.
        struct Cell {
            var position = SIMD3<Float>.zero
            var color = SIMD3<Float>.zero
            var rayDirection = SIMD3<Float>.zero
            var weightSum: Float = 0
            var bestTrust: Float = 0
            var range: Float = 0
            var onEdge = false
        }
        var cells: [SIMD3<Int32>: Cell] = [:]
        cells.reserveCapacity(Swift.min(targetSeeds * 2, 400_000))

        var considered = 0
        var rejected = 0
        var framesUsed = 0

        // Every third native sample in each axis: 256x192 down to ~86x64 per
        // frame. At a 2 cm voxel with a hundred keyframes that is still many
        // measurements per cell, and it is nine times less work.
        let stride = 3

        for frame in keyframes {
            // The refined pose when the pre-pass produced one, the frame's own
            // refined pose next, and the raw VIO pose last. All three are real
            // poses; the order is best-evidence-first, never a fabrication.
            let pose = prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
            guard let depth = depthCache.depth(for: frame, at: ref) else { continue }
            let image = imageCache.image(for: frame, at: ref)
            let edgeMap = edges?.map(for: frame.index) ?? []
            let affine = trust?.depthAffine(frame: frame.index) ?? .identity
            framesUsed += 1

            let cameraCenter = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse

            var v = 0
            while v < depthHeight {
                var u = 0
                while u < depthWidth {
                    let index = v * depthWidth + u
                    considered += 1
                    let raw = depth[index]
                    guard raw > 0.05, raw < lidarMaxRange else {
                        rejected += 1
                        u += stride
                        continue
                    }
                    let z = affine.apply(raw)

                    let edgeClass: EdgeClass = index < edgeMap.count ? edgeMap[index] : .none
                    // A band sample is where the upsampled map is simply
                    // wrong; seeding there puts a Gaussian on the smear.
                    if edgeClass == .band {
                        rejected += 1
                        u += stride
                        continue
                    }

                    let trustWeight = trust?.weight(frame: frame.index, sampleIndex: index) ?? 0.5
                    let authorityValue = authority?.authority(frame: frame.index, sampleIndex: index) ?? 0.5
                    let regime = authority?.regime(frame: frame.index, sampleIndex: index) ?? .near
                    // The far regime is the background model's job, not a
                    // Gaussian's, and seeding it is how a sky ends up as a
                    // wall of floaters two metres from the camera.
                    if regime == .far {
                        rejected += 1
                        u += stride
                        continue
                    }

                    let pixel = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)
                    let camPoint = SmartCamera.unproject(pixel, depthZ: z, nativeK)
                    let world = SmartCamera.cameraToWorld(pose, camPoint)
                    guard world.x.isFinite, world.y.isFinite, world.z.isFinite else {
                        rejected += 1
                        u += stride
                        continue
                    }

                    var colour = SIMD3<Float>(repeating: 0.5)
                    if let image {
                        let p = SmartCamera.nativePixelInImage(
                            u: u, v: v,
                            depthWidth: depthWidth, depthHeight: depthHeight,
                            imageWidth: image.width, imageHeight: image.height
                        )
                        colour = image.rgbNearest(p)
                    }

                    let rayWorld = rotationInverse.act(simd_normalize(camPoint))
                    let combinedTrust = TrainerMath.clamp(trustWeight * authorityValue, 0, 1)
                    let contribution = Swift.max(combinedTrust, 0.05)

                    let key = SIMD3<Int32>(
                        Int32((world.x / voxel).rounded(.down)),
                        Int32((world.y / voxel).rounded(.down)),
                        Int32((world.z / voxel).rounded(.down))
                    )
                    var cell = cells[key] ?? Cell()
                    cell.position += world * contribution
                    cell.color += colour * contribution
                    cell.rayDirection += rayWorld * contribution
                    cell.weightSum += contribution
                    if combinedTrust > cell.bestTrust {
                        cell.bestTrust = combinedTrust
                        cell.range = simd_length(camPoint)
                    }
                    if edgeClass == .geometric { cell.onEdge = true }
                    cells[key] = cell

                    u += stride
                }
                v += stride
            }

            // A whole-house walk can hand us thousands of keyframes. Stop once
            // the voxel grid is already carrying more than the budget can hold:
            // more frames past that point only refine colours we are about to
            // average anyway.
            if cells.count > targetSeeds * 3 { break }
        }

        guard !cells.isEmpty else {
            return TrainerSeedResult(
                seeds: [], source: depthSeedSourceName, framesUsed: framesUsed,
                samplesConsidered: considered, samplesRejected: rejected,
                medianSpacingMeters: 0
            )
        }

        // Local spacing, measured rather than assumed: for each occupied
        // voxel, the distance to the nearest occupied neighbour in the 26-cell
        // ring, falling back to the voxel edge when the cell stands alone.
        var seeds: [TrainerSeed] = []
        seeds.reserveCapacity(cells.count)
        var spacings: [Float] = []
        spacings.reserveCapacity(cells.count)

        for (key, cell) in cells {
            guard cell.weightSum > 1e-6 else { continue }
            let position = cell.position / cell.weightSum
            let colour = simd_clamp(cell.color / cell.weightSum, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            let ray = simd_length(cell.rayDirection) > 1e-6
                ? simd_normalize(cell.rayDirection)
                : SIMD3<Float>(0, 0, 1)

            var nearest = Float.greatestFiniteMagnitude
            for dz in -1...1 {
                for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0 && dz == 0) {
                        let neighbourKey = SIMD3<Int32>(
                            key.x &+ Int32(dx), key.y &+ Int32(dy), key.z &+ Int32(dz)
                        )
                        guard let neighbour = cells[neighbourKey], neighbour.weightSum > 1e-6
                        else { continue }
                        let d = simd_distance(position, neighbour.position / neighbour.weightSum)
                        if d > 1e-5 { nearest = Swift.min(nearest, d) }
                    }
                }
            }
            let spacing = nearest.isFinite ? Swift.min(nearest, voxel * 2) : voxel
            spacings.append(spacing)

            let trusted = cell.bestTrust >= settings.minimumAuthorityForDepth * 4
                && cell.bestTrust >= 0.35
            var seed: TrainerSeed

            if trusted {
                // Thin, opaque, pinned. The disc lies ACROSS the ray, which is
                // the best available stand-in for the surface normal before any
                // normal has been estimated.
                let radius = Swift.max(spacing * 0.5, 0.002)
                let thickness = Swift.max(radius * 0.1, 0.0008)
                seed = TrainerSeed(
                    position: position,
                    rotation: TrainerMath.quaternionAligningZ(to: ray),
                    logScale: SIMD3<Float>(logf(radius), logf(radius), logf(thickness)),
                    opacityLogit: TrainerMath.logit(0.9),
                    colorDC: dcFromColor(colour),
                    flags: TrainerSplatFlag.positionPinned
                        | (cell.onEdge ? TrainerSplatFlag.onEdgeCurve : 0)
                )
            } else {
                // Elongated along the ray, translucent, free. The uncertainty
                // in a doubtful depth sample is along the ray and nowhere else,
                // so that is the axis the Gaussian is allowed to be long in.
                let radius = Swift.max(spacing * 0.5, 0.003)
                // Uncertainty grows with range: a doubtful 4 m sample is
                // uncertain over tens of centimetres, a doubtful 0.5 m one
                // over a couple.
                let along = Swift.max(radius * 3, cell.range * 0.05)
                seed = TrainerSeed(
                    position: position,
                    rotation: TrainerMath.quaternionAligningZ(to: ray),
                    logScale: SIMD3<Float>(logf(radius), logf(radius), logf(along)),
                    opacityLogit: TrainerMath.logit(0.25),
                    colorDC: dcFromColor(colour),
                    flags: TrainerSplatFlag.elongatedAlongRay
                        | (cell.onEdge ? TrainerSplatFlag.onEdgeCurve : 0)
                )
            }
            seeds.append(seed)
        }

        // Same rule as the pre-pass path above: leave densification real room
        // to grow into rather than handing it a full cap and zero headroom.
        seeds = thin(seeds, to: seedTarget(forSplatCap: budget.splatCap))
        spacings.sort()
        let median = spacings.isEmpty ? voxel : spacings[spacings.count / 2]

        return TrainerSeedResult(
            seeds: seeds,
            source: depthSeedSourceName,
            framesUsed: framesUsed,
            samplesConsidered: considered,
            samplesRejected: rejected,
            medianSpacingMeters: median
        )
    }

    // MARK: - Helpers

    /// The INRIA / SPZ convention, inverted: `colour = 0.5 + 0.282095 * dc`.
    /// The same constant `Sources/Export` writes and reads, so a seeded cloud
    /// and an exported one mean the same thing by "red".
    private static func dcFromColor(_ colour: SIMD3<Float>) -> SIMD3<Float> {
        (colour - SIMD3<Float>(repeating: 0.5)) / 0.282095_017
    }

    /// Reduces a seed set to fit the cap, keeping a SPATIALLY UNIFORM subset
    /// rather than a prefix. A prefix of a voxel-hash enumeration is whichever
    /// corner of the room the hash happened to bucket first, and starting a
    /// scan with one wall dense and the rest empty is a hole densification
    /// then spends its whole budget trying to fill.
    private static func thin(_ seeds: [TrainerSeed], to cap: Int) -> [TrainerSeed] {
        guard cap > 0, seeds.count > cap else { return seeds }
        let keepRatio = Double(cap) / Double(seeds.count)
        var kept: [TrainerSeed] = []
        kept.reserveCapacity(cap)
        var accumulator = 0.0
        for seed in seeds {
            accumulator += keepRatio
            if accumulator >= 1 {
                accumulator -= 1
                kept.append(seed)
                if kept.count == cap { break }
            }
        }
        return kept
    }

    /// Median distance to the nearest other seed, sampled rather than computed
    /// exhaustively (an exact all-pairs pass on 300k points is minutes).
    /// Used only for the plain-language summary, never for a scale.
    private static func medianNearestSpacing(of seeds: [TrainerSeed]) -> Float {
        guard seeds.count > 8 else { return 0 }
        // A coarse hash makes "nearest neighbour" a local question.
        var grid: [SIMD3<Int32>: [Int]] = [:]
        let cell: Float = 0.05
        for (i, seed) in seeds.enumerated() {
            let key = SIMD3<Int32>(
                Int32((seed.position.x / cell).rounded(.down)),
                Int32((seed.position.y / cell).rounded(.down)),
                Int32((seed.position.z / cell).rounded(.down))
            )
            grid[key, default: []].append(i)
        }

        var distances: [Float] = []
        let sampleStride = Swift.max(seeds.count / 2000, 1)
        var i = 0
        while i < seeds.count {
            let position = seeds[i].position
            let key = SIMD3<Int32>(
                Int32((position.x / cell).rounded(.down)),
                Int32((position.y / cell).rounded(.down)),
                Int32((position.z / cell).rounded(.down))
            )
            var best = Float.greatestFiniteMagnitude
            for dz in -1...1 {
                for dy in -1...1 {
                    for dx in -1...1 {
                        let neighbourKey = SIMD3<Int32>(
                            key.x &+ Int32(dx), key.y &+ Int32(dy), key.z &+ Int32(dz)
                        )
                        for j in grid[neighbourKey] ?? [] where j != i {
                            best = Swift.min(best, simd_distance(position, seeds[j].position))
                        }
                    }
                }
            }
            if best.isFinite { distances.append(best) }
            i += sampleStride
        }
        guard !distances.isEmpty else { return 0 }
        distances.sort()
        return distances[distances.count / 2]
    }

    // MARK: - Upload

    /// Writes a seed set into the GPU buffers, along with the SH tail (zeroed:
    /// the view-dependent terms start at zero and are learned, which is what
    /// coarse-to-fine SH activation expects) and clean Adam state.
    static func upload(
        _ seeds: [TrainerSeed],
        into resources: TrainerResources
    ) -> Int {
        let count = Swift.min(seeds.count, resources.splatCapacity)
        guard count > 0 else { return 0 }

        var splats = [TrainerSplat](repeating: TrainerSplat(), count: count)
        let shPerSplat = resources.shFloatsPerSplat
        var sh = [Float](repeating: 0, count: count * shPerSplat)

        for i in 0..<count {
            let seed = seeds[i]
            var splat = TrainerSplat()
            splat.rotation = seed.rotation
            splat.mean = seed.position
            splat.logScale = seed.logScale
            splat.opacityLogit = seed.opacityLogit
            splat.flags = seed.flags
            splats[i] = splat

            let base = i * shPerSplat
            sh[base + 0] = seed.colorDC.x
            sh[base + 1] = seed.colorDC.y
            sh[base + 2] = seed.colorDC.z
        }

        resources.splats.writeArray(splats)
        resources.sh.writeArray(sh)

        // Everything derived starts clean. The stats buffer in particular
        // carries `filter3D`, and a stale filter from a previous run would
        // make every Gaussian the wrong size until the next sweep.
        resources.stats.zeroAll()
        resources.adamM.zeroAll()
        resources.adamV.zeroAll()
        resources.shAdamM.zeroAll()
        resources.shAdamV.zeroAll()
        resources.samplingTopK.zeroAll()

        return count
    }
}
