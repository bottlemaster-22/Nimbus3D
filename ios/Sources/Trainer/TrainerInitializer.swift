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
//  WHERE THE TRUSTED / DOUBTFUL LINE IS DRAWN, and why it is not a constant:
//
//    RELATIVE TO THIS SCAN, exactly as `PrePassInitialSplatBuilder` draws it:
//    the better half of the trust scores this scan actually produced, with a
//    low absolute floor underneath so a scan that measured nothing cannot have
//    half of nothing called solid. Both seeders now mean the same thing by
//    "trusted", which matters because the answer decides the SHAPE of the seed
//    (a thin pinned disc against a stretched translucent blob) and the two
//    shapes train completely differently.
//
//    The gate that used to be here was absolute and it passed nothing:
//
//        trusted = bestTrust >= minimumAuthorityForDepth * 4 && bestTrust >= 0.35
//
//    `bestTrust` is not a trust weight. It is `trustWeight * authority`, and
//    authority is itself `0.25 + 0.75 * trustWeight` times the range, glass,
//    saturation and parallax ramps, so that product is a trust weight
//    multiplied by a function of itself. Clearing 0.35 on the product needs a
//    trust weight of 0.537 even when every other ramp is perfect, which is a
//    HARDER bar than the flat 0.5 that had already been measured rejecting one
//    hundred per cent of a real handheld scan. The first clause was dead:
//    `minimumAuthorityForDepth` is 0.05, so it read 0.20, and 0.20 can never
//    bind under 0.35. It only made the gate look softer than it was.
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

/// The trusted / doubtful line the depth-map seeder drew, and the distribution
/// it drew it from.
///
/// Recorded rather than recomputed, because a cut is only ever wrong RELATIVE
/// to the data it was applied to: "the cut was 0.35" says nothing, and "the cut
/// was 0.35 and the whole scan topped out at 0.21" says everything. That pair
/// of numbers is what the previous absolute gate never wrote down, which is why
/// it survived a review and a real scan.
struct TrainerSeedTrustCut {
    /// False when this scan had no trust field at all. The seeder then calls
    /// nothing trusted, which is the same choice `PrePassInitialSplatBuilder`
    /// makes when its trust weights are missing (its weights read as zero and
    /// nothing clears the floor). It is a DIFFERENT state from "the gate
    /// rejected everything", and a census that cannot tell them apart sends
    /// someone hunting for a bug in a threshold that was never consulted.
    var wasMeasured: Bool
    /// The score a cell had to reach to be laid as a solid disc.
    var cut: Float
    /// The absolute floor under the quantile, and the quantile itself.
    var floor: Float
    var quantile: Float
    /// The distribution the cut came out of.
    var p05: Float
    var median: Float
    var p95: Float
    /// How many occupied voxels the distribution was taken over.
    var cellsConsidered: Int
}

/// What a seeding run produced, plus the honest count of what it skipped.
struct TrainerSeedResult {
    var seeds: [TrainerSeed]
    var source: String
    var framesUsed: Int
    var samplesConsidered: Int
    var samplesRejected: Int
    var medianSpacingMeters: Float
    /// Set only by the depth-map path, which is the only path that draws this
    /// line here. `nil` on the pre-pass path, which drew its own and already
    /// recorded it in `PrePassCensus.seeding`. Optional rather than zeroed so
    /// that nothing downstream can print a cut for a run that never took one.
    var trustCut: TrainerSeedTrustCut?

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
        // A spacing of zero means it could not be measured, not that the points
        // are on top of each other, so the sentence simply stops rather than
        // telling the owner something that was never true. The upper guard is
        // belt and braces on a Float-to-Int conversion that traps out of range.
        guard medianSpacingMeters > 0, medianSpacingMeters < 1_000 else {
            return "Started from \(seeds.count) points taken from \(framesUsed) photos."
        }
        let millimetres = Int((medianSpacingMeters * 1000).rounded())
        return "Started from \(seeds.count) points taken from \(framesUsed) photos, "
            + "spaced about \(millimetres) mm apart."
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

    /// The fraction of THIS scan's occupied voxels that the depth-map seeder is
    /// willing to call trusted, before the floor below is applied.
    ///
    /// The same 0.5 `PrePassInitialSplatBuilder.Settings.trustedQuantile` uses,
    /// and the same reasoning: the useful part of a trust field is the ORDERING
    /// it measured, not its absolute level, because the level is dominated by
    /// how steady the hands were. The better half of a shaky scan is still
    /// meaningfully better than the worse half of it.
    static let trustedQuantile: Float = 0.5

    /// The absolute floor under that quantile, on the COMBINED score
    /// (`trustWeight * authority`) this seeder ranks cells by.
    ///
    /// 0.08 is not a taste. It is the pre-pass's own trust-weight floor of 0.20
    /// carried onto this scale exactly. Authority contains
    /// `0.25 + 0.75 * trustWeight`, so a sample at weight 0.20 whose range,
    /// glass, saturation and parallax ramps are all perfect scores
    ///
    ///     0.20 * (0.25 + 0.75 * 0.20) = 0.20 * 0.40 = 0.08
    ///
    /// and the inverse holds: 0.08 on the product maps back to exactly 0.20 on
    /// the weight. So the two seeders now insist on the same evidence.
    ///
    /// One asymmetry, said out loud rather than left to be discovered: when the
    /// authority map is missing but the trust field is not, `authority` falls
    /// back to a flat 0.5, so this floor maps to a trust weight of 0.16 rather
    /// than 0.20. The quantile dominates in that case anyway, which is the
    /// point of making the decision relative.
    ///
    /// The floor is load-bearing here, not decoration. `rangeAuthority` is
    /// `smoothdrop(4.5, 5.5, z)` while the far regime does not start until
    /// 30 m, so every surface past 5.5 m is kept by this seeder and scores
    /// exactly zero. In a big room that can be most of the cells, and without a
    /// floor the median of mostly-zeros would happily call zero "the better
    /// half" and pin a wall of blind guesses.
    static let trustedCombinedFloor: Float = 0.08

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
    /// That was the rule until the LiDAR-initialisation finding below. It is
    /// left in place because it explains what the numbers used to mean, not
    /// because it still describes what this function does.
    ///
    /// REVISED. The `cap / 2` above was the operative branch for every cap
    /// this app produces (the `min(20_000, ...)` term is dead for any cap
    /// above 40,000), so a room threw away 84 per cent of the LiDAR seeds -
    /// 888,951 measured points reduced to 150,000 by a uniform stride that
    /// is spatially arbitrary, not quality-ranked - and then spent the first
    /// 600 iterations cloning the population back up to the cap it had just
    /// been cut below. The header's rationale, "a seeder that fills the splat
    /// cap leaves densification nothing to do", is true of a random or SfM
    /// initialisation and false of a metric depth sensor: the points being
    /// discarded were MEASURED, and what replaces them is guessed.
    ///
    /// PocketGS (arXiv:2601.17354, iPhone 15 / A16, 500 iterations) ablates
    /// prior-conditioned initialisation and reports about 1.2 dB of PSNR lost
    /// AND runtime rising from 255.2 s to 319.5 s without it. We are a
    /// partial version of that ablation, keeping half rather than none.
    ///
    /// `fillFraction` is the switch: pass 0.5 to restore the old behaviour.
    /// The cost is real and should be expected in the census - the first few
    /// hundred iterations now carry the full cap instead of half of it, which
    /// the finding priced at roughly +4 s on a 65 s run.
    /// MEASURED, AND REVERTED. Build 182 ran with fillFraction 1.0 and the
    /// census settles the argument the finding and the file header were
    /// having: the header was right.
    ///
    /// Growth collapsed from 152,991 created Gaussians to 6,900. Filling the
    /// cap at iteration 0 leaves `splatCap - splatCount` at zero, so
    /// densification has no slots and never runs, which means the 150,000
    /// Gaussians that USED to be placed by gradient - where the photographs
    /// say detail is missing - were replaced by 150,000 more LiDAR points
    /// placed by an arbitrary voxel stride. Same final count, worse
    /// placement. Held-out PSNR fell 17.30 to 13.69 while TRAINED-view rose
    /// 19.46 to 20.65: more raw capacity, fitted to the training views and
    /// generalising worse. Training also rose 71 s to 92 s, because the run
    /// now carries the full cap from iteration 0 instead of growing into it.
    ///
    /// PocketGS's 1.2 dB init ablation is real, but it is about the QUALITY
    /// of the initialisation, not about spending the whole budget on it.
    /// 0.5 restores the split the depth-map path was already sized for.
    static func seedTarget(forSplatCap cap: Int, fillFraction: Float = 0.5) -> Int {
        guard cap > 0 else { return 0 }
        // Clamped to 0.05...1 BEFORE the conversion, and NaN takes the `else`
        // branch because every comparison against NaN is false. So the product
        // is at most `cap`, which is an Int already, and the rounded Double
        // cannot be non-finite or out of Int's range.
        let f = fillFraction.isFinite
            ? Double(Swift.min(Swift.max(fillFraction, 0.05), 1))
            : 1.0
        let wanted = Int((Double(cap) * f).rounded())
        return Swift.max(1, Swift.min(cap, wanted))
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

        // `settings` is not forwarded. The depth-map seeder used to read
        // `settings.minimumAuthorityForDepth` in its trust gate, where it was
        // both the wrong quantity and a clause that could never bind; the gate
        // now draws its line from this scan's own trust distribution. The
        // parameter stays on this entry point because it is the trainer's
        // contract with the seeder and the pre-pass path may want it back.
        let seeded = seedFromDepth(
            bundle: bundle,
            prePass: prePass,
            at: ref,
            keyframes: keyframes,
            budget: budget,
            trust: trust,
            authority: authority,
            edges: edges
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
        // Said out loud. The pre-pass result can be REUSED from an earlier
        // session (`ScanProcessingCoordinator` takes the saved one when the
        // intent allows), so a reference can outlive the file it points at, and
        // a silent `nil` here is the difference between "the fallback seeder
        // ran" and nobody ever knowing why.
        guard FileManager.default.fileExists(atPath: url.path) else {
            TrainerLog.general.error(
                "The pre-pass recorded starting points at \(refs.path, privacy: .public) but the file is not there; seeding from the depth maps instead"
            )
            return nil
        }
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
            medianSpacingMeters: spacing,
            // The pre-pass drew its own trusted/doubtful line and recorded it
            // in `PrePassCensus.seeding`; these flags came out of that decision
            // already made. Nothing here re-decides it, so there is no cut to
            // report and this must not pretend otherwise.
            trustCut: nil
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
        edges: NativeDepthEdgeClassifier?
    ) -> TrainerSeedResult {

        let depthWidth = bundle.settings.depthWidth
        let depthHeight = bundle.settings.depthHeight
        let sampleCount = depthWidth * depthHeight
        guard sampleCount > 0, !keyframes.isEmpty else {
            return TrainerSeedResult(
                seeds: [], source: depthSeedSourceName, framesUsed: 0,
                samplesConsidered: 0, samplesRejected: 0, medianSpacingMeters: 0,
                trustCut: nil
            )
        }

        let nativeK = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: depthWidth, depthHeight: depthHeight
        )
        let lidarMaxRange = bundle.settings.lidarMaxRangeMeters

        // Voxel size: aim for one seed per voxel and as many voxels as the
        // thinning below will actually keep, from the scene's measured extent.
        // Never a fixed 1 cm.
        // The measured extent when the capture recorded one, otherwise the
        // median camera-to-surface distance from the QC card, which is a
        // measurement too. Never a constant.
        let extent = Swift.max(
            bundle.sceneBounds?.longestEdgeMeters
                ?? (prePass.qcCard.medianCameraToSurfaceMeters * 3),
            0.5
        )
        // `seedTarget`, the SAME number the thinning below reduces to. It used
        // to be `max(splatCap / 2, 20_000)`, which has no three-quarter ceiling
        // and so disagrees with `seedTarget` for every cap under about 26,600.
        // At the 5,000 emergency floor that is 20,000 against 3,750: the voxel
        // was sized to produce 5.3 times more cells than would survive, the
        // thinning threw away 81 per cent of them, and `medianSpacingMeters`
        // (measured BEFORE the thinning) came out about 2.3 times smaller than
        // the spacing the seeds actually ended up at. Every disc radius is cut
        // from that spacing, so the whole set was seeded 2.3x too small and the
        // holes between them were densification's problem to find.
        let targetSeeds = seedTarget(forSplatCap: budget.splatCap)
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
                    // Finite AND inside a plausible world, not merely finite.
                    // `voxel` floors at 0.005 m, so the division below
                    // multiplies this position by up to two hundred before it
                    // reaches an `Int32`, and `Int32(someFloat)` TRAPS out of
                    // range in release as well as in debug. The depth affine
                    // and the pose both arrive from disk and can be finite and
                    // still absurd. A position we cannot trust must not be
                    // allowed to mark a real part of the room, so it is
                    // rejected here rather than clamped onto a real voxel.
                    guard world.x.isFinite, world.y.isFinite, world.z.isFinite,
                          world.x.magnitude < 1_000_000,
                          world.y.magnitude < 1_000_000,
                          world.z.magnitude < 1_000_000
                    else {
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

                    // `PrePassVoxelFrame.cellIndex`, not a bare `Int32(...)`.
                    // The same non-trapping floor-and-clamp the pre-pass reader
                    // uses, called rather than copied: three bare conversions
                    // exactly like these were the crash in five separate files.
                    let key = SIMD3<Int32>(
                        PrePassVoxelFrame.cellIndex(world.x / voxel),
                        PrePassVoxelFrame.cellIndex(world.y / voxel),
                        PrePassVoxelFrame.cellIndex(world.z / voxel)
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
                medianSpacingMeters: 0, trustCut: nil
            )
        }

        // Local spacing, measured rather than assumed: for each occupied
        // voxel, the distance to the nearest occupied neighbour in the 26-cell
        // ring, falling back to the voxel edge when the cell stands alone.
        var seeds: [TrainerSeed] = []
        seeds.reserveCapacity(cells.count)
        var spacings: [Float] = []
        spacings.reserveCapacity(cells.count)

        // --- Where the trusted / doubtful line falls FOR THIS SCAN -------------
        //
        // Computed once over every occupied voxel, before a single seed is
        // shaped, so it is a property of the scan rather than of the order the
        // dictionary happens to enumerate in. Same rule as the pre-pass
        // seeder's `trustCut`: a quantile of the distribution actually
        // observed, held up by a low absolute floor.
        //
        // The same guard as the shaping loop below, so the count here and the
        // number of seeds are taken over the same population.
        var trustScores: [Float] = []
        trustScores.reserveCapacity(cells.count)
        for cell in cells.values where cell.weightSum > 1e-6 {
            if cell.bestTrust.isFinite { trustScores.append(cell.bestTrust) }
        }
        trustScores.sort()

        func trustQuantile(_ q: Float) -> Float {
            guard !trustScores.isEmpty else { return 0 }
            let clamped = Swift.max(0, Swift.min(1, q))
            let index = Swift.min(
                trustScores.count - 1,
                Swift.max(0, Int(Float(trustScores.count - 1) * clamped))
            )
            return trustScores[index]
        }

        // No trust field means no evidence about which samples are better, and
        // the honest shape for evidence-free geometry is the cheap one: a
        // translucent blob the optimiser can delete, not a pinned disc it can
        // barely move. `PrePassInitialSplatBuilder` lands in the same place
        // when its trust field is missing, because it feeds zero weights in and
        // nothing then clears its floor.
        //
        // This case matters here specifically. Without the flag, a missing
        // trust field would send every `trustWeight` and every `authority` to
        // their 0.5 fallbacks, so EVERY cell would score exactly 0.25, the
        // median would be 0.25, and `>= 0.25` would be true for all of them:
        // a quantile alone would pin the entire scan as solid on the strength
        // of no measurement whatsoever. That is the old fault inverted, and it
        // would be just as quiet.
        //
        // The two failures are independent, before anyone reads a link into
        // this: a pre-pass with no trust field still writes an initial-splat
        // set (its builder keeps zero-weight samples, it only refuses to call
        // them trusted), so a missing trust field does not by itself send the
        // trainer down this path.
        let trustWasMeasured = trust != nil
        let trustCut = Swift.max(
            trustQuantile(TrainerInitializer.trustedQuantile),
            TrainerInitializer.trustedCombinedFloor
        )

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

            // One gate, on one quantity, against a line drawn from that same
            // quantity's own distribution. `settings.minimumAuthorityForDepth`
            // is deliberately NOT in here any more: it is a floor on AUTHORITY,
            // `bestTrust` is trust times authority, and since trust is at most
            // 1 the product can never exceed the authority. Any clause pairing
            // the two is therefore either wrong or dead, and the one that was
            // here was dead.
            let trusted = trustWasMeasured && cell.bestTrust >= trustCut
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

        let cutRecord = TrainerSeedTrustCut(
            wasMeasured: trustWasMeasured,
            cut: trustCut,
            floor: TrainerInitializer.trustedCombinedFloor,
            quantile: TrainerInitializer.trustedQuantile,
            p05: trustQuantile(0.05),
            median: trustQuantile(0.5),
            p95: trustQuantile(0.95),
            cellsConsidered: trustScores.count
        )
        // Written whether or not anything cleared the line, because "the cut
        // was 0.08 and the scan's 95th percentile was 0.02" is the sentence
        // that would have ended the old gate's career on the first run.
        //
        // Built as a plain String, a piece at a time: an `os.Logger` message is
        // a literal with its own interpolation type and two of them cannot be
        // joined, and a long `+` chain of interpolated literals is one of the
        // shapes this project's type checker has given up on before. A log line
        // is not worth a build failure.
        var cutLine = "Seed trust line: "
        if trustWasMeasured {
            cutLine += "cut \(cutRecord.cut)"
            cutLine += " (floor \(cutRecord.floor), quantile \(cutRecord.quantile))"
            cutLine += " over \(cutRecord.cellsConsidered) cells;"
            cutLine += " p05 \(cutRecord.p05)"
            cutLine += ", median \(cutRecord.median)"
            cutLine += ", p95 \(cutRecord.p95)"
        } else {
            cutLine += "this scan had no trust field, so all "
            cutLine += "\(cutRecord.cellsConsidered) starting points were laid "
            cutLine += "as stretched blobs rather than pinned discs"
        }
        TrainerLog.general.info("\(cutLine, privacy: .public)")

        return TrainerSeedResult(
            seeds: seeds,
            source: depthSeedSourceName,
            framesUsed: framesUsed,
            samplesConsidered: considered,
            samplesRejected: rejected,
            medianSpacingMeters: median,
            trustCut: cutRecord
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
    ///
    /// It also RESIZES what it keeps. Every seed's radius is cut from its own
    /// local sample spacing (`PrePassInitialSplats`: `radius = 0.5 * max(
    /// spacing, sensorSpacing)`), which is the spacing of the FULL set. Drop
    /// one seed in N and the surviving points are sqrt(N) further apart, so
    /// discs sized for the dense set leave holes between them that
    /// densification then has to find and fill - and the fallback depth
    /// seeder in this same file already carries a comment describing this
    /// exact bug and saying it was fixed there. It was never fixed on the
    /// preferred path.
    ///
    /// Only the two largest axes are grown. The third is the disc's normal,
    /// which is set from sensor noise and has nothing to do with how far
    /// apart the samples are; scaling it too would inflate flat surfaces into
    /// slabs.
    private static func thin(_ seeds: [TrainerSeed], to cap: Int) -> [TrainerSeed] {
        guard cap > 0, seeds.count > cap else { return seeds }
        let keepRatio = Double(cap) / Double(seeds.count)
        let growth = logf(sqrtf(Float(seeds.count) / Float(cap)))
        var kept: [TrainerSeed] = []
        kept.reserveCapacity(cap)
        var accumulator = 0.0
        for seed in seeds {
            accumulator += keepRatio
            if accumulator >= 1 {
                accumulator -= 1
                var resized = seed
                if growth > 0 {
                    // Grow the two in-plane axes; leave the normal alone.
                    let s = resized.logScale
                    var order = [0, 1, 2]
                    order.sort { s[$0] > s[$1] }
                    resized.logScale[order[0]] = s[order[0]] + growth
                    resized.logScale[order[1]] = s[order[1]] + growth
                }
                kept.append(resized)
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

        // A coarse hash makes "nearest neighbour" a local question, and the
        // cell has to be MEASURED rather than fixed, for the same reason every
        // other scale in this file is.
        //
        // The search only looks at the 26 neighbouring cells, so a cell narrower
        // than the spacing throws the answer away: at the 0.05 m constant that
        // used to be here, a set spaced 0.12 m apart (exactly what the pre-pass
        // clamps to, before the thinning above widens it further) has its
        // nearest neighbour three cells away, outside the ring, and every
        // sample is dropped. The median is then taken over the closest pairs
        // only, or over nothing at all, and "spaced about 0 mm apart" reaches
        // the owner's screen and the census as though it had been measured.
        //
        // Seeds sit on surfaces, so the count grows with the SQUARE of extent
        // over spacing, the same relation the depth seeder sizes its voxels by.
        // Two estimated spacings per cell, clamped either side.
        var lo = seeds[0].position
        var hi = seeds[0].position
        for seed in seeds {
            lo = simd_min(lo, seed.position)
            hi = simd_max(hi, seed.position)
        }
        let span = hi - lo
        let longestEdge = Swift.max(span.x, Swift.max(span.y, span.z))
        let estimatedSpacing = longestEdge / Swift.max(sqrtf(Float(seeds.count)), 1)
        let cell = Swift.max(0.02, Swift.min(1.0, estimatedSpacing * 2))

        var grid: [SIMD3<Int32>: [Int]] = [:]
        for (i, seed) in seeds.enumerated() {
            let p = seed.position
            // A pre-pass set read back off disk can carry NaN positions: any
            // scan recorded before the capture tracking gate landed still has
            // them in `prepass/init_splats.ply`, and this is the PREFERRED
            // seeding path, so it is the ordinary case rather than the exotic
            // one. `Int32(someFloat)` TRAPS on NaN, which killed the trainer
            // here while it was only measuring spacing for a log line.
            // A seed with no trustworthy position is not the nearest neighbour
            // of anything, so it is left out of the grid entirely rather than
            // being given a bucket of its own.
            guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
            let key = SIMD3<Int32>(
                PrePassVoxelFrame.cellIndex(p.x / cell),
                PrePassVoxelFrame.cellIndex(p.y / cell),
                PrePassVoxelFrame.cellIndex(p.z / cell)
            )
            grid[key, default: []].append(i)
        }

        var distances: [Float] = []
        let sampleStride = Swift.max(seeds.count / 2000, 1)
        var i = 0
        while i < seeds.count {
            let position = seeds[i].position
            // Same reason as the grid above, one stack frame earlier than the
            // `best < greatestFiniteMagnitude` test at the bottom of the loop:
            // that test could never run, because the conversion below had
            // already trapped on the NaN.
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite else {
                i += sampleStride
                continue
            }
            let key = SIMD3<Int32>(
                PrePassVoxelFrame.cellIndex(position.x / cell),
                PrePassVoxelFrame.cellIndex(position.y / cell),
                PrePassVoxelFrame.cellIndex(position.z / cell)
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
            // `best < greatestFiniteMagnitude`, NOT `best.isFinite`. The
            // sentinel this starts at IS finite, so the old test appended
            // 3.4e38 metres as a measured distance every time a seed had no
            // neighbour inside the ring. More than half the samples like that
            // and the median came back as 3.4e38, which
            // `Int((median * 1000).rounded())` in `summary` and in
            // `MetalSplatTrainer` then converts, and a Float-to-Int conversion
            // out of range TRAPS in release as well as debug. A sparse pre-pass
            // set in a large room was one crash away, on the preferred seeding
            // path, and the only symptom would have been the trainer dying at
            // the line that logs how it started. An infinity from a corrupt
            // position is dropped by the same test.
            if best < Float.greatestFiniteMagnitude { distances.append(best) }
            i += sampleStride
        }
        // Empty means the answer is UNKNOWN, not that the seeds are on top of
        // each other. `summary` says nothing about spacing when it reads zero,
        // rather than telling the owner they are "about 0 mm apart".
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
