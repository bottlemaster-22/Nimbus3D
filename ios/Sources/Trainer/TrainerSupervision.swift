//
//  TrainerSupervision.swift
//  Trainer
//
//  WHAT ONE KEYFRAME SUPERVISES, BUILT ON THE CPU ONCE PER USE.
//
//  The GPU depth-loss kernel is a flat loop over an array with no branching on
//  module state. That is a deliberate split, stated at the top of
//  `TrainerDepthSample` in TrainerGPULayouts.swift: everything that needs a
//  protocol call (`TrustField.weight`, `BackgroundModel.authority`,
//  `EdgeClassifier.map`) happens here, once, and lands in a plain array.
//
//  THE RULES THIS FILE ENFORCES (F3, F5, F6), each one a thing a generic 3DGS
//  trainer gets wrong:
//
//   1. DEPTH IS SUPERVISED ONLY AT THE NATIVE SAMPLES. 256x192 real returns,
//      not the 1920-wide map ARKit will hand you. Everything between them is
//      the interpolator's opinion, and training against an interpolator
//      teaches the model to reproduce the interpolator.
//
//   2. THE DILATED EDGE BAND IS ZEROED, NOT DOWN-WEIGHTED. Inside the band the
//      upsampled depth is WRONG rather than noisy, and averaging a wrong value
//      in at low weight still moves the surface.
//
//   3. THE AUTHORITY MAP DECIDES WHO OWNS A PIXEL. Where authority is at or
//      below the floor - past LiDAR range, at glass, at a saturated window, or
//      with no parallax - the pixel gets NO depth supervision and is left to
//      the background model. Pulling those pixels to a wrong distance is the
//      exact failure the authority map exists to prevent.
//
//   4. UNKNOWN IS MARKED EVEN WHEN IT CARRIES NO WEIGHT, because the unknown
//      mask is what switches late opacity binarization off for the Gaussians
//      that land there.
//
//   5. FREE SPACE IS EVIDENCE. A return at 2.4 m certifies the 2.35 m in front
//      of it as empty, which becomes a hinge: the rendered surface may not be
//      nearer than that, whatever the photometry would prefer. A no-return
//      sample borrows the bound from its valid neighbours, which is how the
//      space in front of a window is recovered without inventing anything
//      beyond the glass.
//

import Foundation
import simd

/// Everything one training step needs about one keyframe.
struct TrainerFrameSupervision {
    var frame: FrameID
    /// Ground-truth RGB at the render resolution, three floats per pixel.
    var groundTruth: [Float]
    /// Background radiance at the render resolution, three floats per pixel.
    /// Empty when no background model is available, and `hasBackground` is
    /// then false rather than a black image being passed off as a far field.
    var background: [Float]
    var hasBackground: Bool
    var depthSamples: [TrainerDepthSample]
    var renderSize: TrainerRenderSize
    /// The intrinsics rescaled to `renderSize`.
    var intrinsics: CameraIntrinsics
    var pose: Pose
    /// The frame's own QC weight, 0...1.
    var qcWeight: Float
    /// Mean per-pixel authority over this frame, for the log.
    var meanAuthority: Float
    /// How many native samples actually carry depth weight. Zero is a real and
    /// reportable answer for a frame shot entirely through a window.
    var supervisedSampleCount: Int
}

/// Builds `TrainerFrameSupervision`, and caches the decoded images so a
/// keyframe visited twice in a row is not decoded twice.
final class TrainerSupervisionBuilder {

    private let bundle: CaptureBundle
    private let prePass: PrePassResult
    private let ref: CaptureBundleRef
    private let settings: SmartLossSettings
    private let tuning: TrainerTuning

    private let trust: TwoScaleTrustField?
    private let authority: SmartAuthorityMap?
    private let edges: NativeDepthEdgeClassifier?
    private let background: DirectionalBackgroundModel?

    private var imageCache: SmartImageCache
    private let depthCache: SmartDepthCache

    private let depthWidth: Int
    private let depthHeight: Int
    private let nativeIntrinsics: CameraIntrinsics
    private let lidarMaxRange: Float

    /// Fixed once the first frame is decoded, so every frame in a run shares a
    /// pixel grid and the render buffers never have to be reshaped mid-loop.
    private(set) var renderSize: TrainerRenderSize?
    private(set) var renderIntrinsics: CameraIntrinsics?

    /// Set once a photo has come back a different shape from the camera that
    /// describes it, so that failure is reported once instead of once per
    /// iteration.
    private var loggedShapeMismatch = false

    /// The long edge the photos are decoded at. Not fixed for the life of the
    /// run: the budget governor can lower the render resolution mid-run to cool
    /// the phone or free memory, and the supervision grid has to follow it down
    /// (see `lowerLongEdge(to:)`).
    private var requestedLongEdge: Int

    init(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        longEdgePixels: Int,
        settings: SmartLossSettings,
        tuning: TrainerTuning,
        trust: TwoScaleTrustField?,
        authority: SmartAuthorityMap?,
        edges: NativeDepthEdgeClassifier?,
        background: DirectionalBackgroundModel?
    ) {
        self.bundle = bundle
        self.prePass = prePass
        self.ref = ref
        self.settings = settings
        self.tuning = tuning
        self.trust = trust
        self.authority = authority
        self.edges = edges
        self.background = background
        self.requestedLongEdge = Swift.max(longEdgePixels, 64)

        depthWidth = Swift.max(bundle.settings.depthWidth, 1)
        depthHeight = Swift.max(bundle.settings.depthHeight, 1)
        lidarMaxRange = bundle.settings.lidarMaxRangeMeters
        nativeIntrinsics = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: depthWidth, depthHeight: depthHeight
        )

        // Three frames of cache: the current one, the one the densifier may
        // re-render, and one spare. A house scan's frames are 2 MB decoded at
        // 720 px, so this is single-digit megabytes, not a gamble.
        imageCache = SmartImageCache(capacity: 3, longEdge: self.requestedLongEdge)
        depthCache = SmartDepthCache(capacity: 3, sampleCount: depthWidth * depthHeight)
    }

    /// Adopts a smaller supervision grid part way through a run.
    ///
    /// The budget governor lowers the render resolution when the phone is hot
    /// or short of memory, and it resizes the GPU buffers itself. It cannot
    /// resize THIS, and this is where the pixel grid is actually decided: the
    /// grid is fixed from the first photo decoded and every later frame is
    /// measured against it. Without this call the next frame arrives at the old
    /// size, the training loop grows the buffers straight back to match it, and
    /// the cut that was made to cool the phone is undone a moment later, over
    /// and over, with a full buffer reallocation each time.
    ///
    /// Only downward, and only when it really is smaller: the governor's
    /// resolution ladder is one way, and re-decoding a run's photos larger part
    /// way through would spend memory at exactly the moment there is none.
    ///
    /// The cached decodes are dropped with the old grid, because they are at
    /// the old size. Returns whether anything changed, for the caller's log.
    @discardableResult
    func lowerLongEdge(to pixels: Int) -> Bool {
        let target = Swift.max(pixels, 64)
        guard target < requestedLongEdge else { return false }
        requestedLongEdge = target
        imageCache = SmartImageCache(capacity: 3, longEdge: target)
        // Cleared, not recomputed: the next decodable frame fixes the new grid
        // and rescales the intrinsics to it, by the same path the first frame
        // of the run took.
        renderSize = nil
        renderIntrinsics = nil
        return true
    }

    /// The pose the trainer should render this frame from: the pre-pass's
    /// refined pose when there is one, otherwise the frame's own.
    func pose(for frame: CaptureFrame) -> Pose {
        prePass.refinedPose(for: frame.index) ?? frame.refinedPose ?? frame.rawPose
    }

    /// Decodes one keyframe and builds everything the GPU needs from it.
    /// Returns nil when the photo could not be read, which is logged and
    /// skipped rather than substituted with grey.
    func build(
        frame: CaptureFrame,
        iteration: Int,
        totalIterations: Int
    ) -> TrainerFrameSupervision? {

        guard let image = imageCache.image(for: frame, at: ref) else {
            TrainerLog.general.error(
                "Photo for frame \(frame.index) could not be read; that frame is skipped"
            )
            return nil
        }

        let size = TrainerRenderSize(width: image.width, height: image.height)
        if renderSize == nil {
            // The photo has to be the same SHAPE as the intrinsics that
            // describe it. `scaled(toWidth:height:)` scales fx and fy by
            // width and height separately, so a photo that came back turned a
            // quarter turn (a landscape sensor frame decoded as portrait,
            // which is what an EXIF orientation tag would do) is not caught by
            // that scale: it is absorbed into a camera that is wrong in both
            // axes, and the run trains against it without ever complaining.
            //
            // Capture writes its JPEGs with no orientation tag precisely so
            // that pixels, intrinsics and poses stay in one frame, so this
            // should never fire. If it ever does, the cause is upstream and
            // saying so is far better than quietly fitting a wrong camera.
            let sourceWidth = bundle.intrinsics.width
            let sourceHeight = bundle.intrinsics.height
            let sourceAspect = Float(sourceWidth) / Float(Swift.max(sourceHeight, 1))
            let decodedAspect = Float(size.width) / Float(Swift.max(size.height, 1))
            guard sourceAspect > 0,
                  abs(decodedAspect - sourceAspect) <= sourceAspect * 0.02
            else {
                // Said once. The grid stays unfixed while this is true, so
                // every frame of the run comes back here, and a line per
                // iteration would bury the one line that matters.
                if !loggedShapeMismatch {
                    loggedShapeMismatch = true
                    let frameIndex = frame.index
                    TrainerLog.general.error(
                        """
                        Photo for frame \(frameIndex) decoded \(size.width)x\(size.height), \
                        which is not the shape of the capture's \
                        \(sourceWidth)x\(sourceHeight) camera; those frames are skipped
                        """
                    )
                }
                return nil
            }
            renderSize = size
            renderIntrinsics = bundle.intrinsics.scaled(
                toWidth: image.width, height: image.height
            )
        }
        guard let fixedSize = renderSize, let k = renderIntrinsics else { return nil }
        // Every frame of a capture comes from the same camera at the same
        // resolution, so this should never fire. If it does, the frame is
        // skipped rather than sampled into the wrong grid.
        guard size == fixedSize else {
            TrainerLog.general.error(
                "Frame \(frame.index) decoded at \(size.width)x\(size.height), not \(fixedSize.width)x\(fixedSize.height); skipped"
            )
            return nil
        }

        let pixelCount = fixedSize.pixelCount
        var groundTruth = [Float](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            let rgb = image.rgb[i]
            groundTruth[i * 3 + 0] = rgb.x
            groundTruth[i * 3 + 1] = rgb.y
            groundTruth[i * 3 + 2] = rgb.z
        }

        let framePose = pose(for: frame)

        var backgroundPixels: [Float] = []
        var hasBackground = false
        if let background {
            backgroundPixels = backgroundImage(
                background: background, pose: framePose, intrinsics: k, size: fixedSize
            )
            hasBackground = true
        }

        let samples = depthSamples(
            frame: frame,
            pose: framePose,
            renderIntrinsics: k,
            size: fixedSize,
            iteration: iteration,
            totalIterations: totalIterations
        )

        let supervised = samples.reduce(into: 0) { $0 += ($1.weight > 0 ? 1 : 0) }
        let meanAuthority = authority?.map(for: frame.index)?.meanAuthority ?? 0

        return TrainerFrameSupervision(
            frame: frame.index,
            groundTruth: groundTruth,
            background: backgroundPixels,
            hasBackground: hasBackground,
            depthSamples: samples,
            renderSize: fixedSize,
            intrinsics: k,
            pose: framePose,
            qcWeight: TrainerMath.clamp(frame.qc.weight, 0, 1),
            meanAuthority: meanAuthority,
            supervisedSampleCount: supervised
        )
    }

    // MARK: - Depth supervision

    // swiftlint:disable:next function_body_length - one pass over the native
    // samples with every F3/F5/F6 rule visible in order; splitting it would
    // hide the ordering, which is the part that matters.
    private func depthSamples(
        frame: CaptureFrame,
        pose: Pose,
        renderIntrinsics k: CameraIntrinsics,
        size: TrainerRenderSize,
        iteration: Int,
        totalIterations: Int
    ) -> [TrainerDepthSample] {

        guard frame.depthPath != nil,
              let depth = depthCache.depth(for: frame, at: ref)
        else { return [] }

        let edgeMap = edges?.map(for: frame.index) ?? []
        let affine = trust?.depthAffine(frame: frame.index) ?? .identity
        let qcWeight = TrainerMath.clamp(frame.qc.weight, 0, 1)
        let modeRadius = Swift.max(settings.modeWindowRadius, 1)
        let freeSpaceMargin = settings.freeSpaceBoundMarginMeters

        var samples: [TrainerDepthSample] = []
        samples.reserveCapacity(depthWidth * depthHeight / 2)

        for v in 0..<depthHeight {
            for u in 0..<depthWidth {
                let index = v * depthWidth + u

                // Where does this native sample land on the render grid? The
                // depth sensor is registered to the colour camera by ARKit, so
                // this is a pure rescale, not a reprojection.
                let px = Int((Float(u) + 0.5) * Float(size.width) / Float(depthWidth))
                let py = Int((Float(v) + 0.5) * Float(size.height) / Float(depthHeight))
                guard px >= 0, px < size.width, py >= 0, py < size.height else { continue }
                let pixelIndex = py * size.width + px

                let rawEdge: EdgeClass = index < edgeMap.count ? edgeMap[index] : .none
                let regime = authority?.regime(frame: frame.index, sampleIndex: index) ?? .near
                let authorityValue = authority?.authority(frame: frame.index, sampleIndex: index)
                    ?? 0.5

                var sample = TrainerDepthSample()
                sample.pixelIndex = UInt32(pixelIndex)
                sample.edgeClass = UInt32(rawEdge.rawValue)

                let raw = depth[index]
                let hasReturn = raw > 0.05 && raw < lidarMaxRange
                let z = hasReturn ? affine.apply(raw) : 0

                // RULE 3 and RULE 4. Anything the authority map does not back
                // is UNKNOWN: it is marked so binarization is switched off for
                // whatever Gaussians cover it, and it carries no weight, so the
                // far field keeps ownership of the pixel.
                let belowAuthorityFloor = authorityValue <= settings.minimumAuthorityForDepth
                if !hasReturn || regime != .near || belowAuthorityFloor || rawEdge == .unknown {
                    sample.edgeClass = UInt32(EdgeClass.unknown.rawValue)
                    sample.weight = 0
                    sample.depth = 0
                    // RULE 5 for a no-return: borrow a bound from the valid
                    // neighbours, which certifies only the space in FRONT of
                    // the nearest known surface around it and nothing beyond.
                    if !hasReturn {
                        sample.freeSpaceBound = borrowedFreeSpaceBound(
                            depth: depth, u: u, v: v, margin: freeSpaceMargin, affine: affine
                        )
                    }
                    samples.append(sample)
                    continue
                }

                sample.depth = z

                // RULE 2. Zero inside the dilation band, and keep the class so
                // the GPU kernel takes the same branch.
                if rawEdge == .band {
                    sample.weight = 0
                    sample.freeSpaceBound = Swift.max(z - freeSpaceMargin, 0)
                    samples.append(sample)
                    continue
                }

                // The trust weight is inverse-variance and soft by
                // construction; the authority is the per-pixel right to be
                // believed; the QC weight is how good the photo was. All three
                // multiply, which is what the depth-sample contract asks for.
                let trustWeight = trust?.weight(frame: frame.index, sampleIndex: index) ?? 0.5
                var weight = trustWeight * authorityValue * qcWeight
                if rawEdge == .geometric {
                    // The sharpen half of the WHERE/WHAT split: a real depth
                    // step is the most informative sample in the frame.
                    weight *= tuning.geometricEdgeBoost
                }
                sample.weight = TrainerMath.clamp(weight, 0, 4)

                // Huber transition grows with range: 2 cm at 40 cm and 2 cm at
                // 5 m are not the same event.
                let sigma = trust?.sigmaMeters(frame: frame.index, sampleIndex: index)
                sample.huberDelta = Swift.max(
                    sigma ?? (settings.huberDeltaMeters * Swift.max(z / 2, 1)),
                    0.004
                )

                // RULE 5 for a valid return.
                sample.freeSpaceBound = Swift.max(z - freeSpaceMargin, 0)

                // F4 bimodal supervision, and only at a real depth step. Away
                // from one, both modes are the sample itself, which makes the
                // bimodal term vanish exactly rather than approximately.
                if rawEdge == .geometric {
                    let modes = localModes(
                        depth: depth, u: u, v: v, radius: modeRadius, affine: affine
                    )
                    sample.mode0 = modes.near
                    sample.mode1 = modes.far
                } else {
                    sample.mode0 = z
                    sample.mode1 = z
                }

                samples.append(sample)
            }
        }
        return samples
    }

    /// The two local depth modes around a geometric edge: the mean of the
    /// nearer half and the mean of the farther half of the window's valid
    /// returns, split at the midpoint of its range. A two-cluster split rather
    /// than a histogram because the window is 25 samples and a histogram of 25
    /// values is noise with extra steps.
    private func localModes(
        depth: [Float],
        u: Int,
        v: Int,
        radius: Int,
        affine: SmartDepthAffine
    ) -> (near: Float, far: Float) {
        var values: [Float] = []
        values.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
        for dv in -radius...radius {
            for du in -radius...radius {
                let x = u + du, y = v + dv
                guard x >= 0, x < depthWidth, y >= 0, y < depthHeight else { continue }
                let raw = depth[y * depthWidth + x]
                guard raw > 0.05, raw < lidarMaxRange else { continue }
                values.append(affine.apply(raw))
            }
        }
        guard values.count >= 4 else {
            let centre = depth[v * depthWidth + u]
            let z = centre > 0 ? affine.apply(centre) : 0
            return (z, z)
        }
        let lo = values.min() ?? 0
        let hi = values.max() ?? 0
        let split = (lo + hi) * 0.5
        var nearSum: Float = 0, nearCount: Float = 0
        var farSum: Float = 0, farCount: Float = 0
        for value in values {
            if value <= split { nearSum += value; nearCount += 1 }
            else { farSum += value; farCount += 1 }
        }
        guard nearCount > 0, farCount > 0 else { return (lo, hi) }
        let near = nearSum / nearCount
        let far = farSum / farCount
        // The kernel only applies the bimodal term when mode1 > mode0, so a
        // window that turned out to be flat disables it by itself.
        return far > near ? (near, far) : (near, near)
    }

    /// The free-space bound for a sample with no return: the NEAREST valid
    /// return in the neighbourhood, less a margin. Nearest and not farthest,
    /// because the beam demonstrably reached at least as far as the closest
    /// thing around it and claiming more than that is inventing geometry.
    /// Returns 0, which disables the hinge, when nothing valid is nearby.
    private func borrowedFreeSpaceBound(
        depth: [Float],
        u: Int,
        v: Int,
        margin: Float,
        affine: SmartDepthAffine
    ) -> Float {
        let radius = 3
        var nearest = Float.greatestFiniteMagnitude
        for dv in -radius...radius {
            for du in -radius...radius {
                let x = u + du, y = v + dv
                guard x >= 0, x < depthWidth, y >= 0, y < depthHeight else { continue }
                let raw = depth[y * depthWidth + x]
                guard raw > 0.05, raw < lidarMaxRange else { continue }
                nearest = Swift.min(nearest, affine.apply(raw))
            }
        }
        guard nearest.isFinite else { return 0 }
        return Swift.max(nearest - margin, 0)
    }

    // MARK: - Background

    /// Rasterises the direction-only far field into the render grid, so the
    /// photometric kernel can composite it behind the Gaussians with the
    /// accumulated alpha it already has.
    ///
    /// One cubemap lookup per pixel at 480-720 px is a few hundred thousand
    /// trilinear fetches, which is milliseconds; it is not worth a shader.
    private func backgroundImage(
        background: DirectionalBackgroundModel,
        pose: Pose,
        intrinsics k: CameraIntrinsics,
        size: TrainerRenderSize
    ) -> [Float] {
        var pixels = [Float](repeating: 0, count: size.pixelCount * 3)
        let rotationInverse = pose.rotation.simd.inverse
        for y in 0..<size.height {
            for x in 0..<size.width {
                let ray = SmartCamera.ray(SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5), k)
                let worldRay = rotationInverse.act(ray)
                let radiance = background.radiance(forDirection: Vector3(worldRay))
                let i = (y * size.width + x) * 3
                pixels[i + 0] = radiance.x
                pixels[i + 1] = radiance.y
                pixels[i + 2] = radiance.z
            }
        }
        return pixels
    }
}
