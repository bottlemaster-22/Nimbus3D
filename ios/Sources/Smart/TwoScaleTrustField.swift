//
//  TwoScaleTrustField.swift
//  Smart
//
//  F6: trust, done at two scales that are never mixed.
//
//  ---------------------------------------------------------------------------
//  WHY TWO SCALES
//  ---------------------------------------------------------------------------
//  A depth sample can be wrong in two completely different ways, and lumping
//  them together destroys both signals.
//
//   * BIAS is spatial and systematic. A whole patch of wall reads 3 cm too
//     near because of the incidence angle, or a dark sofa reads long because
//     it swallows the emitted photons. That is estimable only by AVERAGING
//     over many samples in a neighbourhood, which is exactly what the coarse
//     25-50 cm voxel field does.
//   * NOISE / OUTLIER is per-sample. One return bounced off a mirror and came
//     back at 9 m. Averaging that into its neighbours is how you get a
//     confidently wrong surface, so the per-sample field is NEVER spatially
//     smoothed. Not as an optimisation choice - as a contract
//     (`docs/DATA_FORMAT.md` section 7).
//
//  The prior is physics (`SmartDepthNoiseModel`): error grows with range and
//  with grazing incidence. The likelihood is measurement: reproject each
//  sample into co-visible frames and look at the spread of what they say.
//  Where the budget allows, a targeted ZNCC plane sweep around the sample adds
//  a photometric opinion that is independent of the sensor entirely.
//
//  ARKit's own confidence is used, but only as a RANKING. It flags roughly
//  1.6% of samples low, which cannot be read as "1.6% of samples are wrong";
//  it is remapped through the residuals actually observed at each level.
//

import Foundation
import simd

// MARK: - Streaming writer

/// Appends to a file in bounded-memory chunks.
///
/// `trust_noise.bin` for a 500-frame scan is 500 * 49152 * 4 = 98 MB. Building
/// that as one `Data` and writing it in one shot is a straightforward way to
/// get jetsammed on a phone, so it is flushed every few megabytes instead.
final class SmartChunkedWriter {
    private let handle: FileHandle
    private var pending: Data
    private let flushThreshold: Int
    private(set) var bytesWritten: Int = 0

    init(url: URL, flushThreshold: Int = 4 << 20) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw SmartError.missingSidecar(url.lastPathComponent)
        }
        handle = try FileHandle(forWritingTo: url)
        pending = Data(capacity: flushThreshold + 4096)
        self.flushThreshold = flushThreshold
    }

    func append(_ data: Data) throws {
        pending.append(data)
        bytesWritten += data.count
        if pending.count >= flushThreshold { try flush() }
    }

    func appendFloats(_ values: [Float]) throws {
        var chunk = Data(capacity: values.count * 4)
        for v in values { SmartBinary.append(v, to: &chunk) }
        try append(chunk)
    }

    func flush() throws {
        guard !pending.isEmpty else { return }
        try handle.write(contentsOf: pending)
        pending.removeAll(keepingCapacity: true)
    }

    func close() throws {
        try flush()
        try handle.close()
    }
}

// MARK: - Per-frame learned sensor affine

/// The per-frame scale and shift on SENSOR depth (F6).
///
/// TIGHTLY constrained on purpose. This exists to absorb a genuine per-frame
/// sensor drift of a percent or two, not to let the optimiser explain a
/// mis-posed frame by stretching its depth map. `SmartLossSettings` clamps
/// both, and a correction that hits the clamp is logged rather than silently
/// applied, because a large correction means something else is wrong.
public struct SmartDepthAffine: Codable, Hashable, Sendable {
    public var scale: Float
    public var shiftMeters: Float

    public init(scale: Float = 1, shiftMeters: Float = 0) {
        self.scale = scale
        self.shiftMeters = shiftMeters
    }

    public static let identity = SmartDepthAffine()

    @inline(__always)
    public func apply(_ z: Float) -> Float { scale * z + shiftMeters }

    public var isIdentity: Bool { scale == 1 && shiftMeters == 0 }
}

// MARK: - Bias record

/// One coarse voxel of the bias field, as it sits in `prepass/trust_bias.bin`.
public struct SmartTrustBiasCell: Hashable, Sendable {
    public var mortonKey: UInt64
    /// Running mean SIGNED residual, metres. Positive means the sensor reads
    /// LONG here relative to what the other views say.
    public var mean: Float
    public var variance: Float
    public var sampleCount: UInt32
    /// How many distinct capture times contributed. This is what separates
    /// "measured 400 times in one sweep of the arm" from "confirmed on three
    /// separate visits", and only the second is real evidence.
    public var distinctTimeCount: UInt16

    public init(
        mortonKey: UInt64,
        mean: Float,
        variance: Float,
        sampleCount: UInt32,
        distinctTimeCount: UInt16
    ) {
        self.mortonKey = mortonKey
        self.mean = mean
        self.variance = variance
        self.sampleCount = sampleCount
        self.distinctTimeCount = distinctTimeCount
    }

    /// Bytes per record in `prepass/trust_bias.bin`.
    static let byteSize = 24
}

// MARK: - TwoScaleTrustField

/// F6. Implements `TrustField` (CONTRACTS.md section 5).
public final class TwoScaleTrustField: TrustField {

    // MARK: Configuration

    private let settings: SmartLossSettings
    private let noiseModel: SmartDepthNoiseModel

    /// Sigma written for a sample that has no usable return at all. Read back
    /// through `weight`, this is ~0, which is the intent: not "slightly
    /// distrusted", but "this sample says nothing".
    static let hopelessSigma: Float = 1e3

    // MARK: Loaded state

    private let lock = NSLock()
    private var samplesPerFrame: Int = 0
    private var depthWidth: Int = 0
    private var depthHeight: Int = 0
    private var loadedRefs: TrustFieldRefs?

    private var noiseReader: SmartSampleFieldReader?
    private var confidenceReader: SmartSampleFieldReader?
    private var biasData: Data?
    private var biasCellCount: Int = 0
    private var biasVoxelSize: Float = 0.25
    private var affineByFrame: [FrameID: SmartDepthAffine] = [:]

    public init(
        settings: SmartLossSettings = .default,
        noiseModel: SmartDepthNoiseModel = .default
    ) {
        self.settings = settings
        self.noiseModel = noiseModel
    }

    // MARK: - TrustField: build

    public func build(
        bundle: CaptureBundle,
        prePassPoses: [String: Pose],
        at ref: CaptureBundleRef
    ) async throws -> TrustFieldRefs {
        let width = bundle.settings.depthWidth
        let height = bundle.settings.depthHeight
        let perFrame = width * height
        guard perFrame > 0 else {
            throw SmartError.malformedSidecar(
                path: "capture_bundle.json settings.depthWidth/Height",
                expectedBytes: 1,
                actualBytes: perFrame
            )
        }

        lock.lock()
        depthWidth = width
        depthHeight = height
        samplesPerFrame = perFrame
        lock.unlock()

        let folder = BrandConfig.Folder.prePass
        let noisePath = "\(folder)/trust_noise.bin"
        let biasPath = "\(folder)/trust_bias.bin"
        let affinePath = "\(folder)/depth_affine.bin"
        let confidencePath = "\(folder)/confidence_recal.bin"

        // Frame-major files are indexed by FrameID, so every slot from 0 to
        // the highest frame index must exist even if that frame had no depth.
        // A missing frame is written as "maximally distrusted", which reads
        // back as weight ~0 rather than as an off-by-one into another frame's
        // samples.
        let maxIndex = bundle.frames.map(\.index).max().map { Int($0) } ?? -1
        let slotCount = maxIndex + 1
        var framesByIndex: [Int: CaptureFrame] = [:]
        framesByIndex.reserveCapacity(bundle.frames.count)
        for frame in bundle.frames { framesByIndex[Int(frame.index)] = frame }

        let poses = Self.poseTable(bundle: bundle, prePassPoses: prePassPoses)
        let nativeK = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: width, depthHeight: height
        )

        // WHAT THIS BUILD IS ALLOWED TO COST.
        //
        // `SmartLossSettings.economy` exists for "a phone that is already
        // warm, or a house-sized scan where the trust build would otherwise
        // dominate the pre-pass". Both of those are measurable right here, so
        // the choice is made here rather than being left to a caller. It is
        // logged either way: a cheaper build is a real difference in what was
        // verified, and a difference nobody can see is how a quality change
        // gets mistaken for a bad scan.
        let costChoice = SmartLossSettings.trustBuildCost(
            requested: settings,
            frameCount: bundle.frames.count,
            thermalLevel: ThermalLevel(ProcessInfo.processInfo.thermalState)
        )
        let cost = costChoice.settings
        if let reason = costChoice.downshiftReason {
            SmartLog.trust.notice(
                """
                Trust build running the economy preset because \
                \(reason, privacy: .public): \(cost.trustPartnerFrames, privacy: .public) \
                partner frames, stride \(cost.trustSampleStride, privacy: .public), \
                plane-sweep budget \(cost.planeSweepSampleBudget, privacy: .public). \
                Depth is verified less thoroughly than on a cool phone.
                """
            )
        }

        let partners = Self.coVisibilityTable(
            bundle: bundle, poses: poses, maxPartners: cost.trustPartnerFrames
        )

        var accumulator = SmartBiasAccumulator(voxelSizeMeters: 0.25)
        // Residual magnitudes split by the ARKit confidence level they came
        // from. This is the raw material for the recalibration in step 5:
        // ARKit's 0/1/2 is a ranking, and these buckets say what that ranking
        // is actually worth on THIS scan.
        var residualsByLevel: [[Float]] = [[], [], []]
        var affines: [(FrameID, SmartDepthAffine)] = []

        let noiseWriter = try SmartChunkedWriter(url: ref.url(forRelativePath: noisePath))

        // There is no confidence writer here on purpose.
        //
        // This loop used to stream a placeholder pass into
        // confidence_recal.bin, one full frame of floats per slot, and
        // step 5 then called rewriteConfidence, whose first act is to
        // DELETE that file and build it again from nothing. Every byte of
        // the first pass was thrown away unread: 196,608 bytes per frame,
        // so about 177 MB on a 900-frame scan, a quarter of everything
        // the whole pre-pass writes.
        //
        // iOS was already complaining. The phone reported 1 GB of file
        // writes in a 32-minute session and 4 GB across one day, breaching
        // both tiers of its daily write budget, which costs real flash
        // endurance on the user's phone.
        //
        // Nothing is lost by not writing it. The real remap needs the
        // residual statistics of the whole scan, which is exactly why it
        // cannot run until this loop has finished, and rewriteConfidence
        // walks every slot from 0 itself rather than patching what was
        // here. If build() throws before step 5 the file is now absent
        // rather than full of zeros, which is the more honest of the two.

        let depthCache = SmartDepthCache(
            capacity: Swift.max(4, cost.trustPartnerFrames + 2), sampleCount: perFrame
        )
        let imageCache = SmartImageCache(
            capacity: Swift.max(3, cost.trustPartnerFrames + 1),
            longEdge: Swift.max(width, height)
        )
        var planeSweepBudget = cost.planeSweepSampleBudget
        let startingSweepBudget = planeSweepBudget

        for slot in 0..<slotCount {
            if Task.isCancelled {
                try? noiseWriter.close()
                throw NimbusError.cancelled
            }

            // One pool per frame slot, for the reason spelled out in
            // SmartImageLoader.load: a Swift concurrency job drains its
            // autorelease pool when the job ENDS, not between iterations, and
            // this is the longest-running loop in the pre-pass. It walks EVERY
            // slot from 0 to the highest frame index, not just the keyframes,
            // so on the 900-frame scan this file's own comments describe it is
            // 900 passes, and each pass opens this frame's depth and
            // confidence sidecars, hands a 196,608-byte chunk to the noise
            // writer, and can reach into the image cache for the plane sweep.
            // None of that is meant to outlive the slot it was read for, so
            // draining per slot keeps one frame resident instead of the scan.
            //
            // The skip path RETURNS rather than using `continue`, which cannot
            // cross a closure boundary. It means exactly what the `continue`
            // meant: the maximally distrusted frame HAS been written and this
            // iteration is over. That ordering is load bearing. trust_noise.bin
            // is frame-major with no header, so a return placed above the
            // append would leave the file one frame short and shear every later
            // lookup into another frame's samples.
            //
            // The cancellation check stays outside the pool because its throw
            // has to leave the loop rather than the closure, and the writer has
            // to be closed before it does.
            try autoreleasepool { () throws -> Void in
                guard
                    let frame = framesByIndex[slot],
                    let pose = poses[frame.index],
                    let depth = depthCache.depth(for: frame, at: ref)
                else {
                    try noiseWriter.appendFloats(
                        [Float](repeating: Self.hopelessSigma, count: perFrame)
                    )
                    return
                }

                let confidence = depthCache.confidence(for: frame, at: ref)

                // --- 1. The prior, for every sample. --------------------------
                // A few flops, so it is computed for the whole grid; only the
                // cross-frame verification below is strided.
                var sigma = [Float](repeating: Self.hopelessSigma, count: perFrame)
                for v in 0..<height {
                    for u in 0..<width {
                        let i = v * width + u
                        let z = depth[i]
                        guard z > 0, SmartMath.isUsableDepth(z) else { continue }
                        let cosIncidence = Self.incidenceCosine(
                            depth: depth, u: u, v: v, width: width, height: height, k: nativeK
                        )
                        sigma[i] = noiseModel.sigma(rangeMeters: z, incidenceCosine: cosIncidence)
                    }
                }

                // --- 2. Cross-frame verification, strided. --------------------
                let sampleStride = Swift.max(1, cost.trustSampleStride)
                var partnerFrames: [CaptureFrame] = []
                for id in partners[frame.index] ?? [] {
                    if let partner = framesByIndex[Int(id)] { partnerFrames.append(partner) }
                }

                var measuredRatios: [Float] = []
                var affineSensor: [Float] = []
                var affineTarget: [Float] = []

                if !partnerFrames.isEmpty {
                    var v = 0
                    while v < height {
                        var u = 0
                        while u < width {
                            defer { u += sampleStride }
                            let i = v * width + u
                            let z = depth[i]
                            guard z > 0, SmartMath.isUsableDepth(z),
                                  sigma[i] < Self.hopelessSigma
                            else { continue }

                            let pixel = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)
                            let cameraPoint = SmartCamera.unproject(pixel, depthZ: z, nativeK)
                            let world = SmartCamera.cameraToWorld(pose, cameraPoint)

                            var residuals: [Float] = []
                            residuals.reserveCapacity(partnerFrames.count)
                            for partner in partnerFrames {
                                guard
                                    let partnerPose = poses[partner.index],
                                    let partnerDepth = depthCache.depth(for: partner, at: ref)
                                else { continue }
                                let pc = SmartCamera.worldToCamera(partnerPose, world)
                                guard let pp = SmartCamera.project(pc, nativeK) else { continue }
                                // Trapping conversion: guard before, not after.
                                // `project` only rules out points behind the
                                // camera, so a partner pose with a huge or
                                // non-finite translation still lands here.
                                guard
                                    let partnerPixel = SmartMath.pixelIndex(
                                        pp, width: width, height: height
                                    )
                                else { continue }
                                let pz = partnerDepth[partnerPixel.y * width + partnerPixel.x]
                                guard pz > 0, SmartMath.isUsableDepth(pz) else { continue }
                                // Positive residual: this frame's sample sits
                                // BEYOND where the partner sees the surface.
                                residuals.append(pc.z - pz)
                            }
                            guard residuals.count >= 2 else { continue }

                            let medianResidual = SmartMath.median(residuals)
                            // Robust spread, not a standard deviation: one partner
                            // looking through a doorway must not get to decide
                            // this sample's noise. 1.4826 makes the median
                            // absolute deviation comparable to a sigma.
                            let spread = SmartMath.median(residuals.map { abs($0 - medianResidual) })
                                * 1.4826

                            var measured = sqrt(
                                sigma[i] * sigma[i]
                                    + spread * spread
                                    + medianResidual * medianResidual
                            )

                            // Optional photometric second opinion, budgeted.
                            if planeSweepBudget > 0,
                               let sweep = planeSweep(
                                   frame: frame,
                                   pose: pose,
                                   partners: partnerFrames,
                                   poses: poses,
                                   u: u,
                                   v: v,
                                   depth: z,
                                   nativeK: nativeK,
                                   imageCache: imageCache,
                                   ref: ref
                               )
                            {
                                planeSweepBudget -= 1
                                // A sharp, high-NCC peak that agrees with the
                                // sensor is real evidence the sample is good; a
                                // flat peak is a textureless patch and its
                                // location means nothing.
                                let agreement = SmartMath.smoothdrop(
                                    0.02, settings.planeSweepRangeMeters, abs(sweep.offsetMeters)
                                )
                                let quality = SmartMath.clamp(sweep.peakNCC, 0, 1) * sweep.sharpness
                                let trustBoost = 0.5 + 0.5 * agreement * quality
                                measured /= Swift.max(trustBoost, 0.25)
                                accumulator.add(
                                    world: world,
                                    residual: sweep.offsetMeters,
                                    timeSeconds: frame.timestampSeconds
                                )
                            }

                            sigma[i] = measured
                            let headOnPrior = Swift.max(
                                noiseModel.sigma(rangeMeters: z, incidenceCosine: 1), 1e-4
                            )
                            measuredRatios.append(measured / headOnPrior)

                            accumulator.add(
                                world: world,
                                residual: medianResidual,
                                timeSeconds: frame.timestampSeconds
                            )

                            let level = Int(i < confidence.count ? confidence[i] : 1)
                            if level >= 0, level < 3 {
                                residualsByLevel[level].append(abs(medianResidual))
                            }

                            affineSensor.append(z)
                            affineTarget.append(z - medianResidual)
                        }
                        v += sampleStride
                    }
                }

                // --- 3. Frame-level inflation for unverified samples. ----------
                // NOT spatial averaging: one scalar per frame, applied uniformly,
                // which cannot move an outlier into its neighbours. It says "on
                // this frame the sensor turned out to be 1.6x worse than the
                // physics prior expected", which is a frame property (motion blur,
                // a warm sensor, a dark room), not a place property.
                let inflation = measuredRatios.isEmpty
                    ? 1
                    : Swift.max(1, SmartMath.percentile(measuredRatios, 0.5))
                if inflation > 1 {
                    for v in 0..<height {
                        for u in 0..<width {
                            let i = v * width + u
                            guard sigma[i] < Self.hopelessSigma else { continue }
                            let verified = (v % sampleStride == 0) && (u % sampleStride == 0)
                            if !verified { sigma[i] *= inflation }
                        }
                    }
                }

                // --- 4. Per-frame sensor affine. -------------------------------
                let affine = Self.fitAffine(
                    sensor: affineSensor,
                    target: affineTarget,
                    maxScaleDeviation: settings.maxDepthScaleDeviation,
                    maxShiftMeters: settings.maxDepthShiftMeters
                )
                if !affine.isIdentity {
                    affines.append((frame.index, affine))
                    let scaleClamped =
                        abs(affine.scale - 1) >= settings.maxDepthScaleDeviation * 0.999
                    let shiftClamped =
                        abs(affine.shiftMeters) >= settings.maxDepthShiftMeters * 0.999
                    if scaleClamped || shiftClamped {
                        SmartLog.trust.notice(
                            """
                            Frame \(frame.index) depth affine hit its clamp \
                            (scale \(affine.scale), shift \(affine.shiftMeters) m). \
                            A correction this large usually means the POSE is wrong, not the depth.
                            """
                        )
                    }
                }

                try noiseWriter.appendFloats(sigma)
            }
        }

        try noiseWriter.close()

        // --- 5. Recalibrate ARKit confidence. -----------------------------
        let levelProbabilities = Self.recalibrate(residualsByLevel: residualsByLevel)
        SmartLog.trust.info(
            """
            ARKit confidence remapped from ranking to probability: \
            low=\(levelProbabilities[0]) medium=\(levelProbabilities[1]) \
            high=\(levelProbabilities[2])
            """
        )
        try rewriteConfidence(
            at: ref.url(forRelativePath: confidencePath),
            slotCount: slotCount,
            samplesPerFrame: perFrame,
            framesByIndex: framesByIndex,
            depthCache: depthCache,
            bundleRef: ref,
            levelProbabilities: levelProbabilities
        )

        // --- 6. Write the bias field and the affines. ---------------------
        let biasCells = accumulator.sortedCells()
        var biasBytes = Data(capacity: biasCells.count * SmartTrustBiasCell.byteSize)
        for cell in biasCells {
            SmartBinary.append(cell.mortonKey, to: &biasBytes)
            SmartBinary.append(cell.mean, to: &biasBytes)
            SmartBinary.append(cell.variance, to: &biasBytes)
            SmartBinary.append(cell.sampleCount, to: &biasBytes)
            SmartBinary.append(cell.distinctTimeCount, to: &biasBytes)
            SmartBinary.append(UInt16(0), to: &biasBytes)
        }
        try SmartBinary.write(biasBytes, to: ref.url(forRelativePath: biasPath))

        var affineBytes = Data(capacity: affines.count * 12)
        for (index, a) in affines {
            SmartBinary.append(index, to: &affineBytes)
            SmartBinary.append(a.scale, to: &affineBytes)
            SmartBinary.append(a.shiftMeters, to: &affineBytes)
        }
        try SmartBinary.write(affineBytes, to: ref.url(forRelativePath: affinePath))

        let refs = TrustFieldRefs(
            biasFieldPath: biasPath,
            biasVoxelSizeMeters: accumulator.voxelSizeMeters,
            noiseFieldPath: noisePath,
            depthAffinePath: affinePath,
            recalibratedConfidencePath: confidencePath
        )

        let sweepsRun = startingSweepBudget - planeSweepBudget
        let presetName = costChoice.downshiftReason == nil ? "full" : "economy"
        SmartLog.trust.info(
            """
            Trust field built: \(slotCount) frame slots, \(biasCells.count) bias cells, \
            \(affines.count) per-frame depth affines, \(sweepsRun) plane-sweep verifications, \
            \(presetName, privacy: .public) cost preset \
            (\(cost.trustPartnerFrames) partner frames, stride \(cost.trustSampleStride))
            """
        )

        try await load(refs, at: ref)
        return refs
    }

    // MARK: - TrustField: load

    public func load(_ refs: TrustFieldRefs, at ref: CaptureBundleRef) async throws {
        // The per-sample files are frame-major with no header, so their stride
        // has to come from somewhere. It comes from `build` in the same
        // process, or from `prepare(depthWidth:depthHeight:)` when a field
        // built in an earlier session is being re-opened. Guessing it would
        // silently shear every lookup, so it is required, not assumed.
        lock.lock()
        let known = samplesPerFrame > 0 ? samplesPerFrame : depthWidth * depthHeight
        lock.unlock()
        guard known > 0 else {
            throw SmartError.notPrepared(
                "TwoScaleTrustField.load needs the native depth grid size; "
                    + "call prepare(depthWidth:depthHeight:) first"
            )
        }
        let perFrame = known

        let noise = try SmartSampleFieldReader(
            url: ref.url(forRelativePath: refs.noiseFieldPath),
            samplesPerFrame: perFrame,
            defaultValue: Self.hopelessSigma
        )

        var confidence: SmartSampleFieldReader?
        if let path = refs.recalibratedConfidencePath {
            confidence = try? SmartSampleFieldReader(
                url: ref.url(forRelativePath: path),
                samplesPerFrame: perFrame,
                defaultValue: 0.5
            )
        }

        let bias = try? SmartBinary.map(ref.url(forRelativePath: refs.biasFieldPath))

        var affines: [FrameID: SmartDepthAffine] = [:]
        if let path = refs.depthAffinePath,
           let data = try? SmartBinary.map(ref.url(forRelativePath: path))
        {
            let records = data.count / 12
            data.withUnsafeBytes { raw in
                for r in 0..<records {
                    let base = r * 12
                    let idx = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base, as: UInt32.self)
                    )
                    let scaleBits = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 4, as: UInt32.self)
                    )
                    let shiftBits = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 8, as: UInt32.self)
                    )
                    affines[idx] = SmartDepthAffine(
                        scale: Float(bitPattern: scaleBits),
                        shiftMeters: Float(bitPattern: shiftBits)
                    )
                }
            }
        }

        let cellCount = (bias?.count ?? 0) / SmartTrustBiasCell.byteSize

        lock.lock()
        samplesPerFrame = perFrame
        loadedRefs = refs
        noiseReader = noise
        confidenceReader = confidence
        biasData = bias
        biasCellCount = cellCount
        biasVoxelSize = Swift.max(refs.biasVoxelSizeMeters, 1e-3)
        affineByFrame = affines
        lock.unlock()

        SmartLog.trust.info(
            """
            Trust field loaded: \(noise.frameCount) frames of noise, \
            \(cellCount) bias cells, \(affines.count) depth affines
            """
        )
    }

    /// Tells this instance the native grid shape before `load`, so a field
    /// built in an earlier session can be read back with the right stride.
    ///
    /// Not part of the `TrustField` protocol: the protocol only has to cover
    /// build-then-use inside one process. This is what makes "capture on
    /// Monday, train on Tuesday" work.
    public func prepare(depthWidth: Int, depthHeight: Int) {
        lock.lock()
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        samplesPerFrame = Swift.max(0, depthWidth * depthHeight)
        lock.unlock()
    }

    /// Convenience over `prepare(depthWidth:depthHeight:)`.
    public func prepare(bundle: CaptureBundle) {
        prepare(
            depthWidth: bundle.settings.depthWidth,
            depthHeight: bundle.settings.depthHeight
        )
    }

    // MARK: - TrustField: query

    public func weight(frame: FrameID, sampleIndex: Int) -> Float {
        lock.lock()
        let reader = noiseReader
        let confidence = confidenceReader
        lock.unlock()

        guard let reader else { return 0 }
        let sigma = reader.value(frame: frame, sampleIndex: sampleIndex)
        guard sigma.isFinite, sigma > 0, sigma < 1e2 else { return 0 }

        // Inverse-variance, written so a sample at exactly the reference noise
        // level scores 0.5. Smooth everywhere: a hard per-region switch puts a
        // visible seam through the middle of a wall, which is precisely the
        // artefact this design exists to avoid.
        let ratio = sigma / Swift.max(settings.noiseReferenceMeters, 1e-4)
        var w = 1 / (1 + ratio * ratio)

        if let confidence {
            // The recalibrated confidence is a probability the sample is
            // usable at all, so it multiplies. The 0.25 floor stops a
            // pessimistic recalibration from zeroing the depth term outright.
            let p = SmartMath.clamp(confidence.value(frame: frame, sampleIndex: sampleIndex), 0, 1)
            w *= (0.25 + 0.75 * p)
        }
        return SmartMath.clamp(w, 0, 1)
    }

    public func bias(atWorldPoint point: Vector3) -> (mean: Float, variance: Float)? {
        lock.lock()
        let data = biasData
        let count = biasCellCount
        let voxel = biasVoxelSize
        lock.unlock()

        guard let data, count > 0 else { return nil }
        let key = SmartMorton.key(point.simd, voxelSize: voxel)

        // Records are sorted ascending by key, so this is a binary search over
        // a memory-mapped file rather than a resident dictionary. A whole
        // house at 25 cm is about twenty probes.
        return data.withUnsafeBytes { raw -> (mean: Float, variance: Float)? in
            var lo = 0
            var hi = count - 1
            while lo <= hi {
                let mid = (lo + hi) / 2
                let base = mid * SmartTrustBiasCell.byteSize
                let k = UInt64(
                    littleEndian: raw.loadUnaligned(fromByteOffset: base, as: UInt64.self)
                )
                if k == key {
                    let meanBits = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 8, as: UInt32.self)
                    )
                    let varianceBits = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 12, as: UInt32.self)
                    )
                    let samples = UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 16, as: UInt32.self)
                    )
                    let times = UInt16(
                        littleEndian: raw.loadUnaligned(fromByteOffset: base + 20, as: UInt16.self)
                    )
                    // Too few samples, or all from one pass of the arm, is not
                    // a bias estimate. nil is the honest answer, and every
                    // caller has a defined behaviour for nil.
                    guard samples >= 8, times >= 2 else { return nil }
                    return (Float(bitPattern: meanBits), Float(bitPattern: varianceBits))
                }
                if k < key { lo = mid + 1 } else { hi = mid - 1 }
            }
            return nil
        }
    }

    public func depthLossScale(iteration: Int, of total: Int) -> Float {
        Self.depthLossScale(iteration: iteration, of: total, floor: settings.depthScheduleFloor)
    }

    /// The schedule, as a pure function so the Trainer can plot it and the
    /// Booster's Python port can match it exactly.
    ///
    /// Full strength while the geometry is being placed, then a cosine taper.
    /// The floor is deliberately not zero: depth stays a weak prior all the
    /// way to the end, because a late photometric fit left completely
    /// unconstrained will happily walk a featureless wall a metre backwards to
    /// buy a fraction of a dB.
    public static func depthLossScale(iteration: Int, of total: Int, floor: Float) -> Float {
        guard total > 0 else { return 1 }
        let i = Swift.max(0, Swift.min(iteration, total))
        // "Strong for the first 3-7k iterations" is written for a desktop
        // run. On a phone doing 500-3000 total, the same intent is the first
        // third; the absolute cap keeps the desktop behaviour identical.
        let strongEnd = Swift.min(Int(Float(total) * 0.35), 7000)
        if i <= strongEnd { return 1 }
        let span = Swift.max(total - strongEnd, 1)
        let t = SmartMath.clamp(Float(i - strongEnd) / Float(span), 0, 1)
        let cosineTaper = 0.5 * (1 + cos(t * .pi))          // 1 -> 0
        let f = SmartMath.clamp(floor, 0, 1)
        return f + (1 - f) * cosineTaper
    }

    /// The per-frame learned sensor scale and shift (F6), or identity when
    /// this frame had no correction.
    ///
    /// Not part of the protocol, because `TrustField` is about weights and
    /// this is about the target those weights are applied to.
    /// `SmartDepthLoss` needs both.
    public func depthAffine(frame: FrameID) -> SmartDepthAffine {
        lock.lock()
        defer { lock.unlock() }
        return affineByFrame[frame] ?? .identity
    }

    /// Raw per-sample sigma in metres, for the artefact heatmap and the
    /// floorplan trust layer. `nil` when nothing is loaded or the sample has
    /// no usable return.
    public func sigmaMeters(frame: FrameID, sampleIndex: Int) -> Float? {
        lock.lock()
        let reader = noiseReader
        lock.unlock()
        guard let reader else { return nil }
        let s = reader.value(frame: frame, sampleIndex: sampleIndex)
        return s.isFinite && s < 1e2 ? s : nil
    }

    public var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return noiseReader != nil
    }

    // MARK: - Recalibration

    /// ARKit's confidence is a RANKING, not a probability. It flags roughly
    /// 1.6% of samples low, which cannot be read as "1.6% of samples are
    /// wrong". So each level is scored by the residuals actually observed at
    /// that level on THIS scan: the fraction of them inside a 2 cm tolerance.
    ///
    /// Monotonicity is then enforced, because a level ordering that inverts is
    /// a sign of too little data rather than of ARKit being backwards.
    static func recalibrate(residualsByLevel: [[Float]]) -> [Float] {
        let tolerance: Float = 0.02
        // Priors, used for any level that did not collect enough residuals to
        // say anything. Ordered, and deliberately not extreme.
        var p: [Float] = [0.30, 0.60, 0.85]
        for level in 0..<3 {
            let residuals = residualsByLevel[level]
            guard residuals.count >= 32 else { continue }
            var inside = 0
            for r in residuals where r <= tolerance { inside += 1 }
            p[level] = SmartMath.clamp(Float(inside) / Float(residuals.count), 0.02, 0.99)
        }
        p[1] = Swift.max(p[1], p[0])
        p[2] = Swift.max(p[2], p[1])
        return p
    }

    private func rewriteConfidence(
        at url: URL,
        slotCount: Int,
        samplesPerFrame: Int,
        framesByIndex: [Int: CaptureFrame],
        depthCache: SmartDepthCache,
        bundleRef: CaptureBundleRef,
        levelProbabilities: [Float]
    ) throws {
        let writer = try SmartChunkedWriter(url: url)
        // The SECOND full walk over every slot in this build, and it costs the
        // same as the first: the depth cache holds only a handful of frames, so
        // nearly every slot here is a miss that opens this frame's depth and
        // confidence sidecars again, and every slot writes another
        // `samplesPerFrame` floats. Same pool, same reason as the loop in
        // `build`, and the same reshaping: `continue` cannot cross a closure
        // boundary, so the skip path writes its all-zero frame and RETURNS.
        // confidence_recal.bin is frame-major like the noise field, so that
        // write has to happen before the return or every later frame's
        // confidence reads back off by one frame.
        for slot in 0..<slotCount {
            try autoreleasepool { () throws -> Void in
                guard
                    let frame = framesByIndex[slot],
                    let depth = depthCache.depth(for: frame, at: bundleRef)
                else {
                    try writer.appendFloats([Float](repeating: 0, count: samplesPerFrame))
                    return
                }
                let confidence = depthCache.confidence(for: frame, at: bundleRef)
                var out = [Float](repeating: 0, count: samplesPerFrame)
                for i in 0..<samplesPerFrame {
                    guard i < depth.count, depth[i] > 0, SmartMath.isUsableDepth(depth[i])
                    else { continue }
                    let level = Swift.max(0, Swift.min(2, Int(i < confidence.count ? confidence[i] : 1)))
                    out[i] = levelProbabilities[level]
                }
                try writer.appendFloats(out)
            }
        }
        try writer.close()
    }

    // MARK: - Plane sweep

    struct PlaneSweepResult {
        /// Where the photometric peak sits relative to the sensor's depth,
        /// metres. Positive means the photometry wants the surface further.
        var offsetMeters: Float
        var peakNCC: Float
        /// 0...1. How far the peak stands above the rest of the sweep. A flat
        /// curve is a textureless patch and its peak location means nothing.
        var sharpness: Float
    }

    /// Targeted plane-sweep photometric verification around one native LiDAR
    /// sample (F6): +/- 15 cm in ~32 steps, ZNCC over the co-visible frames.
    ///
    /// Real, and deliberately budgeted: at ~49k samples per frame across
    /// hundreds of frames this would take longer than the training run, so
    /// `SmartLossSettings.planeSweepSampleBudget` caps how many samples across
    /// the whole scan get one. `0` disables it, which is the economy preset.
    private func planeSweep(
        frame: CaptureFrame,
        pose: Pose,
        partners: [CaptureFrame],
        poses: [FrameID: Pose],
        u: Int,
        v: Int,
        depth: Float,
        nativeK: CameraIntrinsics,
        imageCache: SmartImageCache,
        ref: CaptureBundleRef
    ) -> PlaneSweepResult? {
        let steps = settings.planeSweepSteps
        guard steps >= 5 else { return nil }
        guard let reference = imageCache.image(for: frame, at: ref) else { return nil }

        let radius = Swift.max(1, settings.planeSweepPatchRadius)
        let referenceScaleX = Float(reference.width) / Float(Swift.max(nativeK.width, 1))
        let referenceScaleY = Float(reference.height) / Float(Swift.max(nativeK.height, 1))
        let centre = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)

        var referencePatch: [Float] = []
        referencePatch.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
        for dy in -radius...radius {
            for dx in -radius...radius {
                let p = SIMD2<Float>(
                    (centre.x + Float(dx)) * referenceScaleX,
                    (centre.y + Float(dy)) * referenceScaleY
                )
                referencePatch.append(reference.lumaBilinear(p))
            }
        }
        guard let referenceStats = Self.zeroMeanUnitNorm(referencePatch) else { return nil }

        let half = settings.planeSweepRangeMeters
        var scores = [Float](repeating: 0, count: steps)
        var counts = [Int](repeating: 0, count: steps)

        for partner in partners {
            guard
                let partnerPose = poses[partner.index],
                let partnerImage = imageCache.image(for: partner, at: ref)
            else { continue }
            let partnerScaleX = Float(partnerImage.width) / Float(Swift.max(nativeK.width, 1))
            let partnerScaleY = Float(partnerImage.height) / Float(Swift.max(nativeK.height, 1))

            for s in 0..<steps {
                let t = Float(s) / Float(steps - 1)
                let candidateDepth = depth + (t * 2 - 1) * half
                guard candidateDepth > 0.05 else { continue }

                var patch: [Float] = []
                patch.reserveCapacity(referencePatch.count)
                var usable = true
                for dy in -radius...radius {
                    for dx in -radius...radius {
                        let pixel = SIMD2<Float>(centre.x + Float(dx), centre.y + Float(dy))
                        let cameraPoint = SmartCamera.unproject(
                            pixel, depthZ: candidateDepth, nativeK
                        )
                        let world = SmartCamera.cameraToWorld(pose, cameraPoint)
                        let pc = SmartCamera.worldToCamera(partnerPose, world)
                        guard let pp = SmartCamera.project(pc, nativeK) else {
                            usable = false
                            break
                        }
                        let inPartner = SIMD2<Float>(pp.x * partnerScaleX, pp.y * partnerScaleY)
                        guard inPartner.x >= 0, inPartner.y >= 0,
                              inPartner.x < Float(partnerImage.width),
                              inPartner.y < Float(partnerImage.height)
                        else {
                            usable = false
                            break
                        }
                        patch.append(partnerImage.lumaBilinear(inPartner))
                    }
                    if !usable { break }
                }
                guard usable, let partnerStats = Self.zeroMeanUnitNorm(patch) else { continue }

                var ncc: Float = 0
                for j in 0..<referenceStats.count { ncc += referenceStats[j] * partnerStats[j] }
                scores[s] += ncc
                counts[s] += 1
            }
        }

        var bestIndex = -1
        var best: Float = -2
        var sum: Float = 0
        var evaluated = 0
        for s in 0..<steps where counts[s] > 0 {
            let averaged = scores[s] / Float(counts[s])
            sum += averaged
            evaluated += 1
            if averaged > best {
                best = averaged
                bestIndex = s
            }
        }
        guard bestIndex >= 0, evaluated >= 3 else { return nil }

        let mean = sum / Float(evaluated)
        let sharpness = SmartMath.clamp(best - mean, 0, 1)
        let t = Float(bestIndex) / Float(steps - 1)
        return PlaneSweepResult(
            offsetMeters: (t * 2 - 1) * half,
            peakNCC: best,
            sharpness: sharpness
        )
    }

    /// Zero-mean, unit-norm form of a patch, which is what makes the dot
    /// product above an actual ZNCC. Returns nil for a flat patch, where the
    /// normalisation is a division by zero and the correlation would be
    /// meaningless anyway.
    static func zeroMeanUnitNorm(_ patch: [Float]) -> [Float]? {
        guard patch.count > 1 else { return nil }
        var sum: Float = 0
        for p in patch { sum += p }
        let mean = sum / Float(patch.count)
        var centred = patch.map { $0 - mean }
        var norm: Float = 0
        for c in centred { norm += c * c }
        norm = sqrt(norm)
        guard norm > 1e-4 else { return nil }
        for i in 0..<centred.count { centred[i] /= norm }
        return centred
    }

    // MARK: - Static helpers

    /// Refined poses where the pre-pass produced them, raw VIO otherwise.
    static func poseTable(bundle: CaptureBundle, prePassPoses: [String: Pose]) -> [FrameID: Pose] {
        var table: [FrameID: Pose] = [:]
        table.reserveCapacity(bundle.frames.count)
        for frame in bundle.frames {
            table[frame.index] =
                prePassPoses[String(frame.index)] ?? frame.refinedPose ?? frame.rawPose
        }
        return table
    }

    /// Cheap co-visibility: the frames whose optical centre sits at a useful
    /// baseline and whose view direction still overlaps.
    ///
    /// Too close and the residual carries no information about depth (zero
    /// parallax); too far and the surface is not the same surface any more.
    static func coVisibilityTable(
        bundle: CaptureBundle,
        poses: [FrameID: Pose],
        maxPartners: Int,
        minBaselineMeters: Float = 0.05,
        maxBaselineMeters: Float = 2.5,
        minDirectionDot: Float = 0.5
    ) -> [FrameID: [FrameID]] {
        guard maxPartners > 0 else { return [:] }
        var table: [FrameID: [FrameID]] = [:]
        let frames = bundle.frames
        let idealBaseline =
            minBaselineMeters + 0.4 * (maxBaselineMeters - minBaselineMeters)

        for frame in frames {
            guard let pose = poses[frame.index] else { continue }
            let centre = pose.center.simd
            let forward = pose.forward.simd

            var candidates: [(id: FrameID, cost: Float)] = []
            for other in frames where other.index != frame.index {
                guard let otherPose = poses[other.index] else { continue }
                let baseline = simd_length(otherPose.center.simd - centre)
                guard baseline >= minBaselineMeters, baseline <= maxBaselineMeters else { continue }
                guard simd_dot(forward, otherPose.forward.simd) >= minDirectionDot else { continue }
                candidates.append((other.index, abs(baseline - idealBaseline)))
            }
            candidates.sort { $0.cost < $1.cost }
            table[frame.index] = candidates.prefix(maxPartners).map(\.id)
        }
        return table
    }

    /// `|dot(surface normal, view direction)|` from the local depth gradient.
    /// 1 is head-on, 0 is fully grazing.
    ///
    /// Returns 1 when the neighbourhood is incomplete, which is the
    /// conservative answer: it makes the physics prior its smallest, so a
    /// measured residual has to do the work rather than the prior asserting
    /// distrust it cannot justify.
    static func incidenceCosine(
        depth: [Float],
        u: Int,
        v: Int,
        width: Int,
        height: Int,
        k: CameraIntrinsics
    ) -> Float {
        guard u > 0, v > 0, u + 1 < width, v + 1 < height else { return 1 }
        let i = v * width + u
        let z = depth[i]
        let zLeft = depth[i - 1], zRight = depth[i + 1]
        let zUp = depth[i - width], zDown = depth[i + width]
        guard SmartMath.isUsableDepth(z),
              SmartMath.isUsableDepth(zLeft), SmartMath.isUsableDepth(zRight),
              SmartMath.isUsableDepth(zUp), SmartMath.isUsableDepth(zDown)
        else { return 1 }

        let p = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)
        let centre = SmartCamera.unproject(p, depthZ: z, k)
        let left = SmartCamera.unproject(p - SIMD2<Float>(1, 0), depthZ: zLeft, k)
        let right = SmartCamera.unproject(p + SIMD2<Float>(1, 0), depthZ: zRight, k)
        let up = SmartCamera.unproject(p - SIMD2<Float>(0, 1), depthZ: zUp, k)
        let down = SmartCamera.unproject(p + SIMD2<Float>(0, 1), depthZ: zDown, k)

        let normal = simd_cross(right - left, down - up)
        let length = simd_length(normal)
        guard length > 1e-8, simd_length(centre) > 1e-6 else { return 1 }
        let view = simd_normalize(centre)
        return SmartMath.clamp(abs(simd_dot(normal / length, view)), 0, 1)
    }

    /// Least-squares `target ~= scale * sensor + shift`, then clamped hard.
    ///
    /// The clamp is the point of this function, not an afterthought. An
    /// unconstrained affine on depth is a licence for the optimiser to explain
    /// a mis-posed frame by stretching its depth map, which produces a
    /// beautiful loss curve and a warped room.
    static func fitAffine(
        sensor: [Float],
        target: [Float],
        maxScaleDeviation: Float,
        maxShiftMeters: Float
    ) -> SmartDepthAffine {
        guard sensor.count == target.count, sensor.count >= 64 else { return .identity }
        let n = Float(sensor.count)
        var sx: Float = 0, sy: Float = 0, sxx: Float = 0, sxy: Float = 0
        for i in 0..<sensor.count {
            let x = sensor[i], y = target[i]
            guard x.isFinite, y.isFinite else { return .identity }
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
        }
        let denominator = n * sxx - sx * sx
        guard abs(denominator) > 1e-6 else { return .identity }
        var scale = (n * sxy - sx * sy) / denominator
        guard scale.isFinite else { return .identity }
        var shift = (sy - scale * sx) / n
        guard shift.isFinite else { return .identity }

        scale = SmartMath.clamp(scale, 1 - maxScaleDeviation, 1 + maxScaleDeviation)
        shift = SmartMath.clamp(shift, -maxShiftMeters, maxShiftMeters)
        return SmartDepthAffine(scale: scale, shiftMeters: shift)
    }
}

// MARK: - Bias accumulator

/// Welford accumulation of the coarse bias field, plus a distinct-capture-time
/// count per cell.
///
/// The time count is bucketed at 2 s, so a single sweep of the arm across a
/// wall counts once and coming back to the same wall five minutes later counts
/// again. That distinction is the difference between "measured a lot" and
/// "confirmed", and `TwoScaleTrustField.bias` refuses to report a cell that has
/// only ever been seen in one pass.
struct SmartBiasAccumulator {
    let voxelSizeMeters: Float
    private let timeBucketSeconds: Double = 2

    private struct Cell {
        var count: UInt32 = 0
        var mean: Float = 0
        var sumSquaredDelta: Float = 0
        var times: Set<Int32> = []
    }

    private var cells: [UInt64: Cell] = [:]

    init(voxelSizeMeters: Float) {
        self.voxelSizeMeters = Swift.max(voxelSizeMeters, 0.05)
    }

    /// Ignores anything past 5 m of signed residual: at that magnitude the
    /// sample is not a biased measurement of this surface, it is a
    /// measurement of a different surface, and folding it into a running mean
    /// would corrupt the cell.
    mutating func add(world: SIMD3<Float>, residual: Float, timeSeconds: Double) {
        guard residual.isFinite, abs(residual) < 5 else { return }
        let key = SmartMorton.key(world, voxelSize: voxelSizeMeters)
        var cell = cells[key] ?? Cell()
        cell.count += 1
        let delta = residual - cell.mean
        cell.mean += delta / Float(cell.count)
        cell.sumSquaredDelta += delta * (residual - cell.mean)
        if cell.times.count < 4096, timeSeconds.isFinite {
            // `Int(someDouble)` traps on anything outside Int's range, and
            // this timestamp is read out of capture_bundle.json. Finite is
            // not enough on its own. At two seconds per bucket the window
            // below is longer than any scan will ever be.
            let bucket = (timeSeconds / timeBucketSeconds).rounded(.down)
            if bucket > -1e15, bucket < 1e15 {
                cell.times.insert(Int32(truncatingIfNeeded: Int(bucket)))
            }
        }
        cells[key] = cell
    }

    func sortedCells() -> [SmartTrustBiasCell] {
        cells.map { key, cell in
            let variance = cell.count > 1 ? cell.sumSquaredDelta / Float(cell.count - 1) : 0
            return SmartTrustBiasCell(
                mortonKey: key,
                mean: cell.mean,
                variance: variance,
                sampleCount: cell.count,
                distinctTimeCount: UInt16(Swift.min(cell.times.count, Int(UInt16.max)))
            )
        }
        .sorted { $0.mortonKey < $1.mortonKey }
    }
}

// MARK: - Depth cache

/// Small LRU over decoded native depth and confidence maps.
///
/// The verification inner loop touches each partner frame's depth map once per
/// sample of the reference frame; without this it would re-read and re-decode
/// the same 96 KB file tens of thousands of times.
final class SmartDepthCache {
    private var depthOrder: [FrameID] = []
    private var depths: [FrameID: [Float]] = [:]
    private var confidenceOrder: [FrameID] = []
    private var confidences: [FrameID: [UInt8]] = [:]
    private let capacity: Int
    private let sampleCount: Int
    private let lock = NSLock()

    init(capacity: Int, sampleCount: Int) {
        self.capacity = Swift.max(1, capacity)
        self.sampleCount = Swift.max(1, sampleCount)
    }

    func depth(for frame: CaptureFrame, at ref: CaptureBundleRef) -> [Float]? {
        lock.lock()
        if let hit = depths[frame.index] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        guard let path = frame.depthPath else { return nil }
        guard let values = try? SmartBinary.readDepth16(
            ref.url(forRelativePath: path), count: sampleCount
        ) else {
            SmartLog.trust.error("Depth unreadable for frame \(frame.index), skipped")
            return nil
        }

        lock.lock()
        depths[frame.index] = values
        depthOrder.append(frame.index)
        while depthOrder.count > capacity {
            depths.removeValue(forKey: depthOrder.removeFirst())
        }
        lock.unlock()
        return values
    }

    /// Raw ARKit confidence, or an all-medium map when the sidecar is absent.
    /// Absent confidence costs precision in the recalibration and nothing
    /// else, so it degrades rather than fails.
    func confidence(for frame: CaptureFrame, at ref: CaptureBundleRef) -> [UInt8] {
        lock.lock()
        if let hit = confidences[frame.index] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        var values = [UInt8](repeating: 1, count: sampleCount)
        if let path = frame.confidencePath,
           let read = try? SmartBinary.readConfidence8(
               ref.url(forRelativePath: path), count: sampleCount
           )
        {
            values = read
        }

        lock.lock()
        confidences[frame.index] = values
        confidenceOrder.append(frame.index)
        while confidenceOrder.count > capacity {
            confidences.removeValue(forKey: confidenceOrder.removeFirst())
        }
        lock.unlock()
        return values
    }
}


// MARK: - One frame's trust, taken once

/// One frame's trust arrays, taken once, read without a lock.
///
/// `TwoScaleTrustField.weight(frame:sampleIndex:)` and `sigmaMeters` each take
/// the field's lock and then call through to a SmartSampleFieldReader, which
/// takes ITS lock and returns the frame's whole [Float] slice BY VALUE. Per
/// sample that is four to six lock/unlock pairs and three or four whole-array
/// retain/release cycles, to read two floats.
///
/// The pre-pass seeding stage calls both for 8,538,104 samples: roughly 34
/// million lock acquisitions, and every call for a given frame reads the SAME
/// two arrays.
///
/// The arithmetic here is copied expression for expression from those two
/// methods, fallbacks included, so the seeds come out bit for bit the same.
public struct SmartFrameTrust: Sendable {
    let noise: [Float]
    let confidence: [Float]?
    let noiseDefault: Float
    let confidenceDefault: Float
    let noiseReferenceMeters: Float

    @inline(__always)
    private func noiseValue(_ i: Int) -> Float {
        guard i >= 0, i < noise.count else { return noiseDefault }
        let v = noise[i]
        return v.isFinite ? v : noiseDefault
    }

    /// Same expression as `TwoScaleTrustField.weight`, without the locks.
    public func weight(sampleIndex: Int) -> Float {
        let sigma = noiseValue(sampleIndex)
        guard sigma.isFinite, sigma > 0, sigma < 1e2 else { return 0 }

        let ratio = sigma / Swift.max(noiseReferenceMeters, 1e-4)
        var w = 1 / (1 + ratio * ratio)

        if let confidence {
            var raw = confidenceDefault
            if sampleIndex >= 0, sampleIndex < confidence.count {
                let c = confidence[sampleIndex]
                raw = c.isFinite ? c : confidenceDefault
            }
            let p = SmartMath.clamp(raw, 0, 1)
            w *= (0.25 + 0.75 * p)
        }
        return SmartMath.clamp(w, 0, 1)
    }

    /// Same expression as `TwoScaleTrustField.sigmaMeters`, without the locks.
    public func sigmaMeters(sampleIndex: Int) -> Float? {
        let s = noiseValue(sampleIndex)
        return s.isFinite && s < 1e2 ? s : nil
    }
}

extension TwoScaleTrustField {

    /// This frame's trust arrays, taken under the lock ONCE.
    ///
    /// nil when there is no noise field at all, which is exactly the condition
    /// `weight` answers 0 to and `sigmaMeters` answers nil to, so a caller
    /// getting nil here keeps the fallbacks it already had.
    public func frameTrust(frame: FrameID) -> SmartFrameTrust? {
        lock.lock()
        let reader = noiseReader
        let confidence = confidenceReader
        let reference = settings.noiseReferenceMeters
        lock.unlock()

        guard let reader else { return nil }
        return SmartFrameTrust(
            noise: reader.slice(frame: frame),
            confidence: confidence?.slice(frame: frame),
            noiseDefault: reader.defaultValue,
            confidenceDefault: confidence?.defaultValue ?? 1,
            noiseReferenceMeters: reference
        )
    }
}
