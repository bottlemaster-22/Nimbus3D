//
//  DirectionalBackgroundModel.swift
//  Smart
//
//  F5, second half: the three-regime background, composited behind the
//  Gaussians using accumulated alpha.
//
//  ---------------------------------------------------------------------------
//  THE THREE REGIMES
//  ---------------------------------------------------------------------------
//   NEAR  0 to ~4.5 m       LiDAR is authoritative. Nothing here.
//   MID   ~4.5 to ~30 m     Monocular metric depth, scale-anchored to the
//                           LiDAR-valid pixels of the SAME frame. On the PC
//                           Booster that is Prompt Depth Anything, running for
//                           real. ON DEVICE there is no published Core ML
//                           export, so this regime is an honest STUB
//                           (`SmartMonocularDepthStub`) and falls back to
//                           parallax routing plus the far layer.
//   FAR   beyond ~30 m,     A direction-only field. It has no position,
//         and every          because anything this far away has no measurable
//         aperture with      parallax across a room-sized walk, so pretending
//         no parallax        it has a position is inventing geometry.
//
//  ---------------------------------------------------------------------------
//  WHY DIRECTION-ONLY, GLOBAL, AND FROZEN
//  ---------------------------------------------------------------------------
//  A background model with any positional freedom is the single most effective
//  way to destroy a splat scene. It becomes an infinitely flexible sponge that
//  soaks up every foreground residual: a poorly fitted chair leg is cheaper to
//  explain by bending the background around it than by fixing the chair.
//
//  Three constraints stop that, and all three are load-bearing:
//    * DIRECTION-ONLY. Radiance is a function of the world-space view
//      direction and nothing else. It cannot be parallaxed, so it cannot
//      pretend to be a near object.
//    * GLOBAL. One field for the whole scan, not per-frame, so it cannot
//      absorb per-frame exposure error (that is the trainer's per-frame
//      exposure/gain, which is a different and correctly-shaped parameter).
//    * FROZEN after warm-up. Once the Gaussians start converging, the
//      background stops learning. After the freeze the only way to reduce
//      residual is to fix the actual geometry.
//
//  REPRESENTATION: a low-resolution cubemap, not an MLP.
//  The spec allows either. The cubemap wins on a phone for reasons that are
//  not aesthetic: it is exactly-differentiable with a four-texel bilinear
//  footprint, it needs no autodiff graph or matrix library, it is trivially
//  serialisable, and at 32x32x6 it is 18 KB. An MLP of comparable capacity
//  needs a forward and backward pass per pixel per iteration and buys nothing
//  the sky actually has - the sky is low frequency, which is precisely what a
//  cubemap represents well.
//

import Foundation
import simd

// MARK: - Cubemap

/// The direction-only far field: six square faces of linear RGB.
///
/// Faces are indexed in the standard cubemap order (+X, -X, +Y, -Y, +Z, -Z) in
/// WORLD space, which is ARKit's: right-handed, Y up, metres. This is not the
/// camera frame, and it deliberately does not follow the camera convention -
/// the field is global, so its axes must be too.
public struct SmartBackgroundCubemap: Sendable {
    public static let faceCount = 6

    public private(set) var faceSize: Int
    /// `6 * faceSize * faceSize` linear-RGB texels, face-major then row-major.
    public private(set) var texels: [SIMD3<Float>]

    public init(faceSize: Int = 32, fill: SIMD3<Float> = SIMD3<Float>(repeating: 0.5)) {
        let size = Swift.max(2, faceSize)
        self.faceSize = size
        self.texels = [SIMD3<Float>](
            repeating: fill, count: Self.faceCount * size * size
        )
    }

    public init(faceSize: Int, texels: [SIMD3<Float>]) {
        let size = Swift.max(2, faceSize)
        self.faceSize = size
        if texels.count == Self.faceCount * size * size {
            self.texels = texels
        } else {
            self.texels = [SIMD3<Float>](
                repeating: SIMD3<Float>(repeating: 0.5),
                count: Self.faceCount * size * size
            )
        }
    }

    /// Face index and face-local `(u, v)` in `0...1` for a world direction.
    ///
    /// The standard cube projection: pick the major axis, divide the other two
    /// by its magnitude. The `+ 1) * 0.5` maps `-1...1` to `0...1`.
    public static func faceAndUV(for direction: SIMD3<Float>) -> (face: Int, u: Float, v: Float) {
        let d = simd_length(direction) > 1e-8
            ? simd_normalize(direction)
            : SIMD3<Float>(0, 1, 0)
        let a = abs(d)

        var face = 0
        var sc: Float = 0
        var tc: Float = 0
        var ma: Float = 1

        if a.x >= a.y && a.x >= a.z {
            ma = a.x
            if d.x > 0 { face = 0; sc = -d.z; tc = -d.y }
            else { face = 1; sc = d.z; tc = -d.y }
        } else if a.y >= a.z {
            ma = a.y
            if d.y > 0 { face = 2; sc = d.x; tc = d.z }
            else { face = 3; sc = d.x; tc = -d.z }
        } else {
            ma = a.z
            if d.z > 0 { face = 4; sc = d.x; tc = -d.y }
            else { face = 5; sc = -d.x; tc = -d.y }
        }

        let m = Swift.max(ma, 1e-8)
        return (face, (sc / m + 1) * 0.5, (tc / m + 1) * 0.5)
    }

    /// The four texel indices and weights a direction lands on, which is the
    /// same footprint the gradient is scattered back into. Sharing this
    /// between read and write is what makes the fit consistent.
    ///
    /// Clamped at the face edges rather than wrapped across the seam. A seam
    /// texel therefore blends only within its own face, which produces a
    /// hairline discontinuity at the cube edges - invisible at this
    /// resolution on low-frequency sky, and vastly simpler than getting
    /// cross-face addressing right.
    ///
    /// Static, taking `faceSize` explicitly, so a caller holding a lock over a
    /// mutable cubemap can compute a footprint without touching the shared
    /// value at all.
    static func bilinearFootprint(
        _ direction: SIMD3<Float>,
        faceSize: Int
    ) -> [(index: Int, weight: Float)] {
        let (face, u, v) = Self.faceAndUV(for: direction)
        let n = Swift.max(2, faceSize)
        let x = SmartMath.clamp(u * Float(n) - 0.5, 0, Float(n - 1))
        let y = SmartMath.clamp(v * Float(n) - 0.5, 0, Float(n - 1))
        let x0 = Int(x), y0 = Int(y)
        let x1 = Swift.min(x0 + 1, n - 1), y1 = Swift.min(y0 + 1, n - 1)
        let fx = x - Float(x0), fy = y - Float(y0)
        let base = face * n * n
        return [
            (base + y0 * n + x0, (1 - fx) * (1 - fy)),
            (base + y0 * n + x1, fx * (1 - fy)),
            (base + y1 * n + x0, (1 - fx) * fy),
            (base + y1 * n + x1, fx * fy)
        ]
    }

    func bilinearFootprint(_ direction: SIMD3<Float>) -> [(index: Int, weight: Float)] {
        Self.bilinearFootprint(direction, faceSize: faceSize)
    }

    /// The same blend `bilinearFootprint` describes, WITHOUT the array.
    ///
    /// `bilinearFootprint` returns `[(index, weight)]`, and a Swift array
    /// of tuples is a heap allocation. This function used to call it, and
    /// `TrainerSupervision.backgroundImage` calls this function once per
    /// pixel of the frame it builds: 388,800 allocations and frees per
    /// iteration at 720x540, three thousand times, on the CPU, with the
    /// GPU idle behind it. Measured at 18 to 33 ms of an 82 ms iteration,
    /// which is the largest single item anywhere in the training loop and
    /// is not GPU work at all.
    ///
    /// The header above this file says one cubemap lookup per pixel "is
    /// milliseconds; it is not worth a shader". The lookup was never the
    /// cost. The allocation was.
    ///
    /// EXACT: the four indices, the four weights, the accumulation order
    /// and the `index < texels.count` guard are all what the array version
    /// produced, so the same direction returns the same bits.
    public func radiance(_ direction: SIMD3<Float>) -> SIMD3<Float> {
        let (face, u, v) = Self.faceAndUV(for: direction)
        let n = Swift.max(2, faceSize)
        let x = SmartMath.clamp(u * Float(n) - 0.5, 0, Float(n - 1))
        let y = SmartMath.clamp(v * Float(n) - 0.5, 0, Float(n - 1))
        let x0 = Int(x), y0 = Int(y)
        let x1 = Swift.min(x0 + 1, n - 1), y1 = Swift.min(y0 + 1, n - 1)
        let fx = x - Float(x0), fy = y - Float(y0)
        let base = face * n * n
        let count = texels.count

        // Guarded per entry rather than once, because that is what the
        // `where index < texels.count` clause did: a short texel array
        // dropped individual corners rather than the whole sample.
        var out = SIMD3<Float>.zero
        let i00 = base + y0 * n + x0
        let i01 = base + y0 * n + x1
        let i10 = base + y1 * n + x0
        let i11 = base + y1 * n + x1
        if i00 < count { out += texels[i00] * ((1 - fx) * (1 - fy)) }
        if i01 < count { out += texels[i01] * (fx * (1 - fy)) }
        if i10 < count { out += texels[i10] * ((1 - fx) * fy) }
        if i11 < count { out += texels[i11] * (fx * fy) }
        return out
    }

    mutating func setTexels(_ values: [SIMD3<Float>]) {
        guard values.count == texels.count else { return }
        texels = values
    }

    mutating func addToTexel(_ index: Int, _ delta: SIMD3<Float>) {
        guard index >= 0, index < texels.count else { return }
        texels[index] += delta
    }

    // MARK: Serialisation

    /// `model/background.bin`: little-endian `Float32` RGB triples, face-major
    /// then row-major. Shape lives in `model/background.json`, which is the
    /// convention every other sidecar in `docs/DATA_FORMAT.md` follows.
    public func encoded() -> Data {
        var data = Data(capacity: texels.count * 12)
        for t in texels {
            SmartBinary.append(t.x, to: &data)
            SmartBinary.append(t.y, to: &data)
            SmartBinary.append(t.z, to: &data)
        }
        return data
    }

    public static func decoded(_ data: Data, faceSize: Int) -> SmartBackgroundCubemap? {
        let size = Swift.max(2, faceSize)
        let expected = faceCount * size * size
        guard data.count >= expected * 12 else { return nil }
        let floats = SmartBinary.readFloats(data, offset: 0, count: expected * 3)
        guard floats.count == expected * 3 else { return nil }
        var texels = [SIMD3<Float>](repeating: .zero, count: expected)
        for i in 0..<expected {
            texels[i] = SIMD3<Float>(floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2])
        }
        return SmartBackgroundCubemap(faceSize: size, texels: texels)
    }
}

/// `model/background.json`. Small on purpose: everything a reader needs to
/// interpret `background.bin` and nothing it does not.
public struct SmartBackgroundHeader: Codable, Sendable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    /// Always `"cubemap"` today. Present so a future MLP variant can be added
    /// without a reader silently misinterpreting the bytes.
    public var kind: String
    public var faceSize: Int
    /// Iterations of joint warm-up before the freeze.
    public var warmUpIterations: Int
    /// True once `freeze()` has been called. A model written unfrozen is a
    /// mid-run snapshot, not a result.
    public var frozen: Bool
    /// Far-field pixels that contributed to the fit. Small means the scan
    /// barely saw any sky, and the field is mostly its neutral prior.
    public var contributingSamples: Int
    /// One line of plain language for the QC card.
    public var summary: String

    /// What actually produced the mid-regime (4.5 to 30 m) depth on the run
    /// that wrote this file, in one sentence.
    ///
    /// Optional so a `background.json` written before this field existed still
    /// decodes; a reader that finds it absent knows only that the run did not
    /// record it, which is different from knowing the regime was real.
    public var midRegimeProvenance: String?

    /// False when the mid regime was the on-device stub rather than a real
    /// monocular depth model. Recorded because a run that fell back to
    /// parallax routing produced different geometry from one that did not, and
    /// the difference must be legible in the file a year later.
    public var midRegimeIsReal: Bool?

    public init(
        formatVersion: Int = SmartBackgroundHeader.currentFormatVersion,
        kind: String = "cubemap",
        faceSize: Int,
        warmUpIterations: Int,
        frozen: Bool,
        contributingSamples: Int,
        summary: String,
        midRegimeProvenance: String? = nil,
        midRegimeIsReal: Bool? = nil
    ) {
        self.formatVersion = formatVersion
        self.kind = kind
        self.faceSize = faceSize
        self.warmUpIterations = warmUpIterations
        self.frozen = frozen
        self.contributingSamples = contributingSamples
        self.summary = summary
        self.midRegimeProvenance = midRegimeProvenance
        self.midRegimeIsReal = midRegimeIsReal
    }
}

// MARK: - DirectionalBackgroundModel

/// F5. Implements `BackgroundModel` (CONTRACTS.md section 5).
public final class DirectionalBackgroundModel: BackgroundModel {

    // MARK: Configuration

    private let settings: SmartLossSettings
    private let faceSize: Int
    private let midRegimeProvider: SmartMidRegimeDepthProvider

    // MARK: State

    private let lock = NSLock()
    private var cubemap: SmartBackgroundCubemap
    private var frozen = false
    private var warmUpIterationsRun = 0
    private var contributingSamples = 0

    /// Accumulated gradient and its weight, per texel. Applied and cleared by
    /// `applyAccumulatedGradient`.
    private var gradientAccumulator: [SIMD3<Float>]
    private var gradientWeight: [Float]

    /// Supplied by `prepare`; nil until then, and every authority query then
    /// returns 0, which routes everything to infinity. That is the honest
    /// default: with no data, nothing is authoritative.
    private var authorityMap: SmartAuthorityMap?

    public init(
        settings: SmartLossSettings = .default,
        // 64 since build 326 (was 32): four times the texels, so a window or a
        // doorway keeps its shape instead of a 32-pixel-per-face smear. The
        // trainer samples twice as many pixels per step for its gradient to
        // match, and its GPU copy is sized to 64 (TrainerResources).
        faceSize: Int = 64,
        midRegimeProvider: SmartMidRegimeDepthProvider = SmartMonocularDepthStub()
    ) {
        self.settings = settings
        self.faceSize = Swift.max(2, faceSize)
        self.midRegimeProvider = midRegimeProvider
        // Mid grey, not black. A black prior makes every unobserved direction
        // read as a hole, and a hole is what a user reports as a bug; grey
        // reads as "nothing was seen here", which is the truth.
        self.cubemap = SmartBackgroundCubemap(faceSize: self.faceSize)
        let count = SmartBackgroundCubemap.faceCount * self.faceSize * self.faceSize
        self.gradientAccumulator = [SIMD3<Float>](repeating: .zero, count: count)
        self.gradientWeight = [Float](repeating: 0, count: count)
    }

    /// Wires the authority map. Not part of the protocol: `BackgroundModel`
    /// exposes `authority(frame:sampleIndex:)` as a query, and this is how the
    /// data behind that query arrives.
    public func prepare(_ map: SmartAuthorityMap) {
        lock.lock()
        authorityMap = map
        lock.unlock()
    }

    /// What the mid regime is actually doing on this device, in one sentence.
    /// Surfaced in the QC card so a stubbed run is never mistaken for a real
    /// one.
    public var midRegimeProvenance: String { midRegimeProvider.provenance }

    public var isMidRegimeReal: Bool { midRegimeProvider.isAvailable }

    // MARK: - BackgroundModel: warm-up

    /// Fits the far field from the frames' own far-field pixels.
    ///
    /// HOW THE FIT WORKS, and why it is a weighted mean rather than a gradient
    /// descent: the protocol hands this method a capture bundle and nothing
    /// else. It has no access to the Gaussians, so there is no rendered image
    /// to take a photometric residual against. What it CAN do exactly is the
    /// closed-form answer to "what colour was seen looking in this direction,
    /// over every far-field pixel of every frame, weighted by how sure we are
    /// that the pixel really was far field" - which is the weighted mean, and
    /// is the optimum of the L2 objective a gradient descent would be
    /// crawling towards anyway.
    ///
    /// The joint part of "trained jointly with the Gaussians" is then done by
    /// the trainer through `accumulateGradient` / `applyAccumulatedGradient`
    /// during its warm-up window. This method produces the starting point;
    /// those two refine it; `freeze` ends it.
    ///
    /// `iterations` bounds the work: frames are strided so that roughly
    /// `iterations` of them are visited, because on a 2000-frame house scan
    /// reading every JPEG here would cost more than the training run.
    public func warmUp(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        iterations: Int
    ) async throws {
        lock.lock()
        let alreadyFrozen = frozen
        let map = authorityMap
        lock.unlock()

        guard !alreadyFrozen else {
            SmartLog.background.notice("warmUp called after freeze; ignored, the field is final")
            return
        }

        // Said once per run, at the top, so a log from a scan that came out
        // wrong shows immediately whether the 4.5 to 30 m band was measured or
        // routed around.
        if isMidRegimeReal {
            SmartLog.background.info(
                "Mid regime: \(self.midRegimeProvenance, privacy: .public)"
            )
        } else {
            SmartLog.background.notice(
                """
                Mid regime (4.5 to 30 m) is NOT measured on this device: \
                \(self.midRegimeProvenance, privacy: .public)
                """
            )
        }

        let width = bundle.settings.depthWidth
        let height = bundle.settings.depthHeight
        guard width > 0, height > 0, !bundle.frames.isEmpty else { return }

        let texelCount = SmartBackgroundCubemap.faceCount * faceSize * faceSize
        var sums = [SIMD3<Float>](repeating: .zero, count: texelCount)
        var weights = [Float](repeating: 0, count: texelCount)
        var contributed = 0

        // Visit ~`iterations` frames, spread evenly across the walk rather
        // than taking the first N: the sky a user saw in the last room is as
        // much a part of the background as the sky in the first.
        let visitTarget = Swift.max(1, Swift.min(bundle.frames.count, Swift.max(iterations, 8)))
        let frameStride = Swift.max(1, bundle.frames.count / visitTarget)

        let images = SmartImageCache(capacity: 2, longEdge: Swift.max(width, height))
        let nativeK = SmartCamera.nativeIntrinsics(
            bundle.intrinsics, depthWidth: width, depthHeight: height
        )

        // THE PER-FRAME PREPARATION ON EVERY CORE (build 298). Each visited
        // frame needs a photo decode, a luma and an RGB resample and its
        // authority map (built on a miss), 120 frames one at a time inside
        // the trainer's smartLayer (1.3 s on 276). Those are pure functions of
        // the frame (the image cache and authority map lock their own tables),
        // so they run 12 frames at a time on every core, and the accumulation
        // below walks the prepared frames STRICTLY in the old order: the same
        // frames visited, the same pixels, the same float sums in the same
        // order, the same cubemap to the bit.
        var visitFrames: [CaptureFrame] = []
        var frameCursor = 0
        while frameCursor < bundle.frames.count {
            let frame = bundle.frames[frameCursor]
            frameCursor += frameStride
            guard frame.bracket == .normal else { continue }
            visitFrames.append(frame)
        }

        struct PreparedFrame {
            let luma: [Float]
            let rgb: [SIMD3<Float>]
            let qcWeight: Float
            let authority: SmartFrameAuthority?
            let rotationInverse: simd_quatf
        }

        var visited = 0
        var next = 0
        let prepareChunk = 12
        let frameList = visitFrames
        while next < frameList.count {
            if Task.isCancelled { throw NimbusError.cancelled }
            let base = next
            let chunk = Swift.min(prepareChunk, frameList.count - base)
            var prepared = [PreparedFrame?](repeating: nil, count: chunk)
            prepared.withUnsafeMutableBufferPointer { out in
                DispatchQueue.concurrentPerform(iterations: chunk) { c in
                    let frame = frameList[base + c]
                    let pose = frame.refinedPose ?? frame.rawPose
                    guard let image = images.image(for: frame, at: ref) else { return }
                    let qcWeight = SmartMath.clamp(frame.qc.weight, 0, 1)
                    out[c] = PreparedFrame(
                        luma: SmartAuthorityMap.resampleLuma(image, width: width, height: height),
                        rgb: Self.resampleRGB(image, width: width, height: height),
                        qcWeight: qcWeight,
                        // Only where the serial loop asked for it (past the QC
                        // gate), so the authority cache fills exactly as before.
                        authority: qcWeight > 0.05 ? map?.map(for: frame.index) : nil,
                        rotationInverse: pose.rotation.simd.inverse
                    )
                }
            }
            next += chunk

            for c in 0..<chunk {
                // Per frame, as the serial loop checked it (build 318).
                if Task.isCancelled { throw NimbusError.cancelled }
                guard let frameData = prepared[c] else { continue }
                visited += 1
                let luma = frameData.luma
                let rgb = frameData.rgb
                let qcWeight = frameData.qcWeight
                guard qcWeight > 0.05 else { continue }
                let frameAuthority = frameData.authority
                let rotationInverse = frameData.rotationInverse

                for v in 0..<height {
                    for u in 0..<width {
                        let i = v * width + u

                        let authority = frameAuthority?.value(sampleIndex: i) ?? 0
                        let farness = 1 - SmartMath.clamp(authority, 0, 1)
                        guard farness > 0.5 else { continue }

                        guard luma[i] < settings.saturationLuma else { continue }

                        let pixel = SIMD2<Float>(Float(u) + 0.5, Float(v) + 0.5)
                        let rayCamera = SmartCamera.ray(pixel, nativeK)
                        let direction = rotationInverse.act(rayCamera)

                        let w = farness * qcWeight
                        let colour = rgb[i]
                        let footprint = SmartBackgroundCubemap.bilinearFootprint(
                            direction, faceSize: faceSize
                        )
                        for (index, footprintWeight) in footprint {
                            guard index < texelCount else { continue }
                            let contribution = w * footprintWeight
                            sums[index] += colour * contribution
                            weights[index] += contribution
                        }
                        contributed += 1
                    }
                }
            }
        }

        var texels = [SIMD3<Float>](repeating: SIMD3<Float>(repeating: 0.5), count: texelCount)
        var filled = 0
        for i in 0..<texelCount where weights[i] > 1e-4 {
            texels[i] = sums[i] / weights[i]
            filled += 1
        }

        lock.lock()
        cubemap.setTexels(texels)
        warmUpIterationsRun = iterations
        contributingSamples = contributed
        for i in 0..<gradientAccumulator.count {
            gradientAccumulator[i] = .zero
            gradientWeight[i] = 0
        }
        lock.unlock()

        SmartLog.background.info(
            """
            Background warm-up: \(visited) frames visited, \(contributed) far-field pixels, \
            \(filled) of \(texelCount) texels observed. Mid regime: \
            \(self.midRegimeProvider.provenance, privacy: .public)
            """
        )
    }

    public func freeze() {
        lock.lock()
        let wasFrozen = frozen
        frozen = true
        for i in 0..<gradientAccumulator.count {
            gradientAccumulator[i] = .zero
            gradientWeight[i] = 0
        }
        lock.unlock()
        if !wasFrozen {
            SmartLog.background.info(
                "Background frozen. From here, residual can only be reduced by fixing geometry."
            )
        }
    }

    public var isFrozen: Bool {
        lock.lock()
        defer { lock.unlock() }
        return frozen
    }

    // MARK: - BackgroundModel: query

    /// The whole cubemap, taken once under the lock, to be read without it.
    ///
    /// `radiance(forDirection:)` below locks on EVERY call, and the trainer
    /// calls it once per pixel to build the background image for a frame:
    /// 388,800 lock/unlock pairs per iteration at 720x540, on the CPU, while
    /// the GPU waits. That is the single largest CPU item on the training
    /// loop and it buys nothing, because every one of those calls reads the
    /// same map.
    ///
    /// Taking it once is also MORE correct. Sampling a map that training is
    /// still updating means one image can be built from two different
    /// backgrounds, torn somewhere down the frame. A snapshot cannot tear.
    ///
    /// Cheap: `SmartBackgroundCubemap` is a struct whose storage is a Swift
    /// array, so this is a reference bump under copy-on-write, not a copy of
    /// the texels. If the model mutates afterwards, the snapshot keeps the
    /// values it was taken with, which is the point.
    public var cubemapSnapshot: SmartBackgroundCubemap {
        lock.lock()
        defer { lock.unlock() }
        return cubemap
    }

    public func radiance(forDirection direction: Vector3) -> SIMD3<Float> {
        lock.lock()
        defer { lock.unlock() }
        return cubemap.radiance(direction.simd)
    }

    /// The composite the renderer performs: Gaussians over background, using
    /// the accumulated alpha the rasteriser already has.
    ///
    /// This exists here rather than in the shader so both ends agree exactly
    /// on the operator. `accumulatedAlpha` is the front-to-back accumulated
    /// coverage, so `1 - alpha` is the light that reached infinity.
    public func composite(
        gaussianColor: SIMD3<Float>,
        accumulatedAlpha: Float,
        direction: Vector3
    ) -> SIMD3<Float> {
        let a = SmartMath.clamp(accumulatedAlpha, 0, 1)
        return gaussianColor + (1 - a) * radiance(forDirection: direction)
    }

    public func authority(frame: FrameID, sampleIndex: Int) -> Float {
        lock.lock()
        let map = authorityMap
        lock.unlock()
        // No authority map means no evidence has been assembled, and the
        // honest answer to "may I believe this depth" is then "no".
        return map?.authority(frame: frame, sampleIndex: sampleIndex) ?? 0
    }

    /// Which regime a sample fell into. Not on the protocol; the trainer needs
    /// it to decide whether a pixel gets depth supervision at all.
    public func regime(frame: FrameID, sampleIndex: Int) -> SmartDepthRegime {
        lock.lock()
        let map = authorityMap
        lock.unlock()
        return map?.regime(frame: frame, sampleIndex: sampleIndex) ?? .far
    }

    // MARK: - Joint refinement during warm-up

    /// Scatters one pixel's photometric residual into the field.
    ///
    /// `dLossDRadiance` is `d(loss) / d(background radiance)` for this pixel,
    /// which for the usual L1/L2 photometric term the trainer already has.
    /// `weight` should carry the `(1 - accumulatedAlpha)` factor, because a
    /// pixel the Gaussians already cover opaquely tells you nothing about what
    /// is behind them.
    ///
    /// Silently ignored after `freeze()`. That is the intended behaviour and
    /// it is checked, not assumed: a trainer that keeps calling this past the
    /// freeze must not quietly keep training the sky.
    public func accumulateGradient(
        direction: Vector3,
        dLossDRadiance: SIMD3<Float>,
        weight: Float
    ) {
        guard weight > 0, dLossDRadiance.x.isFinite,
              dLossDRadiance.y.isFinite, dLossDRadiance.z.isFinite
        else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !frozen else { return }
        let footprint = SmartBackgroundCubemap.bilinearFootprint(
            direction.simd, faceSize: faceSize
        )
        for (index, footprintWeight) in footprint {
            guard index >= 0, index < gradientAccumulator.count else { continue }
            let w = weight * footprintWeight
            gradientAccumulator[index] += dLossDRadiance * w
            gradientWeight[index] += w
        }
    }

    /// Applies one step of the accumulated gradient and clears it.
    ///
    /// Normalised per texel by the weight that landed on it, so a texel the
    /// camera stared at for two hundred frames does not take a step two
    /// hundred times larger than one glimpsed once. Returns the mean step
    /// magnitude, which the trainer can log to see the field settling.
    @discardableResult
    public func applyAccumulatedGradient(learningRate: Float) -> Float {
        lock.lock()
        defer { lock.unlock() }
        guard !frozen, learningRate > 0 else { return 0 }

        var totalStep: Float = 0
        var stepped = 0
        for i in 0..<gradientAccumulator.count {
            let w = gradientWeight[i]
            guard w > 1e-6 else { continue }
            let step = (gradientAccumulator[i] / w) * learningRate
            cubemap.addToTexel(i, -step)
            totalStep += simd_length(step)
            stepped += 1
            gradientAccumulator[i] = .zero
            gradientWeight[i] = 0
        }
        // Radiance below zero is not a colour; above 8 it is a numerical
        // runaway rather than a bright sky. Both are clamped rather than
        // allowed to poison a frozen, exported field.
        var texels = cubemap.texels
        for i in 0..<texels.count {
            texels[i] = simd_clamp(texels[i], SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 8))
            if !texels[i].x.isFinite || !texels[i].y.isFinite || !texels[i].z.isFinite {
                texels[i] = SIMD3<Float>(repeating: 0.5)
            }
        }
        cubemap.setTexels(texels)

        return stepped > 0 ? totalStep / Float(stepped) : 0
    }

    // MARK: - BackgroundModel: write

    /// Writes `model/background.bin` and `model/background.json`, and returns
    /// the relative path of the binary, which is what `SplatModel`'s
    /// `backgroundModelPath` holds.
    public func write(to ref: CaptureBundleRef) async throws -> String {
        lock.lock()
        let snapshot = cubemap
        let isFrozen = frozen
        let iterations = warmUpIterationsRun
        let samples = contributingSamples
        lock.unlock()

        let folder = BrandConfig.Folder.model
        let binaryPath = "\(folder)/background.bin"
        let headerPath = "\(folder)/background.json"

        try SmartBinary.write(snapshot.encoded(), to: ref.url(forRelativePath: binaryPath))

        var summary: String
        if samples == 0 {
            summary = "No sky or distant surface was ever visible, so the background is neutral grey."
        } else if isFrozen {
            summary = "Distant surfaces were fitted from \(samples) pixels and then locked."
        } else {
            summary = "Distant surfaces were fitted from \(samples) pixels and are still being refined."
        }

        // THE LIMITATION THIS FILE IS ALLOWED TO ADMIT.
        //
        // The mid regime (4.5 to 30 m) is a stub on the phone: it returns
        // nothing and that range falls back to parallax routing plus this far
        // field. `midRegimeProvenance` was written to say so and nothing read
        // it, so no scan has ever recorded which of the two it got. It goes
        // into the summary as well as its own field, because a reader that
        // only shows the one line still has to see it.
        let midProvenance = midRegimeProvenance
        let midIsReal = isMidRegimeReal
        if !midIsReal {
            summary += " Mid-range depth (4.5 to 30 m) was not measured: \(midProvenance)"
        }

        let header = SmartBackgroundHeader(
            faceSize: snapshot.faceSize,
            warmUpIterations: iterations,
            frozen: isFrozen,
            contributingSamples: samples,
            summary: summary,
            midRegimeProvenance: midProvenance,
            midRegimeIsReal: midIsReal
        )
        let json = try ContractsJSON.encoder().encode(header)
        try SmartBinary.write(json, to: ref.url(forRelativePath: headerPath))

        SmartLog.background.info(
            "Background written to \(binaryPath, privacy: .public) (frozen: \(isFrozen))"
        )
        return binaryPath
    }

    /// Reads back a field written by `write`, for the viewer and for resuming
    /// a Booster job. Loaded fields arrive frozen: a field that has already
    /// been exported is a result, not a starting point.
    public func load(from ref: CaptureBundleRef) throws {
        let folder = BrandConfig.Folder.model
        let headerURL = ref.url(forRelativePath: "\(folder)/background.json")
        let binaryURL = ref.url(forRelativePath: "\(folder)/background.bin")

        let headerData = try SmartBinary.map(headerURL)
        let header = try ContractsJSON.decoder().decode(SmartBackgroundHeader.self, from: headerData)
        // A reader that sees a version or a representation it does not know
        // must refuse, not guess (docs/DATA_FORMAT.md section 9).
        guard header.formatVersion == SmartBackgroundHeader.currentFormatVersion else {
            throw NimbusError.malformedData(
                "background.json is format version \(header.formatVersion); "
                    + "this build reads version \(SmartBackgroundHeader.currentFormatVersion)."
            )
        }
        guard header.kind == "cubemap" else {
            throw NimbusError.malformedData(
                "background.json describes a '\(header.kind)' background; "
                    + "this build only reads 'cubemap'."
            )
        }

        let binary = try SmartBinary.map(binaryURL)
        guard let decoded = SmartBackgroundCubemap.decoded(binary, faceSize: header.faceSize)
        else {
            throw SmartError.malformedSidecar(
                path: "\(folder)/background.bin",
                expectedBytes: SmartBackgroundCubemap.faceCount
                    * header.faceSize * header.faceSize * 12,
                actualBytes: binary.count
            )
        }

        lock.lock()
        cubemap = decoded
        frozen = true
        warmUpIterationsRun = header.warmUpIterations
        contributingSamples = header.contributingSamples
        lock.unlock()
    }

    /// Snapshot for the viewer and for tests.
    public var currentCubemap: SmartBackgroundCubemap {
        lock.lock()
        defer { lock.unlock() }
        return cubemap
    }

    // MARK: - Helpers

    static func resampleRGB(_ image: SmartImage, width: Int, height: Int) -> [SIMD3<Float>] {
        if image.width == width && image.height == height { return image.rgb }
        var out = [SIMD3<Float>](repeating: .zero, count: width * height)
        guard image.width > 0, image.height > 0, width > 0, height > 0 else { return out }
        for y in 0..<height {
            let y0 = y * image.height / height
            let y1 = Swift.max(y0 + 1, (y + 1) * image.height / height)
            for x in 0..<width {
                let x0 = x * image.width / width
                let x1 = Swift.max(x0 + 1, (x + 1) * image.width / width)
                var sum = SIMD3<Float>.zero
                var count = 0
                for iy in y0..<Swift.min(y1, image.height) {
                    for ix in x0..<Swift.min(x1, image.width) {
                        sum += image.rgb[iy * image.width + ix]
                        count += 1
                    }
                }
                out[y * width + x] = count > 0 ? sum / Float(count) : .zero
            }
        }
        return out
    }
}
