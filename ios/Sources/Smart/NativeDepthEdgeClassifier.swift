//
//  NativeDepthEdgeClassifier.swift
//  Smart
//
//  F3: native depth, the edge band, and the WHERE / WHAT split.
//
//  ---------------------------------------------------------------------------
//  THE IDEA, because it is the one most likely to be "optimised" away later
//  ---------------------------------------------------------------------------
//  ARKit will happily hand you a depth map at RGB resolution. It is 256x192
//  real samples with an edge-aware upsampler between them. Supervising a splat
//  field against those interpolated pixels does not teach it the scene; it
//  teaches it the interpolator, and the interpolator's signature failure is
//  precisely at depth discontinuities, where it smears the foreground into the
//  background over several pixels.
//
//  So: depth supervision happens ONLY at the ~49k native samples. Depth edges
//  are found on the NATIVE map, dilated by the upsample ratio, and inside that
//  band the depth loss is zero. Not down-weighted - zero. The upsampled value
//  there is not noisy, it is wrong, and averaging a wrong value in is worse
//  than having no value.
//
//  Then the WHERE / WHAT split. An edge in the image is one of three things:
//
//    WHERE   `geometric`  the depth genuinely steps. Sharpen: this is real
//                         geometry and the bimodal loss in SmartDepthLoss
//                         pushes rendered depth onto one side or the other
//                         rather than letting it sit in the middle.
//    WHAT    `texture`    a strong image gradient across flat depth: a poster,
//                         a rug, a skirting board's paint line. Flatten. A
//                         3DGS optimiser left alone will invent a ridge here,
//                         because a ridge is a cheap way to explain a hard
//                         colour edge.
//    neither `unknown`    glass, beyond range, or a sample we do not believe.
//                         Ignore. Do not guess, in either direction.
//

import Foundation
import simd

/// F3. Implements `EdgeClassifier` (CONTRACTS.md §5).
public final class NativeDepthEdgeClassifier: EdgeClassifier {

    // MARK: Configuration

    private let settings: SmartLossSettings

    // MARK: State

    private var depthWidth: Int = 0
    private var depthHeight: Int = 0
    private var lidarMaxRange: Float = 5
    private var refs: EdgeClassificationRefs?
    private var bundleRef: CaptureBundleRef?
    /// Relative `prepass/edges/frame_*.edge8` path per frame.
    private var pathsByFrame: [FrameID: String] = [:]
    /// Every frame with depth, so a map nobody has built yet can be built
    /// on first use. Set by `classify` and by `load`.
    private var framesByID: [FrameID: CaptureFrame] = [:]

    private var cacheOrder: [FrameID] = []
    private var cache: [FrameID: [EdgeClass]] = [:]
    private let cacheCapacity: Int
    private let lock = NSLock()

    /// SIZED TO COVER A KEYFRAME CYCLE, not to be small.
    ///
    /// The default was 6. The trainer shuffles its keyframes ONCE and then
    /// walks them round-robin, so the reuse distance is the whole cycle: about
    /// 114 frames on the owner's scan. An LRU smaller than the cycle has a 0%
    /// hit rate, not a low one, so every iteration rebuilt a frame it had
    /// already built about 26 times. Raising 6 to 24 would have changed
    /// nothing for the same reason; only covering the cycle helps.
    ///
    /// 128 is chosen from memory, not from the cycle: an edge map is about 49 KB, so 128 of them is roughly 6 MB. That is affordable
    /// against a 687 MB peak and a 1.59 GB ceiling, and it is bounded no
    /// matter how large the capture is, which a "just cache everything" rule
    /// would not be.
    public init(settings: SmartLossSettings = .default, cacheCapacity: Int = 128) {
        self.settings = settings
        self.cacheCapacity = Swift.max(1, cacheCapacity)
    }

    // MARK: - EdgeClassifier

    public func classify(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> EdgeClassificationRefs {
        let width = bundle.settings.depthWidth
        let height = bundle.settings.depthHeight
        guard width > 1, height > 1 else {
            throw SmartError.malformedSidecar(
                path: "capture_bundle.json settings.depthWidth/Height",
                expectedBytes: 2,
                actualBytes: width * height
            )
        }

        // BUILT ON FIRST USE, NOT ALL UP FRONT.
        //
        // This used to walk EVERY frame in the capture, 868 on the owner's
        // scan, decoding a JPEG, reading two sidecars and writing a map for
        // each, on the pre-pass critical path. The readers are the seeder
        // (174 keyframes) and the trainer (its own ~120 keyframes plus the
        // held-out frames), so roughly two thirds of that work was written
        // and never opened. `map(for:)` now builds a missing map the first
        // time anyone asks for it, writes it where this used to, and caches
        // it: the seeder pays for its frames, the trainer pays for its frames
        // on its supervision prefetch worker, off the critical path.
        //
        // The directory is CLEARED first, which is what rewriting every file
        // used to amount to, so `load` in the trainer can never register a
        // stale map left by an older run of the same scan.
        let directory = "\(BrandConfig.Folder.prePass)/edges"
        let directoryURL = ref.url(forRelativePath: directory)
        try? FileManager.default.removeItem(at: directoryURL)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)

        let produced = EdgeClassificationRefs(
            directory: directory,
            bandRadiusNativePixels: settings.edgeBandRadiusNativePixels
        )
        var frames: [FrameID: CaptureFrame] = [:]
        for frame in bundle.frames where frame.depthPath != nil {
            frames[frame.index] = frame
        }

        lock.lock()
        depthWidth = width
        depthHeight = height
        lidarMaxRange = bundle.settings.lidarMaxRangeMeters
        refs = produced
        bundleRef = ref
        framesByID = frames
        pathsByFrame = [:]
        cache.removeAll()
        cacheOrder.removeAll()
        lock.unlock()

        SmartLog.edges.info("Edge maps for \(frames.count) frames will be built on first use")
        return produced
    }

    /// The classification map for one frame, `depthWidth * depthHeight` bytes.
    ///
    /// Returns an EMPTY array when this frame has no map - because the frame
    /// had no depth sidecar, or because `load` was never called. Empty means
    /// "no opinion", and every caller in this module treats it that way
    /// (`SmartDepthLoss` falls back to `.none` for every sample, which is the
    /// generic-3DGS behaviour, not a crash and not a guess).
    public func map(for frame: FrameID) -> [EdgeClass] {
        lock.lock()
        if let hit = cache[frame] {
            lock.unlock()
            return hit
        }
        let path = pathsByFrame[frame]
        let ref = bundleRef
        let expected = depthWidth * depthHeight
        let captureFrame = framesByID[frame]
        let directory = refs?.directory
        lock.unlock()

        guard let ref, expected > 0 else { return [] }

        var classes = [EdgeClass](repeating: .none, count: expected)
        if let path {
            guard let data = try? SmartBinary.map(ref.url(forRelativePath: path)),
                  data.count >= expected
            else {
                SmartLog.edges.error("Edge map for frame \(frame) missing or short at \(path, privacy: .public)")
                return []
            }
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                for i in 0..<expected {
                    classes[i] = EdgeClass(rawValue: raw[i]) ?? .none
                }
            }
        } else {
            // Not built yet: build it now. Computed OUTSIDE the lock, so the
            // trainer's main thread and its prefetch worker never wait on
            // each other; if both ask for the same frame at once they both
            // build identical bytes, which is wasted work and nothing worse.
            guard let captureFrame, let directory,
                  let built = buildMap(frame: captureFrame, ref: ref, directory: directory)
            else { return [] }
            classes = built.classes
            lock.lock()
            pathsByFrame[frame] = built.path
            lock.unlock()
        }

        lock.lock()
        cache[frame] = classes
        cacheOrder.append(frame)
        while cacheOrder.count > cacheCapacity {
            let evicted = cacheOrder.removeFirst()
            cache.removeValue(forKey: evicted)
        }
        lock.unlock()
        return classes
    }

    /// One frame's map: exactly what `classify` used to do per frame, now
    /// run on first use. Returns the classes and, when the write succeeded,
    /// the relative path, so later calls read the file instead of rebuilding.
    private func buildMap(
        frame: CaptureFrame, ref: CaptureBundleRef, directory: String
    ) -> (classes: [EdgeClass], path: String?)? {
        return autoreleasepool { () -> (classes: [EdgeClass], path: String?)? in
            lock.lock()
            let width = depthWidth
            let height = depthHeight
            lock.unlock()
            let sampleCount = width * height
            guard sampleCount > 0, let depthPath = frame.depthPath else { return nil }

            let depth: [Float]
            do {
                depth = try SmartBinary.readDepth16(
                    ref.url(forRelativePath: depthPath), count: sampleCount
                )
            } catch {
                SmartLog.edges.error(
                    "Frame \(frame.index) depth unreadable, skipped: \(String(describing: error), privacy: .public)"
                )
                return nil
            }

            let confidence: [UInt8]
            if let confidencePath = frame.confidencePath,
               let read = try? SmartBinary.readConfidence8(
                   ref.url(forRelativePath: confidencePath), count: sampleCount
               ) {
                confidence = read
            } else {
                confidence = [UInt8](repeating: 1, count: sampleCount)
            }

            let imageCache = SmartImageCache(capacity: 1, longEdge: Swift.max(width, height))
            let luma = nativeLuma(
                frame: frame, ref: ref, cache: imageCache, width: width, height: height
            )
            let classes = classifyFrame(
                depth: depth,
                confidence: confidence,
                luma: luma,
                width: width,
                height: height
            )

            let relative = "\(directory)/\(Self.stem(forImagePath: frame.imagePath)).edge8"
            var bytes = Data(capacity: sampleCount)
            for c in classes { bytes.append(c.rawValue) }
            let written = (try? SmartBinary.write(bytes, to: ref.url(forRelativePath: relative))) != nil
            return (classes: classes, path: written ? relative : nil)
        }
    }

    // MARK: - Loading maps somebody else wrote

    /// Points this instance at maps produced by an earlier pre-pass run, so
    /// the trainer can use them without re-classifying.
    ///
    /// Not part of the `EdgeClassifier` protocol: the protocol only has to
    /// cover produce-and-read within one process. This is what makes
    /// "capture on Monday, train on Tuesday" work.
    public func load(
        _ refs: EdgeClassificationRefs,
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) {
        var localPaths: [FrameID: String] = [:]
        var frames: [FrameID: CaptureFrame] = [:]
        for frame in bundle.frames where frame.depthPath != nil {
            frames[frame.index] = frame
            let relative = "\(refs.directory)/\(Self.stem(forImagePath: frame.imagePath)).edge8"
            if FileManager.default.fileExists(atPath: ref.url(forRelativePath: relative).path) {
                localPaths[frame.index] = relative
            }
        }

        lock.lock()
        depthWidth = bundle.settings.depthWidth
        depthHeight = bundle.settings.depthHeight
        lidarMaxRange = bundle.settings.lidarMaxRangeMeters
        self.refs = refs
        bundleRef = ref
        pathsByFrame = localPaths
        framesByID = frames
        cache.removeAll()
        cacheOrder.removeAll()
        lock.unlock()

        SmartLog.edges.info("Loaded \(localPaths.count) existing edge maps from \(refs.directory, privacy: .public)")
    }

    /// True when `map(for:)` can return something for at least one frame.
    public var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        // Loaded means "can answer", and a map not built yet is built on
        // first use, so this no longer needs a single file on disk.
        return bundleRef != nil && !framesByID.isEmpty
    }

    // MARK: - The classifier itself

    /// Pure function, no IO, so it is trivially testable and reusable by the
    /// Booster's Python port as a reference.
    ///
    /// Order of precedence is deliberate and is the whole design:
    /// `unknown` beats `geometric` beats `band` beats `texture` beats `none`.
    /// A sample we do not believe never becomes evidence of an edge, and a
    /// real depth step is never demoted to "texture" just because there also
    /// happens to be paint on it.
    func classifyFrame(
        depth: [Float],
        confidence: [UInt8],
        luma: [Float]?,
        width: Int,
        height: Int
    ) -> [EdgeClass] {
        let n = width * height
        var classes = [EdgeClass](repeating: .none, count: n)
        guard depth.count >= n else { return classes }

        // --- 1. Validity. No return, out of range, or lowest confidence. ----
        var valid = [Bool](repeating: false, count: n)
        for i in 0..<n {
            let z = depth[i]
            let hasReturn = z > 0 && SmartMath.isUsableDepth(z)
            let inRange = z <= lidarMaxRange
            let believable = i < confidence.count ? confidence[i] > 0 : true
            valid[i] = hasReturn && inRange && believable
            if !valid[i] { classes[i] = .unknown }
        }

        // --- 2. Depth steps on the NATIVE map. ------------------------------
        // Forward differences against the 4-neighbourhood. The threshold is
        // relative to range because sensor noise is, so a 4 cm step at 40 cm
        // is an edge and the same 4 cm at 5 m is not.
        var isStep = [Bool](repeating: false, count: n)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                guard valid[i] else { continue }
                let z = depth[i]
                let threshold = Swift.max(
                    settings.geometricEdgeMinStepMeters,
                    settings.geometricEdgeRelativeStep * z
                )
                var maxStep: Float = 0
                if x > 0, valid[i - 1] { maxStep = Swift.max(maxStep, abs(z - depth[i - 1])) }
                if x + 1 < width, valid[i + 1] { maxStep = Swift.max(maxStep, abs(z - depth[i + 1])) }
                if y > 0, valid[i - width] { maxStep = Swift.max(maxStep, abs(z - depth[i - width])) }
                if y + 1 < height, valid[i + width] { maxStep = Swift.max(maxStep, abs(z - depth[i + width])) }

                // A valid sample whose every neighbour is a no-return is a
                // silhouette against nothing: that is a depth edge too, and it
                // is exactly what the edge of a window frame looks like.
                let neighbourInvalid =
                    (x > 0 && !valid[i - 1])
                    || (x + 1 < width && !valid[i + 1])
                    || (y > 0 && !valid[i - width])
                    || (y + 1 < height && !valid[i + width])

                if maxStep > threshold || neighbourInvalid {
                    isStep[i] = true
                }
            }
        }

        for i in 0..<n where isStep[i] {
            classes[i] = .geometric
        }

        // --- 3. Dilate into the band. ---------------------------------------
        // The upsampled RGB-resolution depth is wrong within the upsample
        // ratio of a step. One native pixel is ~7.5 RGB pixels at 1920 wide,
        // so radius 1 here IS the ~8 px band the design calls for.
        let radius = Swift.max(0, settings.edgeBandRadiusNativePixels)
        if radius > 0 {
            var band = [Bool](repeating: false, count: n)
            for y in 0..<height {
                for x in 0..<width where isStep[y * width + x] {
                    let y0 = Swift.max(0, y - radius), y1 = Swift.min(height - 1, y + radius)
                    let x0 = Swift.max(0, x - radius), x1 = Swift.min(width - 1, x + radius)
                    for by in y0...y1 {
                        for bx in x0...x1 {
                            band[by * width + bx] = true
                        }
                    }
                }
            }
            for i in 0..<n where band[i] && classes[i] == .none {
                classes[i] = .band
            }
        }

        // --- 4. Texture: strong image gradient over flat depth. -------------
        guard let luma, luma.count >= n else { return classes }

        // Sobel on the native-resolution luma, normalised so the threshold is
        // resolution independent. The 1/8 is the Sobel kernel's own gain.
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                guard classes[i] == .none, valid[i] else { continue }

                let tl = luma[i - width - 1], tc = luma[i - width], tr = luma[i - width + 1]
                let ml = luma[i - 1], mr = luma[i + 1]
                let bl = luma[i + width - 1], bc = luma[i + width], br = luma[i + width + 1]

                let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
                let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
                let gradient = sqrt(gx * gx + gy * gy) * 0.125

                guard gradient > settings.textureEdgeGradient else { continue }

                // "Flat depth" is measured, not assumed: the local depth
                // spread must be well under this sample's own step threshold,
                // otherwise this is a geometric edge the step test only just
                // missed and calling it texture would flatten real geometry.
                let z = depth[i]
                var lo = z, hi = z
                var neighbours = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let j = i + dy * width + dx
                        guard valid[j] else { continue }
                        lo = Swift.min(lo, depth[j])
                        hi = Swift.max(hi, depth[j])
                        neighbours += 1
                    }
                }
                let threshold = Swift.max(
                    settings.geometricEdgeMinStepMeters,
                    settings.geometricEdgeRelativeStep * z
                )
                if neighbours >= 6 && (hi - lo) < 0.5 * threshold {
                    classes[i] = .texture
                }
            }
        }

        return classes
    }

    // MARK: - Helpers

    /// The frame's image decoded to exactly the native depth grid, as luma.
    /// Returns nil when the JPEG cannot be read, which downgrades the frame to
    /// depth-only classification (no `texture` class) rather than failing it.
    private func nativeLuma(
        frame: CaptureFrame,
        ref: CaptureBundleRef,
        cache: SmartImageCache,
        width: Int,
        height: Int
    ) -> [Float]? {
        guard let image = cache.image(for: frame, at: ref) else { return nil }
        if image.width == width && image.height == height { return image.luma }

        // Box-filter down to the native grid. A box filter rather than a
        // bilinear point sample on purpose: we are asking "is there a strong
        // gradient in the region this native sample covers", and point
        // sampling a 1920-wide image at 256 columns aliases that question
        // into noise.
        var out = [Float](repeating: 0, count: width * height)
        let sx = Float(image.width) / Float(width)
        let sy = Float(image.height) / Float(height)
        for y in 0..<height {
            let y0 = Int(Float(y) * sy), y1 = Swift.max(y0 + 1, Int(Float(y + 1) * sy))
            for x in 0..<width {
                let x0 = Int(Float(x) * sx), x1 = Swift.max(x0 + 1, Int(Float(x + 1) * sx))
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

    /// `images/frame_20260903_141205_512.jpg` -> `frame_20260903_141205_512`.
    /// The stamp is the join key across every sidecar folder
    /// (`docs/DATA_FORMAT.md` section 1).
    static func stem(forImagePath path: String) -> String {
        let base = path.split(separator: "/").last.map(String.init) ?? path
        guard let dot = base.lastIndex(of: ".") else { return base }
        return String(base[base.startIndex..<dot])
    }
}
