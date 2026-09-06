//
//  PrePassSensorIO.swift
//  PrePass
//
//  READING WHAT THE CAPTURE WROTE, AND THE SPARSE GRID EVERYTHING IS BINNED
//  INTO.
//
//  Three things live here:
//
//   1. `PrePassVoxelHash` - an open-addressing UInt64 -> slot map. Every
//      spatial structure in this module (occupancy, trust bias, surface
//      consensus, co-visibility) is a sparse voxel grid keyed by Morton code,
//      and Swift's `Dictionary` costs roughly 3x the memory and 4x the time
//      for this access pattern. A house scan at 5 cm is tens of millions of
//      cells, so that difference is the difference between running and not.
//
//   2. `PrePassDepthFrame` - the NATIVE 256x192 depth + confidence sidecars
//      (docs/DATA_FORMAT.md section 5), plus unprojection. Never the smoothed
//      or upsampled map: the whole of F3 rests on supervising only the ~49k
//      real samples.
//
//   3. `PrePassImageLoader` - downsampled greyscale / RGB decode of the JPEG
//      frames, via ImageIO's thumbnail path so a 1920x1440 JPEG is never fully
//      decoded just to sample it at 256x192.
//

import Foundation
import simd
import CoreGraphics
import ImageIO
import Accelerate

// MARK: - Sparse voxel hash

/// Open-addressing hash from a 64-bit Morton key to a dense slot index.
///
/// The caller owns the payload arrays and indexes them with the returned slot,
/// which keeps this type free of generics and lets a payload be several
/// parallel arrays of different element types (counts, sums, flags) without
/// boxing anything.
struct PrePassVoxelHash {
    private var keys: [UInt64]
    private var used: [Bool]
    private var slots: [Int32]
    private var mask: UInt64
    /// Number of distinct keys inserted; also the next slot index.
    private(set) var count: Int = 0

    init(expectedCount: Int = 1 << 12) {
        var capacity = 16
        // Keep the load factor under ~0.6, which is where linear probing stops
        // being cheap.
        while capacity < Swift.max(16, expectedCount * 2) { capacity <<= 1 }
        keys = [UInt64](repeating: 0, count: capacity)
        used = [Bool](repeating: false, count: capacity)
        slots = [Int32](repeating: -1, count: capacity)
        mask = UInt64(capacity - 1)
    }

    /// SplitMix64's finaliser. Morton keys have very structured low bits (every
    /// third bit is one axis), and the identity hash would pile every cell of
    /// one row into the same probe sequence.
    @inline(__always)
    private static func scramble(_ key: UInt64) -> UInt64 {
        var z = key &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Slot for `key`, or nil if it was never inserted.
    @inline(__always)
    func index(of key: UInt64) -> Int? {
        var probe = PrePassVoxelHash.scramble(key) & mask
        while used[Int(probe)] {
            if keys[Int(probe)] == key { return Int(slots[Int(probe)]) }
            probe = (probe &+ 1) & mask
        }
        return nil
    }

    /// Slot for `key`, inserting a fresh one if needed.
    /// - Returns: the slot index and whether this call created it. A `true`
    ///   means the caller must append a default payload entry.
    @discardableResult
    mutating func indexOrInsert(_ key: UInt64) -> (index: Int, inserted: Bool) {
        if (count + 1) * 5 >= keys.count * 3 { grow() }
        var probe = PrePassVoxelHash.scramble(key) & mask
        while used[Int(probe)] {
            if keys[Int(probe)] == key { return (Int(slots[Int(probe)]), false) }
            probe = (probe &+ 1) & mask
        }
        let slot = count
        used[Int(probe)] = true
        keys[Int(probe)] = key
        slots[Int(probe)] = Int32(slot)
        count += 1
        return (slot, true)
    }

    private mutating func grow() {
        let newCapacity = keys.count << 1
        var newKeys = [UInt64](repeating: 0, count: newCapacity)
        var newUsed = [Bool](repeating: false, count: newCapacity)
        var newSlots = [Int32](repeating: -1, count: newCapacity)
        let newMask = UInt64(newCapacity - 1)

        for i in 0..<keys.count where used[i] {
            var probe = PrePassVoxelHash.scramble(keys[i]) & newMask
            while newUsed[Int(probe)] { probe = (probe &+ 1) & newMask }
            newUsed[Int(probe)] = true
            newKeys[Int(probe)] = keys[i]
            newSlots[Int(probe)] = slots[i]
        }
        keys = newKeys
        used = newUsed
        slots = newSlots
        mask = newMask
    }

    /// Every (key, slot) pair, in arbitrary order. Callers that need a
    /// deterministic file sort by key afterwards.
    func entries() -> [(key: UInt64, slot: Int)] {
        var out: [(key: UInt64, slot: Int)] = []
        out.reserveCapacity(count)
        for i in 0..<keys.count where used[i] {
            out.append((keys[i], Int(slots[i])))
        }
        return out
    }
}

// MARK: - Voxel coordinate frame

/// The shared "world metres -> integer cell" mapping. Every sparse grid in the
/// module is built on one of these, so the occupancy grid, the trust field and
/// the co-visibility partitioner all agree on where a cell boundary is.
struct PrePassVoxelFrame {
    let origin: SIMD3<Float>
    let voxelSize: Float
    private let inverseVoxelSize: Float

    init(origin: SIMD3<Float>, voxelSize: Float) {
        self.origin = origin
        self.voxelSize = Swift.max(voxelSize, 1e-4)
        self.inverseVoxelSize = 1 / self.voxelSize
    }

    /// A frame covering `bounds` with a margin, so a point right on the
    /// boundary still lands in a representable cell.
    init(bounds: BoundingBox, voxelSize: Float, marginMeters: Float = 1.0) {
        let minimum = bounds.min.simd - SIMD3<Float>(repeating: marginMeters)
        self.init(origin: minimum, voxelSize: voxelSize)
    }

    /// The voxel a world point falls in, WITHOUT the possibility of trapping.
    ///
    /// This is the pre-pass twin of the bug that was killing the app during
    /// capture. `Int32(someFloat)`, like `Int64(someFloat)`, is a trapping
    /// conversion: it kills the process on NaN, on infinity, and on any finite
    /// value outside `Int32`. This used to be three bare `Int32(...)` calls.
    ///
    /// A non-finite point is not hypothetical here. Poses are read back from
    /// disk, and until the capture fix landed a frame with an unusable ARKit
    /// transform could be written down with a NaN pose. Any scan recorded
    /// before that fix still has one on disk, so this path must survive reading
    /// it rather than dying during the check-over.
    ///
    /// Note `key(_:)` below already returns an Optional because
    /// `PrePassMorton.key(cell:)` range-checks the cell, and that check was
    /// useless: it received the value AFTER this function had already trapped
    /// on it. Exactly the same "the guard is one stack frame too late" mistake
    /// the capture crash turned on.
    ///
    /// The sentinel is deliberately far outside any real scan, so
    /// `PrePassMorton.key` rejects it and the sample is dropped rather than
    /// silently landing in a real voxel and corrupting the occupancy grid.
    @inline(__always)
    func cell(_ point: SIMD3<Float>) -> SIMD3<Int32> {
        let local = (point - origin) * inverseVoxelSize
        return SIMD3<Int32>(
            Self.cellIndex(local.x),
            Self.cellIndex(local.y),
            Self.cellIndex(local.z)
        )
    }

    @inline(__always)
    static func cellIndex(_ value: Float) -> Int32 {
        let floored = value.rounded(.down)
        guard floored.isFinite else { return Int32.min }
        let limit: Float = 1_000_000_000
        return Int32(Swift.max(-limit, Swift.min(limit, floored)))
    }

    @inline(__always)
    func key(_ point: SIMD3<Float>) -> UInt64? {
        PrePassMorton.key(cell: cell(point))
    }

    /// Centre of the cell a key names.
    @inline(__always)
    func centre(ofKey key: UInt64) -> SIMD3<Float> {
        let (x, y, z) = PrePassMorton.decode(key)
        return origin + (SIMD3<Float>(Float(x), Float(y), Float(z)) + SIMD3<Float>(repeating: 0.5)) * voxelSize
    }

    /// Minimum corner of the cell a key names.
    @inline(__always)
    func corner(ofKey key: UInt64) -> SIMD3<Float> {
        let (x, y, z) = PrePassMorton.decode(key)
        return origin + SIMD3<Float>(Float(x), Float(y), Float(z)) * voxelSize
    }
}

// MARK: - Native depth sidecars

/// One frame's NATIVE LiDAR depth map and ARKit confidence map, exactly as
/// `docs/DATA_FORMAT.md` section 5 defines them: row-major, top-left origin,
/// `UInt16` little-endian millimetres, `0` meaning no return.
struct PrePassDepthFrame {
    let width: Int
    let height: Int
    /// Millimetres. `0` is a NO RETURN, which is not "distance zero" and is
    /// not "empty" - see F2.
    let depthMillimeters: [UInt16]
    /// ARKit confidence, `0` low / `1` medium / `2` high. Empty when the
    /// sidecar was absent; `confidenceLevel(at:)` then reports medium so a
    /// missing file degrades to "no opinion" rather than to "worthless".
    let confidence: [UInt8]

    var sampleCount: Int { width * height }

    @inline(__always)
    func depthMeters(at index: Int) -> Float {
        Float(depthMillimeters[index]) * 0.001
    }

    @inline(__always)
    func hasReturn(at index: Int) -> Bool {
        depthMillimeters[index] != 0
    }

    @inline(__always)
    func confidenceLevel(at index: Int) -> UInt8 {
        confidence.isEmpty ? 1 : confidence[index]
    }

    /// Reads the two sidecars for one frame, collapsing two very different
    /// facts into one `nil`.
    ///
    /// NIL MEANS EITHER OF TWO THINGS AND THIS SIGNATURE CANNOT TELL THEM
    /// APART:
    ///
    ///   * the frame never recorded depth at all (`CaptureFrame.depthPath` is
    ///     optional, and a frame captured without the laser legitimately has
    ///     none) - normal, and not a fault;
    ///   * the frame DID record depth and the file would not open - missing,
    ///     unreadable, or deleted after capture, which is lost measurement.
    ///
    /// A scan whose depth sidecars had all been lost therefore reads back
    /// through this function exactly like a scan taken on a phone with no
    /// laser in it. Any caller that counts, reports or alarms MUST use
    /// `loadOutcome(frame:settings:at:)` instead, which names all three
    /// outcomes; this wrapper is for callers that genuinely only want the
    /// frame or nothing.
    ///
    /// Throws only when a file DID open and is the wrong size, or when the
    /// capture recorded an impossible depth map size, because that is
    /// corruption and guessing at it would poison everything downstream.
    static func load(
        frame: CaptureFrame,
        settings: CaptureSettings,
        at ref: CaptureBundleRef
    ) throws -> PrePassDepthFrame? {
        switch try loadOutcome(frame: frame, settings: settings, at: ref) {
        case .loaded(let depthFrame):
            return depthFrame
        case .noDepthRecorded, .unreadable:
            return nil
        }
    }

    /// The same read, with the two `nil` cases kept apart. One implementation
    /// backs both, so the two views of a read cannot drift.
    ///
    /// - Throws: `NimbusError.malformedData` when the depth map size the
    ///   capture recorded is impossible, or when a file opened and is the
    ///   wrong length. A file that will not open is NOT a throw: it is
    ///   `.unreadable`, so a caller can count it and carry on rather than
    ///   losing every later frame to one bad sidecar.
    static func loadOutcome(
        frame: CaptureFrame,
        settings: CaptureSettings,
        at ref: CaptureBundleRef
    ) throws -> PrePassDepthLoad {
        guard let depthPath = frame.depthPath else { return .noDepthRecorded }
        let width = settings.depthWidth
        let height = settings.depthHeight
        guard width > 0, height > 0 else {
            throw NimbusError.malformedData(
                "the capture recorded a \(width)x\(height) depth map size"
            )
        }
        let sampleCount = width * height

        let depthURL = ref.url(forRelativePath: depthPath)
        guard let depthData = try? Data(contentsOf: depthURL, options: .mappedIfSafe) else {
            // A path WAS recorded and the file will not open. Named rather
            // than returned as a bare nil: this is the case the honesty
            // counters exist for.
            return .unreadable(path: depthPath)
        }
        guard depthData.count == sampleCount * 2 else {
            throw NimbusError.malformedData(
                "depth map \(depthURL.lastPathComponent) is \(depthData.count) bytes, "
                    + "expected \(sampleCount * 2)"
            )
        }

        var depth = [UInt16](repeating: 0, count: sampleCount)
        depth.withUnsafeMutableBytes { destination in
            depthData.copyBytes(to: destination)
        }
        // The file is little-endian by contract; on every Apple silicon target
        // that is already the host order, but say so rather than assume it.
        if UInt16(littleEndian: 0x0100) != 0x0100 {
            for i in 0..<sampleCount { depth[i] = UInt16(littleEndian: depth[i]) }
        }

        var confidence: [UInt8] = []
        if let confidencePath = frame.confidencePath {
            let confidenceURL = ref.url(forRelativePath: confidencePath)
            if let confidenceData = try? Data(contentsOf: confidenceURL, options: .mappedIfSafe),
               confidenceData.count == sampleCount {
                confidence = [UInt8](confidenceData)
            }
        }

        return .loaded(
            PrePassDepthFrame(
                width: width,
                height: height,
                depthMillimeters: depth,
                confidence: confidence
            )
        )
    }
}

/// What a depth-sidecar read actually found.
///
/// Three outcomes, because two of them used to be the same `nil` and the
/// difference between them is the difference between "this scan never had
/// laser data" and "this scan's laser data was lost". The first needs no
/// action; the second is the thing the QC card's `unreadable_depth` finding
/// and the census's depth-missing counts exist to say out loud.
enum PrePassDepthLoad {
    /// The sidecar opened and is the length the recorded map size implies.
    case loaded(PrePassDepthFrame)
    /// `CaptureFrame.depthPath` is nil: no depth was ever recorded for this
    /// frame. Normal.
    case noDepthRecorded
    /// A depth path WAS recorded and the file would not open. Lost
    /// measurement. The relative path travels with it so a caller can name
    /// the first casualty instead of only counting them.
    case unreadable(path: String)
}

/// Intrinsics resampled to the native depth resolution, plus the per-pixel
/// ray directions that every stage needs. Cached once per capture: the ray
/// table is the same for every frame, and recomputing 49k normalisations per
/// frame is pure waste.
struct PrePassDepthGeometry {
    let width: Int
    let height: Int
    let intrinsics: CameraIntrinsics
    /// Unit direction in CAMERA space (+X right, +Y down, +Z forward) for each
    /// native pixel, row-major.
    let rayDirections: [SIMD3<Float>]
    /// `1 / Zc` scaling: multiply by depth to get the camera-space point
    /// without a normalise. Row-major, same order.
    let rayPlaneCoordinates: [SIMD3<Float>]

    init(rgbIntrinsics: CameraIntrinsics, settings: CaptureSettings) {
        let width = Swift.max(settings.depthWidth, 1)
        let height = Swift.max(settings.depthHeight, 1)
        self.width = width
        self.height = height
        // ARKit's depth map is spatially aligned to the colour image, so the
        // depth intrinsics are the colour intrinsics scaled by the resolution
        // ratio. Nothing else changes: same principal point, same aspect.
        let scaled = rgbIntrinsics.scaled(toWidth: width, height: height)
        self.intrinsics = scaled

        var directions = [SIMD3<Float>](repeating: .zero, count: width * height)
        var planes = [SIMD3<Float>](repeating: .zero, count: width * height)
        let inverseFX = 1 / scaled.fx
        let inverseFY = 1 / scaled.fy
        for v in 0..<height {
            // Pixel centres: the COLMAP PINHOLE convention in
            // docs/DATA_FORMAT.md puts cx at width/2 - 0.5, so the centre of
            // pixel (0,0) is at (0.5, 0.5).
            let y = (Float(v) + 0.5 - scaled.cy) * inverseFY
            for u in 0..<width {
                let x = (Float(u) + 0.5 - scaled.cx) * inverseFX
                let plane = SIMD3<Float>(x, y, 1)
                let index = v * width + u
                planes[index] = plane
                directions[index] = simd_normalize(plane)
            }
        }
        self.rayDirections = directions
        self.rayPlaneCoordinates = planes
    }

    /// Camera-space point for a native sample at range `depthMeters`.
    /// `depthMeters` is the Z coordinate (a depth map, not a range image),
    /// which is what ARKit's `sceneDepth` actually stores.
    @inline(__always)
    func cameraPoint(index: Int, depthMeters: Float) -> SIMD3<Float> {
        rayPlaneCoordinates[index] * depthMeters
    }

    /// True range along the ray, for the physics prior (which grows with
    /// distance travelled, not with the Z coordinate).
    @inline(__always)
    func range(index: Int, depthMeters: Float) -> Float {
        simd_length(rayPlaneCoordinates[index]) * depthMeters
    }

    /// Projects a camera-space point onto the native grid. Returns nil behind
    /// the camera or outside the image.
    @inline(__always)
    func project(cameraPoint p: SIMD3<Float>) -> SIMD2<Float>? {
        guard p.z > 1e-4 else { return nil }
        let u = intrinsics.fx * (p.x / p.z) + intrinsics.cx
        let v = intrinsics.fy * (p.y / p.z) + intrinsics.cy
        guard u >= 0, v >= 0, u < Float(width), v < Float(height) else { return nil }
        return SIMD2<Float>(u, v)
    }
}

// MARK: - Image decode

/// A single-channel 8-bit image, row-major, no padding.
struct PrePassGrayImage {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    @inline(__always)
    func value(x: Int, y: Int) -> Float {
        Float(pixels[y * width + x])
    }

    /// Bilinear sample. Out-of-range coordinates clamp to the edge rather than
    /// throwing, because every caller here is already inside a bounds check
    /// and the clamp is the cheaper way to keep a patch window whole.
    func sampleBilinear(x: Float, y: Float) -> Float {
        let cx = Swift.min(Swift.max(x, 0), Float(width - 1))
        let cy = Swift.min(Swift.max(y, 0), Float(height - 1))
        let x0 = Int(cx), y0 = Int(cy)
        let x1 = Swift.min(x0 + 1, width - 1)
        let y1 = Swift.min(y0 + 1, height - 1)
        let fx = cx - Float(x0), fy = cy - Float(y0)
        let p00 = value(x: x0, y: y0), p10 = value(x: x1, y: y0)
        let p01 = value(x: x0, y: y1), p11 = value(x: x1, y: y1)
        let top = p00 + (p10 - p00) * fx
        let bottom = p01 + (p11 - p01) * fx
        return top + (bottom - top) * fy
    }
}

/// A three-channel 8-bit image, row-major, 4 bytes per pixel (RGBX).
struct PrePassColorImage {
    let width: Int
    let height: Int
    /// RGBA bytes, alpha ignored.
    var pixels: [UInt8]

    @inline(__always)
    func rgb(x: Int, y: Int) -> SIMD3<Float> {
        let i = (y * width + x) * 4
        return SIMD3<Float>(Float(pixels[i]), Float(pixels[i + 1]), Float(pixels[i + 2])) / 255
    }

    /// Fraction of pixels with any channel at or above `threshold`.
    ///
    /// A whole-frame summary, deliberately NOT the F5 saturation mask. That
    /// mask is per pixel and lives in `Sources/Smart`'s `SmartAuthorityMap`,
    /// which ramps each sample's authority down against
    /// `SmartLossSettings.saturationLuma`; the glass detector likewise has its
    /// own `brightLuma` test on the greyscale image and does not call this.
    /// Nothing calls this today. It is kept as the one cheap "how blown out
    /// was this frame" number, and anything that starts using it should say so
    /// through `PrePassCensus` rather than quietly.
    func saturatedFraction(threshold: UInt8 = 250) -> Float {
        guard width * height > 0 else { return 0 }
        var saturated = 0
        var i = 0
        while i < pixels.count {
            if pixels[i] >= threshold || pixels[i + 1] >= threshold || pixels[i + 2] >= threshold {
                saturated += 1
            }
            i += 4
        }
        return Float(saturated) / Float(width * height)
    }
}

/// Downsampled JPEG decode.
///
/// Uses ImageIO's thumbnail path (`kCGImageSourceThumbnailMaxPixelSize`) so a
/// 1920x1440 frame is decoded straight to the size actually wanted. Decoding
/// full size and then scaling costs roughly 8x the memory bandwidth and is the
/// single easiest way to make a pre-pass over 3000 frames take minutes.
enum PrePassImageLoader {

    /// Greyscale at exactly `width` x `height`.
    static func loadGray(url: URL, width: Int, height: Int) -> PrePassGrayImage? {
        guard width > 0, height > 0,
              let cgImage = decode(url: url, maxPixelSize: Swift.max(width, height))
        else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let drawn: Bool = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let base = buffer.baseAddress,
                  let context = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: width,
                      space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.none.rawValue
                  )
            else { return false }
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        return PrePassGrayImage(width: width, height: height, pixels: pixels)
    }

    /// RGB at exactly `width` x `height`, 4 bytes per pixel.
    static func loadColor(url: URL, width: Int, height: Int) -> PrePassColorImage? {
        guard width > 0, height > 0,
              let cgImage = decode(url: url, maxPixelSize: Swift.max(width, height))
        else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        let drawn: Bool = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let base = buffer.baseAddress,
                  let context = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: width * 4,
                      space: colorSpace,
                      bitmapInfo: bitmapInfo
                  )
            else { return false }
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        return PrePassColorImage(width: width, height: height, pixels: pixels)
    }

    private static func decode(url: URL, maxPixelSize: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            // Ask for a little more than needed so the final resample in the
            // CGContext is a downscale, never an upscale of a too-small
            // thumbnail.
            kCGImageSourceThumbnailMaxPixelSize: Swift.max(maxPixelSize * 2, 64)
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }
}

// MARK: - Image gradients

enum PrePassImageOps {
    /// Sobel gradient magnitude, normalised to roughly 0...1 for an 8-bit
    /// input. Border pixels get their nearest interior value rather than zero,
    /// so a strong edge running along the image border is not silently erased.
    static func sobelMagnitude(_ image: PrePassGrayImage) -> [Float] {
        let w = image.width, h = image.height
        var out = [Float](repeating: 0, count: w * h)
        guard w >= 3, h >= 3 else { return out }

        image.pixels.withUnsafeBufferPointer { p in
            for y in 1..<(h - 1) {
                let rowAbove = (y - 1) * w
                let row = y * w
                let rowBelow = (y + 1) * w
                for x in 1..<(w - 1) {
                    let a = Float(p[rowAbove + x - 1]), b = Float(p[rowAbove + x]), c = Float(p[rowAbove + x + 1])
                    let d = Float(p[row + x - 1]), f = Float(p[row + x + 1])
                    let g = Float(p[rowBelow + x - 1]), hh = Float(p[rowBelow + x]), i = Float(p[rowBelow + x + 1])
                    let gx = (c + 2 * f + i) - (a + 2 * d + g)
                    let gy = (g + 2 * hh + i) - (a + 2 * b + c)
                    // Divide by 4*255: the maximum |gx| for 8-bit input is 4*255.
                    out[row + x] = ((gx * gx + gy * gy).squareRoot()) / 1020
                }
            }
            // Replicate the border from its neighbour.
            for x in 0..<w {
                out[x] = out[Swift.min(w + x, w * h - 1)]
                out[(h - 1) * w + x] = out[(h - 2) * w + x]
            }
            for y in 0..<h {
                out[y * w] = out[y * w + 1]
                out[y * w + w - 1] = out[y * w + w - 2]
            }
        }
        return out
    }

    /// Shi-Tomasi corner response (the smaller eigenvalue of the structure
    /// tensor) over a `radius`-sized window. Used to pick the features the
    /// time-offset sweep tracks.
    static func shiTomasiResponse(_ image: PrePassGrayImage, radius: Int = 2) -> [Float] {
        let w = image.width, h = image.height
        var response = [Float](repeating: 0, count: w * h)
        guard w >= 2 * radius + 3, h >= 2 * radius + 3 else { return response }

        // First derivatives, central differences.
        var ix = [Float](repeating: 0, count: w * h)
        var iy = [Float](repeating: 0, count: w * h)
        image.pixels.withUnsafeBufferPointer { p in
            for y in 1..<(h - 1) {
                let row = y * w
                for x in 1..<(w - 1) {
                    ix[row + x] = (Float(p[row + x + 1]) - Float(p[row + x - 1])) * 0.5
                    iy[row + x] = (Float(p[row + w + x]) - Float(p[row - w + x])) * 0.5
                }
            }
        }

        for y in radius..<(h - radius) {
            for x in radius..<(w - radius) {
                var sxx: Float = 0, syy: Float = 0, sxy: Float = 0
                for dy in -radius...radius {
                    let row = (y + dy) * w
                    for dx in -radius...radius {
                        let gx = ix[row + x + dx]
                        let gy = iy[row + x + dx]
                        sxx += gx * gx
                        syy += gy * gy
                        sxy += gx * gy
                    }
                }
                // Smaller eigenvalue of [[sxx, sxy], [sxy, syy]].
                let trace = sxx + syy
                let determinant = sxx * syy - sxy * sxy
                let discriminant = Swift.max(trace * trace * 0.25 - determinant, 0)
                response[y * w + x] = trace * 0.5 - discriminant.squareRoot()
            }
        }
        return response
    }

    /// Zero-mean normalised cross-correlation of two square patches, in
    /// -1...1. Returns -1 (the worst possible score) when either patch is
    /// flat, so a featureless wall never wins a match by accident.
    ///
    /// Deliberately hand-rolled rather than routed through vDSP: the patch is
    /// 81 floats and the per-call setup of four vDSP reductions costs more
    /// than the arithmetic. (vDSP earns its keep in `PrePassTrustFieldBuilder`,
    /// where the vectors are millions of elements long.)
    static func zncc(
        _ a: PrePassGrayImage,
        centerA: SIMD2<Float>,
        _ b: PrePassGrayImage,
        centerB: SIMD2<Float>,
        radius: Int,
        scratchA: inout [Float],
        scratchB: inout [Float]
    ) -> Float {
        let side = 2 * radius + 1
        let count = side * side
        if scratchA.count != count { scratchA = [Float](repeating: 0, count: count) }
        if scratchB.count != count { scratchB = [Float](repeating: 0, count: count) }

        var index = 0
        var sumA: Float = 0, sumB: Float = 0
        for dy in -radius...radius {
            for dx in -radius...radius {
                let va = a.sampleBilinear(x: centerA.x + Float(dx), y: centerA.y + Float(dy))
                let vb = b.sampleBilinear(x: centerB.x + Float(dx), y: centerB.y + Float(dy))
                scratchA[index] = va
                scratchB[index] = vb
                sumA += va
                sumB += vb
                index += 1
            }
        }

        let inverseCount = 1 / Float(count)
        let meanA = sumA * inverseCount
        let meanB = sumB * inverseCount

        var dot: Float = 0, energyA: Float = 0, energyB: Float = 0
        for i in 0..<count {
            let da = scratchA[i] - meanA
            let db = scratchB[i] - meanB
            dot += da * db
            energyA += da * da
            energyB += db * db
        }

        let denominator = (energyA * energyB).squareRoot()
        guard denominator > 1e-3 else { return -1 }
        let value = dot / denominator
        return value.isFinite ? Swift.min(Swift.max(value, -1), 1) : -1
    }
}
