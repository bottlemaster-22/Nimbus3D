//
//  ObservedDirectionField.swift
//  Viewer
//
//  THE HONESTY MASK'S DATA.
//
//  `docs/DATA_FORMAT.md` section 8 says what this file is FOR:
//
//      model/observed_directions.bin is the honesty mask's backing store: a
//      coarse per-voxel bitmask of which directions each part of the scene was
//      actually looked at from. The viewer hatches any pixel whose viewing ray
//      falls in an unobserved direction.
//
//  It does not fix the bytes, because the Viewer is the only reader and the
//  only writer. This file fixes them, and the layout below is normative for
//  this project. The GPU reads the record array directly, so the record is
//  exactly the `ulong2` that `viewer_composite_fragment` binary-searches.
//
//  ---------------------------------------------------------------------------
//  model/observed_directions.bin
//  ---------------------------------------------------------------------------
//
//  Header, 32 bytes, little-endian:
//
//      offset  type      meaning
//      0       char[8]   magic, "NB3DOBS1"
//      8       UInt32    format version, currently 1
//      12      UInt32    record count
//      16      Float32   voxel size, metres
//      20      Float32   grid origin x   (world space, metres)
//      24      Float32   grid origin y
//      28      Float32   grid origin z
//
//  Then `record count` records of 16 bytes, SORTED ASCENDING BY KEY:
//
//      offset  type      meaning
//      0       UInt64    Morton key of the voxel (3 x 21 bits, ViewerMorton)
//      8       UInt64    direction bitmask, one bit per octahedral bin
//
//  The sort is not a nicety: the fragment shader binary-searches this array
//  once per pixel, and an unsorted array would silently return "unobserved"
//  for cells that are present, which would hatch a correctly captured wall.
//
//  A voxel that does not appear has an implicit mask of zero, i.e. "never
//  observed from anywhere", which is the honest answer for space nobody looked
//  at and the one that makes the mask hatch rather than quietly pass.
//
//  ---------------------------------------------------------------------------
//  Direction bins
//  ---------------------------------------------------------------------------
//
//  64 octahedral bins over the full sphere, 8 x 8. `bin(for:)` below and
//  `viewer_direction_bin` in SplatRenderShaders.metal are the same function
//  written twice; there is no shared header in a single XcodeGen target that
//  both the Swift and the Metal compiler can see, so they are kept in step by
//  hand and by `ObservedDirectionField.selfCheck()`.
//
//  The direction stored is FROM THE SURFACE TOWARDS THE CAMERA - the same
//  sense `CaptureCoverageField.directionBucket` uses for the live coverage
//  HUD, so "the user walked around it" means the same thing in both places
//  even though the two use different bin counts (32 there, 64 here: the HUD
//  wants coarse buckets that map onto "walk around it a bit more", the honesty
//  mask wants fine ones because it is deciding per pixel).
//

import Foundation
import simd

/// A coarse per-voxel record of which directions the scene was looked at from.
///
/// Value type, `Sendable`, so it can be built on a background task and handed
/// to the main-actor renderer without a lock.
struct ObservedDirectionField: Sendable {

    /// 8 x 8 octahedral bins. Fixed: the mask is a `UInt64`.
    static let directionBinCount = 64

    /// Magic, then version. Both are checked on read; a version this build
    /// does not know is refused, not guessed at.
    static let magic: [UInt8] = Array("NB3DOBS1".utf8)
    static let formatVersion: UInt32 = 1
    static let headerByteCount = 32
    static let recordByteCount = 16

    /// World position of the minimum corner of cell (0, 0, 0).
    let origin: SIMD3<Float>
    let voxelSizeMeters: Float
    /// Morton keys, ascending. Parallel to `masks`.
    let keys: [UInt64]
    /// One direction bitmask per key.
    let masks: [UInt64]

    var cellCount: Int { keys.count }

    init(
        origin: SIMD3<Float>,
        voxelSizeMeters: Float,
        keys: [UInt64],
        masks: [UInt64]
    ) {
        precondition(keys.count == masks.count, "keys and masks must be parallel")
        self.origin = origin
        self.voxelSizeMeters = Swift.max(voxelSizeMeters, 0.01)
        self.keys = keys
        self.masks = masks
    }

    // MARK: - Direction bins

    /// Octahedral bin index for a unit-ish direction, 0..<64.
    ///
    /// MUST stay identical to `viewer_direction_bin` in
    /// `SplatRenderShaders.metal`. If you edit one, edit the other, then run
    /// `selfCheck()`.
    @inline(__always)
    static func bin(for direction: SIMD3<Float>) -> Int {
        let d = ViewerMath.safeNormalize(direction)
        let denom = Swift.max(abs(d.x) + abs(d.y) + abs(d.z), 1e-8)
        var p = SIMD2<Float>(d.x, d.y) / denom
        if d.z < 0 {
            let sx: Float = p.x >= 0 ? 1 : -1
            let sy: Float = p.y >= 0 ? 1 : -1
            p = SIMD2<Float>((1 - abs(p.y)) * sx, (1 - abs(p.x)) * sy)
        }
        let ux = ViewerMath.clamp(p.x * 0.5 + 0.5, 0, 0.999_999)
        let uy = ViewerMath.clamp(p.y * 0.5 + 0.5, 0, 0.999_999)
        let bx = Swift.min(Int(ux * 8), 7)
        let by = Swift.min(Int(uy * 8), 7)
        return by * 8 + bx
    }

    // MARK: - Cells

    /// Grid cell for a world point, or nil if it falls outside the encodable
    /// range (which can only happen for a point below the origin or absurdly
    /// far from it - both mean "not part of this scan").
    @inline(__always)
    func cell(forWorld point: SIMD3<Float>) -> SIMD3<UInt32>? {
        let rel = (point - origin) / voxelSizeMeters
        guard rel.x >= 0, rel.y >= 0, rel.z >= 0,
              rel.x.isFinite, rel.y.isFinite, rel.z.isFinite
        else { return nil }
        let maxCoord = Float(ViewerMorton.maxCoordinate)
        guard rel.x < maxCoord, rel.y < maxCoord, rel.z < maxCoord else { return nil }
        return SIMD3<UInt32>(UInt32(rel.x), UInt32(rel.y), UInt32(rel.z))
    }

    /// The direction mask recorded at a world point. Zero means "no record",
    /// which is the same as "never observed".
    func mask(atWorld point: SIMD3<Float>) -> UInt64 {
        guard let cell = cell(forWorld: point) else { return 0 }
        return mask(forKey: ViewerMorton.key(cell))
    }

    /// Binary search, matching `viewer_lookup_direction_mask` on the GPU.
    func mask(forKey key: UInt64) -> UInt64 {
        var lo = 0
        var hi = keys.count
        while lo < hi {
            let mid = lo + (hi - lo) / 2
            let k = keys[mid]
            if k < key {
                lo = mid + 1
            } else if k > key {
                hi = mid
            } else {
                return masks[mid]
            }
        }
        return 0
    }

    /// Was the point at `world` ever looked at from roughly `viewDirection`
    /// (a direction pointing FROM the surface TOWARDS a camera)?
    func wasObserved(world: SIMD3<Float>, fromDirection viewDirection: SIMD3<Float>) -> Bool {
        let m = mask(atWorld: world)
        guard m != 0 else { return false }
        return (m >> UInt64(Self.bin(for: viewDirection))) & 1 == 1
    }

    /// How many distinct directions a point was seen from, 0...64. The
    /// heatmap's "sparse directions" term.
    func observedDirectionCount(atWorld point: SIMD3<Float>) -> Int {
        mask(atWorld: point).nonzeroBitCount
    }

    // MARK: - Reading

    static func read(from url: URL) throws -> ObservedDirectionField {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw ViewerError.fileMissing(url.lastPathComponent)
        }
        return try decode(data, name: url.lastPathComponent)
    }

    static func decode(_ data: Data, name: String) throws -> ObservedDirectionField {
        guard data.count >= headerByteCount else {
            throw ViewerError.malformedFile("\(name) is too short to be an observation record")
        }
        var reader = ViewerByteReader(data)
        let magicBytes = try reader.readBytes(8)
        guard Array(magicBytes) == magic else {
            throw ViewerError.malformedFile("\(name) is not an observation record")
        }
        let version = try reader.readUInt32()
        guard version == formatVersion else {
            // "A reader that sees a version it does not know must refuse and
            // say so, not guess" - docs/DATA_FORMAT.md section 9.
            throw ViewerError.malformedFile(
                "\(name) is version \(version); this build understands version \(formatVersion)"
            )
        }
        let count = Int(try reader.readUInt32())
        let voxel = try reader.readFloat()
        let ox = try reader.readFloat()
        let oy = try reader.readFloat()
        let oz = try reader.readFloat()

        guard voxel.isFinite, voxel > 0 else {
            throw ViewerError.malformedFile("\(name) has a nonsense voxel size (\(voxel))")
        }
        guard data.count >= headerByteCount + count * recordByteCount else {
            throw ViewerError.malformedFile(
                "\(name) claims \(count) cells but is only \(data.count) bytes"
            )
        }

        var keys = [UInt64]()
        var masks = [UInt64]()
        keys.reserveCapacity(count)
        masks.reserveCapacity(count)
        var previousKey: UInt64 = 0
        var sorted = true
        for i in 0..<count {
            let key = try reader.readUInt64()
            let mask = try reader.readUInt64()
            if i > 0 && key < previousKey { sorted = false }
            previousKey = key
            keys.append(key)
            masks.append(mask)
        }
        guard sorted else {
            // Refusing beats hatching a correctly captured wall because a
            // binary search walked off an unsorted array.
            throw ViewerError.malformedFile("\(name) is not sorted by cell key")
        }

        return ObservedDirectionField(
            origin: SIMD3<Float>(ox, oy, oz),
            voxelSizeMeters: voxel,
            keys: keys,
            masks: masks
        )
    }

    // MARK: - Writing

    func encoded() -> Data {
        var data = Data()
        data.reserveCapacity(Self.headerByteCount + keys.count * Self.recordByteCount)
        data.append(contentsOf: Self.magic)
        data.appendLittle(Self.formatVersion)
        data.appendLittle(UInt32(keys.count))
        data.appendLittle(voxelSizeMeters)
        data.appendLittle(origin.x)
        data.appendLittle(origin.y)
        data.appendLittle(origin.z)
        for i in 0..<keys.count {
            data.appendLittle(keys[i])
            data.appendLittle(masks[i])
        }
        return data
    }

    func write(to url: URL) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try encoded().write(to: url, options: [.atomic])
    }

    // MARK: - Self check

    /// Cheap invariants worth asserting once in DEBUG: the encoding round
    /// trips, the bins are distinct and total 64, and the Morton codec is its
    /// own inverse. Returns nil when everything holds.
    static func selfCheck() -> String? {
        var problems: [String] = []

        // Morton round trip.
        for coordinate: UInt32 in [0, 1, 2, 1023, 0x1F_FFFF] {
            let cell = SIMD3<UInt32>(coordinate, coordinate / 2, coordinate / 3)
            let round = ViewerMorton.cell(ViewerMorton.key(cell))
            if round != cell {
                problems.append("Morton round trip failed for \(cell) -> \(round)")
            }
        }

        // Every one of the 64 bins is reachable, and no direction lands
        // outside 0..<64.
        var seen = Set<Int>()
        let steps = 40
        for i in 0..<steps {
            for j in 0..<steps {
                let theta = Float(i) / Float(steps - 1) * .pi
                let phi = Float(j) / Float(steps - 1) * 2 * .pi
                let d = SIMD3<Float>(
                    sin(theta) * cos(phi),
                    cos(theta),
                    sin(theta) * sin(phi)
                )
                let b = bin(for: d)
                if b < 0 || b >= directionBinCount {
                    problems.append("direction bin \(b) is out of range for \(d)")
                }
                seen.insert(b)
            }
        }
        if seen.count != directionBinCount {
            problems.append(
                "only \(seen.count) of \(directionBinCount) direction bins are reachable"
            )
        }

        // Encode / decode round trip.
        let field = ObservedDirectionField(
            origin: SIMD3<Float>(-1, -2, -3),
            voxelSizeMeters: 0.25,
            keys: [1, 5, 900],
            masks: [0b1, 0xFFFF_FFFF_FFFF_FFFF, 0b1010]
        )
        do {
            let round = try decode(field.encoded(), name: "selfcheck")
            if round.keys != field.keys || round.masks != field.masks
                || round.voxelSizeMeters != field.voxelSizeMeters
                || round.origin != field.origin {
                problems.append("observation record did not survive a round trip")
            }
        } catch {
            problems.append("observation record failed to decode: \(error)")
        }

        return problems.isEmpty ? nil : problems.joined(separator: "; ")
    }
}

// MARK: - Building the field

/// Builds an `ObservedDirectionField` from a capture bundle.
///
/// This is the "per-voxel direction bitmask built from the coverage grid" the
/// review UX needs. It is built here, in the Viewer, rather than taken on
/// faith from the trainer, for one blunt reason: the trainer may not have
/// written one (an imported model, a Booster result from an older build, a
/// scan that has only ever been pre-passed), and a review screen that silently
/// stops telling the user which parts of their scan are invented is worse than
/// no review screen at all.
///
/// The evidence is the same evidence the live coverage HUD used: every native
/// LiDAR return is a point on a real surface that was really looked at from
/// the camera that returned it. So for each frame, a subsampled grid of native
/// depth samples is back-projected to world space and the direction from that
/// point back to the camera sets one bit in that point's voxel.
///
/// What this deliberately does NOT do is mark free space as observed. Only
/// surface returns set bits. Empty air was seen *through*, not seen.
enum ObservedDirectionBuilder {

    struct Options: Sendable {
        /// Coarse on purpose. 25 cm matches the trust field's coarse scale and
        /// keeps a whole-house field in a few hundred kilobytes.
        var voxelSizeMeters: Float = 0.25
        /// Take every Nth native depth sample in each axis. 4 turns 256x192
        /// into 64x48 = 3072 samples per frame, which is plenty for a 25 cm
        /// grid and roughly 16x cheaper than every sample.
        var sampleStride: Int = 4
        /// Frames beyond this are subsampled in time as well; a 20 minute
        /// house scan does not need every frame to answer "was this looked at
        /// from over there".
        var maxFrames: Int = 900
        /// Depth beyond this is not a LiDAR return worth trusting (F2/F5: the
        /// near regime is 0 to ~4.5 m and beyond ~5 m "no return" means
        /// unknown, never observed).
        var maxDepthMeters: Float = 5.0
        /// Below this, the sample is inside the sensor's blind zone.
        var minDepthMeters: Float = 0.15
        /// Skip frames the capture QC scored below this. A smeared frame did
        /// look at the surface, but claiming it as a clean observation is the
        /// kind of small lie this whole feature exists to prevent.
        var minFrameQCWeight: Float = 0.15

        init() {}
    }

    /// Result of a build, including the honest count of what was skipped.
    struct BuildReport: Sendable {
        var field: ObservedDirectionField
        var framesUsed: Int
        var framesSkippedMissingDepth: Int
        var framesSkippedLowQuality: Int
        var samplesUsed: Int

        /// One plain sentence for the review screen.
        var summary: String {
            var parts = ["Built from \(framesUsed) frames"]
            if framesSkippedMissingDepth > 0 {
                parts.append("\(framesSkippedMissingDepth) had no depth recorded")
            }
            if framesSkippedLowQuality > 0 {
                parts.append("\(framesSkippedLowQuality) were too blurred to count")
            }
            return parts.joined(separator: ", ") + "."
        }
    }

    /// Builds the field. Pure computation plus file reads; call it off the
    /// main actor.
    ///
    /// - Parameter isCancelled: polled every frame so the review screen can
    ///   drop the work when the user leaves.
    static func build(
        bundle: CaptureBundle,
        prePass: PrePassResult?,
        paths: ViewerScanPaths,
        options: Options = Options(),
        isCancelled: () -> Bool = { false }
    ) throws -> BuildReport {

        let depthWidth = bundle.settings.depthWidth
        let depthHeight = bundle.settings.depthHeight
        guard depthWidth > 0, depthHeight > 0 else {
            throw ViewerError.malformedFile(
                "this scan records a \(depthWidth)x\(depthHeight) depth map, which cannot be read"
            )
        }

        // The intrinsics in the bundle describe the RGB frame. The depth map
        // is the same camera at a lower resolution, so the one correct
        // rescaling is Core's own.
        let depthIntrinsics = bundle.intrinsics.scaled(
            toWidth: depthWidth,
            height: depthHeight
        )

        // Choose the frames to use.
        let allFrames = bundle.frames
        let frameStride = Swift.max(1, allFrames.count / Swift.max(1, options.maxFrames))
        var chosen: [CaptureFrame] = []
        chosen.reserveCapacity(allFrames.count / frameStride + 1)
        var index = 0
        while index < allFrames.count {
            chosen.append(allFrames[index])
            index += frameStride
        }

        // Grid origin: the scene bounds if the capture recorded them, else the
        // extent of the camera track padded by the LiDAR range, so no surface
        // the sensor could have reached falls below the origin.
        let origin = gridOrigin(bundle: bundle, prePass: prePass, options: options)

        var cells: [UInt64: UInt64] = [:]
        cells.reserveCapacity(4096)

        var framesUsed = 0
        var skippedDepth = 0
        var skippedQuality = 0
        var samplesUsed = 0

        let stride = Swift.max(1, options.sampleStride)
        let maxCoordinate = Float(ViewerMorton.maxCoordinate)

        for frame in chosen {
            if isCancelled() { break }

            guard frame.qc.weight >= options.minFrameQCWeight else {
                skippedQuality += 1
                continue
            }
            guard let depthPath = frame.depthPath else {
                skippedDepth += 1
                continue
            }
            let depthURL = paths.url(depthPath)
            guard let depth = try? Data(contentsOf: depthURL, options: [.mappedIfSafe]),
                  depth.count >= depthWidth * depthHeight * 2
            else {
                skippedDepth += 1
                continue
            }

            // The refined pose is the one to use when the pre-pass has run:
            // an unrefined pose puts the observation in the wrong voxel, which
            // is exactly the error this field exists to detect.
            let pose = prePass?.refinedPose(for: frame.index)
                ?? frame.refinedPose
                ?? frame.rawPose

            let cameraCenter = pose.center.simd
            // Camera -> world rotation. `Pose.rotation` is world -> camera.
            let cameraToWorld = pose.rotation.simd.inverse

            framesUsed += 1

            depth.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                for y in Swift.stride(from: 0, to: depthHeight, by: stride) {
                    for x in Swift.stride(from: 0, to: depthWidth, by: stride) {
                        let byteOffset = (y * depthWidth + x) * 2
                        let millimetres = UInt16(
                            littleEndian: raw.loadUnaligned(
                                fromByteOffset: byteOffset,
                                as: UInt16.self
                            )
                        )
                        guard millimetres != 0 else { continue }
                        let z = Float(millimetres) / 1000
                        guard z >= options.minDepthMeters, z <= options.maxDepthMeters
                        else { continue }

                        // Back-project. Camera frame is +X right, +Y down,
                        // +Z forward, so this is the plain pinhole model with
                        // no sign flips.
                        let camera = SIMD3<Float>(
                            (Float(x) - depthIntrinsics.cx) / depthIntrinsics.fx * z,
                            (Float(y) - depthIntrinsics.cy) / depthIntrinsics.fy * z,
                            z
                        )
                        let world = cameraToWorld.act(camera) + cameraCenter

                        let rel = (world - origin) / options.voxelSizeMeters
                        guard rel.x >= 0, rel.y >= 0, rel.z >= 0,
                              rel.x < maxCoordinate,
                              rel.y < maxCoordinate,
                              rel.z < maxCoordinate
                        else { continue }

                        let key = ViewerMorton.key(
                            SIMD3<UInt32>(UInt32(rel.x), UInt32(rel.y), UInt32(rel.z))
                        )
                        // From the surface towards the camera.
                        let bin = ObservedDirectionField.bin(for: cameraCenter - world)
                        cells[key, default: 0] |= (1 << UInt64(bin))
                        samplesUsed += 1
                    }
                }
            }
        }

        let sortedKeys = cells.keys.sorted()
        let masks = sortedKeys.map { cells[$0] ?? 0 }

        return BuildReport(
            field: ObservedDirectionField(
                origin: origin,
                voxelSizeMeters: options.voxelSizeMeters,
                keys: sortedKeys,
                masks: masks
            ),
            framesUsed: framesUsed,
            framesSkippedMissingDepth: skippedDepth,
            framesSkippedLowQuality: skippedQuality,
            samplesUsed: samplesUsed
        )
    }

    /// The minimum corner of the grid. Morton keys are unsigned, so every
    /// point of interest has to sit above the origin; the padding is the LiDAR
    /// range, because a surface can be that far from any camera position.
    private static func gridOrigin(
        bundle: CaptureBundle,
        prePass: PrePassResult?,
        options: Options
    ) -> SIMD3<Float> {
        let pad = Swift.max(options.maxDepthMeters, bundle.settings.lidarMaxRangeMeters) + 1

        if let bounds = bundle.sceneBounds {
            return bounds.min.simd - SIMD3<Float>(repeating: pad)
        }

        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var any = false
        for frame in bundle.frames {
            let pose = prePass?.refinedPose(for: frame.index) ?? frame.rawPose
            let c = pose.center.simd
            guard c.x.isFinite, c.y.isFinite, c.z.isFinite else { continue }
            lo = simd_min(lo, c)
            any = true
        }
        guard any else { return SIMD3<Float>(repeating: -pad) }
        return lo - SIMD3<Float>(repeating: pad)
    }
}
