//
//  PrePassBinaryIO.swift
//  PrePass
//
//  WHERE THE PRE-PASS PUTS THINGS, AND THE EXACT BYTES IT PUTS THERE.
//
//  Every binary layout in this file is quoted from `docs/DATA_FORMAT.md`
//  section 7 in the comment above the writer that produces it. If the two ever
//  disagree, the document wins and this file is the bug.
//
//  Two rules that are not negotiable and are therefore enforced here rather
//  than left to each call site:
//
//   1. Little-endian, always. Both ends of this project are little-endian, so
//      no swapping happens - but the intent is written down (and the swap is
//      present, compiled out on every real target) so nobody has to
//      reverse-engineer it from a hex dump later.
//
//   2. A non-finite float never reaches a file. One diverged residual writing
//      a NaN into `trust_noise.bin` would be read back by the trainer as
//      truth and would poison a whole scan silently. `appendFloat` clamps.
//
//  The per-sample fields (`trust_noise.bin`, `confidence_recal.bin`) are
//  frame-major and indexed by `FrameID` DIRECTLY: block `i` of the file is the
//  frame whose `CaptureFrame.index == i`. That is what
//  `Sources/Smart/SmartCore.swift`'s `SmartSampleFieldReader` assumes
//  (`index * samplesPerFrame * 4`), so a frame index missing from the capture
//  gets a default-filled block rather than shifting every later frame by one.
//  Getting this wrong would misalign every sample in the scan by a whole frame
//  and would look like a plausible-but-wrong trust field, which is the worst
//  kind of bug.
//

import Foundation
import simd

// MARK: - Paths

/// Relative paths, from the scan folder root, of everything the pre-pass
/// writes. All derived from `BrandConfig.Folder.prePass` so the folder name
/// lives in exactly one place.
enum PrePassPaths {
    static let directory = BrandConfig.Folder.prePass

    static var result: String { "\(directory)/prepass_result.json" }
    /// The splat census: what every stage of the pre-pass counted about
    /// itself. A separate file from `prepass_result.json` because the result
    /// type lives in `Sources/Core`, which this module does not own, and
    /// because a report about the work must never be able to break the work.
    static var census: String { "\(directory)/census.json" }
    static var occupancy: String { "\(directory)/occupancy.bin" }
    // The four F6 fields and the F3 edge folder below are WRITTEN by
    // `Sources/Smart` (`TwoScaleTrustField`, `NativeDepthEdgeClassifier`),
    // which builds the identical strings from `BrandConfig.Folder.prePass`
    // itself instead of reading them from here. Verified identical, byte
    // layouts included, on 2026-09-06. They are declared here anyway because
    // this enum is the module's statement of where the pre-pass puts things,
    // and `PrePassPaths.missing(_:at:)` uses them after those stages to catch
    // the day the two spellings stop matching. Renaming one of these alone
    // changes nothing on disk: change the Smart writer in the same commit.
    static var trustBias: String { "\(directory)/trust_bias.bin" }
    static var trustNoise: String { "\(directory)/trust_noise.bin" }
    static var depthAffine: String { "\(directory)/depth_affine.bin" }
    static var confidenceRecalibrated: String { "\(directory)/confidence_recal.bin" }
    static var edgesDirectory: String { "\(directory)/edges" }
    static var initialSplats: String { "\(directory)/init_splats.ply" }
    static var initialSplatFlags: String { "\(directory)/init_splats.flags" }
    static var refinedModelDirectory: String { "\(directory)/sparse_refined" }
    static var refinedCameras: String { "\(refinedModelDirectory)/cameras.txt" }
    static var refinedImages: String { "\(refinedModelDirectory)/images.txt" }
    static var refinedPoints: String { "\(refinedModelDirectory)/points3D.txt" }

    /// `images/frame_20260903_141205_512.jpg` -> `frame_20260903_141205_512`.
    ///
    /// The timestamp stamp is the join key between an image and its sidecars,
    /// so the edge map for a frame has to be named from the image the frame
    /// actually references, never re-derived from the frame's own timestamp
    /// (which would round differently and produce a file nothing can find).
    static func stem(ofRelativePath path: String) -> String {
        let last = path.split(separator: "/").last.map(String.init) ?? path
        guard let dot = last.lastIndex(of: ".") else { return last }
        return String(last[last.startIndex..<dot])
    }

    /// `prepass/edges/frame_20260903_141205_512.edge8` for a given frame.
    static func edgeMap(for frame: CaptureFrame) -> String {
        "\(edgesDirectory)/\(stem(ofRelativePath: frame.imagePath)).edge8"
    }

    /// Which of `paths` are NOT on disk under `ref`.
    ///
    /// This exists because four of the names above (`trustBias`, `trustNoise`,
    /// `confidenceRecalibrated`, `edgesDirectory`) are produced by
    /// `Sources/Smart`, which builds the same strings itself from
    /// `BrandConfig.Folder.prePass` rather than reading them from here. The two
    /// spellings agree today. Nothing enforced that they keep agreeing, and a
    /// silent disagreement looks exactly like a stage that ran and produced
    /// nothing: the pre-pass reports success, the file the trainer opens is not
    /// there, and every depth sample falls back to "no opinion".
    ///
    /// So the constants get used for the one thing they can be used for from
    /// this side: after a stage says it wrote its files, check the files are
    /// where this enum says they are, and put it on the QC card when they are
    /// not. A drift that used to be silent becomes a sentence on the card the
    /// owner reads after every scan.
    static func missing(_ paths: [String], at ref: CaptureBundleRef) -> [String] {
        var absent: [String] = []
        for path in paths {
            let url = ref.url(forRelativePath: path)
            if !FileManager.default.fileExists(atPath: url.path) {
                absent.append(path)
            }
        }
        return absent
    }
}

// MARK: - Little-endian appenders

enum PrePassBinary {

    @inline(__always)
    static func append(_ value: UInt8, to data: inout Data) {
        data.append(value)
    }

    @inline(__always)
    static func append(_ value: UInt16, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    @inline(__always)
    static func append(_ value: UInt32, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    @inline(__always)
    static func append(_ value: UInt64, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    /// Non-finite values become `fallback`. See the file header: a NaN on disk
    /// is read back as truth by something that cannot tell.
    @inline(__always)
    static func appendFloat(_ value: Float, to data: inout Data, fallback: Float = 0) {
        let safe = value.isFinite ? value : fallback
        withUnsafeBytes(of: safe.bitPattern.littleEndian) { data.append(contentsOf: $0) }
    }

    /// Writes atomically, creating the parent directory. Atomic matters here:
    /// a pre-pass interrupted by the user backgrounding the app must leave
    /// either the previous file or the new one, never half of the new one.
    static func write(_ data: Data, to url: URL) throws {
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: url, options: .atomic)
        } catch {
            throw NimbusError.prePassFailed(
                "could not write \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }
}

// MARK: - Occupancy grid file

/// `prepass/occupancy.bin`: a flat run of 12-byte records sorted ascending by
/// Morton key.
///
/// | offset | type   | meaning                                   |
/// |--------|--------|-------------------------------------------|
/// | 0      | UInt64 | Morton key                                |
/// | 8      | UInt8  | state: 0 unknown, 1 empty, 2 surface      |
/// | 9      | UInt8  | reserved, 0                               |
/// | 10     | UInt16 | hit count, saturating                     |
///
/// Sorted so a lookup is a binary search and a merge is a linear scan. A cell
/// absent from the file is `.unknown`, and `.unknown` is never `.empty`.
enum PrePassOccupancyFile {
    static let recordSize = 12

    static func encode(keys: [UInt64], states: [UInt8], hits: [UInt16]) -> Data {
        precondition(keys.count == states.count && keys.count == hits.count)
        var data = Data(capacity: keys.count * recordSize)
        for i in 0..<keys.count {
            PrePassBinary.append(keys[i], to: &data)
            PrePassBinary.append(states[i], to: &data)
            PrePassBinary.append(UInt8(0), to: &data)
            PrePassBinary.append(hits[i], to: &data)
        }
        return data
    }

    /// Reads the file back into two parallel arrays, keys already sorted.
    /// Returns nil when the file length is not a whole number of records,
    /// which is corruption: the caller degrades to "no grid", never to a grid
    /// truncated at an arbitrary point.
    static func decode(_ data: Data) -> (keys: [UInt64], states: [UInt8])? {
        guard data.count % recordSize == 0 else { return nil }
        let count = data.count / recordSize
        guard count > 0 else { return ([], []) }
        var keys = [UInt64](repeating: 0, count: count)
        var states = [UInt8](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let offset = i * recordSize
                // `loadUnaligned` because a mapped file gives no alignment
                // guarantee and a 12-byte stride puts every other key on an
                // odd 8-byte boundary.
                let key = raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
                keys[i] = UInt64(littleEndian: key)
                states[i] = raw.loadUnaligned(fromByteOffset: offset + 8, as: UInt8.self)
            }
        }
        return (keys, states)
    }
}

// MARK: - Trust bias field file

/// `prepass/trust_bias.bin`: 24-byte records at 25-50 cm voxels.
///
/// | offset | type    | meaning                                 |
/// |--------|---------|-----------------------------------------|
/// | 0      | UInt64  | Morton key                              |
/// | 8      | Float32 | running mean SIGNED depth residual, m   |
/// | 12     | Float32 | variance of that residual               |
/// | 16     | UInt32  | sample count                            |
/// | 20     | UInt16  | number of DISTINCT capture times        |
/// | 22     | UInt16  | reserved                                |
///
/// The distinct-times count is the whole reason this file is not just a mean:
/// 400 samples from one pass past a wall is one observation repeated, and only
/// separate visits earn real trust.
///
/// NOT CALLED. `Sources/Smart` writes this file with its own inline
/// appenders (`SmartBinary`), which produce byte-identical records and
/// apply the same non-finite guard. Kept because the layout table above
/// is the one written down next to `docs/DATA_FORMAT.md`; the Smart
/// writer has no table. See INTEGRATION_REQUESTS.md.
enum PrePassTrustBiasFile {
    static let recordSize = 24

    static func encode(
        keys: [UInt64],
        mean: [Float],
        variance: [Float],
        sampleCount: [UInt32],
        distinctTimes: [UInt16]
    ) -> Data {
        var data = Data(capacity: keys.count * recordSize)
        for i in 0..<keys.count {
            PrePassBinary.append(keys[i], to: &data)
            PrePassBinary.appendFloat(mean[i], to: &data)
            PrePassBinary.appendFloat(variance[i], to: &data)
            PrePassBinary.append(sampleCount[i], to: &data)
            PrePassBinary.append(distinctTimes[i], to: &data)
            PrePassBinary.append(UInt16(0), to: &data)
        }
        return data
    }
}

// MARK: - Per-frame affine depth correction

/// `prepass/depth_affine.bin`: `(UInt32 frameIndex, Float32 scale,
/// Float32 shift)`, 12 bytes per record, one per frame that has a correction.
/// Tightly constrained around `(1, 0)`; a large correction means something
/// else is wrong and the QC card says so rather than quietly applying it.
///
/// NOT CALLED. `Sources/Smart` writes this file with its own inline
/// appenders (`SmartBinary`), which produce byte-identical records and
/// apply the same non-finite guard. Kept because the layout table above
/// is the one written down next to `docs/DATA_FORMAT.md`; the Smart
/// writer has no table. See INTEGRATION_REQUESTS.md.
enum PrePassDepthAffineFile {
    static let recordSize = 12

    static func encode(_ records: [(frame: FrameID, scale: Float, shift: Float)]) -> Data {
        var data = Data(capacity: records.count * recordSize)
        for record in records {
            PrePassBinary.append(record.frame, to: &data)
            PrePassBinary.appendFloat(record.scale, to: &data, fallback: 1)
            PrePassBinary.appendFloat(record.shift, to: &data)
        }
        return data
    }
}

// MARK: - Frame-major per-sample Float32 fields

/// `prepass/trust_noise.bin` and `prepass/confidence_recal.bin`: one Float32
/// per native depth sample, frame-major, block `i` == frame index `i`.
///
/// See the file header for why the block index is the frame index rather than
/// the frame's position in the array.
///
/// NOT CALLED. `Sources/Smart` writes this file with its own inline
/// appenders (`SmartBinary`), which produce byte-identical records and
/// apply the same non-finite guard. Kept because the layout table above
/// is the one written down next to `docs/DATA_FORMAT.md`; the Smart
/// writer has no table. See INTEGRATION_REQUESTS.md.
enum PrePassSampleFieldFile {

    /// - Parameters:
    ///   - blocks: value slices keyed by frame index. A frame absent from the
    ///     dictionary is written as a whole block of `defaultValue`.
    ///   - frameCount: how many blocks to write, i.e. `maxFrameIndex + 1`.
    static func encode(
        blocks: [FrameID: [Float]],
        frameCount: Int,
        samplesPerFrame: Int,
        defaultValue: Float
    ) -> Data {
        var data = Data(capacity: frameCount * samplesPerFrame * 4)
        let filler = [Float](repeating: defaultValue, count: samplesPerFrame)
        for i in 0..<frameCount {
            let slice = blocks[FrameID(i)] ?? filler
            if slice.count == samplesPerFrame {
                for value in slice { PrePassBinary.appendFloat(value, to: &data, fallback: defaultValue) }
            } else {
                // A wrong-length slice is a programming error upstream, not a
                // data condition; write the default block so the file stays
                // aligned and let the QC card carry the complaint.
                for value in filler { PrePassBinary.appendFloat(value, to: &data) }
            }
        }
        return data
    }
}

// MARK: - COLMAP text model

/// Writes `prepass/sparse_refined/{cameras,images,points3D}.txt`.
///
/// A whole second COLMAP model rather than an `images_refined.txt` sidecar,
/// because that way any COLMAP-reading tool can be pointed straight at the
/// folder and just work, with no awareness of this app at all
/// (`docs/DATA_FORMAT.md` section 4).
///
/// The quaternion order trap lives here and only here: in memory and in JSON a
/// `Quaternion` is `(x, y, z, w)`; COLMAP's `images.txt` wants `QW QX QY QZ`.
enum PrePassColmapWriter {

    static func camerasText(_ intrinsics: CameraIntrinsics) -> String {
        var out = "# Camera list with one line of data per camera:\n"
        out += "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        out += "# Number of cameras: 1\n"
        out += String(
            format: "1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n",
            intrinsics.width, intrinsics.height,
            Double(intrinsics.fx), Double(intrinsics.fy),
            Double(intrinsics.cx), Double(intrinsics.cy)
        )
        return out
    }

    /// `IMAGE_ID` is `CaptureFrame.index + 1` (COLMAP ids are 1-based), and
    /// the second line of every record is empty and MUST be present: this app
    /// does not triangulate, so there are no POINTS2D, but half the tools that
    /// claim to read COLMAP cannot parse the file if the line is missing.
    static func imagesText(frames: [CaptureFrame], poses: [String: Pose]) -> String {
        var out = "# Image list with two lines of data per image:\n"
        out += "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        out += "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        out += "# Number of images: \(frames.count), mean observations per image: 0\n"
        for frame in frames {
            let pose = poses[String(frame.index)] ?? frame.refinedPose ?? frame.rawPose
            let q = pose.rotation.normalized
            let t = pose.translation
            let name = PrePassPaths.stem(ofRelativePath: frame.imagePath)
                + fileExtension(of: frame.imagePath)
            // The name is concatenated rather than passed through `%@`: the
            // C-variadic bridge for a Swift String is one more thing that can
            // behave differently than it reads, for no benefit here.
            out += String(
                format: "%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 ",
                Int(frame.index) + 1,
                Double(q.w), Double(q.x), Double(q.y), Double(q.z),
                Double(t.x), Double(t.y), Double(t.z)
            )
            out += name
            out += "\n"
            // The mandatory empty POINTS2D line. Not decorative.
            out += "\n"
        }
        return out
    }

    /// `ERROR` is the expected METRIC error in metres from the physics prior,
    /// not COLMAP's reprojection error. That reinterpretation is documented in
    /// `docs/DATA_FORMAT.md` section 4; leaving the slot at 0 would be a lie,
    /// and it is the only per-point uncertainty the format has.
    static func pointsText(
        positions: [SIMD3<Float>],
        colors: [SIMD3<Float>],
        expectedErrorMeters: [Float]
    ) -> String {
        var out = "# 3D point list with one line of data per point:\n"
        out += "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        out += "# Number of points: \(positions.count), mean track length: 0\n"
        out.reserveCapacity(positions.count * 72)
        for i in 0..<positions.count {
            let p = positions[i]
            let c = i < colors.count ? colors[i] : SIMD3<Float>(repeating: 0.5)
            let e = i < expectedErrorMeters.count ? expectedErrorMeters[i] : 0
            let r = colorByte(c.x)
            let g = colorByte(c.y)
            let b = colorByte(c.z)
            out += String(
                format: "%d %.6f %.6f %.6f %d %d %d %.6f\n",
                i + 1,
                Double(p.x), Double(p.y), Double(p.z),
                Int(r), Int(g), Int(b),
                Double(e.isFinite ? e : 0)
            )
        }
        return out
    }

    /// One colour channel, 0...1, as the byte `points3D.txt` wants.
    ///
    /// The clamp this replaced was written the wrong way round and was a
    /// trapping conversion waiting for a bad colour:
    ///
    ///     UInt8(Swift.min(Swift.max(c.x * 255, 0), 255))
    ///
    /// Swift's `min` and `max` are `y < x ? y : x` and `y >= x ? y : x`, so
    /// when the comparison is false they hand back their FIRST argument. NaN
    /// makes every comparison false, so with the value written first the NaN
    /// is what survives the clamp and lands on `UInt8(...)`, which kills the
    /// process. With the literal written first the clamp absorbs it instead.
    ///
    /// A channel we cannot trust becomes mid grey, the same neutral the splat
    /// seeder already substitutes when there is no image to sample. Dropping
    /// the point is not an option here: `points3D.txt` has to stay in the same
    /// order, and with the same count, as the PLY it was written beside.
    @inline(__always)
    private static func colorByte(_ value: Float) -> UInt8 {
        guard value.isFinite else { return 128 }
        return UInt8(Swift.max(0, Swift.min(255, value * 255)))
    }

    private static func fileExtension(of path: String) -> String {
        let last = path.split(separator: "/").last.map(String.init) ?? path
        guard let dot = last.lastIndex(of: ".") else { return "" }
        return String(last[dot...])
    }

    static func write(text: String, to url: URL) throws {
        guard let data = text.data(using: .utf8) else {
            throw NimbusError.prePassFailed("could not encode \(url.lastPathComponent) as UTF-8")
        }
        try PrePassBinary.write(data, to: url)
    }
}
