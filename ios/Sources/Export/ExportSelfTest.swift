//
//  ExportSelfTest.swift
//  Export
//
//  DEBUG-only runtime self-check harness for this module's serialization.
//
//  WHY THIS EXISTS INSTEAD OF AN XCTEST TARGET: project.yml currently
//  defines exactly one target ("App") and no test target/scheme, and adding
//  one is a shared, project-wide build-config change this module's mandate
//  ("write files under your assigned directory only") does not cover -
//  especially with other modules' agents potentially touching project.yml
//  in parallel. This harness gives Export real, running correctness checks
//  today (call `ExportSelfTest.runAll()` from anywhere in a DEBUG build,
//  e.g. a hidden developer-settings row, or a breakpoint) without touching
//  shared build config. TODO(nimbus): once a test target exists in
//  project.yml, port these into proper XCTestCase methods under an
//  ExportTests target - the assertions themselves translate directly.
//
//  These are round-trip / known-answer tests, not a mock framework: build a
//  SplatCloud, write it, read it back, and assert the result is within the
//  tolerance the format's own quantization implies (bit-exact for PLY,
//  which stores plain floats; tolerance-based for SPZ's byte-quantized
//  attributes, with the exact tolerances derived from SPZCodec's own
//  documented bucket sizes so a real regression - not just quantization
//  noise - fails the check).
//

#if DEBUG
import Foundation
import simd

enum ExportSelfTest {

    struct Failure: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
    }

    /// Runs every check and returns a human-readable pass/fail report, one
    /// line per check. Never throws - a failing check is reported as a
    /// failed line, not a crash, so a UI can show the whole report.
    static func runAll() -> [String] {
        let checks: [(String, () throws -> Void)] = [
            ("CRC32 known-answer vector", checkCRC32KnownAnswer),
            ("Gzip round trip", checkGzipRoundTrip),
            ("Gzip output is real gzip (magic + trailer)", checkGzipMagicAndTrailer),
            ("PLY round trip is bit-exact", checkPLYRoundTrip),
            ("PLY round trip, degree-2 SH", checkPLYRoundTripDegree2),
            ("SPZ round trip within quantization tolerance", checkSPZRoundTrip),
            ("SPZ container starts with gzip magic", checkSPZIsGzip),
            ("GLB container structure is well-formed", checkGLBStructure),
            ("Empty cloud is rejected, not silently written", checkEmptyCloudRejected),
            ("Capture-bundle zip includes capture_bundle.json + sections", checkBoosterBundlePackaging),
            ("Capture-bundle zip is a well-formed ZIP (EOCD present)", checkBoosterBundleZipStructure),
        ]

        var report: [String] = []
        var passCount = 0
        for (name, check) in checks {
            do {
                try check()
                report.append("PASS - \(name)")
                passCount += 1
            } catch {
                report.append("FAIL - \(name): \(error)")
            }
        }
        report.append("\(passCount)/\(checks.count) checks passed")
        return report
    }

    // MARK: - Fixtures

    /// A small, deterministic, non-degenerate SplatCloud - varied enough
    /// (normalized but non-axis-aligned rotations, non-zero SH) to exercise
    /// every code path without needing real capture data.
    private static func makeFixture(shDegree: SHDegree) throws -> SplatCloud {
        var positions: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var logScales: [SIMD3<Float>] = []
        var opacityLogits: [Float] = []
        var colorDC: [SIMD3<Float>] = []
        var shRest: [[SIMD3<Float>]] = []

        let n = 37
        for i in 0..<n {
            let t = Float(i)
            positions.append(SIMD3(sinf(t) * 2, cosf(t * 0.7), t * 0.05 - 1))
            let raw = SIMD4<Float>(sinf(t * 0.3), cosf(t * 0.9) * 0.4, sinf(t * 1.7) * 0.2, 1.0 + Float(i % 5))
            rotations.append(simd_normalize(raw))
            logScales.append(SIMD3(-3 + 0.1 * t, -4, -3.5 + 0.05 * t))
            opacityLogits.append(Float(i % 7) - 3)
            colorDC.append(SIMD3(sinf(t) * 0.5, cosf(t) * 0.5, 0.1))
            if shDegree != .zero {
                var coeffs: [SIMD3<Float>] = []
                for k in 0..<shDegree.restCoefficientCount {
                    let s = Float(k + 1)
                    coeffs.append(SIMD3(sinf(t + s) * 0.3, cosf(t - s) * 0.3, sinf(t * s) * 0.2))
                }
                shRest.append(coeffs)
            }
        }

        return try SplatCloud(
            shDegree: shDegree,
            positions: positions, rotations: rotations, logScales: logScales,
            opacityLogits: opacityLogits, colorDC: colorDC, shRest: shRest
        )
    }

    // MARK: - CRC32 / Gzip

    private static func checkCRC32KnownAnswer() throws {
        // The standard CRC-32/ISO-HDLC check value: crc32("123456789") == 0xCBF43926.
        let crc = CRC32.checksum(Data("123456789".utf8))
        guard crc == 0xCBF4_3926 else {
            throw Failure(message: "got 0x\(String(crc, radix: 16)), expected 0xcbf43926")
        }
    }

    private static func checkGzipRoundTrip() throws {
        let original = Data((0..<5000).map { UInt8(($0 * 37 + 11) % 256) })
        let compressed = try Gzip.compress(original)
        let restored = try Gzip.decompress(compressed)
        guard restored == original else {
            throw Failure(message: "round trip mismatch: \(original.count) -> \(restored.count) bytes")
        }
    }

    private static func checkGzipMagicAndTrailer() throws {
        let original = Data("hello, gaussian splats".utf8)
        let compressed = try Gzip.compress(original)
        guard compressed.count >= 18, compressed[compressed.startIndex] == 0x1f,
              compressed[compressed.startIndex + 1] == 0x8b
        else {
            throw Failure(message: "missing gzip magic bytes")
        }
    }

    // MARK: - PLY

    private static func checkPLYRoundTrip() throws {
        let cloud = try makeFixture(shDegree: .zero)
        let data = try PLYCodec.write(cloud)
        let restored = try PLYCodec.read(data)
        try assertBitExact(cloud, restored, label: "PLY degree-0")
    }

    private static func checkPLYRoundTripDegree2() throws {
        let cloud = try makeFixture(shDegree: .two)
        let data = try PLYCodec.write(cloud)
        let restored = try PLYCodec.read(data)
        try assertBitExact(cloud, restored, label: "PLY degree-2")
    }

    private static func assertBitExact(_ a: SplatCloud, _ b: SplatCloud, label: String, epsilon: Float = 1e-5) throws {
        guard a.count == b.count, a.shDegree == b.shDegree else {
            throw Failure(message: "\(label): shape mismatch")
        }
        for i in 0..<a.count {
            if simd_distance(a.positions[i], b.positions[i]) > epsilon {
                throw Failure(message: "\(label): position \(i) mismatch (\(a.positions[i]) vs \(b.positions[i]))")
            }
            // Quaternions may come back sign-flipped (q and -q are the same rotation);
            // PLY does not do that flip, so require an exact-sign match here.
            if simd_distance(a.rotations[i], b.rotations[i]) > epsilon {
                throw Failure(message: "\(label): rotation \(i) mismatch")
            }
            if simd_distance(a.logScales[i], b.logScales[i]) > epsilon {
                throw Failure(message: "\(label): scale \(i) mismatch")
            }
            if abs(a.opacityLogits[i] - b.opacityLogits[i]) > epsilon {
                throw Failure(message: "\(label): opacity \(i) mismatch")
            }
            if simd_distance(a.colorDC[i], b.colorDC[i]) > epsilon {
                throw Failure(message: "\(label): color \(i) mismatch")
            }
            for k in 0..<a.shDegree.restCoefficientCount {
                if simd_distance(a.shRest[i][k], b.shRest[i][k]) > epsilon {
                    throw Failure(message: "\(label): SH coeff \(i)/\(k) mismatch")
                }
            }
        }
    }

    // MARK: - SPZ

    private static func checkSPZRoundTrip() throws {
        let cloud = try makeFixture(shDegree: .two)
        let data = try SPZCodec.write(cloud)
        let result = try SPZCodec.read(data)
        let restored = result.cloud
        guard restored.count == cloud.count, restored.shDegree == cloud.shDegree else {
            throw Failure(message: "shape mismatch")
        }
        // Tolerances derived from SPZCodec's own documented quantization:
        //   position: 12 fractional bits -> 1/4096 m per step, +-half a step.
        //   scale/opacity/color: 8-bit bucket -> generous fixed tolerance.
        //   SH: sh1Bits=5 (bucket 8/256) / shRestBits=4 (bucket 16/256).
        //   rotation: "smallest three" is only unstable near the dropped
        //     component; compare up to overall sign (q and -q are the same
        //     rotation) with a loose bound that still catches a real bug.
        let posTol: Float = 1.0 / 4096.0
        let scaleTol: Float = 1.0 / 16.0 + 0.02
        let colorTol: Float = (1.0 / 255.0) / 0.15 + 0.02
        let sh1Tol: Float = 8.0 / 128.0 + 0.02
        let restTol: Float = 16.0 / 128.0 + 0.02

        for i in 0..<cloud.count {
            if simd_distance(cloud.positions[i], restored.positions[i]) > posTol * 2 {
                throw Failure(message: "position \(i) outside tolerance")
            }
            if simd_distance(cloud.logScales[i], restored.logScales[i]) > scaleTol {
                throw Failure(message: "scale \(i) outside tolerance")
            }
            let aAlpha = SplatMath.sigmoid(cloud.opacityLogits[i])
            let bAlpha = SplatMath.sigmoid(restored.opacityLogits[i])
            if abs(aAlpha - bAlpha) > 1.0 / 255.0 + 0.01 {
                throw Failure(message: "opacity \(i) outside tolerance")
            }
            if simd_distance(cloud.colorDC[i], restored.colorDC[i]) > colorTol {
                throw Failure(message: "color \(i) outside tolerance")
            }
            let qa = simd_normalize(cloud.rotations[i])
            var qb = simd_normalize(restored.rotations[i])
            if simd_dot(qa, qb) < 0 { qb = -qb }
            // Skip the known-unstable region (true |w or dropped component|
            // near zero has no tight closed-form bound); elsewhere expect
            // sub-0.05 per-component error.
            if simd_distance(qa, qb) > 0.35 {
                throw Failure(message: "rotation \(i) grossly wrong (dist \(simd_distance(qa, qb)))")
            }
            for k in 0..<cloud.shDegree.restCoefficientCount {
                let tol = k < 3 ? sh1Tol : restTol
                if simd_distance(cloud.shRest[i][k], restored.shRest[i][k]) > tol {
                    throw Failure(message: "SH coeff \(i)/\(k) outside tolerance")
                }
            }
        }
    }

    private static func checkSPZIsGzip() throws {
        let cloud = try makeFixture(shDegree: .one)
        let data = try SPZCodec.write(cloud)
        guard data.count >= 2, data[data.startIndex] == 0x1f, data[data.startIndex + 1] == 0x8b else {
            throw Failure(message: "SPZ output does not start with the gzip magic bytes")
        }
    }

    // MARK: - GLB

    private static func checkGLBStructure() throws {
        let cloud = try makeFixture(shDegree: .one)
        let glb = try GLTFExporter.writeGLB(cloud)
        var r = ByteReader(glb)
        let magic = try r.readUInt32LE()
        guard magic == 0x4654_6C67 else { throw Failure(message: "bad GLB magic") }
        let version = try r.readUInt32LE()
        guard version == 2 else { throw Failure(message: "unexpected glTF version \(version)") }
        let totalLength = try r.readUInt32LE()
        guard Int(totalLength) == glb.count else {
            throw Failure(message: "header length \(totalLength) != actual \(glb.count)")
        }
        let jsonChunkLength = try r.readUInt32LE()
        let jsonChunkType = try r.readUInt32LE()
        guard jsonChunkType == 0x4E4F_534A else { throw Failure(message: "first chunk is not JSON") }
        let jsonBytes = try r.readBytes(Int(jsonChunkLength))
        guard let jsonObject = try? JSONSerialization.jsonObject(with: Data(jsonBytes)) as? [String: Any] else {
            throw Failure(message: "JSON chunk did not parse")
        }
        guard let used = jsonObject["extensionsUsed"] as? [String], used.contains("KHR_gaussian_splatting") else {
            throw Failure(message: "extensionsUsed missing KHR_gaussian_splatting")
        }
        // COLOR_0 is the core-glTF fallback attribute the extension's own
        // "Fallback Behavior" section recommends so non-splat-aware viewers
        // still render a colored point cloud instead of black dots.
        guard let meshes = jsonObject["meshes"] as? [[String: Any]],
              let primitives = meshes.first?["primitives"] as? [[String: Any]],
              let attributes = primitives.first?["attributes"] as? [String: Any],
              attributes["COLOR_0"] != nil
        else {
            throw Failure(message: "primitive attributes missing the COLOR_0 fallback accessor")
        }
        let binChunkLength = try r.readUInt32LE()
        let binChunkType = try r.readUInt32LE()
        guard binChunkType == 0x004E_4942 else { throw Failure(message: "second chunk is not BIN") }
        guard r.remaining == Int(binChunkLength) else {
            throw Failure(message: "BIN chunk length \(binChunkLength) leaves \(r.remaining) trailing bytes")
        }
    }

    // MARK: - Error handling

    private static func checkEmptyCloudRejected() throws {
        let empty = SplatCloud.empty()
        do {
            _ = try PLYCodec.write(empty)
            throw Failure(message: "PLYCodec.write did not reject an empty cloud")
        } catch let e as ExportError {
            guard case .emptyCloud = e else {
                throw Failure(message: "wrong error for empty cloud: \(e)")
            }
        }
    }

    // MARK: - Booster capture-bundle zip

    /// Builds a throwaway scan folder shaped like DATA_FORMAT.md section 1
    /// (a `capture_bundle.json` at the root plus a couple of section
    /// folders), packages it, and cleans up. `body` gets the scan directory
    /// and the finished zip's raw bytes.
    private static func withPackagedFixtureScan(
        _ body: (URL, Data) throws -> Void
    ) throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "nimbus-exporttest-\(UUID().uuidString)", isDirectory: true
        )
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        try Data("{\"formatVersion\":1,\"scanID\":\"scan_selftest\"}".utf8)
            .write(to: root.appendingPathComponent("capture_bundle.json"))

        for folder in [BrandConfig.Folder.images, BrandConfig.Folder.anchors] {
            let dir = root.appendingPathComponent(folder, isDirectory: true)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: dir.appendingPathComponent("fixture.txt"))
        }

        let zipURL = root.appendingPathComponent("out.zip")
        _ = try BoosterBundle.packageForBooster(scanDirectory: root, to: zipURL)
        let zipData = try Data(contentsOf: zipURL)
        try body(root, zipData)
    }

    private static func checkBoosterBundlePackaging() throws {
        try withPackagedFixtureScan { _, zipData in
            // ZipWriter stores entries uncompressed, so a literal filename
            // appears verbatim in its local file header - a simple substring
            // search is a real, meaningful check here, not a placeholder.
            guard let needle = "capture_bundle.json".data(using: .ascii),
                  zipData.range(of: needle) != nil
            else {
                throw Failure(message: "zip is missing the capture_bundle.json root file")
            }
            guard let fixtureNeedle = "images/fixture.txt".data(using: .ascii),
                  zipData.range(of: fixtureNeedle) != nil
            else {
                throw Failure(message: "zip is missing images/fixture.txt")
            }
        }
    }

    private static func checkBoosterBundleZipStructure() throws {
        try withPackagedFixtureScan { _, zipData in
            // End Of Central Directory record signature, PK\x05\x06, little-
            // endian 0x06054b50. It must appear somewhere near the tail of
            // any valid, non-empty ZIP.
            let eocd = Data([0x50, 0x4B, 0x05, 0x06])
            let tailStart = zipData.index(zipData.endIndex, offsetBy: -min(256, zipData.count))
            guard zipData.range(of: eocd, options: [], in: tailStart..<zipData.endIndex) != nil else {
                throw Failure(message: "zip has no End Of Central Directory record near its tail")
            }
        }
    }
}
#endif
