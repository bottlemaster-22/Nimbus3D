//
//  GLTFExporter.swift
//  Export
//
//  Writes a .glb (binary glTF 2.0) containing one KHR_gaussian_splatting
//  point primitive. Verified against the RATIFIED Khronos specification
//  (github.com/KhronosGroup/glTF, extensions/2.0/Khronos/KHR_gaussian_splatting,
//  status "Complete, Ratified by the Khronos Group" as of 2026-02) fetched
//  and read in full on 2026-09-03, not reconstructed from memory - the
//  attribute semantics, accessor types, required properties and coordinate
//  handling below are transcribed from that spec's README and JSON schema.
//
//  Attribute semantics (per spec):
//    POSITION                                VEC3 float, required
//    KHR_gaussian_splatting:ROTATION         VEC4 float, unit quaternion (x,y,z,w)
//    KHR_gaussian_splatting:SCALE            VEC3 float, LINEAR, must not be negative
//    KHR_gaussian_splatting:OPACITY          SCALAR float, linear [0,1]
//    KHR_gaussian_splatting:SH_DEGREE_0_COEF_0        VEC3 float, required
//    KHR_gaussian_splatting:SH_DEGREE_1_COEF_[0-2]    VEC3 float, degree 1
//    KHR_gaussian_splatting:SH_DEGREE_2_COEF_[0-4]    VEC3 float, degree 2
//    KHR_gaussian_splatting:SH_DEGREE_3_COEF_[0-6]    VEC3 float, degree 3
//    COLOR_0 (core glTF, not extension-namespaced)     VEC4 float, linear, optional
//                                                       fallback for viewers that don't
//                                                       understand the extension at all
//                                                       (spec's own "Fallback Behavior"
//                                                       section: RGB from the degree-0 SH
//                                                       coefficient decoded to linear,
//                                                       alpha = opacity)
//
//  SplatCloud stores scale on a log scale and opacity as a pre-sigmoid
//  logit (see SplatCloud.swift); this exporter is the one place that
//  activates them (exp / sigmoid), because glTF - unlike PLY and SPZ - wants
//  the linear, already-activated values.
//
//  Coordinate frame: glTF's coordinate system (+X right, +Y up, +Z toward
//  the viewer / camera looks down -Z) is the SAME RUB convention SplatCloud
//  already stores everything in (see SplatCloud.swift's header comment), so
//  - unlike PLYCodec - no flip happens here.
//
//  Scope: writes only. This module's import-back mandate (per the task
//  brief) is .ply and .spz, both long-established interchange formats;
//  KHR_gaussian_splatting was only ratified in February 2026 and this app
//  is itself the producer of any .glb it would need to read, so a GLB
//  reader is not implemented. `KHR_spz_gaussian_splats_compression` (a
//  compression extension building on this base one) was still an open,
//  unmerged pull request against the glTF repository as of this writing, so
//  this exporter deliberately emits the RATIFIED uncompressed base
//  extension only, not that draft.
//
//  Encoding: STORE, no compression (glTF/GLB has no notion of compressing
//  the binary chunk itself; a compressed transmission format is a separate,
//  unratified extension - see above). Every accessor is plain float32.
//

import Foundation
import simd

enum GLTFExporter {

    private static let colorSpace = "srgb_rec709_display"

    static func writeGLB(_ cloud: SplatCloud) throws -> Data {
        guard cloud.count > 0 else { throw ExportError.emptyCloud }
        let n = cloud.count
        let shDim = cloud.shDegree.restCoefficientCount

        // MARK: Binary chunk - one contiguous, 4-byte-aligned block per accessor.

        var bin = Data(capacity: n * (12 + 16 + 12 + 4 + 12 + shDim * 12))
        var accessors: [(semantic: String, byteOffset: Int, byteLength: Int, type: String, min: [Float]?, max: [Float]?)] = []

        func appendVec3Block(_ semantic: String, _ values: [SIMD3<Float>], withMinMax: Bool = false) {
            let byteOffset = bin.count
            var minV = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var maxV = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            for v in values {
                bin.appendFloat32LE(v.x); bin.appendFloat32LE(v.y); bin.appendFloat32LE(v.z)
                if withMinMax {
                    minV = simd_min(minV, v)
                    maxV = simd_max(maxV, v)
                }
            }
            accessors.append((
                semantic, byteOffset, bin.count - byteOffset, "VEC3",
                withMinMax ? [minV.x, minV.y, minV.z] : nil,
                withMinMax ? [maxV.x, maxV.y, maxV.z] : nil
            ))
        }

        // POSITION - required min/max per the glTF core spec.
        appendVec3Block("POSITION", cloud.positions, withMinMax: true)

        // KHR_gaussian_splatting:ROTATION - VEC4 (x, y, z, w), normalized.
        do {
            let byteOffset = bin.count
            for r in cloud.rotations {
                let lengthSquared = simd_length_squared(r)
                let q = (lengthSquared.isFinite && lengthSquared > 1e-12)
                    ? simd_normalize(r) : SIMD4<Float>(0, 0, 0, 1)
                bin.appendFloat32LE(q.x); bin.appendFloat32LE(q.y)
                bin.appendFloat32LE(q.z); bin.appendFloat32LE(q.w)
            }
            accessors.append(("KHR_gaussian_splatting:ROTATION", byteOffset, bin.count - byteOffset, "VEC4", nil, nil))
        }

        // KHR_gaussian_splatting:SCALE - VEC3, linear (exp of the stored log-scale).
        appendVec3Block(
            "KHR_gaussian_splatting:SCALE",
            cloud.logScales.map { SIMD3(expf($0.x), expf($0.y), expf($0.z)) }
        )

        // KHR_gaussian_splatting:OPACITY - SCALAR, linear [0, 1] (sigmoid of the stored logit).
        do {
            let byteOffset = bin.count
            for a in cloud.opacityLogits {
                bin.appendFloat32LE(min(1, max(0, SplatMath.sigmoid(a))))
            }
            accessors.append(("KHR_gaussian_splatting:OPACITY", byteOffset, bin.count - byteOffset, "SCALAR", nil, nil))
        }

        // KHR_gaussian_splatting:SH_DEGREE_0_COEF_0 - VEC3, raw DC coefficient.
        appendVec3Block("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", cloud.colorDC)

        // COLOR_0 - core glTF fallback attribute, per this extension's own "Fallback
        // Behavior" section (verified against the ratified README on 2026-09-03): a
        // plain point-cloud/model viewer that does not understand
        // KHR_gaussian_splatting at all still renders every splat center as a
        // correctly colored point instead of a black or default-white dot. Per the
        // spec: RGB = clamp(0.282095 * f_dc + 0.5, 0, 1) in this file's declared
        // `colorSpace` ("srgb_rec709_display"), decoded to LINEAR because the glTF
        // core spec requires COLOR_0 to hold linear values; alpha = the same
        // activated opacity written to KHR_gaussian_splatting:OPACITY above ("the
        // alpha channel SHOULD contain the opacity of the splat").
        do {
            let byteOffset = bin.count
            for i in 0..<n {
                let dc = cloud.colorDC[i]
                let displayR = min(1, max(0, dc.x * SplatMath.shDCToColor + 0.5))
                let displayG = min(1, max(0, dc.y * SplatMath.shDCToColor + 0.5))
                let displayB = min(1, max(0, dc.z * SplatMath.shDCToColor + 0.5))
                bin.appendFloat32LE(srgbDisplayToLinear(displayR))
                bin.appendFloat32LE(srgbDisplayToLinear(displayG))
                bin.appendFloat32LE(srgbDisplayToLinear(displayB))
                bin.appendFloat32LE(min(1, max(0, SplatMath.sigmoid(cloud.opacityLogits[i]))))
            }
            accessors.append(("COLOR_0", byteOffset, bin.count - byteOffset, "VEC4", nil, nil))
        }

        // Higher-degree coefficients, one VEC3 accessor per (degree, order) pair,
        // gathered from SplatCloud.shRest[i][k] across all points.
        let bandStartIndex = [1: 0, 2: 3, 3: 8]   // first shRest index for each degree
        let bandCoeffCount = [1: 3, 2: 5, 3: 7]
        for degree in 1...3 where cloud.shDegree.rawValue >= degree {
            let start = bandStartIndex[degree]!
            let count = bandCoeffCount[degree]!
            for order in 0..<count {
                let k = start + order
                let gathered = cloud.shRest.map { $0[k] }
                appendVec3Block("KHR_gaussian_splatting:SH_DEGREE_\(degree)_COEF_\(order)", gathered)
            }
        }

        // Pad the binary chunk to a 4-byte boundary (already true here since
        // every block is a whole number of float32s, but GLB requires it be
        // explicit and zero-padded per spec).
        while bin.count % 4 != 0 { bin.append(0) }

        // MARK: JSON chunk

        var bufferViewsJSON: [JSONValue] = []
        var accessorsJSON: [JSONValue] = []
        var attributePairs: [(String, JSONValue)] = []

        for (i, acc) in accessors.enumerated() {
            bufferViewsJSON.append(.object([
                ("buffer", .int(0)),
                ("byteOffset", .int(acc.byteOffset)),
                ("byteLength", .int(acc.byteLength)),
                ("target", .int(34962)),  // ARRAY_BUFFER
            ]))
            var accessorFields: [(String, JSONValue)] = [
                ("bufferView", .int(i)),
                ("componentType", .int(5126)),  // FLOAT
                ("count", .int(n)),
                ("type", .string(acc.type)),
            ]
            if let mn = acc.min, let mx = acc.max {
                accessorFields.append(("min", .array(mn.map { .double(Double($0)) })))
                accessorFields.append(("max", .array(mx.map { .double(Double($0)) })))
            }
            accessorsJSON.append(.object(accessorFields))
            attributePairs.append((acc.semantic, .int(i)))
        }

        let extensionObject: JSONValue = .object([
            ("kernel", .string("ellipse")),
            ("colorSpace", .string(colorSpace)),
            ("projection", .string("perspective")),
            ("sortingMethod", .string("cameraDistance")),
        ])

        let primitive: JSONValue = .object([
            ("attributes", .object(attributePairs)),
            ("mode", .int(0)),  // POINTS, required by the extension
            ("extensions", .object([("KHR_gaussian_splatting", extensionObject)])),
        ])

        let root: JSONValue = .object([
            ("asset", .object([
                ("version", .string("2.0")),
                ("generator", .string("\(BrandConfig.productName) \(BrandConfig.versionString)")),
            ])),
            ("extensionsUsed", .array([.string("KHR_gaussian_splatting")])),
            ("buffers", .array([.object([("byteLength", .int(bin.count))])])),
            ("bufferViews", .array(bufferViewsJSON)),
            ("accessors", .array(accessorsJSON)),
            ("meshes", .array([.object([("primitives", .array([primitive]))])])),
            ("nodes", .array([.object([("mesh", .int(0))])])),
            ("scenes", .array([.object([("nodes", .array([.int(0)]))])])),
            ("scene", .int(0)),
        ])

        var jsonText = root.serialize()
        // GLB pads the JSON chunk with trailing spaces (0x20) to a 4-byte boundary.
        while (jsonText.utf8.count) % 4 != 0 { jsonText += " " }
        let jsonBytes = Data(jsonText.utf8)

        // MARK: GLB container

        var glb = Data(capacity: 12 + 8 + jsonBytes.count + 8 + bin.count)
        glb.appendUInt32LE(0x4654_6C67)  // "glTF"
        glb.appendUInt32LE(2)            // version
        let totalLength = 12 + 8 + jsonBytes.count + 8 + bin.count
        glb.appendUInt32LE(UInt32(totalLength))

        glb.appendUInt32LE(UInt32(jsonBytes.count))
        glb.appendUInt32LE(0x4E4F_534A)  // "JSON"
        glb.append(jsonBytes)

        glb.appendUInt32LE(UInt32(bin.count))
        glb.appendUInt32LE(0x004E_4942)  // "BIN\0" (bytes 0x42 'B', 0x49 'I', 0x4E 'N', 0x00)
        glb.append(bin)

        return glb
    }

    static func writeGLB(_ cloud: SplatCloud, to url: URL) throws {
        let data = try writeGLB(cloud)
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw ExportError.ioFailure("writing \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// sRGB electro-optical transfer function (display/encoded -> linear),
    /// IEC 61966-2-1. Used only for the `COLOR_0` fallback attribute: the
    /// glTF core spec requires `COLOR_0` to be linear, while the value this
    /// exporter's `colorSpace` ("srgb_rec709_display") computes is
    /// display-referred per KHR_gaussian_splatting's own fallback formula.
    private static func srgbDisplayToLinear(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }
}
