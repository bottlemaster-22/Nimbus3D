//
//  SplatPLYParser.swift — SplatRender module.
//
//  REAL, on-device parser for the 3D Gaussian Splatting PLY format produced by
//  the Brush / INRIA trainers. Handles ASCII and binary_little_endian bodies and
//  arbitrary property order/type. Decodes each Gaussian into a GPUSplat:
//
//    scale_i (stored log-space)  -> exp(scale_i)
//    opacity (stored logit)      -> sigmoid(opacity)
//    f_dc_i  (SH degree-0 coeff) -> color = 0.5 + C0 * f_dc_i          (C0 = 0.2820948)
//    rot_0..3 (quaternion w,x,y,z, unnormalized) -> normalized rotation R
//    covariance Σ = (R S)(R S)ᵀ, S = diag(exp(scale))                  (world space)
//
//  Higher-order SH (f_rest_*) is intentionally NOT uploaded for the live preview:
//  storing degree-3 SH is ~48 floats/splat (≈290 MB at 1.5M splats) which is
//  impractical for a phone preview. The preview shows the view-independent DC
//  color, which is the correct base color of the model.
//  TODO(nimbus): for a "final render" mode, upload f_rest_* and evaluate
//  view-dependent SH per-fragment (needs a larger per-splat buffer + SH degree
//  from SplatModel, and probably chunked/streamed buffers to fit memory).
//

import Foundation
import simd

enum SplatParseError: Error, CustomStringConvertible {
    case notPLY
    case badHeader(String)
    case unsupportedFormat(String)
    case missingProperty(String)
    case truncated

    var description: String {
        switch self {
        case .notPLY: return "File does not start with a PLY magic header."
        case .badHeader(let m): return "Malformed PLY header: \(m)"
        case .unsupportedFormat(let m): return "Unsupported PLY format: \(m)"
        case .missingProperty(let m): return "PLY is missing required splat property: \(m)"
        case .truncated: return "PLY body is shorter than the header declares."
        }
    }
}

struct ParsedSplats: Sendable {
    var splats: [GPUSplat]
    var boundingMin: SIMD3<Float>
    var boundingMax: SIMD3<Float>
}

enum SplatPLYParser {

    /// SH degree-0 basis constant (Y_0^0 = 0.5 / sqrt(pi)).
    static let shC0: Float = 0.28209479177387814

    private enum PLYFormat { case ascii, binaryLE, binaryBE }

    private struct Property {
        var name: String
        var size: Int          // bytes
        var readFloat: (UnsafeRawPointer, Int, Bool) -> Float  // (base, offset, littleEndian)
    }

    static func parse(contentsOf url: URL) throws -> ParsedSplats {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try parse(data: data)
    }

    static func parse(data: Data) throws -> ParsedSplats {
        // --- Locate end of ASCII header ---
        guard let headerEnd = range(of: "end_header", in: data) else {
            throw SplatParseError.badHeader("no end_header line")
        }
        // Body starts after the newline that follows "end_header".
        var bodyStart = headerEnd.upperBound
        if bodyStart < data.count, data[bodyStart] == 0x0D { bodyStart += 1 } // CR
        if bodyStart < data.count, data[bodyStart] == 0x0A { bodyStart += 1 } // LF

        let headerData = data.subdata(in: 0..<headerEnd.lowerBound)
        guard let headerText = String(data: headerData, encoding: .ascii) else {
            throw SplatParseError.badHeader("header is not ASCII")
        }

        // --- Parse header lines ---
        var format: PLYFormat?
        var vertexCount = 0
        var properties: [Property] = []
        var inVertexElement = false
        var sawPly = false

        for rawLine in headerText.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let tokens = rawLine.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            guard let keyword = tokens.first else { continue }
            switch keyword {
            case "ply":
                sawPly = true
            case "format":
                guard tokens.count >= 2 else { throw SplatParseError.badHeader("format line") }
                switch tokens[1] {
                case "ascii": format = .ascii
                case "binary_little_endian": format = .binaryLE
                case "binary_big_endian": format = .binaryBE
                default: throw SplatParseError.unsupportedFormat(tokens[1])
                }
            case "element":
                guard tokens.count >= 3 else { throw SplatParseError.badHeader("element line") }
                if tokens[1] == "vertex" {
                    inVertexElement = true
                    vertexCount = Int(tokens[2]) ?? 0
                } else {
                    inVertexElement = false
                }
            case "property":
                guard inVertexElement else { continue } // ignore face/list props
                guard tokens.count >= 3 else { throw SplatParseError.badHeader("property line") }
                if tokens[1] == "list" {
                    // Vertex elements in splat PLYs never use list props; skip defensively.
                    continue
                }
                let type = tokens[1]
                let name = tokens[2]
                guard let (size, reader) = scalarReader(for: type) else {
                    throw SplatParseError.unsupportedFormat("property type \(type)")
                }
                properties.append(Property(name: name, size: size, readFloat: reader))
            default:
                continue
            }
        }

        guard sawPly else { throw SplatParseError.notPLY }
        guard let fmt = format else { throw SplatParseError.badHeader("missing format line") }
        guard vertexCount > 0 else { return ParsedSplats(splats: [], boundingMin: .zero, boundingMax: .zero) }

        // --- Property offsets + required-name lookup ---
        var offsets: [String: (offset: Int, prop: Property)] = [:]
        var stride = 0
        for p in properties {
            offsets[p.name] = (stride, p)
            stride += p.size
        }

        func require(_ names: String...) throws -> [(offset: Int, prop: Property)] {
            try names.map { name in
                guard let hit = offsets[name] else { throw SplatParseError.missingProperty(name) }
                return hit
            }
        }

        let px = try require("x", "y", "z")
        let fdc = try require("f_dc_0", "f_dc_1", "f_dc_2")
        let opacity = try require("opacity")[0]
        let scale = try require("scale_0", "scale_1", "scale_2")
        let rot = try require("rot_0", "rot_1", "rot_2", "rot_3")

        switch fmt {
        case .ascii:
            return try parseASCII(data: data, bodyStart: bodyStart, vertexCount: vertexCount,
                                  properties: properties,
                                  px: px, fdc: fdc, opacity: opacity, scale: scale, rot: rot)
        case .binaryLE, .binaryBE:
            let little = (fmt == .binaryLE)
            return try parseBinary(data: data, bodyStart: bodyStart, vertexCount: vertexCount,
                                   stride: stride, littleEndian: little,
                                   px: px, fdc: fdc, opacity: opacity, scale: scale, rot: rot)
        }
    }

    // MARK: - Binary body

    private static func parseBinary(data: Data, bodyStart: Int, vertexCount: Int, stride: Int,
                                    littleEndian: Bool,
                                    px: [(offset: Int, prop: Property)],
                                    fdc: [(offset: Int, prop: Property)],
                                    opacity: (offset: Int, prop: Property),
                                    scale: [(offset: Int, prop: Property)],
                                    rot: [(offset: Int, prop: Property)]) throws -> ParsedSplats {
        let required = bodyStart + vertexCount * stride
        guard data.count >= required else { throw SplatParseError.truncated }

        var splats = [GPUSplat]()
        splats.reserveCapacity(vertexCount)
        var bmin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var bmax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!
            for i in 0..<vertexCount {
                let rowBase = bodyStart + i * stride
                func read(_ hit: (offset: Int, prop: Property)) -> Float {
                    hit.prop.readFloat(base, rowBase + hit.offset, littleEndian)
                }
                let pos = SIMD3<Float>(read(px[0]), read(px[1]), read(px[2]))
                let dc = SIMD3<Float>(read(fdc[0]), read(fdc[1]), read(fdc[2]))
                let op = read(opacity)
                let sc = SIMD3<Float>(read(scale[0]), read(scale[1]), read(scale[2]))
                let q = SIMD4<Float>(read(rot[0]), read(rot[1]), read(rot[2]), read(rot[3]))

                let splat = makeSplat(position: pos, dc: dc, opacityLogit: op, logScale: sc, quat: q)
                splats.append(splat)
                bmin = simd.min(bmin, pos)
                bmax = simd.max(bmax, pos)
            }
        }
        if splats.isEmpty { bmin = .zero; bmax = .zero }
        return ParsedSplats(splats: splats, boundingMin: bmin, boundingMax: bmax)
    }

    // MARK: - ASCII body

    private static func parseASCII(data: Data, bodyStart: Int, vertexCount: Int,
                                   properties: [Property],
                                   px: [(offset: Int, prop: Property)],
                                   fdc: [(offset: Int, prop: Property)],
                                   opacity: (offset: Int, prop: Property),
                                   scale: [(offset: Int, prop: Property)],
                                   rot: [(offset: Int, prop: Property)]) throws -> ParsedSplats {
        // Map required properties to their column index (order in `properties`).
        func columnIndex(ofOffset off: Int) -> Int {
            var acc = 0
            for (idx, p) in properties.enumerated() {
                if acc == off { return idx }
                acc += p.size
            }
            return 0
        }
        let colX = columnIndex(ofOffset: px[0].offset)
        let colY = columnIndex(ofOffset: px[1].offset)
        let colZ = columnIndex(ofOffset: px[2].offset)
        let colDC = fdc.map { columnIndex(ofOffset: $0.offset) }
        let colOp = columnIndex(ofOffset: opacity.offset)
        let colSc = scale.map { columnIndex(ofOffset: $0.offset) }
        let colRot = rot.map { columnIndex(ofOffset: $0.offset) }

        let bodyData = data.subdata(in: bodyStart..<data.count)
        guard let bodyText = String(data: bodyData, encoding: .ascii) else {
            throw SplatParseError.badHeader("ASCII body is not ASCII")
        }

        var splats = [GPUSplat]()
        splats.reserveCapacity(vertexCount)
        var bmin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var bmax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        var parsed = 0
        for line in bodyText.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            if parsed >= vertexCount { break }
            let cols = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            if cols.isEmpty { continue }
            func f(_ c: Int) -> Float { c < cols.count ? (Float(cols[c]) ?? 0) : 0 }
            let pos = SIMD3<Float>(f(colX), f(colY), f(colZ))
            let dc = SIMD3<Float>(f(colDC[0]), f(colDC[1]), f(colDC[2]))
            let op = f(colOp)
            let sc = SIMD3<Float>(f(colSc[0]), f(colSc[1]), f(colSc[2]))
            let q = SIMD4<Float>(f(colRot[0]), f(colRot[1]), f(colRot[2]), f(colRot[3]))

            splats.append(makeSplat(position: pos, dc: dc, opacityLogit: op, logScale: sc, quat: q))
            bmin = simd.min(bmin, pos)
            bmax = simd.max(bmax, pos)
            parsed += 1
        }
        if splats.isEmpty { bmin = .zero; bmax = .zero }
        return ParsedSplats(splats: splats, boundingMin: bmin, boundingMax: bmax)
    }

    // MARK: - Gaussian assembly

    private static func makeSplat(position: SIMD3<Float>, dc: SIMD3<Float>,
                                  opacityLogit: Float, logScale: SIMD3<Float>,
                                  quat: SIMD4<Float>) -> GPUSplat {
        // Decode activations.
        let color = simd_clamp(SIMD3<Float>(repeating: 0.5) + shC0 * dc,
                               SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        let opacity = 1.0 / (1.0 + exp(-opacityLogit))
        let scale = SIMD3<Float>(exp(logScale.x), exp(logScale.y), exp(logScale.z))

        // Normalize quaternion (stored order rot_0..3 = w, x, y, z).
        var q = quat
        let n = simd_length(q)
        if n > 1e-8 { q /= n } else { q = SIMD4<Float>(1, 0, 0, 0) }
        let w = q.x, x = q.y, y = q.z, z = q.w

        // Rotation matrix (column-major columns).
        let xx = x * x, yy = y * y, zz = z * z
        let xy = x * y, xz = x * z, yz = y * z
        let wx = w * x, wy = w * y, wz = w * z
        let c0 = SIMD3<Float>(1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy))
        let c1 = SIMD3<Float>(2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx))
        let c2 = SIMD3<Float>(2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy))

        // M = R * diag(scale)  (scale each rotation column).
        let m0 = c0 * scale.x
        let m1 = c1 * scale.y
        let m2 = c2 * scale.z
        let M = simd_float3x3(columns: (m0, m1, m2))
        let sigma = M * M.transpose  // symmetric 3x3 world covariance

        // Extract 6 unique entries. simd column-major: sigma[col][row].
        let a0 = sigma[0][0]  // Σ00
        let a1 = sigma[1][0]  // Σ01
        let a2 = sigma[2][0]  // Σ02
        let b0 = sigma[1][1]  // Σ11
        let b1 = sigma[2][1]  // Σ12
        let b2 = sigma[2][2]  // Σ22

        return GPUSplat(px: position.x, py: position.y, pz: position.z,
                        a0: a0, a1: a1, a2: a2,
                        b0: b0, b1: b1, b2: b2,
                        cr: color.x, cg: color.y, cb: color.z, ca: opacity)
    }

    // MARK: - Scalar readers

    /// Returns (byteSize, reader) for a PLY scalar type name.
    private static func scalarReader(for type: String)
        -> (Int, (UnsafeRawPointer, Int, Bool) -> Float)? {
        switch type {
        case "char", "int8":
            return (1, { base, off, _ in Float(base.loadUnaligned(fromByteOffset: off, as: Int8.self)) })
        case "uchar", "uint8":
            return (1, { base, off, _ in Float(base.loadUnaligned(fromByteOffset: off, as: UInt8.self)) })
        case "short", "int16":
            return (2, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt16.self)
                return Float(Int16(bitPattern: le ? v : v.byteSwapped)) })
        case "ushort", "uint16":
            return (2, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt16.self)
                return Float(le ? v : v.byteSwapped) })
        case "int", "int32":
            return (4, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt32.self)
                return Float(Int32(bitPattern: le ? v : v.byteSwapped)) })
        case "uint", "uint32":
            return (4, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt32.self)
                return Float(le ? v : v.byteSwapped) })
        case "float", "float32":
            return (4, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt32.self)
                return Float(bitPattern: le ? v : v.byteSwapped) })
        case "double", "float64":
            return (8, { base, off, le in
                let v = base.loadUnaligned(fromByteOffset: off, as: UInt64.self)
                return Float(Double(bitPattern: le ? v : v.byteSwapped)) })
        default:
            return nil
        }
    }

    // MARK: - Header scan helper

    /// Finds the byte range of the first occurrence of an ASCII needle in `data`.
    private static func range(of needle: String, in data: Data) -> Range<Int>? {
        let bytes = Array(needle.utf8)
        guard !bytes.isEmpty, data.count >= bytes.count else { return nil }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Range<Int>? in
            let p = raw.bindMemory(to: UInt8.self)
            let last = data.count - bytes.count
            var i = 0
            while i <= last {
                if p[i] == bytes[0] {
                    var match = true
                    var k = 1
                    while k < bytes.count {
                        if p[i + k] != bytes[k] { match = false; break }
                        k += 1
                    }
                    if match { return i..<(i + bytes.count) }
                }
                i += 1
            }
            return nil
        }
    }
}
