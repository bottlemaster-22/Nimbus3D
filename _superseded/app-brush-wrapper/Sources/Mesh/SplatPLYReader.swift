//
//  SplatPLYReader.swift — Mesh module (REAL)
//
//  Parses a Gaussian-splat PLY (Brush's native training output) into `[GaussianSplat]`.
//  Supports `format binary_little_endian 1.0` (the format Brush/3DGS actually writes)
//  and `format ascii 1.0` as a fallback. Handles arbitrary property ordering by name,
//  which is important because different trainers emit the SH `f_rest_*` block in
//  different widths.
//
//  Field conventions from the 3DGS reference exporter:
//    - x, y, z            : centre in world metres (stored as-is)
//    - scale_0..scale_2   : log-space std-dev  -> world scale = exp(value)
//    - opacity            : logit             -> linear opacity = sigmoid(value)
//    - rot_0..rot_3       : quaternion (unused by the density field; dropped)
//    - f_dc_*, f_rest_*   : spherical-harmonic colour (dropped — meshing is geometry-only)
//

import Foundation
import simd

enum SplatPLYReader {

    enum ParseError: Error, CustomStringConvertible {
        case notPLY
        case unsupportedFormat(String)
        case missingProperty(String)
        case truncated

        var description: String {
            switch self {
            case .notPLY: return "File does not start with a PLY magic header."
            case .unsupportedFormat(let f): return "Unsupported PLY format line: \(f)"
            case .missingProperty(let p): return "PLY is missing required splat property '\(p)'."
            case .truncated: return "PLY body is shorter than the declared vertex count."
            }
        }
    }

    private enum ScalarType {
        case float32, float64, int32, uint32, int16, uint16, int8, uint8

        init?(ply token: String) {
            switch token {
            case "float", "float32": self = .float32
            case "double", "float64": self = .float64
            case "int", "int32": self = .int32
            case "uint", "uint32": self = .uint32
            case "short", "int16": self = .int16
            case "ushort", "uint16": self = .uint16
            case "char", "int8": self = .int8
            case "uchar", "uint8": self = .uint8
            default: return nil
            }
        }

        var byteWidth: Int {
            switch self {
            case .float64: return 8
            case .float32, .int32, .uint32: return 4
            case .int16, .uint16: return 2
            case .int8, .uint8: return 1
            }
        }
    }

    private struct Property {
        var name: String
        var type: ScalarType
        /// Byte offset of this property within one vertex record (binary path).
        var offset: Int
    }

    /// Reads and decodes the splat PLY at `url`.
    static func read(contentsOf url: URL) throws -> [GaussianSplat] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try read(data: data)
    }

    static func read(data: Data) throws -> [GaussianSplat] {
        // --- Locate the "end_header\n" boundary without decoding the whole (possibly huge) blob. ---
        guard data.count >= 3,
              data[0] == UInt8(ascii: "p"),
              data[1] == UInt8(ascii: "l"),
              data[2] == UInt8(ascii: "y") else {
            throw ParseError.notPLY
        }

        let marker = Array("end_header".utf8)
        guard let headerEnd = range(of: marker, in: data) else { throw ParseError.truncated }
        // Advance past the newline that follows end_header.
        var bodyStart = headerEnd + marker.count
        while bodyStart < data.count, data[bodyStart] == 0x0D || data[bodyStart] == 0x0A {
            bodyStart += 1
        }

        let headerData = data.subdata(in: 0..<headerEnd)
        guard let headerText = String(data: headerData, encoding: .ascii)
                ?? String(data: headerData, encoding: .utf8) else {
            throw ParseError.truncated
        }

        // --- Parse header lines. ---
        var isBinaryLE = false
        var isASCII = false
        var vertexCount = 0
        var properties: [Property] = []
        var runningOffset = 0
        var inVertexElement = false

        for rawLine in headerText.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            let tokens = line.split(separator: " ").map(String.init)
            switch tokens.first {
            case "format":
                guard tokens.count >= 2 else { throw ParseError.unsupportedFormat(line) }
                switch tokens[1] {
                case "binary_little_endian": isBinaryLE = true
                case "ascii": isASCII = true
                case "binary_big_endian": throw ParseError.unsupportedFormat("binary_big_endian (unsupported)")
                default: throw ParseError.unsupportedFormat(tokens[1])
                }
            case "element":
                guard tokens.count >= 3 else { break }
                if tokens[1] == "vertex" {
                    inVertexElement = true
                    vertexCount = Int(tokens[2]) ?? 0
                } else {
                    inVertexElement = false
                }
            case "property":
                guard inVertexElement, tokens.count >= 3,
                      let type = ScalarType(ply: tokens[1]) else { break }
                // Note: list properties (tokens[1] == "list") never appear on splat vertices.
                let name = tokens[tokens.count - 1]
                properties.append(Property(name: name, type: type, offset: runningOffset))
                runningOffset += type.byteWidth
            default:
                break
            }
        }

        guard isBinaryLE || isASCII else { throw ParseError.unsupportedFormat("no format line") }
        let stride = runningOffset

        func index(_ name: String) throws -> Property {
            guard let p = properties.first(where: { $0.name == name }) else {
                throw ParseError.missingProperty(name)
            }
            return p
        }

        let px = try index("x"), py = try index("y"), pz = try index("z")
        let s0 = try index("scale_0"), s1 = try index("scale_1"), s2 = try index("scale_2")
        let op = try index("opacity")

        var out = [GaussianSplat]()
        out.reserveCapacity(vertexCount)

        if isBinaryLE {
            let body = data.subdata(in: bodyStart..<data.count)
            guard body.count >= stride * vertexCount else { throw ParseError.truncated }
            body.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                for i in 0..<vertexCount {
                    let rec = base + i * stride
                    func f(_ prop: Property) -> Float { readScalar(rec, prop) }
                    let pos = SIMD3<Float>(f(px), f(py), f(pz))
                    let scale = SIMD3<Float>(expf(f(s0)), expf(f(s1)), expf(f(s2)))
                    let opacity = sigmoid(f(op))
                    out.append(GaussianSplat(position: pos, scale: scale, opacity: opacity))
                }
            }
        } else {
            // ASCII path: whitespace-separated values, one vertex per line, column order = property order.
            guard let bodyText = String(data: data.subdata(in: bodyStart..<data.count), encoding: .ascii)
                    ?? String(data: data.subdata(in: bodyStart..<data.count), encoding: .utf8) else {
                throw ParseError.truncated
            }
            let nameToColumn = Dictionary(uniqueKeysWithValues: properties.enumerated().map { ($1.name, $0) })
            let cx = nameToColumn["x"]!, cy = nameToColumn["y"]!, cz = nameToColumn["z"]!
            let cs0 = nameToColumn["scale_0"]!, cs1 = nameToColumn["scale_1"]!, cs2 = nameToColumn["scale_2"]!
            let cop = nameToColumn["opacity"]!
            var parsed = 0
            for line in bodyText.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
                if parsed >= vertexCount { break }
                let cols = line.split(separator: " ").compactMap { Float($0) }
                let maxCol = max(cx, cy, cz, cs0, cs1, cs2, cop)
                guard cols.count > maxCol else { continue }
                let pos = SIMD3<Float>(cols[cx], cols[cy], cols[cz])
                let scale = SIMD3<Float>(expf(cols[cs0]), expf(cols[cs1]), expf(cols[cs2]))
                out.append(GaussianSplat(position: pos, scale: scale, opacity: sigmoid(cols[cop])))
                parsed += 1
            }
        }

        return out
    }

    // MARK: - Scalar decode

    private static func readScalar(_ rec: UnsafeRawPointer, _ prop: Property) -> Float {
        let p = rec + prop.offset
        switch prop.type {
        case .float32: return p.loadUnaligned(as: Float.self)
        case .float64: return Float(p.loadUnaligned(as: Double.self))
        case .int32:   return Float(p.loadUnaligned(as: Int32.self))
        case .uint32:  return Float(p.loadUnaligned(as: UInt32.self))
        case .int16:   return Float(p.loadUnaligned(as: Int16.self))
        case .uint16:  return Float(p.loadUnaligned(as: UInt16.self))
        case .int8:    return Float(p.loadUnaligned(as: Int8.self))
        case .uint8:   return Float(p.loadUnaligned(as: UInt8.self))
        }
    }

    private static func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }

    /// Naive byte-sequence search — the header is tiny, so this is not a hot path.
    private static func range(of pattern: [UInt8], in data: Data) -> Int? {
        guard !pattern.isEmpty, data.count >= pattern.count else { return nil }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
            let bytes = raw.bindMemory(to: UInt8.self)
            let last = data.count - pattern.count
            var i = 0
            while i <= last {
                var match = true
                for j in 0..<pattern.count where bytes[i + j] != pattern[j] {
                    match = false; break
                }
                if match { return i }
                i += 1
            }
            return nil
        }
    }
}
