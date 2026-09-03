//
//  PLYBoundingBox.swift
//  Nimbus3D - SplatEngine module
//
//  Reads a Gaussian-splat PLY back from disk to recover the axis-aligned
//  bounding box and the authoritative splat (vertex) count for SplatModel.
//
//  Brush exports binary_little_endian 1.0 PLYs whose first three vertex
//  properties are `float x, y, z` (the 3DGS convention). This parser is written
//  against the header rather than assuming a fixed layout: it reads the property
//  table, locates x/y/z by name, and computes per-record offsets. It also
//  supports ASCII PLYs as a fallback. Big-endian PLYs are rejected explicitly
//  (iOS is little-endian and Brush never emits them).
//

import Foundation
import simd

enum PLYBoundingBox {

    /// Returns (bounding box over all vertices, vertex count).
    static func compute(url: URL) throws -> (AxisAlignedBoundingBox, Int) {
        let data = try Data(contentsOf: url)

        guard let bodyStart = headerEnd(in: data) else {
            throw NimbusError.trainingFailed("invalid PLY: no end_header marker in \(url.lastPathComponent)")
        }

        let headerText = String(decoding: data.subdata(in: 0..<bodyStart), as: UTF8.self)
        let header = try parseHeader(headerText)

        guard header.vertexCount > 0 else {
            // A valid but empty model; return a degenerate box rather than fail.
            return (AxisAlignedBoundingBox(minCorner: .zero, maxCorner: .zero), 0)
        }
        guard let xi = header.propertyIndex["x"],
              let yi = header.propertyIndex["y"],
              let zi = header.propertyIndex["z"] else {
            throw NimbusError.trainingFailed("PLY is missing x/y/z vertex properties")
        }

        let body = data.subdata(in: bodyStart..<data.count)
        let box: AxisAlignedBoundingBox
        switch header.format {
        case .binaryLittleEndian:
            box = try binaryBounds(body: body, header: header, xi: xi, yi: yi, zi: zi)
        case .ascii:
            box = try asciiBounds(body: body, header: header, xi: xi, yi: yi, zi: zi)
        case .binaryBigEndian:
            throw NimbusError.trainingFailed("big-endian PLY is not supported")
        }
        return (box, header.vertexCount)
    }

    // MARK: - Header

    private enum Format { case ascii, binaryLittleEndian, binaryBigEndian }

    private struct Header {
        var format: Format = .ascii
        var vertexCount: Int = 0
        var propertyTypes: [String] = []      // scalar type per vertex property, in order
        var propertyOffsets: [Int] = []       // byte offset within a vertex record
        var recordStride: Int = 0             // bytes per vertex record (binary)
        var propertyIndex: [String: Int] = [:] // property name -> position
    }

    private static func parseHeader(_ text: String) throws -> Header {
        var header = Header()
        var inVertexElement = false
        var names: [String] = []

        for rawLine in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            let tokens = line.split(separator: " ").map(String.init)
            guard let keyword = tokens.first else { continue }

            switch keyword {
            case "format":
                if tokens.count >= 2 {
                    switch tokens[1] {
                    case "ascii": header.format = .ascii
                    case "binary_little_endian": header.format = .binaryLittleEndian
                    case "binary_big_endian": header.format = .binaryBigEndian
                    default:
                        throw NimbusError.trainingFailed("unknown PLY format: \(tokens[1])")
                    }
                }
            case "element":
                if tokens.count >= 3 {
                    inVertexElement = (tokens[1] == "vertex")
                    if inVertexElement {
                        header.vertexCount = Int(tokens[2]) ?? 0
                    }
                }
            case "property":
                // Only scalar properties on the vertex element concern us. A
                // `property list ...` (faces) would break fixed striding, but
                // splat PLYs have none; reject to avoid silent misreads.
                guard inVertexElement else { break }
                if tokens.count >= 3 && tokens[1] == "list" {
                    throw NimbusError.trainingFailed("unexpected list property on PLY vertex element")
                }
                if tokens.count >= 3 {
                    header.propertyTypes.append(tokens[1])
                    names.append(tokens[tokens.count - 1])
                }
            default:
                break
            }
        }

        var offset = 0
        for (i, type) in header.propertyTypes.enumerated() {
            header.propertyOffsets.append(offset)
            offset += byteSize(of: type)
            header.propertyIndex[names[i]] = i
        }
        header.recordStride = offset
        return header
    }

    // MARK: - Bounds

    private static func binaryBounds(body: Data, header: Header,
                                     xi: Int, yi: Int, zi: Int) throws -> AxisAlignedBoundingBox {
        let stride = header.recordStride
        let required = stride * header.vertexCount
        guard body.count >= required else {
            throw NimbusError.trainingFailed("PLY body truncated: need \(required) bytes, have \(body.count)")
        }

        var minCorner = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxCorner = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        body.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for i in 0..<header.vertexCount {
                let base = i * stride
                let x = readScalar(raw, base + header.propertyOffsets[xi], header.propertyTypes[xi])
                let y = readScalar(raw, base + header.propertyOffsets[yi], header.propertyTypes[yi])
                let z = readScalar(raw, base + header.propertyOffsets[zi], header.propertyTypes[zi])
                let v = SIMD3<Float>(x, y, z)
                minCorner = simd_min(minCorner, v)
                maxCorner = simd_max(maxCorner, v)
            }
        }
        return AxisAlignedBoundingBox(minCorner: minCorner, maxCorner: maxCorner)
    }

    private static func asciiBounds(body: Data, header: Header,
                                    xi: Int, yi: Int, zi: Int) throws -> AxisAlignedBoundingBox {
        let text = String(decoding: body, as: UTF8.self)
        var minCorner = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxCorner = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var parsed = 0

        for rawLine in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            if parsed >= header.vertexCount { break }
            let values = rawLine.split(separator: " ").compactMap { Float($0) }
            guard values.count >= header.propertyTypes.count,
                  xi < values.count, yi < values.count, zi < values.count else { continue }
            let v = SIMD3<Float>(values[xi], values[yi], values[zi])
            minCorner = simd_min(minCorner, v)
            maxCorner = simd_max(maxCorner, v)
            parsed += 1
        }
        guard parsed > 0 else {
            throw NimbusError.trainingFailed("ASCII PLY declared \(header.vertexCount) vertices but none parsed")
        }
        return AxisAlignedBoundingBox(minCorner: minCorner, maxCorner: maxCorner)
    }

    // MARK: - Scalar decoding

    private static func byteSize(of type: String) -> Int {
        switch type {
        case "char", "int8", "uchar", "uint8": return 1
        case "short", "int16", "ushort", "uint16": return 2
        case "int", "int32", "uint", "uint32", "float", "float32": return 4
        case "double", "float64", "int64", "uint64": return 8
        default: return 4
        }
    }

    /// Reads one scalar at `offset` as Float. Little-endian load; the buffer is
    /// treated as unaligned because vertex records pack heterogeneous types.
    private static func readScalar(_ raw: UnsafeRawBufferPointer, _ offset: Int, _ type: String) -> Float {
        switch type {
        case "float", "float32":
            return raw.loadUnaligned(fromByteOffset: offset, as: Float.self)
        case "double", "float64":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: Double.self))
        case "uchar", "uint8":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: UInt8.self))
        case "char", "int8":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: Int8.self))
        case "ushort", "uint16":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        case "short", "int16":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: Int16.self))
        case "uint", "uint32":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        case "int", "int32":
            return Float(raw.loadUnaligned(fromByteOffset: offset, as: Int32.self))
        default:
            return raw.loadUnaligned(fromByteOffset: offset, as: Float.self)
        }
    }

    /// Byte index immediately after the header's terminating newline.
    private static func headerEnd(in data: Data) -> Int? {
        let marker = Data("end_header".utf8)
        guard let range = data.range(of: marker) else { return nil }
        var index = range.upperBound
        if index < data.count, data[index] == 0x0D { index += 1 } // CR
        if index < data.count, data[index] == 0x0A { index += 1 } // LF
        return index
    }
}
