//
//  EXRWriter.swift — HDRI module.
//
//  Minimal, spec-conformant OpenEXR writer: single-part, scanline, 32-bit FLOAT,
//  three channels (B, G, R), NO_COMPRESSION, INCREASING_Y.
//
//  Why hand-rolled: iOS ships no reliable OpenEXR *encoder* (ImageIO's EXR
//  support is read-oriented and version-dependent). Rather than fake HDR output
//  through a lossy container, this writes the real OpenEXR byte layout so the
//  file opens as true 32-bit linear float in Blender / Nuke / macOS Preview.
//
//  Format reference: openexr.com "Technical Introduction" / OpenEXRFileLayout.
//  Layout: magic, version, header attributes, per-scanline offset table, then one
//  uncompressed scanline block per row (int y, int dataSize, then channels in
//  alphabetical order — each channel a contiguous run of `width` little-endian
//  floats).
//

import Foundation

enum EXRWriter {

    /// Write linear RGB radiance to a 32-bit float equirectangular EXR.
    /// - Parameter rgb: row-major, top-origin (`rgb[y*width + x]`), length width*height.
    static func write(rgb: [SIMD3<Float>], width: Int, height: Int, to url: URL) throws {
        precondition(rgb.count == width * height, "pixel count must equal width*height")
        precondition(width > 0 && height > 0, "EXR dimensions must be positive")

        let header = buildHeader(width: width, height: height)

        let dataSize = width * 3 * 4            // 3 channels * float32
        let blockSize = 8 + dataSize            // int y + int dataSize + pixels
        let headerLen = header.count
        let offsetTableSize = height * 8
        let firstBlock = headerLen + offsetTableSize

        var offsets = Data(capacity: offsetTableSize)
        for y in 0..<height {
            offsets.appendUInt64LE(UInt64(firstBlock + y * blockSize))
        }

        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            throw NimbusError.hdriAssemblyFailed("Cannot open EXR for writing at \(url.path)")
        }
        defer { try? fh.close() }

        do {
            try fh.write(contentsOf: header)
            try fh.write(contentsOf: offsets)

            // Reusable per-channel float row buffer to avoid per-pixel allocations.
            var chan = [Float](repeating: 0, count: width)
            var row = Data(capacity: blockSize)

            for y in 0..<height {
                row.removeAll(keepingCapacity: true)
                row.appendInt32LE(Int32(y))
                row.appendInt32LE(Int32(dataSize))
                let base = y * width
                // Alphabetical channel order: B, G, R.
                appendChannel(&row, &chan, rgb, base, width, keyPath: \.z)
                appendChannel(&row, &chan, rgb, base, width, keyPath: \.y)
                appendChannel(&row, &chan, rgb, base, width, keyPath: \.x)
                try fh.write(contentsOf: row)
            }
        } catch {
            throw NimbusError.hdriAssemblyFailed("Failed writing EXR bytes: \(error.localizedDescription)")
        }
    }

    private static func appendChannel(_ row: inout Data,
                                      _ scratch: inout [Float],
                                      _ rgb: [SIMD3<Float>],
                                      _ base: Int, _ width: Int,
                                      keyPath: KeyPath<SIMD3<Float>, Float>) {
        for x in 0..<width { scratch[x] = rgb[base + x][keyPath: keyPath] }
        // ARM is little-endian; Float array bytes are already the required layout.
        scratch.withUnsafeBytes { row.append(contentsOf: $0) }
    }

    private static func buildHeader(width: Int, height: Int) -> Data {
        var h = Data()
        h.appendUInt32LE(0x0131_2F76)   // magic (20000630)
        h.appendUInt32LE(2)             // version 2, flags 0 (single-part scanline)

        writeAttr(&h, name: "channels", type: "chlist") { v in
            for name in ["B", "G", "R"] {
                v.appendNullString(name)
                v.appendInt32LE(2)      // pixel type: FLOAT
                v.appendUInt8(0)        // pLinear
                v.appendUInt8(0); v.appendUInt8(0); v.appendUInt8(0) // reserved
                v.appendInt32LE(1)      // xSampling
                v.appendInt32LE(1)      // ySampling
            }
            v.appendUInt8(0)            // channel-list terminator
        }
        writeAttr(&h, name: "compression", type: "compression") { $0.appendUInt8(0) } // NO_COMPRESSION
        writeAttr(&h, name: "dataWindow", type: "box2i") { v in
            v.appendInt32LE(0); v.appendInt32LE(0)
            v.appendInt32LE(Int32(width - 1)); v.appendInt32LE(Int32(height - 1))
        }
        writeAttr(&h, name: "displayWindow", type: "box2i") { v in
            v.appendInt32LE(0); v.appendInt32LE(0)
            v.appendInt32LE(Int32(width - 1)); v.appendInt32LE(Int32(height - 1))
        }
        writeAttr(&h, name: "lineOrder", type: "lineOrder") { $0.appendUInt8(0) } // INCREASING_Y
        writeAttr(&h, name: "pixelAspectRatio", type: "float") { $0.appendFloatLE(1.0) }
        writeAttr(&h, name: "screenWindowCenter", type: "v2f") { v in
            v.appendFloatLE(0); v.appendFloatLE(0)
        }
        writeAttr(&h, name: "screenWindowWidth", type: "float") { $0.appendFloatLE(1.0) }

        h.appendUInt8(0)                // end-of-header
        return h
    }

    private static func writeAttr(_ d: inout Data, name: String, type: String,
                                  _ body: (inout Data) -> Void) {
        d.appendNullString(name)
        d.appendNullString(type)
        var value = Data()
        body(&value)
        d.appendInt32LE(Int32(value.count))
        d.append(value)
    }
}

private extension Data {
    mutating func appendUInt8(_ v: UInt8) { append(v) }

    mutating func appendUInt32LE(_ v: UInt32) {
        var x = v.littleEndian
        Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) }
    }

    mutating func appendInt32LE(_ v: Int32) { appendUInt32LE(UInt32(bitPattern: v)) }

    mutating func appendUInt64LE(_ v: UInt64) {
        var x = v.littleEndian
        Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) }
    }

    mutating func appendFloatLE(_ v: Float) { appendUInt32LE(v.bitPattern) }

    mutating func appendNullString(_ s: String) {
        append(contentsOf: Array(s.utf8))
        append(0)
    }
}
