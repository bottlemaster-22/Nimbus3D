//
//  GzipCodec.swift
//  Export
//
//  A real, from-scratch gzip (RFC 1952) container built on Apple's system
//  Compression framework.
//
//  WHY THIS EXISTS: Apple's Compression framework's COMPRESSION_ZLIB
//  algorithm produces RAW DEFLATE (RFC 1951) only - no zlib header/trailer
//  (RFC 1950), no gzip header/trailer (RFC 1952). Apple documents the
//  algorithm as equivalent to zlib's deflateInit2 called with windowBits -15,
//  which is precisely "raw deflate, no wrapper". The legacy .spz container
//  (versions 1-3 of the Niantic SPZ format; see SPZCodec.swift) is a single
//  gzip stream, and this project ships no dependency manager at all
//  (project.yml deliberately sets packages: {} so CI never resolves
//  anything), so linking libz/zstd as an external library is not an option
//  either. The fix is small and standard: wrap Apple's raw deflate output in
//  a hand-written 10-byte gzip header and 8-byte CRC32+ISIZE trailer
//  ourselves. That IS gzip - a thin, well-documented container around a raw
//  deflate stream - so the result is a byte-for-byte valid .gz file, readable
//  by any gzip-compatible tool, not a look-alike.
//
//  Scope: encode always writes the minimal header (no filename, no mtime, no
//  extra fields, FLG byte 0x00) so output is deterministic byte-for-byte for
//  the same input. Decode also accepts the optional FEXTRA / FNAME /
//  FCOMMENT / FHCRC fields, since a file this app did not write itself (an
//  .spz downloaded from some other tool) may set them.
//

import Foundation
#if canImport(Compression)
import Compression
#endif

enum GzipError: Error, CustomStringConvertible {
    case notGzip
    case unsupportedCompressionMethod(UInt8)
    case truncated
    case deflateFailed
    case inflateFailed
    case crcMismatch(expected: UInt32, actual: UInt32)
    case sizeMismatch(expected: UInt32, actual: Int)

    var description: String {
        switch self {
        case .notGzip: return "Not a gzip stream (bad magic bytes)."
        case .unsupportedCompressionMethod(let m): return "Unsupported gzip compression method \(m)."
        case .truncated: return "Gzip stream is truncated."
        case .deflateFailed: return "Deflate compression failed."
        case .inflateFailed: return "Deflate decompression failed."
        case .crcMismatch(let e, let a): return "Gzip CRC32 mismatch: expected \(e), got \(a)."
        case .sizeMismatch(let e, let a): return "Gzip size mismatch: header claims \(e) bytes, got \(a)."
        }
    }
}

enum Gzip {

    // MARK: - Public API

    /// Compresses `data` into a byte-exact, minimal-header gzip stream.
    static func compress(_ data: Data) throws -> Data {
        let deflated = try rawDeflate(data)

        var out = Data(capacity: deflated.count + 18)
        out.append(0x1f)
        out.append(0x8b)
        out.append(8)      // CM = 8 (deflate)
        out.append(0)      // FLG = 0 (no extra fields)
        out.append(contentsOf: [0, 0, 0, 0])  // MTIME = 0 (deterministic output)
        out.append(0)      // XFL
        out.append(0xFF)   // OS = 255 (unknown), so output never claims a platform
        out.append(deflated)

        let crc = CRC32.checksum(data)
        out.append(contentsOf: littleEndianBytes(crc))
        let isize = UInt32(truncatingIfNeeded: data.count)
        out.append(contentsOf: littleEndianBytes(isize))
        return out
    }

    /// Decompresses a gzip stream, verifying the trailer CRC32 and size.
    static func decompress(_ data: Data) throws -> Data {
        let bytes = [UInt8](data)
        guard bytes.count >= 18 else { throw GzipError.truncated }
        guard bytes[0] == 0x1f, bytes[1] == 0x8b else { throw GzipError.notGzip }
        let method = bytes[2]
        guard method == 8 else { throw GzipError.unsupportedCompressionMethod(method) }
        let flg = bytes[3]

        var offset = 10
        if flg & 0x04 != 0 {  // FEXTRA
            guard offset + 2 <= bytes.count else { throw GzipError.truncated }
            let xlen = Int(bytes[offset]) | (Int(bytes[offset + 1]) << 8)
            offset += 2 + xlen
        }
        if flg & 0x08 != 0 {  // FNAME - NUL-terminated
            offset = try skipCString(bytes, from: offset)
        }
        if flg & 0x10 != 0 {  // FCOMMENT - NUL-terminated
            offset = try skipCString(bytes, from: offset)
        }
        if flg & 0x02 != 0 {  // FHCRC
            offset += 2
        }
        guard offset <= bytes.count - 8 else { throw GzipError.truncated }

        let trailerStart = bytes.count - 8
        let body = Data(bytes[offset..<trailerStart])
        let expectedCRC = readUInt32LE(bytes, at: trailerStart)
        let expectedSize = readUInt32LE(bytes, at: trailerStart + 4)

        let inflated = try rawInflate(body, expectedSize: Int(expectedSize))

        let actualCRC = CRC32.checksum(inflated)
        guard actualCRC == expectedCRC else {
            throw GzipError.crcMismatch(expected: expectedCRC, actual: actualCRC)
        }
        guard UInt32(truncatingIfNeeded: inflated.count) == expectedSize else {
            throw GzipError.sizeMismatch(expected: expectedSize, actual: inflated.count)
        }
        return inflated
    }

    // MARK: - Raw DEFLATE via Compression framework

    private static func rawDeflate(_ data: Data) throws -> Data {
        #if canImport(Compression)
        if data.isEmpty {
            // A zero-length "stored, final" raw-deflate block is well
            // defined (final-block bit + BTYPE=00 + LEN=0 + NLEN=0xFFFF);
            // synthesizing it directly sidesteps any cross-OS-version
            // behavior difference in compression_encode_buffer on an empty
            // source.
            return Data([0x01, 0x00, 0x00, 0xFF, 0xFF])
        }
        // Raw deflate's worst case (incompressible input) expands by a few
        // bytes per 64 KB block; +50% and a fixed pad is an enormous margin
        // for splat/point-cloud data, which compresses well in practice.
        let capacity = max(64, data.count + data.count / 2 + 4096)
        var dest = [UInt8](repeating: 0, count: capacity)
        let written: Int = dest.withUnsafeMutableBytes { destPtr in
            data.withUnsafeBytes { srcPtr in
                compression_encode_buffer(
                    destPtr.bindMemory(to: UInt8.self).baseAddress!, capacity,
                    srcPtr.bindMemory(to: UInt8.self).baseAddress!, data.count,
                    nil, COMPRESSION_ZLIB
                )
            }
        }
        guard written > 0 else { throw GzipError.deflateFailed }
        return Data(dest[0..<written])
        #else
        throw GzipError.deflateFailed
        #endif
    }

    private static func rawInflate(_ body: Data, expectedSize: Int) throws -> Data {
        #if canImport(Compression)
        guard expectedSize > 0 else { return Data() }
        var dest = [UInt8](repeating: 0, count: expectedSize)
        let written: Int = dest.withUnsafeMutableBytes { destPtr in
            body.withUnsafeBytes { srcPtr -> Int in
                guard let src = srcPtr.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return compression_decode_buffer(
                    destPtr.bindMemory(to: UInt8.self).baseAddress!, expectedSize,
                    src, body.count,
                    nil, COMPRESSION_ZLIB
                )
            }
        }
        guard written == expectedSize else { throw GzipError.inflateFailed }
        return Data(dest)
        #else
        throw GzipError.inflateFailed
        #endif
    }

    // MARK: - Byte helpers

    private static func littleEndianBytes(_ v: UInt32) -> [UInt8] {
        [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
    }

    private static func readUInt32LE(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
    }

    private static func skipCString(_ bytes: [UInt8], from start: Int) throws -> Int {
        var i = start
        while i < bytes.count, bytes[i] != 0 { i += 1 }
        guard i < bytes.count else { throw GzipError.truncated }
        return i + 1
    }
}
