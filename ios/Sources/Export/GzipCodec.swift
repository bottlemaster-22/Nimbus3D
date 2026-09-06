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
    case declaredSizeImplausible(declared: Int, bodyBytes: Int, allowed: Int)

    var description: String {
        switch self {
        case .notGzip: return "Not a gzip stream (bad magic bytes)."
        case .unsupportedCompressionMethod(let m): return "Unsupported gzip compression method \(m)."
        case .truncated: return "Gzip stream is truncated."
        case .deflateFailed: return "Deflate compression failed."
        case .inflateFailed: return "Deflate decompression failed."
        case .crcMismatch(let e, let a): return "Gzip CRC32 mismatch: expected \(e), got \(a)."
        case .sizeMismatch(let e, let a): return "Gzip size mismatch: header claims \(e) bytes, got \(a)."
        case .declaredSizeImplausible(let declared, let bodyBytes, let allowed):
            return "Gzip trailer claims \(declared) uncompressed bytes, past the "
                + "\(allowed)-byte limit for a \(bodyBytes)-byte deflate body. "
                + "The file is truncated or corrupt."
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

    /// The most a raw-deflate stream can expand, in output bytes per input
    /// byte. This is a property of the format rather than a guess. The longest
    /// run a single length/distance pair can encode is 258 bytes, and in a
    /// DYNAMIC Huffman block both alphabets can be shrunk until the length
    /// code and the distance code are one bit each, so those 258 bytes can
    /// cost as little as two bits: 258 * 8 / 2 = 1032 bytes out per byte in.
    /// That is the same 1032:1 ceiling zlib documents.
    ///
    /// It is worth being explicit that 1032 is the DYNAMIC-Huffman figure and
    /// not the fixed-table one, because the fixed tables are the easier thing
    /// to reach for and they give a much smaller answer: under RFC 1951's
    /// fixed tables length code 285 costs 8 bits and every distance code costs
    /// 5, so a maximal match costs 13 bits and the fixed ceiling is only about
    /// 159:1. Bounding with the larger 1032 is what makes this check safe
    /// against any encoder rather than only against fixed-table ones. Real
    /// payloads are nowhere near either figure (a quantized .spz body
    /// compresses at roughly 2:1), so this rejects nothing legitimate while
    /// still catching a trailer that claims more than its own compressed body
    /// could ever have produced.
    private static let deflateMaxExpansionRatio = 1032

    /// An absolute ceiling on what this app will inflate in one allocation.
    ///
    /// The ratio bound above is exact, but it scales with the compressed body,
    /// so a large file carrying a corrupt trailer could still pass it while
    /// asking for more memory than the phone has. This second bound is sized
    /// from what this app's files actually are. SPZCodec.write sizes its own
    /// buffer as `16 + n * (9 + 1 + 3 + 3 + 4 + shDim * 3)`, so a splat costs
    /// 20 bytes of position, alpha, color, scale and rotation plus 3 bytes per
    /// SH coefficient: 20 bytes per splat at SH degree 0 and 65 at degree 3,
    /// where shDim is 15. The largest cloud the trainer will ever produce is
    /// 500,000 splats, the house-sized `scaleCap` in
    /// TrainingBudget.recommended, so a legitimate model.spz is about 33 MB
    /// uncompressed at its very worst, PC-Booster-trained ones included.
    /// 512 MiB sits more than an order of magnitude above that, which leaves
    /// room for an .spz written by some other tool (this decoder deliberately
    /// accepts those; see the file header) while still refusing the
    /// gigabyte-scale request a corrupt trailer asks for.
    ///
    /// Two honest limits on this number, so nobody reads it as a memory
    /// guarantee it is not. The true peak is twice it, because rawInflate
    /// hands back `Data(dest)`, which copies the buffer. And once the
    /// compressed body is larger than about 508 KB (512 MiB / 1032) the ratio
    /// bound already exceeds this ceiling, so from there upwards this ceiling
    /// is the only thing bounding the allocation. What the pair actually buys
    /// is the case that actually happens: a truncated file whose trailer asks
    /// for four gigabytes now throws instead of being jetsammed.
    private static let maxDecompressedBytes = 512 * 1024 * 1024

    private static func rawInflate(_ body: Data, expectedSize: Int) throws -> Data {
        #if canImport(Compression)
        guard expectedSize > 0 else { return Data() }

        // Bound expectedSize BEFORE allocating with it. It arrives straight out
        // of the gzip trailer's ISIZE field, which is entirely under the
        // control of the file: a model.spz whose export or Booster download was
        // interrupted mid-write ends wherever the write stopped, so the four
        // bytes read as ISIZE are whatever happened to land there and can say
        // as much as 4,294,967,295. `decompress` does check the inflated size
        // against ISIZE, but that check runs AFTER this allocation, which is
        // too late to help: iOS kills the process for a four-gigabyte request
        // before any error can be thrown, so a damaged file presents to the
        // user as "the app died" instead of "that file is damaged". Checking
        // here, immediately next to the allocation, is what makes it a thrown
        // error, and keeping the check inside this private function means no
        // future caller can route around it.
        //
        // `body.count` counts bytes already resident in memory, so the
        // multiplication cannot overflow a 64-bit Int on any device that could
        // hold `body` at all. The small addend is slack, so a minimal stream
        // sitting near the boundary is never rejected over a couple of bytes.
        let ratioBound = body.count * deflateMaxExpansionRatio + 64
        let allowed = Swift.min(ratioBound, maxDecompressedBytes)
        guard expectedSize <= allowed else {
            throw GzipError.declaredSizeImplausible(
                declared: expectedSize,
                bodyBytes: body.count,
                allowed: allowed
            )
        }

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
