//
//  CRC32.swift
//  Export
//
//  Standard CRC-32 (IEEE 802.3 / zlib / gzip / PKZIP polynomial 0xEDB88320).
//  Needed because gzip (used by the legacy .spz container) and ZIP (used by
//  the Booster capture bundle) both embed a CRC32 of the uncompressed data,
//  and Apple's Compression framework does not expose one - it only does raw
//  DEFLATE, leaving the container format (and its checksum) to the caller.
//
//  This is the textbook table-driven implementation; the table is generated
//  at first use rather than transcribed by hand, which is both simpler to
//  verify by inspection and immune to a transcription typo in a 256-entry
//  literal table.
//

import Foundation

enum CRC32 {
    private static let table: [UInt32] = {
        var table = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1 != 0) ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1)
            }
            table[i] = c
        }
        return table
    }()

    /// CRC-32 of `data`, matching `zlib.crc32` / `gzip` / PKZIP exactly.
    static func checksum(_ data: Data) -> UInt32 {
        var incremental = Incremental()
        incremental.update(data)
        return incremental.finalize()
    }

    /// Streaming CRC-32, for callers (ZipWriter) that read a large source
    /// file in bounded-size chunks instead of loading it into memory whole.
    struct Incremental {
        private var crc: UInt32 = 0xFFFF_FFFF

        // Explicit no-argument initializer: a `private` stored property
        // would otherwise make the compiler-synthesized memberwise
        // initializer `private` too (scoped to this file), which would
        // break `CRC32.Incremental()` call sites in other files (ZipWriter,
        // ExportSelfTest) even though every property has a default value.
        init() {}

        mutating func update(_ data: Data) {
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                for byte in raw {
                    let index = Int((crc ^ UInt32(byte)) & 0xFF)
                    crc = table[index] ^ (crc >> 8)
                }
            }
        }

        func finalize() -> UInt32 { crc ^ 0xFFFF_FFFF }
    }
}
