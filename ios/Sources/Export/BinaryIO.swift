//
//  BinaryIO.swift
//  Export
//
//  Small, allocation-conscious little-endian read/write helpers shared by
//  every codec in this module (PLY, SPZ, glTF/GLB, ZIP). Everything here
//  avoids `withUnsafeBytes { $0.load(as:) }` on arbitrary offsets: on Apple
//  platforms an unaligned typed load through a raw pointer is undefined
//  behavior (it can trap on some architectures/optimization levels), and
//  file-format offsets are essentially never naturally aligned. Byte-by-byte
//  reconstruction is a few instructions slower and always correct.
//

import Foundation

extension Data {
    mutating func appendUInt16LE(_ v: UInt16) {
        append(UInt8(v & 0xFF))
        append(UInt8((v >> 8) & 0xFF))
    }

    mutating func appendUInt32LE(_ v: UInt32) {
        append(UInt8(v & 0xFF))
        append(UInt8((v >> 8) & 0xFF))
        append(UInt8((v >> 16) & 0xFF))
        append(UInt8((v >> 24) & 0xFF))
    }

    mutating func appendInt32LE(_ v: Int32) {
        appendUInt32LE(UInt32(bitPattern: v))
    }

    mutating func appendFloat32LE(_ v: Float) {
        appendUInt32LE(v.bitPattern)
    }

    mutating func appendASCII(_ s: String) {
        append(contentsOf: Array(s.utf8))
    }
}

/// A forward-only cursor over a byte buffer, used by every reader in this
/// module. Bounds-checked; every read throws `ExportError.malformedFile`
/// instead of trapping on truncated input (import-back must never crash the
/// app on a hand-edited or corrupt file).
struct ByteReader {
    let bytes: [UInt8]
    private(set) var offset: Int = 0

    init(_ data: Data) { self.bytes = [UInt8](data) }
    init(_ bytes: [UInt8]) { self.bytes = bytes }

    var remaining: Int { bytes.count - offset }

    mutating func seek(to newOffset: Int) throws {
        guard newOffset >= 0, newOffset <= bytes.count else {
            throw ExportError.malformedFile("seek out of range")
        }
        offset = newOffset
    }

    mutating func skip(_ n: Int) throws {
        try seek(to: offset + n)
    }

    mutating func readBytes(_ n: Int) throws -> [UInt8] {
        guard n >= 0, offset + n <= bytes.count else {
            throw ExportError.malformedFile("unexpected end of file (need \(n) more bytes, have \(remaining))")
        }
        let slice = Array(bytes[offset..<offset + n])
        offset += n
        return slice
    }

    mutating func readUInt8() throws -> UInt8 {
        try readBytes(1)[0]
    }

    mutating func readUInt16LE() throws -> UInt16 {
        let b = try readBytes(2)
        return UInt16(b[0]) | (UInt16(b[1]) << 8)
    }

    mutating func readUInt32LE() throws -> UInt32 {
        let b = try readBytes(4)
        return UInt32(b[0]) | (UInt32(b[1]) << 8) | (UInt32(b[2]) << 16) | (UInt32(b[3]) << 24)
    }

    mutating func readInt32LE() throws -> Int32 {
        Int32(bitPattern: try readUInt32LE())
    }

    mutating func readFloat32LE() throws -> Float {
        Float(bitPattern: try readUInt32LE())
    }

    /// Reads one line terminated by `\n` (an optional trailing `\r` is
    /// stripped), consuming the terminator. Used only for PLY header
    /// parsing, which is defined to be ASCII.
    mutating func readLine() throws -> String {
        guard offset < bytes.count else {
            throw ExportError.malformedFile("unexpected end of file while reading a header line")
        }
        var end = offset
        while end < bytes.count, bytes[end] != 0x0A { end += 1 }
        guard end < bytes.count else {
            throw ExportError.malformedFile("header line missing newline terminator")
        }
        var lineBytes = bytes[offset..<end]
        if lineBytes.last == 0x0D { lineBytes = lineBytes.dropLast() }
        offset = end + 1
        return String(decoding: lineBytes, as: UTF8.self)
    }
}
