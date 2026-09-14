//
//  ZipWriter.swift
//  Export
//
//  A minimal, from-scratch ZIP (PKWARE APPNOTE) archive writer: local file
//  headers, a central directory, and an end-of-central-directory record.
//  This is what BoosterBundle.swift uses to package a capture bundle
//  directory into the single .zip the LAN Booster protocol transfers.
//
//  Deliberate scope: STORE (method 0, no compression) only, no Zip64. A
//  capture bundle's bulk is JPEG/HEIC frames and binary depth sidecars,
//  already compressed formats that DEFLATE would barely shrink; STORE keeps
//  this writer simple, deterministic, and trivially correct to reason about
//  without a Swift compiler on hand to test against (see MODULE_STATUS.md).
//  The Zip64 ceiling this therefore accepts (4 GiB per entry / per archive,
//  65 535 entries) is generous for one phone capture session's frame count
//  and comfortably above what F10's densification/keyframe budget implies,
//  but `finish()` throws a clear, typed error rather than silently emitting
//  a corrupt archive if a bundle ever grows past it - see
//  ExportError.archiveLimitExceeded.
//
//  Streams every entry from disk in bounded chunks (never loads a whole
//  source file into memory), because the bundle this packages can be a full
//  scan's worth of RGB frames - exactly the kind of allocation the product
//  spec's budget/thermal-gating philosophy (F10) says never to assume fits.
//

import Foundation

final class ZipWriter {

    private struct CentralDirectoryEntry {
        let nameBytes: [UInt8]
        let crc32: UInt32
        let size: UInt32
        let localHeaderOffset: UInt32
        let dosTime: UInt16
        let dosDate: UInt16
    }

    private static let maxEntrySize: UInt64 = 0xFFFF_FFFE
    private static let maxEntryCount = 65_534

    private let output: FileHandle
    private let destinationURL: URL
    private var centralDirectory: [CentralDirectoryEntry] = []
    private var offset: UInt64 = 0
    private var finished = false

    /// Creates (overwriting) the destination file and opens it for
    /// streaming writes.
    init(creatingAt url: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            try? fm.removeItem(at: url)
        }
        guard fm.createFile(atPath: url.path, contents: nil) else {
            throw ExportError.ioFailure("could not create \(url.lastPathComponent)")
        }
        guard let handle = FileHandle(forWritingAtPath: url.path) else {
            throw ExportError.ioFailure("could not open \(url.lastPathComponent) for writing")
        }
        self.output = handle
        self.destinationURL = url
    }

    deinit {
        try? output.close()
    }

    /// Adds one entry, streaming its bytes from `fileURL` in chunks.
    /// `archivePath` is the forward-slash entry name inside the zip (e.g.
    /// `"images/frame_20260903_141205_512.jpg"`, the
    /// `frame_YYYYMMDD_HHMMSS_mmm` stamp per docs/DATA_FORMAT.md section 2).
    func addEntry(archivePath: String, fileURL: URL, chunkSize: Int = 4 * 1024 * 1024) throws {
        guard centralDirectory.count < Self.maxEntryCount else {
            throw ExportError.archiveLimitExceeded(
                "more than \(Self.maxEntryCount) files; this packager does not support Zip64"
            )
        }
        guard let input = FileHandle(forReadingAtPath: fileURL.path) else {
            throw ExportError.ioFailure("could not open \(fileURL.lastPathComponent) for reading")
        }
        defer { try? input.close() }

        let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path)
        let modDate = (attrs?[.modificationDate] as? Date) ?? Date()
        let (dosTime, dosDate) = Self.dosTimestamp(from: modDate)

        let nameBytes = Array(archivePath.utf8)
        let localHeaderOffset = offset

        // First pass: stream-read to compute CRC32 and total size without
        // holding the file in memory.
        var crc = CRC32.Incremental()
        var totalSize: UInt64 = 0
        while let chunk = try input.read(upToCount: chunkSize), !chunk.isEmpty {
            crc.update(chunk)
            totalSize += UInt64(chunk.count)
        }
        guard totalSize <= Self.maxEntrySize else {
            throw ExportError.archiveLimitExceeded(
                "\(archivePath) is \(totalSize) bytes; this packager does not support Zip64 entries"
            )
        }
        let finalCRC = crc.finalize()
        let size32 = UInt32(totalSize)

        try writeLocalHeader(
            nameBytes: nameBytes, crc32: finalCRC, size: size32, dosTime: dosTime, dosDate: dosDate
        )

        // Second pass: stream the actual bytes.
        try input.seek(toOffset: 0)
        while let chunk = try input.read(upToCount: chunkSize), !chunk.isEmpty {
            try output.write(contentsOf: chunk)
        }
        // writeLocalHeader() already advanced `offset` past the header; only
        // the data bytes just written remain to be accounted for.
        offset += totalSize

        centralDirectory.append(CentralDirectoryEntry(
            nameBytes: nameBytes, crc32: finalCRC, size: size32,
            localHeaderOffset: UInt32(localHeaderOffset), dosTime: dosTime, dosDate: dosDate
        ))
    }

    /// Adds one entry from an in-memory buffer (used for small, generated
    /// files like manifest.json, not for streamed capture data).
    func addEntry(archivePath: String, data: Data, modificationDate: Date = Date()) throws {
        guard centralDirectory.count < Self.maxEntryCount else {
            throw ExportError.archiveLimitExceeded(
                "more than \(Self.maxEntryCount) files; this packager does not support Zip64"
            )
        }
        guard UInt64(data.count) <= Self.maxEntrySize else {
            throw ExportError.archiveLimitExceeded(
                "\(archivePath) is \(data.count) bytes; this packager does not support Zip64 entries"
            )
        }
        let nameBytes = Array(archivePath.utf8)
        let (dosTime, dosDate) = Self.dosTimestamp(from: modificationDate)
        let crc = CRC32.checksum(data)
        let localHeaderOffset = offset

        try writeLocalHeader(
            nameBytes: nameBytes, crc32: crc, size: UInt32(data.count), dosTime: dosTime, dosDate: dosDate
        )
        try output.write(contentsOf: data)
        // writeLocalHeader() already advanced `offset` past the header; only
        // the data bytes just written remain to be accounted for.
        offset += UInt64(data.count)

        centralDirectory.append(CentralDirectoryEntry(
            nameBytes: nameBytes, crc32: crc, size: UInt32(data.count),
            localHeaderOffset: UInt32(localHeaderOffset), dosTime: dosTime, dosDate: dosDate
        ))
    }

    /// Writes the central directory and end-of-central-directory record.
    /// Must be called exactly once, after all entries are added.
    @discardableResult
    func finish() throws -> URL {
        guard !finished else { throw ExportError.ioFailure("ZipWriter.finish() called twice") }
        finished = true

        let centralDirectoryStart = offset
        for entry in centralDirectory {
            var header = Data(capacity: 46 + entry.nameBytes.count)
            header.appendUInt32LE(0x0201_4b50)
            header.appendUInt16LE(0x0014)          // version made by (2.0, generic)
            header.appendUInt16LE(0x0014)          // version needed to extract (2.0)
            header.appendUInt16LE(0x0800)          // general purpose flag: UTF-8 name
            header.appendUInt16LE(0)                // compression method: STORE
            header.appendUInt16LE(entry.dosTime)
            header.appendUInt16LE(entry.dosDate)
            header.appendUInt32LE(entry.crc32)
            header.appendUInt32LE(entry.size)       // compressed size == size (STORE)
            header.appendUInt32LE(entry.size)       // uncompressed size
            header.appendUInt16LE(UInt16(entry.nameBytes.count))
            header.appendUInt16LE(0)                // extra field length
            header.appendUInt16LE(0)                // file comment length
            header.appendUInt16LE(0)                // disk number start
            header.appendUInt16LE(0)                // internal file attributes
            header.appendUInt32LE(0)                // external file attributes
            header.appendUInt32LE(entry.localHeaderOffset)
            header.append(contentsOf: entry.nameBytes)
            try output.write(contentsOf: header)
            offset += UInt64(header.count)
        }
        let centralDirectorySize = offset - centralDirectoryStart

        var eocd = Data(capacity: 22)
        eocd.appendUInt32LE(0x0605_4b50)
        eocd.appendUInt16LE(0)                                  // this disk
        eocd.appendUInt16LE(0)                                  // disk with central directory start
        eocd.appendUInt16LE(UInt16(centralDirectory.count))     // entries on this disk
        eocd.appendUInt16LE(UInt16(centralDirectory.count))     // total entries
        eocd.appendUInt32LE(UInt32(centralDirectorySize))
        eocd.appendUInt32LE(UInt32(centralDirectoryStart))
        eocd.appendUInt16LE(0)                                  // comment length
        try output.write(contentsOf: eocd)

        try output.close()
        return destinationURL
    }

    // MARK: - Local header

    private func writeLocalHeader(
        nameBytes: [UInt8], crc32: UInt32, size: UInt32, dosTime: UInt16, dosDate: UInt16
    ) throws {
        var header = Data(capacity: 30 + nameBytes.count)
        header.appendUInt32LE(0x0403_4b50)
        header.appendUInt16LE(0x0014)   // version needed to extract
        header.appendUInt16LE(0x0800)   // general purpose flag: UTF-8 name
        header.appendUInt16LE(0)        // compression method: STORE
        header.appendUInt16LE(dosTime)
        header.appendUInt16LE(dosDate)
        header.appendUInt32LE(crc32)
        header.appendUInt32LE(size)     // compressed size == size (STORE)
        header.appendUInt32LE(size)     // uncompressed size
        header.appendUInt16LE(UInt16(nameBytes.count))
        header.appendUInt16LE(0)        // extra field length
        header.append(contentsOf: nameBytes)
        try output.write(contentsOf: header)
        offset += UInt64(header.count)
    }

    // MARK: - DOS date/time

    private static func dosTimestamp(from date: Date) -> (time: UInt16, date: UInt16) {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        let c = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
        // Spelled out one Int at a time. As a single expression the optional
        // unwraps, the shifts and the UInt16 conversion gave the type checker
        // more overload combinations than it would finish in reasonable time,
        // and it gave up rather than compiling. Same arithmetic, same result.
        let year: Int = max(1980, min(2107, c.year ?? 1980))
        let month: Int = c.month ?? 1
        let day: Int = c.day ?? 1
        let hour: Int = c.hour ?? 0
        let minute: Int = c.minute ?? 0
        let second: Int = c.second ?? 0

        let dateBits: Int = ((year - 1980) << 9) | (month << 5) | day
        let timeBits: Int = (hour << 11) | (minute << 5) | (second / 2)

        let dosDate = UInt16(truncatingIfNeeded: dateBits)
        let dosTime = UInt16(truncatingIfNeeded: timeBits)
        return (dosTime, dosDate)
    }
}
