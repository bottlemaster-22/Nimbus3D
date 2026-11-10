//
//  SplatExporter.swift — Export module
//
//  Places the trained Gaussian-splat file into the export directory.
//
//  HONEST STATUS:
//   - .spz input  -> REAL byte-for-byte copy into `<root>/splat.spz`.
//   - .ply input  -> the .ply is copied verbatim into `<root>/splat.ply` (it is a
//                    valid, usable splat file), but it is NOT recompressed to .spz.
//                    A correct PLY->SPZ encoder must be byte-exact against Niantic's
//                    format; writing one blind would risk emitting corrupt files that
//                    "look" real, which violates the project's no-vibecoding rule.
//                    See TODO below.
//

import Foundation

enum SplatExporter {

    struct Result {
        var url: URL
        /// True when the output is genuinely `.spz`. False when a `.ply` was passed
        /// through uncompressed (caller/UI should surface this honestly).
        var isCompressedSPZ: Bool
    }

    /// Copies the splat into `directory`. Never throws on format mismatch — it always
    /// produces a usable file — but reports via `isCompressedSPZ` whether real SPZ
    /// compression happened.
    static func export(_ model: SplatModel, into directory: URL) throws -> Result {
        let fm = FileManager.default
        switch model.format {
        case .spz:
            let dest = directory.appendingPathComponent("splat.spz")
            try replaceItem(at: dest, withCopyOf: model.splatFileURL, fm: fm)
            return Result(url: dest, isCompressedSPZ: true)

        case .ply:
            // TODO(nimbus): implement a byte-exact PLY -> SPZ encoder (Niantic .spz:
            //   NGSP header {magic 0x5053474e, version, numPoints, shDegree,
            //   fractionalBits, flags}, then quantized positions(24-bit fixed) / alpha /
            //   color / scale / rotation / SH sections, whole stream gzip-compressed).
            //   Preferred: expose it from the Rust/Brush core (NimbusSplatCore) which
            //   already owns the splat representation, rather than re-deriving it here.
            let dest = directory.appendingPathComponent("splat.ply")
            try replaceItem(at: dest, withCopyOf: model.splatFileURL, fm: fm)
            return Result(url: dest, isCompressedSPZ: false)
        }
    }

    private static func replaceItem(at dest: URL, withCopyOf src: URL, fm: FileManager) throws {
        if fm.fileExists(atPath: dest.path) {
            try fm.removeItem(at: dest)
        }
        try fm.copyItem(at: src, to: dest)
    }
}
