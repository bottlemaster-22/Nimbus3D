//
//  ExportManifest.swift — Export module
//
//  A small `manifest.json` written into each export directory. It records what was
//  produced (relative filenames + honest flags) so the Library tab can list exports
//  without re-parsing glTF, and so a moved/copied export folder stays self-describing.
//  Filenames are relative to the export directory, so the folder is portable.
//

import Foundation

struct ExportManifest: Codable, Equatable {
    /// Bumped if the manifest layout changes.
    var manifestVersion: Int = 1
    var assetID: UUID
    var createdAt: Date

    var glbFilename: String?
    var gltfFilename: String?
    var splatFilename: String?
    /// False when a `.ply` was passed through instead of being compressed to `.spz`.
    var splatIsCompressedSPZ: Bool?
    var hdriFilename: String?
    var usdzFilename: String?
    var imageFilenames: [String]

    var materialClass: String?
    var vertexCount: Int
    var triangleCount: Int

    static func read(from directory: URL) -> ExportManifest? {
        let url = directory.appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ExportManifest.self, from: data)
    }

    func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: directory.appendingPathComponent(ExportManifest.fileName), options: .atomic)
    }

    static let fileName = "manifest.json"
}
