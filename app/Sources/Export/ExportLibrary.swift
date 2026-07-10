//
//  ExportLibrary.swift — Export module
//
//  The Library tab: lists finished exports by scanning Documents/Exports/<uuid>/manifest.json.
//  Real, self-contained SwiftUI + file I/O. The App shell can drop `ExportLibraryView()` into
//  its Library NavigationStack (this module owns that screen per NIMBUS_CONTRACTS.md).
//

import Foundation
import SwiftUI

/// One row in the Library, derived from an export directory's manifest.
public struct ExportSummary: Identifiable, Equatable, Sendable {
    public var id: UUID
    public var directory: URL
    public var createdAt: Date
    public var materialClass: String?
    public var vertexCount: Int
    public var triangleCount: Int
    public var glbURL: URL?
    public var gltfURL: URL?
    public var usdzURL: URL?
    public var splatURL: URL?
    public var splatIsCompressedSPZ: Bool?
    public var hdriURL: URL?

    /// A single file suitable for a share sheet: prefer USDZ (Quick Look), then GLB, then glTF.
    public var primaryShareURL: URL? { usdzURL ?? glbURL ?? gltfURL }
}

@MainActor
public final class ExportLibraryStore: ObservableObject {
    @Published public private(set) var exports: [ExportSummary] = []

    public init() {}

    /// Default location written by `NimbusAssetExporter` (Documents/Exports).
    public static func defaultExportsRoot() -> URL? {
        try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: false)
            .appendingPathComponent("Exports", isDirectory: true)
    }

    public func reload(root: URL? = ExportLibraryStore.defaultExportsRoot()) {
        guard let root = root else { exports = []; return }
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(at: root,
                                                     includingPropertiesForKeys: nil,
                                                     options: [.skipsHiddenFiles]) else {
            exports = []
            return
        }
        var summaries: [ExportSummary] = []
        for dir in dirs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            guard let m = ExportManifest.read(from: dir) else { continue }
            func resolve(_ name: String?) -> URL? {
                guard let name = name else { return nil }
                return dir.appendingPathComponent(name)
            }
            summaries.append(ExportSummary(id: m.assetID,
                                           directory: dir,
                                           createdAt: m.createdAt,
                                           materialClass: m.materialClass,
                                           vertexCount: m.vertexCount,
                                           triangleCount: m.triangleCount,
                                           glbURL: resolve(m.glbFilename),
                                           gltfURL: resolve(m.gltfFilename),
                                           usdzURL: resolve(m.usdzFilename),
                                           splatURL: resolve(m.splatFilename),
                                           splatIsCompressedSPZ: m.splatIsCompressedSPZ,
                                           hdriURL: resolve(m.hdriFilename)))
        }
        exports = summaries.sorted { $0.createdAt > $1.createdAt }
    }

    public func delete(_ summary: ExportSummary) {
        try? FileManager.default.removeItem(at: summary.directory)
        exports.removeAll { $0.id == summary.id }
    }
}

// MARK: - Views

public struct ExportLibraryView: View {
    @StateObject private var store = ExportLibraryStore()

    public init() {}

    public var body: some View {
        Group {
            if store.exports.isEmpty {
                ContentUnavailableView("No Exports Yet",
                                       systemImage: "shippingbox",
                                       description: Text("Finished 3D assets from the Export module appear here."))
            } else {
                List {
                    ForEach(store.exports) { export in
                        ExportRow(export: export)
                    }
                    .onDelete { indexSet in
                        for index in indexSet { store.delete(store.exports[index]) }
                    }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            Button {
                store.reload()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
        }
        .onAppear { store.reload() }
    }
}

private struct ExportRow: View {
    let export: ExportSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(export.materialClass.map { $0.capitalized } ?? "Model")
                    .font(.headline)
                Spacer()
                if let url = export.primaryShareURL {
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
            }
            Text("\(export.vertexCount) verts · \(export.triangleCount) tris")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                if export.glbURL != nil { tag("GLB") }
                if export.gltfURL != nil { tag("glTF") }
                if export.usdzURL != nil { tag("USDZ") }
                if let isSPZ = export.splatIsCompressedSPZ {
                    tag(isSPZ ? "SPZ" : "PLY")
                }
                if export.hdriURL != nil { tag("HDRI") }
            }
            Text(export.createdAt, style: .date)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
    }

    private func tag(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
    }
}
