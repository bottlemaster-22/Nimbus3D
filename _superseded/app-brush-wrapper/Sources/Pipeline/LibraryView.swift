//
//  LibraryView.swift
//  Nimbus3D - Pipeline module
//
//  The "Library" tab. `LibraryRootView` is the entry point the App shell drops
//  into the Library tab (replacing LibraryPlaceholderView in NimbusApp.swift).
//  It lists every ExportedAsset the pipeline has produced (AssetLibraryStore),
//  and a detail screen shows the on-disk deliverables with a share sheet.
//
//  Note: the 3D preview of a splat is owned by the SplatRender module
//  (SplatRenderer). This screen lists assets and exposes their files; when
//  SplatRender ships, its preview view can be embedded in LibraryDetailView.
//

import SwiftUI
import Observation

@MainActor
@Observable
final class LibraryViewModel {
    private(set) var assets: [ExportedAsset] = []

    func refresh() {
        assets = AssetLibraryStore.loadAll()
    }

    func delete(_ asset: ExportedAsset) {
        try? AssetLibraryStore.delete(asset)
        assets.removeAll { $0.id == asset.id }
    }

    func delete(at offsets: IndexSet) {
        for index in offsets { delete(assets[index]) }
    }
}

// MARK: - Root screen

public struct LibraryRootView: View {
    @State private var model = LibraryViewModel()

    public init() {}

    public var body: some View {
        Group {
            if model.assets.isEmpty {
                ContentUnavailableView(
                    "No Exports Yet",
                    systemImage: "square.grid.2x2",
                    description: Text("Process a capture in the Process tab. Finished glTF/GLB assets show up here.")
                )
            } else {
                List {
                    ForEach(model.assets) { asset in
                        NavigationLink {
                            LibraryDetailView(asset: asset) {
                                model.delete(asset)
                                model.refresh()
                            }
                        } label: {
                            AssetRow(asset: asset)
                        }
                    }
                    .onDelete { model.delete(at: $0) }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    model.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Reload library")
            }
        }
        .onAppear { model.refresh() }
    }
}

// MARK: - Row

private struct AssetRow: View {
    let asset: ExportedAsset

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "cube.transparent.fill")
                .font(.title2)
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(asset.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.body.weight(.medium))
                Text(asset.contentSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Detail

struct LibraryDetailView: View {
    let asset: ExportedAsset
    var onDelete: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showingDeleteConfirm = false

    var body: some View {
        List {
            Section("Deliverables") {
                fileRow("GLB", url: asset.glbURL)
                fileRow("glTF", url: asset.gltfURL)
                fileRow("Splat (.spz)", url: asset.splatURL)
                fileRow("HDRI (.exr)", url: asset.hdriURL)
            }

            if let materials = asset.materialSet {
                Section("Material Set") {
                    LabeledContent("Class", value: materials.classification.materialClass.rawValue.capitalized)
                    LabeledContent("Confidence",
                                   value: "\(Int((materials.classification.confidence * 100).rounded()))%")
                    LabeledContent("Resolution", value: "\(materials.textureResolution)px")
                    fileRow("Albedo", url: materials.albedoURL)
                    fileRow("Normal", url: materials.normalURL)
                    fileRow("Height", url: materials.heightURL)
                    fileRow("Roughness", url: materials.roughnessURL)
                    fileRow("Metallic", url: materials.metallicURL)
                    fileRow("Ambient Occlusion", url: materials.ambientOcclusionURL)
                }
            }

            Section {
                if let primary = asset.glbURL ?? asset.gltfURL, FileManager.default.fileExists(atPath: primary.path) {
                    ShareLink(item: primary) {
                        Label("Share Asset", systemImage: "square.and.arrow.up")
                    }
                }
                Button(role: .destructive) {
                    showingDeleteConfirm = true
                } label: {
                    Label("Delete Asset", systemImage: "trash")
                }
            }
        }
        .navigationTitle("Exported Asset")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Delete this asset and all its files?",
                            isPresented: $showingDeleteConfirm,
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                onDelete()
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    @ViewBuilder
    private func fileRow(_ label: String, url: URL?) -> some View {
        if let url {
            let exists = FileManager.default.fileExists(atPath: url.path)
            HStack {
                Label(label, systemImage: exists ? "doc.fill" : "doc.badge.ellipsis")
                    .labelStyle(.titleOnly)
                Spacer()
                Text(exists ? fileSize(url) : "missing")
                    .font(.caption.monospaced())
                    .foregroundStyle(exists ? .secondary : .red)
            }
        }
    }

    private func fileSize(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

// MARK: - Summary helper

private extension ExportedAsset {
    var contentSummary: String {
        var parts: [String] = []
        if glbURL != nil { parts.append("GLB") }
        if gltfURL != nil { parts.append("glTF") }
        if materialSet != nil { parts.append("materials") }
        if hdriURL != nil { parts.append("HDRI") }
        if splatURL != nil { parts.append("splat") }
        return parts.isEmpty ? "empty export" : parts.joined(separator: " · ")
    }
}
