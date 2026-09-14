//
//  PipelineStores.swift
//  Nimbus3D - Pipeline module
//
//  On-disk stores the Process and Library screens read:
//    - CaptureBundleStore: scans Documents/Captures/*/manifest.json (written by
//      the Capture module per the CaptureService contract).
//    - AssetLibraryStore: persists an asset.json manifest inside each
//      ExportedAsset.rootDirectory and scans Documents/Exports/*/asset.json
//      (the AssetExporter contract's default output location).
//
//  iOS moves the app container between updates, so absolute URLs stored in
//  manifests can go stale. Both stores rebase decoded URLs onto the directory
//  the manifest was actually found in.
//

import Foundation

// MARK: - Shared helpers

enum PipelinePaths {
    static func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func capturesDirectory() -> URL {
        documentsDirectory().appendingPathComponent("Captures", isDirectory: true)
    }

    static func exportsDirectory() -> URL {
        documentsDirectory().appendingPathComponent("Exports", isDirectory: true)
    }

    /// Re-anchors `url` from `oldRoot` to `newRoot`, preserving the relative
    /// path. Returns `url` unchanged if it is not under `oldRoot`.
    static func rebased(_ url: URL, from oldRoot: URL, to newRoot: URL) -> URL {
        let oldPath = oldRoot.standardizedFileURL.path
        var path = url.standardizedFileURL.path
        guard path.hasPrefix(oldPath) else { return url }
        path.removeFirst(oldPath.count)
        while path.hasPrefix("/") { path.removeFirst() }
        return path.isEmpty ? newRoot : newRoot.appendingPathComponent(path)
    }

    /// Decodes JSON trying the date strategies other modules might have used,
    /// since the manifest encoding is not pinned by the contract.
    static func decodeFlexible<T: Decodable>(_ type: T.Type, from data: Data) -> T? {
        let strategies: [JSONDecoder.DateDecodingStrategy] = [
            .deferredToDate, .iso8601, .secondsSince1970, .millisecondsSince1970,
        ]
        for strategy in strategies {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = strategy
            if let value = try? decoder.decode(type, from: data) {
                return value
            }
        }
        return nil
    }
}

// MARK: - Capture bundles

public enum CaptureBundleStore {

    /// All capture bundles on disk, newest first. Bundles whose manifest cannot
    /// be decoded are skipped.
    public static func loadAll() -> [CaptureBundle] {
        let root = PipelinePaths.capturesDirectory()
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else {
            return []
        }
        var bundles: [CaptureBundle] = []
        for dir in children {
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
            let manifest = dir.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifest),
                  let bundle = PipelinePaths.decodeFlexible(CaptureBundle.self, from: data) else {
                continue
            }
            bundles.append(rebase(bundle, to: dir))
        }
        return bundles.sorted { $0.createdAt > $1.createdAt }
    }

    /// Rebases every file URL in the bundle onto the directory the manifest
    /// actually lives in (handles container moves between app launches).
    static func rebase(_ bundle: CaptureBundle, to newRoot: URL) -> CaptureBundle {
        let oldRoot = bundle.rootDirectory
        guard oldRoot.standardizedFileURL.path != newRoot.standardizedFileURL.path else { return bundle }
        var rebased = bundle
        rebased.rootDirectory = newRoot
        rebased.frames = bundle.frames.map { frame in
            var f = frame
            f.imageURL = PipelinePaths.rebased(f.imageURL, from: oldRoot, to: newRoot)
            f.depthMapURL = f.depthMapURL.map { PipelinePaths.rebased($0, from: oldRoot, to: newRoot) }
            f.depthConfidenceURL = f.depthConfidenceURL.map { PipelinePaths.rebased($0, from: oldRoot, to: newRoot) }
            return f
        }
        rebased.hdriBrackets = bundle.hdriBrackets.map { bracket in
            var b = bracket
            b.imageURL = PipelinePaths.rebased(b.imageURL, from: oldRoot, to: newRoot)
            return b
        }
        return rebased
    }
}

// MARK: - Exported asset library

public enum AssetLibraryStore {

    static let manifestName = "asset.json"

    /// Writes the asset manifest into asset.rootDirectory so the Library tab
    /// can list it across launches.
    public static func save(_ asset: ExportedAsset) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(asset)
        try FileManager.default.createDirectory(at: asset.rootDirectory, withIntermediateDirectories: true)
        try data.write(to: asset.rootDirectory.appendingPathComponent(manifestName), options: .atomic)
    }

    /// All exported assets on disk, newest first.
    public static func loadAll() -> [ExportedAsset] {
        let root = PipelinePaths.exportsDirectory()
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else {
            return []
        }
        var assets: [ExportedAsset] = []
        for dir in children {
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
            let manifest = dir.appendingPathComponent(manifestName)
            guard let data = try? Data(contentsOf: manifest),
                  let asset = PipelinePaths.decodeFlexible(ExportedAsset.self, from: data) else {
                continue
            }
            assets.append(rebase(asset, to: dir))
        }
        return assets.sorted { $0.createdAt > $1.createdAt }
    }

    /// Deletes the asset's whole directory.
    public static func delete(_ asset: ExportedAsset) throws {
        try FileManager.default.removeItem(at: asset.rootDirectory)
    }

    static func rebase(_ asset: ExportedAsset, to newRoot: URL) -> ExportedAsset {
        let oldRoot = asset.rootDirectory
        guard oldRoot.standardizedFileURL.path != newRoot.standardizedFileURL.path else { return asset }
        func move(_ url: URL?) -> URL? {
            url.map { PipelinePaths.rebased($0, from: oldRoot, to: newRoot) }
        }
        var rebased = asset
        rebased.rootDirectory = newRoot
        rebased.gltfURL = move(asset.gltfURL)
        rebased.glbURL = move(asset.glbURL)
        rebased.hdriURL = move(asset.hdriURL)
        rebased.splatURL = move(asset.splatURL)
        if var materials = asset.materialSet {
            materials.albedoURL = PipelinePaths.rebased(materials.albedoURL, from: oldRoot, to: newRoot)
            materials.normalURL = move(materials.normalURL)
            materials.heightURL = move(materials.heightURL)
            materials.roughnessURL = move(materials.roughnessURL)
            materials.metallicURL = move(materials.metallicURL)
            materials.ambientOcclusionURL = move(materials.ambientOcclusionURL)
            rebased.materialSet = materials
        }
        return rebased
    }
}
