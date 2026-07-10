//
//  NimbusAssetExporter.swift — Export module
//
//  Concrete `AssetExporter`: writes a game-ready deliverable to a single directory —
//  glTF and/or GLB (mesh + PBR MaterialSet), the .spz/.ply splat, the equirect .exr
//  HDRI, an optional USDZ, and a manifest.json describing them all.
//
//  What is REAL here: glTF/GLB writing, ORM texture packing, HDRI/splat file placement,
//  and the manifest are genuine, deterministic on-device work. USDZ is real but gated on
//  platform capability; PLY->SPZ recompression is honestly deferred (see SplatExporter).
//

import Foundation

public final class NimbusAssetExporter: AssetExporter {

    public init() {}

    public func export(mesh: MeshAsset,
                       materials: MaterialSet?,
                       splat: SplatModel?,
                       hdri: HDRIEnvironment?,
                       options: ExportOptions,
                       progress: @escaping ProgressHandler) async throws -> ExportedAsset {
        do {
            return try await run(mesh: mesh,
                                 materials: materials,
                                 splat: splat,
                                 hdri: hdri,
                                 options: options,
                                 progress: progress)
        } catch let error as NimbusError {
            throw error
        } catch is CancellationError {
            throw NimbusError.cancelled
        } catch {
            throw NimbusError.exportFailed(String(describing: error))
        }
    }

    // MARK: - Implementation

    private func run(mesh: MeshAsset,
                     materials: MaterialSet?,
                     splat: SplatModel?,
                     hdri: HDRIEnvironment?,
                     options: ExportOptions,
                     progress: @escaping ProgressHandler) async throws -> ExportedAsset {
        try Task.checkCancellation()
        guard !options.formats.isEmpty else {
            throw NimbusError.exportFailed("no export formats requested")
        }

        let assetID = UUID()
        let fm = FileManager.default
        let root = try makeRootDirectory(options: options, assetID: assetID, fm: fm)
        report(progress, 0.05, "Preparing export")

        // HDRI first so its relative path can be recorded in the glTF scene extras.
        var hdriURL: URL? = nil
        var sceneExtras: NimbusSceneExtras? = nil
        if options.includeHDRI, let hdri = hdri {
            try Task.checkCancellation()
            let dest = root.appendingPathComponent("environment.exr")
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.copyItem(at: hdri.equirectangularEXRURL, to: dest)
            hdriURL = dest
            sceneExtras = NimbusSceneExtras(environmentEXR: "environment.exr")
            report(progress, 0.15, "Copied HDRI environment")
        }

        // Build the container-agnostic glTF graph once (textures packed here).
        try Task.checkCancellation()
        report(progress, 0.25, "Building glTF (\(mesh.vertexCount) verts, \(mesh.triangleCount) tris)")
        let assembly = try GLTFAssembly.build(mesh: mesh, materials: materials, sceneExtras: sceneExtras)

        var glbURL: URL? = nil
        var gltfURL: URL? = nil
        let imageFilenames: [String] = assembly.imageSlots.map { $0.filename }

        if options.formats.contains(.glb) {
            try Task.checkCancellation()
            let data = try assembly.glb()
            let dest = root.appendingPathComponent("model.glb")
            try data.write(to: dest, options: .atomic)
            glbURL = dest
            report(progress, 0.5, "Wrote model.glb")
        }

        if options.formats.contains(.gltf) {
            try Task.checkCancellation()
            let (json, bin, files) = try assembly.gltf(binFilename: "model.bin")
            try json.write(to: root.appendingPathComponent("model.gltf"), options: .atomic)
            try bin.write(to: root.appendingPathComponent("model.bin"), options: .atomic)
            for slot in files {
                try slot.bytes.write(to: root.appendingPathComponent(slot.filename), options: .atomic)
            }
            gltfURL = root.appendingPathComponent("model.gltf")
            report(progress, 0.65, "Wrote model.gltf")
        }

        // Splat
        var splatURL: URL? = nil
        var splatIsSPZ: Bool? = nil
        if options.includeSplat, let splat = splat {
            try Task.checkCancellation()
            let result = try SplatExporter.export(splat, into: root)
            splatURL = result.url
            splatIsSPZ = result.isCompressedSPZ
            report(progress, 0.8, result.isCompressedSPZ ? "Copied .spz splat"
                                                         : "Copied .ply splat (SPZ compression deferred)")
        }

        // USDZ (best-effort)
        try Task.checkCancellation()
        let usdzURL = USDZExporter.export(mesh: mesh,
                                          materials: materials,
                                          to: root.appendingPathComponent("model.usdz"))
        if usdzURL != nil { report(progress, 0.9, "Wrote model.usdz") }

        // Manifest
        let manifest = ExportManifest(assetID: assetID,
                                      createdAt: Date(),
                                      glbFilename: glbURL?.lastPathComponent,
                                      gltfFilename: gltfURL?.lastPathComponent,
                                      splatFilename: splatURL?.lastPathComponent,
                                      splatIsCompressedSPZ: splatIsSPZ,
                                      hdriFilename: hdriURL?.lastPathComponent,
                                      usdzFilename: usdzURL?.lastPathComponent,
                                      imageFilenames: imageFilenames,
                                      materialClass: materials?.classification.materialClass.rawValue,
                                      vertexCount: mesh.vertexCount,
                                      triangleCount: mesh.triangleCount)
        try manifest.write(to: root)

        report(progress, 1.0, "Export complete")
        return ExportedAsset(id: assetID,
                             rootDirectory: root,
                             gltfURL: gltfURL,
                             glbURL: glbURL,
                             materialSet: materials,
                             hdriURL: hdriURL,
                             splatURL: splatURL)
    }

    // MARK: - Helpers

    private func makeRootDirectory(options: ExportOptions, assetID: UUID, fm: FileManager) throws -> URL {
        let root: URL
        if let dir = options.outputDirectory {
            root = dir
        } else {
            let documents = try fm.url(for: .documentDirectory,
                                       in: .userDomainMask,
                                       appropriateFor: nil,
                                       create: true)
            root = documents.appendingPathComponent("Exports", isDirectory: true)
                            .appendingPathComponent(assetID.uuidString, isDirectory: true)
        }
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func report(_ handler: ProgressHandler, _ fraction: Double, _ message: String) {
        handler(PipelineProgress(stage: .export, fractionCompleted: fraction, message: message))
    }
}
