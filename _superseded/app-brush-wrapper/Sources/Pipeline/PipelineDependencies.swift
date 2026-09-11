//
//  PipelineDependencies.swift
//  Nimbus3D - Pipeline module
//
//  Dependency container for every stage the orchestrator drives. The Pipeline
//  module depends ONLY on the Core protocols; concrete implementations are
//  registered here by their owning modules (or by the app shell) at launch:
//
//      PipelineServices.shared.splatTrainer = BrushSplatTrainer(...)
//
//  Until a module registers, the corresponding Unwired* stub is in place. It
//  throws NimbusError.notImplemented with a message naming the missing module,
//  so the Process screen reports exactly what is not wired instead of faking a
//  result. These stubs are wiring placeholders, not fake implementations.
//

import Foundation

/// All stage implementations the orchestrator needs.
/// CaptureService and SplatRenderer are deliberately absent: capture happens in
/// the Capture tab before a pipeline run, and preview rendering is driven by the
/// render loop, not by the job orchestrator.
public struct PipelineDependencies: Sendable {
    public var splatTrainer: any SplatTrainer
    public var meshExtractor: any MeshExtractor
    public var delighter: any Delighter
    public var materialClassifier: any MaterialClassifier
    public var textureBuilder: any DynamicTextureBuilder
    public var hdriCapture: any HDRICapture
    public var assetExporter: any AssetExporter

    public init(splatTrainer: any SplatTrainer,
                meshExtractor: any MeshExtractor,
                delighter: any Delighter,
                materialClassifier: any MaterialClassifier,
                textureBuilder: any DynamicTextureBuilder,
                hdriCapture: any HDRICapture,
                assetExporter: any AssetExporter) {
        self.splatTrainer = splatTrainer
        self.meshExtractor = meshExtractor
        self.delighter = delighter
        self.materialClassifier = materialClassifier
        self.textureBuilder = textureBuilder
        self.hdriCapture = hdriCapture
        self.assetExporter = assetExporter
    }

    /// Every slot filled with an honest "not wired" stub.
    public static var unwired: PipelineDependencies {
        PipelineDependencies(splatTrainer: UnwiredSplatTrainer(),
                             meshExtractor: UnwiredMeshExtractor(),
                             delighter: UnwiredDelighter(),
                             materialClassifier: UnwiredMaterialClassifier(),
                             textureBuilder: UnwiredDynamicTextureBuilder(),
                             hdriCapture: UnwiredHDRICapture(),
                             assetExporter: UnwiredAssetExporter())
    }
}

/// Global registration point. Modules assign their implementations at app
/// launch (main thread), before any pipeline run starts.
@MainActor
public enum PipelineServices {
    public static var shared: PipelineDependencies = .unwired
}

// MARK: - Unwired stubs (throw, never fake)

private func unwiredMessage(_ proto: String, module: String) -> String {
    "\(proto) is not wired. The \(module) module must register its implementation in PipelineServices.shared at app launch."
}

public final class UnwiredSplatTrainer: SplatTrainer {
    public init() {}
    public func train(_ bundle: CaptureBundle,
                      config: SplatTrainingConfig,
                      progress: @escaping ProgressHandler) async throws -> SplatModel {
        throw NimbusError.notImplemented(unwiredMessage("SplatTrainer", module: "SplatEngine"))
    }
    public func cancelTraining() async {}
}

public final class UnwiredMeshExtractor: MeshExtractor {
    public init() {}
    public func extractMesh(from model: SplatModel,
                            options: MeshExtractionOptions,
                            progress: @escaping ProgressHandler) async throws -> MeshAsset {
        throw NimbusError.notImplemented(unwiredMessage("MeshExtractor", module: "Mesh"))
    }
}

public final class UnwiredDelighter: Delighter {
    public init() {}
    public func delight(albedoURL: URL,
                        mesh: MeshAsset,
                        environment: HDRIEnvironment?,
                        progress: @escaping ProgressHandler) async throws -> URL {
        throw NimbusError.notImplemented(unwiredMessage("Delighter", module: "Materials"))
    }
}

public final class UnwiredMaterialClassifier: MaterialClassifier {
    public init() {}
    public func classify(albedoURL: URL) async throws -> MaterialClassification {
        throw NimbusError.notImplemented(unwiredMessage("MaterialClassifier", module: "Materials"))
    }
}

public final class UnwiredDynamicTextureBuilder: DynamicTextureBuilder {
    public init() {}
    public func buildMaterialSet(delitAlbedoURL: URL,
                                 mesh: MeshAsset,
                                 classification: MaterialClassification,
                                 options: TextureBuildOptions,
                                 progress: @escaping ProgressHandler) async throws -> MaterialSet {
        throw NimbusError.notImplemented(unwiredMessage("DynamicTextureBuilder", module: "Materials"))
    }
}

public final class UnwiredHDRICapture: HDRICapture {
    public init() {}
    public func assembleHDRI(from bundle: CaptureBundle,
                             options: HDRIAssemblyOptions,
                             progress: @escaping ProgressHandler) async throws -> HDRIEnvironment {
        throw NimbusError.notImplemented(unwiredMessage("HDRICapture", module: "HDRI"))
    }
}

public final class UnwiredAssetExporter: AssetExporter {
    public init() {}
    public func export(mesh: MeshAsset,
                       materials: MaterialSet?,
                       splat: SplatModel?,
                       hdri: HDRIEnvironment?,
                       options: ExportOptions,
                       progress: @escaping ProgressHandler) async throws -> ExportedAsset {
        throw NimbusError.notImplemented(unwiredMessage("AssetExporter", module: "Export"))
    }
}
