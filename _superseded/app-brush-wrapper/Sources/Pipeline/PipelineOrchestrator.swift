//
//  PipelineOrchestrator.swift
//  Nimbus3D - Pipeline module
//
//  Runs the full stage chain against the Core protocols only:
//
//    CaptureBundle -> SplatTrainer -> MeshExtractor -> HDRICapture (early, so
//    the environment can inform delighting) -> captured-albedo bake (placeholder,
//    see CapturedAlbedoBaker) -> Delighter -> MaterialClassifier ->
//    DynamicTextureBuilder -> AssetExporter
//
//  Failure policy, chosen so honest stubs do not brick the whole pipeline:
//    - splatTraining, meshExtraction, export: REQUIRED. Any error is fatal.
//    - hdriAssembly, delighting, materialClassification, textureBuild: OPTIONAL.
//      NimbusError.notImplemented (a stub) marks the stage skipped; other errors
//      mark it failed; either way the run continues with a degraded but honest
//      result (lit albedo instead of de-lit, .unknown material, albedo-only
//      material set, no HDRI).
//    - Cancellation (Task cancellation or cancel()) always aborts with
//      NimbusError.cancelled.
//

import Foundation

// MARK: - Run options

public struct PipelineRunOptions: Codable, Sendable, Equatable {
    public var training: SplatTrainingConfig
    public var meshExtraction: MeshExtractionOptions
    public var textureBuild: TextureBuildOptions
    public var hdriAssembly: HDRIAssemblyOptions
    public var export: ExportOptions
    /// When true (default), optional stages that throw NimbusError.notImplemented
    /// are marked skipped and the run continues. When false, a stubbed optional
    /// stage fails the run.
    public var continueOnNotImplemented: Bool

    public init(training: SplatTrainingConfig = SplatTrainingConfig(),
                meshExtraction: MeshExtractionOptions = MeshExtractionOptions(),
                textureBuild: TextureBuildOptions = TextureBuildOptions(),
                hdriAssembly: HDRIAssemblyOptions = HDRIAssemblyOptions(),
                export: ExportOptions = ExportOptions(),
                continueOnNotImplemented: Bool = true) {
        self.training = training
        self.meshExtraction = meshExtraction
        self.textureBuild = textureBuild
        self.hdriAssembly = hdriAssembly
        self.export = export
        self.continueOnNotImplemented = continueOnNotImplemented
    }
}

// MARK: - Stage status and events

/// UI-facing status of one pipeline stage during a run.
public enum PipelineStageStatus: Sendable, Equatable {
    case pending
    case running(fraction: Double, message: String, isIndeterminate: Bool)
    case completed
    /// Stage did not run (stubbed dependency, missing input) but the run continued.
    case skipped(reason: String)
    /// Stage errored. For optional stages the run may still have continued.
    case failed(message: String)
}

/// Events the orchestrator emits while a run is in flight.
public enum PipelineEvent: Sendable {
    case stageStatus(PipelineStage, PipelineStageStatus)
    /// Raw per-stage progress plus the weighted 0...1 whole-pipeline fraction.
    case stageProgress(PipelineProgress, overallFraction: Double)
}

// MARK: - Orchestrator

public actor PipelineOrchestrator {

    /// Relative contribution of each stage to overall progress. Sums to 1.0.
    /// capture is 0 because a run starts from an already-finished CaptureBundle.
    public static let stageWeights: [PipelineStage: Double] = [
        .capture: 0.0,
        .splatTraining: 0.55,
        .meshExtraction: 0.12,
        .hdriAssembly: 0.06,
        .delighting: 0.06,
        .materialClassification: 0.02,
        .textureBuild: 0.11,
        .export: 0.08,
    ]

    private let deps: PipelineDependencies
    private var cancelRequested = false

    public init(dependencies: PipelineDependencies) {
        self.deps = dependencies
    }

    /// Requests cancellation. The in-flight run(...) throws NimbusError.cancelled
    /// at the next stage boundary; the trainer is asked to stop immediately.
    public func cancel() async {
        cancelRequested = true
        await deps.splatTrainer.cancelTraining()
    }

    /// Runs the whole pipeline for one capture bundle. Emits PipelineEvents
    /// (from arbitrary threads; hop to MainActor before touching UI state) and
    /// returns the final ExportedAsset, which is also persisted to the library.
    public func run(bundle: CaptureBundle,
                    options: PipelineRunOptions,
                    onEvent: @escaping @Sendable (PipelineEvent) -> Void) async throws -> ExportedAsset {
        cancelRequested = false
        var completedWeight = 0.0

        func weight(_ stage: PipelineStage) -> Double {
            Self.stageWeights[stage] ?? 0
        }

        func report(_ stage: PipelineStage, _ status: PipelineStageStatus) {
            onEvent(.stageStatus(stage, status))
        }

        /// Marks a stage resolved (completed, skipped, or non-fatally failed),
        /// banks its weight, and emits the new overall fraction.
        func resolve(_ stage: PipelineStage, _ status: PipelineStageStatus, message: String) {
            completedWeight = min(1.0, completedWeight + weight(stage))
            onEvent(.stageStatus(stage, status))
            onEvent(.stageProgress(PipelineProgress(stage: stage, fractionCompleted: 1.0, message: message),
                                   overallFraction: completedWeight))
        }

        /// Progress handler for one stage. Captures the weight completed so far
        /// as a constant so the @Sendable closure stays data-race free.
        func handler(for stage: PipelineStage) -> ProgressHandler {
            let base = completedWeight
            let w = weight(stage)
            return { p in
                let clamped = max(0.0, min(1.0, p.fractionCompleted))
                onEvent(.stageStatus(p.stage, .running(fraction: clamped,
                                                       message: p.message,
                                                       isIndeterminate: p.isIndeterminate)))
                onEvent(.stageProgress(p, overallFraction: min(1.0, base + w * clamped)))
            }
        }

        func checkCancelled() throws {
            if cancelRequested || Task.isCancelled { throw NimbusError.cancelled }
        }

        func isCancellation(_ error: Error) -> Bool {
            if error is CancellationError { return true }
            if case NimbusError.cancelled = error { return true }
            return false
        }

        // Stage 0: capture already happened; surface it as done so the UI shows
        // the full chain.
        resolve(.capture, .completed, message: "Capture loaded: \(bundle.frames.count) frames")

        // Stage 1: splat training (required).
        try checkCancelled()
        report(.splatTraining, .running(fraction: 0, message: "Starting splat training", isIndeterminate: true))
        let splat: SplatModel
        do {
            splat = try await deps.splatTrainer.train(bundle,
                                                      config: options.training,
                                                      progress: handler(for: .splatTraining))
        } catch {
            let mapped: Error = isCancellation(error) ? NimbusError.cancelled : error
            report(.splatTraining, .failed(message: mapped.localizedDescription))
            throw mapped
        }
        resolve(.splatTraining, .completed, message: "Trained \(splat.splatCount) splats")

        // Stage 2: mesh extraction (required; export needs a mesh).
        try checkCancelled()
        report(.meshExtraction, .running(fraction: 0, message: "Extracting mesh", isIndeterminate: true))
        let mesh: MeshAsset
        do {
            mesh = try await deps.meshExtractor.extractMesh(from: splat,
                                                            options: options.meshExtraction,
                                                            progress: handler(for: .meshExtraction))
        } catch {
            let mapped: Error = isCancellation(error) ? NimbusError.cancelled : error
            report(.meshExtraction, .failed(message: mapped.localizedDescription))
            throw mapped
        }
        resolve(.meshExtraction, .completed,
                message: "Mesh: \(mesh.triangleCount) triangles, \(mesh.uvs.isEmpty ? "no UVs" : "UV unwrapped")")

        // Stage 3: HDRI assembly (optional). Runs before delighting so the
        // environment can inform light separation.
        try checkCancelled()
        var hdri: HDRIEnvironment?
        if bundle.hdriBrackets.isEmpty {
            resolve(.hdriAssembly, .skipped(reason: "No exposure brackets in this capture"),
                    message: "HDRI skipped: no brackets")
        } else {
            report(.hdriAssembly, .running(fraction: 0, message: "Merging exposure brackets", isIndeterminate: true))
            do {
                hdri = try await deps.hdriCapture.assembleHDRI(from: bundle,
                                                               options: options.hdriAssembly,
                                                               progress: handler(for: .hdriAssembly))
                resolve(.hdriAssembly, .completed, message: "HDRI assembled")
            } catch let error where isCancellation(error) {
                report(.hdriAssembly, .failed(message: "Cancelled"))
                throw NimbusError.cancelled
            } catch NimbusError.notImplemented(let m) where options.continueOnNotImplemented {
                resolve(.hdriAssembly, .skipped(reason: "Stubbed: \(m)"), message: "HDRI skipped (stub)")
            } catch {
                resolve(.hdriAssembly, .failed(message: error.localizedDescription),
                        message: "HDRI failed; continuing without environment")
            }
        }

        // Captured-albedo bake. The contract has no texture-projection stage yet,
        // so the orchestrator produces a clearly-labelled placeholder albedo
        // (mean capture color). See CapturedAlbedoBaker for the honest limits.
        try checkCancelled()
        var capturedAlbedoURL: URL?
        do {
            let workDir = try Self.workDirectory(for: bundle)
            report(.delighting, .running(fraction: 0,
                                         message: "Estimating captured albedo (placeholder average color)",
                                         isIndeterminate: true))
            capturedAlbedoURL = try CapturedAlbedoBaker.bakePlaceholderAlbedo(
                from: bundle,
                resolution: options.textureBuild.textureResolution,
                outputDirectory: workDir)
        } catch let error where isCancellation(error) {
            throw NimbusError.cancelled
        } catch {
            capturedAlbedoURL = nil
        }

        // Stage 4: delighting (optional).
        var workingAlbedoURL = capturedAlbedoURL
        if let albedo = capturedAlbedoURL {
            do {
                workingAlbedoURL = try await deps.delighter.delight(albedoURL: albedo,
                                                                    mesh: mesh,
                                                                    environment: hdri,
                                                                    progress: handler(for: .delighting))
                resolve(.delighting, .completed, message: "Albedo de-lit")
            } catch let error where isCancellation(error) {
                report(.delighting, .failed(message: "Cancelled"))
                throw NimbusError.cancelled
            } catch NimbusError.notImplemented(let m) where options.continueOnNotImplemented {
                resolve(.delighting, .skipped(reason: "Stubbed: \(m)"),
                        message: "Delighting skipped; using captured albedo")
            } catch {
                resolve(.delighting, .failed(message: error.localizedDescription),
                        message: "Delighting failed; using captured albedo")
            }
        } else {
            resolve(.delighting, .skipped(reason: "No captured albedo available"),
                    message: "Delighting skipped: no albedo")
        }

        // Stage 5: material classification (optional).
        try checkCancelled()
        var classification = MaterialClassification(materialClass: .unknown, confidence: 0)
        if let albedo = workingAlbedoURL {
            report(.materialClassification, .running(fraction: 0, message: "Classifying material", isIndeterminate: true))
            do {
                classification = try await deps.materialClassifier.classify(albedoURL: albedo)
                resolve(.materialClassification, .completed,
                        message: "Material: \(classification.materialClass.rawValue)")
            } catch let error where isCancellation(error) {
                report(.materialClassification, .failed(message: "Cancelled"))
                throw NimbusError.cancelled
            } catch NimbusError.notImplemented(let m) where options.continueOnNotImplemented {
                resolve(.materialClassification, .skipped(reason: "Stubbed: \(m)"),
                        message: "Classification skipped; material unknown")
            } catch {
                resolve(.materialClassification, .failed(message: error.localizedDescription),
                        message: "Classification failed; material unknown")
            }
        } else {
            resolve(.materialClassification, .skipped(reason: "No albedo to classify"),
                    message: "Classification skipped")
        }

        // Stage 6: dynamic texture build (optional; falls back to albedo-only).
        try checkCancelled()
        var materials: MaterialSet?
        if let albedo = workingAlbedoURL {
            do {
                materials = try await deps.textureBuilder.buildMaterialSet(delitAlbedoURL: albedo,
                                                                           mesh: mesh,
                                                                           classification: classification,
                                                                           options: options.textureBuild,
                                                                           progress: handler(for: .textureBuild))
                resolve(.textureBuild, .completed, message: "PBR maps built")
            } catch let error where isCancellation(error) {
                report(.textureBuild, .failed(message: "Cancelled"))
                throw NimbusError.cancelled
            } catch NimbusError.notImplemented(let m) where options.continueOnNotImplemented {
                materials = MaterialSet(albedoURL: albedo,
                                        classification: classification,
                                        textureResolution: options.textureBuild.textureResolution)
                resolve(.textureBuild, .skipped(reason: "Stubbed: \(m)"),
                        message: "Texture synthesis skipped; exporting albedo only")
            } catch {
                materials = MaterialSet(albedoURL: albedo,
                                        classification: classification,
                                        textureResolution: options.textureBuild.textureResolution)
                resolve(.textureBuild, .failed(message: error.localizedDescription),
                        message: "Texture synthesis failed; exporting albedo only")
            }
        } else {
            resolve(.textureBuild, .skipped(reason: "No albedo available"),
                    message: "Texture build skipped")
        }

        // Stage 7: export (required). The exporter contract requires non-empty
        // mesh UVs whenever materials are provided.
        try checkCancelled()
        let exportMaterials: MaterialSet? = mesh.uvs.isEmpty ? nil : materials
        report(.export, .running(fraction: 0, message: "Exporting assets", isIndeterminate: true))
        do {
            let asset = try await deps.assetExporter.export(mesh: mesh,
                                                            materials: exportMaterials,
                                                            splat: splat,
                                                            hdri: hdri,
                                                            options: options.export,
                                                            progress: handler(for: .export))
            // Persist the library manifest so the Library tab can list it.
            // Best effort: a manifest write failure must not lose the export.
            try? AssetLibraryStore.save(asset)
            resolve(.export, .completed, message: "Export finished")
            return asset
        } catch {
            let mapped: Error = isCancellation(error) ? NimbusError.cancelled : error
            report(.export, .failed(message: mapped.localizedDescription))
            throw mapped
        }
    }

    /// Documents/Processing/<captureID>/ scratch directory for intermediates.
    public static func workDirectory(for bundle: CaptureBundle) throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory,
                                               in: .userDomainMask,
                                               appropriateFor: nil,
                                               create: true)
        let dir = docs
            .appendingPathComponent("Processing", isDirectory: true)
            .appendingPathComponent(bundle.id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
