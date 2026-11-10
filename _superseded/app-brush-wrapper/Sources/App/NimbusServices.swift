//
//  NimbusServices.swift
//  Nimbus3D - App shell (integration wiring)
//
//  Single dependency-injection point for the whole app. Each module ships a
//  concrete implementation of a Core protocol (Sources/Core/Contracts.swift);
//  the Pipeline orchestrator depends ONLY on those protocols and reads whatever
//  is registered in `PipelineServices.shared` when a run starts. This file wires
//  the real implementations in at launch so the Process screen drives the actual
//  trainer / mesher / material / HDRI / export code instead of the honest
//  "not wired" stubs that `PipelineDependencies.unwired` installs by default.
//
//  Honesty note: registering a service here does NOT imply the service is fully
//  real. Several are honest PARTIAL/STUB implementations behind the shared
//  protocol (delighting has no model, splat->clean-mesh is a density-envelope
//  approximation, PLY->SPZ is deferred). They report their own limits through
//  progress messages / NimbusError.notImplemented; the orchestrator degrades
//  gracefully. See MODULE_STATUS.md for the per-module truth.
//

import Foundation

@MainActor
enum NimbusServices {

    /// Guards against re-registering if the app shell calls this more than once.
    private static var didRegister = false

    /// Wires every module's concrete service into the Pipeline dependency
    /// container. Call once, on the main thread, at app launch and before any
    /// pipeline run. Idempotent.
    static func registerAll() {
        guard !didRegister else { return }
        didRegister = true

        // SplatEngine (Rust/Brush via NimbusSplatCore.xcframework). Constructs
        // the Rust trainer handle now; actual training happens per run.
        PipelineServices.shared.splatTrainer = BrushSplatTrainer()

        // Mesh (splat -> triangle mesh, on-device Metal/CPU).
        PipelineServices.shared.meshExtractor = SplatMeshExtractor()

        // HDRI (bracket merge -> equirectangular .exr).
        PipelineServices.shared.hdriCapture = HDRIAssembler()

        // Export (glTF/GLB + PBR + .exr + splat).
        PipelineServices.shared.assetExporter = ExportModule.makeExporter()

        // Materials wires all three of its services (Delighter, MaterialClassifier,
        // DynamicTextureBuilder) itself. Delighter is an honest passthrough when no
        // Core ML model is bundled (the shipping case): it does not fail the run,
        // it reports that lighting was NOT removed.
        MaterialsModule.register()

        // CaptureService and SplatRenderer are intentionally NOT registered here:
        // capture runs in the Capture tab before a pipeline run, and preview
        // rendering is driven by the SplatRender render loop, not the orchestrator.
    }
}
