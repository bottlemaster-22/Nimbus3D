//
//  MaterialDelighter.swift
//  Nimbus3D (Materials module)
//
//  HONEST STUB for delighting. Removing baked-in lighting from a captured albedo
//  so it can be relit under any environment has no proven, self-contained
//  on-device implementation today, so this does NOT actually de-light anything.
//
//  It ships as a model SLOT + a labelled passthrough:
//   - If a compiled Core ML delighting model (NimbusDelighter.mlmodelc) is present
//     in the app bundle, `hasModel` is true and a real inference path can be added
//     where the TODO marks it (kept as a stub for now: even with a model, the
//     tiling / tensor I/O still has to be written, so we do not pretend).
//   - With no model (the shipping case), it copies the captured albedo to a new
//     file whose name says plainly that it is NOT de-lit, and reports that through
//     `progress`. Downstream stages still get a usable albedo so the REAL material
//     pipeline (library substitution, tinting, POM) can run end to end; nothing is
//     claimed to have had its lighting removed.
//
//  This keeps the pipeline runnable while being explicit about the gap. Set
//  `throwWhenNoModel: true` if a caller would rather fail loudly than pass through.
//
//  TODO(nimbus): a real on-device delighter needs a trained intrinsic-image /
//  albedo-estimation model (e.g. a small U-Net predicting reflectance from a lit
//  crop, optionally conditioned on the capture-time HDRI's dominant light
//  direction from `environment`), exported with coremltools as an image->image
//  model named NimbusDelighter.mlmodelc. The inference path would tile the UV
//  albedo, run the model per tile, blend seams, and write the reflectance result.
//

import Foundation
import CoreML

public final class MaterialDelighter: Delighter, @unchecked Sendable {

    /// Compiled Core ML model name expected in the app bundle for a real delight pass.
    public static let bundledModelName = "NimbusDelighter"

    /// Loaded delighting model, or nil when none is bundled (the shipping case).
    private let model: MLModel?

    /// When true and no model is bundled, `delight` throws instead of passing through.
    private let throwWhenNoModel: Bool

    /// True when a real Core ML delighting model was found and loaded.
    public var hasModel: Bool { model != nil }

    public init(bundle: Bundle = .main, throwWhenNoModel: Bool = false) {
        self.throwWhenNoModel = throwWhenNoModel
        if let modelURL = bundle.url(forResource: Self.bundledModelName, withExtension: "mlmodelc") {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            self.model = try? MLModel(contentsOf: modelURL, configuration: configuration)
        } else {
            self.model = nil
        }
    }

    public func delight(albedoURL: URL,
                        mesh: MeshAsset,
                        environment: HDRIEnvironment?,
                        progress: @escaping ProgressHandler) async throws -> URL {
        progress(PipelineProgress(stage: .delighting,
                                  fractionCompleted: 0.0,
                                  message: "Preparing albedo for delighting",
                                  isIndeterminate: true))

        if model != nil {
            // A model is present, but the real inference path is not written yet.
            // Do NOT fake it: fail honestly rather than return an un-de-lit result
            // dressed up as de-lit.
            throw NimbusError.notImplemented(
                "A NimbusDelighter model is bundled, but the on-device delighting inference path is not implemented yet. See TODO(nimbus) in MaterialDelighter.swift.")
        }

        guard !throwWhenNoModel else {
            throw NimbusError.notImplemented(
                "No delighting model is available. Delighting has no on-device implementation yet (Materials module). Bundle NimbusDelighter.mlmodelc or run with passthrough to proceed with the still-lit albedo.")
        }

        // Passthrough: copy the captured albedo unchanged. The filename and the
        // progress message both state that lighting was NOT removed.
        progress(PipelineProgress(stage: .delighting,
                                  fractionCompleted: 0.5,
                                  message: "Delighting model unavailable: passing captured albedo through UNCHANGED (baked lighting is NOT removed)"))

        let outputURL = albedoURL
            .deletingLastPathComponent()
            .appendingPathComponent("albedo_NOT_delit_passthrough.png")

        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try? fm.removeItem(at: outputURL)
        }
        do {
            try fm.copyItem(at: albedoURL, to: outputURL)
        } catch {
            throw NimbusError.delightingFailed("Could not copy albedo for passthrough: \(error.localizedDescription)")
        }

        progress(PipelineProgress(stage: .delighting,
                                  fractionCompleted: 1.0,
                                  message: "Delighting skipped (no model); albedo passed through unchanged"))
        return outputURL
    }
}
