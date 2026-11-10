//
//  MaterialsModule.swift — module anchor. Owned by the Materials agent.
//
//  This module implements `Delighter`, `MaterialClassifier`, and
//  `DynamicTextureBuilder` (Sources/Core/Contracts.swift). The classifier is
//  Core ML; texture map synthesis can be image-processing (Metal/vImage) where
//  real, and neural pieces (delighting, neural material synthesis) must be
//  labelled stubs throwing NimbusError.notImplemented until a real on-device
//  model exists. Do not edit Core contracts.
//

/// Namespace marker for the Materials module.
public enum MaterialsModule {
    public static let name = "Materials"

    /// Registers this module's implementations into the Pipeline's dependency
    /// container. Call once at app launch (main thread) so the Process screen
    /// drives the real classifier / texture builder instead of the unwired stubs.
    ///
    /// - `Delighter` is `MaterialDelighter` (honest stub: no on-device delight
    ///   model ships, so it passes the captured albedo through unchanged and says
    ///   so). Pass `delighterThrowsWithoutModel: true` to fail loudly instead.
    /// - `MaterialClassifier` is `CoreMLMaterialClassifier` (real Vision path;
    ///   returns .unknown at zero confidence until a model is bundled, which the
    ///   MaterialConfirmationView turns into a required user choice).
    /// - `DynamicTextureBuilder` is `LibraryDynamicTextureBuilder` (real library
    ///   substitution + capture tinting + POM parameters).
    @MainActor
    public static func register(bundle: Bundle = .main,
                                delighterThrowsWithoutModel: Bool = false) {
        PipelineServices.shared.delighter =
            MaterialDelighter(bundle: bundle, throwWhenNoModel: delighterThrowsWithoutModel)
        PipelineServices.shared.materialClassifier =
            CoreMLMaterialClassifier(bundle: bundle)
        PipelineServices.shared.textureBuilder =
            LibraryDynamicTextureBuilder(bundle: bundle)
    }
}
