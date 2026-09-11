//
//  PipelineModule.swift — module anchor. Owned by the Pipeline agent.
//
//  This module orchestrates the full stage chain using ONLY the Core protocols:
//  CaptureBundle -> SplatTrainer -> MeshExtractor -> Delighter ->
//  MaterialClassifier -> DynamicTextureBuilder -> HDRICapture -> AssetExporter,
//  forwarding PipelineProgress to the Process tab UI. It must depend on the
//  protocols, never on concrete module types, so stubs and real
//  implementations are interchangeable.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the Pipeline module.
public enum PipelineModule {
    public static let name = "Pipeline"
}
