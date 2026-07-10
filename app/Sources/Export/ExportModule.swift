//
//  ExportModule.swift — module anchor. Owned by the Export agent.
//
//  This module implements `AssetExporter` (Sources/Core/Contracts.swift):
//  writing glTF/.glb (mesh + PBR MaterialSet), copying the .exr HDRI, and
//  converting/copying the .spz splat into an ExportedAsset directory. It also
//  owns the Library tab UI listing exported assets.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the Export module.
public enum ExportModule {
    public static let name = "Export"

    /// The module's `AssetExporter` implementation. Pipeline wires this in as the
    /// final stage; the App's Library tab uses `ExportLibraryView` (same module).
    public static func makeExporter() -> AssetExporter {
        NimbusAssetExporter()
    }
}
