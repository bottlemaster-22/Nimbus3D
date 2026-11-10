//
//  MeshModule.swift — module anchor. Owned by the Mesh agent.
//
//  This module implements `MeshExtractor` (Sources/Core/Contracts.swift):
//  splat -> triangle-mesh surface reconstruction, decimation, and UV unwrap.
//  NOTE: robust splat-to-clean-mesh reconstruction has no proven off-the-shelf
//  on-device implementation today; per the honesty contract, anything not real
//  must be a labelled stub throwing NimbusError.notImplemented.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the Mesh module.
public enum MeshModule {
    public static let name = "Mesh"
}
