//
//  HDRIModule.swift — module anchor. Owned by the HDRI agent.
//
//  This module implements `HDRICapture` (Sources/Core/Contracts.swift):
//  merging pose-tagged bracketed exposures (CaptureBundle.hdriBrackets) into
//  an equirectangular OpenEXR environment map.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the HDRI module.
public enum HDRIModule {
    public static let name = "HDRI"
}
