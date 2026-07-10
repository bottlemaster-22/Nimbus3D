//
//  CaptureModule.swift — module anchor. Owned by the Capture agent.
//
//  This module implements `CaptureService` (Sources/Core/Contracts.swift) using
//  ARKit world tracking + AVFoundation, writing CaptureBundle data to disk,
//  including optional LiDAR depth and bracketed-exposure HDRI frames.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the Capture module.
public enum CaptureModule {
    public static let name = "Capture"
}
