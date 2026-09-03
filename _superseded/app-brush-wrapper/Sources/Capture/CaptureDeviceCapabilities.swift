//
//  CaptureDeviceCapabilities.swift — ARKit support probing + session configuration.
//  Owned by the Capture agent.
//
//  Answers "can this device do world tracking / LiDAR depth?" and builds the
//  ARWorldTrackingConfiguration that ARCaptureService runs. Runtime-guards every
//  LiDAR-only feature so the app still works on non-Pro iPhones (RGB-only capture).
//

import ARKit

enum CaptureDeviceCapabilities {

    /// World tracking is required for any capture. False on the simulator and very old devices.
    static var isWorldTrackingSupported: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    /// True on LiDAR-equipped devices (iPhone Pro / Pro Max, iPad Pro). Enables the depth prior.
    static var supportsLiDARDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    /// Temporally smoothed LiDAR depth, when available. Slightly cleaner than raw sceneDepth.
    static var supportsSmoothedDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
    }

    /// Builds the world-tracking configuration for a capture session based on the requested options.
    static func makeConfiguration(options: CaptureOptions) -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        // Gravity alignment gives a stable up-axis so coverage/azimuth math is meaningful.
        config.worldAlignment = .gravity
        config.isAutoFocusEnabled = true
        // We are scanning a single object, not building a scene graph. No plane/anchor overhead.
        config.planeDetection = []
        config.environmentTexturing = .none

        if options.captureLiDARDepth && supportsLiDARDepth {
            config.frameSemantics.insert(.sceneDepth)
            if supportsSmoothedDepth {
                config.frameSemantics.insert(.smoothedSceneDepth)
            }
        }

        // Prefer the highest-resolution streaming video format the device offers, so saved
        // frames carry as much detail as possible for splat training.
        if let hiRes = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = hiRes
        }

        return config
    }
}
