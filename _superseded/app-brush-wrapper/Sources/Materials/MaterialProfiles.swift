//
//  MaterialProfiles.swift
//  Nimbus3D (Materials module)
//
//  Per-class surface profiles: parallax-occlusion parameters, normal-from-height
//  strength, and fallback roughness/metallic constants used only when the bundled
//  library entry does not ship that map. These are hand-tuned defaults, all real
//  and deterministic; they encode "what relief a masonry / wood / metal surface
//  should read like", they do NOT measure relief from the capture.
//

import Foundation

/// Surface-shading defaults for one coarse material class.
struct MaterialSurfaceProfile {
    /// Parallax-occlusion-mapping parameters written into the MaterialSet.
    var parallax: ParallaxOcclusionParameters
    /// Grayscale byte used for a flat roughness map when the library entry has none.
    var fallbackRoughness: UInt8
    /// Grayscale byte used for a flat metallic map when the library entry has none.
    var fallbackMetallic: UInt8
    /// Sobel strength for the normal-from-height baseline (used only when the
    /// library entry ships no ready-made normal map).
    var normalStrength: Float

    /// Profile for a coarse class. `nil` (no library substitute available:
    /// plastic / glass / unknown) yields a flat, POM-disabled profile.
    static func profile(for coarse: CoarseMaterial?) -> MaterialSurfaceProfile {
        switch coarse {
        case .masonry:
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: true, heightScale: 0.020, minSamples: 8, maxSamples: 32),
                fallbackRoughness: 217, fallbackMetallic: 0, normalStrength: 3.0)
        case .fabric:
            // Fine weave: normal mapping reads better than POM, and POM ghosts at
            // grazing angles on soft cloth. Parallax disabled on purpose.
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: false, heightScale: 0.006, minSamples: 8, maxSamples: 16),
                fallbackRoughness: 235, fallbackMetallic: 0, normalStrength: 2.0)
        case .granular:
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: true, heightScale: 0.015, minSamples: 8, maxSamples: 48),
                fallbackRoughness: 230, fallbackMetallic: 0, normalStrength: 3.5)
        case .wood:
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: true, heightScale: 0.008, minSamples: 8, maxSamples: 24),
                fallbackRoughness: 153, fallbackMetallic: 0, normalStrength: 2.0)
        case .metal:
            // Mostly flat; POM overkill and it exaggerates panel seams. Normal only.
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: false, heightScale: 0.004, minSamples: 8, maxSamples: 16),
                fallbackRoughness: 89, fallbackMetallic: 255, normalStrength: 1.5)
        case .tile:
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: true, heightScale: 0.010, minSamples: 8, maxSamples: 32),
                fallbackRoughness: 77, fallbackMetallic: 0, normalStrength: 2.5)
        case nil:
            // No sensible library substitute: keep it flat and honest.
            return MaterialSurfaceProfile(
                parallax: ParallaxOcclusionParameters(enabled: false, heightScale: 0.0, minSamples: 8, maxSamples: 16),
                fallbackRoughness: 204, fallbackMetallic: 0, normalStrength: 2.0)
        }
    }
}
