//
//  ParallaxShading.swift
//  Nimbus3D (Materials module)
//
//  Swift mirror of the `POMUniforms` struct in ParallaxOcclusion.metal, plus a
//  factory that fills it from a MaterialSet's parallax parameters and a material
//  class. A preview/export renderer (SplatRender or Export) can bind this at
//  buffer(1) of the pom_vertex / pom_fragment pipeline.
//
//  Field order and types match the Metal struct exactly. `simd` types carry the
//  same alignment as their MSL counterparts (float3 / float3x3 are 16-byte
//  aligned on both sides), so this struct is layout-compatible when passed as a
//  Metal constant buffer.
//

import Foundation
import simd

/// CPU-side mirror of `POMUniforms` (ParallaxOcclusion.metal). Bind at buffer(1).
public struct ParallaxShadingUniforms {
    public var modelMatrix: simd_float4x4
    public var viewProjectionMatrix: simd_float4x4
    public var normalMatrix: simd_float3x3
    public var cameraWorldPosition: SIMD3<Float>
    public var lightDirection: SIMD3<Float>
    public var lightColor: SIMD3<Float>
    public var ambientColor: SIMD3<Float>
    public var heightScale: Float
    public var uvScale: Float
    public var minSamples: Int32
    public var maxSamples: Int32
    public var parallaxEnabled: Int32
    public var selfShadowEnabled: Int32

    public init(modelMatrix: simd_float4x4,
                viewProjectionMatrix: simd_float4x4,
                normalMatrix: simd_float3x3,
                cameraWorldPosition: SIMD3<Float>,
                lightDirection: SIMD3<Float>,
                lightColor: SIMD3<Float> = SIMD3<Float>(repeating: 1),
                ambientColor: SIMD3<Float> = SIMD3<Float>(repeating: 0.08),
                heightScale: Float,
                uvScale: Float = 1,
                minSamples: Int32 = 8,
                maxSamples: Int32 = 32,
                parallaxEnabled: Bool = true,
                selfShadowEnabled: Bool = false) {
        self.modelMatrix = modelMatrix
        self.viewProjectionMatrix = viewProjectionMatrix
        self.normalMatrix = normalMatrix
        self.cameraWorldPosition = cameraWorldPosition
        self.lightDirection = lightDirection
        self.lightColor = lightColor
        self.ambientColor = ambientColor
        self.heightScale = heightScale
        self.uvScale = uvScale
        self.minSamples = minSamples
        self.maxSamples = maxSamples
        self.parallaxEnabled = parallaxEnabled ? 1 : 0
        self.selfShadowEnabled = selfShadowEnabled ? 1 : 0
    }

    /// Builds uniforms from a MaterialSet's parallax parameters and the transforms
    /// a caller already has. `uvScale` should reflect real-world tiling: pass
    /// (surface span in meters / library material physicalSizeMeters).
    public static func make(materialSet: MaterialSet,
                            modelMatrix: simd_float4x4,
                            viewProjectionMatrix: simd_float4x4,
                            normalMatrix: simd_float3x3,
                            cameraWorldPosition: SIMD3<Float>,
                            lightDirection: SIMD3<Float>,
                            uvScale: Float = 1,
                            selfShadowEnabled: Bool = false) -> ParallaxShadingUniforms {
        let p = materialSet.parallax
        return ParallaxShadingUniforms(modelMatrix: modelMatrix,
                                       viewProjectionMatrix: viewProjectionMatrix,
                                       normalMatrix: normalMatrix,
                                       cameraWorldPosition: cameraWorldPosition,
                                       lightDirection: lightDirection,
                                       heightScale: p.heightScale,
                                       uvScale: uvScale,
                                       minSamples: Int32(p.minSamples),
                                       maxSamples: Int32(p.maxSamples),
                                       parallaxEnabled: p.enabled,
                                       selfShadowEnabled: selfShadowEnabled)
    }
}
