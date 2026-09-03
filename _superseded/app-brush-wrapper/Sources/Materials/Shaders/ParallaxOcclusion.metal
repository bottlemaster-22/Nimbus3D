//
//  ParallaxOcclusion.metal
//  Nimbus3D (Materials module)
//
//  Parallax Occlusion Mapping surface shader with a normal-mapping baseline.
//  This is REAL Metal that compiles into default.metallib. It shades a
//  mesh with a Nimbus MaterialSet so the substituted library relief reads as
//  actual depth under a directional light.
//
//  It needs no per-vertex tangents: the tangent frame is reconstructed in the
//  fragment shader from screen-space derivatives (Mikkelsen's cotangent-frame
//  trick), so it works with the contract's MeshAsset (positions/normals/uvs).
//
//  Vertex attributes (MTLVertexDescriptor):
//     attribute(0) float3 position   (object space)
//     attribute(1) float3 normal     (object space)
//     attribute(2) float2 uv
//  Uniforms: constant POMUniforms at buffer(1) (mirror: ParallaxShadingUniforms).
//  Textures: 0 albedo (sRGB), 1 normalGL (linear), 2 height (linear R),
//            3 roughness (linear R). Sampler 0: repeat-addressing, linear filter.
//
//  When uniforms.parallaxEnabled == 0 this degrades to the plain normal-mapping
//  baseline (same shader, POM step skipped) so both paths share one pipeline.
//

#include <metal_stdlib>
using namespace metal;

struct POMUniforms {
    float4x4 modelMatrix;
    float4x4 viewProjectionMatrix;
    float3x3 normalMatrix;          // inverse-transpose of the model upper-left 3x3
    float3   cameraWorldPosition;
    float3   lightDirection;        // world space, unit vector TOWARD the light
    float3   lightColor;
    float3   ambientColor;
    float    heightScale;           // tangent-space parallax displacement scale
    float    uvScale;               // texture tiling (tiles per UV unit)
    int      minSamples;
    int      maxSamples;
    int      parallaxEnabled;       // 0 = normal-mapping baseline, 1 = full POM
    int      selfShadowEnabled;     // 0/1 soft parallax self-shadow along the light
};

struct POMVertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct POMVertexOut {
    float4 clipPosition [[position]];
    float3 worldPosition;
    float3 worldNormal;
    float2 uv;
};

vertex POMVertexOut pom_vertex(POMVertexIn in [[stage_in]],
                               constant POMUniforms& u [[buffer(1)]]) {
    float4 world = u.modelMatrix * float4(in.position, 1.0);
    POMVertexOut out;
    out.worldPosition = world.xyz;
    out.clipPosition  = u.viewProjectionMatrix * world;
    out.worldNormal   = normalize(u.normalMatrix * in.normal);
    out.uv            = in.uv;
    return out;
}

// Cotangent frame (tangent, bitangent, normal) reconstructed from derivatives.
// Columns are T, B, N in world space; transpose maps world -> tangent space.
static float3x3 cotangentFrame(float3 N, float3 p, float2 uv) {
    float3 dp1 = dfdx(p);
    float3 dp2 = dfdy(p);
    float2 duv1 = dfdx(uv);
    float2 duv2 = dfdy(uv);

    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    float invmax = rsqrt(max(dot(T, T), dot(B, B)));
    return float3x3(T * invmax, B * invmax, N);
}

// Height convention: sampled value 1.0 = top surface, 0.0 = deepest.
// We march "into" the surface, so depth = 1 - height.
static float2 parallaxOcclusionUV(texture2d<float> heightMap,
                                   sampler s,
                                   float2 uv,
                                   float3 viewTS,
                                   int minSamples,
                                   int maxSamples,
                                   float heightScale) {
    float numLayers = mix(float(maxSamples), float(minSamples), clamp(abs(viewTS.z), 0.0, 1.0));
    numLayers = max(numLayers, 1.0);
    float layerDepth = 1.0 / numLayers;

    // Total UV shift from top to bottom of the height field.
    float2 P = (viewTS.xy / max(viewTS.z, 0.01)) * heightScale;
    float2 deltaUV = P / numLayers;

    float currentLayerDepth = 0.0;
    float2 currentUV = uv;
    float currentDepth = 1.0 - heightMap.sample(s, currentUV).r;

    // Dynamic bound; guard against very large maxSamples with a hard cap.
    int hardCap = min(maxSamples, 256);
    for (int i = 0; i < hardCap; ++i) {
        if (currentLayerDepth >= currentDepth) break;
        currentUV -= deltaUV;
        currentDepth = 1.0 - heightMap.sample(s, currentUV).r;
        currentLayerDepth += layerDepth;
    }

    // Interpolate between the last two layers for a smooth intersection.
    float2 prevUV = currentUV + deltaUV;
    float afterDepth  = currentDepth - currentLayerDepth;
    float beforeDepth = (1.0 - heightMap.sample(s, prevUV).r) - currentLayerDepth + layerDepth;
    float denom = afterDepth - beforeDepth;
    float weight = denom != 0.0 ? afterDepth / denom : 0.0;
    return mix(currentUV, prevUV, clamp(weight, 0.0, 1.0));
}

// Soft self-shadow: march from the shaded point toward the light and count
// how many steps are occluded by the height field.
static float parallaxSelfShadow(texture2d<float> heightMap,
                                sampler s,
                                float2 uv,
                                float3 lightTS,
                                int samples,
                                float heightScale,
                                float surfaceDepth) {
    if (lightTS.z <= 0.0) return 1.0;
    int n = clamp(samples, 1, 64);
    float layerDepth = surfaceDepth / float(n);
    float2 delta = (lightTS.xy / max(lightTS.z, 0.01)) * heightScale / float(n);

    float2 curUV = uv;
    float curDepth = surfaceDepth;
    float occluded = 0.0;
    for (int i = 0; i < n; ++i) {
        curUV += delta;
        curDepth -= layerDepth;
        float sampledDepth = 1.0 - heightMap.sample(s, curUV).r;
        if (sampledDepth < curDepth) occluded += 1.0;
    }
    return 1.0 - clamp(occluded / float(n), 0.0, 1.0);
}

fragment float4 pom_fragment(POMVertexOut in [[stage_in]],
                             constant POMUniforms& u [[buffer(1)]],
                             texture2d<float> albedoMap    [[texture(0)]],
                             texture2d<float> normalMap    [[texture(1)]],
                             texture2d<float> heightMap    [[texture(2)]],
                             texture2d<float> roughnessMap [[texture(3)]],
                             sampler samp [[sampler(0)]]) {
    float3 N = normalize(in.worldNormal);
    float3 V = normalize(u.cameraWorldPosition - in.worldPosition);

    float2 baseUV = in.uv * u.uvScale;
    float3x3 TBN = cotangentFrame(N, in.worldPosition, baseUV);
    float3x3 worldToTangent = transpose(TBN);

    float3 viewTS = normalize(worldToTangent * V);

    float2 uv = baseUV;
    float surfaceDepth = 0.0;
    if (u.parallaxEnabled != 0) {
        uv = parallaxOcclusionUV(heightMap, samp, baseUV, viewTS,
                                 u.minSamples, u.maxSamples, u.heightScale);
        surfaceDepth = 1.0 - heightMap.sample(samp, uv).r;
    }

    float3 albedo = albedoMap.sample(samp, uv).rgb;

    float3 nTS = normalMap.sample(samp, uv).xyz * 2.0 - 1.0;
    nTS = normalize(nTS);
    float3 Nw = normalize(TBN * nTS);

    float3 L = normalize(u.lightDirection);
    float ndotl = max(dot(Nw, L), 0.0);

    float shadow = 1.0;
    if (u.parallaxEnabled != 0 && u.selfShadowEnabled != 0) {
        float3 lightTS = normalize(worldToTangent * L);
        shadow = parallaxSelfShadow(heightMap, samp, uv, lightTS,
                                    u.minSamples, u.heightScale, surfaceDepth);
    }

    // Simple energy-preserving-ish diffuse preview. Roughness is sampled so the
    // binding is meaningful even though this preview does not do full specular.
    float roughness = roughnessMap.sample(samp, uv).r;
    float diffuseScale = mix(1.0, 0.85, roughness);

    float3 lit = albedo * (u.ambientColor + u.lightColor * ndotl * shadow * diffuseScale);
    return float4(lit, 1.0);
}
