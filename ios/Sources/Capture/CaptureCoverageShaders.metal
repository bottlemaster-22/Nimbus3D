//
//  CaptureCoverageShaders.metal
//  Capture
//
//  THE CAPTURE HUD'S GPU SIDE: the camera feed, and the ARKit classified mesh
//  painted with the three coverage channels (F9).
//
//  Every symbol in this file is prefixed `capture_`. The whole app is ONE
//  Xcode target, so every .metal file under Sources/ compiles into ONE
//  default.metallib and two functions with the same name anywhere in the
//  project is a link error. Sources/Viewer owns `viewer_*`, Sources/Trainer
//  owns `trainer_*`; this file owns `capture_*` and nothing else.
//
//  WHY THE COVERAGE IS A VERTEX ATTRIBUTE AND NOT A TEXTURE. The thing being
//  painted is ARKit's scene mesh, which has no UV parameterisation - there is
//  no texture to paint into, and building one every time the mesh re-chunks
//  would be a per-frame atlas allocation for a HUD. The coverage field is a
//  sparse world voxel grid (see CaptureCoverageField.swift); the CPU samples
//  it once per vertex per refresh and hands the result over as a plain
//  per-vertex float4. Interpolation across the triangle then does exactly what
//  is wanted: a smooth gradient from a covered patch to an uncovered one,
//  which reads as "keep going in that direction" rather than as a checkerboard
//  of voxel squares.
//
//  THE THREE CHANNELS AND WHY THEY ARE NOT AVERAGED. `coverage.xyz` is
//  (angles seen, best distance, best sharpness), each 0...1, each with its own
//  remedy: walk around it / walk toward it / slow down. A single blended score
//  would tell the user a patch is bad without telling them what to do, which
//  is the failure mode of every coverage HUD that shows one colour ramp.
//
//  `coverage.w` packs two flags, because a fourth float channel would
//  otherwise be spent on a boolean:
//      bit 0 (+1.0)  this patch has been observed at all
//      bit 1 (+2.0)  this patch is optically unreliable (window / glass / sky)
//  An unreliable patch is drawn calm rather than angry: you cannot walk around
//  a window, and nagging someone to do the impossible is how a guidance UI
//  loses their trust.
//

#include <metal_stdlib>
using namespace metal;

// =============================================================================
//  MARK: - Shared layouts
//
//  Byte-matched to CaptureCoverageRenderer.swift. Both structs are built from
//  float4x4 / float4 / float3x3 only: those have identical size and alignment
//  in Metal and in simd, whereas a packed_float3 next to a float does not, and
//  that mismatch is the classic way a uniform buffer silently shifts by four
//  bytes and the whole overlay ends up somewhere else.
// =============================================================================

struct CaptureCoverageUniforms {
    float4x4 viewProjection;
    float4   cameraPosition;   // xyz world position, w unused
    // x: overlay opacity 0...1
    // y: the per-channel "done" threshold (CaptureTuning.coverageChannelDoneThreshold)
    // z: channel to isolate, -1 for all three at once
    // w: unused
    float4   params;
    // Build 450: the saturation view samples the camera image under each
    // mesh fragment: drawable size in pixels (xy) and the same view-UV ->
    // image-UV transform the background uses.
    float4   screen;
    float3x3 textureTransform;
};

struct CaptureBackgroundUniforms {
    // Maps a view-space UV in 0...1 (origin top-left) to the camera image's
    // own UV. It is the INVERSE of ARFrame.displayTransform, computed on the
    // CPU where CGAffineTransform can do it exactly.
    float3x3 textureTransform;
};

// =============================================================================
//  MARK: - Camera background
//
//  A full-screen triangle rather than a quad: three vertices instead of six,
//  no diagonal seam, and the rasteriser clips the overspill for free.
// =============================================================================

struct CaptureBackgroundVaryings {
    float4 position [[position]];
    float2 uv;
};

vertex CaptureBackgroundVaryings capture_background_vertex(
    uint vertexID [[vertex_id]],
    constant CaptureBackgroundUniforms &uniforms [[buffer(0)]]
) {
    // Clip-space corners of a triangle that covers the whole viewport.
    const float2 positions[3] = {
        float2(-1.0, -3.0),
        float2(-1.0,  1.0),
        float2( 3.0,  1.0),
    };
    // Matching UVs with the origin at the TOP-LEFT of the view, which is the
    // space ARFrame.displayTransform is defined in.
    const float2 uvs[3] = {
        float2(0.0, 2.0),
        float2(0.0, 0.0),
        float2(2.0, 0.0),
    };

    CaptureBackgroundVaryings out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    float3 uv = uniforms.textureTransform * float3(uvs[vertexID], 1.0);
    out.uv = uv.xy / uv.z;
    return out;
}

fragment float4 capture_background_fragment(
    CaptureBackgroundVaryings in [[stage_in]],
    texture2d<float, access::sample> lumaTexture   [[texture(0)]],
    texture2d<float, access::sample> chromaTexture [[texture(1)]]
) {
    constexpr sampler linearSampler(
        filter::linear,
        mip_filter::none,
        address::clamp_to_edge
    );

    float  y  = lumaTexture.sample(linearSampler, in.uv).r;
    float2 cbcr = chromaTexture.sample(linearSampler, in.uv).rg;

    // ARKit delivers full-range 4:2:0 YCbCr. This is the standard BT.601 full
    // range matrix, written out rather than pulled from a constant so the one
    // place it lives is visible next to the sampling that uses it.
    const float4x4 ycbcrToRGB = float4x4(
        float4( 1.0000,  1.0000,  1.0000, 0.0000),
        float4( 0.0000, -0.3441,  1.7720, 0.0000),
        float4( 1.4020, -0.7141,  0.0000, 0.0000),
        float4(-0.7010,  0.5291, -0.8860, 1.0000)
    );

    float4 rgb = ycbcrToRGB * float4(y, cbcr.x, cbcr.y, 1.0);
    // Build 450: the world starts grey; scanned surfaces regain their colour
    // in the coverage pass drawn over this.
    float grey = dot(rgb.rgb, float3(0.299, 0.587, 0.114));
    return float4(float3(grey * 0.9), 1.0);
}

// =============================================================================
//  MARK: - Coverage mesh
// =============================================================================

struct CaptureCoverageVaryings {
    float4 position [[position]];
    float3 worldPosition;
    float3 worldNormal;
    float4 coverage;
};

vertex CaptureCoverageVaryings capture_coverage_vertex(
    uint vertexID [[vertex_id]],
    // float3 (16-byte stride), NOT packed_float3 (12). Swift's SIMD3<Float>
    // has a stride of 16, so a packed pointer here would walk the array four
    // bytes short per vertex and shear the whole mesh - a bug that looks like
    // a pose problem and is not one.
    device const float3 *positions [[buffer(0)]],
    device const float3 *normals   [[buffer(1)]],
    device const float4 *coverage  [[buffer(2)]],
    constant CaptureCoverageUniforms &uniforms [[buffer(3)]]
) {
    float3 world = positions[vertexID];

    CaptureCoverageVaryings out;
    out.position = uniforms.viewProjection * float4(world, 1.0);
    out.worldPosition = world;
    out.worldNormal = normals[vertexID];
    out.coverage = coverage[vertexID];
    return out;
}

// Colours. Each deficient channel owns one, and each colour appears nowhere
// else in the capture UI, so a colour is a fix instruction rather than a mood.
constant float3 kCaptureColorSatisfied  = float3(0.16, 0.84, 0.40);  // green
constant float3 kCaptureColorAngles     = float3(0.68, 0.36, 1.00);  // violet
constant float3 kCaptureColorDistance   = float3(0.24, 0.60, 1.00);  // blue
constant float3 kCaptureColorSharpness  = float3(1.00, 0.70, 0.16);  // amber
constant float3 kCaptureColorUnseen     = float3(0.62, 0.62, 0.66);  // grey
constant float3 kCaptureColorUnreliable = float3(0.35, 0.78, 0.85);  // teal

fragment float4 capture_coverage_fragment(
    CaptureCoverageVaryings in [[stage_in]],
    constant CaptureCoverageUniforms &uniforms [[buffer(0)]],
    texture2d<float, access::sample> lumaTexture   [[texture(0)]],
    texture2d<float, access::sample> chromaTexture [[texture(1)]]
) {
    const float opacity   = uniforms.params.x;
    const float threshold = uniforms.params.y;
    const int   isolate   = int(uniforms.params.z);

    const float flags = in.coverage.w;
    const bool seen        = flags >= 0.5;
    const bool unreliable  = flags >= 1.5;

    // Fade faces seen edge-on. A triangle whose normal is perpendicular to the
    // eye covers a lot of screen for very little surface, and letting those
    // dominate makes the overlay look like a smear of colour across the room.
    float3 toEye = normalize(uniforms.cameraPosition.xyz - in.worldPosition);
    float3 normal = normalize(in.worldNormal);
    float facing = clamp(abs(dot(normal, toEye)), 0.0, 1.0);
    float facingFade = mix(0.25, 1.0, facing);

    // BUILD 450: THE SATURATION VIEW (the default; the coloured channel view
    // below is kept for when one channel is isolated). The camera image is
    // sampled under this fragment and drawn from grey towards full colour by
    // how complete this patch of surface is: the weakest of angles, distance
    // and sharpness against the done threshold. Unseen surface stays grey;
    // glass and other unreliable surface tops out at half colour.
    if (isolate < 0) {
        constexpr sampler cameraSampler(filter::linear, address::clamp_to_edge);
        const float2 viewUV = in.position.xy / max(uniforms.screen.xy, float2(1.0));
        const float3 mapped = uniforms.textureTransform * float3(viewUV, 1.0);
        const float2 uv = mapped.xy / mapped.z;
        const float y = lumaTexture.sample(cameraSampler, uv).r;
        const float2 cbcr = chromaTexture.sample(cameraSampler, uv).rg;
        const float3 rgb = clamp(float3(
            y + 1.4020 * (cbcr.y - 0.5),
            y - 0.3441 * (cbcr.x - 0.5) - 0.7141 * (cbcr.y - 0.5),
            y + 1.7720 * (cbcr.x - 0.5)
        ), 0.0, 1.0);
        const float grey = dot(rgb, float3(0.299, 0.587, 0.114));
        float done = 0.0;
        if (seen) {
            const float3 c = clamp(in.coverage.xyz / max(threshold, 1e-4), 0.0, 1.0);
            done = min(c.x, min(c.y, c.z));
            if (unreliable) { done = min(done, 0.5); }
        }
        const float saturation = smoothstep(0.0, 1.0, done);
        const float3 shade = mix(float3(grey * 0.9), rgb, saturation);
        return float4(shade, 1.0);
    }

    if (!seen) {
        return float4(kCaptureColorUnseen, opacity * 0.22 * facingFade);
    }

    float3 channels = clamp(in.coverage.xyz, 0.0, 1.0);

    // Isolate mode: one channel, as a ramp from its own colour at 0 to green
    // at the threshold. Used by the HUD's channel picker so a user can ask
    // "show me just the angles" and get an answer, not a blend.
    if (isolate >= 0 && isolate <= 2) {
        float value = (isolate == 0) ? channels.x
                    : (isolate == 1) ? channels.y
                                     : channels.z;
        float3 deficientColor = (isolate == 0) ? kCaptureColorAngles
                              : (isolate == 1) ? kCaptureColorDistance
                                               : kCaptureColorSharpness;
        float t = clamp(value / max(threshold, 1e-4), 0.0, 1.0);
        float3 rgb = mix(deficientColor, kCaptureColorSatisfied, t);
        float alpha = opacity * mix(0.85, 0.28, t) * facingFade;
        return float4(rgb, alpha);
    }

    if (unreliable) {
        // A window cannot be walked around and cannot be measured. It is drawn
        // in its own calm colour so the user can see the app KNOWS what it is
        // looking at, without being told to fix something that has no fix.
        return float4(kCaptureColorUnreliable, opacity * 0.35 * facingFade);
    }

    float3 deficit = max(float3(threshold) - channels, float3(0.0));
    float worst = max(deficit.x, max(deficit.y, deficit.z));

    if (worst <= 0.0) {
        return float4(kCaptureColorSatisfied, opacity * 0.28 * facingFade);
    }

    float3 rgb = (deficit.x >= worst) ? kCaptureColorAngles
               : (deficit.y >= worst) ? kCaptureColorDistance
                                      : kCaptureColorSharpness;

    // Alpha grows with how far short the worst channel is, so the eye is drawn
    // to the patch that needs the most work rather than to the nearest one.
    float severity = clamp(worst / max(threshold, 1e-4), 0.0, 1.0);
    float alpha = opacity * mix(0.30, 0.88, severity) * facingFade;
    return float4(rgb, alpha);
}
