//
//  SplatRasterize.metal — SplatRender module.
//
//  REAL EWA (Elliptical Weighted Average) 3D Gaussian Splatting rasterizer.
//  Each splat is drawn as a screen-space quad (2 triangles, instanced). The
//  quad is sized/oriented by projecting the splat's 3D covariance into a 2D
//  screen-space covariance, and the fragment shader evaluates the 2D Gaussian.
//
//  Conventions (must match the caller — SplatRenderer contract says "standard
//  right-handed view/projection"):
//    * View space: camera looks down -Z, so a visible point has cam.z < 0.
//    * Projection: standard RH perspective with clip.w = -cam.z.
//    * NDC: +x right, +y up (Metal). The 2D covariance is derived in the same
//      +x-right/+y-up image basis so the ellipse orientation is consistent with
//      the NDC we offset in.
//
//  Blending (set on the CPU side): premultiplied "over" —
//    src = ONE, dst = ONE_MINUS_SRC_ALPHA — with splats drawn back-to-front
//    (sorted by SplatSort.metal). The fragment outputs premultiplied color.
//

#include <metal_stdlib>
using namespace metal;

struct GPUSplat {
    packed_float3 position;  // world-space mean
    packed_float3 cov3d_a;   // Σ00, Σ01, Σ02
    packed_float3 cov3d_b;   // Σ11, Σ12, Σ22
    packed_float4 color;     // linear rgb + opacity(0..1)
};

struct SortEntry {
    float key;
    uint index;
};

struct SplatUniforms {
    float4x4 view;
    float4x4 projection;
    float2 viewport;  // pixels
    float2 focal;     // (fx, fy) pixels
};

struct RasterVertexOut {
    float4 position [[position]];
    float2 local;    // quad coordinate, ~[-2,2]; Gaussian arg = -dot(local,local)
    float4 color;    // rgb + opacity
};

// Quad corners for a 4-vertex triangle strip. ±2 covers ~2.8 sigma.
constant float2 kQuadCorners[4] = {
    float2(-2.0, -2.0),
    float2( 2.0, -2.0),
    float2(-2.0,  2.0),
    float2( 2.0,  2.0)
};

// A degenerate, clipped vertex (z > w => outside Metal's [0,w] clip range).
static inline RasterVertexOut culledVertex() {
    RasterVertexOut o;
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.local = float2(0.0);
    o.color = float4(0.0);
    return o;
}

vertex RasterVertexOut splatVertex(uint vid [[vertex_id]],
                                   uint iid [[instance_id]],
                                   const device GPUSplat* splats [[buffer(0)]],
                                   const device SortEntry* order [[buffer(1)]],
                                   constant SplatUniforms& u [[buffer(2)]]) {
    const uint idx = order[iid].index;
    const GPUSplat s = splats[idx];
    const float3 center = float3(s.position);

    const float4 cam = u.view * float4(center, 1.0);
    const float4 clip = u.projection * cam;

    // Cull: behind camera or well outside the frustum.
    const float lim = 1.2 * clip.w;
    if (clip.w <= 0.0 ||
        clip.z < -clip.w || clip.z > clip.w ||
        clip.x < -lim || clip.x > lim ||
        clip.y < -lim || clip.y > lim) {
        return culledVertex();
    }

    // World-space 3D covariance (symmetric), rebuilt from the 6 stored entries.
    const float3 A = float3(s.cov3d_a);
    const float3 B = float3(s.cov3d_b);
    const float3x3 Vrk = float3x3(float3(A.x, A.y, A.z),   // col0
                                  float3(A.y, B.x, B.y),   // col1
                                  float3(A.z, B.y, B.z));  // col2

    // View rotation (upper-left 3x3 of the view matrix), columns.
    const float3x3 R = float3x3(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);

    // Jacobian of the pinhole projection about cam (image basis u=+x, v=+y up):
    //   uimg = -fx * cam.x / cam.z ,  vimg = -fy * cam.y / cam.z
    // J (as columns) — third row is zero (we only need the 2D block).
    const float invz = 1.0 / cam.z;
    const float invz2 = invz * invz;
    const float3x3 J = float3x3(
        float3(-u.focal.x * invz, 0.0, 0.0),                        // ∂/∂cam.x
        float3(0.0, -u.focal.y * invz, 0.0),                        // ∂/∂cam.y
        float3(u.focal.x * cam.x * invz2, u.focal.y * cam.y * invz2, 0.0) // ∂/∂cam.z
    );

    // 2D covariance = (J R) Vrk (J R)ᵀ ; take the upper-left 2x2.
    const float3x3 T = J * R;
    float3x3 cov = T * Vrk * transpose(T);

    // Low-pass (antialiasing) dilation: guarantees at least ~1px footprint.
    float cov00 = cov[0][0] + 0.3;
    float cov01 = cov[1][0];
    float cov11 = cov[1][1] + 0.3;

    // Eigen-decomposition of the symmetric 2x2 covariance.
    const float mid = 0.5 * (cov00 + cov11);
    const float disc = sqrt(max(0.0, mid * mid - (cov00 * cov11 - cov01 * cov01)));
    const float lambda1 = mid + disc;
    const float lambda2 = mid - disc;
    if (lambda1 <= 0.0) {
        return culledVertex();
    }

    // Principal axes in pixel units (clamp to keep huge splats bounded).
    float2 diag;
    if (abs(cov01) < 1e-9 && abs(cov00 - cov11) < 1e-9) {
        diag = float2(1.0, 0.0);
    } else {
        diag = normalize(float2(cov01, lambda1 - cov00));
    }
    const float2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diag;
    const float2 minorAxis = min(sqrt(2.0 * max(lambda2, 0.0)), 1024.0) * float2(diag.y, -diag.x);

    const float2 corner = kQuadCorners[vid];
    const float2 ndcCenter = clip.xy / clip.w;

    // Pixel offset -> NDC offset (NDC spans 2 over `viewport` pixels).
    const float2 pixelOffset = corner.x * majorAxis + corner.y * minorAxis;
    const float2 ndcOffset = pixelOffset / (0.5 * u.viewport);

    RasterVertexOut out;
    out.position = float4(ndcCenter + ndcOffset, 0.0, 1.0);
    out.local = corner;
    out.color = s.color;
    return out;
}

fragment float4 splatFragment(RasterVertexOut in [[stage_in]]) {
    // 2D Gaussian: value = exp(-|local|^2). See SplatPLYParser note on why the
    // ±2 quad + this exponent reproduce the covariance-weighted falloff.
    const float power = -dot(in.local, in.local);
    const float g = exp(power);
    const float alpha = min(0.99, in.color.a * g);
    if (alpha < (1.0 / 255.0)) {
        discard_fragment();
    }
    // Premultiplied output for "over" blending.
    return float4(in.color.rgb * alpha, alpha);
}
