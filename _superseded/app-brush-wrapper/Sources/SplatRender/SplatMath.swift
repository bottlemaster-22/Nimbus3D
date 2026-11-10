//
//  SplatMath.swift — SplatRender module.
//
//  Right-handed camera math matching the SplatRenderer contract and the shader
//  conventions in SplatRasterize.metal:
//    * lookAt builds a world->view matrix where the camera looks down -Z.
//    * perspective builds a standard RH projection with clip.w = -cam.z and
//      Metal's [0,1] clip-space depth range.
//

import simd

enum SplatMath {

    /// Right-handed look-at (world -> view). Camera looks toward `center`, -Z forward.
    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = simd_normalize(center - eye)          // forward (+ toward target)
        var s = simd_cross(f, up)
        let sl = simd_length(s)
        s = sl > 1e-6 ? s / sl : SIMD3<Float>(1, 0, 0) // right
        let u = simd_cross(s, f)                       // true up

        // Columns of the view matrix (column-major).
        let c0 = SIMD4<Float>(s.x, u.x, -f.x, 0)
        let c1 = SIMD4<Float>(s.y, u.y, -f.y, 0)
        let c2 = SIMD4<Float>(s.z, u.z, -f.z, 0)
        let c3 = SIMD4<Float>(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)
        return simd_float4x4(columns: (c0, c1, c2, c3))
    }

    /// Right-handed perspective, -Z forward, Metal depth range [0, 1].
    static func perspective(fovyRadians: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovyRadians * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)             // maps near->0, far->1 after divide
        let c0 = SIMD4<Float>(xs, 0, 0, 0)
        let c1 = SIMD4<Float>(0, ys, 0, 0)
        let c2 = SIMD4<Float>(0, 0, zs, -1)     // clip.w = -cam.z
        let c3 = SIMD4<Float>(0, 0, zs * near, 0)
        return simd_float4x4(columns: (c0, c1, c2, c3))
    }
}
