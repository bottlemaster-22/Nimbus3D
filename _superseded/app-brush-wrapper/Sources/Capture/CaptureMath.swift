//
//  CaptureMath.swift — small simd helpers used across the Capture module.
//  Owned by the Capture agent. Keep this file dependency-free (simd only).
//

import simd

extension SIMD4 where Scalar == Float {
    /// Drops the w component. Useful for pulling a position/direction out of a 4x4 column.
    var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

enum CaptureMath {
    /// Angle in radians between two (not necessarily normalized) vectors. Returns 0 for degenerate input.
    static func angleBetween(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let la = simd_length(a), lb = simd_length(b)
        guard la > 1e-6, lb > 1e-6 else { return 0 }
        let c = simd_dot(a, b) / (la * lb)
        return acos(max(-1, min(1, c)))
    }

    static func degreesToRadians(_ d: Float) -> Float { d * .pi / 180 }
}
