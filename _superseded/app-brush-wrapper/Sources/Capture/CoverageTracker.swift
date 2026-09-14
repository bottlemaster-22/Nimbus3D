//
//  CoverageTracker.swift — angular coverage estimation for capture guidance.
//  Owned by the Capture agent.
//
//  Given each posed camera during an object scan, this estimates the subject center
//  (where the user is orbiting) and reports how much of the 360-degree circle around
//  that subject has been observed. The capture UI turns `coverage` into a progress ring.
//
//  This is a real, self-contained heuristic (no ML): the subject center is the running
//  mean of per-frame look-at points (camera position pushed forward by the measured or
//  assumed distance to the object), and coverage is the fraction of azimuth sectors that
//  contain at least one camera viewpoint.
//

import simd

final class CoverageTracker {

    /// Number of azimuth sectors around the subject (15 degrees each).
    private let azimuthSectors = 24
    /// Used when no LiDAR depth is available to estimate how far the object is.
    private let assumedObjectDistance: Float = 0.45

    private var lookAtPoints: [SIMD3<Float>] = []
    private var cameraPositions: [SIMD3<Float>] = []
    private var filledSectors = Set<Int>()

    /// Running estimate of the subject's world-space center. Nil before the first frame.
    private(set) var subjectCenter: SIMD3<Float>?

    /// 0...1 fraction of azimuth sectors observed.
    var coverage: Float {
        Float(filledSectors.count) / Float(azimuthSectors)
    }

    /// Approximate object radius in meters (mean camera distance to the estimated center),
    /// used as a trainer init hint. Nil before enough data.
    var estimatedRadius: Float? {
        guard let center = subjectCenter, !cameraPositions.isEmpty else { return nil }
        let total = cameraPositions.reduce(Float(0)) { $0 + simd_distance($1, center) }
        let radius = total / Float(cameraPositions.count)
        return radius.isFinite && radius > 0 ? radius : nil
    }

    func reset() {
        lookAtPoints.removeAll()
        cameraPositions.removeAll()
        filledSectors.removeAll()
        subjectCenter = nil
    }

    /// Register a tracked camera. `centerDepth` is the LiDAR depth (meters) at the image
    /// center if available; otherwise nil and an assumed distance is used.
    func registerFrame(cameraTransform: simd_float4x4, centerDepth: Float?) {
        let camPos = cameraTransform.columns.3.xyz
        // ARKit camera looks down its local -Z axis.
        let forward = -simd_normalize(cameraTransform.columns.2.xyz)
        let distance = centerDepth.map { max(0.1, min($0, 5.0)) } ?? assumedObjectDistance
        let lookAt = camPos + forward * distance

        lookAtPoints.append(lookAt)
        cameraPositions.append(camPos)

        let mean = lookAtPoints.reduce(SIMD3<Float>(repeating: 0), +) / Float(lookAtPoints.count)
        subjectCenter = mean

        // Azimuth of the camera around the subject in the gravity (x/z) plane.
        let offset = camPos - mean
        let azimuth = atan2(offset.z, offset.x) // -pi ... pi
        let normalized = (azimuth + .pi) / (2 * .pi) // 0 ... 1
        var sector = Int(normalized * Float(azimuthSectors))
        if sector >= azimuthSectors { sector = azimuthSectors - 1 }
        if sector < 0 { sector = 0 }
        filledSectors.insert(sector)
    }
}
