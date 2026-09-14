//
//  HDRIAssembler.swift — HDRI module.
//
//  Concrete `HDRICapture` implementation: takes a CaptureBundle's pose-tagged
//  bracketed exposures and produces a 32-bit linear equirectangular OpenEXR
//  environment map, plus dominant-light and average-luminance analysis.
//
//  Pipeline:
//    1. Group brackets by camera pose (multi-EV frames of the same view).
//    2. Debevec-style log-domain radiance merge per group  (RadianceMerge).
//    3. Gather-project all merged views into an equirect panorama, feather-blended
//       (EquirectProjector).
//    4. Analyze (dominant light direction, mean luminance) using Accelerate.
//    5. Write the panorama as a real 32-bit float .exr (EXRWriter).
//

import Foundation
import Accelerate
import simd

public final class HDRIAssembler: HDRICapture {

    public init() {}

    public func assembleHDRI(from bundle: CaptureBundle,
                             options: HDRIAssemblyOptions,
                             progress: @escaping ProgressHandler) async throws -> HDRIEnvironment {

        let brackets = bundle.hdriBrackets
        guard !brackets.isEmpty else {
            throw NimbusError.hdriAssemblyFailed(
                "Capture bundle contains no HDRI bracket frames.")
        }

        report(progress, 0.0, "Preparing bracketed exposures", indeterminate: true)
        try Task.checkCancellation()

        // 1. Group by pose (frames of the same view captured at different EVs).
        let groups = groupByPose(brackets)

        // 2. Merge each group to linear radiance.
        var images: [RadianceImage] = []
        images.reserveCapacity(groups.count)
        for (i, group) in groups.enumerated() {
            try Task.checkCancellation()
            images.append(try RadianceMerge.merge(group))
            report(progress, 0.05 + 0.35 * Double(i + 1) / Double(groups.count),
                   "Merging exposures \(i + 1)/\(groups.count)")
        }
        guard !images.isEmpty else {
            throw NimbusError.hdriAssemblyFailed("No usable bracket frames after merge.")
        }

        // 3. Project into the equirectangular panorama.
        let W = max(options.outputWidth, 2)
        let H = W / 2
        let projGranularity = max(H / 20, 1)
        let projected = EquirectProjector.project(images: images, outputWidth: W) { row in
            if row % projGranularity == 0 {
                report(progress, 0.4 + 0.45 * Double(row) / Double(H),
                       "Projecting panorama")
            }
        }
        try Task.checkCancellation()

        if projected.coverage < 0.5 {
            report(progress, 0.85,
                   String(format: "Partial coverage (%.0f%%); gaps filled",
                          projected.coverage * 100))
        }

        // 4. Analyze lighting.
        report(progress, 0.88, "Analyzing lighting")
        let analysis = analyze(projected)

        // 5. Write EXR.
        report(progress, 0.92, "Writing OpenEXR")
        let dir = bundle.rootDirectory.appendingPathComponent("hdri", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let exrURL = dir.appendingPathComponent("environment.exr")
        try EXRWriter.write(rgb: projected.pixels, width: projected.width,
                            height: projected.height, to: exrURL)

        report(progress, 1.0, "HDRI ready")

        return HDRIEnvironment(equirectangularEXRURL: exrURL,
                               width: projected.width,
                               height: projected.height,
                               dominantLightDirection: analysis.lightDirection,
                               averageLuminance: analysis.averageLuminance)
    }

    // MARK: - Pose grouping

    /// Group frames that share (approximately) the same camera pose so that a set
    /// of same-view multi-EV brackets merges together. In practice ARKit-driven
    /// captures are mostly single-exposure-per-pose, so groups of one are normal
    /// and handled by `RadianceMerge.merge`.
    private func groupByPose(_ frames: [ExposureBracketFrame]) -> [[ExposureBracketFrame]] {
        var groups: [[ExposureBracketFrame]] = []
        let posTol: Float = 0.02      // 2 cm
        let angTol: Float = 0.9995    // ~1.8 degrees

        for f in frames {
            let m = f.pose.matrix
            let pos = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            let fwd = simd_normalize(-SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z))

            var placed = false
            for i in groups.indices {
                let gm = groups[i][0].pose.matrix
                let gpos = SIMD3<Float>(gm.columns.3.x, gm.columns.3.y, gm.columns.3.z)
                let gfwd = simd_normalize(-SIMD3<Float>(gm.columns.2.x, gm.columns.2.y, gm.columns.2.z))
                if simd_distance(pos, gpos) < posTol && simd_dot(fwd, gfwd) > angTol {
                    groups[i].append(f)
                    placed = true
                    break
                }
            }
            if !placed { groups.append([f]) }
        }
        return groups
    }

    // MARK: - Analysis

    private struct Analysis {
        var lightDirection: SIMD3<Float>?
        var averageLuminance: Float?
    }

    /// Rec.709 luminance-weighted analysis: mean scene luminance (Accelerate) and
    /// the direction toward the single brightest panorama texel.
    private func analyze(_ r: EquirectProjector.Result) -> Analysis {
        let n = r.pixels.count
        guard n > 0 else { return Analysis(lightDirection: nil, averageLuminance: nil) }

        var lum = [Float](repeating: 0, count: n)
        var maxLum: Float = -1
        var maxIdx = 0
        for i in 0..<n {
            let p = r.pixels[i]
            let l = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z
            lum[i] = l
            if l > maxLum { maxLum = l; maxIdx = i }
        }

        let avg = vDSP.mean(lum)
        let lx = maxIdx % r.width
        let ly = maxIdx / r.width
        let dir = EquirectProjector.direction(x: lx, y: ly, width: r.width, height: r.height)

        return Analysis(lightDirection: maxLum > 0 ? simd_normalize(dir) : nil,
                        averageLuminance: avg)
    }

    // MARK: - Progress

    private func report(_ handler: ProgressHandler, _ fraction: Double,
                        _ message: String, indeterminate: Bool = false) {
        handler(PipelineProgress(stage: .hdriAssembly,
                                 fractionCompleted: min(max(fraction, 0), 1),
                                 message: message,
                                 isIndeterminate: indeterminate))
    }
}
