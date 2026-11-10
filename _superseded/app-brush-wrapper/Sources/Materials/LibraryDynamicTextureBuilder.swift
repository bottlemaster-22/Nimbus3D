//
//  LibraryDynamicTextureBuilder.swift
//  Nimbus3D (Materials module)
//
//  REAL implementation of DynamicTextureBuilder following the Dynamic Textures
//  design: it SUBSTITUTES a class-appropriate tileable PBR material from the
//  bundled CC0 ambientCG library and tints it toward the captured surface colour.
//  It does NOT extract fine surface relief from the splat, and it does not claim
//  to: the normal/height/AO detail comes from the library material, tinted to the
//  measured capture colour so it reads as the same surface.
//
//  Pipeline for one build:
//    1. Measure the mean linear colour of the (de-lit or passthrough) albedo.
//    2. Pick the library entry for the confirmed material class.
//    3. Tint the library colour map toward the captured colour (per-channel gain
//       in linear space), resample to the requested resolution -> albedo.png.
//    4. Copy/resample the library normal (or synthesise one from the library
//       displacement via the Sobel baseline), height, roughness, metallic, AO.
//    5. Attach class-appropriate parallax-occlusion parameters.
//
//  When the confirmed class has no library substitute (plastic / glass / unknown)
//  or the library is unavailable, it falls back to an honest albedo-only set:
//  the captured albedo, a flat normal, a constant roughness, POM disabled.
//

import Foundation
import CoreGraphics
import simd

public final class LibraryDynamicTextureBuilder: DynamicTextureBuilder, @unchecked Sendable {

    /// Bundled CC0 material library. Nil if the manifest failed to load; the
    /// builder then always uses the albedo-only fallback.
    private let library: MaterialLibrary?

    public init(bundle: Bundle = .main) {
        self.library = try? MaterialLibrary(bundle: bundle)
    }

    /// Convenience for callers that already hold a library instance.
    public init(library: MaterialLibrary?) {
        self.library = library
    }

    public func buildMaterialSet(delitAlbedoURL: URL,
                                 mesh: MeshAsset,
                                 classification: MaterialClassification,
                                 options: TextureBuildOptions,
                                 progress: @escaping ProgressHandler) async throws -> MaterialSet {
        let res = clampResolution(options.textureResolution)
        let setID = UUID()
        let outDir = delitAlbedoURL
            .deletingLastPathComponent()
            .appendingPathComponent("materialset_\(setID.uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 0.05,
                                  message: "Measuring captured surface colour"))

        let capturedImage = try MaterialImageIO.loadCGImage(at: delitAlbedoURL)
        let capturedMean = try MaterialImageIO.meanLinearRGB(of: capturedImage)

        let coarse = CoarseMaterial(contractClass: classification.materialClass)
        let profile = MaterialSurfaceProfile.profile(for: coarse)
        let entry = library?.entry(for: classification.materialClass)

        if let library, let entry {
            return try buildFromLibrary(setID: setID,
                                        outDir: outDir,
                                        res: res,
                                        options: options,
                                        classification: classification,
                                        profile: profile,
                                        library: library,
                                        entry: entry,
                                        capturedMean: capturedMean,
                                        progress: progress)
        } else {
            return try buildFallback(setID: setID,
                                     outDir: outDir,
                                     res: res,
                                     options: options,
                                     classification: classification,
                                     profile: profile,
                                     capturedImage: capturedImage,
                                     progress: progress)
        }
    }

    // MARK: - Library substitution path

    private func buildFromLibrary(setID: UUID,
                                  outDir: URL,
                                  res: Int,
                                  options: TextureBuildOptions,
                                  classification: MaterialClassification,
                                  profile: MaterialSurfaceProfile,
                                  library: MaterialLibrary,
                                  entry: MaterialLibraryEntry,
                                  capturedMean: SIMD3<Double>,
                                  progress: @escaping ProgressHandler) throws -> MaterialSet {
        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 0.2,
                                  message: "Substituting \(entry.displayName) and tinting to capture"))

        // --- Albedo: tint the library colour toward the captured colour ---
        let libColorImage = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: entry.maps.color))
        let libMean = try MaterialImageIO.meanLinearRGB(of: libColorImage)
        let gain = tintGain(captured: capturedMean, library: libMean)
        let tintedAlbedo = try MaterialImageIO.tinted(libColorImage, gain: gain, size: res)
        let albedoURL = outDir.appendingPathComponent("albedo.png")
        try MaterialImageIO.writePNG(tintedAlbedo, to: albedoURL)

        // --- Height / displacement ---
        var heightURL: URL?
        var libDisplacementImage: CGImage?
        if let dispName = entry.maps.displacement {
            let dispImage = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: dispName))
            libDisplacementImage = dispImage
            if options.generateHeight {
                let heightGray = try MaterialImageIO.resampledGray(dispImage, to: res)
                let url = outDir.appendingPathComponent("height.png")
                try MaterialImageIO.writePNG(heightGray, to: url)
                heightURL = url
            }
        }

        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 0.5,
                                  message: "Building normal / roughness / AO maps"))

        // --- Normal: prefer the ready-made library normal; else Sobel-from-height ---
        var normalURL: URL?
        if options.generateNormal {
            let url = outDir.appendingPathComponent("normal.png")
            if let normalName = entry.maps.normalGL {
                let normalImage = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: normalName))
                let resampled = try MaterialImageIO.resampled(normalImage, to: res)
                try MaterialImageIO.writePNG(resampled, to: url)
                normalURL = url
            } else if let dispImage = libDisplacementImage {
                let heightPixels = try MaterialImageIO.gray8Pixels(of: dispImage, size: res)
                let normal = try MaterialImageIO.normalMap(fromHeightPixels: heightPixels,
                                                           size: res,
                                                           strength: profile.normalStrength,
                                                           wrap: true)
                try MaterialImageIO.writePNG(normal, to: url)
                normalURL = url
            }
        }

        // --- Roughness ---
        var roughnessURL: URL?
        if options.generateRoughness {
            let url = outDir.appendingPathComponent("roughness.png")
            if let roughName = entry.maps.roughness {
                let img = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: roughName))
                try MaterialImageIO.writePNG(try MaterialImageIO.resampledGray(img, to: res), to: url)
            } else {
                try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: profile.fallbackRoughness, size: res), to: url)
            }
            roughnessURL = url
        }

        // --- Metallic ---
        var metallicURL: URL?
        if options.generateMetallic {
            let url = outDir.appendingPathComponent("metallic.png")
            if let metalName = entry.maps.metalness {
                let img = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: metalName))
                try MaterialImageIO.writePNG(try MaterialImageIO.resampledGray(img, to: res), to: url)
            } else {
                try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: profile.fallbackMetallic, size: res), to: url)
            }
            metallicURL = url
        }

        // --- Ambient occlusion ---
        var aoURL: URL?
        if options.generateAmbientOcclusion {
            let url = outDir.appendingPathComponent("ao.png")
            if let aoName = entry.maps.ambientOcclusion {
                let img = try MaterialImageIO.loadCGImage(at: library.url(forMapFileName: aoName))
                try MaterialImageIO.writePNG(try MaterialImageIO.resampledGray(img, to: res), to: url)
            } else {
                try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: 255, size: res), to: url)
            }
            aoURL = url
        }

        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 1.0,
                                  message: "Material set ready (\(entry.displayName), \(res)x\(res))"))

        return MaterialSet(id: setID,
                           albedoURL: albedoURL,
                           normalURL: normalURL,
                           heightURL: heightURL,
                           roughnessURL: roughnessURL,
                           metallicURL: metallicURL,
                           ambientOcclusionURL: aoURL,
                           classification: classification,
                           parallax: profile.parallax,
                           textureResolution: res)
    }

    // MARK: - Fallback (no library substitute)

    private func buildFallback(setID: UUID,
                               outDir: URL,
                               res: Int,
                               options: TextureBuildOptions,
                               classification: MaterialClassification,
                               profile: MaterialSurfaceProfile,
                               capturedImage: CGImage,
                               progress: @escaping ProgressHandler) throws -> MaterialSet {
        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 0.3,
                                  message: "No library substitute for \(classification.materialClass.rawValue); using captured albedo only"))

        // Albedo = the captured/de-lit albedo, resampled. No relief invented.
        let albedoURL = outDir.appendingPathComponent("albedo.png")
        try MaterialImageIO.writePNG(try MaterialImageIO.resampled(capturedImage, to: res), to: albedoURL)

        // Flat tangent-space normal (0,0,1) via a zero-relief height buffer.
        var normalURL: URL?
        if options.generateNormal {
            let flat = [UInt8](repeating: 128, count: res * res)
            let normal = try MaterialImageIO.normalMap(fromHeightPixels: flat, size: res, strength: 0, wrap: true)
            let url = outDir.appendingPathComponent("normal.png")
            try MaterialImageIO.writePNG(normal, to: url)
            normalURL = url
        }

        var roughnessURL: URL?
        if options.generateRoughness {
            let url = outDir.appendingPathComponent("roughness.png")
            try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: profile.fallbackRoughness, size: res), to: url)
            roughnessURL = url
        }

        var metallicURL: URL?
        if options.generateMetallic {
            let url = outDir.appendingPathComponent("metallic.png")
            try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: profile.fallbackMetallic, size: res), to: url)
            metallicURL = url
        }

        var aoURL: URL?
        if options.generateAmbientOcclusion {
            let url = outDir.appendingPathComponent("ao.png")
            try MaterialImageIO.writePNG(try MaterialImageIO.constantGray(value: 255, size: res), to: url)
            aoURL = url
        }

        progress(PipelineProgress(stage: .textureBuild,
                                  fractionCompleted: 1.0,
                                  message: "Albedo-only material set ready (\(res)x\(res), no substituted relief)"))

        // POM disabled in the fallback: there is no height detail to march.
        return MaterialSet(id: setID,
                           albedoURL: albedoURL,
                           normalURL: normalURL,
                           heightURL: nil,
                           roughnessURL: roughnessURL,
                           metallicURL: metallicURL,
                           ambientOcclusionURL: aoURL,
                           classification: classification,
                           parallax: ParallaxOcclusionParameters(enabled: false),
                           textureResolution: res)
    }

    // MARK: - Helpers

    /// Per-channel linear gain that shifts the library colour toward the captured
    /// colour, clamped so a near-black library texel cannot blow up the tint.
    private func tintGain(captured: SIMD3<Double>, library: SIMD3<Double>) -> SIMD3<Double> {
        let eps = 1e-3
        func channel(_ c: Double, _ l: Double) -> Double {
            let g = c / max(l, eps)
            return min(max(g, 0.2), 5.0)
        }
        return SIMD3<Double>(channel(captured.x, library.x),
                             channel(captured.y, library.y),
                             channel(captured.z, library.z))
    }

    private func clampResolution(_ requested: Int) -> Int {
        // Keep to a sane power-of-two-ish range for on-device memory.
        min(max(requested, 256), 4096)
    }
}
