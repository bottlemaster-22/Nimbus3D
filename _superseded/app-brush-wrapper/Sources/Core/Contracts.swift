//
//  Contracts.swift
//  Nimbus3D
//
//  SHARED CONTRACTS for the entire pipeline. Every module imports and implements
//  against these exact types and protocols. Do NOT change a signature here without
//  updating NIMBUS_CONTRACTS.md and every conforming module.
//
//  Pipeline order:
//    CaptureService -> SplatTrainer -> (SplatRenderer for preview)
//      -> MeshExtractor -> Delighter -> MaterialClassifier -> DynamicTextureBuilder
//      -> HDRICapture -> AssetExporter
//

import Foundation
import simd
import Metal
import QuartzCore

// MARK: - Progress

/// Every long-running stage reports progress through this shared type.
public struct PipelineProgress: Codable, Sendable, Equatable {
    public var stage: PipelineStage
    /// 0.0 ... 1.0 within the current stage. Ignored when `isIndeterminate` is true.
    public var fractionCompleted: Double
    /// Short human-readable status, e.g. "Training splats: iteration 4200/8000".
    public var message: String
    public var isIndeterminate: Bool

    public init(stage: PipelineStage,
                fractionCompleted: Double,
                message: String,
                isIndeterminate: Bool = false) {
        self.stage = stage
        self.fractionCompleted = fractionCompleted
        self.message = message
        self.isIndeterminate = isIndeterminate
    }
}

public enum PipelineStage: String, Codable, Sendable, CaseIterable {
    case capture
    case splatTraining
    case meshExtraction
    case delighting
    case materialClassification
    case textureBuild
    case hdriAssembly
    case export
}

/// Shared progress-callback signature used by all stage protocols.
public typealias ProgressHandler = @Sendable (PipelineProgress) -> Void

// MARK: - Geometry primitives

/// Codable/Sendable wrapper for a column-major 4x4 camera-to-world transform.
/// (simd_float4x4 itself is not Codable, hence the column storage.)
public struct CameraPose: Codable, Sendable, Equatable {
    public var c0: SIMD4<Float>
    public var c1: SIMD4<Float>
    public var c2: SIMD4<Float>
    public var c3: SIMD4<Float>

    public init(matrix m: simd_float4x4) {
        self.c0 = m.columns.0
        self.c1 = m.columns.1
        self.c2 = m.columns.2
        self.c3 = m.columns.3
    }

    public var matrix: simd_float4x4 {
        simd_float4x4(columns: (c0, c1, c2, c3))
    }

    public static let identity = CameraPose(matrix: matrix_identity_float4x4)
}

/// Pinhole camera intrinsics in pixel units, matching ARKit's ARCamera.intrinsics layout.
public struct CameraIntrinsics: Codable, Sendable, Equatable {
    /// (fx, fy) focal length in pixels.
    public var focalLength: SIMD2<Float>
    /// (cx, cy) principal point in pixels.
    public var principalPoint: SIMD2<Float>
    public var imageWidth: Int
    public var imageHeight: Int

    public init(focalLength: SIMD2<Float>,
                principalPoint: SIMD2<Float>,
                imageWidth: Int,
                imageHeight: Int) {
        self.focalLength = focalLength
        self.principalPoint = principalPoint
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
    }
}

public struct AxisAlignedBoundingBox: Codable, Sendable, Equatable {
    public var minCorner: SIMD3<Float>
    public var maxCorner: SIMD3<Float>

    public init(minCorner: SIMD3<Float>, maxCorner: SIMD3<Float>) {
        self.minCorner = minCorner
        self.maxCorner = maxCorner
    }

    public var center: SIMD3<Float> { (minCorner + maxCorner) * 0.5 }
    public var extents: SIMD3<Float> { maxCorner - minCorner }
}

// MARK: - Capture types

/// One posed RGB frame captured during scanning, stored on disk.
public struct CapturedFrame: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    /// Sequential capture index within the bundle (0-based).
    public var index: Int
    /// HEIC or JPEG color image on disk.
    public var imageURL: URL
    /// Optional 32-bit float LiDAR depth map (binary, row-major, meters). Nil on non-LiDAR devices.
    public var depthMapURL: URL?
    /// Optional ARKit depth-confidence map matching `depthMapURL` dimensions.
    public var depthConfidenceURL: URL?
    /// Camera-to-world transform at capture time (ARKit world space, meters).
    public var pose: CameraPose
    public var intrinsics: CameraIntrinsics
    /// Seconds since capture-session start.
    public var timestamp: TimeInterval
    public var exposureDuration: TimeInterval
    public var iso: Float

    public init(id: UUID = UUID(),
                index: Int,
                imageURL: URL,
                depthMapURL: URL? = nil,
                depthConfidenceURL: URL? = nil,
                pose: CameraPose,
                intrinsics: CameraIntrinsics,
                timestamp: TimeInterval,
                exposureDuration: TimeInterval,
                iso: Float) {
        self.id = id
        self.index = index
        self.imageURL = imageURL
        self.depthMapURL = depthMapURL
        self.depthConfidenceURL = depthConfidenceURL
        self.pose = pose
        self.intrinsics = intrinsics
        self.timestamp = timestamp
        self.exposureDuration = exposureDuration
        self.iso = iso
    }
}

/// One frame of a bracketed-exposure sweep used for HDRI assembly.
public struct ExposureBracketFrame: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var imageURL: URL
    /// EV bias relative to metered exposure (e.g. -2, 0, +2).
    public var exposureBias: Float
    public var exposureDuration: TimeInterval
    public var iso: Float
    /// Camera-to-world pose so brackets can be projected into an equirect panorama.
    public var pose: CameraPose
    public var intrinsics: CameraIntrinsics

    public init(id: UUID = UUID(),
                imageURL: URL,
                exposureBias: Float,
                exposureDuration: TimeInterval,
                iso: Float,
                pose: CameraPose,
                intrinsics: CameraIntrinsics) {
        self.id = id
        self.imageURL = imageURL
        self.exposureBias = exposureBias
        self.exposureDuration = exposureDuration
        self.iso = iso
        self.pose = pose
        self.intrinsics = intrinsics
    }
}

/// Everything a capture session produces: the input to splat training and HDRI assembly.
/// All referenced files live under `rootDirectory` so a bundle can be moved/deleted as a unit.
public struct CaptureBundle: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var createdAt: Date
    /// Directory containing all frame images, depth maps, and this bundle's manifest.
    public var rootDirectory: URL
    public var frames: [CapturedFrame]
    /// Bracketed-exposure frames for HDRI assembly. Empty if HDRI capture was skipped.
    public var hdriBrackets: [ExposureBracketFrame]
    public var hasLiDARDepth: Bool
    /// Approximate scene radius in meters, if estimated during capture (helps trainer init).
    public var sceneBoundingRadius: Float?

    public init(id: UUID = UUID(),
                createdAt: Date = Date(),
                rootDirectory: URL,
                frames: [CapturedFrame],
                hdriBrackets: [ExposureBracketFrame] = [],
                hasLiDARDepth: Bool = false,
                sceneBoundingRadius: Float? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.rootDirectory = rootDirectory
        self.frames = frames
        self.hdriBrackets = hdriBrackets
        self.hasLiDARDepth = hasLiDARDepth
        self.sceneBoundingRadius = sceneBoundingRadius
    }
}

public struct CaptureOptions: Codable, Sendable, Equatable {
    /// Frames the guidance UI aims for (typical: 60-150).
    public var targetFrameCount: Int
    public var captureLiDARDepth: Bool
    public var captureHDRIBrackets: Bool
    /// Where the bundle directory is created. Nil = app's Documents/Captures/<uuid>.
    public var outputDirectory: URL?

    public init(targetFrameCount: Int = 100,
                captureLiDARDepth: Bool = true,
                captureHDRIBrackets: Bool = true,
                outputDirectory: URL? = nil) {
        self.targetFrameCount = targetFrameCount
        self.captureLiDARDepth = captureLiDARDepth
        self.captureHDRIBrackets = captureHDRIBrackets
        self.outputDirectory = outputDirectory
    }
}

public enum TrackingQuality: String, Codable, Sendable {
    case notAvailable
    case limited
    case normal
}

/// Live events the capture UI subscribes to while scanning.
public enum CaptureEvent: Sendable, Equatable {
    case trackingStateChanged(TrackingQuality)
    case frameCaptured(index: Int, target: Int)
    /// 0...1 estimate of angular coverage around the subject.
    case coverageUpdated(Float)
    case warning(String)
}

// MARK: - Splat types

public enum SplatFileFormat: String, Codable, Sendable {
    /// Uncompressed Gaussian-splat PLY (Brush's native training output).
    case ply
    /// Niantic .spz compressed splat format (export/interchange).
    case spz
}

/// Handle to a trained Gaussian-splat model on disk, plus training metadata.
public struct SplatModel: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var splatFileURL: URL
    public var format: SplatFileFormat
    public var splatCount: Int
    public var trainingIterations: Int
    /// CaptureBundle.id this model was trained from.
    public var sourceCaptureID: UUID
    public var boundingBox: AxisAlignedBoundingBox
    public var createdAt: Date

    public init(id: UUID = UUID(),
                splatFileURL: URL,
                format: SplatFileFormat,
                splatCount: Int,
                trainingIterations: Int,
                sourceCaptureID: UUID,
                boundingBox: AxisAlignedBoundingBox,
                createdAt: Date = Date()) {
        self.id = id
        self.splatFileURL = splatFileURL
        self.format = format
        self.splatCount = splatCount
        self.trainingIterations = trainingIterations
        self.sourceCaptureID = sourceCaptureID
        self.boundingBox = boundingBox
        self.createdAt = createdAt
    }
}

public struct SplatTrainingConfig: Codable, Sendable, Equatable {
    /// Total optimization iterations (on-device typical: 3000-10000).
    public var iterations: Int
    /// Hard cap on splat count to bound memory (0 = engine default).
    public var maxSplatCount: Int
    /// Long-edge pixel cap for training images (downscale above this).
    public var maxImageDimension: Int
    /// Spherical-harmonics degree for view-dependent color (0-3; higher = larger model).
    public var shDegree: Int

    public init(iterations: Int = 6000,
                maxSplatCount: Int = 1_500_000,
                maxImageDimension: Int = 1280,
                shDegree: Int = 2) {
        self.iterations = iterations
        self.maxSplatCount = maxSplatCount
        self.maxImageDimension = maxImageDimension
        self.shDegree = shDegree
    }
}

// MARK: - Mesh types

/// Triangle mesh extracted from a splat model. Buffers are held in memory;
/// large meshes should be decimated before construction (see MeshExtractionOptions).
public struct MeshAsset: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var positions: [SIMD3<Float>]
    /// Per-vertex normals; same count as `positions`.
    public var normals: [SIMD3<Float>]
    /// Per-vertex UVs; same count as `positions`. Empty until unwrapped.
    public var uvs: [SIMD2<Float>]
    /// Triangle list: 3 indices per triangle into the vertex arrays.
    public var indices: [UInt32]
    /// True when decimated to a game-ready low-poly budget.
    public var isLowPoly: Bool
    /// SplatModel.id this mesh was extracted from, if any.
    public var sourceSplatID: UUID?

    public var vertexCount: Int { positions.count }
    public var triangleCount: Int { indices.count / 3 }

    public init(id: UUID = UUID(),
                positions: [SIMD3<Float>],
                normals: [SIMD3<Float>],
                uvs: [SIMD2<Float>] = [],
                indices: [UInt32],
                isLowPoly: Bool,
                sourceSplatID: UUID? = nil) {
        self.id = id
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.indices = indices
        self.isLowPoly = isLowPoly
        self.sourceSplatID = sourceSplatID
    }
}

public struct MeshExtractionOptions: Codable, Sendable, Equatable {
    /// Decimation target. 0 = keep full resolution.
    public var targetTriangleCount: Int
    /// Marks the result low-poly and applies aggressive simplification.
    public var lowPoly: Bool
    /// Generate a UV unwrap (required before texture baking).
    public var unwrapUVs: Bool

    public init(targetTriangleCount: Int = 50_000,
                lowPoly: Bool = true,
                unwrapUVs: Bool = true) {
        self.targetTriangleCount = targetTriangleCount
        self.lowPoly = lowPoly
        self.unwrapUVs = unwrapUVs
    }
}

// MARK: - Material types

public enum MaterialClass: String, Codable, Sendable, CaseIterable {
    case brick
    case wood
    case metal
    case stone
    case concrete
    case fabric
    case plastic
    case glass
    case ceramic
    case leather
    case foliage
    case ground
    case unknown
}

public struct MaterialClassification: Codable, Sendable, Equatable {
    public var materialClass: MaterialClass
    /// 0...1 confidence for `materialClass`.
    public var confidence: Float
    /// Remaining class scores, highest first (may be empty).
    public var alternatives: [MaterialScore]

    public init(materialClass: MaterialClass,
                confidence: Float,
                alternatives: [MaterialScore] = []) {
        self.materialClass = materialClass
        self.confidence = confidence
        self.alternatives = alternatives
    }
}

public struct MaterialScore: Codable, Sendable, Equatable {
    public var materialClass: MaterialClass
    public var confidence: Float

    public init(materialClass: MaterialClass, confidence: Float) {
        self.materialClass = materialClass
        self.confidence = confidence
    }
}

/// Parameters for parallax occlusion mapping, tuned per material class.
public struct ParallaxOcclusionParameters: Codable, Sendable, Equatable {
    public var enabled: Bool
    /// World-space displacement scale applied to the height map.
    public var heightScale: Float
    public var minSamples: Int
    public var maxSamples: Int

    public init(enabled: Bool = false,
                heightScale: Float = 0.02,
                minSamples: Int = 8,
                maxSamples: Int = 32) {
        self.enabled = enabled
        self.heightScale = heightScale
        self.minSamples = minSamples
        self.maxSamples = maxSamples
    }
}

/// Complete PBR texture set for a mesh. All maps are image files on disk;
/// nil means that map was not generated.
public struct MaterialSet: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var albedoURL: URL
    public var normalURL: URL?
    public var heightURL: URL?
    public var roughnessURL: URL?
    public var metallicURL: URL?
    public var ambientOcclusionURL: URL?
    public var classification: MaterialClassification
    public var parallax: ParallaxOcclusionParameters
    /// Square texture edge length in pixels (e.g. 1024, 2048).
    public var textureResolution: Int

    public init(id: UUID = UUID(),
                albedoURL: URL,
                normalURL: URL? = nil,
                heightURL: URL? = nil,
                roughnessURL: URL? = nil,
                metallicURL: URL? = nil,
                ambientOcclusionURL: URL? = nil,
                classification: MaterialClassification,
                parallax: ParallaxOcclusionParameters = ParallaxOcclusionParameters(),
                textureResolution: Int) {
        self.id = id
        self.albedoURL = albedoURL
        self.normalURL = normalURL
        self.heightURL = heightURL
        self.roughnessURL = roughnessURL
        self.metallicURL = metallicURL
        self.ambientOcclusionURL = ambientOcclusionURL
        self.classification = classification
        self.parallax = parallax
        self.textureResolution = textureResolution
    }
}

public struct TextureBuildOptions: Codable, Sendable, Equatable {
    public var textureResolution: Int
    public var generateNormal: Bool
    public var generateHeight: Bool
    public var generateRoughness: Bool
    public var generateMetallic: Bool
    public var generateAmbientOcclusion: Bool

    public init(textureResolution: Int = 2048,
                generateNormal: Bool = true,
                generateHeight: Bool = true,
                generateRoughness: Bool = true,
                generateMetallic: Bool = true,
                generateAmbientOcclusion: Bool = true) {
        self.textureResolution = textureResolution
        self.generateNormal = generateNormal
        self.generateHeight = generateHeight
        self.generateRoughness = generateRoughness
        self.generateMetallic = generateMetallic
        self.generateAmbientOcclusion = generateAmbientOcclusion
    }
}

// MARK: - HDRI types

/// Equirectangular HDR environment map assembled from bracketed captures.
public struct HDRIEnvironment: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    /// OpenEXR file on disk, 2:1 equirectangular, linear Rec.709.
    public var equirectangularEXRURL: URL
    public var width: Int
    public var height: Int
    /// Unit vector toward the brightest light source, if detected.
    public var dominantLightDirection: SIMD3<Float>?
    /// Mean scene luminance in relative linear units, if computed.
    public var averageLuminance: Float?

    public init(id: UUID = UUID(),
                equirectangularEXRURL: URL,
                width: Int,
                height: Int,
                dominantLightDirection: SIMD3<Float>? = nil,
                averageLuminance: Float? = nil) {
        self.id = id
        self.equirectangularEXRURL = equirectangularEXRURL
        self.width = width
        self.height = height
        self.dominantLightDirection = dominantLightDirection
        self.averageLuminance = averageLuminance
    }
}

public struct HDRIAssemblyOptions: Codable, Sendable, Equatable {
    /// Output width in pixels; height is width / 2.
    public var outputWidth: Int

    public init(outputWidth: Int = 4096) {
        self.outputWidth = outputWidth
    }
}

// MARK: - Export types

public enum ExportFormat: String, Codable, Sendable, CaseIterable {
    /// .gltf with external .bin and texture files.
    case gltf
    /// Single-file binary .glb with embedded buffers/textures.
    case glb
}

public struct ExportOptions: Codable, Sendable, Equatable {
    public var formats: Set<ExportFormat>
    public var includeSplat: Bool
    public var includeHDRI: Bool
    /// Where the export directory is created. Nil = app's Documents/Exports/<uuid>.
    public var outputDirectory: URL?

    public init(formats: Set<ExportFormat> = [.glb],
                includeSplat: Bool = true,
                includeHDRI: Bool = true,
                outputDirectory: URL? = nil) {
        self.formats = formats
        self.includeSplat = includeSplat
        self.includeHDRI = includeHDRI
        self.outputDirectory = outputDirectory
    }
}

/// Final on-disk deliverable. Every URL lives under `rootDirectory`.
public struct ExportedAsset: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var rootDirectory: URL
    /// Present when ExportFormat.gltf was requested.
    public var gltfURL: URL?
    /// Present when ExportFormat.glb was requested.
    public var glbURL: URL?
    /// The PBR set referenced by the glTF/GLB, if materials were built.
    public var materialSet: MaterialSet?
    /// Copied .exr environment, if includeHDRI.
    public var hdriURL: URL?
    /// Compressed .spz splat file, if includeSplat.
    public var splatURL: URL?
    public var createdAt: Date

    public init(id: UUID = UUID(),
                rootDirectory: URL,
                gltfURL: URL? = nil,
                glbURL: URL? = nil,
                materialSet: MaterialSet? = nil,
                hdriURL: URL? = nil,
                splatURL: URL? = nil,
                createdAt: Date = Date()) {
        self.id = id
        self.rootDirectory = rootDirectory
        self.gltfURL = gltfURL
        self.glbURL = glbURL
        self.materialSet = materialSet
        self.hdriURL = hdriURL
        self.splatURL = splatURL
        self.createdAt = createdAt
    }
}

// MARK: - Errors

public enum NimbusError: Error, LocalizedError, Sendable {
    case captureFailed(String)
    case trainingFailed(String)
    case renderingFailed(String)
    case meshExtractionFailed(String)
    case delightingFailed(String)
    case classificationFailed(String)
    case textureBuildFailed(String)
    case hdriAssemblyFailed(String)
    case exportFailed(String)
    case cancelled
    /// Thrown by honest stubs: the capability has no real on-device implementation yet.
    case notImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .captureFailed(let m): return "Capture failed: \(m)"
        case .trainingFailed(let m): return "Splat training failed: \(m)"
        case .renderingFailed(let m): return "Rendering failed: \(m)"
        case .meshExtractionFailed(let m): return "Mesh extraction failed: \(m)"
        case .delightingFailed(let m): return "Delighting failed: \(m)"
        case .classificationFailed(let m): return "Material classification failed: \(m)"
        case .textureBuildFailed(let m): return "Texture build failed: \(m)"
        case .hdriAssemblyFailed(let m): return "HDRI assembly failed: \(m)"
        case .exportFailed(let m): return "Export failed: \(m)"
        case .cancelled: return "Operation cancelled."
        case .notImplemented(let m): return "Not implemented: \(m)"
        }
    }
}

// MARK: - Stage protocols

/// Implemented by the Capture module (ARKit + AVFoundation).
public protocol CaptureService: AnyObject {
    /// True between startCapture and finish/cancel.
    var isCapturing: Bool { get }
    /// Live guidance/tracking events for the capture UI. Valid for the life of the service.
    var events: AsyncStream<CaptureEvent> { get }
    /// Begin an AR-tracked capture session. Frames accumulate until finished.
    func startCapture(options: CaptureOptions) async throws
    /// Stop the session and assemble the on-disk bundle (writes manifest.json in the bundle dir).
    func finishCapture() async throws -> CaptureBundle
    /// Abort and delete any partial on-disk data.
    func cancelCapture() async
}

/// Implemented by the SplatEngine module (Rust/Brush core via NimbusSplatCore.xcframework).
public protocol SplatTrainer: AnyObject, Sendable {
    /// Trains a Gaussian-splat model from a capture bundle, fully on-device.
    /// Long-running (minutes). Reports progress on the `splatTraining` stage.
    func train(_ bundle: CaptureBundle,
               config: SplatTrainingConfig,
               progress: @escaping ProgressHandler) async throws -> SplatModel
    /// Requests cancellation of an in-flight train(); train() then throws NimbusError.cancelled.
    func cancelTraining() async
}

/// Implemented by the SplatRender module (Metal). Not Sendable by design:
/// create and drive it from the thread/actor that owns the render loop
/// (typically MainActor via an MTKView delegate).
public protocol SplatRenderer: AnyObject {
    var isModelLoaded: Bool { get }
    /// Loads splat data into GPU buffers. Replaces any previously loaded model.
    func load(_ model: SplatModel) async throws
    func unload()
    /// Sorts (as needed) and rasterizes the loaded splats into the drawable.
    /// Matrices are standard right-handed view/projection; viewportSize in pixels.
    func render(to drawable: CAMetalDrawable,
                viewMatrix: simd_float4x4,
                projectionMatrix: simd_float4x4,
                viewportSize: SIMD2<UInt32>) throws
}

/// Implemented by the Mesh module: splat -> triangle mesh surface reconstruction
/// plus decimation and UV unwrap.
public protocol MeshExtractor: AnyObject, Sendable {
    func extractMesh(from model: SplatModel,
                     options: MeshExtractionOptions,
                     progress: @escaping ProgressHandler) async throws -> MeshAsset
}

/// Implemented by the Materials module: removes baked-in lighting from a
/// captured albedo texture so it can be relit under any environment.
public protocol Delighter: AnyObject, Sendable {
    /// - Parameters:
    ///   - albedoURL: lit (captured) albedo texture for `mesh`'s UV layout.
    ///   - environment: capture-time lighting, if available (improves separation).
    /// - Returns: URL of the new de-lit albedo texture on disk.
    func delight(albedoURL: URL,
                 mesh: MeshAsset,
                 environment: HDRIEnvironment?,
                 progress: @escaping ProgressHandler) async throws -> URL
}

/// Implemented by the Materials module (Core ML classifier).
public protocol MaterialClassifier: AnyObject, Sendable {
    /// Classifies the dominant surface material from an albedo texture.
    func classify(albedoURL: URL) async throws -> MaterialClassification
}

/// Implemented by the Materials module: synthesizes the remaining PBR maps
/// (normal/height/roughness/metallic/AO) from a de-lit albedo + material class,
/// and fills in class-appropriate parallax parameters.
public protocol DynamicTextureBuilder: AnyObject, Sendable {
    func buildMaterialSet(delitAlbedoURL: URL,
                          mesh: MeshAsset,
                          classification: MaterialClassification,
                          options: TextureBuildOptions,
                          progress: @escaping ProgressHandler) async throws -> MaterialSet
}

/// Implemented by the HDRI module: merges bracketed exposures into an
/// equirectangular OpenEXR environment map.
public protocol HDRICapture: AnyObject, Sendable {
    func assembleHDRI(from bundle: CaptureBundle,
                      options: HDRIAssemblyOptions,
                      progress: @escaping ProgressHandler) async throws -> HDRIEnvironment
}

/// Implemented by the Export module: writes glTF/GLB (+ .spz, + .exr) to disk.
public protocol AssetExporter: AnyObject, Sendable {
    /// `mesh.uvs` must be non-empty when `materials` is provided.
    func export(mesh: MeshAsset,
                materials: MaterialSet?,
                splat: SplatModel?,
                hdri: HDRIEnvironment?,
                options: ExportOptions,
                progress: @escaping ProgressHandler) async throws -> ExportedAsset
}
