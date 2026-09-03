# Nimbus3D Contracts (single source of truth)

Every module implements against `Sources/Core/Contracts.swift`. Do not change a
name or signature there without updating this file and every conforming module.
No em dashes anywhere in code comments or copy.

## Pipeline flow

```
CaptureService                (Capture)      -> CaptureBundle
SplatTrainer                  (SplatEngine)  -> SplatModel        [Rust/Brush via NimbusSplatCore.xcframework]
SplatRenderer                 (SplatRender)  -> on-screen preview [Metal]
MeshExtractor                 (Mesh)         -> MeshAsset
Delighter                     (Materials)    -> de-lit albedo URL
MaterialClassifier            (Materials)    -> MaterialClassification  [Core ML]
DynamicTextureBuilder         (Materials)    -> MaterialSet
HDRICapture                   (HDRI)         -> HDRIEnvironment
AssetExporter                 (Export)       -> ExportedAsset
Pipeline module orchestrates all of the above via the protocols only.
```

## Module ownership

| Module (Sources/...) | Implements | Consumes | Produces |
|---|---|---|---|
| Capture | `CaptureService` | ARKit/AVFoundation | `CaptureBundle` (frames + poses + optional LiDAR depth + HDRI brackets, all on disk under `rootDirectory`) |
| SplatEngine | `SplatTrainer` | `CaptureBundle`; Rust Brush core (builds `Frameworks/NimbusSplatCore.xcframework`) | `SplatModel` (.ply/.spz on disk + metadata) |
| SplatRender | `SplatRenderer` | `SplatModel`; Metal (.metal shaders in `Sources/SplatRender/Shaders/`) | rendered frames into a `CAMetalDrawable` |
| Mesh | `MeshExtractor` | `SplatModel` | `MeshAsset` (positions/normals/uvs/indices + `isLowPoly`) |
| Materials | `Delighter`, `MaterialClassifier`, `DynamicTextureBuilder` | albedo URL + `MeshAsset` (+ optional `HDRIEnvironment`) | de-lit albedo URL; `MaterialClassification`; `MaterialSet` (PBR map URLs + class + POM params) |
| HDRI | `HDRICapture` | `CaptureBundle.hdriBrackets` | `HDRIEnvironment` (equirect .exr) |
| Export | `AssetExporter`; Library tab UI | `MeshAsset` + `MaterialSet?` + `SplatModel?` + `HDRIEnvironment?` | `ExportedAsset` (gltf/glb + PBR + .exr + .spz under one directory) |
| Pipeline | orchestration + Process tab UI | all protocols above (never concrete types) | end-to-end runs, `PipelineProgress` to UI |
| App | app shell (`NimbusApp.swift`) | placeholder tabs Capture / Process / Library | navigation shell modules fill in |
| Core | contracts only | nothing | `Contracts.swift` (this contract) |

## Shared data types (all in Contracts.swift, all Codable + Sendable)

- `CameraPose`: codable 4x4 camera-to-world (column storage; `.matrix` gives `simd_float4x4`).
- `CameraIntrinsics`: fx/fy, cx/cy in pixels + image size (matches ARKit layout).
- `AxisAlignedBoundingBox`: `minCorner`/`maxCorner` (`SIMD3<Float>`).
- `CapturedFrame`: image URL, optional depth + confidence URLs, pose, intrinsics, timestamp, exposure, ISO.
- `ExposureBracketFrame`: image URL, EV bias, exposure, ISO, pose, intrinsics.
- `CaptureBundle`: id, `rootDirectory`, `frames`, `hdriBrackets`, `hasLiDARDepth`, optional scene radius.
- `SplatModel`: splat file URL + `SplatFileFormat` (.ply/.spz), splat count, iterations, `sourceCaptureID`, bounding box.
- `MeshAsset`: `positions`/`normals`/`uvs`/`indices` arrays + `isLowPoly` + optional `sourceSplatID`.
- `MaterialClass` (enum): brick, wood, metal, stone, concrete, fabric, plastic, glass, ceramic, leather, foliage, ground, unknown.
- `MaterialClassification`: top class + confidence + `alternatives: [MaterialScore]`.
- `ParallaxOcclusionParameters`: enabled, heightScale, min/max samples.
- `MaterialSet`: albedo URL (required) + optional normal/height/roughness/metallic/AO URLs + classification + POM params + resolution.
- `HDRIEnvironment`: equirect .exr URL, width/height, optional dominant light direction + average luminance.
- `ExportedAsset`: `rootDirectory` + optional gltf/glb/hdri/splat URLs + optional `MaterialSet`.
- `PipelineProgress` + `PipelineStage` (enum of the 8 stages) + `ProgressHandler` typealias.
- Options structs: `CaptureOptions`, `SplatTrainingConfig`, `MeshExtractionOptions`, `TextureBuildOptions`, `HDRIAssemblyOptions`, `ExportOptions`.
- `NimbusError`: one case per stage + `.cancelled` + `.notImplemented(String)` for honest stubs.

## Protocol signatures (exact)

```swift
protocol CaptureService: AnyObject {
    var isCapturing: Bool { get }
    var events: AsyncStream<CaptureEvent> { get }
    func startCapture(options: CaptureOptions) async throws
    func finishCapture() async throws -> CaptureBundle
    func cancelCapture() async
}

protocol SplatTrainer: AnyObject, Sendable {
    func train(_ bundle: CaptureBundle, config: SplatTrainingConfig,
               progress: @escaping ProgressHandler) async throws -> SplatModel
    func cancelTraining() async
}

protocol SplatRenderer: AnyObject {   // NOT Sendable; drive from the render loop's actor
    var isModelLoaded: Bool { get }
    func load(_ model: SplatModel) async throws
    func unload()
    func render(to drawable: CAMetalDrawable, viewMatrix: simd_float4x4,
                projectionMatrix: simd_float4x4, viewportSize: SIMD2<UInt32>) throws
}

protocol MeshExtractor: AnyObject, Sendable {
    func extractMesh(from model: SplatModel, options: MeshExtractionOptions,
                     progress: @escaping ProgressHandler) async throws -> MeshAsset
}

protocol Delighter: AnyObject, Sendable {
    func delight(albedoURL: URL, mesh: MeshAsset, environment: HDRIEnvironment?,
                 progress: @escaping ProgressHandler) async throws -> URL
}

protocol MaterialClassifier: AnyObject, Sendable {
    func classify(albedoURL: URL) async throws -> MaterialClassification
}

protocol DynamicTextureBuilder: AnyObject, Sendable {
    func buildMaterialSet(delitAlbedoURL: URL, mesh: MeshAsset,
                          classification: MaterialClassification, options: TextureBuildOptions,
                          progress: @escaping ProgressHandler) async throws -> MaterialSet
}

protocol HDRICapture: AnyObject, Sendable {
    func assembleHDRI(from bundle: CaptureBundle, options: HDRIAssemblyOptions,
                      progress: @escaping ProgressHandler) async throws -> HDRIEnvironment
}

protocol AssetExporter: AnyObject, Sendable {
    func export(mesh: MeshAsset, materials: MaterialSet?, splat: SplatModel?,
                hdri: HDRIEnvironment?, options: ExportOptions,
                progress: @escaping ProgressHandler) async throws -> ExportedAsset
}
```

## Build rules

- Project is generated by XcodeGen from `project.yml` (run `xcodegen generate` on macOS/CI).
- iOS 17.0 minimum, iPhone only, bundle id `com.nimbus3d.app`, Swift 5.10.
- All Swift under `Sources/` compiles into the single app target; `.metal` files anywhere under `Sources/` are compiled into `default.metallib` automatically.
- The Rust core must exist at `Frameworks/NimbusSplatCore.xcframework` before `xcodebuild`; a pre-build script fails the build with a clear message if it is missing. The SplatEngine module owns the Rust crate and its build script; CI runs that step first.
- No agent may edit `Sources/Core/Contracts.swift`, `Sources/App/NimbusApp.swift` (beyond swapping placeholder views), or another module's directory.

## Honesty contract

Real capability gaps today: splat-to-clean-mesh surface reconstruction, delighting,
and neural material synthesis have no proven on-device implementation. Those must be
clean, typed stubs that throw `NimbusError.notImplemented(...)` (or return clearly
labelled placeholder data) with a `// TODO(nimbus): <what a real impl needs>` comment.
Never claim a stub works. Record REAL / PARTIAL / STUB in `MODULE_STATUS.md`.
