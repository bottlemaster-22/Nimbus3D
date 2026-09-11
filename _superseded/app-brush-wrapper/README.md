# Nimbus3D

**Turn a phone scan of a real object into a game-ready 3D asset, entirely on the device.**

Nimbus3D is an iOS (iPhone) app. You scan an object with the camera (and LiDAR
if you have it), and the app runs a full on-device pipeline: it trains a Gaussian
splat model of the object, reconstructs a triangle mesh, builds PBR textures,
optionally assembles an HDRI environment from the lighting you captured, and
exports a glTF/GLB you can drop into a game engine. No server, no upload.

This repo is honest about where the technology genuinely works today and where a
capability has no proven on-device implementation yet. Those gaps are typed stubs
that say so, never faked. See **[MODULE_STATUS.md](MODULE_STATUS.md)** for the
per-module truth and the "What is real vs stubbed" section at the bottom of this file.

---

## On-device architecture

Everything compiles into **one iOS app target**. Modules are folders under
`Sources/`, coordinated by one contract file. There are no Swift package
sub-modules, so **type names must be globally unique across folders.**

```
Capture ─▶ SplatEngine ─▶ (SplatRender preview)
                │
                ▼
             Mesh ─▶ HDRI ─▶ Delight ─▶ Classify ─▶ TextureBuild ─▶ Export
```

| Stage | Module | Protocol (in `Sources/Core/Contracts.swift`) | Tech |
|---|---|---|---|
| Guided scan -> `CaptureBundle` | `Capture` | `CaptureService` | ARKit + AVFoundation + LiDAR |
| Train Gaussian splats -> `SplatModel` | `SplatEngine` | `SplatTrainer` | Rust **Brush** engine (wgpu/Metal) via `NimbusSplatCore.xcframework` |
| Live 3D preview | `SplatRender` | `SplatRenderer` | Metal (EWA splatting, GPU sort) |
| Splat -> `MeshAsset` | `Mesh` | `MeshExtractor` | Density field + marching cubes + QEM + UV atlas (Metal/CPU) |
| Remove baked lighting | `Materials` | `Delighter` | Core ML slot (honest passthrough today) |
| Material class | `Materials` | `MaterialClassifier` | Vision + Core ML |
| Build PBR maps -> `MaterialSet` | `Materials` | `DynamicTextureBuilder` | ambientCG CC0 library substitution + tint + POM |
| Brackets -> `HDRIEnvironment` (.exr) | `HDRI` | `HDRICapture` | Debevec merge + equirect projection + OpenEXR writer |
| Write glTF/GLB -> `ExportedAsset` | `Export` | `AssetExporter` | Deterministic glTF 2.0 / GLB, ORM packing |
| Orchestrate the whole run | `Pipeline` | (uses protocols only) | Actor job runner + SwiftUI Process/Library tabs |

**Contract rule:** modules depend only on the protocols and shared types in
`Sources/Core/Contracts.swift`, never on each other's concrete types. That is what
lets a real implementation and an honest stub be swapped freely.

### App shell and dependency injection

- `Sources/App/NimbusApp.swift` is the entry point. Three tabs: **Capture**
  (`CaptureRootView`), **Process** (`ProcessRootView`), **Library** (`LibraryRootView`).
- `Sources/App/NimbusServices.swift` runs once at launch and registers each
  module's concrete service into `PipelineServices.shared` (the DI container).
  Before registration, every slot holds an honest "not wired" stub that throws
  `NimbusError.notImplemented` naming the missing module, so an unwired stage is
  reported, never faked.
- The `Pipeline` orchestrator reads `PipelineServices.shared` when a run starts.
  Required stages (training, mesh, export) are fatal on error; optional stages
  (HDRI, delight, classify, texture) skip or degrade honestly and the run continues.

### On-disk layout (all under the app's Documents)

```
Documents/
  Captures/<uuid>/manifest.json + frames/…      (Capture output)
  Models/<uuid>/model.ply                        (SplatEngine output)
  Processing/<captureId>/…                        (pipeline scratch)
  Exports/<assetId>/  asset.json + manifest.json + model.glb + textures + environment.exr + splat.ply
```

The stores rebase stale absolute URLs, because iOS moves the app container
between launches.

---

## How to build

You need **macOS + Xcode** and the **Rust** toolchain. The project is generated
from `project.yml` by [XcodeGen], and the splat trainer core is a Rust crate
compiled into an `.xcframework`. `xcodebuild` cannot run on a non-macOS machine;
use the CI workflow or a Mac.

### 1. Build the Rust core (produces the xcframework the app links)

```bash
# from the app directory (the folder with project.yml)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
bash Sources/SplatEngine/build-xcframework.sh
# -> Frameworks/NimbusSplatCore.xcframework
```

A pre-build script in `project.yml` fails the Xcode build with a clear message if
`Frameworks/NimbusSplatCore.xcframework` is missing, so this step must run first.

### 2. Generate the Xcode project and build

```bash
brew install xcodegen
xcodegen generate          # -> Nimbus3D.xcodeproj
open Nimbus3D.xcodeproj     # build/run on an ARKit-capable iPhone (iOS 17+)
```

Camera + ARKit usage strings and the `arkit`/`metal` device requirements are
generated into `Support/Info.plist` from the `info:` block in `project.yml`.

### 3. CI (the supported path)

`.github/workflows/ios.yml` does all of the above on a `macos-15` runner and
produces an **unsigned** IPA:

1. `build-xcframework.sh` (cached cargo builds for device + simulator),
2. `xcodegen generate`,
3. `xcodebuild archive` with `CODE_SIGNING_ALLOWED=NO`,
4. hand-zips `Payload/Nimbus3D.app` into `Nimbus3D-unsigned.ipa` (uploaded as the
   `Nimbus3D-unsigned-ipa` artifact; attached to a Release on `v*` tags).

See **[BOTTLE_HOOKS.md](BOTTLE_HOOKS.md)**. The workflow assumes the app directory
is the git repo root; move it to `<repo-root>/.github/workflows/` if the app is
nested.

> **Current build reality:** the Swift/Metal side is written to compile. The Rust
> step is the gating one: it builds directly against the Brush engine, and a few
> FFI seams in `Sources/SplatEngine/rust/.../brush_glue.rs` need to be reconciled
> against a real Brush revision before the first green build. There is no offline
> stub for the trainer core, by design.

---

## What is real vs stubbed right now

Honest summary. Full detail per module in [MODULE_STATUS.md](MODULE_STATUS.md).

**Real and working (as written; not yet device-verified here):**

- **Capture** ARKit posed-frame + LiDAR-depth capture and bundle persistence.
- **SplatRender** Metal Gaussian-splat renderer (parse, GPU sort, EWA raster, orbit preview).
- **HDRI** bracket merge -> equirectangular OpenEXR, with light/luminance analysis.
- **Mesh** the full splat->mesh chain (density field, marching cubes, QEM decimation, box UV atlas).
- **Export** deterministic glTF/GLB writing with ORM texture packing and a self-describing manifest.
- **Pipeline** orchestration, progress, failure policy, and the Process/Library UI.
- **App** shell + dependency wiring; **CI** workflow.

**Honest stubs / limits (labelled in code with `// TODO(nimbus:` and reported at runtime):**

- **SplatEngine (trainer):** wired for real against Brush, but Rust FFI seams and a
  few config fields need a first on-device build to reconcile; the crate does not
  compile offline yet.
- **Delighting (Materials):** no on-device model ships. It passes the captured
  albedo through unchanged and reports that lighting was **not** removed. Removing
  baked-in lighting has no proven self-contained on-device implementation today.
- **Neural material synthesis (Materials):** fine surface relief is **substituted**
  from the bundled CC0 library and tinted to the captured color, never extracted
  from the splat. No trained material classifier model ships, so classification is
  `.unknown` at 0 confidence until one is bundled.
- **Clean-mesh reconstruction (Mesh):** marching cubes over a Gaussian *density
  envelope* is watertight-ish but puffy; the state-of-the-art clean-mesh methods
  (2DGS / GOF / MILo) are CUDA-only with no on-device port.
- **Captured-albedo bake (Pipeline):** a flat mean-capture-color placeholder,
  because the contract defines no UV-space projection-bake stage yet.
- **PLY -> SPZ (Export):** deferred; a `.ply` splat is copied through uncompressed
  and flagged as not-SPZ. **USDZ** export is real but gated on OS capability.

Nothing above is presented as working when it is not. If you hit a stubbed stage
during a run, the Process screen shows it as **Skipped** with the exact reason.

[XcodeGen]: https://github.com/yonaskolb/XcodeGen
