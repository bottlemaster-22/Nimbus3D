# SplatEngine

On-device 3D Gaussian Splat training. Implements the `SplatTrainer` protocol
(`Sources/Core/Contracts.swift`): `CaptureBundle` in, `SplatModel` (.ply on disk)
out, with progress and cancellation.

Training runs on-device via the **Brush** engine
(github.com/ArthurBrussee/brush), which trains 3D Gaussian Splats on the GPU with
`burn` + `wgpu` (Metal on iOS). A thin Rust crate wraps Brush behind a C ABI; a
build script packages it as `Frameworks/NimbusSplatCore.xcframework`, which
`project.yml` embeds and code-signs.

## Layout

```
Sources/SplatEngine/
  BrushSplatTrainer.swift          SplatTrainer impl: bundle -> dataset -> Rust core -> SplatModel
  NerfstudioDatasetExporter.swift  CaptureBundle -> transforms.json + images (Brush-readable)
  PLYBoundingBox.swift             reads the exported .ply back for AABB + splat count
  build-xcframework.sh             builds the Rust core into NimbusSplatCore.xcframework
  rust/nimbus-splat-core/
    include/nimbus_splat_core.h    C ABI (imported by Swift as `import NimbusSplatCore`)
    src/lib.rs                     FFI boundary: opaque handle, panic containment, marshalling
    src/brush_glue.rs              real Brush integration (create_process streaming loop)
    Cargo.toml                     staticlib + cdylib; depends on brush-process (git)
```

## Building the Rust core

CI (`.github/workflows/ios.yml`) runs this from the `app/` directory **before**
`xcodegen generate` / `xcodebuild`:

```sh
bash Sources/SplatEngine/build-xcframework.sh
```

Requirements (macOS only): `rustup` with targets `aarch64-apple-ios` and
`aarch64-apple-ios-sim`, `cargo`, and Xcode (for `xcodebuild -create-xcframework`).
The script produces `app/Frameworks/NimbusSplatCore.xcframework`. Until it has run,
the app's pre-build script fails with a clear error (by design).

## Data flow

1. `NerfstudioDatasetExporter` writes `transforms.json` (PINHOLE, per-frame ARKit
   intrinsics + camera-to-world pose) plus copied frame images into a cache dir.
   No COLMAP structure-from-motion is needed: ARKit already supplies world-space poses.
2. `BrushSplatTrainer` calls `nimbus_trainer_run()` on a dedicated thread, bridging
   the C progress callback to `ProgressHandler` and Task/`cancelTraining()` to the
   core's atomic cancel flag.
3. `brush_glue::run_training` drives Brush's `create_process` message stream to
   completion and lets Brush's exporter write the final `.ply`.
4. `PLYBoundingBox` reads that `.ply` back for the authoritative bounding box and
   splat count stored on `SplatModel`.

## Honesty: what is real vs. seam

REAL and compiling as written: the FFI boundary, panic containment, cancellation,
the Nerfstudio exporter, the PLY reader, the Swift wrapper, and the xcframework
build. The Brush call sites are the genuine integration but carry `SEAM(...)` /
`TODO(nimbus)` markers where an exact Brush API detail could not be verified
offline and must be confirmed on the first networked macOS/device build:

- **Brush API surface** (`brush_glue.rs`): `burn_init_setup`, `DataSource::Path`,
  `create_process`, `TrainStreamConfig`/`ProcessConfig` field names, and the
  `ProcessMessage`/`TrainMessage` variants. Written against Brush `main` as
  researched; reconcile against a pinned revision.
- **Config knobs not yet applied**: `sh_degree`, `max_splat_count`, and
  `max_image_dimension` are accepted by the FFI but not yet wired onto Brush's
  config (marked in the header and `brush_glue.rs`). Training currently uses
  Brush's own densification/limits for these.
- **Coordinate convention** (`NerfstudioDatasetExporter.swift`): ARKit and
  Nerfstudio share the OpenGL camera convention, so the pose is emitted unchanged;
  validate on-device and flip Y/Z if the reconstruction comes out mirrored.
- **HEIC images**: if capture emits HEIC, Brush's image loader may need JPEG/PNG;
  transcode at the marked seam in the exporter.
- **Dependency pin**: `brush-process` tracks `branch = "main"`; pin `rev` after the
  first green build for reproducible CI.
