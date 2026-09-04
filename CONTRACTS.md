# Contracts

The foundation every module builds against. If you are an agent about to write
code in this repo, read this first and then read
`ios/Sources/Core/Contracts.swift`, which is the machine-readable version of
everything below.

---

## 1. Hard build rules

These are not preferences. Breaking one of them breaks CI, and CI producing an
installable unsigned IPA is the whole point of the setup.

1. **Pure Swift and Metal on iOS. No Rust, ever.** The previous attempt at this
   product died on a Rust cross-compile in CI. `ios/project.yml` declares
   `packages: {}` and it stays that way: no Swift Package, no CocoaPods, no
   Carthage, no xcframework, nothing to resolve and nothing to cross-compile.
2. **No cloud, no paid services, no accounts.** The only network code in the
   app is the LAN Booster client, and it is optional.
3. **This is our own trainer.** Not a wrapper around Brush, Nerfstudio,
   Postshot or LichtFeld. On the PC side only, the `gsplat` CUDA rasterisation
   *library* (Apache-2.0) may be used as kernels underneath our own trainer
   logic. A library, not a pipeline.
4. **One brand config.** The product name is a working name. It is written in
   exactly two places, `ios/project.yml`'s `settingGroups.brand` block and, for
   the Blender add-on which cannot read it, `blender_addon/brand.py`. No other
   Swift, Metal, Python or Markdown file may contain the product name as a
   string literal. Read it from `BrandConfig`.
5. **iOS 17+, iPhone only, Swift language mode 5.** `SWIFT_VERSION` is `5.0`
   because that setting is the language *mode* and only accepts 4.0 / 4.2 /
   5.0 / 6.0. Xcode 16 ships a Swift 6 compiler running in mode 5, which is
   what "Swift 5.10" means in practice here.
6. **Every module writes only inside its own directory.** The one exception is
   appending your own row to `MODULE_STATUS.md`.
7. **Honesty contract.** Write real, compiling code wherever the technology
   supports it. Where a capability genuinely needs an unavailable model or
   asset, implement a clean typed stub behind the contract that returns a
   clearly labelled result plus a `TODO(nimbus): <what a real impl needs>`
   comment. Never fake functionality. Never claim a stub works.

---

## 2. The single-target reality

The app is **one Xcode target**. "Module" means "a directory under
`ios/Sources` with one owning agent", not a Swift module. There is no
`import Core`, no access-control wall between directories, and, crucially, **no
namespacing**: two files anywhere in the project may not declare the same
top-level type name, or the build fails with a duplicate-symbol error.

So: before you declare a public type, check that the name is free.

```bash
grep -rn "struct YourName\|class YourName\|enum YourName\|protocol YourName" ios/Sources
```

Fixed concrete type names are in section 5. Do not use a name assigned to
another module.

---

## 3. Module ownership

| Directory | Owns | Depends on |
|---|---|---|
| `ios/Sources/Core` | Contracts, brand config, geometry conventions, service registry | nothing |
| `ios/Sources/App` | `@main`, compatibility gate, tab bar, module wiring | Core, and whichever modules are present |
| `ios/Sources/Onboarding` | Device detection, tier assignment, first-run screens | Core |
| `ios/Sources/Capture` | ARKit session, sidecar logging, coverage HUD, haptics and audio guidance | Core |
| `ios/Sources/PrePass` | Submaps, revisits, pose graph, time-offset calibration, carving, edges, trust, QC card | Core |
| `ios/Sources/Trainer` | Metal 3DGS rasteriser and optimiser, budget and thermal control | Core, Smart |
| `ios/Sources/Smart` | Losses, background model, trust field, edge classifier | Core |
| `ios/Sources/Viewer` | Preview renderer, honesty mask, A/B slider, artefact heatmap, scan library | Core, Export |
| `ios/Sources/Export` | PLY / SPZ / GLB writers and readers, capture-bundle zip | Core |
| `ios/Sources/Booster` | Bonjour discovery, pairing, chunked upload, job stream, download | Core |
| `booster/` | Python PC companion: LAN listener, our trainer on gsplat kernels, PySide6 GUI | `docs/BOOSTER_PROTOCOL.md`, `docs/DATA_FORMAT.md` |
| `blender_addon/` | Blender importer for our `.ply` / `.spz` | `docs/DATA_FORMAT.md` |

`Core`, `App`, `Export` and `Booster` are **not** optional in
`ios/project.yml`: if one of those goes missing, CI must fail loudly. The rest
are marked `optional: true`, which means only "skip this path if the directory
is absent" so that a not-yet-written module cannot break `xcodegen generate`
for everybody.

**`optional: true` is not a safety net for a directory that *does* exist.** The
moment a module's directory is on disk, every Swift file in it is compiled into
the single app target. A half-written module therefore breaks the whole build,
exactly as if it were required. See section 6.7.

### What is actually on disk (audited 2026-09-04)

| Directory | On disk | Compiles | Wired into the app |
|---|---|---|---|
| `Core` | yes | yes | n/a |
| `App` | yes | yes | n/a |
| `Export` | yes | yes | yes |
| `Booster` | yes | yes | yes |
| `Onboarding` | yes | **no** - 3 undeclared types | no |
| `Capture` | yes (helpers only) | yes | no - no `ARCaptureService` |
| `PrePass` | yes (math + sensor IO only) | yes | no - no `PrePassPipeline` |
| `Smart` | yes (`NativeDepthEdgeClassifier`) | yes | n/a - not a registry service |
| `Viewer` | yes (GPU layouts + shaders) | yes | no - no `MetalSplatRenderer` |
| `Trainer` | **no** | n/a | no |

---

## 4. The contract types

All in `ios/Sources/Core/Contracts.swift`. Everything is `Codable` where it
touches disk or the wire, and `Sendable` throughout.

### Geometry and conventions

| Type | What it is |
|---|---|
| `ScanID` | `String`. The scan's folder name, `scan_YYYYMMDD_HHMMSS`. |
| `FrameID` | `UInt32`. Capture-order frame index; the join key for every sidecar. |
| `SubmapID` | `UInt32`. |
| `Vector3`, `Quaternion` | Codable simd bridges. Encode as `[x,y,z]` and `[x,y,z,w]`. |
| `Pose` | Rigid **world to camera**: `X_cam = R * X_world + t`. |
| `BoundingBox`, `CameraIntrinsics` | Axis-aligned box; single shared PINHOLE camera. |
| `ContractsJSON` | The one JSON encoder/decoder configuration for every sidecar. |

**World frame:** right-handed, Y up, metres, ARKit's, unchanged.
**Camera frame:** +X right, +Y **down**, +Z **forward**. COLMAP/OpenCV, not
ARKit's.
**The single conversion** is `Pose.fromARKitCameraTransform(_:)`. Nowhere else
in this project may negate an axis. If you are writing a stray minus sign to
make something line up, the bug is elsewhere.

### Capture

`CaptureBundle` is the index of one scan, serialised to
`capture_bundle.json`. It holds `[CaptureFrame]` (image, native depth and
confidence paths, raw pose, refined pose, exposure, ISO, gyro, `FrameQC`
weight, bracket flag, submap), `CameraIntrinsics`, `CaptureSettings`, the
camera-to-IMU `cameraToIMUTimeOffsetSeconds`, `anchorsDuringSession` and
`anchorsAtEndOfSession` (`AnchorRecord`), `meshChunks` (`MeshChunkRef` plus
per-face `SurfaceClass`), `revisitPairs` (`RevisitPair`), and the scene bounds.
`CaptureBundleRef` pairs it with the one absolute URL needed to open a file.

Supporting enums: `TrackingQuality`, `ExposureBracket`, `SurfaceClass`,
`RevisitMethod`.

### Pre-pass

`PrePassResult` is serialised to `prepass/prepass_result.json` and holds
`[Submap]`, `refinedPoses` (keyed by `String(frameIndex)`), the calibrated time
offset, `OccupancyGridRef` (with `OccupancyState` = `unknown` / `empty` /
`surface`), `TrustFieldRefs` (coarse bias field plus never-averaged per-sample
noise field, plus depth affine and recalibrated confidence),
`EdgeClassificationRefs` (with `EdgeClass` = `none` / `geometric` / `texture` /
`unknown` / `band`), `[GlassRegion]`, `InitialSplatSetRef`, a `QCCard`
(`QCFinding` list included), and a `suggestedBudget`.

### Device and budget

`DeviceTier` (`full` / `limited` / `incompatible`), `DeviceCapabilityReport`
(LiDAR presence, chip, RAM, available memory, iOS version, Metal family,
sustained-performance class, a `[FeatureAvailability]` list, and, for the
incompatible case, the plain-language `incompatibleReason`).

`TrainingBudget` (splat cap, iterations, render long edge, SH degree, keyframe
count, memory ceiling, `ThermalPolicy`, fp16, sparse Adam, `TrainingTarget`)
plus `TrainingBudget.recommended(for:sceneExtentMeters:availableMemoryBytes:)`,
which produces a real starting budget in the ranges the spec fixes.
`ThermalLevel` wraps `ProcessInfo.ThermalState` so it can be stored.

**The budget is a ceiling, not a target.** A trainer may lower it live as it
measures real memory and heat. It may never raise it.

### Model, progress, preview, export

`SplatModel` (the `model/model.json` index: ply/spz refs, splat count, SH
degree, bounds, iterations, budget used, background, exposure, observed
directions, held-out PSNR), `ModelSource`, `TrainerStage`, `TrainerProgress`,
`PreviewCameraPath`, `ExportedAsset`.

`TrainerProgress.fractionComplete` is optional on purpose. Omit it when you do
not know. The UI draws an indeterminate spinner for `nil`, which is honest, and
a bar frozen at 0%, which is not.

### Booster

`BoosterJob` is the app-level view of a job. The wire types
(`BoosterManifest`, `BoosterProgressEvent`, `BoosterJobStage`, `BoosterInfo`,
the pairing and job bodies) already live in
`ios/Sources/Booster/BoosterProtocol.swift` and are specified byte-for-byte in
`docs/BOOSTER_PROTOCOL.md`.

### Errors

`NimbusError` is what crosses a module boundary. Each module keeps its own
richer type (`ExportError`, `BoosterError`) for its internals.

---

## 5. The protocols, and who implements what

Every protocol is in Core. Every concrete type name is reserved for its owning
module and **must not be declared anywhere else**.

| Protocol (Core) | Concrete type | Owner |
|---|---|---|
| `DeviceCompatibilityService` | `DeviceCompatibilityProbe` | Onboarding |
| `CaptureService` | `ARCaptureService` | Capture |
| `PrePassService` | `PrePassPipeline` | PrePass |
| `PoseRefiner` | `SubmapPoseRefiner` | PrePass |
| `FreeSpaceCarver` | `VoxelFreeSpaceCarver` | PrePass |
| `EdgeClassifier` | `NativeDepthEdgeClassifier` | Smart |
| `TrustField` | `TwoScaleTrustField` | Smart |
| `BackgroundModel` | `DirectionalBackgroundModel` | Smart |
| `SplatTrainer` | `MetalSplatTrainer` | Trainer |
| `SplatRenderer` | `MetalSplatRenderer` | Viewer |
| `SplatExporting` | `ExportService` (already exists) | Export |
| `BoosterService` | `BoosterClient` (already exists) | Booster |

Signatures are in `Core/Contracts.swift` and are the authority; this table is
just the map. Async shapes worth knowing without opening the file:

* `PrePassService.run(bundle:at:)` returns
  `AsyncThrowingStream<PrePassResult, Error>`: it yields a partial result as
  soon as the QC card exists (the card must be on screen in under three
  seconds) and keeps yielding as the heavier stages land.
* `SplatTrainer.train(bundle:prePass:at:budget:)` returns
  `AsyncThrowingStream<TrainerProgress, Error>`, then `finishedModel()` and
  `snapshot()` give you the result and a live preview cloud.
* `CaptureService` and `SplatRenderer` and `BoosterService` are `@MainActor`.
  The rest are not.

### Registering a module

Two registries, both in Core:

* `NimbusServices.shared` holds the service implementations.
* `NimbusUI.shared` holds screens, as `() -> AnyView` closures, so the app
  shell can show a module's screen without knowing that module's type names.

Both are filled in from exactly one place: the **integration block** at the top
of `ios/Sources/App/NimbusApp.swift`. When your module lands, uncomment its
lines there. That block is deliberately one visible list rather than a
scattering of registration side effects nobody can find, and it doubles as an
honest at-a-glance answer to "what is actually built?".

Anything left `nil` renders as a placeholder that names the module which will
fill it. That is not a bug to route around; it is the design.

---

## 6. Reconciliation: what the already-built modules need

`ios/Sources/Export` and `ios/Sources/Booster` were written **before** these
contracts existed and had to invent their own. They were read line by line
before this file was written, and the contracts were shaped around them.

**Nothing below is required for the build to work today.** The app compiles and
runs as-is. These are the exact, small diffs an integrator may apply later, and
the reasoning, so nobody has to rediscover it.

### 6.1 Types promoted rather than re-declared

Because a duplicate type name is a hard compile error, Core does **not**
re-declare types those modules already own. It references them, and they are
part of the contract surface where they sit:

| Type | Declared in | Status |
|---|---|---|
| `SplatCloud`, `SHDegree`, `ExportError`, `ExportFormat` | `Export/SplatCloud.swift`, `Export/ExportService.swift` | Promoted to contract types in place |
| `BoosterJobStage`, `BoosterProgressEvent`, `BoosterManifest`, `BoosterManifestFile`, `BoosterInfo`, pairing and job bodies, `BoosterAPI` | `Booster/BoosterProtocol.swift` | Promoted; `docs/BOOSTER_PROTOCOL.md` is the normative spec |
| `BoosterDevice`, `BoosterError` | `Booster/` | Promoted |

**Optional later cleanup:** move `SplatCloud.swift` from `Sources/Export` to
`Sources/Core` unchanged. It is a file move with no content change and no
call-site change, because there is no `import` to update in a single target. Do
it when Export's agent is not mid-flight, or never; it costs nothing to leave.

**One conformance that Core owns:** `SHDegree` is declared without `Codable`,
and `TrainingBudget` and `SplatModel` both carry one and both serialise, so
`Contracts.swift` adds `extension SHDegree: Codable {}`. Export **must not**
also declare that conformance: two conformances of the same type to the same
protocol is a hard compile error.

### 6.2 Two protocol names that had to differ

| Task's name | Core's name | Why |
|---|---|---|
| `ExportService` | `SplatExporting` | `ExportService` is already Export's concrete class, and `ExportServicing` is already Export's own local protocol. Taking either name would force a rename inside shipped code. |
| `BoosterClient` | `BoosterService` | `BoosterClient` is already Booster's concrete class. |

Both concrete classes conform via a short extension in
`Sources/App/NimbusApp.swift`. Nothing was deleted, renamed or moved.

`ExportService` keeps its own `ExportServicing` protocol; the two coexist
happily. If Export's agent ever wants to tidy up, deleting `ExportServicing`
and conforming directly to `SplatExporting` is a safe, self-contained change.

### 6.3 The zip versus per-file question, settled

Export ships `packageForBooster(scanID:)`, which zips a scan folder. Booster
ships a per-file, chunked, resumable manifest upload. Both modules flagged the
overlap honestly and neither could resolve it without this document.

**Decision: the per-file chunked manifest transfer in
`docs/BOOSTER_PROTOCOL.md` is the Booster transport.** A zip cannot be resumed
part-way, has to be fully written to the phone's storage before the first byte
leaves (doubling peak disk use on a multi-gigabyte house scan), and cannot be
verified until it has been unpacked.

The zip keeps a real and different job: a single file the user can AirDrop,
back up, or carry to a PC by hand when there is no Wi-Fi. Its reconciliation is
therefore a **rename only**, `packageForBooster` to `packageCaptureBundle`, and
Core's `SplatExporting.packageCaptureBundle` already exposes it under the
better name, so no caller has to change.

### 6.4 Booster job record

`BoosterJobRecord` (Booster's on-disk form) and `BoosterJob` (Core's app-level
form) have identical fields. `NimbusApp.swift` maps one to the other in six
lines. Collapsing them into one type is optional tidying, not a fix.

### 6.5 Scan library

`BoosterScanLister` reads scan folders directly off disk because there was no
shared library type. That is still true and still fine:
`Sources/Viewer` will own the real library screen, and when it does, the
Booster tab's scan picker can switch to it. Until then the folder scan is
correct, because folders are exactly what the format defines.

### 6.6 Blender add-on

`blender_addon/splat_data.py` guessed the PLY property convention. It guessed
right: it matches `Export/PLYCodec.swift` exactly (`rot_0..3` as `(w,x,y,z)`,
natural-log scales, logit opacity, channel-major `f_rest_*`). Section 8 of
`docs/DATA_FORMAT.md` now states it normatively. No change needed.

Its `.spz` reader remains unverified against a real sample and is honestly
flagged PARTIAL. Our own `SPZCodec` writes version 3; the add-on should be
tested against a file our exporter actually produced, once one exists.

### 6.7 The one thing standing between this repo and a green CI run

Everything in 6.1 to 6.6 is optional tidying. **This is not.**

`ios/Sources/Onboarding` exists but does not compile. `DeviceCompatibilityProbe`
and its supporting probes are real and good; the module was simply cut off
before its copy and persistence files were written. Three types are called and
never declared anywhere in the project:

| Missing type | Called from `Onboarding/DeviceCompatibilityProbe.swift` at |
|---|---|
| `OnboardingCopy` | lines 75, 234, 397, 455, 473, 528 |
| `OnboardingFormat` | line 527 |
| `DeviceReportStore` | lines 267, 285 |

From the call sites, what they have to be:

* `OnboardingCopy` - a caseless `enum` of static functions returning the
  plain-language sentences: `lastResortReason` (a `String`),
  `incompatibleReason(_ context: ReasonContext) -> String`,
  `noScannerFeatureDetail(_ context: ReasonContext) -> String`,
  `trainingFeatureDetail(tier: DeviceTier, context: ReasonContext) -> String`,
  `highDetailFeatureDetail(tier: DeviceTier, context: ReasonContext) -> String`,
  `storageDetail(freeBytes: UInt64) -> String`. `ReasonContext` already exists
  at `DeviceCompatibilityProbe.swift:364`. This is where the owner's exact
  requirement lives: warm, specific, never generic, and it must name the actual
  missing thing on **that** phone.
* `OnboardingFormat` - `bytes(_:) -> String`, a human byte formatter.
* `DeviceReportStore` - a `shared` singleton with
  `save(_ findings: DeviceCompatibilityFindings)` and
  `load() -> DeviceCompatibilityFindings?`, persisting the report so the
  onboarding screens do not re-probe.

There is also no `OnboardingFlowView` yet, which is why
`NimbusUI.shared.onboardingFlow` stays nil and the app shell falls back to
`MinimalCompatibilitySummaryView`.

**Two ways to a green build, and only these two:**

1. **Finish the module** (correct). Onboarding's agent writes those three types
   plus `OnboardingFlowView`, then uncomments its lines in the integration
   block in `NimbusApp.swift`.
2. **Remove the directory** (stopgap). Delete `ios/Sources/Onboarding`; the
   `optional: true` entry in `project.yml` then genuinely skips it, and the app
   runs with `deviceCompatibility` nil and no first-run gate.

Do **not** try to fix this by editing the integration block or `project.yml`.
Neither is the cause. Nothing in `Core`, `App`, `Export` or `Booster` needs to
change.

### 6.8 One stale comment, no code impact

`ios/Sources/Export/ZipWriter.swift:73` gives
`"images/frame_000123_143001_500.jpg"` as an example entry name. That is the
old `frame_<index>_<HHMMSS>_<mmm>` shape, not the format's
`frame_<YYYYMMDD>_<HHMMSS>_<mmm>`. It is an illustrative doc comment only -
`CaptureScanFolder.swift` builds the real stamp correctly and the on-disk format
is consistent. Worth a one-line fix by Export's agent; nothing depends on it.

---

## 7. Where the numbers live

Values the whole product agrees on, so nobody has to re-derive them:

| Thing | Value | Where |
|---|---|---|
| Native depth map | 256 x 192, `UInt16` mm, little-endian | `DATA_FORMAT.md` section 5 |
| LiDAR useful range | ~5 m | `CaptureSettings.lidarMaxRangeMeters` |
| Occupancy voxel | ~5 cm | `OccupancyGridRef.voxelSizeMeters` |
| Trust bias voxel | 25 to 50 cm | `TrustFieldRefs.biasVoxelSizeMeters` |
| Edge dilation | upsample ratio, ~7.5 native to RGB px | `EdgeClassificationRefs` |
| Blur meter | `deg/s * exposure / 0.0426`, amber 2 px, red 4 px | `FrameQC.motionBlurPixels` |
| Submap window | 15 to 30 s, 20 to 30% overlap | `Submap` |
| Time-offset sweep | -50 to +50 ms, 5 ms steps | `PoseRefiner.calibrateTimeOffset` |
| Near / mid / far | 0 to 4.5 m, 4.5 to 30 m, beyond | `BackgroundModel.authority` |
| Preview FOV | 100 to 110 deg, path deviation <= 0.20 m | `PreviewCameraPath` |
| Booster chunk | 1 MiB | `BoosterAPI.chunkSize` |
| Booster port | 8760 | `BrandConfig.boosterDefaultPort` |

---

## 8. Documents

| File | What it settles |
|---|---|
| `CONTRACTS.md` | This file: ownership, names, build rules, reconciliation |
| `docs/DATA_FORMAT.md` | Every filename, byte layout and coordinate convention |
| `docs/BOOSTER_PROTOCOL.md` | The phone-to-PC wire protocol, byte for byte |
| `MODULE_STATUS.md` | Honest per-module status. Append your own row; never edit another module's |
| `ios/project.yml` | The build. The brand block at the top is the only place the product is named |
