# Module status

One row per module. One current status, one line of specifics. This file was
rewritten by the integrator on 2026-09-04: it had grown to fifty-odd kilobytes
of pass-by-pass history that nobody could read, and history belongs in
`SESSION_JOURNAL.md`, not here.

**What the words mean**

* **REAL** - the code does what the row says it does. Nothing in it is faked.
* **PARTIAL** - most of it is real, and the part that is not is named on the
  same line, not left for you to find.
* **STUB** - a typed placeholder behind the real contract, clearly labelled at
  the call site, never pretending to work.
* **WRITTEN, REACHABLE, NEVER YET EXECUTED** - the code is all there and
  something in the app now calls it, but no run of it has ever happened. Added
  on 2026-09-04 for the pre-pass, the trainer and the new pipeline module,
  because an audit was right that calling them "REAL" and "working" overstated
  what anyone could actually know: until that day nothing called them at all.

**One thing that is true of every row.** There is no macOS and no Xcode on the
machine this was written on, so no Swift here has ever been compiled. It was
written against real framework signatures and checked mechanically: every type,
function and shader name referenced across a module boundary was swept against
the declarations in `ios/Sources`, top-level type names were checked for
collisions (a duplicate is a hard build failure in a single-target app), and
Metal function names were matched both ways between the Swift and the `.metal`
files. That is not the same thing as a compiler saying yes. GitHub Actions is
the first thing that will actually compile this.

| Module | Status | What is real, and what is not |
|---|---|---|
| **core** (`ios/Sources/Core/`) | REAL | The shared vocabulary: every cross-module type and protocol, the brand config that reads the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit-to-COLMAP pose conversion, and `TrainingBudget.recommended` with real numbers. Nothing here is a placeholder. |
| **app-shell** (`ios/Sources/App/`) | REAL | `@main`, the first-run compatibility gate, the three tabs, and the single integration block that registers all nine modules. Every module is wired in and none of the registration lines is commented out any more. If a screen ever fails to register, the tab says so in plain words instead of showing a blank. |
| **onboarding** (`ios/Sources/Onboarding/`) | REAL | Probes ARKit, Metal, memory and the device model on first launch, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. An incompatible device is now routed to this module's own screen, which has the measurements behind the verdict and offers a re-check, rather than to the shell's shorter one. The chip lookup table's iPhone 17 rows are inferred from the naming pattern, not confirmed against shipped hardware, and say so in the file. |
| **capture** (`ios/Sources/Capture/`) | REAL | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a blur meter driven by the gyro, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. The post-capture quality card is the pre-pass's to produce and is called, never reimplemented. |
| **prepass** (`ios/Sources/PrePass/`) | WRITTEN, REACHABLE, NEVER YET EXECUTED | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection by pose proximity and depth ICP, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a full second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. `PrePassService.run(bundle:at:)` now has a caller: `ScanProcessingCoordinator` in `ios/Sources/Pipeline`, reached from the scan library. Only `quickQCCard` had one before, so the full pass had never been reachable. Exactly what that means and no more: the code is written, it is now reachable from a screen, and it has never been executed on hardware or compiled by any compiler. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, and the per-pixel authority map. STUB, and the only one in the iOS app: `SmartMonocularDepthStub`, the mid-range depth estimator. It returns nil and says so through `provenance`, because no Core ML export of a metric monocular depth model is bundled. `SmartAuthorityMap.swift` lists the four specific things a real one needs. |
| **trainer** (`ios/Sources/Trainer/`) | WRITTEN, REACHABLE, NEVER YET EXECUTED | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge for scenes too big for memory, and a governor that lowers the budget as the phone heats up or runs short. It writes `model.ply`, `model.json`, per-frame exposure, and the held-out frame list the review screen needs. `SplatTrainer.train(bundle:prePass:at:budget:)` now has a caller: `ScanProcessingCoordinator` in `ios/Sources/Pipeline`. It had none at all before, so not one of these 4,100 lines had ever been reachable. Exactly what that means and no more: written, now reachable from a screen, never executed on hardware and never compiled. Honest limit that is already known: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked. |
| **pipeline** (`ios/Sources/Pipeline/`) | WRITTEN, NEVER YET EXECUTED | The processing flow that was missing: `ScanProcessingCoordinator` drives `PrePassService.run` and then `SplatTrainer.train` for one scan, reuses an existing `prepass/` rather than redoing twenty minutes of work, sizes a `TrainingBudget` from the compatibility probe's tier (read back from the pre-pass pipeline, not copied), the scan's own measured extent and `os_proc_available_memory()`, surfaces the real stage names, keeps every budget reduction the trainer reports where the user can read it, compares the finished model's `budgetUsed` against what was asked for, and cancels for real. `ScanProcessingScreen` is the screen with the buttons, reached from the scan library, and it routes a finished model into the Viewer's existing review screen rather than building a second preview. Never executed: no phone has run it and no compiler has seen it. |
| **viewer** (`ios/Sources/Viewer/`) | REAL | The Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. The slider's held-out list now comes from the trainer's own record when a model was trained on the phone; for a model from the PC Booster it still falls back to every 20th frame and says on screen that this is a spot check, not proof. |
| **export** (`ios/Sources/Export/`) | PARTIAL | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the ratified `KHR_gaussian_splatting` extension plus the fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. Labelled stub: `.spz` version 4 read, which refuses with a named reason rather than guessing at a format nobody has published a sample of. |
| **booster-client** (`ios/Sources/Booster/`) | REAL | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, resumable download of the result, and the Booster tab. Pure Swift, no third-party dependencies. All twelve routes and every payload field name were re-checked against `docs/BOOSTER_PROTOCOL.md` and the Python server on 2026-09-04 and they agree. |
| **booster-pc** (`booster/`) | PARTIAL | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry - depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving, and `model.ply` / `model.spz` / `model.json` / a preview image the phone can download. Not written yet, and said plainly in the code: the photometric optimisation loop (`trainer/train.py`). Until it exists the PC gives you back the laser measurement as splats, with no optimisation sharpening it. It also needs an NVIDIA GPU with CUDA PyTorch and gsplat installed for that step, which is a real constraint of that library and is reported to the user in plain language rather than hidden. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `.spz` decode is written from the format description without a real sample file to test against, and says so. |
| **ci** (`.github/workflows/ios.yml`) | REAL, never yet run green | XcodeGen generates the project, `xcodebuild archive` builds it unsigned, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, discovering the product name by globbing so a rename cannot break it. Uploaded on every run, attached to a GitHub Release on a `v*` tag. A separate Ubuntu job byte-compiles and lints `booster/`, which passes today. The macOS job has never actually executed, because there is no macOS here. It is the first real compile this project will get. |

---

## The gap that was not inside any single module

**It is closed in code and open in evidence.**

The gap was that no screen started the pre-pass or the trainer, so 4,100 lines
of trainer and the whole pre-pass had never been reachable from anywhere in the
app. `ios/Sources/Pipeline` is that screen and the coordinator behind it: the
scan library pushes it for any scan without a model, and its buttons call
`PrePassService.run` and then `SplatTrainer.train`.

What is honestly claimed now: the route exists. A recorded scan has an on-device
path to a finished model, and each state in the library has a button that
advances it.

What is not claimed: that any of it works. Nothing on this machine can compile
Swift, so the pre-pass and the trainer have still never been executed, and now
they have never been executed while also being reachable. The first time either
one runs will be on a phone. Until that happens, the only route that has been
seen to produce a model is still the PC Booster.
