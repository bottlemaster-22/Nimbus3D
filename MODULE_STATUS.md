# Module status

One row per module, one current status, one line of specifics. No history: that
belongs in `SESSION_JOURNAL.md`. Rewritten from the source on 2026-09-04 by the
auditor, who read every claim below against the file it describes.

---

## Read this before you read any row

**Nothing in this repository has ever been through a Swift compiler, and
nothing in it has ever run on a phone.** There is no macOS and no Xcode on the
machine it was written on. Every row below describes code that has been written
and read, and checked mechanically: every cross-module type and function name
swept against its declaration, top-level type names checked for collisions,
Metal entry points matched both ways between the `.metal` files and the Swift
that names them, and all eleven source directories confirmed present in
`ios/project.yml`. That is not a compiler saying yes, and it is not a phone
saying yes. GitHub Actions (`.github/workflows/ios.yml`) is the first thing
that will actually compile this, and it has never run because it has never had
a reason to.

So do not read "COMPLETE" below as "working". It means "everything this row
claims is written, and none of it is faked". Whether it runs is unknown for
every single row, the COMPLETE ones included.

**The three words, and they mean the same thing in every row:**

* **COMPLETE** - everything the row claims is written, and nothing in it is a
  placeholder pretending to work.
* **PARTIAL** - most of it is written, and the part that is not is named on the
  same line rather than left for you to find.
* **STUB** - a typed placeholder behind a real contract, labelled at the call
  site, returning nothing and saying so.

The "Reached from" column is separate from the status on purpose. Code can be
COMPLETE and reachable by nothing, which is the state most of this project was
in until recently, and several entries are still in it today.

---

| Module | Status | Reached from | What is real, and what is not |
|---|---|---|---|
| **core** (`ios/Sources/Core/`) | COMPLETE | Every other module | The shared vocabulary: the cross-module types and protocols, `BrandConfig` reading the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit-to-COLMAP pose conversion, `TrainingBudget.recommended`. Nothing here is a placeholder. Two known gaps in the contracts, both worked around at the call site and both filed in `INTEGRATION_REQUESTS.md`: `SplatTrainer` has `cancel()` and no way to ask whether a run has actually ended, so Pipeline downcasts to `MetalSplatTrainer` for `isTraining` and `waitUntilIdle()`; and `TrainerProgress` has no field for "I lowered the budget", so Pipeline recognises one by matching the wording of a sentence. |
| **app-shell** (`ios/Sources/App/`) | PARTIAL | `@main` | Real: the first-run compatibility gate, the three tabs, and one integration block that registers Capture, PrePass, Trainer, Viewer, Export and Booster with no line commented out. A tab whose screen failed to register says so in plain words instead of going blank. Not real: `AppNavigation` (NimbusApp.swift line 443) gives the `TabView` the selection binding it was missing, and nothing calls it. `showSavedScan`, `showScans` and `scanToLeadWith` have no callers and no readers anywhere in the tree, so finishing a capture still leaves the user sitting on the Capture tab. The two changes that would close it are named in `INTEGRATION_REQUESTS.md` and live in Capture's and Viewer's files. |
| **onboarding** (`ios/Sources/Onboarding/`) | COMPLETE | The app shell, on first launch | Probes ARKit, Metal, memory and the device model, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. An incompatible device gets this module's own screen, with the measurements behind the verdict and a re-check. The iPhone 17 rows in the chip table are inferred from the naming pattern rather than confirmed against shipped hardware, and `AppleSiliconCatalog.swift` says so in the file. |
| **capture** (`ios/Sources/Capture/`) | COMPLETE | The Scan tab | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars taken from `sceneDepth` and never the smoothed variant, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a gyro-driven blur meter, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. The post-capture quality card comes from the pre-pass and is not reimplemented here. |
| **prepass** (`ios/Sources/PrePass/`) | COMPLETE | `ScanProcessingCoordinator`, and the capture report for the quick card | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection by pose proximity and depth ICP, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. One honest weakness: `run(bundle:at:)` has no re-entrancy guard and no "have I finished unwinding?" hook, so a stop during the check-over cannot be waited for, and a second check-over can be started on top of one that is still winding down. Filed in `INTEGRATION_REQUESTS.md`. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | PrePass and Trainer, which construct these directly | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, the per-pixel authority map, and the disparity-space anchoring that would turn a relative monocular depth map into a metric one. STUB, and the only one in the iOS app: `SmartMonocularDepthStub` (SmartAuthorityMap.swift line 146), the mid-range depth estimator. `isAvailable` is false, `estimate` returns nil, and `provenance` says why: no Core ML export of a metric monocular depth model is bundled. |
| **trainer** (`ios/Sources/Trainer/`) | COMPLETE | `ScanProcessingCoordinator` | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge for scenes too big for memory, and a governor that lowers the budget as the phone heats up or runs short. It writes `model.ply`, `model.json`, per-frame exposure and the held-out frame list the review screen needs. One run at a time is now enforced rather than assumed: a second `train()` arriving while one is in flight is refused outright (MetalSplatTrainer.swift line 110) instead of resetting the cancel flag and un-cancelling the first loop, and `latestModel`, `latestSnapshot`, `exposureRecords` and `heldOutFrameIndices` are cleared per run rather than carried from one scan into the next. Honest limit: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked for. |
| **pipeline** (`ios/Sources/Pipeline/`) | PARTIAL | The scan library, and the review screen's "Next step" link | `ScanProcessingCoordinator` drives `PrePassService.run` and then `SplatTrainer.train` for one scan, offers an existing `prepass/` for reuse rather than silently redoing twenty minutes of work, sizes a `TrainingBudget` from the device tier plus the scan's measured extent plus `os_proc_available_memory()`, keeps every budget reduction the trainer reports where the user can read it, and refuses to file a finished model under a scan it does not belong to (line 549). `ScanProcessingScreen` is the screen with the buttons, and it shows a live preview of the model as it is being built. Stopping is cooperative and the screen says "Stopping" until the run has genuinely returned. For the training half that wait is real (`waitUntilIdle`), and Stop followed immediately by Start does not take. Not real, and the file's own header at line 59 currently claims otherwise: the same guarantee does not hold for the check-over, where a second run can begin while `PrePassPipeline` is still unwinding. |
| **viewer** (`ios/Sources/Viewer/`) | PARTIAL | The Scans tab | Real: the Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. `load(_ cloud:)` now has a caller (Pipeline's live training preview) where before it had none. The slider's held-out list comes from the trainer's own record for an on-device model and falls back to every 20th frame for a Booster model, saying on screen that this is a spot check and not proof. Not real: `ScanLibraryReader.readDetail` writes a plain sentence into `summary.problem` when a file is in a format version this build does not understand, and no screen reads it, so the comment above that function claiming the screens display it is untrue today. `nextUpScan`, `nextUpScanID`, `ScanSummary.primaryActionTitle` and `isWaitingOnYou` are written and nothing calls any of them. |
| **export** (`ios/Sources/Export/`) | PARTIAL | The export screen, the Booster, the trainer | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the `KHR_gaussian_splatting` extension plus a fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. A labelled refusal rather than a guess: `.spz` version 4 read throws `unsupportedVersion` (SPZCodec.swift line 227), because nobody has published a sample of that format. `ExportSelfTest.runAll()` is written and nothing calls it; it is a DEBUG-only harness and says so in its own header. |
| **booster-client** (`ios/Sources/Booster/`) | COMPLETE | The Booster tab, and the "send it to a computer" section of the processing screen | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, and resumable download of the result. Pure Swift, no third-party dependencies. The routes and payload field names agree with `docs/BOOSTER_PROTOCOL.md` and the Python server. |
| **booster-pc** (`booster/`) | PARTIAL | Run by hand on a PC | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry (depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving) and writes `model.ply`, `model.spz`, `model.json` and a preview image the phone can download. Not written, and the absence is visible in the file listing: there is no `trainer/train.py`, so there is no photometric optimisation loop. Until there is, the PC hands back the laser measurement as splats with nothing sharpening it, and that step will need an NVIDIA GPU with CUDA PyTorch and gsplat. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Installed by hand in Blender | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `.spz` decode is written from the format description with no real sample file to test against, and says so. |
| **ci** (`.github/workflows/ios.yml`) | COMPLETE | GitHub Actions | XcodeGen generates the project, `xcodebuild archive` builds it unsigned on `macos-15`, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, finding the product name by globbing so a rename cannot break it. A separate Ubuntu job byte-compiles and lints `booster/`. The macOS job has never executed. It is the first real compile this project will get, and the first honest answer about whether any of the rows above build. |

---

## The one thing that is true of the whole project

The route exists on paper. A recorded scan can be opened from the library,
checked over, built into a model, watched while it builds, and reviewed. Every
link in that chain has been read by hand and the symbols on both sides of each
link exist.

What has not happened, and what no row above should be read as claiming: no
compiler has seen any of it, no phone has run any of it, and the only route
ever observed to produce a model is still the PC Booster. The first run of the
pre-pass and the first run of the trainer will both happen on a phone, on the
first day somebody installs a build. Expect that day to be about compile errors
rather than about looking at a 3D model.
