# Module status

One row per module, one current status, one line of specifics. No history: that
belongs in `SESSION_JOURNAL.md`. Rewritten from the source on 2026-09-06 by the
compile gate that reviewed the six parallel agents, who read every claim below
against the file it describes.

---

## Read this before you read any row

**The app compiles, it is installed on the owner's iPhone, and it has produced
exactly ONE real scan. Almost nothing below is proven working.**

Those three facts, precisely:

* **It compiles.** The last commit, `e3ec117`, went green through
  `.github/workflows/ios.yml` and was released as v0.1.1. That is the previous
  gate's record and this gate did not re-run it.
* **Nothing since then has been compiled by anything.** The working tree now
  carries 23 modified files and 3,321 added lines against `e3ec117`, from six
  agents that ran in parallel today plus this gate. CI has not seen one line of it. Every
  claim in this file about today's work is a claim about SOURCE, not about a
  build.
* **It is installed.** On the owner's iPhone 17 Pro Max, signed through Bottle
  with a free Apple ID.
* **It has run once.** One three minute scan of a room, on 2026-09-05. It came
  out looking like nothing. Tracing that produced the fixes in v0.1.1 and the
  six changes made today, and **not one of them has been tested on hardware.**
  No scan has been made since.

So a row saying COMPLETE means the code is written, its symbols resolve, and
nothing in it is a placeholder pretending to work. It does NOT mean the code has
ever done its job. For most of what is below, the honest answer to "does this
work" is that nobody knows, because it has been executed either once or never.

**The three words, and they mean the same thing in every row:**

* **COMPLETE** means everything the row claims is written, and nothing in it is
  a placeholder pretending to work.
* **PARTIAL** means most of it is written, and the part that is not is named on
  the same line rather than left for you to find.
* **STUB** means a typed placeholder behind a real contract, labelled at the
  call site, returning nothing and saying so.

"Reached from" is separate from the status on purpose. Code can be COMPLETE and
reachable by nothing, and that is the single most common fault this project has
found in itself.

---

## What this gate checked mechanically, and what it did not

Checked, over all 122 Swift and Metal files in `ios/Sources`:

* Top-level type names, for collisions. 0 duplicates.
* Braces, parentheses and brackets balanced in every file, with line comments,
  block comments and string literals (including multi-line ones and
  interpolations) stripped first. 0 imbalances.
* Metal entry points against Swift. 40 `kernel` / `vertex` / `fragment`
  functions across the three `.metal` files. Every one is named by a string
  literal somewhere in Swift, and no Swift string shaped like a kernel name
  (`trainer_*`, `viewer_*`, `capture_*`) is missing from the shaders. No
  orphans in either direction.
* Metal buffer indices. Every kernel's `[[buffer(N)]]` attributes are unique
  and contiguous from 0. No duplicates anywhere.
* GPU struct strides. All thirteen trainer structs counted field by field
  against `TrainerGPULayouts.swift`, and `TrainerShaders.metal` now carries a
  `static_assert(sizeof(X) == N)` for each, so the next mismatch fails in CI on
  the wrong line instead of at start-up on the owner's phone.
* Every `Type.member` reference in the 18 source files changed today, resolved
  against a declaration in the repo or a known system type. Everything
  resolves.
* By name, the exact things that have broken this project's CI before: bare
  `abs` resolving to C's integer `abs` (six uses, every argument a `Float`, so
  every one binds to `Swift.abs`), Swift 6 only syntax under language mode 5.0
  (none), iOS 18 API without a guard (none), `Data.append` and
  `withUnsafeBytes` patterns (unchanged from the shapes already compiling).

NOT checked exhaustively, and named so nobody reads more into this than is
there: protocol requirements against every conformer, and non-defaulted stored
properties against every initialiser across all 122 files. Both were checked for
the structs changed today, not repo-wide.

There is no SwiftLint step in CI, so `swiftlint:disable` comments and line
lengths cannot fail the build.

That is a static sweep, not a compiler. CI is still the only compiler.

---

| Module | Status | Reached from | What is real, and what is not |
|---|---|---|---|
| **core** (`ios/Sources/Core/`) | COMPLETE | Every other module | The shared vocabulary: cross-module types and the twelve service protocols, `BrandConfig` reading the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit to COLMAP pose conversion, `TrainingBudget.recommended`. Nothing here is a placeholder. NEW today: `TrainerProgress` gained `gradientStepsCompleted` and `consecutiveZeroGrowthPasses`, both optional, both appended last with `nil` defaults so no existing call site changed, and both actually filled by the trainer's in-loop tick. `SplatModel.iterationsCompleted` now says in a doc comment that it counts times round the loop and not optimisation steps, which is what it always counted. `SplatModel.heldOutPSNR` now names the three ways held-out evaluation differs from the viewer's render. Two known contract gaps remain, both worked around at the call site and both filed in `INTEGRATION_REQUESTS.md`: `SplatTrainer` has `cancel()` and no way to ask whether a run has ended, so Pipeline downcasts to `MetalSplatTrainer` for `isTraining` and `waitUntilIdle()`; and `TrainerProgress` still has no field for "I lowered the budget", so Pipeline recognises one by matching the wording of a sentence. |
| **app-shell** (`ios/Sources/App/`) | COMPLETE | `@main` | The first-run compatibility gate, the three tabs, and one integration block that registers Capture, PrePass, Trainer, Viewer, Export and Booster with no line commented out. A tab whose screen failed to register says so in plain words instead of going blank. `AppNavigation.shared.showSavedScan(scanID)` is called from `CaptureScreen.swift`, and `ScanLibraryScreen.swift` reads `navigation.scanToLeadWith` in a `.task(id:)` and clears it, so finishing a capture lands on the new scan. Untouched today. |
| **onboarding** (`ios/Sources/Onboarding/`) | COMPLETE | The app shell, on first launch | Probes ARKit, Metal, memory and the device model, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. An incompatible device gets this module's own screen, with the measurements behind the verdict and a re-check. Labelled uncertainty: the iPhone 17 rows in the chip table are inferred from the naming pattern rather than confirmed against shipped hardware, and `AppleSiliconCatalog.swift` says so in a `TODO(nimbus)`. The owner's phone got past this gate, so the "not incompatible" branch has been exercised once. Untouched today. |
| **capture** (`ios/Sources/Capture/`) | COMPLETE | The Scan tab | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars taken from `sceneDepth` and never the smoothed variant, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a gyro-driven blur meter, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. **The one part of this app with real hardware evidence.** It recorded a full three minute scan. The owner's complaint was that the blur warning fires too readily for shaky hands; the fix was shorter exposure at a normal frame rate rather than a looser warning, and it is untested. Untouched today. |
| **prepass** (`ios/Sources/PrePass/`) | COMPLETE | `ScanProcessingCoordinator`, and the capture report for the quick card | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. NEW today, and all three close a "total failure looks like a normal empty result" hole: a carve that proves NO air empty now throws `NimbusError.prePassFailed` instead of storing a grid that answers "unknown" to every question for the rest of the run, and `VoxelFreeSpaceCarver.load` now actually reads `OccupancyGridRef.cellCount` / `.emptyCellCount`, which had one writer and zero readers; `PrePassDepthFrame.loadOutcome` splits "this frame recorded no depth" from "this frame's depth file would not open", with `load` kept as a four-line wrapper so the two cannot drift; and `PrePassSurvey` now uses it, which makes the `unreadable_depth` QC finding reachable for its own named cause for the first time in this app's history. `PrePassCensus.Carving` gained `keyframesDepthUnreadable` (optional, so an older file reads as "not recorded" and not as zero) and its keyframe line now says "could not be read" instead of the wrong "had no depth to read". Unchanged weakness: `run(bundle:at:)` has no re-entrancy guard and no "have I finished unwinding?" hook, so a stop during the check-over cannot be waited for. Filed in `INTEGRATION_REQUESTS.md`. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | PrePass and Trainer, which construct these directly | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, the per-pixel authority map, and the disparity-space anchoring that would turn a relative monocular depth map into a metric one. STUB, and still the only one in the iOS app: `SmartMonocularDepthStub` (`SmartAuthorityMap.swift` line 146), the mid-range depth estimator, constructed as the default `midRegimeProvider` in `DirectionalBackgroundModel`. `isAvailable` is false, `estimate` returns nil, and `provenance` says why, which is that no Core ML export of a metric monocular depth model is bundled. Untouched today. |
| **trainer** (`ios/Sources/Trainer/`) | COMPLETE | `ScanProcessingCoordinator` | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge, and a governor that lowers the budget as the phone heats up. **Four changes today, and the first is the largest single correction this trainer has had.** (1) Every geometry term in `trainer_loss_depth` was an unnormalised sum over the 49,152 native depth samples while both photometric terms were per-pixel means, so the laser outweighed the photograph by roughly four orders of magnitude and the densification statistic was ranking Gaussians by where the LASER disagreed. All five terms are now divided by the supervised sample count, forward value and gradient together, in one kernel. (2) `runIteration` was still being handed the PLANNED slice budget while the loop exits at `effectiveTotal`, so on any shortened run the back half of six schedules never executed: the SH degree ramp, the frequency-blur decay, the depth-loss decay, late opacity binarization, the position learning-rate decay and the warm-up end. All eight consumers now use `effectiveTotal`. (3) Times round the loop and gradient steps actually taken are now two separate numbers and the run's outcome is judged on the second. (4) The Mip-Splatting 3D low-pass filter is now folded into the exported scale and opacity, so what leaves the trainer is finally the model that was fitted. `TrainerCensus` records every densify pass with the densifier's own verdict on why it grew what it grew, every budget reduction with the live splat count at that moment, every silently skipped iteration, the seeder's trust line against the distribution it was drawn on, and how much of each frame the laser actually got a vote on. Honest limits: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked for; and `TrainerTuning.absGradThreshold` is read by nothing, which `TrainerSupport.swift` line 472 states and the census refuses to report as a live gate. **Executed exactly once, and that run produced almost nothing.** |
| **pipeline** (`ios/Sources/Pipeline/`) | PARTIAL | The scan library, and the review screen's "Next step" link | `ScanProcessingCoordinator` drives `PrePassService.run` then `SplatTrainer.train` for one scan, offers an existing `prepass/` for reuse rather than silently redoing twenty minutes of work, sizes a `TrainingBudget` from the device tier plus the scan's measured extent plus `os_proc_available_memory()`, keeps every budget reduction where the user can read it, and refuses to file a finished model under a scan it does not belong to. `ScanProcessingScreen` shows a live preview of the model as it is being built, and that preview now carries the 3D-filter flag through its downsample instead of losing it. PARTIAL for one reason only, and the file's own header states it: stopping is cooperative and genuinely waited for on the training half (`waitUntilIdle`), and the same guarantee does not hold for the check-over, because `PrePassService` has nothing to ask. |
| **viewer** (`ios/Sources/Viewer/`) | PARTIAL | The Scans tab | Real: the Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. The splat census puts one sentence at the top of the review screen naming the step that lost the most geometry, the full ladder behind a tap, every number carrying the file it was counted in, and a missing number rendered as "not recorded" rather than as zero. NEW today: the renderer's 2D filter variance was 0.3 against the trainer's 0.25, a mismatch nothing would ever have reported, and both are now 0.25; and `ScanCensus.Drawable.filter3DFused` is stamped from the loaded cloud, so a model drawn sharper and more solid than it was built now says so on the "Points that can actually draw" rung instead of only in the log. Still not real: `nextUpScan`, `nextUpScanID`, `ScanSummary.primaryActionTitle` and `isWaitingOnYou` form a chain that ends in nothing calling it. `ScanSummary.heldOutPSNR` is stored and displayed nowhere. |
| **export** (`ios/Sources/Export/`) | PARTIAL | The export screen, the Booster, the trainer | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the `KHR_gaussian_splatting` extension plus a fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. NEW today: `SplatCloud.fuse3DFilter(_:)` folds the trainer's Mip-Splatting 3D low-pass filter into the stored log-scale and opacity, exactly (`R S^2 R^T + f^2 I == R (S^2 + f^2 I) R^T`), so `blender_addon/`, SuperSplat and every other reader draws the model that was actually fitted. It is called from `MetalSplatTrainer.readCloud`, which is worth saying because for most of today it was written and called from nowhere. `SplatCloud.filter3DFused` is `true` / `false` / `nil` and `nil` means "a file that cannot say", never a guess. A labelled refusal rather than a guess: `.spz` version 4 read throws `unsupportedVersion`, because nobody has published a sample of that format. `ExportSelfTest.runAll()` is written and nothing calls it; it is a DEBUG-only harness with a `TODO(nimbus)` saying it wants a test target. |
| **booster-client** (`ios/Sources/Booster/`) | COMPLETE | The Booster tab, and the "send it to a computer" section of the processing screen | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, and resumable download of the result. Pure Swift, no third-party dependencies. The routes and payload field names agree with `docs/BOOSTER_PROTOCOL.md` and the Python server. Never run against a real Booster. Untouched today. |
| **booster-pc** (`booster/`) | PARTIAL | Run by hand on a PC | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry (depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving) and writes `model.ply`, `model.spz`, `model.json` and a preview image the phone can download. Not written, and it says so rather than pretending: there is no `trainer/train.py`, so there is no photometric optimisation loop. `trainer/pipeline.py` line 26 states this in the file, `optimiser_available()` returns false, and the result carries `optimised: bool = False` so nothing downstream can mistake a raw initialisation for a trained model. That step will need an NVIDIA GPU with CUDA PyTorch and gsplat. Untouched today. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Installed by hand in Blender | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `spz_reader.py` is written from the format description with no real sample file to test against, and says so. Untouched today, and it is the main beneficiary of the fuse above: it will now draw the trained model rather than a sharper, more solid version of it. |
| **ci** (`.github/workflows/ios.yml`) | COMPLETE | GitHub Actions | XcodeGen generates the project from `ios/project.yml`, `xcodebuild archive` builds it unsigned on `macos-15`, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, finding the product name by globbing so a rename cannot break it. A separate Ubuntu job byte-compiles and lints `booster/`. Source directories are listed recursively, so a new `.swift` or `.metal` file is picked up with no edit here. It remains the only compiler this project has, and it has not yet seen today's work. |

---

## The one thing that is true of the whole project

The route exists end to end and has been walked once, by a person, on a phone.
A scan was recorded, checked over, built into a model, and reviewed. What came
out of it was almost nothing.

Every fault found since has had the same shape: something was silently doing
nothing, and the app reported success. Today added four more of those to the
list, and the largest of them was that the photographs had almost no say in the
geometry at all. The census exists so that the next time this happens, the
answer is a sentence on the review screen instead of a day of reading code.

The next thing that matters is a green CI run, and then a second scan.
