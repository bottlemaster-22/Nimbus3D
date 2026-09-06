# Module status

One row per module, one current status, one line of specifics. No history: that
belongs in `SESSION_JOURNAL.md`. Rewritten from the source on 2026-09-06 by the
integration gate that reviewed the five parallel dead-wire agents, reading every
claim below against the file it describes.

---

## Read this before you read any row

**The app compiles, it is installed on the owner's iPhone, and it has produced
exactly ONE real scan. Almost nothing below is proven working.**

Those four facts, precisely:

* **It compiled once.** Commit `4bebe6e` is the last thing CI saw. Everything
  described below as "this round" is on top of it.
* **Nothing since then has been compiled by anything.** The working tree carries
  30 modified files against `4bebe6e`: 26 Swift, 1 Metal, 2 Markdown and the CI
  workflow, 1,604 lines added and 58 removed. There is no Swift toolchain on the
  machine this gate ran on. Every claim here about this round is a claim about
  SOURCE, not about a build.
* **It is installed.** On the owner's iPhone 17 Pro Max, signed through Bottle
  with a free Apple ID.
* **It has run once.** One three minute scan of a room, on 2026-09-05. It came
  out looking like nothing. Everything done since has been tracing why, and
  **not one of those fixes has been tested on hardware.** No scan has been made
  since.

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

**Two rounds happened on 2026-09-06.** Rows say which. "Committed earlier today"
means it is in `4bebe6e` and CI has seen it. "This round" means it is in the
working tree and nothing has compiled it.

---

## What this gate checked mechanically, and what it did not

Checked, over all 122 Swift and Metal files in `ios/Sources`:

* Top-level type and typealias names, for collisions across the one Xcode
  target. 0 duplicates.
* Braces, parentheses and brackets balanced in every file changed this round,
  with line comments, block comments and string literals stripped first. Also
  `#if` against `#endif`, and triple-quote counts. 0 imbalances in 26 files.
* Every cross-file symbol the new code names, resolved against its declaration:
  `BrandConfig.loggingSubsystem` / `.versionString` / `.Folder.exports` /
  `.Folder.sparseModel`, `NimbusServices.installedModules`,
  `StoredDeviceReport.isStale`, `DeviceReportStore.load()`,
  `OnboardingFormat.yesNo`, `OnboardingFactRow(symbol:label:value:note:)`,
  `QCFinding(code:severity:message:fixHint:)`,
  `CaptureBundleRef.url(forRelativePath:)`,
  `ARCaptureService.setBracketingEnabled` / `.unlockExposureAndWhiteBalance`,
  `ThermalLevel(ProcessInfo.ThermalState)` and its `Comparable`,
  `TrainerBudgetGovernor.resolutionLadder`, `TrainerRenderSize.tileCount`,
  `ExportedAsset.init(...)`, `ScanSummary.rootURL` / `.scanID`,
  `ScanLibraryStore.nextUpScan`, `ScanSummary.primaryActionTitle`,
  `ScanProcessingScreen(summary:)`. All resolve.
* `DeviceCompatibilityProbe.findingsForDisplay()` changed its return type this
  round. It is NOT a `DeviceCompatibilityService` protocol requirement
  (`Contracts.swift:2060` requires only `evaluate()`), and its one caller was
  updated. No conformance broke.
* Actor isolation on everything new: `NimbusServices` and `NimbusBootstrap` are
  `@MainActor`, so is `ScreenUnavailableView` now, and all three of its call
  sites are `@ViewBuilder` members of MainActor views. `ARCaptureService` and
  `CaptureScreenModel` are both `@MainActor`. Everything crossing a
  `Task.detached` boundary is `Sendable`: `URL`, `ScanID` (a `String`),
  `[ExportedAsset]` (`Codable, Hashable, Sendable`), `DeviceFindingsForDisplay`.
* `SmartAuthorityMap.map(for:)` takes a non-recursive `NSLock`. The new
  glass-counting block sits AFTER the `lock.unlock()` at line 778 and takes the
  lock again for three statements. No re-entrant acquisition, no deadlock.
* The 13 `static_assert(sizeof(...))` lines in `TrainerShaders.metal` against
  the 13 stride checks in `TrainerGPULayouts.verify()`. They match one for one.
  No GPU struct changed this round, so none of the asserts moved.
* **The new SSIM window, in float32 arithmetic, because it now gates the trainer
  starting at all.** `verify()` re-derives the eleven taps from `ssimSigma` and
  fails the run if the shipped table is off by more than `1e-6` on any tap or
  `1e-5` on the sum. Recomputed here: worst tap error `2.98e-8`, sum exactly
  `1.0`. It passes with about 30x margin. The table it replaced summed to
  `0.99752`.
* The 16-bit tile ceiling the same function now enforces: the ladder tops out at
  720 px, which is 45x45 = 2,025 tiles against a 65,535 limit. Passes.
* 40 Metal entry points across the three `.metal` files (4 capture, 28 trainer,
  8 viewer). Unchanged this round.
* By name, the exact things that have broken this project's CI before: bare
  `abs` binding to C's integer `abs` (16 uses, every argument a `Float` or
  `Double`, all pre-existing; the one new comparison uses `Swift.abs`
  explicitly), Swift 6 only syntax under language mode 5.0 (none), iOS 18 API
  without a guard (none), a stored property no initialiser assigns (none in the
  four structs that gained fields; all four are optional or defaulted),
  `Data.append` and `withUnsafeBytes` (untouched), a public init exposing an
  internal type (`DeviceFindingsForDisplay` and
  `SmartLossSettings.trustBuildCost` both take and return public types only).
* `.github/workflows/ios.yml` parsed as YAML. Three jobs, the two `if:`
  expressions are well-formed, and `build` has no `needs:` on the new check.

NOT checked exhaustively, and named so nobody reads more into this than is
there: protocol requirements against every conformer, and non-defaulted stored
properties against every initialiser, across all 122 files. Both were checked
for what changed this round, not repo-wide. Type-checker time on the longer
string concatenations is a guess, not a measurement.

There is no SwiftLint step in CI, so `swiftlint:disable` comments and line
lengths cannot fail the build.

That is a static sweep, not a compiler. **CI is still the only compiler.**

---

## The dead-wire gate, new this round

`tools/deadwire.py` reports every declaration whose name appears nowhere else in
the module. It found 135 candidates; five agents triaged their own file sets and
wired up the real ones, and this gate triaged Booster, which nobody had been
given. 112 remain and every one of them is listed in
`tools/deadwire_allowlist.txt` with a reason.

The script now reads that allowlist, excludes it, and **exits non-zero on
anything else**, so the build goes red for exactly one reason: somebody wrote
new code that nothing calls. A `deadwire` job in `.github/workflows/ios.yml`
runs it on every push, on a cheap Ubuntu runner. It does not block the IPA.

Verified both ways before it was committed: exit 0 on the current tree, and exit
1 with the offending lines named when a decoy declaration was added.

---

| Module | Status | Reached from | What is real, and what is not |
|---|---|---|---|
| **core** (`ios/Sources/Core/`) | COMPLETE | Every other module | The shared vocabulary: cross-module types and the twelve service protocols, `BrandConfig` reading the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit to COLMAP pose conversion, `TrainingBudget.recommended`. Nothing here is a placeholder. Committed earlier today: `TrainerProgress` gained `gradientStepsCompleted` and `consecutiveZeroGrowthPasses`, both optional and appended last so no call site changed. Untouched this round. Two known contract gaps remain, both worked around at the call site and both filed in `INTEGRATION_REQUESTS.md`: `SplatTrainer` has `cancel()` and no way to ask whether a run has ended, so Pipeline downcasts to `MetalSplatTrainer` for `isTraining` and `waitUntilIdle()`; and `TrainerProgress` still has no field for "I lowered the budget", so Pipeline recognises one by matching the wording of a sentence. |
| **app-shell** (`ios/Sources/App/`) | COMPLETE | `@main` | The first-run compatibility gate, the three tabs, and one integration block that registers Capture, PrePass, Trainer, Viewer, Export and Booster with no line commented out. `AppNavigation.shared.showSavedScan(scanID)` is called from `CaptureScreen.swift` and cleared in `ScanLibraryScreen.swift`, so finishing a capture lands on the new scan. THIS ROUND: the app can finally say what is in the build. `NimbusServices.installedModules` existed and nothing read it, so the registration list was only checkable by opening this file; `NimbusBootstrap.install` now logs it at launch and logs an `error` naming any of the seven expected modules that did NOT register, and `ScreenUnavailableView` prints what did load underneath the name of what did not. Read out over a phone call that is the difference between "it is broken" and one named missing module. |
| **onboarding** (`ios/Sources/Onboarding/`) | COMPLETE | The app shell, on first launch | Probes ARKit, Metal, memory and the device model, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. THIS ROUND, two honesty fixes. `findingsForDisplay()` used to return a bare `DeviceCompatibilityFindings`, so a report read back off disk and one measured a moment ago produced the same struct and the same screen; it now returns `DeviceFindingsForDisplay` carrying `StoredDeviceReport.isStale`, and the "Checked" row says outright when the numbers came from an older build, an older iOS, or over a month ago. And the technical details table never stated whether the phone HAS the laser scanner, the one measurement the whole verdict turns on; it is now a row, with a note when running in the Simulator where the answer is a stub. Labelled uncertainty, unchanged: the iPhone 17 rows in the chip table are inferred from the naming pattern rather than confirmed against shipped hardware, and `AppleSiliconCatalog.swift` says so in a `TODO(nimbus)`. The owner's phone got past this gate, so the "not incompatible" branch has been exercised once. |
| **capture** (`ios/Sources/Capture/`) | COMPLETE | The Scan tab | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars from `sceneDepth` and never the smoothed variant, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a gyro-driven blur meter, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. **The one part of this app with real hardware evidence.** THIS ROUND, three fixes, and the first is a correction to the app's own definition of "done". (1) The sharpness channel returned `bestSharpness` RAW while the other two channels were normalised against their targets, and `CaptureTuning.coverageSharpnessTarget = 0.55` was referenced nowhere. Since `CaptureSharpnessMeter` normalises against the SESSION RUNNING MAXIMUM, the coverage bar was not asking "was this patch seen sharply enough" but "was it seen by a frame within 70 per cent of the sharpest frame in the whole scan", a moving reference and a bar 1.8x too high. The error made the percentage pessimistic, not optimistic: it never said "done" early, it stalled short and gave no way to clear the patches it held back, and it drove the spoken hint to the wrong channel. (2) `coverageDirectionBuckets = 32` appeared only in a doc comment while the bucket maths hardcoded 8 azimuth by 4 elevation; azimuth bins are now derived from the constant, with a hard 32-bit ceiling, so the mask and the tuning cannot disagree. Same 31 maximum as before, so behaviour is unchanged today. (3) "Hold the brightness" was a one-way door: lock the exposure at a window, walk into a dark hallway, and every remaining frame was exposed for the window with no control anywhere to give it back. There is now a release on the window card and a second one in the settings panel that stays on screen after the card has gone, plus a user switch for the darker window shots that reads its initial state from the service instead of assuming. One dead spare deleted: `CaptureHUDPalette.panel`, confirmed unreferenced. Unchanged and untested: the owner's complaint that the blur warning fires too readily for shaky hands. |
| **prepass** (`ios/Sources/PrePass/`) | COMPLETE | `ScanProcessingCoordinator`, and the capture report for the quick card | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. **The refined poses were checked end to end this round and they are fine**: computed at `PrePassPipeline.swift:477`, improved by the LiDAR-anchored bundle adjustment, written onto `CaptureFrame.refinedPose`, carried in `PrePassResult.refinedPoses`, and written to `prepass/sparse_refined/` before the four optional later stages so a crash in carving cannot cost them. THIS ROUND: `PrePassPaths` declares five sidecar paths that `Sources/Smart` writes by building the identical strings itself, and nothing enforced that the two spellings keep agreeing. A silent disagreement would look exactly like a stage that ran and produced nothing. `PrePassPaths.missing(_:at:)` is new and the pipeline now calls it right after the F6 and F3 stages report success, raising `trust_files_missing` or `edge_maps_missing` at `.problem` severity on the QC card. That detects drift; it does not prevent it, and the refactor that would is rejected in writing in `INTEGRATION_REQUESTS.md`. Unchanged weakness: `run(bundle:at:)` has no re-entrancy guard and no "have I finished unwinding?" hook. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | PrePass and Trainer, which construct these directly | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, the per-pixel authority map, and the disparity-space anchoring that would turn a relative monocular depth map into a metric one. STUB, and still the only one in the iOS app: `SmartMonocularDepthStub`, the mid-range depth estimator, the default `midRegimeProvider` in `DirectionalBackgroundModel`. THIS ROUND, three things that were measured and thrown away. (1) `SmartLossSettings.economy` was declared, documented for "a phone that is already warm, or a house-sized scan", and selected by NOTHING, so every trust build ran at full cost regardless. `trustBuildCost(requested:frameCount:thermalLevel:)` now makes that choice from two measurements, `TwoScaleTrustField.build` calls it at the top of every build, and it logs which preset ran and why, because a cheaper build is a real difference in what was verified. Only the three build-cost knobs are taken from the preset; the calibration fields are kept, so two runs stay comparable. (2) `midRegimeProvenance` and `isMidRegimeReal` existed so a stubbed run could never be mistaken for a real one, and nothing read them, so no scan on disk recorded which it got; they now go into `model/background.json` as two new optional fields, into the summary sentence, and into the log once per run. (3) `SmartGlassMask.confirmedFraction` was measured per frame and read by nothing, so a scan of a conservatory could lose most of its geometry with nothing saying why. `SmartAuthorityMap` now warns once and keeps a running `glassDominatedFrameCount`, **and this gate added the `builtFrameCount` denominator it needs**, because a count without one cannot say whether it is a catastrophe or a footnote. |
| **trainer** (`ios/Sources/Trainer/`) | COMPLETE | `ScanProcessingCoordinator` | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge, and a governor that lowers the budget as the phone heats up. Committed earlier today, and the largest single correction this trainer has had: every geometry term in `trainer_loss_depth` was an unnormalised sum over 49,152 native depth samples while both photometric terms were per-pixel means, so the laser outweighed the photograph by roughly four orders of magnitude; plus six schedules that never ran their back half on a shortened run, gradient steps split from times round the loop, and the Mip-Splatting 3D filter folded into the export. THIS ROUND: **the SSIM blur window on the GPU had drifted from the sigma written down in Swift.** The eleven literal taps in `trainer_blur_h` and `trainer_blur_v` summed to `0.99752` with a centre tap of `0.26361500` where the normalised sigma-1.5 Gaussian wants `0.26601172`, so every separable blur lost about half a per cent of its energy and the SSIM half of the loss ran on slightly wrong local means. Nothing compared the two, because nothing could. `TrainerGPULayouts.verify()` now re-derives the window from `ssimSigma` and `ssimWindowRadius` and refuses to start if the three disagree, and it also enforces the 16-bit tile ceiling that was written down and never checked. Census additions this round: `authorityFramesBuilt`, `glassDominatedFrames`, `midRegimeIsReal` and `midRegimeProvenance`, a ledger line for each, and a `glass_dominated_frames` alert at a third of frames. Honest limits: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked for; `TrainerTuning.absGradThreshold`, `splitChildCount` and `lossReadbackIntervalIterations` are read by nothing and each says so at its declaration, so the census cannot present a dead setting as a live one. **Executed exactly once, and that run produced almost nothing.** |
| **pipeline** (`ios/Sources/Pipeline/`) | PARTIAL | The scan library, and the review screen's "Next step" link | `ScanProcessingCoordinator` drives `PrePassService.run` then `SplatTrainer.train` for one scan, offers an existing `prepass/` for reuse rather than silently redoing twenty minutes of work, sizes a `TrainingBudget` from the device tier plus the scan's measured extent plus `os_proc_available_memory()`, keeps every budget reduction where the user can read it, and refuses to file a finished model under a scan it does not belong to. `ScanProcessingScreen` shows a live preview of the model as it is built. Untouched this round. PARTIAL for one reason only, and the file's own header states it: stopping is cooperative and genuinely waited for on the training half (`waitUntilIdle`), and the same guarantee does not hold for the check-over, because `PrePassService` has nothing to ask. |
| **viewer** (`ios/Sources/Viewer/`) | PARTIAL | The Scans tab | Real: the Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. The splat census puts one sentence at the top of the review screen naming the step that lost the most geometry, every number carrying the file it was counted in, and a missing number rendered as "not recorded" rather than as zero. Committed earlier today: the renderer's 2D filter variance was 0.3 against the trainer's 0.25 and both are now 0.25. THIS ROUND, three things that were written and never reached. (1) `ScanLibraryStore.nextUpScan` and `ScanSummary.primaryActionTitle` were both written for a "Next up" card that no screen drew, so coming back from the Capture tab a freshly recorded scan was one more identical row; the card now exists at the top of the library with the words of the button that starts the next step. That chain, through `isWaitingOnYou`, is now live. (2) `produced` only ever held files exported in the CURRENT session, so a file exported yesterday sat in the scan's export folder with no row and no share button anywhere in the app; the export screen now lists what is on disk, off the main actor, merged so a file made this session keeps its splat count. (3) `ExportSelfTest.runAll()` now has a caller. Still not real: `ScanSummary.heldOutPSNR` is assigned at `ScanLibraryStore.swift:332` and displayed by nothing. Still flat grey: nothing in this module reads `model/background.bin`, so the trained far field appears in no preview. That is the largest thing left open and it is rejected in writing, with reasons, in `INTEGRATION_REQUESTS.md`. |
| **export** (`ios/Sources/Export/`) | PARTIAL | The export screen, the Booster, the trainer | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the `KHR_gaussian_splatting` extension plus a fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. Committed earlier today: `SplatCloud.fuse3DFilter(_:)` folds the Mip-Splatting 3D low-pass filter into the stored log-scale and opacity, exactly, so `blender_addon/`, SuperSplat and every other reader draws the model that was actually fitted; it is called from `MetalSplatTrainer.readCloud`, which is worth saying because for most of that day it was written and called from nowhere. `SplatCloud.filter3DFused` is `true` / `false` / `nil`, and `nil` means "a file that cannot say", never a guess. A labelled refusal rather than a guess: `.spz` version 4 read throws `unsupportedVersion`, because nobody has published a sample. THIS ROUND: no file in this module changed, but `ExportSelfTest.runAll()` finally runs. Its 368 lines of round-trip and known-answer checks, including "PLY round trip is bit-exact", "SPZ round trip within quantization tolerance" and "empty cloud is rejected, not silently written", had **zero callers anywhere in the app** and had therefore never executed once. A DEBUG-only section of the export screen now runs them on open and shows each line, failures in orange. Release builds compile none of it. The `TODO(nimbus)` asking for a real test target still stands. |
| **booster-client** (`ios/Sources/Booster/`) | COMPLETE | The Booster tab, and the "send it to a computer" section of the processing screen | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, and resumable download of the result. Pure Swift, no third-party dependencies. The routes and payload field names agree with `docs/BOOSTER_PROTOCOL.md` and the Python server. Untouched this round, but **no agent was assigned this module and this gate triaged its six dead-wire candidates rather than leave them unlooked-at.** All six are safe and now carry reasons: `BoosterAPI.pathPrefix` is used eight times and only inside string interpolations the scanner strips, so every Booster URL really is built from it; `pollStatus` is a spare confirmed against `booster/server/pairing.py:137`, which resolves a confirm synchronously so the poll path is never needed; `gpuName` and `queueDepth` are wire-contract fields the Python server really sends and no screen shows yet; `allPairings` and `removeJob` are spare store reads. **Never run against a real Booster.** |
| **booster-pc** (`booster/`) | PARTIAL | Run by hand on a PC | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry (depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving) and writes `model.ply`, `model.spz`, `model.json` and a preview image the phone can download. Not written, and it says so rather than pretending: there is no `trainer/train.py`, so there is no photometric optimisation loop. `trainer/pipeline.py` line 26 states this in the file, `optimiser_available()` returns false, and the result carries `optimised: bool = False` so nothing downstream can mistake a raw initialisation for a trained model. That step will need an NVIDIA GPU with CUDA PyTorch and gsplat. Untouched this round. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Installed by hand in Blender | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `spz_reader.py` is written from the format description with no real sample file to test against, and says so. Untouched this round, and it remains the main beneficiary of the 3D-filter fuse: it will now draw the trained model rather than a sharper, more solid version of it. |
| **ci** (`.github/workflows/ios.yml`) | COMPLETE | GitHub Actions | XcodeGen generates the project from `ios/project.yml`, `xcodebuild archive` builds it unsigned on `macos-15`, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, finding the product name by globbing so a rename cannot break it. A separate Ubuntu job byte-compiles and lints `booster/`. Source directories are listed recursively, so a new `.swift` or `.metal` file is picked up with no edit here. THIS ROUND: a third job, `deadwire`, runs `tools/deadwire.py` against the allowlist on every push, on Ubuntu, in well under a minute. `on.push.branches` widened from `[main]` to `["**"]` so it sees every push, and the two expensive jobs carry an `if:` that reproduces their old trigger exactly (main, a `v*` tag, any PR, manual dispatch), so a feature-branch push runs the check and skips the 90-minute macOS runner. The archive job deliberately has no `needs:` on the check: a lint finding must never be the reason the owner cannot install a build. **It remains the only compiler this project has, and it has not yet seen this round.** |

---

## The one thing that is true of the whole project

The route exists end to end and has been walked once, by a person, on a phone.
A scan was recorded, checked over, built into a model, and reviewed. What came
out of it was almost nothing.

Every fault found since has had the same shape: something was silently doing
nothing, and the app reported success. Four of those were scan-destroying and
all four were found by accident.

This round went looking on purpose. The five agents reported fourteen real
findings between them (capture 3, prepass 0, trainer 2, viewer and export 3,
core and smart and app 6); this gate re-read the load-bearing ones against the
source rather than taking them on trust, and added a fifteenth. What it leaves
behind is a check that fails the build the next time somebody writes code that
nothing calls.

The strongest evidence that the check is needed is that fifteenth. It was
introduced earlier the same day, by the work that was hunting the other
fourteen: a public property, counted every frame, read by nothing, with a log
line pointing the reader at it.

The next thing that matters is a green CI run, and then a second scan.
