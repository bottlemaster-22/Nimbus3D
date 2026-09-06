# Module status

One row per module, one current status, one line of specifics. No history: that
belongs in `SESSION_JOURNAL.md`. Rewritten from the source on 2026-09-06 by the
compile gate, who read every claim below against the file it describes.

---

## Read this before you read any row

**The app compiles, it is installed on the owner's iPhone, and it has produced
exactly ONE real scan. Almost nothing below is proven working.**

Those three facts, precisely:

* **It compiles.** GitHub Actions run 33988092964 was green, and the artifact
  was read rather than assumed: an arm64 Mach-O of 3,305,248 bytes and a
  `default.metallib` of 299,112 bytes, which means all three `.metal` files
  compiled and linked. Released as v0.1.1.
* **It is installed.** On the owner's iPhone 17 Pro Max, signed through Bottle
  with a free Apple ID.
* **It has run once.** One three minute scan of a room, on 2026-09-05. It came
  out looking like nothing. Tracing that produced four fixes, all in v0.1.1,
  and **not one of those fixes has been tested on hardware.** No scan has been
  made since.

So a row saying COMPLETE means the code is written, its symbols resolve, and
nothing in it is a placeholder pretending to work. It does NOT mean the code
has ever done its job. For most of what is below, the honest answer to "does
this work" is that nobody knows, because it has been executed either once or
never.

**The three words, and they mean the same thing in every row:**

* **COMPLETE** means everything the row claims is written, and nothing in it is
  a placeholder pretending to work.
* **PARTIAL** means most of it is written, and the part that is not is named on
  the same line rather than left for you to find.
* **STUB** means a typed placeholder behind a real contract, labelled at the
  call site, returning nothing and saying so.

"Reached from" is separate from the status on purpose. Code can be COMPLETE and
reachable by nothing.

**What was checked mechanically for this rewrite**, over all 122 Swift and
Metal files in `ios/Sources`: every top-level type name checked for collisions
(412 types, none duplicated); every `Type.member` reference resolved against a
declaration; every protocol requirement matched to a member on every conformer;
every non-optional stored property with no default checked against every
explicit initialiser; every `public` signature checked for an internal type;
every Metal entry point matched both ways between the `.metal` files and the
Swift that names them (38 entry points, no orphans in either direction); every
Viewer buffer index matched between `ViewerBufferIndex` and the shader
attributes; braces balanced in every file. Also swept, by name, for the exact
things that have broken this project's CI before: bare `abs` resolving to C's
integer `abs`, expressions with enough mixed conversions to stall the type
checker, `withUnsafeBytes` binding to the instance method inside an extension,
`Data.append` with no `[UInt8]` overload, Swift 6 only syntax under language
mode 5.0, and iOS 18 API without a guard. None of those are present today.

That is a static sweep, not a compiler. CI is still the only compiler.

---

| Module | Status | Reached from | What is real, and what is not |
|---|---|---|---|
| **core** (`ios/Sources/Core/`) | COMPLETE | Every other module | The shared vocabulary: cross-module types and the twelve service protocols, `BrandConfig` reading the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit to COLMAP pose conversion, `TrainingBudget.recommended`. Nothing here is a placeholder. Two known contract gaps, both worked around at the call site and both filed in `INTEGRATION_REQUESTS.md`: `SplatTrainer` has `cancel()` and no way to ask whether a run has ended, so Pipeline downcasts to `MetalSplatTrainer` for `isTraining` and `waitUntilIdle()`; and `TrainerProgress` has no field for "I lowered the budget", so Pipeline recognises one by matching the wording of a sentence. |
| **app-shell** (`ios/Sources/App/`) | COMPLETE | `@main` | The first-run compatibility gate, the three tabs, and one integration block that registers Capture, PrePass, Trainer, Viewer, Export and Booster with no line commented out. A tab whose screen failed to register says so in plain words instead of going blank. The navigation hole named in the previous version of this file is CLOSED: `AppNavigation.shared.showSavedScan(scanID)` is called from `CaptureScreen.swift` line 100, and `ScanLibraryScreen.swift` line 105 reads `navigation.scanToLeadWith` in a `.task(id:)` and clears it, so finishing a capture now lands on the new scan. |
| **onboarding** (`ios/Sources/Onboarding/`) | COMPLETE | The app shell, on first launch | Probes ARKit, Metal, memory and the device model, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. An incompatible device gets this module's own screen, with the measurements behind the verdict and a re-check. Labelled uncertainty: the iPhone 17 rows in the chip table are inferred from the naming pattern rather than confirmed against shipped hardware, and `AppleSiliconCatalog.swift` line 332 says so in a `TODO(nimbus)`. The owner's phone is an iPhone 17 Pro Max, it got past this gate, so at least the "not incompatible" branch has been exercised once. |
| **capture** (`ios/Sources/Capture/`) | COMPLETE | The Scan tab | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars taken from `sceneDepth` and never the smoothed variant, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a gyro-driven blur meter, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. **The one part of this app with real hardware evidence.** It recorded a full three minute scan. The owner's complaint was that the blur warning fires too readily for shaky hands; the fix was shorter exposure at a normal frame rate rather than a looser warning, and it is untested. The post-capture quality card comes from the pre-pass and is not reimplemented here. |
| **prepass** (`ios/Sources/PrePass/`) | COMPLETE | `ScanProcessingCoordinator`, and the capture report for the quick card | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection by pose proximity and depth ICP, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. NEW: `PrePassCensus` counts every stage against itself and writes `prepass/census.json`, and its alarms are added to the QC card the owner already reads. One honest weakness, unchanged: `run(bundle:at:)` has no re-entrancy guard and no "have I finished unwinding?" hook, confirmed by reading it, so a stop during the check-over cannot be waited for. Filed in `INTEGRATION_REQUESTS.md`. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | PrePass and Trainer, which construct these directly | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, the per-pixel authority map, and the disparity-space anchoring that would turn a relative monocular depth map into a metric one. STUB, and still the only one in the iOS app: `SmartMonocularDepthStub` (`SmartAuthorityMap.swift` line 146), the mid-range depth estimator. `isAvailable` is false, `estimate` returns nil, and `provenance` says why, which is that no Core ML export of a metric monocular depth model is bundled. |
| **trainer** (`ios/Sources/Trainer/`) | COMPLETE | `ScanProcessingCoordinator` | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge for scenes too big for memory, and a governor that lowers the budget as the phone heats up. Writes `model.ply`, `model.json`, per-frame exposure and the held-out frame list. One run at a time is enforced rather than assumed: a second `train()` arriving while one is in flight is refused (`MetalSplatTrainer.swift` line 100). NEW: `TrainerCensus` records every densify pass, every budget reduction with the live splat count at that moment, and every silently skipped iteration, and writes both `model/train_census.json` and the flat `model/census.json` the review screen reads. Honest limits: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked for; and `TrainerTuning.absGradThreshold` is now read by nothing, which `TrainerSupport.swift` line 472 states and the census refuses to report as a live gate. **Executed exactly once, and that run produced almost nothing.** |
| **pipeline** (`ios/Sources/Pipeline/`) | PARTIAL | The scan library, and the review screen's "Next step" link | `ScanProcessingCoordinator` drives `PrePassService.run` then `SplatTrainer.train` for one scan, offers an existing `prepass/` for reuse rather than silently redoing twenty minutes of work, sizes a `TrainingBudget` from the device tier plus the scan's measured extent plus `os_proc_available_memory()`, keeps every budget reduction where the user can read it, and refuses to file a finished model under a scan it does not belong to (line 549). `ScanProcessingScreen` shows a live preview of the model as it is being built. Stopping is cooperative and the screen says "Stopping" until the run has genuinely returned; for the training half that wait is real (`waitUntilIdle`). PARTIAL for one reason only, and the file's own header at line 57 now states it correctly rather than claiming otherwise: the same guarantee does not hold for the check-over, because `PrePassService` has nothing to ask. |
| **viewer** (`ios/Sources/Viewer/`) | PARTIAL | The Scans tab | Real: the Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. NEW and the most useful thing here: the splat census. One sentence at the top of the review screen naming the step that lost the most geometry, the full ladder behind a tap, every number carrying the file it was counted in, and a missing number rendered as "not recorded" rather than as zero. It also measures the finished `.ply` itself while the renderer has it in memory, which is how "480,000 points in the file and 3,000 of them can draw" becomes sayable. The stale claim in the previous version of this file is corrected: `summary.problem` IS displayed, through `ScanSummary.nextStep` at `ScanLibraryStore.swift` line 104. Still not real: `nextUpScan`, `nextUpScanID`, `ScanSummary.primaryActionTitle` and `isWaitingOnYou` are written and nothing calls any of them. |
| **export** (`ios/Sources/Export/`) | PARTIAL | The export screen, the Booster, the trainer | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the `KHR_gaussian_splatting` extension plus a fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. A labelled refusal rather than a guess: `.spz` version 4 read throws `unsupportedVersion` (`SPZCodec.swift` line 227), because nobody has published a sample of that format. `ExportSelfTest.runAll()` is written and nothing calls it; it is a DEBUG-only harness with a `TODO(nimbus)` saying it wants a test target. |
| **booster-client** (`ios/Sources/Booster/`) | COMPLETE | The Booster tab, and the "send it to a computer" section of the processing screen | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, and resumable download of the result. Pure Swift, no third-party dependencies. The routes and payload field names agree with `docs/BOOSTER_PROTOCOL.md` and the Python server. Never run against a real Booster. |
| **booster-pc** (`booster/`) | PARTIAL | Run by hand on a PC | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry (depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving) and writes `model.ply`, `model.spz`, `model.json` and a preview image the phone can download. Not written, and it says so rather than pretending: there is no `trainer/train.py`, so there is no photometric optimisation loop. `trainer/pipeline.py` line 26 states this in the file, `optimiser_available()` returns false, and the result carries `optimised: bool = False` so nothing downstream can mistake a raw initialisation for a trained model. That step will need an NVIDIA GPU with CUDA PyTorch and gsplat. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Installed by hand in Blender | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `spz_reader.py` is written from the format description with no real sample file to test against, and says so. |
| **ci** (`.github/workflows/ios.yml`) | COMPLETE | GitHub Actions | XcodeGen generates the project, `xcodebuild archive` builds it unsigned on `macos-15`, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, finding the product name by globbing so a rename cannot break it. A separate Ubuntu job byte-compiles and lints `booster/`. This has now run and gone green, most recently on the four-bug fix commit, first attempt. It remains the only compiler this project has. |

---

## The one thing that is true of the whole project

The route exists end to end and has been walked once, by a person, on a phone.
A scan was recorded, checked over, built into a model, and reviewed. That is
more than this project could say two days ago.

What came out of it was almost nothing, for four reasons that have all been
found and fixed, and none of those four fixes has been tested. The whole point
of the census added today is that the next time this happens, the answer is one
sentence on the review screen instead of a day of reading code. The census
itself has also never run.

The next thing that matters is a second scan.
