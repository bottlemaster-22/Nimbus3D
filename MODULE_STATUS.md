# Module status

One row per module, one current status, one line of specifics. No history: that
belongs in `SESSION_JOURNAL.md`. Rewritten from the source on 2026-09-06 by the
integration gate that reviewed the five parallel trapping-conversion agents,
reading every claim below against the file it describes.

---

## Read this before you read any row

**The app compiles, it is installed on the owner's iPhone, and it has produced
exactly ONE real scan. Almost nothing below is proven working.**

Those four facts, precisely:

* **It compiled once.** Commit `4bebe6e` is the last thing CI saw. Everything
  described below as "earlier today" or "this round" is on top of it.
* **Nothing since then has been compiled by anything.** There is no Swift
  toolchain on the machine these gates ran on. Every claim here about the last
  two rounds is a claim about SOURCE, not about a build.
* **It is installed.** On the owner's iPhone 17 Pro Max, signed through Bottle
  with a free Apple ID.
* **It has run once.** One three minute scan of a room, on 2026-09-05. It came
  out looking like nothing, and since then the app has been **crashing during
  use, with memory free**. Everything done since has been tracing why, and
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

**Three rounds happened on 2026-09-06.** Rows say which. "Committed earlier
today" means it is in `4bebe6e` and CI has seen it. "The dead-wire round" means
the morning pass that hunted code nothing calls. "This round" means the
trapping-conversion pass, which is the one chasing the crash, and nothing has
compiled it.

---

## The crash this round was chasing, in four lines

1. `ARCaptureService` read ARKit's tracking state and used it for one bracket
   check. There was no tracking gate.
2. `Pose.fromARKitCameraTransform` inverted the camera transform with no
   determinant check, and `simd_inverse` of a singular matrix is silently NaN
   in every element.
3. Every world position from that frame was NaN.
4. Each one landed on `Int64((position.x / size).rounded(.down))`, and
   **`Int64(someFloat)` is a TRAPPING conversion**: it kills the process, in
   release as well as debug, on NaN, on infinity, and on any finite value
   outside the destination range. Memory is never involved, which is exactly
   why "it crashes with RAM free" was the clue that cracked it.

Those three lines were found in five separate files. Steps 1 and 2 were fixed
earlier today. This round finished the rest and put a gate in front of it.

---

## What this gate checked mechanically, and what it did not

Seventeen Swift files changed in this round: eleven by the five parallel agents
and six more by this gate. No `.metal` file changed (the newest is timestamped
before the round started), and `Contracts.swift` was appended to and reverted
byte for byte during a negative control, so it carries a modified timestamp and
no change.

Checked, over all 119 Swift files in `ios/Sources`:

* **Braces, parentheses and brackets**, with line comments, block comments and
  string literals stripped first. 0 imbalances, and no depth going negative
  anywhere.
* **Top-level type, protocol and typealias names**, for collisions across the
  one Xcode target: 415 names, **0 duplicates**.
* Every cross-file symbol the new code names, resolved against its declaration:
  `PrePassVoxelFrame.cellIndex(_:)` (internal `static func` on an internal
  struct, in the same single target, called six times from
  `TrainerInitializer.swift`), `CaptureCoverageField.voxelIndex(_:_:)` (called
  from `CapturePointCloudAccumulator` and twice from `CaptureRevisitDetector`
  rather than copied), `SmartMath.pixelIndex(_:width:height:)` (new, called
  from `SmartCore.rgbNearest` and `TwoScaleTrustField.swift:363`),
  `PrePassColmapWriter.colorByte(_:)` (new, private, called three times inside
  its own enum, whose closing brace is at `PrePassBinaryIO.swift:459` so the
  helper really is inside it), `SmartCamera.nativePixelInImage(u:v:...)`
  against its `SIMD2<Float>` return and `SmartImage.rgbNearest(_:)`,
  `Pose.isFinite` (`Contracts.swift:251`), `ViewerMath.clamp` in both
  overloads. All resolve, all in-target, none needing a cross-module change.
* **No new top-level declaration and no new stored property** in any of the
  seventeen files. The only two new declarations are the private/internal
  static funcs above, and both are wired: `tools/deadwire.py` still reports
  exactly 112 candidates, the same number as before this round.
* **No `@available` and no `#available` added**, so nothing new can be an
  unguarded iOS 18 API on an iOS 17 deployment target.
* **Concurrency**: nothing new crosses an isolation boundary. The two new
  functions are pure and take value types; the new log line in
  `ARCaptureService.finish()` reads two private `Int` counters from inside a
  method of the same type, the same way the sibling log line at line 624
  already does.
* By name, the exact things that have broken this project's CI before: bare
  `abs` binding to C's integer `abs` (19 uses in the changed files, every
  argument a `Float` or a `Double` except one pre-existing `Int` at
  `ARCaptureService.swift:1308`, which is Swift's own `abs`), Swift 6 only
  syntax under language mode 5.0 (none), a stored property no initialiser
  assigns (no stored property was added), `Data.append` and `withUnsafeBytes`
  (untouched), a public init exposing an internal type (no new init).
* The 13 `static_assert(sizeof(...))` lines in `TrainerShaders.metal` against
  the 13 stride checks in `TrainerGPULayouts.verify()`. They match one for one,
  and no GPU struct changed this round, so none of the asserts moved.
* `.github/workflows/ios.yml` parsed as YAML. Three jobs, the two `if:`
  expressions well-formed, and `build` still has no `needs:` on the checks.

NOT checked exhaustively, and named so nobody reads more into this than is
there: protocol requirements against every conformer, and non-defaulted stored
properties against every initialiser, across all 119 files. Both were checked
for what changed this round, not repo-wide. Type-checker time on the longer
expressions is a judgement, not a measurement.

That is a static sweep, not a compiler. **CI is still the only compiler.**

---

## The two gates, and what they cost

### `tools/deadwire.py` (dead-wire, from the morning round)

Reports every declaration whose name appears nowhere else in the module. 112
candidates remain and every one is listed with a reason in
`tools/deadwire_allowlist.txt`. Exits non-zero on anything else, so the build
goes red for exactly one reason: somebody wrote new code that nothing calls.

**Exit code on the current tree: 0.**

### `tools/trapconv.py` (trapping conversions, new this round)

Reports every float-to-integer conversion in the tree. It is deliberately
over-eager: it flags anything converting a float-shaped expression, and it
knows one thing that is easy for a person to miss. Swift's `min(x, y)` is
`y < x ? y : x` and `max(x, y)` is `y >= x ? y : x`, so NaN compares false, the
SECOND argument is absorbed and the FIRST propagates:

    UInt8(max(0, min(255, x)))          -> 255 when x is NaN.  SAFE.
    UInt8(min(max(x * 255, 0), 255))    -> NaN when x is NaN.  TRAPS.

Those two lines look identical. `PrePassBinaryIO.swift:410` was the second
form. `ViewerMath.clamp` was the second form. `PrePassMath.percentile` was the
second form.

76 candidates on the current tree, every one read in the source and listed with
a written reason in `tools/trapconv_allowlist.txt`, so it goes red for exactly
one reason: somebody added a NEW untriaged conversion.

**Exit code on the current tree: 0.** Verified both ways: a decoy
`Int((v / 2).rounded()` added to `Contracts.swift` produced exit 1 naming the
file and line, and was then removed.

**A bug in the detector was fixed while curating the list, and it mattered.**
Its comment/string stripper blanked a backslash and the character after it, and
inside a Swift multi-line string a backslash at the end of a line is a
continuation, so that deleted a newline. Ten files were affected and
`MetalSplatTrainer.swift` was reported fifteen lines out. Every line-keyed
allowlist entry written from the old output would have pointed at innocent
code. Newline parity now holds in all 119 files.

Both gates run as two Python steps in one job, `Static gates (dead code,
trapping conversions)`, on a cheap Ubuntu runner, on every push to every
branch. **Neither blocks the IPA**: the archive job has no `needs:` on them,
because the phone build is how the owner gets a working app and a lint finding
must never be the reason he cannot install one. The job's display name changed
this round, which matters only if a branch-protection rule names the old
string.

---

| Module | Status | Reached from | What is real, and what is not |
|---|---|---|---|
| **core** (`ios/Sources/Core/`) | COMPLETE | Every other module | The shared vocabulary: cross-module types and the twelve service protocols, `BrandConfig` reading the product name out of the Info.plist so no Swift file hardcodes it, the one ARKit to COLMAP pose conversion, `TrainingBudget.recommended`. Committed earlier today, and it is step 2 of the crash: `Pose.fromARKitCameraTransform` now takes `simd_determinant` of the camera-to-world matrix and returns identity, not NaN, when it is non-finite or under `1e-6`. Verified in place at `Contracts.swift:273`, with `Pose.isFinite` at line 251 as the second half of the answer. **Unchanged this round**: the file was appended to and reverted byte for byte during the gate's negative control, so its timestamp moved and its contents did not. `Contracts.swift:1648` is on the trapconv allowlist as UInt64 integer division with a guarded divisor. Two known contract gaps remain, both worked around at the call site and both filed in `INTEGRATION_REQUESTS.md`: `SplatTrainer` has `cancel()` and no way to ask whether a run has ended, so Pipeline downcasts to `MetalSplatTrainer`; and `TrainerProgress` still has no field for "I lowered the budget", so Pipeline recognises one by matching the wording of a sentence. A third is newly refused in writing: a `framesSkippedNoTracking` field on the capture bundle, refused for this pass because adding a stored property to a hand-initialised `Codable` struct is one of the two shapes that has actually broken this CI, and it must not land in the same change as a crash fix. |
| **app-shell** (`ios/Sources/App/`) | COMPLETE | `@main` | The first-run compatibility gate, the three tabs, and one integration block that registers Capture, PrePass, Trainer, Viewer, Export and Booster with no line commented out. `NimbusBootstrap.install` logs the registered module list at launch and logs an `error` naming any of the seven expected modules that did NOT register; `ScreenUnavailableView` prints what did load underneath the name of what did not. Untouched this round. |
| **onboarding** (`ios/Sources/Onboarding/`) | COMPLETE | The app shell, on first launch | Probes ARKit, Metal, memory and the device model, decides full / limited / incompatible, and explains the verdict in sentences about this specific phone. `findingsForDisplay()` returns `DeviceFindingsForDisplay` carrying `StoredDeviceReport.isStale`, so the "Checked" row says outright when the numbers came from an older build, an older iOS, or over a month ago, and the details table has a row for whether the phone has the laser scanner at all. Untouched this round. Labelled uncertainty, unchanged: the iPhone 17 rows in the chip table are inferred from the naming pattern rather than confirmed against shipped hardware, and `AppleSiliconCatalog.swift` says so in a `TODO(nimbus)`. The owner's phone got past this gate, so the "not incompatible" branch has been exercised once. |
| **capture** (`ios/Sources/Capture/`) | COMPLETE | The Scan tab | The ARKit session and everything it writes: native 256x192 depth and confidence sidecars from `sceneDepth` and never the smoothed variant, the crash-safe `frames.jsonl` log, the COLMAP text model, mesh chunks with per-face classification, anchors logged twice for a free drift measurement. Live: three-channel coverage painted on the room, a gyro-driven blur meter, spoken and haptic guidance, window mode, exposure bracketing, thermal and storage guards. **The one part of this app with real hardware evidence, and the origin of the crash.** Steps 1 and 2 of the crash are fixed and verified in place: `ARCaptureService.swift:614` now gates on `quality != .notAvailable, quality != .limitedInitializing` and then on `pose.isFinite`, counting each rejection separately. `CaptureCoverageField.voxelIndex` guards `isFinite` and clamps to the 21-bit key range BEFORE `Int64(...)`, in the absorbing argument order, and `CapturePointCloudAccumulator` and `CaptureRevisitDetector` both CALL it rather than carrying a fourth and fifth copy, which is the duplication that spread this bug in the first place. THIS ROUND: `CaptureCoverageField.directionBucket` clamped its azimuth as an `Int` after the conversion, one step too late, and now clamps the `Float` before it, matching the elevation line beside it; and the tracking-drop counter that the gate's own comment promised to the census is finally readable, because `finish()` now logs both drop counts next to the frames kept. Verified SAFE and allowlisted rather than changed: `CaptureCOLMAPWriter.swift:167-169` is `SIMD3<UInt8>`, not a float at all. Unchanged and untested: the owner's complaint that the blur warning fires too readily for shaky hands. |
| **prepass** (`ios/Sources/PrePass/`) | COMPLETE | `ScanProcessingCoordinator`, and the capture report for the quick card | The no-training pass, orchestrated by `PrePassPipeline`: submaps, revisit detection, pose-graph optimisation, camera-to-IMU offset calibration, free-space carving where unknown is never treated as empty, glass detection, the initial Gaussian set, the quality card, and a second COLMAP model under `prepass/sparse_refined/`. Per-stage error boundaries, so a failed carve cannot cost you the poses. **The module with the most real traps, and every one of them turned on a number read back off disk.** `PrePassVoxelFrame.cellIndex` is the shared non-trapping floor-and-clamp and is now the only place in the module that converts a voxel coordinate. `PrePassColmapWriter.pointsText` wrote colour bytes as `UInt8(min(max(c.x * 255, 0), 255))`, the propagating order, and now goes through one `colorByte` helper with the literals first. THIS ROUND the gate found four more that the module's own agent did not: `PrePassCarver.swift:321` was the one reader of `lidarMaxRangeMeters` that did not clamp it, so a corrupt bundle's range killed the carve; `PrePassPoseRefiner` built `duration` as `Swift.max(end - start, 0)` over two timestamps decoded from the frame sidecar with no validation, and that order propagates NaN into a trapping `Int(...)` sixteen lines later; `PrePassSurvey` divided the covered-cell count by a cell count that can be zero, and `0/0` is NaN, not zero, which then reached the QC card; and `PrePassSurvey.format` guarded `isFinite` only, which lets through every finite value up to 3.4e38 while the conversion traps above 9.2e18. All four are fixed. `PrePassMath.percentile` had the same wrong-order clamp as `ViewerMath.clamp` and is now written literal-first. Unchanged weakness: `run(bundle:at:)` has no re-entrancy guard and no "have I finished unwinding?" hook. |
| **smart** (`ios/Sources/Smart/`) | PARTIAL | PrePass and Trainer, which construct these directly | Real: the two-scale trust field (bias averaged, noise never averaged), the direction-only frozen background model, the native-resolution depth edge classifier, the per-pixel authority map, and the disparity-space anchoring that would turn a relative monocular depth map into a metric one. STUB, and still the only one in the iOS app: `SmartMonocularDepthStub`, the mid-range depth estimator, the default `midRegimeProvider` in `DirectionalBackgroundModel`. THIS ROUND: every projected pixel in the module now goes through one new `SmartMath.pixelIndex(_:width:height:)`, which guards `isFinite` and the image bounds and returns an Optional, so a partner pose with a huge or non-finite translation drops one sample instead of the app. It is called from `SmartImage.rgbNearest` and from the plane sweep at `TwoScaleTrustField.swift:363`; both call sites were verified, which is what keeps it off the dead-wire list. Verified SAFE and allowlisted rather than changed: `SmartMath.clamp` and `TrainerMath.clamp` are both `simd_clamp`, which is the `fmin`/`fmax` family and absorbs NaN from either side, so the four percentile and gate conversions built on them cannot trap. Carried over from the dead-wire round and still true: `SmartLossSettings.economy` now selects itself from two measurements and logs which preset ran; `midRegimeProvenance` reaches `model/background.json`; `SmartGlassMask.confirmedFraction` warns once with a denominator behind it. |
| **trainer** (`ios/Sources/Trainer/`) | COMPLETE | `ScanProcessingCoordinator` | The owner's own 3D Gaussian splatting trainer on Metal: full forward and backward rasterisation, SSIM, depth and regularisation losses, budget-first densification that relocates rather than exceeding the cap, slice-train-merge, and a governor that lowers the budget as the phone heats up. **THIS ROUND holds the fix most likely to be what the owner was actually hitting.** `TrainerInitializer.loadPrePassSet` reads `prepass/init_splats.ply` and copied `cloud.positions[i]` into a seed with no finite check, then handed those positions to `medianNearestSpacing`, which converted them with six bare `Int32((position.x / cell).rounded(.down))` calls. Any scan recorded before the capture tracking gate landed has a NaN position on disk, and this is the PREFERRED seeding path, so it was the ordinary case, not the exotic one: the trainer died at slice start on the line that measures spacing for a log message, with memory free. Both loops now drop non-finite positions and both call `PrePassVoxelFrame.cellIndex`. The depth fallback seeder additionally requires each axis under 1e6 metres, because `voxel` floors at 0.005 m and multiplies the position by up to 200 before the conversion, so a finite-but-absurd depth affine read off disk traps just as hard as a NaN. One hardening by the gate: the thermal pause slept for `UInt64(Swift.max(interval, 1) * 1e9)` in the propagating order and is now literal-first and capped at 60 seconds. Committed earlier today and unchanged: the depth-loss normalisation (the laser had outweighed the photograph by roughly four orders of magnitude), the SSIM window re-derived from `ssimSigma` at startup, the 16-bit tile ceiling. Honest limits: every GPU struct is fp32, so `useHalfPrecision` is recorded as false whatever was asked for; three `TrainerTuning` fields are read by nothing and each says so at its declaration. **Executed exactly once, and that run produced almost nothing.** |
| **pipeline** (`ios/Sources/Pipeline/`) | PARTIAL | The scan library, and the review screen's "Next step" link | `ScanProcessingCoordinator` drives `PrePassService.run` then `SplatTrainer.train` for one scan, offers an existing `prepass/` for reuse rather than silently redoing twenty minutes of work, sizes a `TrainingBudget` from the device tier plus the scan's measured extent plus `os_proc_available_memory()`, keeps every budget reduction where the user can read it, and refuses to file a finished model under a scan it does not belong to. THIS ROUND: `ProcessingFormat.meters` guarded `value.isFinite` and nothing else, so a finite-but-absurd scene extent from a half-written pose file killed the app on the check-over screen, in the sentence that explains the plan before training starts. It now refuses above 100 km and says "an unknown distance" rather than inventing a number. PARTIAL for one reason only, and the file's own header states it: stopping is cooperative and genuinely waited for on the training half (`waitUntilIdle`), and the same guarantee does not hold for the check-over, because `PrePassService` has nothing to ask. |
| **viewer** (`ios/Sources/Viewer/`) | PARTIAL | The Scans tab | Real: the Metal splat renderer, the honesty mask that hatches directions the scan never saw, the artefact heatmap, the fly-through pinned to where you actually walked, the photo-versus-scan slider, and the library, review and export screens. THIS ROUND: `ViewerMath.clamp` was `Swift.min(Swift.max(v, lo), hi)`, the NaN-propagating order, in both the `Float` and the `Double` overload. It is the module's shared clamp, so it was feeding two separate crashes downstream, including the census histogram bin at `ScanCensus.swift:521`, where a splat whose log-scales are all above about 88.8 passes the `isFinite` check and then overflows `exp()`. Both overloads are now literal-first, identical for every finite input. `PreviewCameraPathBuilder` gained finite checks on the camera centres it samples and a guarded `ceil` on the decimation stride, because a zero denominator there makes `Int(infinity)`. Verified SAFE and allowlisted rather than changed: both `ObservedDirectionField` cell conversions, where `rel >= 0` rejects NaN and negatives and `rel < maxCoordinate` rejects infinity, and the quarter-turn count in `ViewerSupport`. Still not real: `ScanSummary.heldOutPSNR` is assigned at `ScanLibraryStore.swift:332` and displayed by nothing. Still flat grey: nothing in this module reads `model/background.bin`, so the trained far field appears in no preview. That is the largest thing left open and it is rejected in writing, with reasons, in `INTEGRATION_REQUESTS.md`. |
| **export** (`ios/Sources/Export/`) | PARTIAL | The export screen, the Booster, the trainer | Real: `.ply` write and read, `.spz` versions 1 to 3 write and read, `.glb` with the `KHR_gaussian_splatting` extension plus a fallback colour attribute so a plain glTF viewer still shows something, ZIP packaging of a capture bundle, and the share sheet. `SplatCloud.fuse3DFilter(_:)` folds the Mip-Splatting 3D low-pass filter into the stored log-scale and opacity, exactly, and is called from `MetalSplatTrainer.readCloud`. `SplatCloud.filter3DFused` is `true` / `false` / `nil`, and `nil` means "a file that cannot say", never a guess. A labelled refusal rather than a guess: `.spz` version 4 read throws `unsupportedVersion`, because nobody has published a sample. **No file in this module changed this round.** Its one trapconv candidate, the quaternion packer at `SPZCodec.swift:361`, was read and is genuinely safe: the rotation is length-checked and replaced with identity before `simd_normalize`, fourteen lines above the conversion. `ExportSelfTest.runAll()` runs in DEBUG builds only, from the export screen; the `TODO(nimbus)` asking for a real test target still stands. |
| **booster-client** (`ios/Sources/Booster/`) | COMPLETE | The Booster tab, and the "send it to a computer" section of the processing screen | Bonjour discovery, code-confirmed pairing with the token in the Keychain, chunked resumable upload, live progress over a WebSocket with a polling fallback, and resumable download of the result. Pure Swift, no third-party dependencies. The routes and payload field names agree with `docs/BOOSTER_PROTOCOL.md` and the Python server. THIS ROUND: `BoosterUploadManager` gained `clampedOffset(_:fileByteCount:)`, which forces a byte offset the server reported into the file's real range before it is used, because a negative offset traps the `UInt64(_:)` conversion the seeks need and an offset past the end would seek off the back of the file. Its one trapconv candidate, the retry backoff at line 213, was read and is safe: `withRetry` is private with one call site that never overrides the default of three attempts, so the exponent is 0 or 1. **Never run against a real Booster.** |
| **booster-pc** (`booster/`) | PARTIAL | Run by hand on a PC | Real: the aiohttp server, pairing, resumable chunked transfer, job queue, the PySide6 window, and a pipeline that turns an uploaded scan into real geometry (depth unprojected on its native grid, normal-aligned Gaussian discs, free-space carving) and writes `model.ply`, `model.spz`, `model.json` and a preview image the phone can download. Not written, and it says so rather than pretending: there is no `trainer/train.py`, so there is no photometric optimisation loop. `trainer/pipeline.py` line 26 states this in the file, `optimiser_available()` returns false, and the result carries `optimised: bool = False` so nothing downstream can mistake a raw initialisation for a trained model. That step will need an NVIDIA GPU with CUDA PyTorch and gsplat. Untouched this round. Note for whoever writes it: **Python has none of this problem.** `int(float('nan'))` raises a catchable `ValueError`, it does not kill the process, so the trap class this round eliminated does not port to the PC side. |
| **blender-addon** (`blender_addon/`) | PARTIAL | Installed by hand in Blender | Real: `.ply` import and a geometry-nodes plus shader setup that renders the splats in Blender 4.x. Partial: `spz_reader.py` is written from the format description with no real sample file to test against, and says so. Untouched this round, and it remains the main beneficiary of the 3D-filter fuse: it will now draw the trained model rather than a sharper, more solid version of it. |
| **ci** (`.github/workflows/ios.yml`) | COMPLETE | GitHub Actions | XcodeGen generates the project from `ios/project.yml`, `xcodebuild archive` builds it unsigned on `macos-15`, and the workflow hand-zips `Payload/<Product>.app` into an `.ipa`, finding the product name by globbing so a rename cannot break it. A separate Ubuntu job byte-compiles and lints `booster/`. Source directories are listed recursively, so a new `.swift` or `.metal` file is picked up with no edit here. THIS ROUND: the checks job gained a second Python step running `tools/trapconv.py`, and the job's display name changed from `Dead-wire check (code nothing calls)` to `Static gates (dead code, trapping conversions)` to match what it now does. Same runner, same ten-minute timeout, same absence of a `needs:` from the archive job. Both steps fail with an explicit `::error::` if their script or their allowlist has been deleted, because a check that silently stops checking is worse than no check. **It remains the only compiler this project has, and it has not yet seen either of today's last two rounds.** |

---

## The one thing that is true of the whole project

The route exists end to end and has been walked once, by a person, on a phone.
A scan was recorded, checked over, built into a model, and reviewed. What came
out of it was almost nothing, and since then the app has been dying mid-use with
memory free.

Every fault found has had one of two shapes. Either something was silently doing
nothing and the app reported success, or something was silently doing arithmetic
it could not survive and the app died without a word. The first shape now has a
gate. As of this round, so does the second.

Counted honestly, this round changed a float-to-integer conversion, or the
guard in front of one, in sixteen files. Five of those the gate rates as traps
that could fire on the owner's phone as it stands, and they are, in the order
they would be met:

* `TrainerInitializer.medianNearestSpacing` (training). The strongest one.
  `prepass/init_splats.ply` is read straight into seed positions with no finite
  check, any scan recorded before this morning's tracking gate has a NaN
  position in it, and this is the preferred seeding path. It would have fired
  on the next attempt.
* `ProcessingFormat.meters` (check-over), on the screen that explains the plan
  before training starts, if the scene extent read back off disk is finite and
  absurd.
* `PrePassSurvey.coverageFraction` (check-over), on any scan that recorded
  depth and did not record one usable cell, because `0/0` is NaN.
* `ViewerMath.clamp` feeding `ScanCensus` (review), for a model whose
  log-scales overflow `exp()`.
* `PrePassColmapWriter.pointsText` (exporting), for any future caller handing
  it a colour it did not read out of an image.

The rest are the same shape without a live trigger today: a clamp written in
the propagating order, or a guard that checks `isFinite` and not range. They
were changed anyway, because "safe by agreement with a guard one stack frame
away" is exactly the arrangement that let the original crash through.

Five separate sites were found by this gate AFTER all five agents reported
success, in `PrePassCarver`, `PrePassPoseRefiner`, `PrePassSurvey` (twice) and
`PrePassMath`. That is the argument for the gate, and for the tool: the agents
were reading a report, and the report's line numbers were fifteen lines out in
the file with the most of them.

The next thing that matters is a green CI run, and then a second scan.
