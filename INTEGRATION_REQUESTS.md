# Integration requests

Changes one module needs in another module's files, written down instead of
made. Every module writes only inside its own directory (CONTRACTS.md section
1, rule 6), so a cross-module need lands here as one bullet naming the file,
the symbol and the change. Append; never overwrite; never delete someone
else's bullet.

---

## From Capture (ios/Sources/Capture)

- **`ios/Sources/App/NimbusApp.swift`, `NimbusBootstrap.registerAvailableModules()`:**
  uncomment the two Capture lines, and use the shared instance rather than a
  fresh one:
  `services.capture = ARCaptureService.shared` and
  `ui.captureScreen = { AnyView(CaptureScreen()) }`.
  It has to be `.shared` because `ARCaptureService` owns an `ARSession` and two
  sessions competing for the same camera give you neither. `CaptureScreen`
  already resolves the registry first and falls back to `.shared`, so the
  screen works either way, but a second instance created here would sit there
  holding a camera nobody is looking at. The comment above those lines
  ("`ARCaptureService` ... does not exist yet, and neither does the capture
  screen") is now out of date: both exist.

- **`ios/Sources/Core/Contracts.swift`, `AnchorRecord` (optional, not
  blocking):** add an optional provenance field, for example
  `public var isUserMarked: Bool = false`. Window mode lets the user tap "mark
  this window", which places a named `ARAnchor`; `CaptureAnchorRecorder` writes
  it out classified `.window`, which is true but indistinguishable from a label
  ARKit produced. The pre-pass's glass detector (F5) would reasonably want to
  weight a human's opinion differently from the classifier's. Capture writes
  the honest label today and does not need this to work.

- **`ios/Sources/PrePass` (no change, a dependency worth knowing about):** the
  post-capture quality card comes from `PrePassService.quickQCCard(bundle:at:)`
  through `NimbusServices.shared.prePass`. While that service is nil the
  capture screen says so in plain words and shows only what the recording
  itself measured. Registering `PrePassPipeline` is all that is needed to turn
  the real card on; nothing in Capture has to change.

---

## From PrePass (reconstructed by Capture's agent, 2026-09-04)

- **`ios/Sources/App/NimbusApp.swift`, `NimbusBootstrap.registerAvailableModules()`:**
  uncomment `services.prePass = PrePassPipeline()`.
  RECONSTRUCTED, NOT WRITTEN BY PREPASS: this file was created by Capture's
  agent while PrePass's agent was working in parallel, and PrePass's own bullet
  may have been overwritten in the process. The wording above is taken from the
  PrePass row in MODULE_STATUS.md ("the module is not yet wired into
  NimbusApp.swift's integration block, that is one commented line in another
  module's file, requested in INTEGRATION_REQUESTS.md") and not from the
  original bullet. PrePass's agent should replace this with whatever it
  actually needs.

---

## Resolved by the integrator, 2026-09-04

Every request above has been actioned. Nothing is left open. Kept rather than
deleted, because the reasoning is the useful part.

- **Capture's first bullet (wire the Capture lines in `NimbusApp.swift`):
  DONE, exactly as asked.** The integration block now reads
  `services.capture = ARCaptureService.shared` and
  `ui.captureScreen = { AnyView(CaptureScreen()) }`. The `.shared` reasoning is
  right and is now written into the comment above those lines so the next
  person does not undo it. The stale "does not exist yet" comment is gone.

- **Capture's second bullet (`AnchorRecord.isUserMarked`): DONE, and taken
  further than requested.** `ios/Sources/Core/Contracts.swift` now declares
  `public var isUserMarked: Bool`, defaulted to `false` in the memberwise
  initialiser so no existing call site changes. It needed one thing the request
  did not mention: `AnchorRecord` gained a hand-written `init(from:)` that
  reads the new key with `decodeIfPresent`, because Swift's synthesised decoder
  treats a missing non-optional key as a hard failure and an
  `anchors_session.json` written by an earlier build has to still open.
  `CaptureAnchorRecorder.record(for:firstSeenFrame:)` now sets it from
  `anchor.name == userMarkedWindowAnchorName`, so the flag carries real
  provenance rather than always reading false, and
  `docs/DATA_FORMAT.md` section 6 documents the field and its optionality.
  Nothing consumes it yet; the glass detector may.

- **Capture's third bullet (PrePass registration): DONE.**
  `services.prePass = PrePassPipeline()` is live, so the post-capture screen
  now gets the real quality card from `quickQCCard(bundle:at:)` instead of the
  fallback sentence. Capture needed no change, as it said.

- **PrePass's reconstructed bullet: DONE, and the reconstruction was correct.**
  `services.prePass = PrePassPipeline()` is registered. One thing was added
  that the bullet did not ask for and that the module clearly wants:
  `RootView.runCompatibilityCheck()` now sets `PrePassPipeline.deviceTier` from
  the compatibility report, so the suggested training budget is sized for the
  actual phone rather than falling back to "whatever this process can allocate
  right now".

### Two cross-module fixes the integrator made that nobody had filed

- **`ios/Sources/Trainer/MetalSplatTrainer.swift` now writes
  `model/held_out_frames.json`.** `Sources/Viewer/HeldOutFrames.swift` reads
  that file to prove which photos the photo-versus-scan slider is allowed to
  use, and falls back to "every 20th frame, treat this as a spot check" when it
  is absent. The trainer already knew the answer (`slice.heldOutKeyframes`) and
  was throwing it away. Written as a bare array of frame indices, which is one
  of the two shapes the reader already accepts, and only when the set is
  non-empty (an empty file would be a different and weaker claim). This was
  logged as "THE GAP" in Viewer's status row; it is closed for on-device
  models. A model that came back from the PC Booster still has no such list.

- **`MetalSplatTrainer` now clears `exposureRecords` at the start of each
  run.** It was accumulating for the life of the object, and the app registers
  one trainer for the life of the process, so a second scan's `exposure.bin`
  would have carried the first scan's per-frame exposures. `heldOutFrameIndices`
  is cleared in the same place for the same reason.

---

## From Pipeline (ios/Sources/Pipeline), 2026-09-04

The processing coordinator and the "Finish this scan" screen were written
against the contracts exactly as they stand. Nothing below is blocking: each
one is a place where the code has to work around a contract rather than use it,
and the workaround is documented at the site.

- **`ios/Sources/Core/Contracts.swift`, `TrainerProgress`: add an explicit
  signal that the budget was lowered.** For example
  `public var budgetChangeMessage: String?` (nil on every ordinary tick), or a
  `TrainerStage.budgetLowered`-style flag alongside the stage. WHY: the
  `SplatTrainer` contract says the trainer "owns the right to LOWER `budget` ...
  and reports having done so in `TrainerProgress`", but `TrainerProgress` has no
  field for it, so the only channel is the free-text `message`.
  `ScanProcessingCoordinator.recordNotices(from:)` therefore recognises a
  reduction by matching the "Because ..." prefix that
  `TrainerBudgetChange.message` always produces. That is a workaround against
  wording, not against a contract: it can miss a notice or keep a harmless one,
  and the sentence it shows is the trainer's own, but it should not have to
  guess. The run's final claim about size does NOT depend on it - that comes
  from comparing `SplatModel.budgetUsed` with the budget passed in, which is
  contractual.

- **`ios/Sources/Trainer/MetalSplatTrainer.swift` (would become unnecessary if
  the above lands): nothing to change today.** Recording it here so the next
  person does not "fix" the prefix match by editing the trainer's wording. If
  those sentences in `TrainerBudget.swift` are ever reworded, the coordinator
  stops noticing budget reductions mid-run. The end-of-run comparison still
  works, so the failure mode is a quieter screen, not a wrong one.

- **`ios/Sources/PrePass/PrePassPipeline.swift`, `progress`: consider a
  per-run stream instead of one for the object's lifetime.** `progress` is
  created in `init` and finishes only in `deinit`, and dropping an
  `AsyncStream`'s iterator finishes it permanently, so a consumer that
  subscribes per run silently loses every stage name from the second run
  onwards. `ScanProcessingCoordinator` works around this by being a singleton
  that subscribes exactly once and never cancels. A stream returned per `run`
  call, or a stage tick folded into the `AsyncThrowingStream<PrePassResult, _>`
  the protocol already returns, would remove the constraint. Not blocking, and
  the current shape works.

- **`ios/Sources/Booster` (no change, a dependency worth knowing about):** the
  processing screen offers the Booster as the alternative for a scan too big
  for the phone. It lists `BoosterClient.shared.discovery.devices` filtered to
  paired and reachable, calls `BoosterService.sendScan(scanID:scanDirectory:to:)`
  and then sends the user to the Booster tab to watch it, rather than
  duplicating that tab's progress UI. It calls `discovery.start()` and
  deliberately never calls `stop()`, because that object is shared with the
  Booster tab and stopping it from another screen could take the tab's own
  device list away.

### Two documents outside Pipeline's reach that are now out of date

- **`README.md`, "What is not finished", item 1.** It says "Nothing on the
  phone starts the model builder yet ... The scan list even says 'Ready to
  build the 3D model' and there is nothing to tap." There is now something to
  tap: the library pushes `ScanProcessingScreen` for any scan without a model.
  Replace the item with the honest successor rather than deleting it - the
  route exists and has still never been executed on a phone, which is exactly
  what `MODULE_STATUS.md` now says for the pre-pass, the trainer and Pipeline.

- **`CONTRACTS.md`, the module table (around line 71).** It lists ten source
  directories and their owners; `ios/Sources/Pipeline` is missing. Its
  dependencies are Core, Viewer (it reads `ScanLibraryReader` / `ViewerScanPaths`
  and pushes `ScanReviewScreen`) and Booster (the hand-off), and it owns no
  contract type of its own.

### From the app shell and the scan library (defects 3 and 4)

- **`Sources/Capture/CaptureScreen.swift`, the post-capture report.** Where the
  report panel currently only calls `model.dismissReport()`, also call
  `AppNavigation.shared.showSavedScan(scanID)` with the saved scan's id (the
  `CaptureBundleRef.scanID` the capture just wrote). That one call switches the
  user to the Scans tab and asks the library to lead with the scan they just
  recorded. `AppNavigation` is declared in `Sources/App/NimbusApp.swift`, is
  `@MainActor`, and is reachable from Capture: everything is one target.
  `AppNavigation.shared.showScans()` is there too, for a "See my scans" button
  that should not single a scan out.

- **`Sources/Viewer/ScanLibraryScreen.swift`, the list.** `ScanLibraryStore`
  now publishes `nextUpScan` / `nextUpScanID` (the newest scan waiting on the
  user), and `ScanSummary` publishes `primaryActionTitle` and `isWaitingOnYou`.
  Please lead the list with that scan: a section above the rows, or a marked
  first row, with a button whose label is `primaryActionTitle` rather than a
  generic "Open". The button words are written to match the row's own
  `nextStep` sentence, so the two must be shown together, not one without the
  other. If `AppNavigation.shared.scanToLeadWith` names a scan, lead with that
  one instead and set `scanToLeadWith = nil` once you have.

- **`Sources/Pipeline/ScanProcessingScreen.swift`.** `ScanLibraryReader.readDetail`
  now writes a plain-language sentence into `detail.summary.problem` when a
  `capture_bundle.json` or `prepass_result.json` is in a format version this
  build does not understand. The screen has that string and does not show it
  yet; showing it would replace an empty-looking screen with the reason.

- **`Sources/Pipeline/ScanProcessingCoordinator.swift`, line ~244.** Now that
  `readDetail` refuses a bundle whose `formatVersion` is unknown, `detail.bundle`
  is nil in that case and the user reads "this scan has no readable index
  (capture_bundle.json) in it", which is not what happened. Consider reporting
  `detail.summary.problem` when it is non-nil before falling back to that
  sentence. The version guard at line ~249 stays correct either way; it is now
  a second line of defence rather than the only one.

### From Pipeline: two contract gaps that "stop" runs into

Both come from the same fact: stopping is cooperative. `Task.cancel()` sets a
flag, and the work behind it carries on until it reaches a place where it looks
at that flag. So "cancel returned" and "it has stopped" are two different
moments, and until now nothing in the contracts let a caller tell them apart.

- **`ios/Sources/Core/Contracts.swift`, `SplatTrainer`.** The protocol has
  `cancel()` and no way to ask whether the run has actually ended.
  `ScanProcessingCoordinator` must know, because starting a second run while
  the first is still on the GPU would put two loops on one set of buffers. Two
  members would close it, and `MetalSplatTrainer` already implements both under
  these names, so this is a promotion rather than new work:

      /// True while a run is on the GPU, or still unwinding after a cancel.
      var isTraining: Bool { get }

      /// Returns once a run that was told to stop has genuinely stopped.
      func waitUntilIdle() async

  Until they are in the protocol, Pipeline downcasts to `MetalSplatTrainer` and
  a trainer that is anything else is simply not waited for. That is a real hole,
  not a tidy workaround: it is safe only because the app registers exactly one
  trainer and it is that class.

- **`ios/Sources/Core/Contracts.swift`, `PrePassService`.** Same gap, no
  workaround available: there is no concrete hook to downcast to. A stop during
  the check-over cancels the stream, which cancels the work behind it, and the
  coordinator's task then returns without being able to wait for the pre-pass's
  own task to unwind. The screen is honest about this (it says "Stopping. It
  finishes the bit it is in the middle of first."), but the guarantee is weaker
  than the one training now has. The same two members, or a single
  `func waitUntilIdle() async`, would make it the same guarantee.

### From Pipeline: the live training preview is now wired

Not a request, a note for whoever owns Viewer, because three pieces that had no
call sites now have one.

- **`ios/Sources/Viewer/MetalSplatRenderer.swift`, `load(_ cloud: SplatCloud)`**
  is called by `Sources/Pipeline/TrainingPreview.swift` every few seconds during
  a training run, from a private `MetalSplatRenderer` that Pipeline owns (not
  the one `NimbusApp` registers, which is left alone).
- **`ios/Sources/Viewer/SplatPreviewView.swift`** is used as-is, with
  `gesturesEnabled: false` and `isAnimating` driven by a short burst after each
  load, so the surface is parked between refreshes rather than redrawing at
  60 Hz while the trainer needs the GPU. Nothing in Viewer was changed for it.
- Nothing is asked for. If `SplatPreviewView` ever grows a "draw once and stop"
  mode, Pipeline would use it instead of the burst, because the burst exists
  only to let `residencyStep` finish revealing a freshly loaded cloud.

### From Trainer: the sideways preview must be undone in the viewer, and only there

Not a request for a change to Trainer, a request that nobody make one. Whoever
fixes the 90 degree roll needs this on the record, because the roll can be
cancelled in several places and cancelling it in two of them at once would look
right on the phone while quietly corrupting every export.

I read the trainer's frame end to end and it is self-consistent, so the model it
builds is upright in world space:

- `ios/Sources/Trainer/MetalSplatTrainer.swift:1176` and `:1458` both set
  `camera.viewMatrix` to `supervision.pose(for:).matrix`, optionally with the
  learned per-frame delta on the left. No orientation term, no axis flip.
- `ios/Sources/Trainer/TrainerSupervision.swift`, `pose(for:)` returns the
  persisted pose unchanged, and `renderIntrinsics` is
  `bundle.intrinsics.scaled(toWidth:height:)`, a scale only.
- `ios/Sources/Trainer/TrainerShaders.metal:605` projects with
  `cam.viewMatrix * float4(meanWorld, 1)` and `:688` with
  `fx * X / Z + cx`, `fy * Y / Z + cy`, which is the same OpenCV pinhole the
  persisted intrinsics describe, against ground truth packed row-major from the
  decoded JPEG.

So the photograph, the intrinsics and the pose are all in ARKit's native
landscape sensor frame, and they agree with each other. The phone's roll is
present in the image content AND in the camera pose, and the two cancel inside
the optimiser. The world the splats live in is ARKit's gravity-aligned world
(`ARCaptureService` sets `worldAlignment = .gravity`), so a `.ply` or `.gltf`
opens upright in Blender today.

What follows for the fix:

1. Do NOT rotate anything in `Sources/Trainer`, and do not rotate the persisted
   poses, intrinsics or pixels to compensate. The roll is introduced where a
   sensor-frame pose is replayed as a display camera, which is
   `ios/Sources/Viewer/SplatPreviewView.swift` in `.path` mode (the walk-through
   and the A/B compare) feeding `MetalSplatRenderer`'s view matrix. Free look
   builds its own upright basis and should already look level, which is the
   cheapest way for the owner to confirm the diagnosis in five seconds.
2. If capture ever starts writing an EXIF orientation tag, or the pixels start
   being rotated on disk, the trainer will now say so rather than train a wrong
   camera: `TrainerSupervisionBuilder.build` refuses a photo whose aspect does
   not match the capture intrinsics (it would otherwise be absorbed silently,
   because `CameraIntrinsics.scaled(toWidth:height:)` scales fx and fy
   independently). If a bundle needs a display orientation recorded, record it
   as a separate field and leave the pixels and the geometry alone.

---

## From Capture (ios/Sources/Capture), after the first run on real hardware

Two bugs came back from the first ever scan on a phone. The blur half was
entirely inside Capture and is fixed there. The rotation half is not, and this
is the handover.

### What Capture concluded about the 90 degree roll, and why

Read before changing anything, because fixing this in two places would
double-rotate it: it would look right in the preview and corrupt every export.

Capture's persisted data is CORRECT and self-consistent, and must not be
touched. Verified by reading, not inferred:

- `ARCaptureService.resolvedIntrinsics(from:)` copies `camera.imageResolution`
  and `camera.intrinsics` verbatim, so width/height and cx/cy are paired
  correctly and are landscape.
- `Pose.fromARKitCameraTransform` applies only `diag(1, -1, -1, 1)`, an
  ARKit-to-OpenCV camera axis relabel. It contains no roll about the optical
  axis and no orientation term, so the stored pose is in the same landscape
  sensor frame as the intrinsics.
- `CapturePixelBuffer.jpegData()` encodes the untouched buffer with no rotation
  and, by explicit design, no EXIF orientation tag. So the JPEG on disk is the
  landscape sensor image at exactly the resolution `cameras.txt` claims.

Pixels, intrinsics and poses are therefore all in ONE frame and they agree.
`u = fx * Xc / Zc + cx` projects correctly into the photo that is actually on
disk. There is no landscape-versus-portrait mismatch inside the persisted
geometry, so the model is not trained rotated and a `.ply` or `.gltf` opens
upright in Blender. Normalising at write time would also be wrong for a second
reason: `project.yml` permits portrait and both landscapes, `CaptureCOLMAPWriter`
writes ONE shared PINHOLE camera, and the sensor frame is the only frame that
stays constant if the phone is turned mid-scan.

The roll appears where a sensor-frame pose is replayed as a DISPLAY camera into
a portrait drawable. That is Viewer's `.path` mode, not Capture's writers.

### Requests

- **`ios/Sources/Viewer/SplatPreviewView.swift`, `ViewerCameraController` (or
  `MetalSplatRenderer`, whichever the Viewer owner prefers):** in `.path` mode
  the recorded pose is used verbatim (`pose = sample.pose`), while `.freeOrbit`
  rebuilds an upright basis through `ViewerPoseMath.lookAt`. Roll the replayed
  pose about the camera's own +Z (optical) axis before it becomes a view
  matrix. Left-multiply, and rebuild the translation from the CENTRE so the
  optical centre does not move: `r = (roll * pose.rotation.simd).normalized`,
  `t = -(r.act(pose.center.simd))`. Leave `.freeOrbit` alone, it is already
  level. Capture cannot determine the SIGN of the quarter turn from source, and
  will not assert it: put it behind one named constant so flipping it after one
  look on the phone is a one-character change. A cheap discriminating test for
  the owner: if Free look is upright and Walk-through is on its side, this is
  it; if BOTH are on their side, this diagnosis is wrong and the fault is
  further upstream.
- **`ios/Sources/Viewer/ScanReviewModel.swift`:** Compare mode goes through the
  same path, and puts the render beside the raw JPEG (which is landscape and
  untagged). Roll the photo by the same quarter turn, or the A/B wipe stops
  lining up once the render is levelled. Its `compareAspectRatio` needs to flip
  to height/width for an odd number of quarter turns too.
- **`ios/Sources/Core/Contracts.swift`, `CaptureSettings`:** add one OPTIONAL
  field so the viewer has an exact answer instead of an inferred one, for
  example `public var imageQuarterTurnsToUpright: Int = 0`, with a defaulted
  initializer parameter (the memberwise init currently has no defaults, so a
  default keeps every call site compiling) and a `decodeIfPresent(...) ?? 0` if
  the Codable conformance is ever written by hand. Document in the field's own
  comment that it is PRESENTATION ONLY and that nothing in the geometry may
  read it. Capture will populate it from the interface orientation captured at
  session start as soon as the field exists; until then the viewer can fall
  back to the `width > height` heuristic. Per docs/DATA_FORMAT.md section 9 an
  added optional field does not bump `formatVersion`.

### Documentation that Capture's blur change makes stale

- **`CONTRACTS.md` line 413**, the blur meter row, currently reads
  "`deg/s * exposure / 0.0426`, amber 2 px, red 4 px" as if all four numbers
  were contract. They are not the same kind of thing. The formula and the
  0.0426 divisor ARE contract: `FrameQC.motionBlurPixels` persists that exact
  number and nothing about it has changed. The amber and red thresholds are
  CALIBRATION, they are consumed only inside `ios/Sources/Capture`, and they
  are now 4 px and 8 px. Please split the row so the next reader does not
  re-tighten them believing they were fixed by contract. The derivation, in
  full, is in the doc comment on `CaptureTuning.blurAmberPixels`: capture is
  ~1920 px wide, `TrainerBudget.resolutionLadder` tops out at 720 px on the
  long edge and only ever descends, so every capture pixel is divided by 2.67
  before it reaches a Gaussian. The old amber of 2 px was 0.75 of a pixel in
  the image the trainer actually fits, which is below what that grid can
  represent, and the old red of 4 px interrupted the user at 1.5 trainer
  pixels. That is why a person with shaky hands was permanently in the red.
- **`docs/DATA_FORMAT.md` section 2**: worth stating in one sentence that the
  camera frame is ARKit's NATIVE LANDSCAPE SENSOR frame, orientation
  independent, and that a display-upright replay has to apply a roll about
  camera +Z. Nothing on disk says this today, which is why the viewer had
  nothing to read.
- **`PROPOSALS.md` line 52** is where amber 2 / red 4 came from, in the same
  breath as "lock shutter fast". Worth annotating that the shutter cap was
  never built, so those thresholds shipped without the thing that justified
  them.

### Deliberately NOT requested

- **`ios/Sources/PrePass/PrePassPoseRefiner.swift` lines 572 and 581**, the
  `qc.motionBlurPixels < 3` gate on the time-offset sweep, is CORRECT as it is
  and should NOT be moved to track the new HUD thresholds. It is measured on
  the full-size JPEG, where ZNCC really is resolving a shift of about one
  capture pixel, so it lives in capture pixels for a real reason. Under the new
  amber of 4 px, some frames the HUD calls green will be excluded from
  calibration. That is right, and it is the only place in the tree where those
  two scales legitimately disagree.


---

## From Core (ios/Sources/Core), after the first run on real hardware

Core owns the shared vocabulary, so its part in the sideways preview is the
convention itself: the app documented a camera frame without ever saying that
that frame is LANDSCAPE whichever way the phone is held. Every module read the
convention, every module implemented it correctly, and the picture still came
out on its side, because the one fact that mattered was not written down
anywhere. That is now fixed in `Contracts.swift`.

### Which diagnosis Core followed, and why

Twelve diagnoses were filed and two of them are incompatible: one says the
model itself is built rotated, the rest say the model is upright and the
preview camera is rolled. Core checked and agrees with the majority, and this
is on the record because acting on both would double-rotate the fly-through: it
would look right on the phone while corrupting every `.ply` and `.glb`.

Read, not inferred:

- `ARCaptureService.resolvedIntrinsics(from:)` copies `camera.imageResolution`
  and `camera.intrinsics` straight across, so width pairs with cx and height
  with cy, and both are landscape.
- `ARCaptureService` line 569 stores `Pose.fromARKitCameraTransform(camera.transform)`,
  and that function applies only `diag(1, -1, -1, 1)`. That is an ARKit to
  OpenCV axis relabel with no roll about the optical axis, so the stored pose
  sits in the same landscape frame as the intrinsics.
- `CapturePixelBuffer.jpegData()` encodes the untouched buffer, no rotation and
  deliberately no EXIF orientation tag, so the JPEG matches what `cameras.txt`
  claims.
- Nothing in `Sources/Viewer` or `Sources/Core` consults an orientation at all:
  grepping the tree for `interfaceOrientation`, `displayTransform`,
  `viewMatrix(for` and `projectionMatrix(for` hits only
  `Capture/CaptureCoverageRenderer.swift` and `Capture/CaptureScreen.swift`,
  which are the live HUD.

So the three legs on disk agree with each other, the trainer's world is ARKit's
gravity-aligned world, and the roll only appears at the point where a
sensor-frame pose is handed to a renderer as a display camera. THE PERSISTED
DATA IS CORRECT AND MUST NOT BE ROTATED. The fix is presentation, it belongs in
Viewer, and it belongs there once.

### What changed in Core

1. **The geometry-conventions block at the top of `Contracts.swift`** gained a
   "Sensor frame" section: the camera frame is the native landscape sensor
   frame, ARKit does not turn it when the user turns the phone, everything
   persisted lives in it and agrees, and therefore anything putting a captured
   image or pose in front of a person has to turn it upright itself.

2. **`Pose.rolledForDisplay(quarterTurnsClockwise:)`** (new, presentation only).
   Left-multiplies a roll about camera +Z and rolls the translation with it, so
   `center` is unchanged to the last bit: the camera turns on the spot, which
   keeps the fly-through inside its 0.20 m deviation promise by construction.

3. **`CameraIntrinsics.rotatedForDisplay(quarterTurnsClockwise:)`** (new,
   presentation only). Swaps width and height, swaps fx and fy, and carries the
   principal point round with them.

   These two exist together so the SIGN can never disagree between the picture
   and the geometry again. One quarter turn clockwise sends the image's +X to
   the display's +Y and its +Y to the display's -X, which is a right-handed
   +90 degrees about camera +Z. Checked numerically rather than asserted: four
   turns is the identity; 1920x1440 with the principal point at (959.5, 719.5)
   becomes 1440x1920 at (719.5, 959.5), still dead centre; the source top-left
   pixel lands top-right; and for a phone held portrait upright, world up sits
   at camera (-1, 0, 0) before the roll (screen-left, which is exactly the
   "rotated left" the owner reported) and (0, -1, 0) after it, which is
   screen-up.

4. **`CaptureSettings.imageQuarterTurnsClockwiseToUpright: Int?`** (new,
   optional). This is the field Capture asked for above, with two deliberate
   differences from the shape Capture proposed, both of which matter:
   - It is `Int?`, not `Int = 0`. A synthesised `Codable` does NOT fall back to
     a property's default value for a missing key: only an Optional property is
     decoded with `decodeIfPresent`. `CaptureSettings` uses the synthesised
     conformance, so a non-optional `Int` would have made every scan already on
     the phone fail to decode.
   - `nil` means "not recorded", never "zero". A consumer must fall back, not
     assume landscape.
   The name says CLOCKWISE because the missing sign is what caused this bug.

5. **`FrameQC.motionBlurPixels` and `CaptureLiveState.motionBlurPixels`** no
   longer quote "amber at 2 px, red at 4 px" in their doc comments. Those
   numbers are calibration, they live in `CaptureTuning`, and Capture has since
   moved them to 4 and 8. The contract now says what the field IS (a physical
   prediction of smear at capture resolution, never softened for a shaky hand
   and never rescaled to the trainer's smaller supervision size) and says
   explicitly that the HUD colour, the spoken guidance, the QC weight, the
   keyframe gate and the QC card all read one number, so a change has to move
   them together or the app will say one thing and do another.

6. **`PreviewCameraPath.Keyframe.pose` and `SplatRenderer.setCamera(pose:intrinsics:)`**
   gained one doc note each, at the exact seam the bug went through: a keyframe
   derived from a capture inherits the sensor frame, while a renderer expects a
   DISPLAY camera. Renderers draw what they are given and must not go hunting
   for an orientation of their own.

No signatures changed, no field changed meaning, and both new initializer
parameters are trailing and defaulted, so every existing call site still
compiles. Per docs/DATA_FORMAT.md section 9 an added optional field does not
bump `formatVersion`, and it has not been bumped.

### Requests: Viewer (ios/Sources/Viewer)

The roll is yours, and it is one line of policy plus two call sites.

- **`SplatPreviewView.swift`, `ViewerCameraController.recompute()`.** The
  `.path` branch is `pose = sample.pose`, a sensor-frame camera used verbatim;
  `.freeOrbit` goes through `ViewerPoseMath.lookAt` and is already upright.
  Roll the replayed pose:
  `pose = sample.pose.rolledForDisplay(quarterTurnsClockwise: turns)`.
  Do not write the roll by hand: the helper is in Core precisely so the sign
  matches `CameraIntrinsics.rotatedForDisplay(quarterTurnsClockwise:)`, and it
  preserves the optical centre, which a hand-rolled version usually does not.
- **Where `turns` comes from.** Read
  `bundle.settings.imageQuarterTurnsClockwiseToUpright` when it is there. It is
  `nil` on every scan that exists today, and nothing populates it yet, so you
  need a fallback and the honest one is:
  `turns = recorded ?? (intrinsics.width > intrinsics.height ? 1 : 0)`.
  1 is the ordinary portrait hold, which is how the app is actually used
  (`project.yml` even says "Capture is handheld portrait"). Put the fallback
  behind one named constant with a comment, so that when the owner looks at the
  phone and finds it a quarter turn the other way, it is a one-character change
  rather than a hunt. Core derived the sign from the geometry and checked the
  arithmetic, but nobody has yet confirmed it against a real ARKit buffer, and
  Core will not claim more than that.
- **`ScanReviewModel.swift`, compare mode.** Two things have to move together
  or the A/B wipe stops lining up:
  - the photo itself needs the same quarter turn at display time (the JPEG is
    landscape and untagged on purpose, and it must stay that way on disk), and
    `compareAspectRatio` has to flip to height/width on an odd number of turns;
  - the FOV it hands the preview path is
    `bundle.intrinsics.horizontalFOVDegrees`, which is the LANDSCAPE horizontal
    field of view. Once the camera is rolled, that belongs across the other
    axis. Take it from the rotated copy instead:
    `bundle.intrinsics.rotatedForDisplay(quarterTurnsClockwise: turns).horizontalFOVDegrees`.
- **`ScanLibraryScreen.swift` thumbnails.** Same untagged landscape JPEG, drawn
  with `scaledToFill()` and no rotation, so scan thumbnails are a quarter turn
  out for the same reason. Presentation only, `.rotationEffect` is enough, and
  nothing on disk changes.
- **Do NOT** re-level the fly-through by rebuilding its rotation from a
  look-at. `PreviewCameraPathBuilder`'s whole promise is that the preview
  camera sits where the user actually stood and looks where they actually
  looked; a quarter-turn roll about the view axis keeps that promise, and a
  re-derived basis quietly breaks it.

### Requests: Capture (ios/Sources/Capture)

- **`ARCaptureService.currentSettings()`.** The field you asked for exists now,
  as `CaptureSettings.imageQuarterTurnsClockwiseToUpright: Int?`. Populate it
  from the interface orientation read at session start and everything above
  stops guessing. Mapping, with Core's confidence stated honestly:
  - `.portrait` is 1. Confident: an iPhone rear-camera buffer shot in portrait
    is the case iOS itself tags EXIF orientation 6, "rotate 90 clockwise to
    display", and the app deliberately writes no tag.
  - `.portraitUpsideDown` is 3.
  - `.landscapeLeft` and `.landscapeRight` are 0 and 2, one each. Core will NOT
    assert which way round, because `UIInterfaceOrientation`'s left and right
    are the opposite of `UIDeviceOrientation`'s and that is exactly the kind of
    thing worth being wrong about quietly. It is a two-second check on the
    phone, and getting it backwards shows up as an upside-down preview, not a
    subtle one.
  - `.unknown` should map to `.portrait`'s value rather than being passed
    through. `CaptureScreen.refreshOrientation()`'s `?? .portrait` only covers
    a missing scene, not a scene reporting `.unknown`.
  Nothing else needs to change: the pixels, intrinsics and poses stay exactly
  as they are.
- **A live-preview defect spotted while tracing this, outside Core's reach.**
  Nothing in `ios/Sources` ever calls
  `UIDevice.current.beginGeneratingDeviceOrientationNotifications()` (the only
  `UIDevice.current` in the tree is `BoosterPairingManager.swift:51`, for the
  device name), yet `CaptureScreen` subscribes to
  `UIDevice.orientationDidChangeNotification` to refresh the orientation it
  feeds `CaptureCoverageRenderer`. Apple documents that notification as
  requiring generation to have been begun, so as written the live coverage
  overlay very likely never re-orients after its first read. Driving the value
  from the window scene each frame would sidestep it entirely. Flagged, not
  claimed: Core could not run this.

### Requests: documentation

- **`docs/DATA_FORMAT.md` section 2** already has the request from Capture to
  say the camera frame is the native landscape sensor frame. Seconding it, with
  the wording Core has now committed to in `Contracts.swift`: the frame is
  landscape whichever way the phone was held; the pixels, the intrinsics and
  the poses all live in it and agree with each other, which is why a portrait
  scan still trains gravity-upright; and a display-upright replay must apply a
  roll about camera +Z, which is `Pose.rolledForDisplay(quarterTurnsClockwise:)`
  together with `CameraIntrinsics.rotatedForDisplay(quarterTurnsClockwise:)`.
  Section 5's "no EXIF orientation games" sentence is correct and should stay.
- **`docs/DATA_FORMAT.md` section 3**, the `capture_bundle.json` description:
  `settings` has a new optional key,
  `imageQuarterTurnsClockwiseToUpright`, absent on older scans and presentation
  only.
- **`CONTRACTS.md` line 413**, the blur meter row: Capture has already asked
  for the formula and the calibration to be split. Core seconds it, and
  `FrameQC.motionBlurPixels`'s own doc comment now says the same thing, so the
  table row is the only place left that reads as though amber 2 and red 4 were
  fixed by contract.

### Deliberately NOT done in Core

- **No attenuated or stabilisation-aware blur field was added to `FrameQC`**,
  although two diagnoses proposed one. `motionBlurPixels` is persisted, and the
  QC card, the pre-pass time-offset gate and anyone reading the bundle later
  all rely on it being the honest physical figure. A second, kinder number next
  to it in the same struct is exactly the shape that ends with the HUD saying
  one thing while the keyframe gate does another, which is the failure the
  owner would notice last and mind most. If Capture concludes it genuinely
  needs an attenuated value persisted, ask, and it goes in as a separate
  clearly-named optional rather than by redefining this one.
- **No orientation field on `CaptureFrame`**, only the session-level one on
  `CaptureSettings`. That is what Capture asked for and what Capture can
  populate at session start. Its one honest limit is written into the field's
  doc comment: a user who turns the phone mid-scan will have later frames a
  quarter turn out ON SCREEN. The data stays correct either way, and if it ever
  matters the per-frame field is an additive change on the same pattern.
- **Nothing in `Pose.fromARKitCameraTransform` was touched.** It is correct,
  and it is still the only place in the app allowed to negate an axis. The new
  display helpers are rotations, not sign flips, and say so.
---

## From Viewer (ios/Sources/Viewer), after the first run on real hardware

The 90 degree roll is fixed, in the Viewer only, exactly as Trainer, Capture and
Core all asked. Nothing persisted moved, no exporter changed, and the roll is
applied in ONE place per path so it cannot be applied twice.

### What the Viewer now does

- `PreviewCameraPathBuilder.build` rolls each keyframe pose with Core's
  `Pose.rolledForDisplay(quarterTurnsClockwise:)`, from a new
  `Options.uprightQuarterTurns`. Applying it where the path is MADE rather than
  in `ViewerCameraController.recompute()` (which is where Capture and Core
  suggested it) means the sampler, `PreviewPathSampler.worstDeviation`, the
  A/B compare and the switch back to free look all see one consistent set of
  poses, and there is no per-frame work. `adopt(path:)` now carries a doc
  comment saying the poses arrive already rolled and must not be rolled again.
- `ScanReviewModel` does the same for the single-frame compare path, turns the
  compare photo by the same quarter turn (`ViewerPhoto.upright`, which sets the
  UIImage orientation flag and copies no pixels), flips `compareAspectRatio`
  with it, and now takes the compare field of view from
  `intrinsics.rotatedForDisplay(quarterTurnsClockwise:).horizontalFOVDegrees`
  as Core asked. It also hands the renderer the rotated intrinsics, so the
  pixel aspect the renderer takes from them turns with everything else.
- `ScanLibraryScreen` thumbnails get the same quarter turn, from a new
  `ScanSummary.photoQuarterTurns` filled in by `ScanLibraryReader.readSummary`.
- Free look is untouched: it builds its own upright basis and is already level.
  The training preview in `Sources/Pipeline` only ever uses free orbit
  (`TrainingPreview.swift:231`), so it needs nothing and got nothing.
- `MetalSplatRenderer` and `SplatRenderShaders.metal` were NOT changed. The
  five things the ruled-out diagnoses suspected there (swapped viewport,
  transposed view matrix, flipped texture coordinates, hardcoded aspect,
  uniform layout) were checked and all five are correct.

### One deliberate deviation from Core's request, and why

Core asked for the fallback
`turns = recorded ?? (intrinsics.width > intrinsics.height ? 1 : 0)` behind a
named constant, because nobody could confirm the sign from source. The Viewer
does something else: `ViewerPoseMath.uprightQuarterTurns(of:)` MEASURES the
turn from the scan's own poses. The world frame is gravity aligned, so where
world up lands in the image says how the phone was held; every pose votes,
weighted by how much of world up was in the image plane at all, so frames
pointed at a ceiling or a floor fall out of the average on their own, and the
result is quantised to a quarter turn so a tilted wrist is still replayed as it
happened.

Why that rather than the heuristic, in order of importance:

1. The heuristic assumes portrait. `intrinsics.width > intrinsics.height` is
   true for every scan ever recorded on this app, portrait or landscape, since
   it describes the sensor and not the hold, so a scan shot in landscape would
   have been rolled a quarter turn INTO being sideways. The measurement gets
   all four holds right.
2. There is no unknown sign left to confirm on the phone. The angle is derived
   from the data, not assumed, so there is no constant to flip.
3. It degrades to today's behaviour, never to something worse: a scan with
   nothing but ceiling or floor in it has nothing to measure and returns 0,
   which replays exactly what was recorded.

It was checked numerically, not just reasoned about: for the four holds the
measurement returns portrait 1, sensor-native landscape 0, the other landscape
2, upside down 3, and after Core's `rolledForDisplay` world up lands exactly up
the screen in all four, with the camera centre and the view direction moved by
0 (so the fly-through's ~0.20 m deviation promise is untouched). A simulated
220 frame portrait walk with an 8 degree wrist tremor and 20 ceiling frames
still returns 1. On the documented iPhone camera (1920x1440, fx = fy = 1454.203)
the turned intrinsics are 1440x1920 with the principal point still centred, and
the compare panel's render comes back to a 66.86 degree vertical field of view,
which is the capture's horizontal one, so the two halves of the A/B wipe line
up to within rounding.

The recorded value still WINS wherever a scan has one. The measurement is only
the fallback for scans written before the field existed, which is every scan on
the phone today.

### Requests

- **`ios/Sources/Capture`, `ARCaptureService.currentSettings()`:** please do
  populate `CaptureSettings.imageQuarterTurnsClockwiseToUpright`. The Viewer
  reads it first and only measures when it is nil. The one thing that has to be
  right is the convention, which is Core's: 1 is the ordinary portrait hold, 0
  is the sensor's own landscape, 2 is upside down, 3 is the other landscape.
  There is a free check on it: `ScanReviewModel` works out the measured value
  as well, and logs `viewer.review` "capture recorded N quarter turns to
  upright, the poses measure M" whenever the two disagree. On a scan with a
  horizon in it they should agree, so one line in Console after the next scan
  settles whether the mapping is right, without anyone having to judge it by
  eye.
- **Whoever changes the QC weight curve (`CaptureQCEvaluator.weight`):** two
  gates in the Viewer read `qc.weight` and will move with it, both in the
  direction of keeping more data, which is the right direction:
  `PreviewCameraPathBuilder.Options.minFrameQCWeight` (0.20, which frames may
  hold a review camera) and `ObservedDirectionBuilder.Options.minFrameQCWeight`
  (0.15, which frames count as having really observed a surface). Neither needs
  changing, and neither was changed. Flagging them so the change is not a
  surprise.

### Deliberately NOT done in the Viewer

- **No second roll in `ViewerCameraController.recompute()`.** Every path that
  reaches `.path` mode comes through `PreviewCameraPathBuilder.build` or
  `ScanReviewModel.showCompareFrame`, and both roll already. Adding it in
  `recompute()` as well would turn the picture half a turn.
- **No re-levelling of the fly-through against gravity.** A look-at rebuilt
  from world up would straighten out the user's own wrist and point the preview
  camera somewhere they never looked, which is exactly what
  `PreviewCameraPathBuilder`'s header exists to prevent. A quarter-turn roll
  about the view axis keeps the promise; a re-derived basis quietly breaks it.
- **No hand-written roll.** Core's `Pose.rolledForDisplay` and
  `CameraIntrinsics.rotatedForDisplay` are the only implementations used, so
  the pose and the picture cannot end up turning different ways.
- **No change to the fly-through's 105 degree field of view.** It is applied
  across the drawable's width, and the walk-through surface is the full screen
  width by 360 points, so it is close to square and the vertical field comes
  out near 100 degrees, which is wide but not distorted. It would be worth
  revisiting only if that surface ever becomes tall and narrow.
- **Nothing in `Sources/Capture`, `Sources/Trainer` or `Sources/Export` was
  touched**, and no persisted file changed shape. The blur meter is not the
  Viewer's and nothing in the Viewer reads `motionBlurPixels`.
