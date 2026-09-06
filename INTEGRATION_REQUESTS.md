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

---

## From Viewer (the splat census), 2026-09-05

The Viewer now shows a **splat census**: one plain sentence at the top of the
review screen naming the step that lost the most geometry, with the full ladder
of counts behind a tap. It exists because the first real scan took a day of
code reading to explain, and all four faults that caused it were "a number went
down at a named step" - invisible only because nobody was counting.

**Everything the Viewer can measure on its own, it already measures.** Nothing
below is needed for the census to work today. What is below is the part of the
ladder only PrePass and the Trainer can see, and each missing rung currently
reads "not recorded" on screen (never zero, and the headline says the breakdown
is incomplete). Filling these in is what turns the next bug of this shape from a
day of archaeology into a five second read.

### The ask, in one line

Write one small JSON file at the end of your stage. PrePass writes
`prepass/census.json`; the Trainer writes `model/census.json`. Two files, each
in the folder its writer already owns, so nothing races.

### The schema

Declared as `ScanCensus.Record` in `ios/Sources/Viewer/ScanCensus.swift`.
**Do not re-declare that type in your module** - this is one Xcode target and a
duplicate top-level type name is a hard link error. Write the JSON with a local
private struct, or a dictionary, or whatever suits you. The reader does not care
what wrote it.

Rules the reader already enforces, so you do not have to be careful:

- **Every key is optional.** Write only what you genuinely counted. A missing
  key renders as "not recorded" with your stage named as the reason. A key you
  write as `0` renders as a measured zero, which is a much stronger claim -
  please do not write 0 for "I did not count this".
- Unknown extra keys are ignored, so you may add fields ahead of this reader.
- Keys are camelCase, encoded with `ContractsJSON.encoder()`.
- `formatVersion` should be `1`. A file with a higher version is refused and
  the refusal is shown on screen rather than guessed at.

```json
{
  "formatVersion": 1,
  "writtenBy": "trainer",

  "depthSamplesOffered": 812004,
  "depthSamplesAccepted": 796110,
  "seedsWritten": 43112,
  "seedsShapedAsConfidentDiscs": 41980,
  "trustGateSigmaMeters": 0.02,
  "measuredMedianSigmaMeters": 0.031,

  "splatsAtStart": 43112,
  "splatsCreatedByDensification": 268430,
  "densificationPassCount": 27,
  "densificationCandidateCount": 291004,
  "splatsDeletedByPruning": 31004,
  "pruningPassCount": 10,
  "splatsDeletedByHeatCut": 0,
  "heatCutCount": 0,
  "splatsAtEnd": 280538,

  "plannedSplatCap": 300000,
  "finalSplatCap": 300000
}
```

### For `ios/Sources/PrePass` (the first six keys)

- **`depthSamplesOffered` / `depthSamplesAccepted`.** Counted either side of the
  trust gate in `PrePassInitialSplats`. This is the pair that makes a misplaced
  gate obvious: the census prints "94% of the depth readings were rejected
  before the build even began" instead of leaving someone to work out from a
  photograph of a smear that `trustedWeight` was set at a sigma no handheld
  phone reaches.
- **`trustGateSigmaMeters` / `measuredMedianSigmaMeters`.** The sigma the gate
  demanded, and the sigma this scan actually measured. When both are present the
  census says, in the user's language, that the test asked for 2 cm and the scan
  delivers 3 cm, so the test is in the wrong place. Neither number can be
  recovered from anything on disk today.
- **`seedsWritten`** is a cross-check against
  `InitialSplatSetRef.splatCount`, which the Viewer already reads. If the two
  ever disagree, one of them is wrong and it is worth knowing which.
- **`seedsShapedAsConfidentDiscs`.** How many seeds came out as pinned discs
  rather than ray-elongated blobs (`InitialSplatSetRef.flagsPath` bit 0 / bit 1).
  A field of blobs with almost no discs is the exact fingerprint of the trust
  gate rejecting everything, and the Viewer can only see the CONSEQUENCE of it
  in the finished model, hours later.

### For `ios/Sources/Trainer` (the rest)

- **`splatsCreatedByDensification` and `densificationPassCount` are the most
  valuable two numbers in this whole file.** The densification threshold in the
  wrong units made that step a no-op on every run that has ever happened, and
  nothing anywhere said so. With these two, the census leads with "The step that
  adds detail ran 27 times and created nothing at all", which is the entire
  investigation.
- **`densificationCandidateCount`** separates two different faults that look
  identical from outside: zero candidates means the test that chooses points is
  wrong; many candidates and zero created means the creation path is wrong.
- **`splatsDeletedByPruning` / `pruningPassCount`.** The pruning-window settings
  that were declared and read by nothing are fixed, but a pass count on screen
  is what makes a regression obvious rather than inferred.
- **`splatsDeletedByHeatCut` / `heatCutCount`.** `TrainerBudgetGovernor` already
  computes exactly this when it lowers the cap; it just does not write it down.
  The Viewer can already see that the final `splatCap` is below the pre-pass's
  suggestion, but it CANNOT tell heat from an ordinary lower starting budget, so
  without `heatCutCount` the census deliberately refuses to say heat was the
  cause. It says "a build lowers its own limit when the phone gets warm or when
  memory runs short; this one did not record which".
- **`splatsAtStart` / `splatsAtEnd` / `plannedSplatCap` / `finalSplatCap`.**
  The end-to-end arithmetic. `finalSplatCap` below `splatsAtEnd` is the
  signature of the thermal ratchet that destroyed the first scan, and the census
  calls that out by name as an impossible state worth reporting.
- **Iterations are deliberately NOT asked for.** `model.json` already records
  `iterationsCompleted` and `budgetUsed.iterations`, and the census reads both
  from there. Asking a second writer for the same number would only create a
  disagreement nobody would know how to resolve.

### Names this module has taken

New top-level type names now declared in `ios/Sources/Viewer`, listed so nobody
picks the same one: `ScanCensus`, `ScanCensusSummarySection`, `ScanCensusCard`,
`ScanCensusDetailSheet`, `ScanCensusStageRow`, `ScanCensusLibraryLine`.
Everything else the census uses is nested inside `ScanCensus`.

### Nothing else is asked for

No Core change is needed and none was made. `ScanSummary` gained
`censusInputs` with a default value, so no existing call site changed, and
`ScanLibraryReader.readDetail` carries it through unaltered for
`ScanProcessingCoordinator` and `ScanProcessingScreen`.

---

## Resolved by the integrator (the census sweep), 2026-09-06

All four census requests above are now closed. Two were actioned, one was
rejected with its reasoning, and one was already done. Nothing is left open.

### ACTIONED: the trainer was writing the wrong filename, in the wrong shape

`TrainerCensusWriter.relativePath` was `model/train_census.json`, and
`ViewerScanPaths.modelCensusJSON` reads `model/census.json`. Two different
files. The Viewer would have found nothing on every scan ever built, and every
trainer rung of the ladder would have read "this build did not write
model/census.json" for good.

The shapes did not match either. `TrainerCensus` is a deep diagnostic:
`finalSplatCount`, `slices[]`, `densifyPasses[]`. `ScanCensus.Record` is flat:
`splatsAtEnd`, `splatsCreatedByDensification`. Because every `Record` field is
optional, decoding `train_census.json` into it would have SUCCEEDED and
produced a record with nothing in it, which is worse than failing.

Fixed by keeping both files. `model/train_census.json` stays exactly as it was
and is the Trainer's own. `model/census.json` is new, flat, and is derived
from the same sealed `TrainerCensus` by `TrainerCensus.sharedRecord`, so the
two physically cannot disagree. See `TrainerCensusSharedRecord` in
`ios/Sources/Trainer/TrainerCensus.swift`.

Honesty rules applied to that mapping:

- Every loop-derived key is ABSENT until `trainingBegan` is true, which needs
  at least one slice to have actually put its seeds on the GPU. A run that
  died during buffer allocation has an empty `densifyPasses` array for reasons
  that have nothing to do with densification, and "0 points created" would be
  a number nobody took. `TrainerCensusSlice.seedsUploadedCounted` is the flag,
  set on the same line as the count it vouches for.
- `heatCutCount` and `splatsDeletedByHeatCut` are written ALWAYS, including as
  zero, because the governor records every reduction from the moment the run
  starts, so an empty list really is "no thermal cut happened". That measured
  zero is what lets `capAlert` rule heat out instead of saying it cannot tell.
- `splatsAtEnd` is absent unless the run reached the merge.
- `pruningPassCount` now counts passes where pruning of ANY kind removed
  something, matching `splatsDeletedByPruning`, which is pruning of any kind.
  Counting the numerator one way and the denominator another produced two
  numbers that looked like a pair and were not.
- The thermal reason string is now `TrainerCensusBudgetReduction.heatReason`,
  referenced by `TrainerBudgetChange.Reason.censusReason` rather than repeated.
  `model/census.json` decides whether a cap cut was thermal by matching that
  exact string, and two copies of a phrase is how a heat report silently
  becomes a zero when somebody rewords one of them.

### REJECTED: `depthSamplesOffered` and `depthSamplesAccepted`

PrePass declined these and PrePass is right. Confirmed against the source
rather than the note: in `PrePassInitialSplatBuilder.build`, the trust weight
is used at two places and neither of them drops a sample. It picks the better
of two samples competing for one voxel (`if !slot.inserted, sampleWeight <=
weight[slot.index] { continue }`), and it decides a seed's SHAPE
(`let trusted = weight[i] >= trustCut && hasNormal[i]`). There is no gate
either side of which a sample count could be taken. Those two keys describe a
step this pipeline does not have, and they will stay absent.

Two consequences, both fixed in `ios/Sources/Viewer/ScanCensus.swift`:

1. **The sigma comparison was dead code.** `trustGateAlert` guarded on the two
   sample counts before it would print the sigma pair, so
   `trustGateSigmaMeters` and `measuredMedianSigmaMeters` - which the pre-pass
   DOES write, and which are the two numbers that name the fourth fault from
   the first real scan - could never reach the screen. They are now their own
   alert, `trustGateSigmaAlert`, which fires whenever the gate is tighter than
   the measurement, and it is attached to the "Starting points pinned to a
   surface" rung next to `discAlert`. It can also lead the headline.

2. **Two permanent "not recorded" rows.** The two depth-reading rungs now
   appear only when a writer actually supplied one of the keys. This is a
   deliberate exception to the "a missing key reads as not recorded" rule, and
   the only one: printing two blanks on every scan forever, for a step that
   does not exist, teaches the eye to skip a screen whose whole job is to be
   read.

### ALREADY DONE: the other four pre-pass keys, and the reader

`seedsWritten`, `seedsShapedAsConfidentDiscs`, `trustGateSigmaMeters` and
`measuredMedianSigmaMeters` are written at the top level of
`prepass/census.json` by `PrePassCensus.fillSharedKeys()`, from the same
source as the detailed `seeding` section, so they cannot drift. The sigma is
written only when it was genuinely measured (`seeding.sigmaIsMeasured`); the
physics prior is a prediction and is never reported under a name that says
"measured". `ScanLibraryReader.readDetail` reads both sidecars into
`ScanSummary.censusInputs`. Verified end to end, by name, in the source.

### Documentation

`docs/DATA_FORMAT.md` section 8 now documents `prepass/census.json` and
`model/census.json` as one shared format with one key table, and states in
writing that an absent key means "not counted" while a written `0` is a
measured zero. Both files are in the folder tree at the top of that document.

---

## From Trainer / TrainerShaders.metal (loss normalisation, 2026-09-06)

Context, so the three bullets below read as one thing rather than three.
`trainer_loss_photometric` and `trainer_ssim_stats` are per-pixel MEANS: both
divide by `pixelCount`. Every geometry term in `trainer_loss_depth` (depth
Huber, F4 bimodal, F4 transition width, F6 alpha supervision, F2 free-space
hinge) was an unnormalised per-sample SUM over the whole native depth grid,
256 x 192 = 49,152 samples. The geometry side was therefore larger than the
photometric side by roughly the number of contributing samples, order 10^4.
Two consequences: the photographs were effectively inert for every geometry
parameter, and the AbsGS densification statistic (which is
`length(dLdMean2D)`, and `dLdMean2D` is fed by the depth channel as well as
the colour channel) was ranking Gaussians by where the LASER disagreed rather
than by where the PICTURE was wrong.

`trainer_loss_depth` now divides every one of those five terms, loss value and
gradient together, by `u.depthSampleCount`. That is the right SHAPE but the
wrong COUNT: `depthSampleCount` is the number of samples DISPATCHED, which is
the full native grid including every sample that carries no weight. The
divisor should be the number that actually contributed.

- **`ios/Sources/Trainer/TrainerGPULayouts.swift`, `TrainerLossUniforms`:**
  append one field after `ssimC2`, at offset 64:
  `var depthSupervisedCount: UInt32 = 0  // offset 64`.
  The struct goes from 64 bytes to 68, align 4. It is uploaded with
  `setBytes`, not from an `MTLBuffer`, so no allocation changes and no other
  binding moves. The `check("TrainerLossUniforms", ..., 64)` line in the same
  file's layout verification needs its expected stride changed to 68, and the
  header comment at the top of the file (`TrainerLossUniforms 64 bytes,
  align 4`) needs the same edit. Nothing else in the file is affected: the
  field is appended, so every existing offset is unchanged.

- **`ios/Sources/Trainer/MetalSplatTrainer.swift`, `runIteration`, next to
  `loss.depthSampleCount = UInt32(sampleCount)` (about line 1261):** add
  `loss.depthSupervisedCount = UInt32(supervision.supervisedSampleCount)`.
  NO NEW WORK AND NO NEW SYNC: `TrainerFrameSupervision.supervisedSampleCount`
  is already computed, once per frame, on the CPU, at
  `TrainerSupervision.swift` line 281
  (`samples.reduce(into: 0) { $0 += ($1.weight > 0 ? 1 : 0) }`), and today it
  is assigned and read by NOTHING. It is exactly the count the kernel needs:
  the kernel's gate is `w > 0` where `w = depthScale * s.weight`, and
  `depthScale` has a floor of 0.05, so `w > 0` is `s.weight > 0`, which is the
  predicate that reduce already counts. It is also already clamped correctly
  by construction: `TrainerSupervision` sets `weight = 0` for every BAND
  sample, every UNKNOWN sample, every no-return, and everything below
  `minimumAuthorityForDepth`.
  One caution: `sampleCount` in `runIteration` is
  `min(supervision.depthSamples.count, resources.depthSampleCapacity)`, so if
  the capacity ever truncates the array, the supervised count must be
  recounted over the prefix that was actually uploaded rather than taken from
  the full array.

- **`ios/Sources/Trainer/TrainerShaders.metal` (Trainer's own file, listed here
  only so the three land together):** once the field exists, change the one
  line
  `const float invSamples = 1.0f / float(max(u.depthSampleCount, 1u));`
  in `trainer_loss_depth` to use `u.depthSupervisedCount`, and delete the
  paragraph of the block comment above that kernel headed "THE DENOMINATOR IS
  NOT YET THE ONE IT SHOULD BE". The comment states the interim in full so it
  cannot become a silent approximation.

- **`ios/Sources/Trainer/TrainerCensus.swift` (wanted, not blocking):** the
  supervised sample count is the number that says how much of each frame the
  laser actually got a vote on, and it is the denominator the loss balance now
  rests on. A per-slice mean of `supervision.supervisedSampleCount` alongside
  the 49,152 that were dispatched would make "the LiDAR supervised 8 per cent
  of this frame" visible on the census page instead of inferable. Trainer's
  shader measures nothing new here, so this is a request rather than a
  parallel mechanism: the number exists on the CPU already, and per rule 5 it
  should reach the eye through `TrainerCensus` or not at all.

### Two things found next door while doing this, NOT changed, evidence attached

- **`ios/Sources/Trainer/TrainerSupervision.swift`, `depthSamples(...)`: the
  borrowed free-space bound is computed and then thrown away.** For a sample
  with no return the function calls `borrowedFreeSpaceBound(...)` and stores it
  in `sample.freeSpaceBound`, with a comment saying that is RULE 5. For a BAND
  sample it stores `max(z - freeSpaceMargin, 0)`. Both of those samples are
  then classified UNKNOWN or BAND, and `trainer_loss_depth` returns on exactly
  those two classes BEFORE it reaches the F2 free-space hinge. So neither bound
  has ever been used, in the whole history of this app: the free-space term
  only ever fires on samples that already have a valid depth and a Huber term
  pulling them to the same place. `borrowedFreeSpaceBound` is the
  "documented with worked examples and called from nowhere" pattern, one step
  removed.
  NOT fixed on purpose, and this is the reason: an UNKNOWN sample is typically
  a pixel with nothing rendered in it, so `alpha` is near zero, `expected =
  accumulated / max(alpha, 1e-4)` is near zero, the hinge sees a violation
  equal to the entire borrowed bound, and the chain rule then multiplies the
  gradient by `1 / safeAlpha` and `accumulated / safeAlpha^2`, that is by up to
  1e4 and 1e8. Enabling the branch as it stands would put an amplifier of that
  size on the emptiest pixels in the frame. It needs an alpha floor or an
  alpha-weighted hinge designed with it, not an unblocked early return, and
  that is a change to what the loss MEANS rather than to its scale, which is
  not what this pass was for.

- **`ios/Sources/Trainer/TrainerShaders.metal`, `trainer_loss_depth`, the same
  `1 / safeAlpha` and `accumulated / safeAlpha^2` factors, on the samples that
  DO fire today.** Early in a run, or anywhere the splats have not yet covered
  a surface the laser can see, `alpha` is small and legitimately so, and the
  depth gradient is multiplied by up to 1e4. That is real Jacobian, not a bug,
  but it is an unbounded one, and it is now the largest remaining term-scale
  hazard in the depth path. Worth measuring (the distribution of `alpha` at
  supervised pixels over the first few hundred iterations) before anyone
  chooses a floor by feel.

### Addendum: the reported loss number is not mostly the photograph either

`lossEMA` is display and census only (nothing gates on it, checked:
`MetalSplatTrainer.swift` lines 1580 to 1582 write it, and every other
reference passes it to a progress callback), so this is a reporting fault
rather than a training fault, and it is left alone for now. But it is worth
knowing while reading a run: `lossAccum[0]` is one scalar carrying four
different scales at once. The photometric and SSIM terms are per-pixel means
of order 0.01 to 0.1. The five depth terms are now per-sample means of the
same order. The three regulariser terms in `trainer_regularizer` are still
unnormalised sums over the whole population, up to 300,000 Gaussians: the disc
prior alone at `discWeight = 0.01` contributes of order 100, roughly three
orders of magnitude above the photometric term. So the number on the screen is
essentially the regulariser's population sum, and photometric progress is
invisible in it.

The regulariser sums are NOT wrong and should not be divided by the splat
count. Each thread there writes the prior's gradient into that Gaussian's OWN
slot, so the prior's strength per Gaussian is correctly independent of how many
Gaussians exist; dividing by the population would make every prior weaken as
the model grows, which is worse than the reporting problem. Only the reported
total mixes scales.

The cheap fix, if anyone wants it, is four scalars instead of one.
`resources.lossAccum` is already 16 bytes and `TrainerResources` already
allocates all four; `TrainerPipelines.clearPerIteration` zeroes only the first
(`fillFloat(encoder, buffer: resources.lossAccum, count: 1, value: 0)`) and
`MetalSplatTrainer` reads only element 0. Change the fill to `count: 4` and
read all four, and `TrainerShaders.metal` will write photometric into 1,
geometry into 2 and regulariser into 3 alongside the existing total in 0. Not
done yet on purpose: writing slots that nothing zeroes and nothing reads would
be a measurement with no reader, which is the fault this whole pass exists to
remove.

---

## From `ios/Sources/Trainer/MetalSplatTrainer.swift`: the loop can no longer claim work it did not do

Three things were fixed inside `MetalSplatTrainer.swift` itself and need
nothing from anyone. They are listed first only so the requests below have
their context.

1. `runIteration` was still being handed `totalIterations` (the PLANNED slice
   budget) while the loop exits at `effectiveTotal`. That one argument drives
   the SH degree ramp, the frequency-blur decay, the depth-loss decay, the late
   opacity binarization, the position learning-rate decay and the warm-up end.
   The `effectiveTotal` fix had been applied to `progressFraction`, to
   `supervision.build` and to the loop exit, and to nothing else, so on any
   shortened run the entire back half of every one of those schedules still
   never executed. Now `effectiveTotal`.
2. The stage label and the stage sentence in the progress tick were still keyed
   to `totalIterations`, so the screen showed a stage the loop was no longer in.
   Now `effectiveTotal`.
3. `iteration` (times round the loop) and gradient steps actually taken are now
   two separate numbers, and the run's outcome is judged on the second.

### Requests

- **`ios/Sources/Core/Contracts.swift`, `TrainerProgress`: two optional fields,
  both with defaults so no existing call site breaks.**
  ```swift
  /// Iterations so far that ran a real forward, backward and Adam step, as
  /// opposed to times round the loop. nil when nobody counted.
  public var gradientStepsCompleted: Int? = nil
  /// Densification passes in a row that were allowed to add geometry and
  /// added none. nil when nobody counted.
  public var consecutiveZeroGrowthPasses: Int? = nil
  ```
  and the same two as defaulted parameters on the public `init`, appended
  LAST so the existing positional and labelled calls keep compiling.
  Why: today the trainer can only reach the screen through
  `TrainerProgress.message`, so it says "Still working, but no new detail has
  been added for 12 passes in a row" as prose. That is better than silence, but
  it is a sentence the review screen cannot style, count or act on. Two numbers
  would let the UI show it as a state rather than as a sentence.

- **`ios/Sources/Trainer/TrainerCensus.swift`, `TrainerCensusSlice`: one field.**
  ```swift
  /// Iterations that ran a real gradient step, as opposed to times round the
  /// loop. `iterationsCompleted` minus the three skip counters.
  var iterationsWithGradientStep: Int = 0
  ```
  The number is already derivable (`iterationsCompleted` minus
  `iterationsSkippedNoSupervision + iterationsSkippedGrowingTileBuffer +
  iterationsSkippedNothingToRender`) and the trainer now measures it directly,
  so this is naming a number that exists, not inventing one. Storing it also
  makes the two sides checkable against each other: if the stored count and the
  subtraction ever disagree, a skip path stopped being counted.

- **`ios/Sources/Trainer/TrainerCensus.swift`, alert 13
  (`many_iterations_did_no_work`) should escalate.** It currently fires at
  `severity: "check"` for anything over 5 per cent. A run where MOST iterations
  took no optimisation step is not a thing to check, it is a failed run wearing
  a finished run's clothes, and it is the exact case where the app reported
  success and produced almost nothing. Suggested: keep "check" from 5 per cent,
  and add a `"loud"` alert, code `most_iterations_did_no_work`, when
  `skipped * 2 > iterationsCompleted`.

- **`ios/Sources/Trainer/TrainerCensus.swift`: a longest-zero-growth-streak
  total and a loud alert for it.** Everything needed is already in
  `densifyPasses`: each row carries `growthWindowOpen`, `headroom` and the added
  counts. The aggregate wanted is the longest run of CONSECUTIVE passes for
  which `growthWindowOpen && headroom > 0 && addedBySplit + addedByClone == 0`,
  and a `"loud"` alert (suggested code `densification_stalled`) when that run
  reaches 10. Alert 1 (`densification_created_nothing`) only fires when growth
  was allowed and NOTHING was ever created anywhere in the run, so a run that
  densified normally for a while and then stalled for two thousand iterations
  passes it cleanly today. The trainer already tracks and logs this streak
  live; this is the same measurement surfaced through the census rather than a
  second mechanism.

- **`ios/Sources/Core/Contracts.swift`, `SplatModel.iterationsCompleted`:
  needs either a doc comment or a companion.** `MetalSplatTrainer` fills it
  from `totalIterationsRun`, which is times round the loop, and it is written
  into the finished model as a claim about how much training the model had. It
  has no doc comment, so a reader has no way to know it is not the count of
  optimisation steps. Either document it as "loop iterations, not necessarily
  optimisation steps" or add `gradientStepsCompleted: Int? = nil` next to it.
  NOT changed unilaterally: it is a public field in another module's contract
  and the review screen may already be showing it.

### One thing found next door, NOT changed, evidence attached

- **`ios/Sources/Trainer/MetalSplatTrainer.swift` passes `census: inout
  TrainerCensus` into `trainSlice`, which is `async throws`.** Every census
  write a slice makes goes through that `inout`. Swift's in-out parameters are
  specified as copy-in copy-out, with pass-by-address only as an
  OPTIMISATION when the argument is a value at a physical address. On the
  copy-in copy-out path, a callee that throws does not write back. The whole
  reason the census write sits in a `defer` in `train` is so that a run which
  THREW still leaves a census behind, and a run that threw inside `trainSlice`
  is precisely the run whose slice row would be empty if the writeback did not
  happen.
  In practice the argument is a local `var` in `train`, so the address
  optimisation applies and the writes survive today. It is not guaranteed by
  the language, and it is exactly the class of fault this project has been
  digging out: correct by accident, invisible when it stops being correct.
  NOT changed here because the robust fix is to hold the census in a small
  reference box and pass that, which is about thirty mechanical edits across a
  file no one can compile locally, and CI is the only compiler. Worth doing
  deliberately, by whoever owns `TrainerCensus.swift`, as a
  `final class TrainerCensusRecorder` wrapping the struct.

---

## From the owner of `PrePassCarver.swift` + `PrePassSensorIO.swift` (free-space carving and depth sidecar reads)

Two "total failure looks exactly like a normal empty result" faults were fixed
inside those two files. Three things they touch live in files owned by other
agents, so they are asked for here instead.

### 1. `PrePassSurvey.swift`: the `unreadable_depth` finding has never been able to fire

`PrePassSurvey.swift` line 232 does `result.unreadableDepthFrames += 1`, and it
sits in the `catch` around `PrePassDepthFrame.load`. Line 782 turns any
non-zero value into the `unreadable_depth` QC finding.

That counter could not reach 1 for its own named cause. `load` returned NIL,
not a throw, when a frame HAS a `depthPath` and the file will not open, so the
`catch` never ran for it and `guard let depthFrame else { continue }` swallowed
it. The only things that ever reached that `catch` were a wrong-LENGTH file and
an impossible recorded map size. A scan whose depth sidecars had all been lost
produced `unreadableDepthFrames == 0` and no finding.

`PrePassDepthFrame.load` keeps its exact signature and its exact nil/throw
behaviour, so nothing breaks. Alongside it there is now:

```swift
enum PrePassDepthLoad {
    case loaded(PrePassDepthFrame)
    case noDepthRecorded          // frame.depthPath == nil. Normal.
    case unreadable(path: String) // a path WAS recorded, the file will not open.
}

static func loadOutcome(frame:settings:at:) throws -> PrePassDepthLoad
```

`load` is now a four-line wrapper over `loadOutcome`, so the two cannot drift.

ASK: in `PrePassSurvey.swift`, replace the `load` call with `loadOutcome` and
increment `unreadableDepthFrames` on the `.unreadable` case. That is a five-line
change and it is the only thing standing between that finding and the screen.

The same swap is worth making, but is less urgent, in:

* `PrePassInitialSplats.swift` line 293. It already counts BOTH nil branches
  into `keyframesDepthMissing`, so its total is right today; the swap would only
  let it separate "no laser on this frame" from "this frame's laser data was
  lost", which are different problems with different fixes.
* `PrePassBundleAdjuster.swift` line 247 and `PrePassGlassDetector.swift` line
  169. Both `continue` on either branch and count nothing at all.
* `PrePassPoseRefiner.swift` lines 439 and 779. Both use a bare `try ... else
  { continue }` / `else { return nil }`, so an unreadable sidecar is skipped
  and never counted.

DELIBERATELY NOT DONE: making `load` itself throw on an unreadable file. It
would have fixed the Survey counter with no edit to any other file, but the two
`PrePassPoseRefiner` sites propagate rather than catch, so ONE bad sidecar would
have gone from "skip this pair" to "abandon the whole pose-refinement stage".
Turning a partial degradation into a total one is the wrong trade, and it is not
a change to make in a file I do not own and cannot fix if it lands badly.

### 2. `PrePassCensus.swift`: `carving.keyframesDepthMissing` is mislabelled, and the split now exists

Line 859 renders that count as `"... had no depth to read"`. In the carve loop
that label is wrong. `VoxelFreeSpaceCarver.keyframes(from:)` selects with
`for frame in frames where frame.depthPath != nil`, so EVERY keyframe the carve
tries to open has a depth path recorded, and every one of those misses is an
unreadable file, never a frame that had no depth. The label reads as "this phone
had no laser here" and the fact is "this scan's laser data was lost".

The carver now counts the two apart (`keyframesDepthUnreadable`,
`keyframesNoDepthRecorded`, plus the path of the first casualty) and logs the
split at `.error` on every carve where either is non-zero. It still reports the
SUM into `keyframesDepthMissing`, because that is the only field there is.

ASK, in whichever order suits: (a) change that detail string to "could not be
read", which is true either way; and/or (b) add
`public var keyframesDepthUnreadable = 0` to `PrePassCensus.Carving` (plus its
CodingKey, decode and encode lines) and I will fill it on the next pass. Until
(b) exists the number is measured and logged but has no census slot, and per the
rule that a census must never report a number the code did not measure, nothing
is being written under a borrowed name.

### 3. `Contracts.swift` / `MetalSplatTrainer.swift`: the occupancy grid now refuses to load empty

`OccupancyGridRef.cellCount`, `.emptyCellCount` and `.surfaceCellCount` were
written at exactly one place (`PrePassCarver.carve`) and read at ZERO places in
all of `ios/Sources`. Grepped. They are now read, in
`VoxelFreeSpaceCarver.load`, which throws `NimbusError.malformedData` when the
file's record count disagrees with `cellCount`, and when `emptyCellCount` is 0.

Consequence for `MetalSplatTrainer.loadSmartLayer` (line 2241): a scan whose
occupancy grid proves no air empty now leaves `layer.carver` nil and logs the
sentence that is already there, "The free-space map could not be read (...);
nothing is deleted on free-space grounds this run". That sentence is now true
instead of unreachable. Behaviour is otherwise identical: an all-unknown grid
and a nil carver both delete exactly nothing, the difference is that one of them
says so.

NOTE for whoever owns the trainer census: `TrainerCensus` cannot currently tell
"F2 had no grid" from "F2 had a grid and it licensed no deletes" - both are
`carvedFromEmptySpace == 0`. A `freeSpaceGridLoaded` flag would close that.

---

## From Viewer + Export (ios/Sources/Viewer, ios/Sources/Export)

### The trainer's 3D low-pass filter has to be fused into the cloud before it leaves the trainer

**What is wrong today.** `trainer_preprocess` renders every Gaussian with the
Mip-Splatting 3D filter applied: it widens the covariance to
`Sigma + filter3D^2 * I` (TrainerShaders.metal lines 620-625) and multiplies the
alpha by `comp3D = sqrt(det(Sigma) / det(Sigma + filter3D^2 * I))` (line 632,
used at line 713 as `comp = comp2D * comp3D`). Every opacity the optimiser ever
fitted was fitted against that dimming, and every scale against that widening.

`filter3D` exists in exactly one place: `TrainerSplatStats.filter3D`
(TrainerGPULayouts.swift:220), written by `trainer_filter3d_finalize`.
`MetalSplatTrainer.readCloud` (MetalSplatTrainer.swift, around line 1945) does
not read the stats buffer at all. So the value is dropped there, and the
`SplatCloud` that goes to the viewer, to `.ply`, to `.spz` and to `.glb` is a
DIFFERENT MODEL from the one that was trained: narrower, and too opaque, in the
direction that makes a good run look like noise.

**The decision, and why it is a bake and not a new field.** Carrying `filter3D`
through would need a non-standard `.ply` property, a break in the `.spz` binary
layout and a vendor glTF extension, and even then only this app's own viewer
would benefit: the Blender add-on in this repo, SuperSplat and every other
reader would still draw the wrong model. Baking is also what upstream
Mip-Splatting itself does for released models. The one thing baking costs is the
raw optimiser parameters, and nothing in this app resumes optimisation from a
`SplatCloud`: `TrainerInitializer` seeds from `PrePassResult.initialSplats`
(`TrainerSeed`), never from one of these. So the cost is zero here and the
benefit is every reader on earth.

**The exact change.** In `ios/Sources/Trainer/MetalSplatTrainer.swift`,
`readCloud(resources:count:shDegree:)`:

1. Read the stats alongside the splats:
   `let stats = resources.stats.readArray(TrainerSplatStats.self, count: count)`
2. Build a `filters: [Float]` in the SAME order as the arrays the loop appends
   to. Append `stats[i].filter3D` inside the existing
   `for (i, splat) in splats.enumerated()` loop, in the same place the other
   `append` calls happen, AFTER the non-finite `guard ... else { continue }`.
   Indexing a post-filter array with `i` is the one way this can silently
   mis-pair filters with splats, and it would look like a slightly wrong model
   rather than like a bug.
3. After the `SplatCloud` is built, fuse and record:

```swift
var cloud = try SplatCloud(...)          // exactly as today
let changed = try cloud.fuse3DFilter(filters)
TrainerLog.general.notice(
    "Fused the 3D low-pass filter into \(changed, privacy: .public) of \(cloud.count, privacy: .public) splats."
)
return cloud
```

`SplatCloud.fuse3DFilter(_:)` is already written and lives in Export
(`ios/Sources/Export/SplatCloud.swift`), together with the two pieces of maths
it uses, `SplatMath.filter3DCompensation` and `SplatMath.fusing3DFilter`. Do
NOT re-derive the formula in the trainer: one implementation is the whole point
of putting it there. It sets `cloud.filter3DFused = true`; it throws
`ExportError.alreadyFused` if it is ever called twice on the same cloud; it
throws `ExportError.inconsistentAttributeCounts` if the array length does not
match `cloud.count`; and it leaves any splat whose filter is zero, negative or
non-finite exactly as it was, so "no filter" always means "no compensation" and
never a crash or a NaN.

The fuse is EXACT for geometry, not an approximation:
`R S^2 R^T + f^2 I == R (S^2 + f^2 I) R^T`, so the widened Gaussian is the same
Gaussian with the same quaternion and per-axis sigma `sqrt(s^2 + f^2)`. Nothing
about the shape is lost. Only the pre-filter parameters are unrecoverable.

**Also set the flag honestly on the paths that do not fuse.** Three other places
build a `SplatCloud`, and they must not all silently read as `nil`:

- `MetalSplatTrainer.mergePreview(completedParts:current:)` builds a fresh cloud
  from merged arrays, which throws the flag away. Carry it across, or a fused
  preview will report itself unknown.
- The slice merge at `TrainerSlices.swift:402` has the same problem across
  slices. If every part is `true` the merged cloud is `true`; if any part is
  `false` it is `false`; otherwise `nil`.
- `PrePassInitialSplats.swift:603` builds the seed cloud. It has never been
  trained and has no filter, so `filter3DFused = false` is the honest value:
  it is a cloud whose producer KNOWS there is no filter to apply. Not `true`,
  and not `nil`.

**Until this lands, the viewer says so out loud.** `MetalSplatRenderer.load(_:)`
now logs a warning on `viewer.renderer` whenever a cloud arrives with
`filter3DFused == false`. It deliberately does not warn on `nil`: a cloud read
back from a file genuinely cannot know, and a warning that fired on every
import would be noise rather than information.

### `ios/Sources/Viewer/ScanCensus.swift`, `ScanCensus.Drawable`: one field wanted

Per rule 5 this belongs in the census rather than in a parallel mechanism, and
`Drawable` already has the precedent: `sourceFile` is a `var` that
`MetalSplatRenderer` stamps after `measure` has run. Add alongside it:

```swift
/// Whether the model being drawn had the trainer's 3D low-pass filter fused
/// into its scales and opacities. `nil` means the file could not say.
var filter3DFused: Bool?
```

and the renderer will stamp it in `load(_:)` exactly as it stamps `sourceFile`,
next to the line that already reads `cloud.filter3DFused`. On the review page
`false` should read as something like "this model is drawn more solid than it
was trained" rather than as a technical term. `Drawable` has no explicit
initialiser, so a field with a `nil` default added at the end does not disturb
`measure`.

### `docs/DATA_FORMAT.md`: the meaning of two stored fields has changed

No field is added or removed and no byte layout moves, so every existing file
still parses and every existing reader still works. What changes is what
`opacity` and `scale_0..2` MEAN. The doc currently describes them as raw
trainer parameters (the round-trip note near line 556 and the `.spz` note near
line 575). Please add, in the storage-conventions section:

> **Scale and opacity are draw-ready, not raw optimiser parameters.** The
> trainer fits each Gaussian through a Mip-Splatting 3D low-pass filter of a
> per-Gaussian width in metres: it renders the covariance widened to
> `Sigma + f^2 * I` and the opacity multiplied by
> `sqrt(det(Sigma) / det(Sigma + f^2 * I))`. That width is a training-time
> quantity with nowhere to live in `.ply`, `.spz` or `.glb`, so it is folded
> into the stored scale and opacity once, on the way out of the trainer, by
> `SplatCloud.fuse3DFilter`. A reader therefore draws the model that was
> actually fitted, with no extra field and no extra code, which is what makes
> the Blender add-on and any third-party viewer correct by default. The cost is
> that the pre-filter parameters cannot be recovered from a file, so a file
> written by this app is a finished model and not a training checkpoint.
> Files written before this change carry unfused values. They still read
> correctly; they simply draw a little sharper and more solid than they should.

### Optional, for a future re-import guard

`SplatCloud.filter3DFused` is `nil` for anything parsed out of a file, so
re-fusing an already-fused `.ply` cannot be detected. If that ever becomes a
real risk, `PLYCodec` could write one line, `comment nimbus filter3d fused`, and
set the flag when it reads it back. Not requested now: nothing in the app
re-fuses, and an unused marker is exactly the "written and never wired" pattern
this project keeps finding.

### What `MetalSplatTrainer.swift` now depends on in `TrainerDensifier.swift`

Coded against the file as it stands right now. Whoever owns that file: if any
of these three are renamed, the trainer stops compiling.

- `TrainerDensifyOutcome.created` (the computed `cloned + split`). Used as the
  test for "did this pass add anything", because relocation is correctly not in
  it. If `created` is ever redefined to include relocations, the stall detector
  goes blind, which is the fault it was built to catch.
- `TrainerDensifyOutcome.growthAllowed` and `.headroomAtStart`. Together they
  are the gate on the streak: only passes that were ALLOWED to add and had ROOM
  to add can count towards it. A pass that adds nothing outside the densify
  window is behaving correctly and must never raise the alarm.
- `.growthAllowance`, `.splatsScored`, `.splatsWithNonZeroScore` and
  `.candidatesAfterVisibilityFilter`, printed on the stall line.

Once `TrainerDensifyGrowthVerdict` / `growthVerdict` settles, the stall line in
`MetalSplatTrainer.trainSlice` should print the verdict instead of the four raw
counters. It was not used yet because that type is being written as this was
written and CI is the only compiler. Say when it is stable and the swap is one
line.

### The held-out PSNR is not measured on the picture the user sees, and the app should say so

Read this next to the fuse request above, because the two are connected.

**Before the fuse lands, the number is not about his model at all.** Held-out
evaluation runs `gpu.preprocess`, which is `trainer_preprocess`, which applies
the 3D low-pass filter and `comp3D`. The viewer applies neither. So today the
PSNR grades a model the user has never once looked at.

**After the fuse lands, it is about his model, but still not about his picture.**
`evaluateHeldOut` (MetalSplatTrainer.swift, the pixel loop around line 2179)
differs from the viewer in three ways that no fuse can close, and all three are
correct for what a held-out PSNR is supposed to mean:

1. It renders at `renderSize`, whose long edge is 720 and can be dropped to
   600, 480 or 384 by the thermal governor. The viewer renders at the
   drawable's own size.
2. It applies the learned per-frame exposure, `exposure.x * value + exposure.y`.
   The viewer has no exposure term at all.
3. It composites `transmittance[i] * background[i]` behind the render when the
   frame has a supervision background. The viewer composites against its own
   clear colour.

It also uses the capture's real intrinsics, where `renderIntrinsics` in the
viewer recentres the principal point on the drawable by design.

So the honest sentence is: **the PSNR says how well this model predicts a photo
it was never trained on. It does not say how good the preview looks, and it is
not measured on the preview.** That is the right number to report and the wrong
number to read as "how pretty is my scan".

**Where to put it, since the Viewer/Export agent owns none of these strings:**

- `ios/Sources/Trainer/MetalSplatTrainer.swift`, `doneMessage(cloud:psnr:governor:)`
  around line 2359, which currently renders as ". . . decibels." Add one plain
  sentence after it, something like: "That score compares the model against
  photos it never trained on, at the size it was trained at. It is not a score
  for how the preview looks on screen."
- `ios/Sources/Viewer/ScanReviewScreen.swift` / `ScanLibraryStore.swift`
  wherever `heldOutPSNR` is shown, the same sentence as a caption under the
  number.

Do not shorten it to "measured on held-out frames". The owner is not a graphics
engineer, and "held-out frames" is exactly the phrase that lets a wrong reading
survive.

### The seeder now records the trust line it drew; the census should print it

From the TrainerInitializer.swift agent. Nothing here is urgent and nothing here
blocks: the fix is complete and self-contained inside that file. These are the
two places where a number it now measures stops at the file boundary.

**1. `TrainerSeedResult.trustCut` is measured and logged, and the census cannot
see it.**

`TrainerInitializer` gained `TrainerSeedTrustCut`, carried on
`TrainerSeedResult.trustCut` (optional, nil on the pre-pass path because that
path takes no cut of its own; `PrePassCensus.seeding` already records its one).
It holds `wasMeasured`, `cut`, `floor`, `quantile`, `p05`, `median`, `p95`,
`cellsConsidered`. It is written, read and logged inside TrainerInitializer, so
it is not a dead field, but the trainer census cannot report it.

Two things would follow, when whoever owns those files is free:

- `MetalSplatTrainer.trainSlice`, in the block around line 655 that already
  copies `seedResult` into `census.slices[censusRow]`, could copy these across.
- `TrainerCensus.TrainerCensusSlice` would need the fields, and the alarms at
  the bottom of TrainerCensus.swift could then separate two states that read
  identically today. Alarm 7, `no_seed_was_trusted`, currently fires on
  `seedsPinnedAsDiscsTotal == 0` and says the trust gate rejected everything.
  With `wasMeasured == false` that sentence is wrong: no gate was consulted,
  the scan simply had no trust field, and the honest line is "this scan had no
  depth-reliability measurements, so every starting point was laid cautiously".
  Sending the owner to look for a broken threshold that was never read is the
  same waste the census exists to prevent. Only worth doing WITH the fields
  above; the census must not guess it.
- The pair worth printing together is `cut` against `p95`. A cut at the floor
  with a 95th percentile below it is the exact fingerprint of the fault that was
  just removed, and one line catches it returning.

**2. `MetalSplatTrainer.swift` line 729 converts a Float to an Int unguarded.**

    Int((seedResult.medianSpacingMeters * 1000).rounded())

A Float-to-Int conversion out of Int's range traps, in release as well as debug.
Until today `medianNearestSpacing` could return `Float.greatestFiniteMagnitude`
(its "no neighbour found" sentinel was tested with `.isFinite`, and that
sentinel is finite), so a sparse pre-pass seed set in a large room would have
killed the trainer on the line that logs how it started. The sentinel is fixed
in TrainerInitializer.swift and the value reaching that line is now always a
real distance or zero, so this is no longer reachable. It is still an unguarded
conversion on a Float that arrives from another file, and it costs one `min` to
make it unreachable by construction rather than by argument.

Also: `seedMedianSpacingMillimetres` in the census has no "not measured" state,
so an unmeasurable spacing lands there as 0 mm. `TrainerSeedResult.summary` now
drops the spacing clause rather than claiming "about 0 mm apart"; the census row
still says 0. Worth an optional or a sentinel if that field is ever read into a
sentence.

### The densifier now names WHY a pass grew nothing; the census can record it

From the TrainerDensifier.swift agent. Nothing here blocks: both changes are
complete and self-contained inside TrainerDensifier.swift and TrainerSupport.swift,
and everything new is already read from inside those files, so none of it is a
declared-and-never-wired setting. These are the places where a value it now
derives stops at the file boundary.

**1. `TrainerDensifyOutcome.growthVerdict` should be one more column in the
census row.**

`TrainerDensifier` gained `TrainerDensifyGrowthVerdict`, a `String`-raw-value
`Codable` enum, exposed as the computed property `TrainerDensifyOutcome
.growthVerdict`. It has nine cases: `populationEmpty`, `buffersReadShort`,
`grew`, `nothingScored`, `nothingVisible`, `growthWindowClosed`,
`atTheCapRelocated`, `atTheCapNoDonors`, `unexplained`. Nothing is stored: every
case is decided by a counter the pass actually measured, so it cannot report a
cause the code did not observe, and `unexplained` exists specifically so it
never has to invent one.

The reason it is worth a column: the census today reconstructs the same question
from seven separate counters at two different call sites (`couldHaveAdded` in
`MetalSplatTrainer.trainSlice`, and `passesWithGrowthWindowOpenAndHeadroom`
against `t.added` in `TrainerCensus.buildAlerts`), and neither of them can
distinguish `nothingScored` from `nothingVisible`, which are a dead gradient
signal and a dead visibility accumulator: two different faults with two
different fixes. The order the tests run in also matters and is easy to get
wrong. `growthVerdict` checks the scoring faults BEFORE the window and the cap,
because "the window was shut" and "the budget was full" are both perfectly
reasonable-looking explanations that will stand in front of a dead signal and
hide it. That ordering is now written down in exactly one place.

Two things would follow, when whoever owns those files is free:

- `TrainerCensusDensifyPass` (TrainerCensus.swift, around line 184) could take
  `var growthVerdict: String` and set it in its `init` from
  `outcome.growthVerdict.rawValue`. It is a pure add: no existing field changes
  and no existing number moves.
- `TrainerCensus.buildAlerts` alarm 1, `densification_created_nothing`, could
  then name the dominant verdict across the open passes instead of only quoting
  the scored counts. "Not one of 300,000 points had a gradient above zero" and
  "255,035 scored and none of them was visible in any frame" send the owner to
  two completely different places.

Also worth knowing: `TrainerDensifyOutcome.summary` MUST stay optional and MUST
stay nil for a pass that changed nothing. `MetalSplatTrainer.trainSlice` around
line 1213 branches on exactly that: nil is what routes a zero-growth pass to the
loud branch that prints the counters and the streak length. A non-nil string
there would quietly send the most important passes in the run back to `.info`.
This is now documented on the property itself.

**2. `trainer_reset_densify_stats` is documented as running after every pass and
does not.**

The shader's own comment (TrainerShaders.metal, above the kernel at line 395)
says "Called after every densification pass, never between iterations".
`MetalSplatTrainer.trainSlice` calls it under
`if outcome.changedTopology || outcome.relocated > 0`. So a pass that changed
nothing does NOT reset the accumulators, and `absGrad2D`, `denom`, `visAccum`
and `unknownAccum` keep accumulating into the next interval.

`absGrad2D / denom` is a mean, so it is largely self-correcting. `visAccum` is
not: it is only ever compared against zero, and it never resets on a quiet pass,
so the "something actually looked at it" filter gets steadily more permissive
the longer densification goes without changing anything. That is the wrong
direction. A stage that is producing nothing should not also be quietly
loosening the gate that decides whether it has candidates.

Not fixed here because the call is in MetalSplatTrainer.swift and the comment is
in TrainerShaders.metal, and neither is this agent's file. Either the call
should be unconditional, or the shader comment should say what actually happens.
It should not be left saying one thing while the code does another: that is the
same class of thing as the settings that were declared and never read.

**3. What this cost, for whoever is tracking the thermal budget.**

The candidate ranking in `TrainerDensifier.run` was a full sort of an index
array covering very nearly the whole population, run every
`densifyIntervalIterations` for the whole run (29 passes on a 3,000 iteration
slice, not only the 19 inside the growth window). The donor ranking on the
relocation path was worse: a full sort whose comparator called `sigmoid`, and so
`expf`, twice per comparison rather than once per element.

Both are now bounded selections of only the part that is ever read. Counted
exactly at a 300,000 point population: 4,242,295 comparisons against 1,104,232.
Measured on a desktop transcription of exactly the new code, as a ratio and not
as a device timing: candidate ranking 101.0 ms against 18.5, donor ranking 335.9
ms against 17.7 in its worst case, and with the growth window shut the pass now
does no ranking at all, 101.0 ms against 0.9. The score floor is unchanged and
is still `score[i] > 0`, so nothing here reintroduces a magnitude threshold.

`TrainerTuning.maxGrowthFractionPerPass` and `maxRelocationFractionPerPass` now
also set the size of that selection, which is documented on both. Setting
`maxGrowthFractionPerPass` to 1.0 restores the full-population sort.

---

## Resolved by the compile gate, 2026-09-06 (the six-agent pass)

Every request filed above by the six agents that ran in parallel today is
answered here: made, or refused with a reason. Nothing is left open without a
line saying so. Verified against the source, not against the notes.

### ACTIONED: the fuse was written and wired to nothing

**This was the largest hole in the set.** `SplatCloud.fuse3DFilter(_:)`,
`SplatMath.filter3DCompensation` and `SplatMath.fusing3DFilter` were complete
and correct, and `grep` found **zero callers**. `SplatCloud.filter3DFused` was
read in exactly one place (`MetalSplatRenderer.load(_:)`, the warning) and set
to `true` in exactly one place (inside `fuse3DFilter` itself), so the warning
branch was unreachable and every exported model was still the wrong model. The
viewer's `filterVariancePx` was correctly changed to 0.25 to match the trainer,
which closed the 2D half of the mismatch and left the 3D half entirely open.

Made, exactly as the request specified:

* `MetalSplatTrainer.readCloud(resources:count:shDegree:)` now reads
  `resources.stats` alongside `resources.splats`, builds `filters: [Float]`
  **inside the existing loop and after the same non-finite `continue`** so it
  stays index-for-index with `logScales`, and calls `cloud.fuse3DFilter(filters)`.
  `readArray` returns an EMPTY array when the buffer is short, so
  `haveFilters = stats.count == splats.count` is a real test; both failure
  paths set `filter3DFused = false` and log at `.error`, and neither throws the
  cloud away.
* The fuse is deliberately outside the `do` that builds the cloud. A cloud that
  cannot be fused is still a cloud; discarding it would turn a cosmetic loss
  into an empty model.
* Flag carried across every other construction site, using one shared rule,
  `SplatCloud.mergedFilter3DFused(_:)` (false if any part is false, true only
  if every part is true, nil otherwise, nil for an empty list):
  `MetalSplatTrainer.mergePreview`, `TrainerSlices.merge`, and
  `TrainingPreview`'s downsample (a subset of the same splats inherits the same
  fact). `PrePassInitialSplats` sets `false`, with a comment saying plainly that
  nothing reads it today and why it is still set.
* `ScanCensus.Drawable.filter3DFused` added and stamped by
  `MetalSplatRenderer.load(_:)` next to `sourceFile`, and surfaced on the
  review page through `drawableAlert` on the "Points that can actually draw"
  rung. `nil` says nothing, as asked.
* `docs/DATA_FORMAT.md` section 8 now states that scale and opacity are
  draw-ready rather than raw optimiser parameters.

### ACTIONED: the depth-loss denominator is now the supervised count

All three bullets of the loss-normalisation request, together:

* `TrainerLossUniforms` gained `depthSupervisedCount: UInt32` at offset 64.
  Stride 64 -> 68, align 4. Header comment, the `check(...)` line in
  `TrainerGPULayouts.verify()`, and the Metal struct all updated.
* `MetalSplatTrainer.runIteration` sets it from
  `supervision.supervisedSampleCount`, and **recounts over the uploaded prefix**
  when `sampleCount` truncated the array, exactly as the caution asked.
* `trainer_loss_depth` divides by it.

**One deviation, and it is a correction rather than a preference.** The request
said to use `u.depthSupervisedCount` outright. The kernel falls back to
`u.depthSampleCount` when it is zero, because the F2 free-space hinge does NOT
carry `s.weight`: a frame whose photo QC weight is zero has no supervised
samples and can still have thousands of live hinge terms, and a divisor of one
there would put an unnormalised population sum straight back into the loss.
That is the fault the whole change exists to remove, so the fallback is
load-bearing, not defensive. It is written down in the kernel comment.

### ACTIONED: `trainer_reset_densify_stats` now runs after every pass

The call in `MetalSplatTrainer.trainSlice` was under
`if outcome.changedTopology || outcome.relocated > 0`, and the shader's comment
said "after every densification pass". The call is now unconditional and the
shader comment says why that matters. `visAccum` is only ever compared against
zero, so a pass that skipped the reset made the candidate filter steadily more
permissive the longer densification went without doing anything, which is the
wrong direction. Cost: one extra command buffer per empty pass, roughly 29 per
slice, one kernel over the population.

### ACTIONED: the census

* `TrainerCensusDensifyPass.growthVerdict: String`, copied from
  `outcome.growthVerdict.rawValue`. Not re-derived.
* `TrainerCensus.buildAlerts` alarm 1 now names the dominant verdict across the
  open passes.
* New loud alarm `densification_stalled`, on
  `Totals.longestZeroGrowthStreak >= 10` with something created earlier in the
  run. The streak is counted per slice (two slices are two populations and a
  streak must not be stitched across the join).
* Alarm 13 split: `most_iterations_did_no_work` at "loud" when more than half a
  run took no optimisation step, `many_iterations_did_no_work` at "check" from
  5 per cent, as suggested.
* `TrainerCensusSlice.iterationsWithGradientStep`, measured by the loop, plus a
  new "check" alarm `gradient_step_count_does_not_reconcile` that fires when
  the stored count and the skip-counter subtraction disagree. Storing both
  sides is what makes a skip path that stops being counted visible.
* `depthSamplesPerFrameTotal`, `depthSamplesSupervisedTotal` and
  `depthSupervisionFramesMeasured` per slice, accumulated ONLY on `.stepped`,
  plus a "check" alarm when the supervised fraction is under 5 per cent. The
  divisor is stored so a mean is only ever printed when there was one to take.
* `TrainerSeedTrustCut` copied into the slice row as eight optional fields.
  Alarm 7 now splits: `no_depth_reliability_to_trust` when
  `wasMeasured == false` (no gate was consulted, so do not send anyone hunting
  a threshold), and `no_seed_was_trusted` otherwise, now printing the cut
  against the 95th percentile it was applied to.
* `seedMedianSpacingMillimetres` is now `Int?`. 0 mm is not a spacing.
* `PrePassCensus.Carving.keyframesDepthUnreadable: Int?` added, with its
  CodingKey, `decodeIfPresent` and `encodeIfPresent`, filled by the carver's
  `defer`. The detail string reads "could not be read" instead of "had no depth
  to read", and appends the split only when the writer measured it.
* The carving ledger line now distinguishes "there was no map of empty air" from
  "a map was loaded and licensed no deletions", using the `carverAvailable`
  flag that was already recorded per pass. **The requested
  `freeSpaceGridLoaded` flag was not added: it already exists** as
  `TrainerCensusDensifyPass.carverAvailable` and was simply not being rendered.

### ACTIONED: the small ones

* `PrePassSurvey` now calls `PrePassDepthFrame.loadOutcome` and increments
  `unreadableDepthFrames` on `.unreadable`. The `unreadable_depth` QC finding
  can fire for its own named cause for the first time.
* `TrainerProgress.gradientStepsCompleted` and `.consecutiveZeroGrowthPasses`,
  appended last with `nil` defaults, and **wired**: the in-loop progress tick
  fills both. The stage-change ticks around the loop leave them nil, because
  nil means nobody counted and a stale count presented at the wrong moment
  would be worse than silence.
* `SplatModel.iterationsCompleted` now carries the doc comment saying it is
  times round the loop, not optimisation steps, and names where the real count
  lives.
* `SplatModel.heldOutPSNR` documents the three ways held-out evaluation differs
  from the viewer, and `doneMessage` now says the plain sentence in full. The
  requested caption in `ScanReviewScreen` / `ScanLibraryStore` was NOT added,
  because neither of those displays `heldOutPSNR` at all: `ScanSummary` stores
  it and nothing reads it. There is no string to caption.
* `MetalSplatTrainer.swift`'s unguarded `Int((medianSpacingMeters * 1000).rounded())`
  is now clamped and finiteness-checked before the conversion.
* The stall line in `trainSlice` prints `outcome.growthVerdict.rawValue`
  alongside the counters, as asked once the type settled. It is stable.
* `TrainerShaders.metal` gained `static_assert(sizeof(X) == N)` for all
  thirteen GPU structs. `TrainerGPULayouts.verify()` already checked these, but
  at start-up on the owner's phone; these fail in CI instead, on the wrong line.
  The viewer's shader has carried the same guard from the start.

### REFUSED, with reasons

* **Carrying `filter3D` through the file formats instead of baking it.** Agreed
  with the filter3d agent's reasoning and it is now moot: the bake has landed.
  A `.ply` `comment nimbus filter3d fused` marker is also refused for now.
  Nothing in the app re-fuses, `fuse3DFilter` already throws
  `ExportError.alreadyFused` on the one cloud that could, and an unused marker
  is the written-and-never-wired pattern this pass exists to remove.
* **Unblocking the F2 free-space hinge on UNKNOWN and BAND samples.** The
  agent's own evidence stands: `expected = accumulated / max(alpha, 1e-4)` on
  an empty pixel puts a factor of up to 1e4 and 1e8 on the chain rule, so
  enabling the branch as it stands is an amplifier on the emptiest pixels in
  the frame. `borrowedFreeSpaceBound` stays computed and unused, and that is
  the lesser fault. It needs an alpha floor designed with it.
* **Splitting `lossAccum` into four scalars.** Correct diagnosis: the reported
  loss is mostly the regulariser's population sum and photometric progress is
  invisible in it. Not done, because `lossEMA` is display and census only,
  nothing gates on it, and this pass was not the moment to change four kernels
  and a readback for a reporting improvement.
* **Holding the census in a reference box instead of `inout`.** The hazard is
  real and correctly described. It is roughly thirty mechanical edits across
  the largest file in the project, on a day when six agents had already
  rewritten parts of it and CI is the only compiler. Deliberately deferred, not
  forgotten.
* **Swapping `load` for `loadOutcome` in `PrePassInitialSplats`,
  `PrePassBundleAdjuster`, `PrePassGlassDetector` and `PrePassPoseRefiner`.**
  The Survey one was the urgent case and it is done. These four change no total
  that is wrong today; they would only split a count that is currently correct.
  Worth doing on a quiet day, not on this one.
* **`ScanReviewScreen` PSNR caption.** See above: there is nothing on that
  screen to caption yet.

### One thing found by the gate, NOT changed, evidence attached

`trainer_loss_depth` writes `gradDepth[s.pixelIndex] +=` and
`gradTFinal[s.pixelIndex] +=` non-atomically, and several depth samples sharing
one render pixel would lose updates. It cannot happen today and the reason is
arithmetic, not luck: `TrainerSupervision` maps a 256 x 192 native sample to
`px = floor((u + 0.5) * W / 256)`, and every rung of
`TrainerBudgetGovernor.resolutionLadder` (720, 600, 480, 384 long edge at 4:3)
gives a render grid of at least 384 x 288, so the map is injective in both axes
in portrait and in landscape. If anyone ever adds a rung below 256 columns or
192 rows, this becomes a silent race in the depth gradient. Worth a comment at
the ladder if that is ever considered.

---

## From `Sources/PrePass` to `Sources/Smart`: route the F6 and F3 sidecar paths and byte layouts through `PrePassPaths` / `PrePassBinaryIO`

Raised 2026-09-06 during the dead-wire audit. Nothing is broken today. This is
a request to remove a duplication that would fail silently if it ever drifted.

**What is duplicated.** `PrePassPaths` declares `trustBias`, `trustNoise`,
`depthAffine`, `confidenceRecalibrated` and `edgesDirectory`, and
`PrePassBinaryIO` declares the record layouts for them in
`PrePassTrustBiasFile`, `PrePassDepthAffineFile` and `PrePassSampleFieldFile`,
each with the layout table quoted from `docs/DATA_FORMAT.md`. None of those five
constants and none of those three codecs is called by anything.

The files are still produced, because `Sources/Smart` builds the same strings
and the same bytes itself:

- `Smart/TwoScaleTrustField.swift` lines 212-215 build
  `"\(BrandConfig.Folder.prePass)/trust_noise.bin"` and the other three as
  local string literals.
- `Smart/TwoScaleTrustField.swift` lines 492-507 serialise the bias records and
  the affine records inline with `SmartBinary`.
- `Smart/NativeDepthEdgeClassifier.swift` line 84 builds
  `"\(BrandConfig.Folder.prePass)/edges"` the same way, and its
  `stem(forImagePath:)` at line 442 duplicates `PrePassPaths.stem`.

**Verified identical on 2026-09-06**, so this is not a bug report:

- bias record: `UInt64` key, `Float32` mean, `Float32` variance, `UInt32`
  count, `UInt16` distinct times, `UInt16` reserved = 24 bytes, both sides.
- affine record: `UInt32` frame, `Float32` scale, `Float32` shift = 12 bytes,
  both sides.
- `SmartBinary.append(_ value: Float,...)` applies the same non-finite guard as
  `PrePassBinary.appendFloat`, so the file header's rule 2 holds on both paths.
- the two `stem` implementations are character for character the same.

**Why it still matters.** A rename or a layout change on one side alone
produces no compile error and no runtime error. The pre-pass would report the
stage as successful, the trainer's `TwoScaleTrustField.load` would find nothing,
and every depth sample would silently fall back to "no opinion" while the
census still said the field was built. That is the same shape as the four
scan-destroying bugs this audit exists to find.

**Interim mitigation, already landed in `Sources/PrePass`.**
`PrePassPaths.missing(_:at:)` is new, and `PrePassPipeline` now calls it right
after the F6 and F3 stages report success. A disagreement now raises the QC
findings `trust_files_missing` and `edge_maps_missing` at `.problem` severity
instead of passing unnoticed. This detects drift; it does not prevent it.

**The ask, for whoever owns `Sources/Smart`:**

1. Replace the five local path strings with `PrePassPaths.trustBias`,
   `.trustNoise`, `.depthAffine`, `.confidenceRecalibrated`,
   `.edgesDirectory`, and the per-frame edge name with `PrePassPaths.edgeMap`.
2. Replace the inline bias and affine serialisation with
   `PrePassTrustBiasFile.encode` and `PrePassDepthAffineFile.encode`, and the
   `.appendFloats` block writer with `PrePassSampleFieldFile.encode` if the
   streaming writer can be reworked without losing its memory ceiling. If it
   cannot, say so and the codec should be deleted rather than left as a
   lookalike, since a second definition of a layout is worse than none.
3. Once 1 and 2 land, the `PrePassPaths.missing` checks can stay as cheap
   insurance or be dropped. They cost one `fileExists` per stage.

There is no module-boundary objection to this: `project.yml` builds every
`Sources/*` folder into the ONE Xcode target `App`, so `Sources/Smart` can name
`PrePassPaths` directly with no import and no dependency cycle (`PrePassPipeline`
already names `TwoScaleTrustField` the same way). If the folders are ever split
into real modules, move `PrePassPaths` and the three codecs into `Sources/Core`
instead so both sides read one definition. Either resolution is fine; two
definitions is not.

---

## From Capture: `ARCaptureService.hasOpenScan` is a guard nobody asks for

`ARCaptureService.hasOpenScan` (`phase == .recording || phase == .halted`) and
`isRecording` are both public and both read nowhere in the app. They are
harmless as they stand, but `hasOpenScan` is the exact question the app shell
should be asking before it lets the user navigate away from the capture screen
or open a second scan: a scan that is open but not finished has frames on disk
and no index written for them.

`App/NimbusApp.swift` and whatever owns navigation are not mine to touch. If
the shell wants that guard, the property is there and is already correct.

## From Capture: point-cloud coarsening is logged but not in the bundle

`CapturePointCloudAccumulator.didDegrade` and `effectiveVoxelSizeMeters` record
that the cloud had to coarsen past the format's nominal 1 cm to stay inside its
cap. The coarsening logs itself (old size, new size) so it is not hidden, but
`CaptureBundle` has no field for it, so a downstream stage reading the bundle
cannot tell that the cloud it was handed is lower resolution than the format
says. Adding a field to the bundle contract is a Core change, not a Capture one.

---

## From Core/Smart/App/Onboarding (deadwire pass): five cross-module wirings

Five things I found unwired live in files I do not own. Each names the exact
symbol and the exact call site.

### 1. Trainer: the census cannot say the mid regime is a stub

`DirectionalBackgroundModel.midRegimeProvenance` and `.isMidRegimeReal` exist so
a stubbed run is never mistaken for a real one. On the phone
`SmartMonocularDepthStub.isAvailable` is `false` and `estimate(...)` returns nil,
so the 4.5 to 30 m band is routed around rather than measured, and until now
nothing anywhere said so.

Done on my side: `write(to:)` now puts the provenance into
`model/background.json` (`summary`, plus new optional `midRegimeProvenance` and
`midRegimeIsReal` fields), and `warmUp` logs it once per run.

Wanted from Trainer: `MetalSplatTrainer` builds the model at
`MetalSplatTrainer.swift:2646` (`DirectionalBackgroundModel(settings: settings)`).
`TrainerCensus` should carry `background.isMidRegimeReal` and
`background.midRegimeProvenance` so `model/train_census.json` states which of the
two ran. Both are cheap property reads; neither infers anything.

### 2. Trainer: the glass confirmation rate is measured and unread

`SmartGlassMask.confirmedFraction` is the fraction of a frame's samples confirmed
as glass. A confirmed pane multiplies authority by ZERO, so a frame that is
mostly window contributes almost no depth evidence, however good the capture was.

Done on my side: `SmartAuthorityMap.map(for:)` now reads it, warns once per
prepared scan when a frame is at least half confirmed glass, and keeps a measured
running total in the new `SmartAuthorityMap.glassDominatedFrameCount`.

Wanted from Trainer: `TrainerSupervision.swift:282` already reads
`authority?.map(for: frame.index)?.meanAuthority`. Read
`authorityMap.glassDominatedFrameCount` at the end of the run and put it in
`TrainerCensus` next to the frame count. It is a count, not an estimate; report
it with the number of frames built or it cannot be interpreted.

### 3. PrePass: the economy preset now selects itself, and the pipeline should
know it can

`SmartLossSettings.economy` was declared, documented ("a phone that is already
warm, or a house-sized scan") and selected by nothing, so every trust build ran
at full cost: 4 partner frames, stride 2, a 20 000-sample plane-sweep budget.
`PrePassPipeline.swift:249` constructs `TwoScaleTrustField()` with `.default` in
`init`, before `deviceTier` is known, so the pipeline could not have chosen.

Done on my side: `SmartLossSettings.trustBuildCost(requested:frameCount:thermalLevel:)`
makes the choice from two measurements, and `TwoScaleTrustField.build` calls it
at the top of every build and logs which preset ran and why.

Wanted from PrePass, optional: `deviceTier` is a better signal than either of
mine for a phone that is merely small rather than hot. If `PrePassPipeline`
rebuilds its trust field in `run` once the tier is known, pass
`SmartLossSettings.economy` for `.limited` explicitly rather than leaving it to
the thermal check.

### 4. Viewer: the trained background is never shown

`Viewer/SplatRenderShaders.metal:673` composites a flat grey
(`float3 background = float3(0.09f, 0.095f, 0.105f)`) behind the splats. Nothing
in `Sources/Viewer` reads `model/background.bin`, and
`DirectionalBackgroundModel.load(from:)` and `.currentCubemap` are both
documented "for the viewer" and called by nothing. So the direction-only far
field the trainer fitted and froze is not in any preview or review render: every
sky and every distant wall shows as that grey.

`DirectionalBackgroundModel.composite(gaussianColor:accumulatedAlpha:direction:)`
is the exact operator the trainer's photometric kernel uses
(`splat + T * bg`, `TrainerShaders.metal:963`), written as a CPU mirror so both
ends agree. It is the reference for whoever wires this up.

### 5. Capture: `sparse/0` is spelled out rather than read from the brand block

`BrandConfig.Folder.sparseModel` is `"sparse/0"`, the COLMAP model path
`docs/DATA_FORMAT.md` specifies, and nothing reads it.
`CaptureScanFolder.swift:48` builds the same path as
`Folder.sparse` + `"0"`, and line 81 as `"\(BrandConfig.Folder.sparse)/0/points3D.txt"`.
No drift today, because both spell "0" the same way. The brand block exists so
there is one place to change it; two hand-written `"0"`s is where that stops
being true.

### 6. Docs: `model/background.json` has two new optional fields

`docs/DATA_FORMAT.md:605` describes `background.bin` + `background.json` in one
sentence and does not list the header's fields, so nothing there is now wrong.
The header gained `midRegimeProvenance` (String, optional) and `midRegimeIsReal`
(Bool, optional). Both are absent in files written by older builds and decode as
nil; the format version is unchanged because no byte layout moved and no reader
needs them.

---

# RESOLUTIONS - 2026-09-06, integration gate

Every request open above is answered here: made, or rejected in writing with the
reason. Nothing is left "noted". Where a request was made, the call site is
named so it can be checked without trusting this note.

## MADE

### Trainer census now records the mid regime (request 1)

`TrainerCensus` gained `midRegimeIsReal: Bool?` and `midRegimeProvenance:
String?`, and `MetalSplatTrainer` fills both from `smart.background` right after
`census.finalSplatCount` is set, at the end of the run. `buildLedger()` adds a
sentence when the band was NOT measured: "depth from 4.5 to 30 m was not
measured on this device and was worked out from camera movement instead."

Left nil when the far field could not be fitted at all. That is a third case,
distinct from `false`, and recording it as `false` would claim a measurement
that never happened.

`docs/DATA_FORMAT.md` section 8 now lists both fields, and the `background.json`
paragraph now has a table for the matching pair on that side.

### Trainer census now records the glass-dominated frame count (request 2)

Made, and with the denominator the request itself asked for. Reading
`glassDominatedFrameCount` alone would have been useless: "4 glass-dominated
frames" is a catastrophe out of 5 and a footnote out of 400, and the property's
own doc comment says so.

So `SmartAuthorityMap` also gained `builtFrameCount`, backed by a `builtFrames`
counter incremented inside the existing cache-insert lock in `map(for:)` and
reset in `prepare(...)`. It counts frames whose map was actually BUILT rather
than served from the LRU cache, which is the honest denominator.

`TrainerCensus` gained `authorityFramesBuilt: Int?` and `glassDominatedFrames:
Int?`, a ledger line, and a new check-severity alert `glass_dominated_frames`
that fires at or above `glassDominatedRunPercent` (33 per cent). Below a third,
a few windows in a room is normal, and saying it every time would train the
reader to skip the alerts.

Note for the record: when this gate started, `glassDominatedFrameCount` was
itself dead. It was added earlier the same day, exposed publicly, and read by
nothing. The log line it introduced even ended "The running total is in
glassDominatedFrameCount", pointing at a number no screen and no file carried.
That is the exact bug class this whole pass exists to find, re-introduced by the
work that was doing the finding, which is the strongest argument there is for
the CI gate described at the end of this section.

### `sparse/0` now comes from the brand block (request 5)

`BrandConfig.Folder.sparseModel` is now read by both places in
`Capture/CaptureScanFolder.swift` that used to spell the `"0"` by hand:
`sparseModelDirectory` and `pointCloudRelativePath`.

The directory version appends one component at a time, in a plain loop, rather
than passing `"sparse/0"` to `appendingPathComponent` in one go. The one-shot
form is almost certainly fine, but "almost certainly" is not good enough for the
path the COLMAP export writes to, and there is no compiler or device in this
session to settle it. The loop makes no assumption at all and costs nothing.

`BrandConfig.Folder.sparse` is still live (`Export/BoosterBundle.swift:47`), so
nothing was orphaned by the change.

### `docs/DATA_FORMAT.md` updated (request 6)

Done, and wider than asked. The request only pointed out that
`background.json`'s two new optional fields were undocumented. The same section
did not list ANY of that header's fields, and section 8's census table did not
carry the four fields added today either. Both are now written down, including
what an ABSENT optional means in each case, because "absent" and "false" are
different statements and a reader a year from now cannot tell them apart from
the code.

`glass_dominated_frames` is added to the stable `alerts[].code` list.

## REJECTED, with reasons

### Route Smart's F6/F3 sidecar writers through `PrePassPaths` / `PrePassBinaryIO`

REJECTED FOR NOW. The analysis is right and the duplication is real:
`Sources/Smart` builds the five sidecar paths and serialises the bias and affine
records itself, byte-identically, and a rename on one side alone would produce
no compile error and no runtime error.

But the fix asked for is a rewrite of the writers that produce the files the
trainer reads, performed with no Swift compiler and no device anywhere in this
session. The failure mode of getting it subtly wrong is precisely the failure
mode being defended against: the pre-pass reports success and the trainer opens
nothing. Trading a hypothetical future drift for a possible present breakage is
a bad trade on a project whose one real scan already came out looking like
nothing.

The mitigation that landed with the request is the part that actually matters,
and it is already in: `PrePassPaths.missing(_:at:)` is called right after the F6
and F3 stages report success, and a disagreement now raises
`trust_files_missing` or `edge_maps_missing` at `.problem` severity on the QC
card the owner reads after every scan. Silent drift is no longer possible. Loud
drift is survivable.

Do the refactor in a session that can build and run it, and do it in the
direction the request's own last paragraph suggests: move `PrePassPaths` and the
three codecs into `Sources/Core` so both sides read one definition.

### `deviceTier` should also select the economy trust preset (request 3)

REJECTED. The request marks itself optional, and the gap it describes is already
closed by two measurements: `SmartLossSettings.trustBuildCost` downshifts on
thermal state at or above `.serious`, or on more than 1,200 frames, and
`TwoScaleTrustField.build` calls it at the top of every build and logs which
preset ran and why.

Honouring `deviceTier` as a third signal means restructuring when the trust
field is constructed: `PrePassPipeline.swift:249` builds it in `init`, before
the tier is known. Reordering construction to gain a third opinion about
something two measurements already decide is not worth the risk today.

### Show the trained background in the viewer (request 4)

REJECTED AS OUT OF SCOPE FOR THIS GATE, and it is the most valuable thing left
open. The finding is correct and it is not cosmetic: the trainer fits and
freezes a direction-only far field, and `Viewer/SplatRenderShaders.metal:673`
composites a flat grey behind the splats regardless. Every sky and every distant
wall in every preview is that grey.

It is rejected here because it is a FEATURE, not a wiring fix. It needs
`model/background.bin` loaded, a cubemap uploaded as a texture, a new shader
input, a decision about what to draw when the file is absent, and a look at the
result on a screen. Every one of those steps is unverifiable in this session.
Wiring a render path blind is how you ship a black screen, and this app has
already shown the owner one thing that looked like nothing.

`DirectionalBackgroundModel.composite(gaussianColor:accumulatedAlpha:direction:)`
is kept, allowlisted and documented precisely so that whoever does this has the
CPU reference for the operator the GPU uses. It is the first thing to pick up
next.

### `hasOpenScan` as a navigation guard (Capture to App)

REJECTED. The property is correct and costs nothing where it is. Adding a
navigation guard to the app shell is a UX change with a real failure mode of its
own: a guard that misjudges "open" traps the user on the capture screen with no
way out, which is worse than the problem it solves. The request itself says the
properties are harmless as they stand.

Both `hasOpenScan` and `isRecording` stay, allowlisted with that reasoning. The
offer stands for whoever takes on navigation deliberately.

### Point-cloud coarsening should be a field on `CaptureBundle` (Capture to Core)

REJECTED. `CaptureBundle` is a `Codable` contract written to disk and read by
three modules; adding a field to it is a format change. The coarsening is not
hidden today: `CapturePointCloudAccumulator` logs the old and new voxel size
when it degrades.

The value of the change is that a downstream stage could react to it, and no
downstream stage wants to yet. Change the contract in the commit that adds the
first reader, and not before.

## The gate that stops this list growing silently

`tools/deadwire.py` now reads `tools/deadwire_allowlist.txt` and exits non-zero
on any unreferenced declaration that is NOT on it, and a `deadwire` job in
`.github/workflows/ios.yml` runs it on every push. The 112 known-harmless
candidates are listed with a reason each. A NEW declaration that nothing calls
fails the check.

It deliberately does not block the IPA. The archive job has no `needs:` on it,
because the phone build is how the owner gets a working app and a lint finding
must never be the reason he cannot install one.
