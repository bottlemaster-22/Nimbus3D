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
