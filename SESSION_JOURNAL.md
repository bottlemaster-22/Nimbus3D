# Nimbus3D : Session Journal

Append-only continuity log. Newest entry at the bottom. Read the latest entry to resume.

---

## 2026-07-09 : Day 1, concept + naming

**What happened**
- New TOMBLINE project born: a Gaussian splat capture app that outputs a game-ready 3D asset (best-topology low-vertex mesh) plus lighting.
- Ran a deep research pass (6 dimensions + 2 adversarial fact-checks) on the 2024-2026 state of the art.
- Named it **Nimbus3D** (splat = cloud of blobs; nimbus also nods to sky/HDRI environment).

**Decisions locked**
- Target user: game devs / 3D artists (topology + low vertex count is the moat).
- Scope: FULL pipeline (capture -> splat -> clean mesh + HDRI). v1 narrowed to single opaque matte props.
- Name: Nimbus3D.

**Key findings (see BRIEF.md)**
- "Free HDRI as a byproduct of the splat" does NOT work (lighting is baked into color; LDR clips the highlights an HDRI needs). Reframe: capture the HDRI deliberately as a quick second step in the same session.
- Auto splat -> clean low-poly is only partially automatic: great for static props, still needs a human pass for deforming hero characters.
- The market's triple-intersection (real object + best-topology low-poly + real captured HDRI + delit) is EMPTY. That's the product.
- Closest competitor: KIRI Engine (splat -> quad mesh + PBR), but no real lighting, no proper delight.

**New idea raised (feasibility under research)**
- "Dynamic Textures": detect surface type (carpet/brick/etc.), then fake surface depth cheaply via height/displacement maps + parallax occlusion mapping instead of geometry. More realism, less intensive, more optimization. Stacks on the delighting work.

**Open**
- Platform: phone-only vs cloud-assisted for heavy steps.
- Dynamic Textures: confirm feasible, pick faithful-bake vs smart-material-substitution vs hybrid.

**Next step**
- Finish Dynamic Textures feasibility research, then refine BRIEF.md and pick platform direction.

---

## 2026-07-09 : Day 1 (cont.), platform + delivery locked

**Decisions locked**
- Platform: **native iOS app**. (User corrected an earlier bad assumption that phones are too weak; modern iPhones handle the heavy work, Scaniverse proves it.)
- Dynamic Textures: **hybrid** flavor confirmed (faithful bake + smart material enhancement).
- Build approach: user wants ONE Fable 5 agent workflow that builds everything in a single pass, NO phased hand-offs.

**Delivery / distribution figured out**
- Ship by sideloading through the user's own **Bottle** stack (`C:\Users\Undea\Documents\Bottle`). Bottle = self-hosted iOS sideloader: Node broker signs an UNSIGNED IPA with a free Apple ID (zsign + GrandSlam auth), hosts it + an Ed25519-signed manifest, and installs via a Rust agent over USB (usbmuxd -> AFC -> installation_proxy Upgrade) or a computer-free LAN relay. 7-day cert, auto-refreshed.
- **GitHub confirmed usable**: `gh` authed as `bottlemaster-22`, scopes `repo` + `workflow`. Plan: GitHub Actions macOS runner builds an UNSIGNED Nimbus3D IPA (`CODE_SIGNING_ALLOWED=NO`) -> GitHub Release asset. No Apple certs in CI (Bottle signs).
- Wrote `BOTTLE_INTEGRATION.md`: exact, code-grounded instructions for the OTHER (Bottle) Claude session to add an "Install Nimbus3D" button (add app to source.json, make `resolveUnsignedIpa()` app-aware, per-app entitlements, add bundle_id to the identity/state DB key, wire button to `POST /sideload/build`). Pipeline is otherwise app-agnostic; near-zero changes to signer/installer.

**Open**
- Dynamic Textures feasibility research still running -> then build the plan.

**Next step**
- On research completion: write the implementation plan, then author the single Fable 5 build workflow.

---

## 2026-07-09 : Day 1 (cont.), Dynamic Textures verdict + PLAN.md written

**Dynamic Textures research done (verdict: PARTIALLY FEASIBLE)**
- The rendering cheat (normal map + Parallax Occlusion Mapping) is SOLVED and cheap; POM is not deprecated in 2026 (still the default for granular/aggregate surfaces). The name the user forgot = parallax occlusion mapping; height map = the grayscale asset.
- The two fragile, human-in-the-loop parts: (1) material-type DETECTION (no shippable general classifier for consumer splats; ~60-80% on casual captures), (2) DELIGHTING.
- KEY design change: you CANNOT recover real micro-relief from a splat (view-dependent color + splat-to-mesh smooths detail). Fine detail must be SUBSTITUTED/SYNTHESIZED, not extracted. So the hybrid becomes: capture -> color + coarse form; library/synth -> fine relief. Detection = one-tap SUGGESTION, never silent.
- Best-ship path (from verdict): substitution over extraction. Curated ambientCG (CC0) tileable materials, pre-delit, keyed by coarse class, tinted from capture; normal+POM baseline; optional Nanite/tessellation hero toggle.
- POM caveat: flat silhouette + grazing-angle breakdown, which lands inside a free-orbit viewer. Use POM on the game mesh; real displacement for hero close-ups.

**PLAN.md written**
- Honest architecture: iOS app (capture/preview/HDRI) + GPU backend (the heavy conversion) + GitHub Actions CI (unsigned IPA) + Bottle sideload.
- Pipeline stages tagged off-the-shelf / integrate / R&D. v1 = single opaque matte props.
- Stated plainly: one Fable 5 pass yields a runnable END-TO-END SKELETON to harden, NOT a finished product. Delight + material-detect are the quality-risk stages; on-device heavy processing is the big unknown.

**Two decisions put to the owner (blocking the Fable 5 build)**
1. Where the heavy GPU processing runs: own PC/server vs rented cloud GPU vs fully on-device.
2. What the first big build produces: full end-to-end skeleton vs backend-first vs app-first.

**Next step**
- Get the two decisions, finalize PLAN.md, then author the single Fable 5 build workflow.

---

## 2026-07-10 : Build executed, repo live, CI at the Brush gate

**Owner decisions (FIRM)**
- Processing: FULLY ON-DEVICE on iPhone. Build: ONE workflow, everything at once, NO phases. He does not do phases (do not offer staged plans again).
- Wanted purely Fable 5 agents. RUN THEM IN PARALLEL (he was explicit; do not make them sequential).

**Build workflow saga (lessons)**
- Script: `workflows/scripts/nimbus3d-build-wf_d65351cd-f29.js` (run id wf_d65351cd-f29).
- It kept dying because the host Claude Code PROCESS restarted repeatedly (limits + crashes); background workflows run in-process, so they died with it. Verify a task is really running with `TaskOutput block=false` (a resumed Workflow gets a NEW task id; a dead one returns "No task found"). Don't claim "it's running" on faith. [[verify-claims-before-asserting]]
- ROOT CAUSE of "stalled, no tokens": the `model: 'fable'` override was NOT executing in this session (Fable unresponsive). Switching the module/integrate/review agents OFF fable to the inherited session model fixed it instantly (contracts stayed fable-cached and replayed free). If a workflow shows agents dispatched but burning zero tokens, suspect the model override.
- The per-agent Tokens column in /workflows only fills in AFTER an agent completes; in-progress agents show blank though they ARE spending tokens. Verify progress by FILES ON DISK, not that column.
- Final successful run: parallel, on the session model, all 12 agents done, 0 errors, ~1.26M tokens, ~30 min.

**Build result (honest, reviewer verdict = needs-significant-fixes; DOES NOT COMPILE yet)**
- Whole app on disk under `Nimbus3D/app/` (~40 Swift files + Metal shaders + Rust FFI + CI). See `app/MODULE_STATUS.md` (authoritative per-module REAL/PARTIAL/STUB table) and `app/README.md`.
- REAL Swift/Metal: SplatRender (Metal EWA renderer + GPU sort), Mesh (marching cubes + QEM + UV), HDRI (hand-rolled OpenEXR writer + Debevec merge), Export (glTF/GLB), Pipeline (actor orchestrator + UI), Capture (ARKit+LiDAR), Materials (POM shader + ambientCG substitution + CoreML classifier path), App shell + DI, CI YAML.
- THE ONE GATE: SplatEngine Rust/Brush trainer (`app/Sources/SplatEngine/rust/nimbus-splat-core/src/brush_glue.rs`) is written against UNVERIFIED Brush API names, and it is unknown whether brush-process/burn/wgpu even cross-compiles to aarch64-apple-ios (may pull egui/winit/rerun viewer deps). Top fixes are in the reviewer verdict (journal line 55 of the build run).
- Honest STUBS (not faked): MaterialDelighter (no model; writes `albedo_NOT_delit_passthrough.png`), classifier (returns .unknown until a model bundled), CapturedAlbedoBaker (flat placeholder), PLY->SPZ (deferred). Reviewer found ZERO faked functionality. Nothing device-verified (no macOS here).

**Repo + CI (live)**
- Private repo: https://github.com/bottlemaster-22/Nimbus3D (git root = `Nimbus3D/`, commit 07bce87, 118 files). gh authed as bottlemaster-22 (repo+workflow scopes).
- First CI run (29112027764) reached the Brush Rust build step and is actively cross-compiling (did NOT fail fast). Awaiting outcome. Note: duplicate inert `app/.github/` copy exists; clean up later.

**Next step**
- Get the Brush CI step outcome. If it fails: reconcile brush_glue.rs against a pinned real Brush revision + confirm iOS cross-compile (feature-gate out viewer deps). If it passes: the xcodebuild archive step then tests the Swift/Metal side.
- Still to do: Bottle "Install Nimbus3D" button (hand `BOTTLE_INTEGRATION.md` to the Bottle Claude session); fill the real `downloadURL` in `BOTTLE_HOOKS.md` after first green tagged build.

---

## 2026-07-10 : CI round 1 failed on `rfd`; forked Brush to fix iOS

**CI run 1 (29112027764) FAILED at the Brush Rust step.** Real error: `error: could not compile 'rfd' due to 12 previous errors` (E0277, dialog traits unimplemented). Root cause: Brush's `crates/rrfd` declares `rfd = "0.17"` for `cfg(all(not(wasm), not(android)))` — iOS is NOT excluded, but rfd has no iOS backend. Chain: brush-process -> brush-vfs -> rrfd -> rfd. No feature to disable it; even Brush's own `apps/brush-c` would hit this (nobody has built Brush for iOS).

**Fix = forked Brush.** Fork: https://github.com/bottlemaster-22/brush (main @ **de605e7**). Changed `crates/rrfd` to treat iOS like Android/wasm: excluded iOS from the `rfd` target-dep in Cargo.toml and added compile-only iOS stubs for pick_file/pick_directory/save_file in src/lib.rs (never called at runtime; training uses DataSource::Path). Fork working copy: `scratchpad/brush-fork` (shallow clone).

**Our repo:** `app/Sources/SplatEngine/rust/nimbus-splat-core/Cargo.toml` now pins `brush-process = { git = "https://github.com/bottlemaster-22/brush", rev = "de605e7862b315d6cd25ee5c94e0997ba6e52157" }` (was upstream branch=main). Pushed -> CI run 2 (29113334942) in progress.

**What CI run 2 tells us:** (a) THE BIG UNKNOWN — does the whole burn/wgpu/brush compute stack actually cross-compile to aarch64-apple-ios past rfd? (b) then our `brush_glue.rs` compiles last and will likely surface the API-name mismatches the reviewer flagged (create_process signature, config fields, ProcessMessage/TrainMessage variants). The real Brush API to reconcile against is readable in `scratchpad/brush-fork/crates/brush-process/src/` (create_process in lib.rs; config/, message/, train_stream/ modules; DataSource re-exported from brush-vfs).

**Next step**
- Read CI run 2 result. If deps compile + brush_glue errors: reconcile brush_glue.rs against the real Brush source in the fork clone. If another desktop-only dep fails: same fork-and-cfg pattern. If Rust core builds: on to the Swift xcodebuild archive step.

---

## 2026-07-10 : HARD RESET to the user's real vision (SMART LiDAR splatting, his own trainer)

**The reset.** He supplied consolidated context from his France sessions: he is building his OWN ARKit LiDAR capture app + OWN trainer, explicitly declined Brush/Nerfstudio/Postshot/LichtFeld, name "Nimbus3D" NOT final (do not bake in). So the Fable-built Brush-wrapper scaffold + brush fork + Nimbus3D repo from this session are SUPERSEDED (kept, not resumed). New rules: never Fable agents; Opus 5 for hard work, Sonnet 5 for easy; iPhone 17 Pro Max now; "if PC training wouldn't work, do it on the app." Wants 5-10 experimental features proposed FIRST (plain language), he picks, then build everything; UI must be usable.

**His capture/trainer code is NOT on this machine** (only the superseded scaffold; `RenderEngine` is a Blender/Cycles project, unrelated). Asked him where it lives or whether to build fresh around the verified export format. UNANSWERED.

**Research done** (8 parallel Opus agents, ~872k tokens, run wf_374a4b66-405; digest per agent in scratchpad/digest_1..8.md). Consensus headline: **raw ARKit VIO poses (no BA) are the number-one cause of soft edges / ghosting / bad far field** (~13 px misalignment vs sub-pixel need; arXiv 2608.21008 Aug 2026: ARKit 16.55 dB -> refined 17.48 -> GT 18.29). Fix poses FIRST or every trust/edge technique mislabels drift as sensor noise. Naive COLMAP triangulate-then-BA from ARKit prior made poses WORSE in 15/15 rooms; use structureless BA + loop closure + time-sliced submap pose graph.

**Verdicts on his ideas:** dual-path = right philosophy, wrong mechanism (one solver, three witnesses, not two reconstructions voting); 30cm clusters = wrong resolution + neighbour agreement != accuracy (use two scales: coarse bias, per-sample noise); smart edges = correct but 5 causes, pose dominant, and Apple's upsampled depth INVENTS edges (export native 256x192); smart background = two problems (far diffuse vs through-glass at infinity), window is a CAPTURE problem (bracket exposure); no-training pre-pass = his best idea but target is pose consistency not distances; slicing = better than framed, as a TIME-sliced drift absorber, scene fits 8GB unsliced, "train only seams" fixes 1 of 3 seam causes, overlapping splats add opacity (one owner per region); dynamic textures drop = agreed.

**10 features proposed, written to `Nimbus3D/PROPOSALS.md`** (full evidence + build order + honest measurement plan): F1 poses-first (submaps+loop closure+LiDAR-anchored structureless BA), F2 LiDAR free-space carving (biggest overlooked win; UNKNOWN not EMPTY for glass), F3 native-depth + edge band + WHERE/WHAT edge fusion, F4 AbsGS/Mip filters/SAD-GS/bimodal edges, F5 three-regime background (Prompt Depth Anything fits his exact input) + parallax routing + capture-time bracketing, F6 two-scale trust + plane-sweep second opinion + recalibrated ARKit confidence + structural trust, F7 slicing upgraded (doorway cuts, 7-number merge), F8 capture-side logging (time-offset scalar, end-of-session anchors, QC weights, exposure), F9 in-room QC + Fix-this AR + audio/haptic + on-path honest preview + A/B slider, F10 PC trains / phone previews + memory levers (SH 3->1 = 2.5-4x splats; 8GB not binding).

**Next step**
- Await his picks (or "all") AND the code-location answer. Then build with Opus/Sonnet agents in parallel, no phases.

---

## 2026-07-10 : GO. Mobile-first build launched (Opus/Sonnet, parallel)

**His answers:** F1-F6, F8, F9 yes; F7 optional; F10 accepted once clarified that the phone CAN train (budget-scaled) and PC is optional. Prefers mobile; no money; no cloud. No capture/trainer code exists (prior IPA never installed) -> build fresh. New asks: device-compat onboarding screen ("Why is my device incompatible?"), Blender add-on, PC Booster listener + minimal GUI + configurable save dir. Corrections: don't claim his scene fits 8GB (design measures/adapts); far-background flattening is a LiDAR-logic failure (F5).

**Actions:** moved `app/` (Brush wrapper) to `_superseded/app-brush-wrapper/`. Launched the build workflow: architect (Opus) -> 11 parallel modules (Opus for onboarding/capture/pre-pass/Metal trainer/smart losses/viewer/booster server; Sonnet for export/booster client/Blender/CI) -> integrator (Opus) -> reviewer (Opus). Pure Swift/Metal iOS at `ios/`, Python booster at `booster/`, add-on at `blender_addon/`, CI at `.github/`. Honesty contract enforced (REAL/PARTIAL/STUB, no faking).

**Next step:** on completion: read reviewer verdict + MODULE_STATUS, push, let CI build the IPA, fix what CI surfaces, hand Bottle the hooks.

**Build run 1 result (wf_f3ae4b97-25b):** PARTIAL. 10/14 agents failed on server-side API errors (500 / 529 Overloaded), ALL of them Opus; all 4 Sonnet agents succeeded. On disk and real: ios/Sources/Export (PLY/SPZ/GLB, ~2.5k lines), ios/Sources/Booster client (~2.7k lines), blender_addon (headless-tested on this PC with Blender 5.2), .github/workflows/ios.yml + BOTTLE_HOOKS.md (bundle id com.tombline.nimbus, target/scheme named "App", product name decoupled). Partial architect files: ios/project.yml, ios/Sources/Core/BrandConfig.swift. MISSING: Contracts.swift/docs, onboarding, capture, prepass, metal-trainer, smart-losses, viewer, booster-server, integrate, review. Fix: added an agentRetry wrapper (3 tries) to the script + told architect/modules to build on existing files; resuming.

**Build resumes 2-3 (diagnosis):** the architect NEVER completed (0 results; every attempt hit API 500/529 while Opus was overloaded), but the script did not abort on architect failure: it dispatched all 11 modules with null contracts (42 distinct agent keys in the journal). The 4 Sonnet modules re-ran and correctly preserved prior work. Stopped the run before the 7 contract-less Opus modules got far. FIX: architect now retries 5x with 60s-step backoff and the run THROWS if it still fails (no contract-less module builds). Also: always write the script with LF endings (a CRLF rewrite once broke the permission validator). Host process restarts have also killed runs twice; resume is always safe (cached).

**Build progress (2026-07-10, after abort-guard relaunch):** ARCHITECT COMPLETED and is cached (Contracts.swift, BrandConfig, docs/DATA_FORMAT.md, docs/BOOSTER_PROTOCOL.md, CONTRACTS.md). Core modules wrote partial real files before a usage-limit cut: Capture 10, booster/ 13, Onboarding 3, PrePass 2, Smart 2, Viewer 2; Trainer 0 (hardest, not yet written). Opus API 500/529 errors killed many module attempts (retry wrapper fired repeatedly). Resumed as task wvpseonxg; modules improve their existing files on re-run (monotonic progress). Local git checkpoint 9838d04 (209 files), NOT pushed (would trigger a doomed CI run until the trainer exists). NEXT on completion: reviewer verdict -> MODULE_STATUS -> push -> CI IPA -> Bottle hooks. If Opus stays flaky, option (owner must approve): allow Sonnet fallback on module retries.

---

## 2026-09-04 : The build is further along than the journal said. Only the DRIVERS were missing.

**Corrected picture.** A run on 2026-09-04 (14:18-14:29) landed a lot more than the previous journal entry recorded, then hit the session limit mid-trainer. Actual state verified on disk by file count and by grepping for declarations, not by trusting MODULE_STATUS.md:

| Module | Files | Reality |
|---|---|---|
| Core | 2 | Contracts.swift (2170 lines) + BrandConfig. REAL. |
| Onboarding | 10 | **Compile blocker FIXED**: OnboardingCopy.swift, OnboardingFormat.swift, DeviceReportStore.swift all now exist. |
| Capture | 22 | All the parts. `ARCaptureService` + `CaptureScreen` MISSING. |
| PrePass | 6 | All the parts. `PrePassPipeline` MISSING. |
| Trainer | 2 | TrainerShaders.metal = 2042 lines, **28 kernels, complete forward AND backward 3DGS** (preprocess, radix sort, tile ranges, rasterize fwd/bwd, SSIM, depth loss, regularizer, Adam splat/SH). TrainerGPULayouts.swift = 604 lines of struct layouts + TrainerBufferIndex. `MetalSplatTrainer` (the whole CPU side) MISSING. |
| Smart | 5 | Trust field, authority map, edge classifier, background model. |
| Viewer | 8 | Renderer + shaders + honesty direction field + camera path + library store. Nothing declares `SplatRenderer` conformance. |
| Export | 12 | REAL, verified 3 passes. |
| Booster (iOS) | 20 | REAL, verified 3 passes. |
| booster/ (PC) | 30 py | server 631 lines aiohttp, trainer ~2400 lines. `gui/` is an EMPTY package. |
| blender_addon | 10 | REAL, 43/43 tests on installed Blender 5.2.0. |

**The pattern:** every module wrote its internals but not its top-level driver, the one class that owns the parts and conforms to the protocol. Five drivers missing: `MetalSplatTrainer`, `ARCaptureService` (+ `CaptureScreen`), `PrePassPipeline`, the Viewer's `SplatRenderer` conformance, and the Booster GUI. `NimbusApp.swift` is also STALE: it still says Trainer's "Directory absent" and Onboarding is blocked, neither of which is true now.

**Launched** a focused finishing workflow (run `wf_8eca2052-692`, task `wdflz1ifv`, script `scratchpad/nimbus-finish.js`, verified LF-only, 0 control chars, `node --check` clean, status confirmed `running`): 5 driver agents in PARALLEL (all Opus, trainer/capture/prepass at high effort) -> integrate (Opus high) -> adversarial review (Opus high, schema'd). Prompts carry hard rules: improve never clobber, stay in your own directory (cross-module asks go to INTEGRATION_REQUESTS.md), grep every referenced symbol because CI is the only compiler, honest REAL/PARTIAL/STUB, no em dashes.

**Next step:** read the reviewer verdict, fix real blockers, then push and let CI produce the unsigned IPA, then hand Bottle the install hooks.

---

## 2026-09-04 : Drivers landed (7/7 agents, 0 errors). Audit found the app was a dead end after capture.

**Run `wf_8eca2052-692` completed clean:** 7 agents, 0 errors, 2.37M subagent tokens, 696 tool uses, ~74 min. All five drivers written. File counts after: Trainer 2 -> 11, Capture 22 -> 28, PrePass 6 -> 11, Viewer 8 -> 13, booster gui 0 -> 4 py. Repo is now ~47,700 lines across 111 Swift + 3 Metal files.

**Every mechanical check the reviewer could run without macOS PASSED:** no duplicate top-level type names; all 12 protocols in Contracts.swift have exactly one conforming type with every required member present; all 40 Metal entry points referenced from Swift exist AND all 40 declared entry points are referenced (perfect 1:1); `TrainerBind` matches the `[[buffer(n)]]` attributes of all 28 trainer kernels argument for argument; Viewer and Capture shader indices match; braces balance everywhere; no Swift 6-only construct, no iOS 18+ API; all ten source dirs and all three .metal files covered by project.yml.

**THE REAL FINDING (verified independently by grep before acting):** `PrePassService.run(bundle:at:)` had ZERO call sites and `SplatTrainer.train(...)` had ZERO call sites. The only pre-pass entry point called anywhere was `quickQCCard`, from CaptureScreen.swift:382. So the app could record a scan and show a quality card, then stop forever. `hasPrePass` could never become true, making the "Ready to build the 3D model" branch unreachable dead code, while every scan sat on "Recorded. Not checked over yet." promising a step that could not start. 4,100 lines of trainer plus the whole pre-pass were unreachable. The services WERE registered correctly in NimbusApp.swift; nothing invoked them.

**Fixed by hand:** `ios/Support/.gitkeep` (the dir was untracked and empty, and project.yml points INFOPLIST_FILE and CODE_SIGN_ENTITLEMENTS into it, so a fresh CI checkout would have died on a missing Info.plist for a reason unrelated to the Swift). Also corrected the false "Depends on nothing" comment on Sources/Core in project.yml: Contracts.swift borrows SHDegree, SplatCloud, ExportFormat from Export and BoosterJobStage from Booster, and extends SHDegree with Codable.

**Launched `wf_1f37af9b-954` (task `wkiaa84ix`, verified running):** 2 parallel Opus agents, then an adversarial audit. (1) build the processing flow: a new Sources/Pipeline coordinator driving prePass.run -> trainer.train with honest progress, resumability, real cancel, budget sized from device tier and measured scan size, plus the library buttons that make each `nextStep` sentence true and the Booster hand-off for scans too big for the phone. (2) verify the one claim nothing in the repo can prove: GLTFExporter.swift asserts it was checked against a RATIFIED Khronos glTF Gaussian-splat spec, and every attribute name and the COLOR_0 fallback in the .glb path is transcribed from it. Agent has web access and must check the Khronos registry itself, fix the code if wrong, and correct the comment if the extension is not actually ratified.

**Deliberately NOT pushed yet.** Pushing before the pipeline is reachable would burn CI on an app that cannot finish a scan.

---

## 2026-09-04 : Pipeline is reachable end to end. Audit says buildReady, with four real defects to fix.

**Run `wf_1f37af9b-954` completed clean** (3 agents, 0 errors, 611k tokens, ~34 min).

**The pipeline is now genuinely reachable, traced by hand by the auditor, not asserted:** Your scans tab (NimbusApp.swift:87, rendered :454) -> tap a scan with no model -> ScanProcessingScreen (ScanLibraryScreen.swift:94-98, second route from ScanReviewScreen.swift:152-153) -> three buttons (ScanProcessingScreen.swift:147/159/173) -> ScanProcessingCoordinator.start (:202, :235) -> prePass.run at :343 into the real PrePassPipeline (:250) -> trainer.train at :395 into the real MetalSplatTrainer (:87). Both services registered non-nil (NimbusApp.swift:66, :69). New `Sources/Pipeline` module; `optional: true` removed from all twelve source dirs in project.yml now that every one exists.

**glTF VERIFIED against the live Khronos registry** (this was the one claim nothing in the repo could prove). Checked `extensions/README.md`, the `KHR_gaussian_splatting` README and its JSON schema, plus the commit log (81762cc, 2026-09-03). Result: extension name correct, KHR_ prefix correct, listed under "Ratified Khronos Extensions", and EVERY field matches: mode POINTS(0), required colorSpace/kernel, ROTATION VEC4 (x,y,z,w), SCALE VEC3 with expf activation, OPACITY sigmoid, SH band layout (bandStartIndex 1:0/2:3/3:8, bandCoeffCount 1:3/2:5/3:7 matching restCoefficientCount 3/8/15), and the COLOR_0 fallback including the sRGB decode most implementations skip. ONE claim was FALSE: "ratified as of 2026-02" was the RELEASE CANDIDATE announcement (2026-02-03), not ratification. Comments corrected; no executable line changed.

**FOUR DEFECTS FOUND (now dispatched as `wf_2494865d-344`, task `wd5ivc3uo`, verified running):**
1. **Cancel re-entrancy (real).** `cancel()` sets `runTask = nil` and phase `.cancelled` synchronously, so `isRunning` goes false before anything has stopped. Swift cancellation is cooperative and the trainer only checks between GPU steps. User can press Stop then immediately Start: the old task's unwind clobbers the new run's phase, and `train()` resets `cancelRequested = false`, UN-cancelling the old loop. Single registered trainer instance, no re-entrancy guard, so two loops would share one set of GPU resources. The header comment claiming "nothing is left running behind a screen that says it stopped" is stronger than the code.
2. **Cross-scan model contamination (latent).** `runTraining` writes `trainer.finishedModel()` into this scan's folder without checking `model.scanID`. `latestModel` is never cleared between runs. Every current failure path throws so it cannot bite today, but any clean-exit-without-model path would write scan A's model into scan B.
3. **Format-version gate inconsistent.** `readSummary` refuses an unknown `formatVersion`; `readDetail` does not, and the coordinator uses `readDetail`. A future-version prepass reads as "not checked over" in the library but "reusable" in the coordinator, which feeds it to the trainer. DATA_FORMAT.md section 9 says refuse, do not guess.
4. **No route from capture to processing.** TabView has no selection binding and the capture report only dismisses itself. The whole flow is reachable only if you work out unaided that you must leave the Capture tab.

**Also dispatched:** wire the live training preview. `snapshot()`, `load(_ cloud:)` and `previewAvailable` are all written and have zero call sites, so the user watches a spinner while the model forms, which is what F9 existed to prevent.

**Checkpoint `1aa3082`** (296 files tracked, __pycache__ now gitignored). Still NOT pushed.

---

## 2026-09-04 : Four defects fixed. Then owner raised the bar: exhaustive audit, not three-agent workflows.

**The four audit blockers were fixed** (`wf_2494865d-344`, both fix agents succeeded; the verify agent died on a StructuredOutput retry cap, a tooling failure not a code one, and the schema was dropped in favour of plain text). Verified on disk by hand:
- **Cancel re-entrancy CLOSED.** `canStartNewRun` is a genuine three-way guard (`runTask == nil`, `!isStopping`, and asks the trainer `metal.isTraining`). `cancel()` now KEEPS runTask and sets `isStopping`, so the window where the screen said "stopped" while the GPU was live is gone. The agent added something not asked for and correct: a **generation counter**, so a cancel whose async hop lands after this run finished and a later one began refuses rather than stopping the wrong run. `MetalSplatTrainer` now throws `TrainerError.alreadyRunning` on a re-entrant `train()` instead of blindly resetting `cancelRequested`.
- **Cross-scan contamination CLOSED** at both ends: `guard model.scanID == bundle.scanID` before writing (coordinator:549), and `latestModel = nil` per run (trainer:128) alongside `exposureRecords` and the held-out list, which were also carrying over.
- **Format-version gate + capture-to-processing route** fixed in ScanLibraryStore and NimbusApp.
- **Live preview WIRED** via new `Sources/Pipeline/TrainingPreview.swift`: refreshes from `trainer.snapshot()` on `previewAvailable` (coordinator:523), with an explicit "a failed refresh is not a failed build" rule so a bad snapshot keeps the last good frame rather than killing the run.

**Checkpoint `1aa3082`** (296 files). Push path confirmed ready: remote `bottlemaster-22/Nimbus3D`, gh authenticated as bottlemaster-22, CI triggers on push to main, PR to main, or workflow_dispatch. Remote main still at the old Brush-wrapper commit aef06f2; 3 commits unpushed.

**OWNER DIRECTIVE (new, standing):** *"If you're running workflows, do many more agents. If you are deploying three agents in a workflow, forget the workflow and simply do it yourself."* Ultracode also enabled: optimise for the most exhaustive correct answer, token cost is not a constraint. Stopped the 3-agent verify run mid-flight and replaced it.

**Launched `wf_36e0a9ff-76e` (task `wpz630w5d`, verified running): the exhaustive audit.**
- **60 finders** = 50 Swift (11 modules crossed with 6 lenses: compile-blockers, strict-concurrency, framework/availability legality, algorithmic/mathematical correctness, faked-functionality, and Swift-to-Metal ABI) + 10 cross-cutting slices (Python correctness, Python protocol agreement, Python trainer maths vs the iOS versions of the same algorithms, Blender add-on vs the Swift writer itself rather than its own fixtures, build/CI on a fresh checkout, DATA_FORMAT agreement across all six readers/writers, brand rename-in-one-place, every user-facing string as a body of writing, and cross-module contract seams).
- **Three-lens adversarial verification** on every blocker/major finding: a refuter checking the code says what the finding claims, one checking a real user can actually reach it, and one judging against THIS project's rules (language mode 5.0, targeted not complete concurrency, labelled STUBs are legitimate). Majority rules, default-to-refuted, because a false finding sends a repair agent to break working code.
- **Repair partitioned by owning directory** so no two agents can touch the same file.
- **Five critics**: final compile gate over the parallel edits, completeness critic (what could this audit's structure not see?), first-run simulation on a phone, second-scan singleton state survival, and contract drift + draining INTEGRATION_REQUESTS.md.

Sizes audited: Trainer 9193 lines, Capture 8988, PrePass 7463, Viewer 6414, Smart 4148, Onboarding 3403, Booster 2728, Export 2688, Core 2432, Pipeline 2308, App 623; Python 10196; Blender 1923.

---

## 2026-09-05 : IT BUILDS. Unsigned IPA produced and verified.

**Owner pushback that changed the approach:** *"When am I gonna have a base app cause currently I have nothing"* and, earlier, *"If you're running workflows, do many more agents. If you are deploying three agents in a workflow, forget the workflow and simply do it yourself."* Both correct. I had been running audits as a PROXY for a compiler when the real compiler was one push away. The exhaustive 316-agent audit (`wf_36e0a9ff-76e`) died on the session limit with 19/316 done, and its 24 findings were reported as "refuted" only because all their verifiers had died (`survived 0/0`) - a bug in my own workflow logic, since verifier death is not refutation.

**Pushed and let CI be the compiler. Four rounds, about 40 minutes.**

| Round | Run | Result |
|---|---|---|
| 1 | 33972909584 | 25 errors, 9 files |
| 2 | 33973151718 | 1 error |
| 3 | 33973240389 | 1 error (new, previously masked) |
| 4 | 33973412814 | **SUCCESS** |

The PC Booster job (`python -m compileall` + pyflakes) passed on ALL FOUR rounds, untouched.

**What CI found that no amount of grepping would have:**
1. **`Foundation.abs` is C's `abs(Int32) -> Int32`**, not Swift's generic `abs`. 9 occurrences in PrePassMath poisoned 12 of the 25 errors. Needs real overload resolution to see.
2. **Two expressions the type checker refused to finish**: the DOS timestamp packing (ZipWriter) and the Catmull-Rom spline (PreviewCameraPathBuilder). Splitting the spline into 4 SIMD terms was NOT enough (round 2 still failed on it); it took collecting the basis into one plain `Float` coefficient per control point so every vector op has exactly one overload. Verified algebraically identical over 20k random inputs: max diff 7.1e-14.
3. **Actor isolation**: `CaptureAnchorRecorder.write` and `BoosterDiscovery.hostString` are pure functions on their arguments; their types' `@MainActor` protects accumulated state neither touches. Now `nonisolated`.
4. **`SmartLossSettings` never initialized 3 of its own properties** (`geometricEdgeSharpenBoost`, `textureFlattenWeight`, `minimumAuthorityForDepth`), and `minimumAuthorityForDepth` IS read by the trainer twice. Masked in rounds 1-2 by earlier errors in the module.
5. Missing Combine imports (found and fixed pre-push), `Data.append` with no `[UInt8]` overload, `withUnsafeBytes` binding to the instance method inside `extension Data`, a public init exposing internal `TrainerTuning`, an actor property needing explicit capture, and `showsUnavailable` used but never declared.

**Bug-class sweep instead of one-per-round:** wrote a scanner for "non-optional stored property with no default, never assigned in a designated init". It was WRONG three times (470 hits counting optionals; 342 with broken multi-line-init body capture; 54 demanding `self.x =` when bare `x =` is legal) and then VACUOUSLY clean (Windows Python could not see the `/tmp` test path). Validated against the pre-fix file until it found exactly the 3 real bugs, then confirmed the rest of ios/Sources is clean. Scanner kept at `scratchpad/initscan.py`.

**THE ARTIFACT, verified not assumed:**
- `Payload/Nimbus3D.app/Nimbus3D` - Mach-O `0xfeedfacf` MH_MAGIC_64, cputype arm64, 3,287,720 bytes
- `Payload/Nimbus3D.app/default.metallib` - 299,112 bytes, so all three .metal files compiled AND linked
- Info.plist: `com.tombline.nimbus`, MinimumOSVersion 17.0, UIDeviceFamily [1], UIRequiredDeviceCapabilities [arkit, metal, arm64], NSBonjourServices [_nimbusboost._tcp], all three usage strings present in plain language, and the NBBrand* keys proving the single brand block flowed through.
- Built with Xcode 16F6 / iOS SDK 18.5 on macos-15.

**PR:** https://github.com/bottlemaster-22/Nimbus3D/pull/1 (branch `build/first-ci`, not merged to main).

**Next step:** sign via Bottle and install on the iPhone 17 Pro Max. THEN the real unknowns start, because compiling is not running: nothing in this app has ever executed. Expect the first real failures at ARSession start, Metal pipeline construction (TrainerGPULayouts.verify() throws on any Swift/Metal stride disagreement), and the first training loop.

---

## 2026-09-05 : First hardware feedback. Four real bugs found, all fixed, CI green first try.

**The owner installed v0.1.0 and reported three things.** This is the first real feedback this project has ever had, and it was worth more than every audit combined.

1. *"you have to move so super slowly or it yells at you for being blurry... my hands shake a lot"*
2. *"the preview comes out on it's side as if it were rotated left on the (z?) axis"*
3. *"it looks REALLY sparse, I did a 3 min scan of my room and it came out looking like nothing"*

He also asked, unprompted and correctly reasoned: *"Can you run the camera at 240 fps like my camera supports?"*

**Two workflows, 32 diagnosis agents.** `wf_aef4e7b7-0cc` (13 agents: 7 rotation hypotheses, 6 blur) and `wf_86a737ab-a43` (14 agents, one per stage where a splat can die). Both lost their final agent to the session limit; everything else landed.

### THE SPARSE SCAN: four bugs, in order of damage

**1. DENSIFICATION NEVER FIRED ONCE.** `absGradThreshold = 0.0006` was copied from the reference 3DGS implementation, which multiplies its gradient by `0.5 * width` BEFORE storing, making it an NDC number. This rasteriser accumulates `length(dLdMean2D)` raw, in PIXELS (`TrainerShaders.metal` builds delta as `mean2D - pixelCenter`), which at 720 px is smaller by a factor of a few hundred. Nothing ever cleared the bar. Zero splats created, split or cloned, ever. **The stage whose entire job is growing a sparse seed into a dense model was a silent no-op.** Fixed: gate on `> 0` and let the ranked truncation to budget do the cutting (scale-free, cannot break on units again). Threshold constant dropped to 1e-9 with the units documented.

**2. THE THERMAL CAP WAS A ONE-WAY RATCHET THAT ATE LIVE GEOMETRY.** `degradeForHeat` cut towards `min(current.splatCap, currentSplatCount) * 0.75`. Live count is always below the ceiling early (seeder aims for splatCap/2), so `min()` picked the LIVE count and 0.75 of it landed BELOW the existing population. `applyBudgetChange` then TRIMMED real Gaussians and `resizeSplatCapacity` rebuilt the GPU buffer smaller: geometry physically deleted, not merely disallowed. Then `headroom = splatCap - splatCount == 0` switched densification off permanently. One warm phone in minute one and the model could never grow again. Fixed: cut the CEILING, never below the live count, floor is now `ceiling.splatCap / 4` (scene-derived) instead of a flat 20,000.

**3. THE PRUNE SCHEDULE WAS DEAD CODE.** `pruneStartFraction` (0.15) and `pruneEndFraction` (0.80) were declared, defaulted, assigned, and read NOWHERE (repo-wide grep). Pruning rode the densify interval: every 100 iterations from 100 to the end, 29 passes on a 3,000-iteration run instead of ~10, including before convergence and during late opacity binarization (deleting splats only briefly pushed toward zero). Fixed with an `allowPrune` window. Non-finite pruning still runs every pass: hygiene, not judgement.

**4. THE TRUST GATE REJECTED ~100% OF A SHAKY SCAN.** `trustedWeight = 0.5` needs measured sigma under ~2 cm even at perfect confidence. That sigma comes from cross-frame residuals so it absorbs POSE error, and `PrePassPipeline.swift:19-20` says so itself: "A 2 cm pose error reads as 2 cm of sensor noise everywhere." On a first handheld scan 2-4 cm is ordinary, so every seed became a stretched translucent blob instead of a solid disc. Fixed: relative to the scan (better half by trust) with a low absolute floor.

### ROTATION
`rot-display-transform` **confirmed cause**. Critically, `rot-trainer-frame` was **RULED OUT**, so the model was never rotated and exports were always fine: only the preview. Also ruled out: colmap writer, viewer camera, preview path, metal viewport. This mattered enormously, because "fixing" it in two places would have looked right in the preview while corrupting every .ply and .glb.

### 240 FPS
`blur-framerate` **ruled out**, as suspected: that is an AVFoundation slow-motion path, not something ARKit offers alongside sceneDepth and tracking. But his reasoning was right from the wrong end. At 240 fps a hand-held phone moves ~2 mm between frames, which is no new parallax; splatting wants viewpoint DIVERSITY, not temporal density. The lever he actually wanted is SHORTER EXPOSURE at a normal frame rate, which is the real cure for tremor and is now in the blur fix.

### RESULT
Commit `85e3f09`, ~20 files. **CI run 33988092964 GREEN ON THE FIRST ATTEMPT** despite the size of the change. Artifact verified by reading it: arm64 Mach-O 3,305,248 bytes, default.metallib 299,112 bytes, com.tombline.nimbus, minOS 17.0. Published as release **v0.1.1** and handed to the owner.

**Next step:** he scans again. That is the only test that matters. If it is still sparse, the instrumentation agent's splat census (frames -> keyframes -> seeds -> after densify -> after prune -> after carve -> final) is the next thing to build, because we are still inferring where geometry goes instead of measuring it. Also outstanding: the `gate` and `converge` agents never ran (session limit), so nothing has re-audited these fixes as a set.

---

## 2026-09-08 : 247s to 71s of training, and the accounting error that made it look better than it is

_Appended at the bottom to match this file's stated convention ("newest entry at the bottom"), not the session-continuity skill's newest-on-top default._

**Since the last entry (2026-09-05):** three days of nothing but performance and quality work on the on-device trainer, none of it journalled until now. That gap is why a context compaction hurt so much: the entire cost model, every failed experiment, and the reason for each revert lived only in conversation. This entry closes it.

### THE NUMBER THAT MATTERS, AND THE MISTAKE IN IT

`scratchpad/census.py` prints a line labelled `wall`. It is computed from `train_census.json`'s `startedAt` to `finishedAt`. **That window covers TRAINING ONLY.** The pre-pass has its own census file and is printed separately, below it. Total processing is the sum, and nothing prints the sum.

I quoted the owner "86s to 71s" this session. That was wrong: 86 was a TOTAL and 71 was TRAINING ONLY. Corrected, measured, same scan (`scan_20260906_164840`), 3000 iterations:

| build | pre-pass | training | **total** | splats | held-out PSNR |
|---|---|---|---|---|---|
| 102 | not timed | 187 | ? | 152,335 | 16.78 |
| 130 | not timed | 80 | ? | 151,761 | 16.34 |
| 144 | 21.7 | 68 | 89.7 | 153,164 | 15.94 |
| 156 | 20.9 | 65 | **85.9** | 150,554 | 16.00 |
| 164 | 21.3 | 64 | 85.3 | 158,220 | 14.53 (half-conic regression, reverted) |
| **172** | **17.8** | **71** | **88.8** | **299,883** | **17.30** |

So training did climb 65 to 71, because the model now carries twice the splats. Pre-pass fell 20.9 to 17.8. **Total went 85.9 to 88.8: 2.9 s worse.** In exchange: +99% splats and +1.30 dB, against a measured noise band of +/-0.4 dB. A good trade, but it is a trade, and the headline "71s" was never a total.

### VERIFIED (each with its proof)

- **Trainer 247 s to 65 s of training time**, six changes, each measured on the same scan. Full table and the failed list live in the `likova-trainer-perf` memory file, which is the authoritative copy.
- **The pattern held 6 for 6 against 0 for 5:** changes that REDUCE the quantity of work paid; changes that keep the same work and rearrange it never did.
- **Binarization was destroying half the model.** `binarizeLastFraction` 0.2 starts the opacity ramp at 80% of the run while the growth window closes at 85%, so roughly 90,000 splats were driven to zero and pruned after densification could no longer replace them. Set to 0 as a null test in build 172: splats 150,554 to 299,883, PSNR 16.00 to 17.30. Proof: build 172 census.
- **Seeding's 15.9 s was mostly file writing, not the per-sample loop.** Two prior optimisations of that loop moved 16.4 to 16.38 to 15.89, which should have been the clue. `Data.appendUInt32LE` was four one-byte `Data.append` calls; the PLY write is ~15M values, so 60M appends where 15M do. Plus `points3D.txt`, a `String(format:)` loop over 888,951 points with ZERO consumers in the app, now off by default (`PrePassPipeline.writeRefinedColmapModel = false`). Proof: seeding 15.9 to 12.93, pre-pass 21.3 to 17.8.
- **Thermals are not the constraint at this length.** `nominal` for 99.4% of a 71 s run, peak `nominal`. Proof: build 172 thermals block.
- **`tools/splat_to_blender.py` works.** Converts any 3DGS .ply into a Blender-readable coloured point cloud (INRIA `0.5 + 0.282095*dc`, sigmoid on the opacity logit, RDF to Z-up). Verified on the 427,710-splat Scaniverse capture: 427,550 kept, mean sampled colour 120/108/101, so mid-range rather than clipped.
- **Build 172 is green and published** on `build/first-ci`.

### ASSUMED / UNVERIFIED

- **Splat SIZE is the whole Scaniverse gap.** The reasoning is mechanical and consistent (splats only shrink by SPLITTING; clone copies at the same size; build 172 fired 190 splits against 152,801 clones; our median splat is 13.2 mm against their 3.4 mm) but no build has yet changed the routing, so this is inference, not measurement.
- **The 300,000 cap is now the binding constraint.** 299,883 final strongly implies it, but no run has been done with a higher cap.
- **Raising SH degree 1 to 3 is worth its cost.** Scaniverse ships degree 3. Untested here, and it is 3x the colour coefficients through every gradient.

### DECISIONS & RATIONALE

- **Binarization stays off**, pending one confirming run. It was costing half the model to save file size we are not short of.
- **Do not ship the split/clone routing change alone.** Routing split on SCREEN radius instead of world scale raises the split rate, and against the CURRENT one-axis shrink that makes tile count worse, which is the opposite of the point. It ships together with three-axis split geometry or not at all.
- **Journal at the bottom, not the top.** This file has said "newest entry at the bottom" since day one and eight entries follow that. Consistency beats the skill's default.
- **No workflow this session.** The owner was at 78% of the 5-hour limit and said so. The work is being done inline instead.

### OPEN THREADS (priority order)

1. **Split/clone routing + three-axis split geometry.** The direct attack on 13.2 mm vs 3.4 mm. Must ship as one change.
2. **Raise the 300,000 splat cap** now that it binds, and measure memory and energy at the new count on the A19 Pro.
3. **The GPU timing buckets are blind.** In build 172 `gpuForward`, `gpuLosses`, `gpuBackward` and `gpuOptimiser` all read exactly 0.00 while `gpuSort` reads 18.81 ms/iter and `gpuScan` 1.83, summing to `gpuBusy` 20.69 exactly. That is not a sort regression, it is every stage in that command buffer being credited to the sort bucket. Until this is fixed we cannot see where GPU time goes.
4. **SH degree 1 to 3.**
5. **Repo is TEMPORARILY PUBLIC** for free CI minutes. Flip back with `gh repo edit bottlemaster-22/Nimbus3D --visibility private` as soon as billing allows. Owner: "that's how our hard work gets scraped."
6. Tag-gated CI builds, offered and never actioned, to stop burning Actions minutes on every push.

### NEXT ACTION

Fix the timing buckets (item 3) before any further GPU work, because every remaining speed decision is currently being made blind.

### MAP

- `ios/Sources/Trainer/TrainerShaders.metal` : all Metal kernels. `TrainerSplatRaster` is the 40-byte compact record the hot loops read. The alpha-threshold tile footprint in `trainer_preprocess` was the single biggest speed win.
- `ios/Sources/Trainer/MetalSplatTrainer.swift` : the loop, the command buffers, `timings` reset per run.
- `ios/Sources/Trainer/TrainerSupport.swift` : `binarizeLastFraction`, `splitScaleFraction` (0.004), `snapshotIntervalIterations` (200).
- `ios/Sources/PrePass/PrePassPipeline.swift` : `writeRefinedColmapModel` (now false).
- `ios/Sources/Export/BinaryIO.swift` : `appendUInt32LE`, the bulk-append fix.
- `ios/Sources/Core/Contracts.swift:1820` : iteration count per tier.
- `tools/splat_to_blender.py` : 3DGS .ply to Blender-readable coloured points.
- `scratchpad/census.py` : reads the latest diagnostics. NOT in the repo, lives in the session scratchpad.

### GOTCHAS

- **`wall` in census.py is TRAINING ONLY.** Add the pre-pass line to it before quoting a processing time. This caused a wrong claim to the owner today.
- **Inside an extension on `Data`, bare `withUnsafeBytes` binds to Data's own instance method.** Build 170 failed on exactly this. It must be written `Swift.withUnsafeBytes`.
- **A SIMD-group reduction can gate work inside a threadgroup-uniform loop; it can never decide how many times that loop runs.** Shipped as a model-corrupting bug in builds 132 to 138.
- **`MTL_FAST_MATH: YES` implies `-ffinite-math-only`,** so `isfinite()` folds to constant true. Use a bitwise exponent test.
- **`simd_sum`/`simd_max` are Apple GPU family 7+.** Gate behind a function constant; A12 devices reach this code.
- **Half precision on the conic cost 2 dB.** Reverted. Do not retry.
- **The diagnostics arrive through the Booster**, not by hunting the filesystem.
- CI is the only compiler available; there is no Xcode on the owner's Windows machine.

---

## 2026-09-08 (later) : Eleven changes in one build, mined from seven workflows that had been left unused

**Since the last entry:** the owner said, plainly and for about the sixth time, that shipping one change per build is a waste of his time, money and usage, and that he would rather ship a batch and bisect if something breaks. He also ruled out new workflows for this session (usage at 78%) and pointed out that seven workflow runs from this session were sitting on findings nobody had acted on. He was right.

`.claude`-less salvage route, for the next time this is needed: the workflow journals live under `~/.claude/projects/<project>/<session>/subagents/workflows/wf_*/journal.jsonl`, one JSON object per line, `{type: "started"|"result", agentId, key, result}`. The `result` objects with a `findings` array are the research agents; the ones without are the verifiers, carrying `kept` and `killed`. `wf_5567f899-6d6` alone held **106 findings, 99 of which survived adversarial verification.**

### WHAT SHIPPED, and why each one

Every change is a single named constant with its reasoning written on it, specifically so a bad result is a constant flip and not a diff read.

**The splat size gap** (ours 13.2 mm, Scaniverse 3.4 mm on the same room):

- `splitScreenRadiusPx = 10`. Split also fires on screen radius now. The world-scale test needed a 21.6 mm splat in a model whose p99 is 3.85 cm and whose single largest Gaussian is 7.86 cm, so it selected almost nothing: 190 splits against 152,801 clones. Splats only ever shrink by splitting.
- `splitShrinkAllAxes = true`. Divides the whole scale vector by 0.8*N, which for N=2 is 1.6 and is exactly our existing `splitShrink`, with the offset sampled from the parent's covariance instead of stepped along one axis. This MUST move with the criterion above: the one-axis version makes two children cover 1.25x the parent's screen area, so raising the split rate against it increases tile count.
- `pruneOpacity` 0.005 to 0.02. It sat at minAlpha (0.00392), so a Gaussian had to stop being drawn before it was eligible for recycling. An exported run measured 50.1% of the population below the render threshold: invisible, ungraded (Adam gates on visibleFlag) and un-prunable at once.

**The shortened schedule:**

- `lateLRFraction = 0.1`. Scale, rotation, opacity and SH now decay across the run. Only position did, because that is what the reference does at 30,000 iterations; we run 3,000.
- `shDCLRMultiplier = 4`. Sparse visibility-masked Adam gives a typical splat ~450 steps; at 0.0025 that is at most 0.32 of the colour range against a measured RMSE of 0.117.

**The initialisation:**

- `seedTarget` fills the cap instead of half of it. It was discarding 84% of 888,951 MEASURED LiDAR points by arbitrary uniform stride, then spending 600 iterations cloning them back.
- `thin` now grows the two in-plane axes by sqrt(dropped ratio). Every seed radius is cut from the FULL set's spacing, so thinning left discs sized for a density that no longer existed. The fallback depth seeder in the same file carries a comment saying this was fixed there. It was never fixed on the path we use.

**The prior:** `discPriorWeight` 0.01 to 0.001. It produced a scale gradient of order 0.024 per Gaussian per iteration against a photometric contribution of 2.5e-5 to 7.5e-5, so shape was set by the prior and the photographs were only allowed to nudge it.

**Measurement, without which three of the above are unreadable:**

- Held-out PSNR was `min` across slices, reporting the WORST part of the scene as the model's score.
- `evaluateHeldOut` summed MSE and converted once at the end, which by Jensen's inequality is always the lower number and matches no published figure.
- `heldOutFraction` 0.05 to 0.10: six held-out frames became twelve. Six is why the noise band is +/-0.4 dB.
- Command buffer B was labelled "the tile sort" while holding the sort, forward raster, losses, backward raster and optimiser. That is the whole explanation for build 172's `gpuSort` 18.81 ms with the other four buckets at exactly 0.00. New `gpuStep` bucket.

**Speed, both pure deletions:**

- `trainer_radix_scatter` had a dead 16 KB threadgroup array, written and read back by the same thread, saturating half the 32 KB budget for nothing.
- Both rasterisers now reject with a compare instead of an `exp()`. `alpha = min(0.99, opacity*exp(power)) < minAlpha` is exactly `power < log(minAlpha/opacity)`; the cutoff is computed once per splat in `trainer_preprocess` and rides in `pad0`, a field the 40-byte record already zeroed. Biased down 0.01 so the cheap test can only let through what the exact test would keep, and the exact test still runs. Kills roughly four exp evaluations in five.

### VERIFIED

- **Build 180 is green and published.** Builds 176 and 178 preceded it: 176 failed the trapping-conversion gate, 178 was cancelled by my own follow-up push.

- `tools/trapconv.py` caught `Int((Float(cap) * ...).rounded())` in the new `seedTarget` and was right to: nothing stopped a caller passing NaN. Fixed with an isFinite check and a 0.05...1 clamp before the multiply, then allowlisted with that reasoning. That gate exists because of a crash the owner reported as "crashing quite a bit, even with RAM free".
- `tools/deadwire.py` passes.

### ASSUMED / UNVERIFIED

**All eleven changes.** None has been run on the device. The batch is deliberately large because the owner asked for it and accepted bisecting.

### GOTCHAS

- **Build 178 shows as `cancelled`.** That is my own follow-up push superseding it, not a failure.
- A workflow journal's `result` for a verify agent has `kept`/`killed`, not `findings`; filtering on `findings` silently drops every verifier.

### WHAT I EXPECT TO GO WRONG, written before the run rather than after

1. Three-axis shrink gives up the SAD-GS property the densifier header argues for: a disc that was the right size across a flat measured surface no longer is, and most of this scene is flat measured surface. If geometry gets noisier on walls, `splitShrinkAllAxes = false` first.
2. Filling the cap with seeds costs roughly 4 s and makes the first few hundred iterations carry 300k splats.
3. A genuinely live 300k population costs backward-raster time that a half-dead one did not.
4. Weakening the disc prior costs surface normals, which the mesh export depends on.

### NEXT ACTION

Read the census when the owner tests: splat count, held-out PSNR, `addedBySplit`, clone/split ratio, `gpuStep`, and the pre-pass total. `addedBySplit` is the one that says whether the headline change did anything at all.

---

## 2026-09-08 (night) : An offline copy of the rasteriser, and six research findings put to the test

**Since the last entry:** the owner cannot test again until tomorrow and asked for proof rather than guesses. So the work moved off the device entirely. `tools/offline/` is now a numpy reimplementation of the trainer's hot path, run against the model, poses and intrinsics already sitting in `Documents/LiKOVA/Scans/diagnostics/`.

### THE HARNESS, AND WHY IT CAN BE TRUSTED

- `project.py` : `trainer_preprocess`. 3D covariance from log-scale and quaternion, EWA projection, the Mip-Splatting 2D low-pass and its compensation, the alpha-threshold tile box, tile counts.
- `raster.py` : the rasteriser inner loop, front-to-back, on a sample of tiles. Exposes `geometry()` and `composite()`.
- `backward.py` : what the backward pass walks under each of the four possible gating schemes.

**Validation: on frame 4 it computes 1,059,741 tile instances against the census's recorded `peakTileInstances` of 1,058,989. 0.07 per cent.** Two traps cost an hour and are worth recording: the PLY is written in RDF while the trainer works in RUB, so BOTH the position and the quaternion's imaginary part need y and z negated; and `refinedPoses` in `prepass_result.json` is a dict keyed by frame index as a STRING, not a list.

### SIX FINDINGS TESTED. TWO SURVIVED.

**1. Split on screen radius: WRONG, and shipped wrong in build 182.** The research put the median projected half-extent at 4.8 px, so `splitScreenRadiusPx = 10` was meant to select the top few per cent. Measured, the median of the statistic the code actually reads is **40 to 49 px**, and 10 px selects **99.18 per cent** of the drawn population. 16 px selects 96. 48 px still selects 52. It cannot be rescued by moving the number: `maxRadiusPxBits` is a MAX over every view in the interval and the 3-sigma radius scales with 1/z, so one close-up frame sets it, and the median CLIMBS as more views are sampled. 4.2 per cent of drawn Gaussians record a max radius wider than the whole 720 px frame. Mean alpha extent is no better: 8 px selects 99.46 per cent, 24 px selects 47.89, no knee anywhere. Replaced with `splitShareOfGrowth = 0.2`, a RATIO of the already-ranked candidate list. A share has no unit to be wrong about.

**2. `if (power > 0) continue;` is provably dead.** 100.00 per cent of 7,719,936 pairs pass it. `power` equals `-(0.5/det) * (quadratic form of the adjugate of Sigma2D)`, and the adjugate of a positive definite 2x2 is positive definite, so it cannot be positive wherever a splat is emitted at all. A compare and a branch on every pair, in a loop entered ~294 million times an iteration in each of two kernels. Removed.

**3. "Early termination never fires, the model is a fog": WRONG.** It fires on **93.71 per cent** of pixels, median final transmittance 0.0000. The finding reasoned from a median peak alpha of 0.0038 taken over the WHOLE population including the invisible half, while the splats that reach a pixel are nowhere near median: a typical pixel gets 296 alpha-passing contributions. The measurement it rested on also predates binarization being switched off. An entropy prior and a scale-reset schedule were the proposed fix; both would have been a day spent on a problem that does not exist.

**4. The backward's traversal is at 1.12x the theoretical ideal.** Pairs walked on 40 tiles of frame 4: forward 4,381,243, ideal per-pixel backward 4,381,243, **SIMD-group gate as shipped 4,912,416**, threadgroup gate 5,487,616, no gate 7,719,936. So the backward walks only 12 per cent more than the forward and costs 3.6x. The cost is per-pair work, the twelve gradients and their atomics, NOT traversal. Three separate findings proposing smarter traversal are dead; the one blaming the atomics is right. It also explains why the SIMD-reduction attempt came out 6 per cent slower: right target, wrong instrument.

**5. `filter2DVariance` contributes 0.2 per cent of tile cost** across its whole plausible range (0.0 to 0.5). Our Gaussians are tens of pixels wide, so half a pixel of blur is nothing. Dead lever.

**6. The `pad0` exp cutoff: right, but I overstated it.** I wrote "roughly four out of five" exp evaluations killed. Measured **60.76 per cent**. Still the largest single item in that loop.

### WHAT IT ALL POINTS AT

Per pixel on frame 4: **754 pairs evaluated, 296 clear the alpha test, 178 genuinely composited.** 3.54 tile instances per splat on average, 10.35 on a busy frame. Every one of those numbers is set by how big the Gaussians are on screen, and the median largest axis is 19.6 mm against Scaniverse's 3.4 mm on the same room.

**And splat size has always equalled SEED size,** because clone copies the parent's size and split was firing 190 times against 152,801 clones. That is why the model has never gone finer than LiDAR sample spacing. With split now routed by share, this is the first build that can.

### DECISIONS

- **Seed spacing compensation OFF** (`seedSpacingCompensation = 0`). Geometrically correct in isolation, but it took the median from 13.2 mm to 19.6 mm and tile instances per splat from 2.55 to 3.54, and it amplifies as the seed count falls (2.43x at fillFraction 0.5 against the 1.72x build 182 ran). The whole gap being chased is that our Gaussians are too big.
- **Do not optimise backward traversal.** Measured at 1.12x optimal.
- **Do not build the entropy prior / scale reset.** The problem it targets does not exist in this model.
- **No risky densification-statistics subsampling for now.** It is worth about 5 per cent of `gpuStep` and it degrades the AbsGS score and the visibility test, and quality is currently the binding constraint.

### VERIFIED

Builds 184, 186 and 188 all green and published. `trapconv` and `deadwire` pass. The harness reproduces the census to 0.07 per cent.

### ASSUMED / UNVERIFIED

Everything about what these changes do to a real run. Nothing since build 182 has been tested on the device.

### NEXT ACTION

The owner tests build 188. Read `addedBySplit` and the clone/split ratio first: with `splitShareOfGrowth = 0.2` and a restored growth budget, the ratio should land near 80/20, and the median splat size should fall below the seed spacing for the first time.

### GOTCHAS

- The PLY is RDF, the trainer is RUB: negate y and z on BOTH position and quaternion imaginary part when reading a model back.
- `refinedPoses` is a dict keyed by a STRING frame index.
- `maxRadiusPxBits` is a max over views and scales with 1/z. It is useless as a fixed threshold and its median grows with the number of views sampled.
- A statistic's median measured over the whole population says nothing about the splats that actually reach a pixel.

---

## 2026-09-08 (late) : Relocation is the mechanism that matters, and it was starved, mis-shaped, and then mis-fixed

**Since the last entry:** the harness answered the "what is wrong" question (the size distribution is 1.8x wide against a good model's 8.9x). This entry is about finding the mechanism that can widen it, which turned out not to be the one I had been changing.

### RELOCATION IS A SPLIT, AND IT IS THE ONLY ONE RUNNING FOR 80% OF THE RUN

Build 182's densify counters, summed over 29 passes:

| counter | value |
|---|---|
| addedBySplit | 6,836 |
| addedByClone | 64 |
| **relocated** | **21,000** |

Read the relocation code and it is a split: it takes a Gaussian contributing nothing, shrinks the TARGET, and places the pair at +/- an offset from the target's centre with an opacity correction. It uses a dead Gaussian's slot instead of a new one. And because the population reaches its cap around iteration 600, growth closes and relocation is the ONLY thing still able to make a Gaussian smaller for the remaining 80 per cent of the run.

**So every split fix so far had gone onto the branch that barely fires.** `splitGeometry` is now one function with two callers.

### THE DONOR POOL WAS EMPTY BY CONSTRUCTION, AND I EMPTIED IT

`relocationDonorOpacity` and `pruneOpacity` were both 0.02. A Gaussian below `pruneOpacity` is deleted at the end of the same pass, so the band BETWEEN the two thresholds IS the donor pool, and equal thresholds make it empty. Measured on build 182's model: **113 Gaussians of 299,209 (0.04%)** below 0.02. Every donor the census recorded came from the other half of the test, `visAccum <= 0` ("not seen this interval"), giving 835 a pass against an allowance of 15,000. Donors were the binding limit by 18x.

I raised `pruneOpacity` from 0.005 to 0.02 two builds earlier, for a good reason, without checking who else read it. Donor threshold is 0.05 now (22,005 Gaussians, 7.35%, just clearing the 5% allowance), and **the invariant is written on the constant**.

### AND THEN I BROKE THE OPACITY CORRECTION

Unifying `splitGeometry` handed relocation three-axis shrink. But relocation corrects opacity with `o_new = 1 - sqrt(1 - o_old)`, an identity derived for two Gaussians COVERING WHAT THE PARENT COVERED. Three-axis shrink leaves the pair at about half the parent's volume, so the correction dims a region it has also made smaller. At 15,000 relocations a pass that is the whole model losing brightness and coverage at once.

`preserveCoverage: Bool` now makes each caller state its intent. Growth shrinks three axes (reference behaviour, no correction, densification fills the gap). Relocation shrinks one (coverage-preserving, which is what its correction assumes).

### THE PATTERN OF THE NIGHT, WORTH NAMING

Three regressions, all from changes that were individually defensible and interacted:

1. seed fill-the-cap x seed resize
2. `pruneOpacity` x `relocationDonorOpacity`
3. unified split geometry x the opacity correction

Every one was caught by reading MEASURED counters, not by reasoning forward. The audit that found the third was done deliberately before the owner tests, because the first two cost a device run each.

### THE FULL UNTESTED STACK (build 182 was the last one measured)

| change | from | to |
|---|---|---|
| seed fill fraction | 1.0 | 0.5 |
| `seedSpacingCompensation` | on | 0 |
| `splitScreenRadiusPx` | 10 | 0 |
| `splitShareOfGrowth` | n/a | 0.2 |
| `relocationDonorOpacity` | 0.02 | 0.05 |
| relocation split geometry | one-axis, fixed offset | unchanged, now explicit |
| growth split geometry | one-axis | three-axis + covariance offset |
| `power > 0` branch | present | removed (provably dead) |
| exp cutoff in `pad0` | n/a | shipped |
| dead `tgCount` array | 16 KB | removed |

Also still untested from the build-180 batch: `lateLRFraction` 0.1, `shDCLRMultiplier` 4, `discPriorWeight` 0.001, `pruneOpacity` 0.02, `heldOutFraction` 0.10, per-image PSNR averaging, mean-across-slices PSNR, `gpuStep`.

### NEXT ACTION

Owner tests build 198. Read in this order:

1. **`relocated`** and **`addedBySplit`** vs the clone count. Relocation should be far above 21,000; the growth clone/split ratio should be near 80/20.
2. **median splat size**, via `tools/offline/analyse.py`. It has never gone below LiDAR sample spacing. Anything under 13 mm is the first time.
3. **the p90/p10 spread.** 1.8x is the disease. Movement toward 8.9x is the cure working.
4. held-out PSNR, remembering the measurement itself changed (mean not min, per-image not MSE-domain, 12 frames not 6) so it reads higher for free.
5. `gpuStep` and the pre-pass total.

If the model comes back dim or dissolving, `relocationDonorOpacity` back to 0.02 first: 5% of the population churned per pass with fresh Adam state is the largest single behavioural change in the stack.

---

## 2026-09-08 (night, part two) : Twenty measured agents, 29 survivors, and the seeder was handing out one size

**Since the last entry:** the owner pointed out that "no workflows" was my own inference, not his instruction, and that usage was at 7 per cent with four hours to reset. So a second workflow ran, with one change from the first: every agent was told that FOUR of the last six findings were wrong when finally measured, two of them shipped, and that any claim about a magnitude had to name a command it actually ran. **29 findings survived, 45 were killed.** The first round was 99 of 106.

He then, correctly, got angry that I reported three of the 29 and sat on the rest. Everything below is the rest.

### THE ROOT CAUSE, AND IT IS IN THE SEEDER

**Every seed in this scan was 7.397 mm.** Not approximately: the same number at p1, p50 and p99, for 100 per cent of 888,951 seeds. sd(log10) exactly 0.0000 decades. `radius = 0.5 * max(spacing, sensorSpacing)` has `spacing` as one global constant and `sensorSpacing` only overtakes it beyond 2.726 m, while the p99 distance from any point to its nearest camera here is 1.32 m.

Nothing downstream can repair that: clone copies the parent's scale, split divides by 1.6. Densification was being asked to manufacture the reference model's 0.64 decades of spread out of zero.

**And the fix is the LATTICE, not the radius.** I shipped the radius-only version first and it is the wrong half. Measured: optimising every seed's radius against its own detail bucket, at the shipped 300,000 cap, reaches 16.90 dB against the as-built 16.97. Size has to follow SPACING, and on a uniform lattice spacing is a constant. The reference model's log-size on log-spacing slope is 0.94 (correlation 0.79); ours was 0.18 (0.17).

So the seeder has TWO lattices now. Flat interiors (usable normal, no edge flag) go on one `flatInteriorCoarsening` times coarser; edges and normal-less samples keep the fine one. Radius is already cut from cell size, so size follows spacing by construction. They share one hash safely: a Morton key uses 21 bits per axis and `spread(z) << 2` tops out at bit 62, so bit 63 tags the lattice.

### THE OTHER LEVER THAT BINDS

**A clone copies the parent's scale exactly, and 80 per cent of growth was clones.** Simulated at identical population and identical total created: `splitShareOfGrowth` 0.2 gives 2.132x spread, 0.8 gives 3.820x, 1.0 gives 5.183x, against the reference's 8.9x. Now 0.8. Not 1.0 because a split shrinks covered volume to 48.8 per cent while a clone adds coverage.

Where 0.2 came from: a published 80/20 ratio for stock 3DGS, whose initialisation already HAS a size distribution. Wrong reference class.

### A REAL BUG

`trainer_regularizer` accumulated the loss `w * 0.5 * (rank - target)^2` and applied a gradient of `-4 w residual rank p_j (log p_j + H)`. The derivative is **-2**. Two agents confirmed without sharing code: analytically, and by central differences over 9,000 components of 3,000 real splats giving a ratio of exactly 2.000000 at three step sizes, 1.000000 after. The effective disc weight has moved 20x from where it started, not the 10x I thought.

### THE HOT LOOP TOUCHED TWO CACHE LINES

Every contributing pair issued nine atomics into `splatGrad2D` and three into `stats`, a different buffer, roughly 69 million times an iteration, while the gradient record was nine floats plus `float pad[7]`.

The naive merge is a silent correctness break and this is the crux worth remembering: `clearPerIteration` blit-fills ALL SIXTEEN floats of that record every iteration, while `stats` is zeroed once per densify pass. The three accumulators must survive a whole interval. So they accumulate in the gradient line for one iteration and are folded into `stats` by `trainer_preprocess_backward`, which owns the row and already loads both. Same atomics, same values, half the lines. `stats` is no longer bound to the backward rasteriser; index 14 is a documented hole rather than renumbered.

### MEASUREMENT HONESTY

Held-out frames were scored at **identity exposure and identity pose** while trained frames got both fitted, because `exposures` and `cameraDeltas` are only written for `slice.keyframes`. Part of build 182's 6.96 dB gap was protocol. `heldOutPSNRExposureFitted` now sits beside the raw number: closed-form least-squares gain and bias against the frame's own render, clamped to the trainer's own range. It corrects exposure, not pose, and says so.

### CORRECTIONS I OWE

- **Scaniverse's 7,389 mm p99 is a background dome, not a size tail.** Exactly 10,242 splats at contiguous indices 0 to 10241, on a 240.00001 m sphere, isotropic at log-scale exactly 2.0, alpha 253/255, all 45 higher SH coefficients zero. 10,242 is a 5-times-subdivided icosphere's vertex count. Their real p99 is 82 mm. We already have `DirectionalBackgroundModel`. I told him they had metre-scale splats covering walls; they do not.
- The `pad0` cutoff comment quoted 60.76 per cent of exp() calls wasted. That was pre-cutoff. Post-cutoff it is **0.58 per cent**: 75,529,323 pairs reach exp and 75,089,548 clear alpha. That work is finished.
- The radix comment described a 24-bit six-pass design the code has never implemented, above a constant reading 32. Only pass 7 is actually dead; pass 6 carries tile bits for most of a 1,530-tile grid.

### WHAT THE WORKFLOW TALKED ME OUT OF

- **Halving `densifyIntervalIterations`.** Growth is DELETION-limited, not fraction-limited: after the cap fills, the allowance equals the previous pass's total deletions to the unit on all 20 growth passes. 45,000 authorised, 208 to 625 granted. Halving gives 154,872 created against 154,408, and spread 2.150x against 2.132x, for double the CPU.
- **Reasoning about the disc prior from gradient magnitudes.** Adam's `mhat / (sqrt(vhat) + epsilon)` at epsilon 1e-15 is scale-invariant, so a term 1000x larger produces a step of the learning rate, not a 1000x step. The decisive number is that the prior flips the SIGN on ~48 per cent of components. And it must not be changed yet: with headroom at zero it is the only thing setting shape, so it is confounded.
- **Treating SH degree 3 as a config change.** The training kernels implement degree 2 forward and backward and nothing higher. Degree 3 needs new Metal in `trainer_evalSH` and the SH-gradient kernel before it can even be timed.

### VERIFIED

Builds 202 through 212 all green and published. `trapconv` and `deadwire` pass. The Gini and blob/disc/needle numbers were reproduced locally rather than taken on trust: 27.8 per cent and 0.434 against the reference's 61.2 and 0.753.

### ASSUMED / UNVERIFIED

Every behavioural change since build 182. Nothing has run on the device.

### NEXT ACTION

Owner tests. `cd tools/offline && python census.py` prints the verdict: the clone/split ratio, `relocated`, the size distribution with its p90/p10 spread, the aspect ratio, and the adaptivity pair. The spread is the line that matters, 1.76x today.

### GOTCHAS ADDED

- A Morton key here leaves bit 63 free (21 bits x 3 axes, top at bit 62), which is what makes a second lattice safe in one hash.
- `clearPerIteration` wipes all 16 floats of `splatGrad2D` every iteration; `stats` survives a densify interval. Never move an interval-scoped accumulator into that record without a fold.
- Growth allowance after the cap equals the previous pass's TOTAL deletions, including `carvedFromEmptySpace`, not just `prunedLowOpacity`.

---

## 2026-09-08 (very late) : The seeding fix was the wrong half, and the capture side got its first attention

**Since the last entry:** two things. I measured my own seeding change and it was wrong, and then the capture-side findings from the FIRST workflow, which nobody had ever touched, finally got acted on.

### I SHIPPED THE WRONG HALF OF THE SEEDING FIX, THEN MEASURED IT

Build 212 coarsened flat interiors onto a second lattice. Re-voxelising the owner's real geometry says that was a bad trade:

| config | seeds | slope | p10 | p50 | p90 | spread | smax/nn1 |
|---|---|---|---|---|---|---|---|
| one lattice (before) | 248,057 | 0.000 | 7.40 | 7.40 | 7.40 | 1.00x | 0.54 |
| build 212, coarsen 80% | 170,836 | 0.186 | 7.40 | **14.79** | 14.79 | 2.00x | **0.92** |
| refine edges only | 253,815 | 0.043 | 7.40 | 7.40 | 7.40 | 1.00x | 0.55 |
| **shipped, build 216** | 224,515 | 0.218 | **3.70** | **7.40** | 14.79 | **4.00x** | **0.54** |

The predicate "has a normal and is not an edge" selects about **80 per cent** of the scan, because the census records 69,029 of 888,951 seeds on an edge and 773,856 with a normal. Coarsening the MAJORITY moves the median by definition, and the median is the exact thing this work exists to shrink. Our model runs about 2.6x its seed size, so 7.40 mm seeds would have become roughly 38 mm.

Refining edges alone does nothing either: 7.8 per cent of samples never reach the p10 line.

Build 216 is three levels, and is strictly better than the original on every measured number: same median, same packing, 9 per cent fewer seeds, spread 1.00x to 4.00x. The 46 per cent coarse share is not tuned; it is what "high depth confidence, usable normal, not an edge" actually selects. Three levels need two tag bits, which fit because a Morton key with cells under 2^20 tops out at bit 59.

### THE CAPTURE SIDE, UNTOUCHED UNTIL NOW

- **A window painted itself green.** The LiDAR confidence map is written every frame and read nowhere in the coverage path. Now it attenuates the sample's sharpness (0.35 low, 0.75 medium, 1.0 high) rather than rejecting it, because `observe` keeps `bestSharpness` as a MAX: a patch seen only through doubtful returns never satisfies the quality channel, while one good look still settles it. Rejecting outright would read as "not scanned" and lengthen every scan.
- **The angle channel counted looks, never sides.** `directionCount / 4` against a 0.7 threshold means three buckets OF ANY KIND, so one direction at three heights passed. That is the owner's 50-degree dead zone, invisible to the HUD built to prevent it. Now `min(count, azimuthSpread/90)`. The research proposed "no gap wider than 90 degrees", which nothing can satisfy: a wall patch is visible only from the hemisphere in front of it, so the largest gap is always at least 180. The reachable question is the width of the OBSERVED arc. Verified on all eight occupancy cases including the wrap-around before shipping.
- **Capture wrote 868 frames; the largest consumer reads 240.** Seeder 174, survey 48, glass 40, trainer 120 or 240. About 630 MB written, some 450 MB never opened, 9 to 13 s of JPEG encode with the heat behind it. `keyframeMinIntervalSeconds` 0.30 to 0.45 still writes ~610, 2.5x the largest consumer. **This is the one change tonight whose trade could not be measured offline**: a thinner pool gives `keyframeMinQCWeight` fewer alternatives on a shaky scan.
- **HDR was never requested.** Guarded on `videoFormat.isVideoHDRSupported`, so a format carrying sceneDepth that lacks HDR makes it a no-op rather than a failure. One to two stops of highlight headroom is the difference between a window as pure white, with no gradient for any loss to fit, and one with recoverable structure.

### A RESEARCH CLAIM REFUTED WITH THE DEVICE'S OWN NUMBERS

The blur meter's 0.0426 deg/px divisor was said to under-report smear by 17 per cent, from a 0.0365 derived from Apple's 24 mm-equivalent MARKETING figure. This app writes the real intrinsics to disk: fx 1381.8971 at width 1920, a 69.575 degree horizontal field of view, on-axis pitch 2*atan(0.5/fx) = 0.04146. The constant is **2.7 per cent** high, not 17. It stays: the amber and red thresholds are calibrated against it, CONTRACTS.md fixes it, and 2.7 per cent on a "you are moving too fast" warning is far below the spread of a hand-held scan.

### VERIFIED

Builds 216 through 224 green and published. `trapconv` and `deadwire` pass on every one. The azimuth gap walk was checked against eight hand-computed cases; the blur pitch was computed from the capture bundle on disk.

### ASSUMED / UNVERIFIED

Every capture-side change. None of them can be measured offline at all, unlike the trainer work, because they alter what the device records rather than what is computed from a recording.

### OPEN, AND IT NEEDS A DECISION RATHER THAN A CONSTANT

An unseen ceiling is not in the coverage denominator. `recomputeFraction` iterates `voxels.values`, and unobserved space is not a voxel, so a ceiling nobody looked at can never make the percentage go down. Fixing it means deciding what "space you should have scanned" IS: the ARKit scene mesh is the obvious candidate since `meshStore` already exists, but that is a design question, not a tuning one, and guessing it would be the same mistake as the seeding predicate.

### NEXT ACTION

Owner tests build 224. `cd tools/offline && python census.py` prints the verdict. Read the size distribution's p90/p10 spread first: 1.76x is the disease, and seeding alone should now supply 4x before densification's contribution.

---

## 2026-09-10 : Build 250 measured, relocation found dead since build 182, build 252 batches everything that survived

**Since the last entry:** builds 226 to 250 happened (the 30,000-iteration ceiling run, early stopping, SSIM beside PSNR, the edge-rank mistake in 244, and the `densifyEndFraction` 0.5 leftover that froze half of builds 244 to 246). Build 250 is the first clean measurement after all of that. Two workflows then re-judged the remaining refutations (123 in total, not 73) and dug into the 250 census.

### BUILD 250, MEASURED

| | 250 |
|---|---|
| pre-pass | 17.6 s (seeding 12.7) |
| training | 77 s, 19.2 ms/iter (gpuStep 55.0, gpuScan 6.7, earlyStopEval 2.8) |
| **total** | **94.6 s** |
| splats | 299,888 |
| held-out PSNR | best 20.392 at 3,600 (exported), end state 20.038 raw / 20.444 fitted |
| SSIM | 0.6618 |
| size | p10 3.62, p50 7.83, p90 17.22 mm, spread 4.75x |
| shape | needles 9.1, discs 86.2, blobs 4.7 per cent |

What 250's four changes did: `earlyStopEvalIntervalIterations` 400 halved earlyStopEval (5.7 to 2.8 s); the splatGrad2D self-clear took gpuScan 7.6 to 6.7 s (that drop calibrates blit fill at **83 GB/s**); the tangent clamp removed every splat above 720 px and 13.3 per cent of tile instances; and `pruneMaxScreenRadiusPx` 720 pruned **nothing**, because the clamp had already removed every splat it targeted (clamped max radius 243.7 px). The 13.44 per cent finding behind it was measured before the clamp existed.

### RELOCATION HAD BEEN DEAD SINCE BUILD 182

`relocated 0` in all 39 passes. The gate was `growthAllowance == 0`, i.e. headroom EXACTLY zero. After the cap fills, the low-opacity prune frees 25 to 526 slots a pass and growth refills exactly those (`added == growthAllowance == headroom` on every pass from 6 on, zero mismatches), so headroom is never zero. Build 182 seeded the cap full, so headroom WAS zero there, and relocation ran 21,000 times; lowering the fill to 0.5 silently killed it. The `allowGrowth &&` conjunct also killed it after the growth window shut. And the census field that would have exposed it, `relocationDonorsAvailable`, is only written inside the dead branch, so it read 0 meaning "never measured", not "empty pool".

The pool is real: 9,799 splats below 0.05 opacity, 1.63x the median size of the rest. Build 252 relocates whenever growth is underfed (`growthAllowance < wantedGrowth`), in the same pass, behind growth's share of the ranked list, with a mask so no donor is a slot this pass used, a skip for targets too faint to survive the split, and filter3D carried with the moved splat. **Expected effect is NOT measured.** A static sim says p50 7.83 to 6.71 mm with uniform targets, but targets are ranked by AbsGS score, which grows with pixel coverage and so favours large splats, and the same sim says that case RAISES p50 to 9.1. Kill switch: `maxRelocationFractionPerPass = 0`.

### BUILD 252, EVERYTHING ELSE THAT SURVIVED

Speed (bit-exact or model-neutral): splatGrad and shGrad self-clear in preprocess_backward (28.8 MB of fill); the regulariser skips undrawn splats like both Adam kernels already do (77.8 per cent of its threads wrote gradients nobody read); 24-bit radix key with 12-bit LOG depth, six passes not eight (log so no room is ever too deep; 48.81 dB against exact order, within 0.003 dB of it against the photo); two finiteness tests per backward pair instead of twelve (the roots are `dLdPower` and `weight`, NOT `dLdAlpha`: Metal's `min(0.99, NaN)` is 0.99); the held-out eval uses the GPU background kernel instead of a CPU loop it ran 132 times, and the duplicate final evaluation is folded into the first. Estimated 5 to 10 s in total, none of it device-measured.

Correctness: export now ships the end state when it scores at least as well as the checkpoint (250 threw away 20.444 for 20.392, and shipped a 3,600 cloud with 4,000 exposures and poses); held-out views no longer leak denom and visibleFlag into the next densify pass (the eval ran BEFORE densifier.run, and the reset runs AFTER it, so the old comment's claim was false); the backward now differentiates the forward's clamped Jacobian (latent today, 0 drawn splats bind); `carveIntervalIterations` reads 500, which is what 250 against a 100 densify interval was actually running; the seed load is timed as `prologue`, since 4.03 s of training wall sat in no bucket.

Rejected with measurements: moving the disc target toward rank 3 (post-hoc isotropy costs up to 1.8 dB; rank is not the lever, and 0.001 to 0.005 of weight all act the same because the prior dominates 98.7 per cent of discs); lowering the 720 prune (90 px buys 1.27 per cent of tile instances and costs one held-out view 35 dB); raising iterations on an extrapolated curve (1,000 more cost about 19 s).

### VERIFIED

Every edit applied by exact-string match with an all-or-nothing script. trapconv, deadwire and brandmatch pass on the committed tree. Builds 251 and 252 green (commit 608bc36).

### A LINT THAT LIED TO ME

`python tools/trapconv.py | tail -3; echo $?` prints TAIL's exit code. It read 0 while trapconv had failed on the one growth line I rewrote: the allowlist matches an expression's exact text, so moving an allowlisted conversion into a new variable un-allowlists it. Build 176 went red on exactly this. Never pipe a lint before reading its exit code.

### ASSUMED / UNVERIFIED

Everything in 252. The relocation change is the largest behavioural change since build 182 and its sign on p50 is genuinely unknown.

### OPEN

- **Adam bias correction after a split or relocation.** Both zero `adamM`/`adamV` for the rewritten slot but leave `stats.stepCount`, so bias correction treats fresh moments as mature and the first steps run at about 3x the learning rate. Resetting stepCount alone would desynchronise the SH moments, which are not zeroed. Needs its own change covering both paths.
- **SH degree 1.** `budgetAsRun.shDegree` is 1 on this scan (4 coefficients). Scaniverse ships degree 3.
- Pending verifiers: the command-buffer merge (about 1.0 s, large surgery) and the gpuStep breakdown (50 per cent backward raster, 17 per cent loss chain, 14 per cent forward).

### NEXT ACTION

Owner tests build 252. Read in this order: `relocated` per pass (0 means the gate is still dead; about 10k at iteration 600 then donor-limited is expected), `relocationDonorsAvailable` (first real measurement since 182), p50 and spread, best held-out PSNR against 20.39 +/-0.4 and SSIM against 0.6618 (a drop over 0.4 dB means set `maxRelocationFractionPerPass` to 0), then gpuScan (expect about -1.4 s), gpuStep, earlyStopEval, `prologue`, and the total against 94.6 s.

### ADDENDUM: build 254, the rest of the gpuStep findings, before 252 was tested

The owner had not yet tested 252 when the last verifiers returned, so everything model-neutral that survived went on top of it as 254, a strict superset. Test 254, not 252.

- `draws[gid].opacity = 0` in trainer_preprocess's prologue was a dead 19.2 MB scattered store: every reader of `draws` is gated on tilesTouched != 0, which implies the full `draws[gid] = d` store ran.
- `trainer_loss_photometric` wrote `composited` (4.67 MB) that nothing reads.
- The regulariser and both Adam kernels gate on `tilesTouched` instead of `stats.visibleFlag`: the identical predicate, read from a dense 1.2 MB array rather than a 32-byte-strided stats line. That makes `visibleFlag` dead, so `trainer_reset_visibility` (9.6 MB a step and a dispatch) and the store in preprocess are gone, with every Swift declaration that pointed at them.
- SSIM: X*X, Y*Y and X*Y are formed per tap inside trainer_blur_h's moments pass (flagged by the spare `pad0`), so `trainer_ssim_prepare` is gone; `trainer_ssim_backward` is folded into `trainer_loss_finalize`, which writes the full gradient back into gradFinal because the CPU background accumulator reads it.
- `timings.encodeStep` times the GPU-idle gap between buffers A and B, which is what prices the A/B merge. census.py prints it and a lower bound on untimed wall.

Estimated 2 to 4.7 s, all from a bandwidth model; the verifiers had no on-device timing for any of it. Held back deliberately, each for a build of its own: the pipelined A/B merge (about 1.0 s, a large restructure with an overflow-retry path) and batching the filter sweep 8 cameras a submission (about 0.15 s, but it partly undoes the mitigation for an old iteration-500 watchdog crash).

Failure signatures to read on 254 if anything went wrong: a wrong Adam binding freezes the model (PSNR collapse, sizes stop moving); a silent regulariser shows as discs falling from 86.2 per cent and s3/s2 rising from 0.20; a stuck `pad0` corrupts the SSIM gradient and SSIM collapses from 0.66.

Builds 253 and 254 green (commit c8d3738). Owner to test 254.

---

## 2026-09-10 (later) : "Seeding 12.7 s" was four stages, and the edge maps were built for 868 frames to serve about 300

**Owner, verbatim:** "Not gonna test that one ngl, just keep building." Workflows are off (usage 55 per cent), so this is solo work.

### THE PRE-PASS CLOCK WAS BILLING FOUR STAGES AS ONE

`markStage` records the time since the previous mark. After `markStage(\.carving)` the pipeline runs the trust fields, the edge classifier and glass detection, and only then builds the seeds and marks `\.seeding`. So the 12.69 s "seeding" on build 250 was trust + edges + glass + seeds, and nothing has ever said which. Three clocks added: `trust`, `edges`, `glass`. census.py prints them in pipeline order.

### EDGE MAPS: 868 BUILT, ABOUT 300 READ

`NativeDepthEdgeClassifier.classify` walked every frame in the capture: decode a JPEG thumbnail, read the depth and confidence sidecars, classify, write a 49 KB map. Readers: the seeder's 174 keyframes, the trainer's own keyframes (about 120, chosen later by its own selector, so the pre-pass cannot know them) and the held-out frames. Now `classify` only registers the frames and clears the directory, and `map(for:)` builds a missing map on first use, writes it where it always went, and caches it. The seeder pays for its 174; the trainer pays for its own on the supervision prefetch worker, off the critical path. Same maps, byte for byte, for every frame anyone reads. Clearing the directory is what rewriting every file used to amount to, and it stops `load` registering a stale map from an older run of the same scan. The PC Booster is unaffected: it ships its own `classify_edges`.

**Not measured.** The saving is whatever share of the 12.7 s the edge stage was, times about two thirds. The new clocks will say.

### LOOKED AT AND LEFT ALONE

- **Adam after a split.** A split child's moments are zeroed while `stepCount` is not, so its first steps run 3 to 6x the learning rate. That is exactly what reference 3DGS does: PyTorch Adam keeps one `step` per parameter tensor, so new points get fresh moments under a mature step count. Changing it would be a departure from the reference with an unknown sign, not a fix.
- **Trust fields over every frame.** Also built for all frames, but the trainer reads them for keyframes the pre-pass cannot know in advance, and each frame's trust is verified against partner frames, so it is not separable per frame the way edges are. It gets a clock first.
- **The 4.03 s training prologue.** The PLY parse is 46 MB copied once and about 14 M float reads, and the nearest-spacing pass is a hash grid over 2,000 samples: neither is seconds. The likelier owner is building about 28 compute pipelines at trainer start, which no clock covers. `prologue` (since 252) times only the seed load, so it will say whether that part matters.

### NEXT ACTION

Keep building. The next census to arrive, from any build since 256, shows `trust`, `edges`, `glass` and `seeding` separately; whichever is largest is the next target.

Builds 255 and 256 green. 256 is the build to test: it contains every change since 250.

---

## 2026-09-10 (night) : Build 256 measured, 94.6 s to 83.6 s, relocation alive, and the trust loop's 40 million allocations

### BUILD 256, MEASURED

| | 250 | 256 |
|---|---|---|
| pre-pass | 17.6 s | **13.6 s** |
| training | 77 s | **70 s** |
| **total** | 94.6 s | **83.6 s** |
| best held-out PSNR | 20.39 | 20.41 |
| SSIM | 0.6618 | **0.6647** |
| p50 / spread | 7.83 mm / 4.75x | **7.48 mm / 5.00x** |
| needles / discs / blobs | 9.1 / 86.2 / 4.7 | 11.2 / 82.2 / 6.7 |
| densest 10% / Gini | 45.2% / 0.579 | 47.7% / 0.598 |

**Relocation ran 25,935 times** (0 on 250), donor-limited on every pass exactly as predicted, and `prunedLowOpacity` fell 2,299 to 75 because the faint splats now get moved instead of deleted. p50 fell rather than rose, so the feared large-target bias did not dominate. The export fix works: the end state (fitted 20.463) shipped over the 3,600 checkpoint (20.410).

**The pre-pass, split for the first time:** trust 6.42 s (47 per cent), carving 3.09, seeding 1.77, glass 0.28, edges 0.04. Lazy edges took about 4 s out of the pre-pass, but put about 2.5 s back into training: the loop waited 4.0 s for supervision against 1.5, because the trainer's maps were then built one at a time on the prefetch worker.

**Two hypotheses closed by the new clocks:** `encodeStep` is 0.1 s for the whole run, so the pipelined A/B merge is worth only its submit latency, about 1 s, for a five-file restructure. Not worth it yet. `prologue` (the seed load) is 0.5 s. 3.9 s of training wall is still untimed; the pipelines compile BEFORE the census opens, so the suspect is `loadSmartLayer`, now clocked as `smartLayer`.

### BATCH FOR 258

- **Edge maps warmed on every core** at the start of each slice (the slice's keyframes and held-out frames, via `concurrentPerform` on a background queue), so the first cycle stops waiting on the prefetch worker. The classifier is `@unchecked Sendable`: its state is lock-protected and a map is built outside the lock.
- **`smartLayer` clock** around `loadSmartLayer`.
- **The trust loop's per-sample work.** At stride 2 and 4 partners the cross-frame check visits about 10.7 million samples a pass. For each it did a dictionary read and a LOCKED depth-cache read per partner (about 43 million lock round trips for values constant within a frame) and allocated up to four small arrays (residuals, a sorted copy for the median, the deviations, their sorted copy). Partner poses and depths are now fetched once per frame, and the median runs by insertion sort over a reused buffer. Same values, same order, same result to the bit. This is the pattern that has paid every time on this project: `radiance()` returning a Swift array per call was 388,800 mallocs an iteration and 42 s of training.

**Not measured.** The trust stage should fall well below 6.42 s; how far says how much of it was allocation and locking versus the plane sweep and the per-frame I/O.

### NEXT

If trust is still large after this, the per-frame loop parallelises across cores: the depth and image caches are already locked, and the only order-dependent state is the global 20,000-sample plane-sweep budget.

Builds 257 and 258 green. 258 is the build to test.

---

## 2026-09-10 (late night) : Build 260, carving on every core, and what was deliberately NOT done

The owner asked for more before the next test, so 258 was not tested and 260 stacks on it.

- **Free-space carving on every core, same bytes on disk.** 868 keyframes, 10.7 M rays, one core, 3.09 s on 256. The grid does not depend on ray order: a cell is SURFACE if any ray ended in it, its hit count is the number that did (saturating), and the file is sorted by key. So each core carves every Nth keyframe into its own table and the tables merge by exactly those rules. The helpers became static and take the shard, so nothing on the carver is written from two threads; the cap flag they used to set on `self` is now a field of the shard. Only difference, and only at the cell cap: each shard is capped on its own. The room uses 152,180 cells.
- **The seeder's edge maps warmed on every core** before its loop, the same trick as the trainer's in 258. Since 256 the seeder built its 174 maps one at a time on its critical path.
- **tgIndex dropped from trainer_rasterize_backward.** An experiment, bit-identical output: the loop re-reads `values` (an L1 broadcast, every lane the same address) instead of a 1 KB threadgroup copy, taking the footprint from 11,776 to the forward rasteriser's 10,752 bytes. One verifier said that buys a third resident threadgroup; another said the occupancy argument does not hold and the device read could be slower. gpuStep on 256 was 49.0 s and nothing else in 258 or 260 touches the GPU, so the next census reads it directly. If gpuStep rises, revert this one commit.

**Deliberately not done: the trust build across cores.** 258 removed its per-sample allocations and locks and that result is not measured yet. Parallelising it means a 250-line restructure whose failure mode is silent: `trust_noise.bin` is frame-major with no header, so any ordering slip shears every later frame's trust into another frame's samples. The design, if the next census still shows trust large: run the plane-sweep frames serially until the 20,000-sample budget is spent (a few frames), then compute the rest on every core in chunks and apply the four order-sensitive outputs (noise append, bias accumulator, residuals by confidence level, affines) in slot order.

Next test: 260. It carries every change since 250.

Builds 259 and 260 green (commit 9ca7fa5). 260 is the build to test.

---

## 2026-09-11 : Build 260 measured, and build 262: trust across cores, tgIndex reverted

### BUILD 260, MEASURED

| | 256 | 260 |
|---|---|---|
| pre-pass | 13.6 s | **9.5 s** |
| carving | 3.09 | **1.05** (every core, identical grid) |
| trust | 6.42 | **4.37** (per-sample allocations and locks gone) |
| seeding | 1.77 | 1.70 (the edge warm-up bought almost nothing: seeding is its sample loop, not its maps) |
| training | 70 s | 73 s |
| gpuStep | 49.0 | **52.2** |
| total | 83.6 s | 82.5 s |
| best held-out PSNR / SSIM | 20.41 / 0.6647 | **20.47 / 0.6668**, best yet |

**tgIndex removal made gpuStep 6.5 per cent SLOWER.** It was the only GPU change in 258 and 260. Reverted in 262, with the measurement written on the declaration. Same direction as build 92's SIMD-group reduction: on this GPU, threadgroup memory is not what limits the backward rasteriser, and a device re-read in its innermost loop costs more than the occupancy it might buy.

Supervision wait was 3.5 s (4.0 on 256, 1.5 on 250): 258's background warm-up raced the prefetch worker for the same cores and frames. 262 blocks on it instead.

`smartLayer` is 1.7 s and untimed fell to 2.3 s.

### BUILD 262

- **tgIndex restored.**
- **Trainer edge warm-up blocks** until every map for the slice is built.
- **The trust build across cores, bit-exact.** Each slot's work is independent except four ordered outputs: its row in `trust_noise.bin` (frame-major, no header), the bias accumulator (float sums), the residuals by confidence level, and the affines. `computeSlot` (a local function, so it captures the build's state without new plumbing) produces all four without touching them; `apply` commits them strictly in slot order. The plane-sweep budget is first-come in slot and sample order, so slots run serially exactly as before until it is spent, then the rest run on every core in chunks, each worker walking eight consecutive slots through its own depth cache so neighbours share their partners' loads. A missing result throws rather than letting the noise file shear. If the budget is never spent everything stays serial: correct, not faster. A log line records the split.

Expected: trust 4.37 s down by the parallel share of its compute; the serial `apply` (about 10.7 M accumulator adds) is the floor. Total should land near 9.5 - 2.5 + 73 - 3.2 - 2 = about 75 s if all three hold. Not measured.

Builds 261 and 262 green (commit d2a3439).

### BUILD 264: two more serial loops and a keyframe selector that read the wrong field

- **Background warm-up hoists.** The far-field fit visits about 120 frames and, per sample, called `authority(frame:sampleIndex:)`, which is `map(for:)` (the authority map's lock plus a dictionary lookup) followed by an index: about 5.9 M lock round trips for one object per frame. It also rebuilt the pose's rotation inverse per sample. Both hoisted to once per frame. Same object, same values. This is inside `smartLayer` (1.7 s on 260).
- **Time-offset calibration on every core.** Pair preparation (a depth load, two grey images, a Shi-Tomasi response, a feature pick) runs per pair on every core, compacted in pair order, with the first load error still thrown in pair order. The sweep scores each candidate timing on its own core, same pairs, same samples, summed in the same order, with the offsets generated by the same accumulation. Every cost is the serial one. 1.08 s of the pre-pass on 260.
- **Keyframe look-ahead picks by motion blur.** Its own comment said "take the SHARPEST"; the code compared `qc.weight`, the product of blur, sharpness, depth coverage, exposure jump and tracking. Now `motionBlurPixels`. A quality change, the only one in 262 to 264; it changes which frames train, not how.

### THE REFUTATION LEDGER, ANSWERED FOR THE OWNER

Asked whether every recovered refutation has now been added: no, and deliberately. Of the 42 recovered in the first pass and the survivors of the second: shipped as listed across 250 to 264; measured and dropped (tgIndex, 6.5 per cent slower; the A/B merge, worth about 1 s by the encodeStep clock); refused because they cut the seed count (three findings, the owner's floor is 800k); already adopted before this session (schedules re-indexed to 4,000 iterations in 244); held as quality experiments that cost time (SH degree 2, supervision resolution); left alone because the proposed fix makes it worse (the SH band gate would open view dependence at 35 per cent instead of 12). Open and actionable: seeding across cores (1.70 s, needs an order-exact merge because PLY order drives the trainer's thinning), filter-sweep batching (0.15 s, an old watchdog crash lives there), and the exposure model.

Builds 263 and 264 green (commit 0dd3787). 264 is the build to test.

---

## 2026-09-11 : Build 264 measured: fastest yet, and the keyframe selector change cost 0.7 dB

| | 260 | 264 |
|---|---|---|
| pre-pass | 9.5 s | **7.8 s** (trust 3.38, timeOffset 0.38, carving 1.11) |
| training | 73 s | **67 s** (gpuStep 46.4: tgIndex revert plus) |
| **total** | 82.5 s | **74.8 s** |
| best held-out PSNR | 20.47 | 19.73 |
| SSIM | 0.6668 | 0.6414 |

The speed work all landed. The picture did not survive the one quality change: picking the look-ahead candidate by `motionBlurPixels` instead of `qc.weight`. held_out_frames.json moved from [18 ... 421] to [23 ... 515], so the test set itself changed, but SSIM and trained-view PSNR fell too. Mechanism, read from the selector and refutation [40]: the walk goes in frame order and stops at 120; the least blurred candidate sits further ahead in the window, lastCenter jumps further, and the same 120 keyframes spread over more of a capture whose second half is a REVISIT of the first (every late camera within 1 m of an early one). [40]'s offline measurement priced exactly that spread as a loss: 13.9 views per splat to 11.6 to 12.1. Reverted in 266, which also records the keyframe span in the census. If 266 returns to about 20.47 it also confirms the parallel trust build is exact.

**Standing rule from the owner, 2026-09-11:** no finding is refuted or dropped without my own check against the code or the data. All 223 refutation records are being re-read by hand.

---

## 2026-09-11 (later) : Every refutation re-read by hand, and what came back

Owner's rule: nothing refuted without my own check. All 223 records read (73 refuted findings, 50 missed ones, 100 patch-review notes). Verdicts, and what I checked for each, are in `tools/offline/REFUTATION_LEDGER.md`.

**Refutations that were wrong, with the measurement that shows it:** A03 (CPU parallelism called "rearrangement"; carving, trust and time-offset since paid 2.0, 1.0 and 0.7 s), M16 (memcpy called 0.5%; it was 126 s to 101 s of training), M19 (clear-folding called free of byte savings; gpuScan went 6.7 to 3.8 s), A26 (exposure called constant from frames 0-15 only; the capture's shutter spans 0.74 stops).

**Reopened:**
- **Near-camera veil.** Found inside the A26/A27 refutations and never followed up: deleting every splat within 0.5 m of the camera raised PSNR on the 16 photographed frames by a mean +2.2 dB (up to +5.5) on 264's model. Those splats are wrong, not real geometry, or removing them would lower the score. The offline tools had NO tangent clamp in any of their three Jacobian copies (project.py, raster.py, detail.py), so that number is being re-measured with the device's clamp before anything is believed.
- **Registration** (A11, A27, M36, M45): pose error up to 8.5 render px against a 3.7 px median splat; the ICP cap tries 600 of 24,916 candidates.
- **Supervision worker** (M00, M02): the loop now waits 4.2 s for it because the GPU caught up. Per-sample trust lookups took ~5 locks each; every decode built a luma array supervision never reads. Both fixed locally (see below).
- **Experiments that cost no speed:** disc prior off (A25, A69), SH rest learning rate (A59: our band 1 is 4.8x weaker than Scaniverse's), split share 1.0 with shrink 2.0 (A54), image-detail seed sizing (A62, A64), the backward's p/q reformulation (A45).
- **Keyframes** (A13, M47): rotation admits 104 of 120 keyframes, which is what ends the walk at frame ~515 of 868; sharper supervision needs MORE keyframes (264 proved a thinner spread costs 0.7 dB).

**Applied locally, NOT pushed** (holding until the owner tests 266): supervision fetches each frame's trust slices once (SmartFrameTrust, identical values, the nil-trust 0.5 vs no-noise-reader 0 distinction kept); supervision's image cache decodes rgb only. trapconv, deadwire and brandmatch pass.

**The veil is not real.** Re-measured with the device's tangent clamp in all four offline Jacobian copies: deleting splats within 0.5 m of the camera changes PSNR by 0.000 dB on all 16 frames. The +2.2 dB was the offline renderer drawing unclamped off-axis near-plane splats that the phone has clamped since 250. The same fix exposes a wider problem: every offline render since 250 understated the model by about 2 dB, so offline render-based conclusions from 250 to now must be re-run (A27's pose-shift gain is one). `project.TANGENT_CLAMP` now governs all of them.

---

## 2026-09-11 : Build 266 measured, quality restored, and a speed conclusion withdrawn

Best held-out PSNR **20.47**, SSIM **0.6654**, 299,976 splats: back to 260 exactly. 264's 0.7 dB loss was the keyframe selector, and the parallel trust build is confirmed exact. Total 81.6 s (pre-pass 8.6, training 73).

**Withdrawn:** "tgIndex removal made gpuStep 6.5 per cent slower." 266 restored tgIndex and measured gpuStep 52.4 s, the same as 260 without it (52.2). gpuStep on identical GPU code moves about +/-3 s between runs, and the keyframe SET moves it about 6 s (264's frames rendered cheaper: 46.4). A 3 s difference between two single runs is not evidence. tgIndex stays restored (the original code) and the question is open.

**Keyframe span, first measurement:** frames 0 to 439 of 868, 51 per cent of the capture; 428 frames after the last keyframe never train.

### BUILD 268

The two supervision fixes held back for 266: per-frame trust slices instead of ~5 locks per sample, and rgb-only decoding for supervision's image cache. The loop waited 3.7 s for supervision on 266.

---

## 2026-09-11 : Build 270, the scanner never had its camera, and clocks inside trust and seeding

### THE SCANNER FINDING (owner asked whether the scanner could be improved)

scan_20260906_164840 was captured on app build 44 (bd2f4e7), which already carried the ISO record (since 1aa3082) and the 1/60 s shutter cap (a6091b6). Yet not one of 868 frames carries an `iso` key (nil is omitted, like `refinedPose`), every frame's bracket is `normal`, and the settings record `bracketEveryNFrames 0`. `currentISO` is nil only when `device` is nil, so `configurableCaptureDeviceForPrimaryCamera` returned nil for the whole session. Apple documents nil only on phones without an ultra-wide camera; this phone has one. The code asked the base class `ARConfiguration`; Apple's examples ask `ARWorldTrackingConfiguration`. Fixed to the subclass. NOT PROVEN to be the cause: the next scan's ISO values say.

What was silently off: the shutter cap, the ISO record, the brightness hold, the white-balance hold, window brackets. What the photos show: motion blur p10/p50/p90/max 1.7 / 3.9 / 7.2 / 11.6 px; 68 percent of frames over 3 px, 20 percent over 6 px; shutter 1/100 to 1/60 s (free auto-exposure, capped by the 60 fps frame time, so the 1/60 cap was a no-op even had it applied). qc.sharpness median 0.27.

**Dark brackets are PARKED** (`CaptureTuning.bracketingParked`). Nothing downstream reads `bracket`: the trainer, pre-pass, seeder and keyframe selector would train a 3-stop-dark frame as an ordinary photo, and the only reader (the background model) skips them. With the device fixed, bracketing would have started firing every 12 keyframes. The HUD switch is hidden while parked.

**Next on the scanner, needs the owner's call and a new scan:** a 1/120 s shutter cap would halve blur at the cost of one stop of ISO noise (which the trainer averages across views; blur it cannot). Decide after one scan records ISO so the headroom is known. A new scan also breaks comparability with every trainer A/B so far, so the old scan stays the trainer benchmark.

### CLOCKS

- `PrePassCensus.trustBuild`: setup, serial plane-sweep prefix (with its slot count), parallel slots (count, cores), confidence rewrite, writes. Trust is 3.85 s, the largest pre-pass stage, and nothing said which part.
- `PrePassCensus.seeding.seconds*`: before-loop (edge warm-up split out), sample loop, shaping, write. Decides A36/A03 (an order-exact parallel seeder is only worth it if the loop is the bulk).
- Both optional, so older census files still decode. census.py prints them.
- tgIndex comment corrected (the 6.5 percent claim was noise).

Research workflows running: keyframe coverage, registration, size gap, remaining speed (8 agents) and the exposure model (2 agents).

---

## 2026-09-11 : Build 270 measured, fastest yet: 69.9 s

| | 266 | 270 |
|---|---|---|
| pre-pass | 8.6 s | **7.9 s** |
| training | 73 s | **62 s** |
| **total** | 81.6 s | **69.9 s** |
| gpuStep | 52.4 | 45.5 |
| supervision wait | 3.7 | **1.1** |
| supervision worker (CPU, off the critical path) | 43.8 | **29.5** |
| best held-out PSNR / SSIM | 20.47 / 0.6654 | 20.42 / 0.6661 |

Same keyframes (0 to 439), same scan. The only trainer changes since 266 are 268's supervision fixes (per-frame trust slices, rgb-only decode): 14 s less CPU work in the worker. gpuStep fell 7 s with no GPU change. Two readings, not separated: run-to-run noise (+/-3 s seen before), or the phone's CPU and GPU sharing one power and memory budget, so a worker burning 14 s less CPU leaves the GPU more. If the second, CPU work that overlaps training is not free even when it is off the critical path.

**Trust split (first measurement):** setup 0.02, serial plane-sweep prefix **1.30 s for 13 slots** (100 ms a slot), parallel 1.21 s for 855 slots on 6 cores, confidence rewrite **0.87 s** (serial), writes 0.00.
**Seeding split:** edge warm-up 0.25, sample loop **1.06**, shaping 0.14, write 0.19.

### BUILD 272

- **Confidence rewrite on every core**, rows written in slot order: same bytes. 0.87 s serial.

Build 271/272 green (ae3d513).

---

## 2026-09-11 : Build 274, the keyframe look-ahead leaked the test set; one quality change plus exact speed work

### THE LEAK (research workflow, confirmed by my own run and reading)

`keyframeSharpnessLookahead` 5 (since 244) picked the best-weighted frame up to 5 ahead, but the walk carried on from the gate frame, so the next frames passed the turn gate measured against the pick and chose it again. kf_exact.py reproduces held_out_frames.json and the span exactly and prints 29 duplicate slots of 120. splitHeldOut takes every tenth entry, so a back-to-back duplicate lands in both sets: 6 of the 12 held-out frames (18, 164, 209, 233, 355, 394) also trained. EVERY held-out PSNR from 244 to 272 is flattered. The only quality change in 274: look-ahead 0 (the pre-244 greedy). Predicted exactly: 120 distinct keyframes, 108 trained, span 0..485, held_out_frames.json [17, 61, 109, 155, 199, 231, 272, 314, 352, 392, 425, 460]. Offline: views per Gaussian 11.2 to 14.2, share seen by 3 or fewer views 10.2 to 6.0 percent, tile work -4.5 percent; training-frame blur p50 1.09 to 1.46 render px.

**The held-out number will FALL and that is not the model getting worse.** The estimate for 266's clean half is about 18.9 dB (assumes the leaked half scored near the trained-view 22.0). Success line from the researcher: best held-out >= 20.0 and SSIM >= 0.66 on the clean set; failure below 19.0 or SSIM below 0.64. From 274 on, compare against 274, not against 20.47.

### EXACT CHANGES (output identical; census signatures say if not)

- Trust: `appendFloats` one bulk buffer (85 M four-byte appends); plane sweep's ZNCC in place (5.12 M allocations); bias accumulator mutated in place (one hash, no Set copy); residuals by confidence level replaced by the two counts `recalibrate` ever read (10.7 M appends).
- Trust clocks: `parallelApplySeconds`, `sweepsRun`. Seeding: `secondsPrefetchWait`.
- PLY writer: one bulk buffer (14.1 M appends and a 56.6 MB copy). Same bytes, incl. no NaN sanitising.
- Held-out eval builds no depth samples (it never read them).
- Revisit ICP on every core, folded in candidate order: must reproduce icpConverged 132, icpRejected 468, confirmedPairs 132, poseGraph.initialCost 4520.09.
- Census: trainer camera deltas per slice (median, max, common-mode). Ceiling ~0.9 deg / 6.3 cm.

Registration research (ledger A27, A27t, TEAR): the rigid image shift is worth +0.25 dB cross-validated, it is not a timing error, and the pose graph TEARS the path at all 18 submap boundaries (median 1.18 deg / 3.4 cm, worst 2.08 deg / 23.9 cm between consecutive frames). The continuous-correction fix is the next quality A/B, its own build.
