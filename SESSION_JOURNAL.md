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
