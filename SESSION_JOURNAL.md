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
