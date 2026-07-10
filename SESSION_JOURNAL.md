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
