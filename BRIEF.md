# Nimbus3D : Product Brief

_Living design doc. Last updated 2026-07-09._

## One-line
Scan a real object with a Gaussian splat, and get back a game-ready, correctly-lit 3D asset: best-topology low-vertex mesh, delit PBR materials, and a real captured HDRI, all from one capture session.

## Who it's for
Game developers and 3D artists. They care intensely about clean topology and low vertex counts, so "best topology, least vertices" is the killer feature, not a nice-to-have.

## The core insight (the market gap we own)
Three camps exist and NONE owns our combination:
- Splat capture apps (Scaniverse, Polycam, KIRI): only KIRI gives a clean mesh; none capture real lighting.
- AI generators (Meshy, Rodin, Tripo): clean low-poly, but hallucinated geometry (not your real object) and no captured lighting.
- World models (World Labs Marble): splat + mesh but heavy (~600k tris) and hallucinated, no real light.
- HDRI tools (HDReye): real HDRI, but zero geometry.

The intersection (real object + best-topology low-poly + real captured HDRI + delit so it lights correctly in-engine) is empty. That is what Nimbus3D is.

Closest competitor: KIRI Engine does one-click splat -> quad mesh + PBR. It does NOT capture real lighting or delight properly. Our edge is the full correctly-lit loop, not "another splat scanner."

## Honest constraints (verified via adversarial research, 2026-07-09)
1. **"Free HDRI as a byproduct of the splat" does NOT work.** A splat bakes lighting into color; there is no HDRI inside it. LDR phone cameras clip bright light sources, which is exactly what an HDRI needs, and that info is physically gone. AI methods (LuxDiT, GaSLight) hallucinate a plausible guess, not a measurement.
   - **Fix / reframe:** capture the HDRI deliberately as a second quick step in the same session (guided bracketed 360 sweep, HDReye-style). Real, measured, shippable. Still "one session, model + real light."
2. **Auto splat -> clean low-poly is only PARTIALLY automatic.** Great for static/opaque props with little or no cleanup. Hero characters that deform still need a short human retopology pass (edge-loop flow for animation is not reliably automatic in 2026). Reflective/transparent/glass still break everything.
3. **Splat color is baked lighting, not clean materials.** Getting a delit albedo + PBR maps needs a delighting step almost no consumer tool does. This is where our captured HDRI pays off (delight, then light with the real HDRI).

## v1 scope (narrow to win)
Single, matte-ish, opaque objects (props). NOT hero characters, NOT glass/mirrors. Land "scan a prop -> perfectly-lit game-ready asset in one export." Expand later.

## Differentiator: Dynamic Textures (researched + verdict in, 2026-07-09)
Drive surface detail through material maps + parallax occlusion mapping (the cheap in-engine "cheat" that fakes 3D depth without adding vertices) instead of geometry. Adversarial verdict = PARTIALLY FEASIBLE, and it changed the design: DO NOT try to extract real fine relief from the splat (a splat's detail is largely view-dependent color, and splat-to-mesh smooths away carpet pile / brick pores). Fine detail must be SUBSTITUTED/SYNTHESIZED, not recovered.

Refined hybrid (final):
- Use the capture for overall shape (baked normal/AO) + base color (after delighting).
- DETECT surface type as a coarse bucket (masonry/fabric/granular/wood/metal/tile), shown as a ONE-TAP suggestion the user confirms/overrides (detection is only ~60-80% reliable on casual captures, never silent).
- On confirm, snap in a clean pre-delit TILEABLE PBR material (with height) from a curated ambientCG (CC0) library, tinted from the capture.
- Render with normal + POM baseline (cheap, mobile-safe = "more realism, less intensive"); optional desktop-only Nanite/tessellation "hero detail" toggle.
- Caveat to design around: POM keeps a FLAT silhouette and weakens at grazing angles; since assets are free-orbited, POM for surface detail on the game mesh, real displacement for hero close-ups.

Full detail + build steps in `PLAN.md`.

## Tech stack (mostly open, buildable now)
- Splat training: `gsplat` (Nerfstudio, Apache-2.0). NOT the original Inria code (non-commercial license). LANDMINE.
- Splat -> mesh: MILo (lightest meshes, 2025 SOTA) or 2DGS (reliable default).
- Retopology: Instant Meshes / QuadriFlow (free) or Quad Remesher (~$110; its Indie tier is non-commercial, LANDMINE).
- UV + bake + delight: xatlas for UVs, bake normal/height/albedo, plus a delighting pass.
- HDRI: guided bracketed 360 capture (real, ship first); AI-estimated lighting later, labeled "approximate."
- Formats: `.spz` (compressed splat) + glTF/GLB (becoming the Khronos standard, Feb 2026).
- Compute: phone-first capture (Scaniverse proves full on-device splatting on modern iPhones); heaviest steps (retopo, delight, HDRI merge) on newer devices or a per-job cloud GPU (~$0.80/scene).

## Platform & delivery (decided 2026-07-09)
- **Native iOS app** (iPhone). Modern iPhones can do the heavy on-device work (Scaniverse proves full on-device splatting); heaviest steps may still offload to a per-job cloud GPU. On-device Rust/wgpu path = Brush.
- **Distribution = sideload via Bottle** (the user's own iOS sideloading stack at `C:\Users\Undea\Documents\Bottle`). Bottle signs an UNSIGNED IPA with a free Apple ID and installs over USB/relay, auto-refreshing the 7-day cert.
- **CI = GitHub Actions** (authed as `bottlemaster-22`, `repo` + `workflow` scopes). A macOS runner builds an UNSIGNED IPA (`CODE_SIGNING_ALLOWED=NO`) and publishes it as a GitHub Release asset. No Apple certs in CI (Bottle signs). See `BOTTLE_INTEGRATION.md` for the Bottle-side button.

## Open decisions
- Dynamic Textures: hybrid flavor CONFIRMED (faithful bake + smart material enhancement). Feasibility research pass in progress.
- Build approach: user wants a single Fable 5 agent workflow that builds it all in one pass, no phased hand-offs. Plan to follow once research lands.

## Build estimate
Focused 2-4 person effort, ~6-12 months to a credible v1. Mostly integration of existing open components, not fresh research (except the delighting + dynamic-textures differentiators).
