# Nimbus3D Implementation Plan

_Draft v1, 2026-07-09. Two decisions still needed from the owner (see end)._

## The honest shape of this
Nimbus3D is not one app, it is a small **system**. That matters for "build it all in one go," so it is stated up front:
- The core tech (splat training, mesh extraction, retopology, delighting, material synthesis) is overwhelmingly **Python/CUDA or desktop tools**. Almost none of it runs natively on iOS today.
- So a realistic Nimbus3D is: a **native iOS app** for capture + viewing, talking to a **GPU backend** that does the heavy conversion, plus **CI** that builds the app and **Bottle** that installs it.
- A single Fable 5 workflow can build a real, runnable end-to-end **skeleton** of this. It cannot, in one generation pass, produce a polished full product, because two stages (delighting, material detection) are genuine engineering/R&D, and on-device heavy processing is an open research problem. The plan is honest about which parts are wiring vs which are hard.

## Architecture (recommended)
Four pieces:
1. **iOS app** (Swift / SwiftUI + Metal + ARKit): guided capture (photo/video, LiDAR depth prior on Pro devices), live splat preview, the bracketed HDRI capture flow, upload, and an in-app 3D preview of the finished asset.
2. **GPU backend** (the conveyor belt): the processing pipeline below. Runs on your own PC/server or a rented cloud GPU (decision 1).
3. **CI**: GitHub Actions macOS runner builds an UNSIGNED IPA and publishes it as a Release asset. No Apple certs in CI.
4. **Distribution**: sideload via Bottle (see `BOTTLE_INTEGRATION.md`).

## The processing pipeline (backend), stage by stage
Buildability tags: [OFF-THE-SHELF] drop-in, [INTEGRATE] wire an existing open engine, [R&D] genuine engineering/quality risk.

1. Camera poses: COLMAP or GLOMAP. [INTEGRATE]
2. Train the splat: `gsplat` (Apache-licensed). [INTEGRATE]
3. Splat to dense mesh: MILo (lightest) or 2DGS (reliable). [INTEGRATE]
4. Retopology to clean low-poly: Instant Meshes / QuadriFlow (free) or Quad Remesher. [INTEGRATE]
5. UV unwrap: xatlas. [OFF-THE-SHELF]
6. Delight (strip baked shadows from color): Agisoft De-Lighter / AI delight / open reimpl. [R&D] fragile
7. **Dynamic Textures** (see below): classify surface, substitute a clean tileable PBR material, tint from capture, bake normal + height. [INTEGRATE] + [R&D] for the classifier
8. Capture the HDRI: merge the bracketed 360 into a 32-bit EXR. [OFF-THE-SHELF]
9. Export: glTF/GLB mesh + PBR maps + the HDRI, plus `.spz` splat. [OFF-THE-SHELF]

## Dynamic Textures: final design (revised by adversarial research)
The research verdict was **partially feasible**, and it changed the design in one important way: **do not try to extract real fine detail from the splat.** A splat's apparent detail is largely a trick of the light (view-dependent color), and splat-to-mesh smooths away the fine relief (carpet pile, brick pores) you most want. That micro-detail has to be **substituted or synthesized**, not recovered.

So the refined **hybrid** is:
- Use the capture for what it is good at: overall shape (baked normal/AO from the dense mesh) and **base color** (after delighting).
- **Detect** the surface type as a coarse bucket (masonry / fabric-soft / granular / wood / metal / tile) and show it as a **one-tap suggestion the user confirms or overrides** (never a silent automatic decision, because detection on casual captures is only ~60-80% reliable).
- On confirm, **snap in a clean, pre-delit, tileable PBR material** (with a height map) from a small curated **ambientCG** library (CC0, commercial-safe), then tint/scale it from the capture so it matches the real object.
- **Render** with normal map + Parallax Occlusion Mapping as the universal, cheap, mobile-safe baseline. This delivers "more realism, less intensive."
- Optional desktop-only **"hero detail"** toggle swaps POM for real UE5 Nanite / Unity tessellation displacement on close-up assets.

Honest caveat to design around: POM keeps a **flat silhouette** and its illusion weakens at grazing angles. Since Nimbus3D assets can be freely orbited, we use POM for surface detail on the exported game mesh (fine for engines) and reserve real displacement for hero close-ups.

## v1 scope (narrow to win)
- **Single, opaque, matte-ish props.** Not hero characters, not glass/mirrors.
- Flow: capture on iPhone -> backend converts -> clean low-poly mesh + delit PBR (with substitution-based Dynamic Textures) + a real bracketed HDRI -> download + preview in-app.
- Nimbus3D itself is delivered by sideloading the app through Bottle.

## What the FIRST Fable 5 workflow builds (proposal)
One workflow, internally parallel (you do not manage stages), producing a runnable end-to-end skeleton:
- iOS app scaffold: capture + upload + finished-asset viewer.
- Backend service scaffold with the pipeline wired end-to-end using the open engines above, containerized; the two fragile stages (delight, material detect) ship as a working baseline to be hardened later.
- GitHub Actions CI building the unsigned IPA + the Bottle hooks.
- Result: a capture actually travels all the way to a downloadable asset, even at v0 quality.

## The two decisions I need from you
1. **Where the heavy GPU processing runs** (your own PC/server, a rented cloud GPU, or fully on-device). This sets the whole backend.
2. **What the first big build produces** (the full end-to-end skeleton, or backend-first, or app-first).

## Honest risks
- One generation pass yields a **working skeleton to harden**, not a finished product. Anyone promising otherwise for a system this size is bluffing.
- **Delighting** and **material detection** are the two quality-risk stages; budget iteration there.
- **On-device heavy processing** is the biggest unknown; today it implies porting CUDA tools to Metal/Rust, which is a large effort.
- Free-orbit + POM silhouette limit (mitigated by the hero-displacement toggle).
