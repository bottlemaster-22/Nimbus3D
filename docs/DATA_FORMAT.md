# Capture data format

What one scan looks like on disk: a COLMAP-compatible text model plus the
sidecars that make this app smart rather than generic.

This document is **normative** for every filename, byte layout and coordinate
convention below. `ios/Sources/Core/Contracts.swift` is its Swift binding;
`ios/Sources/Core/BrandConfig.swift` owns the folder names.

## Why COLMAP-compatible at all

Nothing in this app runs COLMAP, and nothing should: reconstructing poses from
scratch throws away a good ARKit prior, and triangulate-then-bundle-adjust from
that prior measurably *degraded* it in 15 of 15 test rooms. The format is
COLMAP-shaped for a much duller reason - it is the lingua franca of every
splat and photogrammetry tool in existence, so a scan from this app can be
opened by something else without a converter, and a scan from something else
can be dropped in here.

The sidecars are where the actual value is. A COLMAP model records where the
camera was. The sidecars record how much to believe it, what the laser
actually measured, where the beam went through empty air, which pixels are
glass, and how hard the phone was shaking - and that is what separates a good
scan from a fog of splats.

---

## 1. Directory layout

A scan lives at `Documents/<brand>/Scans/<scanID>/` on the phone, and at
`Documents/<brand>/Scans/Incoming|Completed/<scanID>/` on the PC Booster. The
tree is identical in both places; the Booster receives a byte-for-byte mirror.

```
<scanID>/
  capture_bundle.json          the index of everything below (Core.CaptureBundle)
  images/                      RGB frames
    frame_20260903_141205_512.jpg
  sensor_data/                 per-frame sensor sidecars
    frames.jsonl               append-only live log, one record per frame
    depth/
      frame_20260903_141205_512.depth16
    confidence/
      frame_20260903_141205_512.conf8
  sparse/
    0/                         COLMAP text model, RAW ARKit poses
      cameras.txt
      images.txt
      points3D.txt
  anchors/
    anchors_session.json       anchors as logged live
    anchors_final.json         the same anchors re-read after the session ended
  mesh/
    mesh_index.json
    chunk_0000.ply             ARKit scene mesh geometry
    chunk_0000.cls             one classification byte per face
  prepass/                     everything the no-training pre-pass produced
    prepass_result.json        the index (Core.PrePassResult)
    census.json                what every pre-pass stage counted (diagnostic)
    sparse_refined/
      cameras.txt              identical to sparse/0/cameras.txt
      images.txt               REFINED poses
      points3D.txt             carved / filtered cloud
    occupancy.bin
    trust_bias.bin
    trust_noise.bin
    depth_affine.bin
    confidence_recal.bin
    edges/
      frame_20260903_141205_512.edge8
    init_splats.ply
    init_splats.flags
  model/                       the trained result (local or from the Booster)
    model.json                 (Core.SplatModel)
    model.ply
    model.spz
    background.bin
    background.json
    exposure.bin
    observed_directions.bin
    held_out_frames.json
    train_census.json        where the trainer's geometry went (diagnostic)
    census.json                the flat build record the review screen reads
  export/                      user-facing files
    <scanID>.ply
    <scanID>.spz
    <scanID>.glb
    <scanID>-capture-bundle.zip
  cache/                       scratch, safe to delete at any moment
```

The folder names come from `BrandConfig.Folder` and do **not** change when the
product is renamed - they are part of this format, not part of the brand.

Every path recorded inside a JSON file is **relative to the scan root**, POSIX,
forward-slash separated. There is no absolute path anywhere in a scan, which is
what lets the whole folder be zipped, sent over the LAN, and opened on a PC.

### `<scanID>`

```
scan_YYYYMMDD_HHMMSS
```

Local device time at the moment capture started, e.g.
`scan_20260903_141150`. Unique per device; the app appends `_2`, `_3` … in the
astronomically unlikely case of a collision.

### Frame stamp

Every per-frame file across every folder shares one stamp so they can be joined
by filename alone:

```
frame_YYYYMMDD_HHMMSS_mmm
```

`mmm` is milliseconds, zero-padded. `frame_20260903_141205_512` is 14:12:05.512
local time. The stamp is derived from the frame's own capture timestamp, not
from when it was written to disk.

Two frames can in principle land in the same millisecond at 60 fps + a bracket
exposure; if that happens the second one gets `_512b`, `_512c`. The
authoritative ordering is always `CaptureFrame.index`, never the filename.

---

## 2. Coordinate conventions

Get these wrong and everything downstream is subtly, expensively broken.

**World frame.** Right-handed, gravity-aligned, **Y vertical (up)**, metres.
This is ARKit's world frame, used unchanged. The origin is where the session
started.

**Camera frame.** `+X` right, `+Y` **down**, `+Z` **forward** (into the scene).
This is the COLMAP / OpenCV convention. It is **not** ARKit's (`+Y` up, `-Z`
forward).

**A pose is world -> camera:**

```
X_cam = R * X_world + t
```

so the camera centre in world space is `C = -Rᵀ t`.

**Projection** (single shared PINHOLE camera, no distortion):

```
u = fx * (Xc / Zc) + cx
v = fy * (Yc / Zc) + cy       with Zc > 0 in front of the camera
```

**The one conversion.** ARKit gives `ARCamera.transform`, a camera->world
matrix in ARKit's camera convention. Converting it is a 180 degree rotation
about the camera X axis, applied camera-side, then inverted:

```
T_world_cvcam = ARCamera.transform * diag(1, -1, -1, 1)
pose          = inverse(T_world_cvcam)
```

This lives in exactly one function, `Pose.fromARKitCameraTransform(_:)` in
`Core/Contracts.swift`. Nowhere else in this project may negate an axis. If you
find yourself writing a stray minus sign to make something line up, the bug is
somewhere else.

**Splat storage frame.** RUB - `+X` right, `+Y` up, `+Z` back. That is the
world frame with no rotation applied, so a trained splat needs no conversion
for `.spz` or `.glb`. Only `.ply` differs (the INRIA reference convention is
RDF), and that flip is applied inside `PLYCodec`, on the way out and back.

---

## 3. `capture_bundle.json`

The index of the whole capture, written when the session ends. Its Swift type
is `Core.CaptureBundle`; the encoding is UTF-8 JSON with **ISO-8601 dates to
whole seconds** and sorted keys (`Core.ContractsJSON`).

`sensor_data/frames.jsonl` is the crash-safe live log written during capture -
one JSON `CaptureFrame` object per line, appended as each frame lands. If the
app is killed mid-capture, the `.jsonl` is what survives, and
`capture_bundle.json` can be rebuilt from it. When both exist,
`capture_bundle.json` wins.

Abbreviated example (comments are not legal JSON, they are here for reading):

```jsonc
{
  "formatVersion": 1,
  "scanID": "scan_20260903_141150",
  "createdAt": "2026-09-03T14:11:50Z",
  "displayName": "Living room",
  "deviceModel": "iPhone17,2",
  "appVersion": "0.1.0 (1)",

  "intrinsics": { "width": 1920, "height": 1440,
                  "fx": 1454.2, "fy": 1454.2, "cx": 959.5, "cy": 719.5 },

  "settings": {
    "bracketEveryNFrames": 12,      // 0 = bracketing was off
    "bracketStops": 3.0,
    "exposureLocked": false,
    "whiteBalanceLocked": false,
    "depthWidth": 256,              // measured, never assumed
    "depthHeight": 192,
    "lidarMaxRangeMeters": 5.0
  },

  // The calibrated camera-to-IMU offset, seconds. ADD it to a camera
  // timestamp to get the IMU timestamp that really matches it. null when
  // the sweep found no clear minimum - which is reported, not hidden.
  "cameraToIMUTimeOffsetSeconds": 0.0135,

  "frames": [
    {
      "index": 0,
      "timestampSeconds": 71234.512,       // device mach-continuous clock
      "imagePath": "images/frame_20260903_141205_512.jpg",
      "depthPath": "sensor_data/depth/frame_20260903_141205_512.depth16",
      "confidencePath": "sensor_data/confidence/frame_20260903_141205_512.conf8",
      "rawPose": { "rotation": [0.0, 0.0, 0.0, 1.0],
                   "translation": [0.0, 0.0, 0.0] },
      "refinedPose": null,                 // filled in by the pre-pass
      "exposureDurationSeconds": 0.0083,
      "exposureOffsetEV": 0.0,
      "iso": 320.0,                        // null when unobtainable
      "angularVelocity": [0.01, -0.22, 0.03],   // rad/s, device body frame
      "qc": {
        "angularSpeedRadPerSec": 0.223,
        "motionBlurPixels": 2.49,
        "sharpness": 0.81,
        "depthValidFraction": 0.93,
        "exposureJumpEV": 0.0,
        "trackingQuality": "normal",
        "weight": 0.74
      },
      "bracket": "normal",                 // or "darker"
      "submap": null                       // assigned by the pre-pass
    }
  ],

  "anchorsDuringSession": [ … ],
  "anchorsAtEndOfSession": [ … ],
  "meshChunks": [ … ],
  "revisitPairs": [ … ],
  "sceneBounds": { "min": [-2.1, -0.1, -3.4], "max": [3.0, 2.6, 1.2] },
  "pointCloudPath": "sparse/0/points3D.txt"
}
```

Note the compact array encodings: a `Vector3` is `[x, y, z]` and a
`Quaternion` is `[x, y, z, w]`. A four-thousand-frame house walk has twelve
thousand of these; spelling out `{"x": …, "y": …, "z": …}` would make the
sidecar several times larger than the useful data in it.

**Quaternion order is a classic source of silent breakage.** In memory and in
JSON it is `(x, y, z, w)`. COLMAP's `images.txt` writes `QW QX QY QZ`. The PLY
splat convention writes `rot_0..rot_3` as `(w, x, y, z)`. Each writer does its
own re-ordering; nothing re-orders in memory.

---

## 4. The COLMAP model

Standard COLMAP **text** format, readable by COLMAP, `pycolmap`, nerfstudio,
gsplat and every splat trainer worth using.

### `sparse/0/cameras.txt`

Exactly one camera, shared by every image.

```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 1920 1440 1454.203 1454.203 959.5 719.5
```

`PINHOLE` (not `SIMPLE_PINHOLE`, not `OPENCV`): ARKit hands back already
rectified frames with independent `fx`/`fy` and no usable distortion
coefficients, so four parameters is exactly right and any distortion model
would be inventing numbers.

### `sparse/0/images.txt`

**Raw ARKit poses.** This file is written once and never rewritten - the raw
VIO track is evidence, and any refinement has to stay auditable against it.

```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 0.998531 0.001204 -0.054118 0.000772 0.014200 -0.003100 0.000000 1 frame_20260903_141205_512.jpg

2 0.998012 0.002011 -0.062940 0.001103 0.031000 -0.004900 0.002100 1 frame_20260903_141209_012.jpg

```

* `IMAGE_ID` is `CaptureFrame.index + 1` (COLMAP ids are 1-based).
* `QW QX QY QZ TX TY TZ` is the **world -> camera** pose, matching COLMAP's own
  definition, and matching `Pose` after the ARKit conversion in section 2.
* `NAME` is the image's filename relative to `images/`, not a path.
* **The second line of each record is empty**, always. This app does not
  triangulate 2D-3D correspondences, so there are no `POINTS2D`. COLMAP's
  reader requires the line to be there; omitting it makes the file unparseable
  by half the tools that claim to read it.

### `sparse/0/points3D.txt`

The LiDAR cloud, baked in, so a downstream tool gets real metric geometry
instead of an empty initialisation.

```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 1.204000 0.881000 -2.310000 148 132 119 0.008000
```

* Colour is sampled from the frame the point was measured in.
* `ERROR` is the **expected metric error in metres**, from the physics prior:
  it grows with range and with grazing incidence angle. This is not COLMAP's
  reprojection error, and it is a deliberate reinterpretation - it is the only
  per-point uncertainty slot the format has, and leaving it at 0 would be a
  lie. It is documented here so nothing downstream mistakes it for pixels.
* `TRACK` is empty for the same reason `POINTS2D` is.
* The cloud is voxel-downsampled at 1 cm before writing; a raw union of every
  frame's 49k samples over a four-minute walk is hundreds of millions of
  redundant points.

### `prepass/sparse_refined/`

The same three files after pose refinement, with `images.txt` carrying the
refined poses and `points3D.txt` carrying the carved and filtered cloud.
`cameras.txt` is byte-identical to `sparse/0/cameras.txt`.

A whole second model rather than an `images_refined.txt` sidecar, because that
way any COLMAP-reading tool can be pointed at `prepass/sparse_refined/` and
just work, with no awareness of this app at all.

---

## 5. Sensor sidecars

### `sensor_data/depth/frame_*.depth16`

**The native LiDAR depth map at its true resolution.** 256 x 192 on every LiDAR
iPhone to date, but the real dimensions are recorded in
`settings.depthWidth`/`depthHeight` and must be read from there, not assumed.

| Property | Value |
|---|---|
| Layout | row-major, top-left origin, no header, no padding |
| Element | `UInt16`, **little-endian** |
| Unit | millimetres |
| Size | `depthWidth * depthHeight * 2` bytes (98304 at 256x192) |
| No return | `0` |

This is `ARFrame.sceneDepth.depthMap`, converted from `Float32` metres to
`UInt16` millimetres. It is **never** `smoothedSceneDepth`, and it is never the
upsampled map ARKit will happily hand you at RGB resolution. The whole of F3
depends on this: depth supervision happens only at these ~49k real samples per
frame, because everything between them is interpolation, and training against
interpolation teaches the model to reproduce the interpolator.

`0` means the beam came back with nothing. That is not "distance zero" and it
is not "empty" either: it is a no-return, and section 6 explains why the
difference matters at a window.

### `sensor_data/confidence/frame_*.conf8`

| Property | Value |
|---|---|
| Layout | row-major, same dimensions as the depth map |
| Element | `UInt8` |
| Values | `0` low, `1` medium, `2` high (`ARConfidenceLevel`) |
| Size | `depthWidth * depthHeight` bytes (49152 at 256x192) |

Store it, but do not believe it. ARKit flags roughly 1.6% of samples as low and
the three levels are not calibrated to anything - a "high" sample next to a
window is routinely worse than a "low" one on a matte wall. The pre-pass treats
these values as a **ranking** and remaps them through observed revisit
residuals into an actual probability, written to
`prepass/confidence_recal.bin`. Consumers should prefer the recalibrated file
and fall back to this one only if it is absent.

### `sensor_data/frames.jsonl`

One JSON object per line, each one a `CaptureFrame`, appended live. Newline
delimited (`\n`, not `\r\n`), UTF-8, no trailing comma, no enclosing array.

Purpose is crash-safety: an append-only log survives a kill -9 in a way that a
single large JSON document rewritten every frame does not.

### `images/frame_*.jpg`

JPEG, quality 0.92, at the capture resolution in `intrinsics`. sRGB, no EXIF
orientation games - pixels are stored in the same orientation the intrinsics
describe.

Bracketed frames (F5) are ordinary JPEGs in the same folder; the only thing
marking them is `"bracket": "darker"` on their `CaptureFrame`, plus the
exposure fields. Do not filter them out of a photometric loss - they are the
only correctly-exposed evidence you have for a bright window - but do use the
per-frame exposure so they are compared on equal terms.

---

## 6. ARKit anchors and mesh

### `anchors/anchors_session.json`, `anchors/anchors_final.json`

Both are a JSON array of `Core.AnchorRecord`:

```json
[
  { "identifier": "9C1F…-UUID",
    "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 1.2,0.8,-2.3,1],
    "firstSeenFrame": 42,
    "classification": "wall",
    "isUserMarked": false }
]
```

`isUserMarked` is provenance, not a second opinion. It is `true` only for an
anchor a person placed by tapping "that is a window" in window mode;
`classification` reads `window` either way. The field is optional on read: a
file written before it existed decodes as `false`, which is why adding it did
not bump `formatVersion`.

`transform` is 16 floats, **column-major**, anchor -> world, in ARKit's own
convention, unmodified.

The two files exist because ARKit silently moves anchors when it relocalises.
`anchors_session.json` is what was logged as the walk happened;
`anchors_final.json` is the same anchors re-read once the session ended.
Differencing them is a free, direct measurement of how far the map slid, with
no extra computation and no assumptions - and it is one of the inputs to the
QC card's drift number.

### `mesh/`

`chunk_NNNN.ply` is a binary little-endian PLY with `vertex` (x, y, z, nx, ny,
nz) and `face` elements, in world space.

`chunk_NNNN.cls` is one byte per face, in the same order as the PLY's face
list, holding a `SurfaceClass` value:

```
0 none   1 wall   2 floor   3 ceiling   4 table
5 seat   6 window 7 door    8 glass     9 sky
```

`8 glass` and `9 sky` are ours, not ARKit's - the glass detector (F5) writes
them when it finds a LiDAR-silent, image-bright, planar region inside an
otherwise planar wall.

`mesh_index.json` is a JSON array of `Core.MeshChunkRef` listing every chunk,
its face count and its bounds.

---

## 7. Pre-pass outputs

`prepass/prepass_result.json` (`Core.PrePassResult`) indexes all of it. The
binary files are all little-endian and headerless; their shape lives in the
JSON, so nothing has to guess.

### `prepass/occupancy.bin` - free-space carving (F2)

A sorted, flat run of records:

| Offset | Type | Meaning |
|---|---|---|
| 0 | `UInt64` | Morton (Z-order) key of the voxel |
| 8 | `UInt8` | state: `0` unknown, `1` empty, `2` surface |
| 9 | `UInt8` | reserved, `0` |
| 10 | `UInt16` | hit count, saturating |

12 bytes per record, sorted ascending by key, so a lookup is a binary search
and a merge is a linear scan. Voxel size, grid origin and record count are in
`prepass_result.json`.

**A cell not present in the file is `unknown`, and `unknown` is not `empty`.**
This is the entire point of the structure. A cell is `empty` only when a LiDAR
beam demonstrably passed *through* it and returned from something further away.
Space beyond the sensor's range, and space behind a no-return, stays `unknown`
forever. Deleting a Gaussian is licensed only by `empty`, never by `unknown` -
which is what stops the carver from erasing everything visible through a
window, since glass returns nothing and "nothing came back" is not evidence of
absence.

The useful exception: a no-return ray still certifies the space between the
sensor and the *known* wall plane it passes through as empty, which is how the
free-space bound at a window is recovered without inventing geometry beyond it.

### `prepass/trust_bias.bin` - coarse bias field (F6)

| Offset | Type | Meaning |
|---|---|---|
| 0 | `UInt64` | Morton key, at 25-50 cm voxels |
| 8 | `Float32` | running mean **signed** depth residual, metres |
| 12 | `Float32` | variance of that residual |
| 16 | `UInt32` | sample count |
| 20 | `UInt16` | number of **distinct capture times** that contributed |
| 22 | `UInt16` | reserved |

24 bytes per record. The distinct-times count is what separates "measured
repeatedly from one pass" from "confirmed on separate visits", and only the
latter earns real trust.

### `prepass/trust_noise.bin` - per-sample noise (F6)

One `Float32` per native depth sample, frame-major, in the same order as the
depth sidecars: frame 0's `width*height` samples, then frame 1's, and so on.

**This file is never spatially averaged.** That is not an optimisation note, it
is the contract: averaging an outlier into its neighbours is exactly how a
wrong surface becomes a confident wrong surface.

### `prepass/depth_affine.bin`

`(UInt32 frameIndex, Float32 scale, Float32 shift)`, 12 bytes per record, one
per frame that has a learned correction. Tightly constrained around
`scale = 1, shift = 0`; a large correction means something else is wrong.

### `prepass/confidence_recal.bin`

One `Float32` in `0...1` per native depth sample, frame-major, same ordering as
`trust_noise.bin`. The recalibrated replacement for `.conf8`.

### `prepass/edges/frame_*.edge8` - edge classification (F3)

One byte per **native depth** pixel (not per RGB pixel), row-major:

```
0 none        not an edge
1 geometric   a real depth step: sharpen here
2 texture     strong image gradient, flat depth: flatten, do not invent geometry
3 unknown     glass, out of range, or low confidence: ignore, do not guess
4 band        inside the dilation around a geometric edge: depth loss is zeroed
```

Edges are computed on the **native** map and then dilated by the upsample
ratio. At 256 native columns against 1920 RGB columns that ratio is 7.5, giving
the ~8 px band the design calls for, expressed at RGB resolution. Class `4` is
the band itself, and the reason it exists is that the upsampled depth map is
simply wrong within it - supervising there teaches the model the interpolator's
smearing, which is worse than no supervision at all.

### `prepass/init_splats.ply` and `.flags`

The initial Gaussian set, in the same PLY layout `Sources/Export/PLYCodec.swift`
reads and writes (INRIA/gsplat property names: `x y z`, `nx ny nz`,
`f_dc_0..2`, `f_rest_*`, `opacity`, `scale_0..2`, `rot_0..3`), so it round-trips
through the same reader as everything else.

`.flags` is one byte per splat, in PLY vertex order:

| Bit | Meaning |
|---|---|
| 0 | position pinned (trusted sample: thin, opaque, held in place) |
| 1 | elongated along the viewing ray (doubtful: translucent, free to slide) |
| 2 | lies on a detected 3D edge curve - exempt from the disc/effective-rank prior |

---

## 8. The trained model

`model/model.json` is `Core.SplatModel`. The splats themselves are in
`model/model.ply` and/or `model/model.spz`, written by `Sources/Export`:

* **`.ply`** - INRIA/gsplat convention, binary little-endian, RDF coordinate
  frame, `rot_0..3` as `(w, x, y, z)`, log scales, logit opacity, raw SH
  coefficients. Read by every splat viewer, and by `blender_addon/`.
* **`.spz`** - version 3, gzip container, RUB frame. Roughly 10x smaller. The
  quantisation constants match Niantic's reference implementation exactly.
* **`.glb`** - `KHR_gaussian_splatting`, ratified, RUB frame, with the spec's
  recommended fallback `COLOR_0` so a plain glTF viewer shows a coloured point
  cloud instead of black dots.

`model/background.bin` + `background.json` hold the frozen direction-only far
field (F5). `model/exposure.bin` holds `(UInt32 frameIndex, Float32 gain,
Float32 bias)` per frame.

`model/held_out_frames.json` is which frames the training run kept out of
training, so the viewer's photo-versus-scan slider can compare against photos
the model provably never saw. Either a bare array of frame indices,
`[10, 30, 50]`, or `{"frames": [10, 30, 50]}`; readers accept both. Optional,
and its absence means something specific: nobody wrote down what was held out,
so a reader must fall back to a deterministic guess and say on screen that it
is a guess. An empty array is NOT the same thing as an absent file, and a
writer that held nothing out must omit the file rather than write `[]`.

`model/observed_directions.bin` is the honesty mask's backing store: a coarse
per-voxel bitmask of which directions each part of the scene was actually
looked at from. The viewer hatches any pixel whose viewing ray falls in an
unobserved direction. Showing the user which parts of their scan are invented
is not a nice-to-have; it is the difference between a tool and a toy.

### `model/train_census.json` - where the geometry went

Written by `Sources/Trainer` (`TrainerCensus.swift`) once per training run, at
the very end. It is a **diagnostic sidecar**: nothing reads it to render or
export anything, and its absence changes no behaviour. It exists because the
first real scan this app produced came out looking like nothing, and four
separate faults had each destroyed most of the model without one of them
logging, warning or failing. The census is the trainer measuring its own
behaviour so the next fault of that shape is a five second read instead of a
day of archaeology.

**Written even when the run fails.** The trainer writes it from a `defer`, so a
run that threw, was cancelled, or was stopped by heat still leaves one behind.
`outcome` says which: `"completed"`, `"stopped early"`, `"cancelled"` or
`"did not reach the end"`.

**It is cheap by construction.** Every field is an integer counter accumulated
in memory during the run. There is exactly one file write, at the end. Nothing
in it forces a GPU synchronisation the trainer would not have performed anyway.

Top level:

| key | what it is |
| --- | --- |
| `formatVersion` | `1`. Bumped only when a field changes MEANING. |
| `scanID`, `startedAt`, `finishedAt`, `outcome` | which run this was and how it ended |
| `alerts` | ordered worst first: `{severity, code, detail}`. The five second read. |
| `ledger` | ordered plain sentences, seeds to final count, one per stage |
| `budgetRequested` / `budgetAsRun` | the budget asked for, and the one actually run |
| `budgetReductions` | every lowering event, with its reason, the iteration, and the live splat count at that moment |
| `gates` | the schedule constants the loop actually READ, captured at the point of use |
| `slices` | one row per time slice: seeding, caps, iterations, skips, what it handed to the merge |
| `densifyPasses` | one row per densification / prune / carve pass |
| `merge` | what the slice merge kept, dropped to another owner, and trimmed |
| `keyframesSelected`, `sliceCount`, `iterationsRequested`, `iterationsCompleted`, `finalSplatCount` | the totals |

A `densifyPasses` row carries the counts AND the context that makes them
readable: `growthWindowOpen`, `pruneWindowOpen`, `carveRan`, `carverAvailable`,
`capInForce`, `headroom`, `growthAllowance`, `splatsScored`,
`splatsWithNonZeroScore`, `candidatesAfterVisibilityFilter`,
`relocationDonorsAvailable`, then `splatCountBefore`, `addedBySplit`,
`addedByClone`, `relocated`, `splatCountAfterGrowth`, `prunedNonFinite`,
`prunedLowOpacity`, `prunedOversized`, `carvedFromEmptySpace`, `trimmedToCap`
and `splatCountAfter`. Growth, pruning and carving all happen inside one pass,
so one row covers all three. `splatCountAfterGrowth` is the only derived value
and it is exact arithmetic, not an estimate: `before + split + clone`.

`alerts[].code` is stable and machine-readable, so a screen can match on it
without parsing English. The current set:

`densification_created_nothing`, `growth_window_never_open`,
`no_headroom_whenever_growth_was_allowed`,
`splat_cap_cut_below_live_population`,
`splat_cap_cut_to_exactly_the_live_population`,
`judgement_pruning_outside_its_window`,
`seeding_rejected_almost_every_sample`, `no_seed_was_trusted`,
`almost_no_seed_was_trusted`, `most_of_the_model_disappeared`,
`final_count_far_below_the_cap`, `free_space_carving_removed_the_most`,
`non_finite_points_reached_the_readback`, `merge_dropped_most_of_the_parts`,
`run_ended_before_its_budget`, `many_iterations_did_no_work`.

An empty `alerts` array means the census ran its checks and none of them
tripped. A MISSING file means nobody wrote one, which is a different and weaker
statement, and a reader must not report it as "clean".

### `prepass/census.json` and `model/census.json` - the build record

Two small, FLAT files with the same key set, each written by the stage that
owns its folder. The review screen in `Sources/Viewer` reads both, through
`ScanCensus.Record`. They exist so that "which step lost the geometry" is a
line on a screen rather than a day of reading code.

The rules are the whole design and a writer must follow them exactly:

* **Every key is optional.** Write only what was genuinely counted.
* **An absent key means "not counted"** and renders as "not recorded", naming
  the stage as the reason.
* **A key written as `0` is a MEASURED ZERO**, which is a much stronger claim.
  Never write `0` for a number nobody took. This is not a style preference: it
  is the one rule that stops this file from reproducing the fault it was built
  to catch.
* Unknown extra keys are ignored, so a writer may run ahead of the reader.
* Keys are camelCase, `formatVersion` is `1`, encoded with
  `ContractsJSON.encoder()`. A higher version is refused and the refusal is
  shown, not guessed at.

| key | written by | what it is |
| --- | --- | --- |
| `formatVersion`, `writtenBy` | both | `1`, and `"prepass"` or `"trainer"` |
| `seedsWritten` | pre-pass | starting points written to `init_splats.ply` |
| `seedsShapedAsConfidentDiscs` | pre-pass | of those, how many were pinned flat rather than stretched along the viewing ray |
| `trustGateSigmaMeters` | pre-pass | the depth accuracy the trust test demanded, metres |
| `measuredMedianSigmaMeters` | pre-pass | the accuracy this scan actually measured, metres. Written ONLY when the trust field really measured it; the physics prior is a prediction and is never reported under this name |
| `splatsAtStart` | trainer | points on the GPU when training began |
| `splatsCreatedByDensification`, `densificationPassCount`, `densificationCandidateCount` | trainer | the step that adds detail: what it created, how often it ran, how many points it considered |
| `splatsDeletedByPruning`, `pruningPassCount` | trainer | what tidying removed and in how many passes |
| `splatsDeletedByHeatCut`, `heatCutCount` | trainer | Gaussians above the ceiling at the moment a warm phone lowered it, and how many times that happened. Always written, so the screen can RULE HEAT OUT rather than say it cannot tell |
| `splatsAtEnd` | trainer | points alive when the run finished. Absent unless the run reached the merge |
| `plannedSplatCap`, `finalSplatCap` | trainer | the cap asked for and the cap ended on. `finalSplatCap` below `splatsAtEnd` is an impossible state and the screen reports it by name |

`depthSamplesOffered` and `depthSamplesAccepted` are in the reader and are
**deliberately not written by this pipeline.** They are defined as the counts
either side of a trust gate that REJECTS depth readings, and there is no such
gate: `PrePassInitialSplatBuilder` uses the trust weight to decide whether a
seed is laid as a disc or as a ray-stretched blob, never to discard a reading.
Writing the seeding funnel's numbers under those two names would put true
numbers under a false label. The funnel itself lives in
`prepass/census.json` under `seeding`, in full.

The trainer writes `model/census.json` from the SAME sealed census as
`model/train_census.json`, so the two files cannot disagree, and every
loop-derived key is left absent until at least one slice has actually put its
seeds on the GPU.

---

## 9. Compatibility notes

**Reading a scan produced by something else.** Point any COLMAP-reading tool at
`sparse/0/` and it works. Every sidecar is optional from a reader's point of
view: absent depth means no depth supervision, absent `prepass/` means the raw
poses are all there is. Nothing hard-fails on a missing sidecar; it degrades to
generic 3DGS, which is exactly what this app is trying to be better than.

**Version.** `formatVersion` is `1` in both `capture_bundle.json` and
`prepass_result.json`. A reader that sees a version it does not know must
refuse and say so, not guess. Adding an optional field does not bump it;
changing what a field means does.

**Endianness.** Every binary sidecar is little-endian. Both ends of this
project (arm64 iPhone, x86-64/arm64 PC) are little-endian, so no byte swapping
happens anywhere - but it is written down so that when it eventually matters,
nobody has to reverse-engineer it from a hex dump.
