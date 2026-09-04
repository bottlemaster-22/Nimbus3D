# Nimbus3D Blender add-on

Optional. Nothing about Nimbus3D on the phone needs this. It exists so a scan
can be dropped into Blender for a look, a render, or as a modelling
reference, without going through a third-party converter first.

"Nimbus3D" is a working name and not final -- see `brand.py`, the one file
in this add-on allowed to say the product name; if the product is renamed,
that's the only file to touch.

## What it does

*File > Import > Nimbus3D Splat (.ply / .spz)* reads a splat file and:

1. Parses it into a point cloud (position, per-point scale, rotation,
   colour, opacity -- see "PLY convention" below).
2. Builds a Blender mesh with one vertex per splat, and four named
   point-domain attributes (`splat_scale`, `splat_rotation_euler`,
   `splat_color`, `splat_opacity`) holding the rest of each splat's data.
3. Adds a Geometry Nodes modifier (built in Python at import time, not a
   shipped `.blend` asset) that instances a small oriented, per-axis-scaled
   ico-sphere at every point, using those attributes.
4. Applies a shared unlit, alpha-blended material that reads
   `splat_color` / `splat_opacity` straight off the geometry.

The result renders and orbits in the viewport immediately, in EEVEE or
Cycles, with no further setup.

## How this differs from a real Gaussian-splat renderer (read this)

This is **not** a Gaussian rasterizer. Blender's renderers don't have an
anisotropic-Gaussian primitive, and writing one would mean a custom Cycles
OSL/compiled kernel -- out of scope for an import add-on. What you get
instead:

- **Solid ico-spheres, not soft Gaussian falloff.** Each splat is a small
  low-poly ellipsoid (an ico-sphere squashed per-axis by the splat's own
  scale, oriented by its own rotation), not a soft alpha-weighted blob.
  A near-planar splat (two large scale axes, one tiny one -- the common
  case after training) reads as a thin oriented disc, which is where "an
  approximated splat point cloud" comes from. It will not look identical to
  Nimbus3D's own on-device/Booster preview, or to a real-time splat viewer.
- **Only the flat (SH degree-0/DC) colour is shown.** Higher
  spherical-harmonic bands (view-dependent colour, i.e. how a splat's
  colour shifts as you move the camera) are parsed and available on the
  cloud object in memory, but are **not** written onto the mesh or
  evaluated by the shipped shader -- that needs a real per-shader SH
  evaluation against the camera vector, genuine node work and out of scope
  here for a first pass. Flagged, not faked.
- **No depth sorting.** Overlapping translucent discs are composited in
  whatever order Blender's alpha-hashed (or blend) mode gives you, not
  splatting's proper back-to-front alpha compositing. Fine for a look/
  reference pass; not a substitute for a real splat renderer's output.
- **Large clouds are large meshes.** A few hundred thousand splats becomes
  a few hundred thousand ico-spheres once Geometry Nodes realizes the
  instances (12 verts / 20 faces each at the default subdivision level) --
  a multi-million-splat house scan will be heavy in the viewport. The
  modifier exposes an **Instance Subdivisions** input (drop it to 0 for a
  much cheaper flat-shaded tetrahedron-ish stand-in) and a **Scale
  Multiplier** input for a quick size check.

If you want an accurate, real-time Gaussian-splat viewport inside Blender,
the existing third-party options do that job properly and this add-on does
not try to compete with them:

- **KIRI Innovations' "3D Gaussian Splatting" Blender add-on** -- a
  dedicated splat renderer/importer with a proper EEVEE-based approximation
  and more complete SH support.
- **`https://github.com/aras-p` / community PLY-to-mesh and splat-viewer
  tools**, and web viewers such as `antimatter15/splat` or PlayCanvas'
  SuperSplat, for viewing/converting outside Blender entirely.

Use this add-on when the point is "get my Nimbus3D scan into my Blender
scene quickly, for reference or a render pass," not when you need a
faithful real-time splat preview.

## Data formats

### Coordinate frames (read this if anything looks mirrored or tilted)

`docs/DATA_FORMAT.md` section 2 (now on disk -- it did not exist when this
add-on was first written) fixes three different frames: the world/"splat
storage" frame is RUB (`+X` right, `+Y` up, `+Z` back); `.spz` and `.glb`
use that frame as-is; `.ply` alone is written in the INRIA/gsplat
reference's RDF convention instead (`+X` right, `+Y` down, `+Z` forward),
flipped on the way out by `Sources/Export/PLYCodec.swift`. Blender's own
world is `+Z` up. `splat_data.py` converts both `.ply` (RDF) and `.spz`
(RUB) positions into Blender's frame, **and applies the same rotation to
each splat's orientation quaternion** before it is stored as Euler angles
-- get the position right and the rotation wrong (or vice versa) and every
splat lands in the right place tilted 90 degrees, which looks like a minor
rendering glitch, not the coordinate-frame bug it actually is.
`selftest.py` asserts both the position AND the rotation conversion
against known values, separately for `.ply` and `.spz` (they rotate in
opposite directions), specifically to catch that class of bug.

### `.ply`

Parses the PLY header generically (whatever `element vertex` properties are
declared, in whatever order) rather than assuming a fixed byte layout --
see `ply_reader.py`. It then looks for the de-facto INRIA / gsplat 3D
Gaussian Splatting property names used by essentially every public splat
tool: `x y z`, `scale_0..2` (natural-log scale), `rot_0..3` (unit
quaternion, `w x y z` order), `opacity` (pre-sigmoid logit), `f_dc_0..2`
(degree-0 SH colour), `f_rest_*` (higher SH bands). A PLY that has
positions but none of the Gaussian-specific properties (e.g. a plain
COLMAP `points3D`-style cloud) still imports, as a small-fixed-radius,
optionally vertex-coloured point cloud, rather than being rejected.

`docs/DATA_FORMAT.md` and `ios/Sources/Export/PLYCodec.swift` now exist and
were checked against this convention: no change was needed, the property
names and quantisation already matched. If that ever changes, the fix is
contained to the handful of named constants at the top of `splat_data.py`
-- every other file in this add-on is written against the resulting
`SplatCloud` shape, not against PLY property names.

### `.spz`

Decoder for Niantic's open `.spz` container (`github.com/nianticlabs/spz`),
now reconciled formula-by-formula against `ios/Sources/Export/SPZCodec.swift`
(our own exporter, itself verified against Niantic's C++ reference) instead
of only the format's public docs. That reconciliation found and fixed four
real bugs an earlier draft had: it only accepted versions 1-2 (our exporter
writes version 3 by default -- every real file would have been rejected),
it only implemented the "first three" quaternion decoding (version 3 uses
"smallest three" -- every real rotation would have been silently wrong, not
rejected), it treated the stored colour byte as an already-finished 0..1
colour instead of a quantised SH-DC coefficient needing the same
`colorScale`/`SH_C0` dequantisation the `.ply` path already applies (colours
would have been visibly washed out), and it read the SH-rest bands as
signed int8 instead of unsigned-with-128-bias (wrong scale and zero point,
though not visible today since those bands aren't rendered yet). All four
are fixed; `spz_reader.py`'s module docstring has the full detail, and
`selftest.py` now round-trips a version-3 file (what our exporter actually
writes) with known position/rotation/colour values and asserts the decoded
result against them, plus a version-1 (float16-position) file for that
separate code path.

**Honesty note, still true:** none of this has been verified against a
byte-exact REAL `.spz` file -- there is no macOS/Blender-side export
pipeline reachable from this environment to produce one. Every formula was
cross-checked against `SPZCodec.swift`'s source text and independently
re-derived for the test fixtures, not against an actual reference file.
Treat `.spz` import as **PARTIAL**; prefer `.ply`, which needed no such
reconciliation.

Either format, anything structurally wrong (bad magic, truncated data,
an unreadable header) is rejected with a specific reported error rather
than silently importing a wrong or partial cloud.

## Installation (Blender 4.x)

This ships as a classic add-on package (a `bl_info` dict + `register()` /
`unregister()`), which every 4.x release still supports installing from a
zip via *Edit > Preferences > Add-ons > Install*, alongside newer 4.2+
Extension-platform add-ons.

1. Zip the `blender_addon/` folder's contents (not the folder itself) as
   `nimbus3d_blender_addon.zip`, or point Blender's Install dialog at this
   folder directly.
2. *Edit > Preferences > Add-ons > Install*, pick the zip, enable
   "Nimbus3D" in the list.
3. *File > Import > Nimbus3D Splat (.ply / .spz)*.

## How this was tested

There is no macOS build machine in this environment and no real Nimbus3D
export yet to test against, but Blender itself **is** installed here
(Blender 5.2.0 LTS, `C:\Program Files\Blender Foundation\Blender 5.2\`),
so this add-on is tested for real rather than only read for plausibility.
`selftest.py`, checked into this directory (not shipped as part of the
installable add-on zip -- it's a developer/CI tool, not something
`__init__.py` imports), is a headless, reproducible self-test:

```
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python selftest.py
```

It registers the real add-on package inside an actual Blender process,
builds synthetic `.ply` and `.spz` files by hand from each format's own
documented byte layout, imports them through the real
`bpy.ops.import_scene.*` operator, and asserts on the real, evaluated
result -- not just that no exception was thrown. It currently makes
**43 assertions, all passing**, most recently run and confirmed against
Blender 5.2.0 LTS. What it covers:

- the operator registers under `bpy.ops` and the import menu, and
  unregisters cleanly,
- a full Gaussian-splat PLY (position, log-scale, quaternion rotation,
  opacity logit, degree-1 SH) imports with the right vertex count and all
  four expected per-point attributes, and opacity is correctly
  sigmoid-dequantised from its stored logit,
- **the RDF -> Blender coordinate-frame conversion is checked against known
  values, for both position and orientation separately** -- a synthetic
  PLY with known vertex positions and an identity rotation is imported and
  the resulting mesh vertex coordinates AND the `splat_rotation_euler`
  attribute are compared against the exact expected `(x, z, -y)` position
  and the resulting Euler angles, not just "an object was created",
- **the Geometry Nodes modifier is evaluated through the dependency
  graph** and produces real instanced ico-sphere geometry (not just "the
  modifier exists" -- the evaluated mesh's vertex/face counts are checked
  against the expected per-instance ico-sphere counts for the default
  subdivision level),
- the splat material is actually present on the *evaluated* mesh (a
  Geometry-Nodes "Set Material" assignment doesn't populate
  `object.data.materials`, only the realized mesh -- the test checks the
  place where it actually matters) and its node graph contains the
  expected Emission/Transparent/Mix/Attribute/Output nodes,
- re-importing the same file does not duplicate the shared node-group or
  material datablocks,
- a version-3 `.spz` file (24-bit fixed positions, "smallest three"
  quaternion compression, quantised SH-DC colour -- what our own exporter
  actually writes) imports with the right vertex count, and its RUB ->
  Blender position/rotation conversion AND its colour dequantisation
  (`colorScale` + `SH_C0`, not a raw byte) are each checked against known
  values the same way the PLY case is,
- a version-1 `.spz` file (float16 positions, a different codec path
  entirely) imports and decodes its positions correctly,
- a plain (non-Gaussian) point-cloud PLY -- positions plus `uchar`
  vertex colour only, no `scale_*`/`rot_*`/`opacity`/`f_dc_*` -- takes the
  "degrades gracefully" path and still imports, with the uchar colour
  correctly dequantised to `0..1`,
- a zero-vertex PLY and two structurally-corrupt files (bad PLY magic,
  bad `.spz` magic) are each rejected with a specific, on-topic error
  message and create no object -- not a crash, and not a silently
  half-imported cloud. (Note: calling an operator directly via
  `bpy.ops.category.name(...)` and having it report an `ERROR` plus
  return `CANCELLED` makes *Blender itself* raise that report as a
  Python `RuntimeError` -- that's `bpy.ops`'s own standard behaviour, not
  a defect in this add-on; the test accounts for it rather than treating
  it as a bare pass/fail on "no exception".)

What is **not** tested: real exported Nimbus3D data (none exists yet,
so every synthetic file above was built directly from each format's
published spec, not from a real export), Blender's actual UI (headless
mode only -- the file browser, menu clicks, and viewport rendering are
not exercised), and Blender versions below 5.2 (the node-group interface
API this add-on uses was introduced in 4.0 and is unlikely to have
changed before 5.2, but that is inference, not a claim of having run it
on an earlier 4.x point release).

## Files

| File | What it does |
|---|---|
| `brand.py` | The one place the product name/id live. |
| `ply_reader.py` | Generic PLY header + body parser (no assumed property layout). |
| `spz_reader.py` | Best-effort `.spz` container decoder. |
| `splat_data.py` | Turns either parser's output into one `SplatCloud` shape. |
| `geonode_setup.py` | Builds the Geometry Nodes group (disc/ellipsoid instancing). |
| `material_setup.py` | Builds the unlit, alpha-blended splat material. |
| `operators.py` | The File > Import operator and menu entry. |
| `__init__.py` | `bl_info`, `register()` / `unregister()`. |
| `selftest.py` | Headless, real-Blender self-test (dev/CI tool, not shipped in the add-on zip). |
