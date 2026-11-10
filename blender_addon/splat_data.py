# =============================================================================
#  splat_data.py -- turns a raw parsed file (ply_reader / spz_reader output)
#  into one common SplatCloud the rest of the add-on builds a mesh from.
#
#  PLY property-name convention: this reader follows the de-facto INRIA /
#  gsplat 3D Gaussian Splatting PLY layout used by every public splat tool
#  (the original "3D Gaussian Splatting for Real-Time Radiance Field
#  Rendering" reference implementation, and every viewer that reads its
#  output): x,y,z position; scale_0..2 NATURAL-LOG scale; rot_0..3 unit
#  quaternion in (w,x,y,z) order; opacity as a LOGIT (pre-sigmoid); f_dc_0..2
#  the degree-0 spherical-harmonic (DC / "flat colour") coefficient per
#  channel; f_rest_* the higher SH bands, channel-major (all of R's bands,
#  then all of G's, then all of B's). This is CONFIRMED (not guessed) to
#  match our own exporter as of docs/DATA_FORMAT.md section 8 and
#  ios/Sources/Export/PLYCodec.swift, which now exist on disk.
#
#  COORDINATE FRAMES (docs/DATA_FORMAT.md section 2, cross-checked against
#  ios/Sources/Export/PLYCodec.swift and SPZCodec.swift):
#    - World / "splat storage" frame is RUB: +X right, +Y up, +Z back.
#    - `.ply` is written in the INRIA/gsplat reference's RDF convention
#      instead (+X right, +Y down, +Z forward) -- PLYCodec.swift flips Y and
#      Z on the way out, so this reader flips them back on the way in.
#    - `.spz` keeps the RUB storage frame as-is (SPZCodec.swift applies no
#      flip), so only the RUB -> Blender step below applies to it.
#  Blender's own world is right-handed with +Z up. Converting a Y-up-forward
#  world (RUB, which is exactly OpenGL/glTF's convention: +X right, +Y up,
#  +Z BACK i.e. -Z forward) into Blender's +Z-up world is the same swap
#  Blender's own glTF importer uses: blender = (x, -z, y). Composed with the
#  RDF -> RUB un-flip (y,z negated) for `.ply`, the net PLY -> Blender map is
#  (x, y, z) -> (x, z, -y). Both maps are proper rotations (a fixed +/-90
#  degree turn about the shared X axis), so the SAME transform is applied to
#  positions (a linear map) and to each splat's orientation quaternion (by
#  composing rotations) -- get one right and forget the other and every
#  splat still lands in the right PLACE but is tilted 90 degrees, which is a
#  much harder bug to notice than a gross position error.
# =============================================================================

import math

from . import ply_reader
from . import spz_reader

SH_C0 = 0.28209479177387814  # degree-0 real spherical harmonic normalisation

_SQRT1_2 = 0.7071067811865476

# Pre-rotation quaternions (w, x, y, z), applied as q_new = Q_FRAME * q_read,
# that carry the READ position transform below. Both are a 90-degree turn
# about the shared, unaffected X axis -- see the module docstring's proof.
_Q_RDF_TO_BLENDER = (_SQRT1_2, -_SQRT1_2, 0.0, 0.0)   # .ply: -90 deg about X
_Q_RUB_TO_BLENDER = (_SQRT1_2, _SQRT1_2, 0.0, 0.0)    # .spz: +90 deg about X


def _rdf_to_blender(x, y, z):
    """PLY's RDF world position -> Blender's +Z-up world position."""
    return (x, z, -y)


def _rub_to_blender(x, y, z):
    """SPZ's RUB world position -> Blender's +Z-up world position."""
    return (x, -z, y)


def _quat_multiply_wxyz(a, b):
    """Hamilton product a*b, both (w,x,y,z). a applied AFTER b (a is the
    outer/world-side rotation, matching q_new = Q_FRAME * q_read below)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


class SplatCloud:
    __slots__ = (
        "count",
        "positions",       # flat [x0,y0,z0, x1,y1,z1, ...], length count*3
        "scales",           # flat linear (already exp'd) [sx0,sy0,sz0, ...]
        "rotations_euler",  # flat XYZ Euler radians [ex0,ey0,ez0, ...]
        "colors_rgb",       # flat [r0,g0,b0, ...] in 0..1
        "opacities",        # flat [a0, ...] in 0..1
        "sh_rest",           # flat, rests_per_channel*3 per point, or []
        "sh_degree",
        "is_full_gaussian",  # False if scale/rotation were not present in the file
        "source_format",     # "ply" or "spz", for the operator's report line
    )


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _quat_wxyz_to_euler_xyz(w, x, y, z):
    """Unit quaternion (w,x,y,z) -> intrinsic XYZ Euler radians, using only
    stdlib math so this file has no dependency on `mathutils` and can be
    unit-tested outside Blender. Standard closed-form conversion.
    """
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / n, x / n, y / n, z / n

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def from_ply(filepath):
    count, props = ply_reader.read_ply_vertices(filepath)
    cloud = SplatCloud()
    cloud.count = count
    cloud.source_format = "ply"

    if not all(k in props for k in ("x", "y", "z")):
        raise ValueError(f"{filepath}: PLY has no x/y/z vertex properties")

    px, py, pz = props["x"], props["y"], props["z"]
    positions = [0.0] * (count * 3)
    for i in range(count):
        bx, by, bz = _rdf_to_blender(px[i], py[i], pz[i])
        positions[i * 3 + 0] = bx
        positions[i * 3 + 1] = by
        positions[i * 3 + 2] = bz
    cloud.positions = positions

    has_scale = all(f"scale_{i}" in props for i in range(3))
    has_rot = all(f"rot_{i}" in props for i in range(4))
    cloud.is_full_gaussian = has_scale and has_rot

    if has_scale:
        s0, s1, s2 = props["scale_0"], props["scale_1"], props["scale_2"]
        scales = [0.0] * (count * 3)
        for i in range(count):
            scales[i * 3 + 0] = math.exp(s0[i])
            scales[i * 3 + 1] = math.exp(s1[i])
            scales[i * 3 + 2] = math.exp(s2[i])
        cloud.scales = scales
    else:
        cloud.scales = [0.01] * (count * 3)  # small uniform dot, not a real splat extent

    if has_rot:
        r0, r1, r2, r3 = props["rot_0"], props["rot_1"], props["rot_2"], props["rot_3"]
        eulers = [0.0] * (count * 3)
        for i in range(count):
            # INRIA convention: rot_0..3 = w,x,y,z, in the PLY's RDF frame --
            # rotate into Blender's frame the same way positions were above
            # (see module docstring) before converting to Euler.
            qw, qx, qy, qz = _quat_multiply_wxyz(
                _Q_RDF_TO_BLENDER, (r0[i], r1[i], r2[i], r3[i])
            )
            roll, pitch, yaw = _quat_wxyz_to_euler_xyz(qw, qx, qy, qz)
            eulers[i * 3 + 0] = roll
            eulers[i * 3 + 1] = pitch
            eulers[i * 3 + 2] = yaw
        cloud.rotations_euler = eulers
    else:
        cloud.rotations_euler = [0.0] * (count * 3)

    if "opacity" in props:
        op = props["opacity"]
        cloud.opacities = [_sigmoid(op[i]) for i in range(count)]
    else:
        cloud.opacities = [1.0] * count

    has_dc = all(f"f_dc_{i}" in props for i in range(3))
    if has_dc:
        d0, d1, d2 = props["f_dc_0"], props["f_dc_1"], props["f_dc_2"]
        colors = [0.0] * (count * 3)
        for i in range(count):
            colors[i * 3 + 0] = _clamp01(0.5 + SH_C0 * d0[i])
            colors[i * 3 + 1] = _clamp01(0.5 + SH_C0 * d1[i])
            colors[i * 3 + 2] = _clamp01(0.5 + SH_C0 * d2[i])
        cloud.colors_rgb = colors
    elif all(k in props for k in ("red", "green", "blue")):
        r, g, b = props["red"], props["green"], props["blue"]
        cloud.colors_rgb = _flatten_bytes(r, g, b, count)
    elif all(k in props for k in ("r", "g", "b")):
        r, g, b = props["r"], props["g"], props["b"]
        cloud.colors_rgb = _flatten_bytes(r, g, b, count)
    else:
        cloud.colors_rgb = [0.7] * (count * 3)

    # Higher SH bands: keep them, but do not attempt to guess a property
    # naming scheme beyond f_rest_N -- if absent, sh_degree is 0.
    rest_names = sorted(
        (k for k in props if k.startswith("f_rest_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    if rest_names:
        n_rest = len(rest_names)
        rests_per_channel = n_rest // 3
        cloud.sh_degree = _degree_from_rests(rests_per_channel)
        flat = [0.0] * (count * n_rest)
        for band_i, name in enumerate(rest_names):
            col = props[name]
            for i in range(count):
                flat[i * n_rest + band_i] = col[i]
        cloud.sh_rest = flat
    else:
        cloud.sh_degree = 0
        cloud.sh_rest = []

    return cloud


def from_spz(filepath):
    raw = spz_reader.decode_spz(filepath)
    cloud = SplatCloud()
    cloud.count = raw.count
    cloud.source_format = "spz"

    positions = [0.0] * (raw.count * 3)
    for i in range(raw.count):
        x, y, z = raw.positions[i * 3 + 0], raw.positions[i * 3 + 1], raw.positions[i * 3 + 2]
        bx, by, bz = _rub_to_blender(x, y, z)
        positions[i * 3 + 0] = bx
        positions[i * 3 + 1] = by
        positions[i * 3 + 2] = bz
    cloud.positions = positions
    cloud.scales = raw.scales
    cloud.is_full_gaussian = True

    eulers = [0.0] * (raw.count * 3)
    for i, (x, y, z, w) in enumerate(raw.rotations_xyzw):
        # .spz stores (x,y,z,w) in the RUB frame -- rotate into Blender's
        # frame the same way positions were above (see module docstring)
        # before converting to Euler.
        qw, qx, qy, qz = _quat_multiply_wxyz(_Q_RUB_TO_BLENDER, (w, x, y, z))
        roll, pitch, yaw = _quat_wxyz_to_euler_xyz(qw, qx, qy, qz)
        eulers[i * 3 + 0] = roll
        eulers[i * 3 + 1] = pitch
        eulers[i * 3 + 2] = yaw
    cloud.rotations_euler = eulers

    # colors_dc is the raw (dequantised) degree-0 SH coefficient, same shape
    # as a PLY's f_dc_0..2 -- turn it into displayable colour the same way
    # from_ply does, rather than treating the quantised byte as if it were
    # already a final 0..1 colour (see spz_reader.py's colorScale note).
    colors = [0.0] * (raw.count * 3)
    for i, (dc_r, dc_g, dc_b) in enumerate(raw.colors_dc):
        colors[i * 3 + 0] = _clamp01(0.5 + SH_C0 * dc_r)
        colors[i * 3 + 1] = _clamp01(0.5 + SH_C0 * dc_g)
        colors[i * 3 + 2] = _clamp01(0.5 + SH_C0 * dc_b)
    cloud.colors_rgb = colors

    cloud.opacities = list(raw.opacities)
    cloud.sh_degree = raw.sh_degree
    cloud.sh_rest = raw.sh_rest
    return cloud


def _degree_from_rests(rests_per_channel):
    for degree, n in ((1, 3), (2, 8), (3, 15)):
        if rests_per_channel == n:
            return degree
    return 0


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _flatten_bytes(r, g, b, count):
    out = [0.0] * (count * 3)
    for i in range(count):
        out[i * 3 + 0] = r[i] / 255.0
        out[i * 3 + 1] = g[i] / 255.0
        out[i * 3 + 2] = b[i] / 255.0
    return out
