# =============================================================================
#  spz_reader.py -- decoder for the .spz Gaussian-splat container (Niantic's
#  open-sourced compact format, github.com/nianticlabs/spz, MIT licensed).
#
#  Every byte offset, constant and quantisation formula below is transcribed
#  from `ios/Sources/Export/SPZCodec.swift`'s `write()` (our own exporter,
#  the ground truth for what a real Nimbus3D .spz file actually contains),
#  which was itself verified against Niantic's C++ reference
#  (src/cc/load-spz.cc / .h, src/cc/splat-utils.h) on 2026-09-03. That file
#  did not exist on disk when this decoder was first written; this pass
#  reconciled every formula against it line by line:
#
#    - SECTION ORDER: SPZCodec.swift's `write()` lays the body out as
#      positions, ALPHAS, COLORS, SCALES, rotations, sh -- an earlier draft
#      of this decoder read positions, SCALES, ROTATIONS, alphas, colors,
#      sh (scales and rotations swapped forward, ahead of alphas/colors).
#      Every per-field formula below was individually correct, but reading
#      them in the wrong section order means each field was actually
#      decoding SOMEONE ELSE's bytes: a real 3-version-3 file's alpha bytes
#      would have been consumed as if they were scale bytes, its colour
#      bytes as if they were rotation bytes, and so on, corrupting the
#      whole record. Fixed to match SPZCodec.swift's actual write order
#      exactly; `selftest.py`'s synthetic-.spz body layout was updated to
#      match (it had the same wrong order, which is why the self-test did
#      not catch this: it was internally consistent with the buggy decoder,
#      not with the real format).
#    - versions 1-3 supported (SPZCodec.swift's own read() accepts 1...3,
#      and WRITES version 3 by default -- a decoder that only understood
#      1-2, as an earlier draft of this file did, would reject every file
#      our own exporter actually produces. Fixed.)
#    - position: v1 is THREE float16 halfs (6 bytes/point); v2-v3 are 24-bit
#      signed fixed-point, 3 bytes/component (9 bytes/point), scaled by
#      1 / 2^fractionalBits (the header's OWN fractionalBits byte, not a
#      hardcoded 12 -- SPZCodec.swift writes 12 but a reader must not assume
#      every writer does).
#    - rotation: v1-v2 use "first three" (x,y,z as signed-normalised bytes,
#      w = sqrt(1 - x^2-y^2-z^2), 3 bytes/point); v3 uses "smallest three"
#      (largest-magnitude component's INDEX + the other three, sign + 9-bit
#      magnitude each, packed into one little-endian uint32, 4 bytes/point).
#      Both are decoded here now; an earlier draft only implemented "first
#      three" and would have silently mis-decoded every v3 rotation (wrong
#      orientation, not a crash -- the worst kind of bug). Fixed.
#    - colour: the stored byte is a quantised SH DC coefficient, not a
#      finished 0..1 colour -- SPZCodec.swift's write is
#      `byte = dc * (colorScale * 255) + 0.5 * 255` with `colorScale = 0.15`.
#      This reader dequantises it back to the raw DC coefficient
#      (`colors_dc`); splat_data.py then applies the SAME
#      `clamp(0.5 + SH_C0 * dc)` step it already applies to a PLY's
#      `f_dc_0..2`, so both formats go through one shared, correct formula
#      instead of two different (and previously, one WRONG) ones. An earlier
#      draft returned `byte / 255.0` directly as if it were already the
#      final colour, which is only coincidentally close to right at dc=0 and
#      visibly washed-out/wrong everywhere else. Fixed.
#    - SH-rest bands: UNSIGNED byte, `(byte - 128) / 128.0` (SPZCodec.swift's
#      `unquantizeSH`). An earlier draft read these as SIGNED int8 and
#      divided by 127 -- wrong zero point and wrong scale. These bands are
#      still not evaluated by the default material (see material_setup.py),
#      so this could not have corrupted what earlier renders looked like,
#      but it is fixed now regardless rather than left wrong-but-unused.
#
#  HONESTY NOTE (still true): this decoder has not been round-tripped
#  against a byte-exact REAL .spz file produced by our own exporter or by
#  Niantic's own tools -- there is no macOS/Blender-side export pipeline
#  reachable from this environment to produce one, and none was supplied.
#  Every formula above was cross-checked against SPZCodec.swift's source
#  text, not against a real decoded file, and `selftest.py` builds its
#  synthetic .spz fixtures by hand from this same documented layout (an
#  independent encoder, not a copy of this decoder, so it is a real test of
#  agreement between two independent implementations of the spec -- but
#  neither has been checked against the third: an actual reference file).
#  Prefer `.ply`, which this add-on parses generically from the file's own
#  header with no guessed constants. TODO(nimbus): once a real .spz sample
#  (from Export or from spz's own reference tools) is available, round-trip
#  it through this decoder and drop this note if it matches.
#
#  If ANYTHING about the file does not match the documented shape (bad
#  magic, unsupported version, truncated data, point count that would
#  allocate an absurd amount of memory), this module raises SpzParseError
#  with a specific reason rather than returning a partially-decoded,
#  silently-wrong cloud.
# =============================================================================

import gzip
import math
import struct

MAGIC = 0x5053474E  # "NGSP" in file byte order, little-endian uint32.
HEADER_FORMAT = "<IIIBBBB"  # magic, version, numPoints, shDegree, fractionalBits, flags, reserved
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 16 bytes

FLAG_ANTIALIASED = 1 << 0
FLAG_EXTENSIONS = 1 << 1  # matches SPZCodec.swift's `flags & 0x02`

# SPZCodec.swift: `colorScale: Float = 0.15`. The stored byte is
# `dc * (colorScale * 255) + 0.5 * 255`; dequantising undoes exactly that.
_COLOR_SCALE = 0.15

# SH rest coefficients per channel for a given degree (bands 1..degree),
# i.e. (degree+1)^2 - 1. Matches the same convention used by the raw PLY
# f_rest_* layout that splat_data.py reads.
_SH_RESTS_PER_CHANNEL = {0: 0, 1: 3, 2: 8, 3: 15}

# A point count above this is almost certainly a corrupt/foreign file, not
# a real scan (our own budget-first densification hard-caps well under
# this -- see PRODUCT SPEC F10). Guards against gzip-bomb-style OOM.
_MAX_SANE_POINTS = 50_000_000

_SQRT1_2 = 0.70710678  # matches SPZCodec.swift's `sqrt1_2` constant exactly


class SpzParseError(Exception):
    """Raised for anything about the file that does not match the
    documented .spz container shape."""


class SpzCloud:
    """Decoded splat cloud, in the SAME shape splat_data.SplatCloud expects
    (see splat_data.py) so operators.py can treat a .ply and a .spz import
    identically after this point. `colors_dc` holds the raw (dequantised)
    SH degree-0 coefficient -- NOT a finished colour, see module docstring
    -- so splat_data.from_spz can apply the one shared DC-to-RGB formula it
    already uses for .ply.
    """
    __slots__ = (
        "count", "positions", "scales", "rotations_xyzw",
        "colors_dc", "opacities", "sh_rest", "sh_degree", "antialiased",
    )


def _read_u8_array(buf, offset, n):
    return struct.unpack_from(f"<{n}B", buf, offset), offset + n


def _decode_positions_fixed24(buf, offset, count, fractional_bits):
    """v2/v3: 24-bit signed fixed-point, 3 bytes/component."""
    scale = 1.0 / (1 << fractional_bits)
    out = [0.0] * (count * 3)
    pos = offset
    for i in range(count * 3):
        b0, b1, b2 = buf[pos], buf[pos + 1], buf[pos + 2]
        pos += 3
        raw = b0 | (b1 << 8) | (b2 << 16)
        if raw & 0x800000:  # sign bit of a 24-bit two's complement int
            raw -= 0x1000000
        out[i] = raw * scale
    return out, pos


def _decode_positions_half(buf, offset, count):
    """v1: three IEEE-754 binary16 halfs per point, widened to float."""
    n = count * 3
    out = list(struct.unpack_from(f"<{n}e", buf, offset))
    return out, offset + n * 2


def _decode_scales(buf, offset, count):
    # log-scale byte -> linear scale: logScale = byte/16 - 10 ; scale = e^logScale.
    out = [0.0] * (count * 3)
    pos = offset
    for i in range(count * 3):
        b = buf[pos]
        pos += 1
        out[i] = math.exp(b / 16.0 - 10.0)
    return out, pos


def _decode_rotations_first_three(buf, offset, count):
    """v1-v2: x,y,z as signed-normalised bytes; w reconstructed from the
    unit-quaternion constraint (encoder always emits w >= 0). Matches
    SPZCodec.swift's `unpackFirstThree`: byte/127.5 - 1.0 (algebraically
    identical to byte/255*2 - 1)."""
    out = [(0.0, 0.0, 0.0, 1.0)] * count
    pos = offset
    for i in range(count):
        bx, by, bz = buf[pos], buf[pos + 1], buf[pos + 2]
        pos += 3
        x = bx / 127.5 - 1.0
        y = by / 127.5 - 1.0
        z = bz / 127.5 - 1.0
        w_sq = 1.0 - (x * x + y * y + z * z)
        w = math.sqrt(w_sq) if w_sq > 0.0 else 0.0
        out[i] = (x, y, z, w)
    return out, pos


def _decode_rotations_smallest_three(buf, offset, count):
    """v3: the largest-magnitude quaternion component's index (top 2 bits of
    a little-endian uint32) plus the other three components (sign bit +
    9-bit magnitude each), scaled by sqrt(1/2). Matches SPZCodec.swift's
    `unpackSmallestThree` exactly, including bit order."""
    c_mask = (1 << 9) - 1
    out = [(0.0, 0.0, 0.0, 1.0)] * count
    pos = offset
    for i in range(count):
        comp = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        i_largest = comp >> 30
        rotation = [0.0, 0.0, 0.0, 0.0]  # (x, y, z, w), same order as SPZCodec.swift's SIMD4
        shifted = comp
        sum_squares = 0.0
        for k in (3, 2, 1, 0):
            if k == i_largest:
                continue
            mag = shifted & c_mask
            neg_bit = (shifted >> 9) & 0x1
            shifted >>= 10
            value = _SQRT1_2 * mag / c_mask
            if neg_bit:
                value = -value
            rotation[k] = value
            sum_squares += value * value
        rotation[i_largest] = math.sqrt(max(0.0, 1.0 - sum_squares))
        out[i] = tuple(rotation)
    return out, pos


def _decode_alphas(buf, offset, count):
    out, pos = _read_u8_array(buf, offset, count)
    return [v / 255.0 for v in out], pos


def _decode_colors_dc(buf, offset, count):
    """Dequantise the stored byte back to the raw SH degree-0 coefficient
    (SPZCodec.swift write: `byte = dc*(colorScale*255) + 0.5*255`). NOT a
    finished colour -- see module docstring."""
    out = [(0.0, 0.0, 0.0)] * count
    pos = offset
    for i in range(count):
        r, g, b = buf[pos], buf[pos + 1], buf[pos + 2]
        pos += 3
        out[i] = (
            (r / 255.0 - 0.5) / _COLOR_SCALE,
            (g / 255.0 - 0.5) / _COLOR_SCALE,
            (b / 255.0 - 0.5) / _COLOR_SCALE,
        )
    return out, pos


def _decode_sh_rest(buf, offset, count, rests_per_channel):
    """Matches SPZCodec.swift's `unquantizeSH`: UNSIGNED byte, (byte-128)/128."""
    if rests_per_channel == 0:
        return [], offset
    n_per_point = rests_per_channel * 3
    total = count * n_per_point
    if total == 0:
        return [], offset
    raw = struct.unpack_from(f"<{total}B", buf, offset)  # unsigned uint8
    out = [(v - 128.0) / 128.0 for v in raw]
    return out, offset + total


def decode_spz(filepath):
    """Read and decode a .spz file. Returns an SpzCloud. Raises
    SpzParseError on anything unsupported or structurally inconsistent.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    try:
        buf = gzip.decompress(raw)
    except OSError as exc:
        raise SpzParseError(
            f"{filepath} is not a gzip stream (.spz files are gzip-"
            f"compressed): {exc}"
        ) from exc

    if len(buf) < HEADER_SIZE:
        raise SpzParseError(
            f"decompressed .spz data is only {len(buf)} bytes, smaller "
            f"than the {HEADER_SIZE}-byte header"
        )

    magic, version, num_points, sh_degree, fractional_bits, flags, _reserved = (
        struct.unpack_from(HEADER_FORMAT, buf, 0)
    )

    if magic != MAGIC:
        raise SpzParseError(
            f"bad .spz magic number 0x{magic:08X} (expected "
            f"0x{MAGIC:08X}) -- not a recognised .spz file"
        )
    if version not in (1, 2, 3):
        if version == 4:
            raise SpzParseError(
                ".spz version 4 (NGSP, Zstandard multi-stream container) is not "
                "supported -- this add-on (and our own exporter) only read/write "
                "versions 1-3. Re-export as SPZ version 2 or 3, or use .ply."
            )
        raise SpzParseError(
            f".spz version {version} is not one this decoder was written "
            f"against (supports 1-3) -- refusing to guess the layout"
        )
    if num_points <= 0 or num_points > _MAX_SANE_POINTS:
        raise SpzParseError(f"implausible point count {num_points} in .spz header")
    if sh_degree not in _SH_RESTS_PER_CHANNEL:
        raise SpzParseError(f"unsupported SH degree {sh_degree} in .spz header")

    rests_per_channel = _SH_RESTS_PER_CHANNEL[sh_degree]
    offset = HEADER_SIZE

    # Order matches SPZCodec.swift's write() EXACTLY: positions, alphas,
    # colors, scales, rotations, sh -- see the module docstring's "SECTION
    # ORDER" note. Do not reorder these without re-checking that file.
    if version == 1:
        positions, offset = _decode_positions_half(buf, offset, num_points)
    else:
        positions, offset = _decode_positions_fixed24(buf, offset, num_points, fractional_bits)
    alphas, offset = _decode_alphas(buf, offset, num_points)
    colors_dc, offset = _decode_colors_dc(buf, offset, num_points)
    scales, offset = _decode_scales(buf, offset, num_points)
    if version >= 3:
        rotations, offset = _decode_rotations_smallest_three(buf, offset, num_points)
    else:
        rotations, offset = _decode_rotations_first_three(buf, offset, num_points)
    sh_rest, offset = _decode_sh_rest(buf, offset, num_points, rests_per_channel)

    if offset > len(buf):
        raise SpzParseError(
            f"decoded past end of .spz data ({offset} > {len(buf)} bytes) "
            f"-- header fields do not match the body length, refusing the "
            f"partially-decoded result"
        )

    cloud = SpzCloud()
    cloud.count = num_points
    cloud.positions = positions          # flat [x0,y0,z0, x1,y1,z1, ...]
    cloud.scales = scales                # flat, linear (already exp'd), same layout
    cloud.rotations_xyzw = rotations     # list of (x,y,z,w) unit quaternions
    cloud.colors_dc = colors_dc          # list of raw (r,g,b) SH degree-0 coefficients
    cloud.opacities = alphas             # list of floats in 0..1
    cloud.sh_rest = sh_rest              # flat, degree*3 per point, or [] if degree 0
    cloud.sh_degree = sh_degree
    cloud.antialiased = bool(flags & FLAG_ANTIALIASED)
    return cloud
