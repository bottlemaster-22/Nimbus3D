"""Niantic SPZ version 3, byte-compatible with ``SPZCodec.swift``.

Every constant, offset and quantisation formula below was transcribed from
``ios/Sources/Export/SPZCodec.swift``, which was in turn transcribed from
Niantic's own ``src/cc/load-spz.cc`` (MIT). Nothing here is remembered.

Container: a **single gzip stream** wrapping a 16-byte header followed by six
concatenated attribute streams, in this order - positions, alphas, colours,
scales, rotations, spherical harmonics.

Why version 3 and not 2: v1/v2 store a quaternion's ``(x, y, z)`` and
reconstruct ``w = sqrt(1 - x^2 - y^2 - z^2)``, which is numerically unstable
whenever the true ``w`` is near zero - a routine case for camera-facing splats.
v3's "smallest three" always drops whichever component is *largest*, never
``w`` specifically, and is stable everywhere on the unit sphere. v3 is also the
de facto interchange version, so this is not a compatibility trade-off.

Version 4 ("NGSP") is a multi-stream Zstandard container. This module does not
write it and does not pretend to read it.

Coordinate frame: SPZ's native frame is RUB, which is exactly how
:class:`SplatCloud` stores everything - so unlike the PLY writer, no flip
happens here at all.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from .splat_cloud import ExportError, SplatCloud, sigmoid

MAGIC = 0x5053_474E  # "NGSP" little-endian
WRITE_VERSION = 3
FRACTIONAL_BITS = 12  # 1/4096 m, about 0.24 mm
SH1_BITS = 5
SH_REST_BITS = 4
COLOR_SCALE = 0.15
SQRT1_2 = 0.707_106_78

#: Positions are 24-bit fixed point with 12 fractional bits, so the whole
#: scene must live inside +-2048 m of the world origin. A house scan is three
#: orders of magnitude inside that; a mis-scaled import is not, which is why
#: the writer refuses rather than wrapping silently.
_MAX_FIXED = (1 << 23) - 1
_MIN_FIXED = -(1 << 23)


def _clamp_to_u8(values: np.ndarray) -> np.ndarray:
    """Round to nearest and clamp to 0..255, with non-finite pinned to an end."""
    values = np.asarray(values, dtype=np.float64)
    out = np.where(np.isfinite(values), values, np.where(values > 0, 255.0, 0.0))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _pack_smallest_three(rotations: np.ndarray) -> np.ndarray:
    """(N, 4) ``(x, y, z, w)`` -> (N,) uint32, the v3 rotation encoding.

    Drop whichever component has the largest magnitude (it is recoverable from
    the unit constraint), record its index in the top two bits, and store the
    other three as a sign bit plus 9 magnitude bits scaled by ``sqrt(2)``,
    most-significant component first in the original 0..3 order.
    """
    q = np.asarray(rotations, dtype=np.float64).reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    bad = (~np.isfinite(norm)) | (norm < 1e-12)
    q = np.where(bad, np.array([0.0, 0.0, 0.0, 1.0]), q / np.where(bad, 1.0, norm))

    largest = np.argmax(np.abs(q), axis=1)
    rows = np.arange(q.shape[0])
    negate = q[rows, largest] < 0.0

    packed = largest.astype(np.uint32)
    for i in range(4):
        keep = largest != i
        component = q[:, i]
        neg_bit = ((component < 0.0) != negate) & keep
        magnitude = np.minimum(
            np.floor((511.0 * (np.abs(component) / SQRT1_2)) + 0.5).astype(np.int64),
            0x1FF,
        )
        magnitude = np.maximum(magnitude, 0)
        contribution = (neg_bit.astype(np.uint32) << np.uint32(9)) | magnitude.astype(
            np.uint32
        )
        # Only the three kept components shift the accumulator; the dropped one
        # contributes nothing and does not advance it.
        packed = np.where(keep, (packed << np.uint32(10)) | contribution, packed)
    return packed.astype("<u4")


def _quantize_sh(values: np.ndarray, bucket: int) -> np.ndarray:
    """SH coefficient -> byte, rounded to the band's bucket size."""
    values = np.asarray(values, dtype=np.float64)
    safe = np.where(np.isfinite(values), values, 0.0)
    q = np.rint(safe * 128.0).astype(np.int64) + 128
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def write_spz(cloud: SplatCloud, path: Path) -> Path:
    """Write ``cloud`` as an SPZ v3 file. Returns ``path``."""
    if len(cloud) == 0:
        raise ExportError("There are no splats to export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cloud)
    rest = cloud.rest_count

    body = bytearray()
    body += struct.pack(
        "<IIIBBBB",
        MAGIC,
        WRITE_VERSION,
        n,
        int(cloud.sh_degree),
        FRACTIONAL_BITS & 0xFF,
        0,  # flags: not antialiased, no extensions
        0,  # reserved
    )

    # -- positions: 24-bit fixed point, little-endian, 9 bytes per splat ----
    scale = float(1 << FRACTIONAL_BITS)
    fixed = np.rint(cloud.positions.astype(np.float64) * scale)
    if not np.all(np.isfinite(fixed)):
        raise ExportError(
            "A splat position is not a finite number, so this scan cannot be "
            "written as .spz."
        )
    if np.any(fixed < _MIN_FIXED) or np.any(fixed > _MAX_FIXED):
        worst = float(np.max(np.abs(cloud.positions)))
        raise ExportError(
            "A splat sits {d:.0f} m from the origin, outside the +-2048 m range "
            "the .spz format can store. The .ply export is unaffected.".format(d=worst)
        )
    as_int = fixed.astype(np.int64).reshape(-1)
    triples = np.stack(
        [
            (as_int & 0xFF).astype(np.uint8),
            ((as_int >> 8) & 0xFF).astype(np.uint8),
            ((as_int >> 16) & 0xFF).astype(np.uint8),
        ],
        axis=1,
    )
    body += np.ascontiguousarray(triples).tobytes()

    # -- alphas: sigmoid(logit) -> byte ------------------------------------
    body += _clamp_to_u8(sigmoid(cloud.opacity_logits) * 255.0).tobytes()

    # -- colours: raw DC -> wide-range byte ---------------------------------
    colors = cloud.color_dc.astype(np.float64) * (COLOR_SCALE * 255.0) + 0.5 * 255.0
    body += np.ascontiguousarray(_clamp_to_u8(colors)).tobytes()

    # -- scales: (logScale + 10) * 16 -> byte -------------------------------
    scales = (cloud.log_scales.astype(np.float64) + 10.0) * 16.0
    body += np.ascontiguousarray(_clamp_to_u8(scales)).tobytes()

    # -- rotations: smallest three, 4 bytes per splat ------------------------
    body += np.ascontiguousarray(_pack_smallest_three(cloud.rotations)).tobytes()

    # -- spherical harmonics: coefficient-major, channel-minor ---------------
    if rest > 0:
        assert cloud.sh_rest is not None
        sh1_bucket = 1 << (8 - SH1_BITS)
        rest_bucket = 1 << (8 - SH_REST_BITS)
        # The first three coefficients (degree 1) get finer buckets than the
        # rest; that split is in the reference and is not an approximation.
        buckets = np.where(np.arange(rest) < 3, sh1_bucket, rest_bucket)
        quantised = np.empty((n, rest, 3), dtype=np.uint8)
        for k in range(rest):
            quantised[:, k, :] = _quantize_sh(cloud.sh_rest[:, k, :], int(buckets[k]))
        body += np.ascontiguousarray(quantised).tobytes()

    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as handle:
        # mtime=0 so the same cloud produces the same bytes twice, which makes
        # a manifest hash reproducible and a re-send genuinely resumable.
        with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=6, mtime=0) as gz:
            gz.write(bytes(body))
    temporary.replace(path)
    return path


def read_spz_header(path: Path) -> dict:
    """Peek at an SPZ file's header without decoding its attribute streams.

    Enough to answer "is this the right version and how many splats" for a log
    line or a sanity check. A v4 file is reported as such rather than decoded:
    v4 is a Zstandard multi-stream container and this module does not implement
    it.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        prefix = handle.read(8)
    if len(prefix) >= 8:
        magic, version = struct.unpack("<II", prefix)
        if magic == MAGIC and version == 4:
            return {"version": 4, "supported": False}
    with gzip.open(str(path), "rb") as gz:
        header = gz.read(16)
    if len(header) < 16:
        raise ExportError("The .spz data stream is truncated: header.")
    magic, version, count, sh_degree, fractional, flags, _reserved = struct.unpack(
        "<IIIBBBB", header
    )
    if magic != MAGIC:
        raise ExportError("Not an .spz file (bad magic number).")
    return {
        "version": int(version),
        "supported": version in (1, 2, 3),
        "count": int(count),
        "shDegree": int(sh_degree),
        "fractionalBits": int(fractional),
        "flags": int(flags),
    }
