"""The binary sidecars: native depth, confidence, edges, trust, occupancy.

``docs/DATA_FORMAT.md`` sections 5 and 7. Every file is little-endian and
headerless; its shape lives in JSON so nothing has to guess. Both ends of this
project are little-endian, so no byte swapping happens - but the dtypes below
say ``<u2`` rather than ``u2`` anyway, so that when it eventually matters
nobody has to reverse-engineer it from a hex dump.

The one thing worth restating because it decides whether F2 and F5 work at all:

    **A no-return is ``0``, and ``0`` is not "distance zero" and not "empty".**

A beam that came back with nothing is a *no-return*. Space beyond the sensor's
range, and space behind a no-return, stays ``unknown`` forever. Deleting a
Gaussian is licensed only by ``empty``, never by ``unknown`` - which is what
stops the carver from erasing everything visible through a window, since glass
returns nothing and "nothing came back" is not evidence of absence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Occupancy states (F2). Mirrors the byte written into occupancy.bin.
# --------------------------------------------------------------------------

OCC_UNKNOWN = 0
OCC_EMPTY = 1
OCC_SURFACE = 2

# --------------------------------------------------------------------------
# Edge classes (F3). One byte per NATIVE depth pixel, not per RGB pixel.
# --------------------------------------------------------------------------

EDGE_NONE = 0
EDGE_GEOMETRIC = 1
EDGE_TEXTURE = 2
EDGE_UNKNOWN = 3
EDGE_BAND = 4

#: ARKit mesh face classification plus the two the glass detector adds.
SURFACE_CLASS_NAMES = (
    "none",
    "wall",
    "floor",
    "ceiling",
    "table",
    "seat",
    "window",
    "door",
    "glass",
    "sky",
)


# --------------------------------------------------------------------------
# Per-frame native sensor maps
# --------------------------------------------------------------------------


def read_depth16(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    """Native LiDAR depth as float32 **metres**, ``0`` where there was no return.

    ``UInt16`` little-endian millimetres on disk, row-major, top-left origin,
    no header and no padding. The dimensions come from the bundle's
    ``settings``; a file whose size disagrees is refused rather than reshaped,
    because a reshape that happens to divide evenly produces a plausible-looking
    sheared depth map and hours of confusion.
    """
    path = Path(path)
    if width <= 0 or height <= 0 or not path.is_file():
        return None
    expected = width * height * 2
    actual = path.stat().st_size
    if actual != expected:
        return None
    raw = np.fromfile(str(path), dtype="<u2")
    if raw.size != width * height:
        return None
    return (raw.reshape(height, width).astype(np.float32)) * 0.001


def read_confidence8(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    """ARKit confidence as uint8: 0 low, 1 medium, 2 high.

    Store it, but do not believe it. ARKit flags roughly 1.6% of samples as low
    and the three levels are not calibrated to anything - a "high" sample next
    to a window is routinely worse than a "low" one on a matte wall. Prefer
    ``prepass/confidence_recal.bin`` when it exists; this is the fallback.
    """
    path = Path(path)
    if width <= 0 or height <= 0 or not path.is_file():
        return None
    if path.stat().st_size != width * height:
        return None
    return np.fromfile(str(path), dtype=np.uint8).reshape(height, width)


def read_edge8(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    """Per-native-pixel edge classification, one of the ``EDGE_*`` constants."""
    path = Path(path)
    if width <= 0 or height <= 0 or not path.is_file():
        return None
    if path.stat().st_size != width * height:
        return None
    return np.fromfile(str(path), dtype=np.uint8).reshape(height, width)


def write_edge8(path: Path, edges: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(edges, dtype=np.uint8).tofile(str(path))


# --------------------------------------------------------------------------
# Morton (Z-order) keys, shared by occupancy.bin and trust_bias.bin
# --------------------------------------------------------------------------

_MORTON_BITS = 21  # 3 * 21 = 63 bits, fits a UInt64 with one to spare
_MORTON_OFFSET = 1 << (_MORTON_BITS - 1)


def _spread3(value: np.ndarray) -> np.ndarray:
    """Interleave two zero bits after each of the low 21 bits of each value."""
    x = value.astype(np.uint64) & np.uint64((1 << _MORTON_BITS) - 1)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_encode(ijk: np.ndarray) -> np.ndarray:
    """(N, 3) signed voxel indices -> (N,) uint64 Z-order keys.

    Indices are biased by 2^20 before interleaving so a negative coordinate -
    which is completely normal, the ARKit origin is wherever the user stood -
    encodes without wrapping. The bias is part of the format: a reader that
    skips it gets keys that sort correctly but decode to nonsense.
    """
    biased = np.asarray(ijk, dtype=np.int64) + _MORTON_OFFSET
    if np.any(biased < 0) or np.any(biased >= (1 << _MORTON_BITS)):
        raise ValueError(
            "voxel index outside the +-1,048,576 cell range a 21-bit Morton "
            "key can hold; the voxel size is too small for this scene"
        )
    return (
        _spread3(biased[:, 0])
        | (_spread3(biased[:, 1]) << np.uint64(1))
        | (_spread3(biased[:, 2]) << np.uint64(2))
    )


def _compact3(value: np.ndarray) -> np.ndarray:
    x = value.astype(np.uint64) & np.uint64(0x1249249249249249)
    x = (x | (x >> np.uint64(2))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x >> np.uint64(4))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x >> np.uint64(8))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x >> np.uint64(32))) & np.uint64((1 << _MORTON_BITS) - 1)
    return x


def morton_decode(keys: np.ndarray) -> np.ndarray:
    """(N,) uint64 keys -> (N, 3) signed voxel indices. Inverse of the above."""
    keys = np.asarray(keys, dtype=np.uint64)
    out = np.stack(
        [
            _compact3(keys),
            _compact3(keys >> np.uint64(1)),
            _compact3(keys >> np.uint64(2)),
        ],
        axis=1,
    ).astype(np.int64)
    return out - _MORTON_OFFSET


# --------------------------------------------------------------------------
# occupancy.bin (F2)
# --------------------------------------------------------------------------

#: 12 bytes per record: key, state, reserved, hit count.
OCCUPANCY_DTYPE = np.dtype(
    [("key", "<u8"), ("state", "u1"), ("reserved", "u1"), ("hits", "<u2")]
)
assert OCCUPANCY_DTYPE.itemsize == 12


@dataclass
class OccupancyGrid:
    """A sparse voxel grid where the absence of a record means ``unknown``.

    That is the whole point of the structure and it is a contract, not an
    optimisation: a cell is ``empty`` only when a LiDAR beam demonstrably
    passed *through* it and returned from something further away.
    """

    voxel_meters: float
    origin: np.ndarray
    keys: np.ndarray  # (M,) uint64, sorted ascending
    states: np.ndarray  # (M,) uint8
    hits: np.ndarray  # (M,) uint16

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def indices_for(self, points: np.ndarray) -> np.ndarray:
        """World-space points -> integer voxel indices."""
        relative = (np.asarray(points, dtype=np.float64) - self.origin) / self.voxel_meters
        return np.floor(relative).astype(np.int64)

    def state_at(self, points: np.ndarray) -> np.ndarray:
        """Per-point occupancy state. Not present in the file means ``unknown``.

        Binary search over the sorted key column, which is why the file is
        sorted: a lookup is O(log M) and a merge of two grids is a linear scan.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(self) == 0 or points.shape[0] == 0:
            return np.full(points.shape[0], OCC_UNKNOWN, dtype=np.uint8)
        query = morton_encode(self.indices_for(points))
        position = np.searchsorted(self.keys, query)
        position = np.clip(position, 0, len(self) - 1)
        hit = self.keys[position] == query
        out = np.full(query.shape[0], OCC_UNKNOWN, dtype=np.uint8)
        out[hit] = self.states[position[hit]]
        return out

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        order = np.argsort(self.keys, kind="stable")
        records = np.zeros(len(self), dtype=OCCUPANCY_DTYPE)
        records["key"] = self.keys[order]
        records["state"] = self.states[order]
        records["hits"] = self.hits[order]
        records.tofile(str(path))

    @staticmethod
    def read(path: Path, voxel_meters: float, origin: np.ndarray) -> "OccupancyGrid":
        path = Path(path)
        if not path.is_file():
            return OccupancyGrid.empty(voxel_meters, origin)
        records = np.fromfile(str(path), dtype=OCCUPANCY_DTYPE)
        order = np.argsort(records["key"], kind="stable")
        records = records[order]
        return OccupancyGrid(
            voxel_meters=float(voxel_meters),
            origin=np.asarray(origin, dtype=np.float64).reshape(3),
            keys=np.ascontiguousarray(records["key"]),
            states=np.ascontiguousarray(records["state"]),
            hits=np.ascontiguousarray(records["hits"]),
        )

    @staticmethod
    def empty(voxel_meters: float, origin: np.ndarray) -> "OccupancyGrid":
        return OccupancyGrid(
            voxel_meters=float(voxel_meters),
            origin=np.asarray(origin, dtype=np.float64).reshape(3),
            keys=np.zeros(0, dtype=np.uint64),
            states=np.zeros(0, dtype=np.uint8),
            hits=np.zeros(0, dtype=np.uint16),
        )


# --------------------------------------------------------------------------
# trust_bias.bin, trust_noise.bin, depth_affine.bin, confidence_recal.bin (F6)
# --------------------------------------------------------------------------

#: 24 bytes per record.
TRUST_BIAS_DTYPE = np.dtype(
    [
        ("key", "<u8"),
        ("meanResidual", "<f4"),
        ("variance", "<f4"),
        ("count", "<u4"),
        ("distinctTimes", "<u2"),
        ("reserved", "<u2"),
    ]
)
assert TRUST_BIAS_DTYPE.itemsize == 24

#: 12 bytes per record: which frame, and its learned depth scale and shift.
DEPTH_AFFINE_DTYPE = np.dtype(
    [("frameIndex", "<u4"), ("scale", "<f4"), ("shift", "<f4")]
)
assert DEPTH_AFFINE_DTYPE.itemsize == 12

#: 12 bytes per record: which frame, and its learned exposure gain and bias.
EXPOSURE_DTYPE = np.dtype([("frameIndex", "<u4"), ("gain", "<f4"), ("bias", "<f4")])
assert EXPOSURE_DTYPE.itemsize == 12


def read_trust_bias(path: Path) -> np.ndarray:
    """The coarse (25-50 cm) voxel bias field as a structured array.

    ``distinctTimes`` is the field that matters and the reason the record is
    24 bytes instead of 16: it separates "measured repeatedly from one pass"
    from "confirmed on separate visits", and only the latter earns real trust.
    """
    path = Path(path)
    if not path.is_file():
        return np.zeros(0, dtype=TRUST_BIAS_DTYPE)
    return np.fromfile(str(path), dtype=TRUST_BIAS_DTYPE)


def write_trust_bias(path: Path, records: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = np.asarray(records, dtype=TRUST_BIAS_DTYPE)
    records = records[np.argsort(records["key"], kind="stable")]
    records.tofile(str(path))


def read_frame_major_floats(
    path: Path, frame_count: int, samples_per_frame: int
) -> Optional[np.ndarray]:
    """``trust_noise.bin`` / ``confidence_recal.bin``: (frames, samples) float32.

    Frame-major, in the same order as the depth sidecars: frame 0's
    ``width*height`` samples, then frame 1's, and so on.

    **These files are never spatially averaged.** That is not an optimisation
    note, it is the contract: averaging an outlier into its neighbours is
    exactly how a wrong surface becomes a confident wrong surface.
    """
    path = Path(path)
    if not path.is_file() or frame_count <= 0 or samples_per_frame <= 0:
        return None
    expected = frame_count * samples_per_frame * 4
    if path.stat().st_size != expected:
        return None
    return np.fromfile(str(path), dtype="<f4").reshape(frame_count, samples_per_frame)


def read_depth_affine(path: Path) -> Dict[int, Tuple[float, float]]:
    """``{frameIndex: (scale, shift)}``, tightly constrained around (1, 0).

    A large correction is a symptom, not a fix: it means something else is
    wrong, most often a pose. Reported by the pipeline, never silently applied
    at full strength.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    records = np.fromfile(str(path), dtype=DEPTH_AFFINE_DTYPE)
    return {
        int(r["frameIndex"]): (float(r["scale"]), float(r["shift"])) for r in records
    }


def write_exposure(path: Path, gains: Dict[int, Tuple[float, float]]) -> None:
    """``model/exposure.bin``: per-frame learned ``(gain, bias)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = np.zeros(len(gains), dtype=EXPOSURE_DTYPE)
    for i, index in enumerate(sorted(gains)):
        gain, bias = gains[index]
        records[i]["frameIndex"] = np.uint32(index)
        records[i]["gain"] = np.float32(gain)
        records[i]["bias"] = np.float32(bias)
    records.tofile(str(path))


# --------------------------------------------------------------------------
# prepass_result.json
# --------------------------------------------------------------------------


def read_prepass_result(scan_root: Path) -> Optional[Dict[str, object]]:
    """``prepass/prepass_result.json``, or ``None`` when the pre-pass never ran.

    Section 9: absent ``prepass/`` means the raw poses are all there is, and
    the pipeline degrades to generic 3DGS rather than refusing to run. That
    degradation is reported to the user, not hidden - a scan trained without
    the pre-pass is measurably worse and they should know why.
    """
    path = Path(scan_root) / "prepass" / "prepass_result.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None
