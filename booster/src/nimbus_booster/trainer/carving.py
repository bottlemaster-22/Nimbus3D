"""F2: free-space carving. Empty air is evidence.

An occupancy grid at ~5 cm, carved from **every** LiDAR ray of **every** frame:

* cells a beam passed through are ``EMPTY``;
* the beam's endpoint cell is ``SURFACE``;
* space beyond the sensor's max range (~5 m), and space behind a **no-return**,
  stays ``UNKNOWN`` - never ``EMPTY``.

That last clause is the whole design. Glass returns nothing, and "nothing came
back" is not evidence of absence. A carver that treats a no-return as an
infinitely long empty ray erases everything visible through a window, which is
precisely the failure this feature exists to avoid.

The useful exception, which recovers most of the value at a window without
inventing geometry beyond it: a no-return ray still certifies the space between
the sensor and the **known wall plane it passes through** as empty. That is a
free-space *lower bound*, and :func:`carve_frame` implements it by clipping a
no-return ray at the nearest surface evidence its neighbours already provide,
rather than at infinity.

Deletion is licensed by ``EMPTY`` and by nothing else. :func:`prune_mask` is
the only function here that a trainer calls, and it returns a mask of
Gaussians whose centres sit in a **certified** empty cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from ..data.sidecars import (
    OCC_EMPTY,
    OCC_SURFACE,
    OCC_UNKNOWN,
    OccupancyGrid,
    morton_encode,
)

#: 5 cm. Fine enough that a Gaussian sitting in mid-air is unambiguously in a
#: carved cell; coarse enough that a whole house is a few million records.
DEFAULT_VOXEL_METERS = 0.05

#: A cell needs this many independent through-beams before it is trusted as
#: empty. One beam through a cell is as likely to be a depth outlier as a
#: measurement; three from different frames is not.
MIN_EMPTY_HITS = 3

#: How close to the endpoint a traversed cell has to be before it is treated as
#: surface rather than free space, in multiples of the voxel size. Without this
#: the last cell of every ray is carved empty by the ray that terminates in it.
_SURFACE_BAND_VOXELS = 1.0


@dataclass
class CarvingStats:
    """What the carve actually observed. Reported, not just logged."""

    rays_cast: int = 0
    rays_no_return: int = 0
    rays_beyond_range: int = 0
    cells_empty: int = 0
    cells_surface: int = 0

    @property
    def no_return_fraction(self) -> float:
        return self.rays_no_return / max(1, self.rays_cast)


class OccupancyAccumulator:
    """Builds the grid frame by frame, in a sparse dict of counters.

    Two counters per cell, not one: a cell is emptied by through-beams and
    filled by endpoints, and a surface seen through a doorway from one angle
    and hit from another must resolve to ``SURFACE``. Keeping the two counts
    separate is what makes that resolution a rule rather than a race.
    """

    def __init__(
        self,
        voxel_meters: float = DEFAULT_VOXEL_METERS,
        origin: Optional[np.ndarray] = None,
    ) -> None:
        self.voxel_meters = float(voxel_meters)
        self.origin = (
            np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
        )
        # key -> [through count, endpoint count]. A dict rather than a dense
        # array because a house at 5 cm is 10^9 cells and 10^7 of them matter.
        self._through: dict = {}
        self._endpoint: dict = {}
        self.stats = CarvingStats()

    # -- accumulation ------------------------------------------------------

    def _keys(self, points: np.ndarray) -> np.ndarray:
        indices = np.floor(
            (points - self.origin) / self.voxel_meters
        ).astype(np.int64)
        return morton_encode(indices)

    def _bump(self, table: dict, keys: np.ndarray) -> None:
        if keys.size == 0:
            return
        unique, counts = np.unique(keys, return_counts=True)
        for key, count in zip(unique.tolist(), counts.tolist()):
            table[key] = table.get(key, 0) + count

    def add_frame(
        self,
        origin: np.ndarray,
        endpoints: np.ndarray,
        valid: np.ndarray,
        max_range_meters: float,
        no_return_clip_meters: Optional[np.ndarray] = None,
    ) -> None:
        """Carve one frame's worth of beams.

        ``endpoints`` are the unprojected world positions of every native depth
        sample, ``valid`` marks the ones with a real return. Invalid samples are
        **not** discarded: they are the no-returns, and they carry the free-space
        lower bound described in this module's docstring. Their ray is walked
        only as far as ``no_return_clip_meters`` says a wall is known to be, and
        beyond that the cells stay ``UNKNOWN`` forever.
        """
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        endpoints = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        self.stats.rays_cast += int(endpoints.shape[0])

        direction = endpoints - origin
        distance = np.linalg.norm(direction, axis=1)
        usable = valid & np.isfinite(distance) & (distance > 1e-4)

        # Beyond the sensor's usable range a "return" is not a measurement.
        # The ray is walked only to the range limit; past it, nothing is known.
        beyond = usable & (distance > max_range_meters)
        self.stats.rays_beyond_range += int(np.count_nonzero(beyond))

        # -- rays with a real return -----------------------------------------
        good = usable & ~beyond
        if np.any(good):
            self._carve_segments(
                origin, direction[good], distance[good], mark_endpoint=True
            )

        # -- rays that ran past the range limit -------------------------------
        if np.any(beyond):
            unit = direction[beyond] / distance[beyond][:, None]
            self._carve_segments(
                origin,
                unit * max_range_meters,
                np.full(int(np.count_nonzero(beyond)), float(max_range_meters)),
                mark_endpoint=False,
            )

        # -- no-returns: free space only as far as something else knows --------
        no_return = ~valid
        self.stats.rays_no_return += int(np.count_nonzero(no_return))
        if np.any(no_return) and no_return_clip_meters is not None:
            clip = np.asarray(no_return_clip_meters, dtype=np.float64).reshape(-1)
            clip = clip[no_return]
            unit = direction[no_return]
            norm = np.linalg.norm(unit, axis=1)
            keep = np.isfinite(clip) & (clip > 1e-3) & (norm > 1e-6)
            if np.any(keep):
                unit = unit[keep] / norm[keep][:, None]
                length = clip[keep]
                self._carve_segments(
                    origin, unit * length[:, None], length, mark_endpoint=False
                )

    def _carve_segments(
        self,
        origin: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        mark_endpoint: bool,
    ) -> None:
        """Sample each ray at voxel-sized steps and count the cells it crosses.

        Uniform sampling at half the voxel size rather than an exact 3D-DDA
        traversal. The trade is deliberate and worth stating: a DDA visits
        every cell exactly once and is the right thing for a single ray, but it
        is a per-ray Python loop, and this runs over roughly 49,000 rays per
        frame times a few thousand frames. Half-voxel sampling misses a cell
        only when a ray clips a corner, which costs a missed *empty* vote - and
        a missed empty vote is a Gaussian that survives, not one that is
        wrongly deleted. The error is one-directional and on the safe side.
        """
        if offsets.shape[0] == 0:
            return
        step = self.voxel_meters * 0.5
        max_length = float(np.max(lengths))
        steps = int(np.ceil(max_length / step)) + 1
        if steps <= 1:
            return
        # Cap the work per batch: a 5 m ray at 2.5 cm steps is 200 samples, and
        # 49k rays x 200 x 3 floats is ~235 MB. Chunk the rays instead.
        chunk = max(1, int(2_000_000 // max(1, steps)))
        unit = offsets / np.maximum(lengths, 1e-9)[:, None]

        for begin in range(0, offsets.shape[0], chunk):
            end = min(begin + chunk, offsets.shape[0])
            u = unit[begin:end]
            length = lengths[begin:end]
            t = np.arange(steps, dtype=np.float64) * step
            # (rays, steps) mask of which samples are actually on the segment,
            # stopping one voxel short of the endpoint so the terminating cell
            # is not carved empty by its own ray.
            limit = length[:, None] - self.voxel_meters * _SURFACE_BAND_VOXELS
            on_segment = (t[None, :] > 0.0) & (t[None, :] < limit)
            if not np.any(on_segment):
                continue
            points = origin[None, None, :] + u[:, None, :] * t[None, :, None]
            self._bump(self._through, self._keys(points[on_segment]))

            if mark_endpoint:
                self._bump(self._endpoint, self._keys(origin + u * length[:, None]))

    # -- resolution --------------------------------------------------------

    def build(self, min_empty_hits: int = MIN_EMPTY_HITS) -> OccupancyGrid:
        """Resolve counters into states and produce the sorted grid.

        The resolution rule, in priority order:

        1. Any endpoint at all makes a cell ``SURFACE``. One real return beats
           any number of through-beams: a beam that passed through a cell is
           evidence about a *different* moment, and a thin surface seen edge-on
           is crossed far more often than it is hit.
        2. Otherwise, ``min_empty_hits`` independent through-beams make it
           ``EMPTY``.
        3. Otherwise it is not written at all, and absence means ``UNKNOWN``.
        """
        keys = sorted(set(self._through) | set(self._endpoint))
        if not keys:
            return OccupancyGrid.empty(self.voxel_meters, self.origin)

        key_array = np.asarray(keys, dtype=np.uint64)
        states = np.full(key_array.shape[0], OCC_UNKNOWN, dtype=np.uint8)
        hits = np.zeros(key_array.shape[0], dtype=np.uint16)

        for i, key in enumerate(keys):
            endpoint_count = self._endpoint.get(key, 0)
            through_count = self._through.get(key, 0)
            if endpoint_count > 0:
                states[i] = OCC_SURFACE
                hits[i] = np.uint16(min(endpoint_count, 65535))
            elif through_count >= min_empty_hits:
                states[i] = OCC_EMPTY
                hits[i] = np.uint16(min(through_count, 65535))

        keep = states != OCC_UNKNOWN
        key_array = key_array[keep]
        states = states[keep]
        hits = hits[keep]
        order = np.argsort(key_array, kind="stable")

        self.stats.cells_empty = int(np.count_nonzero(states == OCC_EMPTY))
        self.stats.cells_surface = int(np.count_nonzero(states == OCC_SURFACE))
        return OccupancyGrid(
            voxel_meters=self.voxel_meters,
            origin=self.origin,
            keys=key_array[order],
            states=states[order],
            hits=hits[order],
        )


def prune_mask(grid: OccupancyGrid, positions: np.ndarray) -> np.ndarray:
    """Boolean mask of Gaussians that may be **hard-deleted**.

    True only where the centre sits in a cell certified ``EMPTY``. A cell that
    is ``UNKNOWN`` - which is every cell the file does not mention - licenses
    nothing. This is the one function the training loop calls, and it is one
    line long on purpose: the rule it enforces is the only thing standing
    between the carver and a scan with the view through every window deleted.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(grid) == 0 or positions.shape[0] == 0:
        return np.zeros(positions.shape[0], dtype=bool)
    return grid.state_at(positions) == OCC_EMPTY


def no_return_clip_from_neighbours(
    depth: np.ndarray, radius: int = 6
) -> np.ndarray:
    """Per-sample free-space lower bound for the no-return pixels of one frame.

    For each no-return sample, the shortest **valid** depth in its local
    neighbourhood: the ray is certified free at least that far, because a
    neighbouring beam pointing within a couple of degrees of it came back from
    something at that range, and the surface it came back from is planar over
    that scale far more often than not.

    Returns metres, with ``0`` where no neighbour offers a bound - and ``0``
    means "carve nothing", not "carve to zero".

    This is the mechanism behind the "no-return rays as a free-space lower
    bound at windows" clause of F2, and it is deliberately conservative: it
    never claims free space *past* the nearest thing anyone measured nearby.
    """
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros(height * width)

    # A min-filter over the valid samples, done as a rolling minimum with
    # invalid samples set to +inf so they never win.
    filled = np.where(valid, depth, np.inf)
    best = filled.copy()
    step = max(1, int(radius))
    for shift in range(1, step + 1):
        best = np.minimum(best, np.roll(filled, shift, axis=0))
        best = np.minimum(best, np.roll(filled, -shift, axis=0))
        best = np.minimum(best, np.roll(filled, shift, axis=1))
        best = np.minimum(best, np.roll(filled, -shift, axis=1))
    best[~np.isfinite(best)] = 0.0
    # Back off by 10%: the neighbour's surface may be nearer along this ray
    # than along its own, and over-carving is the one mistake with no recovery.
    return (best * 0.9).reshape(-1)


def carve_bundle(
    frames: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    max_range_meters: float,
    voxel_meters: float = DEFAULT_VOXEL_METERS,
    origin: Optional[np.ndarray] = None,
) -> Tuple[OccupancyGrid, CarvingStats]:
    """Carve a whole scan.

    ``frames`` yields ``(camera_centre, world_points, valid_mask, native_depth)``
    per frame. The depth map comes along so the no-return free-space bound can
    be computed on the native grid, where the samples are genuinely adjacent.
    """
    accumulator = OccupancyAccumulator(voxel_meters=voxel_meters, origin=origin)
    for centre, points, valid, depth in frames:
        clip = no_return_clip_from_neighbours(depth) if depth is not None else None
        accumulator.add_frame(
            origin=centre,
            endpoints=points,
            valid=valid,
            max_range_meters=max_range_meters,
            no_return_clip_meters=clip,
        )
    grid = accumulator.build()
    return grid, accumulator.stats
