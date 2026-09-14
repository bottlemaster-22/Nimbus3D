"""F3: native depth edges, the dilation band, and the where/what classification.

Depth supervision happens **only** at the ~49k native 256x192 samples per
frame. Everything between them is interpolation, and training against
interpolation teaches the model to reproduce the interpolator.

Edges are computed on the **native** map and then dilated by the upsample ratio
(~7.5 at 256 native columns against 1920 RGB ones, giving the ~8 px band the
design calls for at RGB resolution). Inside that band the depth loss is zeroed,
because the upsampled depth map is simply wrong there and supervising on it
teaches the model the interpolator's smearing - which is worse than no
supervision at all.

Three edge kinds, and each one earns a different treatment:

``geometric``
    A real depth step outside the band. Sharpen here.
``texture``
    A strong image gradient over flat depth. Flatten: do not invent geometry
    where the only evidence is a painted line.
``unknown``
    Glass, out of range, or low confidence. Ignore. Do not guess.

The byte values written to ``prepass/edges/frame_*.edge8`` are the ``EDGE_*``
constants from :mod:`nimbus_booster.data.sidecars`; class ``4`` is the band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..data.sidecars import (
    EDGE_BAND,
    EDGE_GEOMETRIC,
    EDGE_NONE,
    EDGE_TEXTURE,
    EDGE_UNKNOWN,
)


@dataclass
class EdgeParameters:
    """Thresholds, all in units that mean something physically.

    A depth step is judged **relative to range**, not in absolute metres: LiDAR
    noise grows with distance, so a 3 cm step is a real edge at 1 m and is
    noise at 5 m. That ratio is the single most important number here.
    """

    #: A depth discontinuity is a step larger than this fraction of the local
    #: range. 3% at 2 m is 6 cm, which is a door frame, not sensor noise.
    relative_step: float = 0.03
    #: Absolute floor so a step at 20 cm range is not called an edge by the
    #: relative rule alone.
    min_step_meters: float = 0.02
    #: Normalised image gradient above which a pixel is "strongly textured".
    texture_gradient: float = 0.18
    #: ARKit confidence at or below this is treated as no measurement at all.
    low_confidence: int = 0
    #: Band width is this many native pixels, before the upsample ratio is
    #: applied. One native pixel already covers ~7.5 RGB pixels.
    band_native_pixels: int = 1


def _finite_diff_step(depth: np.ndarray) -> np.ndarray:
    """Largest absolute depth difference to any 4-neighbour, in metres.

    A max over neighbours rather than a Sobel magnitude on purpose: a Sobel
    filter smooths a one-pixel step into a two-pixel ramp, and on a 256-wide
    map a two-pixel ramp is 15 RGB pixels of misplaced band.
    """
    valid = np.isfinite(depth) & (depth > 0.0)
    filled = np.where(valid, depth, np.nan)
    step = np.zeros_like(depth)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        neighbour = np.roll(filled, shift, axis=axis)
        difference = np.abs(filled - neighbour)
        step = np.fmax(step, np.nan_to_num(difference, nan=0.0))
    step[~valid] = 0.0
    return step


def _image_gradient(gray: np.ndarray) -> np.ndarray:
    """Normalised central-difference gradient magnitude of a 0..1 gray image."""
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    return np.sqrt(gx * gx + gy * gy)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element dilation, as a rolling OR."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for shift in range(1, radius + 1):
        out |= np.roll(mask, shift, axis=0)
        out |= np.roll(mask, -shift, axis=0)
        out |= np.roll(mask, shift, axis=1)
        out |= np.roll(mask, -shift, axis=1)
    return out


def upsample_ratio(rgb_width: int, native_width: int) -> float:
    """How many RGB pixels one native depth pixel covers along a row.

    7.5 at 1920 / 256. This is the number the band is dilated by, expressed in
    native pixels: ``ceil(band_native_pixels)`` native pixels of dilation is
    ``ratio * band_native_pixels`` RGB pixels of band.
    """
    if native_width <= 0:
        return 1.0
    return float(rgb_width) / float(native_width)


def classify_edges(
    depth: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    gray_native: Optional[np.ndarray] = None,
    max_range_meters: float = 5.0,
    parameters: Optional[EdgeParameters] = None,
) -> np.ndarray:
    """Classify every native depth pixel. Returns an ``(H, W)`` uint8 map.

    ``gray_native`` is the RGB frame's luminance resampled **down** to the
    native depth grid, in 0..1. It may be ``None``, in which case no pixel is
    classified ``texture`` - the honest outcome, since texture is by definition
    an image-side observation and there is no image-side observation to make.
    """
    parameters = parameters or EdgeParameters()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    out = np.full((height, width), EDGE_NONE, dtype=np.uint8)

    valid = np.isfinite(depth) & (depth > 0.0)

    # -- unknown: no return, past range, or ARKit says it does not know -----
    unknown = ~valid
    unknown |= valid & (depth > max_range_meters)
    if confidence is not None:
        unknown |= np.asarray(confidence) <= parameters.low_confidence

    # -- geometric: a depth step large relative to the local range ----------
    step = _finite_diff_step(depth)
    threshold = np.maximum(
        parameters.min_step_meters, parameters.relative_step * np.abs(depth)
    )
    geometric = valid & (step > threshold)

    # -- texture: strong image gradient over flat depth ---------------------
    texture = np.zeros_like(geometric)
    if gray_native is not None:
        gradient = _image_gradient(np.asarray(gray_native, dtype=np.float64))
        texture = valid & (~geometric) & (gradient > parameters.texture_gradient)

    # -- band: the dilation around geometric edges --------------------------
    band = _dilate(geometric, max(0, int(parameters.band_native_pixels)))
    band &= ~geometric

    # Priority: unknown beats everything (we genuinely do not know), then the
    # geometric edge itself, then its band, then texture. Ordering matters:
    # a texture edge inside a geometric band is still band, because the depth
    # there is still the interpolator's invention.
    out[texture] = EDGE_TEXTURE
    out[band] = EDGE_BAND
    out[geometric] = EDGE_GEOMETRIC
    out[unknown] = EDGE_UNKNOWN
    return out


def depth_supervision_mask(edges: np.ndarray) -> np.ndarray:
    """Where the depth loss is allowed to act at all.

    Excludes ``band`` (the upsampled depth is wrong there) and ``unknown``
    (there is nothing to compare against). Geometric edges are **kept**: they
    are the samples the bimodal edge supervision in F4 needs.
    """
    edges = np.asarray(edges)
    return (edges != EDGE_BAND) & (edges != EDGE_UNKNOWN)


def bimodal_edge_targets(
    depth: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two local depth modes at each geometric edge, and the gap between.

    F4's bimodal edge supervision: at a depth discontinuity, penalise the
    rendered depth against whichever of the two local modes it is **nearer**,
    rather than against their average. Supervising against the average pulls
    the surface into the gap between a door frame and the wall behind it and
    fills it with a smear of translucent Gaussians, which is the classic
    3DGS-at-an-edge artefact.

    Returns ``(near, far, is_edge)``. ``near`` and ``far`` are metres; where
    ``is_edge`` is False both hold the sample's own depth and the caller should
    fall back to ordinary single-target supervision.
    """
    depth = np.asarray(depth, dtype=np.float64)
    is_edge = np.asarray(edges) == EDGE_GEOMETRIC
    valid = np.isfinite(depth) & (depth > 0.0)
    filled = np.where(valid, depth, np.nan)

    stack = [filled]
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        stack.append(np.roll(filled, shift, axis=axis))
    neighbourhood = np.stack(stack, axis=0)

    with np.errstate(invalid="ignore"):
        near = np.nanmin(neighbourhood, axis=0)
        far = np.nanmax(neighbourhood, axis=0)
    near = np.nan_to_num(near, nan=0.0)
    far = np.nan_to_num(far, nan=0.0)

    near = np.where(is_edge, near, filled)
    far = np.where(is_edge, far, filled)
    return (
        np.nan_to_num(near, nan=0.0),
        np.nan_to_num(far, nan=0.0),
        is_edge & valid,
    )


def resample_to_native(
    image: np.ndarray, native_width: int, native_height: int
) -> np.ndarray:
    """Box-average an RGB-resolution channel down onto the native depth grid.

    Area averaging, not nearest-neighbour: nearest would sample one RGB pixel
    out of the ~56 that a native depth pixel covers, so a single specular
    highlight would decide whether a whole depth sample counts as textured.
    """
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape[:2]
    if native_width <= 0 or native_height <= 0:
        return image
    if (height, width) == (native_height, native_width):
        return image

    row_edges = np.linspace(0, height, native_height + 1).astype(int)
    col_edges = np.linspace(0, width, native_width + 1).astype(int)
    # Two passes of np.add.reduceat: rows first, then columns.
    rows = np.add.reduceat(image, row_edges[:-1], axis=0)
    row_counts = np.diff(row_edges).astype(np.float64)
    rows = rows / row_counts.reshape(-1, *([1] * (image.ndim - 1)))
    cols = np.add.reduceat(rows, col_edges[:-1], axis=1)
    col_counts = np.diff(col_edges).astype(np.float64)
    shape = [1] * cols.ndim
    shape[1] = -1
    return cols / col_counts.reshape(shape)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma of an ``(H, W, 3)`` image in 0..1."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
