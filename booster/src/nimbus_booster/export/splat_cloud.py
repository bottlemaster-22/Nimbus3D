"""The Python mirror of ``Export.SplatCloud``, backed by numpy arrays.

Storage conventions, transcribed from ``ios/Sources/Export/SplatCloud.swift``
so the two ends cannot drift:

* **Frame**: RUB - ``+X`` right, ``+Y`` up, ``+Z`` back. That is the world
  frame with no rotation applied, so a trained splat needs no conversion for
  ``.spz`` or ``.glb``. Only ``.ply`` differs (INRIA's convention is RDF) and
  that flip is applied inside the PLY writer, on the way out and back.
* **Rotations** are ``(x, y, z, w)``, not necessarily normalised on input;
  every writer normalises.
* **Scales** are stored as **log** scale. ``exp(logScales[i])`` is the
  world-space standard deviation along the corresponding rotated axis.
* **Opacity** is stored as a pre-sigmoid **logit**.
* **Colour** is a raw, unnormalised SH coefficient. Display colour is
  ``0.5 + 0.282095 * dc``, evaluated by each consumer and never baked into
  storage.
* **SH rest** coefficients are ordered by ascending ``(degree, order)``:
  degree 1 gives 3, degree 2 the next 5, degree 3 the next 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: ``Y_0,0 = 0.5 * sqrt(1/pi)``. Matches the INRIA reference and SPZ exactly.
SH_DC_TO_COLOR = 0.282095_017

#: How many *rest* (non-DC) coefficients each SH degree carries. Degree 4 is
#: out of scope: the ratified ``KHR_gaussian_splatting`` extension caps at 3
#: and the product spec asks for 0-2.
REST_COEFFICIENT_COUNT = {0: 0, 1: 3, 2: 8, 3: 15}


class ExportError(RuntimeError):
    """Something could not be written. The message is shown to a person."""


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: no ``exp`` of a large positive number."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def inverse_sigmoid(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit, clamped away from 0 and 1 so nothing becomes +-infinity."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class SplatCloud:
    """A cloud of 3D Gaussians in the storage convention above.

    Every array is checked for length agreement on construction, because the
    failure mode of a mismatched attribute array is a file that writes without
    complaint and renders as noise.
    """

    sh_degree: int
    positions: np.ndarray  # (N, 3) float32, RUB world metres
    rotations: np.ndarray  # (N, 4) float32, (x, y, z, w)
    log_scales: np.ndarray  # (N, 3) float32
    opacity_logits: np.ndarray  # (N,) float32
    color_dc: np.ndarray  # (N, 3) float32, raw SH degree-0
    sh_rest: Optional[np.ndarray] = None  # (N, rest, 3) float32, or None

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, np.float32).reshape(-1, 3)
        n = self.positions.shape[0]
        self.rotations = np.ascontiguousarray(self.rotations, np.float32).reshape(-1, 4)
        self.log_scales = np.ascontiguousarray(self.log_scales, np.float32).reshape(-1, 3)
        self.opacity_logits = np.ascontiguousarray(
            self.opacity_logits, np.float32
        ).reshape(-1)
        self.color_dc = np.ascontiguousarray(self.color_dc, np.float32).reshape(-1, 3)
        if int(self.sh_degree) not in REST_COEFFICIENT_COUNT:
            raise ExportError(
                "Unsupported spherical-harmonics degree: {d}".format(d=self.sh_degree)
            )
        rest = REST_COEFFICIENT_COUNT[int(self.sh_degree)]
        if rest == 0:
            self.sh_rest = None
        else:
            if self.sh_rest is None:
                self.sh_rest = np.zeros((n, rest, 3), dtype=np.float32)
            self.sh_rest = np.ascontiguousarray(self.sh_rest, np.float32).reshape(
                -1, rest, 3
            )
        counts = {
            "positions": n,
            "rotations": self.rotations.shape[0],
            "log_scales": self.log_scales.shape[0],
            "opacity_logits": self.opacity_logits.shape[0],
            "color_dc": self.color_dc.shape[0],
        }
        if self.sh_rest is not None:
            counts["sh_rest"] = self.sh_rest.shape[0]
        if len(set(counts.values())) != 1:
            raise ExportError(
                "Splat data is inconsistent: "
                + " ".join("{k}={v}".format(k=k, v=v) for k, v in counts.items())
            )

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def rest_count(self) -> int:
        return REST_COEFFICIENT_COUNT[int(self.sh_degree)]

    def normalized_rotations(self) -> np.ndarray:
        """Unit quaternions, with the identity substituted for degenerate rows."""
        q = np.asarray(self.rotations, dtype=np.float64)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (~np.isfinite(norm)) | (norm < 1e-12)
        return np.where(bad, np.array([0.0, 0.0, 0.0, 1.0]), q / np.where(bad, 1.0, norm))

    def bounds(self):
        if len(self) == 0:
            return (np.zeros(3, np.float32), np.zeros(3, np.float32))
        return (self.positions.min(axis=0), self.positions.max(axis=0))

    def select(self, mask: np.ndarray) -> "SplatCloud":
        """A new cloud holding only the splats ``mask`` selects."""
        mask = np.asarray(mask)
        return SplatCloud(
            sh_degree=self.sh_degree,
            positions=self.positions[mask],
            rotations=self.rotations[mask],
            log_scales=self.log_scales[mask],
            opacity_logits=self.opacity_logits[mask],
            color_dc=self.color_dc[mask],
            sh_rest=None if self.sh_rest is None else self.sh_rest[mask],
        )

    def display_colors(self) -> np.ndarray:
        """(N, 3) float32 in 0..1: the degree-0 term only, for previews.

        Deliberately ignores view-dependent SH: a preview and a point-cloud
        fallback both want the average colour, and evaluating direction-
        dependent terms without a direction would be inventing one.
        """
        return np.clip(0.5 + SH_DC_TO_COLOR * self.color_dc, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def empty(sh_degree: int = 0) -> "SplatCloud":
        rest = REST_COEFFICIENT_COUNT[int(sh_degree)]
        return SplatCloud(
            sh_degree=sh_degree,
            positions=np.zeros((0, 3), np.float32),
            rotations=np.zeros((0, 4), np.float32),
            log_scales=np.zeros((0, 3), np.float32),
            opacity_logits=np.zeros((0,), np.float32),
            color_dc=np.zeros((0, 3), np.float32),
            sh_rest=None if rest == 0 else np.zeros((0, rest, 3), np.float32),
        )
