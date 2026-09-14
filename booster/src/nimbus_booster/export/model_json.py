"""``model/model.json`` - the Python side of ``Core.SplatModel``.

The phone decodes this with Foundation's synthesised ``Codable``, which means
the two rules from ``docs/BOOSTER_PROTOCOL.md`` section 4 apply here too even
though this is a file rather than a wire message:

* **Dates are ISO-8601 to whole seconds.** ``JSONDecoder`` with the ``.iso8601``
  strategy rejects fractional seconds and takes the whole document with it.
* **A non-optional field must be present.** Everything typed ``T?`` in
  ``Contracts.swift`` may be omitted; nothing else may.

``SHDegree`` encodes as a bare integer 0-3 (Core adds the ``Codable``
conformance to a ``RawRepresentable`` whose ``RawValue`` is ``Int``), not as a
string and not as an object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..protocol.wire import iso8601


@dataclass
class TrainingBudgetJSON:
    """``Core.TrainingBudget``, as the Booster actually ran it.

    Written out rather than assumed so the phone can show what the run really
    did after any live degradation, instead of what was requested. Everything
    here is a measured or chosen number, never a placeholder.
    """

    splatCap: int
    iterations: int
    renderLongEdgePixels: int
    shDegree: int
    keyframeCount: int
    memoryCeilingBytes: int
    useHalfPrecision: bool = True
    useSparseAdam: bool = True
    #: ``TrainingTarget``. Always ``booster`` for a run that happened here.
    target: str = "booster"
    #: ``ThermalPolicy``. A desktop GPU has no thermal gate of the kind the
    #: phone needs, but the field is non-optional on the Swift side, so the
    #: defaults are written rather than omitted.
    thermalPolicy: Dict[str, Any] = field(
        default_factory=lambda: {
            "degradeAt": 1,
            "pauseAt": 2,
            "abortAt": 3,
            "sampleIntervalSeconds": 5,
        }
    )

    def to_json(self) -> Dict[str, Any]:
        return {
            "splatCap": int(self.splatCap),
            "iterations": int(self.iterations),
            "renderLongEdgePixels": int(self.renderLongEdgePixels),
            "shDegree": int(self.shDegree),
            "keyframeCount": int(self.keyframeCount),
            "memoryCeilingBytes": int(self.memoryCeilingBytes),
            "useHalfPrecision": bool(self.useHalfPrecision),
            "useSparseAdam": bool(self.useSparseAdam),
            "target": self.target,
            "thermalPolicy": dict(self.thermalPolicy),
        }


def bounding_box_json(minimum: np.ndarray, maximum: np.ndarray) -> Dict[str, Any]:
    """``Core.BoundingBox``. ``Vector3`` encodes as the compact ``[x, y, z]``."""
    return {
        "min": [float(v) for v in np.asarray(minimum).reshape(3)],
        "max": [float(v) for v in np.asarray(maximum).reshape(3)],
    }


def write_model_json(
    path: Path,
    *,
    model_id: str,
    scan_id: str,
    ply_path: Optional[str],
    spz_path: Optional[str],
    splat_count: int,
    sh_degree: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    iterations_completed: int,
    budget: Optional[TrainingBudgetJSON],
    background_model_path: Optional[str] = None,
    exposure_path: Optional[str] = None,
    observed_directions_path: Optional[str] = None,
    held_out_psnr: Optional[float] = None,
    created_at: Optional[str] = None,
) -> Path:
    """Write ``model.json``. Returns ``path``.

    ``heldOutPSNR`` is written when held-out evaluation ran and omitted when it
    did not. It is never faked and never rounded up: it is reported even when
    it is bad, because a quality number that only appears when it flatters the
    result is not a quality number.
    """
    if not ply_path and not spz_path:
        raise ValueError(
            "at least one of plyPath / spzPath must be set: SplatModel promises it"
        )
    payload: Dict[str, Any] = {
        "modelID": model_id,
        "scanID": scan_id,
        "createdAt": created_at or iso8601(),
        # ModelSource. This run happened on the PC, so it is `booster` even if
        # the phone later re-exports it.
        "source": "booster",
        "splatCount": int(splat_count),
        "shDegree": int(sh_degree),
        "bounds": bounding_box_json(bounds_min, bounds_max),
        "iterationsCompleted": int(iterations_completed),
    }
    if ply_path is not None:
        payload["plyPath"] = ply_path
    if spz_path is not None:
        payload["spzPath"] = spz_path
    if budget is not None:
        payload["budgetUsed"] = budget.to_json()
    if background_model_path is not None:
        payload["backgroundModelPath"] = background_model_path
    if exposure_path is not None:
        payload["exposurePath"] = exposure_path
    if observed_directions_path is not None:
        payload["observedDirectionsPath"] = observed_directions_path
    if held_out_psnr is not None and np.isfinite(held_out_psnr):
        payload["heldOutPSNR"] = float(held_out_psnr)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
