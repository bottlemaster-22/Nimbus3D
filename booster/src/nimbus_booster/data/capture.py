"""The Python mirror of ``Core.CaptureBundle`` and its per-frame records.

``docs/DATA_FORMAT.md`` section 3 is normative. The traps this module exists to
absorb, all of which parse fine and then quietly ruin a reconstruction:

* **Quaternion order is ``(x, y, z, w)`` in JSON and in memory.** COLMAP's
  ``images.txt`` writes ``QW QX QY QZ`` and the splat PLY convention writes
  ``rot_0..rot_3`` as ``(w, x, y, z)``. Every writer does its own reordering;
  nothing reorders in memory. See :mod:`nimbus_booster.data.colmap`.
* **A pose is world -> camera**: ``X_cam = R * X_world + t``, camera ``+Z``
  forward and ``+Y`` down. The camera centre is ``C = -R^T t``.
* **Depth dimensions are read, never assumed.** 256x192 on every LiDAR iPhone
  to date, but ``settings.depthWidth`` / ``depthHeight`` is what the file
  actually contains, and a future device that changes it must not silently
  produce a garbage reshape.
* **A missing sidecar is not an error.** Section 9: every sidecar is optional
  from a reader's point of view, and the pipeline degrades to generic 3DGS
  rather than hard-failing.

``sensor_data/frames.jsonl`` is the crash-safe live log. When both it and
``capture_bundle.json`` exist, the bundle wins; when only the ``.jsonl``
survives (the app was killed mid-capture) :func:`load_bundle` rebuilds from it
and says so in :attr:`CaptureBundle.rebuilt_from_jsonl`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: ``docs/DATA_FORMAT.md`` section 9. A reader that sees a version it does not
#: know must refuse and say so, not guess.
SUPPORTED_FORMAT_VERSION = 1


class CaptureFormatError(ValueError):
    """A bundle this reader must refuse rather than misinterpret."""


# --------------------------------------------------------------------------
# Small value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    """The single shared PINHOLE camera. No distortion model, by design.

    ARKit hands back already-rectified frames with independent ``fx``/``fy``
    and no usable distortion coefficients, so four parameters is exactly right
    and any distortion model would be inventing numbers.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "Intrinsics":
        return Intrinsics(
            width=int(raw["width"]),
            height=int(raw["height"]),
            fx=float(raw["fx"]),
            fy=float(raw["fy"]),
            cx=float(raw["cx"]),
            cy=float(raw["cy"]),
        )

    def scaled(self, long_edge_pixels: int) -> Tuple["Intrinsics", float]:
        """Downscale to a training resolution, returning the new K and factor.

        Intrinsics scale with the image; getting this wrong by half a pixel is
        a systematic reprojection bias that no amount of training removes.
        """
        longest = max(self.width, self.height)
        if long_edge_pixels <= 0 or long_edge_pixels >= longest:
            return self, 1.0
        factor = float(long_edge_pixels) / float(longest)
        return (
            Intrinsics(
                width=max(1, int(round(self.width * factor))),
                height=max(1, int(round(self.height * factor))),
                fx=self.fx * factor,
                fy=self.fy * factor,
                # The half-pixel term: a pixel centre at (0.5, 0.5) in the
                # source must land at (0.5*f, 0.5*f) in the target.
                cx=(self.cx + 0.5) * factor - 0.5,
                cy=(self.cy + 0.5) * factor - 0.5,
            ),
            factor,
        )

    def matrix(self) -> np.ndarray:
        """3x3 ``K``, float32, the layout gsplat's ``Ks`` argument wants."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Pose:
    """World -> camera. ``rotation`` is ``(x, y, z, w)``, ``translation`` metres."""

    rotation: np.ndarray  # (4,) float64, (x, y, z, w)
    translation: np.ndarray  # (3,) float64

    @staticmethod
    def identity() -> "Pose":
        return Pose(
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            translation=np.zeros(3),
        )

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "Pose":
        rotation = np.asarray(raw["rotation"], dtype=np.float64).reshape(4)
        translation = np.asarray(raw["translation"], dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(rotation))
        if not math.isfinite(norm) or norm < 1e-12:
            rotation = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            rotation = rotation / norm
        return Pose(rotation=rotation, translation=translation)

    def to_json(self) -> Dict[str, Any]:
        return {
            "rotation": [float(v) for v in self.rotation],
            "translation": [float(v) for v in self.translation],
        }

    @property
    def matrix_R(self) -> np.ndarray:
        """3x3 rotation, world -> camera."""
        from ..trainer.geometry import quaternion_to_matrix

        return quaternion_to_matrix(self.rotation)

    @property
    def view_matrix(self) -> np.ndarray:
        """4x4 world -> camera, the ``viewmats`` layout gsplat wants."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.matrix_R
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def center(self) -> np.ndarray:
        """Camera optical centre in world space, ``C = -R^T t``."""
        return -self.matrix_R.T @ self.translation

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction in world space (camera ``+Z``)."""
        return self.matrix_R.T @ np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FrameQC:
    """Per-frame quality inputs and the single weight derived from them (F8).

    Every field is kept, not just the weight: the QC card has to be able to
    tell the user *which* problem cost them coverage.
    """

    angularSpeedRadPerSec: float = 0.0
    motionBlurPixels: float = 0.0
    sharpness: float = 1.0
    depthValidFraction: float = 0.0
    exposureJumpEV: float = 0.0
    trackingQuality: str = "normal"
    weight: float = 1.0

    @staticmethod
    def from_json(raw: Optional[Dict[str, Any]]) -> "FrameQC":
        if not isinstance(raw, dict):
            # No QC block at all: a bundle from an older capture, or an
            # imported COLMAP model. Weight 1 is the honest default - "no
            # reason recorded to distrust this frame".
            return FrameQC()
        return FrameQC(
            angularSpeedRadPerSec=float(raw.get("angularSpeedRadPerSec", 0.0)),
            motionBlurPixels=float(raw.get("motionBlurPixels", 0.0)),
            sharpness=float(raw.get("sharpness", 1.0)),
            depthValidFraction=float(raw.get("depthValidFraction", 0.0)),
            exposureJumpEV=float(raw.get("exposureJumpEV", 0.0)),
            trackingQuality=str(raw.get("trackingQuality", "normal")),
            weight=float(raw.get("weight", 1.0)),
        )

    @property
    def is_pose_trustworthy(self) -> bool:
        """Whether this frame may act as a pose-graph constraint (F1).

        A frame captured while tracking was limited is still trained on - it is
        down-weighted, never dropped, because throwing data away is how you get
        a hole in the scan you cannot explain to the user - but it must not
        anchor other frames' poses.
        """
        return self.trackingQuality == "normal"


@dataclass
class CaptureFrame:
    """One captured frame. Paths are POSIX and relative to the scan root."""

    index: int
    timestampSeconds: float
    imagePath: str
    depthPath: Optional[str]
    confidencePath: Optional[str]
    rawPose: Pose
    refinedPose: Optional[Pose]
    exposureDurationSeconds: float
    exposureOffsetEV: float
    iso: Optional[float]
    angularVelocity: np.ndarray
    qc: FrameQC
    bracket: str
    submap: Optional[int]

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "CaptureFrame":
        refined = raw.get("refinedPose")
        angular = raw.get("angularVelocity") or [0.0, 0.0, 0.0]
        return CaptureFrame(
            index=int(raw["index"]),
            timestampSeconds=float(raw.get("timestampSeconds", 0.0)),
            imagePath=str(raw["imagePath"]),
            depthPath=raw.get("depthPath"),
            confidencePath=raw.get("confidencePath"),
            rawPose=Pose.from_json(raw["rawPose"]),
            refinedPose=Pose.from_json(refined) if isinstance(refined, dict) else None,
            exposureDurationSeconds=float(raw.get("exposureDurationSeconds", 0.0)),
            exposureOffsetEV=float(raw.get("exposureOffsetEV", 0.0)),
            iso=(float(raw["iso"]) if raw.get("iso") is not None else None),
            angularVelocity=np.asarray(angular, dtype=np.float64).reshape(3),
            qc=FrameQC.from_json(raw.get("qc")),
            bracket=str(raw.get("bracket", "normal")),
            submap=(int(raw["submap"]) if raw.get("submap") is not None else None),
        )

    @property
    def pose(self) -> Pose:
        """The best pose available: refined if the pre-pass ran, else raw.

        The raw track is never overwritten (it is evidence), so "which pose"
        is a read-time decision and this is the one place that makes it.
        """
        return self.refinedPose or self.rawPose

    @property
    def is_darker_bracket(self) -> bool:
        return self.bracket == "darker"

    @property
    def relative_exposure(self) -> float:
        """Linear exposure scale relative to EV 0, from duration and offset.

        Used to put a deliberately-darker bracketed frame (F5) on equal terms
        with its neighbours before any photometric loss. Bracketed frames are
        the only correctly-exposed evidence for a bright window, so they are
        never filtered out - they are just compared fairly.
        """
        duration = max(1e-6, float(self.exposureDurationSeconds))
        return duration * float(2.0 ** self.exposureOffsetEV)


@dataclass
class CaptureSettings:
    """Capture-time settings a downstream stage must know to read the frames."""

    bracketEveryNFrames: int = 0
    bracketStops: float = 0.0
    exposureLocked: bool = False
    whiteBalanceLocked: bool = False
    depthWidth: int = 0
    depthHeight: int = 0
    lidarMaxRangeMeters: float = 5.0

    @staticmethod
    def from_json(raw: Optional[Dict[str, Any]]) -> "CaptureSettings":
        if not isinstance(raw, dict):
            return CaptureSettings()
        return CaptureSettings(
            bracketEveryNFrames=int(raw.get("bracketEveryNFrames", 0)),
            bracketStops=float(raw.get("bracketStops", 0.0)),
            exposureLocked=bool(raw.get("exposureLocked", False)),
            whiteBalanceLocked=bool(raw.get("whiteBalanceLocked", False)),
            depthWidth=int(raw.get("depthWidth", 0)),
            depthHeight=int(raw.get("depthHeight", 0)),
            lidarMaxRangeMeters=float(raw.get("lidarMaxRangeMeters", 5.0)),
        )

    @property
    def has_native_depth(self) -> bool:
        return self.depthWidth > 0 and self.depthHeight > 0


@dataclass
class RevisitPair:
    """A detected loop closure (F1): the same place, seen again later."""

    frameA: int
    frameB: int
    method: str
    measuredRelativePose: Pose
    translationResidualMeters: float
    rotationResidualDegrees: float
    inlierCount: int
    confidence: float

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "RevisitPair":
        return RevisitPair(
            frameA=int(raw["frameA"]),
            frameB=int(raw["frameB"]),
            method=str(raw.get("method", "unknown")),
            measuredRelativePose=Pose.from_json(raw["measuredRelativePose"]),
            translationResidualMeters=float(raw.get("translationResidualMeters", 0.0)),
            rotationResidualDegrees=float(raw.get("rotationResidualDegrees", 0.0)),
            inlierCount=int(raw.get("inlierCount", 0)),
            confidence=float(raw.get("confidence", 0.0)),
        )


@dataclass
class BoundingBox:
    min: np.ndarray
    max: np.ndarray

    @staticmethod
    def from_json(raw: Optional[Dict[str, Any]]) -> Optional["BoundingBox"]:
        if not isinstance(raw, dict):
            return None
        return BoundingBox(
            min=np.asarray(raw["min"], dtype=np.float64).reshape(3),
            max=np.asarray(raw["max"], dtype=np.float64).reshape(3),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "min": [float(v) for v in self.min],
            "max": [float(v) for v in self.max],
        }

    @property
    def longest_edge_meters(self) -> float:
        return float(np.max(self.max - self.min))


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


@dataclass
class CaptureBundle:
    """One capture session, as an index over the files on disk.

    This is the index, not the data: pixels, depth and mesh stay in their own
    files and are opened lazily, so a whole-house scan does not have to be
    resident in memory to be described.
    """

    root: Path
    formatVersion: int
    scanID: str
    createdAt: str
    displayName: str
    deviceModel: str
    appVersion: str
    intrinsics: Intrinsics
    settings: CaptureSettings
    frames: List[CaptureFrame]
    cameraToIMUTimeOffsetSeconds: Optional[float] = None
    revisitPairs: List[RevisitPair] = field(default_factory=list)
    sceneBounds: Optional[BoundingBox] = None
    pointCloudPath: str = "sparse/0/points3D.txt"
    meshChunks: List[Dict[str, Any]] = field(default_factory=list)
    anchorsDuringSession: List[Dict[str, Any]] = field(default_factory=list)
    anchorsAtEndOfSession: List[Dict[str, Any]] = field(default_factory=list)
    #: True when ``capture_bundle.json`` was absent and this was reconstructed
    #: from the append-only ``frames.jsonl``. Reported, never hidden: a
    #: rebuilt bundle has no anchors, no revisit pairs and no scene bounds.
    rebuilt_from_jsonl: bool = False

    # -- paths -------------------------------------------------------------

    def path(self, relative: Optional[str]) -> Optional[Path]:
        """Resolve a bundle-relative POSIX path against the scan root."""
        if not relative:
            return None
        candidate = self.root
        for part in str(relative).split("/"):
            if part in ("", ".", ".."):
                # There is no absolute path anywhere in a scan and no "..".
                # A bundle containing one is corrupt or hostile; either way it
                # is not opened.
                return None
            candidate = candidate / part
        return candidate

    def image_path(self, frame: CaptureFrame) -> Optional[Path]:
        return self.path(frame.imagePath)

    def depth_path(self, frame: CaptureFrame) -> Optional[Path]:
        return self.path(frame.depthPath)

    def confidence_path(self, frame: CaptureFrame) -> Optional[Path]:
        return self.path(frame.confidencePath)

    @property
    def prepass_dir(self) -> Path:
        return self.root / "prepass"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def has_prepass(self) -> bool:
        return (self.prepass_dir / "prepass_result.json").is_file()

    # -- frames ------------------------------------------------------------

    def frames_with_depth(self) -> List[CaptureFrame]:
        return [f for f in self.frames if f.depthPath]

    def by_index(self) -> Dict[int, CaptureFrame]:
        return {f.index: f for f in self.frames}

    def poses(self) -> np.ndarray:
        """(N, 4, 4) float32 world -> camera, in ``self.frames`` order."""
        return np.stack([f.pose.view_matrix for f in self.frames]).astype(np.float32)

    def centers(self) -> np.ndarray:
        """(N, 3) float64 camera centres in world space."""
        return np.stack([f.pose.center for f in self.frames])

    def estimated_bounds(self) -> BoundingBox:
        """Scene extent, from ``sceneBounds`` when present, else the track.

        The fallback is deliberately the camera track padded by the LiDAR
        range rather than a fixed box: "never assume a scene fits" applies to
        the budget calculation as much as to memory.
        """
        if self.sceneBounds is not None:
            return self.sceneBounds
        if not self.frames:
            return BoundingBox(min=np.zeros(3), max=np.ones(3))
        centers = self.centers()
        pad = float(self.settings.lidarMaxRangeMeters)
        return BoundingBox(
            min=centers.min(axis=0) - pad,
            max=centers.max(axis=0) + pad,
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(root: Path) -> CaptureBundle:
    """Open the scan folder at ``root``.

    Prefers ``capture_bundle.json``; falls back to rebuilding from
    ``sensor_data/frames.jsonl`` when the app was killed mid-capture and only
    the append-only log survived.

    Raises :class:`CaptureFormatError` for a format version this reader does
    not know, per ``docs/DATA_FORMAT.md`` section 9: refuse and say so, do not
    guess.
    """
    root = Path(root)
    bundle_path = root / "capture_bundle.json"
    if bundle_path.is_file():
        raw = _read_json(bundle_path)
        if not isinstance(raw, dict):
            raise CaptureFormatError("capture_bundle.json is not a JSON object.")
        version = int(raw.get("formatVersion", 0))
        if version != SUPPORTED_FORMAT_VERSION:
            raise CaptureFormatError(
                "This scan is in capture format version {v}; this Booster "
                "understands version {ours}. Update the Booster.".format(
                    v=version, ours=SUPPORTED_FORMAT_VERSION
                )
            )
        frames = [CaptureFrame.from_json(f) for f in raw.get("frames", [])]
        frames.sort(key=lambda f: f.index)
        return CaptureBundle(
            root=root,
            formatVersion=version,
            scanID=str(raw["scanID"]),
            createdAt=str(raw.get("createdAt", "")),
            displayName=str(raw.get("displayName", "")),
            deviceModel=str(raw.get("deviceModel", "")),
            appVersion=str(raw.get("appVersion", "")),
            intrinsics=Intrinsics.from_json(raw["intrinsics"]),
            settings=CaptureSettings.from_json(raw.get("settings")),
            frames=frames,
            cameraToIMUTimeOffsetSeconds=raw.get("cameraToIMUTimeOffsetSeconds"),
            revisitPairs=[
                RevisitPair.from_json(p) for p in raw.get("revisitPairs", [])
            ],
            sceneBounds=BoundingBox.from_json(raw.get("sceneBounds")),
            pointCloudPath=str(raw.get("pointCloudPath", "sparse/0/points3D.txt")),
            meshChunks=list(raw.get("meshChunks", [])),
            anchorsDuringSession=list(raw.get("anchorsDuringSession", [])),
            anchorsAtEndOfSession=list(raw.get("anchorsAtEndOfSession", [])),
        )

    jsonl = root / "sensor_data" / "frames.jsonl"
    if not jsonl.is_file():
        raise CaptureFormatError(
            "This folder has neither capture_bundle.json nor "
            "sensor_data/frames.jsonl, so there is no scan in it."
        )
    return _rebuild_from_jsonl(root, jsonl)


def _rebuild_from_jsonl(root: Path, jsonl: Path) -> CaptureBundle:
    """Reconstruct what can be reconstructed from the crash-safe live log.

    Deliberately partial and honest about it: the ``.jsonl`` holds frames and
    nothing else, so anchors, revisit pairs, mesh chunks and scene bounds come
    back empty and ``rebuilt_from_jsonl`` is set. Intrinsics are not in the
    log either, so they are read from ``sparse/0/cameras.txt``, which is the
    only other place they exist.
    """
    from .colmap import read_cameras

    frames: List[CaptureFrame] = []
    with open(jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(CaptureFrame.from_json(json.loads(line)))
            except (ValueError, KeyError):
                # A kill -9 can leave a half-written final line. Everything
                # before it is intact, which is the whole point of the format.
                continue
    frames.sort(key=lambda f: f.index)
    if not frames:
        raise CaptureFormatError("sensor_data/frames.jsonl contains no usable frames.")

    cameras_txt = root / "sparse" / "0" / "cameras.txt"
    if not cameras_txt.is_file():
        raise CaptureFormatError(
            "This scan was interrupted and its camera description "
            "(sparse/0/cameras.txt) is missing, so its frames cannot be used."
        )
    intrinsics = read_cameras(cameras_txt)

    depth_shape = _sniff_depth_shape(root, frames, intrinsics)
    settings = CaptureSettings(
        depthWidth=depth_shape[0],
        depthHeight=depth_shape[1],
    )
    return CaptureBundle(
        root=root,
        formatVersion=SUPPORTED_FORMAT_VERSION,
        scanID=root.name,
        createdAt="",
        displayName=root.name,
        deviceModel="",
        appVersion="",
        intrinsics=intrinsics,
        settings=settings,
        frames=frames,
        rebuilt_from_jsonl=True,
    )


def _sniff_depth_shape(
    root: Path, frames: Sequence[CaptureFrame], intrinsics: Intrinsics
) -> Tuple[int, int]:
    """Recover the native depth dimensions when settings are unavailable.

    Only reached on the rebuild-from-jsonl path. The file size fixes the
    product ``w * h`` (two bytes per sample); the RGB aspect ratio picks the
    factorisation. If nothing fits, this returns ``(0, 0)`` and the pipeline
    runs without depth supervision rather than reshaping garbage.
    """
    for frame in frames:
        if not frame.depthPath:
            continue
        candidate = root
        for part in frame.depthPath.split("/"):
            candidate = candidate / part
        if not candidate.is_file():
            continue
        samples = candidate.stat().st_size // 2
        if samples <= 0:
            continue
        aspect = float(intrinsics.width) / max(1.0, float(intrinsics.height))
        height = int(round(math.sqrt(samples / aspect)))
        for h in range(max(1, height - 4), height + 5):
            if h > 0 and samples % h == 0:
                return (samples // h, h)
        return (0, 0)
    return (0, 0)


def iter_frames_with_images(bundle: CaptureBundle) -> Iterator[CaptureFrame]:
    """Frames whose JPEG is actually on disk.

    A bundle is an index; the Booster receives it over a LAN and a file can be
    missing for boring reasons. Training against a manifest entry with no
    pixels behind it is a crash halfway through a forty-minute run.
    """
    for frame in bundle.frames:
        path = bundle.image_path(frame)
        if path is not None and path.is_file():
            yield frame
