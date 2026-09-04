"""COLMAP text model reader and writer. ``docs/DATA_FORMAT.md`` section 4.

Nothing here runs COLMAP and nothing should: reconstructing poses from scratch
throws away a good ARKit prior, and triangulate-then-bundle-adjust from that
prior measurably *degraded* it in 15 of 15 test rooms (arXiv 2608.21008). The
format is COLMAP-shaped for a duller reason - it is the lingua franca of every
splat tool in existence.

Three details that make the difference between a file COLMAP reads and one it
rejects:

1. ``images.txt`` has **two lines per record** and the second is empty. This
   app does not triangulate 2D-3D correspondences, so there are no
   ``POINTS2D`` - but COLMAP's reader requires the line to exist, and omitting
   it makes the file unparseable by half the tools that claim to read it.
2. Quaternions are written ``QW QX QY QZ``. In memory they are
   ``(x, y, z, w)``. The reorder happens here and nowhere else.
3. ``IMAGE_ID`` is ``CaptureFrame.index + 1`` - COLMAP ids are 1-based.

``points3D.txt``'s ``ERROR`` column is a deliberate reinterpretation: it holds
the **expected metric error in metres** from the physics prior (F6), not
COLMAP's reprojection error in pixels. It is the only per-point uncertainty
slot the format has and leaving it at zero would be a lie. Anything reading a
scan from this app should read that column as metres; it is documented in the
file's own header comment so nothing downstream has to guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .capture import CaptureFrame, Intrinsics, Pose

_CAMERAS_HEADER = (
    "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    "# Number of cameras: 1\n"
)

_IMAGES_HEADER = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
)

_POINTS_HEADER = (
    "# 3D point list with one line of data per point:\n"
    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    "# NOTE: ERROR is the expected metric error in METRES from this app's\n"
    "#   physics prior (grows with range and grazing incidence), not COLMAP's\n"
    "#   reprojection error in pixels. See docs/DATA_FORMAT.md section 4.\n"
)


class ColmapFormatError(ValueError):
    """A COLMAP text file this reader refuses rather than half-understands."""


# --------------------------------------------------------------------------
# cameras.txt
# --------------------------------------------------------------------------


def read_cameras(path: Path) -> Intrinsics:
    """Read the single shared PINHOLE camera.

    Accepts ``PINHOLE`` (four params) and ``SIMPLE_PINHOLE`` (three, shared
    focal) so an imported model from another tool works. Anything with
    distortion coefficients is refused by name rather than silently having them
    dropped, because dropping them is a systematic reprojection error that
    looks like a bad trainer.
    """
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        model = parts[1].upper()
        width, height = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        if model == "PINHOLE" and len(params) >= 4:
            return Intrinsics(width, height, params[0], params[1], params[2], params[3])
        if model == "SIMPLE_PINHOLE" and len(params) >= 3:
            return Intrinsics(width, height, params[0], params[0], params[1], params[2])
        raise ColmapFormatError(
            "This scan's camera model is {model}, which carries lens "
            "distortion this Booster does not correct for. Only PINHOLE and "
            "SIMPLE_PINHOLE are read.".format(model=model)
        )
    raise ColmapFormatError("cameras.txt contains no camera.")


def write_cameras(path: Path, intrinsics: Intrinsics, camera_id: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "{id} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n".format(
        id=camera_id,
        w=intrinsics.width,
        h=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    path.write_text(_CAMERAS_HEADER + body, encoding="utf-8")


# --------------------------------------------------------------------------
# images.txt
# --------------------------------------------------------------------------


def read_images(path: Path) -> Dict[int, Tuple[Pose, str]]:
    """``{image_id: (pose, name)}``. ``pose`` is world -> camera, as stored."""
    out: Dict[int, Tuple[Pose, str]] = {}
    expect_points_line = False
    for line in _data_lines(path, keep_blank=True):
        if expect_points_line:
            # The mandatory second line, empty in our own files and full of
            # 2D observations in a real COLMAP model. Skipped either way.
            expect_points_line = False
            continue
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        tx, ty, tz = (float(v) for v in parts[5:8])
        name = parts[9]
        rotation = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(rotation))
        if norm > 1e-12:
            rotation = rotation / norm
        else:
            rotation = np.array([0.0, 0.0, 0.0, 1.0])
        out[image_id] = (
            Pose(rotation=rotation, translation=np.array([tx, ty, tz])),
            name,
        )
        expect_points_line = True
    return out


def write_images(
    path: Path,
    frames: Sequence[CaptureFrame],
    poses: Optional[Sequence[Pose]] = None,
    camera_id: int = 1,
) -> None:
    """Write one record per frame, each as two lines, the second one empty.

    ``poses`` overrides ``frame.pose`` positionally, which is how
    ``prepass/sparse_refined/images.txt`` is written without mutating the raw
    track in ``sparse/0/``.
    """
    if poses is not None and len(poses) != len(frames):
        raise ValueError("poses and frames must be the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = [
        _IMAGES_HEADER,
        "# Number of images: {n}\n".format(n=len(frames)),
    ]
    for i, frame in enumerate(frames):
        pose = poses[i] if poses is not None else frame.pose
        qx, qy, qz, qw = (float(v) for v in pose.rotation)
        tx, ty, tz = (float(v) for v in pose.translation)
        name = frame.imagePath.rsplit("/", 1)[-1]
        chunks.append(
            "{id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            "{tx:.9f} {ty:.9f} {tz:.9f} {cam} {name}\n".format(
                id=int(frame.index) + 1,  # COLMAP ids are 1-based
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                tx=tx,
                ty=ty,
                tz=tz,
                cam=camera_id,
                name=name,
            )
        )
        # The mandatory empty POINTS2D line. Omitting it makes the file
        # unreadable by half the tools that claim to read COLMAP.
        chunks.append("\n")
    path.write_text("".join(chunks), encoding="utf-8")


# --------------------------------------------------------------------------
# points3D.txt
# --------------------------------------------------------------------------


class PointCloud:
    """The baked LiDAR cloud: positions, colours and per-point metric error."""

    __slots__ = ("xyz", "rgb", "error")

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, error: np.ndarray) -> None:
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
        self.rgb = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1, 3)
        self.error = np.ascontiguousarray(error, dtype=np.float32).reshape(-1)
        n = self.xyz.shape[0]
        if self.rgb.shape[0] != n or self.error.shape[0] != n:
            raise ValueError("point cloud attribute counts disagree")

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_empty:
            return (np.zeros(3, np.float32), np.zeros(3, np.float32))
        return (self.xyz.min(axis=0), self.xyz.max(axis=0))


def read_points3D(path: Path) -> PointCloud:
    """Read ``points3D.txt``. Missing or empty gives an empty cloud, not a raise.

    Section 9: nothing hard-fails on a missing sidecar. An empty cloud means
    the initialiser falls back to random points inside the camera hull, which
    is worse but is still a scan.
    """
    if not Path(path).is_file():
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
    error: List[float] = []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb.append((int(parts[4]), int(parts[5]), int(parts[6])))
            error.append(float(parts[7]))
        except ValueError:
            continue
    if not xyz:
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    return PointCloud(
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rgb, dtype=np.uint8),
        np.asarray(error, dtype=np.float32),
    )


def write_points3D(path: Path, cloud: PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = [
        _POINTS_HEADER,
        "# Number of points: {n}\n".format(n=len(cloud)),
    ]
    for i in range(len(cloud)):
        x, y, z = cloud.xyz[i]
        r, g, b = cloud.rgb[i]
        chunks.append(
            "{id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {e:.6f}\n".format(
                id=i + 1, x=x, y=y, z=z, r=int(r), g=int(g), b=int(b),
                e=float(cloud.error[i]),
            )
        )
    path.write_text("".join(chunks), encoding="utf-8")


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


def _data_lines(path: Path, keep_blank: bool = False) -> Iterable[str]:
    path = Path(path)
    if not path.is_file():
        raise ColmapFormatError("missing COLMAP file: {p}".format(p=path.name))
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith("#"):
                continue
            if not line.strip() and not keep_blank:
                continue
            yield line


def voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, error: np.ndarray, voxel_meters: float
) -> PointCloud:
    """Collapse points onto a regular grid, keeping one representative each.

    ``docs/DATA_FORMAT.md``: the cloud is voxel-downsampled at 1 cm before
    writing, because a raw union of every frame's 49k samples over a
    four-minute walk is hundreds of millions of redundant points.

    The representative is the **mean** position and colour, but the **minimum**
    error, deliberately: averaging uncertainty would make a cell containing one
    good measurement and nine bad ones look mediocre, when in fact the good one
    is what the cell is worth.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0 or voxel_meters <= 0:
        return PointCloud(xyz, rgb, error)
    keys = np.floor(xyz / float(voxel_meters)).astype(np.int64)
    # A structured view gives a single sortable key per row without hashing
    # into a Python dict, which matters at a hundred million points.
    view = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).reshape(-1)
    _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
    n_cells = int(counts.shape[0])

    sums = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(sums, inverse, xyz)
    means = sums / counts[:, None]

    rgb64 = np.ascontiguousarray(rgb, dtype=np.float64).reshape(-1, 3)
    colour_sums = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(colour_sums, inverse, rgb64)
    colours = np.clip(colour_sums / counts[:, None], 0, 255).astype(np.uint8)

    err = np.ascontiguousarray(error, dtype=np.float64).reshape(-1)
    best = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(best, inverse, err)
    best[~np.isfinite(best)] = 0.0

    return PointCloud(means.astype(np.float32), colours, best.astype(np.float32))
