"""One received scan, end to end: bundle on disk in, exported model out.

This is the piece that makes a job arriving over the wire actually reach the
trainer. It is deliberately the only module that knows the whole order of
operations, so the server can stay ignorant of the trainer and the trainer can
stay ignorant of HTTP.

What runs here today, for real:

* :func:`nimbus_booster.data.capture.load_bundle` opens the uploaded scan,
  including the rebuild-from-``frames.jsonl`` path for a capture the phone was
  killed part-way through;
* every depth frame is unprojected on its **native** grid with the correct
  half-pixel convention, normals are estimated on that same grid, and the
  samples are collapsed onto a voxel grid keeping the most trusted sample per
  cell;
* :func:`nimbus_booster.trainer.init_lidar.initial_cloud_from_samples` turns
  those samples into normal-aligned Gaussian discs (F);
* :class:`nimbus_booster.trainer.carving.OccupancyAccumulator` carves free
  space from the same frames, and only cells certified EMPTY license a delete
  (F2);
* the result is written as ``model.ply``, ``model.spz``, ``model.json`` and a
  real EWA-splat ``preview.png``, all of which the phone can download.

What does NOT run here today, stated plainly because the owner is going to be
told the truth rather than a number: **there is no photometric optimisation
loop yet.** ``trainer/train.py`` (the gsplat-kernel optimiser, densification,
background model and pose refinement) has not been written. Until it lands,
what comes back is the LiDAR measurement itself, exported as splats: correct
geometry at the resolution the sensor measured, colour lifted straight from
the photos, and none of the sharpening that optimisation buys.

The hook is already here and is not a stub: if a module named
``nimbus_booster.trainer.train`` exists and exposes ``refine``, this pipeline
calls it and reports a genuinely optimised result. See
:func:`_load_optimiser` for the exact signature expected.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..data import capture as capture_data
from ..data import colmap as colmap_data
from ..data import sidecars
from ..export import model_json, ply, preview, spz
from ..export.splat_cloud import SplatCloud
from . import carving, geometry, init_lidar

log = logging.getLogger(__name__)

#: Phase names handed to the progress callback. They are the trainer's own
#: words, not wire values; the caller maps them onto whatever it reports.
PHASE_READING = "reading"
PHASE_BUILDING = "building"
PHASE_EXPORTING = "exporting"

#: Collapse depth samples onto this grid before initialisation. One centimetre
#: is what docs/DATA_FORMAT.md uses for the baked cloud, and it is finer than
#: the LiDAR resolves at any sensible range, so it costs nothing real.
DEFAULT_SAMPLE_VOXEL_METERS = 0.01

#: Hard ceiling on samples fed to the initialiser regardless of the configured
#: splat cap. Disc initialisation does one 3x3 SVD per sample in Python, so the
#: cost is linear but not free: a few million is minutes, tens of millions is
#: an evening. The cap is named here rather than hidden in a loop.
INITIALISER_SAMPLE_CEILING = 1_500_000

#: Confidence 0/1/2 mapped to a starting trust. This is a MAPPING, not a
#: calibration: ARKit's three levels are a ranking. The real per-sample trust
#: field (F6) is not built, so a scan initialised through here uses these.
_CONFIDENCE_TRUST = (0.35, 0.60, 0.90)

#: Trust used when a frame has no confidence sidecar at all.
_TRUST_WITHOUT_CONFIDENCE = 0.6


class PipelineError(RuntimeError):
    """Something stopped the build. The message is read by a person, verbatim."""


class PipelineCancelled(RuntimeError):
    """The user cancelled. Not an error, and not reported as one."""


@dataclass
class PipelineOptions:
    """The knobs the GUI and the config expose, in the trainer's own terms."""

    max_splats: int = 2_000_000
    sh_degree: int = 2
    iterations: int = 30_000
    render_long_edge: int = 1600
    preview_long_edge: int = 1024
    sample_voxel_meters: float = DEFAULT_SAMPLE_VOXEL_METERS
    #: Frames used for initialisation and carving. Every frame is read for
    #: carving; this caps how many contribute samples, so a ten-minute walk
    #: does not spend an hour on redundant surface.
    max_sample_frames: int = 600
    write_spz: bool = True


@dataclass
class PipelineResult:
    """What was produced, and an honest description of how."""

    result_dir: Path
    files: List[Path] = field(default_factory=list)
    splat_count: int = 0
    #: True only when a real optimisation loop ran over the initialisation.
    optimised: bool = False
    #: ``lidar`` (depth frames) or ``points3D`` (the baked cloud fallback).
    initialisation: str = "lidar"
    iterations_completed: int = 0
    #: One plain sentence for the user, naming what did and did not happen.
    summary: str = ""


ReportCallback = Callable[[str, str, Optional[float]], None]
CancelCallback = Callable[[], bool]


def _noop_report(_phase: str, _message: str, _fraction: Optional[float]) -> None:
    return None


def _never_cancelled() -> bool:
    return False


def _load_optimiser():
    """Return ``trainer.train`` if it exists, else ``None``.

    The module is expected to expose::

        refine(cloud, bundle, options, report, is_cancelled) -> (SplatCloud, int)

    returning the optimised cloud and the number of iterations actually
    completed. Nothing calls into a half-written optimiser: if the import
    fails for any reason the pipeline says so and exports the initialisation,
    which is a real result rather than a blank screen.
    """
    try:
        from . import train  # type: ignore  # noqa: PLC0415
    except ImportError:
        return None
    if not hasattr(train, "refine"):
        log.warning("trainer.train exists but has no refine(); skipping it.")
        return None
    return train


def optimiser_available() -> bool:
    """Whether a real photometric optimisation loop is installed and usable."""
    return _load_optimiser() is not None


def capability_sentence(capability) -> str:
    """What this PC can honestly do right now, for the window and the CLI.

    :func:`nimbus_booster.trainer.torch_support.probe` answers a narrower
    question than a person is asking: it reports whether the gsplat optimiser
    could run. Today the answer to that is irrelevant to whether a scan comes
    back, because the pipeline builds from the depth measurements in numpy and
    needs no GPU at all. Saying "this PC cannot build scans" on a machine that
    demonstrably does build them would be the kind of wrong the owner catches.
    """
    lead = (
        "This PC can receive scans and build them from the depth measurements. "
        "That part works with or without a graphics card."
    )
    if not optimiser_available():
        return (
            lead
            + " The step that refines a scan against your photos is not "
            "written yet, so nothing here uses the graphics card today."
        )
    if getattr(capability, "can_train", False):
        return lead + " " + capability.summary
    return lead + " " + capability.reason


# --------------------------------------------------------------------------
# Reading one frame's samples
# --------------------------------------------------------------------------


@dataclass
class _FrameSamples:
    positions: np.ndarray
    normals: np.ndarray
    view_directions: np.ndarray
    ranges: np.ndarray
    colors: np.ndarray
    trust: np.ndarray
    #: False when the frame's photo could not be decoded and the colour below
    #: is a neutral grey rather than a measurement.
    has_photo: bool = True


def _load_rgb_for_depth_grid(
    bundle: "capture_data.CaptureBundle",
    frame: "capture_data.CaptureFrame",
    depth_width: int,
    depth_height: int,
) -> Optional[np.ndarray]:
    """The frame's photo resampled onto the native depth grid, 0..1 float.

    Pillow is an optional dependency (``[preview]`` / ``[train]``). Without it
    there is no JPEG decoder in the standard library, so this returns ``None``
    and the caller uses a neutral grey and says so. A grey scan that is
    geometrically right beats no scan, and pretending it has colour would be
    the dishonest option.
    """
    path = bundle.image_path(frame)
    if path is None or not path.is_file():
        return None
    try:
        from PIL import Image  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    try:
        with Image.open(str(path)) as image:
            rgb = image.convert("RGB").resize(
                (int(depth_width), int(depth_height)), Image.BILINEAR
            )
            return np.asarray(rgb, dtype=np.float64) / 255.0
    except Exception as error:  # noqa: BLE001 - one bad JPEG is not fatal
        log.warning("Could not read %s: %s", path, error)
        return None


def _frame_samples(
    bundle: "capture_data.CaptureBundle",
    frame: "capture_data.CaptureFrame",
    depth: np.ndarray,
    K_depth: np.ndarray,
) -> Tuple[Optional[_FrameSamples], np.ndarray, np.ndarray, np.ndarray]:
    """Unproject one frame. Returns ``(samples, world, valid, centre)``.

    ``world``, ``valid`` and ``centre`` come back even when there are no usable
    samples, because free-space carving wants every frame's rays, including
    the ones that measured nothing.
    """
    height, width = depth.shape
    view = frame.pose.view_matrix
    centre = frame.pose.center
    world, valid = geometry.unproject_depth_map(depth, K_depth, view)
    if not bool(valid.any()):
        return None, world, valid, centre

    normals = geometry.estimate_normals(world, valid, (height, width))

    indices = np.flatnonzero(valid)
    positions = world[indices]
    offsets = positions - centre[None, :]
    ranges = np.linalg.norm(offsets, axis=1)
    usable = ranges > 1e-4
    if not bool(usable.any()):
        return None, world, valid, centre
    indices = indices[usable]
    positions = positions[usable]
    offsets = offsets[usable]
    ranges = ranges[usable]
    directions = offsets / ranges[:, None]

    confidence_path = bundle.confidence_path(frame)
    confidence = None
    if confidence_path is not None:
        confidence = sidecars.read_confidence8(confidence_path, width, height)
    if confidence is None:
        trust = np.full(indices.shape[0], _TRUST_WITHOUT_CONFIDENCE)
    else:
        levels = np.clip(confidence.reshape(-1)[indices], 0, 2).astype(np.int64)
        trust = np.asarray(_CONFIDENCE_TRUST, dtype=np.float64)[levels]
    # A frame the capture-time QC already distrusted starts lower. weight is
    # 1.0 when no QC block was recorded, so this is a no-op for older bundles.
    trust = np.clip(trust * float(max(0.0, min(1.0, frame.qc.weight))), 0.0, 1.0)

    photo = _load_rgb_for_depth_grid(bundle, frame, width, height)
    if photo is None:
        colors = np.full((indices.shape[0], 3), 0.5)
    else:
        colors = photo.reshape(-1, 3)[indices]

    return (
        _FrameSamples(
            positions=positions,
            normals=normals[indices],
            view_directions=directions,
            ranges=ranges,
            colors=colors,
            trust=trust,
            has_photo=photo is not None,
        ),
        world,
        valid,
        centre,
    )


def _thin_by_voxel(
    samples: _FrameSamples, voxel_meters: float
) -> _FrameSamples:
    """Keep the single most trusted sample in each voxel.

    A mean would be wrong here: the normal, the view direction and the range
    all belong to one measurement, and averaging them across a cell produces a
    disc that no beam ever measured. Picking a representative keeps every
    attribute of one real sample together.
    """
    if voxel_meters <= 0 or samples.positions.shape[0] == 0:
        return samples
    keys = np.floor(samples.positions / float(voxel_meters)).astype(np.int64)
    # Sort by trust descending so np.unique's first occurrence per cell is the
    # most trusted sample in it.
    order = np.argsort(-samples.trust, kind="stable")
    view = np.ascontiguousarray(keys[order]).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).reshape(-1)
    _, first = np.unique(view, return_index=True)
    keep = np.sort(order[first])
    return _FrameSamples(
        positions=samples.positions[keep],
        normals=samples.normals[keep],
        view_directions=samples.view_directions[keep],
        ranges=samples.ranges[keep],
        colors=samples.colors[keep],
        trust=samples.trust[keep],
        has_photo=samples.has_photo,
    )


def _concatenate(chunks: List[_FrameSamples]) -> _FrameSamples:
    if not chunks:
        empty = np.zeros((0, 3))
        return _FrameSamples(empty, empty, empty, np.zeros(0), empty, np.zeros(0))
    return _FrameSamples(
        positions=np.concatenate([c.positions for c in chunks], axis=0),
        normals=np.concatenate([c.normals for c in chunks], axis=0),
        view_directions=np.concatenate([c.view_directions for c in chunks], axis=0),
        ranges=np.concatenate([c.ranges for c in chunks], axis=0),
        colors=np.concatenate([c.colors for c in chunks], axis=0),
        trust=np.concatenate([c.trust for c in chunks], axis=0),
    )


def _cap_samples(samples: _FrameSamples, ceiling: int) -> _FrameSamples:
    """Trim to ``ceiling`` samples, keeping the most trusted ones."""
    count = samples.positions.shape[0]
    if ceiling <= 0 or count <= ceiling:
        return samples
    keep = np.sort(np.argsort(-samples.trust, kind="stable")[:ceiling])
    return _FrameSamples(
        positions=samples.positions[keep],
        normals=samples.normals[keep],
        view_directions=samples.view_directions[keep],
        ranges=samples.ranges[keep],
        colors=samples.colors[keep],
        trust=samples.trust[keep],
        has_photo=samples.has_photo,
    )


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def run_pipeline(
    scan_dir: Path,
    result_dir: Path,
    options: Optional[PipelineOptions] = None,
    report: ReportCallback = _noop_report,
    is_cancelled: CancelCallback = _never_cancelled,
) -> PipelineResult:
    """Build one scan and write the result files. Raises on failure.

    ``scan_dir`` is the uploaded capture bundle (``Incoming/<scanID>`` or,
    after the move, ``Completed/<scanID>``). ``result_dir`` is where the files
    the phone will download are written.
    """
    options = options or PipelineOptions()
    scan_dir = Path(scan_dir)
    result_dir = Path(result_dir)

    def check_cancel() -> None:
        if is_cancelled():
            raise PipelineCancelled()

    report(PHASE_READING, "Opening the scan.", None)
    try:
        bundle = capture_data.load_bundle(scan_dir)
    except capture_data.CaptureFormatError as error:
        raise PipelineError(str(error)) from error
    except (OSError, ValueError, KeyError) as error:
        raise PipelineError(
            "This scan could not be opened. It may not have finished sending."
        ) from error

    check_cancel()
    cloud, initialisation, notes = _initialise(bundle, options, report, check_cancel)
    if len(cloud) == 0:
        raise PipelineError(
            "There was nothing measurable in this scan, so there is nothing to "
            "build. Walking the space more slowly usually fixes it."
        )

    check_cancel()
    optimiser = _load_optimiser()
    iterations_completed = 0
    optimised = False
    if optimiser is not None:
        report(PHASE_BUILDING, "Refining the scan against the photos.", None)
        try:
            cloud, iterations_completed = optimiser.refine(
                cloud, bundle, options, report, is_cancelled
            )
            optimised = True
        except PipelineCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - report, keep the geometry
            log.exception("The optimisation step failed.")
            notes.append(
                "The photo refinement step stopped early, so what came back is "
                "the measured shape without it."
            )
            log.info("refine() raised: %s", error)

    check_cancel()
    result = _export(bundle, cloud, result_dir, options, report)
    result.optimised = optimised
    result.initialisation = initialisation
    result.iterations_completed = iterations_completed
    result.summary = _summary(result, notes, optimised)
    return result


def _initialise(
    bundle: "capture_data.CaptureBundle",
    options: PipelineOptions,
    report: ReportCallback,
    check_cancel: Callable[[], None],
) -> Tuple[SplatCloud, str, List[str]]:
    """Discs from LiDAR where possible, the baked point cloud where not."""
    notes: List[str] = []
    settings = bundle.settings
    depth_frames = bundle.frames_with_depth()

    if not settings.has_native_depth or not depth_frames:
        cloud = _initialise_from_points(bundle, options)
        notes.append(
            "This scan arrived without its depth measurements, so it was built "
            "from the saved point cloud instead. That is a rougher starting "
            "point than a full LiDAR scan."
        )
        return cloud, "points3D", notes

    depth_width = int(settings.depthWidth)
    depth_height = int(settings.depthHeight)
    K_rgb = np.asarray(bundle.intrinsics.matrix(), dtype=np.float64)
    K_depth = geometry.depth_map_intrinsics(
        K_rgb,
        (bundle.intrinsics.width, bundle.intrinsics.height),
        (depth_width, depth_height),
    )
    pixel_angular_size = 2.0 * math.atan(0.5 / max(1e-6, float(K_depth[0, 0])))

    stride = max(1, len(depth_frames) // max(1, int(options.max_sample_frames)))
    accumulator = carving.OccupancyAccumulator()
    carving_usable = True
    chunks: List[_FrameSamples] = []
    depth_missing = 0
    photo_missing = 0

    for position, frame in enumerate(depth_frames):
        check_cancel()
        report(
            PHASE_READING,
            "Reading the measurements from your scan.",
            (position + 1) / float(len(depth_frames)),
        )
        depth_path = bundle.depth_path(frame)
        depth = (
            sidecars.read_depth16(depth_path, depth_width, depth_height)
            if depth_path is not None
            else None
        )
        if depth is None:
            depth_missing += 1
            continue

        samples, world, valid, centre = _frame_samples(bundle, frame, depth, K_depth)
        if carving_usable:
            try:
                accumulator.add_frame(
                    origin=centre,
                    endpoints=world,
                    valid=valid,
                    max_range_meters=float(settings.lidarMaxRangeMeters),
                    no_return_clip_meters=carving.no_return_clip_from_neighbours(depth),
                )
            except (ValueError, MemoryError) as error:
                # A pose that flew off, or a scene too large for the Morton
                # key range. Free-space carving is an optimisation, not the
                # scan: give it up rather than lose the whole build.
                log.warning("Free-space carving stopped early: %s", error)
                carving_usable = False
        if samples is None:
            continue
        if position % stride != 0:
            continue
        if not samples.has_photo:
            photo_missing += 1
        chunks.append(_thin_by_voxel(samples, options.sample_voxel_meters))

    if photo_missing:
        notes.append(
            "The photos for {n} frame(s) could not be opened, so those parts "
            "came out grey rather than in colour.".format(n=photo_missing)
        )
    if depth_missing:
        notes.append(
            "{n} of the depth files could not be read, so those frames were "
            "left out.".format(n=depth_missing)
        )

    samples = _concatenate(chunks)
    del chunks
    samples = _thin_by_voxel(samples, options.sample_voxel_meters)
    ceiling = min(int(options.max_splats), INITIALISER_SAMPLE_CEILING)
    before = samples.positions.shape[0]
    samples = _cap_samples(samples, ceiling)
    if samples.positions.shape[0] < before:
        notes.append(
            "This scan measured more surface than the {cap:,} point limit "
            "allows, so the least certain measurements were left out.".format(
                cap=ceiling
            )
        )

    if samples.positions.shape[0] == 0:
        cloud = _initialise_from_points(bundle, options)
        notes.append(
            "None of the depth frames held usable measurements, so the saved "
            "point cloud was used instead."
        )
        return cloud, "points3D", notes

    check_cancel()
    report(PHASE_BUILDING, "Turning the measurements into splats.", None)
    cloud, _flags = init_lidar.initial_cloud_from_samples(
        positions=samples.positions,
        normals=samples.normals,
        view_directions=samples.view_directions,
        ranges=samples.ranges,
        colors=samples.colors,
        trust=samples.trust,
        pixel_angular_size=pixel_angular_size,
        sh_degree=int(options.sh_degree),
    )

    if carving_usable:
        check_cancel()
        report(PHASE_BUILDING, "Clearing out the empty space.", None)
        try:
            grid = accumulator.build()
        except (ValueError, MemoryError) as error:
            log.warning("Could not finish the free-space carve: %s", error)
            grid = None
        if grid is not None and len(grid):
            deletable = carving.prune_mask(grid, cloud.positions)
            removed = int(deletable.sum())
            if removed and removed < len(cloud):
                cloud = cloud.select(~deletable)
                log.info(
                    "Carving deleted %s splats sitting in certified empty space.",
                    removed,
                )
    return cloud, "lidar", notes


def _initialise_from_points(
    bundle: "capture_data.CaptureBundle", options: PipelineOptions
) -> SplatCloud:
    """The fallback: isotropic blobs from ``sparse/0/points3D.txt``."""
    points_path = bundle.path(bundle.pointCloudPath)
    if points_path is None or not points_path.is_file():
        raise PipelineError(
            "This scan has neither depth measurements nor a saved point cloud, "
            "so there is nothing to build from. Try scanning it again."
        )
    cloud_points = colmap_data.read_points3D(points_path)
    if cloud_points.is_empty:
        raise PipelineError(
            "The saved point cloud in this scan is empty, so there is nothing "
            "to build from. Try scanning it again."
        )
    if len(cloud_points) > options.max_splats:
        keep = np.linspace(
            0, len(cloud_points) - 1, int(options.max_splats)
        ).astype(np.int64)
        xyz = cloud_points.xyz[keep]
        rgb = cloud_points.rgb[keep]
    else:
        xyz = cloud_points.xyz
        rgb = cloud_points.rgb
    return init_lidar.fallback_cloud_from_points(
        xyz, rgb, sh_degree=int(options.sh_degree)
    )


def _preview_camera(
    bundle: "capture_data.CaptureBundle",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """A camera the user actually stood at (F9), or ``None`` for the fallback.

    The middle of the walk, not the first frame: a capture almost always opens
    with the phone pointed at the floor while the user finds the start.
    """
    if not bundle.frames:
        return None, None
    frame = bundle.frames[len(bundle.frames) // 2]
    return frame.pose.view_matrix, np.asarray(
        bundle.intrinsics.matrix(), dtype=np.float64
    )


def _export(
    bundle: "capture_data.CaptureBundle",
    cloud: SplatCloud,
    result_dir: Path,
    options: PipelineOptions,
    report: ReportCallback,
) -> PipelineResult:
    """Write the four result files. A failed optional file is not fatal."""
    result_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    report(PHASE_EXPORTING, "Saving the finished scan.", 0.1)
    ply_path = ply.write_ply(
        cloud,
        result_dir / "model.ply",
        comment="Built by the Booster from scan " + bundle.scanID,
    )
    files.append(ply_path)

    if options.write_spz:
        report(PHASE_EXPORTING, "Saving a smaller copy for your phone.", 0.4)
        try:
            files.append(spz.write_spz(cloud, result_dir / "model.spz"))
        except Exception as error:  # noqa: BLE001 - the .ply is the one that matters
            log.warning("Could not write model.spz: %s", error)

    report(PHASE_EXPORTING, "Rendering a preview to send back.", 0.6)
    preview_path: Optional[Path] = None
    try:
        view, K = _preview_camera(bundle)
        preview_path = preview.render_preview_png(
            cloud,
            result_dir / "preview.png",
            view=view,
            K=K,
            long_edge=int(options.preview_long_edge),
        )
        files.append(preview_path)
    except Exception as error:  # noqa: BLE001 - no preview is not a failed build
        log.warning("Could not render the preview: %s", error)

    report(PHASE_EXPORTING, "Writing the scan details.", 0.85)
    minimum, maximum = cloud.bounds()
    budget = model_json.TrainingBudgetJSON(
        splatCap=int(options.max_splats),
        iterations=int(options.iterations),
        renderLongEdgePixels=int(options.render_long_edge),
        shDegree=int(options.sh_degree),
        keyframeCount=len(bundle.frames),
        memoryCeilingBytes=0,
        useHalfPrecision=False,
        useSparseAdam=False,
    )
    files.append(
        model_json.write_model_json(
            result_dir / "model.json",
            model_id=str(uuid.uuid4()),
            scan_id=bundle.scanID,
            ply_path="model.ply",
            spz_path="model.spz" if (result_dir / "model.spz").is_file() else None,
            splat_count=len(cloud),
            sh_degree=int(cloud.sh_degree),
            bounds_min=minimum,
            bounds_max=maximum,
            iterations_completed=0,
            budget=budget,
        )
    )
    report(PHASE_EXPORTING, "Saved.", 1.0)
    return PipelineResult(
        result_dir=result_dir, files=files, splat_count=len(cloud)
    )


def _summary(result: PipelineResult, notes: List[str], optimised: bool) -> str:
    """One plain paragraph the phone and the GUI both show verbatim."""
    lead = "Your scan is ready: {n:,} splats.".format(n=result.splat_count)
    if optimised:
        lead += " It was refined against your photos."
    else:
        lead += (
            " It was built straight from the depth measurements. The step that "
            "sharpens it against your photos is not built yet, so fine detail "
            "will look softer than the finished product will."
        )
    if notes:
        return lead + " " + " ".join(notes)
    return lead
