"""F1: poses first. Submaps, revisits, the pose graph, and the time offset.

This is the number-one quality blocker and the reason it is worth this much
code. Raw ARKit VIO drifts roughly 0.55 degrees and 19 mm over a long walk;
at 2 m that is about 13 px of reprojection error, and Gaussian splatting needs
sub-pixel. No amount of training fixes a pose error - it just grows fur.

What this module does, in order:

1. **Time-sliced submaps.** 15-30 s windows with 20-30% overlap. VIO is
   near-perfect *locally*, so each window is treated as internally rigid and
   only its single SE(3) placement is optimised. That reduces a 4000-pose
   problem to a 20-pose one, which is why it converges in seconds instead of
   being a research project.
2. **Revisit detection.** Pose proximity, plus view-direction similarity, plus
   point-to-plane ICP on the native depth. All three, because proximity alone
   pairs a frame with the one shot through the wall behind it.
3. **Submap pose graph.** One rigid SE(3) per submap, robust loss (Huber, or
   annealed), solved by Gauss-Newton with Levenberg damping.
4. **Camera-to-IMU time offset.** Sweep -50..+50 ms in 5 ms steps,
   re-interpolate poses with SLERP, keep the offset that minimises the
   revisit residual.

What this module deliberately does **not** do:

* It never runs COLMAP from scratch. Reconstructing poses from nothing throws
  away a good prior.
* It never triangulates and then bundle-adjusts from the ARKit prior. That
  measurably *degraded* the prior in 15 of 15 test rooms (arXiv 2608.21008).
  The stronger optional refinement is a LiDAR-depth-anchored, triangulation-
  free BA, which lives in :func:`refine_depth_anchored` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.capture import CaptureFrame, Pose
from .geometry import (
    matrix_to_quaternion,
    quaternion_slerp,
    se3_exp,
    se3_from,
    se3_inverse,
    se3_log,
)

#: A submap window. Short enough that VIO drift inside one is negligible, long
#: enough that there are few enough of them to solve quickly.
SUBMAP_SECONDS = 22.0
#: Overlap between neighbouring windows, as a fraction. The overlap frames are
#: what tie consecutive submaps together in the graph.
SUBMAP_OVERLAP = 0.25

#: The time-offset sweep from the design: -50..+50 ms in 5 ms steps.
TIME_OFFSET_RANGE_SECONDS = 0.050
TIME_OFFSET_STEP_SECONDS = 0.005


# --------------------------------------------------------------------------
# 1. Submaps
# --------------------------------------------------------------------------


@dataclass
class Submap:
    """A time slice of the walk, treated as internally rigid."""

    submap_id: int
    frame_indices: List[int]
    start_time: float
    end_time: float
    #: The optimised correction, applied on the LEFT in world space:
    #: ``T_world_cam_corrected = correction @ T_world_cam``. Identity until the
    #: graph is solved.
    correction: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __len__(self) -> int:
        return len(self.frame_indices)


def build_submaps(
    frames: Sequence[CaptureFrame],
    window_seconds: float = SUBMAP_SECONDS,
    overlap: float = SUBMAP_OVERLAP,
) -> List[Submap]:
    """Slice the walk into overlapping time windows.

    Time-sliced, not distance-sliced: drift accumulates with *time* integrated
    over IMU noise, not with distance walked, and a user who stands still
    studying one corner for forty seconds accumulates drift the whole while.

    Submaps are always on. F7's co-visibility partitioning is a different thing
    (an optional whole-house mode); these are needed for pose fixing on every
    scan, including a single object.
    """
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: (f.timestampSeconds, f.index))
    times = np.array([f.timestampSeconds for f in ordered], dtype=np.float64)
    if not np.all(np.isfinite(times)) or float(times[-1] - times[0]) <= 0.0:
        # No usable clock (an imported model, say). One submap: the pose graph
        # then has nothing to solve, which is the honest outcome.
        return [
            Submap(
                submap_id=0,
                frame_indices=[f.index for f in ordered],
                start_time=0.0,
                end_time=0.0,
            )
        ]

    stride = max(1e-3, window_seconds * (1.0 - max(0.0, min(0.9, overlap))))
    submaps: List[Submap] = []
    start = float(times[0])
    end_of_walk = float(times[-1])
    submap_id = 0
    while start <= end_of_walk:
        stop = start + window_seconds
        inside = np.nonzero((times >= start) & (times < stop))[0]
        if inside.size > 0:
            submaps.append(
                Submap(
                    submap_id=submap_id,
                    frame_indices=[ordered[i].index for i in inside],
                    start_time=float(times[inside[0]]),
                    end_time=float(times[inside[-1]]),
                )
            )
            submap_id += 1
        if stop >= end_of_walk:
            break
        start += stride

    # A trailing window with one or two frames contributes nothing but a
    # near-singular block to the graph. Fold it into its predecessor.
    if len(submaps) >= 2 and len(submaps[-1]) < 3:
        tail = submaps.pop()
        for index in tail.frame_indices:
            if index not in submaps[-1].frame_indices:
                submaps[-1].frame_indices.append(index)
        submaps[-1].end_time = tail.end_time
    return submaps


def submap_of_frame(submaps: Sequence[Submap]) -> Dict[int, int]:
    """``{frame_index: submap_id}``, resolving overlaps to the earlier submap.

    An overlap frame belongs to both windows by construction. It is *assigned*
    to the earlier one and *observed* by both, which is exactly what makes it a
    constraint between them rather than a duplicated unknown.
    """
    owner: Dict[int, int] = {}
    for submap in submaps:
        for index in submap.frame_indices:
            owner.setdefault(index, submap.submap_id)
    return owner


# --------------------------------------------------------------------------
# 2. Revisit detection
# --------------------------------------------------------------------------


@dataclass
class Revisit:
    """One measured constraint: frame ``b`` sees the same place as frame ``a``."""

    frame_a: int
    frame_b: int
    #: Measured relative transform, camera A -> camera B.
    relative: np.ndarray
    translation_residual: float
    rotation_residual_degrees: float
    inliers: int
    confidence: float
    method: str = "poseProximity"


@dataclass
class RevisitParameters:
    """Gates for a candidate pair. All three must pass."""

    #: Camera centres this close, in metres. Wider than it looks: the point is
    #: to *propose* candidates cheaply, and ICP rejects the bad ones.
    max_center_distance: float = 1.5
    #: Viewing directions within this angle. Two cameras a metre apart looking
    #: opposite ways share nothing.
    max_view_angle_degrees: float = 45.0
    #: Frames must be at least this far apart in time, or every pair is its own
    #: neighbour and the graph learns nothing.
    min_time_separation_seconds: float = 8.0
    #: ICP acceptance: this fraction of sampled points must land within
    #: ``inlier_distance`` of a plane in the other frame.
    min_inlier_fraction: float = 0.35
    inlier_distance_meters: float = 0.06


def find_revisit_candidates(
    frames: Sequence[CaptureFrame],
    parameters: Optional[RevisitParameters] = None,
    max_candidates: int = 400,
) -> List[Tuple[int, int]]:
    """Propose pairs by pose proximity **and** view-direction similarity.

    Both gates, not either: proximity alone happily pairs a frame with the one
    taken from the same spot facing the opposite wall, and a loop closure built
    on that constraint bends the whole map.

    Only candidates come out of here. Every one still has to survive ICP.
    """
    parameters = parameters or RevisitParameters()
    trustworthy = [f for f in frames if f.qc.is_pose_trustworthy]
    if len(trustworthy) < 2:
        return []

    centers = np.stack([f.pose.center for f in trustworthy])
    forwards = np.stack([f.pose.forward for f in trustworthy])
    times = np.array([f.timestampSeconds for f in trustworthy])
    cos_limit = np.cos(np.radians(parameters.max_view_angle_degrees))

    candidates: List[Tuple[float, int, int]] = []
    n = len(trustworthy)
    for i in range(n):
        # Vectorised over j > i: a Python double loop over 4000 frames is 8
        # million iterations and takes minutes.
        delta_time = times[i + 1 :] - times[i]
        far_enough = np.abs(delta_time) >= parameters.min_time_separation_seconds
        if not np.any(far_enough):
            continue
        distance = np.linalg.norm(centers[i + 1 :] - centers[i], axis=1)
        alignment = forwards[i + 1 :] @ forwards[i]
        passes = (
            far_enough
            & (distance <= parameters.max_center_distance)
            & (alignment >= cos_limit)
        )
        for offset in np.nonzero(passes)[0]:
            j = i + 1 + int(offset)
            # Score: near and well-aligned first, so the cap keeps the best.
            score = float(distance[offset]) - 0.5 * float(alignment[offset])
            candidates.append((score, trustworthy[i].index, trustworthy[j].index))

    candidates.sort(key=lambda c: c[0])
    return [(a, b) for _, a, b in candidates[:max_candidates]]


def point_to_plane_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    initial: np.ndarray,
    iterations: int = 12,
    inlier_distance: float = 0.06,
) -> Tuple[np.ndarray, float, int]:
    """Point-to-plane ICP. Returns ``(transform, rms_residual, inlier_count)``.

    Point-to-plane rather than point-to-point because these are LiDAR returns
    off flat walls: point-to-point tries to make individual samples correspond,
    which they do not (the beams landed in different places), while
    point-to-plane only asks each point to lie *on* the other surface, which is
    the thing that is actually true.

    Correspondences are found by nearest neighbour. scipy's KD-tree is used
    when available; the fallback is a chunked brute-force search, which is
    slower but has no dependency and is exact.
    """
    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    target_normals = np.asarray(target_normals, dtype=np.float64).reshape(-1, 3)
    if source_points.shape[0] < 16 or target_points.shape[0] < 16:
        return initial, float("inf"), 0

    query = _nearest_neighbour_index(target_points)
    transform = np.asarray(initial, dtype=np.float64).reshape(4, 4).copy()
    residual = float("inf")
    inliers = 0

    for _ in range(max(1, iterations)):
        moved = source_points @ transform[:3, :3].T + transform[:3, 3]
        indices, distances = query(moved)
        keep = distances < inlier_distance
        inliers = int(np.count_nonzero(keep))
        if inliers < 16:
            break

        p = moved[keep]
        q = target_points[indices[keep]]
        normal = target_normals[indices[keep]]
        normal_length = np.linalg.norm(normal, axis=1, keepdims=True)
        usable = normal_length[:, 0] > 1e-6
        if int(np.count_nonzero(usable)) < 16:
            break
        p, q = p[usable], q[usable]
        normal = normal[usable] / normal_length[usable]

        # Linearised point-to-plane: minimise sum((R p + t - q) . n)^2 over a
        # small twist. The Jacobian row is [n, p x n].
        jacobian = np.concatenate([normal, np.cross(p, normal)], axis=1)
        error = np.sum((p - q) * normal, axis=1)
        # Huber weights: one bad correspondence off a specular surface would
        # otherwise dominate a least-squares solve.
        weight = _huber_weights(error, delta=inlier_distance * 0.5)
        weighted = jacobian * weight[:, None]
        hessian = weighted.T @ jacobian
        gradient = weighted.T @ (-error)
        hessian += np.eye(6) * 1e-9
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        # The twist is [rotation(3), translation(3)] in this parameterisation,
        # which is the opposite order to se3_exp's, hence the reorder.
        transform = se3_exp(np.concatenate([delta[3:], delta[:3]])) @ transform
        residual = float(np.sqrt(np.mean(error * error)))
        if float(np.linalg.norm(delta)) < 1e-7:
            break

    return transform, residual, inliers


def _nearest_neighbour_index(points: np.ndarray):
    """Return a callable ``query(x) -> (indices, distances)``."""
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(points)

        def query_kd(x: np.ndarray):
            distances, indices = tree.query(x, k=1)
            return np.asarray(indices, dtype=np.int64), np.asarray(distances)

        return query_kd
    except Exception:  # noqa: BLE001 - scipy is optional
        pass

    def query_brute(x: np.ndarray):
        indices = np.empty(x.shape[0], dtype=np.int64)
        distances = np.empty(x.shape[0], dtype=np.float64)
        chunk = max(1, 2_000_000 // max(1, points.shape[0]))
        for begin in range(0, x.shape[0], chunk):
            end = min(begin + chunk, x.shape[0])
            difference = x[begin:end, None, :] - points[None, :, :]
            squared = np.einsum("ijk,ijk->ij", difference, difference)
            best = np.argmin(squared, axis=1)
            indices[begin:end] = best
            distances[begin:end] = np.sqrt(squared[np.arange(end - begin), best])
        return indices, distances

    return query_brute


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights for the Huber loss: 1 inside ``delta``, ``delta/|r|`` outside."""
    magnitude = np.abs(np.asarray(residuals, dtype=np.float64))
    return np.where(magnitude <= delta, 1.0, delta / np.maximum(magnitude, 1e-12))


# --------------------------------------------------------------------------
# 3. The submap pose graph
# --------------------------------------------------------------------------


@dataclass
class PoseGraphResult:
    """What the solve produced, including the parts that did not go well."""

    submaps: List[Submap]
    iterations: int
    initial_cost: float
    final_cost: float
    constraint_count: int
    #: Constraints whose robust weight ended below 0.5 - measured, then largely
    #: disbelieved. Reported so a scan with many of them can be flagged rather
    #: than quietly averaged into mush.
    downweighted_count: int = 0
    converged: bool = False

    @property
    def cost_reduction(self) -> float:
        if self.initial_cost <= 0.0:
            return 0.0
        return 1.0 - (self.final_cost / self.initial_cost)


def solve_submap_graph(
    submaps: Sequence[Submap],
    frames_by_index: Dict[int, CaptureFrame],
    revisits: Sequence[Revisit],
    owner: Dict[int, int],
    iterations: int = 25,
    huber_delta: float = 0.05,
) -> PoseGraphResult:
    """Optimise one rigid SE(3) per submap against the revisit constraints.

    Gauss-Newton with Levenberg damping, Huber-weighted, submap 0 gauge-fixed
    (the whole graph is free to slide otherwise, and a solver that discovers
    that on its own wastes its iterations discovering it).

    A constraint between two frames in the **same** submap is skipped: it
    carries no information about the unknowns, because the submap is rigid by
    construction and both frames move together.
    """
    submaps = list(submaps)
    if len(submaps) < 2 or not revisits:
        return PoseGraphResult(
            submaps=submaps,
            iterations=0,
            initial_cost=0.0,
            final_cost=0.0,
            constraint_count=0,
            converged=True,
        )

    index_of = {s.submap_id: i for i, s in enumerate(submaps)}
    n = len(submaps)
    corrections = [np.eye(4) for _ in range(n)]

    # Pre-bake each constraint into the form the residual needs.
    edges: List[Tuple[int, int, np.ndarray, float]] = []
    for revisit in revisits:
        frame_a = frames_by_index.get(revisit.frame_a)
        frame_b = frames_by_index.get(revisit.frame_b)
        if frame_a is None or frame_b is None:
            continue
        submap_a = owner.get(revisit.frame_a)
        submap_b = owner.get(revisit.frame_b)
        if submap_a is None or submap_b is None or submap_a == submap_b:
            continue
        if submap_a not in index_of or submap_b not in index_of:
            continue
        # Measured: camera A -> camera B. Predicted, given corrections:
        #   T_b_pred = (C_b T_wb) (C_a T_wa)^-1
        # so the residual is log(measured^-1 @ predicted).
        edges.append(
            (
                index_of[submap_a],
                index_of[submap_b],
                np.asarray(revisit.relative, dtype=np.float64),
                float(max(0.0, min(1.0, revisit.confidence))),
            )
        )
    if not edges:
        return PoseGraphResult(
            submaps=submaps,
            iterations=0,
            initial_cost=0.0,
            final_cost=0.0,
            constraint_count=0,
            converged=True,
        )

    view_of = {
        index: frames_by_index[index].pose.view_matrix
        for index in frames_by_index
    }
    frame_pairs = [
        (revisit.frame_a, revisit.frame_b)
        for revisit in revisits
        if revisit.frame_a in view_of and revisit.frame_b in view_of
    ][: len(edges)]

    def residuals(current: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((len(edges), 6))
        weights = np.ones(len(edges))
        for k, (ia, ib, measured, confidence) in enumerate(edges):
            frame_a, frame_b = frame_pairs[k]
            view_a = current[ia] @ view_of[frame_a]
            view_b = current[ib] @ view_of[frame_b]
            predicted = view_b @ se3_inverse(view_a)
            out[k] = se3_log(se3_inverse(measured) @ predicted)
            weights[k] = confidence
        return out, weights

    def cost_of(residual: np.ndarray, weights: np.ndarray) -> float:
        magnitude = np.linalg.norm(residual, axis=1)
        robust = _huber_weights(magnitude, huber_delta)
        return float(np.sum(weights * robust * magnitude * magnitude))

    residual, confidence_weights = residuals(corrections)
    initial_cost = cost_of(residual, confidence_weights)
    cost = initial_cost
    damping = 1e-4
    converged = False
    used = 0

    for used in range(1, max(1, iterations) + 1):
        hessian = np.zeros((6 * n, 6 * n))
        gradient = np.zeros(6 * n)
        magnitude = np.linalg.norm(residual, axis=1)
        robust = _huber_weights(magnitude, huber_delta) * confidence_weights

        for k, (ia, ib, _measured, _c) in enumerate(edges):
            # Numerical Jacobian in the 6-dof twist of each endpoint. The
            # analytic form exists, but the graph has at most a few dozen
            # unknowns and a few hundred edges, so a forward difference costs
            # milliseconds and cannot be silently wrong about an adjoint.
            weight = robust[k]
            for side, block in ((0, ia), (1, ib)):
                for axis in range(6):
                    twist = np.zeros(6)
                    twist[axis] = 1e-6
                    perturbed = list(corrections)
                    perturbed[block] = se3_exp(twist) @ perturbed[block]
                    frame_a, frame_b = frame_pairs[k]
                    view_a = perturbed[ia] @ view_of[frame_a]
                    view_b = perturbed[ib] @ view_of[frame_b]
                    predicted = view_b @ se3_inverse(view_a)
                    perturbed_residual = se3_log(
                        se3_inverse(edges[k][2]) @ predicted
                    )
                    column = (perturbed_residual - residual[k]) / 1e-6
                    rows = slice(6 * block, 6 * block + 6)
                    hessian[rows, rows] += weight * np.outer(column, column)
                    gradient[rows] -= weight * column * residual[k]
                    del side

        # Gauge fix: submap 0 does not move. Without this the normal equations
        # are singular along six directions and the damping term silently
        # decides where the map ends up.
        hessian[0:6, :] = 0.0
        hessian[:, 0:6] = 0.0
        hessian[0:6, 0:6] = np.eye(6)
        gradient[0:6] = 0.0

        try:
            step = np.linalg.solve(hessian + damping * np.eye(6 * n), gradient)
        except np.linalg.LinAlgError:
            break

        trial = [
            se3_exp(step[6 * i : 6 * i + 6]) @ corrections[i] for i in range(n)
        ]
        trial_residual, trial_weights = residuals(trial)
        trial_cost = cost_of(trial_residual, trial_weights)

        if trial_cost < cost:
            corrections = trial
            residual = trial_residual
            confidence_weights = trial_weights
            improvement = cost - trial_cost
            cost = trial_cost
            damping = max(1e-9, damping * 0.5)
            if improvement < 1e-10:
                converged = True
                break
        else:
            damping *= 4.0
            if damping > 1e6:
                converged = True
                break

    magnitude = np.linalg.norm(residual, axis=1)
    downweighted = int(np.count_nonzero(_huber_weights(magnitude, huber_delta) < 0.5))
    for i, submap in enumerate(submaps):
        submap.correction = corrections[i]

    return PoseGraphResult(
        submaps=submaps,
        iterations=used,
        initial_cost=initial_cost,
        final_cost=cost,
        constraint_count=len(edges),
        downweighted_count=downweighted,
        converged=converged,
    )


def apply_corrections(
    frames: Sequence[CaptureFrame],
    submaps: Sequence[Submap],
    owner: Dict[int, int],
) -> Dict[int, Pose]:
    """Turn solved submap corrections into one refined pose per frame.

    The correction is applied on the left, in world space::

        T_world_cam_refined = correction @ T_world_cam_raw

    A frame in the overlap between two submaps gets the correction of its
    owner, not a blend. Blending two rigid corrections is not a rigid
    correction, and the seam it creates lands exactly where two submaps meet -
    which is where the pose error was in the first place.
    """
    out: Dict[int, Pose] = {}
    correction_of = {s.submap_id: s.correction for s in submaps}
    for frame in frames:
        submap_id = owner.get(frame.index)
        correction = correction_of.get(submap_id) if submap_id is not None else None
        view = frame.pose.view_matrix
        if correction is not None:
            view = np.asarray(correction, dtype=np.float64) @ view
        out[frame.index] = Pose(
            rotation=matrix_to_quaternion(view[:3, :3]),
            translation=view[:3, 3].copy(),
        )
    return out


# --------------------------------------------------------------------------
# 4. Camera-to-IMU time offset
# --------------------------------------------------------------------------


def interpolate_pose_at(
    frames: Sequence[CaptureFrame], times: np.ndarray, at: float
) -> Optional[np.ndarray]:
    """Pose at an arbitrary time, SLERP on rotation and linear on translation.

    Interpolating in world -> camera translation directly would move the camera
    along a chord through the scene rather than along its path; the
    interpolation is done on the **camera centre**, which is the thing that
    physically moved, and the translation is rebuilt from it.
    """
    if len(frames) < 2:
        return None
    position = int(np.searchsorted(times, at))
    if position <= 0 or position >= len(frames):
        return None
    before, after = frames[position - 1], frames[position]
    span = float(times[position] - times[position - 1])
    if span <= 1e-9:
        return before.pose.view_matrix
    t = float((at - times[position - 1]) / span)

    rotation = quaternion_slerp(before.pose.rotation, after.pose.rotation, t)
    centre = (1.0 - t) * before.pose.center + t * after.pose.center
    from .geometry import quaternion_to_matrix  # noqa: PLC0415 - avoids a cycle

    R = quaternion_to_matrix(rotation)
    return se3_from(R, -R @ centre)


@dataclass
class TimeOffsetResult:
    """The sweep's outcome. ``offset_seconds`` is ``None`` when it did not converge.

    Reported, not hidden. ``CaptureBundle.cameraToIMUTimeOffsetSeconds`` is
    optional precisely so "the sweep found no clear minimum" can be said out
    loud instead of being rounded to zero and pretended about.
    """

    offset_seconds: Optional[float]
    costs: np.ndarray
    offsets: np.ndarray
    #: How much better the best offset is than zero, as a fraction. Below a few
    #: percent the minimum is not distinguishable from noise.
    improvement: float = 0.0


def calibrate_time_offset(
    frames: Sequence[CaptureFrame],
    revisits: Sequence[Revisit],
    range_seconds: float = TIME_OFFSET_RANGE_SECONDS,
    step_seconds: float = TIME_OFFSET_STEP_SECONDS,
    min_improvement: float = 0.03,
) -> TimeOffsetResult:
    """Sweep the offset and keep the one that minimises the revisit residual.

    A camera timestamp and an IMU timestamp that disagree by 20 ms put every
    pose 20 ms along the walk from where the pixels were taken. At a brisk
    walking pace that is about 2 cm of translation and, at 2 m, several pixels
    of reprojection error - which is the same order as the drift this whole
    module exists to remove, and it is a *constant* bias rather than a random
    one, so it does not average out.

    The metric is the same relative-pose residual the pose graph minimises,
    which is why this runs **after** revisits are detected and before the graph
    is solved.
    """
    offsets = np.arange(
        -range_seconds, range_seconds + 0.5 * step_seconds, step_seconds
    )
    if len(frames) < 3 or not revisits:
        return TimeOffsetResult(offset_seconds=None, costs=np.zeros(0), offsets=offsets)

    ordered = sorted(frames, key=lambda f: f.timestampSeconds)
    times = np.array([f.timestampSeconds for f in ordered], dtype=np.float64)
    by_index = {f.index: f for f in ordered}

    costs = np.full(offsets.shape[0], np.nan)
    for k, offset in enumerate(offsets):
        total = 0.0
        counted = 0
        for revisit in revisits:
            frame_a = by_index.get(revisit.frame_a)
            frame_b = by_index.get(revisit.frame_b)
            if frame_a is None or frame_b is None:
                continue
            view_a = interpolate_pose_at(
                ordered, times, frame_a.timestampSeconds + float(offset)
            )
            view_b = interpolate_pose_at(
                ordered, times, frame_b.timestampSeconds + float(offset)
            )
            if view_a is None or view_b is None:
                continue
            predicted = view_b @ se3_inverse(view_a)
            residual = se3_log(se3_inverse(revisit.relative) @ predicted)
            magnitude = float(np.linalg.norm(residual))
            # Arctan loss: bounded, so one bad ICP result cannot decide the
            # offset, and smooth, so the sweep has a real minimum rather than
            # a staircase.
            total += float(np.arctan(magnitude * 20.0)) * max(0.05, revisit.confidence)
            counted += 1
        costs[k] = total / counted if counted else np.nan

    if not np.any(np.isfinite(costs)):
        return TimeOffsetResult(offset_seconds=None, costs=costs, offsets=offsets)

    best = int(np.nanargmin(costs))
    zero = int(np.argmin(np.abs(offsets)))
    baseline = costs[zero]
    improvement = 0.0
    if np.isfinite(baseline) and baseline > 0:
        improvement = float((baseline - costs[best]) / baseline)

    if improvement < min_improvement:
        # No clear minimum. Say so; do not report zero as if it were measured.
        return TimeOffsetResult(
            offset_seconds=None, costs=costs, offsets=offsets, improvement=improvement
        )

    # Parabolic refinement around the best sample: the true minimum is almost
    # never exactly on a 5 ms grid point.
    offset = float(offsets[best])
    if 0 < best < len(offsets) - 1:
        left, middle, right = costs[best - 1], costs[best], costs[best + 1]
        denominator = left - 2.0 * middle + right
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            shift = 0.5 * (left - right) / denominator
            if abs(shift) <= 1.0:
                offset += float(shift) * step_seconds
    return TimeOffsetResult(
        offset_seconds=offset, costs=costs, offsets=offsets, improvement=improvement
    )


# --------------------------------------------------------------------------
# 5. The optional stronger refinement
# --------------------------------------------------------------------------


def refine_depth_anchored(
    frames: Sequence[CaptureFrame],
    poses: Dict[int, Pose],
    observations: Sequence[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    iterations: int = 20,
    anneal: Sequence[float] = (1e4, 1e3, 1e2),
) -> Dict[int, Pose]:
    """LiDAR-depth-anchored, triangulation-free bundle adjustment.

    The stronger optional refinement from F1, and the reason it is *not* a
    normal BA: there is no triangulation and there are no 3D landmark
    unknowns. Each observation carries its own **ray depth**, taken from the
    LiDAR and held by a confidence-weighted prior, so a feature's position is a
    one-dimensional unknown along a known ray rather than a free point in
    space. Symmetric cross-projection residuals - project A's point into B and
    B's into A - keep the two cameras honest about each other.

    The arctan loss is annealed 1e4 -> 1e3 -> 1e2 across the iterations, which
    starts nearly-quadratic (every observation is heard) and ends strongly
    redescending (only the consistent ones still speak).

    ``observations`` is ``(frame_a, frame_b, points_a_camera, uv_b, weights)``.

    STATUS: this is the honest state of this function. The residual, the
    anneal schedule and the weighting are implemented; the solve below is a
    damped Gauss-Newton over the **relative** pose of each pair, and the
    per-pair results are averaged into per-frame updates rather than solved as
    one joint system. That is a real refinement and it measurably reduces
    cross-projection error, but it is not the full joint solve the design
    describes.
    TODO(nimbus): replace the per-pair averaging with a single sparse joint
    system over all frames (Schur-complement on the per-observation depths,
    which are the only per-observation unknowns), so that a long chain of pairs
    cannot disagree with itself.
    """
    if not observations:
        return dict(poses)

    updated = {index: pose for index, pose in poses.items()}
    schedule = list(anneal) or [1e3]

    for stage, scale in enumerate(schedule):
        per_frame_delta: Dict[int, List[np.ndarray]] = {}
        stage_iterations = max(1, iterations // len(schedule))

        for frame_a, frame_b, points_a, uv_b, weights in observations:
            pose_a = updated.get(frame_a)
            pose_b = updated.get(frame_b)
            if pose_a is None or pose_b is None:
                continue
            relative = pose_b.view_matrix @ se3_inverse(pose_a.view_matrix)
            for _ in range(stage_iterations):
                relative, moved = _cross_projection_step(
                    relative, points_a, uv_b, weights, scale
                )
                if not moved:
                    break
            corrected = relative @ pose_a.view_matrix
            delta = corrected @ se3_inverse(pose_b.view_matrix)
            per_frame_delta.setdefault(frame_b, []).append(se3_log(delta))

        for index, twists in per_frame_delta.items():
            # Average in the tangent space, then exponentiate once. Averaging
            # matrices directly would not produce a rotation.
            mean = np.mean(np.stack(twists, axis=0), axis=0)
            pose = updated[index]
            view = se3_exp(mean * 0.5) @ pose.view_matrix
            updated[index] = Pose(
                rotation=matrix_to_quaternion(view[:3, :3]),
                translation=view[:3, 3].copy(),
            )
        del stage
    return updated


def _cross_projection_step(
    relative: np.ndarray,
    points_a: np.ndarray,
    uv_b: np.ndarray,
    weights: np.ndarray,
    arctan_scale: float,
) -> Tuple[np.ndarray, bool]:
    """One damped Gauss-Newton step on a pair's relative pose.

    ``points_a`` are 3D points in camera A's frame, already at their LiDAR
    depth. ``uv_b`` is where they were observed in B, in **normalised** camera
    coordinates so the step is independent of focal length.
    """
    points_a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if points_a.shape[0] < 8:
        return relative, False

    moved = points_a @ relative[:3, :3].T + relative[:3, 3]
    z = moved[:, 2]
    valid = z > 1e-3
    if int(np.count_nonzero(valid)) < 8:
        return relative, False
    moved = moved[valid]
    target = uv_b[valid]
    weight = weights[valid]
    z = moved[:, 2]

    predicted = moved[:, :2] / z[:, None]
    error = (predicted - target).reshape(-1)

    # d(u,v)/d(twist) for a point in the camera frame, twist = [rho, omega].
    n = moved.shape[0]
    jacobian = np.zeros((2 * n, 6))
    inv_z = 1.0 / z
    x, y = moved[:, 0], moved[:, 1]
    jacobian[0::2, 0] = inv_z
    jacobian[0::2, 2] = -x * inv_z * inv_z
    jacobian[0::2, 3] = -x * y * inv_z * inv_z
    jacobian[0::2, 4] = 1.0 + x * x * inv_z * inv_z
    jacobian[0::2, 5] = -y * inv_z
    jacobian[1::2, 1] = inv_z
    jacobian[1::2, 2] = -y * inv_z * inv_z
    jacobian[1::2, 3] = -(1.0 + y * y * inv_z * inv_z)
    jacobian[1::2, 4] = x * y * inv_z * inv_z
    jacobian[1::2, 5] = x * inv_z

    # Arctan loss weight: d/dr [atan(s r)] / r, redescending as s grows.
    magnitude = np.abs(error)
    robust = 1.0 / (1.0 + (arctan_scale * magnitude) ** 2)
    row_weight = np.repeat(weight, 2) * robust

    weighted = jacobian * row_weight[:, None]
    hessian = weighted.T @ jacobian + np.eye(6) * 1e-8
    gradient = -weighted.T @ error
    try:
        delta = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return relative, False
    if not np.all(np.isfinite(delta)):
        return relative, False
    return se3_exp(delta) @ relative, bool(np.linalg.norm(delta) > 1e-9)
