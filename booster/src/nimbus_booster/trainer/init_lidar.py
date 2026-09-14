"""Initialisation: normal-aligned discs from the LiDAR cloud, not random points.

A generic 3DGS run starts from an isotropic blob at every SfM point and spends
its first few thousand iterations discovering that the world is made of
surfaces. We already measured the surfaces. So the initial Gaussian at a LiDAR
sample is a **disc**: two large axes in the local tangent plane, one small axis
along the normal, oriented so the thin direction points along the surface
normal.

The scales are not arbitrary either. The in-plane radius is the physical
footprint of the sample - the sample spacing at that range, which is the actual
area that one beam measured - so neighbouring discs tile the surface rather than
overlapping into a fog or leaving gaps that later densification has to find.

Structural trust (F6) is applied here, at birth, because it is far cheaper to
start a doubtful sample in the right shape than to teach the optimiser to
stretch it later:

* **trusted** samples get a thin, opaque, position-pinned disc;
* **doubtful** samples get a disc elongated **along the viewing ray** and made
  translucent, so the optimiser can slide it in depth - the axis its
  uncertainty actually lies along - without fighting a shape that says
  otherwise.

Those two states are recorded in ``prepass/init_splats.flags``, bit 0 and bit 1.
Bit 2 marks a splat on a detected 3D edge curve, which is exempt from the
disc/effective-rank prior in F4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..export.splat_cloud import SH_DC_TO_COLOR, SplatCloud, inverse_sigmoid
from .geometry import matrix_to_quaternion

#: Bit 0 of ``init_splats.flags``: trusted sample, thin, opaque, held in place.
FLAG_PINNED = 1 << 0
#: Bit 1: doubtful, elongated along the viewing ray, translucent, free to slide.
FLAG_RAY_ELONGATED = 1 << 1
#: Bit 2: lies on a detected 3D edge curve, exempt from the disc prior.
FLAG_EDGE_CURVE = 1 << 2


@dataclass
class InitParameters:
    """Every number here is physical, not a magic constant.

    ``footprint_multiplier`` is the one to understand: a native depth sample at
    range ``r`` covers roughly ``r * pixel_angular_size`` of surface, and a
    Gaussian whose one-sigma radius equals *half* that footprint tiles the
    surface with neighbours meeting at one sigma. Larger and the surface fogs;
    smaller and it is a field of dots with gaps the densifier has to rediscover.
    """

    footprint_multiplier: float = 0.5
    #: Thickness along the normal, as a fraction of the in-plane radius. A
    #: surface is a surface; 1/8 is thin enough to look like one and thick
    #: enough not to vanish under the 3D low-pass filter.
    thickness_ratio: float = 0.125
    #: Floor and ceiling on the in-plane radius, metres. The floor stops a
    #: 20 cm-range sample from producing a Gaussian smaller than the renderer
    #: can resolve; the ceiling stops a 5 m grazing sample from producing a
    #: dinner plate.
    min_radius_meters: float = 0.004
    max_radius_meters: float = 0.10
    #: Starting opacity for a trusted sample and for a doubtful one.
    trusted_opacity: float = 0.85
    doubtful_opacity: float = 0.25
    #: How much longer a doubtful splat is along the ray than across it.
    ray_elongation: float = 3.0
    #: Trust below this is "doubtful". Trust is the recalibrated confidence
    #: crossed with the F6 noise field, not ARKit's raw three levels.
    trust_threshold: float = 0.5
    #: A sample whose incidence angle exceeds this is grazing, and a grazing
    #: LiDAR return is long. Its disc is widened rather than believed.
    grazing_cos_threshold: float = 0.35


def _tangent_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane orthogonal to each normal.

    The seed axis is chosen per-normal to be the world axis **least** aligned
    with it. Using a fixed seed makes the cross product degenerate for any
    surface parallel to it - which for a fixed ``+Y`` seed is every floor and
    ceiling in the scan.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    absolute = np.abs(normals)
    seed_index = np.argmin(absolute, axis=1)
    seed = np.zeros_like(normals)
    seed[np.arange(normals.shape[0]), seed_index] = 1.0

    u = np.cross(normals, seed)
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.divide(u, np.where(norm < 1e-9, 1.0, norm))
    v = np.cross(normals, u)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, np.where(norm_v < 1e-9, 1.0, norm_v))
    return u, v


def _quaternions_from_frames(
    axis_x: np.ndarray, axis_y: np.ndarray, axis_z: np.ndarray
) -> np.ndarray:
    """Stack three orthonormal axes into per-splat rotation quaternions.

    Columns, not rows: the covariance is ``R diag(s^2) R^T``, so column ``k`` of
    ``R`` must be the direction that scale ``s_k`` stretches along.
    """
    n = axis_x.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        matrix = np.stack([axis_x[i], axis_y[i], axis_z[i]], axis=1)
        # Re-orthonormalise: a cross product of two nearly parallel measured
        # vectors is not exactly orthogonal, and matrix_to_quaternion on a
        # non-rotation matrix produces a quaternion that is quietly wrong.
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, 2] *= -1.0
            rotation = u @ vt
        out[i] = matrix_to_quaternion(rotation)
    return out


def initial_cloud_from_samples(
    positions: np.ndarray,
    normals: np.ndarray,
    view_directions: np.ndarray,
    ranges: np.ndarray,
    colors: np.ndarray,
    trust: np.ndarray,
    pixel_angular_size: float,
    sh_degree: int = 1,
    parameters: Optional[InitParameters] = None,
) -> Tuple[SplatCloud, np.ndarray]:
    """Build the initial Gaussian set. Returns ``(cloud, flags)``.

    ``view_directions`` are unit vectors from the camera **towards** each
    sample, in world space; they define the ray a doubtful sample is allowed to
    slide along. ``pixel_angular_size`` is the angular size of one native depth
    pixel in radians (``2 * atan(0.5 / fx_native)`` is close enough; the caller
    computes it from the real intrinsics rather than assuming 256x192).

    ``trust`` is 0..1 per sample. It comes from F6 - the recalibrated
    confidence crossed with the per-sample noise field - and never from ARKit's
    raw three levels, which are a ranking and not a probability.
    """
    parameters = parameters or InitParameters()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = positions.shape[0]
    if n == 0:
        return SplatCloud.empty(sh_degree), np.zeros(0, dtype=np.uint8)

    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    view_directions = np.asarray(view_directions, dtype=np.float64).reshape(-1, 3)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    trust = np.clip(np.asarray(trust, dtype=np.float64).reshape(-1), 0.0, 1.0)

    # A normal that failed to estimate (all-zero) falls back to facing the
    # camera. Facing the camera is the least-wrong guess for a sample whose
    # neighbourhood was too broken to fit a plane through.
    normal_length = np.linalg.norm(normals, axis=1)
    degenerate = normal_length < 1e-6
    normals = np.where(
        degenerate[:, None], -view_directions, normals / np.where(degenerate, 1.0, normal_length)[:, None]
    )
    # Orient towards the camera: a disc whose thin axis points away is
    # invisible from the only viewpoint that ever measured it.
    facing = np.sum(normals * view_directions, axis=1) > 0.0
    normals = np.where(facing[:, None], -normals, normals)

    # -- in-plane radius: the physical footprint of one sample -------------
    footprint = np.abs(ranges) * float(pixel_angular_size)
    # Grazing incidence spreads one beam over more surface: divide by the
    # cosine of the incidence angle, floored so a near-tangent sample does not
    # produce an infinite disc.
    cos_incidence = np.abs(np.sum(normals * view_directions, axis=1))
    cos_incidence = np.maximum(cos_incidence, parameters.grazing_cos_threshold)
    radius = np.clip(
        footprint * parameters.footprint_multiplier / cos_incidence,
        parameters.min_radius_meters,
        parameters.max_radius_meters,
    )

    doubtful = trust < parameters.trust_threshold

    # -- axes ---------------------------------------------------------------
    tangent_u, tangent_v = _tangent_basis(normals)
    # Axis order is (u, v, normal), so scale index 2 is the thin one.
    quaternions = _quaternions_from_frames(tangent_u, tangent_v, normals)

    scales = np.empty((n, 3), dtype=np.float64)
    scales[:, 0] = radius
    scales[:, 1] = radius
    scales[:, 2] = radius * parameters.thickness_ratio

    # A doubtful sample's uncertainty lies along the ray, and for a sample seen
    # nearly head-on the ray IS the normal - so lengthening the third axis is
    # exactly "elongated along the viewing ray". Doing it in the disc's own
    # frame keeps the rotation valid without a second SVD.
    scales[doubtful, 2] *= parameters.ray_elongation / parameters.thickness_ratio
    scales[doubtful, 2] = np.minimum(
        scales[doubtful, 2], parameters.max_radius_meters * parameters.ray_elongation
    )

    opacity = np.where(
        doubtful, parameters.doubtful_opacity, parameters.trusted_opacity
    )

    # -- colour -------------------------------------------------------------
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if colors.max(initial=0.0) > 1.5:
        colors = colors / 255.0
    color_dc = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_DC_TO_COLOR

    cloud = SplatCloud(
        sh_degree=sh_degree,
        positions=positions.astype(np.float32),
        rotations=quaternions.astype(np.float32),
        log_scales=np.log(np.maximum(scales, 1e-8)).astype(np.float32),
        opacity_logits=inverse_sigmoid(opacity).astype(np.float32),
        color_dc=color_dc.astype(np.float32),
        sh_rest=None,
    )

    flags = np.zeros(n, dtype=np.uint8)
    flags[~doubtful] |= FLAG_PINNED
    flags[doubtful] |= FLAG_RAY_ELONGATED
    return cloud, flags


def write_flags(path, flags: np.ndarray) -> None:
    """``prepass/init_splats.flags``: one byte per splat, in PLY vertex order."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(flags, dtype=np.uint8).tofile(str(path))


def read_flags(path, expected_count: int) -> Optional[np.ndarray]:
    """Read the flags back, or ``None`` if absent or the wrong length.

    A wrong length is refused rather than padded: flags are positional against
    the PLY's vertex order, and a silently truncated array pins the wrong
    splats, which looks exactly like a trainer that will not converge.
    """
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size != expected_count:
        return None
    return data


def fallback_cloud_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int = 1,
    radius_meters: float = 0.02,
) -> SplatCloud:
    """Isotropic blobs from a bare point cloud, when no normals are available.

    Only reached when the scan carries ``points3D.txt`` but no per-frame depth
    sidecars - an imported COLMAP model, or a bundle whose depth files did not
    survive the transfer. Labelled here rather than buried: this is the generic
    3DGS initialisation, and a scan started this way is measurably worse than
    one started from discs. The pipeline says so in the job message.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = xyz.shape[0]
    if n == 0:
        return SplatCloud.empty(sh_degree)
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    return SplatCloud(
        sh_degree=sh_degree,
        positions=xyz.astype(np.float32),
        rotations=np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n, 1)),
        log_scales=np.full((n, 3), np.log(radius_meters), np.float32),
        opacity_logits=np.full(n, float(inverse_sigmoid(np.array(0.5))), np.float32),
        color_dc=((np.clip(rgb, 0, 1) - 0.5) / SH_DC_TO_COLOR).astype(np.float32),
        sh_rest=None,
    )
