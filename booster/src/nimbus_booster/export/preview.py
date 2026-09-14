"""A preview render of the trained splats, and a dependency-free PNG writer.

The preview exists so the phone can show the user what came back before it
downloads a hundred megabytes of ``.ply``. It is a genuine EWA-style splat
rasterisation - the 3D covariance is projected through the perspective
Jacobian to a real 2D covariance and each splat is evaluated over its own
3-sigma footprint, back to front, with "over" compositing - not a point
scatter dressed up as a render.

It is deliberately **not** the trainer's rasteriser. gsplat's CUDA kernels do
that, on the GPU, differentiably. This is CPU numpy, runs in a couple of
seconds on a whole-house cloud, and needs neither torch nor a GPU, so a
preview still comes back from a machine where training failed - which is
exactly when the user most wants to see something.

PNG is written here rather than with Pillow because Pillow is an optional
dependency and a preview that silently does not exist is worse than a hundred
lines of zlib.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..trainer.geometry import quaternion_to_matrix
from .splat_cloud import SplatCloud, sigmoid

#: Evaluate each splat out to this many standard deviations. Beyond 3 sigma a
#: Gaussian contributes under 1.2% of its peak, which is below the 1/255
#: quantisation of the output for any splat that is not nearly opaque.
_SIGMA_CUTOFF = 3.0

#: Anything wider than this in pixels is a splat that has been dragged across
#: the frame by a near-degenerate projection. Clamping the footprint bounds the
#: cost of one bad splat instead of letting it paint the whole preview.
_MAX_RADIUS_PIXELS = 256


def write_png(path: Path, rgb: np.ndarray) -> Path:
    """Write an ``(H, W, 3)`` uint8 array as an 8-bit RGB PNG.

    Plain PNG: one IHDR, one zlib-compressed IDAT with filter byte 0 on every
    scanline, one IEND. No interlacing, no palette, no ancillary chunks.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("write_png wants an (H, W, 3) uint8 array")
    height, width, _ = rgb.shape

    # Filter type 0 (None) prepended to each scanline.
    raw = np.concatenate(
        [np.zeros((height, 1), np.uint8), rgb.reshape(height, width * 3)], axis=1
    )
    compressed = zlib.compress(raw.tobytes(), 6)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def _covariance_3d(cloud: SplatCloud) -> np.ndarray:
    """(N, 3, 3) world-space covariance ``R S S^T R^T`` from scale and rotation."""
    rotation = quaternion_to_matrix(cloud.normalized_rotations())  # (N, 3, 3)
    scale = np.exp(np.clip(cloud.log_scales.astype(np.float64), -20.0, 5.0))
    scaled = rotation * scale[:, None, :]  # R @ diag(s)
    return scaled @ np.transpose(scaled, (0, 2, 1))


def render_preview(
    cloud: SplatCloud,
    view: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    background: Tuple[float, float, float] = (0.06, 0.07, 0.09),
    max_splats: int = 400_000,
) -> np.ndarray:
    """Rasterise ``cloud`` from one camera. Returns ``(H, W, 3)`` uint8.

    ``view`` is 4x4 world -> camera, ``K`` is 3x3, both in the conventions of
    ``docs/DATA_FORMAT.md`` section 2 (camera ``+Z`` forward, ``+Y`` down).

    When the cloud is larger than ``max_splats`` the preview keeps the splats
    with the largest ``opacity * projected area``, which is the honest ranking
    for "what would a viewer actually see": dropping by index would thin the
    scene unevenly, and dropping by opacity alone would delete every large,
    faint background splat that holds the picture together.
    """
    background_rgb = np.asarray(background, dtype=np.float64).reshape(3)
    canvas = np.repeat(
        background_rgb.reshape(1, 1, 3), width, axis=1
    ).repeat(height, axis=0)
    if len(cloud) == 0:
        return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    view = np.asarray(view, dtype=np.float64).reshape(4, 4)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rotation = view[:3, :3]
    translation = view[:3, 3]

    camera_xyz = cloud.positions.astype(np.float64) @ rotation.T + translation
    depth = camera_xyz[:, 2]
    in_front = depth > 1e-3
    if not np.any(in_front):
        return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    safe_depth = np.where(in_front, depth, 1.0)
    u = fx * camera_xyz[:, 0] / safe_depth + cx
    v = fy * camera_xyz[:, 1] / safe_depth + cy

    # Perspective Jacobian of (x, y, z) -> (u, v), evaluated at each centre.
    # This is what makes the footprint an ellipse that grows correctly with
    # distance and shears correctly off-axis, rather than a circle.
    inv_z = 1.0 / safe_depth
    inv_z2 = inv_z * inv_z
    jacobian = np.zeros((len(cloud), 2, 3), dtype=np.float64)
    jacobian[:, 0, 0] = fx * inv_z
    jacobian[:, 0, 2] = -fx * camera_xyz[:, 0] * inv_z2
    jacobian[:, 1, 1] = fy * inv_z
    jacobian[:, 1, 2] = -fy * camera_xyz[:, 1] * inv_z2

    cov3d = _covariance_3d(cloud)
    cov_cam = rotation @ cov3d @ rotation.T
    cov2d = jacobian @ cov_cam @ np.transpose(jacobian, (0, 2, 1))
    # The low-pass filter every 3DGS rasteriser applies so a splat smaller than
    # a pixel does not alias into nothing. 0.3 px^2 matches the usual choice.
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    determinant = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    valid = in_front & np.isfinite(determinant) & (determinant > 1e-9)

    radius = np.zeros(len(cloud))
    trace_half = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    discriminant = np.maximum(trace_half * trace_half - determinant, 0.0)
    largest_eigenvalue = trace_half + np.sqrt(discriminant)
    radius[valid] = _SIGMA_CUTOFF * np.sqrt(np.maximum(largest_eigenvalue[valid], 1e-9))
    radius = np.minimum(radius, _MAX_RADIUS_PIXELS)

    valid &= (
        (u + radius >= 0) & (u - radius < width) & (v + radius >= 0) & (v - radius < height)
    )
    indices = np.nonzero(valid)[0]
    if indices.size == 0:
        return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    alpha = sigmoid(cloud.opacity_logits.astype(np.float64))
    if indices.size > max_splats:
        importance = alpha[indices] * (radius[indices] ** 2)
        keep = np.argpartition(-importance, max_splats - 1)[:max_splats]
        indices = indices[keep]

    # Back to front, so a plain "over" is exact: each splat simply replaces
    # `alpha` of whatever is already behind it. No transmittance bookkeeping,
    # and no approximation from splats that overlap.
    order = indices[np.argsort(-depth[indices], kind="stable")]

    colors = cloud.display_colors().astype(np.float64)
    inv_det = 1.0 / determinant
    # Inverse of a 2x2, written out: this is evaluated per splat and a general
    # inverse would be an order of magnitude slower for no benefit.
    a = cov2d[:, 1, 1] * inv_det
    b = -cov2d[:, 0, 1] * inv_det
    c = cov2d[:, 0, 0] * inv_det

    for i in order:
        r = radius[i]
        x0 = max(0, int(np.floor(u[i] - r)))
        x1 = min(width, int(np.ceil(u[i] + r)) + 1)
        y0 = max(0, int(np.floor(v[i] - r)))
        y1 = min(height, int(np.ceil(v[i] + r)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = (np.arange(x0, x1, dtype=np.float64) + 0.5) - u[i]
        dy = (np.arange(y0, y1, dtype=np.float64) + 0.5) - v[i]
        gx, gy = np.meshgrid(dx, dy)
        power = -0.5 * (a[i] * gx * gx + 2.0 * b[i] * gx * gy + c[i] * gy * gy)
        weight = np.exp(np.maximum(power, -12.0))
        weight[power < -12.0] = 0.0
        contribution = (alpha[i] * weight)[:, :, None]

        window = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = window * (1.0 - contribution) + contribution * colors[i]

    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def preview_camera_from_cloud(
    cloud: SplatCloud, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """A fallback three-quarter view, for when there is no capture pose to use.

    Only reached when the bundle has no usable frames - an imported model, or a
    scan whose poses failed to load. The real preview always uses a camera
    **on the walked path** (F9), because a viewpoint the user never stood at
    shows them geometry nobody ever measured.
    """
    minimum, maximum = cloud.bounds()
    centre = (np.asarray(minimum, np.float64) + np.asarray(maximum, np.float64)) * 0.5
    extent = float(np.max(np.asarray(maximum) - np.asarray(minimum)))
    extent = max(extent, 0.5)

    focal = 0.9 * max(width, height)
    K = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]]
    )

    direction = np.array([0.6, -0.35, 0.72])
    direction /= np.linalg.norm(direction)
    eye = centre - direction * extent * 1.6

    forward = centre - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    # Camera +Y is DOWN in this project's convention, hence the sign.
    down = np.cross(forward, right)

    R = np.stack([right, down, forward], axis=0)
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = -R @ eye
    return view, K


def render_preview_png(
    cloud: SplatCloud,
    path: Path,
    view: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    long_edge: int = 1024,
) -> Path:
    """Render one preview and write it as a PNG. Returns ``path``."""
    if view is None or K is None:
        width = int(long_edge)
        height = int(long_edge * 3 // 4)
        view, K = preview_camera_from_cloud(cloud, width, height)
    else:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        width = int(round(K[0, 2] * 2.0))
        height = int(round(K[1, 2] * 2.0))
        longest = max(width, height)
        if longest > long_edge and longest > 0:
            factor = float(long_edge) / float(longest)
            scaled = np.eye(3)
            scaled[0, 0] = K[0, 0] * factor
            scaled[1, 1] = K[1, 1] * factor
            scaled[0, 2] = (K[0, 2] + 0.5) * factor - 0.5
            scaled[1, 2] = (K[1, 2] + 0.5) * factor - 0.5
            K = scaled
            width = max(1, int(round(width * factor)))
            height = max(1, int(round(height * factor)))
    image = render_preview(cloud, view, K, max(1, width), max(1, height))
    return write_png(path, image)
