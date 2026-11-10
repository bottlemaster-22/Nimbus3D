"""SE(3), quaternions and projection, in numpy. No torch, no scipy required.

The conventions here are ``docs/DATA_FORMAT.md`` section 2 and they are not
negotiable anywhere in this project:

* **World frame**: right-handed, gravity-aligned, ``+Y`` up, metres.
* **Camera frame**: ``+X`` right, ``+Y`` **down**, ``+Z`` **forward**. That is
  COLMAP/OpenCV, not ARKit.
* **A pose is world -> camera**: ``X_cam = R * X_world + t``. The camera centre
  is ``C = -R^T t``.
* **Quaternions are ``(x, y, z, w)`` in memory and in JSON.** Only file writers
  reorder.

If you find yourself writing a stray minus sign to make something line up, the
bug is somewhere else.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# Quaternions, (x, y, z, w)
# --------------------------------------------------------------------------


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Unit-length, with the identity as the fallback for a degenerate input.

    Normalising a zero quaternion produces NaN, and one NaN pose poisons an
    entire pose graph silently. Falling back to the identity is wrong by a
    knowable amount; NaN is wrong by an unknowable one.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    bad = (~np.isfinite(norm)) | (norm < _EPS)
    out = np.where(bad, np.array([0.0, 0.0, 0.0, 1.0]), q / np.where(bad, 1.0, norm))
    return out[0] if single else out


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation. Batched over a leading axis if present."""
    q = quaternion_normalize(q)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix[0] if single else matrix


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x, y, z, w).

    Uses Shepperd's branch on the largest diagonal term rather than the naive
    ``w = sqrt(1 + trace) / 2``, which loses most of its precision - and can
    take the square root of a small negative number - whenever the rotation is
    near 180 degrees. A loop-closure constraint across a doorway is routinely
    near 180 degrees.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quaternion_normalize(np.array([x, y, z, w]))


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, ``(x, y, z, w)`` order, meaning "apply b then a"."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quaternion_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation, used to re-interpolate poses in the F1 sweep.

    The sign flip is not cosmetic: ``q`` and ``-q`` are the same rotation, and
    without it a 5 ms time-offset step can interpolate the long way round and
    produce a pose that is 350 degrees from where it should be.
    """
    a = quaternion_normalize(a)
    b = quaternion_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return quaternion_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b) / sin_theta


# --------------------------------------------------------------------------
# so(3) / se(3) exponential and logarithm
# --------------------------------------------------------------------------


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues: an axis-angle 3-vector -> a 3x3 rotation.

    The small-angle branch is a Taylor series rather than a division by
    ``sin(theta)/theta``; a pose-graph step routinely proposes a correction of
    a few microradians and the naive form divides by zero on the last iteration
    of a converged solve.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    skew = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    if theta < 1e-8:
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * skew + b * (skew @ skew)


def so3_log(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation -> axis-angle 3-vector, the inverse of :func:`so3_exp`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_theta = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    if theta > np.pi - 1e-5:
        # Near 180 degrees the antisymmetric part vanishes; recover the axis
        # from the symmetric part instead. A doorway loop closure hits this.
        symmetric = 0.5 * (matrix + matrix.T) - np.eye(3) * cos_theta
        axis = np.sqrt(np.clip(np.diag(symmetric) / (1.0 - cos_theta), 0.0, None))
        # Signs are ambiguous in the symmetric part; fix them from whichever
        # off-diagonal is largest.
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
            axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
        norm = float(np.linalg.norm(axis))
        if norm < _EPS:
            return np.zeros(3)
        return axis / norm * theta
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """(6,) twist ``[translation(3), rotation(3)]`` -> a 4x4 transform.

    The left Jacobian ``V`` is applied to the translation part, which is what
    makes this a true SE(3) exponential rather than "rotate, then add". The
    difference matters: without ``V``, a Gauss-Newton step that rotates a
    submap also translates it by an amount nobody asked for, and the solve
    oscillates instead of converging.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    rotation = so3_exp(omega)
    skew = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    if theta < 1e-8:
        v = np.eye(3) + 0.5 * skew + (1.0 / 6.0) * (skew @ skew)
    else:
        v = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / (theta * theta)) * skew
            + ((theta - np.sin(theta)) / (theta ** 3)) * (skew @ skew)
        )
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = v @ rho
    return out


def se3_log(matrix: np.ndarray) -> np.ndarray:
    """4x4 transform -> (6,) twist. Inverse of :func:`se3_exp`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    omega = so3_log(matrix[:3, :3])
    theta = float(np.linalg.norm(omega))
    skew = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    if theta < 1e-8:
        v_inv = np.eye(3) - 0.5 * skew + (1.0 / 12.0) * (skew @ skew)
    else:
        half = 0.5 * theta
        cot = half / np.tan(half)
        v_inv = (
            np.eye(3)
            - 0.5 * skew
            + ((1.0 - cot) / (theta * theta)) * (skew @ skew)
        )
    out = np.empty(6)
    out[:3] = v_inv @ matrix[:3, 3]
    out[3:] = omega
    return out


def se3_inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rotation = matrix[:3, :3]
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ matrix[:3, 3]
    return out


def se3_from(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


# --------------------------------------------------------------------------
# Projection and unprojection
# --------------------------------------------------------------------------


def project(
    points_world: np.ndarray, view: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """World points -> ``(uv, depth)``. Points behind the camera get ``depth<=0``.

    Returned ``uv`` for a point behind the camera is meaningless, not clamped:
    the caller must mask on ``depth > 0``, and hiding that behind a clamp is how
    a splat behind the camera ends up supervising a pixel in front of it.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = view[:3, :3]
    translation = view[:3, 3]
    camera = points_world @ rotation.T + translation
    depth = camera[:, 2]
    safe = np.where(np.abs(depth) < _EPS, _EPS, depth)
    u = K[0, 0] * (camera[:, 0] / safe) + K[0, 2]
    v = K[1, 1] * (camera[:, 1] / safe) + K[1, 2]
    return np.stack([u, v], axis=1), depth


def unproject_depth_map(
    depth: np.ndarray, K: np.ndarray, view: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """A native depth map -> world points, plus the boolean mask of valid ones.

    ``depth`` is metres with ``0`` for a no-return. The returned mask is what
    "valid" means everywhere downstream: a positive, finite return. Everything
    else is ``unknown``, which is emphatically not the same as empty.

    Pixel centres are at ``(col + 0.5, row + 0.5)``. Getting that half-pixel
    wrong biases every unprojected point by half a depth-pixel, which at the
    ~8x upsample ratio is four RGB pixels of systematic error.
    """
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    x = (cols - K[0, 2]) / K[0, 0] * depth
    y = (rows - K[1, 2]) / K[1, 1] * depth
    camera = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    rotation = view[:3, :3]
    translation = view[:3, 3]
    world = (camera - translation) @ rotation
    return world, valid.reshape(-1)


def depth_map_intrinsics(K_rgb: np.ndarray, rgb_size, depth_size) -> np.ndarray:
    """Scale RGB intrinsics down to the native depth grid.

    ARKit's depth map covers the same field of view as the RGB frame, so this
    is a pure resolution change and the half-pixel term applies exactly as it
    does in :meth:`Intrinsics.scaled`.
    """
    rgb_w, rgb_h = rgb_size
    depth_w, depth_h = depth_size
    sx = float(depth_w) / float(rgb_w)
    sy = float(depth_h) / float(rgb_h)
    out = np.eye(3, dtype=np.float64)
    out[0, 0] = K_rgb[0, 0] * sx
    out[1, 1] = K_rgb[1, 1] * sy
    out[0, 2] = (K_rgb[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (K_rgb[1, 2] + 0.5) * sy - 0.5
    return out


def estimate_normals(
    points: np.ndarray, valid: np.ndarray, shape: Tuple[int, int], radius: int = 2
) -> np.ndarray:
    """Per-pixel surface normals from the depth map's own grid neighbourhood.

    Cross product of the local horizontal and vertical tangents, computed on the
    **native** grid: it is the only place the samples are genuinely adjacent.
    Doing this on an upsampled map measures the interpolator's smoothness
    instead of the surface's.

    Normals are oriented towards the camera - which is at the origin in camera
    space, so the test is simply "does it point back along the ray". A disc
    initialised with a back-facing normal is invisible from the only viewpoint
    that ever saw it.
    """
    height, width = shape
    grid = points.reshape(height, width, 3)
    mask = valid.reshape(height, width)

    step = max(1, int(radius))
    right = np.roll(grid, -step, axis=1)
    left = np.roll(grid, step, axis=1)
    down = np.roll(grid, -step, axis=0)
    up = np.roll(grid, step, axis=0)
    mask_h = mask & np.roll(mask, -step, axis=1) & np.roll(mask, step, axis=1)
    mask_v = mask & np.roll(mask, -step, axis=0) & np.roll(mask, step, axis=0)

    tangent_u = right - left
    tangent_v = down - up
    normals = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.where(norm < _EPS, 1.0, norm))
    normals[~(mask_h & mask_v)] = 0.0
    return normals.reshape(-1, 3)
