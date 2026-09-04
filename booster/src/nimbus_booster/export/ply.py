"""The INRIA/gsplat ``.ply`` layout, byte-compatible with ``PLYCodec.swift``.

Property order, transcribed from that file (which was itself verified against
Niantic's ``spz`` reference rather than reconstructed from memory)::

    x y z                             position, RDF convention
    nx ny nz                          normal, unused, written 0
    f_dc_0 f_dc_1 f_dc_2              raw degree-0 SH
    f_rest_0 .. f_rest_(3*shDim-1)    higher SH, CHANNEL-MAJOR:
                                        all R coefficients, then all G, then all B
    opacity                           pre-sigmoid logit
    scale_0 scale_1 scale_2           log scale
    rot_0 rot_1 rot_2 rot_3           quaternion, property order (w, x, y, z)

Two conversions happen here and only here:

* **RUB -> RDF.** A 180-degree rotation about X: negate ``y`` and ``z`` of the
  position, and negate the ``y`` and ``z`` components of the quaternion
  (``rot_2`` and ``rot_3``). ``x`` and ``w`` are untouched. Spherical harmonics
  are **not** touched: RUB and RDF differ by a reflection pair within one
  coordinate family, and per-coefficient SH rotation is only needed for a true
  basis rotation.
* **Channel-major SH.** In memory the rest coefficients are ``[N, k, 3]``
  (coefficient-major). The PLY writes all of channel R's coefficients, then all
  of G's, then all of B's. Getting this backwards produces a file that loads
  everywhere and has wrong view-dependent colour everywhere.

``f_rest`` channel order is the single most commonly mis-implemented part of
this format, which is why the transpose below is one explicit line rather than
a clever reshape.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .splat_cloud import REST_COEFFICIENT_COUNT, ExportError, SplatCloud


def build_header(cloud: SplatCloud, comment: str) -> str:
    rest = cloud.rest_count
    lines: List[str] = [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex {n}".format(n=len(cloud)),
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    lines += ["property float f_rest_{i}".format(i=i) for i in range(rest * 3)]
    lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
    ]
    if comment:
        lines.append("comment " + comment.replace("\n", " "))
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_ply(cloud: SplatCloud, path: Path, comment: str = "") -> Path:
    """Write ``cloud`` as a binary little-endian splat PLY. Returns ``path``.

    Written to a temporary file and renamed, so a phone that starts
    downloading while an export is still running never sees a half-written PLY
    whose header promises more vertices than the body contains.
    """
    if len(cloud) == 0:
        raise ExportError("There are no splats to export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rest = cloud.rest_count
    n = len(cloud)

    columns: List[np.ndarray] = []

    # RUB -> RDF: negate y and z.
    positions = cloud.positions.astype(np.float32)
    columns.append(positions[:, 0])
    columns.append(-positions[:, 1])
    columns.append(-positions[:, 2])

    zeros = np.zeros(n, dtype=np.float32)
    columns += [zeros, zeros, zeros]  # nx ny nz, unused

    columns += [cloud.color_dc[:, i].astype(np.float32) for i in range(3)]

    if rest > 0:
        assert cloud.sh_rest is not None
        # (N, k, 3) coefficient-major -> channel-major: R's k coefficients,
        # then G's, then B's.
        channel_major = np.transpose(cloud.sh_rest, (0, 2, 1))  # (N, 3, k)
        flat = np.ascontiguousarray(channel_major, np.float32).reshape(n, rest * 3)
        columns += [flat[:, i] for i in range(rest * 3)]

    columns.append(cloud.opacity_logits.astype(np.float32))
    columns += [cloud.log_scales[:, i].astype(np.float32) for i in range(3)]

    q = cloud.normalized_rotations()  # (N, 4) as (x, y, z, w)
    # rot_0 = w, rot_1 = x, rot_2 = -y, rot_3 = -z.
    columns.append(q[:, 3].astype(np.float32))
    columns.append(q[:, 0].astype(np.float32))
    columns.append(-q[:, 1].astype(np.float32))
    columns.append(-q[:, 2].astype(np.float32))

    body = np.stack(columns, axis=1).astype("<f4", copy=False)

    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as handle:
        handle.write(build_header(cloud, comment).encode("ascii"))
        handle.write(np.ascontiguousarray(body).tobytes())
    temporary.replace(path)
    return path


def read_ply(path: Path) -> SplatCloud:
    """Read a splat PLY back, undoing the RDF flip and the channel-major SH.

    Only ``binary_little_endian`` with all-``float`` vertex properties is
    accepted - the layout in this module's docstring, which covers every
    Gaussian-splat PLY any compatible tool produces. A big-endian file, or a
    generic coloured point cloud with ``uchar`` channels, is refused by name
    rather than silently misread.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        header_bytes = bytearray()
        while True:
            line = handle.readline()
            if not line:
                raise ExportError("The file could not be read: PLY header never ended.")
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode("ascii", errors="replace")
        body = handle.read()

    fmt = None
    count = 0
    properties: List[str] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] != "float":
                raise ExportError(
                    "Unsupported file format: this PLY has a '{t}' vertex "
                    "property ('{n}'). Only all-float Gaussian-splat PLYs are "
                    "read.".format(t=parts[1], n=parts[-1])
                )
            properties.append(parts[2])

    if fmt == "binary_big_endian":
        raise ExportError(
            "Unsupported file format: this PLY is big-endian. Re-export it as "
            "binary little-endian."
        )
    if fmt != "binary_little_endian":
        raise ExportError(
            "Unsupported file format: only binary little-endian PLY is read, "
            "not '{f}'.".format(f=fmt)
        )
    if count <= 0:
        raise ExportError("There are no splats in that file.")

    stride = len(properties)
    expected = count * stride * 4
    if len(body) < expected:
        raise ExportError(
            "The file could not be read: it promises {n} splats but holds "
            "{have} bytes of the {want} needed.".format(
                n=count, have=len(body), want=expected
            )
        )
    table = np.frombuffer(body[:expected], dtype="<f4").reshape(count, stride)
    index = {name: i for i, name in enumerate(properties)}

    def column(name: str) -> np.ndarray:
        if name not in index:
            raise ExportError(
                "The file could not be read: it has no '{n}' property, so it "
                "is not a Gaussian-splat PLY.".format(n=name)
            )
        return table[:, index[name]]

    rest_names = sorted(
        (p for p in properties if p.startswith("f_rest_")),
        key=lambda p: int(p.split("_")[-1]),
    )
    rest_total = len(rest_names)
    if rest_total % 3 != 0:
        raise ExportError(
            "The file could not be read: {n} f_rest properties is not a "
            "multiple of three.".format(n=rest_total)
        )
    rest = rest_total // 3
    degree = next(
        (d for d, k in REST_COEFFICIENT_COUNT.items() if k == rest),
        None,
    )
    if degree is None:
        raise ExportError(
            "Unsupported spherical-harmonics degree: {n} rest coefficients per "
            "channel does not correspond to any degree 0-3.".format(n=rest)
        )

    positions = np.stack([column("x"), -column("y"), -column("z")], axis=1)
    color_dc = np.stack([column("f_dc_{i}".format(i=i)) for i in range(3)], axis=1)
    opacity = column("opacity")
    log_scales = np.stack([column("scale_{i}".format(i=i)) for i in range(3)], axis=1)
    rotations = np.stack(
        [column("rot_1"), -column("rot_2"), -column("rot_3"), column("rot_0")],
        axis=1,
    )

    sh_rest = None
    if rest > 0:
        flat = np.stack([table[:, index[n]] for n in rest_names], axis=1)
        # Channel-major on disk -> coefficient-major in memory.
        sh_rest = np.transpose(flat.reshape(count, 3, rest), (0, 2, 1))

    return SplatCloud(
        sh_degree=degree,
        positions=positions,
        rotations=rotations,
        log_scales=log_scales,
        opacity_logits=opacity,
        color_dc=color_dc,
        sh_rest=sh_rest,
    )
