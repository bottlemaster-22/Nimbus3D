# =============================================================================
#  selftest.py -- headless, real-Blender self-test for this add-on.
#
#  Not shipped as part of the installable add-on zip (it is not imported by
#  __init__.py and is not on bpy's add-on path at runtime); it is a
#  developer/CI tool. Run it directly against a real Blender install:
#
#      "<blender>" --background --factory-startup --python selftest.py
#
#  It registers the REAL add-on package (this directory, imported as
#  `blender_addon`), builds synthetic .ply and .spz files by hand from each
#  format's own documented byte layout (not copies of any file the add-on
#  ships), imports them through the real `bpy.ops.import_scene.*` operator,
#  and asserts on the actual evaluated Blender state -- not just "no
#  exception was thrown". Exits with code 0 iff every assertion passed;
#  non-zero and a traceback otherwise, so this is CI-usable as-is.
#
#  Every assertion prints PASS/FAIL as it runs so a partial failure is easy
#  to localise instead of stopping at the first AssertionError with no
#  context.
# =============================================================================

import gzip
import math
import os
import struct
import sys
import tempfile
import traceback

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ADDON_DIR)  # .../Nimbus3D
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy  # noqa: E402  (must come after sys.path is set up; only real inside Blender)

_PASS = []
_FAIL = []


def check(name, condition, detail=""):
    if condition:
        _PASS.append(name)
        print(f"PASS: {name}")
    else:
        _FAIL.append((name, detail))
        print(f"FAIL: {name}  {detail}")


# -----------------------------------------------------------------------
# Synthetic .ply builder -- de-facto INRIA/gsplat vertex layout, degree-1 SH.
# -----------------------------------------------------------------------

def write_synthetic_ply(path, n=6):
    prop_names = (
        ["x", "y", "z", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
         "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(9)]  # degree-1: 3 rests/channel * 3
    )
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for name in prop_names:
        header += f"property float {name}\n"
    header += "end_header\n"

    rows = []
    for i in range(n):
        x, y, z = float(i), float(i) * 0.5, -float(i)
        scale = (math.log(0.05 + 0.01 * i), math.log(0.05), math.log(0.002))  # thin -> disc-like
        rot = (1.0, 0.0, 0.0, 0.0)  # identity quaternion (w,x,y,z)
        opacity_logit = 2.0  # sigmoid(2.0) ~= 0.88
        f_dc = (0.3 + 0.05 * i, 0.1, -0.1)
        f_rest = [0.01 * j for j in range(9)]
        row = [x, y, z, *scale, *rot, opacity_logit, *f_dc, *f_rest]
        rows.append(struct.pack("<%df" % len(row), *row))

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for row in rows:
            f.write(row)


def write_corrupt_ply(path):
    with open(path, "wb") as f:
        f.write(b"not a ply file at all\nsome garbage\n")


# -----------------------------------------------------------------------
# Synthetic .spz builder -- matches spz_reader.py's documented layout
# exactly (this is a from-scratch encoder, independent of the decoder
# under test, built directly from SPZCodec.swift's own field definitions
# -- not copied from spz_reader.py -- so it is a real check of agreement
# between two independent implementations of the spec, not a tautology).
#
# Body section order is positions, alphas, colors, scales, rotations, sh --
# matching SPZCodec.swift's write() exactly. (An earlier draft of both this
# builder and spz_reader.py's decode_spz used positions/scales/rotations/
# alphas/colors/sh instead: internally consistent with each other, so the
# self-test passed, but wrong against the real format -- see spz_reader.py's
# module docstring "SECTION ORDER" note. Both were fixed together.)
#
# Supports versions 1 (float16 positions, "first three" rotation), 2
# (24-bit fixed positions, "first three" rotation) and 3 (24-bit fixed
# positions, "smallest three" rotation) -- version 3 is what our own
# exporter actually writes by default (SPZCodec.swift `writeVersion = 3`),
# so that is the version every position/rotation/colour correctness
# assertion below is checked against.
# -----------------------------------------------------------------------

_SPZ_SQRT1_2 = 0.70710678
_SPZ_COLOR_SCALE = 0.15


def _spz_pack_first_three(x, y, z):
    bx = int(round((x + 1.0) / 2.0 * 255.0))
    by = int(round((y + 1.0) / 2.0 * 255.0))
    bz = int(round((z + 1.0) / 2.0 * 255.0))
    return bytes((max(0, min(255, bx)), max(0, min(255, by)), max(0, min(255, bz))))


def _spz_pack_smallest_three(x, y, z, w):
    """Independent from-scratch encoder for the "smallest three" quaternion
    compression, built directly from SPZCodec.swift's `packSmallestThree`
    (largest-magnitude component dropped, its index stored in the top 2
    bits, the other three as sign + 9-bit magnitude, packed low-to-high in
    processing order 0..3 skipping the largest)."""
    q = [x, y, z, w]
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    q = [c / n for c in q]
    i_largest = max(range(4), key=lambda i: abs(q[i]))
    negate = q[i_largest] < 0
    comp = i_largest
    cmask = (1 << 9) - 1
    for i in range(4):
        if i == i_largest:
            continue
        neg_bit = 1 if ((q[i] < 0) != negate) else 0
        mag = int(round(cmask * (abs(q[i]) / _SPZ_SQRT1_2)))
        mag = min(mag, cmask)
        comp = (comp << 10) | (neg_bit << 9) | mag
    return struct.pack("<I", comp & 0xFFFFFFFF)


def write_synthetic_spz(path, n=5, sh_degree=1, version=3, positions=None,
                         rotations=None, colors_dc=None):
    """`positions` (if given) is a list of n (x,y,z) tuples in the RUB frame
    the format actually stores; `rotations` a list of n (x,y,z,w) unit
    quaternions; `colors_dc` a list of n (r,g,b) raw SH-DC coefficients.
    Passed explicitly (rather than only derived-from-i) so the caller can
    assert the exact expected decode against a known input."""
    MAGIC = 0x5053474E
    FRACTIONAL_BITS = 12
    FLAGS = 0
    header = struct.pack("<IIIBBBB", MAGIC, version, n, sh_degree, FRACTIONAL_BITS, FLAGS, 0)

    if positions is None:
        positions = [((i - n) * 0.1, 0.2 + 0.01 * i, -0.3 + 0.02 * i) for i in range(n)]
    if rotations is None:
        rotations = [(0.0, 0.0, 0.0, 1.0)] * n  # identity
    if colors_dc is None:
        colors_dc = [(0.2, 0.6, 0.4)] * n

    body = bytearray()

    # positions
    if version == 1:
        for (x, y, z) in positions:
            body += struct.pack("<3e", x, y, z)
    else:
        scale = 1 << FRACTIONAL_BITS
        for (x, y, z) in positions:
            for v in (x, y, z):
                raw = int(round(v * scale)) & 0xFFFFFF  # two's complement wrap into 24 bits
                body += bytes((raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF))

    # alphas: count bytes
    for i in range(n):
        body.append(int(round((0.5 + 0.1 * i) * 255)) & 0xFF)

    # colors: count * 3 bytes, quantised SH-DC coefficient (SPZCodec.swift:
    # byte = dc * (colorScale * 255) + 0.5 * 255)
    for (dc_r, dc_g, dc_b) in colors_dc:
        for dc in (dc_r, dc_g, dc_b):
            b = int(round(dc * (_SPZ_COLOR_SCALE * 255) + 0.5 * 255))
            body.append(max(0, min(255, b)))

    # scales: count*3 bytes, log-scale byte = (logscale + 10) * 16
    for i in range(n * 3):
        log_scale = -3.0 + 0.1 * i
        b = int(round((log_scale + 10.0) * 16.0))
        body.append(max(0, min(255, b)))

    # rotations
    if version >= 3:
        for (x, y, z, w) in rotations:
            body += _spz_pack_smallest_three(x, y, z, w)
    else:
        for (x, y, z, w) in rotations:
            body += _spz_pack_first_three(x, y, z)  # w >= 0 implied, reconstructed on read

    # sh_rest: count * rests_per_channel*3 UNSIGNED int8, degree 1 -> 3 per channel
    rests_per_channel = {0: 0, 1: 3, 2: 8, 3: 15}[sh_degree]
    for i in range(n * rests_per_channel * 3):
        body.append((i * 7) % 256)

    with open(path, "wb") as f:
        f.write(gzip.compress(bytes(header) + bytes(body)))


def write_corrupt_spz(path):
    with open(path, "wb") as f:
        f.write(gzip.compress(b"this is not a valid spz body, too short"))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    import blender_addon as addon
    from blender_addon import brand, geonode_setup, material_setup

    # Fresh factory scene, nothing left over from a previous run in this process.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    addon.register()
    check(
        "operator registered under bpy.ops",
        hasattr(bpy.ops.import_scene, brand.IMPORT_OPERATOR_ID.split(".")[1]),
    )

    tmpdir = tempfile.mkdtemp(prefix="nimbus3d_addon_selftest_")
    ply_path = os.path.join(tmpdir, "synthetic.ply")
    spz_path = os.path.join(tmpdir, "synthetic.spz")
    corrupt_ply_path = os.path.join(tmpdir, "corrupt.ply")
    corrupt_spz_path = os.path.join(tmpdir, "corrupt.spz")

    write_synthetic_ply(ply_path, n=6)
    # Version 3, identity rotations, known positions/colours: this is the
    # version our own exporter actually writes by default (SPZCodec.swift
    # `writeVersion = 3`, "smallest three" quaternion compression), so it is
    # the one whose decode is checked for an exact expected value below.
    spz_n = 5
    spz_positions = [(0.1 * i, 0.2 + 0.01 * i, -0.3 + 0.02 * i) for i in range(spz_n)]
    spz_colors_dc = [(0.5, -0.5, 0.0)] * spz_n
    write_synthetic_spz(
        spz_path, n=spz_n, sh_degree=1, version=3,
        positions=spz_positions, colors_dc=spz_colors_dc,
    )
    write_corrupt_ply(corrupt_ply_path)
    write_corrupt_spz(corrupt_spz_path)

    op_id = brand.IMPORT_OPERATOR_ID  # e.g. "import_scene.nimbus3d_splat"
    op_callable = getattr(bpy.ops.import_scene, op_id.split(".")[1])

    # ---- PLY import ----
    before_objs = set(bpy.data.objects.keys())
    result = op_callable(filepath=ply_path)
    check("PLY import operator returned FINISHED", result == {'FINISHED'}, str(result))

    new_objs = set(bpy.data.objects.keys()) - before_objs
    check("PLY import created exactly one new object", len(new_objs) == 1, str(new_objs))
    ply_obj = bpy.data.objects[list(new_objs)[0]] if new_objs else None

    if ply_obj is not None:
        mesh = ply_obj.data
        check("PLY mesh has 6 vertices", len(mesh.vertices) == 6, str(len(mesh.vertices)))

        attr_names = set(mesh.attributes.keys())
        for expected in ("splat_scale", "splat_rotation_euler", "splat_color", "splat_opacity"):
            check(f"PLY mesh has '{expected}' attribute", expected in attr_names, str(attr_names))

        opacity_attr = mesh.attributes.get("splat_opacity")
        if opacity_attr is not None:
            vals = [d.value for d in opacity_attr.data]
            expected_op = 1.0 / (1.0 + math.exp(-2.0))
            check(
                "PLY opacity dequantised via sigmoid(logit)",
                all(abs(v - expected_op) < 1e-4 for v in vals),
                str(vals),
            )

        # Coordinate frame: docs/DATA_FORMAT.md says .ply is written in the
        # INRIA/gsplat RDF convention (+X right, +Y down, +Z forward), while
        # Blender's world is +Z-up. write_synthetic_ply's raw vertex i is
        # (i, i*0.5, -i) in that RDF frame -- the correct Blender-space
        # position is (x, z, -y) = (i, -i, -i*0.5) (see splat_data.py's
        # module docstring for the derivation). This catches the "every
        # splat lands in the wrong place" class of bug directly, not just
        # "some object got created".
        got_co = [tuple(v.co) for v in mesh.vertices]
        expected_co = [(float(i), -float(i), -float(i) * 0.5) for i in range(6)]
        check(
            "PLY vertex positions converted from RDF to Blender's +Z-up frame",
            all(
                all(abs(g - e) < 1e-4 for g, e in zip(got, exp))
                for got, exp in zip(got_co, expected_co)
            ),
            f"got {got_co[:2]}..., expected {expected_co[:2]}...",
        )

        # Same frame conversion must also rotate each splat's ORIENTATION,
        # not just its position, or every splat lands in the right place
        # tilted 90 degrees -- a much harder bug to notice by eye.
        # write_synthetic_ply uses an identity source quaternion for every
        # point, so after the RDF -> Blender pre-rotation (-90 deg about X,
        # see splat_data.py) every point's Euler should be (-pi/2, 0, 0).
        rot_attr = mesh.attributes.get("splat_rotation_euler")
        if rot_attr is not None:
            eulers = [tuple(d.vector) for d in rot_attr.data]
            expected_euler = (-math.pi / 2, 0.0, 0.0)
            check(
                "PLY identity rotation converted from RDF to Blender's +Z-up frame",
                all(
                    all(abs(g - e) < 1e-4 for g, e in zip(got, expected_euler))
                    for got in eulers
                ),
                f"got {eulers[:2]}..., expected {expected_euler}",
            )

        # Geometry Nodes modifier present and produces real instanced geometry
        # when evaluated through the depsgraph -- not just "the modifier exists".
        mod = ply_obj.modifiers.get(f"{brand.INTERNAL_PREFIX} Splat")
        check("PLY object has the splat Geometry Nodes modifier", mod is not None)
        if mod is not None:
            check("modifier uses the shared node group", mod.node_group is not None
                  and mod.node_group.name == geonode_setup.NODE_GROUP_NAME)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = ply_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        # Default "Instance Subdivisions" = 1 -> base icosahedron per instance:
        # 12 verts / 20 faces. 6 points * 12 = 72 verts, 6*20 = 120 faces.
        check(
            "evaluated (Geometry-Nodes-realized) mesh has the expected "
            "instanced ico-sphere vertex count",
            len(eval_mesh.vertices) == 6 * 12,
            f"got {len(eval_mesh.vertices)}, expected {6 * 12}",
        )
        check(
            "evaluated mesh has the expected instanced ico-sphere face count",
            len(eval_mesh.polygons) == 6 * 20,
            f"got {len(eval_mesh.polygons)}, expected {6 * 20}",
        )
        # The splat material is assigned INSIDE the Geometry Nodes graph (a
        # "Set Material" node), not on obj.data.materials -- that is how a
        # geometry-nodes-only material assignment works in Blender, so the
        # object's own material slots stay empty and the real check is the
        # material baked onto the EVALUATED mesh that came out of the
        # modifier (already fetched above for the vertex/face-count check).
        eval_mat_names = {m.name for m in eval_mesh.materials if m is not None}
        check(
            "splat material assigned on the evaluated (Geometry-Nodes) mesh",
            material_setup.MATERIAL_NAME in eval_mat_names,
            str(eval_mat_names),
        )
        eval_obj.to_mesh_clear()

        mat = bpy.data.materials.get(material_setup.MATERIAL_NAME)
        check("splat material datablock exists", mat is not None)
        if mat is not None:
            node_types = {n.bl_idname for n in mat.node_tree.nodes}
            for expected_node in (
                'ShaderNodeEmission', 'ShaderNodeBsdfTransparent',
                'ShaderNodeMixShader', 'ShaderNodeAttribute',
                'ShaderNodeOutputMaterial',
            ):
                check(f"material graph contains {expected_node}", expected_node in node_types)

    # ---- Re-import: no duplicate shared datablocks ----
    mat_count_before = len([m for m in bpy.data.materials if m.name.startswith(material_setup.MATERIAL_NAME)])
    ng_count_before = len([g for g in bpy.data.node_groups if g.name.startswith(geonode_setup.NODE_GROUP_NAME)])
    op_callable(filepath=ply_path)
    mat_count_after = len([m for m in bpy.data.materials if m.name.startswith(material_setup.MATERIAL_NAME)])
    ng_count_after = len([g for g in bpy.data.node_groups if g.name.startswith(geonode_setup.NODE_GROUP_NAME)])
    check("re-import does not duplicate the shared material datablock",
          mat_count_after == mat_count_before, f"{mat_count_before} -> {mat_count_after}")
    check("re-import does not duplicate the shared node-group datablock",
          ng_count_after == ng_count_before, f"{ng_count_before} -> {ng_count_after}")

    # ---- SPZ import (version 3: 24-bit fixed positions, "smallest three"
    # rotation compression -- what our own exporter actually writes) ----
    before_objs = set(bpy.data.objects.keys())
    result = op_callable(filepath=spz_path)
    check("SPZ import operator returned FINISHED", result == {'FINISHED'}, str(result))
    new_objs = set(bpy.data.objects.keys()) - before_objs
    check("SPZ import created exactly one new object", len(new_objs) == 1, str(new_objs))
    if new_objs:
        spz_obj = bpy.data.objects[list(new_objs)[0]]
        spz_mesh = spz_obj.data
        check("SPZ mesh has 5 vertices", len(spz_mesh.vertices) == 5,
              str(len(spz_mesh.vertices)))

        # Coordinate frame: .spz's native frame is RUB (+X right, +Y up,
        # +Z back) -- Blender position is (x, -z, y). Uses the SAME known
        # positions passed to write_synthetic_spz above.
        got_co = [tuple(v.co) for v in spz_mesh.vertices]
        expected_co = [(x, -z, y) for (x, y, z) in spz_positions]
        check(
            "SPZ vertex positions converted from RUB to Blender's +Z-up frame",
            len(got_co) == len(expected_co) and all(
                all(abs(g - e) < 5e-3 for g, e in zip(got, exp))  # 24-bit fixed-point quantisation
                for got, exp in zip(got_co, expected_co)
            ),
            f"got {got_co}, expected {expected_co}",
        )

        # Same identity-source-rotation check as the PLY case above, but for
        # the RUB -> Blender pre-rotation (+90 deg about X instead of -90),
        # and decoded through the "smallest three" v3 path instead of
        # "first three" -- a different code path than the PLY assertion
        # exercises, so this is not redundant with it.
        spz_rot_attr = spz_mesh.attributes.get("splat_rotation_euler")
        if spz_rot_attr is not None:
            spz_eulers = [tuple(d.vector) for d in spz_rot_attr.data]
            expected_spz_euler = (math.pi / 2, 0.0, 0.0)
            check(
                "SPZ identity rotation converted from RUB to Blender's +Z-up frame",
                all(
                    all(abs(g - e) < 1e-3 for g, e in zip(got, expected_spz_euler))
                    for got in spz_eulers
                ),
                f"got {spz_eulers}, expected {expected_spz_euler}",
            )

        # Colour: the stored byte is a quantised SH-DC coefficient, not a
        # finished colour (see spz_reader.py) -- decode must run it through
        # the same clamp(0.5 + SH_C0*dc) formula the PLY path uses, not
        # return the byte as-is. dc=(0.5,-0.5,0.0) -> expect roughly
        # (0.641, 0.359, 0.5), NOT (0.5,0.5,0.5) (naive byte/255) or
        # (0.5,-0.5,0.0) (raw dc unconverted).
        spz_color_attr = spz_mesh.attributes.get("splat_color")
        if spz_color_attr is not None:
            SH_C0 = 0.28209479177387814
            expected_rgb = (0.5 + SH_C0 * 0.5, 0.5 + SH_C0 * -0.5, 0.5 + SH_C0 * 0.0)
            got_rgb = tuple(spz_color_attr.data[0].color)[:3]
            check(
                "SPZ colour dequantised via colorScale + SH_C0, not left as a raw byte",
                all(abs(g - e) < 0.01 for g, e in zip(got_rgb, expected_rgb)),
                f"got {got_rgb}, expected ~{expected_rgb}",
            )

    # ---- SPZ version 1 (float16 positions, "first three" rotation) --
    # a different position codec entirely, exercised nowhere else. ----
    spz_v1_path = os.path.join(tmpdir, "synthetic_v1.spz")
    spz_v1_positions = [(1.5, -2.25, 0.5), (0.0, 0.0, 0.0), (3.0, 3.0, -3.0)]
    write_synthetic_spz(
        spz_v1_path, n=3, sh_degree=0, version=1, positions=spz_v1_positions,
    )
    before_objs = set(bpy.data.objects.keys())
    result = op_callable(filepath=spz_v1_path)
    check("SPZ v1 (float16 positions) import returned FINISHED", result == {'FINISHED'}, str(result))
    new_objs = set(bpy.data.objects.keys()) - before_objs
    if new_objs:
        v1_obj = bpy.data.objects[list(new_objs)[0]]
        got_v1_co = [tuple(v.co) for v in v1_obj.data.vertices]
        expected_v1_co = [(x, -z, y) for (x, y, z) in spz_v1_positions]
        check(
            "SPZ v1 float16 positions decoded and frame-converted correctly",
            len(got_v1_co) == len(expected_v1_co) and all(
                all(abs(g - e) < 1e-2 for g, e in zip(got, exp))  # float16 precision
                for got, exp in zip(got_v1_co, expected_v1_co)
            ),
            f"got {got_v1_co}, expected {expected_v1_co}",
        )

    # ---- Plain (non-Gaussian) point-cloud PLY: no scale_*/rot_*/opacity/
    # f_dc_* properties at all, just x,y,z + uchar red/green/blue -- the
    # "degrades gracefully" path in splat_data.from_ply, not the common
    # Gaussian-splat path already exercised above.
    plain_ply_path = os.path.join(tmpdir, "plain.ply")
    plain_n = 4
    with open(plain_ply_path, "wb") as f:
        f.write(
            ("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {plain_n}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode("ascii")
        )
        for i in range(plain_n):
            f.write(struct.pack("<fff", float(i), 0.0, 0.0))
            f.write(struct.pack("<BBB", 255, 128, 0))

    before_objs = set(bpy.data.objects.keys())
    result = op_callable(filepath=plain_ply_path)
    check("plain point-cloud PLY import returned FINISHED", result == {'FINISHED'}, str(result))
    new_objs = set(bpy.data.objects.keys()) - before_objs
    if new_objs:
        plain_obj = bpy.data.objects[list(new_objs)[0]]
        check("plain point-cloud PLY has correct vertex count",
              len(plain_obj.data.vertices) == plain_n, str(len(plain_obj.data.vertices)))
        color_attr = plain_obj.data.attributes.get("splat_color")
        first_color = list(color_attr.data[0].color) if color_attr is not None else None
        check(
            "plain point-cloud PLY dequantised uchar color to 0..1",
            first_color is not None
            and abs(first_color[0] - 1.0) < 1e-3
            and abs(first_color[1] - 128 / 255.0) < 1e-3,
            str(first_color),
        )

    # ---- Zero-vertex PLY: structurally valid, but nothing to import ----
    empty_ply_path = os.path.join(tmpdir, "empty.ply")
    with open(empty_ply_path, "wb") as f:
        f.write(
            ("ply\nformat binary_little_endian 1.0\n"
             "element vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
             "end_header\n").encode("ascii")
        )
    before_objs = set(bpy.data.objects.keys())
    try:
        op_callable(filepath=empty_ply_path)
        check("zero-vertex PLY rejected with a specific error", False, "did not raise")
    except RuntimeError as exc:
        check("zero-vertex PLY rejected with a specific error", "zero vertices" in str(exc), str(exc))
    check("zero-vertex PLY rejection created no object",
          set(bpy.data.objects.keys()) == before_objs)

    # ---- Corrupt files rejected cleanly: a specific reported error, no new
    # object, no crash. NOTE: bpy.ops itself converts an operator's
    # self.report({'ERROR'}, ...) + {'CANCELLED'} into a Python RuntimeError
    # when called this way (standard Blender behaviour, not a bug in this
    # add-on) -- so "rejected cleanly" here means "a RuntimeError carrying
    # our specific message, not an unrelated crash, and no object created",
    # not "no exception at all".
    def _check_rejected(path, label, expected_substring):
        before = set(bpy.data.objects.keys())
        try:
            op_callable(filepath=path)
            check(label, False, "operator did not raise for a corrupt file")
        except RuntimeError as exc:
            msg = str(exc)
            check(label, expected_substring in msg, msg)
        after = set(bpy.data.objects.keys())
        check(f"{label} (no object created)", after == before, str(after - before))

    _check_rejected(corrupt_ply_path, "corrupt .ply rejected with a specific error", "not a PLY file")
    _check_rejected(corrupt_spz_path, "corrupt .spz rejected with a specific error", "bad .spz magic number")

    # ---- Menu entry wired up ----
    menu_funcs = bpy.types.TOPBAR_MT_file_import._dyn_ui_initialize()
    from blender_addon import operators as ops_mod
    check("import menu function is registered on File > Import",
          ops_mod.menu_func_import in menu_funcs)

    # ---- Unregister cleanly ----
    addon.unregister()
    check(
        "operator un-registered from bpy.ops",
        not hasattr(bpy.ops.import_scene, op_id.split(".")[1])
        or op_id.split(".")[1] not in dir(bpy.ops.import_scene),
    )

    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILURES:")
        for name, detail in _FAIL:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
