# =============================================================================
#  operators.py -- the File > Import operator. Reads a .ply / .spz into a
#  SplatCloud (splat_data.py), builds a Blender mesh (vertices = splat
#  centres, plus per-point attributes the shader/geometry-nodes graph
#  read by name), and wires up the material + Geometry Nodes modifier so
#  the object renders as splats the moment it appears.
# =============================================================================

import array
import os

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from . import brand
from . import geonode_setup
from . import material_setup
from . import ply_reader
from . import spz_reader
from . import splat_data

SCALE_ATTR = "splat_scale"
ROTATION_ATTR = "splat_rotation_euler"
COLOR_ATTR = "splat_color"
OPACITY_ATTR = "splat_opacity"


def _load_cloud(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".ply":
        return splat_data.from_ply(filepath)
    if ext == ".spz":
        return splat_data.from_spz(filepath)
    raise ValueError(f"unrecognised extension {ext!r} (expected .ply or .spz)")


def _build_mesh(cloud, name):
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(cloud.count)
    mesh.vertices.foreach_set("co", cloud.positions)

    scale_attr = mesh.attributes.new(name=SCALE_ATTR, type='FLOAT_VECTOR', domain='POINT')
    scale_attr.data.foreach_set("vector", cloud.scales)

    rot_attr = mesh.attributes.new(name=ROTATION_ATTR, type='FLOAT_VECTOR', domain='POINT')
    rot_attr.data.foreach_set("vector", cloud.rotations_euler)

    color_attr = mesh.attributes.new(name=COLOR_ATTR, type='FLOAT_COLOR', domain='POINT')
    rgba = array.array('f', [0.0]) * (cloud.count * 4)
    for i in range(cloud.count):
        rgba[i * 4 + 0] = cloud.colors_rgb[i * 3 + 0]
        rgba[i * 4 + 1] = cloud.colors_rgb[i * 3 + 1]
        rgba[i * 4 + 2] = cloud.colors_rgb[i * 3 + 2]
        rgba[i * 4 + 3] = 1.0  # alpha handled separately via OPACITY_ATTR, not here
    color_attr.data.foreach_set("color", rgba)

    opacity_attr = mesh.attributes.new(name=OPACITY_ATTR, type='FLOAT', domain='POINT')
    opacity_attr.data.foreach_set("value", cloud.opacities)

    mesh.update()
    return mesh


class NIMBUS_OT_import_splat(Operator, ImportHelper):
    """Import a Nimbus3D-exported Gaussian splat (.ply or .spz) as a point
    cloud with a Geometry Nodes + shader setup approximating splat
    rendering. See blender_addon/README.md for exactly what this does and
    does not reproduce compared to a real splat renderer."""

    bl_idname = brand.IMPORT_OPERATOR_ID
    bl_label = f"Import {brand.PRODUCT_DISPLAY_NAME} Splat"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".ply"
    filter_glob: StringProperty(
        default="*.ply;*.spz",
        options={'HIDDEN'},
    )

    def execute(self, context):
        filepath = self.filepath
        try:
            cloud = _load_cloud(filepath)
        except (ply_reader.PlyParseError, spz_reader.SpzParseError, ValueError, OSError) as exc:
            self.report({'ERROR'}, f"Could not import {filepath}: {exc}")
            return {'CANCELLED'}

        if cloud.count == 0:
            self.report({'ERROR'}, f"{filepath} has zero vertices; nothing to import")
            return {'CANCELLED'}

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        mesh = _build_mesh(cloud, base_name)

        obj = bpy.data.objects.new(base_name, mesh)
        context.collection.objects.link(obj)
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        material = material_setup.build_splat_material()
        geonode_setup.apply_to_object(obj, material)

        kind = "full Gaussian splat" if cloud.is_full_gaussian else "point cloud (no scale/rotation in file)"
        sh_note = f", SH degree {cloud.sh_degree} (DC only is rendered)" if cloud.sh_degree else ""
        self.report(
            {'INFO'},
            f"{brand.PRODUCT_DISPLAY_NAME}: imported {cloud.count:,} points as a "
            f"{kind} from {cloud.source_format.upper()}{sh_note}.",
        )
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(NIMBUS_OT_import_splat.bl_idname, text=brand.IMPORT_MENU_LABEL)


CLASSES = (NIMBUS_OT_import_splat,)
