# =============================================================================
#  geonode_setup.py -- builds the Geometry Nodes group that turns the
#  imported point-cloud mesh into oriented, per-axis-scaled, coloured splat
#  approximations.
#
#  Built with Python at import time (not shipped as a baked .blend asset)
#  so the add-on is pure .py, has no binary asset to keep in sync, and the
#  node group name still derives from the ONE brand constant (brand.py).
#
#  APPROXIMATION, STATED PLAINLY (see README.md "How this differs from a
#  real Gaussian-splat renderer"): each splat is instanced as a small
#  ico-sphere, non-uniformly scaled by the splat's own (scale_x, scale_y,
#  scale_z) and rotated by its own orientation. A near-planar Gaussian
#  (two large scale axes, one tiny one -- the common case after training)
#  renders as a thin oriented ellipsoid, i.e. a disc. This is a solid-
#  surface stand-in for a soft alpha-weighted Gaussian kernel: Blender's
#  EEVEE/Cycles do not have a native anisotropic-Gaussian-splat primitive,
#  and shipping one would mean a custom Cycles OSL/compiled kernel, out of
#  scope for an import add-on. Ico-sphere (not a flat 2D circle) is used
#  deliberately so the approximation is correct regardless of WHICH local
#  axis happens to be the thin one for a given splat.
# =============================================================================

import bpy

from . import brand

NODE_GROUP_NAME = f"{brand.INTERNAL_PREFIX}_splat_render"

_SCALE_ATTR = "splat_scale"
_ROTATION_ATTR = "splat_rotation_euler"


def _rebuild_interface(ng):
    """Blender 4.x node-group interface API. This add-on targets 4.x only,
    per the product spec, so no 3.x fallback is implemented."""
    iface = ng.interface
    for item in list(iface.items_tree):
        iface.remove(item)

    iface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    scale_in = iface.new_socket(
        name="Scale Multiplier", in_out='INPUT', socket_type='NodeSocketFloat'
    )
    scale_in.default_value = 1.0
    scale_in.min_value = 0.0
    scale_in.subtype = 'FACTOR'

    subdiv_in = iface.new_socket(
        name="Instance Subdivisions", in_out='INPUT', socket_type='NodeSocketInt'
    )
    subdiv_in.default_value = 1
    subdiv_in.min_value = 0
    subdiv_in.max_value = 3

    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')


def build_splat_node_group(material):
    """(Re)builds the shared splat-render node group and returns it. Safe to
    call on every import: an existing group with this name is cleared and
    rebuilt in place, so already-placed modifiers on other objects keep
    working (same datablock, new internals) instead of accumulating
    ``.001``-suffixed duplicates every time the user imports another file.
    """
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if ng is None or ng.bl_idname != 'GeometryNodeTree':
        if ng is not None:
            bpy.data.node_groups.remove(ng)
        ng = bpy.data.node_groups.new(NODE_GROUP_NAME, 'GeometryNodeTree')

    ng.nodes.clear()
    _rebuild_interface(ng)

    nodes = ng.nodes
    links = ng.links

    group_in = nodes.new('NodeGroupInput')
    group_in.location = (-800, 0)

    group_out = nodes.new('NodeGroupOutput')
    group_out.location = (800, 0)

    ico = nodes.new('GeometryNodeMeshIcoSphere')
    ico.location = (-600, -250)
    ico.inputs['Radius'].default_value = 1.0
    ico.inputs['Subdivisions'].default_value = 1

    smooth = nodes.new('GeometryNodeSetShadeSmooth')
    smooth.location = (-400, -250)
    smooth.inputs['Shade Smooth'].default_value = True

    scale_attr = nodes.new('GeometryNodeInputNamedAttribute')
    scale_attr.location = (-800, -450)
    scale_attr.data_type = 'FLOAT_VECTOR'
    scale_attr.inputs['Name'].default_value = _SCALE_ATTR

    scale_mul = nodes.new('ShaderNodeVectorMath')
    scale_mul.location = (-600, -450)
    scale_mul.operation = 'SCALE'

    rot_attr = nodes.new('GeometryNodeInputNamedAttribute')
    rot_attr.location = (-800, -650)
    rot_attr.data_type = 'FLOAT_VECTOR'
    rot_attr.inputs['Name'].default_value = _ROTATION_ATTR

    euler_to_rot = nodes.new('FunctionNodeEulerToRotation')
    euler_to_rot.location = (-600, -650)

    instance = nodes.new('GeometryNodeInstanceOnPoints')
    instance.location = (-200, 0)

    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (200, 0)

    set_mat = nodes.new('GeometryNodeSetMaterial')
    set_mat.location = (500, 0)
    set_mat.inputs['Material'].default_value = material

    # Instance primitive: ico-sphere, subdivisions driven by the group input
    # so the user can trade fidelity for viewport speed on a huge cloud.
    links.new(group_in.outputs['Instance Subdivisions'], ico.inputs['Subdivisions'])
    links.new(ico.outputs['Mesh'], smooth.inputs['Geometry'])

    # Per-point scale, multiplied by the group's Scale Multiplier.
    links.new(scale_attr.outputs['Attribute'], scale_mul.inputs[0])
    links.new(group_in.outputs['Scale Multiplier'], scale_mul.inputs['Scale'])

    # Per-point rotation, stored at import time as XYZ Euler radians
    # (splat_data.py converts the PLY/SPZ quaternion once, at import, so
    # this graph stays simple and version-stable rather than depending on
    # a Vector->Rotation implicit-cast that varies across 4.x point
    # releases).
    links.new(rot_attr.outputs['Attribute'], euler_to_rot.inputs['Euler'])

    links.new(group_in.outputs['Geometry'], instance.inputs['Points'])
    links.new(smooth.outputs['Geometry'], instance.inputs['Instance'])
    links.new(euler_to_rot.outputs['Rotation'], instance.inputs['Rotation'])
    links.new(scale_mul.outputs['Vector'], instance.inputs['Scale'])

    links.new(instance.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], set_mat.inputs['Geometry'])
    links.new(set_mat.outputs['Geometry'], group_out.inputs['Geometry'])

    return ng


def apply_to_object(obj, material):
    """Adds (or replaces) a Geometry Nodes modifier on `obj` using the
    shared splat-render node group."""
    ng = build_splat_node_group(material)

    mod_name = f"{brand.INTERNAL_PREFIX} Splat"
    existing = obj.modifiers.get(mod_name)
    if existing is not None:
        obj.modifiers.remove(existing)

    mod = obj.modifiers.new(name=mod_name, type='NODES')
    mod.node_group = ng
    return mod
