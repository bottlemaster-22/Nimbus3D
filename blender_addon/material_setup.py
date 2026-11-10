# =============================================================================
#  material_setup.py -- the shader that reads the per-point attributes
#  operators.py wrote onto the mesh and turns them into an unlit, alpha-
#  blended look.
#
#  Deliberately Emission, not Principled BSDF: a splat's colour was baked
#  from training against real-world lighting already (it is not an albedo
#  waiting to be re-lit). Feeding it through Principled and the scene's
#  lights would light it TWICE. Emission + Transparent, mixed by the
#  per-point opacity attribute, reproduces the flat "already lit" look a
#  Gaussian-splat viewer shows, while still obeying Blender's own scene
#  lighting for anything else in the same viewport.
#
#  Only the SH degree-0 (DC / "flat colour") term is used here -- see
#  splat_data.py and README.md "Limitations": higher spherical-harmonic
#  bands (view-dependent colour) are imported into the `sh_rest` mesh
#  attributes so they are not thrown away, but this default material does
#  not evaluate them (that needs a per-shader spherical-harmonics
#  evaluation against the camera vector, real but nontrivial node work,
#  and out of scope for a first import add-on -- flagged, not faked).
# =============================================================================

import bpy

from . import brand

MATERIAL_NAME = f"{brand.INTERNAL_PREFIX}_splat_material"

COLOR_ATTR = "splat_color"
OPACITY_ATTR = "splat_opacity"


def build_splat_material():
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(MATERIAL_NAME)
    mat.use_nodes = True

    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    attr_color = tree.nodes.new('ShaderNodeAttribute')
    attr_color.location = (-400, 100)
    attr_color.attribute_type = 'GEOMETRY'
    attr_color.attribute_name = COLOR_ATTR

    attr_opacity = tree.nodes.new('ShaderNodeAttribute')
    attr_opacity.location = (-400, -150)
    attr_opacity.attribute_type = 'GEOMETRY'
    attr_opacity.attribute_name = OPACITY_ATTR

    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = (-100, 150)
    emission.inputs['Strength'].default_value = 1.0

    transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (-100, -50)

    mix = tree.nodes.new('ShaderNodeMixShader')
    mix.location = (150, 0)

    links = tree.links
    links.new(attr_color.outputs['Color'], emission.inputs['Color'])
    links.new(attr_opacity.outputs['Fac'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    # Alpha blending: the exact enum values on Material.blend_method have
    # shifted between EEVEE Legacy (<=4.1) and EEVEE Next (>=4.2). Try the
    # values in order of preference and accept whichever this Blender
    # build actually has rather than hard-failing registration over a
    # cosmetic setting -- a cloud of many overlapping translucent discs
    # has no single correct sort order either way (see README.md).
    for value in ('HASHED', 'BLEND', 'DITHERED'):
        try:
            mat.blend_method = value
            break
        except TypeError:
            continue

    try:
        mat.show_transparent_back = False
    except AttributeError:
        pass  # property does not exist on this Blender version; harmless

    try:
        mat.use_backface_culling = False
    except AttributeError:
        pass

    return mat
