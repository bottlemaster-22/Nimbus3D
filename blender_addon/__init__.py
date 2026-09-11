# =============================================================================
#  Nimbus3D Blender add-on -- entry point.
#
#  Optional. Nobody needs this to use Nimbus3D on the phone; it exists so a
#  scan can be dropped straight into Blender for a look, a render, or as a
#  modelling reference, without going through a third-party converter. See
#  README.md for exactly what it does, its honest limitations, and the
#  existing third-party alternatives.
#
#  Product name/id: brand.py is the ONLY file allowed to contain the
#  literal product name. Every other file in this package (including this
#  one) reads it from there.
# =============================================================================

from . import brand

bl_info = {
    "name": brand.PRODUCT_DISPLAY_NAME,
    "description": (
        f"Import {brand.PRODUCT_DISPLAY_NAME} Gaussian-splat scans "
        f"(.ply / .spz) as an approximated splat point cloud."
    ),
    "author": "TOMBLINE",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "File > Import",
    "warning": "Approximate splat renderer (ico-sphere discs), not a true Gaussian rasterizer -- see README.",
    "doc_url": "",
    "category": "Import-Export",
}

# Deferred: these import bpy at module scope, which is only importable
# inside Blender's own Python. That keeps the bl_info dict above (and this
# file generally) importable without bpy, for tooling and this repo's own
# reviewers.
#
# Loaded with importlib.import_module(), and deliberately NOT pre-declared
# as `operators = None` (etc) up here for type-hinting purposes: per
# Python's import semantics, `from package import name` uses an EXISTING
# attribute of `package` named `name` if one is already present, instead
# of importing the submodule -- so a `None` placeholder on THIS module
# would silently break not just this file's own load, but every OTHER
# file in the package that does `from . import geonode_setup` or
# `from . import material_setup` (operators.py does both), each one
# quietly rebinding to the placeholder instead of the real module, no
# exception raised anywhere. selftest.py's "operator registered under
# bpy.ops" and import assertions would catch a regression here -- left as
# a comment because the failure mode is invisible by design (nothing
# raises) if it ever comes back.
_modules_loaded = False


def _load_modules():
    global operators, geonode_setup, material_setup, _modules_loaded
    if _modules_loaded:
        return
    import importlib
    operators = importlib.import_module(".operators", __package__)
    geonode_setup = importlib.import_module(".geonode_setup", __package__)
    material_setup = importlib.import_module(".material_setup", __package__)
    _modules_loaded = True


def register():
    _load_modules()
    import bpy

    for cls in operators.CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(operators.menu_func_import)


def unregister():
    _load_modules()
    import bpy

    bpy.types.TOPBAR_MT_file_import.remove(operators.menu_func_import)
    for cls in reversed(operators.CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
