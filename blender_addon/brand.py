# =============================================================================
#  brand.py -- THE SINGLE SOURCE OF TRUTH FOR THE PRODUCT NAME IN THIS ADD-ON.
#
#  The product name ("Nimbus3D") is a WORKING NAME, not final. No other file
#  in blender_addon/ may contain the product name as a string literal. Every
#  menu label, panel title, node-group name, material name and operator
#  bl_idname in this add-on is DERIVED from the constants below.
#
#  This file is the Python-side twin of two other single-source-of-truth
#  files in this repo (Python cannot import Swift or read a .pbxproj, so it
#  cannot read from them directly -- keep this file's values in sync by hand):
#
#      ios/Sources/Core/BrandConfig.swift   (iOS app)
#      ios/project.yml -> settingGroups.brand (build settings)
#
#  If the product is renamed, change ONLY the constants below.
# =============================================================================

#: What the user reads: the add-on's display name, the File > Import label
#: prefix, panel titles.
PRODUCT_DISPLAY_NAME = "Nimbus3D"

#: Lowercase, ASCII, no spaces or punctuation. Used to build Blender
#: identifiers that have hard syntax rules: operator bl_idname
#: ("import_scene.<slug>_splat"), node group / material internal names,
#: the add-on's own module-level Python package name convention.
PRODUCT_SLUG = "nimbus3d"

#: Reverse-DNS bundle id, matches NIMBUS_BUNDLE_ID in ios/project.yml.
#: Not used by Blender directly today, but kept here so any future
#: telemetry-free "about" text stays derived rather than re-typed.
BUNDLE_IDENTIFIER = "com.tombline.nimbus"

#: bl_idname for the import operator, e.g. "import_scene.nimbus3d_splat".
IMPORT_OPERATOR_ID = f"import_scene.{PRODUCT_SLUG}_splat"

#: Internal (non-user-facing) name prefix for generated datablocks
#: (node group, material, mesh attribute layers) so they never collide with
#: a user's own datablocks and are easy to spot/clean up.
INTERNAL_PREFIX = PRODUCT_SLUG

#: File > Import submenu label, built once, used once (operators.py).
IMPORT_MENU_LABEL = f"{PRODUCT_DISPLAY_NAME} Splat (.ply / .spz)"
