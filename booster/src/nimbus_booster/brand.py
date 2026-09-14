"""The single source of truth for the product name on the PC side.

CONTRACTS.md rule 4: the product name is a working name, written in exactly two
places (``ios/project.yml``'s brand block and, for tools that cannot read it,
one constant file per language). This is that file for Python.

No other Python file under ``booster/`` may contain the product name as a
string literal. Import from here. Renaming the product is a change to the four
constants below and nothing else.

The values here MUST stay byte-identical to
``ios/Sources/Core/BrandConfig.swift``'s ``Fallback`` block, because the phone
and the PC have to agree on the Bonjour service type or discovery silently
finds nothing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The four brand values. Change these and nothing else to rename the product.
#
# AND CHANGE THE BRAND BLOCK IN ios/project.yml AT THE SAME TIME. These four
# are one half of a pair; the phone carries the other half. When the product
# was renamed to LiKOVA only the phone side was edited, so it began browsing
# for _likovaboost._tcp while this file went on registering _nimbusboost._tcp
# and the two could never see each other again. tools/brandmatch.py is now a
# CI gate over exactly that, because compiling and linting cannot notice two
# constants in two languages drifting apart.
# --------------------------------------------------------------------------

#: What the user reads: window title, "About", every sentence of UI copy.
DISPLAY_NAME = "LiKOVA"

#: Filesystem-safe form, no spaces. Top folder under the user's Documents.
DOCUMENTS_FOLDER_NAME = "LiKOVA"

#: Short lowercase token that derived identifiers are built from.
SLUG = "likova"

#: Reverse-DNS identifier of the phone app we pair with. Informational here.
BUNDLE_IDENTIFIER = "com.tombline.likova"


# --------------------------------------------------------------------------
# Derived. Do not edit these; they follow from the four above.
# --------------------------------------------------------------------------

#: DNS-SD service type, e.g. ``_nimbusboost._tcp``. Must equal
#: ``BrandConfig.boosterServiceType`` on the phone.
BOOSTER_SERVICE_TYPE = "_{slug}boost._tcp".format(slug=SLUG)

#: Bonjour browse domain. Always the local link. zeroconf wants a trailing dot.
BOOSTER_SERVICE_DOMAIN = "local."

#: Default TCP port, matching ``BrandConfig.boosterDefaultPort``.
BOOSTER_DEFAULT_PORT = 8760

# NOTE on the X-Nimbus-* HTTP headers: they are deliberately NOT here.
# They are frozen wire bytes, not brand. The shipped Swift client spells them
# as string literals (`BoosterHTTP.swift` lines 110, 111, 168, 225), so
# deriving them from SLUG would mean a rename silently stopped the Python
# server from recognising a header the phone still sends. They live in
# `protocol/wire.py` beside the rest of the protocol constants, in the same
# category as `_tcp` and the on-disk folder names below: part of the format,
# not part of the name. Changing them is an API version bump.


# --------------------------------------------------------------------------
# Fixed sub-folder names. These are part of the on-disk data format
# (docs/DATA_FORMAT.md) and do NOT change when the product is renamed.
# Mirrors BrandConfig.Folder on the Swift side.
# --------------------------------------------------------------------------

FOLDER_SCANS = "Scans"
FOLDER_IMAGES = "images"
FOLDER_SENSOR_DATA = "sensor_data"
FOLDER_SPARSE = "sparse"
FOLDER_SPARSE_MODEL = "sparse/0"
FOLDER_ANCHORS = "anchors"
FOLDER_MESH = "mesh"
FOLDER_PREPASS = "prepass"
FOLDER_MODEL = "model"
FOLDER_EXPORT = "export"
FOLDER_CACHE = "cache"

#: Booster inbox / outbox, mirrored from BrandConfig.Folder.Booster.
FOLDER_INCOMING = "Incoming"
FOLDER_COMPLETED = "Completed"
FOLDER_FAILED = "Failed"


def assert_consistent() -> None:
    """Shout if a rename broke something that fails silently at runtime.

    Called once at start-up. Every check here corresponds to a failure mode
    that otherwise produces no error at all, just a Booster the phone never
    finds.
    """
    label = BOOSTER_SERVICE_TYPE.split(".")[0]
    # DNS-SD caps the application-protocol label at 15 characters, underscore
    # included. Break this and browsing silently returns nothing at all.
    if len(label) > 15:
        raise ValueError(
            "Bonjour service type label {label!r} is {n} characters; DNS-SD "
            "allows at most 15. Shorten SLUG in brand.py (and "
            "NIMBUS_BRAND_SLUG in ios/project.yml to match).".format(
                label=label, n=len(label)
            )
        )
    if not BOOSTER_SERVICE_TYPE.endswith("._tcp"):
        raise ValueError(
            "Bonjour service type must look like _something._tcp, got "
            + BOOSTER_SERVICE_TYPE
        )
    if "/" in DOCUMENTS_FOLDER_NAME or "\\" in DOCUMENTS_FOLDER_NAME:
        raise ValueError(
            "DOCUMENTS_FOLDER_NAME must be a single path component, got "
            + DOCUMENTS_FOLDER_NAME
        )
