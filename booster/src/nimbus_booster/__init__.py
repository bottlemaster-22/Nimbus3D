"""The optional PC Booster: a LAN listener plus our trainer on gsplat kernels.

Nothing in this package is required for the phone app to work. There is no
cloud in it, no account, and no outbound connection of any kind: it listens on
one TCP port on the local network and answers a phone that has been paired by
a human reading a six-digit code off this screen.

Layout::

    brand.py      the four brand constants, the ONLY place the name appears
    config.py     persisted settings, including the stable boosterID
    protocol/     the wire types, matching docs/BOOSTER_PROTOCOL.md
    server/       aiohttp app, pairing, jobs, Bonjour, progress fan-out
    data/         readers for docs/DATA_FORMAT.md (COLMAP model + sidecars)
    trainer/      our trainer logic (F1-F7), on gsplat rasterisation kernels
    export/       .ply / .spz writers and the preview render
    gui/          the PySide6 window

Import cost is deliberately shallow: importing ``nimbus_booster`` pulls in
nothing heavier than the standard library, so the CLI can print help on a
machine with no torch, no Qt and no CUDA.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Kept in step with pyproject.toml's ``project.version``.
__version__ = "0.1.0"
