"""The LAN listener: aiohttp routes, pairing, job store, Bonjour, progress.

Nothing in here imports torch. The server is useful on its own - it can
receive and verify a scan on a machine that cannot train one - and the trainer
is injected as a callable by :mod:`nimbus_booster.server.runner`.
"""

from __future__ import annotations

__all__ = []
