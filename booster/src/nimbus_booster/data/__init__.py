"""Readers and writers for ``docs/DATA_FORMAT.md``.

The Booster receives a byte-for-byte mirror of the phone's scan folder, so
these modules are the PC's copy of what ``ios/Sources/Core/Contracts.swift``
describes on the phone. Everything is little-endian, every path inside a
bundle is POSIX and relative to the scan root, and nothing here guesses a
dimension that the bundle records (``settings.depthWidth`` is read, never
assumed to be 256).
"""

from __future__ import annotations

__all__ = []
