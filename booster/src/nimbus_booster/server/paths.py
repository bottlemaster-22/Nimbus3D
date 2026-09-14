"""Save-root layout and the relative-path safety check.

``docs/BOOSTER_PROTOCOL.md`` section 12::

    <root>/Incoming/<scanID>/    partially or fully uploaded capture bundles
    <root>/Completed/<scanID>/   finished jobs, capture bundle + result/
    <root>/Failed/<scanID>/      whatever a failed job left behind

The uploaded tree under ``Incoming/<scanID>/`` mirrors the phone's scan folder
exactly, so ``docs/DATA_FORMAT.md`` describes it unchanged and the trainer
opens it with the same code path as a locally-produced capture.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .. import brand


class UnsafeRelativePath(ValueError):
    """A relative path from the wire that must be refused, not sanitised.

    The protocol is explicit about this: "Reject rather than sanitise: a
    caller sending ``../`` is not making a typo."
    """


#: Characters that are illegal in a Windows path component. Refusing them
#: keeps a scan bundle portable: a name that writes fine on Linux but cannot
#: be created on Windows turns into a mid-transfer failure on the user's PC,
#: which is a worse place to discover it than at the first chunk.
_WINDOWS_RESERVED_CHARS = set('<>:"|?*')

#: Windows refuses these as a whole component, with or without an extension.
_WINDOWS_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + ["COM{}".format(i) for i in range(1, 10)]
    + ["LPT{}".format(i) for i in range(1, 10)]
)


def safe_relative_path(raw: str) -> str:
    """Validate one wire relative path, returning it in normalised POSIX form.

    Raises :class:`UnsafeRelativePath` for anything that is absolute, contains
    a ``..`` segment, contains a backslash, contains a NUL, or would escape
    the job directory. The check is on the DECODED string; aiohttp has already
    percent-decoded it by the time we see it.

    >>> safe_relative_path("images/frame_20260903_141205_512.jpg")
    'images/frame_20260903_141205_512.jpg'
    >>> safe_relative_path("../../etc/passwd")
    Traceback (most recent call last):
        ...
    nimbus_booster.server.paths.UnsafeRelativePath: ...
    """
    if not raw:
        raise UnsafeRelativePath("empty relative path")
    if "\x00" in raw:
        raise UnsafeRelativePath("relative path contains a NUL byte")
    if "\\" in raw:
        # A backslash is a separator on Windows, so allowing it would let a
        # caller escape the job directory on exactly the platform this server
        # targets first.
        raise UnsafeRelativePath("relative path contains a backslash: " + raw)
    if raw.startswith("/"):
        raise UnsafeRelativePath("relative path is absolute: " + raw)
    # A Windows drive letter ("C:foo") or a UNC-ish prefix.
    if len(raw) >= 2 and raw[1] == ":":
        raise UnsafeRelativePath("relative path names a drive: " + raw)

    parts = []
    for part in raw.split("/"):
        if part in ("", "."):
            # A doubled slash or a "./" is sloppy rather than hostile, but the
            # protocol says reject rather than sanitise, and the client never
            # produces one.
            raise UnsafeRelativePath("relative path has an empty segment: " + raw)
        if part == "..":
            raise UnsafeRelativePath("relative path escapes the job: " + raw)
        if any(c in _WINDOWS_RESERVED_CHARS for c in part):
            raise UnsafeRelativePath("relative path has an illegal character: " + raw)
        if part.upper().split(".")[0] in _WINDOWS_RESERVED_NAMES:
            raise UnsafeRelativePath("relative path uses a reserved name: " + raw)
        if part.endswith(" ") or part.endswith("."):
            # Windows silently strips these, so two different wire paths would
            # land on one file.
            raise UnsafeRelativePath("relative path segment ends in . or space: " + raw)
        parts.append(part)

    if len(parts) > 32:
        raise UnsafeRelativePath("relative path is implausibly deep: " + raw)
    return "/".join(parts)


def resolve_within(base: Path, relative: str) -> Path:
    """Join a validated relative path onto ``base`` and prove it stayed inside.

    :func:`safe_relative_path` already rejects the obvious attacks; this is the
    belt to its braces, and it is the check that catches the non-obvious one:
    a symlink placed inside the job directory pointing somewhere else. The
    comparison is done on the fully resolved real paths.
    """
    checked = safe_relative_path(relative)
    base_real = Path(os.path.realpath(str(base)))
    candidate = base_real.joinpath(*checked.split("/"))
    # The file itself may not exist yet, so resolve the deepest parent that
    # does and check that.
    probe = candidate
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    probe_real = Path(os.path.realpath(str(probe)))
    try:
        probe_real.relative_to(base_real)
    except ValueError:
        raise UnsafeRelativePath(
            "relative path resolves outside the job directory: " + relative
        ) from None
    return candidate


def default_save_root() -> Path:
    """``~/Documents/<brand>/Scans``, the default the protocol names.

    Falls back to the home directory when there is no ``Documents`` (a Linux
    box without XDG user dirs, a stripped CI container).
    """
    home = Path.home()
    documents = home / "Documents"
    if not documents.is_dir():
        documents = home
    return documents / brand.DOCUMENTS_FOLDER_NAME / brand.FOLDER_SCANS


class SaveRoot:
    """The three-folder save area, created on demand.

    The root is configurable; only the three folder names under it are fixed,
    because they are part of the data format rather than part of the brand.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- the three areas -------------------------------------------------

    @property
    def incoming(self) -> Path:
        return self.root / brand.FOLDER_INCOMING

    @property
    def completed(self) -> Path:
        return self.root / brand.FOLDER_COMPLETED

    @property
    def failed(self) -> Path:
        return self.root / brand.FOLDER_FAILED

    def ensure(self) -> None:
        for path in (self.incoming, self.completed, self.failed):
            path.mkdir(parents=True, exist_ok=True)

    # -- per-scan directories --------------------------------------------

    def incoming_scan(self, scan_id: str) -> Path:
        return self.incoming / _safe_scan_id(scan_id)

    def completed_scan(self, scan_id: str) -> Path:
        return self.completed / _safe_scan_id(scan_id)

    def failed_scan(self, scan_id: str) -> Path:
        return self.failed / _safe_scan_id(scan_id)

    def result_dir(self, scan_id: str) -> Path:
        """``Completed/<scanID>/result/`` - what the result endpoints serve."""
        return self.completed_scan(scan_id) / "result"

    # -- moving a finished or abandoned job -------------------------------

    def move_to(self, scan_id: str, destination: Path) -> Path:
        """Move ``Incoming/<scanID>`` to ``destination/<scanID>``.

        Replaces an existing directory of the same name, because a re-sent scan
        supersedes whatever the previous attempt left behind. Returns the new
        location. A cross-volume move is handled by ``shutil.move``.
        """
        source = self.incoming_scan(scan_id)
        target = destination / _safe_scan_id(scan_id)
        destination.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            return target
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
        shutil.move(str(source), str(target))
        return target

    # -- disk -------------------------------------------------------------

    def free_disk_bytes(self) -> int:
        """Free space on the volume holding the save root.

        Reported in ``/v1/info`` so the phone can warn before a multi-gigabyte
        house scan fills a disk. Walks up to the first parent that exists,
        since the root itself may not have been created yet.
        """
        probe: Optional[Path] = self.root
        while probe is not None and not probe.exists():
            probe = probe.parent if probe.parent != probe else None
        if probe is None:
            return 0
        try:
            return int(shutil.disk_usage(str(probe)).free)
        except OSError:
            return 0


def _safe_scan_id(scan_id: str) -> str:
    """A scan id is a folder name, so it gets the same treatment as any path.

    ``docs/DATA_FORMAT.md`` fixes the shape as ``scan_YYYYMMDD_HHMMSS`` with an
    optional ``_2`` suffix, but the id arrives over the wire and is used to
    build a directory, so it is validated rather than trusted.
    """
    checked = safe_relative_path(scan_id)
    if "/" in checked:
        raise UnsafeRelativePath("scanID must be a single path component: " + scan_id)
    return checked
