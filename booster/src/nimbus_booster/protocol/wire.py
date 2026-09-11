"""The wire types, matching ``docs/BOOSTER_PROTOCOL.md`` byte for byte.

Every field name here is exactly the Swift property name in
``ios/Sources/Booster/BoosterProtocol.swift``. There are no ``CodingKeys`` on
the Swift side, so the JSON key *is* the property name, camelCase, with ``ID``
capitalised: ``scanID``, never ``scanId``.

Two traps this module exists to avoid, both of which parse fine right up until
they do not:

1. **Dates must not carry fractional seconds.** The client decodes with
   Foundation's ``.iso8601`` strategy, which is RFC 3339 to whole seconds only.
   ``2026-09-03T14:12:05Z`` decodes; ``2026-09-03T14:12:05.512Z`` throws and
   takes the whole message with it. :func:`iso8601` is the only place a
   timestamp is formatted.
2. **A non-optional field must be present.** Swift's synthesised decoder throws
   on a missing key. ``None`` is legal only for the fields typed ``T?`` in the
   Swift source; everywhere else it is an error, not a shortcut. Each
   ``to_json`` below drops ``None`` only for the genuinely-optional fields and
   never for the mandatory ones.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Frozen protocol constants. See the note in brand.py: these are wire bytes,
# not brand, and the shipped Swift client spells them as literals.
# --------------------------------------------------------------------------

#: Bumped ONLY for a breaking change: a removed or renamed field, a changed
#: field type, a new enum value the old client cannot decode, or changed
#: resume semantics. Additive optional fields do not bump it.
API_VERSION = "1"

PATH_PREFIX = "/v1"

HEADER_API_VERSION = "X-Nimbus-Api-Version"
HEADER_CHUNK_OFFSET = "X-Nimbus-Chunk-Offset"
HEADER_CHUNK_SHA256 = "X-Nimbus-Chunk-Sha256"

#: Advisory only. The client ignores it and always sends 1 MiB windows; the
#: server must accept whatever size actually arrives. The field exists so a
#: future client can honour it.
CHUNK_SIZE = 1_048_576


# --------------------------------------------------------------------------
# Enums. Lowercase strings, exactly these spellings. An unrecognised value
# fails the decode of the whole message on the client, so a new stage is an
# API version bump, not a quiet addition.
# --------------------------------------------------------------------------


class PairStatusValue:
    """``BoosterPairStatusValue``."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"

    ALL = (PENDING, APPROVED, DENIED, EXPIRED)


class JobStage:
    """``BoosterJobStage``.

    The machine::

        queued -> receiving -> verifying -> training -> exporting -> ready
                                                                  \\-> failed
           (any stage) -> cancelled
    """

    QUEUED = "queued"
    RECEIVING = "receiving"
    VERIFYING = "verifying"
    TRAINING = "training"
    EXPORTING = "exporting"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"

    ALL = (
        QUEUED,
        RECEIVING,
        VERIFYING,
        TRAINING,
        EXPORTING,
        READY,
        FAILED,
        CANCELLED,
    )

    #: The client stops listening after any of these and closes the socket.
    TERMINAL = (READY, FAILED, CANCELLED)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


def iso8601(when: Optional[_dt.datetime] = None) -> str:
    """Format a timestamp the way Foundation's ``.iso8601`` can decode it.

    Whole seconds, UTC, ``Z`` suffix. The ``microsecond=0`` is not cosmetic:
    with it left in, ``JSONDecoder`` throws and the client drops the entire
    message.

    >>> iso8601(_dt.datetime(2026, 9, 3, 14, 12, 5, 512000, _dt.timezone.utc))
    '2026-09-03T14:12:05Z'
    """
    if when is None:
        when = _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    when = when.astimezone(_dt.timezone.utc).replace(microsecond=0)
    return when.isoformat().replace("+00:00", "Z")


def parse_iso8601(text: str) -> _dt.datetime:
    """Read back what :func:`iso8601` wrote. Tolerant of fractional seconds."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return _dt.datetime.fromisoformat(cleaned)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@dataclass
class ManifestFile:
    """``BoosterManifestFile``.

    ``sha256`` is the lowercase hex digest of the WHOLE file, 64 characters,
    no ``0x``. It is used both for resume (a changed hash voids that file's
    progress) and for the verify step at finalize.
    """

    relativePath: str
    byteCount: int
    sha256: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "relativePath": self.relativePath,
            "byteCount": int(self.byteCount),
            "sha256": self.sha256,
        }

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "ManifestFile":
        return ManifestFile(
            relativePath=str(raw["relativePath"]),
            byteCount=int(raw["byteCount"]),
            sha256=str(raw["sha256"]).lower(),
        )


@dataclass
class Manifest:
    """``BoosterManifest``.

    ``totalByteCount`` is a computed property on the Swift side and is
    therefore NOT serialised. Do not expect it and do not require it; sending
    it back is harmless because unknown fields are ignored.
    """

    scanID: str
    files: List[ManifestFile] = field(default_factory=list)

    @property
    def total_byte_count(self) -> int:
        return sum(f.byteCount for f in self.files)

    def sorted_files(self) -> List[ManifestFile]:
        """Files by ``relativePath`` ascending, byte-wise.

        The order is stable so a rebuilt manifest lines up with a
        partially-uploaded job.
        """
        return sorted(self.files, key=lambda f: f.relativePath.encode("utf-8"))

    def by_path(self) -> Dict[str, ManifestFile]:
        return {f.relativePath: f for f in self.files}

    def to_json(self) -> Dict[str, Any]:
        return {
            "scanID": self.scanID,
            "files": [f.to_json() for f in self.sorted_files()],
        }

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "Manifest":
        # scanID is mandatory in the Swift type: a result manifest without it
        # fails to decode on the phone, so refuse to build one without it.
        return Manifest(
            scanID=str(raw["scanID"]),
            files=[ManifestFile.from_json(f) for f in raw.get("files", [])],
        )


# --------------------------------------------------------------------------
# Info
# --------------------------------------------------------------------------


@dataclass
class BoosterInfo:
    """``BoosterInfo``. Served by the one endpoint that needs no token."""

    boosterID: str
    boosterName: str
    apiVersion: str
    freeDiskBytes: int
    queueDepth: int
    gpuName: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "boosterID": self.boosterID,
            "boosterName": self.boosterName,
            "apiVersion": self.apiVersion,
            "freeDiskBytes": int(self.freeDiskBytes),
            "queueDepth": int(self.queueDepth),
        }
        # gpuName is `String?` on the client, so omitting it is legal.
        if self.gpuName is not None:
            out["gpuName"] = self.gpuName
        return out


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


@dataclass
class PairRequestBody:
    """``BoosterPairRequestBody``, sent by the phone."""

    deviceID: str
    deviceName: str
    appVersion: str

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "PairRequestBody":
        return PairRequestBody(
            deviceID=str(raw["deviceID"]),
            deviceName=str(raw["deviceName"]),
            appVersion=str(raw["appVersion"]),
        )


@dataclass
class PairRequestResponse:
    """``BoosterPairRequestResponse``."""

    requestID: str
    awaitingCode: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "requestID": self.requestID,
            "awaitingCode": bool(self.awaitingCode),
        }


@dataclass
class PairConfirmBody:
    """``BoosterPairConfirmBody``."""

    code: str

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "PairConfirmBody":
        return PairConfirmBody(code=str(raw["code"]))


@dataclass
class PairStatusResponse:
    """``BoosterPairStatusResponse``.

    ``token``, ``boosterID`` and ``boosterName`` are optional and are present
    only when ``status`` is ``approved``. If ``approved`` arrives without a
    token the client shows "The Booster approved pairing but did not send a
    token" and stops, so :meth:`to_json` refuses to build that message.
    """

    status: str
    token: Optional[str] = None
    boosterID: Optional[str] = None
    boosterName: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        if self.status not in PairStatusValue.ALL:
            raise ValueError("unknown pair status " + repr(self.status))
        if self.status == PairStatusValue.APPROVED and not self.token:
            raise ValueError(
                "refusing to send status=approved with no token: the client "
                "treats that as a dead end and tells the user the Booster "
                "approved pairing but did not send a token"
            )
        out: Dict[str, Any] = {"status": self.status}
        if self.token is not None:
            out["token"] = self.token
        if self.boosterID is not None:
            out["boosterID"] = self.boosterID
        if self.boosterName is not None:
            out["boosterName"] = self.boosterName
        return out


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class CreateJobBody:
    """``BoosterCreateJobBody``."""

    manifest: Manifest
    appVersion: str

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "CreateJobBody":
        return CreateJobBody(
            manifest=Manifest.from_json(raw["manifest"]),
            appVersion=str(raw.get("appVersion", "")),
        )


@dataclass
class CreateJobResponse:
    """``BoosterCreateJobResponse``.

    ``receivedByteOffsets`` is mandatory: send ``{}`` for a brand-new job, not
    ``null``. It may omit files with zero bytes received; the client reads a
    missing key as 0.
    """

    jobID: str
    chunkSize: int
    receivedByteOffsets: Dict[str, int]

    def to_json(self) -> Dict[str, Any]:
        return {
            "jobID": self.jobID,
            "chunkSize": int(self.chunkSize),
            "receivedByteOffsets": {
                k: int(v) for k, v in self.receivedByteOffsets.items() if v
            },
        }


@dataclass
class ChunkUploadResponse:
    """``BoosterChunkUploadResponse``.

    ``receivedByteOffset`` is the TOTAL bytes now held for this file and is
    server-authoritative. The client trusts it over its own bookkeeping and
    seeks its local read cursor to whatever we return, which is what makes a
    partial write on our side recoverable instead of silently corrupting.
    """

    receivedByteOffset: int

    def to_json(self) -> Dict[str, Any]:
        return {"receivedByteOffset": int(self.receivedByteOffset)}


@dataclass
class FinalizeResponse:
    """``BoosterFinalizeResponse``.

    ``reason`` is shown to the user VERBATIM, so it must read as a plain
    sentence, never as a stack trace or an error code.
    """

    accepted: bool
    reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"accepted": bool(self.accepted)}
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass
class ProgressEvent:
    """``BoosterProgressEvent``, one WebSocket message.

    ``fractionComplete`` is optional ON PURPOSE. Omit it when you genuinely do
    not know: the client draws an indeterminate spinner for a missing fraction
    and a real bar for a present one, and a fake ``0.0`` renders as a bar
    frozen at zero, which reads as "broken" to a non-technical user.

    ``message`` is mandatory and is shown verbatim.
    """

    stage: str
    message: str
    fractionComplete: Optional[float] = None
    timestamp: Optional[_dt.datetime] = None

    def to_json(self) -> Dict[str, Any]:
        if self.stage not in JobStage.ALL:
            raise ValueError("unknown job stage " + repr(self.stage))
        out: Dict[str, Any] = {
            "stage": self.stage,
            "message": self.message,
            "timestamp": iso8601(self.timestamp),
        }
        if self.fractionComplete is not None:
            # Clamp rather than trust a caller's arithmetic: a fraction above
            # 1.0 draws a bar past the end of its track.
            out["fractionComplete"] = max(0.0, min(1.0, float(self.fractionComplete)))
        return out


@dataclass
class JobStatusResponse:
    """``BoosterJobStatusResponse``, the 3-second polling fallback."""

    jobID: str
    stage: str
    message: str
    fractionComplete: Optional[float] = None
    resultManifest: Optional[Manifest] = None

    def to_json(self) -> Dict[str, Any]:
        if self.stage not in JobStage.ALL:
            raise ValueError("unknown job stage " + repr(self.stage))
        out: Dict[str, Any] = {
            "jobID": self.jobID,
            "stage": self.stage,
            "message": self.message,
        }
        if self.fractionComplete is not None:
            out["fractionComplete"] = max(0.0, min(1.0, float(self.fractionComplete)))
        if self.resultManifest is not None:
            out["resultManifest"] = self.resultManifest.to_json()
        return out
