"""The human-witnessed pairing handshake and the bearer tokens it issues.

``docs/BOOSTER_PROTOCOL.md`` section 6. No accounts, no cloud, no QR code, and
no silent trust of whatever answers on the port: the phone asks, the PC window
shows a six-digit code, a human reads it across the room and types it in.

The security properties that actually matter here, and where each one lives:

* the code is CSPRNG, six digits, and **expires after two minutes**
  (:data:`CODE_LIFETIME_SECONDS`);
* at most **five** wrong attempts per request, then the request is burned
  (:data:`MAX_CODE_ATTEMPTS`) - which is what stops a 10^6 brute force from
  being a 10^6 brute force;
* the token is 256 bits of CSPRNG output, compared in constant time, and
  stored only as a SHA-256 hash;
* pairing is revocable from the GUI, and a revoked token starts returning 401.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import secrets
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..config import BoosterConfig, PairedDevice
from ..protocol import wire

#: Two minutes, per the protocol. Long enough to walk to the PC, short enough
#: that a code left on screen is not a standing invitation.
CODE_LIFETIME_SECONDS = 120

#: Five wrong guesses and the request is dead. The phone must start over,
#: which mints a fresh code.
MAX_CODE_ATTEMPTS = 5

#: How long an approved-but-uncollected request stays readable by
#: `GET /v1/pair/requests/{id}`. The client polls this while its code sheet is
#: open; after that the record is just clutter.
APPROVED_RETENTION_SECONDS = 600


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def hash_token(token: str) -> str:
    """SHA-256 hex of a bearer token. What we persist instead of the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class PairRequest:
    """One in-flight pairing attempt."""

    requestID: str
    deviceID: str
    deviceName: str
    appVersion: str
    code: str
    createdAt: _dt.datetime
    status: str = wire.PairStatusValue.PENDING
    attempts: int = 0
    #: Set once, when the code is accepted. Handed to the phone exactly once
    #: in the confirm response; also readable by the poll endpoint until the
    #: record is swept, because the client may race the two.
    token: Optional[str] = None
    decidedAt: Optional[_dt.datetime] = None

    @property
    def expires_at(self) -> _dt.datetime:
        return self.createdAt + _dt.timedelta(seconds=CODE_LIFETIME_SECONDS)

    def seconds_remaining(self, when: Optional[_dt.datetime] = None) -> float:
        delta = self.expires_at - (when or _now())
        return max(0.0, delta.total_seconds())


class PairingManager:
    """Owns pending pair requests and the paired-device list.

    Thread-safe: the aiohttp event loop and the PySide6 GUI thread both touch
    this (the GUI to show a code and to revoke a device), so every mutation
    takes the lock.
    """

    def __init__(
        self,
        config: BoosterConfig,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._requests: Dict[str, PairRequest] = {}
        #: tokenHash -> deviceID, rebuilt from the config on construction so a
        #: restart does not un-pair anyone.
        self._tokens: Dict[str, str] = {
            d.tokenHash: d.deviceID for d in config.pairedDevices if d.tokenHash
        }
        self._on_change = on_change

    # -- notification ------------------------------------------------------

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    # -- request lifecycle -------------------------------------------------

    def create_request(self, body: wire.PairRequestBody) -> PairRequest:
        """Start a handshake and mint the code the PC window will show."""
        with self._lock:
            self._sweep_locked()
            request = PairRequest(
                requestID=str(uuid.uuid4()),
                deviceID=body.deviceID,
                deviceName=body.deviceName or "iPhone",
                appVersion=body.appVersion,
                # secrets.randbelow, not random: this is the only thing between
                # an open port and a paired device.
                code="{:06d}".format(secrets.randbelow(1_000_000)),
                createdAt=_now(),
            )
            self._requests[request.requestID] = request
        self._notify()
        return request

    def get_request(self, request_id: str) -> Optional[PairRequest]:
        with self._lock:
            self._sweep_locked()
            return self._requests.get(request_id)

    def confirm(self, request_id: str, code: str) -> Optional[PairRequest]:
        """Check a typed code and, if it matches, issue the bearer token.

        Returns the request in its new state, or ``None`` if the id is
        unknown. An expired or already-burned request is returned as-is with
        its terminal status, so the caller can answer honestly rather than
        pretending the id does not exist.
        """
        changed = False
        try:
            with self._lock:
                self._sweep_locked()
                request = self._requests.get(request_id)
                if request is None:
                    return None
                if request.status != wire.PairStatusValue.PENDING:
                    return request

                if request.seconds_remaining() <= 0.0:
                    request.status = wire.PairStatusValue.EXPIRED
                    request.decidedAt = _now()
                    changed = True
                    return request

                request.attempts += 1
                # Constant-time: a timing side channel on a six-digit code is
                # not the likeliest attack here, but compare_digest costs
                # nothing and removes the question.
                if hmac.compare_digest(request.code, str(code).strip()):
                    request.status = wire.PairStatusValue.APPROVED
                    request.decidedAt = _now()
                    request.token = secrets.token_urlsafe(32)  # 256 bits
                    self._add_paired_device_locked(request)
                    changed = True
                elif request.attempts >= MAX_CODE_ATTEMPTS:
                    # Burn it. The phone starts over and gets a fresh code,
                    # which is what makes the attempt budget mean anything.
                    request.status = wire.PairStatusValue.EXPIRED
                    request.decidedAt = _now()
                    changed = True
                return request
        finally:
            if changed:
                self._notify()

    def deny(self, request_id: str) -> None:
        """Refuse a request from the GUI. The phone sees ``denied``."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is not None and request.status == wire.PairStatusValue.PENDING:
                request.status = wire.PairStatusValue.DENIED
                request.decidedAt = _now()
        self._notify()

    def pending_requests(self) -> List[PairRequest]:
        """Live requests, for the GUI to display their codes."""
        with self._lock:
            self._sweep_locked()
            return [
                r
                for r in self._requests.values()
                if r.status == wire.PairStatusValue.PENDING
            ]

    # -- tokens ------------------------------------------------------------

    def device_for_token(self, token: str) -> Optional[str]:
        """Resolve a bearer token to a paired ``deviceID``, or ``None``.

        The lookup is by hash, so the plaintext token is never held anywhere
        but in this call's arguments.
        """
        if not token:
            return None
        digest = hash_token(token)
        with self._lock:
            device_id = self._tokens.get(digest)
            if device_id is not None:
                for device in self._config.pairedDevices:
                    if device.deviceID == device_id:
                        device.lastSeenAt = wire.iso8601()
                        break
            return device_id

    def is_authorised(self, token: Optional[str]) -> bool:
        return bool(token) and self.device_for_token(str(token)) is not None

    def revoke(self, device_id: str) -> bool:
        """Un-pair one phone. Its token starts returning 401 immediately."""
        removed = False
        with self._lock:
            keep = []
            for device in self._config.pairedDevices:
                if device.deviceID == device_id:
                    self._tokens.pop(device.tokenHash, None)
                    removed = True
                else:
                    keep.append(device)
            if removed:
                self._config.pairedDevices = keep
                self._config.save()
        if removed:
            self._notify()
        return removed

    def paired_devices(self) -> List[PairedDevice]:
        with self._lock:
            return list(self._config.pairedDevices)

    # -- internals ---------------------------------------------------------

    def _add_paired_device_locked(self, request: PairRequest) -> None:
        assert request.token is not None
        digest = hash_token(request.token)
        # Re-pairing the same phone replaces its old token rather than
        # accumulating one per attempt.
        existing = [
            d for d in self._config.pairedDevices if d.deviceID == request.deviceID
        ]
        for device in existing:
            self._tokens.pop(device.tokenHash, None)
        self._config.pairedDevices = [
            d for d in self._config.pairedDevices if d.deviceID != request.deviceID
        ]
        self._config.pairedDevices.append(
            PairedDevice(
                deviceID=request.deviceID,
                deviceName=request.deviceName,
                tokenHash=digest,
                pairedAt=wire.iso8601(),
                lastSeenAt=wire.iso8601(),
            )
        )
        self._tokens[digest] = request.deviceID
        self._config.save()

    def _sweep_locked(self) -> None:
        """Expire stale pending requests and forget decided ones."""
        now = _now()
        dead = []
        for request_id, request in self._requests.items():
            if request.status == wire.PairStatusValue.PENDING:
                if request.seconds_remaining(now) <= 0.0:
                    request.status = wire.PairStatusValue.EXPIRED
                    request.decidedAt = now
            elif request.decidedAt is not None:
                age = (now - request.decidedAt).total_seconds()
                if age > APPROVED_RETENTION_SECONDS:
                    dead.append(request_id)
        for request_id in dead:
            self._requests.pop(request_id, None)

    def status_response(self, request: PairRequest) -> wire.PairStatusResponse:
        """Build the wire answer for a request in whatever state it is in."""
        if request.status == wire.PairStatusValue.APPROVED:
            return wire.PairStatusResponse(
                status=wire.PairStatusValue.APPROVED,
                token=request.token,
                boosterID=self._config.boosterID,
                boosterName=self._config.boosterName,
            )
        return wire.PairStatusResponse(status=request.status)
