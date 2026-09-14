"""What the window shows, worked out away from Qt so it can be read and tested.

Everything here is plain Python. The window asks these functions what to
display and then displays it, which keeps the copy (the part a person actually
reads) out of the middle of the widget plumbing, and lets the wording be
checked without a display attached.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import List, Optional

from ..protocol import wire

#: How long after the last result read the window keeps saying "sending back".
#: A phone downloading a large model pauses between range requests, so a short
#: window here would flicker.
SENDING_BACK_SECONDS = 20.0

STATUS_STOPPED = "stopped"
STATUS_IDLE = "idle"
STATUS_PAIRING = "pairing"
STATUS_RECEIVING = "receiving"
STATUS_BUILDING = "building"
STATUS_SENDING = "sending"

#: Status key -> (headline, one plain sentence underneath).
_STATUS_TEXT = {
    STATUS_STOPPED: (
        "Not listening",
        "Press Start listening and this PC will wait for your phone.",
    ),
    STATUS_IDLE: (
        "Waiting for your phone",
        "Nothing to do right now. Leave this window open and send a scan.",
    ),
    STATUS_PAIRING: (
        "A phone wants to pair",
        "Type the code below into your phone to let it send scans here.",
    ),
    STATUS_RECEIVING: (
        "Receiving a scan",
        "Your phone is sending a scan over. Keep both on the same Wi-Fi.",
    ),
    STATUS_BUILDING: (
        "Building your scan",
        "This is the slow part. You can leave it running and come back.",
    ),
    STATUS_SENDING: (
        "Sending your scan back",
        "Your phone is collecting the finished scan.",
    ),
}


@dataclass
class JobRow:
    """One line of the queue table."""

    job_id: str
    scan_id: str
    stage: str
    message: str
    fraction: Optional[float]
    is_terminal: bool

    @property
    def stage_text(self) -> str:
        return stage_word(self.stage)

    @property
    def percent(self) -> Optional[int]:
        if self.fraction is None:
            return None
        return int(round(max(0.0, min(1.0, self.fraction)) * 100.0))


def stage_word(stage: str) -> str:
    """The stage in words a person uses, not the protocol's spelling."""
    return {
        wire.JobStage.QUEUED: "Waiting",
        wire.JobStage.RECEIVING: "Arriving",
        wire.JobStage.VERIFYING: "Checking",
        wire.JobStage.TRAINING: "Building",
        wire.JobStage.EXPORTING: "Saving",
        wire.JobStage.READY: "Ready",
        wire.JobStage.FAILED: "Did not work",
        wire.JobStage.CANCELLED: "Cancelled",
    }.get(stage, stage)


def status_key(
    running: bool,
    jobs: List[JobRow],
    has_pending_pair: bool,
    last_result_read: Optional[_dt.datetime] = None,
    now: Optional[_dt.datetime] = None,
) -> str:
    """Work out the one word at the top of the window.

    Order matters: a pairing code on screen is the thing the person in the room
    has to act on, so it beats a transfer that needs nothing from them.
    """
    if not running:
        return STATUS_STOPPED
    if has_pending_pair:
        return STATUS_PAIRING
    for row in jobs:
        if row.stage in (wire.JobStage.TRAINING, wire.JobStage.EXPORTING):
            return STATUS_BUILDING
    for row in jobs:
        if row.stage in (wire.JobStage.RECEIVING, wire.JobStage.VERIFYING):
            return STATUS_RECEIVING
    if last_result_read is not None:
        now = now or _dt.datetime.now(_dt.timezone.utc)
        if (now - last_result_read).total_seconds() <= SENDING_BACK_SECONDS:
            return STATUS_SENDING
    return STATUS_IDLE


def status_text(key: str) -> tuple:
    """``(headline, explanation)`` for a status key."""
    return _STATUS_TEXT.get(key, _STATUS_TEXT[STATUS_IDLE])


def queue_summary(jobs: List[JobRow]) -> str:
    """One line under the table: what the queue is doing, in plain words."""
    if not jobs:
        return "No scans yet."
    active = [j for j in jobs if not j.is_terminal]
    ready = sum(1 for j in jobs if j.stage == wire.JobStage.READY)
    failed = sum(1 for j in jobs if j.stage == wire.JobStage.FAILED)
    parts = []
    if active:
        parts.append(
            "{n} scan in progress".format(n=len(active))
            if len(active) == 1
            else "{n} scans in progress".format(n=len(active))
        )
    if ready:
        parts.append("{n} finished".format(n=ready))
    if failed:
        parts.append("{n} did not work".format(n=failed))
    return ", ".join(parts) if parts else "No scans yet."


def free_space_text(free_bytes: int) -> str:
    gigabytes = float(free_bytes) / (1024.0 ** 3)
    if gigabytes >= 10.0:
        return "{gb:.0f} GB free".format(gb=gigabytes)
    if gigabytes >= 1.0:
        return "{gb:.1f} GB free".format(gb=gigabytes)
    return "{mb:.0f} MB free, which is not much".format(
        mb=float(free_bytes) / (1024.0 ** 2)
    )
