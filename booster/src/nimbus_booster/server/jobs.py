"""Job records, the resume mechanism, chunk writes and finalize verification.

The single most important sentence in ``docs/BOOSTER_PROTOCOL.md`` section 7.2:
**jobs are keyed by ``scanID``**. Posting the same ``scanID`` again returns the
same ``jobID`` and the bytes already on disk, per file. There is no local
resume-state file on the phone, so this server's bookkeeping *is* the resume
mechanism, and it has to survive an app kill, a crash, and a full reinstall on
the phone side.

It also has to survive a restart on this side, which is why every job record is
written to disk beside its bytes as ``.booster_job.json`` and read back at
start-up.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..protocol import wire
from .paths import SaveRoot, UnsafeRelativePath, resolve_within

#: The per-job sidecar this server writes into the scan directory. It starts
#: with a dot and is never part of the capture bundle, so the trainer's file
#: walk ignores it and it never appears in a manifest.
JOB_RECORD_NAME = ".booster_job.json"

#: Read size when hashing a file for verification. Big enough to be fast on a
#: spinning disk, small enough that a 4 GB house scan never lands in RAM.
_HASH_BLOCK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streamed lowercase-hex SHA-256 of a whole file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_BLOCK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Job:
    """One capture bundle in flight, and then in training."""

    jobID: str
    scanID: str
    manifest: wire.Manifest
    deviceID: str
    appVersion: str
    createdAt: str
    stage: str = wire.JobStage.QUEUED
    message: str = "Waiting."
    fractionComplete: Optional[float] = None
    #: Set once the result exists and the result endpoints can serve it.
    resultManifest: Optional[wire.Manifest] = None
    #: Where the bundle currently lives: Incoming, Completed or Failed.
    directory: Optional[str] = None
    updatedAt: str = field(default_factory=wire.iso8601)
    #: Set when the user cancels; the trainer polls it between iterations.
    cancelRequested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.stage in wire.JobStage.TERMINAL

    def to_json(self) -> Dict[str, Any]:
        return {
            "jobID": self.jobID,
            "scanID": self.scanID,
            "manifest": self.manifest.to_json(),
            "deviceID": self.deviceID,
            "appVersion": self.appVersion,
            "createdAt": self.createdAt,
            "stage": self.stage,
            "message": self.message,
            "fractionComplete": self.fractionComplete,
            "resultManifest": (
                self.resultManifest.to_json() if self.resultManifest else None
            ),
            "directory": self.directory,
            "updatedAt": self.updatedAt,
        }

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "Job":
        result_raw = raw.get("resultManifest")
        stage = str(raw.get("stage") or wire.JobStage.QUEUED)
        if stage not in wire.JobStage.ALL:
            stage = wire.JobStage.QUEUED
        return Job(
            jobID=str(raw["jobID"]),
            scanID=str(raw["scanID"]),
            manifest=wire.Manifest.from_json(raw["manifest"]),
            deviceID=str(raw.get("deviceID", "")),
            appVersion=str(raw.get("appVersion", "")),
            createdAt=str(raw.get("createdAt") or wire.iso8601()),
            stage=stage,
            message=str(raw.get("message") or ""),
            fractionComplete=raw.get("fractionComplete"),
            resultManifest=(
                wire.Manifest.from_json(result_raw)
                if isinstance(result_raw, dict)
                else None
            ),
            directory=raw.get("directory"),
            updatedAt=str(raw.get("updatedAt") or wire.iso8601()),
        )


class JobStore:
    """All jobs, their bytes, and the rules about both.

    Thread-safe. The aiohttp handlers, the training worker thread and the GUI
    all reach in here.
    """

    def __init__(
        self,
        save_root: SaveRoot,
        on_change: Optional[Callable[[Job], None]] = None,
    ) -> None:
        self.save_root = save_root
        self._lock = threading.RLock()
        self._by_job_id: Dict[str, Job] = {}
        self._by_scan_id: Dict[str, str] = {}
        self._on_change = on_change
        self.save_root.ensure()

    # -- start-up ---------------------------------------------------------

    def load_existing(self) -> int:
        """Re-adopt jobs left behind by a previous run.

        Without this, a Booster restart mid-upload would hand the phone a new
        ``jobID`` and a zeroed offset table, and a four-gigabyte transfer would
        start again from byte zero.
        """
        found = 0
        areas = [
            self.save_root.incoming,
            self.save_root.completed,
            self.save_root.failed,
        ]
        for area in areas:
            if not area.is_dir():
                continue
            for scan_dir in sorted(area.iterdir()):
                record = scan_dir / JOB_RECORD_NAME
                if not record.is_file():
                    continue
                try:
                    job = Job.from_json(json.loads(record.read_text(encoding="utf-8")))
                except (OSError, ValueError, KeyError):
                    # A corrupt record is not worth crashing over; the bytes
                    # are still there and the phone can re-POST the manifest,
                    # which rebuilds the record from scratch.
                    continue
                job.directory = str(scan_dir)
                # A job caught mid-flight by a restart is not still training.
                if job.stage in (wire.JobStage.TRAINING, wire.JobStage.EXPORTING):
                    job.stage = wire.JobStage.FAILED
                    job.message = (
                        "The Booster was closed while this scan was being "
                        "built. Send it again to start over."
                    )
                    job.fractionComplete = None
                with self._lock:
                    self._by_job_id[job.jobID] = job
                    self._by_scan_id[job.scanID] = job.jobID
                found += 1
        return found

    # -- lookup -----------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._by_job_id.get(job_id)

    def get_by_scan(self, scan_id: str) -> Optional[Job]:
        with self._lock:
            job_id = self._by_scan_id.get(scan_id)
            return self._by_job_id.get(job_id) if job_id else None

    def all_jobs(self) -> List[Job]:
        with self._lock:
            return sorted(self._by_job_id.values(), key=lambda j: j.createdAt)

    def queue_depth(self) -> int:
        """Jobs that still want CPU or GPU time, for ``/v1/info``."""
        with self._lock:
            return sum(1 for j in self._by_job_id.values() if not j.is_terminal)

    def job_dir(self, job: Job) -> Path:
        if job.directory:
            return Path(job.directory)
        return self.save_root.incoming_scan(job.scanID)

    # -- create / resume --------------------------------------------------

    def create_or_resume(
        self, body: wire.CreateJobBody, device_id: str
    ) -> wire.CreateJobResponse:
        """``POST /v1/jobs``. The whole resume mechanism lives here.

        For a known ``scanID`` this returns the SAME ``jobID`` and the real
        per-file byte counts from disk. Where the new manifest disagrees with
        the stored one about a file's size or hash, that file's progress is
        void and it is reported at offset 0, per section 7.2.
        """
        manifest = body.manifest
        with self._lock:
            existing = self.get_by_scan(manifest.scanID)
            if existing is not None and not self._is_restartable(existing):
                job = existing
                previous = job.manifest.by_path()
                job.manifest = manifest
                job.deviceID = device_id or job.deviceID
                job.appVersion = body.appVersion or job.appVersion
                if job.stage in (wire.JobStage.QUEUED, wire.JobStage.RECEIVING):
                    job.stage = wire.JobStage.RECEIVING
                    job.message = "Receiving your scan."
                offsets = self._offsets_locked(job, previous)
            else:
                # Brand new, or the previous attempt is terminal and the phone
                # is starting over. Either way the old bytes are not evidence.
                if existing is not None:
                    self._discard_locked(existing)
                job = Job(
                    jobID=str(uuid.uuid4()),
                    scanID=manifest.scanID,
                    manifest=manifest,
                    deviceID=device_id,
                    appVersion=body.appVersion,
                    createdAt=wire.iso8601(),
                    stage=wire.JobStage.RECEIVING,
                    message="Receiving your scan.",
                )
                directory = self.save_root.incoming_scan(job.scanID)
                directory.mkdir(parents=True, exist_ok=True)
                job.directory = str(directory)
                self._by_job_id[job.jobID] = job
                self._by_scan_id[job.scanID] = job.jobID
                offsets = self._offsets_locked(job, {})
            self._persist_locked(job)

        self._changed(job)
        return wire.CreateJobResponse(
            jobID=job.jobID,
            chunkSize=wire.CHUNK_SIZE,
            receivedByteOffsets=offsets,
        )

    def _is_restartable(self, job: Job) -> bool:
        """True when a re-POST for this scan should mint a fresh job."""
        return job.stage in (wire.JobStage.FAILED, wire.JobStage.CANCELLED)

    def _offsets_locked(
        self, job: Job, previous: Dict[str, wire.ManifestFile]
    ) -> Dict[str, int]:
        """Real bytes held per file, with changed files reset to zero."""
        directory = self.job_dir(job)
        offsets: Dict[str, int] = {}
        for entry in job.manifest.files:
            was = previous.get(entry.relativePath)
            if was is not None and (
                was.sha256 != entry.sha256 or was.byteCount != entry.byteCount
            ):
                # The file changed under us. Whatever is on disk is now
                # unrelated to what the phone intends to send.
                self._truncate_locked(directory, entry.relativePath)
                continue
            try:
                target = resolve_within(directory, entry.relativePath)
            except UnsafeRelativePath:
                continue
            if target.is_file():
                size = target.stat().st_size
                if size > entry.byteCount:
                    # More bytes than the manifest claims: a stale, longer
                    # version of the file. Start it over rather than serve a
                    # truncated hash mismatch at finalize.
                    self._truncate_locked(directory, entry.relativePath)
                    continue
                if size:
                    offsets[entry.relativePath] = size
        return offsets

    def _truncate_locked(self, directory: Path, relative: str) -> None:
        try:
            target = resolve_within(directory, relative)
        except UnsafeRelativePath:
            return
        try:
            if target.is_file():
                target.unlink()
        except OSError:
            pass

    def _discard_locked(self, job: Job) -> None:
        self._by_job_id.pop(job.jobID, None)
        if self._by_scan_id.get(job.scanID) == job.jobID:
            self._by_scan_id.pop(job.scanID, None)

    # -- chunk upload -----------------------------------------------------

    def write_chunk(
        self, job: Job, relative: str, offset: int, payload: bytes, chunk_sha: str
    ) -> int:
        """Append one chunk and return the authoritative total for that file.

        Raises :class:`ChunkChecksumMismatch` when the chunk's own hash does
        not match, which the caller turns into a 409.

        The offset rules, straight from section 7.3:

        * a chunk **behind** what we hold is discarded and answered with our
          current total, not appended;
        * a chunk **ahead** of what we hold is a bug on one side, so we answer
          with our current total and let the client rewind.

        Either way the answer is the truth about our disk, which is exactly
        what the client wants: it seeks its own read cursor to whatever we say.
        """
        expected = hashlib.sha256(payload).hexdigest()
        if chunk_sha and not _constant_time_equal(expected, chunk_sha.lower()):
            raise ChunkChecksumMismatch(
                "That piece of the file arrived damaged. It will be sent again."
            )

        entry = job.manifest.by_path().get(relative)
        if entry is None:
            raise UnknownManifestFile(
                "That file is not part of this scan's manifest: " + relative
            )

        directory = self.job_dir(job)
        target = resolve_within(directory, relative)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            current = target.stat().st_size if target.is_file() else 0
            if offset != current:
                # Out of order in either direction. Do not append; tell the
                # truth and let the client re-align.
                return current
            if current + len(payload) > entry.byteCount:
                # Would overrun the declared size. Refusing is better than
                # writing bytes the finalize hash is guaranteed to reject.
                return current
            with open(target, "r+b" if target.is_file() else "wb") as handle:
                handle.seek(current)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            total = current + len(payload)

        if job.stage == wire.JobStage.QUEUED:
            self.set_stage(job, wire.JobStage.RECEIVING, "Receiving your scan.")
        return total

    def received_bytes(self, job: Job) -> int:
        """Total bytes on disk across every manifest file."""
        directory = self.job_dir(job)
        total = 0
        for entry in job.manifest.files:
            try:
                target = resolve_within(directory, entry.relativePath)
            except UnsafeRelativePath:
                continue
            if target.is_file():
                total += min(target.stat().st_size, entry.byteCount)
        return total

    # -- finalize ---------------------------------------------------------

    def finalize(self, job: Job) -> wire.FinalizeResponse:
        """Verify every file's whole-file hash and completeness.

        ``reason`` is read by a non-technical person, verbatim, so it is a
        plain sentence naming what to do next. A rejected finalize deliberately
        leaves the job resumable: the phone re-POSTs the manifest and the
        offsets tell it exactly what to send again.
        """
        self.set_stage(job, wire.JobStage.VERIFYING, "Checking the scan arrived intact.")
        directory = self.job_dir(job)
        missing: List[str] = []
        short: List[str] = []
        corrupt: List[str] = []

        files = job.manifest.sorted_files()
        for index, entry in enumerate(files):
            if job.cancelRequested:
                return wire.FinalizeResponse(
                    accepted=False, reason="You cancelled this scan."
                )
            self.set_progress(
                job,
                wire.JobStage.VERIFYING,
                "Checking the scan arrived intact.",
                (index + 1) / max(1, len(files)),
            )
            try:
                target = resolve_within(directory, entry.relativePath)
            except UnsafeRelativePath:
                corrupt.append(entry.relativePath)
                continue
            if not target.is_file():
                missing.append(entry.relativePath)
                continue
            size = target.stat().st_size
            if size != entry.byteCount:
                short.append(entry.relativePath)
                continue
            if sha256_file(target) != entry.sha256:
                # Delete it: leaving a wrong-hash file on disk would make the
                # next resume report a full offset for a file that can never
                # verify, and the transfer would loop forever.
                try:
                    target.unlink()
                except OSError:
                    pass
                corrupt.append(entry.relativePath)

        if not missing and not short and not corrupt:
            self.set_stage(job, wire.JobStage.QUEUED, "Waiting to start.")
            return wire.FinalizeResponse(accepted=True)

        reason = _plain_finalize_reason(missing, short, corrupt)
        # Back to receiving, not failed: the job stays resumable.
        self.set_stage(job, wire.JobStage.RECEIVING, reason)
        return wire.FinalizeResponse(accepted=False, reason=reason)

    # -- cancel -----------------------------------------------------------

    def cancel(self, job: Job) -> None:
        """User pressed cancel. Stop training, move the bytes out of the way."""
        with self._lock:
            job.cancelRequested = True
        self.set_stage(job, wire.JobStage.CANCELLED, "You cancelled this scan.")
        # Give a training thread a moment to notice the flag before the
        # directory moves out from under it.
        time.sleep(0.05)
        try:
            moved = self.save_root.move_to(job.scanID, self.save_root.failed)
            job.directory = str(moved)
            self._persist(job)
        except OSError:
            pass

    # -- stage and progress ------------------------------------------------

    def set_stage(
        self, job: Job, stage: str, message: str, fraction: Optional[float] = None
    ) -> None:
        with self._lock:
            job.stage = stage
            job.message = message
            job.fractionComplete = fraction
            job.updatedAt = wire.iso8601()
            self._persist_locked(job)
        self._changed(job)

    def set_progress(
        self, job: Job, stage: str, message: str, fraction: Optional[float]
    ) -> None:
        """Update without re-writing the record every time.

        Progress fires many times a second during training; persisting each one
        would be thousands of pointless disk writes. The record is written on
        stage changes, which is what a restart actually needs.
        """
        with self._lock:
            job.stage = stage
            job.message = message
            job.fractionComplete = fraction
            job.updatedAt = wire.iso8601()
        self._changed(job)

    def set_result(self, job: Job, manifest: wire.Manifest) -> None:
        with self._lock:
            job.resultManifest = manifest
            self._persist_locked(job)
        self._changed(job)

    def status_response(self, job: Job) -> wire.JobStatusResponse:
        return wire.JobStatusResponse(
            jobID=job.jobID,
            stage=job.stage,
            message=job.message,
            fractionComplete=job.fractionComplete,
            resultManifest=job.resultManifest,
        )

    # -- internals ---------------------------------------------------------

    def _changed(self, job: Job) -> None:
        if self._on_change is not None:
            self._on_change(job)

    def _persist(self, job: Job) -> None:
        with self._lock:
            self._persist_locked(job)

    def _persist_locked(self, job: Job) -> None:
        directory = self.job_dir(job)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / JOB_RECORD_NAME
            temporary = directory / (JOB_RECORD_NAME + ".tmp")
            temporary.write_text(
                json.dumps(job.to_json(), indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(str(temporary), str(target))
        except OSError:
            # Losing the record costs a resume, not the bytes. Not worth
            # failing a live upload over.
            pass


class ChunkChecksumMismatch(Exception):
    """A chunk's ``X-Nimbus-Chunk-Sha256`` did not match its bytes -> 409."""


class UnknownManifestFile(Exception):
    """A PUT named a file the job's manifest does not contain -> 409."""


def _constant_time_equal(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    result = 0
    for a, b in zip(left, right):
        result |= ord(a) ^ ord(b)
    return result == 0


def _plain_finalize_reason(
    missing: List[str], short: List[str], corrupt: List[str]
) -> str:
    """Turn three lists of paths into one sentence a person can act on.

    Deliberately does not print the paths: "sensor_data/depth/frame_2026...
    .depth16" tells a non-technical owner nothing except that something broke.
    The count and the instruction are the useful parts.
    """
    total = len(missing) + len(short) + len(corrupt)
    if total == 0:
        return "The scan arrived correctly."
    if len(corrupt) == total:
        noun = "file" if total == 1 else "files"
        return (
            "{n} {noun} in the scan did not arrive correctly. "
            "Sending the scan again will fix it.".format(n=total, noun=noun)
        )
    if len(missing) == total:
        noun = "file" if total == 1 else "files"
        return (
            "{n} {noun} from the scan never arrived. "
            "Sending the scan again will fix it.".format(n=total, noun=noun)
        )
    noun = "part" if total == 1 else "parts"
    return (
        "{n} {noun} of the scan did not arrive correctly. "
        "Sending the scan again will fix it.".format(n=total, noun=noun)
    )
