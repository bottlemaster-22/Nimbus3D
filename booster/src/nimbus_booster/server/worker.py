"""The one background thread that turns a finished upload into a result.

One thread, not a pool. Two scans training at once on one GPU is slower than
two scans training one after the other and it doubles the peak memory, so the
queue is deliberately serial and the GUI shows the depth of it.

Where the bytes go, in order:

1. the upload lands in ``Incoming/<scanID>/`` and is verified there;
2. this worker builds it **in place** and writes ``Incoming/<scanID>/result/``;
3. on success the whole directory moves to ``Completed/<scanID>/``, which
   carries ``result/`` along with it, so the result endpoints find it at
   ``Completed/<scanID>/result/`` exactly as the protocol says;
4. on failure it moves to ``Failed/<scanID>/`` instead, still holding whatever
   was produced, because a failed scan the user can look at beats a deleted one.

Doing the move last is what makes the whole thing crash-safe: a job killed
mid-build is still in ``Incoming`` when the Booster restarts, and
:meth:`JobStore.load_existing` marks it failed and tells the user to send it
again rather than serving half a model.
"""

from __future__ import annotations

import logging
import queue as queue_module
import threading
from pathlib import Path
from typing import Callable, List, Optional

from ..protocol import wire
from ..trainer import pipeline as trainer_pipeline
from ..trainer import torch_support
from .jobs import Job, JobStore, sha256_file

log = logging.getLogger(__name__)

#: Files under ``result/`` that the phone is offered, in the order the manifest
#: lists them. Anything else the pipeline writes is kept on disk but not sent:
#: the manifest is a promise the client verifies file by file.
RESULT_FILES = ("model.json", "model.ply", "model.spz", "preview.png")


class TrainingQueue:
    """A serial build queue with a single worker thread.

    Safe to construct on a machine with no GPU: the capability probe runs once
    here and the answer becomes the job's message, so the phone is told
    "this PC cannot build scans yet, and here is why" instead of watching a bar
    that never moves.
    """

    def __init__(
        self,
        jobs: JobStore,
        options_factory: Callable[[], trainer_pipeline.PipelineOptions],
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._jobs = jobs
        self._options_factory = options_factory
        self._on_log = on_log
        self._queue: "queue_module.Queue[Optional[str]]" = queue_module.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._queued_ids: List[str] = []
        self._current_id: Optional[str] = None
        #: Probed once, lazily, because importing torch takes seconds.
        self._capability: Optional[torch_support.TrainerCapability] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="nimbus-booster-trainer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish. A build in progress is asked to cancel."""
        self._stop.set()
        self._queue.put(None)
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- queueing ----------------------------------------------------------

    def submit(self, job: Job) -> None:
        """Put a verified job in the queue. Returns immediately."""
        with self._lock:
            if job.jobID == self._current_id or job.jobID in self._queued_ids:
                return
            self._queued_ids.append(job.jobID)
            position = len(self._queued_ids)
        if self._current_id is None and position == 1:
            message = "Starting on your scan."
        elif position == 1:
            message = "Next in line. One scan is being built ahead of it."
        else:
            message = "Waiting in line, {n} scans ahead.".format(n=position - 1)
        self._jobs.set_stage(job, wire.JobStage.QUEUED, message)
        self._queue.put(job.jobID)

    def resume_queued(self) -> int:
        """Re-queue jobs that a restart left sitting in ``queued``."""
        count = 0
        for job in self._jobs.all_jobs():
            if job.stage == wire.JobStage.QUEUED and not job.resultManifest:
                self.submit(job)
                count += 1
        return count

    # -- introspection for the GUI -----------------------------------------

    def depth(self) -> int:
        with self._lock:
            return len(self._queued_ids) + (1 if self._current_id else 0)

    def current_job_id(self) -> Optional[str]:
        with self._lock:
            return self._current_id

    def capability(self, strict: bool = False) -> torch_support.TrainerCapability:
        """Probe once and remember it. Never raises."""
        if self._capability is None:
            self._capability = torch_support.probe(strict=strict)
        return self._capability

    # -- the thread --------------------------------------------------------

    def _log(self, message: str) -> None:
        log.info("%s", message)
        if self._on_log is not None:
            try:
                self._on_log(message)
            except Exception:  # noqa: BLE001 - a log sink must never kill a build
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                break
            with self._lock:
                if job_id in self._queued_ids:
                    self._queued_ids.remove(job_id)
                self._current_id = job_id
            try:
                job = self._jobs.get(job_id)
                if job is not None and not job.is_terminal:
                    self._build(job)
            except Exception:  # noqa: BLE001 - one bad scan must not end the queue
                log.exception("The build thread hit an unexpected problem.")
            finally:
                with self._lock:
                    self._current_id = None

    def _build(self, job: Job) -> None:
        scan_dir = self._jobs.job_dir(job)
        if not scan_dir.is_dir():
            self._jobs.set_stage(
                job,
                wire.JobStage.FAILED,
                "The files for this scan are no longer on this PC. Send it again.",
            )
            return

        def is_cancelled() -> bool:
            return bool(job.cancelRequested) or self._stop.is_set()

        def report(phase: str, message: str, fraction: Optional[float]) -> None:
            stage = (
                wire.JobStage.EXPORTING
                if phase == trainer_pipeline.PHASE_EXPORTING
                else wire.JobStage.TRAINING
            )
            self._jobs.set_progress(job, stage, message, fraction)

        self._jobs.set_stage(job, wire.JobStage.TRAINING, "Starting on your scan.")
        self._log("Building " + job.scanID + ".")
        result_dir = scan_dir / "result"
        try:
            result = trainer_pipeline.run_pipeline(
                scan_dir,
                result_dir,
                options=self._options_factory(),
                report=report,
                is_cancelled=is_cancelled,
            )
        except trainer_pipeline.PipelineCancelled:
            # JobStore.cancel already set the stage and moved the directory.
            self._log("Cancelled " + job.scanID + ".")
            return
        except trainer_pipeline.PipelineError as error:
            self._fail(job, str(error))
            return
        except MemoryError:
            self._fail(
                job,
                "This PC ran out of memory building that scan. A smaller splat "
                "limit in the Booster settings usually gets it through.",
            )
            return
        except Exception as error:  # noqa: BLE001
            log.exception("Building %s failed.", job.scanID)
            self._fail(
                job,
                "This PC could not finish building that scan. The Booster log "
                "has the details: {err}".format(err=error),
            )
            return

        if is_cancelled():
            self._log("Cancelled " + job.scanID + ".")
            return

        self._jobs.set_progress(
            job, wire.JobStage.EXPORTING, "Putting your scan somewhere safe.", 0.95
        )
        try:
            moved = self._jobs.save_root.move_to(
                job.scanID, self._jobs.save_root.completed
            )
            job.directory = str(moved)
        except OSError as error:
            log.warning("Could not move %s into Completed: %s", job.scanID, error)

        final_result_dir = self._jobs.save_root.result_dir(job.scanID)
        if not final_result_dir.is_dir():
            # The move did not happen, so the result is still where it was
            # written. Serve it from there rather than claim it is missing.
            final_result_dir = result_dir

        manifest = build_result_manifest(job.scanID, final_result_dir)
        if not manifest.files:
            self._fail(
                job,
                "The scan was built but nothing could be saved to send back. "
                "There may be no space left on this PC.",
            )
            return

        self._jobs.set_result(job, manifest)
        self._jobs.set_stage(job, wire.JobStage.READY, result.summary)
        self._log(
            "Finished {scan}: {n} splats.".format(
                scan=job.scanID, n=result.splat_count
            )
        )

    def _fail(self, job: Job, message: str) -> None:
        self._jobs.set_stage(job, wire.JobStage.FAILED, message)
        try:
            moved = self._jobs.save_root.move_to(job.scanID, self._jobs.save_root.failed)
            job.directory = str(moved)
        except OSError as error:
            log.warning("Could not move %s into Failed: %s", job.scanID, error)
        self._log("Failed " + job.scanID + ": " + message)


def build_result_manifest(scan_id: str, result_dir: Path) -> wire.Manifest:
    """Hash whatever the pipeline wrote and describe it for the phone.

    Only the files in :data:`RESULT_FILES` are offered, and only the ones that
    actually exist: the client verifies every entry's SHA-256 and refuses the
    download on a mismatch, so a manifest must never promise a file that is not
    there.
    """
    files = []
    for name in RESULT_FILES:
        path = Path(result_dir) / name
        if not path.is_file():
            continue
        try:
            files.append(
                wire.ManifestFile(
                    relativePath=name,
                    byteCount=int(path.stat().st_size),
                    sha256=sha256_file(path),
                )
            )
        except OSError as error:
            log.warning("Could not hash %s: %s", path, error)
    return wire.Manifest(scanID=scan_id, files=files)
