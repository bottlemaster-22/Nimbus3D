"""The aiohttp application: every endpoint in ``docs/BOOSTER_PROTOCOL.md`` s5.

Plain HTTP/1.1, no TLS, bound to the LAN. The protocol document explains why at
length; the short version is that a self-signed certificate on a link between
two devices on one home Wi-Fi produces a padlock that means nothing and teaches
the user to click through certificate warnings. The real protection is that
nothing is honoured without a bearer token and the token is issued only after a
human read a code off this screen.

Things here that look odd and are deliberate:

* ``X-Nimbus-Api-Version`` is echoed on **every** response including errors,
  because the client reads it off a 426 to name the mismatch in plain language
  instead of saying "unknown".
* The files route is a catch-all wildcard. iOS's ``.urlPathAllowed`` does not
  escape ``/``, so ``sensor_data/depth/frame_x.depth16`` arrives as real path
  segments, not as one encoded blob.
* A ranged download never answers ``200``. The client reads a ``200`` as "that
  was the whole file" and stops asking for ranges.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from aiohttp import web

from ..config import BoosterConfig
from ..protocol import wire
from .jobs import ChunkChecksumMismatch, JobStore, UnknownManifestFile
from .pairing import PairingManager
from .paths import SaveRoot, UnsafeRelativePath, resolve_within, safe_relative_path
from .progress import HEARTBEAT_SECONDS, ProgressHub

log = logging.getLogger(__name__)

#: Refuse a chunk larger than this outright. The client sends 1 MiB windows;
#: 8 MiB is generous headroom for a future client, and it stops an unbounded
#: request body from being an out-of-memory button on an open LAN port.
MAX_CHUNK_BYTES = 8 * 1024 * 1024

#: Cap on a range response so one request cannot ask us to buffer a whole
#: multi-gigabyte file. The client asks for 1 MiB at a time.
MAX_RANGE_BYTES = 8 * 1024 * 1024

_JOB_KEY = "nimbus_job"


class BoosterServer:
    """Wires the store, the pairing manager and the progress hub into routes.

    The trainer is injected as a callable rather than imported, so the whole
    HTTP layer can be tested (and linted, and started) on a machine with no
    PyTorch and no CUDA. That is not a hypothetical: the server is useful on
    its own for receiving a scan, and the phone should get an honest "this PC
    cannot train yet" rather than a server that will not boot.
    """

    def __init__(
        self,
        config: BoosterConfig,
        save_root: Optional[SaveRoot] = None,
        on_job_change: Optional[Callable[[Any], None]] = None,
        on_pair_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.save_root = save_root or SaveRoot(config.save_root_path)
        self.progress = ProgressHub()
        self.pairing = PairingManager(config, on_change=on_pair_change)
        self._external_job_change = on_job_change
        self.jobs = JobStore(self.save_root, on_change=self._job_changed)
        #: Set by the runner once a trainer is available. Signature:
        #: ``submit(job) -> None``, returning immediately.
        self.submit_job: Optional[Callable[[Any], None]] = None
        #: Optional observer, set by the GUI. Called whenever a phone reads any
        #: part of a finished result, which is the only evidence this server
        #: has that a download is in progress: the window shows "sending your
        #: scan back" off this and nothing else, rather than guessing.
        self.on_result_read: Optional[Callable[[Any], None]] = None
        self._app: Optional[web.Application] = None

    # -- progress plumbing -------------------------------------------------

    def _job_changed(self, job: Any) -> None:
        """One place where a job state change becomes a wire event."""
        event = wire.ProgressEvent(
            stage=job.stage,
            message=job.message,
            fractionComplete=job.fractionComplete,
        )
        self.progress.publish(job.jobID, event)
        if job.is_terminal:
            self.progress.close_job(job.jobID)
        if self._external_job_change is not None:
            self._external_job_change(job)

    # -- application -------------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application(
            middlewares=[self._version_middleware, self._auth_middleware],
            # A house scan's largest single file is well under this; the real
            # per-chunk guard is MAX_CHUNK_BYTES in the handler.
            client_max_size=MAX_CHUNK_BYTES + 65536,
        )
        prefix = wire.PATH_PREFIX
        app.add_routes(
            [
                web.get(prefix + "/info", self.handle_info),
                web.post(prefix + "/pair/requests", self.handle_pair_create),
                web.post(
                    prefix + "/pair/requests/{request_id}/confirm",
                    self.handle_pair_confirm,
                ),
                web.get(prefix + "/pair/requests/{request_id}", self.handle_pair_status),
                web.post(prefix + "/jobs", self.handle_job_create),
                # Wildcard: .urlPathAllowed leaves "/" unescaped, so the
                # relative path arrives as real segments.
                web.put(
                    prefix + "/jobs/{job_id}/files/{rel:.*}", self.handle_chunk_upload
                ),
                web.post(prefix + "/jobs/{job_id}/finalize", self.handle_finalize),
                web.get(prefix + "/jobs/{job_id}/stream", self.handle_stream),
                web.get(
                    prefix + "/jobs/{job_id}/result/manifest",
                    self.handle_result_manifest,
                ),
                web.get(
                    prefix + "/jobs/{job_id}/result/files/{rel:.*}",
                    self.handle_result_file,
                ),
                web.put(
                    prefix + "/diagnostics/{scan_id}/{rel:.*}",
                    self.handle_diagnostics_upload,
                ),
                web.get(prefix + "/jobs/{job_id}", self.handle_job_status),
                web.delete(prefix + "/jobs/{job_id}", self.handle_job_cancel),
            ]
        )
        app.on_startup.append(self._on_startup)
        self._app = app
        return app

    async def _on_startup(self, _app: web.Application) -> None:
        self.progress.bind_loop(asyncio.get_running_loop())

    # -- middleware --------------------------------------------------------

    @web.middleware
    async def _version_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        """Echo our API version on every response, and gate on the client's.

        The echo has to happen even on a failure, which is why it wraps the
        handler in a try/finally-shaped block rather than decorating success.
        """
        client_version = request.headers.get(wire.HEADER_API_VERSION)
        if client_version is not None and client_version.strip() != wire.API_VERSION:
            # 426 with our own version in the header. Do not negotiate: a
            # wrong-version pair failing loudly is a fixable instruction, and
            # half-working is not.
            return self._error(
                web.HTTPUpgradeRequired,
                "This Booster speaks version {ours} of the connection and the "
                "phone speaks version {theirs}. Update both to the same "
                "version.".format(ours=wire.API_VERSION, theirs=client_version.strip()),
            )
        try:
            response = await handler(request)
        except web.HTTPException as http_error:
            http_error.headers[wire.HEADER_API_VERSION] = wire.API_VERSION
            raise
        response.headers[wire.HEADER_API_VERSION] = wire.API_VERSION
        return response

    @web.middleware
    async def _auth_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        """Bearer-token gate. ``/v1/info`` is the one endpoint outside it."""
        path = request.path
        open_paths = (
            wire.PATH_PREFIX + "/info",
            wire.PATH_PREFIX + "/pair/requests",
        )
        # Pairing is by definition unauthenticated: it is how a token is
        # obtained in the first place. Everything else needs one.
        if path in open_paths or path.startswith(wire.PATH_PREFIX + "/pair/requests/"):
            return await handler(request)

        token = _bearer_token(request)
        if not self.pairing.is_authorised(token):
            return self._error(
                web.HTTPUnauthorized, "This phone is not paired with this Booster."
            )
        return await handler(request)

    # -- helpers -----------------------------------------------------------

    def _error(self, kind: Any, message: str) -> web.Response:
        """A refusal, carrying our API version so the client can name it.

        Only 409 has a body the client reads; for everything else the body is
        ignored, so this is for our own logs and for a curl session.
        """
        return kind(
            text=json.dumps({"message": message}),
            content_type="application/json",
            headers={wire.HEADER_API_VERSION: wire.API_VERSION},
        )

    def _json(self, payload: Dict[str, Any], status: int = 200) -> web.Response:
        return web.json_response(
            payload,
            status=status,
            headers={wire.HEADER_API_VERSION: wire.API_VERSION},
            dumps=lambda obj: json.dumps(obj, sort_keys=True),
        )

    def _result_read(self, job: Any) -> None:
        """Tell the observer a phone is collecting this result. Never raises."""
        if self.on_result_read is None:
            return
        try:
            self.on_result_read(job)
        except Exception as error:  # noqa: BLE001 - a watcher must not break a download
            log.debug("on_result_read raised: %s", error)

    def _require_job(self, request: web.Request) -> Any:
        job = self.jobs.get(request.match_info.get("job_id", ""))
        if job is None:
            raise self._error(
                web.HTTPNotFound, "This Booster no longer recognises that scan."
            )
        return job

    # -- endpoints ---------------------------------------------------------

    async def handle_info(self, _request: web.Request) -> web.Response:
        """``GET /v1/info`` - the only endpoint served without a token."""
        info = wire.BoosterInfo(
            boosterID=self.config.boosterID,
            boosterName=self.config.boosterName,
            apiVersion=wire.API_VERSION,
            freeDiskBytes=self.save_root.free_disk_bytes(),
            queueDepth=self.jobs.queue_depth(),
            gpuName=_gpu_name(),
        )
        return self._json(info.to_json())

    async def handle_pair_create(self, request: web.Request) -> web.Response:
        """``POST /v1/pair/requests`` - the PC window starts showing a code."""
        try:
            body = wire.PairRequestBody.from_json(await request.json())
        except (ValueError, KeyError, TypeError):
            return self._error(web.HTTPBadRequest, "That pairing request made no sense.")
        pair_request = self.pairing.create_request(body)
        log.info(
            "Pairing code %s for %s (expires in %ss)",
            pair_request.code,
            body.deviceName,
            int(pair_request.seconds_remaining()),
        )
        return self._json(
            wire.PairRequestResponse(
                requestID=pair_request.requestID, awaitingCode=True
            ).to_json()
        )

    async def handle_pair_confirm(self, request: web.Request) -> web.Response:
        """``POST /v1/pair/requests/{id}/confirm`` - the typed code lands."""
        try:
            body = wire.PairConfirmBody.from_json(await request.json())
        except (ValueError, KeyError, TypeError):
            return self._error(web.HTTPBadRequest, "That confirmation made no sense.")
        request_id = request.match_info["request_id"]
        pair_request = self.pairing.confirm(request_id, body.code)
        if pair_request is None:
            return self._error(
                web.HTTPNotFound, "That pairing attempt has already been cleared."
            )
        return self._json(self.pairing.status_response(pair_request).to_json())

    async def handle_pair_status(self, request: web.Request) -> web.Response:
        """``GET /v1/pair/requests/{id}`` - polled while the code sheet is open."""
        pair_request = self.pairing.get_request(request.match_info["request_id"])
        if pair_request is None:
            return self._error(
                web.HTTPNotFound, "That pairing attempt has already been cleared."
            )
        return self._json(self.pairing.status_response(pair_request).to_json())

    async def handle_job_create(self, request: web.Request) -> web.Response:
        """``POST /v1/jobs`` - create, or resume by ``scanID``."""
        try:
            body = wire.CreateJobBody.from_json(await request.json())
        except (ValueError, KeyError, TypeError) as error:
            log.warning("Bad create-job body: %s", error)
            return self._error(web.HTTPBadRequest, "That scan description made no sense.")
        if not body.manifest.files:
            return self._error(web.HTTPBadRequest, "That scan contains no files.")

        device_id = self.pairing.device_for_token(_bearer_token(request) or "") or ""
        try:
            response = self.jobs.create_or_resume(body, device_id)
        except UnsafeRelativePath as error:
            log.warning("Refused a manifest with an unsafe path: %s", error)
            return self._error(
                web.HTTPBadRequest, "That scan contains a file name this PC cannot use."
            )
        return self._json(response.to_json())

    async def handle_chunk_upload(self, request: web.Request) -> web.Response:
        """``PUT /v1/jobs/{id}/files/{path}`` - one chunk, appended in order."""
        job = self._require_job(request)
        relative = request.match_info.get("rel", "")

        raw_offset = request.headers.get(wire.HEADER_CHUNK_OFFSET, "0")
        try:
            offset = int(raw_offset)
        except ValueError:
            return self._error(web.HTTPBadRequest, "That upload had a bad offset.")
        if offset < 0:
            return self._error(web.HTTPBadRequest, "That upload had a bad offset.")

        chunk_sha = request.headers.get(wire.HEADER_CHUNK_SHA256, "")
        payload = await request.read()
        if len(payload) > MAX_CHUNK_BYTES:
            return self._error(
                web.HTTPRequestEntityTooLarge, "That piece of the file was too big."
            )

        try:
            total = self.jobs.write_chunk(job, relative, offset, payload, chunk_sha)
        except ChunkChecksumMismatch as error:
            # 409 is the one status whose body the client shows verbatim.
            return self._json(
                wire.FinalizeResponse(accepted=False, reason=str(error)).to_json(),
                status=409,
            )
        except UnknownManifestFile as error:
            log.warning("%s", error)
            return self._json(
                wire.FinalizeResponse(
                    accepted=False,
                    reason="That file is not part of this scan. Send the scan again.",
                ).to_json(),
                status=409,
            )
        except UnsafeRelativePath as error:
            log.warning("Refused an unsafe upload path: %s", error)
            return self._error(web.HTTPBadRequest, "That file name cannot be used.")
        except OSError as error:
            log.error("Could not write a chunk: %s", error)
            return self._error(
                web.HTTPInternalServerError, "This PC could not save that file."
            )

        self._republish_receive_progress(job)
        return self._json(wire.ChunkUploadResponse(receivedByteOffset=total).to_json())

    def _republish_receive_progress(self, job: Any) -> None:
        """Turn bytes-on-disk into a fraction for the receiving stage.

        Computed from the manifest total rather than from a counter, so it is
        correct after a resume instead of restarting at zero.
        """
        total = job.manifest.total_byte_count
        if total <= 0:
            return
        received = self.jobs.received_bytes(job)
        fraction = min(1.0, received / float(total))
        self.jobs.set_progress(
            job, wire.JobStage.RECEIVING, "Receiving your scan.", fraction
        )

    async def handle_finalize(self, request: web.Request) -> web.Response:
        """``POST /v1/jobs/{id}/finalize`` - verify, then queue for training."""
        job = self._require_job(request)
        # Hashing a multi-gigabyte bundle is CPU work; keep the loop free so
        # progress sockets and other phones stay responsive.
        response = await asyncio.get_running_loop().run_in_executor(
            None, self.jobs.finalize, job
        )
        if not response.accepted:
            # 409 so the client reads `reason` and shows it verbatim.
            return self._json(response.to_json(), status=409)

        if self.submit_job is not None and self.config.autoTrain:
            self.submit_job(job)
        elif self.submit_job is None:
            self.jobs.set_stage(
                job,
                wire.JobStage.QUEUED,
                "Your scan is here and waiting. This PC cannot build it yet.",
            )
        return self._json(response.to_json())

    async def handle_diagnostics_upload(self, request: web.Request) -> web.Response:
        """``PUT /v1/diagnostics/{scan_id}/{path}`` - one whole file, for a human.

        DEVELOPMENT ONLY, and separate from the job routes on purpose.

        This exists so the phone can hand over a census, a point cloud or a log
        without anyone exporting it by hand, opening the Files app and moving it
        somewhere. It is NOT part of the boost protocol: nothing here creates a
        job, starts a train, or touches the manifest machinery. It writes a file
        into a folder and says how big it was.

        It is deliberately dumb. No chunking, no resume, no checksum, because
        the files are small enough to send in one request and a failed
        diagnostic upload costs a retry rather than a scan. The one thing it is
        NOT casual about is the path: both segments go through
        ``safe_relative_path`` and the result is confined with
        ``resolve_within``, because this is a route that writes attacker-named
        files to disk on a machine that is trusted by the person running it.

        Behind the same bearer-token gate as everything except ``/info`` and
        pairing, so an unpaired phone on the same network cannot use it.

        To remove the feature entirely: delete this method and its ``web.put``
        line above. Nothing else refers to either.
        """
        scan_id = request.match_info.get("scan_id", "")
        relative = request.match_info.get("rel", "")
        try:
            scan_id = safe_relative_path(scan_id)
            relative = safe_relative_path(relative)
        except UnsafeRelativePath as error:
            log.warning("Rejected a diagnostics path: %s", error)
            return self._error(web.HTTPBadRequest, "That file path is not allowed.")
        if "/" in scan_id:
            return self._error(web.HTTPBadRequest, "The scan id must be one segment.")

        base = self.save_root.root / "diagnostics"
        try:
            destination = resolve_within(base, scan_id + "/" + relative)
        except UnsafeRelativePath as error:
            log.warning("Rejected a diagnostics path: %s", error)
            return self._error(web.HTTPBadRequest, "That file path is not allowed.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        # Streamed rather than read() in one go: a model.ply is tens of
        # megabytes and this server also runs on machines with very little to
        # spare.
        with open(destination, "wb") as handle:
            while True:
                block = await request.content.readany()
                if not block:
                    break
                handle.write(block)
                written += len(block)

        log.info("Diagnostics: %s/%s (%d bytes)", scan_id, relative, written)
        return self._json({"ok": True, "bytes": written, "path": str(destination)})

    async def handle_job_status(self, request: web.Request) -> web.Response:
        """``GET /v1/jobs/{id}`` - the 3-second polling fallback."""
        job = self._require_job(request)
        return self._json(self.jobs.status_response(job).to_json())

    async def handle_job_cancel(self, request: web.Request) -> web.Response:
        """``DELETE /v1/jobs/{id}`` - any 2xx; the client ignores the body."""
        job = self._require_job(request)
        await asyncio.get_running_loop().run_in_executor(None, self.jobs.cancel, job)
        return self._json({"cancelled": True})

    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """``WS /v1/jobs/{id}/stream`` - live progress.

        Sends the latest known state immediately on connect (a phone that
        reconnects mid-training would otherwise stare at nothing), then one
        message per update, then a heartbeat every 10 seconds so the socket
        never looks dead, and closes after the terminal event.
        """
        job = self._require_job(request)
        socket = web.WebSocketResponse(heartbeat=30.0)
        await socket.prepare(request)

        queue = self.progress.subscribe(job.jobID)
        if queue is None:
            await socket.close()
            return socket

        try:
            latest = self.progress.latest(job.jobID) or wire.ProgressEvent(
                stage=job.stage,
                message=job.message,
                fractionComplete=job.fractionComplete,
            ).to_json()
            await socket.send_json(latest)
            if latest.get("stage") in wire.JobStage.TERMINAL:
                await socket.close()
                return socket

            while not socket.closed:
                try:
                    payload = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Nothing changed. Re-send the current state rather than a
                    # ping: the protocol asks for an update every 10 seconds,
                    # and a repeated message is the honest one.
                    current = self.jobs.get(job.jobID)
                    if current is None:
                        break
                    await socket.send_json(
                        wire.ProgressEvent(
                            stage=current.stage,
                            message=current.message,
                            fractionComplete=current.fractionComplete,
                        ).to_json()
                    )
                    if current.is_terminal:
                        break
                    continue

                if payload is None:
                    break
                await socket.send_json(payload)
                if payload.get("stage") in wire.JobStage.TERMINAL:
                    break
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.progress.unsubscribe(job.jobID, queue)
            if not socket.closed:
                await socket.close()
        return socket

    async def handle_result_manifest(self, request: web.Request) -> web.Response:
        """``GET /v1/jobs/{id}/result/manifest``.

        Same shape as the upload manifest, ``scanID`` included: it is mandatory
        in the Swift type, and a result manifest without it fails to decode.
        """
        job = self._require_job(request)
        if job.resultManifest is None:
            return self._error(
                web.HTTPNotFound, "This scan does not have a finished result yet."
            )
        self._result_read(job)
        return self._json(job.resultManifest.to_json())

    async def handle_result_file(self, request: web.Request) -> web.StreamResponse:
        """``GET /v1/jobs/{id}/result/files/{path}`` with Range support."""
        job = self._require_job(request)
        self._result_read(job)
        relative = request.match_info.get("rel", "")
        result_dir = self.save_root.result_dir(job.scanID)
        try:
            target = resolve_within(result_dir, relative)
        except UnsafeRelativePath as error:
            log.warning("Refused an unsafe result path: %s", error)
            return self._error(web.HTTPBadRequest, "That file name cannot be used.")
        if not target.is_file():
            return self._error(web.HTTPNotFound, "That part of the result is missing.")

        size = target.stat().st_size
        range_header = request.headers.get("Range")
        if not range_header:
            return await self._send_bytes(target, 0, size, size, partial=False)

        parsed = _parse_range(range_header, size)
        if parsed is None:
            return self._error(
                web.HTTPRequestRangeNotSatisfiable, "That part of the file does not exist."
            )
        start, end = parsed
        length = min(end - start + 1, MAX_RANGE_BYTES)
        # Never answer 200 to a ranged request unless the body really is the
        # whole file: the client reads a 200 as "that was everything" and
        # stops asking for more of this file.
        return await self._send_bytes(target, start, length, size, partial=True)

    async def _send_bytes(
        self, target: Path, start: int, length: int, size: int, partial: bool
    ) -> web.StreamResponse:
        headers = {
            wire.HEADER_API_VERSION: wire.API_VERSION,
            "Accept-Ranges": "bytes",
        }
        if partial:
            end = start + length - 1
            headers["Content-Range"] = "bytes {s}-{e}/{total}".format(
                s=start, e=end, total=size
            )
        response = web.StreamResponse(
            status=206 if partial else 200,
            headers=headers,
        )
        response.content_type = "application/octet-stream"
        response.content_length = length
        # aiohttp needs the request to prepare a StreamResponse; it is taken
        # from the task context by the caller's handler signature.
        return _FileBody(response, target, start, length)

    # -- lifecycle ---------------------------------------------------------

    def start_bonjour(self) -> Any:
        from .discovery import BonjourAdvertiser

        advertiser = BonjourAdvertiser(
            booster_id=self.config.boosterID,
            name=self.config.boosterName,
            port=self.config.port,
        )
        advertiser.start()
        return advertiser


class _FileBody(web.StreamResponse):
    """A ``StreamResponse`` that streams a slice of a file when prepared.

    aiohttp wants ``prepare()`` called with the live request before any body
    is written, and a handler can only return a response object, so the read
    loop is deferred into ``prepare`` rather than run in the handler.

    The attribute names carry a ``file_`` prefix for a reason that cost a
    debugging session: ``StreamResponse`` already has a ``_start`` METHOD, and
    an attribute of that name shadows it with an integer. The failure is a
    ``TypeError: 'int' object is not callable`` raised deep inside aiohttp
    while writing the response, long after this class is out of the picture.
    """

    def __init__(
        self, template: web.StreamResponse, path: Path, start: int, length: int
    ) -> None:
        super().__init__(status=template.status, headers=template.headers)
        self.content_type = "application/octet-stream"
        self.content_length = length
        self._file_path = path
        self._file_start = start
        self._file_length = length

    async def prepare(self, request: web.BaseRequest) -> Any:
        writer = await super().prepare(request)
        if writer is None:
            return None
        remaining = self._file_length
        with open(self._file_path, "rb") as handle:
            handle.seek(self._file_start)
            while remaining > 0:
                block = handle.read(min(65536, remaining))
                if not block:
                    break
                remaining -= len(block)
                await self.write(block)
        await self.write_eof()
        return writer


def _bearer_token(request: web.Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def _parse_range(header: str, size: int) -> Optional[Tuple[int, int]]:
    """Parse ``bytes=START-END``, including the open-ended ``bytes=START-``.

    The client sends an open-ended range when it does not know the remaining
    length, so handling it is not optional.
    """
    text = header.strip().lower()
    if not text.startswith("bytes="):
        return None
    spec = text[6:].split(",")[0].strip()
    if "-" not in spec:
        return None
    left, _, right = spec.partition("-")
    try:
        if not left:
            # A suffix range: the last N bytes.
            suffix = int(right)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            return (start, size - 1)
        start = int(left)
        end = int(right) if right else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return (start, min(end, size - 1))


def _gpu_name() -> Optional[str]:
    """Best-effort GPU name for ``/v1/info``, or ``None`` if unknown.

    Deliberately returns ``None`` rather than a guess: ``gpuName`` is
    ``String?`` on the client precisely so this can say "I do not know".
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - a driver problem is not our business here
        return None
    return None
