"""Starting and stopping the whole Booster, from the CLI or from the window.

There is one class here and it is the only thing that knows how the pieces fit
together: config, save folders, HTTP app, Bonjour, build queue.

**Threading.** The aiohttp server gets its own thread with its own asyncio
event loop. It does not share a loop with anything, and in particular the GUI
does not try to drive it: PySide6 runs its own Qt event loop on the main
thread, and the two talk through thread-safe queues and Qt signals rather than
through a shared loop. That choice is written down in MODULE_STATUS.md because
the alternative (running Qt against the asyncio loop with qasync) is the one
people assume was used.

Everything a caller needs from another thread is safe to touch:
:class:`~nimbus_booster.server.jobs.JobStore`,
:class:`~nimbus_booster.server.pairing.PairingManager` and
:class:`~nimbus_booster.server.progress.ProgressHub` all take locks, and the
progress hub hands events back to the loop with ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, List, Optional

from aiohttp import web

from .. import brand
from ..config import BoosterConfig
from ..trainer import pipeline as trainer_pipeline
from .app import BoosterServer
from .discovery import local_ip_addresses
from .paths import SaveRoot
from .worker import TrainingQueue

log = logging.getLogger(__name__)


class BoosterRunner:
    """One running Booster: HTTP, Bonjour and the build queue together."""

    def __init__(
        self,
        config: BoosterConfig,
        on_job_change: Optional[Callable[[Any], None]] = None,
        on_pair_change: Optional[Callable[[], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.save_root = SaveRoot(config.save_root_path)
        # Created here rather than lazily so a bad save root (a drive that is
        # not plugged in) fails at start-up with a message, not halfway
        # through someone's upload.
        self.save_root.ensure()
        self.server = BoosterServer(
            config,
            save_root=self.save_root,
            on_job_change=on_job_change,
            on_pair_change=on_pair_change,
        )
        self.queue = TrainingQueue(
            self.server.jobs, self._pipeline_options, on_log=on_log
        )
        self.server.submit_job = self.queue.submit
        self._on_log = on_log
        self._advertiser: Any = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._start_error: Optional[BaseException] = None

    # -- settings ----------------------------------------------------------

    def _pipeline_options(self) -> trainer_pipeline.PipelineOptions:
        """Read the knobs fresh each build, so a settings change takes effect."""
        return trainer_pipeline.PipelineOptions(
            max_splats=int(self.config.maxSplats),
            sh_degree=int(self.config.shDegree),
            iterations=int(self.config.iterations),
            render_long_edge=int(self.config.renderLongEdge),
        )

    # -- reporting ---------------------------------------------------------

    def _log(self, message: str) -> None:
        log.info("%s", message)
        if self._on_log is not None:
            try:
                self._on_log(message)
            except Exception:  # noqa: BLE001
                pass

    @property
    def is_running(self) -> bool:
        return self._ready.is_set() and not self._stopped.is_set()

    @property
    def start_error(self) -> Optional[BaseException]:
        return self._start_error

    def urls(self) -> List[str]:
        """Addresses to read out to someone holding the phone."""
        port = int(self.config.port)
        return [
            "http://{ip}:{port}".format(ip=ip, port=port)
            for ip in local_ip_addresses()
        ] or ["http://127.0.0.1:{port}".format(port=port)]

    @property
    def bonjour_error(self) -> Optional[str]:
        advertiser = self._advertiser
        return getattr(advertiser, "last_error", None) if advertiser else None

    # -- start / stop ------------------------------------------------------

    def start(self, timeout: float = 15.0) -> bool:
        """Start on a background thread. Returns True once it is listening.

        On failure :attr:`start_error` holds what went wrong, so the caller can
        put it in front of the user instead of a silent dead window.
        """
        if self._thread is not None:
            return self.is_running
        self._ready.clear()
        self._stopped.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._thread_main, name="nimbus-booster-server", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=timeout)
        return self.is_running

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:  # noqa: BLE001 - carried to the caller
            self._start_error = error
            log.exception("The Booster server stopped with an error.")
        finally:
            self._ready.set()
            self._stopped.set()

    def run_blocking(self) -> None:
        """Run the server on THIS thread until interrupted. Used headless."""
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass
        finally:
            self._stopped.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        adopted = self.server.jobs.load_existing()
        if adopted:
            self._log(
                "Picked up {n} scan(s) left from last time.".format(n=adopted)
            )

        app = self.server.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner
        site = web.TCPSite(runner, self.config.bindHost, int(self.config.port))
        try:
            await site.start()
        except OSError as error:
            await runner.cleanup()
            self._runner = None
            raise OSError(
                "Another program is already using port {port} on this PC, so "
                "the Booster could not start. Change the port in the Booster "
                "settings and try again.".format(port=self.config.port)
            ) from error

        self.queue.start()
        resumed = self.queue.resume_queued()
        if resumed:
            self._log("Restarted {n} scan(s) that were waiting.".format(n=resumed))

        try:
            # In an executor, NOT on this thread. zeroconf's register_service
            # is a blocking call that waits on zeroconf's own event loop, and
            # calling it from inside a running asyncio loop makes zeroconf
            # raise EventLoopBlocked and the Booster silently stops appearing
            # in the phone's list. Learned the hard way; do not inline it.
            self._advertiser = await self._loop.run_in_executor(
                None, self.server.start_bonjour
            )
        except Exception as error:  # noqa: BLE001 - mDNS is never fatal
            log.warning("Bonjour advertisement failed: %s", error)
            self._advertiser = None

        self._log(
            "{name} is listening on port {port}.".format(
                name=brand.DISPLAY_NAME, port=self.config.port
            )
        )
        self._ready.set()

        stop = asyncio.Event()
        self._stop_event = stop
        try:
            await stop.wait()
        finally:
            self.queue.stop()
            advertiser = self._advertiser
            self._advertiser = None
            if advertiser is not None:
                try:
                    # Off the loop thread for the same reason as registering.
                    await asyncio.get_running_loop().run_in_executor(
                        None, advertiser.stop
                    )
                except Exception as error:  # noqa: BLE001
                    log.debug("Stopping Bonjour failed: %s", error)
            await runner.cleanup()
            self._runner = None
            self._loop = None
            self._stop_event = None

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the server to shut down and wait for the thread to finish."""
        loop = self._loop
        event = self._stop_event
        if loop is not None and event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(event.set)
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._stopped.set()
        self._ready.clear()
