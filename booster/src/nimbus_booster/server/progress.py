"""Fan-out of progress events to WebSocket listeners.

``docs/BOOSTER_PROTOCOL.md`` section 8.2. Two requirements shape this module:

* **Send an update at least every 10 seconds even when nothing changed**, so
  the socket does not look dead. :data:`HEARTBEAT_SECONDS`.
* **Close the socket after the terminal event.** The client stops listening
  after ``ready``, ``failed`` or ``cancelled``; leaving the socket open just
  holds a file descriptor for nobody.

Events are produced on a worker thread (the trainer) and consumed on the
asyncio loop, so the hand-off goes through
``loop.call_soon_threadsafe``. Getting that wrong is the classic way to make a
progress bar that updates only when some unrelated request arrives.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Dict, List, Optional, Set

from ..protocol import wire

#: The client tolerates silence, but a non-technical user watching a still
#: screen does not. Ten seconds is the protocol's number.
HEARTBEAT_SECONDS = 10.0

#: Cap per job. A phone opens one socket; anything beyond a handful is a bug
#: or a leak, and an unbounded set is a memory leak with extra steps.
MAX_LISTENERS_PER_JOB = 8


class ProgressHub:
    """Per-job pub/sub for :class:`~nimbus_booster.protocol.wire.ProgressEvent`.

    Each listener gets its own bounded queue. A slow reader drops its oldest
    event rather than stalling the trainer: progress is a snapshot, so the
    newest one is the only one that matters, and back-pressure from a phone
    that went to sleep must never slow down a training loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queues: Dict[str, Set["asyncio.Queue[Optional[dict]]"]] = {}
        self._latest: Dict[str, dict] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the server's event loop so worker threads can reach it."""
        self._loop = loop

    # -- producing ---------------------------------------------------------

    def publish(self, job_id: str, event: wire.ProgressEvent) -> None:
        """Push one event to every listener on ``job_id``. Thread-safe."""
        payload = event.to_json()
        with self._lock:
            self._latest[job_id] = payload
            queues = list(self._queues.get(job_id, ()))
        if not queues:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in queues:
            loop.call_soon_threadsafe(self._offer, queue, payload)

    @staticmethod
    def _offer(queue: "asyncio.Queue[Optional[dict]]", payload: Optional[dict]) -> None:
        """Enqueue, dropping the oldest event if the reader is behind."""
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def latest(self, job_id: str) -> Optional[dict]:
        """The last event published for a job, sent to a socket on connect.

        Without this, a phone that reconnects mid-training sees nothing at all
        until the next update lands.
        """
        with self._lock:
            return self._latest.get(job_id)

    # -- consuming ---------------------------------------------------------

    def subscribe(self, job_id: str) -> Optional["asyncio.Queue[Optional[dict]]"]:
        """Open a listener queue, or ``None`` if this job already has too many."""
        queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue(maxsize=32)
        with self._lock:
            listeners = self._queues.setdefault(job_id, set())
            if len(listeners) >= MAX_LISTENERS_PER_JOB:
                return None
            listeners.add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: "asyncio.Queue[Optional[dict]]") -> None:
        with self._lock:
            listeners = self._queues.get(job_id)
            if listeners is None:
                return
            listeners.discard(queue)
            if not listeners:
                self._queues.pop(job_id, None)

    def close_job(self, job_id: str) -> None:
        """Wake every listener with a sentinel so the socket can close cleanly."""
        with self._lock:
            queues = list(self._queues.get(job_id, ()))
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in queues:
            loop.call_soon_threadsafe(self._offer, queue, None)

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._latest.pop(job_id, None)
            self._queues.pop(job_id, None)

    def listener_counts(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._queues.items()}

    def job_ids(self) -> List[str]:
        with self._lock:
            return list(self._latest.keys())
