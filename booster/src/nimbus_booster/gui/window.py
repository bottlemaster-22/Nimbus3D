"""The Booster window.

Five things, because five things are what the owner asked for and a sixth
would only be in the way:

1. what this PC is doing right now, in one line anybody can read;
2. the queue, one row per scan, with a real progress bar per row;
3. the folder scans are saved in, changeable, and remembered through
   :mod:`nimbus_booster.config`;
4. the pairing code, so the phone can find and trust this PC;
5. a log, for when something goes wrong and "it did not work" is not enough.

**The window never freezes while a scan is building.** The HTTP server has its
own thread and its own asyncio loop (see
:class:`nimbus_booster.server.runner.BoosterRunner`), the build queue has a
second thread, and Qt keeps the main thread to itself. The two sides do not
share an event loop and they never touch each other's objects directly:

* worker threads set a flag or emit a Qt signal, and a 400 ms
  :class:`~PySide6.QtCore.QTimer` on the GUI thread reads the current state out
  of the thread-safe job store. Progress updates arrive hundreds of times a
  second during a build, and turning each one into a queued signal would flood
  the Qt event loop with repaints, so they are coalesced instead;
* the slow, blocking operations a button can trigger (stopping the server,
  cancelling a job, changing the save folder) run on a short-lived thread and
  report back through a signal, so the click returns immediately.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import brand
from ..config import BoosterConfig
from ..protocol import wire
from ..server.paths import SaveRoot
from ..server.runner import BoosterRunner
from ..trainer import pipeline as trainer_pipeline
from . import state

log = logging.getLogger(__name__)

#: How often the window re-reads the world. Fast enough that a progress bar
#: looks alive, slow enough that a build is not competing with repaints.
REFRESH_MILLISECONDS = 400

#: Lines kept in the log pane. Older ones are dropped: this is a window, not
#: an archive, and an unbounded QPlainTextEdit is a slow memory leak.
MAX_LOG_LINES = 2000

_COLUMNS = ("Scan", "What is happening", "Progress", "Details")


class _LogHandler(logging.Handler):
    """Feeds the Python log into the window's log pane."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(record.getMessage())
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


class BoosterWindow(QMainWindow):
    """The whole window. One class, because it is one screen."""

    #: Emitted from any thread. Carries one line for the log pane.
    logLine = Signal(str)
    #: Emitted when a background start/stop/restart finishes.
    busyFinished = Signal(str)

    def __init__(self, config: BoosterConfig) -> None:
        super().__init__()
        self.config = config
        self.runner: Optional[BoosterRunner] = None
        self._row_for_job: dict = {}
        self._jobs_dirty = True
        self._busy = False
        self._last_result_read: Optional[_dt.datetime] = None
        self._shown_pair_request: Optional[str] = None

        self.setWindowTitle(brand.DISPLAY_NAME + " Booster")
        self.resize(940, 720)
        self._build_ui()

        self.logLine.connect(self._append_log)
        self.busyFinished.connect(self._on_busy_finished)

        self._log_handler = _LogHandler(self.logLine.emit)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger("nimbus_booster").addHandler(self._log_handler)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MILLISECONDS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        if self.config.autoStart:
            self.start_listening()
        else:
            self._refresh()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_status_box())
        layout.addWidget(self._build_pairing_box())
        layout.addWidget(self._build_queue_box(), 1)
        layout.addWidget(self._build_folder_box())
        layout.addWidget(self._build_log_box(), 1)
        self.setCentralWidget(central)

    def _build_status_box(self) -> QWidget:
        box = QGroupBox("This PC")
        outer = QVBoxLayout(box)

        self.status_headline = QLabel("Starting up")
        headline_font = QFont(self.status_headline.font())
        headline_font.setPointSize(headline_font.pointSize() + 6)
        headline_font.setBold(True)
        self.status_headline.setFont(headline_font)
        outer.addWidget(self.status_headline)

        self.status_detail = QLabel("")
        self.status_detail.setWordWrap(True)
        outer.addWidget(self.status_detail)

        self.address_label = QLabel("")
        self.address_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.address_label.setWordWrap(True)
        outer.addWidget(self.address_label)

        self.capability_label = QLabel("")
        self.capability_label.setWordWrap(True)
        outer.addWidget(self.capability_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start listening")
        self.start_button.clicked.connect(self.start_listening)
        buttons.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop listening")
        self.stop_button.clicked.connect(self.stop_listening)
        buttons.addWidget(self.stop_button)

        self.autostart_check = QCheckBox("Start listening when this window opens")
        self.autostart_check.setChecked(bool(self.config.autoStart))
        self.autostart_check.toggled.connect(self._on_autostart_toggled)
        buttons.addWidget(self.autostart_check)

        self.autotrain_check = QCheckBox("Build scans as soon as they arrive")
        self.autotrain_check.setChecked(bool(self.config.autoTrain))
        self.autotrain_check.toggled.connect(self._on_autotrain_toggled)
        buttons.addWidget(self.autotrain_check)

        buttons.addStretch(1)
        outer.addLayout(buttons)
        return box

    def _build_pairing_box(self) -> QWidget:
        box = QGroupBox("Phones")
        outer = QHBoxLayout(box)

        left = QVBoxLayout()
        self.pair_label = QLabel("No phone is waiting to pair.")
        self.pair_label.setWordWrap(True)
        left.addWidget(self.pair_label)

        self.pair_code = QLabel("")
        code_font = QFont("Consolas")
        code_font.setPointSize(code_font.pointSize() + 14)
        code_font.setBold(True)
        self.pair_code.setFont(code_font)
        self.pair_code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.pair_code)

        self.deny_button = QPushButton("Refuse this phone")
        self.deny_button.clicked.connect(self._deny_pairing)
        self.deny_button.setEnabled(False)
        left.addWidget(self.deny_button)
        left.addStretch(1)
        outer.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Phones allowed to send scans here:"))
        self.paired_list = QListWidget()
        right.addWidget(self.paired_list, 1)
        self.unpair_button = QPushButton("Remove the selected phone")
        self.unpair_button.clicked.connect(self._unpair_selected)
        right.addWidget(self.unpair_button)
        outer.addLayout(right, 1)
        return box

    def _build_queue_box(self) -> QWidget:
        box = QGroupBox("Scans")
        outer = QVBoxLayout(box)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 160)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.queue_summary = QLabel("No scans yet.")
        row.addWidget(self.queue_summary)
        row.addStretch(1)

        self.build_button = QPushButton("Build the selected scan now")
        self.build_button.clicked.connect(self._build_selected)
        row.addWidget(self.build_button)

        self.cancel_button = QPushButton("Cancel the selected scan")
        self.cancel_button.clicked.connect(self._cancel_selected)
        row.addWidget(self.cancel_button)

        self.reveal_button = QPushButton("Open the scan's folder")
        self.reveal_button.clicked.connect(self._reveal_selected)
        row.addWidget(self.reveal_button)
        outer.addLayout(row)
        return box

    def _build_folder_box(self) -> QWidget:
        box = QGroupBox("Where finished scans are saved")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(self.config.save_root_path))
        self.folder_edit.setReadOnly(True)
        row.addWidget(self.folder_edit, 1)

        self.change_folder_button = QPushButton("Change folder")
        self.change_folder_button.clicked.connect(self._choose_folder)
        row.addWidget(self.change_folder_button)

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self._open_folder)
        row.addWidget(self.open_folder_button)
        outer.addLayout(row)

        self.folder_detail = QLabel("")
        self.folder_detail.setWordWrap(True)
        outer.addWidget(self.folder_detail)
        return box

    def _build_log_box(self) -> QWidget:
        box = QGroupBox("Log")
        outer = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setFont(QFont("Consolas"))
        outer.addWidget(self.log_view)
        return box

    # -- logging -----------------------------------------------------------

    def _append_log(self, message: str) -> None:
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(stamp + "  " + message)

    # -- server lifecycle --------------------------------------------------

    def _make_runner(self) -> BoosterRunner:
        runner = BoosterRunner(
            self.config,
            on_job_change=self._on_job_change,
            on_pair_change=self._on_pair_change,
            on_log=self.logLine.emit,
        )
        runner.server.on_result_read = self._on_result_read
        return runner

    def start_listening(self) -> None:
        if self.runner is not None and self.runner.is_running:
            return
        self._set_busy(True, "Starting.")
        self._run_in_background(self._start_worker)

    def _start_worker(self) -> str:
        try:
            runner = self._make_runner()
        except OSError as error:
            return (
                "The scans folder could not be opened: {err}\n\nPick a "
                "different folder and try again.".format(err=error)
            )
        self.runner = runner
        if not runner.start():
            error = runner.start_error
            self.runner = None
            return str(error) if error else (
                "The Booster did not start, and did not say why. The log may "
                "have more."
            )
        return ""

    def stop_listening(self) -> None:
        if self.runner is None:
            return
        self._set_busy(True, "Stopping.")
        self._run_in_background(self._stop_worker)

    def _stop_worker(self) -> str:
        runner = self.runner
        self.runner = None
        if runner is not None:
            runner.stop()
        return ""

    def _restart(self) -> None:
        """Stop and start again, which is how a save folder change takes hold."""
        self._set_busy(True, "Applying the change.")

        def worker() -> str:
            self._stop_worker()
            return self._start_worker()

        self._run_in_background(worker)

    def _run_in_background(self, work: Callable[[], str]) -> None:
        def wrapped() -> None:
            try:
                message = work()
            except Exception as error:  # noqa: BLE001 - reported, never swallowed
                log.exception("A background action failed.")
                message = str(error)
            self.busyFinished.emit(message)

        threading.Thread(target=wrapped, daemon=True).start()

    def _set_busy(self, busy: bool, note: str = "") -> None:
        self._busy = busy
        for widget in (
            self.start_button,
            self.stop_button,
            self.change_folder_button,
            self.build_button,
            self.cancel_button,
        ):
            widget.setEnabled(not busy)
        if busy and note:
            self.status_headline.setText(note)

    def _on_busy_finished(self, message: str) -> None:
        self._set_busy(False)
        self._jobs_dirty = True
        if message:
            QMessageBox.warning(self, brand.DISPLAY_NAME + " Booster", message)
        self._refresh()

    # -- callbacks from worker threads -------------------------------------

    def _on_job_change(self, _job: object) -> None:
        # Called from the HTTP loop and from the build thread, very often.
        # Setting a flag is the whole handler on purpose: see the module note.
        self._jobs_dirty = True

    def _on_pair_change(self) -> None:
        self._jobs_dirty = True

    def _on_result_read(self, _job: object) -> None:
        self._last_result_read = _dt.datetime.now(_dt.timezone.utc)

    # -- refresh -----------------------------------------------------------

    def _job_rows(self) -> List[state.JobRow]:
        runner = self.runner
        if runner is None:
            return []
        rows = []
        for job in runner.server.jobs.all_jobs():
            rows.append(
                state.JobRow(
                    job_id=job.jobID,
                    scan_id=job.scanID,
                    stage=job.stage,
                    message=job.message,
                    fraction=job.fractionComplete,
                    is_terminal=job.is_terminal,
                )
            )
        rows.sort(key=lambda r: (r.is_terminal, r.scan_id))
        return rows

    def _refresh(self) -> None:
        if self._busy:
            return
        runner = self.runner
        running = runner is not None and runner.is_running
        rows = self._job_rows()
        pending = runner.server.pairing.pending_requests() if running else []

        key = state.status_key(
            running, rows, bool(pending), self._last_result_read
        )
        headline, detail = state.status_text(key)
        self.status_headline.setText(headline)
        self.status_detail.setText(detail)

        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

        self._refresh_addresses(running)
        self._refresh_pairing(pending)
        self._refresh_paired_list()
        self._refresh_table(rows)
        self._refresh_folder()
        self._jobs_dirty = False

    def _refresh_addresses(self, running: bool) -> None:
        runner = self.runner
        if not running or runner is None:
            self.address_label.setText("")
            self.capability_label.setText("")
            return
        addresses = ", ".join(runner.urls())
        line = (
            "Your phone should find \"{name}\" on its own. If it does not, "
            "type this in by hand: {addresses}".format(
                name=self.config.boosterName, addresses=addresses
            )
        )
        bonjour_problem = runner.bonjour_error
        if bonjour_problem:
            line = bonjour_problem + " " + line
        self.address_label.setText(line)

        capability = runner.queue.capability()
        self.capability_label.setText(
            trainer_pipeline.capability_sentence(capability)
        )

    def _refresh_pairing(self, pending: list) -> None:
        if not pending:
            self.pair_label.setText(
                "No phone is waiting to pair. On your phone, choose this PC "
                "and a code will appear here."
            )
            self.pair_code.setText("")
            self.deny_button.setEnabled(False)
            self._shown_pair_request = None
            return
        request = pending[0]
        self._shown_pair_request = request.requestID
        self.pair_label.setText(
            "{device} wants to send scans to this PC. Type this code into the "
            "phone within {seconds} seconds.".format(
                device=request.deviceName,
                seconds=int(request.seconds_remaining()),
            )
        )
        self.pair_code.setText(request.code)
        self.deny_button.setEnabled(True)

    def _refresh_paired_list(self) -> None:
        names = [
            device.deviceName or device.deviceID
            for device in self.config.pairedDevices
        ]
        current = [
            self.paired_list.item(i).text() for i in range(self.paired_list.count())
        ]
        if names == current:
            return
        self.paired_list.clear()
        for name in names:
            self.paired_list.addItem(name)
        self.unpair_button.setEnabled(bool(names))

    def _refresh_table(self, rows: List[state.JobRow]) -> None:
        wanted = [row.job_id for row in rows]
        if list(self._row_for_job.keys()) != wanted:
            self.table.setRowCount(0)
            self._row_for_job = {}
            for index, row in enumerate(rows):
                self.table.insertRow(index)
                self.table.setItem(index, 0, QTableWidgetItem(row.scan_id))
                self.table.setItem(index, 1, QTableWidgetItem(row.stage_text))
                bar = QProgressBar()
                bar.setTextVisible(True)
                self.table.setCellWidget(index, 2, bar)
                self.table.setItem(index, 3, QTableWidgetItem(row.message))
                self._row_for_job[row.job_id] = index

        for row in rows:
            index = self._row_for_job.get(row.job_id)
            if index is None:
                continue
            self.table.item(index, 1).setText(row.stage_text)
            self.table.item(index, 3).setText(row.message)
            bar = self.table.cellWidget(index, 2)
            if bar is None:
                continue
            percent = row.percent
            if row.stage == wire.JobStage.READY:
                bar.setRange(0, 100)
                bar.setValue(100)
            elif row.is_terminal:
                bar.setRange(0, 100)
                bar.setValue(0)
            elif percent is None:
                # No fraction means we genuinely do not know. A busy bar says
                # that; a bar frozen at zero would read as broken.
                bar.setRange(0, 0)
            else:
                bar.setRange(0, 100)
                bar.setValue(percent)

        self.queue_summary.setText(state.queue_summary(rows))

    def _refresh_folder(self) -> None:
        root = SaveRoot(self.config.save_root_path)
        self.folder_edit.setText(str(root.root))
        self.folder_detail.setText(
            "Finished scans go in {completed}. {free}".format(
                completed=str(root.completed),
                free=state.free_space_text(root.free_disk_bytes()),
            )
        )

    # -- actions -----------------------------------------------------------

    def _selected_job_id(self) -> Optional[str]:
        indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not indexes:
            return None
        row = indexes[0].row()
        for job_id, index in self._row_for_job.items():
            if index == row:
                return job_id
        return None

    def _selected_job(self):
        if self.runner is None:
            return None
        job_id = self._selected_job_id()
        return self.runner.server.jobs.get(job_id) if job_id else None

    def _build_selected(self) -> None:
        job = self._selected_job()
        if job is None or self.runner is None:
            QMessageBox.information(
                self,
                brand.DISPLAY_NAME + " Booster",
                "Pick a scan in the list first.",
            )
            return
        if job.stage != wire.JobStage.QUEUED:
            QMessageBox.information(
                self,
                brand.DISPLAY_NAME + " Booster",
                "That scan is not waiting to be built.",
            )
            return
        self.runner.queue.submit(job)
        self._jobs_dirty = True

    def _cancel_selected(self) -> None:
        job = self._selected_job()
        if job is None or self.runner is None:
            return
        answer = QMessageBox.question(
            self,
            brand.DISPLAY_NAME + " Booster",
            "Stop working on {scan} and move it out of the way?".format(
                scan=job.scanID
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        jobs = self.runner.server.jobs
        # cancel() sleeps and then moves a directory, so it does not happen on
        # the thread that draws the window.
        self._run_in_background(lambda: (jobs.cancel(job), "")[1])

    def _reveal_selected(self) -> None:
        job = self._selected_job()
        if job is None or self.runner is None:
            self._open_folder()
            return
        directory = self.runner.server.jobs.job_dir(job)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _deny_pairing(self) -> None:
        if self.runner is None or not self._shown_pair_request:
            return
        self.runner.server.pairing.deny(self._shown_pair_request)
        self._jobs_dirty = True
        self._refresh()

    def _unpair_selected(self) -> None:
        item = self.paired_list.currentItem()
        if item is None:
            return
        name = item.text()
        device = next(
            (
                d
                for d in self.config.pairedDevices
                if (d.deviceName or d.deviceID) == name
            ),
            None,
        )
        if device is None:
            return
        answer = QMessageBox.question(
            self,
            brand.DISPLAY_NAME + " Booster",
            "Stop letting {name} send scans to this PC? It will have to pair "
            "again.".format(name=name),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.runner is not None:
            self.runner.server.pairing.revoke(device.deviceID)
        else:
            self.config.pairedDevices = [
                d for d in self.config.pairedDevices if d.deviceID != device.deviceID
            ]
            self.config.save()
        self._refresh()

    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose where to keep your scans",
            str(self.config.save_root_path),
        )
        if not chosen:
            return
        root = Path(chosen)
        try:
            SaveRoot(root).ensure()
        except OSError as error:
            QMessageBox.warning(
                self,
                brand.DISPLAY_NAME + " Booster",
                "Scans cannot be saved there: {err}".format(err=error),
            )
            return
        self.config.saveRoot = str(root)
        self.config.save()
        self._append_log("Scans will now be saved in " + str(root) + ".")
        self._refresh_folder()
        if self.runner is not None:
            self._restart()

    def _open_folder(self) -> None:
        root = SaveRoot(self.config.save_root_path)
        try:
            root.ensure()
        except OSError as error:
            QMessageBox.warning(
                self,
                brand.DISPLAY_NAME + " Booster",
                "That folder could not be opened: {err}".format(err=error),
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root.root)))

    def _on_autostart_toggled(self, checked: bool) -> None:
        self.config.autoStart = bool(checked)
        self.config.save()

    def _on_autotrain_toggled(self, checked: bool) -> None:
        self.config.autoTrain = bool(checked)
        self.config.save()

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        logging.getLogger("nimbus_booster").removeHandler(self._log_handler)
        self._timer.stop()
        runner = self.runner
        self.runner = None
        if runner is not None:
            runner.stop()
        event.accept()
