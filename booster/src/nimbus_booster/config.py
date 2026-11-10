"""Persisted Booster settings, including the stable ``boosterID``.

``docs/BOOSTER_PROTOCOL.md`` section 2 is blunt about the one thing that must
not change: **``boosterID`` must be stable across restarts.** After pairing,
the phone's identity for this PC *is* that id, so a Booster that regenerates it
looks like a brand-new unpaired device every time it boots. It is generated
once, written next to the config, and never regenerated.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import brand
from .server.paths import default_save_root


def config_dir() -> Path:
    """Where settings and the paired-device list live.

    Windows gets ``%APPDATA%``, everything else the XDG config home. Kept out
    of the save root on purpose: the user is invited to point the save root at
    an external drive, and unplugging it must not lose the pairing.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / brand.DOCUMENTS_FOLDER_NAME / "Booster"


def default_booster_name() -> str:
    """The name the user sees in the phone's device list before pairing.

    Named after the computer, not after the product: the phone shows the
    Bonjour instance name, and "Ollie's PC" is what makes it pickable out of a
    list. Falls back to something honest rather than to the product name.
    """
    try:
        host = socket.gethostname().split(".")[0].strip()
    except OSError:
        host = ""
    return host or "This PC"


@dataclass
class PairedDevice:
    """One phone that completed the code handshake."""

    deviceID: str
    deviceName: str
    #: SHA-256 of the bearer token, hex. The plaintext token is handed to the
    #: phone once and never stored: a stolen config file should not be a
    #: working credential.
    tokenHash: str
    pairedAt: str
    lastSeenAt: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "deviceID": self.deviceID,
            "deviceName": self.deviceName,
            "tokenHash": self.tokenHash,
            "pairedAt": self.pairedAt,
            "lastSeenAt": self.lastSeenAt,
        }

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "PairedDevice":
        return PairedDevice(
            deviceID=str(raw.get("deviceID", "")),
            deviceName=str(raw.get("deviceName", "")),
            tokenHash=str(raw.get("tokenHash", "")),
            pairedAt=str(raw.get("pairedAt", "")),
            lastSeenAt=raw.get("lastSeenAt"),
        )


@dataclass
class BoosterConfig:
    """Everything the Booster remembers between runs."""

    boosterID: str
    boosterName: str
    port: int
    saveRoot: str
    #: Bind address. ``0.0.0.0`` reaches every LAN interface; there is no
    #: authentication before pairing, so this is deliberately not the default
    #: for anything that leaves the LAN. See the note in server/app.py.
    bindHost: str = "0.0.0.0"
    autoStart: bool = True
    #: Train jobs automatically as they finish uploading, or wait for the user
    #: to press Start in the GUI.
    autoTrain: bool = True
    pairedDevices: List[PairedDevice] = field(default_factory=list)
    #: Trainer knobs the GUI exposes. Kept flat so a future field is additive.
    maxSplats: int = 2_000_000
    iterations: int = 30_000
    shDegree: int = 2
    renderLongEdge: int = 1600
    usePartitioner: bool = False

    # -- persistence -----------------------------------------------------

    @property
    def path(self) -> Path:
        return config_dir() / "config.json"

    def to_json(self) -> Dict[str, Any]:
        return {
            "boosterID": self.boosterID,
            "boosterName": self.boosterName,
            "port": int(self.port),
            "saveRoot": self.saveRoot,
            "bindHost": self.bindHost,
            "autoStart": bool(self.autoStart),
            "autoTrain": bool(self.autoTrain),
            "pairedDevices": [d.to_json() for d in self.pairedDevices],
            "maxSplats": int(self.maxSplats),
            "iterations": int(self.iterations),
            "shDegree": int(self.shDegree),
            "renderLongEdge": int(self.renderLongEdge),
            "usePartitioner": bool(self.usePartitioner),
        }

    def save(self) -> None:
        """Write atomically: a half-written config would lose the pairing."""
        directory = config_dir()
        directory.mkdir(parents=True, exist_ok=True)
        target = self.path
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(str(temporary), str(target))

    @staticmethod
    def load() -> "BoosterConfig":
        """Read the config, creating a fresh one on first run.

        A corrupt or unreadable file is replaced rather than crashed on, but
        the ``boosterID`` is preserved out of it if at all possible, because
        losing that un-pairs every phone.
        """
        path = config_dir() / "config.json"
        raw: Dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except (OSError, ValueError):
                raw = {}

        booster_id = str(raw.get("boosterID") or "").strip()
        if not booster_id:
            booster_id = str(uuid.uuid4())

        config = BoosterConfig(
            boosterID=booster_id,
            boosterName=str(raw.get("boosterName") or default_booster_name()),
            port=int(raw.get("port") or brand.BOOSTER_DEFAULT_PORT),
            saveRoot=str(raw.get("saveRoot") or default_save_root()),
            bindHost=str(raw.get("bindHost") or "0.0.0.0"),
            autoStart=bool(raw.get("autoStart", True)),
            autoTrain=bool(raw.get("autoTrain", True)),
            pairedDevices=[
                PairedDevice.from_json(d)
                for d in raw.get("pairedDevices", [])
                if isinstance(d, dict)
            ],
            maxSplats=int(raw.get("maxSplats") or 2_000_000),
            iterations=int(raw.get("iterations") or 30_000),
            shDegree=int(raw.get("shDegree", 2)),
            renderLongEdge=int(raw.get("renderLongEdge") or 1600),
            usePartitioner=bool(raw.get("usePartitioner", False)),
        )
        if not path.is_file():
            config.save()
        return config

    # -- convenience ------------------------------------------------------

    @property
    def save_root_path(self) -> Path:
        return Path(self.saveRoot)
