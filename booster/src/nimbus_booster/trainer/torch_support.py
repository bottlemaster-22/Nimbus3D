"""Is this PC able to train, and if not, what exactly is missing.

The Booster is useful without a GPU: it can receive a scan, verify it, and keep
it. So nothing in the server imports torch, and this module is the one place
that asks the question - in plain language, because the answer is shown to a
non-technical person in the GUI and sent to the phone as a job message.

:func:`probe` never raises and never imports gsplat's CUDA extension eagerly.
It is safe to call at start-up on a laptop with integrated graphics.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Optional

#: The trainer needs an NVIDIA GPU because gsplat's rasterisation kernels are
#: CUDA. That is a real constraint of the library, not a preference, and it is
#: named rather than hidden behind "training unavailable".
_INSTALL_HINT = (
    "Install the training extras with a CUDA build of PyTorch:\n"
    "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
    "  pip install gsplat\n"
    "PyTorch's own index is used because the correct wheel depends on the "
    "CUDA runtime installed on this PC, and PyPI's default wheel is CPU-only."
)


@dataclass
class TrainerCapability:
    """What this PC can actually do, and a sentence explaining why."""

    can_train: bool
    #: Shown to the user verbatim. One plain sentence, no stack trace.
    reason: str
    torch_version: Optional[str] = None
    gsplat_version: Optional[str] = None
    cuda_available: bool = False
    device_name: Optional[str] = None
    total_vram_bytes: int = 0
    #: The longer, technical follow-up. Shown under a disclosure triangle in
    #: the GUI and written to the log, not put in front of the user first.
    detail: str = ""

    @property
    def summary(self) -> str:
        if self.can_train:
            return "Ready to build scans on {gpu}.".format(
                gpu=self.device_name or "this PC's GPU"
            )
        return self.reason


def _module_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a broken install must not be fatal here
        return None
    return str(getattr(module, "__version__", "unknown"))


def probe(strict: bool = False) -> TrainerCapability:
    """Report training capability. Never raises.

    ``strict=False`` (the default) only checks that the packages import and
    that CUDA reports a device. ``strict=True`` additionally allocates a tiny
    tensor and calls into gsplat, which is the only way to catch a driver that
    is present but broken, or a gsplat built against a different CUDA than the
    one installed. The GUI uses strict once, at start-up, in the background.
    """
    if importlib.util.find_spec("torch") is None:
        return TrainerCapability(
            can_train=False,
            reason=(
                "This PC can receive scans but cannot build them yet: PyTorch "
                "is not installed."
            ),
            detail=_INSTALL_HINT,
        )
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
    except Exception as error:  # noqa: BLE001
        return TrainerCapability(
            can_train=False,
            reason=(
                "This PC can receive scans but cannot build them yet: PyTorch "
                "is installed but failed to load."
            ),
            detail="{err}\n\n{hint}".format(err=error, hint=_INSTALL_HINT),
        )

    torch_version = str(torch.__version__)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False

    device_name = None
    total_vram = 0
    if cuda_available:
        try:
            device_name = str(torch.cuda.get_device_name(0))
            total_vram = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:  # noqa: BLE001
            cuda_available = False

    if not cuda_available:
        return TrainerCapability(
            can_train=False,
            reason=(
                "This PC can receive scans but cannot build them: PyTorch "
                "cannot see an NVIDIA graphics card."
            ),
            torch_version=torch_version,
            detail=(
                "torch {v} reports no CUDA device. Either the installed "
                "PyTorch is the CPU-only build, or the graphics driver is not "
                "presenting a CUDA device.\n\n{hint}".format(
                    v=torch_version, hint=_INSTALL_HINT
                )
            ),
        )

    if importlib.util.find_spec("gsplat") is None:
        return TrainerCapability(
            can_train=False,
            reason=(
                "This PC can receive scans but cannot build them yet: the "
                "gsplat rasteriser is not installed."
            ),
            torch_version=torch_version,
            cuda_available=True,
            device_name=device_name,
            total_vram_bytes=total_vram,
            detail=_INSTALL_HINT,
        )
    gsplat_version = _module_version("gsplat")

    capability = TrainerCapability(
        can_train=True,
        reason="Ready.",
        torch_version=torch_version,
        gsplat_version=gsplat_version,
        cuda_available=True,
        device_name=device_name,
        total_vram_bytes=total_vram,
    )
    if not strict:
        return capability

    try:
        import torch as _torch  # noqa: PLC0415
        import gsplat as _gsplat  # noqa: PLC0415

        if not hasattr(_gsplat, "rasterization"):
            # Imported, but not the library we need. A gsplat without the
            # rasteriser is either a very old version or a name collision, and
            # both fail later in a much more confusing place than here.
            raise AttributeError(
                "the installed gsplat has no rasterization function"
            )
        probe_tensor = _torch.zeros(8, 3, device="cuda")
        _ = float(probe_tensor.sum().item())
    except Exception as error:  # noqa: BLE001
        return TrainerCapability(
            can_train=False,
            reason=(
                "This PC has a graphics card but the training software could "
                "not start on it."
            ),
            torch_version=torch_version,
            gsplat_version=gsplat_version,
            cuda_available=True,
            device_name=device_name,
            total_vram_bytes=total_vram,
            detail=(
                "torch {t} / gsplat {g} raised on first use: {err}\n"
                "This is usually a gsplat built against a different CUDA "
                "version than the installed driver. Reinstalling gsplat "
                "against the current torch normally fixes it.".format(
                    t=torch_version, g=gsplat_version, err=error
                )
            ),
        )
    return capability


def recommended_vram_budget(capability: TrainerCapability, headroom: float = 0.75) -> int:
    """Bytes of VRAM the trainer may plan to use.

    Never the whole card: the display driver, the desktop compositor and
    gsplat's own workspace all live there too, and a run that dies with an
    out-of-memory error two hours in has produced nothing. 75% is the default
    headroom, and the budget is a wall the densifier is not allowed to cross,
    not a target to grow into.
    """
    if not capability.can_train or capability.total_vram_bytes <= 0:
        return 0
    return int(capability.total_vram_bytes * max(0.1, min(0.95, headroom)))
