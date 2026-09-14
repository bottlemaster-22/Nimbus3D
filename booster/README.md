# The PC Booster

Optional. The phone app works on its own; this is for when a scan is big
enough that you would rather your PC did the heavy part.

It sits on your home network and waits. The phone finds it, you pair once by
reading a six digit code off the PC screen, and after that the phone can send
a scan over, watch it build, and pull the result back. Nothing leaves your
network. There is no account, no cloud and no outbound connection of any kind.

## Install

Three pieces, split on purpose so you only install what you need.

```
pip install -e .            # the listener. Small, no GPU needed.
pip install -e .[gui]       # adds the window (PySide6).
pip install -e .[train]     # adds PyTorch and gsplat. Large, needs an NVIDIA card.
```

PyTorch is not pinned here because the correct build depends on the CUDA
runtime on your PC, and the default wheel on PyPI is CPU only:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat
```

The listener starts and works without any of that. It will receive and keep a
scan on a laptop with no graphics card, and tell the phone plainly that this PC
cannot build it yet.

## Run

```
nimbus-booster-gui          the window
nimbus-booster serve        the same thing with no window, until Ctrl+C
nimbus-booster info         what it would do, and what this PC can do
nimbus-booster config --save-root D:\Scans      remember a different folder
```

## Where things go

```
Documents/Nimbus3D/Scans/
  Incoming/<scanID>/     a scan that is still arriving, or still building
  Completed/<scanID>/    finished, with result/ holding what the phone collects
  Failed/<scanID>/       kept, not deleted, so you can see what happened
```

The folder is changeable in the window (Change folder) or from the command
line, and the choice is remembered. Only the three folder names inside it are
fixed, because those are part of the file format rather than part of the name.

Settings and the paired phone list live somewhere else, in `%APPDATA%` on
Windows, so pointing the scans folder at an external drive and then unplugging
it does not un-pair your phone.

## What actually happens to a scan

1. It arrives file by file, in one megabyte pieces, each one checksummed. An
   interrupted transfer picks up where it stopped rather than starting again.
2. Every file's whole checksum is verified before anything is built.
3. The build runs: the depth measurements are turned into Gaussian splats, the
   empty space the LiDAR saw through is cleared out, and the result is written
   as `model.ply`, `model.spz`, `model.json` and a `preview.png`.
4. The finished folder moves to `Completed/`, the phone is told, and it
   collects the result.

One honest caveat, which is also in `MODULE_STATUS.md`: the step that refines
the splats against your photos is not written yet. What comes back today is
built from the depth sensor and coloured from the photos, which is real and
correct but softer than the finished product will be.
