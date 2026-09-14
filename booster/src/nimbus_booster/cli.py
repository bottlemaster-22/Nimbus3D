"""``nimbus-booster``: the command line, and the headless way to run.

The window is optional. This is what a PC in a cupboard runs, and it is also
the fastest way to answer "is the Booster actually up" without opening
anything::

    nimbus-booster serve                 listen until Ctrl+C
    nimbus-booster serve --port 8761     just for this run
    nimbus-booster info                  what it would do, and what it can do
    nimbus-booster config --save-root D:\\Scans     remember a new folder
    nimbus-booster gui                   open the window instead

Nothing here imports Qt, torch or gsplat at module level, so ``--help`` works
on a machine with none of them installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, brand
from .config import BoosterConfig, config_dir
from .server.paths import SaveRoot, default_save_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nimbus-booster",
        description=(
            "The optional PC companion for {name}. It sits on your home "
            "network, waits for your phone to send a scan, builds it and sends "
            "the result back. Nothing leaves your network.".format(
                name=brand.DISPLAY_NAME
            )
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print everything, including each pairing code and every request.",
    )
    # Running with no subcommand at all means "serve", so the options that
    # subcommand reads have to exist on the top-level namespace too.
    parser.set_defaults(port=None, host=None, save_root=None, name=None, auto_train=None)
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser(
        "serve", help="Listen for scans until you press Ctrl+C. No window."
    )
    serve.add_argument("--port", type=int, default=None, help="TCP port to listen on.")
    serve.add_argument(
        "--host",
        default=None,
        help="Address to bind. The default reaches every network this PC is on.",
    )
    serve.add_argument(
        "--save-root",
        default=None,
        help="Folder to keep scans in, just for this run.",
    )

    subparsers.add_parser(
        "info", help="Show the folders, the address and whether this PC can build."
    )

    configure = subparsers.add_parser(
        "config", help="Show or change the settings that are remembered."
    )
    configure.add_argument("--save-root", default=None, help="Remember a new folder.")
    configure.add_argument("--port", type=int, default=None, help="Remember a new port.")
    configure.add_argument("--name", default=None, help="Rename this PC on the phone.")
    configure.add_argument(
        "--auto-train",
        choices=("on", "off"),
        default=None,
        help="Start building as soon as a scan finishes arriving.",
    )

    subparsers.add_parser("gui", help="Open the window instead of running headless.")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_info(config: BoosterConfig) -> None:
    from .server.discovery import local_ip_addresses
    from .trainer import pipeline as trainer_pipeline
    from .trainer import torch_support

    save_root = SaveRoot(config.save_root_path)
    save_root.ensure()
    print("{name} Booster {version}".format(name=brand.DISPLAY_NAME, version=__version__))
    print("  This PC appears on the phone as: " + config.boosterName)
    print("  Settings file: " + str(config.path))
    print("  Scans folder:  " + str(save_root.root))
    print("    Arriving:  " + str(save_root.incoming))
    print("    Finished:  " + str(save_root.completed))
    print("    Problems:  " + str(save_root.failed))
    free_gb = save_root.free_disk_bytes() / (1024.0 ** 3)
    print("  Free space:    {gb:.1f} GB".format(gb=free_gb))
    print("  Port:          {port}".format(port=config.port))
    addresses = local_ip_addresses()
    if addresses:
        print("  Reachable at:  " + ", ".join(
            "http://{ip}:{port}".format(ip=ip, port=config.port) for ip in addresses
        ))
    print("  Paired phones: {n}".format(n=len(config.pairedDevices)))
    for device in config.pairedDevices:
        print("    " + device.deviceName + "  (paired " + device.pairedAt + ")")
    capability = torch_support.probe()
    print("  Building:      " + trainer_pipeline.capability_sentence(capability))
    if trainer_pipeline.optimiser_available() and not capability.can_train:
        # Only worth telling someone to install PyTorch once there is
        # something here that would actually use it.
        for line in capability.detail.splitlines():
            print("    " + line)


def _apply_config_changes(config: BoosterConfig, args: argparse.Namespace) -> bool:
    changed = False
    if args.save_root:
        root = Path(args.save_root).expanduser()
        try:
            SaveRoot(root).ensure()
        except OSError as error:
            print(
                "That folder could not be used: {err}".format(err=error),
                file=sys.stderr,
            )
            return False
        config.saveRoot = str(root)
        changed = True
    if args.port:
        if not (1 <= int(args.port) <= 65535):
            print("A port has to be between 1 and 65535.", file=sys.stderr)
            return False
        config.port = int(args.port)
        changed = True
    if args.name:
        config.boosterName = str(args.name).strip() or config.boosterName
        changed = True
    if args.auto_train is not None:
        config.autoTrain = args.auto_train == "on"
        changed = True
    if changed:
        config.save()
        print("Saved to " + str(config.path))
    return True


def _serve(config: BoosterConfig, args: argparse.Namespace) -> int:
    from .server.runner import BoosterRunner

    if args.port:
        config.port = int(args.port)
    if args.host:
        config.bindHost = str(args.host)
    if args.save_root:
        config.saveRoot = str(Path(args.save_root).expanduser())

    try:
        runner = BoosterRunner(config, on_log=lambda message: print(message))
    except OSError as error:
        print(
            "The scans folder could not be opened: {err}".format(err=error),
            file=sys.stderr,
        )
        return 2

    print("Keeping scans in " + str(runner.save_root.root))
    for url in runner.urls():
        print("Reachable at " + url)
    print("Press Ctrl+C to stop.")
    try:
        runner.run_blocking()
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 2
    print("Stopped.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(bool(args.verbose))

    try:
        brand.assert_consistent()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    command = args.command or "serve"
    if command == "gui":
        from .gui.main import main as gui_main

        return gui_main([])

    try:
        config = BoosterConfig.load()
    except OSError as error:
        print(
            "The settings file in {directory} could not be read or written: "
            "{err}".format(directory=config_dir(), err=error),
            file=sys.stderr,
        )
        return 2

    if command == "info":
        _print_info(config)
        return 0
    if command == "config":
        if not _apply_config_changes(config, args):
            return 2
        _print_info(config)
        return 0
    if command == "serve":
        return _serve(config, args)

    parser.print_help()
    return 1


def default_scans_folder() -> Path:
    """Convenience for scripts: where scans go when nothing is configured."""
    return default_save_root()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
