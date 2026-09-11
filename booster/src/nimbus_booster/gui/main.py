"""``nimbus-booster-gui``: open the window.

Qt is an optional dependency, so the very first thing this does is find out
whether it is there and, if not, say so in one sentence a person can act on
rather than a traceback about a missing module.
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

from .. import brand

_INSTALL_HINT = (
    "The {name} Booster window needs one extra piece that is not installed "
    "yet.\n\n"
    "Install it with:\n"
    "    pip install PySide6\n\n"
    "The Booster still works without the window. Run \"nimbus-booster serve\" "
    "in a terminal and it will listen for scans exactly the same way."
).format(name=brand.DISPLAY_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Open the window. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    try:
        brand.assert_consistent()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 3

    from ..config import BoosterConfig
    from .window import BoosterWindow

    arguments = list(sys.argv[:1]) + list(argv or [])
    application = QApplication(arguments)
    application.setApplicationName(brand.DISPLAY_NAME + " Booster")
    application.setOrganizationName(brand.DISPLAY_NAME)

    try:
        config = BoosterConfig.load()
    except OSError as error:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(
            None,
            brand.DISPLAY_NAME + " Booster",
            "The Booster's settings file could not be read or written: "
            "{err}".format(err=error),
        )
        return 2

    window = BoosterWindow(config)
    window.show()
    return int(application.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
