#!/usr/bin/env python
from PySide6.QtWidgets import QApplication
import sys
import os
import logging
import faulthandler

from gui.main_window import MainWindow
from utils.logger import setup_logging
import faulthandler


def main() -> None:
    faulthandler.enable()
    setup_logging()

    disable_fh = False
    if "--no-faulthandler" in sys.argv:
        sys.argv.remove("--no-faulthandler")
        disable_fh = True
    if os.environ.get("NO_FAULTHANDLER"):
        disable_fh = True

    if not disable_fh:
        try:
            faulthandler.enable()
        except Exception as exc:  # pragma: no cover - fail safe
            logging.debug("Failed to enable faulthandler: %s", exc)

    logging.info("Application started")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

