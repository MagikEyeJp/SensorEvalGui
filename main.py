#!/usr/bin/env python
from PySide6.QtWidgets import QApplication
import sys
import os
import logging
import faulthandler
import argparse
from pathlib import Path

from gui.main_window import MainWindow
from utils.logger import setup_logging


def main() -> None:
    setup_logging()

    disable_fh = False
    if "--no-faulthandler" in sys.argv:
        sys.argv.remove("--no-faulthandler")
        disable_fh = True
    if os.environ.get("NO_FAULTHANDLER"):
        disable_fh = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project folder to load", type=str)
    args, qt_args = parser.parse_known_args()

    if not disable_fh:
        try:
            faulthandler.enable()
        except Exception as exc:  # pragma: no cover - fail safe
            logging.debug("Failed to enable faulthandler: %s", exc)

    logging.info("Application started")
    app = QApplication([sys.argv[0]] + qt_args)
    win = MainWindow(Path(args.project) if args.project else None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
