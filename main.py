#!/usr/bin/env python
from PySide6.QtWidgets import QApplication
import sys
import logging

from gui.main_window import MainWindow
from utils.logger import setup_logging
import faulthandler


def main() -> None:
    faulthandler.enable()
    setup_logging()
    logging.info("Application started")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

