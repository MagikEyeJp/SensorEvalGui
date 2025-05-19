# gui/main_window.py – config‑aware implementation

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from utils.config import load_config
from core.loader import load_image_stack
from core.analysis import extract_roi_stats
from core.plotting import plot_snr_vs_signal, plot_snr_vs_exposure
from core.report_gen import save_summary_txt, generate_html_report

import numpy as np


class MainWindow(QMainWindow):
    """Main GUI window for Sensor Evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")

        # ── runtime state ────────────────────────────────────────────────
        self.project_dir: Path | None = None
        self.config: Dict[str, Any] = {}

        # ── widgets ─────────────────────────────────────────────────────
        self.folder_label = QLabel("No project selected")
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)

        self.select_button = QPushButton("Select Project Folder")
        self.select_button.clicked.connect(self.select_project)

        self.run_button = QPushButton("RUN (re‑evaluate)")
        self.run_button.clicked.connect(self.run_evaluation)
        self.run_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ╭──────────────── callbacks ─────────────────╮

    def select_project(self) -> None:
        """User chooses project folder → load config → auto‑run."""
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not folder:
            return

        self.project_dir = Path(folder).resolve()
        self.folder_label.setText(str(self.project_dir))

        # Load config.yaml (fallback to repo default)
        try:
            self.config = load_config(self.project_dir)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        self.run_button.setEnabled(True)
        self.run_evaluation()

    def run_evaluation(self) -> None:
        """Run full evaluation pipeline on current project."""
        if self.project_dir is None:
            QMessageBox.warning(self, "No project", "Please select a project folder first.")
            return

        try:
            # 1. Load image stack(s)
            img_root = self.project_dir / self.config["image_structure"]["graychart_dir"]
            stack = load_image_stack(img_root, self.config)

            # 2. Analysis
            stats = extract_roi_stats(stack, self.config)

            # 3. Dummy plotting (replace with real data later)
            signal = np.linspace(100, 10000, 11)
            snr = signal / np.sqrt(signal)

            output_dir = self.project_dir / self.config["output"]["output_dir"]
            output_dir.mkdir(exist_ok=True)

            sig_png = output_dir / "snr_signal.png"
            exp_png = output_dir / "snr_exposure.png"

            plot_snr_vs_signal(signal, snr, self.config, sig_png)
            plot_snr_vs_exposure(np.array(self.config["exposure_ratios"]), snr, self.config, exp_png)

            if self.config["output"].get("generate_summary_txt", True):
                save_summary_txt(stats, output_dir / "summary.txt")
            if self.config["output"].get("generate_html_report", True):
                generate_html_report(stats, {"snr_signal": sig_png, "snr_exposure": exp_png}, output_dir / "report.html")

            self.output_display.setPlainText(f"Evaluation complete → {output_dir}")

        except Exception as exc:  # pragma: no cover
            self.output_display.setPlainText(f"Error: {exc}")


# ── entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())
