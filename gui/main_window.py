# generated: 2025-05-18T11:35:00Z (auto)
# gui/main_window.py – PySide6 GUI (Spec‑aligned, full analysis pipeline)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from utils.config import load_config, exposure_entries, gain_entries
from core.analysis import extract_roi_stats, calculate_dynamic_range, calculate_snr_curve
from core.plotting import plot_snr_vs_signal, plot_snr_vs_exposure
from core.report_gen import save_summary_txt, report_csv, report_html


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")
        self.project_dir: Path | None = None
        self.config: Dict[str, Any] | None = None
        self._setup_ui()

    # ──────────────────────────────────────────── UI setup
    def _setup_ui(self):
        sel_btn = QPushButton("Select Project Folder")
        sel_btn.clicked.connect(self.select_project)
        run_btn = QPushButton("RUN")
        run_btn.clicked.connect(self.run_analysis)
        self.status = QLabel("Ready")

        lay = QVBoxLayout(); lay.addWidget(sel_btn); lay.addWidget(run_btn); lay.addWidget(self.status)
        container = QWidget(); container.setLayout(lay); self.setCentralWidget(container)

    # ──────────────────────────────────────────── Slots
    def select_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not dir_path:
            return
        self.project_dir = Path(dir_path)
        cfg_path = self.project_dir / "config.yaml"
        if not cfg_path.is_file():
            QMessageBox.critical(self, "Error", "config.yaml not found in project folder")
            return
        self.config = load_config(cfg_path)
        self.status.setText(f"Project loaded: {self.project_dir}")

    def run_analysis(self):
        if self.project_dir is None or self.config is None:
            QMessageBox.warning(self, "No Project", "Select a project folder first.")
            return
        # try:
        self._run_pipeline()
        self.status.setText("Done ✅")
        # except Exception as e:
        #     QMessageBox.critical(self, "Error", str(e))

    # ──────────────────────────────────────────── Core pipeline
    def _run_pipeline(self):
        cfg = self.config; project = self.project_dir

        # 1. ROI‑based statistics
        stats = extract_roi_stats(project, cfg)  # {(gain, ratio): {mean,std,snr}}
        if not stats:
            raise RuntimeError("No valid stacks found for analysis")

        # Flatten dict to lists keeping consistent order (ascending signal)
        tuples = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
        signals = np.array([kv[1]["mean"] for kv in tuples])
        noises  = np.array([kv[1]["std"]  for kv in tuples])
        snr_lin = signals / noises
        snr_db  = 20.0 * np.log10(snr_lin, where=noises!=0)
        ratios  = np.array([kv[0][1] for kv in tuples])

        # 2. Summary metrics
        dyn_range = calculate_dynamic_range(snr_lin, signals, cfg)
        summary = {"Dynamic Range (dB)": dyn_range}

        # 3. Output dir
        out_dir = project / cfg["output"].get("output_dir", "output")
        out_dir.mkdir(exist_ok=True)

        # 4. Save CSV & summary
        stats_rows = [
            {"gain_db": kv[0][0], "ratio": kv[0][1], **kv[1]} for kv in tuples
        ]
        report_csv(stats_rows, cfg, out_dir / "stats.csv")
        save_summary_txt(summary, cfg, out_dir / "summary.txt")

        # 5. Plots
        plot_snr_vs_signal(signals, snr_lin, cfg, out_dir / "snr_signal.png")
        plot_snr_vs_exposure(ratios, snr_lin, cfg, out_dir / "snr_exposure.png")

        # 6. HTML report
        graphs = {
            "snr_signal": out_dir / "snr_signal.png",
            "snr_exposure": out_dir / "snr_exposure.png",
        }
        report_html(summary, graphs, cfg, out_dir / "report.html")


# ──────────────────────────────────────────── Entrypoint
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show(); sys.exit(app.exec())
