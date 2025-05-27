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
    QProgressBar,
)
from PySide6.QtCore import QThread, Signal

from utils.config import load_config
import utils.config as cfgutil
from core.analysis import (
    extract_roi_stats,
    calculate_dark_noise,
    calculate_dark_noise_gain,
    calculate_dn_sat,
    calculate_dynamic_range_dn,
    calculate_pseudo_prnu,
)
from core.plotting import (
    plot_snr_vs_signal,
    plot_snr_vs_exposure,
    plot_prnu_regression,
    plot_heatmap,
)
from core.report_gen import save_summary_txt, report_csv, report_html
from core.loader import load_image_stack


class EvalWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, project: Path, cfg: Dict[str, Any]):
        super().__init__()
        self.project = project
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_pipeline(self.project, self.cfg)
            self.finished.emit(summary)
            self.progress.emit(100)
        except Exception as e:  # pragma: no cover - UI error path
            self.error.emit(str(e))


def run_pipeline(project: Path, cfg: Dict[str, Any]) -> Dict[str, float]:
    # 1. ROI-based statistics
    stats = extract_roi_stats(project, cfg)
    if not stats:
        raise RuntimeError("No valid stacks found for analysis")

    tuples = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
    signals = np.array([kv[1]["mean"] for kv in tuples])
    noises = np.array([kv[1]["std"] for kv in tuples])
    snr_lin = signals / noises
    ratios = np.array([kv[0][1] for kv in tuples])

    # 2. Dark/flat metrics per gain (first gain only for simplicity)
    gain_db = cfgutil.gain_entries(cfg)[0][0]
    dsnu, read_noise, dsnu_map, rn_map = calculate_dark_noise_gain(project, gain_db, cfg)
    flat_folder = cfgutil.find_gain_folder(project, gain_db, cfg) / cfg["measurement"].get("flat_lens_folder", "flat")
    flat_stack = load_image_stack(flat_folder)
    dn_sat = calculate_dn_sat(flat_stack, cfg)
    prnu, prnu_map = calculate_pseudo_prnu(flat_stack, cfg)

    dyn_range = calculate_dynamic_range_dn(dn_sat, read_noise)

    out_dir = project / cfg["output"].get("output_dir", "output")
    out_dir.mkdir(exist_ok=True)

    stats_rows = [{"gain_db": kv[0][0], "ratio": kv[0][1], **kv[1]} for kv in tuples]
    report_csv(stats_rows, cfg, out_dir / "stats.csv")
    summary = {
        "Dynamic Range (dB)": dyn_range,
        "DSNU (DN)": dsnu,
        "Read Noise (DN)": read_noise,
        "DN_sat": dn_sat,
        "PRNU (%)": prnu,
    }
    save_summary_txt(summary, cfg, out_dir / "summary.txt")

    plot_snr_vs_signal(signals, snr_lin, cfg, out_dir / "snr_signal.png")
    plot_snr_vs_exposure(ratios, snr_lin, cfg, out_dir / "snr_exposure.png")
    plot_prnu_regression(signals, noises, out_dir / "prnu_fit.png")
    plot_heatmap(dsnu_map, "DSNU map", out_dir / "dsnu_map.png")
    plot_heatmap(rn_map, "Read noise map", out_dir / "readnoise_map.png")
    plot_heatmap(prnu_map, "PRNU residual", out_dir / "prnu_residual_map.png")

    graphs = {
        "snr_signal": out_dir / "snr_signal.png",
        "snr_exposure": out_dir / "snr_exposure.png",
        "prnu_fit": out_dir / "prnu_fit.png",
        "dsnu_map": out_dir / "dsnu_map.png",
        "readnoise_map": out_dir / "readnoise_map.png",
        "prnu_residual_map": out_dir / "prnu_residual_map.png",
    }
    report_html(summary, graphs, cfg, out_dir / "report.html")
    return summary


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")
        self.project_dir: Path | None = None
        self.config: Dict[str, Any] | None = None
        self.worker: EvalWorker | None = None
        self._setup_ui()

    # ──────────────────────────────────────────── UI setup
    def _setup_ui(self):
        sel_btn = QPushButton("Select Project Folder")
        sel_btn.clicked.connect(self.select_project)
        run_btn = QPushButton("RUN")
        run_btn.clicked.connect(self.run_analysis)
        self.status = QLabel("Ready")
        self.progress = QProgressBar()

        lay = QVBoxLayout()
        lay.addWidget(sel_btn)
        lay.addWidget(run_btn)
        lay.addWidget(self.status)
        lay.addWidget(self.progress)
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
        self.run_analysis()

    def run_analysis(self):
        if self.project_dir is None or self.config is None:
            QMessageBox.warning(self, "No Project", "Select a project folder first.")
            return
        if self.worker is not None:
            return
        self.status.setText("Running...")
        self.worker = EvalWorker(self.project_dir, self.config)
        self.worker.finished.connect(self._analysis_done)
        self.worker.error.connect(self._analysis_error)
        self.worker.progress.connect(self.progress.setValue)
        self.progress.setValue(0)
        self.worker.start()

    # ──────────────────────────────────────────── Core pipeline
    def _analysis_done(self, summary: Dict[str, float]):
        self.status.setText("Done ✅")
        self.progress.setValue(100)
        self.worker = None

    def _analysis_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.status.setText("Error")
        self.progress.setValue(0)
        self.worker = None


# ──────────────────────────────────────────── Entrypoint
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show(); sys.exit(app.exec())
