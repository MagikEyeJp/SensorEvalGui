# generated: 2025-05-18T11:35:00Z (auto)
# gui/main_window.py – PySide6 GUI (Spec‑aligned, full analysis pipeline)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any
import logging
import threading

import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QScrollArea,
    QTabWidget,
    QSplitter,
    QSizePolicy,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QThread, Signal, Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
import matplotlib.pyplot as plt

from utils.config import load_config
import utils.config as cfgutil
from utils.logger import log_memory_usage

# ensure only one pipeline runs at a time
pipeline_lock = threading.Lock()
from core.analysis import (
    extract_roi_stats,
    extract_roi_table,
    calculate_dark_noise,
    calculate_dark_noise_gain,
    calculate_dn_sat,
    calculate_dynamic_range_dn,
    calculate_system_sensitivity,
    collect_mid_roi_snr,
    calculate_dn_at_snr,
    calculate_snr_at_half,
    calculate_dn_at_snr_one,
    calculate_pseudo_prnu,
)
from core.plotting import (
    plot_snr_vs_signal,
    plot_snr_vs_exposure,
    plot_prnu_regression,
    plot_heatmap,
)
from utils.roi import load_rois
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
            logging.info("Evaluation worker started")
            log_memory_usage("before pipeline: ")
            summary = run_pipeline(self.project, self.cfg)
            self.finished.emit(summary)
            self.progress.emit(100)
            logging.info("Evaluation worker finished")
            log_memory_usage("after pipeline: ")
        except Exception as e:  # pragma: no cover - UI error path
            logging.exception("Worker failed: %s", e)
            self.error.emit(str(e))


def run_pipeline(project: Path, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Run full analysis pipeline with logging and memory checks."""
    logging.info("Pipeline start: %s", project)
    with pipeline_lock:
        try:
            log_memory_usage("start: ")

            # 1. ROI-based statistics
            stats = extract_roi_stats(project, cfg)
            if not stats:
                raise RuntimeError("No valid stacks found for analysis")

            log_memory_usage("after roi stats: ")
            tuples = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
            signals = np.array([kv[1]["mean"] for kv in tuples])
            noises = np.array([kv[1]["std"] for kv in tuples])
            snr_lin = signals / noises
            ratios = np.array([kv[0][1] for kv in tuples])

            roi_table = extract_roi_table(project, cfg)
            flat_roi_file = project / cfg["measurement"].get("flat_roi_file")
            flat_rects = load_rois(flat_roi_file)

            # 2. Dark/flat metrics per gain (use first gain for maps)
            debug_stacks = cfg["output"].get("debug_stacks", False)
            out_dir = project / cfg["output"].get("output_dir", "output")
            out_dir.mkdir(exist_ok=True)

            dsnu_list = []
            rn_list = []
            prnu_list = []
            sens_list = []
            first = True
            for gain_db, _ in cfgutil.gain_entries(cfg):
                logging.info("Processing gain %.1f dB", gain_db)
                dsnu, rn, dsnu_map_tmp, rn_map_tmp = calculate_dark_noise_gain(project, gain_db, cfg)
                dsnu_list.append(dsnu)
                rn_list.append(rn)
                flat_folder = cfgutil.find_gain_folder(project, gain_db, cfg) / cfg["measurement"].get("flat_lens_folder", "flat")
                flat_stack = load_image_stack(flat_folder)
                if debug_stacks and first:
                    dark_folder = cfgutil.find_gain_folder(project, gain_db, cfg) / cfg["measurement"].get("dark_folder", "dark")
                    dark_stack = load_image_stack(dark_folder)
                    tifffile.imwrite(out_dir / f"dark_cache_{int(gain_db)}dB.tiff", dark_stack)
                    tifffile.imwrite(out_dir / f"flat_cache_{int(gain_db)}dB.tiff", flat_stack)
                prnu, prnu_map_tmp = calculate_pseudo_prnu(flat_stack, cfg, flat_rects)
                prnu_list.append(prnu)
                sens_list.append(calculate_system_sensitivity(flat_stack, cfg, flat_rects))
                if first:
                    dsnu_map, rn_map, prnu_map = dsnu_map_tmp, rn_map_tmp, prnu_map_tmp
                    dn_sat = calculate_dn_sat(flat_stack, cfg)
                    first = False
                log_memory_usage(f"after gain {gain_db}: ")

            dsnu = float(np.mean(dsnu_list)) if dsnu_list else 0.0
            read_noise = float(np.mean(rn_list)) if rn_list else 0.0
            dyn_range = calculate_dynamic_range_dn(dn_sat, read_noise)
            prnu = float(np.mean(prnu_list)) if prnu_list else float('nan')
            system_sens = float(np.mean(sens_list)) if sens_list else float('nan')
            dn_at_10 = calculate_dn_at_snr(signals, snr_lin, cfg["processing"].get("snr_threshold_dB", 10.0))
            snr_at_50 = calculate_snr_at_half(signals, snr_lin, dn_sat)
            dn_at_0 = calculate_dn_at_snr_one(signals, snr_lin)

            log_memory_usage("after metrics: ")

            stats_rows = roi_table
            report_csv(stats_rows, cfg, out_dir / "roi_stats.csv")
            summary = {
                "Dynamic Range (dB)": dyn_range,
                "DSNU (DN)": dsnu,
                "Read Noise (DN)": read_noise,
                "DN_sat": dn_sat,
                "Pseudo PRNU (%)": prnu,
                "System Sensitivity": system_sens,
                "DN @ 10 dB": dn_at_10,
                "SNR @ 50% (dB)": snr_at_50,
                "DN @ 0 dB": dn_at_0,
            }
            save_summary_txt(summary, cfg, out_dir / "summary.txt")

            mid_idx = (
                cfg.get("reference", {}).get(
                    "roi_mid_index", cfg.get("measurement", {}).get("roi_mid_index", 5)
                )
            )
            exp_data = collect_mid_roi_snr(roi_table, mid_idx)

            plot_snr_vs_signal(signals, snr_lin, cfg, out_dir / "snr_signal.png")
            plot_snr_vs_exposure(exp_data, cfg, out_dir / "snr_exposure.png")
            plot_prnu_regression(signals, noises, cfg, out_dir / "prnu_fit.png")
            plot_heatmap(dsnu_map, "DSNU map", out_dir / "dsnu_map.png")
            plot_heatmap(rn_map, "Read noise map", out_dir / "readnoise_map.png")
            plot_heatmap(prnu_map, "PRNU residual", out_dir / "prnu_residual_map.png")
            log_memory_usage("after plots: ")

            graphs = {
                "snr_signal": out_dir / "snr_signal.png",
                "snr_exposure": out_dir / "snr_exposure.png",
                "prnu_fit": out_dir / "prnu_fit.png",
                "dsnu_map": out_dir / "dsnu_map.png",
                "readnoise_map": out_dir / "readnoise_map.png",
                "prnu_residual_map": out_dir / "prnu_residual_map.png",
            }
            report_html(summary, graphs, cfg, out_dir / "report.html")
            logging.info("Pipeline completed")
            log_memory_usage("pipeline end: ")
            return summary
        except Exception as e:  # pragma: no cover - log path
            logging.exception("Pipeline error: %s", e)
            raise


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")
        self.project_dir: Path | None = None
        self.config: Dict[str, Any] | None = None
        self.worker: EvalWorker | None = None
        self.canvases: list[FigureCanvas] = []
        self._setup_ui()
        self.resize(640, 480)

    # ──────────────────────────────────────────── UI setup
    def _setup_ui(self):
        self.sel_btn = QPushButton("Select Project Folder")
        self.sel_btn.clicked.connect(self.select_project)
        self.run_btn = QPushButton("RUN")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_analysis)

        self.status = QLabel("Ready")
        self.progress = QProgressBar()

        self.summary_view = QTextEdit()
        self.summary_view.setReadOnly(True)

        self.graph_tabs = QTabWidget()

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.summary_view)
        self.splitter.addWidget(self.graph_tabs)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.sel_btn)
        btn_row.addWidget(self.run_btn)

        lay = QVBoxLayout()
        lay.addLayout(btn_row)
        lay.addWidget(self.status)
        lay.addWidget(self.progress)
        lay.addWidget(self.splitter)

        container = QWidget()
        container.setLayout(lay)
        self.setCentralWidget(container)

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
        self.run_btn.setEnabled(True)
        self.run_analysis()

    def run_analysis(self):
        if self.project_dir is None or self.config is None:
            QMessageBox.warning(self, "No Project", "Select a project folder first.")
            return
        if self.worker is not None:
            return
        self.run_btn.setEnabled(False)
        self.sel_btn.setEnabled(False)
        self.summary_view.clear()
        self.graph_tabs.clear()
        self.canvases.clear()
        self.status.setText("Running...")
        plt.switch_backend("Agg")
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
        self.run_btn.setEnabled(True)
        self.sel_btn.setEnabled(True)

        if self.project_dir is None or self.config is None:
            return

        out_dir = self.project_dir / self.config.get("output", {}).get("output_dir", "output")

        summary_path = out_dir / "summary.txt"
        if summary_path.is_file():
            text = summary_path.read_text(encoding="utf-8")
        else:
            text = "\n".join(f"{k}: {v:.3f}" for k, v in summary.items())
        self.summary_view.setPlainText(text)

        self.graph_tabs.clear()

        graph_files = {
            "SNR-Signal": "snr_signal.png",
            "SNR-Exposure": "snr_exposure.png",
            "PRNU Fit": "prnu_fit.png",
            "DSNU Map": "dsnu_map.png",
            "Readnoise Map": "readnoise_map.png",
            "PRNU Residual": "prnu_residual_map.png",
        }

        for title, fname in graph_files.items():
            path = out_dir / fname
            if path.is_file():
                widget = self._create_canvas(path)
                self.graph_tabs.addTab(widget, title)

        self.resize(640, self.height())
        h = self.splitter.height()
        self.splitter.setSizes([int(h * 0.25), int(h * 0.75)])
        self._refresh_canvas_geometry()

    def _create_canvas(self, png_path: Path) -> QWidget:
        """Return QWidget with interactive matplotlib canvas for the PNG."""
        plt.switch_backend("QtAgg")
        img = plt.imread(str(png_path))
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.set_axis_off()
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvases.append(canvas)
        toolbar = NavigationToolbar2QT(canvas, None)
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        w.setLayout(layout)
        return w

    def _refresh_canvas_geometry(self) -> None:
        """Resize matplotlib figures to match their canvas widgets."""
        for canvas in self.canvases:
            dpi = canvas.figure.dpi
            scale = getattr(canvas, "devicePixelRatioF", lambda: 1.0)()
            w = canvas.width() * scale / dpi
            h = canvas.height() * scale / dpi
            canvas.figure.set_size_inches(w, h, forward=True)
            canvas.draw_idle()

    def _analysis_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.status.setText("Error")
        self.progress.setValue(0)
        self.worker = None
        self.sel_btn.setEnabled(True)

    # ──────────────────────────────────────────── Events
    def resizeEvent(self, event):  # pragma: no cover - GUI
        super().resizeEvent(event)
        self._refresh_canvas_geometry()

    def showEvent(self, event):  # pragma: no cover - GUI
        super().showEvent(event)
        self._refresh_canvas_geometry()


# ──────────────────────────────────────────── Entrypoint
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show(); sys.exit(app.exec())
