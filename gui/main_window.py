# generated: 2025-05-18T11:35:00Z (auto)
# gui/main_window.py – PySide6 GUI (Spec‑aligned, full analysis pipeline)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Sequence
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
    QGridLayout,
)
from PySide6.QtGui import QPixmap, QFontDatabase, QKeySequence
from PySide6.QtCore import QThread, Signal, Qt, QTimer

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from utils.config import load_config
import utils.config as cfgutil
from utils.logger import log_memory_usage, apply_logging_config
from core.pipeline import run_pipeline
from core.analysis import (
    extract_roi_stats,
    extract_roi_stats_gainmap,
    extract_roi_table,
    calculate_dark_noise,
    calculate_dark_noise_gain,
    calculate_dn_sat,
    calculate_dynamic_range_dn,
    calculate_system_sensitivity,
    collect_mid_roi_snr,
    collect_gain_snr_signal,
    collect_gain_noise_signal,
    collect_prnu_points,
    calculate_dn_at_snr_pspline,
    calculate_snr_at_half,
    calculate_dn_at_snr_one,
    calculate_prnu_residual,
    calculate_gain_map,
    clear_cache,
)
from core.plotting import (
    plot_snr_vs_signal_multi,
    plot_noise_vs_signal_multi,
    plot_snr_vs_exposure,
    plot_prnu_regression,
    plot_heatmap,
    plot_roi_area,
)
from utils.roi import load_rois
from core.report_gen import (
    save_summary_txt,
    report_csv,
    report_html,
    save_snr_signal_json,
)
from core.loader import load_image_stack, load_first_frame
from typing import Callable, Optional


class EvalWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(self, project: Path, cfg: Dict[str, Any]):
        super().__init__()
        self.project = project
        self.cfg = cfg

    def run(self) -> None:
        try:
            logging.info("Evaluation worker started")
            log_memory_usage("before pipeline: ")
            result = run_pipeline(
                self.project,
                self.cfg,
                progress=self.progress.emit,
                status=self.status.emit,
            )
            self.progress.emit(100)
            logging.info("Evaluation worker finished")
            log_memory_usage("after pipeline: ")
            self.finished.emit(result)
        except Exception as e:  # pragma: no cover - UI error path
            logging.exception("Worker failed: %s", e)
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, autoload: Path | None = None):
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")
        self.project_dir: Path | None = None
        self.config: Dict[str, Any] | None = None
        self.worker: EvalWorker | None = None
        self.canvases: list[FigureCanvas] = []
        self._setup_ui()
        self.resize(640, 480)
        if autoload is not None:
            QTimer.singleShot(0, lambda: self.open_project(Path(autoload)))

    # ──────────────────────────────────────────── UI setup
    def _setup_ui(self):
        self.sel_btn = QPushButton("Select Project Folder")
        self.sel_btn.clicked.connect(self.select_project)
        self.run_btn = QPushButton("RUN")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_analysis)
        self.sel_btn.setShortcut(QKeySequence("Ctrl+O"))
        self.run_btn.setShortcut(QKeySequence("Ctrl+R"))

        self.status = QLabel("Ready")
        self.progress = QProgressBar()

        self.summary_view = QTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setLineWrapMode(QTextEdit.NoWrap)
        self.summary_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

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
        self.open_project(Path(dir_path))

    def open_project(self, path: Path) -> None:
        """Load project at path and start analysis."""
        self.project_dir = path
        cfg_path = self.project_dir / "config.yaml"
        if not cfg_path.is_file():
            QMessageBox.critical(
                self, "Error", "config.yaml not found in project folder"
            )
            return
        self.config = load_config(cfg_path)
        apply_logging_config(self.config)
        self.status.setText(f"Project loaded: {self.project_dir}")
        self.run_btn.setEnabled(True)
        self.run_analysis()

    def run_analysis(self):
        if self.project_dir is None:
            QMessageBox.warning(self, "No Project", "Select a project folder first.")
            return
        cfg_path = self.project_dir / "config.yaml"
        if not cfg_path.is_file():
            QMessageBox.critical(
                self, "Error", "config.yaml not found in project folder"
            )
            return
        # reload configuration fresh each run to avoid stale values
        self.config = load_config(cfg_path)
        apply_logging_config(self.config)
        if self.worker is not None:
            return
        self.run_btn.setEnabled(False)
        self.sel_btn.setEnabled(False)
        self.summary_view.clear()
        self.graph_tabs.clear()
        self._clear_canvases()
        self.status.setText("Running...")
        self.worker = EvalWorker(self.project_dir, self.config)
        self.worker.finished.connect(self._analysis_done)
        self.worker.error.connect(self._analysis_error)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status.setText)
        self.worker.finished.connect(self._worker_finished)
        self.progress.setValue(0)
        self.worker.start()

    # ──────────────────────────────────────────── Core pipeline
    def _analysis_done(self, result: Dict[str, Any]):
        self.status.setText("Done ✅")
        self.progress.setValue(100)
        self.run_btn.setEnabled(True)
        self.sel_btn.setEnabled(True)

        if self.project_dir is None or self.config is None:
            return

        out_dir = self.project_dir / self.config.get("output", {}).get(
            "output_dir", "output"
        )

        summary = result.get("summary", {})
        figures = result.get("figures", {})

        summary_path = out_dir / "summary.txt"
        if summary_path.is_file():
            text = summary_path.read_text(encoding="utf-8")
        else:
            text = "\n".join(f"{k}: {v:.3f}" for k, v in summary.items())
        self.summary_view.setPlainText(text)

        self.graph_tabs.clear()

        graph_groups = {
            "SNR-Signal": ["snr_signal", "noise_signal"],
            "SNR-Exposure": ["snr_exposure"],
            "PRNU Fit": ["prnu_fit"],
            "DSNU Map": ["dsnu_map", "dsnu_map_scaled"],
            "Readnoise Map": ["readnoise_map", "readnoise_map_scaled"],
            "PRNU Residual": [
                "prnu_residual_map",
                "prnu_residual_map_scaled",
                "gain_map",
            ],
            "ROI Area": ["roi_area"],
        }

        for title, names in graph_groups.items():
            figs_to_use = [figures[n] for n in names if n in figures]
            if figs_to_use:
                widget = self._create_canvas(figs_to_use)
                self.graph_tabs.addTab(widget, title)

        # self.resize(640, self.height())
        h = self.splitter.height()
        self.splitter.setSizes([int(h * 0.25), int(h * 0.75)])
        self._refresh_canvas_geometry()

    def _create_canvas(self, figures: Sequence[Figure]) -> QWidget:
        """Return QWidget with interactive matplotlib canvases."""
        w_container = QWidget()
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        for idx, fig in enumerate(figures):
            vbox = QVBoxLayout()
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.canvases.append(canvas)
            toolbar = NavigationToolbar2QT(canvas, None)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas)
            widget = QWidget()
            widget.setLayout(vbox)
            row = idx // 2
            col = idx % 2
            layout.addWidget(widget, row, col)

        w_container.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(w_container)
        return scroll

    def _clear_canvases(self) -> None:
        """Close and delete existing matplotlib canvases."""
        for canvas in self.canvases:
            try:
                plt.close(canvas.figure)
            except Exception:
                pass
            canvas.setParent(None)
            canvas.deleteLater()
        self.canvases.clear()

    def _refresh_canvas_geometry(self) -> None:
        """Resize matplotlib figures to match their canvas widgets."""
        for canvas in self.canvases:
            dpi = canvas.figure.dpi
            scale = getattr(canvas, "devicePixelRatioF", lambda: 1.0)()
            w = canvas.width() * scale / dpi
            h = canvas.height() * scale / dpi
            canvas.figure.set_size_inches(w, h, forward=True)
            canvas.draw_idle()

    def _worker_finished(self, *args) -> None:
        """Cleanup after worker thread finishes."""
        if self.worker is not None:
            self.worker.wait()
            self.worker = None

    def _analysis_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.status.setText("Error")
        self.progress.setValue(0)
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
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
