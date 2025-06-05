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

# ensure only one pipeline runs at a time
pipeline_lock = threading.Lock()
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


def run_pipeline(
    project: Path,
    cfg: Dict[str, Any],
    *,
    progress: Optional[Callable[[int], None]] = None,
    status: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """Run full analysis pipeline with logging and memory checks."""
    clear_cache()
    apply_logging_config(cfg)
    logging.info("Pipeline start: %s", project)
    with pipeline_lock:
        try:
            log_memory_usage("start: ")
            if status:
                status("Loading images...")
            if progress:
                progress(0)

            # 1. ROI-based statistics
            stats = extract_roi_stats(project, cfg, status=status)
            if not stats:
                raise RuntimeError("No valid stacks found for analysis")

            log_memory_usage("after roi stats: ")
            if progress:
                progress(10)
            tuples = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
            signals = np.array([kv[1]["mean"] for kv in tuples])
            noises = np.array([kv[1]["std"] for kv in tuples])
            snr_lin = signals / noises
            ratios = np.array([kv[0][1] for kv in tuples])

            fit_cfg = cfg.get("processing", {}).get("snr_fit", {})
            adc_full_scale = cfgutil.adc_full_scale(cfg)

            black_levels: Dict[float, float] = {}
            for gain_db, _ in cfgutil.gain_entries(cfg):
                _, _, _, _, bl = calculate_dark_noise_gain(
                    project, gain_db, cfg, status=status
                )
                black_levels[gain_db] = bl

            roi_table = extract_roi_table(project, cfg)
            snr_signal_data = collect_gain_snr_signal(roi_table, cfg, black_levels)
            noise_signal_data = collect_gain_noise_signal(roi_table, cfg, black_levels)
            flat_roi_file = project / cfg["measurement"].get("flat_roi_file")
            flat_rects = load_rois(flat_roi_file)

            gain_mode = cfg.get("processing", {}).get("gain_map_mode", "none")
            if gain_mode != "none":
                stats_corr = extract_roi_stats_gainmap(
                    project, cfg, status=status, noise_signals=noise_signal_data
                )
                tuples_c = sorted(stats_corr.items(), key=lambda kv: kv[1]["mean"])
                signals_corr = np.array([kv[1]["mean"] for kv in tuples_c])
                noises_corr = np.array([kv[1]["std"] for kv in tuples_c])
                prnu_stats = stats_corr
            else:
                signals_corr = signals
                noises_corr = noises
                prnu_stats = stats

            prnu_data = collect_prnu_points(prnu_stats)

            # 2. Dark/flat metrics per gain (use first gain for maps)
            debug_stacks = cfg["output"].get("debug_stacks", False)
            out_dir = project / cfg["output"].get("output_dir", "output")
            out_dir.mkdir(exist_ok=True)

            dsnu_list = []
            rn_list = []
            prnu_list = []
            sens_list = []
            black_list = []
            per_gain: Dict[float, Dict[str, float]] = {}
            gains_list = [g for g, _ in cfgutil.gain_entries(cfg)]
            step = 60 / len(gains_list) if gains_list else 60
            first = True
            gain_map = None
            for idx, (gain_db, _) in enumerate(cfgutil.gain_entries(cfg)):
                if status:
                    status(f"Processing gain {gain_db:.1f} dB")
                logging.info("Processing gain %.1f dB", gain_db)
                dsnu, rn, dsnu_map_tmp, rn_map_tmp, black_level = (
                    calculate_dark_noise_gain(project, gain_db, cfg, status=status)
                )
                flat_folder = cfgutil.find_gain_folder(project, gain_db, cfg) / cfg[
                    "measurement"
                ].get("flat_lens_folder", "flat")
                if status:
                    status(f"Loading flat frames for gain {gain_db:.1f} dB")
                flat_stack = load_image_stack(flat_folder)
                dn_sat_gain = calculate_dn_sat(
                    flat_stack, cfg, noise_signal_data.get(gain_db)
                )
                gain_map_tmp = calculate_gain_map(
                    flat_stack,
                    cfg,
                    flat_rects,
                    project,
                    gain_db,
                    noise_signal=noise_signal_data.get(gain_db),
                    dn_sat=dn_sat_gain,
                )
                if debug_stacks and first:
                    dark_folder = cfgutil.find_gain_folder(project, gain_db, cfg) / cfg[
                        "measurement"
                    ].get("dark_folder", "dark")
                    dark_stack = load_image_stack(dark_folder)
                    tifffile.imwrite(
                        out_dir / f"dark_cache_{int(gain_db)}dB.tiff", dark_stack
                    )
                    tifffile.imwrite(
                        out_dir / f"flat_cache_{int(gain_db)}dB.tiff", flat_stack
                    )
                prnu, prnu_map_tmp = calculate_prnu_residual(
                    flat_stack,
                    cfg,
                    flat_rects,
                    project,
                    gain_db,
                    noise_signal=noise_signal_data.get(gain_db),
                    dn_sat=dn_sat_gain,
                )
                gain_mult = cfgutil.gain_ratio(gain_db)
                sens = calculate_system_sensitivity(
                    flat_stack,
                    cfg,
                    flat_rects,
                    ratio=1.0 / gain_mult,
                )
                if first:
                    dsnu_map, rn_map, prnu_map, gain_map = (
                        dsnu_map_tmp,
                        rn_map_tmp,
                        prnu_map_tmp,
                        gain_map_tmp,
                    )
                    dn_sat = dn_sat_gain
                    first = False

                # SNR metrics for this gain
                tuples_g = sorted(
                    (kv for kv in stats.items() if kv[0][0] == gain_db),
                    key=lambda kv: kv[1]["mean"],
                )
                if tuples_g:
                    sig_g = np.array([kv[1]["mean"] for kv in tuples_g])
                    noise_g = np.array([kv[1]["std"] for kv in tuples_g])
                    snr_lin_g = sig_g / noise_g
                    dn_at_10_g = calculate_dn_at_snr_pspline(
                        sig_g,
                        snr_lin_g,
                        cfg["processing"].get("snr_threshold_dB", 10.0),
                        adc_full_scale,
                        black_level,
                        deg=int(fit_cfg.get("deg", 3)),
                        n_splines=fit_cfg.get("n_splines", "auto"),
                        lam=fit_cfg.get("lam"),
                        knot_density=fit_cfg.get("knot_density", "auto"),
                        robust=fit_cfg.get("robust", "huber"),
                        num_points=int(fit_cfg.get("num_points", 400)),
                    )
                    snr_at_50_g = calculate_snr_at_half(sig_g, snr_lin_g, dn_sat_gain)
                    dn_at_0_g = calculate_dn_at_snr_pspline(
                        sig_g,
                        snr_lin_g,
                        0.0,
                        adc_full_scale,
                        black_level,
                        deg=int(fit_cfg.get("deg", 3)),
                        n_splines=fit_cfg.get("n_splines", "auto"),
                        lam=fit_cfg.get("lam"),
                        knot_density=fit_cfg.get("knot_density", "auto"),
                        robust=fit_cfg.get("robust", "huber"),
                        num_points=int(fit_cfg.get("num_points", 400)),
                    )
                else:
                    dn_at_10_g = float("nan")
                    snr_at_50_g = float("nan")
                    dn_at_0_g = float("nan")

                dyn_range_g = calculate_dynamic_range_dn(dn_sat_gain, rn)
                per_gain[gain_db] = {
                    "Dynamic Range": dyn_range_g,
                    "DSNU": dsnu,
                    "Read Noise": rn,
                    "Black level": black_level,
                    "DN_sat": dn_sat_gain,
                    "Pseudo PRNU": prnu,
                    "SensitivityDN": sens,
                    "DN @ 10 dB": dn_at_10_g,
                    "SNR @ 50%": snr_at_50_g,
                    "DN @ 0 dB": dn_at_0_g,
                }

                dsnu_list.append(dsnu)
                rn_list.append(rn)
                prnu_list.append(prnu)
                sens_list.append(sens)
                black_list.append(black_level)
                log_memory_usage(f"after gain {gain_db}: ")
                if progress:
                    progress(int(10 + step * (idx + 1)))

            dsnu = float(np.mean(dsnu_list)) if dsnu_list else 0.0
            read_noise = float(np.mean(rn_list)) if rn_list else 0.0
            dyn_range = calculate_dynamic_range_dn(dn_sat, read_noise)
            prnu = float(np.mean(prnu_list)) if prnu_list else float("nan")
            system_sens = float(np.mean(sens_list)) if sens_list else float("nan")
            black_level = float(np.mean(black_list)) if black_list else 0.0
            dn_at_10 = calculate_dn_at_snr_pspline(
                signals,
                snr_lin,
                cfg["processing"].get("snr_threshold_dB", 10.0),
                adc_full_scale,
                black_level,
                deg=int(fit_cfg.get("deg", 3)),
                n_splines=fit_cfg.get("n_splines", "auto"),
                lam=fit_cfg.get("lam"),
                knot_density=fit_cfg.get("knot_density", "auto"),
                robust=fit_cfg.get("robust", "huber"),
                num_points=int(fit_cfg.get("num_points", 400)),
            )
            snr_at_50 = calculate_snr_at_half(signals, snr_lin, dn_sat)
            dn_at_0 = calculate_dn_at_snr_pspline(
                signals,
                snr_lin,
                0.0,
                adc_full_scale,
                black_level,
                deg=int(fit_cfg.get("deg", 3)),
                n_splines=fit_cfg.get("n_splines", "auto"),
                lam=fit_cfg.get("lam"),
                knot_density=fit_cfg.get("knot_density", "auto"),
                robust=fit_cfg.get("robust", "huber"),
                num_points=int(fit_cfg.get("num_points", 400)),
            )

            log_memory_usage("after metrics: ")
            if progress:
                progress(80)
            if status:
                status("Plotting graphs...")

            stats_rows = roi_table
            report_csv(stats_rows, cfg, out_dir / "roi_stats.csv")
            summary_avg = {
                "Dynamic Range": dyn_range,
                "DSNU": dsnu,
                "Read Noise": read_noise,
                "Black level": black_level,
                "DN_sat": dn_sat,
                "Pseudo PRNU": prnu,
                "SensitivityDN": system_sens,
                "DN @ 10 dB": dn_at_10,
                "SNR @ 50%": snr_at_50,
                "DN @ 0 dB": dn_at_0,
            }
            save_summary_txt(per_gain, cfg, out_dir / "summary.txt")

            dn_sat_map = {g: v.get("DN_sat", float("nan")) for g, v in per_gain.items()}

            mid_idx = cfg.get("reference", {}).get(
                "roi_mid_index", cfg.get("measurement", {}).get("roi_mid_index", 5)
            )
            exp_data = collect_mid_roi_snr(
                roi_table,
                mid_idx,
                black_levels=black_levels,
            )
            sig_data = snr_signal_data

            logging.info("Plotting SNR vs Signal (multi)")
            log_memory_usage("before snr_signal plot: ")
            fig_snr_signal = plot_snr_vs_signal_multi(
                sig_data,
                cfg,
                out_dir / "snr_signal.png",
                return_fig=True,
                interp_points=cfgutil.adc_full_scale(cfg),
                black_levels=black_levels,
                dn_sat=dn_sat_map,
            )
            save_snr_signal_json(
                sig_data,
                cfg,
                out_dir / "snr_signal.json",
                black_levels=black_levels,
                dn_sat=dn_sat_map,
            )
            log_memory_usage("after snr_signal plot: ")

            logging.info("Plotting Noise vs Signal (multi)")
            fig_noise_signal = plot_noise_vs_signal_multi(
                noise_signal_data,
                cfg,
                out_dir / "noise_signal.png",
                return_fig=True,
            )
            log_memory_usage("after noise_signal plot: ")

            logging.info("Plotting SNR vs Exposure")
            log_memory_usage("before snr_exposure plot: ")
            fig_snr_exposure = plot_snr_vs_exposure(
                exp_data, cfg, out_dir / "snr_exposure.png", return_fig=True
            )
            log_memory_usage("after snr_exposure plot: ")

            logging.info("Plotting PRNU regression")
            log_memory_usage("before prnu_fit plot: ")
            fig_prnu_fit = plot_prnu_regression(
                prnu_data, cfg, out_dir / "prnu_fit.png", return_fig=True
            )
            log_memory_usage("after prnu_fit plot: ")

            logging.info("Plotting noise maps")
            log_memory_usage("before dsnu_map plot: ")
            fig_dsnu_map = plot_heatmap(
                dsnu_map, "DSNU map", out_dir / "dsnu_map.png", return_fig=True
            )
            fig_dsnu_map_scaled = plot_heatmap(
                dsnu_map,
                "DSNU map (scaled)",
                out_dir / "dsnu_map_scaled.png",
                vmin=0,
                vmax=float(np.nanpercentile(dsnu_map, 99)),
                return_fig=True,
            )
            log_memory_usage("after dsnu_map plot: ")
            fig_rn_map = plot_heatmap(
                rn_map, "Read noise map", out_dir / "readnoise_map.png", return_fig=True
            )
            fig_rn_map_scaled = plot_heatmap(
                rn_map,
                "Read noise map (scaled)",
                out_dir / "readnoise_map_scaled.png",
                vmin=0,
                vmax=float(np.nanpercentile(rn_map, 99)),
                return_fig=True,
            )
            fig_prnu_map = plot_heatmap(
                prnu_map,
                "PRNU residual",
                out_dir / "prnu_residual_map.png",
                return_fig=True,
            )
            fig_prnu_map_scaled = plot_heatmap(
                prnu_map,
                "PRNU residual (scaled)",
                out_dir / "prnu_residual_map_scaled.png",
                vmin=0,
                vmax=float(np.nanpercentile(prnu_map, 99)),
                return_fig=True,
            )

            fig_gain_map = None
            if gain_map is not None:
                fig_gain_map = plot_heatmap(
                    gain_map,
                    "Gain map",
                    out_dir / "gain_map.png",
                    return_fig=True,
                )

            fig_roi_area = None
            try:
                g0 = cfgutil.nearest_gain(cfg, 0.0)
                r1 = cfgutil.nearest_exposure(cfg, 1.0)
                chart_folder = cfgutil.find_exposure_folder(project, g0, r1, cfg)
                chart_img = load_first_frame(chart_folder)
                flat_folder = cfgutil.find_gain_folder(project, g0, cfg) / cfg[
                    "measurement"
                ].get("flat_lens_folder", "LensFlat")
                flat_img = load_first_frame(flat_folder)
                dark_folder = cfgutil.find_gain_folder(project, g0, cfg) / cfg[
                    "measurement"
                ].get("dark_folder", "dark")
                dark_img = load_first_frame(dark_folder)
                chart_rects = load_rois(project / cfg["measurement"]["chart_roi_file"])
                flat_rects = load_rois(project / cfg["measurement"]["flat_roi_file"])
                fig_roi_area = plot_roi_area(
                    [chart_img, flat_img, dark_img],
                    [chart_rects, flat_rects, flat_rects],
                    ["Grayscale", "Flat", "Dark"],
                    out_dir / "roi_area.png",
                    return_fig=True,
                )
            except Exception as exc:  # pragma: no cover - optional
                logging.info("ROI area plot failed: %s", exc)

            log_memory_usage("after plots: ")
            if progress:
                progress(90)

            graphs = {
                "snr_signal": out_dir / "snr_signal.png",
                "noise_signal": out_dir / "noise_signal.png",
                "snr_exposure": out_dir / "snr_exposure.png",
                "prnu_fit": out_dir / "prnu_fit.png",
                "dsnu_map": out_dir / "dsnu_map.png",
                "dsnu_map_scaled": out_dir / "dsnu_map_scaled.png",
                "readnoise_map": out_dir / "readnoise_map.png",
                "readnoise_map_scaled": out_dir / "readnoise_map_scaled.png",
                "prnu_residual_map": out_dir / "prnu_residual_map.png",
                "prnu_residual_map_scaled": out_dir / "prnu_residual_map_scaled.png",
                "gain_map": out_dir / "gain_map.png",
                "roi_area": out_dir / "roi_area.png",
            }
            figures = {
                "snr_signal": fig_snr_signal,
                "noise_signal": fig_noise_signal,
                "snr_exposure": fig_snr_exposure,
                "prnu_fit": fig_prnu_fit,
                "dsnu_map": fig_dsnu_map,
                "dsnu_map_scaled": fig_dsnu_map_scaled,
                "readnoise_map": fig_rn_map,
                "readnoise_map_scaled": fig_rn_map_scaled,
                "prnu_residual_map": fig_prnu_map,
                "prnu_residual_map_scaled": fig_prnu_map_scaled,
            }
            if fig_gain_map is not None:
                figures["gain_map"] = fig_gain_map
            if fig_roi_area is not None:
                figures["roi_area"] = fig_roi_area
            report_html(summary_avg, graphs, cfg, out_dir / "report.html")
            if progress:
                progress(100)
            logging.info("Pipeline completed")
            log_memory_usage("pipeline end: ")
            return {"summary": summary_avg, "figures": figures}
        except Exception as e:  # pragma: no cover - log path
            logging.exception("Pipeline error: %s", e)
            raise


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
