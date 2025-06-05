# core/pipeline.py – High-level analysis pipeline

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import tifffile
from matplotlib.figure import Figure

__all__ = ["run_pipeline"]

from utils.logger import log_memory_usage, apply_logging_config
import utils.config as cfgutil
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


pipeline_lock = threading.Lock()


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
            black_levels: Dict[float, float] = {}
            for gain_db, _ in cfgutil.gain_entries(cfg):
                _, _, _, _, bl = calculate_dark_noise_gain(
                    project, gain_db, cfg, status=status
                )
                black_levels[gain_db] = bl

            tuples = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
            signals = np.array([kv[1]["mean"] for kv in tuples])
            noises = np.array([kv[1]["std"] for kv in tuples])
            snr_lin = np.array(
                [
                    (kv[1]["mean"] - black_levels.get(kv[0][0], 0.0)) / kv[1]["std"]
                    for kv in tuples
                ]
            )
            snr_lin = np.maximum(snr_lin, 1.0)
            ratios = np.array([kv[0][1] for kv in tuples])

            fit_cfg = cfg.get("processing", {}).get("snr_fit", {})
            adc_full_scale = cfgutil.adc_full_scale(cfg)

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
                    snr_lin_g = np.maximum((sig_g - black_level) / noise_g, 1.0)
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
