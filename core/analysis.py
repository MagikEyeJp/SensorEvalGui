# generated: 2025-05-18T11:25:00Z (auto)
# core/analysis.py – Spec‑aligned analysis (ROI, min_sig_factor, debug_stacks)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import tifffile

from utils import config as cfgutil
from utils.roi import load_rois
from core.loader import load_image_stack

__all__ = [
    "extract_roi_stats",
    "extract_roi_table",
    "calculate_snr_curve",
    "calculate_dynamic_range",
    "calculate_dark_noise",
    "calculate_dark_noise_gain",
    "calculate_dn_sat",
    "calculate_dynamic_range_dn",
    "calculate_system_sensitivity",
    "calculate_dn_at_snr",
    "calculate_pseudo_prnu",
]

# ───────────────────────────── internal helpers

def _mask_from_rects(shape: Tuple[int, int], rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for l, t, w, h in rects:
        mask[t:t + h, l:l + w] = True
    return mask

# ───────────────────────────── public api

def extract_roi_stats(project_dir: Path | str, cfg: Dict[str, Any]) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Return dict keyed by (gain_db, exposure_ratio) with mean/std/snr."""
    project_dir = Path(project_dir)
    res: Dict[Tuple[float, float], Dict[str, float]] = {}

    chart_roi_file = project_dir / cfg["measurement"]["chart_roi_file"]
    flat_roi_file  = project_dir / cfg["measurement"]["flat_roi_file"]
    chart_rects = load_rois(chart_roi_file)
    flat_rects  = load_rois(flat_roi_file)

    snr_thresh = cfg["processing"].get("snr_threshold_dB", 10.0)
    min_sig_factor = cfg["processing"].get("min_sig_factor", 3.0)
    excl_low_snr = cfg["processing"].get("exclude_abnormal_snr", True)
    debug_stacks = cfg["output"].get("debug_stacks", False)

    for gain_db, gfold in cfgutil.gain_entries(cfg):
        for ratio, efold in cfgutil.exposure_entries(cfg):
            folder = project_dir / gfold / efold
            if not folder.is_dir():
                continue
            stack = load_image_stack(folder)
            if debug_stacks:
                tifffile.imwrite(folder / "stack_cache.tiff", stack)

            rects = chart_rects if "chart" in efold.lower() else flat_rects
            mask = _mask_from_rects(stack.shape[1:], rects)
            pix = stack[:, mask]
            mean = float(np.mean(pix))
            std  = float(np.std(pix))
            if std == 0:
                continue
            if mean < min_sig_factor * std:
                continue
            snr = mean / std
            if excl_low_snr and snr < snr_thresh:
                continue
            res[(gain_db, ratio)] = {"mean": mean, "std": std, "snr": snr}
    return res


def extract_roi_table(project_dir: Path | str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of ROI stats rows for csv output."""
    project_dir = Path(project_dir)
    chart_roi_file = project_dir / cfg["measurement"]["chart_roi_file"]
    flat_roi_file = project_dir / cfg["measurement"]["flat_roi_file"]
    chart_rects = load_rois(chart_roi_file)
    flat_rects = load_rois(flat_roi_file)

    debug_stacks = cfg["output"].get("debug_stacks", False)

    rows: List[Dict[str, Any]] = []
    for gain_db, gfold in cfgutil.gain_entries(cfg):
        for ratio, efold in cfgutil.exposure_entries(cfg):
            folder = project_dir / gfold / efold
            if not folder.is_dir():
                continue
            stack = load_image_stack(folder)
            if debug_stacks:
                tifffile.imwrite(folder / "stack_cache.tiff", stack)

            if "chart" in efold.lower():
                roi_type = "grayscale"
                rects = chart_rects
            else:
                roi_type = "flat"
                rects = flat_rects
            for i, r in enumerate(rects):
                mask = _mask_from_rects(stack.shape[1:], [r])
                pix = stack[:, mask]
                mean = float(np.mean(pix))
                std = float(np.std(pix))
                snr_db = float("nan") if std == 0 else float(20 * np.log10(mean / std))
                rows.append({
                    "ROI Type": roi_type,
                    "ROI No": i if roi_type == "grayscale" else "-",
                    "Gain (dB)": gain_db,
                    "Exposure": ratio,
                    "Mean": mean,
                    "Std": std,
                    "SNR (dB)": snr_db,
                })
    return rows


def calculate_snr_curve(signal: np.ndarray, noise: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise == 0, np.nan, signal / noise)
    return snr


def calculate_dynamic_range(snr: np.ndarray, signal: np.ndarray, cfg: Dict[str, Any]) -> float:
    thresh = cfg["processing"].get("snr_threshold_dB", 20.0)
    idx = np.where(snr >= thresh)[0]
    if idx.size == 0:
        return 0.0
    sig_min = signal[idx[0]]
    sig_max = float(np.max(signal))
    if sig_min == 0:
        return 0.0
    return 20.0 * np.log10(sig_max / sig_min)


def _reduce(values: np.ndarray, mode: str) -> float:
    if mode == "rms":
        return float(np.sqrt(np.mean(values ** 2)))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "mad":
        med = np.median(values)
        return float(np.median(np.abs(values - med)))
    raise ValueError(mode)


def calculate_dark_noise(project_dir: Path | str, cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Return (DSNU, read_noise) from dark stack."""
    project_dir = Path(project_dir)
    dark_folder = project_dir / cfg["measurement"].get("dark_folder", "dark")
    if not dark_folder.is_dir():
        raise FileNotFoundError(f"Dark folder not found: {dark_folder}")
    stack = load_image_stack(dark_folder)

    roi_file = project_dir / cfg["measurement"].get("flat_roi_file")
    rects = load_rois(roi_file)
    mask = _mask_from_rects(stack.shape[1:], rects)

    stat_mode = cfg["processing"].get("stat_mode", "rms")

    # DSNU: spatial std of mean frame
    mean_frame = np.mean(stack, axis=0)
    dsnu = _reduce(mean_frame[mask] - np.mean(mean_frame[mask]), stat_mode)

    # Read noise: time-domain std per pixel → reduce spatially
    read_noise_pix = np.std(stack, axis=0)
    read_noise = _reduce(read_noise_pix[mask], stat_mode)

    return dsnu, read_noise


def calculate_dark_noise_gain(project_dir: Path | str, gain_db: float, cfg: Dict[str, Any]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (dsnu, read_noise, dsnu_map, read_noise_map) for given gain."""
    gain_folder = cfgutil.find_gain_folder(project_dir, gain_db, cfg)
    dark_folder = gain_folder / cfg["measurement"].get("dark_folder", "dark")
    stack = load_image_stack(dark_folder)

    roi_file = project_dir / cfg["measurement"].get("flat_roi_file")
    rects = load_rois(roi_file)
    mask = _mask_from_rects(stack.shape[1:], rects)

    stat_mode = cfg["processing"].get("stat_mode", "rms")
    mean_frame = np.mean(stack, axis=0)
    dsnu_map = mean_frame - np.mean(mean_frame[mask])
    dsnu = _reduce(dsnu_map[mask], stat_mode)
    read_noise_map = np.std(stack, axis=0)
    read_noise = _reduce(read_noise_map[mask], stat_mode)
    return dsnu, read_noise, dsnu_map, read_noise_map


def calculate_dn_sat(flat_stack: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Detect DN_sat using multiple heuristics."""
    p999 = float(np.percentile(flat_stack, 99.9))
    sat_factor = cfg.get("illumination", {}).get("sat_factor", 0.95)
    max_from_factor = float(np.max(flat_stack)) / max(sat_factor, 1e-6)
    adc_bits = int(cfg.get("sensor", {}).get("adc_bits", 16))
    adc_max = (1 << adc_bits) - 1
    method3 = adc_max * 0.90
    return max(p999, max_from_factor, method3)


def calculate_dynamic_range_dn(dn_sat: float, read_noise: float) -> float:
    if read_noise == 0:
        return 0.0
    return 20.0 * np.log10(dn_sat / read_noise)


def calculate_pseudo_prnu(flat_stack: np.ndarray, cfg: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    """Return (prnu_percent, residual_map)."""
    mean_frame = np.mean(flat_stack, axis=0)
    std_frame = np.std(flat_stack, axis=0)
    stat_mode = cfg["processing"].get("stat_mode", "rms")
    value = _reduce(std_frame, stat_mode) / max(mean_frame.mean(), 1e-6) * 100.0
    return value, std_frame


def calculate_system_sensitivity(flat_stack: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Return System Sensitivity (DN/µW·cm⁻²·s)."""
    illum = cfg.get("illumination", {})
    power = float(illum.get("power_uW_cm2", 1.0))
    exposure_ms = float(illum.get("exposure_ms", 1.0))
    denom = power * exposure_ms / 1000.0
    if denom == 0:
        return 0.0
    mean_dn = float(np.mean(flat_stack))
    return mean_dn / denom


def calculate_dn_at_snr(signal: np.ndarray, snr_lin: np.ndarray, threshold_db: float) -> float:
    """Interpolate DN where SNR crosses threshold_dB."""
    thr_lin = 10 ** (threshold_db / 20.0)
    idx = np.where(snr_lin >= thr_lin)[0]
    if idx.size == 0:
        return float("nan")
    if idx[0] == 0:
        return float(signal[0])
    x0, x1 = signal[idx[0]-1], signal[idx[0]]
    y0, y1 = snr_lin[idx[0]-1], snr_lin[idx[0]]
    if y1 == y0:
        return float(x1)
    r = (thr_lin - y0) / (y1 - y0)
    return float(x0 + r * (x1 - x0))
