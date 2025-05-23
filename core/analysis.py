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
    "calculate_snr_curve",
    "calculate_dynamic_range",
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
