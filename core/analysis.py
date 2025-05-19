# core/analysis.py – Spec‑aligned analysis functions

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from utils import config as cfgutil
from core.loader import load_image_stack

__all__ = [
    "extract_roi_stats",
    "calculate_snr_curve",
    "calculate_dynamic_range",
]

# -----------------------------------------------------------------------------
# Helpers (ROI load stub)
# -----------------------------------------------------------------------------

def _load_roi_mask(stack: np.ndarray, roi_zip_path: Path) -> np.ndarray:
    """Dummy ROI loader — returns full‑frame mask. TODO: implement ZIP parsing"""
    return np.ones(stack.shape[1:], bool)


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def extract_roi_stats(project_dir: Path | str, cfg: Dict[str, Any]) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Compute mean/std/snr per (gain_db, exposure_ratio) condition."""
    project_dir = Path(project_dir)
    results: Dict[Tuple[float, float], Dict[str, float]] = {}

    roi_zip = Path(cfg["processing"]["roi_zip_file"])

    for gain_db, g_folder in cfgutil.gain_entries(cfg):
        for ratio, e_folder in cfgutil.exposure_entries(cfg):
            folder = project_dir / g_folder / e_folder
            if not folder.is_dir():
                continue
            stack = load_image_stack(folder)
            mask = _load_roi_mask(stack, roi_zip)
            pixels = stack[:, mask]
            mean = float(np.mean(pixels))
            std = float(np.std(pixels))
            snr = mean / std if std else 0.0
            results[(gain_db, ratio)] = {"mean": mean, "std": std, "snr": snr}
    return results


def calculate_snr_curve(signal: np.ndarray, noise: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    exclude = cfg["processing"].get("exclude_abnormal_snr", True)
    noise_safe = np.where(noise == 0, np.nan if exclude else 1e-6, noise)
    return signal / noise_safe


def calculate_dynamic_range(snr: np.ndarray, signal: np.ndarray, cfg: Dict[str, Any]) -> float:
    thresh = cfg["processing"].get("snr_threshold_dB", 20.0)
    valid = snr >= thresh
    if not np.any(valid):
        return 0.0
    sig_min = signal[np.argmax(valid)]
    sig_max = float(np.max(signal))
    return 20.0 * np.log10(sig_max / sig_min) if sig_min else 0.0
