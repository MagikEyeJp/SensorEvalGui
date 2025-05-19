# core/plotting.py – Spec‑aware plotting utilities

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import config as cfgutil

__all__ = [
    "plot_snr_vs_signal",
    "plot_snr_vs_exposure",
]


def _auto_labels(ratios: Sequence[float]) -> list[str]:
    return [f"{r:g}×" for r in ratios]


def plot_snr_vs_signal(signal: np.ndarray, snr: np.ndarray, cfg: Dict[str, Any], output_path: Path):
    """Plot SNR–Signal curve and save PNG."""
    plt.figure()
    plt.plot(signal, snr, marker="o", linestyle="-")
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR")
    plt.title("SNR vs Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_snr_vs_exposure(exposure_ratios: np.ndarray, snr: np.ndarray, cfg: Dict[str, Any], output_path: Path):
    """Plot SNR–Exposure curve with labels from cfg.plot.exposures or auto."""
    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("exposures")
    if labels is None:
        # derive order from measurement.exposures nested‑dict (descending ratio)
        labels = [ratio for ratio, _ in cfgutil.exposure_entries(cfg)]
    label_strs = _auto_labels(labels)

    plt.figure()
    plt.plot(exposure_ratios, snr, marker="s", linestyle="-")
    plt.xticks(exposure_ratios, label_strs, rotation=45)
    plt.xlabel("Exposure Ratio")
    plt.ylabel("SNR")
    plt.title("SNR vs Exposure")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
