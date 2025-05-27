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
    "plot_prnu_regression",
    "plot_heatmap",
]


def _auto_labels(ratios: Sequence[float]) -> list[str]:
    return [f"{r:g}×" for r in ratios]


def plot_snr_vs_signal(signal: np.ndarray, snr: np.ndarray, cfg: Dict[str, Any], output_path: Path):
    """Plot SNR–Signal curve (log–log) with ideal line and threshold."""
    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    thr_lin = 10 ** (thresh / 20.0)
    plt.figure()
    plt.loglog(signal, snr, marker="o", linestyle="-", label="Measured")
    plt.loglog(signal, np.sqrt(signal), linestyle=":", label="Ideal √µ")
    plt.axhline(thr_lin, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR")
    plt.title("SNR vs Signal")
    plt.grid(True, which="both")
    plt.legend()
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

    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    thr_lin = 10 ** (thresh / 20.0)
    plt.figure()
    plt.semilogx(exposure_ratios, snr, marker="s", linestyle="-", label="Measured")
    plt.axhline(thr_lin, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xticks(exposure_ratios, label_strs, rotation=45)
    plt.xlabel("Exposure Ratio")
    plt.ylabel("SNR")
    plt.title("SNR vs Exposure")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prnu_regression(means: np.ndarray, stds: np.ndarray, output_path: Path):
    """Plot PRNU regression (std vs mean) with simple linear fit."""
    plt.figure()
    plt.scatter(means, stds, s=8, alpha=0.6)
    if means.size > 1:
        p = np.polyfit(means, stds, 1)
        x = np.linspace(means.min(), means.max(), 100)
        y = np.polyval(p, x)
        plt.plot(x, y, "r--", label=f"y={p[0]:.3f}x+{p[1]:.3f}")
        plt.legend(fontsize=8)
    plt.xlabel("Mean (DN)")
    plt.ylabel("Std (DN)")
    plt.title("PRNU Regression")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_heatmap(data: np.ndarray, title: str, output_path: Path):
    plt.figure()
    plt.imshow(data, cmap="viridis")
    plt.title(title)
    plt.colorbar(label="DN")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
