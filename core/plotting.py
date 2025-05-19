# core/plotting.py – config‑aware matplotlib routines

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_snr_vs_signal",
    "plot_snr_vs_exposure",
]


def _maybe_color(idx: int, cfg: Dict) -> str | None:
    """Return color if color_by_exposure is True, else None (use default)."""
    if cfg.get("plot", {}).get("color_by_exposure", False):
        # Simple deterministic color map (tab10 cycles)
        return plt.cm.tab10(idx % 10)
    return None


def plot_snr_vs_signal(signal: np.ndarray, snr: np.ndarray, cfg: Dict, output_path: Path) -> None:
    """Plot SNR–Signal curve and save to *output_path*.

    Parameters
    ----------
    signal : np.ndarray
        X‑axis values (DN).
    snr : np.ndarray
        SNR values.
    cfg : dict
        YAML config with keys under cfg["plot"].
    output_path : Path
        Destination PNG path.
    """
    plt.figure()
    color = _maybe_color(0, cfg)
    plt.plot(signal, snr, marker="o", linestyle="-", color=color)
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR")
    plt.title("SNR vs Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_snr_vs_exposure(exposure: np.ndarray, snr: np.ndarray, cfg: Dict, output_path: Path) -> None:
    """Plot SNR–Exposure curve and save to *output_path*."""
    labels = cfg.get("plot", {}).get("exposure_labels", list(map(str, exposure)))
    plt.figure()
    color = _maybe_color(0, cfg)
    plt.plot(exposure, snr, marker="s", linestyle="-", color=color)
    plt.xticks(exposure, labels)
    plt.xlabel("Exposure Ratio")
    plt.ylabel("SNR")
    plt.title("SNR vs Exposure")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
