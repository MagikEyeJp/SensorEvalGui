# core/plotting.py – Spec‑aware plotting utilities

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence
import logging

import matplotlib

matplotlib.use("Agg")  # avoid GUI backend so plotting works inside threads
import matplotlib.pyplot as plt
import numpy as np

from utils.logger import log_memory_usage

from utils import config as cfgutil

__all__ = [
    "plot_snr_vs_signal",
    "plot_snr_vs_signal_multi",
    "plot_snr_vs_exposure",
    "plot_prnu_regression",
    "plot_heatmap",
    "plot_roi_area",
]


def _validate_positive_finite(arr: np.ndarray, name: str) -> np.ndarray:
    """Return ``arr`` if it is non-empty, finite and strictly positive."""
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _auto_labels(ratios: Sequence[float]) -> list[str]:
    return [f"{r:g}×" for r in ratios]


def plot_snr_vs_signal(
    signal: np.ndarray, snr: np.ndarray, cfg: Dict[str, Any], output_path: Path
):
    """Plot SNR–Signal curve (log–log) with ideal line and threshold."""
    signal = _validate_positive_finite(signal, "signal")
    snr = _validate_positive_finite(snr, "snr")
    if signal.size == 1 or snr.size == 1:
        # Avoid singular log scale when only one sample is present
        signal = np.asarray([signal[0] * 0.9, signal[0] * 1.1])
        snr = np.asarray([snr[0] * 0.9, snr[0] * 1.1])
    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    snr_db = 20 * np.log10(snr)
    plt.figure()
    plt.loglog(signal, snr_db, marker="o", linestyle="-", label="Measured")
    plt.loglog(signal, 20 * np.log10(np.sqrt(signal)), linestyle=":", label="Ideal √µ")
    plt.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Signal")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_snr_vs_signal_multi(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
):
    """Plot SNR–Signal curves for multiple gains."""
    logging.info("plot_snr_vs_signal_multi: output=%s", output_path)
    log_memory_usage("plot start: ")

    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    plt.figure()

    all_signals = []
    for gain, (sig, snr) in sorted(data.items()):
        logging.debug(
            "gain %.1f: sig shape=%s snr shape=%s", gain, sig.shape, snr.shape
        )
        sig = _validate_positive_finite(sig, "signal")
        snr = _validate_positive_finite(snr, "snr")
        if sig.size == 1 or snr.size == 1:
            sig = np.asarray([sig[0] * 0.9, sig[0] * 1.1])
            snr = np.asarray([snr[0] * 0.9, snr[0] * 1.1])
        all_signals.append(sig)
        snr_db = 20 * np.log10(snr)
        plt.loglog(sig, snr_db, marker="o", linestyle="-", label=f"{gain:g}dB")

    if all_signals:
        concat = np.concatenate(all_signals)
        x_min = float(concat.min())
        x_max = float(concat.max())
        if x_min == x_max:
            xs = np.asarray([x_min * 0.9, x_max * 1.1])
        else:
            xs = np.linspace(x_min, x_max, 200)
        plt.loglog(xs, 20 * np.log10(np.sqrt(xs)), linestyle=":", label="Ideal √µ")

    plt.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Signal")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log_memory_usage("plot end: ")


def plot_snr_vs_exposure(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
):
    """Plot SNR–Exposure curves per gain."""

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("exposures")
    if labels is None:
        try:
            labels = [ratio for ratio, _ in cfgutil.exposure_entries(cfg)]
        except KeyError:
            labels = []
    base_ms = float(cfg.get("illumination", {}).get("exposure_ms", 1.0))
    xticks = base_ms * np.array(labels)
    label_strs = [f"{t:g}" for t in xticks]

    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)

    plt.figure()
    for gain, (ratios, snr) in sorted(data.items()):
        ratios = _validate_positive_finite(ratios, "exposure ratios")
        snr = _validate_positive_finite(snr, "snr")
        snr_db = 20 * np.log10(snr)
        gain_mult = cfgutil.gain_ratio(gain)
        times = base_ms * ratios / gain_mult
        plt.semilogx(
            times,
            snr_db,
            marker="s",
            linestyle="-",
            label=f"{gain:g} dB",
        )
    plt.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xticks(xticks, label_strs, rotation=45)
    plt.xlabel("Exposure Time (ms)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Exposure")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prnu_regression(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
):
    """Plot PRNU regression per gain with LS or WLS fit."""

    plt.figure()
    fit_mode = cfg.get("processing", {}).get("prnu_fit", "LS").upper()
    cmap = plt.cm.get_cmap("tab10")

    for idx, (gain, (means, stds)) in enumerate(sorted(data.items())):
        means = _validate_positive_finite(means, "mean")
        stds = _validate_positive_finite(stds, "std")
        color = cmap(idx % 10)
        plt.scatter(means, stds, s=8, alpha=0.6, color=color, label=f"{gain:g}dB")
        if means.size > 1:
            if fit_mode == "WLS":
                w = 1.0 / np.maximum(stds, 1e-6)
                p = np.polyfit(means, stds, 1, w=w)
            else:
                p = np.polyfit(means, stds, 1)
            x = np.linspace(means.min(), means.max(), 100)
            y = np.polyval(p, x)
            plt.plot(
                x,
                y,
                linestyle="--",
                color=color,
                label=f"{gain:g}dB fit: y={p[0]:.3f}x+{p[1]:.3f}",
            )

    plt.xlabel("Mean (DN)")
    plt.ylabel("Std (DN)")
    plt.title("PRNU Regression")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_heatmap(
    data: np.ndarray,
    title: str,
    output_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw heatmap with optional value scaling."""

    plt.figure()
    plt.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(label="DN")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roi_area(
    images: Sequence[np.ndarray],
    rects: Sequence[Sequence[tuple[int, int, int, int]]],
    titles: Sequence[str],
    output_path: Path,
):
    """Visualize ROI rectangles on given images."""

    if len(images) != len(rects) or len(images) != len(titles):
        raise ValueError("images, rects and titles must have the same length")

    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i, (img, rs, title) in enumerate(zip(images, rects, titles), start=1):
        ax = plt.subplot(1, n, i)
        ax.imshow(img, cmap="gray")
        for l, t, w, h in rs:
            rect = plt.Rectangle(
                (l, t), w, h, edgecolor="r", facecolor="none", linewidth=1
            )
            ax.add_patch(rect)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
