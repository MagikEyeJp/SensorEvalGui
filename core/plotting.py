# core/plotting.py – Spec‑aware plotting utilities

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence
import logging

import matplotlib

matplotlib.use("Agg")  # avoid GUI backend so plotting works inside threads
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from utils.logger import log_memory_usage
from core import analysis

from utils import config as cfgutil

__all__ = [
    "plot_snr_vs_signal_multi",
    "plot_snr_vs_exposure",
    "plot_prnu_regression",
    "plot_heatmap",
    "plot_roi_area",
    "plot_noise_vs_signal_multi",
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


def plot_snr_vs_signal_multi(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
    interp_points: int | None = None,
    black_levels: Dict[float, float] | None = None,
) -> Figure | None:
    """Plot SNR–Signal curves for multiple gains."""
    logging.info("plot_snr_vs_signal_multi: output=%s", output_path)
    log_memory_usage("plot start: ")

    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    fig, ax_snr = plt.subplots()

    adc_full_scale = cfgutil.adc_full_scale(cfg)

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
        color = ax_snr._get_lines.get_next_color()
        ax_snr.loglog(
            sig,
            20 * np.log10(snr),
            linestyle="None",
            marker="o",
            color=color,
            label=f"{gain:g}dB",
        )

        bl = 0.0 if black_levels is None else float(black_levels.get(gain, 0.0))
        xs, snr_fit = analysis.fit_snr_signal_model(
            sig, snr, adc_full_scale, black_level=bl
        )
        ax_snr.loglog(
            xs, 20 * np.log10(snr_fit), linestyle="-", color=color, label="_nolegend_"
        )

    if all_signals:
        concat = np.concatenate(all_signals)
        x_min = float(concat.min())
        x_max = float(concat.max())
        if x_min == x_max:
            xs = np.asarray([x_min * 0.9, x_max * 1.1])
        else:
            xs = np.linspace(x_min, x_max, 200)
        ax_snr.loglog(xs, 20 * np.log10(np.sqrt(xs)), linestyle=":", label="Ideal √µ")

    ax_snr.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    ax_snr.set_xlabel("Signal (DN)")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_title("SNR vs Signal")
    ax_snr.grid(True, which="both")
    ax_snr.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        log_memory_usage("plot end: ")
        return fig
    plt.close(fig)
    log_memory_usage("plot end: ")
    return None


def plot_noise_vs_signal_multi(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
) -> Figure | None:
    """Plot Noise–Signal curves for multiple gains."""

    logging.info("plot_noise_vs_signal_multi: output=%s", output_path)
    fig, ax = plt.subplots()

    for gain, (sig, noise) in sorted(data.items()):
        sig = _validate_positive_finite(sig, "signal")
        noise = _validate_positive_finite(noise, "noise")
        if sig.size == 1 or noise.size == 1:
            sig = np.asarray([sig[0] * 0.9, sig[0] * 1.1])
            noise = np.asarray([noise[0] * 0.9, noise[0] * 1.1])
        ax.loglog(sig, noise, marker="o", linestyle="-", label=f"{gain:g}dB")

    ax.set_xlabel("Signal (DN)")
    ax.set_ylabel("Noise (DN)")
    ax.set_title("Noise vs Signal")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_snr_vs_exposure(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
) -> Figure | None:
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

    fig = plt.figure()
    sorted_items = sorted(data.items())
    ideal_gain = (
        0.0 if any(abs(g - 0.0) < 1e-6 for g, _ in sorted_items) else sorted_items[0][0]
    )

    for gain, (ratios, snr) in sorted_items:
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
        if gain == ideal_gain:
            base_idx = int(np.argmin(np.abs(ratios - 1.0)))
            base_ratio = ratios[base_idx]
            base_snr = snr[base_idx]
            ideal = base_snr * np.sqrt(ratios / base_ratio)
            ideal_db = 20 * np.log10(ideal)
            plt.semilogx(
                times,
                ideal_db,
                linestyle="--",
                color="k",
                label=f"{gain:g} dB Ideal √k",
            )
    plt.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xticks(xticks, label_strs, rotation=45)
    plt.xlabel("Exposure Time (ms)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Exposure")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_prnu_regression(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
) -> Figure | None:
    """Plot PRNU regression per gain with LS or WLS fit."""

    fig = plt.figure()
    fit_mode = cfg.get("processing", {}).get("prnu_fit", "LS").upper()
    squared = bool(cfg.get("plot", {}).get("prnu_squared", False))
    cmap = plt.cm.get_cmap("tab10")

    for idx, (gain, (means, stds)) in enumerate(sorted(data.items())):
        means = _validate_positive_finite(means, "mean")
        stds = _validate_positive_finite(stds, "std")
        color = cmap(idx % 10)
        x = means**2 if squared else means
        y = stds**2 if squared else stds
        plt.scatter(x, y, s=8, alpha=0.6, color=color, label=f"{gain:g}dB")
        if means.size > 1:
            if fit_mode == "WLS":
                weight = stds ** (2 if squared else 1)
                w = 1.0 / np.maximum(weight, 1e-6)
                p = np.polyfit(x, y, 1, w=w)
            else:
                p = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = np.polyval(p, x_fit)
            plt.plot(
                x_fit,
                y_fit,
                linestyle="--",
                color=color,
                label=f"{gain:g}dB fit: y={p[0]:.3f}x+{p[1]:.3f}",
            )

    xlabel = "Mean^2 (DN^2)" if squared else "Mean (DN)"
    ylabel = "Std^2 (DN^2)" if squared else "Std (DN)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("PRNU Regression")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_heatmap(
    data: np.ndarray,
    title: str,
    output_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    return_fig: bool = False,
) -> Figure | None:
    """Draw heatmap with optional value scaling."""

    fig = plt.figure()
    plt.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(label="DN")
    plt.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_roi_area(
    images: Sequence[np.ndarray],
    rects: Sequence[Sequence[tuple[int, int, int, int]]],
    titles: Sequence[str],
    output_path: Path,
    *,
    return_fig: bool = False,
) -> Figure | None:
    """Visualize ROI rectangles on given images."""

    if len(images) != len(rects) or len(images) != len(titles):
        raise ValueError("images, rects and titles must have the same length")

    n = len(images)
    fig = plt.figure(figsize=(4 * n, 4))
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
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None
