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
from scipy.signal import savgol_filter

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


def _smooth_and_second_derivative(
    signal: np.ndarray,
    snr: np.ndarray,
    window: int = 5,
    poly: int = 2,
    interp_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed SNR and its second derivative using Savitzky-Golay.

    When ``interp_points`` is provided and greater than ``signal.size``, the
    ``signal`` and ``snr`` arrays are linearly interpolated to that length before
    smoothing. This can help produce smoother curves when the original data
    points are sparse.
    """

    idx = np.argsort(signal)
    sig = np.asarray(signal, dtype=float)[idx]
    s = np.asarray(snr, dtype=float)[idx]

    if interp_points is not None and interp_points > sig.size:
        xs = np.linspace(float(sig.min()), float(sig.max()), int(interp_points))
        ys = np.interp(xs, sig, s)
        sig = xs
        s = ys

    win = min(window, sig.size if sig.size % 2 else sig.size - 1)
    if win < poly + 2 or win < 3:
        s_smooth = s
    else:
        if win % 2 == 0:
            win -= 1
        s_smooth = savgol_filter(s, win, poly, mode="interp")

    d1 = np.gradient(s_smooth, sig)
    d2 = np.gradient(d1, sig)
    return sig, s_smooth, d2


def plot_snr_vs_signal(
    signal: np.ndarray,
    snr: np.ndarray,
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
) -> Figure | None:
    """Plot SNR–Signal curve (log–log) with ideal line and threshold."""
    signal = _validate_positive_finite(signal, "signal")
    snr = _validate_positive_finite(snr, "snr")
    if signal.size == 1 or snr.size == 1:
        # Avoid singular log scale when only one sample is present
        signal = np.asarray([signal[0] * 0.9, signal[0] * 1.1])
        snr = np.asarray([snr[0] * 0.9, snr[0] * 1.1])
    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    snr_db = 20 * np.log10(snr)
    fig = plt.figure()
    plt.loglog(signal, snr_db, marker="o", linestyle="-", label="Measured")
    plt.loglog(signal, 20 * np.log10(np.sqrt(signal)), linestyle=":", label="Ideal √µ")
    plt.axhline(thresh, color="r", linestyle="--", label=f"{thresh:g} dB")
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Signal")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_snr_vs_signal_multi(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    output_path: Path,
    *,
    return_fig: bool = False,
    show_derivative: bool = False,
    interp_points: int | None = None,
) -> Figure | None:
    """Plot SNR–Signal curves for multiple gains.

    When ``show_derivative`` is ``True`` an additional subplot is drawn below the
    main plot showing the second derivative of the smoothed SNR curve.
    """
    logging.info("plot_snr_vs_signal_multi: output=%s", output_path)
    log_memory_usage("plot start: ")

    thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    if show_derivative:
        fig, (ax_snr, ax_d2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    else:
        fig, ax_snr = plt.subplots()

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
        if interp_points is not None and interp_points > sig.size:
            xs = np.linspace(float(sig.min()), float(sig.max()), int(interp_points))
            snr = np.interp(xs, sig, snr)
            sig = xs
        snr_db = 20 * np.log10(snr)
        ax_snr.loglog(sig, snr_db, marker="o", linestyle="-", label=f"{gain:g}dB")
        if show_derivative:
            sig_s, snr_smooth, d2 = _smooth_and_second_derivative(
                sig, snr, interp_points=interp_points
            )
            color = ax_snr.get_lines()[-1].get_color()
            ax_snr.loglog(
                sig_s,
                20 * np.log10(snr_smooth),
                linestyle="--",
                color=color,
            )
            ax_d2.semilogx(
                sig_s, d2, marker=".", linestyle="-", color=color, label=f"{gain:g}dB"
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
    if show_derivative:
        ax_d2.set_xlabel("Signal (DN)")
        ax_d2.set_ylabel("d²SNR / dDN²")
        ax_d2.set_title("Second Derivative")
        ax_d2.grid(True, which="both")
        ax_d2.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    if return_fig:
        log_memory_usage("plot end: ")
        return fig
    plt.close(fig)
    log_memory_usage("plot end: ")
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
