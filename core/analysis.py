# generated: 2025-05-18T11:25:00Z (auto)
# core/analysis.py – Spec‑aligned analysis (ROI, min_sig_factor, debug_stacks)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, List
import os
import logging

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
    "collect_mid_roi_snr",
    "collect_gain_snr_signal",
    "calculate_dn_at_snr",
    "calculate_snr_at_half",
    "calculate_dn_at_snr_one",
    "calculate_pseudo_prnu",
]

# ───────────────────────────── internal helpers


def _mask_from_rects(
    shape: Tuple[int, int], rects: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for l, t, w, h in rects:
        mask[t : t + h, l : l + w] = True
    return mask


# ───────────────────────────── public api


def extract_roi_stats(
    project_dir: Path | str, cfg: Dict[str, Any]
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Compute mean, standard deviation and SNR for each ROI.

    Parameters
    ----------
    project_dir:
        Root directory containing measurement folders.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    Dict[Tuple[float, float], Dict[str, float]]
        Mapping ``(gain_db, exposure_ratio)`` to ``{"mean", "std", "snr"}`` values.
    """
    project_dir = Path(project_dir)
    res: Dict[Tuple[float, float], Dict[str, float]] = {}

    chart_roi_file = project_dir / cfg["measurement"]["chart_roi_file"]
    flat_roi_file = project_dir / cfg["measurement"]["flat_roi_file"]
    chart_rects = load_rois(chart_roi_file)
    flat_rects = load_rois(flat_roi_file)

    snr_thresh = cfg["processing"].get("snr_threshold_dB", 10.0)
    min_sig_factor = cfg["processing"].get("min_sig_factor", 3.0)
    excl_low_snr = cfg["processing"].get("exclude_abnormal_snr", True)
    debug_stacks = cfg["output"].get("debug_stacks", False)

    for gain_db, gfold in cfgutil.gain_entries(cfg):
        for ratio, efold in cfgutil.exposure_entries(cfg):
            folder = project_dir / gfold / efold
            if not folder.is_dir():
                logging.info("Skipping missing folder: %s", folder)
                continue
            logging.info(
                "Processing folder %s (%.1f dB, %.3fx)", folder, gain_db, ratio
            )
            stack = load_image_stack(folder)
            if debug_stacks:
                tifffile.imwrite(folder / "stack_cache.tiff", stack)

            rects = chart_rects if "chart" in efold.lower() else flat_rects
            mask = _mask_from_rects(stack.shape[1:], rects)
            pix = stack[:, mask]
            mean = float(np.mean(pix))
            std = float(np.std(pix))
            if std == 0:
                logging.info("Skip due to zero std: %s", folder)
                continue
            if mean < min_sig_factor * std:
                logging.info(
                    "Skip due to low signal: mean %.2f < %.2f × %.2f",
                    mean,
                    min_sig_factor,
                    std,
                )
                continue
            snr = mean / std
            if excl_low_snr and snr < snr_thresh:
                logging.info(
                    "Skip due to low SNR: %.2f dB < %.2f dB",
                    20 * np.log10(snr),
                    snr_thresh,
                )
                continue
            res[(gain_db, ratio)] = {"mean": mean, "std": std, "snr": snr}
    logging.info("Collected %d ROI stats", len(res))
    return res


def extract_roi_table(
    project_dir: Path | str, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Return ROI statistics formatted for CSV output.

    Parameters
    ----------
    project_dir:
        Root directory containing measurement folders.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    List[Dict[str, Any]]
        Rows describing the ROI type, index, gain, exposure and statistics.
    """
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
                rows.append(
                    {
                        "ROI Type": roi_type,
                        "ROI No": i if roi_type == "grayscale" else "-",
                        "Gain (dB)": gain_db,
                        "Exposure": ratio,
                        "Mean": mean,
                        "Std": std,
                        "SNR (dB)": snr_db,
                    }
                )
    return rows


def collect_mid_roi_snr(
    rows: List[Dict[str, Any]], mid_index: int
) -> Dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return SNR curves for the grayscale ROI at ``mid_index``.

    Parameters
    ----------
    rows:
        Rows from :func:`extract_roi_table`.
    mid_index:
        Index of the ROI to analyse.

    Returns
    -------
    Dict[float, tuple[np.ndarray, np.ndarray]]
        Mapping of gain to arrays of exposure ratios and linear SNR values.
    """
    data: Dict[float, List[tuple[float, float]]] = {}
    for row in rows:
        if row.get("ROI Type") != "grayscale" or row.get("ROI No") != mid_index:
            continue
        gain = float(row["Gain (dB)"])
        ratio = float(row["Exposure"])
        mean = float(row["Mean"])
        std = float(row["Std"])
        if std == 0:
            continue
        data.setdefault(gain, []).append((ratio, mean / std))

    res: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gain, items in data.items():
        items.sort(key=lambda x: x[0])
        r, s = zip(*items)
        res[gain] = (np.array(r), np.array(s))
    return res


def collect_gain_snr_signal(
    rows: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return SNR curves indexed by signal level for each gain using ROI rows.

    Parameters
    ----------
    rows:
        ROI table rows from :func:`extract_roi_table`.
    cfg:
        Parsed configuration dictionary controlling filtering.

    Returns
    -------
    Dict[float, tuple[np.ndarray, np.ndarray]]
        Mapping of gain to arrays of signal levels and linear SNR values.
    """

    snr_thresh = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    min_sig_factor = cfg.get("processing", {}).get("min_sig_factor", 3.0)
    excl_low = cfg.get("processing", {}).get("exclude_abnormal_snr", True)

    data: Dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        mean = float(row.get("Mean", 0.0))
        std = float(row.get("Std", 0.0))
        if std == 0:
            continue
        if mean < min_sig_factor * std:
            continue
        snr = mean / std
        if excl_low and 20 * np.log10(snr) < snr_thresh:
            continue
        gain = float(row.get("Gain (dB)", 0.0))
        data.setdefault(gain, []).append((mean, snr))

    res: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gain, items in data.items():
        items.sort(key=lambda x: x[0])
        sig, s = zip(*items)
        res[gain] = (np.array(sig), np.array(s))
    return res


def calculate_snr_curve(
    signal: np.ndarray, noise: np.ndarray, cfg: Dict[str, Any]
) -> np.ndarray:
    """Compute the signal-to-noise ratio for each pixel.

    Parameters
    ----------
    signal, noise:
        Arrays of signal and noise values with matching shape.
    cfg:
        Configuration dictionary (unused).

    Returns
    -------
    np.ndarray
        Linear SNR calculated as ``signal / noise`` with ``NaN`` where noise is zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise == 0, np.nan, signal / noise)
    return snr


def calculate_dynamic_range(
    snr: np.ndarray, signal: np.ndarray, cfg: Dict[str, Any]
) -> float:
    """Estimate dynamic range from the SNR curve.

    Parameters
    ----------
    snr:
        Linear SNR values.
    signal:
        Corresponding DN levels.
    cfg:
        Configuration dictionary providing ``snr_threshold_dB``.

    Returns
    -------
    float
        Dynamic range in decibels based on the threshold crossing.
    """
    thresh_db = cfg["processing"].get("snr_threshold_dB", 20.0)
    thresh = 10 ** (thresh_db / 20)
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
        return float(np.sqrt(np.mean(values**2)))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "mad":
        med = np.median(values)
        return float(np.median(np.abs(values - med)))
    raise ValueError(mode)


def calculate_dark_noise(
    project_dir: Path | str, cfg: Dict[str, Any]
) -> Tuple[float, float]:
    """Calculate DSNU and read noise from a dark frame stack.

    Parameters
    ----------
    project_dir:
        Root directory of the project containing the dark images.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    Tuple[float, float]
        ``(dsnu, read_noise)`` in DN.
    """
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

    mode = int(cfg.get("processing", {}).get("read_noise_mode", 0))
    if mode == 1:
        diff = np.diff(stack, axis=0)
        read_noise_pix = np.std(diff, axis=0) / np.sqrt(2)
    else:
        read_noise_pix = np.std(stack, axis=0)
    read_noise = _reduce(read_noise_pix[mask], stat_mode)

    return dsnu, read_noise


def calculate_dark_noise_gain(
    project_dir: Path | str, gain_db: float, cfg: Dict[str, Any]
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Calculate dark noise metrics for a specific gain setting.

    Parameters
    ----------
    project_dir:
        Root project directory.
    gain_db:
        Sensor gain in decibels to evaluate.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    Tuple[float, float, np.ndarray, np.ndarray]
        ``(dsnu, read_noise, dsnu_map, read_noise_map)`` where maps match the image size.
    """
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
    mode = int(cfg.get("processing", {}).get("read_noise_mode", 0))
    if mode == 1:
        diff = np.diff(stack, axis=0)
        read_noise_map = np.std(diff, axis=0) / np.sqrt(2)
    else:
        read_noise_map = np.std(stack, axis=0)
    read_noise = _reduce(read_noise_map[mask], stat_mode)
    return dsnu, read_noise, dsnu_map, read_noise_map


def calculate_dn_sat(flat_stack: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Estimate the sensor saturation level in DN.

    Parameters
    ----------
    flat_stack:
        Stack of illuminated flat-field frames.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    float
        Detected saturation DN value.
    """
    p999 = float(np.percentile(flat_stack, 99.9))
    sat_factor = cfg.get("illumination", {}).get(
        "sat_factor",
        cfg.get("reference", {}).get("sat_factor", 0.95),
    )
    max_from_factor = float(np.max(flat_stack)) / max(sat_factor, 1e-6)
    adc_bits = int(cfg.get("sensor", {}).get("adc_bits", 16))
    adc_max = (1 << adc_bits) - 1
    method3 = adc_max * 0.90
    return max(p999, max_from_factor, method3)


def calculate_dynamic_range_dn(dn_sat: float, read_noise: float) -> float:
    """Compute dynamic range using DN values.

    Parameters
    ----------
    dn_sat:
        Saturation level in DN.
    read_noise:
        Measured read noise in DN.

    Returns
    -------
    float
        Dynamic range in decibels. Returns ``0.0`` if ``read_noise`` is zero.
    """
    if read_noise == 0:
        return 0.0
    return 20.0 * np.log10(dn_sat / read_noise)


def calculate_pseudo_prnu(
    flat_stack: np.ndarray,
    cfg: Dict[str, Any],
    rects: list[tuple[int, int, int, int]] | None = None,
) -> Tuple[float, np.ndarray]:
    """Estimate pseudo-PRNU within the specified ROI.

    Parameters
    ----------
    flat_stack:
        Stack of flat-field frames.
    cfg:
        Parsed configuration dictionary.
    rects:
        Optional list of ROI rectangles ``(left, top, width, height)``.

    Returns
    -------
    Tuple[float, np.ndarray]
        ``(pseudo_prnu_percent, residual_map)`` where the residual map matches the image shape.
    """

    mask = (
        _mask_from_rects(flat_stack.shape[1:], rects)
        if rects
        else np.ones(flat_stack.shape[1:], bool)
    )

    mean_frame = np.mean(flat_stack, axis=0)
    margin = cfg.get("processing", {}).get("mask_upper_margin")
    if margin is not None:
        dn_sat = calculate_dn_sat(flat_stack, cfg)
        mask &= mean_frame <= margin * dn_sat

    apply_gain = cfg.get("processing", {}).get("apply_gain_map", False)
    order = int(cfg.get("processing", {}).get("plane_fit_order", 0))

    def _fit_gain(frame: np.ndarray, mask: np.ndarray, order: int) -> np.ndarray:
        if order <= 0:
            c = float(np.mean(frame[mask]))
            return np.full_like(frame, c)
        y, x = np.indices(frame.shape)
        xm, ym = x[mask].ravel(), y[mask].ravel()
        z = frame[mask].ravel()
        cols = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                cols.append((xm**i) * (ym**j))
        A = np.vstack(cols).T
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        cols_full = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                cols_full.append((x**i) * (y**j))
        A_full = np.stack(cols_full, axis=0)
        fitted = np.tensordot(coef, A_full, axes=(0, 0))
        return fitted

    if apply_gain:
        gain_map = _fit_gain(mean_frame, mask, order)
        gain_map = np.where(gain_map == 0, 1e-6, gain_map)
        corrected = flat_stack / gain_map
        mean_frame = np.mean(corrected, axis=0)
        std_frame = np.std(corrected, axis=0)
    else:
        std_frame = np.std(flat_stack, axis=0)

    stat_mode = cfg.get("processing", {}).get("stat_mode", "rms")
    value = (
        _reduce(std_frame[mask], stat_mode) / max(mean_frame[mask].mean(), 1e-6) * 100.0
    )
    return value, std_frame


def calculate_system_sensitivity(
    flat_stack: np.ndarray,
    cfg: Dict[str, Any],
    rects: list[tuple[int, int, int, int]] | None = None,
    ratio: float = 1.0,
) -> float:
    """Compute system sensitivity in DN per irradiance and exposure time.

    Parameters
    ----------
    flat_stack:
        Stack of flat-field frames.
    cfg:
        Parsed configuration dictionary containing illumination info.
    rects:
        Optional ROI rectangles over which to compute the mean.

    ratio:
        Exposure multiplier relative to ``exposure_ms``.

    Returns
    -------
    float
        Sensitivity in DN / (µW·cm⁻²·s). Returns ``0.0`` if the denominator is zero.
    """

    mask = (
        _mask_from_rects(flat_stack.shape[1:], rects)
        if rects
        else np.ones(flat_stack.shape[1:], bool)
    )

    illum = cfg.get("illumination", {})
    power = float(illum.get("power_uW_cm2", 1.0))
    exposure_ms = float(illum.get("exposure_ms", 1.0))
    denom = power * exposure_ms * float(ratio) / 1000.0
    if denom == 0:
        return 0.0

    mean_dn = float(np.mean(flat_stack[:, mask]))
    return mean_dn / denom


def calculate_dn_at_snr(
    signal: np.ndarray, snr_lin: np.ndarray, threshold_db: float
) -> float:
    """Interpolate the DN value where the SNR reaches ``threshold_db``.

    Parameters
    ----------
    signal:
        Array of DN levels.
    snr_lin:
        Corresponding SNR values in linear scale.
    threshold_db:
        SNR threshold in decibels.

    Returns
    -------
    float
        Estimated DN at the given threshold or ``NaN`` if it is never reached.
    """
    thr_lin = 10 ** (threshold_db / 20.0)
    idx = np.where(snr_lin >= thr_lin)[0]
    if idx.size == 0:
        return float("nan")
    if idx[0] == 0:
        return float(signal[0])
    x0, x1 = signal[idx[0] - 1], signal[idx[0]]
    y0, y1 = snr_lin[idx[0] - 1], snr_lin[idx[0]]
    if y1 == y0:
        return float(x1)
    r = (thr_lin - y0) / (y1 - y0)
    return float(x0 + r * (x1 - x0))


def calculate_snr_at_half(
    signal: np.ndarray, snr_lin: np.ndarray, dn_sat: float
) -> float:
    """Return the SNR in dB at half of ``dn_sat``.

    Parameters
    ----------
    signal:
        DN levels sorted or unsorted.
    snr_lin:
        SNR values in linear scale.
    dn_sat:
        Saturation DN level.

    Returns
    -------
    float
        Interpolated SNR in decibels at ``dn_sat / 2`` or ``NaN`` if undefined.
    """
    if signal.size == 0:
        return float("nan")
    order = np.argsort(signal)
    sig_sorted = signal[order]
    snr_sorted = snr_lin[order]
    target = 0.5 * dn_sat
    snr_val = float(np.interp(target, sig_sorted, snr_sorted))
    if snr_val <= 0:
        return float("nan")
    return float(20.0 * np.log10(snr_val))


def calculate_dn_at_snr_one(signal: np.ndarray, snr_lin: np.ndarray) -> float:
    """Shortcut for :func:`calculate_dn_at_snr` with a threshold of 0 dB."""
    return calculate_dn_at_snr(signal, snr_lin, 0.0)
