# generated: 2025-05-18T11:25:00Z (auto)
# core/analysis.py – Spec‑aligned analysis (ROI, min_sig_factor, debug_stacks)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Callable
import os
import logging

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import tifffile

from utils import config as cfgutil
from utils.roi import load_rois
from core.loader import load_image_stack

__all__ = [
    "clear_cache",
    "extract_roi_stats",
    "extract_roi_stats_gainmap",
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
    "collect_gain_noise_signal",
    "collect_prnu_points",
    "calculate_dn_at_snr",
    "calculate_snr_at_half",
    "calculate_dn_at_snr_one",
    "fit_gain_map_akima",
    "fit_gain_map_hermite",
    "fit_gain_map_rbf",
    "fit_gain_map",
    "get_gain_map",
    "clipped_snr_model",
    "fit_clipped_snr_model",
    "fit_three_region_snr_model",
    "calculate_gain_map",
    "calculate_pseudo_prnu",
    "calculate_prnu_residual",
]

# ───────────────────────────── cache helpers

# Cache for loaded image stacks and calculated ROI statistics to
# avoid repeated disk I/O and computations when functions are called
# multiple times during a single session.
_stack_cache: Dict[Path, np.ndarray] = {}
_stats_cache: Dict[tuple[Path, float, float, bool], Dict[str, float]] = {}
_dark_cache: Dict[tuple[Path, float], Tuple[float, float, np.ndarray, np.ndarray, float]] = {}


def clear_cache() -> None:
    """Empty cached image stacks and ROI statistics."""
    _stack_cache.clear()
    _stats_cache.clear()
    _dark_cache.clear()


def _load_stack_cached(folder: Path) -> np.ndarray:
    stack = _stack_cache.get(folder)
    if stack is None:
        stack = load_image_stack(folder)
        _stack_cache[folder] = stack
    return stack


# ───────────────────────────── internal helpers


def _mask_from_rects(shape: Tuple[int, int], rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for l, t, w, h in rects:
        mask[t : t + h, l : l + w] = True
    return mask


def _gain_fit_mask(
    stack: np.ndarray,
    cfg: Dict[str, Any],
    *,
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
    dn_sat: float | None = None,
) -> np.ndarray:
    """Return mask for gain-map fitting using ``stack`` and config margins."""

    proc = cfg.get("processing", {})
    lower = proc.get("mask_lower_margin")
    upper = proc.get("mask_upper_margin")
    if lower is None and upper is None:
        return np.ones(stack.shape[1:], bool)

    mean_frame = np.mean(stack, axis=0)
    if dn_sat is None:
        dn_sat = calculate_dn_sat(stack, cfg, noise_signal)
    lo = 0.0 if lower is None else float(lower)
    hi = float("inf") if upper is None else float(upper)
    return (mean_frame >= lo * dn_sat) & (mean_frame <= hi * dn_sat)


def fit_gain_map_rbf(
    frame: np.ndarray,
    mask: np.ndarray,
    smooth: float = 0.0,
    function: str = "thin_plate",
    *,
    subsample_step: int = 1,
    subsample_method: str = "uniform",
) -> np.ndarray:
    """Return RBF-interpolated gain map for ``frame`` using ``mask`` pixels.

    ``subsample_step`` > 1 reduces the number of sample points by selecting
    every ``n``th pixel before fitting, which lowers the memory cost.
    """

    try:
        from scipy.interpolate import Rbf
    except Exception as exc:  # pragma: no cover - runtime dependency missing
        raise RuntimeError("scipy is required for RBF fitting") from exc

    step = max(int(subsample_step), 1)
    y_full, x_full = np.indices(frame.shape)

    if step > 1:
        if subsample_method == "uniform":
            frame_sub = frame[::step, ::step]
            mask_sub = mask[::step, ::step]
            y_sub, x_sub = np.indices(frame_sub.shape)
            xm = (x_sub[mask_sub] * step).ravel()
            ym = (y_sub[mask_sub] * step).ravel()
            z = frame_sub[mask_sub].ravel()
        else:
            coords = np.argwhere(mask)
            n = max(1, coords.shape[0] // (step**2))
            idx = np.random.choice(coords.shape[0], n, replace=False)
            ym = coords[idx, 0]
            xm = coords[idx, 1]
            z = frame[ym, xm]
    else:
        xm = x_full[mask].ravel()
        ym = y_full[mask].ravel()
        z = frame[mask].ravel()

    rbf = Rbf(xm, ym, z, function=function, smooth=smooth)
    fitted = rbf(x_full, y_full)

    fitted = np.where(fitted == 0, 1e-6, fitted)
    gain_max = np.max(fitted[mask])
    fitted /= max(gain_max, 1e-6)
    return fitted


def fit_gain_map_akima(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    subsample_step: int = 1,
    subsample_method: str = "uniform",
) -> np.ndarray:
    """Return gain map using Akima-based interpolation respecting ``mask``."""

    try:
        from scipy.interpolate import griddata
    except Exception as exc:  # pragma: no cover - runtime dependency missing
        raise RuntimeError("scipy is required for Akima fitting") from exc

    step = max(int(subsample_step), 1)
    y_full, x_full = np.indices(frame.shape)

    if step > 1:
        if subsample_method == "uniform":
            mask_sub = mask[::step, ::step]
            frame_sub = frame[::step, ::step]
            y_sub, x_sub = np.indices(frame_sub.shape)
            pts = np.stack([y_sub[mask_sub] * step, x_sub[mask_sub] * step], axis=1)
            vals = frame_sub[mask_sub].ravel()
        else:
            coords = np.argwhere(mask)
            n = max(1, coords.shape[0] // (step**2))
            idx = np.random.choice(coords.shape[0], n, replace=False)
            pts = coords[idx]
            vals = frame[pts[:, 0], pts[:, 1]]
    else:
        pts = np.argwhere(mask)
        vals = frame[mask]

    fitted = griddata(pts, vals, (y_full, x_full), method="cubic")
    missing = np.isnan(fitted)
    if np.any(missing):
        fitted[missing] = griddata(pts, vals, (y_full[missing], x_full[missing]), method="nearest")

    fitted = np.where(fitted == 0, 1e-6, fitted)
    gain_max = np.max(fitted[mask])
    fitted /= max(gain_max, 1e-6)
    return fitted


def fit_gain_map_hermite(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    subsample_step: int = 1,
    subsample_method: str = "uniform",
) -> np.ndarray:
    """Return gain map using Hermite-based interpolation respecting ``mask``."""

    try:
        from scipy.interpolate import griddata
    except Exception as exc:  # pragma: no cover - runtime dependency missing
        raise RuntimeError("scipy is required for Hermite fitting") from exc

    step = max(int(subsample_step), 1)
    y_full, x_full = np.indices(frame.shape)

    if step > 1:
        if subsample_method == "uniform":
            mask_sub = mask[::step, ::step]
            frame_sub = frame[::step, ::step]
            y_sub, x_sub = np.indices(frame_sub.shape)
            pts = np.stack([y_sub[mask_sub] * step, x_sub[mask_sub] * step], axis=1)
            vals = frame_sub[mask_sub].ravel()
        else:
            coords = np.argwhere(mask)
            n = max(1, coords.shape[0] // (step**2))
            idx = np.random.choice(coords.shape[0], n, replace=False)
            pts = coords[idx]
            vals = frame[pts[:, 0], pts[:, 1]]
    else:
        pts = np.argwhere(mask)
        vals = frame[mask]

    fitted = griddata(pts, vals, (y_full, x_full), method="cubic")
    missing = np.isnan(fitted)
    if np.any(missing):
        fitted[missing] = griddata(pts, vals, (y_full[missing], x_full[missing]), method="nearest")

    fitted = np.where(fitted == 0, 1e-6, fitted)
    gain_max = np.max(fitted[mask])
    fitted /= max(gain_max, 1e-6)
    return fitted


def fit_gain_map(
    frame: np.ndarray,
    mask: np.ndarray,
    order: int,
    *,
    method: str = "poly",
    subsample_step: int = 1,
    subsample_method: str = "uniform",
) -> np.ndarray:
    """Return a plane-fit gain map for ``frame`` using ``mask`` pixels.

    The returned map is normalized so that the brightest masked pixel is ``1``.

    Parameters
    ----------
    frame:
        2-D array of pixel values to fit.
    mask:
        Boolean mask selecting pixels to include in the fit.
    order:
        Polynomial order of the surface. ``0`` gives a constant plane.

    Returns
    -------
    np.ndarray
        Fitted gain map matching ``frame`` shape.
    """
    logging.info(
        "Fitting gain map: method=%s order=%d mask_pixels=%d",
        method,
        order,
        int(mask.sum()),
    )

    if method == "rbf":
        return fit_gain_map_rbf(
            frame,
            mask,
            smooth=float(order),
            subsample_step=subsample_step,
            subsample_method=subsample_method,
        )
    if method == "akima":
        return fit_gain_map_akima(
            frame,
            mask,
            subsample_step=subsample_step,
            subsample_method=subsample_method,
        )
    if method == "hermite":
        return fit_gain_map_hermite(
            frame,
            mask,
            subsample_step=subsample_step,
            subsample_method=subsample_method,
        )

    if order <= 0:
        c = float(np.mean(frame[mask]))
        fitted = np.full_like(frame, c)
    else:
        step = max(int(subsample_step), 1)
        y, x = np.indices(frame.shape)
        if step > 1:
            if subsample_method == "uniform":
                frame_sub = frame[::step, ::step]
                mask_sub = mask[::step, ::step]
                y_sub, x_sub = np.indices(frame_sub.shape)
                xm = (x_sub[mask_sub] * step).ravel()
                ym = (y_sub[mask_sub] * step).ravel()
                z = frame_sub[mask_sub].ravel()
            else:
                coords = np.argwhere(mask)
                n = max(1, coords.shape[0] // (step**2))
                idx = np.random.choice(coords.shape[0], n, replace=False)
                ym = coords[idx, 0]
                xm = coords[idx, 1]
                z = frame[ym, xm]
        else:
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

    logging.info(
        "Gain map pre-normalization: min=%.3g max=%.3g",
        fitted[mask].min(),
        fitted[mask].max(),
    )

    fitted = np.where(fitted == 0, 1e-6, fitted)
    gain_max = np.max(fitted[mask])
    fitted /= max(gain_max, 1e-6)
    logging.info(
        "Gain map normalized: gain_max=%.3g min=%.3g max=%.3g",
        gain_max,
        fitted.min(),
        fitted.max(),
    )
    return fitted


def get_gain_map(
    cfg: Dict[str, Any],
    mask: np.ndarray | None,
    project_dir: Path | str | None = None,
    gain_db: float | None = None,
    stack: np.ndarray | None = None,
    *,
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
    dn_sat: float | None = None,
) -> np.ndarray | None:
    """Return gain map according to ``cfg['processing']['gain_map_mode']``.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary.
    mask:
        Boolean mask for plane fitting. If ``None``, it is generated from
        ``stack`` using ``mask_lower_margin`` and ``mask_upper_margin``.
    project_dir:
        Project directory to load reference flats for ``flat_`` modes.
    gain_db:
        Gain level used to locate reference flats.
    stack:
        Image stack used when the mode is ``self_fit`` or to avoid reloading
        reference flats.
    noise_signal:
        Optional tuple of ``(signal, noise)`` arrays used to estimate ``dn_sat``
        when clipping margins are applied.
    dn_sat:
        Precomputed saturation level. If ``None``, it is calculated as needed
        using ``noise_signal``.

    Returns
    -------
    np.ndarray | None
        Gain map normalized by its maximum, or ``None`` if no correction is
        requested.
    """

    mode = cfg.get("processing", {}).get("gain_map_mode", "none")
    if mode == "none":
        logging.info("Gain map mode is 'none'")
        return None

    order = int(cfg.get("processing", {}).get("plane_fit_order", 0))
    method = cfg.get("processing", {}).get("gain_fit_method", "poly")
    logging.info(
        "get_gain_map: mode=%s method=%s order=%d stack_provided=%s",
        mode,
        method,
        order,
        stack is not None,
    )

    if mode == "self_fit" or project_dir is None or gain_db is None:
        if stack is None:
            raise ValueError("stack is required for self_fit gain map")
        mean_src = np.mean(stack, axis=0)
        logging.info(
            "Using provided stack for gain map; mean=%.3g min=%.3g max=%.3g",
            float(np.mean(mean_src)),
            float(np.min(mean_src)),
            float(np.max(mean_src)),
        )
    else:
        if stack is None:
            flat_folder = cfgutil.find_gain_folder(project_dir, gain_db, cfg) / cfg["measurement"].get(
                "flat_lens_folder", "LensFlat"
            )
            stack = load_image_stack(flat_folder)
            mean_src = np.mean(stack, axis=0)
            logging.info(
                "Loaded reference flat stack from %s; mean=%.3g min=%.3g max=%.3g",
                flat_folder,
                float(np.mean(mean_src)),
                float(np.min(mean_src)),
                float(np.max(mean_src)),
            )
        else:
            mean_src = np.mean(stack, axis=0)
            logging.info(
                "Using provided flat stack; mean=%.3g min=%.3g max=%.3g",
                float(np.mean(mean_src)),
                float(np.min(mean_src)),
                float(np.max(mean_src)),
            )

    clip_margin = bool(cfg.get("processing", {}).get("gain_clip_margin", False))

    dn_sat_val = dn_sat

    if clip_margin:
        proc = cfg.get("processing", {})
        lower = proc.get("mask_lower_margin")
        upper = proc.get("mask_upper_margin")
        if lower is not None or upper is not None:
            if dn_sat_val is None:
                dn_sat_val = calculate_dn_sat(stack, cfg, noise_signal)
            lo = 0.0 if lower is None else float(lower) * dn_sat_val
            hi = float("inf") if upper is None else float(upper) * dn_sat_val
            mean_src = np.clip(mean_src, lo, hi)
        mask = np.ones_like(mean_src, bool)
    elif mask is None:
        mask = _gain_fit_mask(stack, cfg, noise_signal=noise_signal, dn_sat=dn_sat_val)

    if mode == "flat_frame":
        gain_map = mean_src
        gain_map = np.where(gain_map == 0, 1e-6, gain_map)
        gain_max = np.max(gain_map[mask])
        gain_map /= max(gain_max, 1e-6)
        logging.info(
            "Flat frame mode: gain_max=%.3g min=%.3g max=%.3g",
            gain_max,
            float(np.min(gain_map)),
            float(np.max(gain_map)),
        )
    else:
        proc = cfg.get("processing", {})
        subsample = int(proc.get("fit_subsample_step", 1))
        subsample_method_cfg = proc.get("subsample_method", "uniform")
        gain_map = fit_gain_map(
            mean_src,
            mask,
            order,
            method=method,
            subsample_step=subsample,
            subsample_method=subsample_method_cfg,
        )
        logging.info(
            "Fit gain map result: min=%.3g max=%.3g",
            float(np.min(gain_map)),
            float(np.max(gain_map)),
        )

    logging.info(
        "Gain map returned: mean=%.3g min=%.3g max=%.3g",
        float(np.mean(gain_map)),
        float(np.min(gain_map)),
        float(np.max(gain_map)),
    )
    return gain_map


# ───────────────────────────── public api


def extract_roi_stats(
    project_dir: Path | str,
    cfg: Dict[str, Any],
    status: Optional[Callable[[str], None]] = None,
    *,
    use_gain_map: bool = False,
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

    mid_idx = cfg.get("reference", {}).get("roi_mid_index", cfg.get("measurement", {}).get("roi_mid_index", 5))

    snr_thresh = cfg["processing"].get("snr_threshold_dB", 10.0)
    min_sig_factor = cfg["processing"].get("min_sig_factor", 3.0)
    excl_low_snr = cfg["processing"].get("exclude_abnormal_snr", True)
    debug_stacks = cfg["output"].get("debug_stacks", False)

    mode = cfg.get("processing", {}).get("gain_map_mode", "none")
    apply_gain = use_gain_map and mode != "none"
    flat_cache: Dict[float, np.ndarray] = {}

    for gain_db, gfold in cfgutil.gain_entries(cfg):
        for ratio, efold in cfgutil.exposure_entries(cfg):
            folder = project_dir / gfold / efold
            if not folder.is_dir():
                logging.info("Skipping missing folder: %s", folder)
                continue
            logging.info("Processing folder %s (%.1f dB, %.3fx)", folder, gain_db, ratio)
            if status:
                status(f"Loading images for gain {gain_db:.1f} dB")
            stack = _load_stack_cached(folder)
            if debug_stacks:
                tifffile.imwrite(folder / "stack_cache.tiff", stack)

            corrected = stack
            if apply_gain:
                flat_stack = None
                if mode != "self_fit":
                    flat_stack = flat_cache.get(gain_db)
                    if flat_stack is None:
                        flat_folder = cfgutil.find_gain_folder(project_dir, gain_db, cfg) / cfg["measurement"].get(
                            "flat_lens_folder", "LensFlat"
                        )
                        if status:
                            status(f"Loading flat frames for gain {gain_db:.1f} dB")
                        flat_stack = _load_stack_cached(flat_folder)
                        flat_cache[gain_db] = flat_stack
                stack_fit = stack if mode == "self_fit" else flat_stack
                gain_map = get_gain_map(
                    cfg,
                    None,
                    project_dir=project_dir,
                    gain_db=gain_db,
                    stack=stack_fit,
                )
                corrected = stack if gain_map is None else stack / gain_map

            rects = chart_rects if "chart" in efold.lower() else flat_rects
            idx = mid_idx if "chart" in efold.lower() else 0
            if idx >= len(rects):
                idx = 0
            mask = _mask_from_rects(stack.shape[1:], [rects[idx]])

            cache_key = (folder, gain_db, ratio, apply_gain)
            cached = _stats_cache.get(cache_key)
            if cached:
                res[(gain_db, ratio)] = cached
                continue

            pix = corrected[:, mask]
            stat_mode = cfg["processing"].get("stat_mode", "rms")
            mean = float(np.mean(pix))
            noise_pix = np.std(pix, axis=0)
            std = float(_reduce(noise_pix, stat_mode))
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
            vals = {"mean": mean, "std": std, "snr": snr}
            res[(gain_db, ratio)] = vals
            _stats_cache[cache_key] = vals
    logging.info(
        "Collected %d ROI stats%s",
        len(res),
        " (gain-corrected)" if apply_gain else "",
    )
    return res


def extract_roi_stats_gainmap(
    project_dir: Path | str,
    cfg: Dict[str, Any],
    status: Optional[Callable[[str], None]] = None,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Compute ROI stats with gain-map correction applied."""
    return extract_roi_stats(
        project_dir,
        cfg,
        status=status,
        use_gain_map=True,
    )


def extract_roi_table(project_dir: Path | str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            stat_mode = cfg["processing"].get("stat_mode", "rms")
            for i, r in enumerate(rects):
                mask = _mask_from_rects(stack.shape[1:], [r])
                pix = stack[:, mask]
                mean = float(np.mean(pix))
                noise_pix = np.std(pix, axis=0)
                std = float(_reduce(noise_pix, stat_mode))
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


def collect_mid_roi_snr(rows: List[Dict[str, Any]], mid_index: int) -> Dict[float, tuple[np.ndarray, np.ndarray]]:
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
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    black_levels: Dict[float, float] | None = None,
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

    # Filtering parameters are read for backward compatibility but are not used
    # because all ROI points should contribute to the SNR–Signal plot.
    _ = cfg.get("processing", {}).get("snr_threshold_dB", 10.0)
    _ = cfg.get("processing", {}).get("min_sig_factor", 3.0)
    _ = cfg.get("processing", {}).get("exclude_abnormal_snr", True)

    data: Dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        mean = float(row.get("Mean", 0.0))
        std = float(row.get("Std", 0.0))
        if std == 0:
            continue
        gain = float(row.get("Gain (dB)", 0.0))
        black = 0.0 if black_levels is None else float(black_levels.get(gain, 0.0))
        snr = (mean - black) / std
        data.setdefault(gain, []).append((mean, snr))

    res: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gain, items in data.items():
        items.sort(key=lambda x: x[0])
        sig, s = zip(*items)
        res[gain] = (np.array(sig), np.array(s))
    return res


def collect_gain_noise_signal(
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    black_levels: Dict[float, float] | None = None,
) -> Dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return noise curves indexed by signal level for each gain using ROI rows."""

    data: Dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        mean = float(row.get("Mean", 0.0))
        std = float(row.get("Std", 0.0))
        if std == 0:
            continue
        gain = float(row.get("Gain (dB)", 0.0))
        data.setdefault(gain, []).append((mean, std))

    res: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gain, items in data.items():
        items.sort(key=lambda x: x[0])
        sig, n = zip(*items)
        res[gain] = (np.array(sig), np.array(n))
    return res


def collect_prnu_points(
    stats: Dict[tuple[float, float], Dict[str, float]],
) -> Dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return mean/std arrays per gain for PRNU regression."""

    data: Dict[float, list[tuple[float, float]]] = {}
    for (gain, _), vals in stats.items():
        std = float(vals.get("std", 0.0))
        if std == 0:
            continue
        mean = float(vals.get("mean", 0.0))
        data.setdefault(gain, []).append((mean, std))

    res: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gain, items in data.items():
        items.sort(key=lambda x: x[0])
        m, s = zip(*items)
        res[gain] = (np.array(m), np.array(s))
    return res


def calculate_snr_curve(signal: np.ndarray, noise: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
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


def calculate_dynamic_range(snr: np.ndarray, signal: np.ndarray, cfg: Dict[str, Any]) -> float:
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


def calculate_dark_noise(project_dir: Path | str, cfg: Dict[str, Any]) -> Tuple[float, float]:
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
    project_dir: Path | str,
    gain_db: float,
    cfg: Dict[str, Any],
    status: Optional[Callable[[str], None]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
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
    Tuple[float, float, np.ndarray, np.ndarray, float]
        ``(dsnu, read_noise, dsnu_map, read_noise_map, black_level)`` where maps match the image size.
    """
    project_dir = Path(project_dir)
    cache_key = (project_dir, float(gain_db))
    cached = _dark_cache.get(cache_key)
    if cached is not None:
        return cached

    gain_folder = cfgutil.find_gain_folder(project_dir, gain_db, cfg)
    dark_folder = gain_folder / cfg["measurement"].get("dark_folder", "dark")
    if status:
        status(f"Loading dark frames for gain {gain_db:.1f} dB")
    stack = load_image_stack(dark_folder)

    roi_file = project_dir / cfg["measurement"].get("flat_roi_file")
    rects = load_rois(roi_file)
    mask = _mask_from_rects(stack.shape[1:], rects)

    stat_mode = cfg["processing"].get("stat_mode", "rms")
    mean_frame = np.mean(stack, axis=0)
    black_level = float(np.mean(mean_frame[mask]))
    dsnu_map = mean_frame - black_level
    dsnu = _reduce(dsnu_map[mask], stat_mode)
    mode = int(cfg.get("processing", {}).get("read_noise_mode", 0))
    if mode == 1:
        diff = np.diff(stack, axis=0)
        read_noise_map = np.std(diff, axis=0) / np.sqrt(2)
    else:
        read_noise_map = np.std(stack, axis=0)
    read_noise = _reduce(read_noise_map[mask], stat_mode)
    result = (dsnu, read_noise, dsnu_map, read_noise_map, black_level)
    _dark_cache[cache_key] = result
    return result


def _estimate_sat_from_snr(signal: np.ndarray, snr: np.ndarray) -> float:
    """Return DN level where the SNR curve drops sharply.

    The function applies a spline fit to the SNR values, computes the second
    derivative, and locates the point with the maximum change starting from the
    highest signal level.

    Parameters
    ----------
    signal:
        Signal levels sorted in ascending order.
    snr:
        Corresponding linear SNR values.

    Returns
    -------
    float
        Estimated DN value where the curve bends. ``NaN`` on failure.
    """

    if signal.size < 3 or snr.size != signal.size:
        return float("nan")

    idx = np.argsort(signal)
    sig = np.asarray(signal, dtype=float)[idx]
    s = np.asarray(snr, dtype=float)[idx]
    logging.debug(
        "_estimate_sat_from_snr: sorted signal=%s",
        np.array2string(sig, precision=3, threshold=10),
    )
    logging.debug(
        "_estimate_sat_from_snr: sorted snr=%s",
        np.array2string(s, precision=3, threshold=10),
    )

    diffs = np.diff(sig)
    close_idx = np.where(np.abs(diffs) < 1.0)[0]
    if close_idx.size > 0:
        logging.debug(
            "_estimate_sat_from_snr: close signal points at idx %s -> diffs=%s",
            close_idx.tolist(),
            np.array2string(diffs[close_idx], precision=3, threshold=10),
        )

        # merge consecutive points closer than 1 DN by averaging
        merged_sig = []
        merged_s = []
        i = 0
        while i < sig.size:
            j = i + 1
            while j < sig.size and abs(sig[j] - sig[j - 1]) < 1.0:
                j += 1
            if j - i > 1:
                merged_sig.append(float(np.mean(sig[i:j])))
                merged_s.append(float(np.mean(s[i:j])))
            else:
                merged_sig.append(float(sig[i]))
                merged_s.append(float(s[i]))
            i = j
        sig = np.asarray(merged_sig)
        s = np.asarray(merged_s)

    # remove any remaining duplicates exactly equal
    uniq_sig, inv_idx = np.unique(sig, return_inverse=True)
    if uniq_sig.size != sig.size:
        uniq_s = [float(np.mean(s[inv_idx == i])) for i in range(uniq_sig.size)]
        sig = uniq_sig
        s = np.asarray(uniq_s)

    # try logistic model fit first using high-signal region
    if sig.size >= 4:

        def _logistic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
            return a + b / (1.0 + np.exp(-(x - c) / d))

        thresh = 0.6 * np.max(sig)
        mask = sig >= thresh
        if np.count_nonzero(mask) >= 4:
            sig_fit = sig[mask]
            s_fit = s[mask]
            init = [
                float(s_fit[-1]),
                float(s_fit[0] - s_fit[-1]),
                float(sig_fit[len(sig_fit) // 2]),
                1.0,
            ]
            try:
                popt, _ = curve_fit(_logistic, sig_fit, s_fit, p0=init, maxfev=10000)
                est = float(popt[2])
                if np.isfinite(est):
                    logging.debug("_estimate_sat_from_snr: logistic fit est=%.3f", est)
                    return est
            except Exception as exc:
                logging.debug("_estimate_sat_from_snr: logistic fit failed due to %s", exc)

    if sig.size >= 4:
        s_val = 0.2
        try:
            spline = UnivariateSpline(sig, s, s=s_val, k=3)
        except Exception as exc:  # pragma: no cover - should not normally fail
            s_val = float(sig.size) * 0.1
            logging.debug(
                "_estimate_sat_from_snr: spline retry with s=%.3f due to %s",
                s_val,
                exc,
            )
            spline = UnivariateSpline(sig, s, s=s_val, k=3)
        d2 = spline.derivative(2)(sig)
        logging.debug(
            "_estimate_sat_from_snr: second derivative=%s",
            np.array2string(d2, precision=3, threshold=10),
        )
    else:
        d1 = np.gradient(s, sig)
        d2 = np.gradient(d1, sig)
        logging.debug(
            "_estimate_sat_from_snr: second derivative (grad)=%s",
            np.array2string(d2, precision=3, threshold=10),
        )

    max_val = np.max(d2)
    idxs = np.where(np.isclose(d2, max_val, rtol=1e-6, atol=0.0))[0]
    if idxs.size == 0:
        logging.debug("_estimate_sat_from_snr: no maxima found")
        return float("nan")
    pos = sig[int(idxs[-1])]
    logging.debug(
        "_estimate_sat_from_snr: max d2 at index %d -> pos=%.3f val=%.3f",
        int(idxs[-1]),
        pos,
        max_val,
    )
    return float(pos)


def _estimate_sat_from_noise(signal: np.ndarray, noise: np.ndarray) -> float:
    """Return DN level where the noise curve starts to drop sharply."""

    if signal.size < 3 or noise.size != signal.size:
        return float("nan")

    idx = np.argsort(signal)
    sig = np.asarray(signal, dtype=float)[idx]
    n = np.asarray(noise, dtype=float)[idx]

    diffs = np.diff(sig)
    close_idx = np.where(np.abs(diffs) < 1.0)[0]
    if close_idx.size > 0:
        merged_sig = []
        merged_n = []
        i = 0
        while i < sig.size:
            j = i + 1
            while j < sig.size and abs(sig[j] - sig[j - 1]) < 1.0:
                j += 1
            if j - i > 1:
                merged_sig.append(float(np.mean(sig[i:j])))
                merged_n.append(float(np.mean(n[i:j])))
            else:
                merged_sig.append(float(sig[i]))
                merged_n.append(float(n[i]))
            i = j
        sig = np.asarray(merged_sig)
        n = np.asarray(merged_n)

    uniq_sig, inv_idx = np.unique(sig, return_inverse=True)
    if uniq_sig.size != sig.size:
        uniq_n = [float(np.mean(n[inv_idx == i])) for i in range(uniq_sig.size)]
        sig = uniq_sig
        n = np.asarray(uniq_n)

    if sig.size >= 4:
        try:
            spline = UnivariateSpline(sig, n, s=0.2, k=3)
            d1 = spline.derivative(1)(sig)
        except Exception:
            d1 = np.gradient(n, sig)
    else:
        d1 = np.gradient(n, sig)

    idx_drop = int(np.argmin(d1))
    return float(sig[idx_drop])


def calculate_dn_sat(
    flat_stack: np.ndarray,
    cfg: Dict[str, Any],
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
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
    logging.info("calculate_dn_sat: p999=%.3f", p999)

    if noise_signal is not None:
        sig, noise = noise_signal
        est = _estimate_sat_from_noise(np.asarray(sig), np.asarray(noise))
        logging.info("calculate_dn_sat: noise_est=%.3f", est)
    else:
        est = float("nan")
        logging.info("calculate_dn_sat: noise_signal not provided")

    if not np.isfinite(est):
        mean_frame = np.mean(flat_stack, axis=0)
        flat_sorted = np.sort(mean_frame.ravel())
        count = max(1, int(flat_sorted.size * 0.01))
        est = float(np.mean(flat_sorted[-count:]))
        logging.info("calculate_dn_sat: fallback_est=%.3f", est)
    sat_factor = cfg.get("illumination", {}).get(
        "sat_factor",
        cfg.get("reference", {}).get("sat_factor", 0.95),
    )

    adc_bits = int(cfg.get("sensor", {}).get("adc_bits", 16))
    lsb_shift = int(cfg.get("sensor", {}).get("lsb_shift", 0))
    adc_full_scale = ((1 << adc_bits) - 1) * (1 << lsb_shift)

    # Reference threshold based on the configured saturation factor
    reference_thresh = adc_full_scale * sat_factor

    logging.info(
        "calculate_dn_sat: sat_factor=%.3f adc_full_scale=%d reference_thresh=%.3f",
        sat_factor,
        adc_full_scale,
        reference_thresh,
    )

    dn_sat = max(est, p999, reference_thresh)
    dn_sat_final = min(dn_sat, adc_full_scale)
    logging.info(
        "calculate_dn_sat: chosen=max(est=%.3f, p999=%.3f, ref=%.3f)=%.3f -> %.3f",
        est,
        p999,
        reference_thresh,
        dn_sat,
        dn_sat_final,
    )
    return dn_sat_final


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
    project_dir: Path | str | None = None,
    gain_db: float | None = None,
    *,
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
    dn_sat: float | None = None,
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
    noise_signal:
        Optional ``(signal, noise)`` tuple used to estimate ``dn_sat`` when
        applying mask margins.
    dn_sat:
        Precomputed saturation level. If ``None``, calculated from
        ``noise_signal`` as needed.

    Returns
    -------
    Tuple[float, np.ndarray]
        ``(pseudo_prnu_percent, residual_map)`` where the residual map matches the image shape.
    """

    mask = _mask_from_rects(flat_stack.shape[1:], rects) if rects else np.ones(flat_stack.shape[1:], bool)

    mean_frame = np.mean(flat_stack, axis=0)
    margin = cfg.get("processing", {}).get("mask_upper_margin")
    dn_sat_val = dn_sat
    if margin is not None:
        if dn_sat_val is None:
            dn_sat_val = calculate_dn_sat(flat_stack, cfg, noise_signal)
        mask &= mean_frame <= margin * dn_sat_val

    logging.info(
        "PRNU residual: initial mean=%.3g min=%.3g max=%.3g",
        float(np.mean(mean_frame)),
        float(np.min(mean_frame)),
        float(np.max(mean_frame)),
    )

    mode = cfg.get("processing", {}).get("gain_map_mode", "none")

    gain_map = get_gain_map(
        cfg,
        None,
        project_dir=project_dir,
        gain_db=gain_db,
        stack=(flat_stack if mode == "self_fit" or project_dir is None or gain_db is None else None),
        noise_signal=noise_signal,
        dn_sat=dn_sat_val,
    )

    if gain_map is not None:
        corrected = flat_stack / gain_map
        mean_frame = np.mean(corrected, axis=0)
        std_frame = np.std(corrected, axis=0)
    else:
        std_frame = np.std(flat_stack, axis=0)

    stat_mode = cfg.get("processing", {}).get("stat_mode", "rms")
    value = _reduce(std_frame[mask], stat_mode) / max(mean_frame[mask].mean(), 1e-6) * 100.0
    return value, std_frame


def calculate_gain_map(
    flat_stack: np.ndarray,
    cfg: Dict[str, Any],
    rects: list[tuple[int, int, int, int]] | None = None,
    project_dir: Path | str | None = None,
    gain_db: float | None = None,
    *,
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
    dn_sat: float | None = None,
) -> np.ndarray | None:
    """Return gain map using flat-field stack and configuration."""

    _ = rects  # ROI unused for gain map fitting
    return get_gain_map(
        cfg,
        None,
        project_dir=project_dir,
        gain_db=gain_db,
        stack=flat_stack,
        noise_signal=noise_signal,
        dn_sat=dn_sat,
    )


def calculate_prnu_residual(
    flat_stack: np.ndarray,
    cfg: Dict[str, Any],
    rects: list[tuple[int, int, int, int]] | None = None,
    project_dir: Path | str | None = None,
    gain_db: float | None = None,
    *,
    noise_signal: tuple[np.ndarray, np.ndarray] | None = None,
    dn_sat: float | None = None,
) -> Tuple[float, np.ndarray]:
    """Calculate PRNU residual from a flat-field stack.

    Parameters
    ----------
    flat_stack:
        Stack of flat-field frames.
    cfg:
        Parsed configuration dictionary.
    rects:
        Optional ROI rectangles ``(left, top, width, height)``.
    noise_signal:
        Optional ``(signal, noise)`` tuple to estimate ``dn_sat`` when needed.
    dn_sat:
        Precomputed saturation level. If ``None``, it is calculated using
        ``noise_signal`` when required.

    Returns
    -------
    Tuple[float, np.ndarray]
        ``(prnu_percent, residual_map)`` where the residual map matches the image size.
    """

    mask = _mask_from_rects(flat_stack.shape[1:], rects) if rects else np.ones(flat_stack.shape[1:], bool)

    mean_frame = np.mean(flat_stack, axis=0)
    margin = cfg.get("processing", {}).get("mask_upper_margin")
    dn_sat_val = dn_sat
    if margin is not None:
        if dn_sat_val is None:
            dn_sat_val = calculate_dn_sat(flat_stack, cfg, noise_signal)
        mask &= mean_frame <= margin * dn_sat_val

    mode = cfg.get("processing", {}).get("gain_map_mode", "none")

    gain_map = get_gain_map(
        cfg,
        None,
        project_dir=project_dir,
        gain_db=gain_db,
        stack=(flat_stack if mode == "self_fit" or project_dir is None or gain_db is None else None),
        noise_signal=noise_signal,
        dn_sat=dn_sat_val,
    )

    if gain_map is not None:
        corrected = flat_stack / gain_map
        mean_frame = np.mean(corrected, axis=0)
        logging.info(
            "Corrected frame: mean=%.3g min=%.3g max=%.3g",
            float(np.mean(mean_frame)),
            float(np.min(mean_frame)),
            float(np.max(mean_frame)),
        )

    roi_mean = float(np.mean(mean_frame[mask]))
    residual_map = mean_frame - roi_mean

    logging.info(
        "Residual map: roi_mean=%.3g min=%.3g max=%.3g",
        roi_mean,
        float(np.min(residual_map)),
        float(np.max(residual_map)),
    )

    stat_mode = cfg.get("processing", {}).get("stat_mode", "rms")
    value = _reduce(residual_map[mask], stat_mode) / max(roi_mean, 1e-6) * 100.0
    return value, residual_map


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

    mask = _mask_from_rects(flat_stack.shape[1:], rects) if rects else np.ones(flat_stack.shape[1:], bool)

    illum = cfg.get("illumination", {})
    power = float(illum.get("power_uW_cm2", 1.0))
    exposure_ms = float(illum.get("exposure_ms", 1.0))
    denom = power * exposure_ms * float(ratio) / 1000.0
    if denom == 0:
        return 0.0

    mean_dn = float(np.mean(flat_stack[:, mask]))
    return mean_dn / denom


def calculate_dn_at_snr(signal: np.ndarray, snr_lin: np.ndarray, threshold_db: float) -> float:
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


def calculate_snr_at_half(signal: np.ndarray, snr_lin: np.ndarray, dn_sat: float) -> float:
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


def clipped_snr_model(
    signal: np.ndarray,
    read_noise: float,
    adc_full_scale: float,
    black_level: float = 0.0,
    *,
    limit_noise: float = 0.0,
    limit_margin: float = 0.05,
) -> np.ndarray:
    """Return SNR curve accounting for ADC clipping and black level.

    Additional ``limit_noise`` can be injected only when the signal is close to
    ``black_level`` or ``adc_full_scale``. The affected region is controlled by
    ``limit_margin`` which specifies the fraction of the usable range.
    """

    from scipy.stats import norm

    sig = np.asarray(signal, dtype=float) - black_level
    sig_pos = np.maximum(sig, 0.0)
    range_val = adc_full_scale - black_level
    if limit_noise > 0.0 and limit_margin > 0.0:
        margin = limit_margin * range_val
        edge_mask = (sig_pos < margin) | ((range_val - sig_pos) < margin)
    else:
        edge_mask = np.zeros_like(sig_pos, dtype=bool)
    sigma = np.sqrt(sig_pos + read_noise**2 + (limit_noise**2) * edge_mask)
    alpha = (0.0 - sig) / sigma
    beta = (range_val - sig) / sigma
    cdf = norm.cdf
    pdf = norm.pdf
    with np.errstate(divide="ignore", invalid="ignore"):
        z = cdf(beta) - cdf(alpha)
        a = pdf(alpha)
        b = pdf(beta)
        mean_adj = (a - b) / z
        var = 1.0 + (alpha * a - beta * b) / z - mean_adj**2
    mean = sig + sigma * mean_adj
    var = sigma**2 * var
    var = np.where(z <= 0, 0.0, var)
    noise = np.sqrt(var)
    mean = np.clip(mean, 0.0, adc_full_scale - black_level)
    return mean / np.maximum(noise, 1e-6)


def fit_clipped_snr_model(
    signal: np.ndarray,
    snr: np.ndarray,
    adc_full_scale: float,
    black_level: float = 0.0,
    *,
    limit_noise: float | None = None,
    limit_margin: float = 0.05,
) -> tuple[float, float]:
    """Fit :func:`clipped_snr_model` and return estimated ``(read_noise, limit_noise)``.

    If ``limit_noise`` is ``None``, both ``read_noise`` and ``limit_noise`` are
    optimized. Otherwise ``limit_noise`` is kept fixed and only ``read_noise`` is
    estimated.
    """

    signal = np.asarray(signal, dtype=float)
    snr = np.asarray(snr, dtype=float)

    mask = np.isfinite(signal) & np.isfinite(snr)
    signal = signal[mask]
    snr = snr[mask]

    # Avoid discarding high-SNR points. They may indicate clipped noise rather
    # than measurement errors, so include all finite samples in the fit.

    if limit_noise is None:

        def _model(x: np.ndarray, r: float, ln: float) -> np.ndarray:
            return clipped_snr_model(
                x,
                r,
                adc_full_scale,
                black_level,
                limit_noise=ln,
                limit_margin=limit_margin,
            )

        try:
            popt, _ = curve_fit(
                _model,
                signal,
                snr,
                p0=[1.0, 0.1],
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=10000,
            )
        except Exception:
            return 1.0, 0.0
        return float(popt[0]), float(popt[1])

    def _model(x: np.ndarray, r: float) -> np.ndarray:
        return clipped_snr_model(
            x,
            r,
            adc_full_scale,
            black_level,
            limit_noise=limit_noise,
            limit_margin=limit_margin,
        )

    try:
        popt, _ = curve_fit(_model, signal, snr, p0=[1.0], bounds=(0.0, np.inf), maxfev=10000)
    except Exception:
        return 1.0, float(limit_noise)
    return float(popt[0]), float(limit_noise)


def fit_three_region_snr_model(
    signal: np.ndarray,
    snr: np.ndarray,
    adc_full_scale: float,
    black_level: float = 0.0,
    *,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed SNR curve using linear interpolation and spline."""

    signal = np.asarray(signal, dtype=float)
    snr = np.asarray(snr, dtype=float)

    mask = np.isfinite(signal) & np.isfinite(snr)
    signal = signal[mask]
    snr = snr[mask]
    if signal.size == 0:
        xs = np.linspace(0.0, 1.0, num_points)
        return xs, np.full_like(xs, np.nan)

    order = np.argsort(signal)
    signal = signal[order]
    snr = snr[order]

    xs = np.linspace(float(signal.min()), float(signal.max()), num_points)
    ys = np.interp(xs, signal, snr)

    if xs.size >= 4:
        try:
            spline = UnivariateSpline(xs, ys, s=0.2, k=3)
            ys = spline(xs)
        except Exception:
            pass

    return xs, ys
