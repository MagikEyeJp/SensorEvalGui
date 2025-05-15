# core/analysis.py

import numpy as np
from typing import Tuple, Dict


def extract_roi_stats(stack: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics for a given image stack within an ROI.
    Assumes stack shape is (N, H, W).
    Returns mean, std, and SNR.
    """
    if stack.ndim != 3:
        raise ValueError("Expected 3D stack (N, H, W)")

    mean = np.mean(stack)
    std = np.std(stack)
    snr = mean / std if std != 0 else 0.0

    return {
        "mean": mean,
        "std": std,
        "snr": snr
    }


def calculate_snr_curve(signal: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Calculates SNR = signal / noise across an array.
    Zeros in noise are replaced with np.nan to avoid divide-by-zero.
    """
    noise_safe = np.where(noise == 0, np.nan, noise)
    return signal / noise_safe


def calculate_dynamic_range(snr_array: np.ndarray, signal_array: np.ndarray, threshold_snr: float = 20.0) -> float:
    """
    Calculate dynamic range as the ratio (in dB) between the max signal and the
    signal level where SNR >= threshold (e.g. 20 dB).
    """
    if not np.any(snr_array >= threshold_snr):
        return 0.0

    signal_min = signal_array[np.where(snr_array >= threshold_snr)[0][0]]
    signal_max = np.max(signal_array)
    return 20 * np.log10(signal_max / signal_min) if signal_min > 0 else 0.0