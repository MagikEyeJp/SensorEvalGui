# core/plotting.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_snr_vs_signal(signal: np.ndarray, snr: np.ndarray, output_path: Path) -> None:
    """
    Plot SNR vs Signal and save as PNG.
    """
    plt.figure()
    plt.plot(signal, snr, marker='o')
    plt.xlabel("Signal (DN)")
    plt.ylabel("SNR")
    plt.title("SNR vs Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_snr_vs_exposure(exposure: np.ndarray, snr: np.ndarray, output_path: Path) -> None:
    """
    Plot SNR vs Exposure and save as PNG.
    """
    plt.figure()
    plt.plot(exposure, snr, marker='s')
    plt.xlabel("Exposure Ratio")
    plt.ylabel("SNR")
    plt.title("SNR vs Exposure")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()