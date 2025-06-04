# core/loader.py – Spec‑compliant loader (nested‑dict gains/exposures)

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import tifffile
import numpy as np

__all__ = [
    "load_image_stack",
    "load_first_frame",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _collect_frames(folder: Path) -> List[Path]:
    """Return sorted list of TIFF files (.tif/.tiff) in *folder*."""
    exts = {".tiff", ".tif"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_image_stack(folder: Path | str) -> np.ndarray:
    """Load all TIFF frames in *folder* and return as (N,H,W) uint16 array."""
    folder = Path(folder)
    files = _collect_frames(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    file_list = [str(f) for f in files]
    # use memory-mapped reading to reduce RAM usage
    return tifffile.imread(file_list, mode="r", out="memmap")


def load_first_frame(folder: Path | str) -> np.ndarray:
    """Load the first TIFF frame in *folder* as ``(H,W)`` array."""

    folder = Path(folder)
    files = _collect_frames(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    return tifffile.imread(str(files[0]))
