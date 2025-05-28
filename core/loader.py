# core/loader.py – Spec‑compliant loader (nested‑dict gains/exposures)

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import tifffile
import numpy as np

__all__ = [
    "find_condition_folders",
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


def find_condition_folders(
    project_dir: Path | str, cfg: Dict
) -> List[Tuple[Path, float, float]]:
    """Discover leaf folders <gain>/<exposure> present on disk.

    Returns list[ (folder_path, gain_db, exposure_ratio) ].
    """
    project_dir = Path(project_dir)

    gains_cfg = cfg["measurement"]["gains"]  # dict: db -> {folder: ..}
    exposures_cfg = cfg["measurement"]["exposures"]  # dict: ratio -> {folder: ..}

    folders: List[Tuple[Path, float, float]] = []
    for gain_db_str, g_meta in gains_cfg.items():
        gain_db = float(gain_db_str)
        gain_folder = project_dir / g_meta["folder"]
        if not gain_folder.is_dir():
            continue  # skip missing gain dir
        for ratio_str, e_meta in exposures_cfg.items():
            ratio = float(ratio_str)
            exp_folder = gain_folder / e_meta["folder"]
            if exp_folder.is_dir() and any(exp_folder.iterdir()):
                folders.append((exp_folder, gain_db, ratio))
    return folders


def load_image_stack(folder: Path | str) -> np.ndarray:
    """Load all TIFF frames in *folder* and return as (N,H,W) uint16 array."""
    folder = Path(folder)
    files = _collect_frames(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    stack = [tifffile.imread(str(f)) for f in files]
    return np.stack(stack, axis=0)


def load_first_frame(folder: Path | str) -> np.ndarray:
    """Load the first TIFF frame in *folder* as ``(H,W)`` array."""

    folder = Path(folder)
    files = _collect_frames(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    return tifffile.imread(str(files[0]))
