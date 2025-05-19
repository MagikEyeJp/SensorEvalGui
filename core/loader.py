# core/loader.py – config-aware implementation

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import tifffile
import numpy as np

__all__ = [
    "load_image_stack",
    "find_condition_folders",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _collect_files(folder: Path, cfg: Dict) -> List[Path]:
    """Return ordered list of image paths based on config rules."""
    image_cfg = cfg.get("image_structure", {})
    ext = image_cfg.get("file_extension", ".tiff")
    rule = image_cfg.get("file_naming_rule")

    if rule:
        files: List[Path] = []
        idx = 0
        while True:
            candidate = folder / (rule % idx)
            if candidate.exists():
                files.append(candidate)
                idx += 1
            else:
                break
        if not files:
            raise FileNotFoundError(f"No files matched naming rule '{rule}' in {folder}")
    else:
        files = sorted(folder.glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(f"No '*{ext}' files found in {folder}")
    return files


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_image_stack(folder: Path | str, cfg: Dict) -> np.ndarray:
    """Load stack of 16‑bit images according to config rules."""
    folder = Path(folder)
    files = _collect_files(folder, cfg)
    stack = [tifffile.imread(str(p)) for p in files]
    return np.stack(stack, axis=0)


def find_condition_folders(root_dir: Path | str, cfg: Dict) -> List[Path]:
    """Find leaf folders that contain at least one image file."""
    root_dir = Path(root_dir)
    ext = cfg.get("image_structure", {}).get("file_extension", ".tiff")
    return [f for f in root_dir.rglob("*") if f.is_dir() and any(p.suffix == ext for p in f.iterdir() if p.is_file())]
