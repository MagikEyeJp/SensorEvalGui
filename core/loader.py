# core/loader.py

from pathlib import Path
import tifffile
import numpy as np


def load_image_stack(folder: Path) -> np.ndarray:
    """
    Load a stack of 16-bit TIFF images from the given folder.
    Returns a NumPy array with shape (N, H, W), where N is the number of images.
    """
    tiffs = sorted(folder.glob("*.tif*"))
    stack = [tifffile.imread(str(p)) for p in tiffs]
    return np.stack(stack, axis=0)


def find_condition_folders(root_dir: Path) -> list[Path]:
    """
    Recursively search for condition folders under the project root.
    Each condition folder typically contains TIFF files for a single exposure/gain condition.
    """
    return [f for f in root_dir.rglob("*") if f.is_dir() and any(p.suffix in ['.tif', '.tiff'] for p in f.iterdir())]
