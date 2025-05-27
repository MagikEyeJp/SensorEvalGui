# generated: 2025-05-18T10:55:00Z (auto)
# utils/roi.py – ROI ZIP / ROI file loader

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import roifile  # type: ignore

__all__ = ["load_rois"]

Rect = Tuple[int, int, int, int]  # (left, top, width, height)


def load_rois(zip_path: Path | str) -> List[Rect]:
    """ImageJ ROI zip (or .roi) → [(x,y,w,h), ...]"""
    path = Path(zip_path)
    rects: List[Rect] = []
    if not path.exists():
        raise FileNotFoundError(f"ROI file not found: {path}")
    rz = roifile.roiread(str(path))
    if not isinstance(rz, list):
        rz = [rz]
    for r in rz:
        l, t = int(r.left), int(r.top)
        w = int(getattr(r, "width", r.right - r.left))
        h = int(getattr(r, "height", r.bottom - r.top))
        rects.append((l, t, w, h))
    if not rects:
        raise ValueError(f"Invalid ROI file: {path}")
    # print(f"ROI count: {len(rects)}, {rects}")
    return rects

