# generated: 2025-05-18T09:57:00Z (auto)
# utils/config.py – Config utilities (Spec-complete, nested‑dict gains/exposures)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml

__all__ = [
    "load_config",
    "gain_entries",
    "exposure_entries",
    "find_gain_folder",
    "find_exposure_folder",
]

# -----------------------------------------------------------------------------
# Load & merge config
# -----------------------------------------------------------------------------
_DEFAULT_CFG_PATH = Path(__file__).parent.parent / "config" / "default_config.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (src overwrites dst)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(project_cfg_path: Path | str) -> Dict[str, Any]:
    """Return merged config dict (default <- project)."""
    default_cfg = _read_yaml(_DEFAULT_CFG_PATH)
    project_cfg = _read_yaml(Path(project_cfg_path))
    return _merge_dict(default_cfg, project_cfg)

# -----------------------------------------------------------------------------
# Measurement helpers (nested-dict access)
# -----------------------------------------------------------------------------

def gain_entries(cfg: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Return sorted list of (gain_db, folder)."""
    gains = cfg["measurement"]["gains"]
    return sorted([(float(db), meta["folder"]) for db, meta in gains.items()], key=lambda x: x[0])


def exposure_entries(cfg: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Return sorted list of (ratio, folder) descending by ratio."""
    exps = cfg["measurement"]["exposures"]
    return sorted([(float(r), meta["folder"]) for r, meta in exps.items()], key=lambda x: -x[0])


def find_gain_folder(project: Path | str, gain_db: float, cfg: Dict[str, Any]) -> Path:
    project = Path(project)
    folder = cfg["measurement"]["gains"][str(int(gain_db))]["folder"]
    return project / folder


def find_exposure_folder(project: Path | str, gain_db: float, ratio: float, cfg: Dict[str, Any]) -> Path:
    gain_path = find_gain_folder(project, gain_db, cfg)
    exp_folder = cfg["measurement"]["exposures"][str(ratio)]["folder"]
    return gain_path / exp_folder
