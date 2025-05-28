# generated: 2025-05-18T10:25:00Z (auto)
# utils/config.py – Config utilities (Spec-complete, nested-dict gains/exposures)

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
    "gain_ratio",
]

# ────────────────────────────────────────────────
# Load & merge config
# ────────────────────────────────────────────────
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
    project_yaml = Path(project_cfg_path)
    if project_yaml.is_dir():
        project_yaml = project_yaml / "config.yaml"
    project_cfg = _read_yaml(project_yaml)
    cfg = _merge_dict(default_cfg, project_cfg)
    meas = cfg.get("measurement", {})
    if "gains" in meas:
        meas["gains"] = {str(k): v for k, v in meas["gains"].items()}
    if "exposures" in meas:
        meas["exposures"] = {str(k): v for k, v in meas["exposures"].items()}
    cfg["measurement"] = meas
    return cfg


# ────────────────────────────────────────────────
# Measurement helpers (nested-dict access)
# ────────────────────────────────────────────────
def gain_entries(cfg: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Return sorted list of (gain_db, folder)."""
    gains = cfg["measurement"]["gains"]
    return sorted(
        [(float(db), meta["folder"]) for db, meta in gains.items()], key=lambda x: x[0]
    )


def exposure_entries(cfg: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Return sorted list of (ratio, folder) descending by ratio."""
    exps = cfg["measurement"]["exposures"]
    return sorted(
        [(float(r), meta["folder"]) for r, meta in exps.items()], key=lambda x: -x[0]
    )


def _lookup_nested(d: Dict[str, Any], key: float | int) -> Dict[str, Any]:
    """Robustly fetch nested-dict item allowing float/int/str key variants."""
    if key in d:
        return d[key]
    if int(key) in d:
        return d[int(key)]
    str_key = str(key)
    if str_key in d:
        return d[str_key]
    raise KeyError(key)


def find_gain_folder(
    project: Path | str, gain_db: float | int, cfg: Dict[str, Any]
) -> Path:
    """Return <project>/<gain_folder>. Accepts gain_db as float or int."""
    project = Path(project)
    folder = _lookup_nested(cfg["measurement"]["gains"], gain_db)["folder"]
    return project / folder


def find_exposure_folder(
    project: Path | str, gain_db: float | int, ratio: float, cfg: Dict[str, Any]
) -> Path:
    gain_path = find_gain_folder(project, gain_db, cfg)
    exp_folder = _lookup_nested(cfg["measurement"]["exposures"], ratio)["folder"]
    return gain_path / exp_folder


def gain_ratio(gain_db: float) -> float:
    """Return linear gain ratio from decibel value."""

    return 10 ** (float(gain_db) / 20.0)
