# generated: 2025-05-18T10:25:00Z (auto)
# utils/config.py – Config utilities (Spec-complete, nested-dict gains/exposures)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Mapping
import yaml

__all__ = [
    "load_config",
    "gain_entries",
    "exposure_entries",
    "find_gain_folder",
    "find_exposure_folder",
    "gain_ratio",
    "nearest_gain",
    "nearest_exposure",
    "adc_full_scale",
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
    proc = cfg.get("processing", {})
    proc["gain_map_mode"] = str(proc.get("gain_map_mode", "none"))
    cfg["processing"] = proc
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
    int_key = int(key)
    str_int = str(int_key)
    if str_int in d:
        return d[str_int]
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


def nearest_gain(cfg: Dict[str, Any], target: float) -> float:
    """Return gain value in config closest to ``target``."""

    gains = [g for g, _ in gain_entries(cfg)]
    if not gains:
        raise ValueError("No gains defined")
    return min(gains, key=lambda g: abs(g - float(target)))


def nearest_exposure(cfg: Dict[str, Any], target: float) -> float:
    """Return exposure ratio in config closest to ``target``."""

    ratios = [r for r, _ in exposure_entries(cfg)]
    if not ratios:
        raise ValueError("No exposures defined")
    return min(ratios, key=lambda r: abs(r - float(target)))


def adc_full_scale(cfg: Mapping[str, Any]) -> int:
    """Return ADC full-scale DN from configuration."""

    adc_bits = int(cfg.get("sensor", {}).get("adc_bits", 0))
    lsb_shift = int(cfg.get("sensor", {}).get("lsb_shift", 0))
    if adc_bits <= 0:
        return 0
    return ((1 << adc_bits) - 1) * (1 << lsb_shift)
