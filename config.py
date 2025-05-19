# utils/config.py
"""Utility functions for loading the evaluation config.

- Looks for a `config.yaml` in the selected *project* directory.
- Falls back to the repository‑level `config/default_config.yaml` if none found.
- Exposes a single helper `load_config(project_dir)` returning a `dict`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Path to the repo‑default config (relative to the repo root)
_REPO_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"


def load_config(project_dir: Path | str) -> Dict[str, Any]:
    """Load *project*/config.yaml, fallback to repo default.

    Parameters
    ----------
    project_dir : Path | str
        The directory selected by the user (top‑level *project* folder).

    Returns
    -------
    dict
        Parsed YAML config as a nested dictionary.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    cfg_path = project_dir / "config.yaml"

    if not cfg_path.exists():
        cfg_path = _REPO_DEFAULT

    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse YAML config: {cfg_path}\n{exc}") from exc

    # Inject convenience keys
    cfg.setdefault("_paths", {})["config_file"] = str(cfg_path)
    cfg.setdefault("_paths", {})["project_dir"] = str(project_dir)

    return cfg
