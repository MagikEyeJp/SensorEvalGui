#!/usr/bin/env python3
import tempfile
from pathlib import Path
import pytest

yaml = pytest.importorskip("yaml")

from utils.config import load_config, gain_entries, gain_ratio


def test_load_config_merges_defaults(tmp_path):
    project_cfg = {
        "measurement": {
            "gains": {0: {"folder": "g0_custom"}},
            "exposures": {1.0: {"folder": "exp1_custom"}},
        },
        "processing": {
            "snr_threshold_dB": 25,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(project_cfg, fh)

    cfg = load_config(cfg_file)

    # project override applied
    assert cfg["measurement"]["gains"]["0"]["folder"] == "g0_custom"
    assert cfg["measurement"]["exposures"]["1.0"]["folder"] == "exp1_custom"
    assert cfg["processing"]["snr_threshold_dB"] == 25

    # default values preserved
    assert cfg["measurement"]["gains"]["6"]["folder"] == "gain_6dB"
    assert cfg["processing"]["min_sig_factor"] == 3


def test_gain_entries_sorted():
    cfg = {
        "measurement": {
            "gains": {
                "6": {"folder": "g6"},
                "0": {"folder": "g0"},
                "12": {"folder": "g12"},
            }
        }
    }
    entries = gain_entries(cfg)
    assert entries == [
        (0.0, "g0"),
        (6.0, "g6"),
        (12.0, "g12"),
    ]


def test_gain_ratio_conversion():
    assert gain_ratio(0.0) == 1.0
    assert pytest.approx(gain_ratio(6.0), abs=1e-6) == 10 ** (6.0 / 20.0)
