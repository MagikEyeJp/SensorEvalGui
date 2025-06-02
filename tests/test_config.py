#!/usr/bin/env python3
import tempfile
from pathlib import Path
import pytest
import logging

yaml = pytest.importorskip("yaml")

from utils.config import (
    load_config,
    gain_entries,
    gain_ratio,
    _lookup_nested,
    nearest_gain,
    nearest_exposure,
)


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


def test_lookup_nested_str_int_key():
    cfg = {"measurement": {"gains": {"0": {"folder": "g0"}}}}
    entries = gain_entries(cfg)
    assert _lookup_nested(cfg["measurement"]["gains"], 0.0)["folder"] == "g0"


def test_nearest_gain_exposure():
    cfg = {
        "measurement": {
            "gains": {"0": {"folder": "g0"}, "6": {"folder": "g6"}},
            "exposures": {"1.0": {"folder": "e1"}, "4.0": {"folder": "e4"}},
        }
    }
    assert nearest_gain(cfg, 2.0) == 0.0
    assert nearest_exposure(cfg, 2.0) == 1.0


def test_apply_logging_config_sets_level():
    from utils.logger import apply_logging_config

    logger = logging.getLogger()
    old_level = logger.level
    cfg = {"logging": {"level": "DEBUG"}}
    apply_logging_config(cfg)
    try:
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(old_level)
