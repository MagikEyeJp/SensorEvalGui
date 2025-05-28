#!/usr/bin/env python3
import sys
import types
import importlib

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tifffile")

import numpy as np
import tifffile


# provide roifile stub before importing modules that require it
class _StubROI:
    left = 0
    top = 0
    width = 2
    height = 2


def _roiread(path):
    return [_StubROI()]


sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)

from utils.config import load_config
import utils.roi as roi  # reload with stubbed roifile

importlib.reload(roi)
import core.analysis as analysis

importlib.reload(analysis)


def test_calculate_dark_noise_gain(tmp_path):
    project = tmp_path
    gain_dir = project / "gain_0dB" / "dark"
    gain_dir.mkdir(parents=True)

    for i in range(2):
        tifffile.imwrite(
            gain_dir / f"frame{i}.tiff", np.full((2, 2), i, dtype=np.uint16)
        )

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "flat_roi_file": str(roi_file),
        }
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)

    dsnu, rn, dsnu_map, rn_map = analysis.calculate_dark_noise_gain(project, 0, cfg)

    assert pytest.approx(dsnu, abs=1e-6) == 0.0
    assert pytest.approx(rn, abs=1e-6) == 0.5
    assert dsnu_map.shape == (2, 2)
    assert rn_map.shape == (2, 2)


def test_calculate_system_sensitivity_ratio():
    stack = np.full((2, 2, 2), 200, dtype=np.uint16)
    cfg = {"illumination": {"power_uW_cm2": 100.0, "exposure_ms": 50}}
    sens = analysis.calculate_system_sensitivity(stack, cfg, ratio=2.0)
    assert pytest.approx(sens, abs=1e-6) == 200.0 / (100.0 * 50 * 2.0 / 1000.0)


def test_collect_gain_snr_signal_rows():
    rows = [
        {
            "ROI Type": "grayscale",
            "ROI No": 0,
            "Gain (dB)": 0.0,
            "Exposure": 1.0,
            "Mean": 10.0,
            "Std": 1.0,
        },
        {
            "ROI Type": "flat",
            "ROI No": "-",
            "Gain (dB)": 0.0,
            "Exposure": 1.0,
            "Mean": 20.0,
            "Std": 2.0,
        },
    ]
    cfg = {"processing": {"exclude_abnormal_snr": False, "min_sig_factor": 0}}
    data = analysis.collect_gain_snr_signal(rows, cfg)
    assert 0.0 in data
    sig, snr = data[0.0]
    assert np.allclose(sig, [10.0, 20.0])
    assert np.allclose(snr, [10.0, 10.0])


def test_extract_roi_stats_mid_index(tmp_path):
    def _roiread_multi(path):
        class _R:
            def __init__(self, left: int):
                self.left = left
                self.top = 0
                self.width = 1
                self.height = 2

        return [_R(0), _R(1), _R(2)]

    sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread_multi)
    importlib.reload(roi)
    importlib.reload(analysis)

    project = tmp_path
    stack = np.stack(
        [
            [[10, 20, 30], [10, 20, 30]],
            [[11, 21, 31], [11, 21, 31]],
        ],
        axis=0,
    ).astype(np.uint16)
    gain_dir = project / "gain_0dB" / "chart_1x"
    gain_dir.mkdir(parents=True)
    for i, frame in enumerate(stack):
        tifffile.imwrite(gain_dir / f"frame{i}.tiff", frame)

    roi_file = project / "roi.zip"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
            "roi_mid_index": 1,
        }
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    stats = analysis.extract_roi_stats(project, cfg)
    assert (0.0, 1.0) in stats
    res = stats[(0.0, 1.0)]
    assert pytest.approx(res["mean"], abs=1e-6) == 20.5
    assert pytest.approx(res["std"], abs=1e-6) == 0.5
