import sys
import types

import numpy as np
import tifffile
import pytest

from pathlib import Path

from utils.config import load_config
from core.pipeline import run_pipeline


# Stub roifile before importing modules that require it
class _Roi:
    left = 0
    top = 0
    width = 2
    height = 2


def _roiread(path):
    return [_Roi()]


sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)


def _write_stack(folder: Path, stack: np.ndarray) -> None:
    folder.mkdir(parents=True)
    for i, frame in enumerate(stack):
        tifffile.imwrite(folder / f"frame{i}.tiff", frame)


def test_run_pipeline_returns_summary(tmp_path):
    project = tmp_path

    chart_dir = project / "gain_0dB" / "chart_1x"
    flat_dir = project / "gain_0dB" / "flat"
    dark_dir = project / "gain_0dB" / "dark"

    _write_stack(
        chart_dir,
        np.stack([np.full((2, 2), 10, np.uint16), np.full((2, 2), 12, np.uint16)]),
    )
    _write_stack(flat_dir, np.stack([np.full((2, 2), 100, np.uint16)] * 2))
    _write_stack(
        dark_dir, np.stack([np.zeros((2, 2), np.uint16), np.ones((2, 2), np.uint16)])
    )

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "flat_lens_folder": "flat",
            "dark_folder": "dark",
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
            "roi_mid_index": 0,
        },
        "processing": {
            "stat_mode": "mean",
            "min_sig_factor": 0,
            "exclude_abnormal_snr": False,
            "gain_map_mode": "none",
            "snr_fit": {"num_points": 5},
        },
        "output": {
            "report_html": False,
            "report_csv": False,
            "report_summary": False,
            "output_dir": "out",
            "snr_signal_data": False,
        },
        "illumination": {"power_uW_cm2": 100.0, "exposure_ms": 50},
        "sensor": {"adc_bits": 10, "lsb_shift": 0},
    }

    cfg_file = project / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    cfg["measurement"]["gains"] = {"0": {"folder": "gain_0dB"}}
    cfg["measurement"]["exposures"] = {"1.0": {"folder": "chart_1x"}}
    result = run_pipeline(project, cfg)
    assert "summary" in result
    assert "Dynamic Range (dB)" in result["summary"]
    assert isinstance(result["summary"]["Dynamic Range (dB)"], float)
