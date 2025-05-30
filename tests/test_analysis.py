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

    dsnu, rn, dsnu_map, rn_map, black = analysis.calculate_dark_noise_gain(
        project, 0, cfg
    )

    assert pytest.approx(dsnu, abs=1e-6) == 0.0
    assert pytest.approx(rn, abs=1e-6) == 0.5
    assert dsnu_map.shape == (2, 2)
    assert rn_map.shape == (2, 2)
    assert pytest.approx(black, abs=1e-6) == 0.5


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


def test_extract_roi_stats_temporal_std(tmp_path):
    sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)
    importlib.reload(roi)
    importlib.reload(analysis)

    project = tmp_path
    stack = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.uint16,
    )
    gain_dir = project / "gain_0dB" / "chart_1x"
    gain_dir.mkdir(parents=True)
    for i, frame in enumerate(stack):
        tifffile.imwrite(gain_dir / f"frame{i}.tiff", frame)

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
        },
        "processing": {
            "stat_mode": "mean",
            "min_sig_factor": 0,
            "exclude_abnormal_snr": False,
        },
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    stats = analysis.extract_roi_stats(project, cfg)
    res = stats[(0.0, 1.0)]
    assert pytest.approx(res["mean"], abs=1e-6) == 4.5
    assert pytest.approx(res["std"], abs=1e-6) == 2.0
    assert pytest.approx(res["snr"], abs=1e-6) == 2.25


def test_extract_roi_table_temporal_std(tmp_path):
    sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)
    importlib.reload(roi)
    importlib.reload(analysis)

    project = tmp_path
    stack = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.uint16,
    )
    gain_dir = project / "gain_0dB" / "chart_1x"
    gain_dir.mkdir(parents=True)
    for i, frame in enumerate(stack):
        tifffile.imwrite(gain_dir / f"frame{i}.tiff", frame)

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
        },
        "processing": {
            "stat_mode": "mean",
            "min_sig_factor": 0,
            "exclude_abnormal_snr": False,
        },
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    rows = analysis.extract_roi_table(project, cfg)
    assert len(rows) == 1
    row = rows[0]
    assert pytest.approx(row["Mean"], abs=1e-6) == 4.5
    assert pytest.approx(row["Std"], abs=1e-6) == 2.0
    assert pytest.approx(row["SNR (dB)"], abs=1e-6) == 20 * np.log10(2.25)


def test_extract_roi_stats_gainmap_self_fit(tmp_path):
    sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)
    importlib.reload(roi)
    importlib.reload(analysis)

    project = tmp_path
    stack = np.stack(
        [
            [[10, 20], [10, 20]],
            [[20, 40], [20, 40]],
        ],
        axis=0,
    ).astype(np.uint16)
    gain_dir = project / "gain_0dB" / "chart_1x"
    gain_dir.mkdir(parents=True)
    for i, frame in enumerate(stack):
        tifffile.imwrite(gain_dir / f"frame{i}.tiff", frame)

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
        },
        "processing": {
            "stat_mode": "rms",
            "min_sig_factor": 0,
            "exclude_abnormal_snr": False,
            "plane_fit_order": 1,
            "gain_map_mode": "self_fit",
        },
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    stats = analysis.extract_roi_stats_gainmap(project, cfg)
    res = stats[(0.0, 1.0)]
    assert pytest.approx(res["mean"], abs=1e-6) == 30.0
    assert pytest.approx(res["std"], abs=1e-6) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize(
    "mode, expected_mean, expected_std",
    [
        ("flat_fit", 22.5, 7.905694150420948),
        ("flat_frame", 22.5, 7.905694150420948),
    ],
)
def test_extract_roi_stats_gainmap_modes(tmp_path, mode, expected_mean, expected_std):
    sys.modules["roifile"] = types.SimpleNamespace(roiread=_roiread)
    importlib.reload(roi)
    importlib.reload(analysis)

    project = tmp_path
    chart_stack = np.stack(
        [
            [[10, 10], [10, 10]],
            [[20, 20], [20, 20]],
        ],
        axis=0,
    ).astype(np.uint16)
    chart_dir = project / "gain_0dB" / "chart_1x"
    chart_dir.mkdir(parents=True)
    for i, frame in enumerate(chart_stack):
        tifffile.imwrite(chart_dir / f"frame{i}.tiff", frame)

    flat_stack = np.stack(
        [
            [[1, 2], [1, 2]],
            [[1, 2], [1, 2]],
        ],
        axis=0,
    ).astype(np.uint16)
    flat_dir = project / "gain_0dB" / "LensFlat"
    flat_dir.mkdir(parents=True)
    for i, frame in enumerate(flat_stack):
        tifffile.imwrite(flat_dir / f"frame{i}.tiff", frame)

    roi_file = project / "roi.roi"
    roi_file.write_text("dummy")

    cfg_data = {
        "measurement": {
            "gains": {0: {"folder": "gain_0dB"}},
            "exposures": {1.0: {"folder": "chart_1x"}},
            "chart_roi_file": str(roi_file),
            "flat_roi_file": str(roi_file),
            "flat_lens_folder": "LensFlat",
        },
        "processing": {
            "stat_mode": "rms",
            "min_sig_factor": 0,
            "exclude_abnormal_snr": False,
            "plane_fit_order": 1,
            "gain_map_mode": mode,
        },
    }
    cfg_file = project / "config.yaml"
    with cfg_file.open("w") as fh:
        import yaml

        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)
    stats = analysis.extract_roi_stats_gainmap(project, cfg)
    res = stats[(0.0, 1.0)]
    assert pytest.approx(res["mean"], abs=1e-6) == expected_mean
    assert pytest.approx(res["std"], abs=1e-6) == expected_std


def test_calculate_prnu_residual_simple():
    stack = np.array(
        [
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
        ],
        dtype=np.uint16,
    )
    cfg = {"processing": {"gain_map_mode": "none", "stat_mode": "rms"}}
    val, res = analysis.calculate_prnu_residual(stack, cfg)
    expected = np.array([[1, 2], [3, 4]], dtype=float) - 2.5
    assert res.shape == (2, 2)
    assert np.allclose(res, expected)
    assert pytest.approx(val, abs=1e-6) == np.sqrt(np.mean(expected**2)) / 2.5 * 100.0


def test_fit_gain_map_basic():
    frame = np.array([[1, 2], [3, 4]], dtype=float)
    mask = np.array([[True, False], [True, False]])
    res = analysis.fit_gain_map(frame, mask, order=0)
    assert np.allclose(res, 1.0)


def test_fit_gain_map_order1_exact():
    frame = np.array([[1, 2], [3, 4]], dtype=float)
    mask = np.ones_like(frame, dtype=bool)
    res = analysis.fit_gain_map(frame, mask, order=1)
    assert np.allclose(res, frame / 4.0)


def test_fit_gain_map_rbf_basic():
    pytest.importorskip("scipy")
    frame = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    mask = np.ones_like(frame, dtype=bool)
    res = analysis.fit_gain_map(frame, mask, order=0, method="rbf")
    assert res.shape == frame.shape
    assert np.allclose(np.max(res), 1.0)
