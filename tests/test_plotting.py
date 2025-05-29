#!/usr/bin/env python3
import pytest

np = pytest.importorskip("numpy")

from core import plotting


def test_plot_snr_vs_signal_invalid(tmp_path):
    sig = np.array([1.0, 0.0, 3.0])
    snr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        plotting.plot_snr_vs_signal(sig, snr, {}, tmp_path / "out.png")


def test_plot_snr_vs_exposure_invalid(tmp_path):
    data = {0.0: (np.array([1.0, -2.0]), np.array([1.0, 2.0]))}
    with pytest.raises(ValueError):
        plotting.plot_snr_vs_exposure(data, {}, tmp_path / "out.png")


def test_plot_snr_vs_signal_single_point(tmp_path):
    plotting.plot_snr_vs_signal(
        np.array([1.0]), np.array([2.0]), {}, tmp_path / "out.png"
    )
    assert (tmp_path / "out.png").is_file()


def test_plot_snr_vs_signal_multi(tmp_path):
    data = {
        0.0: (np.array([1.0, 2.0]), np.array([2.0, 4.0])),
        6.0: (np.array([1.5, 3.0]), np.array([1.0, 2.0])),
    }
    plotting.plot_snr_vs_signal_multi(data, {}, tmp_path / "multi.png")
    assert (tmp_path / "multi.png").is_file()


def test_plot_snr_vs_signal_multi_single_point(tmp_path):
    data = {
        0.0: (np.array([1.0]), np.array([2.0])),
        6.0: (np.array([1.1]), np.array([2.2])),
    }
    plotting.plot_snr_vs_signal_multi(data, {}, tmp_path / "single_multi.png")
    assert (tmp_path / "single_multi.png").is_file()


def test_plot_snr_vs_signal_multi_invalid(tmp_path):
    data = {0.0: (np.array([1.0]), np.array([-1.0]))}
    with pytest.raises(ValueError):
        plotting.plot_snr_vs_signal_multi(data, {}, tmp_path / "bad.png")


def test_plot_roi_area(tmp_path):
    img = np.zeros((4, 4))
    rects = [[(0, 0, 2, 2)], [(1, 1, 2, 2)], []]
    plotting.plot_roi_area(
        [img, img, img], rects, ["a", "b", "c"], tmp_path / "roi.png"
    )
    assert (tmp_path / "roi.png").is_file()


def test_plot_heatmap_vmin_vmax(tmp_path):
    data = np.arange(4).reshape(2, 2)
    plotting.plot_heatmap(
        data,
        "heat",
        tmp_path / "heat.png",
        vmin=0.0,
        vmax=3.0,
    )
    assert (tmp_path / "heat.png").is_file()
