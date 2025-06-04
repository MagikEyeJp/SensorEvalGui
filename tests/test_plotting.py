#!/usr/bin/env python3
import pytest

np = pytest.importorskip("numpy")

from core import plotting


def test_plot_snr_vs_exposure_invalid(tmp_path):
    data = {0.0: (np.array([1.0, -2.0]), np.array([1.0, 2.0]))}
    with pytest.raises(ValueError):
        plotting.plot_snr_vs_exposure(data, {}, tmp_path / "out.png")


def test_plot_snr_vs_exposure_single_ideal(tmp_path):
    data = {
        0.0: (np.array([1.0, 2.0]), np.array([2.0, 2.828])),
        6.0: (np.array([1.0, 2.0]), np.array([1.5, 2.1])),
    }
    fig = plotting.plot_snr_vs_exposure(
        data,
        {},
        tmp_path / "exp.png",
        return_fig=True,
    )
    ideal_lines = [
        l
        for l in fig.axes[0].lines
        if l.get_color() == "k" and l.get_linestyle() == "--"
    ]
    assert len(ideal_lines) == 1


def test_plot_snr_vs_signal_multi(tmp_path):
    data = {
        0.0: (np.array([1.0, 2.0]), np.array([2.0, 4.0])),
        6.0: (np.array([1.5, 3.0]), np.array([1.0, 2.0])),
    }
    plotting.plot_snr_vs_signal_multi(
        data,
        {},
        tmp_path / "multi.png",
        black_levels={0.0: 0.0, 6.0: 0.0},
    )
    assert (tmp_path / "multi.png").is_file()


def test_plot_snr_vs_signal_multi_single_point(tmp_path):
    data = {
        0.0: (np.array([1.0]), np.array([2.0])),
        6.0: (np.array([1.1]), np.array([2.2])),
    }
    plotting.plot_snr_vs_signal_multi(
        data,
        {},
        tmp_path / "single_multi.png",
        black_levels={0.0: 0.0, 6.0: 0.0},
    )
    assert (tmp_path / "single_multi.png").is_file()


def test_plot_snr_vs_signal_multi_interp(tmp_path):
    data = {0.0: (np.array([1.0, 3.0]), np.array([2.0, 6.0]))}
    fig = plotting.plot_snr_vs_signal_multi(
        data,
        {"processing": {"snr_fit": {"num_points": 5}}},
        tmp_path / "interp.png",
        return_fig=True,
        interp_points=5,
        black_levels={0.0: 0.0},
    )
    assert (tmp_path / "interp.png").is_file()
    assert len(fig.axes[0].lines[1].get_xdata()) == 5


def test_plot_snr_vs_signal_multi_invalid(tmp_path):
    data = {0.0: (np.array([1.0]), np.array([-1.0]))}
    with pytest.raises(ValueError):
        plotting.plot_snr_vs_signal_multi(
            data,
            {},
            tmp_path / "bad.png",
            black_levels={0.0: 0.0},
        )


def test_plot_noise_vs_signal_multi(tmp_path):
    data = {
        0.0: (np.array([1.0, 2.0]), np.array([0.5, 1.0])),
        6.0: (np.array([1.5, 3.0]), np.array([0.7, 1.4])),
    }
    plotting.plot_noise_vs_signal_multi(data, {}, tmp_path / "noise.png")
    assert (tmp_path / "noise.png").is_file()


def test_plot_noise_vs_signal_multi_invalid(tmp_path):
    data = {0.0: (np.array([1.0, -2.0]), np.array([0.5, 1.0]))}
    with pytest.raises(ValueError):
        plotting.plot_noise_vs_signal_multi(data, {}, tmp_path / "bad2.png")


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


def _get_scatter_xy(ax):
    coll = ax.collections[0]
    return coll.get_offsets().data[:, 0], coll.get_offsets().data[:, 1]


def test_plot_prnu_regression_labels(tmp_path):
    data = {0.0: (np.array([1.0, 2.0]), np.array([0.5, 1.0]))}
    fig = plotting.plot_prnu_regression(
        data,
        {"plot": {"prnu_squared": False}},
        tmp_path / "prnu.png",
        return_fig=True,
    )
    ax = fig.axes[0]
    x, y = _get_scatter_xy(ax)
    assert ax.get_xlabel() == "Mean (DN)"
    assert ax.get_ylabel() == "Std (DN)"
    assert x[0] == pytest.approx(1.0)
    assert y[0] == pytest.approx(0.5)


def test_plot_prnu_regression_labels_squared(tmp_path):
    data = {0.0: (np.array([2.0]), np.array([3.0]))}
    fig = plotting.plot_prnu_regression(
        data,
        {"plot": {"prnu_squared": True}},
        tmp_path / "prnu_sq.png",
        return_fig=True,
    )
    ax = fig.axes[0]
    x, y = _get_scatter_xy(ax)
    assert ax.get_xlabel() == "Mean^2 (DN^2)"
    assert ax.get_ylabel() == "Std^2 (DN^2)"
    assert x[0] == pytest.approx(4.0)
    assert y[0] == pytest.approx(9.0)
