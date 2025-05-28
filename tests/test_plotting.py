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
    plotting.plot_snr_vs_signal(np.array([1.0]), np.array([2.0]), {}, tmp_path / "out.png")
    assert (tmp_path / "out.png").is_file()

