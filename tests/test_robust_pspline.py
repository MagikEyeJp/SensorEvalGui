import numpy as np

from utils.robust_pspline import robust_p_spline_fit


def test_robust_p_spline_simple():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 20)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.1, size=x.size)
    x_dense, y_pred, upper, lower = robust_p_spline_fit(
        x, y, deg=3, n_splines=15, lam=0.1, knot_density="uniform"
    )
    assert x_dense.size == 200
    assert y_pred.shape == x_dense.shape
    assert np.all(upper >= lower)
