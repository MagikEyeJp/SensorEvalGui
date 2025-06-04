"""Robust P-spline fitting utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve

__all__ = ["robust_p_spline_fit", "plot_fit"]


_DEF_LAM_RANGE = np.logspace(-3, 2, 10)
_DEF_N_RANGE = np.arange(10, 31, 5)


def _design_matrix(
    x: np.ndarray, deg: int, n_splines: int, knot_density: str
) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(x.min())
    x_max = float(x.max())
    n_inner = n_splines - deg - 1
    if n_inner < 0:
        raise ValueError("n_splines too small for degree")
    if knot_density == "auto" and n_inner > 0:
        interior = np.quantile(x, np.linspace(0, 1, n_inner + 2)[1:-1])
    else:
        if n_inner > 0:
            interior = np.linspace(x_min, x_max, n_inner + 2)[1:-1]
        else:
            interior = np.array([])
    knots = np.concatenate(
        [
            np.full(deg + 1, x_min),
            interior,
            np.full(deg + 1, x_max),
        ]
    )
    B = BSpline.design_matrix(x, knots, deg).toarray()
    return knots, B


def _diff_penalty(n: int, order: int = 2) -> np.ndarray:
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def _irls(
    B: np.ndarray,
    y: np.ndarray,
    lam: float,
    D: np.ndarray,
    *,
    robust: str = "huber",
    weights: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weights is None:
        w = np.ones_like(y)
    else:
        w = weights.copy()
    coef = np.zeros(B.shape[1])
    DtD = D.T @ D
    for _ in range(max_iter):
        W = w[:, None]
        left = B.T @ (W * B) + lam * DtD
        right = B.T @ (w * y)
        c, low = cho_factor(left, overwrite_a=False)
        new_coef = cho_solve((c, low), right)
        res = y - B @ new_coef
        sigma = np.median(np.abs(res)) / 0.6745 + 1e-12
        if robust == "huber":
            cval = 1.345 * sigma
            w = np.where(np.abs(res) <= cval, 1.0, cval / np.abs(res))
        elif robust in ("bisquare", "tukey"):
            cval = 4.685 * sigma
            r = res / cval
            mask = np.abs(res) <= cval
            w = np.where(mask, (1 - r**2) ** 2, 0.0)
        else:
            w = np.ones_like(res)
        if np.linalg.norm(new_coef - coef) < tol:
            coef = new_coef
            break
        coef = new_coef
    res = y - B @ coef
    return coef, res, w


def _gcv_score(
    B: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    lam: float,
    D: np.ndarray,
    w: np.ndarray,
) -> float:
    W = w[:, None]
    left = B.T @ (W * B) + lam * (D.T @ D)
    c, low = cho_factor(left)
    inv_left = cho_solve((c, low), np.eye(left.shape[0]))
    S = B @ inv_left @ (B.T * W.T)
    trace = np.trace(S)
    res = y - B @ coef
    rss = np.sum((w * res) ** 2)
    n = y.size
    denom = max(n - trace, 1e-8)
    return n * rss / (denom**2)


def _search_params(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    knot_density: str,
    lam: float | None,
    n_splines: int | str,
    robust: str,
    weights: np.ndarray,
) -> tuple[float, int]:
    lam_vals = _DEF_LAM_RANGE if lam is None else np.array([lam])
    if isinstance(n_splines, str) and n_splines == "auto":
        n_vals = _DEF_N_RANGE
    elif isinstance(n_splines, int):
        n_vals = np.array([n_splines])
    else:
        raise ValueError("n_splines must be int or 'auto'")

    best_score = np.inf
    best = (lam_vals[0], int(n_vals[0]))

    for n in n_vals:
        knots, B = _design_matrix(x, deg, int(n), knot_density)
        D = _diff_penalty(B.shape[1])
        for lam_v in lam_vals:
            coef, _, w = _irls(B, y, lam_v, D, robust=robust, weights=weights)
            score = _gcv_score(B, y, coef, lam_v, D, w)
            if score < best_score:
                best_score = score
                best = (lam_v, int(n))
    return best


def robust_p_spline_fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    deg: int = 3,
    n_splines: int | str = 20,
    lam: float | None = None,
    knot_density: str = "auto",
    robust: str = "huber",
    num_points: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a robust P-spline and return curve and 95% CI."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    if x_arr.size < deg + 1:
        x_dense = np.linspace(
            float(x_arr.min()), float(x_arr.max()), num_points - 1, endpoint=False
        )
        x_dense = np.concatenate(
            [x_dense, [np.nextafter(float(x_arr.max()), float(x_arr.min()))]]
        )
        y_dense = np.interp(x_dense, x_arr, y_arr)
        return x_dense, y_dense, y_dense, y_dense

    x_aug = np.concatenate([np.repeat(x_arr[0], 3), x_arr, np.repeat(x_arr[-1], 3)])
    y_aug = np.concatenate([np.repeat(y_arr[0], 3), y_arr, np.repeat(y_arr[-1], 3)])
    w = np.ones_like(x_aug)
    w[:3] = 10.0
    w[-3:] = 10.0

    lam_opt, n_opt = _search_params(
        x_aug, y_aug, deg, knot_density, lam, n_splines, robust, w
    )

    knots, B = _design_matrix(x_aug, deg, n_opt, knot_density)
    D = _diff_penalty(B.shape[1])
    coef, res, weights = _irls(B, y_aug, lam_opt, D, robust=robust, weights=w)

    x_dense = np.linspace(
        float(x_arr.min()), float(x_arr.max()), num_points - 1, endpoint=False
    )
    x_dense = np.concatenate(
        [x_dense, [np.nextafter(float(x_arr.max()), float(x_arr.min()))]]
    )
    B_dense = BSpline.design_matrix(x_dense, knots, deg).toarray()

    DtD = D.T @ D
    W = weights[:, None]
    left = B.T @ (W * B) + lam_opt * DtD
    c, low = cho_factor(left)
    cov_coef = cho_solve((c, low), np.eye(left.shape[0]))
    sigma2 = np.sum((weights * res) ** 2) / max(
        x_aug.size - np.trace(B @ cov_coef @ (B.T * W.T)), 1
    )
    y_pred = B_dense @ coef
    var_pred = np.sum(B_dense @ cov_coef * B_dense, axis=1)
    ci = 1.96 * np.sqrt(np.maximum(var_pred * sigma2, 0.0))
    return x_dense, y_pred, y_pred + ci, y_pred - ci


def plot_fit(
    x: ArrayLike,
    y: ArrayLike,
    x_dense: ArrayLike,
    y_pred: ArrayLike,
    ci_upper: ArrayLike,
    ci_lower: ArrayLike,
) -> None:
    """Plot fitted P-spline curve with 95% CI."""
    import matplotlib.pyplot as plt

    plt.scatter(x, y, s=10, color="k", label="data")
    plt.plot(x_dense, y_pred, color="C1", label="fit")
    plt.fill_between(x_dense, ci_lower, ci_upper, color="C1", alpha=0.3)
    plt.xlabel("Signal")
    plt.ylabel("SNR")
    plt.legend()
    plt.tight_layout()
    plt.show()
