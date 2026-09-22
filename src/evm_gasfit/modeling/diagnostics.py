"""Per-fit diagnostics: residual curvature and held-out prediction error.

These computations back the qualification layer. They are deterministic
(no bootstrap, no random splits): a model that cannot predict a held-out
session or a held-out workload point is unfit regardless of luck, and
repeated runs must report identical numbers.

Leave-one-out here means *by cluster* for sessions and *by distinct
regressor value* for workload points. Randomly splitting repeated rows of
the same case/session would masquerade as validation while testing nothing
but within-session noise.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def residual_curvature_r2(
    y: np.ndarray, fitted: np.ndarray, resid: np.ndarray
) -> float:
    """R² of residuals regressed on ``fitted`` and ``fitted²``.

    A well-specified linear model leaves residuals uncorrelated with any
    smooth function of the fitted values. Substantial curvature R² means
    systematic structure the design never priced — e.g. a per-word cost that
    a gas-linear model attributes to nothing — which qualification treats as
    inconclusive rather than letting the slope stand.
    """
    x = np.asarray(fitted, dtype=float)
    r = np.asarray(resid, dtype=float)
    if len(x) < 3:
        return float("nan")
    spread = float(np.ptp(x))
    if not np.isfinite(spread) or spread == 0:
        return float("nan")
    # Polynomial rank must not depend on whether runtimes are tiny or huge.
    x = (x - np.min(x)) / spread
    design = np.column_stack([np.ones(len(x)), x, x * x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        # Fitted values collapsed to (near-)constants; curvature is not
        # identifiable on this slice.
        return float("nan")
    rounding_scale = np.finfo(float).eps * float(np.max(np.abs(y))) * 8
    if float(np.max(np.abs(r))) <= rounding_scale:
        return 0.0
    try:
        # This is a diagnostic regression, not the pricing model. Residuals
        # take both signs, so constraining its coefficients to NNLS would hide
        # a U-shaped or inverse-U-shaped misspecification.
        beta, _, _, _ = np.linalg.lstsq(design, r, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    explained = design @ beta
    ss_res = float(np.sum((r - explained) ** 2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _mean_relative_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.abs(np.asarray(actual, dtype=float))
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(predicted[mask] - actual[mask]) / denom[mask]))


def leave_one_group_out_error(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Mean relative prediction error over leave-one-session-out refits.

    Refits the model on all but one cluster and predicts the held-out
    cluster's rows; averages the relative error over every cluster. Any
    refit that fails (e.g. removing a cluster leaves a rank-deficient
    design) yields NaN — an unvalidatable model, surfaced as such.
    """
    uniq = np.unique(np.asarray(groups))
    if len(uniq) < 2:
        return float("nan")
    errors: list[float] = []
    for g in uniq:
        train = groups != g
        test = ~train
        if train.sum() <= X.shape[1] or np.linalg.matrix_rank(X[train]) < X.shape[1]:
            return float("nan")
        try:
            beta, _ = nnls(X[train], y[train])
        except Exception:  # noqa: BLE001 -- numerical refit failures invalidate evidence
            return float("nan")
        error = _mean_relative_error(y[test], X[test] @ beta)
        if not np.isfinite(error):
            return float("nan")
        errors.append(error)
    if not errors or any(not np.isfinite(e) for e in errors):
        return float("nan")
    return float(np.mean(errors))


def leave_one_point_out_error(
    X: np.ndarray,
    y: np.ndarray,
    opcount: np.ndarray,
) -> float:
    """Mean relative prediction error over leave-one-workload-point-out refits.

    Workload points are the distinct ``opcount`` values of the sweep.
    Holding out a point probes interpolation/extrapolation across the
    measured grid, which row-level validation cannot see.
    """
    points = np.unique(np.asarray(opcount, dtype=float))
    if len(points) < 2:
        return float("nan")
    errors: list[float] = []
    for p in points:
        train = opcount != p
        test = ~train
        if train.sum() <= X.shape[1] or np.linalg.matrix_rank(X[train]) < X.shape[1]:
            return float("nan")
        try:
            beta, _ = nnls(X[train], y[train])
        except Exception:  # noqa: BLE001 -- numerical refit failures invalidate evidence
            return float("nan")
        error = _mean_relative_error(y[test], X[test] @ beta)
        if not np.isfinite(error):
            return float("nan")
        errors.append(error)
    if not errors or any(not np.isfinite(e) for e in errors):
        return float("nan")
    return float(np.mean(errors))
