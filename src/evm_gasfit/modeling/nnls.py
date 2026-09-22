"""NNLS regression with bootstrap inference."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .results import NNLSResults


def _cluster_indices(
    groups: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> list[np.ndarray] | None:
    """Pre-draw session-cluster row-index arrays.

    Returns ``None`` when fewer than two sessions exist. That is not a request
    to row-bootstrap: the point estimate remains valid but session-cluster
    uncertainty is unavailable, so the caller preserves all-NaN bootstrap
    draws for qualification to mark inconclusive.
    """
    uniq = np.asarray(sorted(pd.unique(groups), key=str), dtype=object)
    if len(uniq) < 2:
        return None
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    draws: list[np.ndarray] = []
    chosen = rng.integers(0, len(uniq), size=(n_bootstrap, len(uniq)))
    for i in range(n_bootstrap):
        parts = [members[uniq[c]] for c in chosen[i]]
        draws.append(np.concatenate(parts) if parts else np.array([], dtype=int))
    return draws


def fit_nnls(
    feature_df: pd.DataFrame,
    features: list[str],
    target: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    groups: pd.Series | np.ndarray | None = None,
    bootstrap_target: Callable[[int], np.ndarray | None] | None = None,
    uncertainty_conditional: bool = False,
) -> NNLSResults:
    """Fit a non-negative least squares model with bootstrap inference.

    The intercept is fitted alongside every other coefficient under the same
    non-negativity constraint: a column of ones is prepended to the design
    matrix and passed through ``scipy.optimize.nnls``. Bootstrap iterations that
    raise leave a row of NaNs in the coefficient matrix; ``NNLSResults`` filters
    those rows out before computing std errors, confidence intervals, and
    p-values.

    When ``groups`` is provided (typically the per-row session labels of a
    benchmark campaign), inference switches to a **cluster bootstrap**: whole
    groups are resampled with replacement, so rows sharing machine state are
    never treated as independent observations. With fewer than two distinct
    groups, the point estimate remains but all bootstrap draws are unavailable;
    qualification marks insufficient session evidence rather than
    row-resampling.

    Args:
        feature_df: Frame containing the regressors and the target column.
        features: Regressor column names; ``"const"`` is added internally.
        target: Name of the response column in ``feature_df``.
        n_bootstrap: Number of bootstrap resamples used for inference.
        random_seed: Seed threaded into ``numpy.random.default_rng`` so the
            bootstrap is reproducible across runs and platforms.
        groups: Optional per-row cluster labels (e.g. session ids) aligned
            with ``feature_df``'s rows.
        bootstrap_target: Optional callback supplying the full response vector
            for each bootstrap replicate.
        uncertainty_conditional: Whether the replicate draws omit required
            supporting uncertainty and must not support an unconditional CI.

    Returns:
        Results object exposing ``params``, ``pvalues``, ``conf_int``,
        ``rsquared``, ``rsquared_adj``, ``nobs``, ``fittedvalues``,
        ``resid``, ``groups``, and ``summary``.

    Raises:
        ValueError: If ``feature_df`` is empty, the target / feature columns
            are missing, or ``groups`` length does not match the frame.
    """
    if feature_df.empty:
        raise ValueError("feature_df cannot be empty")
    if target not in feature_df.columns:
        raise ValueError(f"feature_df must contain target column {target!r}")
    missing = [f for f in features if f not in feature_df.columns]
    if missing:
        raise ValueError(f"features not found in feature_df: {missing}")

    X = feature_df[features].to_numpy(dtype=float)
    y = feature_df[target].to_numpy(dtype=float)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    feature_names = ["const"] + list(features)

    group_values: np.ndarray | None = None
    if groups is not None:
        group_values = np.asarray(groups)
        if len(group_values) != len(y):
            raise ValueError(
                f"groups length {len(group_values)} does not match "
                f"feature_df rows {len(y)}"
            )

    coefficients, residual_norm = nnls(X_with_const, y)

    rng = np.random.default_rng(random_seed)
    n = len(y)
    n_features = X_with_const.shape[1]
    # NaN-initialized so failed iterations are distinguishable from a legitimate
    # bootstrap draw of zero (which is the NNLS boundary). Downstream inference
    # in NNLSResults filters NaN rows out of std-error / percentile / p-value.
    bootstrap_coefs = np.full((n_bootstrap, n_features), np.nan)
    # Pre-draw all resample indices so seeding stays deterministic regardless
    # of how many iterations fail mid-fit. A session-aware campaign never
    # silently substitutes row resampling: one session leaves uncertainty
    # unavailable rather than fabricating independent replication.
    if group_values is None:
        all_indices = list(rng.integers(0, n, size=(n_bootstrap, n)))
    else:
        cluster_draws = _cluster_indices(group_values, rng, n_bootstrap)
        all_indices = cluster_draws if cluster_draws is not None else []
    for i, idx in enumerate(all_indices):
        with contextlib.suppress(Exception):
            bootstrap_y = y if bootstrap_target is None else bootstrap_target(i)
            if bootstrap_y is None or np.shape(bootstrap_y) != np.shape(y):
                continue
            coef_boot, _ = nnls(X_with_const[idx], bootstrap_y[idx])
            bootstrap_coefs[i] = coef_boot

    return NNLSResults(
        X=X_with_const,
        y=y,
        y_name=target,
        coefficients=coefficients,
        bootstrap_coefs=bootstrap_coefs,
        feature_names=feature_names,
        residual_norm=residual_norm,
        groups=group_values,
        uncertainty_conditional=uncertainty_conditional,
    )
