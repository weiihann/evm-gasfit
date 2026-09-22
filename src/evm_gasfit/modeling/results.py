"""NNLS regression results wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


class NNLSResults:
    """Wrap an NNLS fit with a statsmodels-style read-only surface."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_name: str,
        coefficients: np.ndarray,
        bootstrap_coefs: np.ndarray,
        feature_names: list[str],
        residual_norm: float,
        groups: np.ndarray | None = None,
        uncertainty_conditional: bool = False,
    ) -> None:
        self._X = X
        self._y = y
        self._dep_var = y_name
        self._coefficients = coefficients
        self._bootstrap_coefs = bootstrap_coefs
        self._feature_names = list(feature_names)
        self._residual_norm = residual_norm
        # Per-row cluster labels (e.g. session ids) when the fit was
        # session-aware; ``None`` for an ordinary row bootstrap. Retained so
        # downstream uncertainty propagation can resample the same clusters.
        self._groups = None if groups is None else np.asarray(groups)
        # A mixed glue fit can retain a valid point estimate while lacking the
        # joint partner draws needed for an unconditional interval.
        self._uncertainty_conditional = uncertainty_conditional

        self._fittedvalues = X @ coefficients
        self._resid = y - self._fittedvalues

        ss_res = float(np.sum(self._resid**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        self._rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n = len(y)
        k = len(coefficients) - 1  # exclude intercept
        if n > k + 1:
            self._rsquared_adj = 1.0 - (1.0 - self._rsquared) * (n - 1) / (n - k - 1)
        else:
            self._rsquared_adj = self._rsquared

        self._rmse = float(np.sqrt(np.mean(self._resid**2)))
        self._mae = float(np.mean(np.abs(self._resid)))

        # Successful-iteration mask: NNLS failures show up as all-NaN rows in
        # bootstrap_coefs and are excluded from inference. Cached so that
        # pvalues / conf_int / summary all see the same filtered population.
        self._success_mask: np.ndarray = ~np.isnan(bootstrap_coefs).any(axis=1)
        self._successful_bootstrap: np.ndarray = bootstrap_coefs[self._success_mask]
        self._n_bootstrap_total: int = len(bootstrap_coefs)
        self._n_bootstrap_success: int = int(self._success_mask.sum())

        # Lazy caches.
        self._params_series: pd.Series | None = None
        self._pvalues_series: pd.Series | None = None
        self._std_errors: np.ndarray | None = None
        self._condition_number: float | None = None

    @property
    def params(self) -> pd.Series:
        if self._params_series is None:
            self._params_series = pd.Series(
                self._coefficients, index=self._feature_names
            )
        return self._params_series

    @property
    def pvalues(self) -> pd.Series:
        if self._pvalues_series is None:
            self._pvalues_series = pd.Series(
                self._bootstrap_pvalues(), index=self._feature_names
            )
        return self._pvalues_series

    @property
    def rsquared(self) -> float:
        return self._rsquared

    @property
    def rsquared_adj(self) -> float:
        return self._rsquared_adj

    @property
    def nobs(self) -> int:
        return len(self._y)

    @property
    def X(self) -> np.ndarray:
        """The intercept-augmented design matrix the fit saw."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """The response vector the fit saw."""
        return self._y

    @property
    def resid(self) -> np.ndarray:
        return self._resid

    @property
    def groups(self) -> np.ndarray | None:
        """Per-row cluster labels, or ``None`` for an ordinary row bootstrap."""
        return self._groups

    @property
    def uncertainty_conditional(self) -> bool:
        """Whether the draws omit required supporting uncertainty."""
        return self._uncertainty_conditional

    @property
    def condition_number(self) -> float:
        """Condition number of the column-normalized design matrix.

        Columns are scaled by their own max absolute value before the 2-norm
        condition number is taken, so the metric measures **collinearity**
        (how separable the regressors are from each other and the intercept)
        rather than the arbitrary units of the regressors — an opcount sweep
        reaching 2.4e8 would otherwise dominate the ratio and flag every
        well-conditioned design. Large values flag near-collinear regressors
        — designs where the target coefficient and a support/setup feature
        are only separable thanks to noise. Qualification treats an excessive
        condition number as inconclusive rather than letting the fit stand.
        """
        if self._condition_number is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled = np.array(self._X, dtype=float, copy=True)
                for j in range(scaled.shape[1]):
                    col_max = np.max(np.abs(scaled[:, j]))
                    if col_max > 0:
                        scaled[:, j] = scaled[:, j] / col_max
                cond = float(np.linalg.cond(scaled))
            self._condition_number = cond if np.isfinite(cond) else float("inf")
        return self._condition_number

    def bootstrap_draws(self, feature_name: str) -> np.ndarray:
        """All successful bootstrap draws for one coefficient (may be empty)."""
        col = self._feature_names.index(feature_name)
        return self._successful_bootstrap[:, col]

    def bootstrap_draw_matrix(self, feature_name: str) -> np.ndarray:
        """Return every bootstrap draw for ``feature_name``, preserving failures."""
        col = self._feature_names.index(feature_name)
        return self._bootstrap_coefs[:, col]

    @property
    def session_ids(self) -> frozenset[str] | None:
        """Return the session identities that define cluster-bootstrap draws."""
        if self._groups is None:
            return None
        return frozenset(str(group) for group in pd.unique(self._groups))

    def bootstrap_draw(self, feature_name: str, rng: np.random.Generator) -> float:
        """Draw one coefficient value from the successful bootstrap draws."""
        if self._n_bootstrap_success == 0:
            raise ValueError("no successful bootstrap draws available")
        idx = int(rng.integers(0, self._n_bootstrap_success))
        col = self._feature_names.index(feature_name)
        return float(self._successful_bootstrap[idx, col])

    @property
    def fittedvalues(self) -> np.ndarray:
        return self._fittedvalues

    def _bootstrap_pvalues(self) -> np.ndarray:
        coefs = np.asarray(self._coefficients)
        n_success = self._n_bootstrap_success
        if n_success == 0:
            self._std_errors = np.full(coefs.shape, np.nan)
            return np.ones_like(coefs)

        samples = self._successful_bootstrap
        # Cache std errs while we're walking the bootstrap matrix.
        self._std_errors = np.std(samples, axis=0)
        # Vectorized percentile-style p-value:
        #   constrained-to-zero coefficients → 1.0
        #   else → mean(bootstrap_coef <= eps), floored at 1/n_success since
        #   "zero of n_success draws hit the boundary" means "p below the
        #   bootstrap resolution", not literally zero.
        p_below = (samples <= _EPS).mean(axis=0)
        p_below = np.maximum(p_below, 1.0 / n_success)
        zero_mask = coefs == 0
        return np.where(zero_mask, 1.0, p_below)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        if self._n_bootstrap_success == 0:
            nans = np.full(len(self._feature_names), np.nan)
            return pd.DataFrame({0: nans, 1: nans}, index=self._feature_names)
        samples = self._successful_bootstrap
        lower = np.percentile(samples, 100 * (alpha / 2), axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        return pd.DataFrame({0: lower, 1: upper}, index=self._feature_names)

    def summary(self) -> str:
        # Touch pvalues so std errors are populated.
        pvals = self.pvalues
        ci = self.conf_int()
        width = 78
        lines: list[str] = []
        lines.append("=" * width)
        lines.append(f"{'NNLS Regression Results':^{width}}")
        lines.append("=" * width)
        lines.append(
            f"Dep. Variable:          {self._dep_var}"
            f"{'R-squared:':>{width - 54}}{self.rsquared:>15.3f}"
        )
        lines.append(
            f"Model:                  NNLS"
            f"{'Adj. R-squared:':>{width - 43}}{self.rsquared_adj:>15.3f}"
        )
        lines.append(
            f"No. Observations:       {self.nobs:<7}"
            f"{'RMSE:':>{width - 46}}{self._rmse:>15.2f}"
        )
        lines.append(
            f"Df Residuals:           {self.nobs - len(self._feature_names):<7}"
            f"{'MAE:':>{width - 46}}{self._mae:>15.2f}"
        )
        lines.append(f"Df Model:               {len(self._feature_names) - 1:<7}")
        lines.append("=" * width)
        lines.append(
            f"{'':>14}{'coef':>12}{'std err':>12}{'P-value':>12}"
            f"{'[0.025':>12}{'0.975]':>12}"
        )
        lines.append("-" * width)
        std_errs = self._std_errors
        assert std_errs is not None  # populated by pvalues access above
        for i, name in enumerate(self._feature_names):
            coef = self.params[name]
            pval = pvals[name]
            ci_low = ci.loc[name, 0]
            ci_high = ci.loc[name, 1]
            se = std_errs[i]
            lines.append(
                f"{name:>14}{coef:>12.4f}{se:>12.4f}{pval:>12.3f}"
                f"{ci_low:>12.4f}{ci_high:>12.4f}"
            )
        lines.append("=" * width)
        if self._n_bootstrap_success == self._n_bootstrap_total:
            iter_note = f"({self._n_bootstrap_total} iterations)"
        else:
            iter_note = (
                f"({self._n_bootstrap_success} of "
                f"{self._n_bootstrap_total} iterations succeeded)"
            )
        lines.append(
            f"Notes: Non-negative least squares with bootstrap inference {iter_note}"
        )
        lines.append("=" * width)
        return "\n".join(lines)
