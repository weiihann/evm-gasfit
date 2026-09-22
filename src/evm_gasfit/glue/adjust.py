"""Apply the glue adjustment to each fitted target coefficient.

For every ``(test_name, target_opcode, *model_by, client)`` row in
``results_df``, subtract the contribution of every priced glue opcode that
correlates with the target on that test group: ratio × glue_runtime_ms. A
glue opcode's contribution is included only when its per-client fit passed
both quality gates — ``p_value < p_threshold`` and ``rsquared >= r2_threshold``
— so a noisy glue fit cannot pull the target coefficient down on the
strength of a slope it never measured reliably. Negative adjusted
coefficients are clipped to zero. The point-shifted bounds are retained only
as an explicitly conditional fallback when full uncertainty cannot propagate.

Uncertainty propagation
-----------------------

When the fit objects are available, a shared-session target and glue fit use
the same bootstrap replicate index: cluster draws are generated from stable
session IDs with the same seed, and failed refits stay aligned as unavailable
rows. This retains the session covariance that the cluster bootstrap exists
to represent. Fits with genuinely disjoint session sets may sample independent
marginal draws. Any partial/shared-but-unsynchronizable session overlap, absent
cluster identity, missing draw, or failed aligned refit leaves a point-shifted
interval marked ``glue_interval_conditional = True``; qualification then
withholds it from recommended pricing.

The detector's per-partner ``ratio`` is held fixed during propagation. Its
sampling noise is not available in the stored fit matrices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from evm_gasfit.modeling.results import NNLSResults

_log = logging.getLogger("evm_gasfit.glue")


def _model_by_cols(
    results_df: pd.DataFrame, glue_opcodes_by_test_df: pd.DataFrame
) -> list[str]:
    reserved = {
        "test_name",
        "client_name",
        "target_opcode",
        "glue_opcode",
        "corr",
        "ratio",
    }
    candidates = [c for c in glue_opcodes_by_test_df.columns if c not in reserved]
    return [c for c in candidates if c in results_df.columns]


def _propagated_interval(
    *,
    target_fit: NNLSResults,
    partners: list[tuple[float, NNLSResults, str]],
    confidence_level: float,
    rng: np.random.Generator,
    point_adjustment: float,
    point_target: float,
) -> tuple[float, float, bool]:
    """Adjusted CI from paired grouped-bootstrap draws.

    Returns ``(low, high, conditional)``. ``conditional`` is True when
    propagation could not run and the caller must fall back to the
    point-shifted interval.
    """
    if target_fit.uncertainty_conditional or any(
        partner_fit.uncertainty_conditional for _, partner_fit, _ in partners
    ):
        return float("nan"), float("nan"), True

    target_sessions = target_fit.session_ids
    target_shared_draws = target_fit.bootstrap_draw_matrix("opcount")
    shared_partners: list[tuple[float, np.ndarray]] = []
    independent_partners: list[tuple[float, np.ndarray]] = []

    for ratio, partner_fit, partner_name in partners:
        partner_sessions = partner_fit.session_ids
        if target_sessions is None and partner_sessions is None:
            partner_draws = partner_fit.bootstrap_draws(partner_name)
            if len(partner_draws) == 0:
                return float("nan"), float("nan"), True
            independent_partners.append((ratio, partner_draws))
            continue
        if target_sessions is None or partner_sessions is None:
            return float("nan"), float("nan"), True
        overlap = target_sessions & partner_sessions
        if overlap:
            # A partial overlap cannot be synchronized without reconstructing
            # both fits over a common campaign-session draw.
            if target_sessions != partner_sessions:
                return float("nan"), float("nan"), True
            partner_draws = partner_fit.bootstrap_draw_matrix(partner_name)
            if len(partner_draws) != len(target_shared_draws):
                return float("nan"), float("nan"), True
            shared_partners.append((ratio, partner_draws))
        else:
            partner_draws = partner_fit.bootstrap_draws(partner_name)
            if len(partner_draws) == 0:
                return float("nan"), float("nan"), True
            independent_partners.append((ratio, partner_draws))

    if shared_partners:
        target_draws = target_shared_draws
        valid = np.isfinite(target_draws)
        subtract = np.zeros(len(target_draws))
        for ratio, partner_draws in shared_partners:
            valid &= np.isfinite(partner_draws)
            subtract += ratio * partner_draws
    else:
        target_draws = target_fit.bootstrap_draws("opcount")
        if len(target_draws) == 0:
            return float("nan"), float("nan"), True
        valid = np.ones(len(target_draws), dtype=bool)
        subtract = np.zeros(len(target_draws))

    for ratio, partner_draws in independent_partners:
        subtract += (
            ratio
            * partner_draws[rng.integers(0, len(partner_draws), size=len(target_draws))]
        )
    adjusted = np.maximum(0.0, target_draws[valid] - subtract[valid])
    if len(adjusted) == 0:
        return float("nan"), float("nan"), True
    alpha = 1.0 - confidence_level
    low = float(np.quantile(adjusted, alpha / 2.0))
    high = float(np.quantile(adjusted, 1.0 - alpha / 2.0))
    _ = (point_adjustment, point_target)
    return low, high, False


def compute_glue_adjustment(
    results_df: pd.DataFrame,
    glue_results_df: pd.DataFrame,
    glue_opcodes_by_test_df: pd.DataFrame,
    p_threshold: float,
    r2_threshold: float,
    target_fits: dict[tuple, NNLSResults] | None = None,
    glue_fits: dict[tuple[str, str], NNLSResults] | None = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute per-row glue adjustment plus the clipped target coefficient.

    Returns a DataFrame keyed by ``(source_label, test_name, target_opcode,
    *model_by, client_name)`` with columns ``glue_adjustment``,
    ``adjusted_target_coef_runtime_ms``, ``adjusted_target_coef_conf_int_low``,
    ``adjusted_target_coef_conf_int_high``, and
    ``glue_interval_conditional``. ``source_label`` leads the key so two
    specs that share ``test_name`` + target + ``model_by`` and differ only
    in ``filter_by`` (e.g. a read/write split) each carry their own
    adjustment against their own fitted coefficient instead of colliding.
    """
    model_by_cols = _model_by_cols(results_df, glue_opcodes_by_test_df)
    key_cols = [
        "source_label",
        "test_name",
        "target_opcode",
        *model_by_cols,
        "client_name",
    ]
    rng = np.random.default_rng(random_seed)
    conditional_rows = 0

    rows: list[dict[str, object]] = []
    for _, row in results_df.iterrows():
        # `model_by_cols` is the union of every spec's `model_by` param
        # columns; a given row only populates its own spec's subset and
        # carries NaN in every other spec's columns. Restricting the mask to
        # this row's non-null columns recovers that row's own spec's key —
        # comparing the NaN-filled columns would always be False and zero
        # out `candidates` for every row (`NaN == NaN` is `False`).
        row_model_by_cols = [mb for mb in model_by_cols if pd.notna(row[mb])]
        ratio_mask = (glue_opcodes_by_test_df["test_name"] == row["test_name"]) & (
            glue_opcodes_by_test_df["target_opcode"] == row["target_opcode"]
        )
        for mb in row_model_by_cols:
            ratio_mask &= glue_opcodes_by_test_df[mb] == row[mb]
        candidates = glue_opcodes_by_test_df[ratio_mask]

        adjustment = 0.0
        # (ratio, partner_fit, partner_name) for every gate-passing partner.
        partners: list[tuple[float, NNLSResults, str]] = []
        if not candidates.empty and not glue_results_df.empty:
            glue_for_client = glue_results_df[
                glue_results_df["client_name"] == row["client_name"]
            ]
            for _, cand in candidates.iterrows():
                glue_row_mask = glue_for_client["glue_opcode"] == cand["glue_opcode"]
                glue_row = glue_for_client[glue_row_mask]
                if glue_row.empty:
                    continue
                pval = float(glue_row.iloc[0]["p_value"])
                r2 = float(glue_row.iloc[0]["rsquared"])
                glue_ms = float(glue_row.iloc[0]["glue_runtime_ms"])
                if (
                    np.isnan(pval)
                    or np.isnan(r2)
                    or np.isnan(glue_ms)
                    or pval >= p_threshold
                    or r2 < r2_threshold
                ):
                    continue
                ratio = float(cand["ratio"])
                adjustment += ratio * glue_ms
                if glue_fits is not None:
                    partner_fit = glue_fits.get(
                        (row["client_name"], cand["glue_opcode"])
                    )
                    if partner_fit is not None:
                        partners.append((ratio, partner_fit, cand["glue_opcode"]))

        target = float(row["target_coef_runtime_ms"])
        low = float(row["target_coef_conf_int_low"])
        high = float(row["target_coef_conf_int_high"])
        adjusted_target = max(0.0, target - adjustment)
        adjusted_low = max(0.0, low - adjustment)
        adjusted_high = max(0.0, high - adjustment)

        conditional = False
        target_fit = None
        if target_fits is not None:
            fit_key = (
                row["source_label"],
                row["test_name"],
                row["target_opcode"],
                *[row[mb] for mb in row_model_by_cols],
                row["client_name"],
            )
            target_fit = target_fits.get(fit_key)
        if adjustment > 0.0 and target_fit is not None and partners:
            p_low, p_high, conditional = _propagated_interval(
                target_fit=target_fit,
                partners=partners,
                confidence_level=confidence_level,
                rng=rng,
                point_adjustment=adjustment,
                point_target=target,
            )
            if not conditional:
                adjusted_low = p_low
                adjusted_high = p_high
        elif adjustment > 0.0:
            # No fit objects available — the point-shifted interval stands,
            # explicitly labeled conditional.
            conditional = True
        if conditional:
            conditional_rows += 1

        out_row: dict[str, object] = {
            "source_label": row["source_label"],
            "test_name": row["test_name"],
            "target_opcode": row["target_opcode"],
            "client_name": row["client_name"],
        }
        for mb in model_by_cols:
            out_row[mb] = row[mb]
        out_row.update(
            {
                "glue_adjustment": adjustment,
                "adjusted_target_coef_runtime_ms": adjusted_target,
                "adjusted_target_coef_conf_int_low": adjusted_low,
                "adjusted_target_coef_conf_int_high": adjusted_high,
                "glue_interval_conditional": conditional,
            }
        )
        rows.append(out_row)

    if conditional_rows:
        _log.warning(
            "conditional-glue-interval: %d adjusted interval(s) are conditional "
            "on point glue estimates — supporting-cost uncertainty was not "
            "propagated",
            conditional_rows,
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in key_cols if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
