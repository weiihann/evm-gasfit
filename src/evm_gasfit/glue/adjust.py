"""Apply the glue adjustment to each fitted target coefficient.

For every ``(source_label, test_name, target_opcode, *model_by, client)``
row in ``results_df``, subtract the contribution of every priced glue
opcode that correlates with the target on that test group: ratio ×
glue_runtime_ms. A glue opcode's contribution is included only when its
per-client fit passed both quality gates — ``p_value < p_threshold`` and
``rsquared >= r2_threshold`` — **and** the fit is ``isolated`` (every
count-correlated priced support in its own driver was subtracted), so a
noisy or bundled glue fit can neither pull the target coefficient down on
the strength of a slope it never measured reliably nor subtract a
coefficient that still contains un-attributed partner work. Negative
adjusted coefficients are clipped to zero. The point-shifted bounds are
retained only as an explicitly conditional fallback when full uncertainty
cannot propagate.

Coverage — never silently "isolated"
------------------------------------

Every detected candidate is classified per row:

- ``glue_priced_opcodes`` — subtracted (gate-passing, isolated, reliable
  ratio).
- ``glue_bundled_opcodes`` — not separately priced, but attributed to a
  subtracted partner via wrapper-bundle accounting: the candidate is
  count-collinear with a priced partner on this target group, and that
  partner's calibration driver carries at least as much of the candidate
  per unit (e.g. STOP 1:1 inside a STOP-only STATICCALL callee), so the
  partner subtraction already removed it.
- ``glue_unpriced_opcodes`` — detected, neither subtractable nor covered.
  Its cost remains inside the target coefficient, so the adjusted estimate
  is *not* an isolated cost.

``glue_detection_status`` records whether detection ran at all for the
group (``evaluated`` / ``insufficient_fixture_points`` /
``ambiguous_target``); ``glue_coverage_complete`` is true only when
detection ran, every candidate is priced or bundled, and every kept ratio
was linear-reliable. Rows with incomplete coverage downgrade
``adjusted_estimate_status`` downstream and are blocked from recommended
prices — an un-removable correlated supporter must never silently become
a zero-cost footnote in a deployable price.

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

The detector's per-partner ``ratio`` (OLS count slope) is held fixed during
propagation. Its sampling noise is not available in the stored fit matrices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from evm_gasfit.modeling.results import NNLSResults

from .required import SHAPE_PARAM_BY_SPEC, SPEC_BY_NAME

_log = logging.getLogger("evm_gasfit.glue")

_BUNDLE_SLACK = 1e-9


def _model_by_cols(
    results_df: pd.DataFrame, glue_opcodes_by_test_df: pd.DataFrame
) -> list[str]:
    reserved = {
        "source_label",
        "test_name",
        "client_name",
        "target_opcode",
        "glue_opcode",
        "corr",
        "ratio",
        "ratio_endpoint_delta",
        "ratio_reliable",
        "collinear_with",
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


def _driver_bundle_lookup(
    driver_support_df: pd.DataFrame | None,
) -> dict[tuple[str, str], float]:
    """Map ``(priced driver, support opcode)`` → driver-side per-count ratio."""
    if driver_support_df is None or driver_support_df.empty:
        return {}
    out: dict[tuple[str, str], float] = {}
    for _, row in driver_support_df.iterrows():
        support = str(row["support_opcode"])
        if _candidate_is_priced(support):
            continue
        key = (str(row["glue_opcode"]), support)
        out[key] = float(row["ratio_per_driver_count"])
    return out


def _candidate_is_priced(name: str) -> bool:
    spec = SPEC_BY_NAME.get(name)
    return spec is not None and spec.test_name is not None


def _glue_row_for(
    glue_results_df: pd.DataFrame, client: str, name: str
) -> pd.Series | None:
    if glue_results_df.empty or "glue_opcode" not in glue_results_df.columns:
        return None
    sub = glue_results_df[
        (glue_results_df["client_name"] == client)
        & (glue_results_df["glue_opcode"] == name)
    ]
    return None if sub.empty else sub.iloc[0]


def _shape_verified(name: str, target_row: pd.Series, candidate_row: pd.Series) -> bool:
    """Return whether an input-dependent glue rate has matching shape evidence."""
    if name != "CALLDATACOPY":
        return True
    shape_columns = SHAPE_PARAM_BY_SPEC.get(name, ())
    return bool(shape_columns) and all(
        column in target_row.index
        and column in candidate_row.index
        and pd.notna(target_row[column])
        and pd.notna(candidate_row[column])
        and str(target_row[column]) == str(candidate_row[column])
        for column in shape_columns
    )


def _coverage_status(
    detection_coverage_df: pd.DataFrame | None,
    source_label: object,
    test_name: object,
    target_opcode: object,
    model_by_values: dict[str, object],
) -> str:
    if detection_coverage_df is None or detection_coverage_df.empty:
        return "evaluated"
    mask = (
        (detection_coverage_df["source_label"].astype(str) == str(source_label))
        & (detection_coverage_df["test_name"].astype(str) == str(test_name))
        & (detection_coverage_df["target_opcode"].astype(str) == str(target_opcode))
    )
    for col, value in model_by_values.items():
        if col in detection_coverage_df.columns:
            mask &= detection_coverage_df[col].astype(str) == str(value)
    sub = detection_coverage_df[mask]
    if sub.empty:
        return "evaluated"
    return str(sub.iloc[0]["detection_status"])


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
    detection_coverage_df: pd.DataFrame | None = None,
    driver_support_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-row glue adjustment plus the clipped target coefficient.

    Returns a DataFrame keyed by ``(source_label, test_name, target_opcode,
    *model_by, client_name)`` with columns ``glue_adjustment``,
    ``adjusted_target_coef_runtime_ms``, ``adjusted_target_coef_conf_int_low``,
    ``adjusted_target_coef_conf_int_high``, ``glue_interval_conditional``,
    plus the coverage ledger: ``glue_priced_opcodes``,
    ``glue_bundled_opcodes``, ``glue_unpriced_opcodes``,
    ``glue_coverage_reason``, ``glue_detection_status`` and
    ``glue_coverage_complete``.
    ``source_label`` leads the key so two specs that share ``test_name`` +
    target + ``model_by`` and differ only in ``filter_by`` (e.g. a
    read/write split) each carry their own adjustment against their own
    fitted coefficient instead of colliding.
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
    driver_bundles = _driver_bundle_lookup(driver_support_df)

    rows: list[dict[str, object]] = []
    for _, row in results_df.iterrows():
        # `model_by_cols` is the union of every spec's `model_by` param
        # columns; a given row only populates its own spec's subset and
        # carries NaN in every other spec's columns. Restricting the mask to
        # this row's non-null columns recovers that row's own spec's key —
        # comparing the NaN-filled columns would always be False and zero
        # out `candidates` for every row (`NaN == NaN` is `False`).
        row_model_by_cols = [mb for mb in model_by_cols if pd.notna(row[mb])]
        model_by_values = {mb: row[mb] for mb in row_model_by_cols}
        ratio_mask = (
            (glue_opcodes_by_test_df["test_name"] == row["test_name"])
            & (glue_opcodes_by_test_df["target_opcode"] == row["target_opcode"])
            & (glue_opcodes_by_test_df["source_label"] == row["source_label"])
        )
        for mb in row_model_by_cols:
            ratio_mask &= glue_opcodes_by_test_df[mb] == row[mb]
        candidates = glue_opcodes_by_test_df[ratio_mask]

        adjustment = 0.0
        # (ratio, partner_fit, partner_name) for every gate-passing partner.
        partners: list[tuple[float, NNLSResults, str]] = []
        priced: list[str] = []
        unpriced: list[str] = []
        shape_unverified: set[str] = set()
        coverage_reasons: list[str] = []
        # candidate ratio by name, for bundle attribution math.
        ratios: dict[str, float] = {}
        collinear_by_name: dict[str, set[str]] = {}
        if not candidates.empty:
            for _, cand in candidates.iterrows():
                name = str(cand["glue_opcode"])
                ratio = float(cand["ratio"])
                ratios[name] = ratio
                collinear_by_name[name] = {
                    p
                    for p in str(cand.get("collinear_with") or "").split(";")
                    if p and p != "nan"
                }
                shape_verified = _shape_verified(name, row, cand)
                reliable = bool(cand.get("ratio_reliable", True))
                glue_row = _glue_row_for(
                    glue_results_df, str(row["client_name"]), name
                )
                usable = (
                    shape_verified
                    and reliable
                    and _candidate_is_priced(name)
                    and glue_row is not None
                    and np.isfinite(float(glue_row["glue_runtime_ms"]))
                    and np.isfinite(float(glue_row["p_value"]))
                    and np.isfinite(float(glue_row["rsquared"]))
                    and float(glue_row["p_value"]) < p_threshold
                    and float(glue_row["rsquared"]) >= r2_threshold
                    and (
                        "isolated" not in glue_row.index
                        or bool(glue_row["isolated"])
                    )
                )
                if not usable:
                    unpriced.append(name)
                    if not shape_verified:
                        coverage_reasons.append(f"{name}: input shape unverified")
                        shape_unverified.add(name)
                    continue
                adjustment += ratio * float(glue_row["glue_runtime_ms"])
                priced.append(name)
                if glue_fits is not None:
                    partner_fit = glue_fits.get(
                        (row["client_name"], name)
                    )
                    if partner_fit is not None:
                        partners.append((ratio, partner_fit, name))

        # Wrapper-bundle attribution is exact aggregate accounting. Every
        # priced partner is checked, so an embedded STOP absent from the
        # target is an explicit zero-vs-positive mismatch.
        bundled: list[str] = []
        still_unpriced = list(unpriced)
        target_unpriced = {
            name
            for name in unpriced
            if name not in shape_unverified and not _candidate_is_priced(name)
        }
        aggregate: dict[str, float] = {}
        for partner in priced:
            partner_ratio = ratios.get(partner, 0.0)
            if partner_ratio <= 0:
                continue
            for (driver, support), driver_ratio in driver_bundles.items():
                if driver != partner or _candidate_is_priced(support):
                    continue
                aggregate[support] = (
                    aggregate.get(support, 0.0) + partner_ratio * driver_ratio
                )
        mismatches = {
            support
            for support in set(target_unpriced) | set(aggregate)
            if abs(
                aggregate.get(support, 0.0)
                - (ratios.get(support, 0.0) if support in target_unpriced else 0.0)
            )
            > _BUNDLE_SLACK
        }
        if not mismatches and target_unpriced:
            bundled = sorted(target_unpriced)
            still_unpriced = [
                name for name in still_unpriced if name not in target_unpriced
            ]
        elif mismatches:
            coverage_reasons.append(
                "bundle composition mismatch: "
                + ", ".join(sorted(mismatches))
            )
            still_unpriced = sorted(set(still_unpriced) | mismatches)
        unpriced = sorted(still_unpriced)

        detection_status = _coverage_status(
            detection_coverage_df,
            row["source_label"],
            row["test_name"],
            row["target_opcode"],
            model_by_values,
        )
        unreliable = [
            str(cand["glue_opcode"])
            for _, cand in candidates.iterrows()
            if not bool(cand.get("ratio_reliable", True))
        ]
        coverage_complete = (
            detection_status == "evaluated"
            and not unpriced
            and not unreliable
            and not mismatches
            and not coverage_reasons
        )
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
                "glue_priced_opcodes": ";".join(sorted(priced)),
                "glue_bundled_opcodes": ";".join(sorted(bundled)),
                "glue_unpriced_opcodes": ";".join(sorted(unpriced)),
                "glue_coverage_reason": "; ".join(sorted(set(coverage_reasons))),
                "glue_detection_status": detection_status,
                "glue_coverage_complete": coverage_complete,
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
    incomplete = sum(
        1 for r in rows if not bool(r.get("glue_coverage_complete", True))
    )
    if incomplete:
        _log.warning(
            "glue-coverage: %d adjusted estimate(s) have incomplete supporting-"
            "cost coverage (unpriced or unreliable detected supporters, or "
            "detection could not run); isolated recommendations are blocked",
            incomplete,
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in key_cols if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
