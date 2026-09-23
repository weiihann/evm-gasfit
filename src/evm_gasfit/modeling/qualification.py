"""Explicit qualification status for every planned model.

A planned model is each ``(spec, model_by-combo, client)`` fit the config
asks for. Every planned model receives exactly one status:

- ``failed`` — the primary NNLS solver raised; its error is retained.
- ``inconclusive`` — expected evidence is absent (no matching fixtures,
  insufficient observations, rank-deficient design, or constant target count)
  or an enabled gate failed: ill conditioning, systematic residual structure,
  uncertainty beyond the tolerated band, held-out prediction error beyond
  tolerance, too few sessions, or (when ``enforce_fit_quality`` is on) the
  poor-fit thresholds. Inconclusive models keep their numeric estimates as
  research observations but cannot become recommended prices when
  ``block_unqualified`` is set.
- ``qualified`` — every armed gate passed.

The workload-slope status and the adjusted target-operation status are
kept distinct: glue/overhead adjustment shifts the coefficient, and an
interval that is only point-adjusted (uncertainty not propagated) can hold
the slope's qualification while the adjusted estimate is reported
conditional — and treated conservatively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evm_gasfit.config import Config

from .diagnostics import (
    leave_one_group_out_error,
    leave_one_point_out_error,
    residual_curvature_r2,
)
from .estimate import PlannedFit

STATUS_QUALIFIED = "qualified"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_FAILED = "failed"

QUALIFICATION_COLUMNS = [
    "source_label",
    "test_name",
    "target_opcode",
    "client_name",
    "status",
    "reasons",
    "adjusted_estimate_status",
    "nobs",
    "n_sessions",
    "condition_number",
    "residual_curvature_r2",
    "holdout_session_error",
    "holdout_point_error",
    "relative_ci_width",
    "confidence_level",
]


def fit_key(
    source_label: str,
    test_name: str,
    target_opcode: object,
    group_values: dict[str, object],
    client: str,
) -> tuple:
    """The canonical fit-identity key shared with ``results_df`` routing."""
    return (
        str(source_label),
        str(test_name),
        str(target_opcode or ""),
        *[str(group_values[c]) for c in sorted(group_values)],
        str(client),
    )


@dataclass
class _FitEvidence:
    nobs: float
    n_sessions: float
    condition_number: float
    curvature: float
    holdout_session: float
    holdout_point: float
    relative_ci_width: float
    coefficient_ci_widths: dict[str, float]
    coefficient_pvalues: dict[str, float]
    rsquared: float


def _collect_evidence(config: Config, record: PlannedFit) -> _FitEvidence:
    fit = record.fit
    assert fit is not None  # callers check before invoking
    groups = fit.groups
    n_sessions = float(len(np.unique(groups))) if groups is not None else float("nan")
    holdout_session = (
        leave_one_group_out_error(fit.X, fit.y, groups)
        if groups is not None
        else float("nan")
    )
    # ``features`` always leads with ``opcount`` (estimate._build_design), so
    # column 1 of the intercept-augmented matrix is the target count.
    holdout_point = leave_one_point_out_error(fit.X, fit.y, fit.X[:, 1])
    curvature = residual_curvature_r2(fit.y, fit.fittedvalues, fit.resid)
    alpha = 1.0 - config.qualification.confidence_level
    ci = fit.conf_int(alpha=alpha)
    widths: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    for feature in fit.params.index:
        if feature == "const":
            continue
        point = float(fit.params[feature])
        low = float(ci.loc[feature, 0])
        high = float(ci.loc[feature, 1])
        widths[feature] = (
            float("nan")
            if (
                not np.isfinite(point)
                or point <= 0
                or not np.isfinite(low)
                or not np.isfinite(high)
            )
            else (high - low) / point
        )
        pvalues[feature] = float(fit.pvalues[feature])
    return _FitEvidence(
        nobs=float(fit.nobs),
        n_sessions=n_sessions,
        condition_number=float(fit.condition_number),
        curvature=float(curvature),
        holdout_session=float(holdout_session),
        holdout_point=float(holdout_point),
        relative_ci_width=widths["opcount"],
        coefficient_ci_widths=widths,
        coefficient_pvalues=pvalues,
        rsquared=float(fit.rsquared),
    )


def _skipped_status(reason: str | None) -> str:
    """Classify an absent fit without conflating missing evidence and failure."""
    if reason is not None and reason.startswith("NNLS solver raised"):
        return STATUS_FAILED
    return STATUS_INCONCLUSIVE


def _gate_reasons(config: Config, record: PlannedFit, ev: _FitEvidence) -> list[str]:
    q = config.qualification
    reasons: list[str] = []
    if record.dropped_features:
        features = ", ".join(repr(name) for name in sorted(record.dropped_features))
        reasons.append(
            "configured coefficient(s) were not estimated because they had one "
            f"observed value: {features}"
        )

    # The new design/residual gates diagnose campaign sessions. Applying them
    # to legacy row-bootstrap inputs would retroactively withdraw established
    # prices without supplying the session evidence those gates require.
    session_aware = record.fit is not None and record.fit.groups is not None
    if session_aware and not np.isfinite(ev.condition_number):
        reasons.append("condition number is not finite")
    elif session_aware and ev.condition_number > q.max_condition_number:
        reasons.append(
            f"condition number {ev.condition_number:.3g} exceeds "
            f"{q.max_condition_number:.3g}"
        )
    if session_aware and not np.isfinite(ev.curvature):
        reasons.append(
            "residual curvature is not computable because fitted values do not "
            "identify a quadratic diagnostic"
        )
    elif session_aware and ev.curvature > q.max_residual_curvature_r2:
        reasons.append(
            f"systematic residual curvature R2 {ev.curvature:.3f} exceeds "
            f"{q.max_residual_curvature_r2:.3g}"
        )

    if record.fit is not None and record.fit.groups is not None and ev.n_sessions < 2:
        reasons.append(
            "session-cluster uncertainty unavailable with fewer than two sessions"
        )

    if q.max_relative_uncertainty is not None:
        for feature, width in ev.coefficient_ci_widths.items():
            label = "target_coef" if feature == "opcount" else feature
            if not np.isfinite(width):
                reasons.append(
                    f"relative CI width for coefficient {label!r} is not computable "
                    "(zero-pinned coefficient or no successful bootstrap draws) "
                    "while an uncertainty gate is armed"
                )
            elif width > q.max_relative_uncertainty:
                reasons.append(
                    f"relative CI width for coefficient {label!r} is {width:.3f}, "
                    f"exceeds {q.max_relative_uncertainty:.3g} at confidence "
                    f"{q.confidence_level:.2f}"
                )

    if q.max_holdout_error is not None:
        if record.fit is not None and record.fit.groups is not None:
            if np.isnan(ev.holdout_session):
                reasons.append(
                    "session holdout error not computable while a holdout gate "
                    "is armed (fewer than two sessions or a refit failed)"
                )
            elif ev.holdout_session > q.max_holdout_error:
                reasons.append(
                    f"held-out session error {ev.holdout_session:.3f} exceeds "
                    f"{q.max_holdout_error:.3g}"
                )
        if np.isnan(ev.holdout_point):
            reasons.append(
                "workload-point holdout error not computable while a holdout "
                "gate is armed (a leave-one-point-out refit failed)"
            )
        elif ev.holdout_point > q.max_holdout_error:
            reasons.append(
                f"held-out workload-point error {ev.holdout_point:.3f} exceeds "
                f"{q.max_holdout_error:.3g}"
            )

    if q.min_sessions is not None:
        if record.fit is None or record.fit.groups is None:
            reasons.append(
                "session gate armed but the runtimes input carries no session column"
            )
        elif ev.n_sessions < q.min_sessions:
            reasons.append(
                f"{int(ev.n_sessions)} session(s) present, fewer than the "
                f"required {q.min_sessions}"
            )

    if q.enforce_fit_quality:
        for feature, pvalue in ev.coefficient_pvalues.items():
            label = "target_coef" if feature == "opcount" else feature
            if not np.isfinite(pvalue):
                reasons.append(
                    f"p-value for coefficient {label!r} is not computable while "
                    "the fit-quality gate is armed"
                )
            elif pvalue >= config.modeling.poor_fit_p_value_threshold:
                reasons.append(
                    f"p-value for coefficient {label!r} is {pvalue:.3g}, fails "
                    "the fit-quality threshold "
                    f"{config.modeling.poor_fit_p_value_threshold}"
                )
        if not np.isfinite(ev.rsquared):
            reasons.append(
                "R-squared is not computable while the fit-quality gate is armed"
            )
        elif ev.rsquared < config.modeling.poor_fit_rsquared_threshold:
            reasons.append(
                f"R-squared {ev.rsquared:.3f} below the fit-quality threshold "
                f"{config.modeling.poor_fit_rsquared_threshold}"
            )
    return reasons


def evaluate_qualification(config: Config, planned: list[PlannedFit]) -> pd.DataFrame:
    """Compute the qualification status of every planned model."""
    rows: list[dict[str, object]] = []
    model_by_cols = sorted({c for r in planned for c in r.group_values})
    for record in planned:
        row: dict[str, object] = {
            "source_label": record.source_label,
            "test_name": record.test_name,
            "target_opcode": record.target_opcode or "",
            "client_name": record.client,
            "model_by_cols": None,
        }
        for col in model_by_cols:
            row[f"model_by::{col}"] = record.group_values.get(col)
        if record.fit is None:
            status = _skipped_status(record.skip_reason)
            row.update(
                {
                    "status": status,
                    "reasons": record.skip_reason or "fit did not run",
                    "adjusted_estimate_status": status,
                    "nobs": 0,
                    "n_sessions": float("nan"),
                    "condition_number": float("nan"),
                    "residual_curvature_r2": float("nan"),
                    "holdout_session_error": float("nan"),
                    "holdout_point_error": float("nan"),
                    "relative_ci_width": float("nan"),
                    "confidence_level": config.qualification.confidence_level,
                }
            )
        else:
            ev = _collect_evidence(config, record)
            reasons = _gate_reasons(config, record, ev)
            row.update(
                {
                    "status": (STATUS_INCONCLUSIVE if reasons else STATUS_QUALIFIED),
                    "reasons": "; ".join(reasons),
                    # Provisional; the proposal stage downgrades this when the
                    # adjusted interval is only point-shifted (conditional).
                    "adjusted_estimate_status": (
                        STATUS_INCONCLUSIVE if reasons else STATUS_QUALIFIED
                    ),
                    "nobs": ev.nobs,
                    "n_sessions": ev.n_sessions,
                    "condition_number": ev.condition_number,
                    "residual_curvature_r2": ev.curvature,
                    "holdout_session_error": ev.holdout_session,
                    "holdout_point_error": ev.holdout_point,
                    "relative_ci_width": ev.relative_ci_width,
                    "confidence_level": config.qualification.confidence_level,
                }
            )
        rows.append(row)

    # ``model_by::<col>`` keys become plain columns; the prefix kept them
    # sorted-stable while records carried heterogeneous spec shapes.
    df = pd.DataFrame(rows)
    rename = {c: c.split("model_by::", 1)[1] for c in df.columns if "model_by::" in c}
    df = df.rename(columns=rename).drop(columns=["model_by_cols"], errors="ignore")
    front = [c for c in QUALIFICATION_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df.sort_values(
        ["source_label", "test_name", "client_name"], kind="mergesort"
    ).reset_index(drop=True)


def adjusted_interval_reasons(
    config: Config,
    glue_adjustment_df: pd.DataFrame | None,
    planned: list[PlannedFit],
) -> dict[tuple, list[str]]:
    """Gate the *adjusted* estimate on its propagated interval.

    The raw-status relative-CI gate inspects the unadjusted point and
    interval; a small subtraction can leave an adjusted interval that
    spans zero or whose relative width balloons far past the configured
    tolerance while the raw row stays qualified. The adjusted estimate is
    what feeds an isolated recommendation, so it gets its own check:
    finite bounds, a strictly positive point and upper bound, and — when
    ``max_relative_uncertainty`` is armed — the same relative-width bound
    computed on the adjusted interval. Raw ``status`` is never touched.
    """
    if glue_adjustment_df is None or glue_adjustment_df.empty:
        return {}
    q = config.qualification
    max_rel = q.max_relative_uncertainty
    confidence = q.confidence_level
    by_key: dict[tuple, dict[str, object]] = {}
    model_by_cols = sorted(
        c for c in glue_adjustment_df.columns if c.startswith("param_")
    )
    for _, row in glue_adjustment_df.iterrows():
        if bool(row.get("glue_interval_conditional", False)):
            continue  # conditional intervals are downgraded elsewhere
        key = (
            str(row["source_label"]),
            str(row["test_name"]),
            str(row["target_opcode"]),
            *[str(row[mb]) for mb in model_by_cols if pd.notna(row[mb])],
            str(row["client_name"]),
        )
        by_key[key] = row
    out: dict[tuple, list[str]] = {}
    for record in planned:
        key = fit_key(
            record.source_label,
            record.test_name,
            record.target_opcode,
            record.group_values,
            record.client,
        )
        row = by_key.get(key)
        if row is None:
            continue
        reasons: list[str] = []
        point = float(row["adjusted_target_coef_runtime_ms"])
        low = float(row["adjusted_target_coef_conf_int_low"])
        high = float(row["adjusted_target_coef_conf_int_high"])
        if not (np.isfinite(point) and np.isfinite(low) and np.isfinite(high)):
            reasons.append(
                "adjusted estimate interval is not finite after glue subtraction"
            )
        else:
            if low > high or point < low or point > high:
                reasons.append(
                    f"adjusted interval [{low:.3g}, {high:.3g}] does not "
                    f"contain point estimate {point:.3g} in ordered bounds"
                )
            if point <= 0:
                reasons.append(
                    f"adjusted point estimate {point:.3g} is not positive after "
                    "glue subtraction"
                )
            if low < 0 or high <= 0:
                reasons.append(
                    f"adjusted interval [{low:.3g}, {high:.3g}] cannot support a "
                    "positive isolated price (lower bound negative or upper "
                    "bound non-positive)"
                )
            if max_rel is not None and point > 0:
                width = (high - low) / point
                if not np.isfinite(width):
                    reasons.append(
                        "adjusted relative CI width is not computable while an "
                        "uncertainty gate is armed"
                    )
                elif width > max_rel:
                    reasons.append(
                        f"adjusted relative CI width {width:.3f} exceeds "
                        f"{max_rel:.3g} at confidence {confidence:.2f}"
                    )
        if reasons:
            out[key] = reasons
    return out


def apply_adjusted_statuses(
    qualification_df: pd.DataFrame,
    downgrades: dict[tuple, list[str]],
    planned: list[PlannedFit],
) -> pd.DataFrame:
    """Downgrade adjusted estimates whose supporting evidence is unusable.

    ``downgrades`` maps the canonical ``fit_key`` to reason strings: a
    point-shifted (conditional) interval, incomplete supporting-cost
    coverage, or an adjusted interval that fails the propagated-interval
    gate. Each downgrades ``adjusted_estimate_status`` to inconclusive with
    the explicit reasons appended; ``status`` (the raw timing-model
    verdict) is preserved so budget-style consumers can still use the raw
    qualified estimate. A point-shifted interval or unresolved coverage
    disqualifies the row from recommended pricing regardless of any
    optional CI-width gate.
    """
    if qualification_df.empty or not downgrades:
        return qualification_df
    out = qualification_df.copy()
    for record in planned:
        key = fit_key(
            record.source_label,
            record.test_name,
            record.target_opcode,
            record.group_values,
            record.client,
        )
        reasons = downgrades.get(key)
        if not reasons:
            continue
        mask = (
            (out["source_label"] == record.source_label)
            & (out["test_name"] == record.test_name)
            & (out["client_name"] == record.client)
        )
        if record.target_opcode:
            mask &= out["target_opcode"] == record.target_opcode
        for col, value in record.group_values.items():
            if col not in out.columns:
                continue
            if pd.isna(value):
                mask &= out[col].isna()
            else:
                mask &= out[col] == value
        if not mask.any():
            continue
        if out.loc[mask, "status"].eq(STATUS_QUALIFIED).all():
            out.loc[mask, "adjusted_estimate_status"] = STATUS_INCONCLUSIVE
            prev = out.loc[mask, "reasons"].astype(str)
            out.loc[mask, "reasons"] = prev.map(
                lambda r: r + "".join(
                    f"; {reason}" if reason not in r else "" for reason in reasons
                )
                if r
                else "; ".join(reasons)
            )
    return out
