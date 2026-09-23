"""Per-group ratio/correlation detection of glue opcodes.

Groups fixtures by ``(source_label, test_name, target_opcode, *model_by)``
(no client axis — opcode counts are a property of the fixture). For each
non-target opcode the detector computes the Pearson correlation against
``opcount`` and the count-per-opcount slope; opcodes passing both
thresholds are returned.

Family-member columns (``DUP1``..``DUP16``, ``SWAP1``..``SWAP16``,
``PUSH1``..``PUSH32``) are folded into their canonical name (``DUP``,
``SWAP``, ``PUSH``) before the threshold pass, so a single family row is
emitted instead of one per member. When the target itself is a family
member (e.g. a ``DUP7`` variant), the *whole* canonical family is excluded
from the candidate universe: the family's own work is the target being
priced, and subtracting it as glue would double-count work that the family
parameter already charges for. The result drives ``glue_opcodes_by_test.csv``
and the missing-glue warning.

A spec with ``overhead_baseline_param`` set runs against the baseline-paired
delta (``modeling/estimate.py`` ``_split_baseline_pair``, applied to every
opcode-count column here, not just runtime) rather than raw counts. A glue
opcode whose count is identical between the paired fixtures — the case the
pairing exists to handle — deltas to a constant and fails the correlation
threshold on its own, so it's never flagged and never subtracted a second
time. One that scales with the target's own opcount (e.g. calling-convention
GAS/PUSH/POP around a CALL, which the ``True`` baseline drops along with the
target op) survives the diff and is detected exactly as it would be without
pairing.

Detection runs on the **target lane only** (``param_campaign_role`` ≠
``calibration``): ratios describe contamination inside priced target
workloads, and calibration-lane rows sharing a test name must not leak into
a target group.

Ratio estimators
----------------

``ratio`` is the OLS slope of support-count on opcount (design-consistent:
the target regression attributes per-opcount runtime to the slope, so the
contamination removed per unit opcount is the count slope, not counts
divided by opcount). ``ratio_endpoint_delta`` is the legacy mean-delta
secant estimator, retained for transparency. The two agree for a linear
count relation; material disagreement is recorded as
``ratio_reliable=False`` — a curved or variant-mixed relation has no single
per-opcount ratio, so the adjuster refuses to subtract it and blocks the
row's isolated recommendation instead of subtracting a biased amount.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from evm_gasfit.config import ModelSpec
from evm_gasfit.modeling.estimate import (
    _apply_filters,
    _materialize_derived,
    _resolve_target_opcode,
    _split_baseline_pair,
)

from .lane import split_campaign_lanes
from .required import CANONICAL_TO_MEMBERS, MEMBER_TO_CANONICAL, PRICED_GLUE_OPCODES

_log = logging.getLogger("evm_gasfit.glue")

_MIN_FIXTURE_POINTS = 5
_RATIO_FLOOR = 5e-4
# The OLS slope and the endpoint-delta secant must agree within this
# relative tolerance for the ratio to count as a usable linear per-opcount
# rate. Disagreement means curvature or variant mixing — no single ratio.
_RATIO_LINEARITY_TOLERANCE = 0.2
# Columns on ``fixtures_df`` that are not per-opcode counts.
_NON_OPCODE_COLUMNS: frozenset[str] = frozenset(
    {
        "client_name",
        "fixture_name",
        "test_file",
        "test_name",
        "test_runtime_ms",
        "target_opcode",
        "opcount",
    }
)

DETECTION_STATUS_EVALUATED = "evaluated"
DETECTION_STATUS_INSUFFICIENT_POINTS = "insufficient_fixture_points"
DETECTION_STATUS_AMBIGUOUS_TARGET = "ambiguous_target"


def _spec_groups(
    fixtures_df: pd.DataFrame, spec: ModelSpec, session_column: str = "session_id"
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    """Yield ``(group_values, group_df)`` per spec slice; empty when filters drop everything."""
    slice_df = fixtures_df[fixtures_df["test_name"] == spec.test_name]
    slice_df = _apply_filters(slice_df, spec.filter_by)
    if slice_df.empty:
        return []
    slice_df = _resolve_target_opcode(slice_df, spec)
    pairing_session = session_column if session_column in slice_df.columns else None
    slice_df = _split_baseline_pair(slice_df, spec, pairing_session)
    slice_df = _materialize_derived(slice_df, spec)

    for col in spec.model_by:
        if col not in slice_df.columns:
            return []

    out: list[tuple[dict[str, object], pd.DataFrame]] = []
    if spec.model_by:
        for key, group_df in slice_df.groupby(spec.model_by, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_values = {col: val for col, val in zip(spec.model_by, key_tuple)}
            out.append((group_values, group_df))
    else:
        out.append(({}, slice_df))
    return out


def _family_self_exclusion(target_opcode: str) -> set[str]:
    """Member columns to exclude because the target belongs to their family."""
    canonical = MEMBER_TO_CANONICAL.get(target_opcode)
    if canonical is None:
        return set()
    return set(CANONICAL_TO_MEMBERS[canonical]) - {target_opcode}


def _canonical_columns(
    agg: pd.DataFrame,
    target_opcode: str,
    opcode_columns: set[str],
    count_source: str | None = None,
) -> dict[str, np.ndarray]:
    """Fold actual EVM opcode count columns into canonical-name sums.

    CSV metadata may be numeric, but it is not execution work. The fixture
    builder records the exact opcount columns from the JSON mapping, which is
    the sole candidate universe here. Semantic precompile counters are target
    evidence, not opcode glue, and are excluded even if they happen to vary
    with the sweep. The target's own count column, its count source, and —
    when the target is a family member — every sibling in that canonical
    family are excluded so the target's own work is never offered back as
    glue to subtract from itself.
    """
    excluded = {"opcount", target_opcode} | _family_self_exclusion(target_opcode)
    if count_source is not None:
        excluded.add(count_source)
    raw_cols = [
        column
        for column in opcode_columns
        if column in agg.columns
        and column not in excluded
        and not column.startswith("PRECOMPILE_")
        and pd.api.types.is_numeric_dtype(agg[column])
    ]
    members_by_canonical: dict[str, list[str]] = {}
    for col in raw_cols:
        canonical = MEMBER_TO_CANONICAL.get(col, col)
        members_by_canonical.setdefault(canonical, []).append(col)
    return {
        canonical: agg[cols].astype(float).sum(axis=1).to_numpy()
        for canonical, cols in members_by_canonical.items()
    }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of ``y`` on ``x`` (with intercept)."""
    var_x = float(np.var(x))
    if var_x == 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / var_x)


def _passes_thresholds(
    counts: np.ndarray,
    opcount: np.ndarray,
    eps: float,
) -> tuple[bool, float, float, float, bool]:
    """Return ``(keep, corr, ratio_ols, ratio_endpoint_delta, ratio_reliable)``."""
    if np.std(counts) == 0 or np.std(opcount) == 0:
        return False, float("nan"), float("nan"), float("nan"), False
    corr = float(np.corrcoef(counts, opcount)[0, 1])
    ratio = _ols_slope(opcount.astype(float), counts.astype(float))

    order = np.argsort(opcount, kind="mergesort")
    x_sorted = opcount.astype(float)[order]
    y_sorted = counts.astype(float)[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    duplicate_conflict = False
    for x_value in np.unique(x_sorted):
        values = y_sorted[x_sorted == x_value]
        if len(values) > 1 and not np.allclose(values, values[0]):
            duplicate_conflict = True
        unique_x.append(float(x_value))
        unique_y.append(float(values[0]))
    x_unique = np.asarray(unique_x)
    y_unique = np.asarray(unique_y)
    d_count = np.diff(y_unique)
    d_opcount = np.diff(x_unique)
    endpoint = (
        float(d_count.mean() / d_opcount.mean()) if d_opcount.mean() != 0 else float("nan")
    )
    local_slopes = d_count / d_opcount if len(d_opcount) else np.array([])
    local_reliable = (
        not duplicate_conflict
        and len(local_slopes) > 0
        and np.all(np.isfinite(local_slopes))
        and np.all(local_slopes > 0)
        and np.all(
            np.abs(local_slopes - ratio) <= _RATIO_LINEARITY_TOLERANCE * ratio
        )
    )
    keep = corr >= (1 - eps) and ratio >= _RATIO_FLOOR
    reliable = (
        keep
        and np.isfinite(endpoint)
        and endpoint > 0
        and local_reliable
    )
    return keep, corr, ratio, endpoint, reliable


def compute_glue_opcodes_by_test(
    fixtures_df: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    eps: float,
    session_column: str = "session_id",
) -> pd.DataFrame:
    """Compute the per-test glue opcode ratio table (target lane only).

    Args:
        fixtures_df: Shared fixtures frame.
        model_specs: Iterable of validated ``ModelSpec`` objects.
        eps: ``ratio_corr_eps`` from config; keep opcodes with ``corr >= 1 - eps``.

    Returns:
        DataFrame with columns ``source_label``, ``test_name``,
        ``target_opcode``, every ``model_by`` column observed across specs,
        ``glue_opcode`` (canonical name), ``corr``, ``ratio`` (OLS slope),
        ``ratio_endpoint_delta``, ``ratio_reliable``, and
        ``collinear_with`` (``;``-joined canonical candidates whose counts
        move together within ``1 - eps`` — the evidence wrapper-bundle
        accounting uses to attribute an unpriced opcode to a priced one).
    """
    target_df, _ = split_campaign_lanes(fixtures_df)
    model_by_cols: list[str] = sorted(
        {c for spec in model_specs for c in spec.model_by}
    )
    opcode_columns = set(target_df.attrs.get("opcode_columns", []))
    rows: list[dict[str, object]] = []

    for spec in model_specs:
        for group_values, group_df in _spec_groups(target_df, spec, session_column):
            # Opcounts are a property of the fixture, not the client — collapse
            # to one row per fixture before correlating. Sorting by opcount
            # keeps the endpoint-based ratio well-defined.
            agg = (
                group_df.drop_duplicates(subset="fixture_name")
                .sort_values("opcount", kind="mergesort")
                .reset_index(drop=True)
            )
            if len(agg) < _MIN_FIXTURE_POINTS:
                continue
            target_opcode = agg["target_opcode"].iloc[0]
            opcount = agg["opcount"].astype(float).to_numpy()
            canonical_counts = _canonical_columns(
                agg,
                target_opcode,
                opcode_columns,
                getattr(spec, "target_operation_count_source", None),
            )
            kept: dict[str, np.ndarray] = {}
            stats: dict[str, tuple[float, float, float, bool]] = {}
            for canonical, counts in canonical_counts.items():
                keep, corr, ratio, endpoint, reliable = _passes_thresholds(
                    counts, opcount, eps
                )
                if not keep:
                    continue
                kept[canonical] = counts
                stats[canonical] = (corr, ratio, endpoint, reliable)
            # Pairwise collinearity among the group's detected candidates —
            # counts on the same fixtures, thresholded like detection.
            collinear: dict[str, str] = {}
            for canonical, counts in kept.items():
                partners = sorted(
                    other
                    for other, other_counts in kept.items()
                    if other != canonical
                    and np.std(other_counts) > 0
                    and float(np.corrcoef(counts, other_counts)[0, 1]) >= 1 - eps
                )
                collinear[canonical] = ";".join(partners)
            for canonical, (corr, ratio, endpoint, reliable) in stats.items():
                row: dict[str, object] = {
                    "source_label": spec.source_label,
                    "test_name": spec.test_name,
                    "target_opcode": target_opcode,
                }
                for mb in model_by_cols:
                    row[mb] = group_values.get(mb)
                row["glue_opcode"] = canonical
                row["corr"] = corr
                row["ratio"] = ratio
                row["ratio_endpoint_delta"] = endpoint
                row["ratio_reliable"] = bool(reliable)
                row["collinear_with"] = collinear.get(canonical, "")
                rows.append(row)

    cols = [
        "source_label",
        "test_name",
        "target_opcode",
        *model_by_cols,
        "glue_opcode",
        "corr",
        "ratio",
        "ratio_endpoint_delta",
        "ratio_reliable",
        "collinear_with",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    sort_cols = ["source_label", "test_name", "target_opcode", *model_by_cols, "glue_opcode"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def compute_detection_coverage(
    fixtures_df: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    eps: float,
    session_column: str = "session_id",
) -> pd.DataFrame:
    """Per ``(source_label, test_name, target_opcode, *model_by)`` detection coverage.

    Detection can only claim a group is clean when it actually ran: enough
    distinct fixture points to correlate against (``_MIN_FIXTURE_POINTS``)
    and an unambiguous single target (all rows one target opcode, or all
    members of one canonical family). Groups below that bar are recorded
    with ``detection_status`` != ``evaluated`` so downstream consumers block
    isolated recommendations instead of reading silence as "no support".

    Returns a DataFrame with ``source_label``, ``test_name``,
    ``target_opcode``, the ``model_by`` columns, ``n_fixture_points``,
    ``n_detected`` and ``detection_status``.
    """
    target_df, _ = split_campaign_lanes(fixtures_df)
    model_by_cols: list[str] = sorted(
        {c for spec in model_specs for c in spec.model_by}
    )
    opcode_columns = set(target_df.attrs.get("opcode_columns", []))
    rows: list[dict[str, object]] = []

    for spec in model_specs:
        for group_values, group_df in _spec_groups(target_df, spec, session_column):
            agg = (
                group_df.drop_duplicates(subset="fixture_name")
                .sort_values("opcount", kind="mergesort")
                .reset_index(drop=True)
            )
            targets = [str(t) for t in agg["target_opcode"].unique()]
            target_opcode = targets[0] if targets else ""
            if len(targets) > 1:
                families = {MEMBER_TO_CANONICAL.get(t, t) for t in targets}
                if len(families) > 1:
                    status = DETECTION_STATUS_AMBIGUOUS_TARGET
                else:
                    status = DETECTION_STATUS_EVALUATED
            else:
                status = DETECTION_STATUS_EVALUATED
            if status == DETECTION_STATUS_EVALUATED and len(agg) < _MIN_FIXTURE_POINTS:
                status = DETECTION_STATUS_INSUFFICIENT_POINTS
            n_detected = 0
            if status == DETECTION_STATUS_EVALUATED:
                opcount = agg["opcount"].astype(float).to_numpy()
                count_source = getattr(spec, "target_operation_count_source", None)
                for canonical, counts in _canonical_columns(
                    agg, target_opcode, opcode_columns, count_source
                ).items():
                    keep, *_ = _passes_thresholds(counts, opcount, eps)
                    n_detected += int(keep)
            row: dict[str, object] = {
                "source_label": spec.source_label,
                "test_name": spec.test_name,
                "target_opcode": target_opcode,
            }
            for mb in model_by_cols:
                row[mb] = group_values.get(mb)
            row["n_fixture_points"] = int(len(agg))
            row["n_detected"] = n_detected
            row["detection_status"] = status
            rows.append(row)

    cols = [
        "source_label",
        "test_name",
        "target_opcode",
        *model_by_cols,
        "n_fixture_points",
        "n_detected",
        "detection_status",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    sort_cols = [c for c in cols if c in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def detect_missing_glue(
    fixtures_df: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    eps: float,
    session_column: str = "session_id",
) -> list[tuple[str, str]]:
    """Return sorted ``(test_name, glue_opcode)`` pairs that meet the thresholds but aren't priced."""
    glue_df = compute_glue_opcodes_by_test(
        fixtures_df, model_specs, eps, session_column
    )
    if glue_df.empty:
        return []
    priced = set(PRICED_GLUE_OPCODES)
    pairs = {
        (str(r["test_name"]), str(r["glue_opcode"]))
        for _, r in glue_df.iterrows()
        if r["glue_opcode"] not in priced
    }
    return sorted(pairs)
