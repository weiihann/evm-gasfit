"""Per-group ratio/correlation detection of glue opcodes.

Groups fixtures by ``(test_name, target_opcode, *model_by)`` (no client axis —
opcode counts are a property of the fixture). For each non-target opcode the
detector computes the Pearson correlation against ``opcount`` and the mean
delta ratio; opcodes passing both thresholds are returned.

Family-member columns (``DUP1``..``DUP16``, ``SWAP1``..``SWAP16``,
``PUSH1``..``PUSH32``) are folded into their canonical name (``DUP``,
``SWAP``, ``PUSH``) before the threshold pass, so a single family row is
emitted instead of one per member. The result drives
``glue_opcodes_by_test.csv`` and the missing-glue warning.

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

from .required import MEMBER_TO_CANONICAL, PRICED_GLUE_OPCODES

_log = logging.getLogger("evm_gasfit.glue")

_MIN_FIXTURE_POINTS = 5
_RATIO_FLOOR = 5e-4
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
    with the sweep.
    """
    excluded = {"opcount", target_opcode}
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


def _passes_thresholds(
    counts: np.ndarray,
    opcount: np.ndarray,
    eps: float,
) -> tuple[bool, float, float]:
    if np.std(counts) == 0 or np.std(opcount) == 0:
        return False, float("nan"), float("nan")
    corr = float(np.corrcoef(counts, opcount)[0, 1])
    d_count = np.diff(counts)
    d_opcount = np.diff(opcount)
    if d_opcount.mean() == 0:
        return False, corr, float("nan")
    ratio = float(d_count.mean() / d_opcount.mean())
    keep = corr >= (1 - eps) and ratio >= _RATIO_FLOOR
    return keep, corr, ratio


def compute_glue_opcodes_by_test(
    fixtures_df: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    eps: float,
    session_column: str = "session_id",
) -> pd.DataFrame:
    """Compute the per-test glue opcode ratio table.

    Args:
        fixtures_df: Shared fixtures frame.
        model_specs: Iterable of validated ``ModelSpec`` objects.
        eps: ``ratio_corr_eps`` from config; keep opcodes with ``corr >= 1 - eps``.

    Returns:
        DataFrame with columns ``test_name``, ``target_opcode``, every
        ``model_by`` column observed across specs, ``glue_opcode``
        (canonical name), ``corr``, ``ratio``.
    """
    model_by_cols: list[str] = sorted(
        {c for spec in model_specs for c in spec.model_by}
    )
    opcode_columns = set(fixtures_df.attrs.get("opcode_columns", []))
    rows: list[dict[str, object]] = []

    for spec in model_specs:
        for group_values, group_df in _spec_groups(fixtures_df, spec, session_column):
            # Opcounts are a property of the fixture, not the client — collapse
            # to one row per fixture before correlating. Sorting by opcount
            # keeps the endpoint-based ratio (np.diff().mean()) well-defined.
            agg = (
                group_df.drop_duplicates(subset="fixture_name")
                .sort_values("opcount", kind="mergesort")
                .reset_index(drop=True)
            )
            if len(agg) < _MIN_FIXTURE_POINTS:
                continue
            target_opcode = agg["target_opcode"].iloc[0]
            opcount = agg["opcount"].astype(float).to_numpy()
            for canonical, counts in _canonical_columns(
                agg,
                target_opcode,
                opcode_columns,
                spec.target_operation_count_source,
            ).items():
                keep, corr, ratio = _passes_thresholds(counts, opcount, eps)
                if not keep:
                    continue
                row: dict[str, object] = {
                    "test_name": spec.test_name,
                    "target_opcode": target_opcode,
                }
                for mb in model_by_cols:
                    row[mb] = group_values.get(mb)
                row["glue_opcode"] = canonical
                row["corr"] = corr
                row["ratio"] = ratio
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        cols = [
            "test_name",
            "target_opcode",
            *model_by_cols,
            "glue_opcode",
            "corr",
            "ratio",
        ]
        return pd.DataFrame(columns=cols)
    sort_cols = ["test_name", "target_opcode", *model_by_cols, "glue_opcode"]
    sort_cols = [c for c in sort_cols if c in df.columns]
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
