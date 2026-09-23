"""Per-client and across-client aggregation of fitted runtimes into gas params.

Expands each ``results.csv`` row into one row per ``model_params`` entry
(target_coef + extras), applies glue adjustment to the target row's runtime,
selects per-client and across-client worst-case values, and surfaces the
``poor_fit`` flag.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from evm_gasfit.config import Config


def _all_model_by_cols(config: Config) -> list[str]:
    return sorted({c for spec in config.resolved_models for c in spec.model_by})


def _lookup_glue_adjustment(
    glue_adjustment_df: pd.DataFrame | None,
    source_label: str,
    test_name: str,
    target_opcode: str,
    model_by: list[str],
    model_by_values: dict[str, object],
    client: str,
) -> tuple[float, float | None, float | None, float | None, bool, dict[str, object]]:
    """Return ``(adjustment, adjusted_runtime, adjusted_low, adjusted_high,
    interval_conditional, coverage)`` or zeros.

    ``coverage`` carries the per-row supporting-cost ledger
    (``glue_priced_opcodes`` / ``glue_bundled_opcodes`` /
    ``glue_unpriced_opcodes`` / ``glue_coverage_reason`` /
    ``glue_detection_status`` / ``glue_coverage_complete``) so recommendation
    consumers can see which supporters were subtracted, which were bundled
    into a priced partner, and which blocked isolation.
    """
    empty_coverage = {
        "glue_priced_opcodes": "",
        "glue_bundled_opcodes": "",
        "glue_unpriced_opcodes": "",
        "glue_coverage_reason": "",
        "glue_detection_status": "evaluated",
        "glue_coverage_complete": True,
    }
    if glue_adjustment_df is None or glue_adjustment_df.empty:
        return 0.0, None, None, None, False, empty_coverage
    mask = (
        (glue_adjustment_df["test_name"] == test_name)
        & (glue_adjustment_df["target_opcode"] == target_opcode)
        & (glue_adjustment_df["client_name"] == client)
    )
    # Match the producing spec so two specs sharing test/target/model_by but
    # differing in filter_by (read/write split) don't pick up each other's
    # adjusted coefficient.
    if "source_label" in glue_adjustment_df.columns:
        mask &= glue_adjustment_df["source_label"] == source_label
    for col in model_by:
        if col in glue_adjustment_df.columns:
            mask &= glue_adjustment_df[col] == model_by_values[col]
    sub = glue_adjustment_df[mask]
    if sub.empty:
        return 0.0, None, None, None, False, empty_coverage
    row = sub.iloc[0]
    conditional = (
        bool(row["glue_interval_conditional"])
        if "glue_interval_conditional" in glue_adjustment_df.columns
        and pd.notna(row.get("glue_interval_conditional"))
        else False
    )
    coverage = {
        "glue_priced_opcodes": str(row.get("glue_priced_opcodes") or ""),
        "glue_bundled_opcodes": str(row.get("glue_bundled_opcodes") or ""),
        "glue_unpriced_opcodes": str(row.get("glue_unpriced_opcodes") or ""),
        "glue_coverage_reason": str(row.get("glue_coverage_reason") or ""),
        "glue_detection_status": str(row.get("glue_detection_status") or "evaluated"),
        "glue_coverage_complete": bool(row.get("glue_coverage_complete", True)),
    }
    return (
        float(row["glue_adjustment"]),
        float(row["adjusted_target_coef_runtime_ms"]),
        float(row["adjusted_target_coef_conf_int_low"]),
        float(row["adjusted_target_coef_conf_int_high"]),
        conditional,
        coverage,
    )


def _qualification_lookup(
    config: Config, qualification_df: pd.DataFrame | None
) -> dict[tuple, tuple[str, str]]:
    """Map exact fit identities to slope and adjusted-estimate statuses."""
    if qualification_df is None or qualification_df.empty:
        return {}
    specs = {spec.source_label: spec for spec in config.resolved_models}
    out: dict[tuple, tuple[str, str]] = {}
    for _, row in qualification_df.iterrows():
        spec = specs.get(str(row["source_label"]))
        if spec is None:
            continue
        key = (
            str(row["source_label"]),
            str(row["test_name"]),
            str(row["target_opcode"]),
            *[str(row[column]) for column in sorted(spec.model_by)],
            str(row["client_name"]),
        )
        out[key] = (
            str(row.get("status", "")),
            str(row.get("adjusted_estimate_status", "")),
        )
    return out


def expand_to_per_client(
    results_df: pd.DataFrame,
    config: Config,
    glue_adjustment_df: pd.DataFrame | None,
    qualification_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand one results_df row into N rows per coefficient → gas-param map.

    Each ``results_df`` row is routed back to the exact spec that produced it
    via the ``source_label`` provenance column. This is a 1:1 join — two specs
    that share ``test_name`` + target + ``model_by`` (differing only in
    ``filter_by``) land on identical key columns but distinct ``source_label``
    values, so neither expands the other's fit. Coefficients come from both
    ``model_params`` (opcount interaction terms) and ``setup_params``
    (n-independent terms); ``feature_kind`` records which. Without a pricing
    anchor (``anchor_rate: null``) the ``new_gas_*`` columns are left empty —
    the run is comparison-only and never invents a conversion rate.
    """
    model_by_cols = _all_model_by_cols(config)
    anchor_rate = config.anchor_rate
    specs_by_label = {spec.source_label: spec for spec in config.resolved_models}
    qual = _qualification_lookup(config, qualification_df)

    rows: list[dict[str, object]] = []
    for _, res_row in results_df.iterrows():
        spec = specs_by_label[res_row["source_label"]]
        model_by_values = {c: res_row[c] for c in spec.model_by}
        target_opcode = res_row["target_opcode"]
        client = res_row["client_name"]
        qual_key = (
            str(spec.source_label),
            str(spec.test_name),
            str(target_opcode),
            *[str(model_by_values[c]) for c in sorted(spec.model_by)],
            str(client),
        )
        slope_status, adjusted_status = qual.get(
            qual_key, (str(res_row.get("qualification_status", "")), "")
        )
        (
            glue_adjustment,
            adj_runtime,
            adj_low,
            adj_high,
            interval_conditional,
            glue_coverage,
        ) = _lookup_glue_adjustment(
            glue_adjustment_df,
            spec.source_label,
            spec.test_name,
            target_opcode,
            spec.model_by,
            model_by_values,
            client,
        )

        coefficient_maps = [
            ("target", spec.model_params),
            ("setup", spec.setup_params),
        ]
        for feature_kind, params in coefficient_maps:
            for coef_name, gas_param in params.items():
                if feature_kind == "target":
                    if coef_name == "target_coef":
                        if adj_runtime is not None:
                            runtime_ms = adj_runtime
                            ci_low = adj_low
                            ci_high = adj_high
                        else:
                            runtime_ms = float(res_row["target_coef_runtime_ms"])
                            ci_low = float(res_row["target_coef_conf_int_low"])
                            ci_high = float(res_row["target_coef_conf_int_high"])
                        pvalue = float(res_row["target_coef_pvalue"])
                        row_glue_adjustment = float(glue_adjustment)
                        row_status = adjusted_status or slope_status
                        row_glue_coverage = glue_coverage
                    else:
                        rt_col = f"{coef_name}_runtime_ms"
                        if rt_col not in res_row.index:
                            continue
                        val = res_row[rt_col]
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            continue
                        runtime_ms = float(val)
                        pvalue = float(res_row[f"{coef_name}_pvalue"])
                        ci_low = float(res_row[f"{coef_name}_conf_int_low"])
                        ci_high = float(res_row[f"{coef_name}_conf_int_high"])
                        row_glue_adjustment = 0.0
                        row_status = slope_status
                        row_glue_coverage = {
                            "glue_priced_opcodes": "",
                            "glue_bundled_opcodes": "",
                            "glue_unpriced_opcodes": "",
                            "glue_coverage_reason": "",
                            "glue_detection_status": "evaluated",
                            "glue_coverage_complete": True,
                        }
                else:
                    rt_col = f"{coef_name}_runtime_ms"
                    if rt_col not in res_row.index:
                        continue
                    val = res_row[rt_col]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    runtime_ms = float(val)
                    pvalue = float(res_row[f"{coef_name}_pvalue"])
                    ci_low = float(res_row[f"{coef_name}_conf_int_low"])
                    ci_high = float(res_row[f"{coef_name}_conf_int_high"])
                    row_glue_adjustment = 0.0
                    row_status = slope_status
                    row_glue_coverage = {
                        "glue_priced_opcodes": "",
                        "glue_bundled_opcodes": "",
                        "glue_unpriced_opcodes": "",
                        "glue_coverage_reason": "",
                        "glue_detection_status": "evaluated",
                        "glue_coverage_complete": True,
                    }

                if anchor_rate is not None:
                    new_gas_decimal = float(anchor_rate) * runtime_ms / 1000.0
                    new_gas_rounded: int | None = math.ceil(new_gas_decimal)
                else:
                    new_gas_decimal = float("nan")
                    new_gas_rounded = None

                out: dict[str, object] = {
                    "gas_param": gas_param,
                    "client_name": client,
                    "runtime_ms": runtime_ms,
                    "pvalue": pvalue,
                    "conf_int_low": ci_low,
                    "conf_int_high": ci_high,
                    "test_name": spec.test_name,
                    "target_opcode": target_opcode,
                    "model_coef_name": coef_name,
                    "source_label": spec.source_label,
                    "feature_kind": feature_kind,
                    "glue_adjustment": row_glue_adjustment,
                    "glue_interval_conditional": interval_conditional,
                    "qualification_status": row_status,
                    "rsquared": float(res_row["rsquared"]),
                    "rsquared_adj": float(res_row["rsquared_adj"]),
                }
                out.update(row_glue_coverage)
                for col in model_by_cols:
                    if col in spec.model_by:
                        out[col] = model_by_values[col]
                    else:
                        out[col] = None
                out["new_gas_decimal"] = new_gas_decimal
                out["new_gas_rounded"] = new_gas_rounded
                out["poor_fit"] = False
                out["is_winner"] = False
                rows.append(out)

    cols = (
        [
            "gas_param",
            "client_name",
            "runtime_ms",
            "pvalue",
            "conf_int_low",
            "conf_int_high",
            "test_name",
            "target_opcode",
            "model_coef_name",
            "source_label",
            "feature_kind",
            "glue_adjustment",
            "glue_interval_conditional",
            "glue_coverage_complete",
            "glue_priced_opcodes",
            "glue_bundled_opcodes",
            "glue_unpriced_opcodes",
            "glue_coverage_reason",
            "glue_detection_status",
            "qualification_status",
            "rsquared",
            "rsquared_adj",
        ]
        + model_by_cols
        + ["new_gas_decimal", "new_gas_rounded", "poor_fit", "is_winner"]
    )
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    sort_cols = [
        "gas_param",
        "client_name",
        "test_name",
        "target_opcode",
        "model_coef_name",
        "source_label",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _model_by_combo(row: pd.Series, model_by_cols: list[str]) -> str:
    parts: list[str] = []
    for col in model_by_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        parts.append(str(val))
    return "_".join(parts) if parts else ""


def select_per_client_max(
    new_gas_all_df: pd.DataFrame,
    pvalue_threshold: float,
    rsquared_threshold: float,
    block_unqualified: bool = False,
) -> pd.DataFrame:
    """Select the winning row per ``(gas_param, client_name)``.

    Qualified pool requires ``pvalue < pvalue_threshold`` **and**
    ``rsquared >= rsquared_threshold``. On fallback (no candidate qualifies)
    the winner failed at least one of the two thresholds. When
    ``block_unqualified`` is set, the pool is further restricted to rows whose
    qualification status is ``qualified`` (falling back to the fit-quality
    pool, then to all candidates) — an unqualified winner still surfaces, but
    the proposal stage clears its recommended value.

    Mutates the input frame: flags ``poor_fit = True`` on **every** candidate
    that fails either threshold (not just the chosen winner) so losing weak
    fits stay visible on ``new_gas_all_params.csv``, and ``is_winner = True``
    on every chosen row so callers can route losing candidates downstream. A
    qualified winner keeps ``poor_fit = False``; a fallback winner necessarily
    fails a threshold and is therefore flagged too.
    """
    if new_gas_all_df.empty:
        return new_gas_all_df.copy()

    new_gas_all_df["poor_fit"] = ~(
        (new_gas_all_df["pvalue"] < pvalue_threshold)
        & (new_gas_all_df["rsquared"] >= rsquared_threshold)
    )

    model_by_cols = [
        c
        for c in new_gas_all_df.columns
        if c
        not in {
            "gas_param",
            "client_name",
            "runtime_ms",
            "pvalue",
            "conf_int_low",
            "conf_int_high",
            "test_name",
            "target_opcode",
            "model_coef_name",
            "source_label",
            "feature_kind",
            "glue_adjustment",
            "glue_interval_conditional",
            "qualification_status",
            "rsquared",
            "rsquared_adj",
            "new_gas_decimal",
            "new_gas_rounded",
            "poor_fit",
            "is_winner",
        }
    ]

    df = new_gas_all_df.copy()
    df["_combo"] = df.apply(lambda r: _model_by_combo(r, model_by_cols), axis=1)
    df["_idx"] = df.index

    chosen_indices: list[int] = []
    for (_gp, _client), group in df.groupby(["gas_param", "client_name"], sort=True):
        qualified = group[
            (group["pvalue"] < pvalue_threshold)
            & (group["rsquared"] >= rsquared_threshold)
        ]
        sub = qualified if not qualified.empty else group
        if block_unqualified and "qualification_status" in sub.columns:
            qual_pool = sub[sub["qualification_status"] == "qualified"]
            if not qual_pool.empty:
                sub = qual_pool
        sub = sub.sort_values(
            by=[
                "runtime_ms",
                "pvalue",
                "test_name",
                "target_opcode",
                "model_coef_name",
                "_combo",
                "source_label",
            ],
            ascending=[False, True, True, True, True, True, True],
            kind="mergesort",
        )
        winner_idx = int(sub.iloc[0]["_idx"])
        chosen_indices.append(winner_idx)
        new_gas_all_df.loc[winner_idx, "is_winner"] = True

    chosen = new_gas_all_df.loc[chosen_indices].copy()
    chosen = chosen.sort_values(
        ["gas_param", "client_name"], kind="mergesort"
    ).reset_index(drop=True)
    return chosen


def select_across_client_max(per_client_df: pd.DataFrame) -> pd.DataFrame:
    """For each gas_param, pick the row with the largest ``runtime_ms``."""
    if per_client_df.empty:
        cols = [
            "gas_param",
            "client_name",
            "runtime_ms",
            "conf_int_low",
            "conf_int_high",
            "selected_test",
            "selected_opcode",
            "selected_model_coef_name",
            "feature_kind",
            "glue_adjustment",
            "glue_interval_conditional",
            "qualification_status",
            "new_gas_decimal",
            "new_gas_rounded",
        ]
        return pd.DataFrame(columns=cols)

    reserved = {
        "gas_param",
        "client_name",
        "runtime_ms",
        "pvalue",
        "conf_int_low",
        "conf_int_high",
        "test_name",
        "target_opcode",
        "model_coef_name",
        "source_label",
        "feature_kind",
        "glue_adjustment",
        "glue_interval_conditional",
        "qualification_status",
        "rsquared",
        "rsquared_adj",
        "new_gas_decimal",
        "new_gas_rounded",
        "poor_fit",
        "is_winner",
    }
    model_by_cols = [c for c in per_client_df.columns if c not in reserved]

    chosen_rows: list[pd.Series] = []
    for _gp, group in per_client_df.groupby("gas_param", sort=True):
        sub = group.sort_values(
            by=["runtime_ms", "client_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        chosen_rows.append(sub.iloc[0])

    df = pd.DataFrame(chosen_rows).reset_index(drop=True)
    df = df.rename(
        columns={
            "test_name": "selected_test",
            "target_opcode": "selected_opcode",
            "model_coef_name": "selected_model_coef_name",
        }
    )
    out_cols = (
        [
            "gas_param",
            "client_name",
            "runtime_ms",
            "conf_int_low",
            "conf_int_high",
            "selected_test",
            "selected_opcode",
            "selected_model_coef_name",
            "feature_kind",
            "glue_adjustment",
            "glue_interval_conditional",
            "qualification_status",
        ]
        + model_by_cols
        + ["new_gas_decimal", "new_gas_rounded"]
    )
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols]
    return df.sort_values("gas_param", kind="mergesort").reset_index(drop=True)
