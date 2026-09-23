"""Assemble the final gas-cost proposal from fitted results and glue outputs."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evm_gasfit.config import Config
from evm_gasfit.glue import (
    compute_glue_adjustment,
    detect_missing_glue,
)

from .aggregate import (
    expand_to_per_client,
    select_across_client_max,
    select_per_client_max,
)
from .derived import evaluate

_log = logging.getLogger("evm_gasfit")

# Sentinel for rows that have no underlying fit — emitted when a name in
# ``proposed_by_model_params`` produced no successful regression, or when a
# derived formula resolves to ``None`` through propagation.
NO_FIT_LABEL = "<no-fit>"


@dataclass
class ProposalOutput:
    """Bundle the canonical proposal CSVs plus rendering context."""

    new_gas_all_df: pd.DataFrame
    new_gas_df: pd.DataFrame
    derived_rows: pd.DataFrame
    current_values: dict[str, int]
    warnings: list[str]
    missing_glue_pairs: list[tuple[str, str]]
    glue_opcodes_by_test_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Per-(client, glue_opcode) fit metrics from the glue estimator. Empty
    # when glue is disabled. Consumed by the report to surface glue opcodes
    # whose fits failed the modeling thresholds.
    glue_results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Every per-client candidate row from the expansion step, annotated with
    # ``is_winner`` (set by ``select_per_client_max``) and ``poor_fit``.
    # Consumed by the report to surface losing candidates with weak fits.
    candidates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # The (possibly status-downgraded) qualification table, echoed to
    # ``qualification.csv`` by the API layer.
    qualification_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _empty_glue_opcodes_by_test() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["test_name", "target_opcode", "glue_opcode", "corr", "ratio"]
    )


def build_proposal(
    config: Config,
    results_df: pd.DataFrame,
    glue_estimate_output=None,
    fixtures_df: pd.DataFrame | None = None,
    qualification_df: pd.DataFrame | None = None,
    planned: list | None = None,
) -> ProposalOutput:
    """Build the full proposal pipeline (aggregation + derived + diff baseline).

    ``qualification_df`` / ``planned`` come from the estimate stage; when
    supplied, per-candidate rows carry a ``qualification_status``, unqualified
    winners are blocked from recommended prices (``block_unqualified``), and
    glue-adjusted intervals that could not propagate supporting-cost
    uncertainty downgrade the adjusted estimate's status.
    """
    warnings_list: list[str] = list(config.warnings)
    missing_glue_pairs: list[tuple[str, str]] = []
    glue_opcodes_by_test_df = _empty_glue_opcodes_by_test()
    glue_adjustment_df: pd.DataFrame | None = None

    glue_enabled = config.glue_adjustment.enabled and glue_estimate_output is not None
    if glue_enabled and fixtures_df is not None:
        # Reuse the table the glue estimator already built; rebuilding here
        # would duplicate ~O(N·specs) work and risk drift between the table
        # the mixed-tier fits saw and the table the proposal subtracts from.
        glue_opcodes_by_test_df = glue_estimate_output.glue_opcodes_by_test_df
        # Fit objects let the adjustment propagate supporting-cost uncertainty
        # through paired grouped-bootstrap draws. Keys follow the results-row
        # identity the adjustment iterates: model_by in the spec's order.
        target_fits: dict[tuple, object] = {}
        if planned is not None:
            specs_by_label = {s.source_label: s for s in config.resolved_models}
            for record in planned:
                if record.fit is None:
                    continue
                # Key order follows the canonical sorted(group_values) —
                # the same convention the adjuster and qualification use.
                order = sorted(record.group_values)
                target_fits[
                    (
                        record.source_label,
                        record.test_name,
                        record.target_opcode,
                        *[
                            record.group_values[column]
                            for column in order
                            if pd.notna(record.group_values[column])
                        ],
                        record.client,
                    )
                ] = record.fit
        glue_adjustment_df = compute_glue_adjustment(
            results_df,
            glue_estimate_output.results_df,
            glue_opcodes_by_test_df,
            config.glue_adjustment.glue_contribution_p_value_threshold,
            config.glue_adjustment.glue_contribution_rsquared_threshold,
            target_fits=target_fits or None,
            glue_fits=glue_estimate_output.fits,
            confidence_level=config.qualification.confidence_level,
            random_seed=config.modeling.random_seed,
            detection_coverage_df=glue_estimate_output.detection_coverage_df,
            driver_support_df=glue_estimate_output.driver_support_df,
        )
        missing_glue_pairs = detect_missing_glue(
            fixtures_df,
            config.resolved_models,
            config.glue_adjustment.ratio_corr_eps,
            config.campaign.session_column,
        )
        for test_name, glue_opcode in missing_glue_pairs:
            msg = (
                f"missing-glue: test_name={test_name!r} correlates with non-priced "
                f"opcode {glue_opcode!r}; target coefficient left unadjusted"
            )
            _log.warning(msg)
            warnings_list.append(msg)

    # Downgrade adjusted-estimate statuses: point-shifted (conditional)
    # intervals, incomplete supporting-cost coverage, and adjusted intervals
    # that fail the propagated-interval gate. All three make the adjusted
    # estimate unusable as an isolated recommended price; none touches the
    # raw timing-model status.
    if (
        glue_adjustment_df is not None
        and qualification_df is not None
        and planned is not None
        and not glue_adjustment_df.empty
    ):
        from evm_gasfit.modeling.qualification import (
            adjusted_interval_reasons,
            apply_adjusted_statuses,
        )

        def _row_key(row: pd.Series) -> tuple | None:
            spec = next(
                (
                    candidate
                    for candidate in config.resolved_models
                    if candidate.source_label == row["source_label"]
                ),
                None,
            )
            if spec is None:
                return None
            model_by = [
                c
                for c in sorted(spec.model_by)
                if c in glue_adjustment_df.columns and pd.notna(row[c])
            ]
            return (
                str(row["source_label"]),
                str(row["test_name"]),
                str(row["target_opcode"]),
                *[str(row[c]) for c in model_by],
                str(row["client_name"]),
            )

        downgrades: dict[tuple, list[str]] = {}
        for _, row in glue_adjustment_df.iterrows():
            key = _row_key(row)
            if key is None:
                continue
            reasons: list[str] = []
            if bool(row.get("glue_interval_conditional", False)):
                reasons.append(
                    "adjusted interval conditional on point glue estimate "
                    "(uncertainty not propagated)"
                )
            unpriced = str(row.get("glue_unpriced_opcodes") or "")
            if unpriced:
                reasons.append(
                    f"detected supporting opcode(s) not isolable: {unpriced}"
                )
            coverage_reason = str(row.get("glue_coverage_reason") or "")
            coverage_value = row.get("glue_coverage_complete", True)
            coverage_incomplete = str(coverage_value).strip().lower() in {
                "false",
                "0",
            }
            if coverage_incomplete:
                detail = coverage_reason or "supporting-cost coverage is incomplete"
                reasons.append(f"glue coverage incomplete: {detail}")
            status = str(row.get("glue_detection_status") or "evaluated")
            if status != "evaluated":
                reasons.append(
                    f"support-cost detection could not run ({status})"
                )
            if reasons:
                downgrades.setdefault(key, []).extend(reasons)
        for key, reasons in adjusted_interval_reasons(
            config, glue_adjustment_df, planned
        ).items():
            downgrades.setdefault(key, []).extend(reasons)
        if downgrades:
            qualification_df = apply_adjusted_statuses(
                qualification_df, downgrades, planned
            )

    expanded_df = expand_to_per_client(
        results_df, config, glue_adjustment_df, qualification_df
    )
    per_client_df = select_per_client_max(
        expanded_df,
        config.modeling.poor_fit_p_value_threshold,
        config.modeling.poor_fit_rsquared_threshold,
        block_unqualified=config.qualification.block_unqualified,
    )
    # ``select_per_client_max`` mutates ``expanded_df`` in place, tagging
    # ``is_winner`` on each chosen row and ``poor_fit`` on every candidate that
    # failed a fit-quality threshold. ``candidates_df`` carries the full
    # expanded set so the report can surface losing candidates.
    candidates_df = expanded_df
    # ``new_gas_all_params.csv`` carries every per-client candidate fit (the
    # full spec × results-row × coef expansion), with ``is_winner`` marking the
    # row the per-client worst-case selector picked for each
    # ``(gas_param, client_name)``. It also publishes ``selected_*`` aliases so
    # provenance checks can match either naming. ``new_gas.csv`` selects the
    # across-client worst-case from the per-client winners only.
    new_gas_all_df = candidates_df.copy()
    new_gas_all_df["selected_test"] = new_gas_all_df["test_name"]
    new_gas_all_df["selected_opcode"] = new_gas_all_df["target_opcode"]
    new_gas_all_df["selected_model_coef_name"] = new_gas_all_df["model_coef_name"]
    new_gas_df = select_across_client_max(per_client_df)

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
            "selected_test",
            "selected_opcode",
            "selected_model_coef_name",
            "source_label",
            "feature_kind",
            "glue_adjustment",
            "glue_interval_conditional",
            "glue_priced_opcodes",
            "glue_bundled_opcodes",
            "glue_unpriced_opcodes",
            "glue_coverage_reason",
            "glue_detection_status",
            "glue_coverage_complete",
            "qualification_status",
            "rsquared",
            "rsquared_adj",
            "new_gas_decimal",
            "new_gas_rounded",
            "poor_fit",
            "is_winner",
        }
    ]
    declared_params = [
        gas_param
        for spec in config.resolved_models
        for gas_param in [*spec.model_params.values(), *spec.setup_params.values()]
    ]
    declared_params.extend(config.new_params)
    missing_params = [
        name
        for name in dict.fromkeys(declared_params)
        if name not in set(new_gas_all_df.get("gas_param", pd.Series(dtype=str)))
    ]
    for name in missing_params:
        new_gas_df = pd.concat(
            [
                new_gas_df,
                pd.DataFrame(
                    [_no_fit_summary_row(name, new_gas_df.columns, model_by_cols)]
                ),
            ],
            ignore_index=True,
        )
        new_gas_all_df = pd.concat(
            [
                new_gas_all_df,
                pd.DataFrame(
                    [_no_fit_all_row(name, new_gas_all_df.columns, model_by_cols)]
                ),
            ],
            ignore_index=True,
        )

    # Block unqualified models from becoming recommended prices. The numeric
    # estimate stays visible as a research observation (``runtime_ms`` is
    # untouched); only the recommended value is withheld, so a weak or
    # missing model can never silently price a parameter.
    if config.qualification.block_unqualified and "qualification_status" in (
        new_gas_df.columns
    ):
        status = new_gas_df["qualification_status"].astype(str)
        blocked = (status != "qualified") & new_gas_df["new_gas_rounded"].notna()
        for idx in new_gas_df.index[blocked]:
            gp = str(new_gas_df.loc[idx, "gas_param"])
            msg = (
                f"unqualified-param: {gp!r} blocked from recommended prices "
                f"(status={new_gas_df.loc[idx, 'qualification_status']!r}; "
                f"see qualification.csv)"
            )
            _log.warning(msg)
            warnings_list.append(msg)
        new_gas_df.loc[blocked, "new_gas_decimal"] = float("nan")
        new_gas_df.loc[blocked, "new_gas_rounded"] = pd.NA
        if "qualification_status" in new_gas_all_df.columns:
            all_status = new_gas_all_df["qualification_status"].astype(str)
            all_blocked = (all_status != "qualified") & new_gas_all_df[
                "new_gas_rounded"
            ].notna()
            new_gas_all_df.loc[all_blocked, "new_gas_decimal"] = float("nan")
            new_gas_all_df.loc[all_blocked, "new_gas_rounded"] = pd.NA

    # Derived parameters. Evaluated against the integer worst-case table; the
    # env carries ``None`` for unresolved names so derived formulas propagate.
    env: dict[str, int | float | None] = {}
    for _, row in new_gas_df.iterrows():
        gp = str(row["gas_param"])
        rounded = row["new_gas_rounded"]
        env[gp] = None if pd.isna(rounded) else int(rounded)

    derived_rows: list[dict[str, object]] = []
    for name, (_raw, tree) in config.derived_evaluated.items():
        value = evaluate(tree, env)
        rounded: int | None = None if value is None else math.ceil(value)
        env[name] = rounded

        derived_summary = _derived_summary_row(
            name, value, rounded, new_gas_df.columns, model_by_cols
        )
        new_gas_df = pd.concat(
            [new_gas_df, pd.DataFrame([derived_summary])], ignore_index=True
        )

        all_row = _derived_all_row(
            name, value, rounded, new_gas_all_df.columns, model_by_cols
        )
        new_gas_all_df = pd.concat(
            [new_gas_all_df, pd.DataFrame([all_row])], ignore_index=True
        )
        derived_rows.append(derived_summary)

    derived_rows_df = pd.DataFrame(derived_rows) if derived_rows else pd.DataFrame()

    # Patched fork values augmented with any integer new_params defaults are
    # the diff baseline rendered as ``current_gas`` in the proposal report.
    current_values: dict[str, int] = (
        dict(config.gas_costs_obj.values) if config.gas_costs_obj else {}
    )
    for name, value in config.new_params.items():
        if value is not None:
            current_values[name] = int(value)

    # Coerce ``new_gas_rounded`` to a nullable integer column so the empty
    # cells in placeholder rows survive CSV round-trips.
    new_gas_df["new_gas_rounded"] = new_gas_df["new_gas_rounded"].astype("Int64")
    new_gas_all_df["new_gas_rounded"] = new_gas_all_df["new_gas_rounded"].astype(
        "Int64"
    )

    # Final sort + reset for determinism. Gas params follow their first
    # appearance in the config (presets + custom models, then derived); any
    # name not declared in the config (shouldn't happen given the validators
    # but kept defensive) falls to the end in alphabetical order.
    order_index = _config_param_order_index(config)
    fallback = len(order_index)

    def _pos(series: pd.Series) -> pd.Series:
        return series.astype(str).map(lambda n: order_index.get(n, fallback))

    new_gas_all_df = (
        new_gas_all_df.assign(_pos=_pos(new_gas_all_df["gas_param"]))
        .sort_values(
            [
                "_pos",
                "gas_param",
                "client_name",
                "test_name",
                "target_opcode",
                "model_coef_name",
                "source_label",
            ],
            kind="mergesort",
        )
        .drop(columns="_pos")
        .reset_index(drop=True)
    )
    new_gas_df = (
        new_gas_df.assign(_pos=_pos(new_gas_df["gas_param"]))
        .sort_values(["_pos", "gas_param"], kind="mergesort")
        .drop(columns="_pos")
        .reset_index(drop=True)
    )
    if not candidates_df.empty:
        candidates_df = (
            candidates_df.assign(_pos=_pos(candidates_df["gas_param"]))
            .sort_values(
                [
                    "_pos",
                    "gas_param",
                    "client_name",
                    "test_name",
                    "target_opcode",
                    "model_coef_name",
                    "source_label",
                ],
                kind="mergesort",
            )
            .drop(columns="_pos")
            .reset_index(drop=True)
        )
    _ = np  # quiet linters; numpy imported for future use.

    # Null-baseline warning: any new_params entry with `null` baseline that
    # also lands in the heatmap will render as a blank row (no current gas to
    # ratio against). Flag it so users notice the lost coloring.
    plotted_params = set(
        new_gas_all_df.loc[
            new_gas_all_df["client_name"].astype(str).str.len() > 0, "gas_param"
        ].astype(str)
    )
    for name, value in config.new_params.items():
        if value is None and name in plotted_params:
            msg = (
                f"null-baseline: new_params[{name!r}] has no prior default; "
                f"its heatmap row will be blank (no current gas to ratio against)"
            )
            _log.warning(msg)
            warnings_list.append(msg)

    glue_results_df = (
        glue_estimate_output.results_df if glue_enabled else pd.DataFrame()
    )
    return ProposalOutput(
        new_gas_all_df=new_gas_all_df,
        new_gas_df=new_gas_df,
        derived_rows=derived_rows_df,
        current_values=current_values,
        warnings=warnings_list,
        missing_glue_pairs=missing_glue_pairs,
        glue_opcodes_by_test_df=glue_opcodes_by_test_df,
        glue_results_df=glue_results_df,
        candidates_df=candidates_df,
        qualification_df=(
            qualification_df if qualification_df is not None else pd.DataFrame()
        ),
    )


def _config_param_order_index(config: Config) -> dict[str, int]:
    """Map each declared gas-param name to its first-appearance position.

    Walks ``resolved_models`` in YAML declaration order (presets, then custom)
    and records each ``model_params`` RHS on first sight, then appends
    ``derived`` keys. The returned dict is consumed as a sort key so every
    proposal artifact — CSV, markdown tables, heatmap rows — surfaces gas
    params in the order the user declared them rather than alphabetically.
    """
    order: dict[str, int] = {}
    for spec in config.resolved_models:
        for gas_param in [*spec.model_params.values(), *spec.setup_params.values()]:
            if gas_param not in order:
                order[gas_param] = len(order)
    for name in config.derived_evaluated:
        if name not in order:
            order[name] = len(order)
    return order


def _no_fit_summary_row(
    name: str, columns, model_by_cols: list[str]
) -> dict[str, object]:
    row: dict[str, object] = {
        "gas_param": name,
        "client_name": "",
        "runtime_ms": float("nan"),
        "conf_int_low": float("nan"),
        "conf_int_high": float("nan"),
        "selected_test": NO_FIT_LABEL,
        "selected_opcode": NO_FIT_LABEL,
        "selected_model_coef_name": NO_FIT_LABEL,
        "feature_kind": "",
        "glue_adjustment": float("nan"),
        "glue_interval_conditional": False,
        "qualification_status": "inconclusive",
        "new_gas_decimal": float("nan"),
        "new_gas_rounded": pd.NA,
    }
    for col in model_by_cols:
        row[col] = None
    return {c: row.get(c) for c in columns}


def _no_fit_all_row(name: str, columns, model_by_cols: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "gas_param": name,
        "client_name": "",
        "runtime_ms": float("nan"),
        "pvalue": float("nan"),
        "conf_int_low": float("nan"),
        "conf_int_high": float("nan"),
        "test_name": NO_FIT_LABEL,
        "target_opcode": NO_FIT_LABEL,
        "model_coef_name": NO_FIT_LABEL,
        "source_label": NO_FIT_LABEL,
        "selected_test": NO_FIT_LABEL,
        "selected_opcode": NO_FIT_LABEL,
        "selected_model_coef_name": NO_FIT_LABEL,
        "feature_kind": "",
        "glue_adjustment": float("nan"),
        "glue_interval_conditional": False,
        "qualification_status": "inconclusive",
        "rsquared": float("nan"),
        "rsquared_adj": float("nan"),
        "new_gas_decimal": float("nan"),
        "new_gas_rounded": pd.NA,
        "poor_fit": False,
        "is_winner": False,
    }
    for col in model_by_cols:
        row[col] = None
    return {c: row.get(c) for c in columns}


def _derived_summary_row(
    name: str,
    value: float | None,
    rounded: int | None,
    columns,
    model_by_cols: list[str],
) -> dict[str, object]:
    label = NO_FIT_LABEL if value is None else "<derived>"
    row: dict[str, object] = {
        "gas_param": name,
        "client_name": "",
        "runtime_ms": float("nan"),
        "conf_int_low": float("nan"),
        "conf_int_high": float("nan"),
        "selected_test": label,
        "feature_kind": "derived",
        "glue_adjustment": float("nan") if value is None else 0.0,
        "glue_interval_conditional": False,
        "qualification_status": "qualified" if value is not None else "inconclusive",
        "new_gas_decimal": float("nan") if value is None else float(value),
        "new_gas_rounded": pd.NA if rounded is None else int(rounded),
    }
    for col in model_by_cols:
        row[col] = None
    return {c: row.get(c) for c in columns}


def _derived_all_row(
    name: str,
    value: float | None,
    rounded: int | None,
    columns,
    model_by_cols: list[str],
) -> dict[str, object]:
    label = NO_FIT_LABEL if value is None else "<derived>"
    row: dict[str, object] = {
        "gas_param": name,
        "client_name": "",
        "runtime_ms": float("nan"),
        "pvalue": float("nan"),
        "conf_int_low": float("nan"),
        "conf_int_high": float("nan"),
        "test_name": label,
        "target_opcode": label,
        "model_coef_name": label,
        "source_label": label,
        "selected_test": label,
        "selected_opcode": label,
        "selected_model_coef_name": label,
        "feature_kind": "derived",
        "glue_adjustment": float("nan") if value is None else 0.0,
        "glue_interval_conditional": False,
        "qualification_status": "qualified" if value is not None else "inconclusive",
        "rsquared": float("nan"),
        "rsquared_adj": float("nan"),
        "new_gas_decimal": float("nan") if value is None else float(value),
        "new_gas_rounded": pd.NA if rounded is None else int(rounded),
        "poor_fit": False,
        "is_winner": False,
    }
    for col in model_by_cols:
        row[col] = None
    return {c: row.get(c) for c in columns}
