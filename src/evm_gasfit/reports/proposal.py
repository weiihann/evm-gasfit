"""Final proposal markdown: ``new_gas_proposal.md``."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evm_gasfit.config import Config
from evm_gasfit.proposal.build import ProposalOutput

from .plots import (
    build_provenance_pivot,
    plot_proposal_heatmap,
    plot_proposal_provenance_heatmap,
    slug,
)

_PROVENANCE_NON_ID_COLS = frozenset(
    {
        "gas_param",
        "client_name",
        "runtime_ms",
        "pvalue",
        "conf_int_low",
        "conf_int_high",
        "source_label",
        "glue_adjustment",
        "rsquared",
        "rsquared_adj",
        "new_gas_decimal",
        "new_gas_rounded",
        "poor_fit",
        "is_winner",
    }
)


def _combo_counts_per_param(plottable: pd.DataFrame) -> dict[str, int]:
    """Number of distinct (test_name, target_opcode, model_coef_name, *model_by)
    tuples per gas_param across ``plottable``.

    Rows whose ``client_name`` is empty or whose ``new_gas_rounded`` is null
    are filtered upstream by the caller; this function assumes the slice it
    receives only carries plottable rows.
    """
    id_cols = [c for c in plottable.columns if c not in _PROVENANCE_NON_ID_COLS]
    if plottable.empty or not id_cols:
        return {}
    keys = (
        plottable[id_cols].astype(object).where(plottable[id_cols].notna(), "")
    ).agg(tuple, axis=1)
    df = pd.DataFrame({"gas_param": plottable["gas_param"].astype(str), "_key": keys})
    return df.groupby("gas_param")["_key"].nunique().to_dict()


SENTINEL = "no prior default"

# `missing-glue: test_name='X' correlates with non-priced opcode 'Y'; ...`
_MISSING_GLUE_RE = re.compile(
    r"^missing-glue:\s*test_name=['\"]([^'\"]+)['\"]\s*"
    r"correlates with non-priced opcode\s*['\"]([^'\"]+)['\"]"
)


def _format_anchor_rate_mgas_s(anchor_rate: float) -> str:
    """Render ``anchor_rate`` (gas/s) as a 3-sig-fig ``Mgas/s`` string."""
    mgas = float(anchor_rate) / 1e6
    if mgas == 0:
        return "0 Mgas/s"
    # 3 significant figures; strip trailing zeros after the decimal point so
    # round numbers stay readable (100 Mgas/s, not 100. Mgas/s).
    formatted = f"{mgas:.3g}"
    return f"{formatted} Mgas/s"


def _signed_diff(proposed: int, current: int) -> str:
    diff = proposed - current
    if diff == 0:
        return "0"
    return f"{diff:+d}"


def _signed_pct(proposed: int, current: int) -> str:
    if current == 0:
        return "n/a"
    pct = round((proposed - current) / current * 100)
    if pct == 0:
        return "0%"
    return f"{pct:+d}%"


def _fmt_ms(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{float(value):.{digits}g}"


def _comparison_table_md(proposal_output, config: Config) -> list[str]:
    """Markdown rows for the mandatory compute-vs-current-gas table."""
    from evm_gasfit.reports.comparison import build_comparison_df

    df = build_comparison_df(proposal_output, config)
    if df.empty:
        return ["_No parameters measured._"]
    lines = [
        "| Gas param | Client | Measured runtime (ms) | CI | Current gas | "
        "ms per gas | Relative costliness | Qualification |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        current = row["current_gas"]
        current_cell = "—" if current is None or pd.isna(current) else str(int(current))
        ci = (
            "—"
            if pd.isna(row["conf_int_low"]) or pd.isna(row["conf_int_high"])
            else f"[{_fmt_ms(row['conf_int_low'])}, {_fmt_ms(row['conf_int_high'])}]"
        )
        rel = row["relative_costliness"]
        rel_cell = "—" if rel is None or pd.isna(rel) else f"{float(rel):.2f}×"
        lines.append(
            f"| {row['gas_param']} | {row['client_name'] or '—'} | "
            f"{_fmt_ms(row['runtime_ms'])} | {ci} | {current_cell} | "
            f"{_fmt_ms(row['measured_ms_per_gas'])} | {rel_cell} | "
            f"{row['qualification_status'] or '—'} |"
        )
    return lines


def _qualification_table_md(qualification_df) -> list[str]:
    """Markdown rows summarizing every planned model's qualification status."""
    if qualification_df is None or qualification_df.empty:
        return ["_No planned models recorded._"]
    counts = qualification_df["status"].value_counts().to_dict()
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    lines = [
        f"**{summary}** — one row per planned model; see `qualification.csv` for the full gate evidence.",
        "",
    ]
    lines.append("| Test | Target | Client | Status | Adjusted estimate | Reasons |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in qualification_df.iterrows():
        reasons = str(row.get("reasons", "") or "")
        reasons_cell = reasons if reasons else "—"
        lines.append(
            f"| {row['test_name']} | {row['target_opcode'] or '—'} | "
            f"{row['client_name'] or '—'} | {row['status']} | "
            f"{row.get('adjusted_estimate_status', '')} | {reasons_cell} |"
        )
    return lines


def _scenarios_table_md(proposal_output, config: Config) -> list[str]:
    """Markdown rows for the explicit pricing-scenario section."""
    from evm_gasfit.reports.comparison import build_pricing_scenarios_df

    df = build_pricing_scenarios_df(proposal_output, config)
    lines: list[str] = []
    for scenario in config.pricing_scenarios:
        lines.append(f"### {scenario.name}")
        lines.append("")
        lines.append(
            f"anchor {scenario.anchor_rate:.3g} gas/s · margin "
            f"{scenario.margin_pct:.3g}% — price = ceil(anchor · (1 + margin) "
            "· runtime_ms / 1000)"
        )
        lines.append("")
        sub = df[df["scenario"] == scenario.name] if not df.empty else df
        if sub.empty:
            lines.append("_No measured runtimes to price._")
            lines.append("")
            continue
        lines.append(
            "| Gas param | Measured runtime (ms) | Scenario gas | Qualification |"
        )
        lines.append("| --- | --- | --- | --- |")
        for _, row in sub.iterrows():
            scenario_gas = (
                "withheld"
                if row["scenario_gas"] is None or pd.isna(row["scenario_gas"])
                else str(int(row["scenario_gas"]))
            )
            lines.append(
                f"| {row['gas_param']} | {_fmt_ms(row['runtime_ms'])} | "
                f"{scenario_gas} | {row['qualification_status'] or '—'} |"
            )
        lines.append("")
    return lines


def _direction_counts(
    new_gas_df, current_values: dict[str, int]
) -> tuple[int, int, int, int, int]:
    """Return (n_total, n_increased, n_decreased, n_new, n_unresolved)."""
    n_total = len(new_gas_df)
    inc = dec = new = unresolved = 0
    for _, row in new_gas_df.iterrows():
        gas_param = str(row["gas_param"])
        if pd.isna(row["new_gas_rounded"]):
            unresolved += 1
            continue
        proposed = int(row["new_gas_rounded"])
        if gas_param not in current_values:
            new += 1
            continue
        diff = proposed - int(current_values[gas_param])
        if diff > 0:
            inc += 1
        elif diff < 0:
            dec += 1
    return n_total, inc, dec, new, unresolved


def _partition_warnings(warnings: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split warnings into (missing-glue grouped by test_name) and (other)."""
    missing_by_test: dict[str, list[str]] = defaultdict(list)
    other: list[str] = []
    for w in warnings:
        m = _MISSING_GLUE_RE.match(w)
        if m:
            test_name, opcode = m.group(1), m.group(2)
            missing_by_test[test_name].append(opcode)
        else:
            other.append(w)
    return missing_by_test, other


def _build_partial_fit_rows(
    new_gas_all_df: pd.DataFrame,
    expected_clients: list[str],
) -> list[dict[str, object]]:
    """Per-param list of clients with no estimation, for params that fit on
    *some* clients.

    Fully-unresolved params (no client fit) are filtered out — they surface
    under the ``Missing parameters`` subsection. Derived/placeholder rows
    (empty ``client_name``) are likewise excluded. ``expected_clients`` is the
    universe a param is measured against: callers pass the configured clients
    that produced at least one fit somewhere, so clients with zero fits across
    *all* params don't repeat here (they get the dedicated no-fits callout).
    """
    df = new_gas_all_df[new_gas_all_df["client_name"].astype(str).str.len() > 0]
    df = df[df["new_gas_rounded"].notna()]
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    # ``new_gas_all_df`` arrives in config-declaration order; preserve that
    # by iterating unique values in first-appearance order rather than sorting.
    for gas_param in dict.fromkeys(df["gas_param"].astype(str)):
        fitting_clients = set(
            df[df["gas_param"] == gas_param]["client_name"].astype(str)
        )
        missing = [c for c in expected_clients if c not in fitting_clients]
        if missing:
            rows.append({"gas_param": gas_param, "missing_clients": missing})
    return rows


def _clients_with_no_fits(
    new_gas_all_df: pd.DataFrame,
    configured_clients: list[str],
) -> list[str]:
    """Configured clients that produced zero estimations across all params."""
    df = new_gas_all_df[new_gas_all_df["client_name"].astype(str).str.len() > 0]
    df = df[df["new_gas_rounded"].notna()]
    fitting = set(df["client_name"].astype(str)) if not df.empty else set()
    return [c for c in configured_clients if c not in fitting]


def _poor_fit_failure_label(
    pvalue: float, rsquared: float, pv_thresh: float, r2_thresh: float
) -> str:
    """Return ``p-value``, ``R²``, ``both`` — or empty if the row passes both
    thresholds. ``NaN`` on either metric counts as failing it (defensive: the
    winner selector never produces a NaN on these columns for fitted rows).
    """
    fails_p = pd.isna(pvalue) or float(pvalue) >= pv_thresh
    fails_r2 = pd.isna(rsquared) or float(rsquared) < r2_thresh
    if fails_p and fails_r2:
        return "both"
    if fails_p:
        return "p-value"
    if fails_r2:
        return "R²"
    return ""


def _weak_losing_candidates(
    candidates_df: pd.DataFrame,
    pv_thresh: float,
    r2_thresh: float,
) -> pd.DataFrame:
    """Return losing candidates (``is_winner == False``) that failed at least
    one fit-quality threshold. Pre-sorted upstream by ``_pos`` /
    ``gas_param`` / ``client_name`` / ``test_name`` / ``target_opcode`` /
    ``model_coef_name`` so callers can iterate without re-sorting.

    Multiple ``ModelSpec`` presets can map the same ``results.csv`` row to the
    same gas-param (e.g. several ``cold_account_*`` presets all writing
    ``ACCOUNT_WRITE``), so the expander emits one candidate per (spec × row).
    The dedupe below collapses those into a single entry per distinct fit so
    the report doesn't double-count what is really the same weak regression.
    """
    if candidates_df.empty or "is_winner" not in candidates_df.columns:
        return candidates_df.iloc[0:0]
    losers = candidates_df[candidates_df["is_winner"].eq(False)]
    weak_mask = (losers["pvalue"] >= pv_thresh) | (losers["rsquared"] < r2_thresh)
    weak = losers[weak_mask]
    dedupe_cols = [
        "gas_param",
        "client_name",
        "test_name",
        "target_opcode",
        "model_coef_name",
        "runtime_ms",
        "pvalue",
        "rsquared",
    ]
    dedupe_cols = [c for c in dedupe_cols if c in weak.columns]
    return weak.drop_duplicates(subset=dedupe_cols, keep="first")


def _render_poor_fit_table(
    rows_df: pd.DataFrame, pv_thresh: float, r2_thresh: float
) -> list[str]:
    """Markdown table rendering for a poor-fit subsection.

    Columns: ``Gas param | Client | Test | Target opcode | Coef | runtime_ms
    | pvalue | rsquared | Failed``. Numeric cells use 4-significant-figure
    formatting so very small p-values still read as e.g. ``3.21e-05`` and
    R² stays compact (``0.273``).
    """
    lines = [
        (
            "| Gas param | Client | Test | Target opcode | Coef | "
            "runtime_ms | pvalue | rsquared | Failed |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rows_df.iterrows():
        failed = _poor_fit_failure_label(
            row["pvalue"], row["rsquared"], pv_thresh, r2_thresh
        )
        lines.append(
            f"| `{row['gas_param']}` | `{row['client_name']}` | "
            f"`{row['test_name']}` | `{row['target_opcode']}` | "
            f"`{row['model_coef_name']}` | "
            f"{float(row['runtime_ms']):.4g} | "
            f"{float(row['pvalue']):.4g} | "
            f"{float(row['rsquared']):.4g} | {failed} |"
        )
    return lines


def _render_weak_losers_table(
    rows_df: pd.DataFrame, pv_thresh: float, r2_thresh: float
) -> list[str]:
    """Per-gas-param ``<details>`` blocks for ``### Other weak candidates``.

    One row per distinct combo (``test_name``, ``target_opcode``,
    ``model_coef_name``, plus any ``model_by`` factors that vary within the
    block); failing clients collapse into a single ``Failing clients`` cell
    with each client's failure label in parens. The ``Combo`` cell drops
    ``model_by`` factors that are constant across the block, matching the
    short-label convention used by the worst-case provenance section. Per-fit
    metrics (``runtime_ms``, ``pvalue``, ``rsquared``) are intentionally
    dropped; readers can cross-reference
    ``runtime_estimation_autogenerated_report.md`` when they need the numbers
    behind a particular fit.

    ``rows_df`` is expected to arrive pre-sorted by config gas_param order
    (the upstream ``candidates_df`` sort is preserved through
    ``_weak_losing_candidates``), so iterating gas_params in first-appearance
    order yields declaration order.
    """
    # ``test_name`` / ``target_opcode`` / ``model_coef_name`` already get their
    # own columns in the rendered table, so exclude them from the ``model_by``
    # axis even though they're not in ``_PROVENANCE_NON_ID_COLS``.
    fixed_id_cols = {"test_name", "target_opcode", "model_coef_name"}
    model_by_cols = [
        c
        for c in rows_df.columns
        if c not in _PROVENANCE_NON_ID_COLS and c not in fixed_id_cols
    ]

    seen_params: list[str] = []
    for gp in rows_df["gas_param"].astype(str):
        if gp not in seen_params:
            seen_params.append(gp)

    lines: list[str] = []
    for gas_param in seen_params:
        block = rows_df[rows_df["gas_param"].astype(str) == gas_param]
        # ``model_by`` cols whose values vary within this block. Treat NaN as
        # the empty string so (None, X) counts as two distinct values — a
        # combo where one spec didn't carry the field deserves its own row.
        varying = [
            c
            for c in model_by_cols
            if block[c].astype(object).where(block[c].notna(), "").nunique() > 1
        ]

        group_cols = ["test_name", "target_opcode", "model_coef_name", *varying]
        combo_to_clients: dict[tuple[str, ...], dict[str, str]] = {}
        for _, row in block.iterrows():
            key = tuple(str(row[c]) if pd.notna(row[c]) else "" for c in group_cols)
            label = _poor_fit_failure_label(
                row["pvalue"], row["rsquared"], pv_thresh, r2_thresh
            )
            combo_to_clients.setdefault(key, {})[str(row["client_name"])] = label

        sorted_keys = sorted(combo_to_clients.keys())
        n_combos = len(sorted_keys)
        lines.append("<details>")
        lines.append(
            f"<summary><code>{gas_param}</code> — {n_combos} weak "
            f"combo{'s' if n_combos != 1 else ''}</summary>"
        )
        lines.append("")
        lines.append("| Test | Target opcode | Coef | Combo | Failing clients |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in sorted_keys:
            test_name, target_opcode, coef = key[0], key[1], key[2]
            combo_vals = key[3:]
            if varying:
                parts = [f"`{c}={v}`" for c, v in zip(varying, combo_vals) if v]
                combo_cell = " / ".join(parts) if parts else "—"
            else:
                combo_cell = "—"
            clients = combo_to_clients[key]
            clients_cell = ", ".join(f"`{c}` ({clients[c]})" for c in sorted(clients))
            lines.append(
                f"| `{test_name}` | `{target_opcode}` | `{coef}` | "
                f"{combo_cell} | {clients_cell} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Drop the trailing blank — the caller appends its own section padding.
    if lines and lines[-1] == "":
        lines.pop()
    return lines


_GLUE_RATIO_NON_KEY_COLS = frozenset({"glue_opcode", "corr", "ratio"})


def _gas_params_per_glue_opcode(
    glue_opcodes_by_test_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Map each priced glue opcode → gas params whose target_coef depends on it.

    Joins the detector's ratio table against the per-client candidate pool on
    ``(test_name, target_opcode, *model_by)``. Only ``target_coef`` candidates
    are considered, since glue adjustment is only applied to target
    coefficients (see ``proposal.aggregate.expand_to_per_client``). NaN cells
    on either side compare equal here — both frames carry ``None`` for
    model_by columns outside a spec's own ``model_by``, and pandas' default
    merge skips those rows.
    """
    if glue_opcodes_by_test_df.empty or candidates_df.empty:
        return {}

    cand = candidates_df[candidates_df["client_name"].astype(str).str.len() > 0]
    cand = cand[cand["model_coef_name"] == "target_coef"]
    if cand.empty:
        return {}

    model_by_cols = [
        c
        for c in glue_opcodes_by_test_df.columns
        if c not in _GLUE_RATIO_NON_KEY_COLS
        and c not in {"test_name", "target_opcode"}
        and c in cand.columns
    ]

    out: dict[str, set[str]] = {}
    for _, glue_row in glue_opcodes_by_test_df.iterrows():
        mask = (cand["test_name"] == glue_row["test_name"]) & (
            cand["target_opcode"] == glue_row["target_opcode"]
        )
        for col in model_by_cols:
            val = glue_row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                mask &= cand[col].isna()
            else:
                mask &= cand[col] == val
        if not mask.any():
            continue
        glue_opcode = str(glue_row["glue_opcode"])
        out.setdefault(glue_opcode, set()).update(
            str(p) for p in cand.loc[mask, "gas_param"]
        )

    return {k: sorted(v) for k, v in out.items()}


def _poor_fit_glue_rows(
    glue_results_df: pd.DataFrame,
    glue_opcodes_by_test_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    pv_thresh: float,
    r2_thresh: float,
) -> list[dict[str, object]]:
    """Per-glue-opcode rows for the poor-fit glue table.

    A glue opcode surfaces here when at least one client's fit fails either
    threshold (``p_value >= pv_thresh`` or ``rsquared < r2_thresh``). NaN on
    either metric counts as failing it — defensive only, the estimator emits
    NaN only when the fit was skipped entirely (in which case the row is
    excluded from ``glue_results_df`` anyway).
    """
    if glue_results_df.empty:
        return []
    pval = glue_results_df["p_value"]
    r2 = glue_results_df["rsquared"]
    weak_mask = pval.isna() | (pval >= pv_thresh) | r2.isna() | (r2 < r2_thresh)
    # Skipped fits land in glue_results_df with nobs == 0 and NaN metrics —
    # those rows do not represent a "poor fit", they represent a missing fit,
    # so exclude them from the warning.
    fitted_mask = glue_results_df["nobs"].astype("Int64") > 0
    weak = glue_results_df[weak_mask & fitted_mask]
    if weak.empty:
        return []

    gas_params_by_glue = _gas_params_per_glue_opcode(
        glue_opcodes_by_test_df, candidates_df
    )

    rows: list[dict[str, object]] = []
    for glue_opcode, group in weak.groupby("glue_opcode", sort=False):
        client_labels: list[str] = []
        for _, r in group.sort_values("client_name", kind="mergesort").iterrows():
            label = _poor_fit_failure_label(
                r["p_value"], r["rsquared"], pv_thresh, r2_thresh
            )
            client_labels.append(f"`{r['client_name']}` ({label})")
        rows.append(
            {
                "glue_opcode": str(glue_opcode),
                "clients": client_labels,
                "gas_params": gas_params_by_glue.get(str(glue_opcode), []),
            }
        )
    rows.sort(key=lambda r: r["glue_opcode"])
    return rows


def _build_client_comparison_rows(
    new_gas_all_df: pd.DataFrame,
    fitted_params: list[str],
) -> list[dict[str, object]]:
    """Worst vs. second-worst client per gas param, fitted rows only.

    Skips gas params with fewer than 2 fitted clients (nothing to compare).
    """
    df = new_gas_all_df[new_gas_all_df["client_name"].astype(str).str.len() > 0]
    df = df[df["new_gas_rounded"].notna()]
    rows: list[dict[str, object]] = []
    for gas_param in fitted_params:
        sub = df[df["gas_param"] == gas_param]
        if len(sub) < 2:
            continue
        ordered = sub.sort_values(
            by=["new_gas_rounded", "client_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        worst = ordered.iloc[0]
        second = ordered.iloc[1]
        worst_val = int(worst["new_gas_rounded"])
        second_val = int(second["new_gas_rounded"])
        ratio_cell = "n/a" if second_val == 0 else f"{worst_val / second_val:.2f}×"
        rows.append(
            {
                "gas_param": gas_param,
                "worst_client": str(worst["client_name"]),
                "worst_value": worst_val,
                "second_client": str(second["client_name"]),
                "second_value": second_val,
                "ratio": ratio_cell,
            }
        )
    return rows


def _cell(value: float | None) -> str:
    """Render a pivot cell as either an integer or blank for NaN/None."""
    if value is None:
        return ""
    if isinstance(value, float) and not pd.notna(value):
        return ""
    return f"{int(value)}"


def _render_overview_table(
    plottable: pd.DataFrame,
) -> list[str]:
    """Markdown table rendering of the overview heatmap: rows are gas
    parameters in first-appearance order, columns are clients alphabetically.

    Cells carry ``new_gas_rounded`` integers; missing (param × client) pairs
    render blank. No color encoding — this is the plots-off fallback.
    """
    pivot = plottable.assign(
        new_gas_rounded=plottable["new_gas_rounded"].astype("Float64").astype(float)
    ).pivot_table(
        index="gas_param",
        columns="client_name",
        values="new_gas_rounded",
        aggfunc="max",
    )
    row_order = list(dict.fromkeys(plottable["gas_param"].astype(str)))
    pivot = pivot.reindex([p for p in row_order if p in pivot.index])
    clients = sorted(str(c) for c in pivot.columns)

    header = ["Gas param", *clients]
    rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for gas_param, row in pivot.iterrows():
        cells = [str(gas_param), *(_cell(row[c]) for c in clients)]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _render_provenance_table(
    slice_df: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, str]] | None]:
    """Markdown table rendering of one per-param provenance heatmap.

    Returns ``(table_lines, legend)``. Rows are model combos (labeled the
    same way the heatmap labels them, including the ``M1, M2, …`` collapse
    when labels get too long); columns are clients alphabetically; cells are
    ``new_gas_rounded`` integers, blank for missing combo/client pairs. The
    winning combo per client is rendered in **bold** so the chosen provenance
    is readable without diffing against the proposal table.
    """
    pivot, legend, winners = build_provenance_pivot(slice_df)
    clients = sorted(str(c) for c in pivot.columns)
    header = ["Combo", *clients]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for combo_label, row in pivot.iterrows():
        cells = [f"`{combo_label}`"]
        for c in clients:
            text = _cell(row[c])
            if text and bool(winners.at[combo_label, c]):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines, legend


def _append_provenance_section(
    lines: list[str],
    plottable: pd.DataFrame,
    qualifying: list[str],
    skipped: list[str],
    combo_counts: dict[str, int],
    current_values: dict[str, int],
    out_dir: Path,
    *,
    plots_enabled: bool,
) -> None:
    """Append `## Worst-case provenance per gas param` with one `<details>`
    block per multi-combo gas param.

    Callers compute ``qualifying`` (params with ≥ 2 combos), ``skipped``
    (single-combo params), and ``combo_counts`` once so the TOC can decide
    whether to link to this section. When ``plots_enabled`` is ``True``, each
    block embeds the per-param heatmap PNG; otherwise it embeds a markdown
    table with the same labels.
    """
    if not qualifying:
        return

    lines.append("## Worst-case provenance per gas param")
    lines.append("")
    base_intro = (
        "One collapsible block per gas parameter showing every per-client "
        "candidate that the worst-case selector saw. Rows are model combos "
        "(the source regression's `test_name`, `target_opcode`, "
        "`model_coef_name`, and any `model_by` factors — components constant "
        "within a parameter are dropped from the label). Cells carry each "
        "candidate's proposed gas; the cell the per-client selector picked is"
    )
    if plots_enabled:
        lines.append(
            base_intro + " outlined in black. Colors are `log2(proposed / current)` "
            "against that parameter's baseline on a per-parameter symmetric "
            "scale."
        )
    else:
        lines.append(base_intro + " bolded.")
    lines.append("")
    if skipped:
        skipped_cell = ", ".join(f"`{p}`" for p in skipped)
        lines.append(
            f"_Single-combo parameters omitted (see proposal table for the "
            f"sole estimation): {skipped_cell}._"
        )
        lines.append("")

    for gas_param in qualifying:
        slice_df = plottable[plottable["gas_param"].astype(str) == gas_param]
        current = current_values.get(gas_param)
        n_combos = combo_counts[gas_param]
        n_clients = slice_df["client_name"].astype(str).nunique()
        summary = (
            f"<summary><code>{gas_param}</code> — {n_combos} combos × "
            f"{n_clients} client{'s' if n_clients != 1 else ''}</summary>"
        )
        lines.append("<details>")
        lines.append(summary)
        lines.append("")
        if plots_enabled:
            _, legend = plot_proposal_provenance_heatmap(
                gas_param,
                slice_df,
                current_value=current,
                out_dir=out_dir,
            )
            if legend:
                lines.append("| Label | Combo |")
                lines.append("| --- | --- |")
                for short, full in legend:
                    lines.append(f"| `{short}` | {full} |")
                lines.append("")
            lines.append(f"![](figs/proposal/provenance__{slug(gas_param)}.png)")
        else:
            table_lines, legend = _render_provenance_table(slice_df)
            if legend:
                lines.append("| Label | Combo |")
                lines.append("| --- | --- |")
                for short, full in legend:
                    lines.append(f"| `{short}` | {full} |")
                lines.append("")
            lines.extend(table_lines)
        lines.append("")
        lines.append("</details>")
        lines.append("")


def write_proposal_report(
    out_dir: Path,
    proposal_output: ProposalOutput,
    config: Config,
) -> None:
    """Write the final proposal markdown under ``out_dir``."""
    out_path = out_dir / "new_gas_proposal.md"
    new_gas_df = proposal_output.new_gas_df
    current_values = proposal_output.current_values

    # ``new_gas_all_params.csv`` now carries every per-client candidate fit,
    # with the per-client worst-case pick flagged ``is_winner``. The sections
    # that speak about the *selected* per-client value — poor-fit winners, the
    # overview heatmap, client comparison, and coverage — read the winners
    # only. Placeholder/derived rows (empty ``client_name``) carry
    # ``is_winner = False`` but must survive for the missing-parameter and
    # coverage logic, so keep them too. (Losing candidates are surfaced
    # separately from ``candidates_df``.)
    full_all_df = proposal_output.new_gas_all_df
    if "is_winner" in full_all_df.columns:
        winners_all_df = full_all_df[
            (full_all_df["is_winner"].eq(True))
            | (full_all_df["client_name"].astype(str).str.len() == 0)
        ]
    else:
        winners_all_df = full_all_df

    n_total, n_inc, n_dec, n_new, n_unresolved = _direction_counts(
        new_gas_df, current_values
    )
    missing_by_test, other_warnings = _partition_warnings(proposal_output.warnings)
    n_warn = sum(len(v) for v in missing_by_test.values()) + len(other_warnings)
    poor_fit_rows = (
        winners_all_df[winners_all_df["poor_fit"].eq(True)]
        if "poor_fit" in winners_all_df.columns
        else winners_all_df.iloc[0:0]
    )

    lines: list[str] = ["# New gas proposal", ""]

    # Run metadata.
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    if config.anchor_rate is None:
        anchor_label = "none (comparison-only run)"
    else:
        anchor_label = _format_anchor_rate_mgas_s(config.anchor_rate)
    lines.append(
        f"_Generated {generated} · fork `{config.gas_costs.fork}` · "
        f"anchor_rate {anchor_label}_"
    )
    lines.append("")

    has_anchor = config.anchor_rate is not None
    # Summary line.
    if has_anchor:
        lines.append(
            f"**Summary:** {n_total} parameters proposed — "
            f"{n_inc} increased, {n_dec} decreased, {n_new} new, "
            f"{n_unresolved} unresolved · "
            f"{n_warn} warning{'s' if n_warn != 1 else ''} · "
            f"{len(poor_fit_rows)} poor-fit selection{'s' if len(poor_fit_rows) != 1 else ''}"
        )
    else:
        lines.append(
            f"**Summary:** {n_total} parameters measured — comparison-only "
            f"run (no pricing anchor; proposed-gas columns are empty) · "
            f"{n_warn} warning{'s' if n_warn != 1 else ''} · "
            f"{len(poor_fit_rows)} poor-fit selection{'s' if len(poor_fit_rows) != 1 else ''}"
        )
    lines.append("")

    # Diff table — fitted rows only. Without an anchor nothing is "fitted"
    # in gas terms; the compute-vs-current section carries the run instead.
    fitted_df = new_gas_df[~new_gas_df["new_gas_rounded"].isna()]
    if has_anchor:
        unresolved_df = new_gas_df[new_gas_df["new_gas_rounded"].isna()]
    else:
        unresolved_df = new_gas_df.iloc[0:0]

    # Decide which optional sections will render so the TOC links match.
    plots_enabled = config.output.plots
    new_gas_all_df = winners_all_df
    fitted_params = [str(p) for p in fitted_df["gas_param"]]
    plottable = new_gas_all_df[new_gas_all_df["client_name"].astype(str).str.len() > 0]
    plottable = plottable[plottable["new_gas_rounded"].notna()]
    # Provenance is sourced from the full per-client candidate set (every spec
    # × row × coef expansion, not just the per-client winners), so each
    # (combo, client) cell carries its proposed gas — winners get marked,
    # losers stay visible for comparison.
    candidates_df = proposal_output.candidates_df
    plottable_candidates = candidates_df[
        candidates_df["client_name"].astype(str).str.len() > 0
    ]
    plottable_candidates = plottable_candidates[
        plottable_candidates["new_gas_rounded"].notna()
    ]
    combo_counts = _combo_counts_per_param(plottable_candidates)
    qualifying = [p for p in fitted_params if combo_counts.get(p, 0) >= 2]
    skipped = [p for p in fitted_params if 0 < combo_counts.get(p, 0) < 2]
    has_provenance_section = bool(qualifying) and not plottable_candidates.empty
    has_scenarios = bool(config.pricing_scenarios)

    # TOC.
    lines.append("## Contents")
    lines.append("")
    if has_anchor:
        lines.append("- [Proposed parameters](#proposed-gas-parameters)")
    lines.append("- [Compute vs current gas](#compute-vs-current-gas)")
    lines.append("- [Client comparison](#client-comparison)")
    if has_provenance_section:
        lines.append("- [Worst-case provenance](#worst-case-provenance-per-gas-param)")
    lines.append("- [Qualification](#qualification)")
    if has_scenarios:
        lines.append("- [Pricing scenarios](#pricing-scenarios)")
    lines.append("- [Warnings](#warnings)")
    lines.append("- [Poor-fit selections](#poor-fit-selections)")
    lines.append("")

    if has_anchor:
        lines.append("## Proposed gas parameters")
        lines.append("")
        lines.append("| Gas param | Current gas | Proposed gas | Diff | Diff % |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in fitted_df.iterrows():
            gas_param = str(row["gas_param"])
            proposed = int(row["new_gas_rounded"])
            if gas_param in current_values:
                current_int = int(current_values[gas_param])
                current_cell = str(current_int)
                diff_cell = _signed_diff(proposed, current_int)
                diff_pct_cell = _signed_pct(proposed, current_int)
            else:
                current_cell = SENTINEL
                diff_cell = "n/a"
                diff_pct_cell = "n/a"
            lines.append(
                f"| {gas_param} | {current_cell} | {proposed} | {diff_cell} | "
                f"{diff_pct_cell} |"
            )
        lines.append("")

    # Mandatory compute-vs-current-gas comparison — with or without an
    # anchor. Without one this is the run's primary numeric table: measured
    # cost against the active schedule, never a price.
    lines.append("## Compute vs current gas")
    lines.append("")
    if has_anchor:
        lines.append(
            "Measured worst-case runtime per parameter against the active "
            "schedule. `Relative costliness` normalizes each param's "
            "`runtime_ms / current_gas` by the median across params carrying "
            "a current value — a unitless descriptive index, not a price."
        )
    else:
        lines.append(
            "Comparison-only run: no pricing anchor was supplied, so no "
            "proposed gas is computed. The table reports measured worst-case "
            "runtime per parameter against the active schedule. `Relative "
            "costliness` normalizes each param's `runtime_ms / current_gas` "
            "by the median across params carrying a current value — a "
            "unitless descriptive index, not a price."
        )
    lines.append("")
    comparison_md = _comparison_table_md(proposal_output, config)
    lines.extend(comparison_md)
    lines.append("")

    # Client comparison: worst vs. second-worst per gas param, plus heatmap.
    comparison_rows = _build_client_comparison_rows(new_gas_all_df, fitted_params)

    lines.append("## Client comparison")
    lines.append("")
    if not comparison_rows:
        lines.append(
            "_Not enough clients to compare — every gas parameter was fitted "
            "by a single client._"
        )
        lines.append("")
    else:
        lines.append(
            "Worst client vs. second-worst client per gas parameter. The "
            "`Ratio` column is `worst gas / second-worst gas` — values close "
            "to 1× mean the worst case sits next to the rest of the field, "
            "while large ratios flag the worst client as an outlier."
        )
        lines.append("")
        lines.append(
            "| Gas param | Worst client | Worst gas | Second-worst client | "
            "Second-worst gas | Ratio |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in comparison_rows:
            lines.append(
                f"| {row['gas_param']} | {row['worst_client']} | "
                f"{row['worst_value']} | {row['second_client']} | "
                f"{row['second_value']} | {row['ratio']} |"
            )
        lines.append("")

    if not plottable.empty:
        if plots_enabled:
            plot_proposal_heatmap(
                plottable, current_values=current_values, out_dir=out_dir
            )
            lines.append(
                "Per-client proposed gas for each parameter. Cells are colored "
                "by `log2(proposed / current)` — red means the proposal is more "
                "expensive than the current gas cost, green means cheaper, and "
                "white sits at unchanged. Annotations show the absolute "
                "proposed gas value; blank rows are parameters with no prior "
                "baseline (see warnings below)."
            )
            lines.append("")
            lines.append("![](figs/proposal/heatmap.png)")
            lines.append("")
        else:
            lines.append(
                "Per-client proposed gas for each parameter. Blank cells mark "
                "(parameter, client) pairs with no fit."
            )
            lines.append("")
            lines.extend(_render_overview_table(plottable))
            lines.append("")

    _append_provenance_section(
        lines,
        plottable_candidates,
        qualifying,
        skipped,
        combo_counts,
        current_values,
        out_dir,
        plots_enabled=plots_enabled,
    )

    # Qualification status of every planned model — always rendered.
    lines.append("## Qualification")
    lines.append("")
    q = config.qualification
    lines.append(
        f"Policy — confidence level {q.confidence_level:.2f}; always-on gates: "
        f"condition number ≤ {q.max_condition_number:.3g}, residual curvature "
        f"R² ≤ {q.max_residual_curvature_r2:.3g}; armed gates: "
        + (
            f"relative CI width ≤ {q.max_relative_uncertainty:.3g}"
            if q.max_relative_uncertainty is not None
            else "relative CI width (unset)"
        )
        + ", "
        + (
            f"held-out error ≤ {q.max_holdout_error:.3g}"
            if q.max_holdout_error is not None
            else "held-out error (unset)"
        )
        + ", "
        + (
            f"min sessions {q.min_sessions}"
            if q.min_sessions is not None
            else "min sessions (unset)"
        )
        + ("; poor-fit thresholds enforced" if q.enforce_fit_quality else "")
        + (
            "; unqualified models blocked from recommended prices"
            if q.block_unqualified
            else ""
        )
        + "."
    )
    lines.append("")
    qual_lines = _qualification_table_md(proposal_output.qualification_df)
    lines.extend(qual_lines)
    lines.append("")

    # Optional explicit pricing scenarios — research outputs only.
    if has_scenarios:
        lines.append("## Pricing scenarios")
        lines.append("")
        lines.append(
            "Each scenario applies its own explicit gas/second anchor and "
            "margin policy. These are research outputs; nothing here alters "
            "production constants, and no anchor is invented when none is "
            "supplied."
        )
        lines.append("")
        lines.extend(_scenarios_table_md(proposal_output, config))
        lines.append("")

    # Warnings (with Missing parameters as a subsection — always shown).
    lines.append("## Warnings")
    lines.append("")
    lines.append("### Missing parameters")
    lines.append("")
    if unresolved_df.empty:
        lines.append("_None._")
        lines.append("")
    else:
        lines.append(
            "These names were proposed by a `model_params` RHS or `derived` "
            "entry but produced no value — either every candidate fit was "
            "skipped (constant opcount, insufficient observations, solver "
            "failure) or a referenced upstream value was itself unresolved. "
            "Inspect the `evm_gasfit` warnings in `meta.json` for the cause."
        )
        lines.append("")
        lines.append("| Gas param |")
        lines.append("| --- |")
        for _, row in unresolved_df.iterrows():
            lines.append(f"| `{row['gas_param']}` |")
        lines.append("")

    # Partial fits: gas params with at least one client fit but missing on
    # others — the proposed value still stands but was selected from a
    # smaller pool. Clients that produced zero estimations across *all* params
    # are reported only in the dedicated callout below, so the per-param table
    # considers just the clients that fit somewhere.
    configured_clients = list(config.clients)
    no_fit_clients = _clients_with_no_fits(winners_all_df, configured_clients)
    partial_universe = [c for c in configured_clients if c not in no_fit_clients]
    partial_rows = _build_partial_fit_rows(winners_all_df, partial_universe)
    lines.append("### Incomplete client coverage")
    lines.append("")
    if not partial_rows and not no_fit_clients:
        lines.append("_None._")
        lines.append("")
    else:
        if no_fit_clients:
            names = ", ".join(f"`{c}`" for c in no_fit_clients)
            lines.append(
                f"**Clients with no estimations at all:** {names}. These "
                "configured clients produced no fits for any gas parameter — "
                "check that the runtimes CSV contains their rows and that the "
                "fixture-name conventions match. Inspect the `evm_gasfit` "
                "warnings in `meta.json` for the cause."
            )
            lines.append("")
        if partial_rows:
            lines.append(
                "These gas parameters were fit by at least one client but not "
                "by every configured client — the listed clients produced no "
                "estimation, so the worst-case value was selected from a "
                "smaller pool. Inspect the `evm_gasfit` warnings in "
                "`meta.json` for the cause."
            )
            lines.append("")
            lines.append("| Gas param | Missing clients |")
            lines.append("| --- | --- |")
            for row in partial_rows:
                missing_cell = ", ".join(f"`{c}`" for c in row["missing_clients"])
                lines.append(f"| `{row['gas_param']}` | {missing_cell} |")
            lines.append("")
    glue_pv_thresh = config.glue_adjustment.glue_contribution_p_value_threshold
    glue_r2_thresh = config.glue_adjustment.glue_contribution_rsquared_threshold
    poor_glue_rows = _poor_fit_glue_rows(
        proposal_output.glue_results_df,
        proposal_output.glue_opcodes_by_test_df,
        proposal_output.candidates_df,
        glue_pv_thresh,
        glue_r2_thresh,
    )
    if missing_by_test or poor_glue_rows or other_warnings:
        if missing_by_test or poor_glue_rows:
            lines.append("### Missing glue adjustments")
            lines.append("")
            if missing_by_test:
                n_tests = len(missing_by_test)
                lines.append("<details>")
                lines.append(
                    f"<summary><b>Non-priced opcodes correlated with the "
                    f"target opcount</b> — {n_tests} test"
                    f"{'s' if n_tests != 1 else ''} affected</summary>"
                )
                lines.append("")
                lines.append(
                    "The target coefficient was left unadjusted for these. "
                    "Consider adding them to the glue model or re-designing the "
                    "test to isolate the target opcode."
                )
                lines.append("")
                lines.append("| Test name | Non-priced opcodes |")
                lines.append("| --- | --- |")
                for test_name in sorted(missing_by_test):
                    opcodes = sorted(set(missing_by_test[test_name]))
                    opcode_cell = ", ".join(f"`{op}`" for op in opcodes)
                    lines.append(f"| `{test_name}` | {opcode_cell} |")
                lines.append("")
                lines.append("</details>")
                lines.append("")
            if poor_glue_rows:
                n_skipped = sum(len(r["clients"]) for r in poor_glue_rows)
                lines.append("<details>")
                lines.append(
                    f"<summary><b>Priced glue opcodes with a poor fit</b> — "
                    f"{n_skipped} (glue_opcode, client) "
                    f"fit{'s' if n_skipped != 1 else ''} skipped</summary>"
                )
                lines.append("")
                lines.append(
                    f"`p_value >= glue_contribution_p_value_threshold` "
                    f"({glue_pv_thresh:g}) or "
                    f"`rsquared < glue_contribution_rsquared_threshold` "
                    f"({glue_r2_thresh:g}) — the contribution of these "
                    f"(glue_opcode, client) fits was **skipped** when computing "
                    f"the glue adjustment, so the listed gas params carry a "
                    f"target coefficient that is not net of this glue opcode's "
                    f"runtime on the affected clients. See "
                    f"`glue_opcodes_autogenerated_report.md` for per-fit metrics."
                )
                lines.append("")
                lines.append("| Glue opcode | Affected clients | Affected gas params |")
                lines.append("| --- | --- | --- |")
                for row in poor_glue_rows:
                    gas_params = row["gas_params"]
                    gp_cell = (
                        ", ".join(f"`{p}`" for p in gas_params) if gas_params else "—"
                    )
                    lines.append(
                        f"| `{row['glue_opcode']}` | "
                        f"{', '.join(row['clients'])} | {gp_cell} |"
                    )
                lines.append("")
                lines.append("</details>")
                lines.append("")
        if other_warnings:
            lines.append("### Other")
            lines.append("")
            for w in other_warnings:
                lines.append(f"- {w}")
            lines.append("")

    # Poor-fit selections (target-coef side) uses the modeling thresholds —
    # distinct from the glue gating thresholds above.
    pv_thresh = config.modeling.poor_fit_p_value_threshold
    r2_thresh = config.modeling.poor_fit_rsquared_threshold
    weak_losers_df = _weak_losing_candidates(
        proposal_output.candidates_df, pv_thresh, r2_thresh
    )
    lines.append("## Poor-fit selections")
    lines.append("")
    lines.append(
        f"Rows where the winning fit's p-value exceeded "
        f"`modeling.poor_fit_p_value_threshold` ({pv_thresh:g}) or its "
        f"R² fell below `modeling.poor_fit_rsquared_threshold` ({r2_thresh:g}). "
        f"The failing threshold(s) are noted alongside each row; selections in "
        f"`### Winners with poor fit` still drive the proposal, while "
        f"`### Other weak candidates` lists losing candidates that the "
        f"selector dropped in favor of a qualified alternative. See "
        f"`runtime_estimation_autogenerated_report.md` for per-fit "
        f"`runtime_ms`, `pvalue`, and `rsquared` metrics."
    )
    lines.append("")
    lines.append("### Winners with poor fit")
    lines.append("")
    if poor_fit_rows.empty:
        lines.append("_None._")
        lines.append("")
    else:
        lines.extend(_render_poor_fit_table(poor_fit_rows, pv_thresh, r2_thresh))
        lines.append("")
    lines.append("### Other weak candidates")
    lines.append("")
    if weak_losers_df.empty:
        lines.append("_None._")
        lines.append("")
    else:
        lines.extend(_render_weak_losers_table(weak_losers_df, pv_thresh, r2_thresh))
        lines.append("")

    out_path.write_text("\n".join(lines))
