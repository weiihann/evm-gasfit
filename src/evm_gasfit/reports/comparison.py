"""The mandatory compute-vs-current-gas comparison.

This table exists with or without a pricing anchor. With an anchor it
complements the proposal's diff table; without one (``anchor_rate: null``,
a comparison-only run) it *is* the primary output — measured cost is
reported against the active schedule without inventing a conversion rate
and without falling back to some other fork's anchor.

``relative_costliness`` is a unitless descriptive index: each param's
``runtime_ms / current_gas`` divided by the median of that ratio across
params carrying a current value. It says how expensive the operation is
*relative to its current pricing* compared with its peers in this run —
not a proposed price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evm_gasfit.config import Config
from evm_gasfit.proposal.build import ProposalOutput
from evm_gasfit.proposal.derived import evaluate

COMPARISON_COLUMNS = [
    "gas_param",
    "client_name",
    "runtime_ms",
    "conf_int_low",
    "conf_int_high",
    "current_gas",
    "current_gas_source",
    "measured_ms_per_gas",
    "relative_costliness",
    "feature_kind",
    "qualification_status",
    "new_gas_decimal",
    "new_gas_rounded",
]


def build_comparison_df(
    proposal_output: ProposalOutput, config: Config
) -> pd.DataFrame:
    """Assemble the compute-vs-current-gas comparison table."""
    new_gas_df = proposal_output.new_gas_df
    if new_gas_df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    fork_fields = set(config.raw_fork_fields)
    new_param_values = {k: v for k, v in config.new_params.items() if v is not None}

    rows: list[dict[str, object]] = []
    for _, row in new_gas_df.iterrows():
        gp = str(row["gas_param"])
        if gp in fork_fields:
            current: int | None = int(proposal_output.current_values.get(gp, 0))
            source = "fork"
        elif gp in new_param_values:
            current = int(new_param_values[gp])
            source = "new_params"
        else:
            current = None
            source = "none"
        runtime = row.get("runtime_ms")
        runtime = (
            float(runtime) if runtime is not None and not pd.isna(runtime) else None
        )
        if current is not None and current > 0 and runtime is not None:
            per_gas = runtime / current
        else:
            per_gas = None
        rows.append(
            {
                "gas_param": gp,
                "client_name": row.get("client_name", ""),
                "runtime_ms": runtime,
                "conf_int_low": row.get("conf_int_low"),
                "conf_int_high": row.get("conf_int_high"),
                "current_gas": current,
                "current_gas_source": source,
                "measured_ms_per_gas": per_gas,
                "relative_costliness": None,
                "feature_kind": row.get("feature_kind", ""),
                "qualification_status": row.get("qualification_status", ""),
                "new_gas_decimal": row.get("new_gas_decimal"),
                "new_gas_rounded": row.get("new_gas_rounded"),
            }
        )

    df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    ratios = df["measured_ms_per_gas"].dropna()
    if not ratios.empty:
        median = float(np.median(ratios.to_numpy()))
        if median > 0:
            df["relative_costliness"] = df["measured_ms_per_gas"] / median
    return df.sort_values("gas_param", kind="mergesort").reset_index(drop=True)


def write_comparison_csv(out_dir: Path, comparison_df: pd.DataFrame) -> None:
    """Write ``compute_gas_comparison.csv`` under ``out_dir``."""
    comparison_df.to_csv(
        out_dir / "compute_gas_comparison.csv", index=False, lineterminator="\n"
    )


def build_pricing_scenarios_df(
    proposal_output: ProposalOutput, config: Config
) -> pd.DataFrame:
    """Render every configured explicit anchor+margin scenario.

    Direct prices are ``ceil(anchor_rate · (1 + margin_pct/100) · runtime_ms /
    1000)``. Derived parameters are then evaluated in declaration order against
    those scenario-specific integers; an unavailable dependency withholds its
    derived price.
    """
    if not config.pricing_scenarios:
        return pd.DataFrame(
            columns=[
                "scenario",
                "anchor_rate",
                "margin_pct",
                "gas_param",
                "runtime_ms",
                "scenario_gas",
                "qualification_status",
            ]
        )
    columns = [
        "scenario",
        "anchor_rate",
        "margin_pct",
        "gas_param",
        "runtime_ms",
        "scenario_gas",
        "qualification_status",
    ]
    rows: list[dict[str, object]] = []
    direct_rows = proposal_output.new_gas_df[
        proposal_output.new_gas_df["feature_kind"].astype(str) != "derived"
    ]
    for scenario in config.pricing_scenarios:
        price = scenario.anchor_rate * (1.0 + scenario.margin_pct / 100.0)
        env: dict[str, int | None] = {}
        for _, row in direct_rows.iterrows():
            runtime = row.get("runtime_ms")
            has_runtime = runtime is not None and not pd.isna(runtime)
            qualification = str(row.get("qualification_status", ""))
            qualified = qualification == "qualified"
            scenario_gas = (
                int(np.ceil(price * float(runtime) / 1000.0))
                if qualified and has_runtime
                else None
            )
            gas_param = str(row["gas_param"])
            env[gas_param] = scenario_gas
            rows.append(
                {
                    "scenario": scenario.name,
                    "anchor_rate": scenario.anchor_rate,
                    "margin_pct": scenario.margin_pct,
                    "gas_param": gas_param,
                    "runtime_ms": float(runtime) if has_runtime else float("nan"),
                    "scenario_gas": (
                        scenario_gas if scenario_gas is not None else pd.NA
                    ),
                    "qualification_status": qualification,
                }
            )

        for name, (_raw, tree) in config.derived_evaluated.items():
            value = evaluate(tree, env)
            scenario_gas = None if value is None else int(np.ceil(value))
            env[name] = scenario_gas
            rows.append(
                {
                    "scenario": scenario.name,
                    "anchor_rate": scenario.anchor_rate,
                    "margin_pct": scenario.margin_pct,
                    "gas_param": name,
                    "runtime_ms": float("nan"),
                    "scenario_gas": (
                        scenario_gas if scenario_gas is not None else pd.NA
                    ),
                    "qualification_status": (
                        "qualified" if scenario_gas is not None else "inconclusive"
                    ),
                }
            )
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.sort_values(["scenario", "gas_param"], kind="mergesort").reset_index(
        drop=True
    )


def write_pricing_scenarios_csv(out_dir: Path, scenarios_df: pd.DataFrame) -> None:
    """Write ``pricing_scenarios.csv`` under ``out_dir``."""
    scenarios_df.to_csv(
        out_dir / "pricing_scenarios.csv", index=False, lineterminator="\n"
    )
