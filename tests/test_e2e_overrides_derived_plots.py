"""End-to-end: gas-cost overrides, derived params, and the plots toggle.

Each test exercises one independent concern:

- `gas_costs.overrides` patches the fork defaults; the patched value flows
  through to `new_gas_proposal.md`.
- `derived:` entries (both alias-form and `{formula: ...}` form) evaluate
  against the rounded-integer worst-case table, respect declaration order,
  and appear in `new_gas.csv`.
- `output.plots: false` skips figure rendering *and* image embeds; setting
  it to true populates `figs/runtime/` and embeds via markdown image syntax.
- An unresolved identifier in a `derived:` formula is a load-time error.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    base_config,
    make_block_limit_fixtures,
    run_pipeline,
    write_standard_inputs,
)

_SPECS = [
    ("test_storage_cold_write", "SSTORE", "COLD_STORAGE_WRITE"),
    ("test_storage_cold_access", "SLOAD", "COLD_STORAGE_ACCESS"),
    ("test_account_cold_code", "EXTCODESIZE", "COLD_ACCOUNT_CODE_ACCESS"),
]


def _build_simple_inputs(
    tmp_path: Path,
    *,
    plots: bool,
    overrides: dict | None = None,
    derived: dict | None = None,
    pricing_scenarios: list[dict] | None = None,
) -> tuple[Path, Path, Path, Path]:
    fixtures = []
    for tn, opcode, _gas in _SPECS:
        fixtures.extend(
            make_block_limit_fixtures(
                test_file=tn,
                test_name=tn,
                target_opcode=opcode,
                params={"opcode": opcode},
                target_opcount_per_million=500_000,
            )
        )
    models = {
        "geth": ClientModel(intercept=80.0, slope=2.0e-5),
        "besu": ClientModel(intercept=100.0, slope=2.5e-5),
    }
    config = base_config(
        plots=plots,
        overrides=overrides,
        new_params={"COLD_ACCOUNT_CODE_ACCESS": None},
        models_custom=[
            {
                "test_name": tn,
                "target_operation": opcode,
                "model_params": {"target_coef": gas},
            }
            for tn, opcode, gas in _SPECS
        ],
    )
    if derived is not None:
        config["derived"] = derived
    if pricing_scenarios is not None:
        config["pricing_scenarios"] = pricing_scenarios
    return write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models=models,
        config=config,
        noise_pct=0.002,
        seed=9,
    )


def test_gas_overrides_flow_to_proposal(tmp_path: Path) -> None:
    """`gas_costs.overrides` patches the instantiated `GasCosts`. The proposal
    diff renders the overridden value as the current default."""
    override_value = 9999  # atypical so a substring match is unambiguous.
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_simple_inputs(
        tmp_path,
        plots=False,
        overrides={"COLD_STORAGE_ACCESS": override_value},
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    proposal = (out_dir / "new_gas_proposal.md").read_text()
    assert str(override_value) in proposal


def test_derived_alias_and_formula_evaluate(tmp_path: Path) -> None:
    """Alias-form, formula-form, and a chained formula referencing a prior
    derived name all produce rows in `new_gas.csv`."""
    derived = {
        "ACCESS_LIST_ADDRESS": "COLD_ACCOUNT_CODE_ACCESS",  # alias
        "STORAGE_CLEAR_REFUND": {
            "formula": "(COLD_STORAGE_WRITE + COLD_STORAGE_ACCESS) * 4800 / 5000"
        },
        "DOUBLED_ACCESS_LIST_ADDRESS": {
            "formula": "ACCESS_LIST_ADDRESS * 2"
        },  # chained
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_simple_inputs(
        tmp_path, plots=False, derived=derived
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    gas_params = set(new_gas["gas_param"])
    assert {
        "ACCESS_LIST_ADDRESS",
        "STORAGE_CLEAR_REFUND",
        "DOUBLED_ACCESS_LIST_ADDRESS",
    }.issubset(gas_params)

    def rounded(name: str) -> int:
        return int(new_gas.loc[new_gas["gas_param"] == name, "new_gas_rounded"].iloc[0])

    cold_write = rounded("COLD_STORAGE_WRITE")
    cold_access = rounded("COLD_STORAGE_ACCESS")
    cold_code = rounded("COLD_ACCOUNT_CODE_ACCESS")

    # Derived values are computed against the rounded integer table, then
    # rounded up to an integer themselves.
    assert rounded("ACCESS_LIST_ADDRESS") == cold_code
    assert rounded("STORAGE_CLEAR_REFUND") == math.ceil(
        (cold_write + cold_access) * 4800 / 5000
    )
    assert rounded("DOUBLED_ACCESS_LIST_ADDRESS") == rounded("ACCESS_LIST_ADDRESS") * 2


def test_pricing_scenarios_evaluate_derived_params_in_declaration_order(
    tmp_path: Path,
) -> None:
    """Scenario prices include qualified derived aliases and dependencies."""
    derived = {
        "ACCESS_LIST_ADDRESS": "COLD_ACCOUNT_CODE_ACCESS",
        "DOUBLED_ACCESS_LIST_ADDRESS": {"formula": "ACCESS_LIST_ADDRESS * 2"},
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_simple_inputs(
        tmp_path,
        plots=False,
        derived=derived,
        pricing_scenarios=[
            {"name": "conservative", "anchor_rate": 150_000_000, "margin_pct": 10}
        ],
    )

    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    scenarios = pd.read_csv(out_dir / "pricing_scenarios.csv")
    conservative = scenarios[scenarios["scenario"] == "conservative"]
    assert {
        "ACCESS_LIST_ADDRESS",
        "DOUBLED_ACCESS_LIST_ADDRESS",
    }.issubset(set(conservative["gas_param"]))

    def scenario_price(name: str) -> int:
        return int(
            conservative.loc[conservative["gas_param"] == name, "scenario_gas"].iloc[0]
        )

    assert scenario_price("ACCESS_LIST_ADDRESS") == scenario_price(
        "COLD_ACCOUNT_CODE_ACCESS"
    )
    assert scenario_price("DOUBLED_ACCESS_LIST_ADDRESS") == (
        scenario_price("ACCESS_LIST_ADDRESS") * 2
    )


@pytest.mark.parametrize("plots", [True, False])
def test_plots_toggle(tmp_path: Path, plots: bool) -> None:
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_simple_inputs(
        tmp_path, plots=plots
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    figs_dir = out_dir / "figs"
    report = (out_dir / "runtime_estimation_autogenerated_report.md").read_text()

    if plots:
        runtime_figs = list((figs_dir / "runtime").glob("*.png"))
        assert runtime_figs, "expected PNGs under figs/runtime/ when plots: true"
        assert "figs/runtime/" in report
        assert "![" in report
    else:
        if figs_dir.exists():
            assert not any(figs_dir.rglob("*.png"))
        assert "![" not in report
        assert "<img" not in report


def test_derived_formula_unknown_identifier_is_load_time_error(tmp_path: Path) -> None:
    """A `derived:` formula identifier that resolves against neither the
    fork's raw fields, nor any `model_params` RHS, nor any earlier-declared
    `derived:` key, must fail at `GasFit.from_config(...)`."""
    derived = {"STORAGE_CLEAR_REFUND": {"formula": "COLD_STORAGE_WIRTE + 100"}}  # typo
    config_yaml, _, _, _ = _build_simple_inputs(tmp_path, plots=False, derived=derived)

    from evm_gasfit import GasFit
    from evm_gasfit.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown identifier"):
        GasFit.from_config(config_yaml)
