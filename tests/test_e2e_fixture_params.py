"""End-to-end: derived fixture params (rename + value remap).

Pins the per-spec derived-column mechanic:

- A spec may declare `fixture_params: {<derived>: {source: <raw>}}` to rename a
  raw fixture-param into a canonical name used by `model_by` / `model_params`.
- The optional `values:` mapping remaps non-numeric source values to floats so
  the regressor can consume them.
- Two specs can declare the same derived name from different sources — the
  derived columns live on the per-spec slice of `fixtures_df`, never the
  shared frame.
- An observed source value that is not in `values:` is a fit-time error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    base_config,
    cross_product_fixtures,
    run_pipeline,
    write_standard_inputs,
)

from evm_gasfit.errors import ConfigError

_ACCOUNT_VALUE_SENT_SPEC = {
    "test_name": "test_account_access",
    "target_operation": "BALANCE",
    "model_by": "update",
    "fixture_params": {"update": {"source": "value_sent"}},
    "model_params": {"target_coef": "COLD_ACCOUNT_ACCESS", "update": "ACCOUNT_WRITE"},
}

_SSTORE_REMAP_SPEC = {
    "test_name": "test_sstore_bloated",
    "target_operation": "SSTORE",
    "model_by": "update",
    "fixture_params": {
        "update": {"source": "write_new_value", "values": {"False": 0, "True": 1}},
    },
    "model_params": {"target_coef": "COLD_STORAGE_WRITE", "update": "STORAGE_WRITE"},
}


def _account_fixtures(values=("0", "1", "2")):
    return cross_product_fixtures(
        test_file="test_account_access",
        test_name="test_account_access",
        param_grid={"value_sent": list(values)},
        target_opcode_for="BALANCE",
        target_opcount_per_million=800_000,
    )


def _sstore_fixtures(values=("False", "True")):
    return cross_product_fixtures(
        test_file="test_sstore_bloated",
        test_name="test_sstore_bloated",
        param_grid={"write_new_value": list(values)},
        target_opcode_for="SSTORE",
        target_opcount_per_million=500_000,
    )


def test_fixture_param_rename_passes_through_numeric_value(tmp_path: Path) -> None:
    """`fixture_params` with only `source:` renames a raw param; values pass through as floats."""
    fixtures = _account_fixtures()
    true_slope = 1.0e-5
    models = {"geth": ClientModel(intercept=70.0, slope=true_slope)}
    config = base_config(
        models_custom=[_ACCOUNT_VALUE_SENT_SPEC],
        new_params={"ACCOUNT_WRITE": None},
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models=models,
        config=config,
        noise_pct=0.002,
        seed=3,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    results = pd.read_csv(out_dir / "results.csv")
    assert len(results) == 3  # 3 update values × 1 client
    assert "update" in results.columns
    assert set(results["update"].astype(float)) == {0.0, 1.0, 2.0}

    for _, row in results.iterrows():
        recovered = float(row["target_coef_runtime_ms"])
        assert recovered == pytest.approx(true_slope, rel=0.05), (
            f"update={row['update']}: target_coef {recovered} not ~{true_slope}"
        )


def test_fixture_param_value_remap_translates_strings(tmp_path: Path) -> None:
    """`values:` translates non-numeric source values to floats the regressor can use."""
    fixtures = _sstore_fixtures()
    models = {"geth": ClientModel(intercept=60.0, slope=1.2e-5)}
    config = base_config(
        models_custom=[_SSTORE_REMAP_SPEC],
        new_params={"STORAGE_WRITE": None},
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models=models,
        config=config,
        noise_pct=0.002,
        seed=5,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    results = pd.read_csv(out_dir / "results.csv")
    assert len(results) == 2  # 2 update groups × 1 client
    assert "update" in results.columns
    assert set(results["update"].astype(float)) == {0.0, 1.0}
    # Raw strings did not survive into the per-spec column.
    assert "False" not in set(results["update"].astype(str))
    assert "True" not in set(results["update"].astype(str))


def test_two_specs_can_share_derived_name_with_different_sources(
    tmp_path: Path,
) -> None:
    """The derived column is per-spec — two specs may both declare `update`
    from different raw params."""
    fixtures = _account_fixtures(values=("0", "1")) + _sstore_fixtures()
    models = {
        "geth": ClientModel(
            intercept=80.0,
            slope=1.0e-5,
            extra_coefs={"value_sent": 2.0e-6},
        )
    }
    config = base_config(
        models_custom=[_ACCOUNT_VALUE_SENT_SPEC, _SSTORE_REMAP_SPEC],
        new_params={"ACCOUNT_WRITE": None, "STORAGE_WRITE": None},
        clients=("geth",),
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models=models,
        config=config,
        noise_pct=0.002,
        seed=9,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    results = pd.read_csv(out_dir / "results.csv")
    assert len(results) == 4  # 2 update values per spec × 2 specs × 1 client
    assert set(results["test_name"]) == {"test_account_access", "test_sstore_bloated"}
    for test_name in ("test_account_access", "test_sstore_bloated"):
        sub = results[results["test_name"] == test_name]
        assert len(sub) == 2
        assert set(sub["update"].astype(float)) == {0.0, 1.0}

    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    proposed = set(new_gas["gas_param"])
    # Only target_coef params survive: `update` is constant within each
    # per-group fit and is dropped by the one-value-extras rule.
    expected = {"COLD_ACCOUNT_ACCESS", "COLD_STORAGE_WRITE"}
    assert expected.issubset(proposed), (
        f"new_gas.csv missing params: {expected - proposed}"
    )

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert set(qualification["status"]) == {"inconclusive"}
    assert (
        qualification["reasons"]
        .str.contains("configured coefficient\\(s\\) were not estimated")
        .all()
    )


def test_uncertainty_gate_rejects_zero_pinned_mapped_coefficient(
    tmp_path: Path,
) -> None:
    """Every emitted mapped coefficient must satisfy an armed uncertainty gate."""
    fixtures = cross_product_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        param_grid={"workload": ("0", "1", "2")},
        target_opcode_for="ADD",
        target_opcount_per_million=800_000,
    )
    config = base_config(
        clients=("geth",),
        models_custom=[
            {
                "test_name": "test_arithmetic",
                "target_operation": "ADD",
                "model_params": {"target_coef": "OPCODE_ADD"},
                "setup_params": {"workload": "ACCOUNT_WRITE"},
            }
        ],
        new_params={"ACCOUNT_WRITE": None},
        extra={
            "qualification": {
                "max_relative_uncertainty": 0.1,
                "enforce_fit_quality": True,
            },
            "derived": {"DERIVED_ACCOUNT_WRITE": "ACCOUNT_WRITE"},
            "pricing_scenarios": [{"name": "strict", "anchor_rate": 100_000_000}],
        },
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=80.0, slope=1.0e-5)},
        config=config,
        noise_pct=0.0,
    )

    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert qualification.iloc[0]["status"] == "inconclusive"
    assert (
        "relative CI width for coefficient 'workload' is not computable"
        in qualification.iloc[0]["reasons"]
    )
    assert "p-value for coefficient 'workload'" in qualification.iloc[0]["reasons"]

    scenarios = pd.read_csv(out_dir / "pricing_scenarios.csv")
    derived = scenarios[scenarios["gas_param"] == "DERIVED_ACCOUNT_WRITE"].iloc[0]
    assert derived["qualification_status"] == "inconclusive"
    assert pd.isna(derived["scenario_gas"])


def test_unmapped_source_value_raises(tmp_path: Path) -> None:
    """An observed source value that the `values:` map omits is a fit-time error."""
    fixtures = _sstore_fixtures()
    models = {"geth": ClientModel(intercept=60.0, slope=1.2e-5)}
    spec = {
        **_SSTORE_REMAP_SPEC,
        # "True" deliberately missing from the map.
        "fixture_params": {
            "update": {"source": "write_new_value", "values": {"False": 0}},
        },
    }
    config = base_config(models_custom=[spec], new_params={"STORAGE_WRITE": None})
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models=models,
        config=config,
        noise_pct=0.002,
        seed=13,
    )

    from evm_gasfit import GasFit

    gas_fit = GasFit.from_config(config_yaml)
    gas_fit.load_runtimes(runtimes_csv)
    gas_fit.load_opcounts(opcounts_json)

    # The unmapped value is detected when the spec slice is materialized
    # inside `estimate_models()`.
    from evm_gasfit.errors import ModelingError

    with pytest.raises(ModelingError, match="values map omits observed"):
        gas_fit.estimate_models()
    assert not (out_dir / "results.csv").exists()


def test_explicit_runtime_param_coalesces_with_fixture_identity(tmp_path: Path) -> None:
    """CSV parameters can augment fixture identity without pandas suffixes."""
    fixtures = _account_fixtures(values=("0", "1"))
    config = base_config(
        models_custom=[
            {
                "test_name": "test_account_access",
                "target_operation": "BALANCE",
                "model_by": "workload_shape",
                "model_params": {"target_coef": "COLD_ACCOUNT_ACCESS"},
            }
        ],
        clients=("geth",),
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=70.0, slope=1.0e-5)},
        config=config,
        noise_pct=0.0,
    )
    value_by_fixture = {
        fixture.fixture_name: int(fixture.params["value_sent"]) for fixture in fixtures
    }
    runtimes = pd.read_csv(runtimes_csv)
    runtimes["param_value_sent"] = runtimes["fixture_name"].map(value_by_fixture)
    runtimes["param_workload_shape"] = runtimes["param_value_sent"].map(
        {0: "small", 1: "large"}
    )
    runtimes.to_csv(runtimes_csv, index=False)

    from evm_gasfit import GasFit

    gas_fit = GasFit.from_config(config_yaml)
    gas_fit.load_runtimes(runtimes_csv)
    gas_fit.load_opcounts(opcounts_json)
    gas_fit.estimate_models()
    gas_fit.build_proposal()
    gas_fit.write_reports(out_dir)

    assert gas_fit.fixtures_df is not None
    assert "param_value_sent" in gas_fit.fixtures_df
    assert "param_workload_shape" in gas_fit.fixtures_df
    assert "param_value_sent_x" not in gas_fit.fixtures_df
    assert "param_value_sent_y" not in gas_fit.fixtures_df
    results = pd.read_csv(out_dir / "results.csv")
    assert set(results["param_workload_shape"]) == {"small", "large"}


def test_explicit_runtime_param_conflict_is_rejected(tmp_path: Path) -> None:
    """CSV facts cannot contradict the fixture-name identity."""
    fixtures = _account_fixtures(values=("0",))
    config = base_config(
        models_custom=[
            {
                "test_name": "test_account_access",
                "target_operation": "BALANCE",
                "model_params": {"target_coef": "COLD_ACCOUNT_ACCESS"},
            }
        ],
        clients=("geth",),
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=70.0, slope=1.0e-5)},
        config=config,
        noise_pct=0.0,
    )
    runtimes = pd.read_csv(runtimes_csv)
    runtimes["param_value_sent"] = 999
    runtimes.to_csv(runtimes_csv, index=False)

    with pytest.raises(ConfigError, match="param_value_sent.*conflicts"):
        run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)
