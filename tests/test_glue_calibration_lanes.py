"""Calibration-lane separation and glue-coverage regression tests.

Focused coverage for the explicit two-lane campaign design and the
coverage ledger that gates isolated recommendations:

- ``param_campaign_role=calibration`` rows feed glue drivers only and can
  never reach a target-model fit, even through a selector that matches
  their test name (lane separation).
- A detected count-correlated supporter that cannot be priced blocks the
  adjusted recommendation instead of silently vanishing (unpriced).
- STOP 1:1 behind a priced STATICCALL is attributed by wrapper-bundle
  accounting when the calibration driver itself carries the STOP (bundled).
- A target that is itself a PUSH/DUP/SWAP family member never has its own
  family subtracted back as glue (group self-exclusion).
- Groups with too few fixture points cannot claim clean coverage
  (detection coverage).
- A contaminated pure driver yields a non-isolated coefficient, and using
  it downstream is blocked (isolation).
- Two specs sharing test/target but differing in filter_by keep separate
  detector candidates (source_label routing).
- Slice-local partner detection subtracts pure support from the joint
  cycle fit (cycle partner subtraction).
- Curved count relations produce unreliable ratios (ratio estimators).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    make_block_limit_fixtures,
    make_glue_driver_fixtures,
    base_config,
    run_pipeline,
    write_standard_inputs,
)

from evm_gasfit.errors import ConfigError
from evm_gasfit.glue.lane import (
    CAMPAIGN_ROLE_COLUMN,
    ROLE_CALIBRATION,
    ROLE_TARGET,
    split_campaign_lanes,
)


def _read(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out_dir / name)


# ----- Lane split unit behavior -------------------------------------------


def test_split_campaign_lanes_unit() -> None:
    df = pd.DataFrame(
        {
            "fixture_name": ["a", "b", "c"],
            "test_name": ["t", "t", "t"],
            CAMPAIGN_ROLE_COLUMN: ["target", "Calibration ", None],
        }
    )
    target, calibration = split_campaign_lanes(df)
    assert list(target["fixture_name"]) == ["a", "c"]
    assert list(calibration["fixture_name"]) == ["b"]


def test_split_campaign_lanes_without_column_returns_all_target() -> None:
    df = pd.DataFrame({"fixture_name": ["a"], "test_name": ["t"]})
    target, calibration = split_campaign_lanes(df)
    assert len(target) == 1
    assert calibration.empty


def test_split_campaign_lanes_rejects_unknown_role() -> None:
    df = pd.DataFrame(
        {
            "fixture_name": ["a"],
            "test_name": ["t"],
            CAMPAIGN_ROLE_COLUMN: ["calibraton"],  # typo on purpose
        }
    )
    with pytest.raises(ConfigError, match="param_campaign_role"):
        split_campaign_lanes(df)


def test_lane_constants() -> None:
    assert CAMPAIGN_ROLE_COLUMN == "param_campaign_role"
    assert ROLE_TARGET == "target"
    assert ROLE_CALIBRATION == "calibration"


# ----- Calibration rows never reach target fits ---------------------------


def test_calibration_rows_never_reach_target_fits(tmp_path: Path) -> None:
    """A calibration sweep sharing the target's test name must not leak
    into the priced fit, even though the spec selector matches it."""
    target_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        block_limits=(30, 60, 90, 120, 150, 180, 210, 240),
    )
    # Calibration-lane rows for the SAME test_name: identical shape but a
    # massive SHL contribution. If they leaked into the target regression
    # the fitted slope would explode.
    calibration_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        block_limits=(300, 330, 360, 390, 420, 450, 480, 510),
        extra_opcount_per_million={"SHL": 4_000_000.0},
        campaign_role="calibration",
    )
    all_fixtures = target_fixtures + calibration_fixtures + make_glue_driver_fixtures()
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5, glue_coefs={"SHL": 5e-4})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=11,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    results = _read(out_dir, "results.csv")
    add_rows = results[results["target_opcode"] == "ADD"]
    assert len(add_rows) == 1
    # The target coefficient reflects the target-lane sweep only; the
    # calibration rows (10x counts + huge SHL contamination) did not enter.
    assert float(add_rows.iloc[0]["target_coef_runtime_ms"]) == pytest.approx(
        2.0e-5, rel=0.05
    )
    # Both lanes still feed the glue machinery (drivers priced).
    glue_results = _read(out_dir, "glue_results.csv")
    assert glue_results["glue_runtime_ms"].notna().any()


def test_calibration_rows_leak_changes_slope_without_lane_guard(
    tmp_path: Path,
) -> None:
    """Sanity for the guard above: without the role column the very same
    contaminated rows DO shift the fitted slope, so the previous test is
    actually exercising the lane split (not an inert fixture set)."""
    target_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        block_limits=(30, 60, 90, 120, 150, 180, 210, 240),
    )
    contaminated = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        block_limits=(300, 330, 360, 390, 420, 450, 480, 510),
        extra_opcount_per_million={"SHL": 4_000_000.0},
        campaign_role=None,  # no lane marker → part of the target corpus
    )
    all_fixtures = target_fixtures + contaminated + make_glue_driver_fixtures(
        campaign_role=None
    )
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5, glue_coefs={"SHL": 5e-4})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=11,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)
    results = _read(out_dir, "results.csv")
    add_rows = results[results["target_opcode"] == "ADD"]
    slope = float(add_rows.iloc[0]["target_coef_runtime_ms"])
    assert slope > 4.0 * 2.0e-5  # contamination dragged the slope far up


# ----- Unpriced supporters block isolated recommendations ------------------


def test_unpriced_supporter_blocks_isolated_recommendation(tmp_path: Path) -> None:
    """SHL correlates with the ADD sweep but is not a priced glue opcode:
    its cost cannot be removed, so the adjusted estimate must be labeled
    incomplete and blocked from recommended prices — never silently
    reported as an isolated cost."""
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={"SHL": 500_000.0},
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5, glue_coefs={"SHL": 4.0e-5})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=12,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert add_row["glue_unpriced_opcodes"] == "SHL"
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))

    qualification = _read(out_dir, "qualification.csv")
    qual_row = qualification.iloc[0]
    assert qual_row["status"] == "qualified"
    assert qual_row["adjusted_estimate_status"] == "inconclusive"
    assert "not isolable: SHL" in str(qual_row["reasons"])

    summary = _read(out_dir, "new_gas.csv")
    add_summary = summary[summary["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert bool(pd.isna(add_summary["new_gas_rounded"]))


def test_insufficient_fixture_points_blocks_coverage(tmp_path: Path) -> None:
    """A 4-point sweep cannot run detection; coverage must not claim clean."""
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        block_limits=(30, 90, 150, 240),
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5)}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=13,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    coverage = _read(out_dir, "glue_detection_coverage.csv")
    add = coverage[coverage["test_name"] == "test_arithmetic"].iloc[0]
    assert int(add["n_fixture_points"]) == 4
    assert add["detection_status"] == "insufficient_fixture_points"

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert add_row["glue_detection_status"] == "insufficient_fixture_points"
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))


# ----- Wrapper-bundle accounting ------------------------------------------


def test_stop_bundled_into_priced_staticcall(tmp_path: Path) -> None:
    """A 1:1 STOP behind every STATICCALL is covered when the STATICCALL
    calibration driver itself carries a 1:1 STOP (STOP-only callee)."""
    drivers = [
        f
        for f in make_glue_driver_fixtures()
        if f.test_name != "test_ext_account_query_warm"
    ]
    drivers += make_block_limit_fixtures(
        test_file="test_ext_account_query_warm",
        test_name="test_ext_account_query_warm",
        target_opcode="STATICCALL",
        params={"opcode": "STATICCALL"},
        target_opcount_per_million=2_000_000.0,
        extra_opcount_per_million={"STOP": 2_000_000.0},
        campaign_role="calibration",
    )
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={
            "STATICCALL": 500_000.0,
            "STOP": 500_000.0,
        },
    )
    models = {
        "geth": ClientModel(
            intercept=50.0,
            slope=0.0,
            glue_coefs={"ADD": 2.0e-5, "STATICCALL": 2.0e-5, "STOP": 1.0e-5},
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=main_fixtures + drivers,
        models=models,
        config=base_config(glue_enabled=True),
        seed=14,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    support = _read(out_dir, "glue_driver_support.csv")
    sc_stop = support[
        (support["glue_opcode"] == "STATICCALL") & (support["support_opcode"] == "STOP")
    ]
    assert len(sc_stop) == 1
    assert float(sc_stop.iloc[0]["ratio_per_driver_count"]) == pytest.approx(1.0)

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert add_row["glue_priced_opcodes"] == "STATICCALL"
    assert add_row["glue_bundled_opcodes"] == "STOP"
    assert bool(add_row["glue_coverage_complete"])
    # The STATICCALL coefficient absorbed the callee STOP, so subtracting it
    # removes the full wrapper: adjusted slope recovers the planted cost.
    assert float(add_row["runtime_ms"]) == pytest.approx(2.0e-5, rel=0.1)
    assert not bool(pd.isna(add_row["new_gas_rounded"]))


def test_stop_without_covering_driver_stays_unpriced(tmp_path: Path) -> None:
    """STOP collinear with STATICCALL but no driver-side STOP evidence
    (clean callee) is NOT bundled — it blocks instead of faking zero."""
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={
            "STATICCALL": 500_000.0,
            "STOP": 500_000.0,
        },
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {
        "geth": ClientModel(
            intercept=50.0,
            slope=2.0e-5,
            glue_coefs={"STATICCALL": 2.0e-5, "STOP": 1.0e-5},
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=15,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert "STOP" in add_row["glue_unpriced_opcodes"]
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))


@pytest.mark.parametrize(
    ("driver_support", "expected_complete"),
    [
        ({"STOP": 1.0}, True),
        ({"STOP": 1.0, "SHL": 1.0}, False),
        ({"STOP": 2.0}, False),
    ],
    ids=["exact", "extra-driver-support", "duplicated-stop"],
)
def test_stop_bundle_requires_exact_aggregate_composition(
    driver_support: dict[str, float], expected_complete: bool
) -> None:
    """Bundle coverage rejects extra or duplicated embedded work."""
    from evm_gasfit.glue.adjust import compute_glue_adjustment

    target = pd.DataFrame(
        [
            {
                "source_label": "arithmetic",
                "test_name": "test_arithmetic",
                "target_opcode": "ADD",
                "client_name": "geth",
                "target_coef_runtime_ms": 10.0,
                "target_coef_conf_int_low": 10.0,
                "target_coef_conf_int_high": 10.0,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "source_label": "arithmetic",
                "test_name": "test_arithmetic",
                "target_opcode": "ADD",
                "glue_opcode": "STATICCALL",
                "corr": 1.0,
                "ratio": 1.0,
                "ratio_reliable": True,
                "collinear_with": "STOP",
            },
            {
                "source_label": "arithmetic",
                "test_name": "test_arithmetic",
                "target_opcode": "ADD",
                "glue_opcode": "STOP",
                "corr": 1.0,
                "ratio": 1.0,
                "ratio_reliable": True,
                "collinear_with": "STATICCALL",
            },
        ]
    )
    glue = pd.DataFrame(
        [
            {
                "client_name": "geth",
                "glue_opcode": "STATICCALL",
                "glue_runtime_ms": 2.0,
                "p_value": 0.01,
                "rsquared": 1.0,
                "isolated": True,
            }
        ]
    )
    support_rows = [
        {
            "glue_opcode": "STATICCALL",
            "test_name": "test_ext_account_query_warm",
            "support_opcode": support,
            "ratio_per_driver_count": ratio,
        }
        for support, ratio in driver_support.items()
    ]
    adjusted = compute_glue_adjustment(
        target,
        glue,
        candidates,
        p_threshold=0.05,
        r2_threshold=0.5,
        driver_support_df=pd.DataFrame(support_rows),
    ).iloc[0]

    assert bool(adjusted["glue_coverage_complete"]) is expected_complete
    if expected_complete:
        assert adjusted["glue_bundled_opcodes"] == "STOP"
        assert adjusted["glue_unpriced_opcodes"] == ""
    else:
        assert "STOP" in adjusted["glue_unpriced_opcodes"]
        assert "bundle composition mismatch" in adjusted["glue_coverage_reason"]



def test_calldatacopy_unknown_shape_blocks_coverage() -> None:
    """Input-dependent copy length cannot transfer an unknown driver rate."""
    from evm_gasfit.glue.adjust import compute_glue_adjustment

    target = pd.DataFrame(
        [
            {
                "source_label": "copy",
                "test_name": "test_copy",
                "target_opcode": "ADD",
                "client_name": "geth",
                "target_coef_runtime_ms": 5.0,
                "target_coef_conf_int_low": 5.0,
                "target_coef_conf_int_high": 5.0,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "source_label": "copy",
                "test_name": "test_copy",
                "target_opcode": "ADD",
                "glue_opcode": "CALLDATACOPY",
                "corr": 1.0,
                "ratio": 1.0,
                "ratio_reliable": True,
                "collinear_with": "",
            }
        ]
    )
    glue = pd.DataFrame(
        [
            {
                "client_name": "geth",
                "glue_opcode": "CALLDATACOPY",
                "glue_runtime_ms": 1.0,
                "p_value": 0.01,
                "rsquared": 1.0,
                "isolated": True,
            }
        ]
    )
    adjusted = compute_glue_adjustment(
        target,
        glue,
        candidates,
        p_threshold=0.05,
        r2_threshold=0.5,
    ).iloc[0]
    assert adjusted["glue_unpriced_opcodes"] == "CALLDATACOPY"
    assert not bool(adjusted["glue_coverage_complete"])
    assert "shape unverified" in adjusted["glue_coverage_reason"]



@pytest.mark.parametrize(
    ("target_stop_ratio", "driver_stop_ratio", "driver_extra"),
    [
        (0.0, 1.0, {}),
        (1.0, 2.0, {}),
        (1.0, 1.0, {"SHL": 1_000_000.0}),
    ],
    ids=["absent-target-stop", "duplicated-driver-stop", "extra-driver-support"],
)
def test_bundle_mismatch_withholds_adjusted_recommendation(
    tmp_path: Path,
    target_stop_ratio: float,
    driver_stop_ratio: float,
    driver_extra: dict[str, float],
) -> None:
    """Bundle composition mismatches downgrade qualification and pricing."""
    drivers = [
        fixture
        for fixture in make_glue_driver_fixtures()
        if fixture.test_name != "test_ext_account_query_warm"
    ]
    drivers += make_block_limit_fixtures(
        test_file="test_ext_account_query_warm",
        test_name="test_ext_account_query_warm",
        target_opcode="STATICCALL",
        params={"opcode": "STATICCALL"},
        target_opcount_per_million=2_000_000.0,
        extra_opcount_per_million={
            "STOP": driver_stop_ratio * 2_000_000.0,
            **driver_extra,
        },
        campaign_role="calibration",
    )
    target_extra = {"STATICCALL": 500_000.0}
    if target_stop_ratio:
        target_extra["STOP"] = 500_000.0 * target_stop_ratio
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million=target_extra,
    )
    models = {
        "geth": ClientModel(
            intercept=50.0,
            slope=2.0e-5,
            glue_coefs={
                "STATICCALL": 2.0e-5,
                "STOP": 1.0e-5,
                "SHL": 1.0e-5,
            },
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=main_fixtures + drivers,
        models=models,
        config=base_config(glue_enabled=True),
        seed=19,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    adjusted = _read(out_dir, "new_gas_all_params.csv")
    add_row = adjusted[adjusted["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))
    qualification = _read(out_dir, "qualification.csv").iloc[0]
    assert qualification["adjusted_estimate_status"] == "inconclusive"
# ----- Group self-exclusion for grouped families ---------------------------


def test_dup_family_target_never_subtracts_own_family(tmp_path: Path) -> None:
    """A DUP7 target with sibling DUP2 support must not have canonical DUP
    subtracted back: the family's own work is the target, and the family
    parameter already charges for it."""
    config = base_config(
        models_custom=[
            {
                "test_name": "test_dup",
                "target_operation_param": "opcode",
                "model_by": ["opcode"],
                "model_params": {"target_coef": "OPCODE_DUP"},
            }
        ],
        glue_enabled=True,
    )
    main_fixtures = make_block_limit_fixtures(
        test_file="test_dup",
        test_name="test_dup",
        target_opcode="DUP7",
        params={"opcode": "DUP7"},
        extra_opcount_per_million={"DUP2": 1_000_000.0},
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {"geth": ClientModel(intercept=10.0, slope=2.0e-5, glue_coefs={"DUP2": 1.0e-5})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path, fixtures=all_fixtures, models=models, config=config, seed=16
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_by_test = _read(out_dir, "glue_opcodes_by_test.csv")
    dup_rows = glue_by_test[glue_by_test["test_name"] == "test_dup"]
    assert dup_rows.empty  # no DUP-family candidate for the DUP target group

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    dup_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_DUP"].iloc[0]
    assert float(dup_row["glue_adjustment"]) == 0.0
    assert bool(dup_row["glue_coverage_complete"])
    # Sibling cost stays inside the coefficient (conservative direction).
    assert float(dup_row["runtime_ms"]) == pytest.approx(3.0e-5, rel=0.1)
    assert not bool(pd.isna(dup_row["new_gas_rounded"]))


# ----- Isolation -----------------------------------------------------------


def test_contaminated_pure_driver_blocks_downstream(tmp_path: Path) -> None:
    """ISZERO straight-line driver polluted with JUMPDEST yields a
    non-isolated coefficient; a target that needs ISZERO subtracted is
    blocked rather than adjusted with a bundled slope."""
    drivers = []
    for fixture in make_glue_driver_fixtures():
        if fixture.test_name == "test_iszero_straight":
            fixture.extra_opcounts["JUMPDEST"] = fixture.target_opcount
        drivers.append(fixture)
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={"ISZERO": 500_000.0},
    )
    models = {
        "geth": ClientModel(
            intercept=50.0, slope=2.0e-5, glue_coefs={"ISZERO": 3.0e-5, "JUMPDEST": 2.0e-5}
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=main_fixtures + drivers,
        models=models,
        config=base_config(glue_enabled=True),
        seed=17,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_results = _read(out_dir, "glue_results.csv")
    iszero_row = glue_results[glue_results["glue_opcode"] == "ISZERO"].iloc[0]
    assert not bool(iszero_row["isolated"])
    assert "JUMPDEST" in str(iszero_row["unmodeled_partners"])

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert "ISZERO" in add_row["glue_unpriced_opcodes"]
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))


def test_contaminated_cycle_block_invalidates_all_cycle_members(tmp_path: Path) -> None:
    """One unpriced support in a joint block invalidates every cycle member."""
    drivers = []
    for fixture in make_glue_driver_fixtures():
        if fixture.test_name == "test_dup_straight":
            fixture.extra_opcounts["SHL"] = fixture.target_opcount
        drivers.append(fixture)
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={"DUP1": 500_000.0},
    )
    models = {
        "geth": ClientModel(
            intercept=50.0,
            slope=2.0e-5,
            glue_coefs={"DUP1": 3.0e-5, "SHL": 2.0e-5},
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=main_fixtures + drivers,
        models=models,
        config=base_config(glue_enabled=True),
        seed=18,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_results = _read(out_dir, "glue_results.csv")
    cycle = glue_results[glue_results["tier"] == "cycle"]
    assert not cycle.empty
    assert not cycle["isolated"].fillna(False).any()
    assert cycle["unmodeled_partners"].astype(str).str.contains("SHL").all()

def test_cycle_fit_subtracts_pure_partner(tmp_path: Path) -> None:
    """The joint cycle fit must charge ISZERO background to the ISZERO
    coefficient (pure tier), not absorb it into DUP's."""
    drivers = []
    for fixture in make_glue_driver_fixtures():
        if fixture.test_name == "test_dup_straight":
            fixture.extra_opcounts["ISZERO"] = fixture.target_opcount
        drivers.append(fixture)
    models = {
        "geth": ClientModel(
            intercept=10.0,
            slope=0.0,
            glue_coefs={
                **{f"DUP{i}": 3.0e-5 for i in range(1, 17)},
                "ISZERO": 4.0e-5,
                "ADD": 2.0e-5,
            },
        )
    }
    target_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=drivers + target_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=18,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_results = _read(out_dir, "glue_results.csv")
    dup_row = glue_results[glue_results["glue_opcode"] == "DUP"].iloc[0]
    assert bool(dup_row["isolated"])
    # DUP's coefficient is its own per-count cost; the ISZERO background
    # was subtracted on the LHS via the pure-tier fit.
    assert float(dup_row["glue_runtime_ms"]) == pytest.approx(3.0e-5, rel=0.15)


# ----- Source-label routing ------------------------------------------------


def test_source_label_routes_candidates_across_colliding_specs(tmp_path: Path):
    """Two specs sharing test_name + target but split by filter_by must not
    share detector candidates: only the contaminated variant adjusts."""
    config = base_config(
        models_custom=[
            {
                "test_name": "test_arithmetic",
                "target_operation": "ADD",
                "filter_by": ["variant_fast"],
                "model_params": {"target_coef": "OPCODE_ADD_FAST"},
            },
            {
                "test_name": "test_arithmetic",
                "target_operation": "ADD",
                "filter_by": ["variant_slow"],
                "model_params": {"target_coef": "OPCODE_ADD_SLOW"},
            },
        ],
        new_params={"OPCODE_ADD_FAST": None, "OPCODE_ADD_SLOW": None},
        glue_enabled=True,
    )
    fast = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD", "variant": "fast"},
    )
    slow = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD", "variant": "slow"},
        extra_opcount_per_million={"ISZERO": 500_000.0},
    )
    models = {"geth": ClientModel(intercept=50.0, slope=0.0, glue_coefs={"ADD": 2.0e-5, "ISZERO": 4.0e-5})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fast + slow + make_glue_driver_fixtures(),
        models=models,
        config=config,
        seed=19,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    fast_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD_FAST"].iloc[0]
    slow_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD_SLOW"].iloc[0]
    assert float(fast_row["glue_adjustment"]) == 0.0
    assert slow_row["glue_priced_opcodes"] == "ISZERO"
    assert float(slow_row["glue_adjustment"]) > 0.0
    assert bool(slow_row["glue_coverage_complete"])
    # Slow recovers the planted slope after subtracting the ISZERO bundle.
    assert float(slow_row["runtime_ms"]) == pytest.approx(2.0e-5, rel=0.1)
    assert float(fast_row["runtime_ms"]) == pytest.approx(2.0e-5, rel=0.1)


# ----- Ratio estimators ----------------------------------------------------


def test_passes_thresholds_flags_curved_ratios() -> None:
    from evm_gasfit.glue.detect import _passes_thresholds

    opcount = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    linear = 0.5 * opcount
    keep, corr, ratio, endpoint, reliable = _passes_thresholds(linear, opcount, 0.05)
    assert keep
    assert reliable
    assert ratio == pytest.approx(0.5)
    assert endpoint == pytest.approx(0.5)

    curved = 0.5 * opcount**1.5
    keep, corr, ratio, endpoint, reliable = _passes_thresholds(curved, opcount, 0.05)
    assert keep  # still detected — but flagged, never silently subtracted
    assert not reliable

    # Uniformly spaced quadratic counts make the mean local slope equal the
    # endpoint slope and the OLS slope, so endpoint-vs-OLS alone is insufficient.
    quadratic = 0.5 * opcount**2
    keep, corr, ratio, endpoint, reliable = _passes_thresholds(
        quadratic, opcount, 0.05
    )
    assert keep
    assert not reliable
    assert endpoint == pytest.approx(ratio)


def test_adjuster_blocks_unreliable_ratio(tmp_path: Path) -> None:
    """A variant-mixed support ratio (no single per-opcount rate) is
    detected but flagged unreliable; the row blocks instead of subtracting
    a biased amount."""
    fixtures = []
    ratios = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
    for bl, support in zip((30, 60, 90, 120, 150, 180, 210, 240), ratios):
        from _data_synth import FixtureSpec

        fixtures.append(
            FixtureSpec(
                test_file="test_arithmetic",
                test_name="test_arithmetic",
                params={"opcode": "ADD"},
                block_limit_million=bl,
                target_opcode="ADD",
                target_opcount=bl * 1_000_000.0,
                extra_opcounts={"SHL": bl * 1_000_000.0 * support},
            )
        )
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5, glue_coefs={"SHL": 4.0e-5})}
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures + make_glue_driver_fixtures(),
        models=models,
        config=base_config(glue_enabled=True),
        seed=20,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_by_test = _read(out_dir, "glue_opcodes_by_test.csv")
    shl = glue_by_test[glue_by_test["glue_opcode"] == "SHL"].iloc[0]
    assert not bool(shl["ratio_reliable"])

    new_gas_all = _read(out_dir, "new_gas_all_params.csv")
    add_row = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"].iloc[0]
    assert "SHL" in add_row["glue_unpriced_opcodes"]
    assert not bool(add_row["glue_coverage_complete"])
    assert bool(pd.isna(add_row["new_gas_rounded"]))


# ----- Pure POP anchor and per-driver fixed effects ------------------------


def test_pure_pop_anchor_subtracts_paired_grower_support(tmp_path: Path) -> None:
    """A dedicated constant-seed POP driver identifies POP alone; the push
    growers' 1:1 POP background is then charged to POP's coefficient, not
    absorbed into PUSH's."""
    drivers = []
    for fixture in make_glue_driver_fixtures():
        if fixture.test_name == "test_push_straight":
            fixture.extra_opcounts["POP"] = fixture.target_opcount
        drivers.append(fixture)
    target_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    models = {
        "geth": ClientModel(
            intercept=10.0,
            slope=0.0,
            glue_coefs={
                **{f"PUSH{i}": 3.0e-5 for i in range(33)},
                "POP": 4.0e-5,
                "ADD": 2.0e-5,
            },
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=drivers + target_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=22,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_results = _read(out_dir, "glue_results.csv")
    push_row = glue_results[glue_results["glue_opcode"] == "PUSH"].iloc[0]
    assert bool(push_row["isolated"])
    # POP background was subtracted via the pure anchor fit.
    assert float(push_row["glue_runtime_ms"]) == pytest.approx(3.0e-5, rel=0.15)
    pop_row = glue_results[glue_results["glue_opcode"] == "POP"].iloc[0]
    assert bool(pop_row["isolated"])
    assert float(pop_row["glue_runtime_ms"]) == pytest.approx(4.0e-5, rel=0.15)


def test_cycle_fit_fixed_effects_absorb_constant_setup(tmp_path: Path) -> None:
    """Constant per-driver setup (here a fixed 777-JUMPDEST seed on the DUP
    driver only) must load on that driver's fixed effect, not on any
    coefficient: within-driver demeaning identifies slopes from each
    driver's own sweep."""
    drivers = []
    for fixture in make_glue_driver_fixtures():
        if fixture.test_name == "test_dup_straight":
            fixture.extra_opcounts["JUMPDEST"] = 777.0  # constant seed
        drivers.append(fixture)
    target_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    models = {
        "geth": ClientModel(
            intercept=10.0,
            slope=0.0,
            glue_coefs={
                **{f"DUP{i}": 3.0e-5 for i in range(1, 17)},
                "JUMPDEST": 5.0e-5,
                "ADD": 2.0e-5,
            },
        )
    }
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=drivers + target_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=23,
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    glue_results = _read(out_dir, "glue_results.csv")
    dup_row = glue_results[glue_results["glue_opcode"] == "DUP"].iloc[0]
    assert bool(dup_row["isolated"])
    assert float(dup_row["glue_runtime_ms"]) == pytest.approx(3.0e-5, rel=0.15)
