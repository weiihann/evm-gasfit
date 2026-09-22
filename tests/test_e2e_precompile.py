"""End-to-end: precompile specs disambiguated by synthetic target_operation.

Precompiles have no dedicated opcode mnemonic in ``opcounts.json``. A worker
records the actual destination invocation under
``PRECOMPILE_0x<20-byte-address>``; wrapper ``STATICCALL`` totals are distinct
execution work and must not stand in for that semantic count. A spec keeps the
human-readable ``target_operation`` and points
``target_operation_count_source`` at the address-specific counter.

This test exercises two precompile specs that share ``test_name`` and
``model_by`` shape but use separate semantic counters. Each variant is
synthesized with a different true slope; if the aggregator were unable to
distinguish the specs, both ``PRECOMPILE_*`` rows in ``new_gas.csv`` would
collapse to the larger slope. The test asserts they don't.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    FixtureSpec,
    base_config,
    run_pipeline,
    runtime_for,
    write_config_yaml,
    write_opcounts_json,
)


def _precompile_fixtures(
    *,
    target_opcode: str,
    variant_token: str,
    count_source: str,
    block_limits: Sequence[int] = (30, 60, 90, 120, 150, 180, 210, 240),
    target_opcount_per_million: float = 100_000.0,
) -> list[FixtureSpec]:
    """Build BLS-style fixtures with a semantic precompile count."""
    key, _, value = variant_token.partition("_")
    fixtures: list[FixtureSpec] = []
    for bl in block_limits:
        fixtures.append(
            FixtureSpec(
                test_file="test_bls12_381",
                test_name="test_bls12_381",
                params={key: value},
                block_limit_million=bl,
                target_opcode=target_opcode,
                target_opcount=bl * target_opcount_per_million,
                count_source_opcode=count_source,
                omit_opcode_token=True,
            )
        )
    return fixtures


def test_precompile_count_source_isolates_specs(tmp_path: Path) -> None:
    g1add_fixtures = _precompile_fixtures(
        target_opcode="BLS12_G1ADD",
        variant_token="bls12_g1add",
        count_source="PRECOMPILE_0x000000000000000000000000000000000000000b",
    )
    g2add_fixtures = _precompile_fixtures(
        target_opcode="BLS12_G2ADD",
        variant_token="bls12_g2add",
        count_source="PRECOMPILE_0x000000000000000000000000000000000000000d",
    )

    g1add_slope = 1.0e-5
    g2add_slope = 3.0e-5
    intercept = 100.0
    g1add_model = ClientModel(intercept=intercept, slope=g1add_slope)
    g2add_model = ClientModel(intercept=intercept, slope=g2add_slope)

    runtimes_csv = tmp_path / "runtimes.csv"
    opcounts_json = tmp_path / "opcounts.json"
    config_yaml = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"

    rng = np.random.default_rng(0)
    rows = []
    for variant_fixtures, model in (
        (g1add_fixtures, g1add_model),
        (g2add_fixtures, g2add_model),
    ):
        for spec in variant_fixtures:
            rows.append(
                {
                    "client_name": "geth",
                    "fixture_name": spec.fixture_name,
                    "test_runtime_ms": runtime_for(spec, model, rng, noise_pct=0.003),
                }
            )
    pd.DataFrame(rows).to_csv(runtimes_csv, index=False)

    write_opcounts_json(opcounts_json, g1add_fixtures + g2add_fixtures)

    config = base_config(
        models_custom=[
            {
                "test_name": "test_bls12_381",
                "target_operation": "BLS12_G1ADD",
                "target_operation_count_source": "PRECOMPILE_0x000000000000000000000000000000000000000b",
                "filter_by": ["bls12_g1add"],
                "model_params": {"target_coef": "PRECOMPILE_BLS12_G1ADD"},
            },
            {
                "test_name": "test_bls12_381",
                "target_operation": "BLS12_G2ADD",
                "target_operation_count_source": "PRECOMPILE_0x000000000000000000000000000000000000000d",
                "filter_by": ["bls12_g2add"],
                "model_params": {"target_coef": "PRECOMPILE_BLS12_G2ADD"},
            },
        ],
        new_params={
            "PRECOMPILE_BLS12_G1ADD": None,
            "PRECOMPILE_BLS12_G2ADD": None,
        },
        clients=("geth",),
    )
    write_config_yaml(config_yaml, config)

    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    # ---- results.csv: one row per spec, target_opcode is the display name --
    results = pd.read_csv(out_dir / "results.csv")
    assert len(results) == 2
    assert set(results["target_opcode"]) == {"BLS12_G1ADD", "BLS12_G2ADD"}

    g1 = results[results["target_opcode"] == "BLS12_G1ADD"].iloc[0]
    g2 = results[results["target_opcode"] == "BLS12_G2ADD"].iloc[0]
    assert float(g1["target_coef_runtime_ms"]) == pytest.approx(g1add_slope, rel=0.05)
    assert float(g2["target_coef_runtime_ms"]) == pytest.approx(g2add_slope, rel=0.05)

    # ---- new_gas.csv: each gas_param keeps its own slope ------------------
    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    gas_params = set(new_gas["gas_param"])
    assert {"PRECOMPILE_BLS12_G1ADD", "PRECOMPILE_BLS12_G2ADD"}.issubset(gas_params)

    g1_gas = new_gas[new_gas["gas_param"] == "PRECOMPILE_BLS12_G1ADD"].iloc[0]
    g2_gas = new_gas[new_gas["gas_param"] == "PRECOMPILE_BLS12_G2ADD"].iloc[0]
    assert float(g1_gas["runtime_ms"]) == pytest.approx(g1add_slope, rel=0.05)
    assert float(g2_gas["runtime_ms"]) == pytest.approx(g2add_slope, rel=0.05)
    # The visible cross-contamination check: source labels and semantic
    # target counters keep the two variants isolated.
    assert float(g1_gas["runtime_ms"]) < float(g2_gas["runtime_ms"])
    assert g1_gas["selected_opcode"] == "BLS12_G1ADD"
    assert g2_gas["selected_opcode"] == "BLS12_G2ADD"
