"""End-to-end: R² and p-value thresholds drive the Poor-fit section.

Two specs race for the same gas param (``OPCODE_ADD``) on the same target
opcode (``ADD``) but under different ``test_name`` labels. Multiple clients
get distinct noise budgets so the selector sees:

- one client where both candidates pass thresholds (no surfacing);
- one client where one candidate passes and the other fails R² (the failing
  candidate must appear under ``### Other weak candidates``);
- one client where both candidates fail R² (the fallback winner gets
  ``poor_fit = True`` and surfaces under ``### Winners with poor fit``).

Pins: §4.6 broadened selector (qualified pool requires `pvalue < pv_thresh`
AND `rsquared >= r2_thresh`); ``poor_fit`` is `True` iff the selector fell
back; ``new_gas_all_params.csv`` carries ``rsquared`` / ``rsquared_adj``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from _data_synth import (
    ClientModel,
    base_config,
    make_block_limit_fixtures,
    run_pipeline,
    write_config_yaml,
    write_opcounts_json,
    write_standard_inputs,
)


def _generate_runtimes(
    runtimes_csv: Path,
    fixtures_per_test: dict[str, list],
    models: dict[str, ClientModel],
    noise_table: dict[tuple[str, str], float],
    *,
    seed: int = 42,
) -> None:
    """Write a runtimes CSV with per-(test, client) noise.

    ``noise_table[(test_name, client_name)]`` is the multiplicative noise
    sigma applied to that combination's runtimes. Missing entries default to
    0.003 (matching ``_data_synth.write_runtimes_csv`` default).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for client_name, model in models.items():
        for test_name, fixtures in fixtures_per_test.items():
            noise = noise_table.get((test_name, client_name), 0.003)
            for spec in fixtures:
                val = model.intercept + model.slope * spec.target_opcount
                if noise > 0:
                    val *= 1.0 + rng.normal(0.0, noise)
                rows.append(
                    {
                        "client_name": client_name,
                        "fixture_name": spec.fixture_name,
                        "test_runtime_ms": float(val),
                    }
                )
    pd.DataFrame(rows).to_csv(runtimes_csv, index=False)


def _write_inputs(tmp_path: Path):
    """Materialize the three input files for the poor-fit scenario."""
    fixtures_a = make_block_limit_fixtures(
        test_file="test_a",
        test_name="test_a",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    fixtures_b = make_block_limit_fixtures(
        test_file="test_b",
        test_name="test_b",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    fixtures_per_test = {"test_a": fixtures_a, "test_b": fixtures_b}

    models = {
        "alpha": ClientModel(intercept=80.0, slope=1.0e-5),
        "beta": ClientModel(intercept=90.0, slope=1.2e-5),
        "gamma": ClientModel(intercept=100.0, slope=1.5e-5),
    }
    # noise: 0.6 reliably pushes R² below 0.5 for this block-limit grid (the
    # multiplicative noise stays correlated with the planted slope, so the
    # threshold has to overcome ~σ²·E[x²]/Var(x) ≈ 5σ² of residual fraction);
    # 0.003 keeps R² > 0.99.
    noise_table: dict[tuple[str, str], float] = {
        ("test_a", "alpha"): 0.003,
        ("test_b", "alpha"): 0.003,
        ("test_a", "beta"): 0.003,
        ("test_b", "beta"): 0.6,
        ("test_a", "gamma"): 0.6,
        ("test_b", "gamma"): 0.6,
    }

    config = base_config(
        models_custom=[
            {
                "test_name": "test_a",
                "target_operation": "ADD",
                "model_params": {"target_coef": "OPCODE_ADD"},
            },
            {
                "test_name": "test_b",
                "target_operation": "ADD",
                "model_params": {"target_coef": "OPCODE_ADD"},
            },
        ],
        clients=tuple(models.keys()),
    )

    config_yaml = tmp_path / "config.yaml"
    runtimes_csv = tmp_path / "runtimes.csv"
    opcounts_json = tmp_path / "opcounts.json"
    out_dir = tmp_path / "out"
    write_config_yaml(config_yaml, config)
    write_opcounts_json(opcounts_json, fixtures_a + fixtures_b)
    _generate_runtimes(runtimes_csv, fixtures_per_test, models, noise_table, seed=17)
    return config_yaml, runtimes_csv, opcounts_json, out_dir


def test_poor_fit_section_surfaces_winners_and_losing_candidates(
    tmp_path: Path,
) -> None:
    config_yaml, runtimes_csv, opcounts_json, out_dir = _write_inputs(tmp_path)
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    new_gas_all = pd.read_csv(out_dir / "new_gas_all_params.csv")
    assert "rsquared" in new_gas_all.columns
    assert "rsquared_adj" in new_gas_all.columns

    # Sanity: the noise table planted should produce the expected R² shape.
    add_rows = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"]
    assert {"alpha", "beta", "gamma"}.issubset(set(add_rows["client_name"]))

    # gamma's per-client winner falls back to the unfiltered pool, so it
    # carries ``poor_fit = True``; alpha (both clean) and beta (one clean,
    # one noisy) both have a qualified candidate, so neither winner is flagged.
    # ``poor_fit`` now flags every failing candidate, so restrict to winners.
    poor_winners = add_rows[
        (add_rows["poor_fit"].eq(True)) & (add_rows["is_winner"].eq(True))
    ]
    assert set(poor_winners["client_name"]) == {"gamma"}
    # The winning row's R² should reflect the actual noisy fit (well below the
    # 0.5 threshold) rather than a clean fit accidentally carried over.
    assert poor_winners.iloc[0]["rsquared"] < 0.5

    # Read the proposal markdown and check both subsections render.
    proposal = (out_dir / "new_gas_proposal.md").read_text()
    assert "## Poor-fit selections" in proposal
    # The intro paragraph names the subsection headers inline (in backticks)
    # to orient the reader, so anchor on the actual line-leading heading.
    win_match = re.search(r"^### Winners with poor fit$", proposal, re.MULTILINE)
    los_match = re.search(r"^### Other weak candidates$", proposal, re.MULTILINE)
    assert win_match and los_match
    assert win_match.start() < los_match.start(), "winners must precede losers"

    winners_block = proposal[win_match.start() : los_match.start()]
    losers_block = proposal[los_match.start() :]

    # Winners block must mention gamma and not the clean clients.
    assert "gamma" in winners_block
    # The Failed cell on the winners table should carry an R²-related label
    # because gamma's noise lives in the R² dimension, not p-value.
    assert "R²" in winners_block or "both" in winners_block

    # Losers block must include beta's noisy losing candidate (test_b on beta
    # failed R², lost to test_a on beta) and gamma's losing candidate.
    assert "beta" in losers_block
    assert "gamma" in losers_block
    assert "_None._" not in losers_block.split("##", 2)[0]

    # The losers section renders one ``<details>`` block per affected
    # gas_param with the new ``Combo`` column. Both specs in this test
    # carry no ``model_by`` fields, so the combo cell collapses to ``—``.
    assert "<summary><code>OPCODE_ADD</code>" in losers_block
    assert "| Test | Target opcode | Coef | Combo | Failing clients |" in losers_block
    assert "| — |" in losers_block


def test_poor_fit_section_renders_none_when_all_fits_clean(tmp_path: Path) -> None:
    """A clean run leaves both Poor-fit subsections at ``_None._``."""
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
    )
    models = {
        "geth": ClientModel(intercept=80.0, slope=1.0e-5),
        "besu": ClientModel(intercept=100.0, slope=1.5e-5),
    }
    config = base_config()
    config_yaml, runtimes_csv, opcounts_json, out_dir = write_standard_inputs(
        tmp_path, fixtures=fixtures, models=models, config=config
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    proposal = (out_dir / "new_gas_proposal.md").read_text()
    win_match = re.search(r"^### Winners with poor fit$", proposal, re.MULTILINE)
    los_match = re.search(r"^### Other weak candidates$", proposal, re.MULTILINE)
    assert win_match and los_match
    # Both subsections collapse to the ``_None._`` placeholder.
    assert "_None._" in proposal[win_match.start() : los_match.start()]
    assert "_None._" in proposal[los_match.start() :]
