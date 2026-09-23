"""End-to-end: sentinel rendering on the proposal + missing-glue warnings.

Two rules share this surface because both hinge on proposal rendering:

- **"no prior default" sentinel**: a `new_params` entry declared with a
  `null` value renders the `no prior default` sentinel in the proposal diff
  column rather than a misleading numeric zero. A raw fork-field row must
  NOT carry the sentinel (regression guard).
- **Missing-glue detection**: glue detection inspects every fitted model's
  group for opcodes that meet the corr/ratio thresholds; an opcode that
  meets the thresholds but is outside the priced set triggers a warning
  and surfaces in the proposal.

The sentinel path is parametrized over two distinct gas-param names
(``BRAND_NEW_GAS_PARAM`` and ``OPCODE_NEVER_EXISTED_GAS``) so each name
independently catches a regression.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    assert_sentinel_near,
    base_config,
    make_block_limit_fixtures,
    make_glue_driver_fixtures,
    run_pipeline,
    runtime_for,
    write_config_yaml,
    write_opcounts_json,
    write_standard_inputs,
)

SENTINEL = "no prior default"
UNKNOWN_PARAMS = ("BRAND_NEW_GAS_PARAM", "OPCODE_NEVER_EXISTED_GAS")
KNOWN_PARAM = "OPCODE_ADD"

# ----- new_params declared without a value renders the sentinel -----------


def _build_new_param_inputs(tmp_path: Path, gas_param: str):
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
    config = base_config(
        models_custom=[
            {
                "test_name": "test_arithmetic",
                "target_operation": "ADD",
                "model_params": {"target_coef": gas_param},
            }
        ],
        new_params=({gas_param: None} if gas_param not in {"OPCODE_ADD"} else None),
    )
    return write_standard_inputs(
        tmp_path, fixtures=fixtures, models=models, config=config, seed=17
    )


@pytest.mark.parametrize("gas_param", UNKNOWN_PARAMS)
def test_declared_new_param_renders_sentinel(tmp_path: Path, gas_param: str) -> None:
    """A `new_params` entry with `null` value: the fit succeeds, the name
    appears in `new_gas.csv`, and the proposal diff column carries the
    `no prior default` sentinel co-located with the param (not a fabricated
    numeric zero)."""
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_new_param_inputs(
        tmp_path, gas_param
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    # ---- row in new_gas.csv --------------------------------------------
    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    assert gas_param in set(new_gas["gas_param"]), (
        f"{gas_param} missing from new_gas.csv gas_param column"
    )

    # ---- "no prior default" sentinel co-located with the new name ------
    proposal = (out_dir / "new_gas_proposal.md").read_text()
    assert_sentinel_near(proposal, gas_param, SENTINEL)

    # ---- no isolated `| 0 |` markdown cell that would imply prior == 0 -
    md_lines = [line for line in proposal.splitlines() if gas_param in line]
    isolated_zero = re.compile(r"\|\s*0\s*\|")
    for line in md_lines:
        assert not isolated_zero.search(line), (
            f"markdown row for {gas_param} renders an isolated `0` table cell:\n{line}"
        )


def test_known_gas_param_does_not_carry_sentinel(tmp_path: Path) -> None:
    """A raw fork-field gas-param must not carry the sentinel — guards
    against a regression where the sentinel leaks into every row."""
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_new_param_inputs(
        tmp_path, KNOWN_PARAM
    )
    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir)

    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    assert KNOWN_PARAM in set(new_gas["gas_param"]), (
        f"{KNOWN_PARAM} missing from new_gas.csv"
    )

    proposal = (out_dir / "new_gas_proposal.md").read_text()
    md_lines = "\n".join(line for line in proposal.splitlines() if KNOWN_PARAM in line)
    assert SENTINEL.lower() not in md_lines.lower(), (
        f"sentinel {SENTINEL!r} leaked into a known-default row for {KNOWN_PARAM}"
    )


# ----- missing-glue detection --------------------------------------------


# A clearly non-glue opcode: arithmetic, three operands, never used as a stitch.
NON_PRICED_OPCODE = "ADDMOD"


def _build_contaminant_inputs(
    tmp_path: Path, *, contaminant: str, contaminant_per_million: float = 500_000
):
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={contaminant: contaminant_per_million},
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {"geth": ClientModel(intercept=50.0, slope=2.0e-5)}
    return write_standard_inputs(
        tmp_path,
        fixtures=all_fixtures,
        models=models,
        config=base_config(glue_enabled=True),
        seed=7,
    )


def test_non_priced_glue_candidate_emits_warning_and_appears_in_proposal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from evm_gasfit.glue.required import PRICED_GLUE_OPCODES

    priced = set(PRICED_GLUE_OPCODES)
    assert NON_PRICED_OPCODE not in priced, (
        f"{NON_PRICED_OPCODE} unexpectedly priced — adjust the test contaminant"
    )

    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_contaminant_inputs(
        tmp_path, contaminant=NON_PRICED_OPCODE
    )

    with caplog.at_level(logging.WARNING, logger="evm_gasfit"):
        run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    pattern = re.compile(rf"\b{re.escape(NON_PRICED_OPCODE)}\b")
    matching = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name.startswith("evm_gasfit")
        and pattern.search(r.getMessage())
    ]
    assert matching, (
        f"expected a WARNING on logger 'evm_gasfit' naming {NON_PRICED_OPCODE!r}; "
        f"got {[(r.name, r.levelname, r.getMessage()) for r in caplog.records]}"
    )

    proposal = (out_dir / "new_gas_proposal.md").read_text()
    assert NON_PRICED_OPCODE in proposal, (
        f"{NON_PRICED_OPCODE!r} expected to appear in new_gas_proposal.md"
    )


# ----- poor-fit glue sub-block on the proposal ----------------------------


def _write_per_client_runtimes(
    path: Path,
    fixtures,
    models,
    noise_table: dict[tuple[str, str], float],
    *,
    default_noise: float = 0.003,
    seed: int = 42,
) -> None:
    """Like `write_runtimes_csv` but per-(client, test_name) noise.

    ``noise_table[(test_name, client_name)]`` is the multiplicative noise
    sigma applied to that combination's runtimes. Missing entries use
    ``default_noise``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for client_name, model in models.items():
        for spec in fixtures:
            sigma = noise_table.get((spec.test_name, client_name), default_noise)
            rows.append(
                {
                    "client_name": client_name,
                    "fixture_name": spec.fixture_name,
                    "test_runtime_ms": runtime_for(spec, model, rng, sigma),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_poor_fit_glue_opcodes_surface_under_missing_glue_section(
    tmp_path: Path,
) -> None:
    """A noisy ISZERO driver fit on one client surfaces under the new
    `_Priced glue opcodes with a poor fit_` sub-block of `### Missing glue
    opcodes`, naming that client and the ADD gas param that depends on the
    ISZERO glue adjustment. Other glue opcodes (with clean fits) do not
    appear in that table.
    """
    main_fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
        params={"opcode": "ADD"},
        extra_opcount_per_million={"ISZERO": 500_000},
    )
    all_fixtures = main_fixtures + make_glue_driver_fixtures()
    models = {
        "alpha": ClientModel(intercept=80.0, slope=1.0e-5),
        "beta": ClientModel(intercept=90.0, slope=1.2e-5),
    }
    # Heavy noise on beta's ISZERO driver alone — every other (test, client)
    # combo stays clean, so only the ISZERO/beta glue fit fails R².
    noise_table = {("test_iszero_straight", "beta"): 0.6}

    config_yaml = tmp_path / "config.yaml"
    runtimes_csv = tmp_path / "runtimes.csv"
    opcounts_json = tmp_path / "opcounts.json"
    out_dir = tmp_path / "out"
    config = base_config(glue_enabled=True)
    config["clients"] = list(models.keys())
    write_config_yaml(config_yaml, config)
    write_opcounts_json(opcounts_json, all_fixtures)
    _write_per_client_runtimes(runtimes_csv, all_fixtures, models, noise_table, seed=11)

    run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    # Sanity: ISZERO did fail R² on beta and only beta.
    glue_results = pd.read_csv(out_dir / "glue_results.csv")
    iszero = glue_results[glue_results["glue_opcode"] == "ISZERO"]
    by_client = dict(zip(iszero["client_name"], iszero["rsquared"]))
    assert by_client["beta"] < 0.5, (
        f"expected beta's ISZERO fit to fail R² < 0.5, got {by_client['beta']}"
    )
    assert by_client["alpha"] >= 0.5, (
        f"expected alpha's ISZERO fit to stay above R² ≥ 0.5, got {by_client['alpha']}"
    )

    # The R² gate on compute_glue_adjustment skips beta's ISZERO contribution
    # while alpha's still applies — so alpha carries a positive
    # glue_adjustment on OPCODE_ADD and beta's is exactly zero.
    new_gas_all = pd.read_csv(out_dir / "new_gas_all_params.csv")
    add_rows = new_gas_all[new_gas_all["gas_param"] == "OPCODE_ADD"]
    by_client_adj = dict(zip(add_rows["client_name"], add_rows["glue_adjustment"]))
    assert by_client_adj["alpha"] > 0, (
        f"expected alpha's OPCODE_ADD glue_adjustment > 0, got {by_client_adj['alpha']}"
    )
    assert by_client_adj["beta"] == 0, (
        f"expected beta's OPCODE_ADD glue_adjustment to be skipped (== 0) "
        f"because the ISZERO fit failed R²; got {by_client_adj['beta']}"
    )

    proposal = (out_dir / "new_gas_proposal.md").read_text()
    # The heading still renders (no missing-glue rows in this run since the
    # contamination uses ISZERO, which is priced — so this fully exercises
    # the new sub-block on its own).
    assert "### Missing glue adjustments" in proposal
    summary_match = re.search(
        r"<summary><b>Priced glue opcodes with a poor fit</b>", proposal
    )
    assert summary_match, (
        "expected a collapsible <details> block with the priced-glue "
        "poor-fit summary under ### Missing glue adjustments"
    )
    # Slice from the <details> open tag preceding the summary to the matching
    # </details>, so per-row assertions stay inside this one block.
    details_open = proposal.rfind("<details>", 0, summary_match.start())
    assert details_open >= 0, "summary not wrapped in a <details> block"
    details_close = proposal.find("</details>", summary_match.end())
    assert details_close >= 0, "<details> block not closed"
    poor_block = proposal[details_open:details_close]
    assert "| Glue opcode | Affected clients | Affected gas params |" in poor_block

    # Pick out the ISZERO row.
    iszero_row = next(
        (
            line
            for line in poor_block.splitlines()
            if line.startswith("|") and "`ISZERO`" in line
        ),
        None,
    )
    assert iszero_row, f"no ISZERO row in poor-fit glue block:\n{poor_block}"
    assert "`beta`" in iszero_row, (
        f"expected beta to be flagged on ISZERO row: {iszero_row}"
    )
    assert "`alpha`" not in iszero_row, (
        f"alpha should not appear — its ISZERO fit is clean: {iszero_row}"
    )
    assert "R²" in iszero_row or "both" in iszero_row, (
        f"expected R² failure label on ISZERO row: {iszero_row}"
    )
    assert "`OPCODE_ADD`" in iszero_row, (
        f"expected OPCODE_ADD in affected-gas-params cell: {iszero_row}"
    )

    # Clean glue opcodes must not have a row in the poor-fit table.
    clean_names = ("`POP`", "`PUSH`", "`DUP`", "`SWAP`", "`MSTORE`")
    table_rows = [line for line in poor_block.splitlines() if line.startswith("| `")]
    for name in clean_names:
        bad = [r for r in table_rows if r.startswith(f"| {name}")]
        assert not bad, f"clean opcode {name} unexpectedly listed: {bad}"


def test_priced_glue_opcode_does_not_emit_missing_glue_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Sanity counterpart: POP is in the priced set, so even when it
    correlates perfectly with the target opcount the missing-glue warning
    does not fire."""
    priced_opcode = "POP"
    config_yaml, runtimes_csv, opcounts_json, out_dir = _build_contaminant_inputs(
        tmp_path, contaminant=priced_opcode
    )

    with caplog.at_level(logging.WARNING, logger="evm_gasfit"):
        run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, glue=True)

    pattern = re.compile(rf"\b{re.escape(priced_opcode)}\b")
    offending = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name.startswith("evm_gasfit")
        and pattern.search(r.getMessage())
    ]
    assert not offending, (
        f"unexpected WARNING(s) naming a priced glue opcode {priced_opcode!r}: "
        f"{[(r.name, r.getMessage()) for r in offending]}"
    )
