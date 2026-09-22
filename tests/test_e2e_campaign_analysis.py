"""End-to-end coverage for campaign-aware analysis and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evm_gasfit import GasFit
from evm_gasfit.campaign import compare_campaigns
from evm_gasfit.errors import ConfigError
from evm_gasfit.provenance import sha256_file
from _data_synth import (
    RESULTS_COLUMNS,
    ClientModel,
    base_config,
    make_block_limit_fixtures,
    write_config_yaml,
    write_standard_inputs,
)


def _manifest(*, hardware: str = "r7a.4xlarge") -> dict[str, object]:
    return {
        "software": {"revision": "candidate"},
        "hardware": {"instance": hardware},
        "workloads": {"corpus": "osaka-compute-v2"},
        "boundary": "evm2_transaction_execution",
        "gas_schedule": "osaka",
    }


def _write_comparison_input(
    directory: Path,
    manifest: dict[str, object],
    row: dict[str, object],
    *,
    config_hash: str = "analysis-config",
    version: str = "test-version",
) -> None:
    new_gas = directory / "new_gas.csv"
    pd.DataFrame([row]).to_csv(new_gas, index=False)
    (directory / "analysis_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evm_gasfit_version": version,
                "inputs": {"config": {"sha256": config_hash}},
                "manifest": {"content": manifest, "sha256": "manifest"},
                "outputs": {"new_gas.csv": sha256_file(new_gas)},
            }
        )
    )


def _run_campaign(
    config_path: Path,
    runtimes_path: Path,
    opcounts_path: Path,
    manifest_path: Path,
    out_dir: Path,
) -> None:
    fit = GasFit.from_config(config_path)
    fit.load_runtimes(runtimes_path)
    fit.load_opcounts(opcounts_path)
    fit.load_manifest(manifest_path)
    fit.estimate_models()
    fit.build_proposal()
    fit.write_reports(out_dir)


def test_campaign_eligibility_session_inference_and_anchorless_outputs(
    tmp_path: Path,
) -> None:
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(
        anchor_rate=None,
        clients=("geth",),
        extra={
            "campaign": {
                "eligible_phases": ["qualification", "performance"],
                "eligible_statuses": ["executed"],
                "require_correctness_passed": True,
            },
            "qualification": {"min_sessions": 2, "block_unqualified": True},
        },
    )
    config_path, runtimes_path, opcounts_path, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )

    runtimes = pd.read_csv(runtimes_path)
    first = runtimes.assign(
        session_id="session-a",
        sample_id=lambda frame: [f"a-{i}" for i in range(len(frame))],
        phase="qualification",
        status="executed",
        correctness_passed=True,
        repetition=0,
        operator_note=7,
    )
    second = first.assign(
        session_id="session-b",
        sample_id=lambda frame: [f"b-{i}" for i in range(len(frame))],
        repetition=1,
    )
    excluded = first.iloc[[0]].assign(
        sample_id="diagnostic-row",
        phase="diagnostic",
        correctness_passed=False,
    )
    incorrect = first.iloc[[0]].assign(
        sample_id="incorrect-row",
        correctness_passed=False,
    )
    pilot = first.iloc[[0]].assign(sample_id="pilot-row", phase="pilot")
    warmup = first.iloc[[0]].assign(sample_id="warmup-row", phase="warmup")
    failed = first.iloc[[0]].assign(sample_id="failed-row", status="failed")
    pd.concat(
        [first, second, excluded, incorrect, pilot, warmup, failed],
        ignore_index=True,
    ).to_csv(runtimes_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))

    _run_campaign(config_path, runtimes_path, opcounts_path, manifest_path, out_dir)

    eligibility = pd.read_csv(out_dir / "eligibility.csv")
    assert len(eligibility) == len(first) * 2 + 5
    assert eligibility["eligible"].sum() == len(first) * 2
    excluded_samples = {
        "diagnostic-row",
        "incorrect-row",
        "pilot-row",
        "warmup-row",
        "failed-row",
    }
    assert not eligibility.loc[
        eligibility["sample_id"].isin(excluded_samples), "eligible"
    ].any()
    assert any(
        "not in eligible_phases" in reason for reason in eligibility["reason"].dropna()
    )

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert set(qualification["status"]) == {"qualified"}
    assert set(qualification["n_sessions"]) == {2}

    proposal = pd.read_csv(out_dir / "new_gas.csv")
    assert proposal["new_gas_decimal"].isna().all()
    assert proposal["new_gas_rounded"].isna().all()

    status = json.loads((out_dir / "analysis_status.json").read_text())
    assert status["inputs"]["config"]["content"] == config_path.read_text()
    assert status["manifest"]["content"] == _manifest()


@pytest.mark.parametrize(
    "campaign",
    [
        {"eligible_phases": ["diagnostic"]},
        {"eligible_phases": ["pilot"]},
        {"eligible_phases": ["warmup"]},
        {"eligible_statuses": ["failed"]},
        {"require_correctness_passed": False},
    ],
)
def test_campaign_config_rejects_ineligible_calibration_overrides(
    tmp_path: Path, campaign: dict[str, object]
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config_yaml(config_path, base_config(extra={"campaign": campaign}))

    with pytest.raises(ConfigError):
        GasFit.from_config(config_path)


def test_campaign_missing_correctness_evidence_fails_closed(tmp_path: Path) -> None:
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(clients=("geth",))
    config_path, runtimes_path, opcounts_path, _ = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )
    runtimes = pd.read_csv(runtimes_path).assign(
        session_id="session-a",
        phase="qualification",
        status="executed",
    )
    runtimes.to_csv(runtimes_path, index=False)

    fit = GasFit.from_config(config_path)
    fit.load_runtimes(runtimes_path)
    fit.load_opcounts(opcounts_path)

    with pytest.raises(ConfigError):
        fit.estimate_models()


def test_campaign_with_no_eligible_rows_writes_inconclusive_analysis(
    tmp_path: Path,
) -> None:
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(
        clients=("geth",),
        models_custom=[
            {
                "test_name": "test_arithmetic",
                "target_operation": "ADD",
                "model_params": {"target_coef": "OPCODE_ADD"},
            },
            {
                "test_name": "test_bitwise",
                "target_operation": "OR",
                "model_params": {"target_coef": "OPCODE_OR"},
            },
        ],
    )
    config_path, runtimes_path, opcounts_path, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )
    runtimes = pd.read_csv(runtimes_path).assign(
        test_runtime_ms=float("nan"),
        session_id="failed-session",
        sample_id=lambda frame: [f"failed-{i}" for i in range(len(frame))],
        phase="qualification",
        status="failed",
        correctness_passed=True,
    )
    runtimes.to_csv(runtimes_path, index=False)
    opcounts_path.write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))

    _run_campaign(config_path, runtimes_path, opcounts_path, manifest_path, out_dir)

    eligibility = pd.read_csv(out_dir / "eligibility.csv")
    assert len(eligibility) == len(runtimes)
    assert not eligibility["eligible"].any()
    assert eligibility["reason"].str.contains("not in eligible_statuses").all()

    results = pd.read_csv(out_dir / "results.csv")
    assert results.empty
    assert RESULTS_COLUMNS <= set(results.columns)

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert set(qualification["test_name"]) == {"test_arithmetic", "test_bitwise"}
    assert set(qualification["status"]) == {"inconclusive"}

    proposal = pd.read_csv(out_dir / "new_gas.csv")
    assert set(proposal["gas_param"]) == {"OPCODE_ADD", "OPCODE_OR"}
    assert proposal["new_gas_rounded"].isna().all()
    assert set(proposal["selected_test"]) == {"<no-fit>"}

    status = json.loads((out_dir / "analysis_status.json").read_text())
    assert len(status["planned_models"]) == 2
    assert {record["status"] for record in status["planned_models"]} == {"inconclusive"}


def test_single_session_campaign_is_inconclusive_without_row_bootstrap(
    tmp_path: Path,
) -> None:
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(
        clients=("geth",),
        extra={"campaign": {"require_correctness_passed": True}},
    )
    config_path, runtimes_path, opcounts_path, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )
    runtimes = pd.read_csv(runtimes_path).assign(
        session_id="only-session",
        sample_id=lambda frame: [f"sample-{i}" for i in range(len(frame))],
        phase="qualification",
        status="executed",
        correctness_passed=True,
    )
    runtimes.to_csv(runtimes_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))

    _run_campaign(config_path, runtimes_path, opcounts_path, manifest_path, out_dir)

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert set(qualification["status"]) == {"inconclusive"}
    assert "fewer than two sessions" in qualification.iloc[0]["reasons"]
    assert pd.read_csv(out_dir / "new_gas.csv")["new_gas_rounded"].isna().all()


def test_missing_configured_client_is_recorded_as_inconclusive(tmp_path: Path) -> None:
    """Configured clients without measurements remain visible in qualification."""
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(clients=("geth", "besu"))
    config_path, runtimes_path, opcounts_path, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )
    write_config_yaml(config_path, config)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))

    _run_campaign(config_path, runtimes_path, opcounts_path, manifest_path, out_dir)

    qualification = pd.read_csv(out_dir / "qualification.csv")
    missing = qualification[qualification["client_name"] == "besu"].iloc[0]
    assert missing["status"] == "inconclusive"
    assert "no eligible fixtures for configured client" in missing["reasons"]


def test_unavailable_campaign_curvature_is_inconclusive(tmp_path: Path) -> None:
    """Collapsed fitted values cannot satisfy the mandatory curvature gate."""
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config(
        clients=("geth",),
        extra={"qualification": {"min_sessions": 2}},
    )
    config_path, runtimes_path, opcounts_path, out_dir = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=0.0)},
        config=config,
        noise_pct=0.0,
    )
    runtimes = pd.read_csv(runtimes_path)
    first = runtimes.assign(session_id="session-a")
    second = runtimes.assign(session_id="session-b")
    pd.concat([first, second], ignore_index=True).to_csv(runtimes_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))

    _run_campaign(config_path, runtimes_path, opcounts_path, manifest_path, out_dir)

    qualification = pd.read_csv(out_dir / "qualification.csv")
    assert qualification.iloc[0]["status"] == "inconclusive"
    assert "residual curvature is not computable" in qualification.iloc[0]["reasons"]


def test_campaign_comparison_allows_software_only_difference(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for directory, revision in ((baseline, "before"), (candidate, "after")):
        directory.mkdir()
        manifest = _manifest()
        manifest["software"] = {"revision": revision}
        _write_comparison_input(
            directory,
            manifest,
            {
                "gas_param": "OPCODE_ADD",
                "runtime_ms": 1.0,
                "new_gas_rounded": 100,
                "qualification_status": "qualified",
            },
        )

    summary = compare_campaigns(baseline, candidate, tmp_path / "comparison")

    assert summary["software_changed"] is True
    assert summary["changed_factors"] == []
    assert (tmp_path / "comparison" / "campaign_comparison.csv").exists()


def test_campaign_comparison_rejects_empty_manifest_factor(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for directory, hardware in (
        (baseline, {"instance": "r7a.4xlarge"}),
        (candidate, {}),
    ):
        directory.mkdir()
        manifest = _manifest()
        manifest["hardware"] = hardware
        _write_comparison_input(
            directory,
            manifest,
            {
                "gas_param": "OPCODE_ADD",
                "runtime_ms": 1.0,
                "qualification_status": "qualified",
            },
        )

    comparison = tmp_path / "comparison"
    with pytest.raises(ConfigError, match="provenance incomplete"):
        compare_campaigns(baseline, candidate, comparison)
    assert not comparison.exists()


def test_campaign_comparison_rejects_unavailable_schedule_identity(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for directory, revision in ((baseline, "before"), (candidate, "after")):
        directory.mkdir()
        manifest = _manifest()
        manifest["comparison_factors"] = {
            "hardware": manifest["hardware"],
            "workload": "osaka-compute-v2",
            "execution_boundary": "evm2_transaction_execution",
            "gas_schedule": {
                "active_schedule_identity": {
                    "status": "unavailable",
                    "reason": "evm2 worker protocol does not expose a schedule",
                }
            },
        }
        _write_comparison_input(
            directory,
            manifest,
            {
                "gas_param": "OPCODE_ADD",
                "runtime_ms": 1.0,
                "qualification_status": "qualified",
            },
        )

    comparison = tmp_path / "comparison"
    with pytest.raises(ConfigError, match="provenance incomplete"):
        compare_campaigns(baseline, candidate, comparison)
    assert not comparison.exists()


@pytest.mark.parametrize(
    ("candidate_config_hash", "candidate_version"),
    [("other-config", "test-version"), ("analysis-config", "other-version")],
)
def test_campaign_comparison_rejects_incompatible_analysis_method(
    tmp_path: Path,
    candidate_config_hash: str,
    candidate_version: str,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    row = {
        "gas_param": "OPCODE_ADD",
        "runtime_ms": 1.0,
        "qualification_status": "qualified",
    }
    _write_comparison_input(baseline, _manifest(), row)
    _write_comparison_input(
        candidate,
        _manifest(),
        row,
        config_hash=candidate_config_hash,
        version=candidate_version,
    )

    comparison = tmp_path / "comparison"
    with pytest.raises(
        ConfigError, match="analysis config or analyzer version differs"
    ):
        compare_campaigns(baseline, candidate, comparison)
    assert not comparison.exists()


def test_campaign_comparison_rejects_unverified_new_gas_csv(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    row = {
        "gas_param": "OPCODE_ADD",
        "runtime_ms": 1.0,
        "qualification_status": "qualified",
    }
    _write_comparison_input(baseline, _manifest(), row)
    _write_comparison_input(candidate, _manifest(), row)
    pd.DataFrame([{**row, "runtime_ms": 2.0}]).to_csv(
        candidate / "new_gas.csv", index=False
    )

    comparison = tmp_path / "comparison"
    with pytest.raises(ConfigError, match="new_gas.csv hash differs"):
        compare_campaigns(baseline, candidate, comparison)
    assert not comparison.exists()


def test_campaign_comparison_rejects_hardware_mismatch_before_output(
    tmp_path: Path,
) -> None:
    fixtures = make_block_limit_fixtures(
        test_file="test_arithmetic",
        test_name="test_arithmetic",
        target_opcode="ADD",
    )
    config = base_config()
    config_path, runtimes_path, opcounts_path, _ = write_standard_inputs(
        tmp_path,
        fixtures=fixtures,
        models={"geth": ClientModel(intercept=1.0, slope=2.0)},
        config=config,
        noise_pct=0.0,
    )

    baseline_manifest = tmp_path / "baseline-manifest.json"
    candidate_manifest = tmp_path / "candidate-manifest.json"
    baseline_manifest.write_text(json.dumps(_manifest()))
    candidate_manifest.write_text(json.dumps(_manifest(hardware="r7a.8xlarge")))
    baseline_out = tmp_path / "baseline"
    candidate_out = tmp_path / "candidate"
    _run_campaign(
        config_path,
        runtimes_path,
        opcounts_path,
        baseline_manifest,
        baseline_out,
    )
    _run_campaign(
        config_path,
        runtimes_path,
        opcounts_path,
        candidate_manifest,
        candidate_out,
    )

    comparison_out = tmp_path / "comparison"
    with pytest.raises(ConfigError, match="hardware differs"):
        compare_campaigns(baseline_out, candidate_out, comparison_out)
    assert not comparison_out.exists()
