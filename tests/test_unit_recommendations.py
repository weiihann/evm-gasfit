"""Focused unit tests for :mod:`evm_gasfit.recommendations`.

These exercise the pricing catalog boundaries (exact Osaka current-charge
formulas), calibration exclusion, missing-coverage blocking, thresholds,
the keccak cache limitation, and the budget-aware lane -- the behaviors the
600M recommendation policy depends on. The module is loaded standalone so
the tests run even where the heavy analysis stack is absent.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "evm_gasfit" / "recommendations.py"
_spec = importlib.util.spec_from_file_location("evm_gasfit_recommendations", _SRC)
rec = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("evm_gasfit_recommendations", rec)
_spec.loader.exec_module(rec)

ANCHOR = 600_000_000.0


# ---------------------------------------------------------------------------
# Rounding and unit conversion.
# ---------------------------------------------------------------------------


def test_ceil_2_significant_boundaries() -> None:
    assert rec.ceil_2_significant(4.569) == pytest.approx(4.6)
    assert rec.ceil_2_significant(4.6) == pytest.approx(4.6)
    assert rec.ceil_2_significant(100.0) == 100.0
    assert rec.ceil_2_significant(101.0) == 110.0
    assert rec.ceil_2_significant(99.0) == 99.0
    assert rec.ceil_2_significant(99.0001) == 100.0
    assert rec.ceil_2_significant(0.851) == pytest.approx(0.86)
    assert rec.ceil_2_significant(0.0) == 0.0
    assert rec.ceil_2_significant(-3.0) == 0.0


def test_gas_ns_conversion_at_600m_anchor() -> None:
    assert rec.gas_from_ns(1000.0, ANCHOR) == pytest.approx(600.0)
    assert rec.ns_from_ms(0.001) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Exact current-charge formulas (Osaka sources).
# ---------------------------------------------------------------------------


def test_modexp_gas_matches_eip7883_examples() -> None:
    # base 8B, exponent 112 x 0xff, modulus 8B (test_modexp mod_even_8b):
    # complexity = 16 (max operand <= 32B), iterations = 16*(112-32) + 255.
    assert rec.modexp_gas(b"\xff" * 8, b"\xff" * 112, b"\xff" * 7 + b"\x00") == 24_560
    # 32-byte all-ones exponent: iterations = 255, complexity = 16.
    assert rec.modexp_gas(b"\xff" * 32, b"\xff" * 32, b"\xff" * 32) == 4_080
    # Zero exponent clamps iterations to 1; floor gas applies.
    assert rec.modexp_gas(b"\x01", b"", b"\x01") == 500
    # 1024B operands with a 128-byte exponent (handoff worked example):
    # 2 * 128^2 * (16*(128-32) + 255) = 32768 * 1791 = 58,687,488.
    assert rec.modexp_gas(b"\xff" * 1024, b"\xff" * 128, b"\xff" * 1024) == 58_687_488


def test_modexp_short_exponent_is_not_right_padded() -> None:
    assert rec.modexp_iterations(b"\x01") == 1
    assert rec.modexp_gas(b"\x01", b"\x01", b"\x01") == 500


def test_modexp_repr_parsing_supports_double_quoted_bytes() -> None:
    # repr switches to double quotes when the operand contains 0x27.
    operand = "base=b\"\\x01'v+{\" exponent=b'\\x03' modulus=b'\\xff'"
    parsed = rec.parse_modexp_operands(operand)
    assert parsed is not None
    base, exponent, modulus = parsed
    assert exponent == b"\x03"
    assert base.endswith(b"'v+{")
    assert modulus == b"\xff"


def test_msm_discount_schedule() -> None:
    # k = 1 pays the undiscounted mul price.
    assert rec.bls_msm_gas(1, 12_000, rec.G1_K_DISCOUNT, 519) == 12_000
    # k = 128 uses the last table entry (519 per-mille for G1).
    assert rec.bls_msm_gas(128, 12_000, rec.G1_K_DISCOUNT, 519) == 797_184
    # k beyond the table uses the max discount.
    assert (
        rec.bls_msm_gas(512, 12_000, rec.G1_K_DISCOUNT, 519)
        == 512 * 12_000 * 519 // 1000
    )
    assert len(rec.G1_K_DISCOUNT) == 128
    assert len(rec.G2_K_DISCOUNT) == 128


def test_osaka_tables_match_gasfit_defaults_when_importable() -> None:
    pytest.importorskip("evm_gasfit.defaults", reason="analysis stack absent")
    from evm_gasfit.defaults import get_gas_costs

    costs = get_gas_costs("osaka")
    for name, value in rec.OSAKA_GAS_COSTS.items():
        if name in costs.field_names:
            assert costs[name] == value, name


def test_bls_map_addresses_are_osaka_0x10_0x11() -> None:
    assert (
        rec.PRECOMPILE_ADDRESSES["BLS12_MAP_FP_TO_G1"]
        == "0x0000000000000000000000000000000000000010"
    )
    assert (
        rec.PRECOMPILE_ADDRESSES["BLS12_MAP_FP2_TO_G2"]
        == "0x0000000000000000000000000000000000000011"
    )
    assert (
        rec.PRECOMPILE_ADDRESSES["BLS12_PAIRING"]
        == "0x000000000000000000000000000000000000000f"
    )


# ---------------------------------------------------------------------------
# Workload fixtures.
# ---------------------------------------------------------------------------


def _case(
    case_id: str,
    family: str,
    operation: str | None,
    status: str = "ready",
    parameters: dict | None = None,
) -> dict:
    return {
        "id": case_id,
        "family": family,
        "target_operation": operation,
        "status": status,
        "parameters": parameters or {},
    }


def _workload(cases: list[dict]) -> dict:
    return {"schema_version": 2, "fork": "Osaka", "generator": {}, "cases": cases}


def _keccak_case(variant: str, opcount: str) -> dict:
    return _case(
        f"tests/benchmark/compute/instruction/test_keccak.py::"
        f"test_keccak_workload_witness[fork_Osaka--{variant}-{opcount}]",
        "keccak",
        "KECCAK256",
        parameters={"target_count_key": "", "source_parameters": {}},
    )


def _keccak_witness_params(input_length: int) -> dict:
    return {
        "target_count_key": "",
        "source_parameters": {"input_length": input_length},
    }


# ---------------------------------------------------------------------------
# Classification boundaries.
# ---------------------------------------------------------------------------


def test_keccak_cache_boundary_and_charges() -> None:
    cases = [
        _case(
            "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_87-opcount_1]",
            "keccak",
            "KECCAK256",
            parameters=_keccak_witness_params(87),
        )
    ]
    info = rec.classify_variant(cases[0]["id"], cases)
    assert info.current_charge_gas == 30 + 6 * 3  # 87B rounds to 3 words
    assert info.keccak_cache_affected is True

    cases[0] = _case(
        "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_136-opcount_1]",
        "keccak",
        "KECCAK256",
        parameters=_keccak_witness_params(136),
    )
    info = rec.classify_variant(cases[0]["id"], cases)
    assert info.current_charge_gas == 60
    assert info.keccak_cache_affected is False

    cases[0] = _case(
        "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_0-opcount_1]",
        "keccak",
        "KECCAK256",
        parameters=_keccak_witness_params(0),
    )
    info = rec.classify_variant(cases[0]["id"], cases)
    assert info.current_charge_gas == 30
    assert info.keccak_cache_affected is False


def test_exp_byte_length_variants() -> None:
    info = rec.classify_variant(
        "t/a.py::test_exp_bench_arithmetic[fork_Osaka--exp_136279841-base_3-opcount_1]",
        [_case("x", "arithmetic", "EXP")],
    )
    assert info.current_charge_gas is None
    assert info.units is None

    info = rec.classify_variant(
        "t/a.py::test_arithmetic[fork_Osaka--opcode_EXP-opcount_1]",
        [_case("x", "arithmetic", "EXP")],
    )
    assert info.current_charge_gas == 10 + 50 * 32  # exponent 2**256 - 1


def test_zero_charge_variants_are_reported_not_priced_by_ratio() -> None:
    info = rec.classify_variant(
        "t/b.py::test_blake2f[fork_Osaka--blake2f_zero_rounds-opcount_1]",
        [_case("x", "precompile", "BLAKE2F")],
    )
    assert info.group_kind == rec.GROUP_KIND_LINEAR
    assert info.current_charge_gas == 0

    info = rec.classify_variant(
        "t/m.py::test_modexp_length_above_upper_bound[fork_Osaka--oversized base-opcount_1]",
        [_case("x", "precompile", "MODEXP")],
    )
    assert info.group_kind == rec.GROUP_KIND_REJECTION
    assert info.current_charge_gas == 0


def test_dynamic_precompile_metadata_never_defaults_to_zero_pairs() -> None:
    blake = rec.classify_variant(
        "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_24-opcount_1]",
        [_case("x", "precompile", "BLAKE2F")],
    )
    assert blake.units == 24
    assert blake.input_bytes == 213
    assert blake.current_charge_gas == 24

    for operation in ("BN128_PAIRING", "BLS12_PAIRING", "BLS12_G1MSM"):
        info = rec.classify_variant(
            f"t/p.py::test_{operation}[fork_Osaka--missing-opcount_1]",
            [_case("x", "precompile", operation)],
        )
        assert info.units is None
        assert info.current_charge_gas is None


def test_exported_transaction_data_resolves_pairing_and_msm_lengths() -> None:
    pair_case = _case("x", "precompile", "BN128_PAIRING")
    pair_case["transactions"] = [{"data": "0x" + "00" * 192}]
    pair = rec.classify_variant(
        "t/p.py::test_bn128_pairings_amortized[fork_Osaka--x-opcount_1]",
        [pair_case],
    )
    assert pair.units == 1

    msm_case = _case("x", "precompile", "BLS12_G1MSM")
    msm_case["transactions"] = [{"data": "0x" + "00" * 160}]
    msm = rec.classify_variant(
        "t/p.py::test_bls12_381_uncachable[fork_Osaka--x-opcount_1]",
        [msm_case],
    )
    assert msm.units == 1

    pair_case["transactions"].append({"data": "0x00"})
    mismatch = rec.classify_variant(
        "t/p.py::test_bn128_pairings_amortized[fork_Osaka--x-opcount_1]",
        [pair_case],
    )
    assert mismatch.units is None


def test_unknown_target_operation_refuses_to_guess() -> None:
    info = rec.classify_variant(
        "t/x.py::test_new_op[fork_Osaka--]", [_case("x", "precompile", "SOMETHING_NEW")]
    )
    assert info.group_kind == rec.GROUP_KIND_UNSUPPORTED
    assert info.current_charge_gas is None
    assert any("refusing to guess" in reason for reason in info.reasons)


# ---------------------------------------------------------------------------
# create-config: calibration exclusion and guards.
# ---------------------------------------------------------------------------


def _mini_workload() -> dict:
    return _workload(
        [
            _case(
                "t/a.py::test_arithmetic[fork_Osaka--opcode_ADD-opcount_1]",
                "arithmetic",
                "ADD",
            ),
            _case(
                "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_24-opcount_1]",
                "precompile",
                "BLAKE2F",
            ),
        ]
    )


def test_create_config_excludes_calibration_from_every_model() -> None:
    workload = _mini_workload()
    workload["cases"].append(
        _case(
            "tests/benchmark/compute/calibration/test_glue.py::"
            "test_calldatasize[fork_Osaka--calib-opcount_1]",
            "precompile",
            "CALLDATASIZE",
            parameters={"campaign_role": "calibration"},
        )
    )
    config, sidecar = rec.build_analysis_config(workload, client="evm2")
    assert len(config["models"]["custom"]) == 2
    assert sidecar["campaign_roles"]["calibration_cases"] == 1
    assert sidecar["campaign_roles"]["calibration_priced"] == 0
    for model in config["models"]["custom"]:
        assert "calibration/" not in model["filter_by"][0]


def test_create_config_rejects_calibration_matching_a_target_filter() -> None:
    workload = _mini_workload()
    workload["cases"].append(
        _case(
            "t/a.py::test_arithmetic[fork_Osaka--opcode_ADD-opcount_calib]",
            "arithmetic",
            "ADD",
            parameters={"campaign_role": "calibration"},
        )
    )
    with pytest.raises(rec.WorkloadError, match="calibration-lane case"):
        rec.build_analysis_config(workload, client="evm2")


def test_load_workload_rejects_foreign_fork_and_schema(tmp_path: Path) -> None:
    good = _mini_workload()
    path = tmp_path / "wl.json"
    path.write_text(json.dumps(good))
    assert rec.load_workload(path)["fork"] == "Osaka"

    bad = _mini_workload()
    bad["fork"] = "Cancun"
    path.write_text(json.dumps(bad))
    with pytest.raises(rec.WorkloadError, match="fork"):
        rec.load_workload(path)

    bad = _mini_workload()
    bad["schema_version"] = 1
    path.write_text(json.dumps(bad))
    with pytest.raises(rec.WorkloadError, match="schema_version"):
        rec.load_workload(path)


def test_create_config_freezes_policy_constants() -> None:
    config, _ = rec.build_analysis_config(_mini_workload(), client="evm2")
    assert config["glue_adjustment"] == {"enabled": True}
    assert config["modeling"]["bootstrap_iterations"] == 1000
    assert config["modeling"]["random_seed"] == 20260922
    assert config["qualification"]["min_sessions"] == 4
    assert config["qualification"]["block_unqualified"] is True
    assert config["campaign"]["eligible_phases"] == ["qualification"]
    assert config["campaign"]["require_correctness_passed"] is True
    assert config["pricing_scenarios"] == [
        {"name": "osaka-600m", "anchor_rate": 600_000_000, "margin_pct": 0.0}
    ]
    assert config["clients"] == ["evm2"]


# ---------------------------------------------------------------------------
# Budget-aware lane (exact charged-gas accounting).
# ---------------------------------------------------------------------------


def _diag(case_id: str, count: int, gas: int) -> dict:
    return {
        "case_id": case_id,
        "phase": "diagnostic",
        "status": "executed",
        "correctness_passed": True,
        "target_count": count,
        "charged_gas": gas,
        "sample_id": f"s-{case_id[-8:-1]}-{count}",
    }


def _blake24_info() -> rec.VariantInfo:
    return rec.classify_variant(
        "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_24]",
        [_case("x", "precompile", "BLAKE2F")],
    )


def test_budget_lane_derives_other_gas_from_exact_affine_charged_gas() -> None:
    info = _blake24_info()
    diagnostics = {
        info.variant_id: [
            _diag(info.variant_id + "-opcount_0.016K", 16, 16 * 142 + 100),
            _diag(info.variant_id + "-opcount_0.256K", 256, 256 * 142 + 100),
        ]
    }
    budget = rec._budget_evidence(info, diagnostics)
    assert budget.affine is True
    assert budget.total_marginal_gas == 142.0
    assert budget.other_marginal_gas == 142.0 - 24.0  # wrapper gas funds the rest

    ev = rec.VariantEvidence(
        qualification_status="qualified",
        raw_point_ms=0.0003,
        raw_low_ms=0.0002,
        raw_high_ms=0.0003634,
    )  # 363.4ns -> 218.04 gas
    need = rec._budget_need_gas(info, budget, ev, ANCHOR)
    assert need == pytest.approx(110.0, abs=0.5)  # max(24, ceil2sig(218.04-118)) -> 110


def test_budget_lane_abstains_on_non_affine_charged_gas() -> None:
    info = _blake24_info()
    diagnostics = {
        info.variant_id: [
            _diag(info.variant_id + "-a", 16, 2400),
            _diag(info.variant_id + "-b", 32, 4700),  # 230/step, not 150
            _diag(info.variant_id + "-c", 64, 9600),
        ]
    }
    budget = rec._budget_evidence(info, diagnostics)
    assert budget.affine is False
    assert budget.total_marginal_gas is None
    assert "abstain" in budget.abstain_reason or "affine" in budget.abstain_reason


def test_budget_lane_never_treats_missing_evidence_as_zero() -> None:
    info = _blake24_info()
    budget = rec._budget_evidence(info, {})
    assert budget.affine is False
    assert rec._budget_need_gas(info, budget, None, ANCHOR) is None


def test_budget_lane_requires_raw_qualification_and_exact_rational_affinity() -> None:
    info = _blake24_info()
    diagnostics = {
        info.variant_id: [
            _diag(info.variant_id + "-a", 1, 101),
            _diag(info.variant_id + "-b", 3, 104),
        ]
    }
    budget = rec._budget_evidence(info, diagnostics)
    assert budget.total_marginal_gas == rec.Fraction(3, 2)
    assert budget.other_marginal_gas == rec.Fraction(-45, 2)
    assert budget.affine is False
    assert "negative" in budget.abstain_reason

    diagnostics[info.variant_id][1]["charged_gas"] = 105
    diagnostics[info.variant_id].append(_diag(info.variant_id + "-dup", 1, 102))
    valid_diagnostics = {
        info.variant_id: [
            _diag(info.variant_id + "-a", 16, 16 * 142 + 100),
            _diag(info.variant_id + "-b", 256, 256 * 142 + 100),
        ]
    }
    budget = rec._budget_evidence(info, valid_diagnostics)
    ev = rec.VariantEvidence(
        raw_point_ms=0.0003,
        raw_low_ms=0.0002,
        raw_high_ms=0.0004,
    )
    assert rec._budget_need_gas(info, budget, ev, ANCHOR) is None
    assert "qualification" in budget.abstain_reason
    budget = rec._budget_evidence(info, diagnostics)
    assert "conflicting duplicate" in budget.abstain_reason


# ---------------------------------------------------------------------------
# End-to-end build over a synthetic analysis.
# ---------------------------------------------------------------------------


def _synthetic_analysis(tmp_path: Path, config: dict) -> Path:
    """Write a minimal completed gasfit analysis for ``config``'s models."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    models = config["models"]["custom"]

    def param_of(model: dict) -> str:
        return model["model_params"]["target_coef"]

    results_rows = []
    proposal_rows = []
    qual_rows = []
    for index, model in enumerate(models):
        label = f"models.custom[{index}]"
        # 100ns/call raw slope; glue adjustment of 20ns -> 80ns adjusted.
        results_rows.append(
            {
                "test_name": model["test_name"],
                "client_name": "evm2",
                "target_opcode": model["target_operation"],
                "source_label": label,
                "target_coef_runtime_ms": "0.0001",
                "target_coef_conf_int_low": "0.00009",
                "target_coef_conf_int_high": "0.00011",
            }
        )
        coverage = "True"
        adjusted_status = "qualified"
        if model["test_name"] == "test_blake2f_benchmark":
            # 24 rounds: current 24 gas; 80ns adjusted -> 48 gas upper
            # 66 gas -> strong increase evidence via lower bound 54/24 > 2.
            proposal_rows.append(
                {
                    "gas_param": param_of(model),
                    "client_name": "evm2",
                    "runtime_ms": "0.00008",
                    "conf_int_low": "0.000075",
                    "conf_int_high": "0.00011",
                    "glue_adjustment": "0.00002",
                    "glue_interval_conditional": "False",
                    "qualification_status": "qualified",
                    "glue_priced_opcodes": "STATICCALL;PUSH",
                    "glue_unpriced_opcodes": "",
                    "glue_bundled_opcodes": "",
                    "glue_detection_status": "evaluated",
                    "glue_coverage_complete": coverage,
                }
            )
        else:
            # ADD: 3 gas current; 80ns point -> 48 gas, lower 45 gas > 2x3.
            proposal_rows.append(
                {
                    "gas_param": param_of(model),
                    "client_name": "evm2",
                    "runtime_ms": "0.00008",
                    "conf_int_low": "0.000075",
                    "conf_int_high": "0.00011",
                    "glue_adjustment": "0.00002",
                    "glue_interval_conditional": "False",
                    "qualification_status": "qualified",
                    "glue_priced_opcodes": "PUSH;DUP",
                    "glue_unpriced_opcodes": "",
                    "glue_bundled_opcodes": "",
                    "glue_detection_status": "evaluated",
                    "glue_coverage_complete": coverage,
                }
            )
        qual_rows.append(
            {
                "source_label": label,
                "test_name": model["test_name"],
                "target_opcode": model["target_operation"],
                "client_name": "evm2",
                "status": "qualified",
                "reasons": "",
                "adjusted_estimate_status": adjusted_status,
            }
        )

    import csv

    def write_csv(name: str, rows: list[dict]) -> None:
        with (analysis / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("results.csv", results_rows)
    write_csv("new_gas_all_params.csv", proposal_rows)
    write_csv("qualification.csv", qual_rows)
    config_content = json.dumps(config)
    config_sha256 = hashlib.sha256(config_content.encode()).hexdigest()
    (analysis / "analysis_status.json").write_text(
        json.dumps(
            {
                "evm_gasfit_version": "0.4.0",
                "campaign": {"manifest_sha256": "deadbeef"},
                "inputs": {
                    "config": {
                        "content": config_content,
                        "path": "/container/workspace/config.json",
                        "sha256": config_sha256,
                    }
                },
            }
        )
    )
    return analysis


def test_build_end_to_end_reports_every_variant_and_group(tmp_path: Path) -> None:
    workload = _mini_workload()
    workload["cases"].append(
        _case(
            "t/c.py::test_clz_diff[fork_Osaka-]",
            "bitwise",
            None,
            status="unsupported",
            parameters={"reason": "no fixed-count generator"},
        )
    )
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    diagnostics = tmp_path / "samples.jsonl"
    blake_variant = "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_24]"
    add_variant = "t/a.py::test_arithmetic[fork_Osaka--opcode_ADD]"
    with diagnostics.open("w") as handle:
        for record in (
            _diag(add_variant + "-opcount_0.25K", 250, 250 * 5 + 21000),
            _diag(add_variant + "-opcount_4K", 4000, 4000 * 5 + 21000),
            _diag(blake_variant + "-opcount_0.016K", 16, 16 * 142 + 100),
            _diag(blake_variant + "-opcount_0.256K", 256, 256 * 142 + 100),
        ):
            handle.write(json.dumps(record) + "\n")

    out_dir = tmp_path / "out"
    rc = rec.main(
        [
            "build",
            "--workload",
            str(tmp_path / "wl.json"),
            "--analysis",
            str(analysis),
            "--diagnostics",
            str(diagnostics),
            "--out",
            str(out_dir),
        ]
    )
    assert rc != 0  # workload file does not exist yet
    (tmp_path / "wl.json").write_text(json.dumps(workload))
    rc = rec.main(
        [
            "build",
            "--workload",
            str(tmp_path / "wl.json"),
            "--analysis",
            str(analysis),
            "--diagnostics",
            str(diagnostics),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    document = json.loads((out_dir / "recommendations.json").read_text())
    assert len(document["variants"]) == 3  # every selected variant, once
    decisions = {v["variant_id"]: v["policy_decision"] for v in document["variants"]}
    assert decisions["t/c.py::test_clz_diff[fork_Osaka-]"] == "unsupported"

    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    # ADD: 45 gas lower bound vs 3 current -> increase candidate.
    add_group = groups["OPCODE_ADD"]
    assert add_group["decision"] == "increase_candidate"
    assert add_group["deployable"] is True
    assert add_group["isolated_candidate"]["params"]["value"] == 66  # 66 gas
    # BLAKE2F: lower 45 gas vs 24 current -> below 2x -> keep by policy.
    blake_group = groups["PRECOMPILE_BLAKE2F"]
    assert blake_group["decision"] == "keep_by_policy"
    # Budget lane: raw upper 66.0 gas - 118 other = 0 -> floor at current
    # per-round 1 gas... but the ADD variant drives its own budget row.
    blake_row = next(
        v for v in document["variants"] if blake_variant in v["variant_id"]
    )
    assert blake_row["budget_aware"]["affine_charged_gas"] is True
    assert blake_row["budget_aware"]["other_marginal_gas"] == 118.0

    csv_text = (out_dir / "recommendations.csv").read_text()
    assert csv_text.count("\n") == 4  # header + 3 rows
    assert "missing_coverage" in csv_text.splitlines()[0]


def test_build_blocks_group_when_glue_coverage_incomplete(tmp_path: Path) -> None:
    workload = _mini_workload()
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    # Corrupt coverage on the ADD proposal row: STOP detected but unpriced.
    rows = list(csv.DictReader((analysis / "new_gas_all_params.csv").open(newline="")))
    for row in rows:
        if row["gas_param"].startswith("WORKLOAD_ADD"):
            row["glue_coverage_complete"] = "False"
            row["glue_unpriced_opcodes"] = "STOP;POP"
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    document = rec.build_recommendations(workload, analysis, None)
    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    add_group = groups["OPCODE_ADD"]
    assert add_group["decision"] == "blocked-coverage"
    assert add_group["deployable"] is False
    assert "STOP" in json.dumps(add_group["missing_coverage"])
    # Incomplete glue coverage blocks deployability and pricing candidates.
    assert "isolated_candidate" not in add_group
    add_variant = "t/a.py::test_arithmetic[fork_Osaka--opcode_ADD]"
    add_row = next(v for v in document["variants"] if add_variant in v["variant_id"])
    assert add_row["policy_decision"] == "blocked-coverage"
    assert any("not isolable" in m for m in add_row["missing_coverage"])


def test_build_marks_missing_glue_columns_as_unknown_coverage(tmp_path: Path) -> None:
    workload = _mini_workload()
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    rows = list(csv.DictReader((analysis / "new_gas_all_params.csv").open(newline="")))
    for row in rows:
        row.pop("glue_coverage_complete", None)
    fields = [k for k in rows[0] if k != ""]
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    document = rec.build_recommendations(workload, analysis, None)
    add_variant = "t/a.py::test_arithmetic[fork_Osaka--opcode_ADD]"
    add_row = next(v for v in document["variants"] if add_variant in v["variant_id"])
    assert add_row["glue_coverage"] == "unknown"
    assert add_row["policy_decision"] == "blocked-coverage"


def test_threshold_boundary_at_2x(tmp_path: Path) -> None:
    workload = _mini_workload()
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    rows = list(csv.DictReader((analysis / "new_gas_all_params.csv").open(newline="")))
    # ADD current 3 gas: lower bound exactly 6.0 gas -> 2.0x triggers.
    for row in rows:
        if row["gas_param"].startswith("WORKLOAD_ADD"):
            row["conf_int_low"] = "0.00001"  # 10ns -> 6 gas
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = rec.build_recommendations(workload, analysis, None)
    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    assert groups["OPCODE_ADD"]["increase_trigger"] is True

    # A hair below the boundary keeps current pricing by policy.
    for row in rows:
        if row["gas_param"].startswith("WORKLOAD_ADD"):
            row["conf_int_low"] = "0.0000099"  # 9.9ns -> 5.94 gas
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = rec.build_recommendations(workload, analysis, None)
    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    assert groups["OPCODE_ADD"]["increase_trigger"] is False
    assert groups["OPCODE_ADD"]["decision"] == "keep_by_policy"


def test_no_decreases_candidate_floors_at_current(tmp_path: Path) -> None:
    workload = _mini_workload()
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    rows = list(csv.DictReader((analysis / "new_gas_all_params.csv").open(newline="")))
    for row in rows:
        if row["gas_param"].startswith("WORKLOAD_ADD"):
            row["runtime_ms"] = "0.0000005"  # 0.5ns -> 0.3 gas (tiny)
            row["conf_int_low"] = "0.0000004"
            row["conf_int_high"] = "0.0000006"
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = rec.build_recommendations(workload, analysis, None)
    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    # Every upper bound <= current: proven adequate, no decrease proposed.
    assert groups["OPCODE_ADD"]["decision"] == "keep_proven_adequate"
    variant_row = next(
        v for v in document["variants"] if "opcode_ADD" in v["variant_id"]
    )
    assert variant_row["conservative_candidate_gas"] == 3.0  # floor at current


def test_cached_keccak_variant_never_drives_increase(tmp_path: Path) -> None:
    cached_variant = "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_32]"
    uncached_variant = (
        "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_136]"
    )
    cached_case = cached_variant[:-1] + "-opcount_1]"
    uncached_case = uncached_variant[:-1] + "-opcount_1]"
    workload = _workload(
        [
            _case(
                cached_case,
                "keccak",
                "KECCAK256",
                parameters=_keccak_witness_params(32),
            ),
            _case(
                uncached_case,
                "keccak",
                "KECCAK256",
                parameters=_keccak_witness_params(136),
            ),
        ]
    )
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    rows = list(csv.DictReader((analysis / "new_gas_all_params.csv").open(newline="")))
    # Cached variant: enormous lower bound; uncached: modest evidence.
    by_param = {row["gas_param"]: row for row in rows}
    cached_param = rec.variant_param(cached_variant, "KECCAK256")
    uncached_param = rec.variant_param(uncached_variant, "KECCAK256")
    by_param[cached_param]["runtime_ms"] = "0.015"
    by_param[cached_param]["conf_int_low"] = "0.01"
    by_param[cached_param]["conf_int_high"] = "0.02"
    by_param[uncached_param]["runtime_ms"] = "0.0000015"
    by_param[uncached_param]["conf_int_low"] = "0.000001"
    by_param[uncached_param]["conf_int_high"] = "0.000002"
    with (analysis / "new_gas_all_params.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = rec.build_recommendations(workload, analysis, None)
    groups = {g["pricing_group"]: g for g in document["pricing_groups"]}
    keccak = groups["OPCODE_KECCAK256"]
    # The absurd cached slope must not fire the trigger; the modest uncached
    # evidence holds current pricing.
    assert keccak["increase_trigger"] is False
    assert cached_variant in keccak["inherent_keccak_cache_variants"]
    cached_row = next(
        v for v in document["variants"] if cached_variant in v["variant_id"]
    )
    assert cached_row["policy_decision"] == "blocked-coverage"
    assert any(
        "keccak cache" in m or "keccak input" in m
        for m in cached_row["missing_coverage"]
    )


def test_resolved_input_sidecar_validation_and_application() -> None:
    case_id = "t/k.py::test_keccak_max_permutations[fork_Osaka--x-opcount_1]"
    workload = _workload(
        [
            _case(
                case_id,
                "keccak",
                "KECCAK256",
                parameters={"source_parameters": {}},
            )
        ]
    )
    infos = rec.classify_workload(workload)
    variant_id = rec._variant_of(case_id)
    entry = {
        "variant_id": variant_id,
        "input_bytes": 65281,
        "case_ids": [case_id],
        "evidence": "reviewed fixture bytecode",
    }
    digest = (
        __import__("hashlib")
        .sha256(json.dumps(workload, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest()
    )
    sidecar = {
        "schema_version": 1,
        "canonical_workload_sha256": digest,
        "inputs": [entry],
    }
    annotations = rec._apply_resolved_inputs(workload, infos, sidecar)
    assert annotations[0]["input_bytes"] == 65281
    assert infos[variant_id].units == (65281 + 31) // 32

    bad_hash = dict(sidecar, canonical_workload_sha256="0" * 64)
    with pytest.raises(rec.BuildError, match="workload hash"):
        rec._apply_resolved_inputs(workload, rec.classify_workload(workload), bad_hash)
    with pytest.raises(rec.BuildError, match="missing variant"):
        rec._apply_resolved_inputs(
            workload,
            rec.classify_workload(workload),
            {"schema_version": 1, "canonical_workload_sha256": digest, "inputs": []},
        )

    known_case = (
        "t/k.py::test_keccak_workload_witness[fork_Osaka--input_length_32-opcount_1]"
    )
    known_workload = _workload(
        [
            _case(
                known_case, "keccak", "KECCAK256", parameters=_keccak_witness_params(32)
            )
        ]
    )
    known_variant = rec._variant_of(known_case)
    known_digest = (
        __import__("hashlib")
        .sha256(
            json.dumps(known_workload, sort_keys=True, separators=(",", ":")).encode()
        )
        .hexdigest()
    )
    with pytest.raises(rec.BuildError, match="overrides known"):
        rec._apply_resolved_inputs(
            known_workload,
            rec.classify_workload(known_workload),
            {
                "schema_version": 1,
                "canonical_workload_sha256": known_digest,
                "inputs": [
                    {
                        "variant_id": known_variant,
                        "input_bytes": 64,
                        "case_ids": [known_case],
                        "evidence": "invalid override",
                    }
                ],
            },
        )


def _qualified_evidence(gas: float) -> rec.VariantEvidence:
    """Build complete isolated and raw evidence at ``gas`` on the fixed anchor."""
    milliseconds = gas * 1000.0 / ANCHOR
    return rec.VariantEvidence(
        source_label="models.custom[0]",
        raw_point_ms=milliseconds,
        raw_low_ms=milliseconds,
        raw_high_ms=milliseconds,
        adjusted_point_ms=milliseconds,
        adjusted_low_ms=milliseconds,
        adjusted_high_ms=milliseconds,
        qualification_status="qualified",
        adjusted_estimate_status="qualified",
        glue_coverage_complete=True,
        glue_priced="STATICCALL",
        glue_detection_status="evaluated",
    )


def test_modexp_floor_candidates_scale_product_and_evaluate_charge() -> None:
    case = _case(
        "t/m.py::test_modexp[fork_Osaka--floor_active-opcount_1]",
        "precompile",
        "MODEXP",
        parameters={
            "source_parameters": {
                "mod_exp_input": "base=b'\\x01' exponent=b'\\x01' modulus=b'\\x01'"
            }
        },
    )
    info = rec.classify_variant(case["id"], [case])
    assert info.current_charge_gas == 500

    evidence = {info.variant_id: _qualified_evidence(1200)}
    diagnostics = {
        info.variant_id: [
            _diag(info.variant_id + "-count-1", 1, 500),
            _diag(info.variant_id + "-count-2", 2, 1000),
        ]
    }
    report = rec._group_report(
        "PRECOMPILE_MODEXP",
        [info],
        {info.variant_id: info},
        evidence,
        diagnostics,
        ANCHOR,
    )
    expected_params = {
        "PRECOMPILE_MODEXP_MULTIPLIER": 75,
        "PRECOMPILE_MODEXP_MIN_GAS": 500,
    }
    for lane in ("isolated_candidate", "budget_aware_candidate"):
        candidate = report[lane]
        assert candidate["product_multiplier"] == 75
        assert candidate["params"] == expected_params
        assert candidate["candidate_charge_gas"] == {info.variant_id: 1200}


def test_blake2_zero_round_shape_blocks_both_candidate_lanes() -> None:
    zero_case = _case(
        "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_0-opcount_1]",
        "precompile",
        "BLAKE2F",
    )
    positive_case = _case(
        "t/b.py::test_blake2f_benchmark[fork_Osaka--num_rounds_24-opcount_1]",
        "precompile",
        "BLAKE2F",
    )
    zero = rec.classify_variant(zero_case["id"], [zero_case])
    positive = rec.classify_variant(positive_case["id"], [positive_case])
    evidence = {
        zero.variant_id: _qualified_evidence(1200),
        positive.variant_id: _qualified_evidence(1200),
    }
    diagnostics = {
        zero.variant_id: [
            _diag(zero.variant_id + "-count-1", 1, 100),
            _diag(zero.variant_id + "-count-2", 2, 200),
        ],
        positive.variant_id: [
            _diag(positive.variant_id + "-count-1", 1, 124),
            _diag(positive.variant_id + "-count-2", 2, 248),
        ],
    }
    report = rec._group_report(
        "PRECOMPILE_BLAKE2F",
        [zero, positive],
        {zero.variant_id: zero, positive.variant_id: positive},
        evidence,
        diagnostics,
        ANCHOR,
    )

    assert report["decision"] == "blocked-shape"
    assert report["deployable"] is False
    assert report["increase_trigger"] is True
    assert report["sensitivity_triggers"]["2.0x"] is True
    assert report["missing_coverage"] == {}
    assert report["isolated_candidate"]["abstained"] is True
    assert report["budget_decision"] == "blocked-shape"
    assert report["budget_increase_trigger"] is True
    assert report["budget_sensitivity_triggers"]["2.0x"] is True
    assert report["budget_coverage"]["complete"] is True
    assert report["budget_aware_candidate"]["abstained"] is True


def test_modexp_input_rejection_budget_lane_is_not_priced() -> None:
    case = _case(
        "t/m.py::test_modexp_length_above_upper_bound[fork_Osaka--oversized-opcount_1]",
        "precompile",
        "MODEXP",
    )
    info = rec.classify_variant(case["id"], [case])
    report = rec._group_report(
        "MODEXP_INPUT_REJECTION",
        [info],
        {info.variant_id: info},
        {},
        {},
        ANCHOR,
    )

    assert report["decision"] == "not-priced"
    assert report["budget_decision"] == "not-priced"
    assert "budget_aware_candidate" not in report


def test_build_uses_embedded_config_json_yaml_and_hash_validation(
    tmp_path: Path,
) -> None:
    import yaml

    workload = _mini_workload()
    config, _ = rec.build_analysis_config(workload, client="evm2")
    analysis = _synthetic_analysis(tmp_path, config)
    status_path = analysis / "analysis_status.json"
    status = json.loads(status_path.read_text())
    config_path = status["inputs"]["config"]["path"]
    assert not Path(config_path).exists()

    document = rec.build_recommendations(workload, analysis, None)
    add = next(
        g for g in document["pricing_groups"] if g["pricing_group"] == "OPCODE_ADD"
    )
    assert add["isolated_candidate"]["params"]["value"] == 66

    yaml_content = yaml.safe_dump(config, sort_keys=False)
    status["inputs"]["config"] = {
        "content": yaml_content,
        "path": "/container/workspace/config.yaml",
        "sha256": hashlib.sha256(yaml_content.encode()).hexdigest(),
    }
    status_path.write_text(json.dumps(status))
    document = rec.build_recommendations(workload, analysis, None)
    add = next(
        g for g in document["pricing_groups"] if g["pricing_group"] == "OPCODE_ADD"
    )
    assert add["isolated_candidate"]["params"]["value"] == 66

    tampered_content = dict(status)
    tampered_content["inputs"] = dict(status["inputs"])
    tampered_content["inputs"]["config"] = dict(status["inputs"]["config"])
    tampered_content["inputs"]["config"]["content"] += "\n# tampered"
    status_path.write_text(json.dumps(tampered_content))
    with pytest.raises(rec.BuildError):
        rec.build_recommendations(workload, analysis, None)

    tampered_hash = dict(status)
    tampered_hash["inputs"] = dict(status["inputs"])
    tampered_hash["inputs"]["config"] = dict(status["inputs"]["config"])
    tampered_hash["inputs"]["config"]["sha256"] = "0" * 64
    status_path.write_text(json.dumps(tampered_hash))
    with pytest.raises(rec.BuildError):
        rec.build_recommendations(workload, analysis, None)


def test_work_size_token_stripping_is_mode_independent() -> None:
    fixed = "t/a.py::test_arithmetic[fork_Osaka--opcode_DIV-0-opcount_0.25K]"
    budget = (
        "t/a.py::test_arithmetic[fork_Osaka--opcode_DIV-0-benchmark-gas-value_120M]"
    )
    variant = "t/a.py::test_arithmetic[fork_Osaka--opcode_DIV-0]"
    assert rec._variant_of(fixed) == variant
    assert rec._variant_of(budget) == variant
    # Budgets sharing a digit prefix stay one variant, not two.
    assert rec._variant_of(budget.replace("120M", "100M")) == variant
