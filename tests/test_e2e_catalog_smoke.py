"""End-to-end: every catalog preset loads, validates, and fits without raising.

The goal is to catch typos in ``test_name`` / ``target_operation`` /
``model_params`` keys at landing time — not to validate fit quality. We
synthesize one minimal block-limit sweep per preset (just enough to satisfy
each preset's ``filter_by``, ``model_by``, ``target_operation_param`` and
``fixture_params`` requirements), drive the full pipeline through a config
that lists every preset, and assert ``results.csv`` carries at least one
row per preset and ``new_gas.csv`` is non-empty.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _data_synth import (
    ClientModel,
    FixtureSpec,
    base_config,
    runtime_for,
    write_config_yaml,
    write_opcounts_json,
)

from evm_gasfit.config import ModelSpec
from evm_gasfit.defaults.models import PRESETS

_BLOCK_LIMITS = (30, 60, 90, 120, 150)
_OPCOUNT_PER_MILLION = 500_000.0

# Names the catalog's presets propose that are not raw osaka fork fields.
# Any preset-only config must declare these in `new_params` for load to succeed.
_CATALOG_NEW_PARAMS: dict[str, int | None] = {
    "COLD_ACCOUNT_CODE_ACCESS": None,
    "COLD_ACCOUNT_NOCODE_ACCESS": None,
    "COLD_ACCOUNT_CODE_WRITE": None,
    "COLD_ACCOUNT_NOCODE_WRITE": None,
    "OPCODE_CALLDATACOPY_PER_WORD": None,
    "OPCODE_CODECOPY_PER_WORD": None,
    "OPCODE_MCOPY_PER_WORD": None,
}


# Representative values for known param columns when the preset declares them
# under model_by or as a fixture_params source. Keys are param column names,
# values are stringified to match the fixture-name parser's output (which
# always yields string values).
_PLACEHOLDER_PARAMS: dict[str, str] = {
    "mod_bits": "256",
    "mem_size": "1024",
    "calldata_size": "32",
    "return_size": "32",
    "returned_size": "32",
    "copy_size": "32",
    "code_size": "32",
    "msg_size": "32",
    "size": "32",
    "block": "5",
    "num_rounds": "10",
    "num_pairs": "2",
    "k": "2",
    "value_sent": "1",
    "write_new_value": "True",
    "existing_slots": "True",
    # `account_mode` is handled by `_representative_account_mode` instead —
    # no single value survives every account-access preset's negation filters.
}


def _placeholder(col: str) -> str:
    return _PLACEHOLDER_PARAMS.get(col, "1")


def _representative_target(spec: ModelSpec) -> str:
    """Pick a target_opcode for a target_operation_param-based spec.

    The catalog uses ``opcode`` as the param key for every such preset; the
    actual value only has to be a valid mnemonic so the opcount invariant has
    a column to match against.
    """
    if spec.target_operation_param == "opcode":
        # PUSH presets specifically filter to opcodes containing "PUSH".
        for token in spec.filter_by:
            if "PUSH" in token:
                return "PUSH1"
        if "stack_dup" in spec.test_name or spec.test_name == "test_dup":
            return "DUP1"
        if spec.test_name == "test_swap":
            return "SWAP1"
        if spec.test_name == "test_create":
            return "CREATE"
        if spec.test_name == "test_ext_account_query_warm":
            return "BALANCE"
        # test_account_access: cold-account family.
        return "BALANCE"
    return "BALANCE"


_ACCOUNT_MODE_CANDIDATES: tuple[str, ...] = (
    "AccountMode.NON_EXISTING_ACCOUNT",
    "AccountMode.EXISTING_EOA",
    "AccountMode.EXISTING_CONTRACT_MINIMAL",
)


def _representative_account_mode(spec: ModelSpec) -> str:
    """Pick an `account_mode` value that survives every `!AccountMode.*`
    negation in `spec.filter_by`.

    No single hardcoded value works for every account-access preset: the
    NOCODE presets exclude `EXISTING_CONTRACT`, and the CODE presets exclude
    both `EXISTING_EOA` and `NON_EXISTING_ACCOUNT` (no `overhead_baseline_True`
    counterpart for either mode) — so the placeholder has to be chosen per
    spec.
    """
    # `_apply_filters` matches by substring, so "!AccountMode.EXISTING_CONTRACT"
    # excludes "AccountMode.EXISTING_CONTRACT_MINIMAL" too — check containment,
    # not exact equality.
    excluded = [
        token[1:] for token in spec.filter_by if token.startswith("!AccountMode.")
    ]
    for candidate in _ACCOUNT_MODE_CANDIDATES:
        if not any(e in candidate for e in excluded):
            return candidate
    return _ACCOUNT_MODE_CANDIDATES[0]


def _filter_tokens_to_params(
    spec: ModelSpec, params: dict[str, str], target_opcode: str
) -> None:
    """Mutate ``params`` so every ``filter_by`` substring will appear in the
    generated fixture name. Auto-generated tokens (``opcode_<X>``) are skipped
    because they're already produced by the target_opcode argument."""
    for token in spec.filter_by:
        if token.startswith("!"):
            # Negation tokens require absence; placeholder fixtures never
            # contain real-fork variant labels, so nothing to inject.
            continue
        if token.startswith("opcode_"):
            # Already covered by the target_opcode argument.
            continue
        if "_" in token:
            key, _, value = token.partition("_")
            # Don't overwrite an existing param; instead store the token under
            # a synthetic key so the whole substring still appears in the name.
            if key in params and params[key] != value:
                params[f"flag_{token}"] = "X"
            else:
                params[key] = value
        else:
            # Standalone tag — embed as `<token>_X` so the parser is happy and
            # the substring `token` still appears in the fixture name.
            params[token] = "X"


def _smoke_fixtures_for(spec: ModelSpec) -> list[FixtureSpec]:
    is_precompile = spec.target_operation_count_source is not None

    if spec.target_operation is not None:
        target_opcode = spec.target_operation
    else:
        target_opcode = _representative_target(spec)

    params: dict[str, str] = {}

    if spec.target_operation_param is not None:
        params[spec.target_operation_param] = target_opcode

    for col in spec.model_by:
        if col == "account_mode":
            params.setdefault(col, _representative_account_mode(spec))
        else:
            params.setdefault(col, _placeholder(col))

    for fp_spec in spec.fixture_params.values():
        if fp_spec.source not in params:
            if fp_spec.values:
                params[fp_spec.source] = next(iter(fp_spec.values.keys()))
            else:
                params[fp_spec.source] = _placeholder(fp_spec.source)

    # Extra model_params coefficients beyond target_coef must each map to a
    # column on the parsed frame. Add a placeholder param so the column
    # exists; the one-value-extras filter then drops it from the design
    # matrix (only target_coef survives, which is fine for a smoke test).
    for coef_name in spec.model_params:
        if coef_name == "target_coef":
            continue
        # If the coef is the name of a derived fixture-param, it's already
        # materialized from its source — don't add a same-named raw param.
        if coef_name in spec.fixture_params:
            continue
        params.setdefault(coef_name, _placeholder(coef_name))

    _filter_tokens_to_params(spec, params, target_opcode)

    if spec.overhead_baseline_param is not None:
        # This spec pairs each fixture against an `overhead_baseline_True`
        # counterpart (see modeling/estimate.py `_split_baseline_pair`) — the
        # smoke fixtures need one, or the pairing step raises.
        params[spec.overhead_baseline_param] = "False"

    fixtures: list[FixtureSpec] = []
    for bl in _BLOCK_LIMITS:
        fixtures.append(
            FixtureSpec(
                test_file=spec.test_name,
                test_name=spec.test_name,
                params=dict(params),
                block_limit_million=bl,
                target_opcode=target_opcode,
                target_opcount=bl * _OPCOUNT_PER_MILLION,
                count_source_opcode=spec.target_operation_count_source,
                omit_opcode_token=is_precompile,
            )
        )
        if spec.overhead_baseline_param is not None:
            true_params = dict(params)
            true_params[spec.overhead_baseline_param] = "True"
            fixtures.append(
                FixtureSpec(
                    test_file=spec.test_name,
                    test_name=spec.test_name,
                    params=true_params,
                    block_limit_million=bl,
                    target_opcode=target_opcode,
                    target_opcount=0.0,
                )
            )
    return fixtures


def _build_runtimes(
    fixtures: Iterable[FixtureSpec], model: ClientModel, *, noise_pct: float = 0.002
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for spec in fixtures:
        rows.append(
            {
                "client_name": "geth",
                "fixture_name": spec.fixture_name,
                "test_runtime_ms": runtime_for(spec, model, rng, noise_pct),
            }
        )
    return pd.DataFrame(rows)


def test_every_catalog_preset_fits_without_raising(tmp_path: Path) -> None:
    """The full catalog drives an end-to-end pipeline run on synthetic data."""
    all_fixtures: list[FixtureSpec] = []
    preset_to_fixture_names: dict[str, list[str]] = {}
    for name, spec in PRESETS.items():
        preset_fixtures = _smoke_fixtures_for(spec)
        all_fixtures.extend(preset_fixtures)
        preset_to_fixture_names[name] = [f.fixture_name for f in preset_fixtures]

    # Deduplicate fixture names: identical fixture names can appear when two
    # presets share test_name + target_opcode but use different filter_by
    # tokens to disambiguate. The opcounts JSON would need a unique key per
    # fixture, so when fixture names collide, the LAST entry wins — that's
    # fine for a smoke test since the opcount invariant is identical for any
    # writer of the same (target, count_source) pair.
    seen_names: set[str] = set()
    deduped_fixtures: list[FixtureSpec] = []
    for spec in all_fixtures:
        if spec.fixture_name in seen_names:
            continue
        seen_names.add(spec.fixture_name)
        deduped_fixtures.append(spec)

    model = ClientModel(intercept=80.0, slope=1.0e-5)
    runtimes_df = _build_runtimes(deduped_fixtures, model)

    runtimes_csv = tmp_path / "runtimes.csv"
    opcounts_json = tmp_path / "opcounts.json"
    config_yaml = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"

    runtimes_df.to_csv(runtimes_csv, index=False)
    write_opcounts_json(opcounts_json, deduped_fixtures)

    config = base_config(
        models_custom=[],
        models_presets=list(PRESETS.keys()),
        new_params=_CATALOG_NEW_PARAMS,
        clients=("geth",),
    )
    write_config_yaml(config_yaml, config)

    from evm_gasfit import GasFit

    gas_fit = GasFit.from_config(config_yaml)
    gas_fit.load_runtimes(runtimes_csv)
    gas_fit.load_opcounts(opcounts_json)
    gas_fit.estimate_models()
    gas_fit.build_proposal()
    gas_fit.write_reports(out_dir)

    results = pd.read_csv(out_dir / "results.csv")
    # At least one row per preset's test_name should be present. Group by
    # test_name (multiple presets can share a test_name) and verify each is
    # represented.
    represented_test_names = set(results["test_name"])
    expected_test_names = {spec.test_name for spec in PRESETS.values()}
    missing = expected_test_names - represented_test_names
    assert not missing, f"these test_names produced no results.csv rows: {missing}"

    new_gas = pd.read_csv(out_dir / "new_gas.csv")
    assert not new_gas.empty, "new_gas.csv must carry at least one proposed param"
    # The proposed params include both raw fork fields (OPCODE_ADD, ...) and
    # the catalog's new names (ACCOUNT_WRITE, STORAGE_WRITE, ...).
    assert "OPCODE_ADD" in set(new_gas["gas_param"])


def test_catalog_requires_new_params_declaration(tmp_path: Path) -> None:
    """A preset-only config that doesn't declare the catalog's new names fails
    to load with a hard `ConfigError`; declaring them lets the config load
    cleanly with no warnings emitted."""
    from evm_gasfit.config import load_config
    from evm_gasfit.errors import ConfigError

    # Without `new_params`, a preset writing a non-raw name is a hard error.
    bare = base_config(
        models_custom=[], models_presets=list(PRESETS.keys()), clients=("geth",)
    )
    bare_yaml = tmp_path / "bare.yaml"
    write_config_yaml(bare_yaml, bare)
    with pytest.raises(ConfigError, match=r"not declared in new_params"):
        load_config(bare_yaml)

    # With `new_params` declared, the catalog loads clean.
    declared = base_config(
        models_custom=[],
        models_presets=list(PRESETS.keys()),
        new_params=_CATALOG_NEW_PARAMS,
        clients=("geth",),
    )
    declared_yaml = tmp_path / "declared.yaml"
    write_config_yaml(declared_yaml, declared)
    loaded = load_config(declared_yaml)
    assert loaded.warnings == []
