"""Helpers for generating synthetic e2e test inputs.

Each test produces three files:

- `runtimes.csv` with columns (at minimum) `client_name, fixture_name,
  test_runtime_ms`.
- `opcounts.json` keyed by fixture_name, each value containing `opcount`
  (the target-opcode count), the target opcode's mnemonic key (which must
  equal `opcount` per the loader invariant), plus per-opcode counts.
- `config.yaml`.

Runtime is generated from a known linear model so tests can assert that the
NNLS regression recovers the true slope (and, for multi-feature models, the
extra coefficients) within a tolerance.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class FixtureSpec:
    """A single benchmark fixture.

    `params` keys/values appear inside the `[...]` of the EEST fixture name,
    in insertion order. `extra_opcounts` lists glue/non-target opcodes used
    by this fixture; values are counts, not ratios.

    `count_source_opcode` is the precompile escape hatch: when set, the
    opcounts JSON entry uses that opcode as the opcount column (instead of
    `target_opcode`, which is then treated as a synthetic display name with
    no opcount column at all). `omit_opcode_token` likewise suppresses the
    default `opcode_<target>` fixture-name token; precompile fixtures
    identify their variant via a `params` entry instead (e.g. `bls12_g1add`).
    """

    test_file: str
    test_name: str
    params: dict[str, str]
    block_limit_million: int
    target_opcode: str
    target_opcount: float
    extra_opcounts: dict[str, float] = field(default_factory=dict)
    count_source_opcode: str | None = None
    omit_opcode_token: bool = False
    # Template for the trailing scan-axis token; `{n}` is replaced with
    # `block_limit_million`. EEST's newer convention is `"benchmark_{n}M"`.
    sweep_token_format: str = "block_limit_million_{n}"
    # Campaign lane (``param_campaign_role``): calibration rows are glue
    # drivers excluded from every target-model fit; target/None rows are
    # the priced corpus. Emitted as an explicit CSV column when any
    # fixture carries a role.
    campaign_role: str | None = None

    @property
    def fixture_name(self) -> str:
        tokens = ["fork_Amsterdam", "benchmark_test"]
        if not self.omit_opcode_token:
            tokens.append(f"opcode_{self.target_opcode}")
        tokens.extend(f"{k}_{v}" for k, v in self.params.items())
        tokens.append(self.sweep_token_format.format(n=self.block_limit_million))
        return f"{self.test_file}.py__{self.test_name}[{'-'.join(tokens)}]"


@dataclass
class ClientModel:
    """The true linear model for one client.

    `runtime = intercept + slope * opcount
               + Σ_i extra_coef_i * opcount * param_i
               + Σ_j glue_coef_j * extra_opcounts[op_j]`

    `extra_coefs` maps a fixture-param name to a coefficient. The param value
    is coerced to float and multiplied by opcount when generating runtimes,
    matching the regression form in plan §4.3.

    `glue_coefs` maps an opcode mnemonic (matching keys in
    `FixtureSpec.extra_opcounts`) to its per-count runtime contribution in
    ms. This models real glue-opcode contamination: when a benchmark
    fixture's bytecode emits ``count_p`` instances of opcode ``p``, the
    measured runtime includes ``glue_coef_p · count_p``. Mixed-tier glue
    fits subtract these contributions on the LHS using the partner's
    estimated coefficient, so synthesizing them lets tests assert recovery
    of the planted target slope.
    """

    intercept: float
    slope: float
    extra_coefs: dict[str, float] = field(default_factory=dict)
    glue_coefs: dict[str, float] = field(default_factory=dict)


def runtime_for(
    spec: FixtureSpec, model: ClientModel, rng: np.random.Generator, noise_pct: float
) -> float:
    val = model.intercept + model.slope * spec.target_opcount
    for param_name, coef in model.extra_coefs.items():
        if param_name not in spec.params:
            # Coefficient doesn't apply to this fixture (e.g. mixed-spec runs
            # where only some fixtures carry the param).
            continue
        param_val = float(spec.params[param_name])
        val += coef * spec.target_opcount * param_val
    for op_name, ms_per_count in model.glue_coefs.items():
        if op_name == spec.target_opcode:
            # Target-opcode contribution: count is target_opcount (the
            # mnemonic and target_opcount are the same column for a
            # well-formed fixture, per the §2.3 invariant).
            val += ms_per_count * spec.target_opcount
        else:
            count = spec.extra_opcounts.get(op_name, 0.0)
            if count:
                val += ms_per_count * count
    if noise_pct > 0:
        val *= 1.0 + rng.normal(0.0, noise_pct)
    return float(val)


def write_runtimes_csv(
    path: Path,
    fixtures: Sequence[FixtureSpec],
    models: Mapping[str, ClientModel],
    *,
    noise_pct: float = 0.0,
    seed: int = 42,
) -> None:
    """Write a runtimes CSV with one row per (client, fixture).

    Uses pandas so the schema (header order, types) is predictable. When any
    fixture carries a ``campaign_role``, a ``param_campaign_role`` column is
    emitted for every row (target rows are explicit).
    """
    rng = np.random.default_rng(seed)
    has_roles = any(f.campaign_role is not None for f in fixtures)
    rows = []
    for client, model in models.items():
        for spec in fixtures:
            row = {
                "client_name": client,
                "fixture_name": spec.fixture_name,
                "test_runtime_ms": runtime_for(spec, model, rng, noise_pct),
            }
            if has_roles:
                row["param_campaign_role"] = spec.campaign_role or "target"
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_opcounts_json(path: Path, fixtures: Sequence[FixtureSpec]) -> None:
    payload: dict[str, dict[str, float]] = {}
    for spec in fixtures:
        # The invariant requires data[fixture]["opcount"] == data[fixture][count_source]
        # where count_source is target_opcode for opcode targets, or the
        # spec's count_source_opcode for precompiles. When count_source_opcode
        # is set, target_opcode is a synthetic display name and is NOT written
        # as a separate opcount column.
        count_key = spec.count_source_opcode or spec.target_opcode
        entry: dict[str, float] = {
            "opcount": spec.target_opcount,
            count_key: float(spec.target_opcount),
        }
        for op, n in spec.extra_opcounts.items():
            if op == count_key:
                raise ValueError(
                    f"extra_opcounts for {spec.fixture_name} duplicates the "
                    f"count-source opcode {count_key!r}"
                )
            if op == spec.target_opcode and spec.count_source_opcode is None:
                raise ValueError(
                    f"extra_opcounts for {spec.fixture_name} duplicates the "
                    f"target opcode {spec.target_opcode!r}"
                )
            entry[op] = float(n)
        payload[spec.fixture_name] = entry
    path.write_text(json.dumps(payload, indent=2))


def write_config_yaml(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False))


# ----- Convenience fixture builders ---------------------------------------


def make_block_limit_fixtures(
    *,
    test_file: str,
    test_name: str,
    target_opcode: str,
    params: dict[str, str] | None = None,
    block_limits: Sequence[int] = (30, 60, 90, 120, 150, 180, 210, 240),
    target_opcount_per_million: float = 1_000_000.0,
    extra_opcount_per_million: Mapping[str, float] | None = None,
    sweep_token_format: str = "block_limit_million_{n}",
    campaign_role: str | None = None,
) -> list[FixtureSpec]:
    """Build fixtures that vary only by block-limit.

    `target_opcount_per_million` is opcount per 1M block-limit. With the
    defaults above we get opcounts in [3e7, 2.4e8], spanning more than one
    order of magnitude, which gives NNLS enough leverage to fit cleanly.
    """
    params = params or {}
    extra_opcount_per_million = extra_opcount_per_million or {}
    fixtures: list[FixtureSpec] = []
    for bl in block_limits:
        fixtures.append(
            FixtureSpec(
                test_file=test_file,
                test_name=test_name,
                params=dict(params),
                block_limit_million=bl,
                target_opcode=target_opcode,
                target_opcount=bl * target_opcount_per_million,
                extra_opcounts={
                    op: bl * per_m for op, per_m in extra_opcount_per_million.items()
                },
                sweep_token_format=sweep_token_format,
                campaign_role=campaign_role,
            )
        )
    return fixtures


def cross_product_fixtures(
    *,
    test_file: str,
    test_name: str,
    param_grid: Mapping[str, Sequence[str]],
    target_opcode_for: Mapping[tuple[tuple[str, str], ...], str] | str,
    block_limits: Sequence[int] = (30, 60, 90, 120, 150, 180, 210, 240),
    target_opcount_per_million: float = 1_000_000.0,
    extra_opcount_per_million: Mapping[str, float] | None = None,
    sweep_token_format: str = "block_limit_million_{n}",
) -> list[FixtureSpec]:
    """Cross-product over `param_grid`, one fixture set per combo.

    `target_opcode_for` can be a literal string (every combo has the same
    target) or a mapping from sorted param items to the target opcode.
    """
    extra_opcount_per_million = extra_opcount_per_million or {}
    keys = list(param_grid.keys())
    fixtures: list[FixtureSpec] = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if isinstance(target_opcode_for, str):
            target = target_opcode_for
        else:
            target = target_opcode_for[tuple(sorted(params.items()))]
        for bl in block_limits:
            fixtures.append(
                FixtureSpec(
                    test_file=test_file,
                    test_name=test_name,
                    params=dict(params),
                    block_limit_million=bl,
                    target_opcode=target,
                    target_opcount=bl * target_opcount_per_million,
                    extra_opcounts={
                        op: bl * per_m
                        for op, per_m in extra_opcount_per_million.items()
                    },
                    sweep_token_format=sweep_token_format,
                )
            )
    return fixtures


def make_glue_driver_fixtures(
    target_opcount_per_million: float = 2_000_000.0,
    sweep_token_format: str = "block_limit_million_{n}",
    campaign_role: str | None = "calibration",
) -> list[FixtureSpec]:
    """Driver fixtures for every pure or cycle priced glue spec.

    Walks `evm_gasfit.glue.required.PRICED_GLUE_SPECS`, generating one
    block-limit sweep per family member of the **pure** and **cycle**
    tiers. Family specs (DUP/SWAP/PUSH) therefore produce one sweep per
    `DUPn`/`SWAPn`/`PUSHn`; the e2e pipeline collapses them into one
    canonical estimate. Specs without a driver test are skipped. Fixtures
    default to the calibration lane, mirroring the real campaign layout.

    Mixed-tier specs (`mixed_a`, `mixed_b`) are intentionally excluded:
    their canonical names (`ADD`, `MSTORE`, `KECCAK256`, ...) are also
    modelspec targets, so the driver fixtures come from the same test
    slices the modelspec pipeline already populates. Generating them
    here would collide on `fixture_name` and overwrite caller-provided
    contamination.
    """
    from evm_gasfit.glue.required import PRICED_GLUE_SPECS

    fixtures: list[FixtureSpec] = []
    for spec in PRICED_GLUE_SPECS:
        if spec.test_name is None:
            continue
        if spec.tier not in ("pure", "cycle"):
            continue
        for member in spec.members:
            fixtures.extend(
                make_block_limit_fixtures(
                    test_file=spec.test_name,
                    test_name=spec.test_name,
                    target_opcode=member,
                    params={"opcode": member},
                    target_opcount_per_million=target_opcount_per_million,
                    sweep_token_format=sweep_token_format,
                    campaign_role=campaign_role,
                )
            )
    return fixtures


# ----- Canonical config + run helper --------------------------------------


_ADD_MODEL_SPEC: dict[str, Any] = {
    "test_name": "test_arithmetic",
    "target_operation": "ADD",
    "model_params": {"target_coef": "OPCODE_ADD"},
}


def base_config(
    *,
    models_custom: Sequence[dict[str, Any]] | None = None,
    models_presets: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    new_params: Mapping[str, int | None] | None = None,
    glue_enabled: bool = False,
    plots: bool = False,
    seed: int | None = None,
    anchor_rate: float = 1.0e8,
    clients: Sequence[str] = ("geth", "besu"),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal config dict with sane defaults.

    `models_custom` defaults to a single ``test_arithmetic`` / ``ADD`` spec
    writing ``OPCODE_ADD`` — the happy path. Pass `models_custom=[]` plus
    `models_presets=[...]` for a preset-only config.
    """
    if models_custom is None and models_presets is None:
        models_custom = [copy.deepcopy(_ADD_MODEL_SPEC)]
    gas_costs: dict[str, Any] = {"fork": "osaka"}
    if overrides is not None:
        gas_costs["overrides"] = dict(overrides)
    cfg: dict[str, Any] = {
        "version": 1,
        "anchor_rate": anchor_rate,
        "clients": list(clients),
        "gas_costs": gas_costs,
        "output": {"plots": plots},
        "models": {
            "presets": list(models_presets) if models_presets is not None else [],
            "custom": list(models_custom) if models_custom is not None else [],
        },
    }
    if new_params is not None:
        cfg["new_params"] = dict(new_params)
    if glue_enabled:
        cfg["glue_adjustment"] = {
            "enabled": True,
            "glue_contribution_p_value_threshold": 0.05,
        }
    if seed is not None:
        cfg["modeling"] = {"random_seed": seed}
    if extra:
        for k, v in extra.items():
            cfg[k] = v
    return cfg


def write_standard_inputs(
    tmp_path: Path,
    *,
    fixtures: Sequence[FixtureSpec],
    models: Mapping[str, ClientModel],
    config: dict[str, Any],
    noise_pct: float = 0.003,
    seed: int = 42,
) -> tuple[Path, Path, Path, Path]:
    """Materialize the three input files (config, runtimes, opcounts) + out_dir.

    Returns `(config_yaml, runtimes_csv, opcounts_json, out_dir)` — the same
    shape every test consumes.
    """
    config_yaml = tmp_path / "config.yaml"
    runtimes_csv = tmp_path / "runtimes.csv"
    opcounts_json = tmp_path / "opcounts.json"
    out_dir = tmp_path / "out"

    write_runtimes_csv(runtimes_csv, fixtures, models, noise_pct=noise_pct, seed=seed)
    write_opcounts_json(opcounts_json, fixtures)
    # The runtimes CSV is the ground truth for the client universe, so the
    # config's mandatory `clients` field must match the keys of `models`.
    config = {**config, "clients": list(models.keys())}
    write_config_yaml(config_yaml, config)
    return config_yaml, runtimes_csv, opcounts_json, out_dir


def run_pipeline(
    config_yaml: Path,
    runtimes_csv: Path,
    opcounts_json: Path,
    out_dir: Path,
    *,
    glue: bool = False,
) -> None:
    """Drive the full pipeline through the public API."""
    from evm_gasfit import GasFit

    gas_fit = GasFit.from_config(config_yaml)
    gas_fit.load_runtimes(runtimes_csv)
    gas_fit.load_opcounts(opcounts_json)
    gas_fit.estimate_models()
    if glue:
        gas_fit.estimate_glue()
    gas_fit.build_proposal()
    gas_fit.write_reports(out_dir)


# ----- Canonical output column sets ---------------------------------------
#
# Single source of truth so tests don't drift if a column is renamed.


RESULTS_COLUMNS: frozenset[str] = frozenset(
    {
        "test_name",
        "client_name",
        "target_opcode",
        "source_label",
        "nobs",
        "intercept_runtime_ms",
        "intercept_pvalue",
        "rsquared",
        "rsquared_adj",
        "target_coef_runtime_ms",
        "target_coef_pvalue",
        "target_coef_conf_int_low",
        "target_coef_conf_int_high",
    }
)

NEW_GAS_ALL_PARAMS_COLUMNS: frozenset[str] = frozenset(
    {
        "gas_param",
        "client_name",
        "runtime_ms",
        "pvalue",
        "conf_int_low",
        "conf_int_high",
        "test_name",
        "target_opcode",
        "model_coef_name",
        "source_label",
        "glue_adjustment",
        "rsquared",
        "rsquared_adj",
        "new_gas_decimal",
        "new_gas_rounded",
        "poor_fit",
        "is_winner",
    }
)

NEW_GAS_COLUMNS: frozenset[str] = frozenset(
    {
        "gas_param",
        "client_name",
        "runtime_ms",
        "conf_int_low",
        "conf_int_high",
        "selected_test",
        "selected_opcode",
        "selected_model_coef_name",
        "glue_adjustment",
        "new_gas_decimal",
        "new_gas_rounded",
    }
)

ALWAYS_ON_ARTIFACTS: tuple[str, ...] = (
    "results.csv",
    "new_gas.csv",
    "new_gas_all_params.csv",
    "runtime_estimation_autogenerated_report.md",
    "new_gas_proposal.md",
)

GLUE_ARTIFACTS: tuple[str, ...] = (
    "glue_results.csv",
    "glue_opcodes_by_test.csv",
    "glue_opcodes_autogenerated_report.md",
)


# ----- Small assertion helpers --------------------------------------------


def assert_columns(df: pd.DataFrame, expected: frozenset[str], label: str) -> None:
    missing = expected - set(df.columns)
    assert not missing, f"{label} missing columns: {missing}"


def assert_sentinel_near(
    text: str,
    needle: str,
    sentinel: str,
    *,
    window_chars: int = 200,
) -> None:
    """Assert `sentinel` (case-insensitive) appears within `window_chars` of
    each `needle` occurrence in `text`."""
    for match in re.finditer(re.escape(needle), text):
        window = text[max(0, match.start() - window_chars) : match.end() + window_chars]
        if re.search(re.escape(sentinel), window, flags=re.IGNORECASE):
            return
    raise AssertionError(
        f"sentinel {sentinel!r} not found within {window_chars} chars of {needle!r}"
    )
