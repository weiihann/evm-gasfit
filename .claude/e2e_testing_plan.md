# `evm-gasfit` — e2e testing plan

This document is the contract between the implementation plan (`.claude/implementation_plan.md`)
and the e2e suite under [tests/](../tests/). It enumerates every test file, every
test function, and the plan rules each one pins down. When a behavior rule in
the plan moves, the corresponding tests here move with it.

Tests are the executable spec. When a behavior rule changes, update the failing
test first, watch it fail against the current code, then make it pass.

---

## 1. Shared infrastructure

| File | Purpose |
| --- | --- |
| [`tests/conftest.py`](../tests/conftest.py) | Adds the tests dir to `sys.path` so `_data_synth` is importable. No fixtures shared at the package level — every test uses `tmp_path`. |
| [`tests/_data_synth.py`](../tests/_data_synth.py) | Single source of synthesized inputs, canonical config builder, run helper, output-column constants, and small assertion helpers. |

### `_data_synth.py` exports

- **Fixture builders**: `make_block_limit_fixtures(...)`, `cross_product_fixtures(...)`, `make_glue_driver_fixtures(...)`. The last walks `evm_gasfit.glue.required.PRICED_GLUE_SPECS`, generating one block-limit sweep per family member (so `DUP` yields 16, `PUSH` yields 32). Specs without a driver test (`POP`, `STOP`) are skipped.
- **Inputs + config**: `base_config(...)` (canonical YAML dict with defaults — `models_custom` defaults to the `test_arithmetic`/`ADD` happy spec), `write_standard_inputs(tmp_path, fixtures, models, config, ...)` returns `(config_yaml, runtimes_csv, opcounts_json, out_dir)`.
- **Pipeline runner**: `run_pipeline(config_yaml, runtimes_csv, opcounts_json, out_dir, *, glue=False)` drives the full public API.
- **Column constants**: `RESULTS_COLUMNS`, `NEW_GAS_ALL_PARAMS_COLUMNS`, `NEW_GAS_COLUMNS`, `ALWAYS_ON_ARTIFACTS`, `GLUE_ARTIFACTS`. Tests import these instead of restating column sets locally.
- **Assertion helpers**: `assert_columns(df, expected, label)`, `assert_sentinel_near(text, needle, sentinel, window_chars=200)`.

### `_data_synth.py` invariants

- runtimes CSV column is **`test_runtime_ms`** (LHS of every regression).
- Fixture-name token convention emits `block_limit_million_{N}` (the parser produces a `block_limit_million` column for every row).
- opcounts JSON writes both `opcount` and `data[fixture][TARGET_OPCODE] = opcount` (the equality is enforced at model estimation, once each spec has resolved its `target_opcode`).
- `runtime = intercept + slope·opcount + Σ_i extra_coef_i · opcount · param_i` matches the multi-feature regression form.

### Conventions every test follows

- `pathlib.Path` for all filesystem paths.
- `from __future__ import annotations` and PEP 604 unions.
- No mentions of "Claude", "AI", "the plan", or `.claude/` in test files.
- No mocks for scipy/pandas/numpy — the real stack runs end to end.
- Column conventions (single source of truth):

| Domain | Plan ref | Column |
| --- | --- | --- |
| runtimes CSV | §2.2 | `test_runtime_ms` |
| `results.csv` coef (ms) | §4.3 | `intercept_runtime_ms`, `target_coef_runtime_ms`, `<param>_runtime_ms` |
| `results.csv` coef stats | §4.3 | `intercept_pvalue`, `target_coef_pvalue`, `target_coef_conf_int_low/high`, `<param>_pvalue`, `<param>_conf_int_low/high` |
| `new_gas.csv` / `new_gas_all_params.csv` value | §4.7 | `new_gas_decimal` (full) + `new_gas_rounded` (`ceil`) |
| `glue_results.csv` per-fit runtime | §5 | `glue_runtime_ms` |
| YAML `models:` shape | §2.1, §2.6 | `{presets: [...], custom: [...]}` — at least one non-empty |
| Time-unit conversion | §4.5 | `new_gas_decimal = anchor_rate · runtime_ms / 1000` |

---

## 2. Test files

Each subsection lists (a) the plan rules pinned, (b) the test functions, and (c) any test-specific input notes. All tests build inputs via the `_data_synth` helpers — local fixtures are only described when they vary the standard shape.

### 2.1 [`tests/test_e2e_cli.py`](../tests/test_e2e_cli.py) — CLI exit-code contract (§8)

Plan rules pinned: §8 exit codes 0/1/2; §2.5 mapping of validation outcomes onto exit codes; §4.0 `ConfigError`/`ModelingError` boundary.

Drives the CLI via either the installed `evm-gasfit` script or `python -m evm_gasfit.cli`.

| Test | Exit | Setup |
| --- | --- | --- |
| `test_cli_run_succeeds_and_populates_out` | 0 | Valid invocation; assert every always-on artifact in §5 exists, plus light schema checks. |
| `test_cli_missing_input_returns_exit_code_1` | 1 | `--runtimes` points at a nonexistent path. |
| `test_cli_unknown_override_key_returns_exit_code_1` | 1 | `gas_costs.overrides: {NOT_A_REAL_FIELD: 123}` — strict §2.5 rule: unknown override key = hard `ConfigError`. |
| `test_cli_no_fits_produces_modeling_error_exit_2` | 2 | Single spec with `filter_by: ["opcode_NEVERMATCHES"]` excludes every fixture; per §4.3 the spec is skipped → no rows in `results.csv` → §4.0 `ModelingError`. |

### 2.2 [`tests/test_e2e_happy_path.py`](../tests/test_e2e_happy_path.py) — minimal pipeline and Prague comparison (§2–§5)

Plan rules pinned: §2.1 config shape; §2.2/§2.3 input loaders; §2.4 Prague
baseline selection and fork-current comparison values; §4.2 single NNLS pass per
`(spec, group, client)`; §4.3 `results.csv` schema; §4.5 time conversion; §4.6
worst-case across clients + winning-row provenance; §4.7 rounding; §5 always-on
artifacts and glue/plots negative checks.

| Test | Coverage |
| --- | --- |
| `test_minimal_pipeline_end_to_end` | One spec (`test_arithmetic` → `OPCODE_ADD`), two clients with distinct slopes, glue off, plots off. Asserts: always-on artifacts; no glue or fig artifacts; `results.csv` column set + 2 rows + slope-within-5% recovery + `rsquared > 0.95`; `new_gas_all_params.csv` schema (including `rsquared` / `rsquared_adj` carried verbatim from `results.csv`, plus the `is_winner` flag) + uniformly-zero `glue_adjustment` + boolean `poor_fit`; `new_gas.csv` schema + worst-case row equals one row in `new_gas_all_params.csv` verbatim (§4.6 provenance); `new_gas_decimal = anchor_rate · runtime_ms / 1000`; `new_gas_rounded == ceil(new_gas_decimal)`. |
| `test_prague_current_gas_comparison_end_to_end` | Runs the same public pipeline under `gas_costs.fork: prague`; verifies the mandatory comparison records `OPCODE_ADD`'s Prague current gas (`3`) as a fork baseline. |

### 2.3 [`tests/test_e2e_multi_model.py`](../tests/test_e2e_multi_model.py) — multi-spec, groups, multi-feature (§2.1, §4.3, §4.6, §5.1)

Plan rules pinned: `target_operation` vs `target_operation_param`; `model_by` grouping; multi-feature regression; figure naming with `__` join separator and per-segment sanitization; `new_gas` worst-case-across-clients copies a single `new_gas_all_params` row verbatim.

| Test | Coverage |
| --- | --- |
| `test_multi_model_with_target_param_and_groups` | Two specs run together: (a) `test_arithmetic` with `target_operation_param: opcode` and `model_by: opcode` over `{ADD, SUB, MUL}` → `OPCODE_GENERIC`; (b) `test_account_access` with `target_operation: BALANCE` and `model_by: cache_strategy` over `{NO_CACHE, HOT}` → `GAS_WARM_ACCESS`. Two clients, `output.plots: true`. Asserts: `results.csv` has `(3+2) × 2 = 10` rows; the `param_opcode` and `param_cache_strategy` columns appear (raw parsed-param `model_by` entries land on output CSVs under the `param_` prefix that `fixtures_df` exposes them under, see implementation plan §4.1); every fit's `target_coef_runtime_ms` is within 5% of the planted slope; PNGs exist under `figs/runtime/` and each filename splits into exactly five `__`-joined segments matching the §5.1 contract; runtime report embeds `figs/runtime/`; `new_gas_all_params.csv` carries **every** candidate fit (`3×2 + 2×2 = 10` rows across the two params, not just the per-client winners) with exactly one `is_winner = true` row per `(gas_param, client)` (`2×2 = 4` winners); `new_gas.csv` carries both gas params and the worst-case row equals exactly one row in `new_gas_all_params.csv` on the provenance columns. |
| `test_multi_feature_regression_recovers_extra_coefficient` | One spec with `model_params: {target_coef: COLD_STORAGE_WRITE, value_sent: STORAGE_WRITE_PER_VALUE}` and a cross product of `value_sent ∈ {"1","5","10"}` × the block-limit grid. Asserts: 1 row in `results.csv`; `target_coef_runtime_ms` ≈ 1.0e-5 and `value_sent_runtime_ms` ≈ 2.0e-6 (both within 5%); the per-feature stat columns are present; `new_gas.csv` carries both gas params. |

Note: the PNG-count assertion was relaxed from `len == 30` (a `3 specs × 2 clients × 3 families` arithmetic check) to "PNGs exist" + per-filename naming check, which is the actual contract. The spot-check on `ADD__test_arithmetic__ADD__geth__{family}.png` still pins the family names.

### 2.4 [`tests/test_e2e_glue.py`](../tests/test_e2e_glue.py) — glue adjustment (§4.4)

Plan rules pinned: glue off by default; the priced canonical-name set is sourced from `evm_gasfit.glue.required.PRICED_GLUE_SPECS` (skipping specs with `test_name=None`); the required-driver-fixture check covers only `pure`/`cycle` drivers — mixed-tier drivers are never required; family aggregation collapses `DUP1`..`DUP16` etc. into one canonical row; the four-pass tier order (`pure → cycle → mixed_a → mixed_b`) and the LHS partner-subtraction formula. `_data_synth.ClientModel.glue_coefs` plants real per-opcode runtime contributions so the recovery tests below can isolate the subtraction effect.

| Test | Coverage |
| --- | --- |
| `test_glue_disabled_skips_files` | Glue toggle omitted (defaults off). Asserts: `glue_results.csv`, `glue_opcodes_by_test.csv`, `glue_opcodes_autogenerated_report.md` absent; `new_gas_all_params.csv` keeps a `glue_adjustment` column that is uniformly zero. |
| `test_glue_enabled_writes_files_and_adjusts_slope` | Derives the active priced-opcode set from `PRICED_GLUE_SPECS` (POP/STOP excluded since they have no driver). Synthesizes main-model fixtures with an `ISZERO` background plus driver fixtures for every spec with a driver via `make_glue_driver_fixtures()`. Enables glue with p-value threshold 0.05. Asserts: glue CSVs + glue markdown exist; `glue_results.csv` columns ⊇ `{client_name, glue_opcode, nobs, glue_runtime_ms, p_value, rsquared}`; the `glue_opcode` set equals the active priced set (all 30 names including the 16 mixed-A and 2 mixed-B entries, since the four-pass loop emits a row per spec even when the slice is empty); `new_gas_all_params.csv` row for `OPCODE_ADD` has `glue_adjustment > 0` and `new_gas_decimal = anchor_rate · runtime_ms / 1000` still holds with the post-adjustment `runtime_ms`. |
| `test_glue_family_collapses_to_single_canonical_row` | Confirms `DUP1`..`DUP16` driver fixtures collapse to exactly one `glue_opcode == "DUP"` row per client (no per-member leakage) and that the recovered slope matches the synthetic model. |
| `test_glue_missing_required_test_raises` | Glue enabled but no glue-required-test fixtures. Asserts any failure surfacing during load or pre-estimation (`pytest.raises(Exception)`). |
| `test_glue_mixed_a_recovers_planted_slope_after_partner_subtraction` | Plant a real ISZERO glue contribution into the ADD runtimes (`ClientModel.glue_coefs={ADD: 2e-5, ISZERO: 1e-5, ...}` over `_main_fixtures(extra_per_million={ISZERO: 500_000})` plus pure+cycle drivers). Asserts: `glue_results.csv` ISZERO `glue_runtime_ms ≈ 1e-5`; `glue_results.csv` ADD `glue_runtime_ms ≈ 2e-5` (clean recovery after LHS subtraction); `results.csv` ADD `target_coef_runtime_ms ≈ 2.5e-5` (the contaminated slope, demonstrating the modelspec-side absorbs both costs). |
| `test_glue_mixed_b_recovers_planted_slope_after_mixed_a_partner` | Mixed-B partners may come from mixed-A: contaminate `test_keccak_diff_mem_msg_sizes` with MSTORE and provide a clean MSTORE modelspec/driver slice. Asserts: `glue_results.csv` MSTORE `glue_runtime_ms ≈ 1.5e-5` (mixed-A clean fit); `glue_results.csv` KECCAK256 `glue_runtime_ms ≈ 3e-5` (clean recovery after subtracting the MSTORE partner produced earlier in the same pass). |
| `test_mixed_glue_partner_draws_reach_adjusted_target_interval` | Upstream supporting-cost bootstrap draws propagate through the mixed fit into the adjusted target interval rather than becoming a fixed point subtraction. |
| `test_glue_mixed_a_does_not_subtract_other_mixed_a_partners` | Tier guard: contaminate ADD with MSTORE so the detector surfaces MSTORE as an ADD partner; assert ADD's mixed-A fit ignores MSTORE (since both are tier `mixed_a`, the static four-pass order forbids the subtraction) and recovers the contaminated slope `2e-5 + 0.5·1.5e-5 = 2.75e-5`. Guards against regressions of `allowed_partner_tiers` enforcement. |
| `test_glue_works_with_benchmark_sweep_token` | Same as `test_glue_enabled_writes_files_and_adjusts_slope` but the scan axis is named `benchmark_<N>M` instead of `block_limit_million_<N>`. Sanity-checks the parser/detector don't depend on a specific token name. |
| `test_glue_missing_optional_driver_does_not_raise` | Driver fixtures present for every required spec; POP/STOP remain absent (they have no driver in any dataset). Asserts: pipeline runs to completion; POP/STOP are absent from `glue_results.csv` rather than appearing as NaN rows. |

### 2.4a [`tests/test_e2e_overhead_baseline.py`](../tests/test_e2e_overhead_baseline.py) — baseline controls (§2.1, §4.3, §4.4)

Plan rules pinned: `overhead_baseline_param` pairs `False` work against
aggregated `True` controls by client and session plus either all other params
(legacy `null`), no optional shape params (`[]`), or the explicit shape list.
The pairing step differences runtime, `opcount`, and all opcode counts before
the target invariant. That permits a native control that executes fixed system
tail work while preserving its variable target count after subtraction.
Unmatched selected controls and invalid match-key configurations are
`ConfigError`s; glue detection runs over the same paired delta.

| Test | Coverage |
| --- | --- |
| `test_baseline_pair_cancels_shared_contamination` | A shared contaminant is cancelled without a glue adjustment. |
| `test_baseline_pair_cancels_matching_contaminant_out_of_glue_detection` | A matching contaminant becomes a constant and is not detected as glue. |
| `test_baseline_pair_still_prices_non_cancelling_glue_opcode` | A False-only calling-convention count survives pairing and is priced as glue. |
| `test_baseline_pair_aggregates_repeated_true_trials` | Repeated controls aggregate before pairing. |
| `test_baseline_pair_raises_on_unmatched_false_row` | An unmatched work row raises `ConfigError`. |
| `test_baseline_pair_accepts_csv_boolean_flags` | Explicit boolean CSV flags preserve control pairing and recover the planted marginal cost. |
| `test_native_nonzero_control_counts_are_subtracted_with_explicit_match_key` | A zero-work control with 18 fixed ADDs pairs against 18 + n measured ADDs using `overhead_baseline_match_params: []`; the invariant sees n and recovers the planted slope. |
| `test_baseline_match_params_require_a_baseline_flag` | A match-key list without `overhead_baseline_param` is rejected during config loading. |

### 2.5 [`tests/test_e2e_overrides_derived_plots.py`](../tests/test_e2e_overrides_derived_plots.py) — overrides, derived params, plots toggle (§2.4, §4.7, §4.8, §5.1)

Plan rules pinned: `gas_costs.overrides` patches the fork defaults and flows to the proposal diff; both `derived:` forms (alias and `{formula: ...}`) evaluate against the worst-case integer table in declaration order; identifier-not-found inside a derived formula is a load-time error (§4.8); the plots-on / plots-off branches in §5.1.

| Test | Coverage |
| --- | --- |
| `test_gas_overrides_flow_to_proposal` | Patches `COLD_STORAGE_ACCESS` to `9999`; the value appears in `new_gas_proposal.md`. |
| `test_derived_alias_and_formula_evaluate` | Declares three derived entries (alias, formula, chained formula). Asserts all three names appear in `new_gas.csv`, and each rounded value satisfies the §4.8 evaluation rule (integer worst-case table, `ceil`). |
| `test_pricing_scenarios_evaluate_derived_params_in_declaration_order` | Every explicit pricing scenario evaluates qualified aliases and chained derived parameters in declaration order. |
| `test_plots_toggle[True]` / `test_plots_toggle[False]` | Parametrized. `plots: true` writes PNGs under `figs/runtime/` and the runtime report embeds at least one `![…]` markdown image. `plots: false` skips figs and embeds. |
| `test_derived_formula_unknown_identifier_is_load_time_error` | A `derived:` formula identifier that doesn't resolve at its declaration point (typo `COLD_STORAGE_WIRTE`) raises at `GasFit.from_config(...)` — not later. |

### 2.6 [`tests/test_e2e_presets.py`](../tests/test_e2e_presets.py) — model preset registry (§2.6)

Plan rules pinned: a preset is a named, frozen `ModelSpec` shipped in `defaults/models.py`; `models.presets:` is resolved at load time and concatenated with `models.custom:`; selection is per-preset, never field-level; unknown preset = config error; duplicate `test_name` across `presets:` and `custom:` is allowed (independent fits); **byte-identical resolved specs = config error**; empty `presets:` *and* empty `custom:` = config error. Specs sharing `test_name` + target + `model_by` and differing only in `filter_by` are routed by `source_label`, not collapsed — see [`tests/test_e2e_spec_collision.py`](../tests/test_e2e_spec_collision.py).

The canonical preset under test is `arithmetic_add` (`test_arithmetic` → `ADD` → `OPCODE_ADD`).

| Test | Coverage |
| --- | --- |
| `test_preset_only_config_runs_pipeline` | `models: {presets: ["arithmetic_add"], custom: []}` runs the full pipeline; `OPCODE_ADD` appears in `new_gas.csv`. |
| `test_preset_plus_custom_concatenate` | One preset + one custom (different `test_name` / target); both gas params appear in `new_gas.csv`. |
| `test_invalid_models_section_is_config_error[unknown_preset_name]` / `[empty_presets_and_empty_custom]` | Parametrized: both invalid `models:` shapes raise at `GasFit.from_config(...)`. |
| `test_duplicate_test_name_across_preset_and_custom_is_allowed` | A preset and a custom spec both target `test_arithmetic`/`ADD` but write to different gas params; both rows appear in `new_gas.csv`. |

### 2.7 [`tests/test_e2e_fixture_params.py`](../tests/test_e2e_fixture_params.py) — derived fixture params (§2.7)
Plan rules pinned: `fixture_params:` materializes per-spec derived columns from
raw parameters; a runtimes CSV may provide `param_<name>` where a canonical
workload parameter is absent from the fixture ID. Explicit and parsed values
coalesce only when equal; a conflict is rejected rather than producing pandas
suffix columns.

| Test | Coverage |
| --- | --- |
| `test_fixture_param_rename_passes_through_numeric_value` | Rename-only derived parameter is numeric in the regressor. |
| `test_fixture_param_value_remap_translates_strings` | String parameter values are remapped before fitting. |
| `test_two_specs_can_share_derived_name_with_different_sources` | Per-spec derived columns remain isolated. |
| `test_uncertainty_gate_rejects_zero_pinned_mapped_coefficient` | A mapped coefficient pinned at zero cannot pass the uncertainty gate merely because the primary target coefficient is precise. |
| `test_unmapped_source_value_raises` | An incomplete value map fails at fit time. |
| `test_explicit_runtime_param_coalesces_with_fixture_identity` | Equal CSV and fixture-ID facts become one usable raw parameter column. |
| `test_explicit_runtime_param_conflict_is_rejected` | Conflicting CSV and fixture-ID facts raise `ConfigError`. |

### 2.7a [`tests/test_e2e_bytes_to_words.py`](../tests/test_e2e_bytes_to_words.py) — `bytes_to_words` fixture-param transform (§2.7)

Plan rules pinned: a `fixture_params` entry may carry `transform: bytes_to_words` to apply `ceil(x / 32)` to a byte-sized raw param before the regression consumes it; the fitted coefficient on the derived column then directly recovers the per-word slope. `transform` and `values` are mutually exclusive.

| Test | Coverage |
| --- | --- |
| `test_bytes_to_words_transform_recovers_per_word_coefficient` | Copy-style sweep with `calldata_size ∈ {32, 64, 96, 128}`; spec declares `fixture_params.calldata_words = {source: calldata_size, transform: bytes_to_words}` and `model_params.calldata_words → OPCODE_CALLDATACOPY_PER_WORD`; the recovered `calldata_words_runtime_ms` matches the planted per-word slope; both BASE and PER_WORD gas params land in `new_gas.csv`. |
| `test_transform_and_values_are_mutually_exclusive` | A spec setting both fields raises at config load. |

### 2.7b [`tests/test_e2e_anchor_filter.py`](../tests/test_e2e_anchor_filter.py) — `filter_by` carve-out semantics (§2.6 / catalog)

Plan rules pinned: (1) when an opcode's `opcode_<X>` token is a prefix of another opcode's token in the same `test_name` slice, the catalog preset must carry a trailing `-` anchor (`filter_by: [opcode_<X>-]`) so the substring match cannot leak the sibling's fixtures into the fit; (2) a `!`-prefixed token in `filter_by` inverts the substring match — `!foo` requires `foo` to be absent from the fixture name — and is ANDed with the positive tokens.

| Test | Coverage |
| --- | --- |
| `test_anchored_filter_excludes_prefix_sibling[add_vs_addmod]` / `[mul_vs_mulmod]` / `[push0_vs_push1]` | Parametrized over the three documented overlap pairs. Synthesizes fixtures for *both* the target opcode and the sibling under one `test_name`, drives the corresponding catalog preset (`arithmetic_add`, `arithmetic_mul`, `stack_push0`), and asserts `results.csv` carries only the target opcode and the recovered slope matches the target's planted value — not the sibling's (5× larger). |
| `test_negation_token_excludes_overlapping_sibling` | Synthesizes ADD + ADDMOD fixtures under `test_arithmetic` with distinct slopes (5× apart) and drives a custom spec whose `filter_by: ["opcode_ADD", "!opcode_ADDMOD"]` pairs a positive substring that would match both opcodes with a negation that excludes the sibling. Asserts `results.csv` carries only `ADD` and the recovered slope matches `ADD`'s planted value — had the negation been ignored, the fit would pull toward `ADDMOD`'s 5× slope. |

### 2.7d [`tests/test_e2e_param_opcode_collision.py`](../tests/test_e2e_param_opcode_collision.py) — parsed-param column name cannot collide with opcode mnemonic (§4.1)

Plan rules pinned: parsed-param tokens land on `fixtures_df` under the `param_<key>` prefix so they cannot clash with opcode-mnemonic columns from the opcounts merge. Synthesizes fixtures whose parametrization token (`ADD_same` / `ADD_diff`) is parsed under the partition fallback to `{ADD: ...}` — without the prefix the resulting `ADD` column would collide with the `ADD` opcode count and the merge would emit `ADD_x`/`ADD_y`.

| Test | Coverage |
| --- | --- |
| `test_parsed_param_does_not_collide_with_opcode_column` | Single spec on `test_arithmetic` with `target_operation: ADD`, `model_by: ADD`, and a parametrization grid over `{ADD: ["same", "diff"]}`. Asserts: `run_pipeline` completes; `results.csv` carries a `param_ADD` column with both values; no `ADD_x` or `ADD_y` columns leak through. |

### 2.7e [`tests/test_e2e_spec_collision.py`](../tests/test_e2e_spec_collision.py) — specs colliding on `(test_name, target, model_by)` (§2.6, §4.6)

Plan rules pinned: `results.csv` carries a `source_label` provenance column; the proposal aggregator routes each row back to its producing spec by `source_label`, so two specs that share `test_name` + target + `model_by` and differ only in `filter_by` neither duplicate nor cross-contaminate each other's candidates (the shipped `cold_account_nocode_access` / `cold_account_code_access` shape); byte-identical resolved specs are a config error.

| Test | Coverage |
| --- | --- |
| `test_filter_by_only_collision_does_not_duplicate_or_contaminate` | Two custom specs on `test_account_access`/`BALANCE`/`model_by: cache_strategy` writing the same `GAS_ACCESS`, partitioning `{NO_CACHE, HOT, COLD}` by `filter_by` (`!HOT` vs `!COLD`, overlapping on `NO_CACHE`). Asserts on `new_gas_all_params.csv`: `source_label` distinguishes the two specs (`models.custom[0]` / `models.custom[1]`); each spec owns only the strategies its `filter_by` keeps (no contamination) at 4 rows each; no row is duplicated on the full identity key including `source_label`; exactly one `is_winner` per `(gas_param, client)`. |
| `test_byte_identical_specs_are_a_config_error` | The same custom spec listed twice raises `ConfigError` (`duplicate model spec`) at `GasFit.from_config(...)`. |

### 2.7c [`tests/test_e2e_catalog_smoke.py`](../tests/test_e2e_catalog_smoke.py) — full preset catalog smoke (§2.6)

Plan rules pinned: every preset in `defaults/models.py::PRESETS` must pass Pydantic validation and drive an end-to-end pipeline run without raising. The goal is to catch typos in `test_name` / `target_operation` / `model_params` keys at landing time, not to validate fit quality.

| Test | Coverage |
| --- | --- |
| `test_every_catalog_preset_fits_without_raising` | Programmatically synthesizes one block-limit sweep per preset by introspecting each `ModelSpec` (covers `target_operation_param`, `model_by`, `fixture_params` sources, non-target `model_params` coefs, precompile-style `target_operation_count_source`, and — for `cold_account_code_*_noncall` — a paired `overhead_baseline_True` fixture per sweep point). Loads all preset names into one config, runs the pipeline, asserts every preset's `test_name` produces at least one row in `results.csv` and `new_gas.csv` is non-empty. |
| `test_catalog_requires_new_params_declaration` | A preset-only config that omits the catalog's non-raw `model_params` RHS names (`OPCODE_*COPY_PER_WORD`, `COLD_ACCOUNT_{NOCODE,CODE}_ACCESS`, `COLD_ACCOUNT_{NOCODE,CODE}_WRITE`) fails to load with a hard `ConfigError`; declaring them lets the config load cleanly with no warnings. |

### 2.7f [`tests/test_e2e_joint_delta.py`](../tests/test_e2e_joint_delta.py) — joint worst-case pricing for access deltas (catalog / §4.8)

Plan rules pinned: the state-access write cost is fitted as a *combined* access+write param (`COLD_STORAGE_WRITE`, `COLD_ACCOUNT_*_WRITE`) selected via `filter_by` on the `write_new_value`/`value_sent` tokens, and the write delta is recovered in `derived` as `max(0, combined − access)`. This bounds the combined op by a single worst-case client (tighter than the subadditive sum of two independent per-param maxima) and floors a degenerate negative delta at zero.

| Test | Coverage |
| --- | --- |
| `test_storage_write_joint_is_tighter_than_independent_max` | Two clients whose worst access and worst combined-write come from *different* clients. Asserts `STORAGE_WRITE == max(0, COLD_STORAGE_WRITE − COLD_STORAGE_ACCESS)` and that it is strictly below the naive per-client `max(combined − access)` read off `new_gas_all_params.csv`. |
| `test_storage_write_clamps_to_zero` | Combined write measures below the cold access for every client; asserts the derived `STORAGE_WRITE` floors at `0`. |
| `test_account_write_takes_worst_context` | Two account contexts (nocode/code) with different write deltas; asserts `ACCOUNT_WRITE == max(0, code_delta, nocode_delta)` equals the worst (code) context. |
| `test_joint_delta_survives_glue_adjustment` | Regression for the glue-adjustment key collision: with glue enabled, the read/write specs share `(test_name, target, model_by)` and differ only in `filter_by`. Asserts the write spec keeps its own (higher) coefficient — the glue adjustment df is keyed on `source_label` — so the delta does not collapse to 0. |

### 2.8 [`tests/test_e2e_determinism.py`](../tests/test_e2e_determinism.py) — determinism contract (§4.0)

Plan rules pinned: "given identical inputs and the same `random_seed`, all CSV and markdown outputs are byte-identical across runs and across platforms"; bootstrap sampling threads the seed into every `numpy.random.Generator`; PNGs are **not** promised byte-identical.

| Test | Coverage |
| --- | --- |
| `test_two_runs_with_same_seed_produce_byte_identical_csvs_and_md` | Two pipeline runs with the same inputs and `random_seed: 42` into distinct `out_dir`s; every always-on CSV and MD compares byte-equal via `Path.read_bytes()`. Markdown artifacts are normalized through `_normalize_for_compare` first — the `_Generated YYYY-MM-DD HH:MM:SSZ` header in `new_gas_proposal.md` is wall-clock metadata, not computed content, and is replaced with `_Generated <TIMESTAMP>` before comparison. |
| `test_different_seed_produces_different_bootstrap_outputs` | Seeds 42 vs 99 against identical inputs; at least one of `target_coef_conf_int_low/high` or `target_coef_pvalue` differs. |
| `test_glue_on_pipeline_is_also_deterministic` | Same byte-equality check as test 1 but with glue enabled, so `glue_results.csv`, `glue_opcodes_by_test.csv`, and `glue_opcodes_autogenerated_report.md` are also compared. |

### 2.9 [`tests/test_e2e_proposal_warnings.py`](../tests/test_e2e_proposal_warnings.py) — warnings + sentinel rendering (§2.5, §4.0, §4.4, §4.6)

Plan rules pinned (merged from three earlier files — see §3):

- §2.5 lenient `model_params` RHS naming an unknown gas-param: pipeline runs, warning fires on the `evm_gasfit` logger, the name appears in `new_gas.csv` + the Warnings section of `new_gas_proposal.md`.
- §4.6 "no prior default" sentinel co-located with the unknown name in either CSV or markdown; no isolated `| 0 |` table cell; a known fork field must NOT carry the sentinel.
- §4.4 missing-glue detection: a non-priced opcode meeting corr/ratio thresholds emits a warning and surfaces in the proposal; a priced opcode does not.
- §4.4 glue-contribution dual-gate: `compute_glue_adjustment` and the mixed-tier partner LHS subtraction skip a glue opcode's contribution on a given client whenever its fit fails either `p_value < glue_contribution_p_value_threshold` or `rsquared >= glue_contribution_rsquared_threshold`. A skipped contribution leaves the target coefficient holding that glue's runtime — visible on `new_gas_all_params.csv` as `glue_adjustment == 0` for the (gas_param, client) row whose fit would otherwise have applied it.
- §5 `### Missing glue adjustments` carries a second sub-block listing priced glue opcodes whose per-client fit failed those same gates, with the failing clients (tagged `p-value` / `R²` / `both`) and the dependent gas params on each row. The sub-block is wrapped in a collapsible `<details>` element whose `<summary>` carries the row count, mirroring the non-priced-opcodes sub-block above it.
- §4.6 poor-fit thresholds: `### Winners with poor fit` lists winners whose `poor_fit = true` (the selector fell back because no candidate passed both thresholds); each row carries a `Failed` cell with `p-value`, `R²`, or `both`. `### Other weak candidates` lists losing candidates that failed at least one threshold, grouped under one `<details>` block per gas_param (summary `<gas_param> — N weak combos`) with body columns `Test`, `Target opcode`, `Coef`, `Combo`, `Failing clients` — clients collapse into one cell each tagged `(p-value)` / `(R²)` / `(both)`, and `Combo` lists the `model_by` factors that vary within the block as `k=v / k=v` (or `—` when none vary).

| Test | Coverage |
| --- | --- |
| `test_unknown_gas_param_emits_warning_and_renders_sentinel[BRAND_NEW_GAS_PARAM]` / `[OPCODE_NEVER_EXISTED_GAS]` | Parametrized over two distinct unknown gas-param names so each independently catches a regression. Single spec, two clients. Asserts: pipeline completes; `caplog.at_level(logging.WARNING, logger="evm_gasfit")` captures a WARNING mentioning the param; row in `new_gas.csv`; Warnings heading in proposal mentions the param; `assert_sentinel_near(...)` finds `no prior default` within 200 chars of the name; no `\|\s*0\s*\|` table cell on rows mentioning the name; any present `current_gas`/`diff`/`gas_diff` column is not numeric 0. |
| `test_known_gas_param_does_not_carry_sentinel` | Counterpart with `OPCODE_ADD`. Asserts the sentinel does NOT appear in the CSV row text or any markdown line mentioning `OPCODE_ADD`. |
| `test_non_priced_glue_candidate_emits_warning_and_appears_in_proposal` | Main `test_arithmetic` / `ADD` fit contaminated with `ADDMOD` counts exactly proportional to the target opcount (corr ≈ 1.0), plus `make_glue_driver_fixtures()`. Glue enabled. Asserts: derived priced set from `PRICED_GLUE_OPCODES` does NOT contain `ADDMOD`; at least one WARNING on an `evm_gasfit`-namespaced logger naming `ADDMOD`; `ADDMOD` appears in `new_gas_proposal.md`. |
| `test_priced_glue_opcode_does_not_emit_missing_glue_warning` | Sanity counterpart with `POP` (priced) as the contaminant. Asserts no `evm_gasfit`-namespaced WARNING names `POP`. |
| `test_poor_fit_glue_opcodes_surface_under_missing_glue_section` | Glue enabled; plant heavy noise on one client's driver fixtures for one glue opcode so its per-client fit fails the `glue_contribution_rsquared_threshold` while the other clients fit cleanly. Drive ADD with a single custom modelspec so the glue→gas-param join is unambiguous. Asserts: (a) `### Missing glue adjustments` renders; under that heading, a collapsible `<details>` block whose `<summary>` starts with `<b>Priced glue opcodes with a poor fit</b>` wraps a `Glue opcode \| Affected clients \| Affected gas params` table; the row for the noisy glue opcode names only the noisy client (tagged with the failure label) and lists `OPCODE_ADD` in the gas-params cell; clean glue opcodes do not appear in that table. (b) On `new_gas_all_params.csv`, the OPCODE_ADD row for the clean client carries a positive `glue_adjustment` while the noisy client's row carries `glue_adjustment == 0` — confirming the R² gate skipped the contribution rather than applying a noisy slope. |
| `test_poor_fit_section_surfaces_winners_and_losing_candidates` | Two specs target the same gas param; one client gets a noisy fit that fails the R² threshold while passing p-value, and a second client has a low-R² candidate that loses selection to a clean alternative. Asserts: `new_gas_all_params.csv` carries `rsquared` / `rsquared_adj` columns; the noisy fallback winner is the only row with `poor_fit = true` **and** `is_winner = true` (since `poor_fit` now flags every failing candidate, the winner is isolated via `is_winner`); `## Poor-fit selections` → `### Winners with poor fit` lists that row with `Failed` = `R²`; `### Other weak candidates` renders one `<details>` block per affected gas_param, with the losing low-R² candidate appearing inside that block tagged `(R²)` in its `Failing clients` cell; when a `model_by` factor varies across the weak losers the corresponding `k=v` substring surfaces in the block's `Combo` cell; both subsections render `_None._` when the run carries no weak fits. |

### 2.9a [`tests/test_e2e_campaign_analysis.py`](../tests/test_e2e_campaign_analysis.py) — campaign eligibility, inference, and provenance (§2.2, §4.2a, §5, §8)

Plan rules pinned: campaign metadata is admissibility/provenance only, not a
design feature; eligible fitting is restricted to executed
qualification/performance rows that passed correctness; session bootstrap is
cluster-aware; anchorless analyses do not manufacture gas prices; comparison
requires complete matching non-software manifest factors.

| Test | Coverage |
| --- | --- |
| `test_campaign_eligibility_session_inference_and_anchorless_outputs` | Two sessions duplicate a clean ADD sweep with numeric bookkeeping metadata. Incorrect, diagnostic, pilot, warmup, and failed rows remain in the eligibility ledger but do not fit; the qualified model reports two sessions, anchorless prices are null, and provenance embeds the exact config and manifest. |
| `test_campaign_config_rejects_ineligible_calibration_overrides` | Campaign configuration cannot admit non-performance phases or failed status, or disable correctness evidence. |
| `test_campaign_missing_correctness_evidence_fails_closed` | Structured campaign rows without their required correctness column fail rather than being treated as verified. |
| `test_single_session_campaign_is_inconclusive_without_row_bootstrap` | One complete ADD sweep carries a single session ID. Asserts the point fit is retained for audit while bootstrap inference is unavailable, qualification is inconclusive even without an explicit `min_sessions` gate, and a blocked price is empty. |
| `test_campaign_with_no_eligible_rows_writes_inconclusive_analysis` | Every campaign runtime is failed, has no duration, and opcounts are empty. Asserts eligibility retains every exclusion, `results.csv` is a schema-correct empty successful-fit table, every configured model is inconclusive in `qualification.csv` and `analysis_status.json`, and unresolved parameters have no recommended price. |
| `test_missing_configured_client_is_recorded_as_inconclusive` | A configured client without measurements still receives an inconclusive planned-model record. |
| `test_unavailable_campaign_curvature_is_inconclusive` | Collapsed fitted values cannot satisfy the mandatory residual-curvature gate. |
| `test_campaign_comparison_allows_software_only_difference` | Complete manifests differ only in software revision. Asserts a comparison artifact is produced and reports the software delta. |
| `test_campaign_comparison_rejects_empty_manifest_factor` | An empty hardware factor is incomplete provenance, not evidence of equality; comparison fails before its output directory exists. |
| `test_campaign_comparison_rejects_hardware_mismatch_before_output` | Two otherwise identical completed analyses carry manifests with different hardware factors. Asserts comparison raises `ConfigError` naming hardware and does not create its output directory. |
| `test_campaign_comparison_rejects_unavailable_schedule_identity` | A nested unavailable gas-schedule identity blocks comparison despite matching fork labels. |
| `test_campaign_comparison_rejects_incompatible_analysis_method` | Different archived analysis-config hashes or analyzer versions block comparison. |
| `test_campaign_comparison_rejects_unverified_new_gas_csv` | A changed proposal CSV whose bytes no longer match its archived hash is rejected before comparison output. |

### 2.10 [`tests/test_e2e_report_format.py`](../tests/test_e2e_report_format.py) — `new_gas_proposal.md` formatting + structural contract (§5)

Plan rules pinned: the headline ordering, column headers, diff-cell rendering, sentinel handling, anchor-rate formatting, gas-param row ordering, and the heatmap embed location described in §5's `new_gas_proposal.md` row. The proposal heatmap colormap rule (`log2(proposed / current)` with `RdYlGn_r` and a blank row for `null` baselines) is exercised end-to-end here.

| Test | Coverage |
| --- | --- |
| `test_anchor_rate_renders_as_mgas_per_second` / `test_anchor_rate_three_sig_fig_smart_format` | `anchor_rate` in the run-metadata line renders as `<N> Mgas/s` with 3 significant figures (`1.0e8 → 100 Mgas/s`, `1.234e8 → 123 Mgas/s`). |
| `test_proposed_table_column_order_and_headers` | `## Proposed gas parameters` table columns are `Gas param \| Current gas \| Proposed gas \| Diff \| Diff %` in that order, title-cased. |
| `test_diff_percent_column_computed_against_current` | Each fitted row's `Diff %` cell equals `round((proposed - current) / current * 100)` with a signed integer percent. |
| `test_new_param_diff_pct_renders_na` | A `new_params: {NAME: null}` row renders `no prior default` in `Current gas` and `n/a` in both `Diff` and `Diff %` (no current baseline to ratio against). |
| `test_client_comparison_section_present_and_populated` | `## Client comparison` sits between `## Proposed gas parameters` and `## Warnings`; its header row matches the documented column order; rows carry worst-then-second-worst clients with a `Ratio` cell (`worst gas / second-worst gas`) formatted as `1.23×`. |
| `test_heatmap_embedded_in_client_comparison` | When `output.plots: true`, the `![](figs/proposal/heatmap.png)` embed lives inside the `## Client comparison` section (no trailing `## Plots` heading, no `by_client.png`). |
| `test_null_baseline_param_in_heatmap_emits_warning` | A model writing to a `new_params: {NAME: null}` entry with plots enabled surfaces a `null-baseline: new_params['NAME']` warning under `## Warnings` and still produces the heatmap PNG (that row renders blank, annotations only). |
| `test_null_baseline_warning_absent_when_all_baselines_known` | The standard config (every gas param backed by a raw fork field) emits no `null-baseline:` warning — the new warning is gated on actual null-baseline declarations. |
| `test_partial_fits_subsection_calls_out_clients_with_no_fits` | A client declared in `config.clients` (§2.1) but absent from the runtimes CSV surfaces under `### Incomplete client coverage` both as a dedicated `Clients with no estimations at all:` callout above the per-param table and as a missing entry on every per-param row. Pins the expected-universe semantics: the partial-fits logic measures against `config.clients`, not whichever clients happened to appear in the CSV. |
| `test_gas_params_follow_config_declaration_order` | Proposed-params table, client-comparison table, and `new_gas.csv` list gas params in the order each first appears in `models.custom` (then `derived` keys) — not alphabetically. Picks `SUB, ADD, MUL` as non-alphabetical opcodes to make the assertion meaningful. |
| `test_provenance_section_present_with_per_param_heatmaps` | When `output.plots: true` and at least one gas param has ≥ 2 distinct model combos, a top-level `## Worst-case provenance per gas param` section sits between `## Client comparison` (after the overview heatmap embed) and `## Warnings`. Each qualifying gas param renders a `<details>` block whose `<summary>` carries the param name and whose body embeds `figs/proposal/provenance__<gas_param>.png`. The PNG exists for each qualifying param. |
| `test_provenance_section_renders_tables_when_plots_disabled` | `output.plots: false` still renders the `## Worst-case provenance per gas param` section: each qualifying gas param's `<details>` body carries a markdown table (combo rows × client columns, cells = `new_gas_rounded` for every per-client candidate; the (combo, client) cell the per-client selector picked is rendered in `**bold**`) instead of a heatmap embed, and no `provenance__*.png` figures are written. The combo-row labels share the heatmap's labeling helper, so they match the plots-on output verbatim. |
| `test_overview_table_replaces_heatmap_when_plots_disabled` | `output.plots: false` swaps the `figs/proposal/heatmap.png` embed inside `## Client comparison` for a markdown table whose header is `\| Gas param \| <client>... \|` (clients alphabetical) and whose rows are the per-client `new_gas_rounded` integers in config-declaration order. The PNG is not written. |

---

## 3. Maintenance log (refactor notes)

Earlier waves of the suite split warnings, sentinel rendering, and missing-glue detection into three sibling files for "fail-independence." That guarantee is now provided by `@pytest.mark.parametrize` on a single test surface, so [`test_e2e_proposal_warnings.py`](../tests/test_e2e_proposal_warnings.py) carries all three concerns. Each test still fails in isolation if the corresponding implementation surface drifts.

A handful of conventions remain shapes worth preserving as the codebase evolves — refactors should keep these stable, and tests should keep asserting them:

- **Logger tree.** `evm_gasfit` is the root; per-area children (`evm_gasfit.estimate`, `evm_gasfit.glue`, `evm_gasfit.defaults`, `evm_gasfit.reports`) emit their own records and propagate up. Tests `caplog`-scope to the parent or a specific descendant; both work. Don't collapse the tree to a single logger, and don't add siblings outside the `evm_gasfit.*` namespace.
- **`runtime_ms` on `new_gas_all_params.csv` is post-glue-adjustment** for the target-coef row (see [`proposal/aggregate.py`](../src/evm_gasfit/proposal/aggregate.py)). The §4.5 conversion `new_gas_decimal == anchor_rate · runtime_ms / 1000` is the binding contract; tests rely on this equality.
- **`poor_fit` is written as a Python bool** ([`proposal/aggregate.py`](../src/evm_gasfit/proposal/aggregate.py)); pandas `read_csv` round-trips it back to bool (auto-inferred), so tests compare with `== True` / `== False`. If the writer ever changes to write `0`/`1` or strings, the comparisons here break by design — surface that, don't paper over it.

Items previously listed here as "open" are now pinned by code: `client_name` is the canonical client column in [`glue_results.csv`](../src/evm_gasfit/glue/estimate.py); `"no prior default"` is the literal `SENTINEL` in [`reports/proposal.py`](../src/evm_gasfit/reports/proposal.py); `## Warnings` is the depth-2 heading; `ModelingError` always exits 2 via [`cli.py`](../src/evm_gasfit/cli.py); `new_gas.csv` carries no `current_gas` column — the diff baseline is applied at markdown-render time.

---

## 4. Coverage matrix

| Plan section | Topic | Pinned by |
| --- | --- | --- |
| §2.1 | YAML config shape, model specs | `test_e2e_happy_path`, `test_e2e_multi_model` |
| §2.2 | Runtimes CSV columns and campaign eligibility | `_data_synth` + `test_e2e_campaign_analysis` |
| §2.3 | Opcounts JSON and target-count invariant | `_data_synth`; semantic precompile coverage in `test_e2e_precompile` and `test_e2e_catalog_smoke` |
| §4.2 / §4.2a | NNLS, session-cluster inference, qualification gates | fit failures in `test_unit_nnls_failures`; session/admissibility provenance in `test_e2e_campaign_analysis` |
| §4.3 | Model formula and setup parameters | `test_e2e_happy_path`, `test_e2e_multi_model` |
| §4.4 | Glue adjustment and propagated/conditional uncertainty | `test_e2e_glue`; missing-glue warning tests |
| §4.5–§4.7 | Optional anchor, scenarios, aggregation, rounding | happy path, glue, and campaign analysis |
| §5 / §8 | Artifacts, immutable provenance, CLI, manifest-gated comparison | report/CLI tests and `test_e2e_campaign_analysis` |
| §3 | Pipeline architecture | implicit via every happy-path test |
| §4.0 | Logging channel, error types, determinism | CLI exit codes + `test_e2e_determinism` (3 tests) |
| §4.1 | Fixture-name parser | implicit via every test |
| §4.8 | Derived params: alias, formula, AST whitelist (incl. `max`/`min`), load-time identifier check, joint access-delta pricing | `test_derived_alias_and_formula_evaluate`, `test_derived_formula_unknown_identifier_is_load_time_error`, `test_e2e_joint_delta` (3 tests), `test_unit_derived` |
| §5.1 | Figure naming/layout and plots toggle | `test_e2e_report_format` |
| §5 (`new_gas_proposal.md`) | Durable proposal artifacts and plots | `test_e2e_report_format` |
| §11 | Docs site | out of scope for the e2e suite |

---

## 5. Unit-test surfaces

The following sad paths are intentionally not e2e tests — each exercises a single module's behavior in isolation, and is pinned by a dedicated suite under [tests/](../tests/):

- §4.2 fit failure modes (`nobs < n_features + 1`, rank-deficient design, constant `opcount`, scipy convergence failure, bootstrap-iteration failure, all-skipped → `ModelingError`) → [`tests/test_unit_nnls_failures.py`](../tests/test_unit_nnls_failures.py).
- §4.3 zero-match specs: the end-of-run skipped-spec summary → [`tests/test_unit_nnls_failures.py`](../tests/test_unit_nnls_failures.py). Two specs, one fitting and one whose `test_name` matches no fixture. `test_zero_match_specs_are_summarized_once_at_end_of_run` asserts exactly one summary WARNING beyond the per-spec skip line, naming the unmatched spec's `source_label` and `test_name` and *not* the spec that matched. `test_skipped_spec_summary_precedes_downstream_warnings` asserts it is the modeling layer's last WARNING, so it lands ahead of the glue/proposal warnings in a full run.
- §4.6 tie-break order (per-client `runtime_ms` → `pvalue` → lexicographic on `(test_name, target_opcode, model_coef_name, model_by-combo, source_label)`; across-client by ascending `client_name`), each with an order-independence counter-test → [`tests/test_unit_aggregate_tie_break.py`](../tests/test_unit_aggregate_tie_break.py).
- §2.4 Prague and Osaka gas-cost source selection (current
  `ethereum.forks.<fork>` path, legacy compatibility path, exact Prague
  fallback baseline, and no Osaka-only leakage) plus the
  `EVM_GASFIT_USE_FALLBACK=1` override →
  [`tests/test_unit_defaults_source.py`](../tests/test_unit_defaults_source.py).
- §4.8 derived mini-language `max`/`min` extension (clamping, variadic/nested forms, `None` propagation, identifier discovery into call args, rejection of non-whitelisted calls / keywords / starred args) → [`tests/test_unit_derived.py`](../tests/test_unit_derived.py).
- §2.1 campaign/pricing numeric validation and immutable-status serialization
  (`NaN`/infinite policy values rejected; model-by group identity retained;
  non-finite evidence encoded as JSON null) →
  [`tests/test_unit_campaign_config.py`](../tests/test_unit_campaign_config.py).

Identically-zero `opcount` is rejected earlier by the §2.3 input invariant as a `ConfigError` (CLI exit 1), not via the fit-skip path; the NNLS suite's `[zero]` parametrize branch covers that route.

---

## 6. How to maintain this document

- When you add or rename a test, update its row in §2 and the matching cell in §4.
- When the plan moves (a column renames, a section is added), update the Conventions table in §1 first, then push the change through to every matching assertion. Tests with broken assertions are the safety net for plan/code drift — don't relax assertions to make them pass; surface the conflict to the spec.
- When adding boilerplate (a new config shape, a new fixture pattern), prefer extending `_data_synth.py` over duplicating in test files. The factory/runner/columns trio is the single source of truth.
