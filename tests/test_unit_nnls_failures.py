"""Unit tests pinning the fit-failure-mode contract for the NNLS regressor.

These exercise ``modeling.estimate.estimate_models`` (which logs the WARNINGs)
and ``modeling.nnls.fit_nnls`` (where the bootstrap loop tolerates iteration
failures). The synthesized inputs go straight to the modeling layer — no YAML,
no runtime/opcount loaders — because each test is about one failure branch.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls as _real_nnls

from evm_gasfit.config import Config
from evm_gasfit.errors import ModelingError
from evm_gasfit.glue.adjust import _propagated_interval
from evm_gasfit.modeling import nnls as nnls_module
from evm_gasfit.modeling.diagnostics import (
    leave_one_group_out_error,
    leave_one_point_out_error,
)
from evm_gasfit.modeling.estimate import estimate_models
from evm_gasfit.modeling.nnls import fit_nnls

_LOGGER_NAME = "evm_gasfit"
_TEST_NAME = "test_arithmetic"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_config(
    *,
    model_params: dict[str, str] | None = None,
    new_params: dict[str, int | None] | None = None,
    bootstrap_iterations: int = 25,
) -> Config:
    """Build a minimal validated Config targeting a single ADD spec."""
    cfg: dict[str, Any] = {
        "version": 1,
        "anchor_rate": 1.0e8,
        "clients": ["geth"],
        "gas_costs": {"fork": "osaka"},
        "modeling": {"bootstrap_iterations": bootstrap_iterations, "random_seed": 7},
        "output": {"plots": False},
        "models": {
            "presets": [],
            "custom": [
                {
                    "test_name": _TEST_NAME,
                    "target_operation": "ADD",
                    "model_params": model_params or {"target_coef": "OPCODE_ADD"},
                }
            ],
        },
    }
    if new_params is not None:
        cfg["new_params"] = dict(new_params)
    return Config.model_validate(cfg)


def _make_fixtures_df(
    *,
    opcounts: list[float],
    runtimes: list[float] | None = None,
    extra_cols: dict[str, list[float]] | None = None,
    client: str = "geth",
) -> pd.DataFrame:
    """Build a fixtures_df slice for ``test_arithmetic`` / target ADD.

    Each row gets a unique ``fixture_name``; ``ADD`` is filled to match
    ``opcount`` so the invariant in ``_enforce_opcount_invariant`` passes.
    """
    n = len(opcounts)
    runtimes = (
        runtimes if runtimes is not None else [100.0 + 1e-5 * c for c in opcounts]
    )
    if len(runtimes) != n:
        raise ValueError("runtimes and opcounts must be the same length")
    df = pd.DataFrame(
        {
            "client_name": [client] * n,
            "fixture_name": [f"f{i}" for i in range(n)],
            "test_file": [_TEST_NAME] * n,
            "test_name": [_TEST_NAME] * n,
            "test_runtime_ms": runtimes,
            "opcount": [float(c) for c in opcounts],
            # Per-opcode count column; the invariant matches opcount to ADD.
            "ADD": [float(c) for c in opcounts],
        }
    )
    if extra_cols:
        for name, values in extra_cols.items():
            if len(values) != n:
                raise ValueError(f"extra_cols[{name!r}] length mismatch")
            df[name] = [float(v) for v in values]
    return df


# ---------------------------------------------------------------------------
# Holdout diagnostics must reject incomplete refit evidence.
# ---------------------------------------------------------------------------


def test_session_holdout_is_nan_when_any_real_refit_loses_rank() -> None:
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([3.0, 5.0, 7.0])
    groups = np.array(["a", "a", "b"])

    # Holding out session ``a`` leaves only one row. The remaining fold cannot
    # identify intercept and slope, so a successful ``b`` fold must not mask it.
    assert np.isnan(leave_one_group_out_error(X, y, groups))


def test_workload_holdout_is_nan_when_any_real_refit_loses_rank() -> None:
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([3.0, 3.0, 5.0])
    opcount = X[:, 1]

    # Both workload-point holdouts leave a constant design. This is a real
    # numerical precondition failure, not a mocked solver error.
    assert np.isnan(leave_one_point_out_error(X, y, opcount))


def test_single_session_keeps_point_estimate_without_row_bootstrap() -> None:
    design = pd.DataFrame(
        {
            "opcount": [1.0, 2.0, 3.0, 4.0],
            "test_runtime_ms": [3.0, 5.0, 7.0, 9.0],
        }
    )

    fit = fit_nnls(
        design,
        features=["opcount"],
        target="test_runtime_ms",
        n_bootstrap=20,
        random_seed=7,
        groups=np.array(["only-session"] * len(design)),
    )

    assert fit.params["opcount"] == pytest.approx(2.0)
    assert len(fit.bootstrap_draws("opcount")) == 0


def test_shared_session_glue_interval_pairs_matching_bootstrap_replicates() -> None:
    groups = np.array(["session-a"] * 4 + ["session-b"] * 4)
    counts = np.tile(np.arange(1.0, 5.0), 2)
    target_design = pd.DataFrame(
        {
            "opcount": counts,
            "test_runtime_ms": 2.0 + 3.0 * counts + np.repeat([0.0, 1.0], 4),
        }
    )
    glue_design = pd.DataFrame(
        {
            "POP": counts,
            "test_runtime_ms": 1.0 + 0.5 * counts + np.repeat([0.0, 0.3], 4),
        }
    )
    target = fit_nnls(
        target_design,
        features=["opcount"],
        target="test_runtime_ms",
        n_bootstrap=40,
        random_seed=11,
        groups=groups,
    )
    glue = fit_nnls(
        glue_design,
        features=["POP"],
        target="test_runtime_ms",
        n_bootstrap=40,
        random_seed=11,
        groups=groups,
    )

    low, high, conditional = _propagated_interval(
        target_fit=target,
        partners=[(1.0, glue, "POP")],
        confidence_level=0.95,
        rng=np.random.default_rng(19),
        point_adjustment=0.0,
        point_target=float(target.params["opcount"]),
    )

    target_draws = target.bootstrap_draw_matrix("opcount")
    glue_draws = glue.bootstrap_draw_matrix("POP")
    valid = np.isfinite(target_draws) & np.isfinite(glue_draws)
    adjusted = np.maximum(0.0, target_draws[valid] - glue_draws[valid])
    assert conditional is False
    assert low == pytest.approx(np.quantile(adjusted, 0.025))
    assert high == pytest.approx(np.quantile(adjusted, 0.975))


# ---------------------------------------------------------------------------
# §4.2 fit-failure tests driven through estimate_models
# ---------------------------------------------------------------------------


def test_skip_when_nobs_below_features_plus_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # n_features+1 = 2 (intercept + opcount); need at least 3 rows. Give 2.
    config = _make_config()
    fixtures_df = _make_fixtures_df(opcounts=[10.0, 20.0])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    with pytest.raises(ModelingError):
        estimate_models(config, fixtures_df)

    skip_records = [r for r in caplog.records if "skipping" in r.getMessage()]
    assert skip_records, "expected a WARNING naming the skipped fit"
    msg = skip_records[0].getMessage()
    assert _TEST_NAME in msg
    assert "geth" in msg
    assert "nobs" in msg


def test_skip_when_design_is_rank_deficient(caplog: pytest.LogCaptureFixture) -> None:
    # Two extras that are equal row-by-row produce perfectly collinear
    # opcount*param columns in the design matrix.
    config = _make_config(
        model_params={
            "target_coef": "OPCODE_ADD",
            "feat_a": "OPCODE_SUB",
            "feat_b": "OPCODE_MUL",
        },
    )
    n = 8
    fixtures_df = _make_fixtures_df(
        opcounts=[10.0 * (i + 1) for i in range(n)],
        extra_cols={
            "feat_a": [1.0 + (i % 3) for i in range(n)],
            "feat_b": [1.0 + (i % 3) for i in range(n)],
        },
    )

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    with pytest.raises(ModelingError):
        estimate_models(config, fixtures_df)

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "skipping" in m and "rank" in m.lower() and _TEST_NAME in m and "geth" in m
        for m in msgs
    ), f"expected a rank-deficient skip warning; got: {msgs}"


@pytest.mark.parametrize(
    ("opcounts", "label"),
    [
        ([5.0, 5.0, 5.0, 5.0], "constant"),
        ([0.0, 0.0, 0.0, 0.0], "zero"),
    ],
)
def test_skip_when_opcount_is_constant_or_zero(
    opcounts: list[float],
    label: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The opcount=0 path also trips the fixtures_df invariant (target opcount
    # must be > 0). Both kill target_coef identifiability and must be skipped.
    config = _make_config()
    fixtures_df = _make_fixtures_df(
        opcounts=opcounts,
        runtimes=[100.0, 101.0, 99.0, 100.5],
    )

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    if label == "zero":
        # _enforce_opcount_invariant rejects opcount=0 outright as a ConfigError
        # before the fit step is reached — exercise that path instead.
        from evm_gasfit.errors import ConfigError

        with pytest.raises(ConfigError, match="opcount=0"):
            estimate_models(config, fixtures_df)
        return

    with pytest.raises(ModelingError):
        estimate_models(config, fixtures_df)

    skip_msgs = [r.getMessage() for r in caplog.records if "skipping" in r.getMessage()]
    assert any(
        "constant" in m and _TEST_NAME in m and "geth" in m for m in skip_msgs
    ), f"expected constant-opcount skip; got: {skip_msgs}"


def test_skip_when_scipy_nnls_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Force the solver to raise on every call: scipy raises RuntimeError on
    # non-convergence in production, so use that real exception type.
    def boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("nnls forced failure")

    monkeypatch.setattr(nnls_module, "nnls", boom)

    config = _make_config()
    fixtures_df = _make_fixtures_df(opcounts=[10.0, 20.0, 30.0, 40.0, 50.0])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    with pytest.raises(ModelingError):
        estimate_models(config, fixtures_df)

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "NNLS solver raised" in m and _TEST_NAME in m and "geth" in m for m in msgs
    ), f"expected scipy-raise skip warning; got: {msgs}"


def test_modeling_error_when_every_fit_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One client, one spec, one group, too-few-rows → the lone fit is skipped
    # and the whole run produces zero result rows.
    config = _make_config()
    fixtures_df = _make_fixtures_df(opcounts=[10.0, 20.0])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    with pytest.raises(ModelingError, match="every model spec was skipped"):
        estimate_models(config, fixtures_df)


# ---------------------------------------------------------------------------
# Zero-match specs — the end-of-run skipped-spec summary
# ---------------------------------------------------------------------------


def _make_config_with_unmatched_spec() -> Config:
    """One ADD spec that fits plus one spec whose test_name matches nothing."""
    return Config.model_validate(
        {
            "version": 1,
            "anchor_rate": 1.0e8,
            "clients": ["geth"],
            "gas_costs": {"fork": "osaka"},
            "modeling": {"bootstrap_iterations": 25, "random_seed": 7},
            "output": {"plots": False},
            "models": {
                "presets": [],
                "custom": [
                    {
                        "test_name": _TEST_NAME,
                        "target_operation": "ADD",
                        "model_params": {"target_coef": "OPCODE_ADD"},
                    },
                    {
                        "test_name": "test_renamed_away",
                        "target_operation": "ADD",
                        "model_params": {"target_coef": "OPCODE_SUB"},
                    },
                ],
            },
        }
    )


def test_zero_match_specs_are_summarized_once_at_end_of_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A spec matching zero fixtures gets its own summary WARNING naming the
    spec by ``source_label``, on top of the per-spec skip warning — so a stale
    ``test_name`` isn't invisible among the other benign warnings."""
    config = _make_config_with_unmatched_spec()
    fixtures_df = _make_fixtures_df(opcounts=[10.0, 20.0, 30.0, 40.0, 50.0])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    estimate_models(config, fixtures_df)

    summaries = [
        r.getMessage()
        for r in caplog.records
        if "matched no fixtures" in r.getMessage()
    ]
    assert len(summaries) == 1, f"expected exactly one summary line; got: {summaries}"
    summary = summaries[0]
    assert "models.custom[1]" in summary
    assert "test_renamed_away" in summary
    # The spec that did match must not be named.
    assert "models.custom[0]" not in summary


def test_skipped_spec_summary_precedes_downstream_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The summary is the last WARNING the modeling layer emits, so it lands
    ahead of the glue/proposal warnings that follow in a full run."""
    config = _make_config_with_unmatched_spec()
    fixtures_df = _make_fixtures_df(opcounts=[10.0, 20.0, 30.0, 40.0, 50.0])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    estimate_models(config, fixtures_df)

    msgs = [r.getMessage() for r in caplog.records]
    assert "matched no fixtures" in msgs[-1], f"summary must come last; got: {msgs}"


# ---------------------------------------------------------------------------
# §4.2 bootstrap-iteration failure — driven through fit_nnls directly
# ---------------------------------------------------------------------------


def test_bootstrap_iteration_failures_dont_break_primary_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Primary fit must succeed; bootstrap iterations after the first should
    # raise. NNLSResults filters NaN rows out of inference and surfaces the
    # reduced success count on its summary string.
    rng = np.random.default_rng(0)
    opcounts = np.linspace(10.0, 100.0, 20)
    runtimes = 5.0 + 0.5 * opcounts + rng.normal(0.0, 0.1, size=opcounts.size)
    df = pd.DataFrame({"opcount": opcounts, "test_runtime_ms": runtimes})

    call_counter = {"n": 0}

    def flaky_nnls(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        call_counter["n"] += 1
        # First call is the primary fit; let it succeed. Every other call (the
        # bootstrap iterations) raises so they end up as NaN rows.
        if call_counter["n"] == 1:
            return _real_nnls(A, b)
        raise RuntimeError("bootstrap iteration forced failure")

    monkeypatch.setattr(nnls_module, "nnls", flaky_nnls)

    n_bootstrap = 8
    result = fit_nnls(
        df,
        features=["opcount"],
        target="test_runtime_ms",
        n_bootstrap=n_bootstrap,
        random_seed=1,
    )

    # Primary fit completed despite every bootstrap iteration failing.
    assert result.nobs == len(opcounts)
    assert float(result.params["opcount"]) > 0
    # All bootstrap iterations failed → success counter is zero, p-values fall
    # back to 1.0 (unidentifiable), and confidence intervals are NaN.
    assert result._n_bootstrap_total == n_bootstrap
    assert result._n_bootstrap_success == 0
    assert (result.pvalues == 1.0).all()
    ci = result.conf_int()
    assert ci.isna().all().all()
    # The summary string surfaces the iteration tally per the §4.2 note.
    summary = result.summary()
    assert f"0 of {n_bootstrap} iterations succeeded" in summary


def test_bootstrap_partial_failures_reduce_n_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mixed-success case: half the bootstrap iterations raise. Inference still
    # runs against the surviving draws; std errors are finite.
    rng = np.random.default_rng(0)
    opcounts = np.linspace(10.0, 100.0, 20)
    runtimes = 5.0 + 0.5 * opcounts + rng.normal(0.0, 0.1, size=opcounts.size)
    df = pd.DataFrame({"opcount": opcounts, "test_runtime_ms": runtimes})

    call_counter = {"n": 0}

    def flaky_nnls(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        call_counter["n"] += 1
        # First call (primary) succeeds; even-numbered subsequent calls raise.
        if call_counter["n"] == 1 or call_counter["n"] % 2 == 1:
            return _real_nnls(A, b)
        raise RuntimeError("forced odd-iteration failure")

    monkeypatch.setattr(nnls_module, "nnls", flaky_nnls)

    n_bootstrap = 10
    result = fit_nnls(
        df,
        features=["opcount"],
        target="test_runtime_ms",
        n_bootstrap=n_bootstrap,
        random_seed=1,
    )

    assert 0 < result._n_bootstrap_success < n_bootstrap
    summary = result.summary()
    assert (
        f"{result._n_bootstrap_success} of {n_bootstrap} iterations succeeded"
        in summary
    )
