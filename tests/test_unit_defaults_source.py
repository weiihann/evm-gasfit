"""Unit tests pinning the gas-cost source-selection contract.

The :mod:`evm_gasfit.defaults` module picks **one** source for the whole run:

- When ``ethereum/execution-specs`` is installed and
  ``EVM_GASFIT_USE_FALLBACK`` is unset, current
  ``ethereum.forks.<fork>.vm.gas.GasCosts`` or legacy
  ``ethereum.<fork>.vm.gas.GasCosts`` is the source of truth. The returned
  ``GasCosts`` reports ``source == "execution-specs"`` and an ``INFO`` line on
  the ``evm_gasfit.defaults`` logger names that source.
- When the extra is absent (or the env var is set to ``"1"``), the bundled
  ``_fallback.py`` table is used and ``source == "fallback"``.

The two paths are **never mixed** within one ``GasCosts`` instance.

The probe of the optional extra and the env var happen at import time, so
each test sets up the environment / ``sys.modules`` stub *before* reloading
``evm_gasfit.defaults`` and restores the module to its on-disk state in
teardown so the rest of the suite is unaffected.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator

import pytest

import evm_gasfit.defaults as defaults_mod

# A non-osaka value the fallback could never produce; lets us tell which
# source backs a returned GasCosts instance without depending on whether the
# real execution-specs package agrees with the bundled mirror.
_SENTINEL_OPCODE = "OPCODE_ADD"
_SENTINEL_SPECS_VALUE = 9999


@pytest.fixture
def reload_defaults() -> Iterator[None]:
    """Restore ``evm_gasfit.defaults`` to its on-disk state after the test.

    Tests that mutate ``sys.modules`` or ``EVM_GASFIT_USE_FALLBACK`` and then
    call ``importlib.reload(defaults_mod)`` must leave the module observable
    in its original form once the test ends — other tests in the suite import
    ``GasCosts`` and ``get_gas_costs`` from this module and would otherwise
    see stale references or a wrongly-cached probe result.
    """
    yield
    # Drop any ``ethereum.*`` stubs a test injected so the reload re-runs the
    # genuine "is the extra installed?" probe.
    for name in [
        m for m in sys.modules if m == "ethereum" or m.startswith("ethereum.")
    ]:
        if getattr(sys.modules[name], "__file__", None) is None:
            del sys.modules[name]
    importlib.reload(defaults_mod)


def _build_specs_stub(fork: str) -> types.ModuleType:
    """Build a minimal ``GasCosts`` module matching the loader's contract.

    The loader reads ``module.GasCosts``, iterates ``dir()`` skipping
    underscore names, and collects integer attributes into a flat dict.
    """

    class GasCosts:
        BASE = 2
        VERY_LOW = 3
        WARM_ACCESS = 100
        COLD_ACCOUNT_ACCESS = 2600
        # Value disagrees with the bundled fallback (which has OPCODE_ADD = 3)
        # so a test can prove the returned table came from this stub.
        OPCODE_ADD = _SENTINEL_SPECS_VALUE
        # A few non-int attributes the loader must ignore.
        _PRIVATE = "skip me"
        FLAG = True  # bools are int subclasses but the loader filters them out.
        NAME = "stub"

    module = types.ModuleType(f"ethereum.{fork}.vm.gas")
    module.GasCosts = GasCosts
    return module


def _install_specs_stub(fork: str) -> None:
    """Insert empty parent packages plus the leaf ``ethereum.<fork>.vm.gas`` stub."""
    for parent in ("ethereum", f"ethereum.{fork}", f"ethereum.{fork}.vm"):
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    sys.modules[f"ethereum.{fork}.vm.gas"] = _build_specs_stub(fork)


def _install_current_specs_stub(fork: str) -> None:
    """Insert a current ``ethereum.forks.<fork>.vm.gas`` stub."""
    for parent in (
        "ethereum",
        "ethereum.forks",
        f"ethereum.forks.{fork}",
        f"ethereum.forks.{fork}.vm",
    ):
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    sys.modules[f"ethereum.forks.{fork}.vm.gas"] = _build_specs_stub(fork)


# --------------------------------------------------------------------------
# execution-specs path.
# --------------------------------------------------------------------------


def test_execution_specs_path_when_extra_installed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reload_defaults: None,
) -> None:
    """Extra present + env var unset → execution-specs path is selected."""
    monkeypatch.delenv("EVM_GASFIT_USE_FALLBACK", raising=False)
    _install_specs_stub("osaka")
    importlib.reload(defaults_mod)

    with caplog.at_level(logging.INFO, logger="evm_gasfit.defaults"):
        gc = defaults_mod.get_gas_costs("osaka")

    assert gc.source == "execution-specs"
    assert gc.fork == "osaka"
    # The stub's value (not the fallback's) is what landed in the table.
    assert gc[_SENTINEL_OPCODE] == _SENTINEL_SPECS_VALUE
    # Non-int attributes (string, bool, dunder) are filtered out.
    assert "NAME" not in gc
    assert "FLAG" not in gc
    assert "_PRIVATE" not in gc
    # One INFO line on the expected logger with the expected shape.
    matches = [
        rec
        for rec in caplog.records
        if rec.name == "evm_gasfit.defaults"
        and rec.levelno == logging.INFO
        and "source=execution-specs" in rec.getMessage()
        and "fork=osaka" in rec.getMessage()
    ]
    assert len(matches) == 1, caplog.text


def test_current_execution_specs_path_loads_prague(
    monkeypatch: pytest.MonkeyPatch,
    reload_defaults: None,
) -> None:
    """The current ``ethereum.forks`` package layout serves Prague."""
    monkeypatch.delenv("EVM_GASFIT_USE_FALLBACK", raising=False)
    _install_current_specs_stub("prague")
    importlib.reload(defaults_mod)

    gc = defaults_mod.get_gas_costs("prague")

    assert gc.source == "execution-specs"
    assert gc.fork == "prague"
    assert gc[_SENTINEL_OPCODE] == _SENTINEL_SPECS_VALUE


# --------------------------------------------------------------------------
# Fallback path.
# --------------------------------------------------------------------------


def test_fallback_path_when_extra_absent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reload_defaults: None,
) -> None:
    """Extra absent → bundled fallback table is selected and logged as ``fallback``."""
    monkeypatch.delenv("EVM_GASFIT_USE_FALLBACK", raising=False)
    importlib.reload(defaults_mod)
    # Simulate "extra not installed" by neutralising the probe handle, the
    # way an unsuccessful import-time probe would have left it.
    monkeypatch.setattr(defaults_mod, "_from_specs", None)

    with caplog.at_level(logging.INFO, logger="evm_gasfit.defaults"):
        gc = defaults_mod.get_gas_costs("osaka")

    from evm_gasfit.defaults._fallback import get_fallback

    expected = get_fallback("osaka")
    assert gc.source == "fallback"
    assert gc.fork == "osaka"
    assert dict(gc.values) == expected
    assert gc[_SENTINEL_OPCODE] == expected[_SENTINEL_OPCODE]
    matches = [
        rec
        for rec in caplog.records
        if rec.name == "evm_gasfit.defaults"
        and rec.levelno == logging.INFO
        and "source=fallback" in rec.getMessage()
        and "fork=osaka" in rec.getMessage()
    ]
    assert len(matches) == 1, caplog.text


def test_prague_fallback_matches_prague_and_excludes_osaka_fields(
    monkeypatch: pytest.MonkeyPatch,
    reload_defaults: None,
) -> None:
    """The bundled Prague baseline is not a relabeled Osaka table."""
    monkeypatch.setenv("EVM_GASFIT_USE_FALLBACK", "1")
    importlib.reload(defaults_mod)

    prague = defaults_mod.get_gas_costs("prague")
    osaka = defaults_mod.get_gas_costs("osaka")

    assert prague.source == "fallback"
    assert prague["BLOB_TARGET_GAS_PER_BLOCK"] == 786432
    assert prague["BLOB_BASE_FEE_UPDATE_FRACTION"] == 5007716
    assert prague["REFUND_AUTH_PER_EXISTING_ACCOUNT"] == 12500
    assert "PRECOMPILE_P256VERIFY" not in prague
    assert "OPCODE_CLZ" not in prague
    assert "BLOB_SCHEDULE_TARGET" not in prague
    assert "BLOCK_ACCESS_LIST_ITEM" not in prague
    assert osaka["PRECOMPILE_P256VERIFY"] == 6900
    assert osaka["OPCODE_CLZ"] == 5


# --------------------------------------------------------------------------
# Env-var override.
# --------------------------------------------------------------------------


def test_env_var_forces_fallback_even_when_extra_installed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reload_defaults: None,
) -> None:
    """``EVM_GASFIT_USE_FALLBACK=1`` overrides an installed extra."""
    monkeypatch.setenv("EVM_GASFIT_USE_FALLBACK", "1")
    _install_specs_stub("osaka")
    importlib.reload(defaults_mod)

    # With the override active, the import-time probe must short-circuit
    # without binding the execution-specs entry point.
    assert defaults_mod._from_specs is None

    with caplog.at_level(logging.INFO, logger="evm_gasfit.defaults"):
        gc = defaults_mod.get_gas_costs("osaka")

    from evm_gasfit.defaults._fallback import get_fallback

    expected = get_fallback("osaka")
    assert gc.source == "fallback"
    # Crucially, the table is the fallback's, not the stub's sentinel value.
    assert gc[_SENTINEL_OPCODE] == expected[_SENTINEL_OPCODE]
    assert gc[_SENTINEL_OPCODE] != _SENTINEL_SPECS_VALUE
    matches = [
        rec
        for rec in caplog.records
        if rec.name == "evm_gasfit.defaults" and "source=fallback" in rec.getMessage()
    ]
    assert len(matches) == 1, caplog.text


def test_env_var_unset_does_not_force_fallback(
    monkeypatch: pytest.MonkeyPatch,
    reload_defaults: None,
) -> None:
    """Counter-test: with the stub installed and the env var unset, the
    execution-specs path is taken — proving the previous test's mechanism
    actually does something."""
    monkeypatch.delenv("EVM_GASFIT_USE_FALLBACK", raising=False)
    _install_specs_stub("osaka")
    importlib.reload(defaults_mod)

    gc = defaults_mod.get_gas_costs("osaka")

    assert gc.source == "execution-specs"
    assert gc[_SENTINEL_OPCODE] == _SENTINEL_SPECS_VALUE


# --------------------------------------------------------------------------
# Sources never mix.
# --------------------------------------------------------------------------


def test_sources_are_never_mixed_within_one_instance(
    monkeypatch: pytest.MonkeyPatch,
    reload_defaults: None,
) -> None:
    """Each ``GasCosts`` instance is sourced from exactly one backend.

    Build one instance via the stubbed execution-specs path and another via
    the fallback; the sentinel field disagrees between the two backends, so
    if any cross-contamination occurred (e.g. the loader merged tables) the
    sentinel would no longer split them cleanly.
    """
    monkeypatch.delenv("EVM_GASFIT_USE_FALLBACK", raising=False)
    _install_specs_stub("osaka")
    importlib.reload(defaults_mod)

    from_specs = defaults_mod.get_gas_costs("osaka")
    assert from_specs.source == "execution-specs"
    assert from_specs[_SENTINEL_OPCODE] == _SENTINEL_SPECS_VALUE

    # Now flip to the fallback path without reload: the env var route is
    # import-time, so we emulate "extra not installed" by clearing the probe
    # handle directly.
    monkeypatch.setattr(defaults_mod, "_from_specs", None)
    from_fallback = defaults_mod.get_gas_costs("osaka")

    from evm_gasfit.defaults._fallback import get_fallback

    expected = get_fallback("osaka")
    assert from_fallback.source == "fallback"
    assert from_fallback[_SENTINEL_OPCODE] == expected[_SENTINEL_OPCODE]
    assert from_fallback[_SENTINEL_OPCODE] != _SENTINEL_SPECS_VALUE
    # The two instances disagree on the sentinel: no mixed-source merging.
    assert from_specs[_SENTINEL_OPCODE] != from_fallback[_SENTINEL_OPCODE]
