"""Pull per-fork gas costs from the optional ``ethereum/execution-specs`` extra."""

from __future__ import annotations

from importlib import import_module


def get_from_execution_specs(fork: str) -> dict[str, int]:
    """Return the fork's gas-cost fields as a flat dict.

    Current execution-specs exposes forks below ``ethereum.forks``. Older
    published packages used ``ethereum.<fork>``; retain that path so existing
    Osaka analyses keep their source-selection behavior.

    Raises:
        ImportError: when the ``execution-specs`` extra is not installed or the
            fork's ``GasCosts`` class cannot be located.
    """
    module = None
    paths = (
        f"ethereum.forks.{fork}.vm.gas",
        f"ethereum.{fork}.vm.gas",
    )
    for path in paths:
        try:
            module = import_module(path)
            break
        except ImportError:
            continue
    if module is None:
        joined = " or ".join(paths)
        raise ImportError(f"could not import {joined}")

    gas_costs = getattr(module, "GasCosts", None)
    if gas_costs is None:
        raise ImportError(f"{module.__name__} has no GasCosts attribute")
    # GasCosts is expected to expose integer-valued attributes; collect the
    # public ones into a plain dict.
    out: dict[str, int] = {}
    for name in dir(gas_costs):
        if name.startswith("_"):
            continue
        value = getattr(gas_costs, name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    if not out:
        raise ImportError(f"{module.__name__}.GasCosts exposes no integer fields")
    return out
