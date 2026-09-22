"""Per-fork ``GasCosts`` defaults.

Selects a single source for the whole run: ``ethereum/execution-specs`` when the
optional extra is installed (and ``EVM_GASFIT_USE_FALLBACK`` is unset), otherwise
the bundled fallback tables in :mod:`evm_gasfit.defaults._fallback`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from difflib import get_close_matches
from importlib import import_module

from evm_gasfit.errors import ConfigError

from ._fallback import get_fallback, known_forks

_log = logging.getLogger(__name__)


def _probe_execution_specs() -> bool:
    """Return True iff a supported execution-specs fork module imports.

    Current execution-specs packages forks below ``ethereum.forks``. Legacy
    package paths remain canaries for existing Prague and Osaka installations.
    """
    for path in (
        "ethereum.forks.prague.vm.gas",
        "ethereum.prague.vm.gas",
        "ethereum.osaka.vm.gas",
    ):
        try:
            import_module(path)
            return True
        except ImportError:
            continue
    return False


# Probe the optional extra once at import time; the whole run uses one source.
_USE_FALLBACK = os.environ.get("EVM_GASFIT_USE_FALLBACK") == "1"
if _USE_FALLBACK or not _probe_execution_specs():
    _from_specs = None
else:
    from ._from_execution_specs import get_from_execution_specs as _from_specs


@dataclass
class GasCosts:
    """A flat, mutable mapping of fork gas-param field names to integer costs.

    The instance is dict-backed so callers can patch individual fields after
    construction (used by ``gas_costs.overrides`` in the YAML config) without
    mutating module-level state. ``field_names`` returns the original fork
    field set, captured at construction time so later additions (e.g. derived
    params) do not pollute the "raw fork fields" universe.

    Attributes:
        values: The underlying ``name -> cost`` table. Prefer the mapping API
            (``gc[name]``, ``name in gc``) over reaching for this directly.
        fork: The fork name this table belongs to, normalized to lowercase.
        source: Either ``"execution-specs"`` or ``"fallback"`` — useful for
            reproducibility tracking.
    """

    values: dict[str, int] = field(default_factory=dict)
    fork: str = ""
    source: str = ""
    _field_names: frozenset[str] = field(default_factory=frozenset)

    @property
    def field_names(self) -> frozenset[str]:
        """The raw fork-field names captured at construction time."""
        return self._field_names

    def __getitem__(self, name: str) -> int:
        return self.values[name]

    def __setitem__(self, name: str, value: int) -> None:
        self.values[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self.values

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self.values.items())


def get_gas_costs(fork: str) -> GasCosts:
    """Build a fresh ``GasCosts`` instance for ``fork``.

    The selected source (``execution-specs`` vs. ``fallback``) is fixed at
    package import time and logged once at ``INFO`` on the
    ``evm_gasfit.defaults`` logger.

    Args:
        fork: Fork name; matched case-insensitively against the bundled set.

    Returns:
        A fresh ``GasCosts`` whose ``values`` can be patched without affecting
        subsequent calls.

    Raises:
        ConfigError: When ``fork`` is not a known fork name.
    """
    normalized = fork.lower()
    table: dict[str, int] | None = None
    source = "fallback"
    if _from_specs is not None:
        try:
            table = _from_specs(normalized)
            source = "execution-specs"
        except ImportError:
            table = None
    if table is None:
        try:
            table = get_fallback(normalized)
        except KeyError:
            known = known_forks()
            hint = get_close_matches(normalized, known, n=1)
            suffix = f"; did you mean {hint[0]!r}?" if hint else ""
            raise ConfigError(
                f"unknown fork {fork!r} (known: {', '.join(known)}){suffix}"
            ) from None
    _log.info("gas costs: fork=%s, source=%s", normalized, source)
    return GasCosts(
        values=table,
        fork=normalized,
        source=source,
        _field_names=frozenset(table),
    )
