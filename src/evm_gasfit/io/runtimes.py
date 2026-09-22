"""Loader for the runtimes CSV input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evm_gasfit.errors import ConfigError

_REQUIRED_COLUMNS: tuple[str, ...] = ("client_name", "fixture_name", "test_runtime_ms")


def load_runtimes(path: Path) -> pd.DataFrame:
    """Load the runtimes CSV.

    Args:
        path: Path to a CSV file containing at minimum the columns
            ``client_name``, ``fixture_name``, and ``test_runtime_ms``.
            Extra columns pass through unchanged. Campaign runs add the
            metadata columns ``session_id``, ``sample_id``, ``phase``,
            ``status``, and ``repetition``; any further numeric metadata
            (gas quantities, sequence numbers) also passes through and is
            never consumed as a regressor or differenced by baseline
            pairing.

    Returns:
        The parsed DataFrame. Duplicate ``(client_name, fixture_name)`` rows
        are preserved as independent observations.

    Raises:
        ConfigError: If the file is missing or a required column is absent.
    """
    if not path.exists():
        raise ConfigError(f"runtimes CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(
            f"runtimes CSV {path} is missing required column(s): {', '.join(missing)}"
        )
    return df
