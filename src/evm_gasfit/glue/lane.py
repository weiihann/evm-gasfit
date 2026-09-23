"""Campaign-lane separation between target and calibration exports.

A campaign may export two lanes in one runtimes CSV:

- ``target`` rows price gas parameters. They are the modelspec corpus and
  the only rows a target-model regression may ever see.
- ``calibration`` rows carry the dedicated glue-driver fixtures (straight
  line sweeps, wrapper-call microbenchmarks). They exist so supporting
  opcodes can be estimated in isolation; they must never leak into a
  target-model fit, even through an overly broad spec selector, because a
  calibration sweep would dominate the target's count axis and silently
  replace the priced quantity.

The lane is carried as the explicit string column ``param_campaign_role``
(the exporter writes ``parameters.campaign_role``). Rows without the column
— legacy three-column CSVs — are all target rows. A row with an empty or
missing value is a target row; only the exact string ``calibration`` (after
whitespace/case normalization) marks the calibration lane. Anything else is
a typo and refuses to run rather than guessing.
"""

from __future__ import annotations

import logging

import pandas as pd

from evm_gasfit.errors import ConfigError

_log = logging.getLogger("evm_gasfit.glue")

#: Column the exporter uses to carry the lane (``parameters.campaign_role``).
CAMPAIGN_ROLE_COLUMN = "param_campaign_role"
#: Lane value for gas-parameter target rows.
ROLE_TARGET = "target"
#: Lane value for glue-driver calibration rows.
ROLE_CALIBRATION = "calibration"

_KNOWN_ROLES = frozenset({ROLE_TARGET, ROLE_CALIBRATION})


def split_campaign_lanes(
    fixtures_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(target_lane, calibration_lane)`` views of ``fixtures_df``.

    Both views share the parent's ``attrs`` (notably ``opcode_columns``) so
    downstream transforms keep working on either lane. When the frame has no
    role column the calibration lane is empty and the target lane is the
    input unchanged.
    """
    if CAMPAIGN_ROLE_COLUMN not in fixtures_df.columns:
        return fixtures_df, fixtures_df.iloc[0:0]
    raw = fixtures_df[CAMPAIGN_ROLE_COLUMN]
    normalized = raw.astype("string").str.strip().str.lower().fillna(ROLE_TARGET)
    unknown = sorted(set(str(v) for v in normalized.unique() if v not in _KNOWN_ROLES))
    if unknown:
        raise ConfigError(
            f"{CAMPAIGN_ROLE_COLUMN} carries unsupported value(s) {unknown!r}; "
            f"expected {' or '.join(sorted(_KNOWN_ROLES))!r}"
        )
    calibration_mask = normalized == ROLE_CALIBRATION
    calibration = fixtures_df[calibration_mask]
    target = fixtures_df[~calibration_mask]
    # ``attrs`` is not guaranteed to survive every pandas op; copy explicitly.
    calibration.attrs = dict(fixtures_df.attrs)
    target.attrs = dict(fixtures_df.attrs)
    return target, calibration


def calibration_lane(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: the calibration-lane rows (empty when no role column)."""
    return split_campaign_lanes(fixtures_df)[1]


def target_lane(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: the target-lane rows (everything without role=calibration)."""
    return split_campaign_lanes(fixtures_df)[0]
