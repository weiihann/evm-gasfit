"""Fixture-name parser and shared ``fixtures_df`` builder."""

from __future__ import annotations

import re

import pandas as pd

from evm_gasfit.errors import ConfigError
from evm_gasfit.io import FixtureMatchResult, report_unmatched_fixtures

# Accepts both the legacy ``file.py__test[...]`` form and the pytest node-ID
# ``path/file.py::test[...]`` form, capturing the basename (path stripped) and
# tolerating a missing ``[...]`` params group.
_FIXTURE_RE = re.compile(
    r"^(?:.*/)?(?P<test_file>test_[^./]+\.py)(?:__|::)"
    r"(?P<test_name>[^\[]+?)"
    r"(?:\[(?P<test_params>.*)\])?$"
)
# Split a token into (key, value) at the first '_' whose value side starts with
# an uppercase letter or a digit — that's the key/value transition. If no such
# transition exists, fall back to the first '_'. Tokens with no '_' don't match
# and become standalone tags.
_KEY_VALUE_RE = re.compile(r"^(?P<key>.+?)_(?P<value>[A-Z0-9].*)$")


def parse_fixture_name(fixture_name: str) -> dict[str, str | list[str]]:
    """Parse an EEST fixture name into ``test_file``, ``test_name``, ``tokens``, and ``params``."""
    match = _FIXTURE_RE.match(fixture_name)
    if not match:
        return {
            "test_file": "",
            "test_name": fixture_name,
            "tokens": [],
            "params": {},
        }
    tokens = match["test_params"].split("-") if match["test_params"] else []
    params: dict[str, str] = {}
    for token in tokens:
        if "_" not in token:
            continue
        kv = _KEY_VALUE_RE.match(token)
        if kv is not None:
            params[kv["key"]] = kv["value"]
        else:
            key, _, value = token.partition("_")
            params[key] = value
    return {
        "test_file": match["test_file"],
        "test_name": match["test_name"],
        "tokens": tokens,
        "params": params,
    }


def _same_param_value(left: object, right: object) -> bool:
    """Return whether CSV and fixture-name representations describe one value."""
    if pd.isna(left) or pd.isna(right):
        return True
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _coalesce_runtime_params(
    runtimes_df: pd.DataFrame, parsed_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coalesce equal explicit and fixture-name ``param_`` columns."""
    parsed_params = {
        column for column in parsed_df.columns if column.startswith("param_")
    }
    explicit_params = {
        column for column in runtimes_df.columns if column.startswith("param_")
    }
    shared = sorted(parsed_params & explicit_params)
    parsed_by_fixture = parsed_df.set_index("fixture_name")
    for column in shared:
        parsed_values = runtimes_df["fixture_name"].map(parsed_by_fixture[column])
        conflicts = [
            fixture
            for fixture, left, right in zip(
                runtimes_df["fixture_name"],
                runtimes_df[column],
                parsed_values,
            )
            if not _same_param_value(left, right)
        ]
        if conflicts:
            raise ConfigError(
                f"runtimes CSV column {column!r} conflicts with fixture-name "
                f"parameter for fixture {conflicts[0]!r}"
            )
        runtimes_df[column] = runtimes_df[column].where(
            runtimes_df[column].notna(), parsed_values
        )
    return runtimes_df, parsed_df.drop(columns=shared)


def build_fixtures_df(
    runtimes_df: pd.DataFrame,
    opcounts: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, FixtureMatchResult]:
    """Build the shared ``fixtures_df`` joining runtimes with parsed params and opcounts.

    Args:
        runtimes_df: DataFrame from :func:`load_runtimes`, with at minimum the
            columns ``client_name``, ``fixture_name``, ``test_runtime_ms``.
        opcounts: Mapping from :func:`load_opcounts`.

    Returns:
        A frame carrying original runtime columns, parsed ``test_file`` and
        ``test_name``, raw ``param_<key>`` fields, ``opcount``, and per-opcode
        counts. Explicit ``param_<key>`` runtime columns carry canonical
        workload parameters omitted from the fixture ID; equal explicit and
        parsed facts coalesce, while conflicts are rejected. The ``param_``
        prefix prevents collisions with opcode mnemonics like ``SSTORE``.
        Fixtures appearing in only one input are dropped with a count-only
        warning per direction on the ``evm_gasfit`` logger; their names are
        returned on ``match_result`` for downstream export to ``meta.json``.
    """
    runtimes_fixtures = set(runtimes_df["fixture_name"].unique())
    opcounts_fixtures = set(opcounts.keys())
    match_result = report_unmatched_fixtures(runtimes_fixtures, opcounts_fixtures)
    matched = match_result.matched

    df = (
        runtimes_df[runtimes_df["fixture_name"].isin(matched)]
        .copy()
        .reset_index(drop=True)
    )

    # Parse fixture names once per unique fixture to avoid redundant work.
    unique_names = df["fixture_name"].drop_duplicates().tolist()
    parsed_rows = []
    for name in unique_names:
        parsed = parse_fixture_name(name)
        row: dict[str, object] = {
            "fixture_name": name,
            "test_file": parsed["test_file"],
            "test_name": parsed["test_name"],
        }
        row.update(
            {f"param_{k}": v for k, v in parsed["params"].items()}  # type: ignore[arg-type]
        )
        parsed_rows.append(row)
    # Supplying columns for populated data would discard parsed parameters.
    parsed_df = pd.DataFrame(
        parsed_rows,
        columns=None if parsed_rows else ["fixture_name", "test_file", "test_name"],
    )
    df, parsed_df = _coalesce_runtime_params(df, parsed_df)
    df = df.merge(parsed_df, on="fixture_name", how="left")

    # Build a wide opcode-count table indexed by fixture_name and left-join it.
    opcounts_df = pd.DataFrame.from_dict(opcounts, orient="index")
    opcounts_df.index.name = "fixture_name"
    opcounts_df = opcounts_df.reset_index()
    # Restrict to matched fixtures so the merge can't reintroduce orphans.
    opcounts_df = opcounts_df[opcounts_df["fixture_name"].isin(matched)]
    # `SHA3` is the legacy mnemonic for opcode 0x20; benchmark opcount tables
    # emit it under that name exclusively, while the rest of the pipeline
    # (glue specs, model specs) only knows the canonical `KECCAK256`. Fold it
    # in here, at the single point every opcode-count column enters the
    # pipeline, so no downstream consumer needs its own alias lookup.
    if "SHA3" in opcounts_df.columns:
        opcounts_df["KECCAK256"] = opcounts_df["SHA3"].fillna(0.0) + (
            opcounts_df["KECCAK256"].fillna(0.0)
            if "KECCAK256" in opcounts_df.columns
            else 0.0
        )
        opcounts_df = opcounts_df.drop(columns="SHA3")
    # Per-opcode missing values are zero counts (sparse JSON is supported).
    opcode_cols = [c for c in opcounts_df.columns if c != "fixture_name"]
    opcounts_df[opcode_cols] = opcounts_df[opcode_cols].fillna(0.0)

    df = df.merge(opcounts_df, on="fixture_name", how="left")
    # Record which columns are per-opcode counts so downstream transforms
    # (notably baseline pairing) can diff exactly ``test_runtime_ms`` plus
    # these — never arbitrary numeric metadata that rode along in the
    # runtimes CSV (repetition counters, gas quantities, session sequence
    # numbers). Read once before any row-wise operation; ``attrs`` is not
    # guaranteed to survive every pandas op.
    df.attrs["opcode_columns"] = list(opcode_cols)
    return df, match_result
