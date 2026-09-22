"""Immutable analysis provenance: ``analysis_status.json``.

The file records everything needed to attribute an analysis output
directory to its exact inputs: input-file SHA-256 hashes, the embedded
campaign manifest (plus its hash), the frozen qualification policy, the
per-planned-model qualification statuses, and hashes of every emitted
artifact. It contains no timestamps and sorts its keys, so two identical
runs produce byte-identical files.

Immutability: writing is refused when a *different* ``analysis_status.json``
already exists in the target directory. Re-running an identical analysis
into the same directory is a no-op; anything else must move to a fresh
directory — provenance is append-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from numbers import Integral, Real
from pathlib import Path

from evm_gasfit.config import Config
from evm_gasfit.errors import ConfigError

_log = logging.getLogger("evm_gasfit")

STATUS_SCHEMA_VERSION = 1
STATUS_FILENAME = "analysis_status.json"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, hex-encoded."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _policy_dict(config: Config) -> dict[str, object]:
    return {
        "anchor_rate": config.anchor_rate,
        "clients": list(config.clients),
        "fork": config.gas_costs.fork,
        "gas_costs_overrides": dict(config.gas_costs.overrides),
        "modeling": {
            "bootstrap_iterations": config.modeling.bootstrap_iterations,
            "poor_fit_p_value_threshold": config.modeling.poor_fit_p_value_threshold,
            "poor_fit_rsquared_threshold": config.modeling.poor_fit_rsquared_threshold,
            "random_seed": config.modeling.random_seed,
        },
        "qualification": config.qualification.model_dump(),
        "campaign": config.campaign.model_dump(),
        "glue_adjustment": config.glue_adjustment.model_dump(),
        "pricing_scenarios": [s.model_dump() for s in config.pricing_scenarios],
    }


def build_analysis_status(
    *,
    evm_gasfit_version: str,
    config: Config,
    input_paths: dict[str, Path | None],
    input_hashes: dict[str, str | None],
    manifest: dict[str, object] | None,
    manifest_sha256: str | None,
    config_document: str | None,
    planned_models: list[dict[str, object]],
    output_hashes: dict[str, str],
) -> dict[str, object]:
    """Assemble the deterministic analysis-status document."""
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "evm_gasfit_version": evm_gasfit_version,
        "inputs": {
            name: {
                "path": str(p) if p else None,
                "sha256": input_hashes.get(name),
                **(
                    {"content": config_document}
                    if name == "config" and config_document is not None
                    else {}
                ),
            }
            for name, p in input_paths.items()
        },
        "manifest": {
            "sha256": manifest_sha256,
            "content": manifest,
        },
        "policy": _policy_dict(config),
        "planned_models": planned_models,
        "outputs": dict(sorted(output_hashes.items())),
    }


def _json_scalar(value: object) -> object | None:
    """Convert a DataFrame scalar to strict JSON, mapping non-finite to null."""
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return str(value)


def _json_safe(value: object) -> object:
    """Recursively replace non-finite numbers before strict JSON encoding."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def planned_models_payload(qualification_df, model_specs) -> list[dict[str, object]]:
    """Flatten planned-model status with its exact model-by identity."""
    if qualification_df is None or qualification_df.empty:
        return []
    specs_by_label = {spec.source_label: spec for spec in model_specs}
    records: list[dict[str, object]] = []
    for _, row in qualification_df.iterrows():
        source_label = str(row.get("source_label", ""))
        spec = specs_by_label.get(source_label)
        group_values = {
            column: _json_scalar(row.get(column))
            for column in (spec.model_by if spec is not None else [])
        }
        record: dict[str, object] = {
            "source_label": source_label,
            "test_name": str(row.get("test_name", "")),
            "target_opcode": str(row.get("target_opcode", "")),
            "client_name": str(row.get("client_name", "")),
            "group_values": group_values,
            "status": str(row.get("status", "")),
            "reasons": str(row.get("reasons", "")),
            "adjusted_estimate_status": str(row.get("adjusted_estimate_status", "")),
        }
        for col in (
            "nobs",
            "n_sessions",
            "condition_number",
            "residual_curvature_r2",
            "holdout_session_error",
            "holdout_point_error",
            "relative_ci_width",
            "confidence_level",
        ):
            record[col] = _json_scalar(row.get(col))
        records.append(record)
    return records


def write_analysis_status(out_dir: Path, status: dict[str, object]) -> None:
    """Write ``analysis_status.json``; refuse to overwrite differing content."""
    out_dir = Path(out_dir)
    payload = (
        json.dumps(_json_safe(status), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    target = out_dir / STATUS_FILENAME
    if target.exists():
        existing = target.read_text()
        if existing == payload:
            return
        raise ConfigError(
            f"{target} already exists with different content; analysis "
            f"provenance is immutable — write to a fresh directory"
        )
    target.write_text(payload)


def read_analysis_status(analysis_dir: Path) -> dict[str, object]:
    """Load a previously written ``analysis_status.json``."""
    path = Path(analysis_dir) / STATUS_FILENAME
    if not path.exists():
        raise ConfigError(f"no {STATUS_FILENAME} under {analysis_dir}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return raw
