"""Manifest-gated comparison of two completed analysis directories.

A comparison is a regression claim only when the campaigns differ in nothing
but software: identical workloads, measurement boundary, hardware, and gas
schedule, with qualified values on both sides. Missing or mismatched
comparability metadata is a hard input error rather than a tempting
descriptive table; callers must repair provenance or choose a valid pair.

Both sides are read from their ``analysis_status.json`` (embedded manifest,
policy, planned-model statuses) plus ``new_gas.csv``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evm_gasfit.errors import ConfigError
from evm_gasfit.provenance import STATUS_FILENAME, read_analysis_status, sha256_file

_log = logging.getLogger("evm_gasfit.campaign")

# Manifest factor → candidate keys (matched case-insensitively, first hit
# wins). The manifest schema is owned by the campaign controller; gasfit
# only reads these descriptive sections to classify comparability.
_FACTOR_KEYS: dict[str, tuple[str, ...]] = {
    "software": ("software", "build", "sources", "revisions"),
    "hardware": ("hardware", "machine", "host"),
    "workloads": ("workloads", "workload", "corpus"),
    "boundary": ("boundary", "execution_boundary", "measurement"),
    "gas_schedule": ("gas_schedule", "schedule", "gas"),
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _gas_schedule_identity(value: object) -> str | None:
    """Return a normalized gas-schedule identity when it is explicit."""
    if isinstance(value, str):
        identity = value.strip()
        if identity and not identity.casefold().startswith(("unavailable", "unknown")):
            return identity.casefold()
        return None
    if not isinstance(value, dict):
        return None
    lowered = {str(key).casefold(): item for key, item in value.items()}
    if "active_schedule_identity" in lowered:
        return _gas_schedule_identity(lowered["active_schedule_identity"])
    for key in ("identity", "schedule", "fork", "name"):
        if key in lowered:
            identity = _gas_schedule_identity(lowered[key])
            if identity is not None:
                return identity
    return None


def _manifest_factor(manifest: dict[str, object] | None, factor: str) -> str | None:
    """Return a substantive canonical factor value, or None if unavailable."""
    if not isinstance(manifest, dict):
        return None
    source = manifest
    comparison_factors = manifest.get("comparison_factors")
    if factor != "software" and isinstance(comparison_factors, dict):
        source = comparison_factors
    lowered = {str(k).lower(): v for k, v in source.items()}
    for key in _FACTOR_KEYS[factor]:
        if key not in lowered:
            continue
        value = lowered[key]
        if factor == "gas_schedule":
            identity = _gas_schedule_identity(value)
            if identity is None:
                return None
            return _canonical(identity)
        if (
            value is None
            or (isinstance(value, str) and not value.strip())
            or value == {}
            or value == []
        ):
            return None
        return _canonical(value)
    return None


def _analysis_method(status: dict[str, object] | None) -> tuple[str, str] | None:
    """Return the archived config hash and analyzer version for comparison."""
    if not isinstance(status, dict):
        return None
    inputs = status.get("inputs")
    config = inputs.get("config") if isinstance(inputs, dict) else None
    config_hash = config.get("sha256") if isinstance(config, dict) else None
    version = status.get("evm_gasfit_version")
    if (
        not isinstance(config_hash, str)
        or not config_hash.strip()
        or not isinstance(version, str)
        or not version.strip()
    ):
        return None
    return config_hash, version


def _classify(
    baseline_manifest: dict[str, object] | None,
    candidate_manifest: dict[str, object] | None,
) -> tuple[str, list[str], bool]:
    """Return verdict, blocking non-software deltas, and software delta flag."""
    changed: list[str] = []
    unknown: list[str] = []
    software_changed = False
    for factor in _FACTOR_KEYS:
        base = _manifest_factor(baseline_manifest, factor)
        cand = _manifest_factor(candidate_manifest, factor)
        if base is None or cand is None:
            unknown.append(factor)
        elif base != cand:
            if factor == "software":
                software_changed = True
            else:
                changed.append(factor)
    if changed:
        return (
            "not comparable: " + ", ".join(f"{f} differs" for f in changed),
            changed,
            software_changed,
        )
    if unknown:
        return (
            "not comparable: provenance incomplete (unknown: "
            + ", ".join(unknown)
            + ")",
            changed,
            software_changed,
        )
    verdict = (
        "comparable: software differs"
        if software_changed
        else "comparable: identical declared conditions"
    )
    return verdict, changed, software_changed


def _load_new_gas(analysis_dir: Path, status: dict[str, object]) -> pd.DataFrame:
    path = Path(analysis_dir) / "new_gas.csv"
    if not path.exists():
        raise ConfigError(f"no new_gas.csv under {analysis_dir}")
    outputs = status.get("outputs")
    expected_hash = outputs.get("new_gas.csv") if isinstance(outputs, dict) else None
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ConfigError(
            "campaign comparison requires archived new_gas.csv hash under "
            f"{analysis_dir}"
        )
    if sha256_file(path) != expected_hash:
        raise ConfigError(
            "campaign comparison rejected: new_gas.csv hash differs under "
            f"{analysis_dir}"
        )
    return pd.read_csv(path)


def compare_campaigns(
    baseline_dir: Path, candidate_dir: Path, out_dir: Path
) -> dict[str, object]:
    """Compare two provenance-complete campaigns; write CSV + markdown.

    Both directories must carry manifests proving identical non-software
    conditions, the same archived analysis method, and a verified
    ``new_gas.csv``. A mismatch raises :class:`ConfigError` before any
    comparison artifact is written.
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    out_dir = Path(out_dir)

    status_paths = {
        "baseline": baseline_dir / STATUS_FILENAME,
        "candidate": candidate_dir / STATUS_FILENAME,
    }
    statuses: dict[str, dict[str, object] | None] = {}
    for side, path in status_paths.items():
        statuses[side] = read_analysis_status(path.parent) if path.exists() else None
    base_status = statuses["baseline"]
    cand_status = statuses["candidate"]

    base_manifest = None
    cand_manifest = None
    if isinstance(base_status, dict):
        block = base_status.get("manifest")
        if isinstance(block, dict):
            content = block.get("content")
            base_manifest = content if isinstance(content, dict) else None
    if isinstance(cand_status, dict):
        block = cand_status.get("manifest")
        if isinstance(block, dict):
            content = block.get("content")
            cand_manifest = content if isinstance(content, dict) else None

    if base_status is None or cand_status is None:
        raise ConfigError(
            "campaign comparison requires analysis_status.json in both inputs"
        )
    base_method = _analysis_method(base_status)
    cand_method = _analysis_method(cand_status)
    if base_method is None or cand_method is None:
        raise ConfigError(
            "campaign comparison requires archived analysis config hash and "
            "analyzer version"
        )
    if base_method != cand_method:
        raise ConfigError(
            "campaign comparison rejected: analysis config or analyzer version differs"
        )

    verdict, changed, software_changed = _classify(base_manifest, cand_manifest)
    if changed or verdict.startswith("not comparable"):
        raise ConfigError("campaign comparison rejected: " + verdict)

    base_df = _load_new_gas(baseline_dir, base_status)
    cand_df = _load_new_gas(candidate_dir, cand_status)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_indexed = base_df.set_index("gas_param", drop=False)
    cand_indexed = cand_df.set_index("gas_param", drop=False)
    all_params = sorted(set(base_indexed.index) | set(cand_indexed.index))

    def _num(df: pd.DataFrame, param: str, col: str) -> float:
        if param not in df.index or col not in df.columns:
            return float("nan")
        val = df.loc[param, col]
        return float(val) if val is not None and not pd.isna(val) else float("nan")

    def _status_of(df: pd.DataFrame, param: str) -> str:
        if "qualification_status" not in df.columns or param not in df.index:
            return ""
        val = df.loc[param, "qualification_status"]
        return "" if val is None or pd.isna(val) else str(val)

    rows: list[dict[str, object]] = []
    for param in all_params:
        base_rt = _num(base_indexed, param, "runtime_ms")
        cand_rt = _num(cand_indexed, param, "runtime_ms")
        ratio = (
            float(cand_rt / base_rt)
            if np.isfinite(base_rt) and base_rt != 0 and np.isfinite(cand_rt)
            else float("nan")
        )
        base_st = _status_of(base_indexed, param)
        cand_st = _status_of(cand_indexed, param)
        both_qualified = base_st == "qualified" and cand_st == "qualified"
        rows.append(
            {
                "gas_param": param,
                "baseline_runtime_ms": base_rt,
                "candidate_runtime_ms": cand_rt,
                "runtime_ratio": ratio,
                "baseline_new_gas": _num(base_indexed, param, "new_gas_rounded"),
                "candidate_new_gas": _num(cand_indexed, param, "new_gas_rounded"),
                "baseline_qualification": base_st,
                "candidate_qualification": cand_st,
                "supported_difference": bool(
                    both_qualified
                    and not changed
                    and np.isfinite(ratio)
                    and abs(ratio - 1.0) > 1e-9
                ),
            }
        )
    comparison_df = pd.DataFrame(rows)

    csv_path = out_dir / "campaign_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, lineterminator="\n")

    lines: list[str] = ["# Campaign comparison", ""]
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(f"- baseline: `{baseline_dir}`")
    lines.append(f"- candidate: `{candidate_dir}`")
    if isinstance(base_status, dict) and isinstance(cand_status, dict):
        base_hash = (
            base_status.get("manifest", {}).get("sha256")
            if isinstance(base_status.get("manifest"), dict)
            else None
        )
        cand_hash = (
            cand_status.get("manifest", {}).get("sha256")
            if isinstance(cand_status.get("manifest"), dict)
            else None
        )
        lines.append(f"- baseline manifest sha256: `{base_hash}`")
        lines.append(f"- candidate manifest sha256: `{cand_hash}`")
    lines.append(f"- software differs: `{software_changed}`")
    lines.append("")
    lines.append(
        "Only rows qualified on both sides count as supported "
        "differences, and software is the only permitted factor delta."
    )
    lines.append("")
    if not comparison_df.empty:
        lines.append(
            "| Gas param | Baseline ms | Candidate ms | Ratio | Base qual | Cand qual | Supported |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in comparison_df.iterrows():

            def _fmt(v: object, digits: int = 4) -> str:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "—"
                if isinstance(v, (int, np.integer)):
                    return str(int(v))
                return f"{float(v):.{digits}g}"

            lines.append(
                f"| {row['gas_param']} | {_fmt(row['baseline_runtime_ms'])} | "
                f"{_fmt(row['candidate_runtime_ms'])} | {_fmt(row['runtime_ratio'])} | "
                f"{row['baseline_qualification'] or '—'} | "
                f"{row['candidate_qualification'] or '—'} | "
                f"{bool(row['supported_difference'])} |"
            )
        lines.append("")
    (out_dir / "campaign_comparison.md").write_text("\n".join(lines))

    return {
        "verdict": verdict,
        "changed_factors": changed,
        "software_changed": software_changed,
        "rows": len(comparison_df),
    }
