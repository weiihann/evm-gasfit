"""Public entry point: :class:`GasFit` drives the pipeline stages end-to-end."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evm_gasfit.config import Config, load_config
from evm_gasfit.errors import ConfigError
from evm_gasfit.glue import GlueEstimateOutput, estimate_glue
from evm_gasfit.io import FixtureMatchResult
from evm_gasfit.io.fixtures import build_fixtures_df
from evm_gasfit.io.opcounts import load_opcounts
from evm_gasfit.io.runtimes import load_runtimes
from evm_gasfit.modeling.estimate import EstimateOutput, estimate_models
from evm_gasfit.modeling.qualification import evaluate_qualification
from evm_gasfit.proposal.build import ProposalOutput, build_proposal
from evm_gasfit.reports.comparison import (
    build_comparison_df,
    build_pricing_scenarios_df,
    write_comparison_csv,
    write_pricing_scenarios_csv,
)
from evm_gasfit.reports.glue import write_glue_report
from evm_gasfit.reports.proposal import write_proposal_report
from evm_gasfit.reports.runtime import write_runtime_report

_log = logging.getLogger("evm_gasfit")


class _WarningCaptureHandler(logging.Handler):
    """Append every ``WARNING+`` record on the ``evm_gasfit`` logger to a list."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(self.format(record))


class GasFit:
    """High-level driver wrapping the pipeline stages.

    Attributes:
        config: The validated configuration loaded from YAML.
        runtimes_df: The raw runtimes frame loaded from CSV.
        opcounts: The parsed opcounts mapping loaded from JSON.
        fixtures_df: The merged per-fixture frame consumed by every stage.
        estimate_output: The output of :meth:`estimate_models`.
        glue_estimate_output: The output of :meth:`estimate_glue`, or ``None``.
        proposal_output: The output of :meth:`build_proposal`.
        qualification_df: Explicit status of every planned model.
        eligibility_df: The campaign eligibility ledger, or ``None`` when the
            runtimes input carried no phase/status metadata columns.
        manifest: The parsed campaign manifest, or ``None``.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.config_path: Path | None = None
        self.runtimes_path: Path | None = None
        self.opcounts_path: Path | None = None
        self.manifest_path: Path | None = None
        self.manifest: dict[str, object] | None = None
        self.manifest_sha256: str | None = None
        self.run_started_at: datetime = datetime.now(timezone.utc).replace(
            microsecond=0
        )
        self.runtimes_df: pd.DataFrame | None = None
        self.opcounts: dict[str, dict[str, float]] | None = None
        self.fixtures_df: pd.DataFrame | None = None
        self.fixture_match_result: FixtureMatchResult | None = None
        self.estimate_output: EstimateOutput | None = None
        self.glue_estimate_output: GlueEstimateOutput | None = None
        self.proposal_output: ProposalOutput | None = None
        self.qualification_df: pd.DataFrame | None = None
        self.eligibility_df: pd.DataFrame | None = None
        self._warnings: list[str] = []
        self._warning_handler = _WarningCaptureHandler(self._warnings)
        _log.addHandler(self._warning_handler)

    @classmethod
    def from_config(cls, path: Path) -> GasFit:
        """Load and validate the YAML config at ``path`` and return a fresh driver."""
        path = Path(path)
        pre_warnings: list[str] = []
        pre_handler = _WarningCaptureHandler(pre_warnings)
        _log.addHandler(pre_handler)
        try:
            config = load_config(path)
        finally:
            _log.removeHandler(pre_handler)
        fit = cls(config)
        fit._warnings[:0] = pre_warnings
        fit.config_path = path
        return fit

    def load_runtimes(self, path: Path) -> None:
        """Load the runtimes CSV at ``path``, restricted to ``config.clients``."""
        path = Path(path)
        df = load_runtimes(path)
        configured = list(self.config.clients)
        configured_set = set(configured)
        present = set(df["client_name"].astype(str))
        missing = sorted(configured_set - present)
        if missing:
            _log.warning(
                "missing-client: %d configured client(s) absent from runtimes CSV: %s",
                len(missing),
                ", ".join(repr(c) for c in missing),
            )
        self.runtimes_df = df[
            df["client_name"].astype(str).isin(configured_set)
        ].reset_index(drop=True)
        self.runtimes_path = path

    def load_opcounts(self, path: Path) -> None:
        """Load the opcounts JSON at ``path``."""
        path = Path(path)
        self.opcounts = load_opcounts(path)
        self.opcounts_path = path

    def load_manifest(self, path: Path) -> None:
        """Load the campaign manifest at ``path`` (JSON object).

        The manifest schema is owned by the campaign controller; gasfit only
        embeds it (and its SHA-256) into the analysis provenance so campaign
        comparisons can attribute differences to software, hardware,
        workload, boundary, or schedule factors.
        """
        from evm_gasfit.provenance import sha256_bytes, sha256_file

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        raw_bytes = path.read_bytes()
        try:
            raw = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest {path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"manifest {path} must be a JSON object")
        self.manifest = raw
        self.manifest_path = path
        # Hash the exact bytes so re-serialization differences never fake a
        # manifest change; the embedded copy is for factor comparison.
        self.manifest_sha256 = sha256_bytes(raw_bytes) or sha256_file(path)
        _log.info(
            "manifest loaded: %s (sha256=%s)",
            path,
            self.manifest_sha256[:12],
        )

    def _apply_eligibility(self) -> None:
        """Split campaign rows into eligible-for-calibration vs excluded.

        A legacy CSV without campaign phase or status metadata is unchanged.
        Campaign metadata is all-or-nothing: phase, status, and explicit
        correctness evidence must all be present before calibration can run.
        """
        assert self.runtimes_df is not None
        df = self.runtimes_df
        camp = self.config.campaign
        phase_col = camp.phase_column
        status_col = camp.status_column
        has_phase = phase_col in df.columns
        has_status = status_col in df.columns
        if not has_phase and not has_status:
            self.eligibility_df = None
            return
        if not has_phase or not has_status:
            raise ConfigError(
                "campaign runtimes require both "
                f"{phase_col!r} and {status_col!r} columns"
            )

        correctness_col = camp.correctness_column
        if correctness_col not in df.columns:
            raise ConfigError(
                "campaign runtimes require explicit "
                f"{correctness_col!r} evidence for calibration"
            )

        phase = df[phase_col].astype(str)
        status = df[status_col].astype(str)
        phase_ok = phase.isin(camp.eligible_phases)
        status_ok = status.isin(camp.eligible_statuses)
        correctness = df[correctness_col].astype(str).str.strip().str.lower()
        correctness_ok = correctness.isin({"true", "1", "yes"})
        eligible = phase_ok & status_ok & correctness_ok

        def _reason(i: int) -> str:
            reasons: list[str] = []
            if not phase_ok.iloc[i]:
                reasons.append(f"phase={phase.iloc[i]!r} not in eligible_phases")
            if not status_ok.iloc[i]:
                reasons.append(f"status={status.iloc[i]!r} not in eligible_statuses")
            if not correctness_ok.iloc[i]:
                reasons.append(
                    f"{correctness_col}={correctness.iloc[i]!r} is not passed"
                )
            return " and ".join(reasons)

        ledger = pd.DataFrame(
            {
                "fixture_name": df["fixture_name"].astype(str),
                "client_name": df["client_name"].astype(str),
                "session_id": (
                    df[camp.session_column].astype(str)
                    if camp.session_column in df.columns
                    else ""
                ),
                "sample_id": (
                    df[camp.sample_column].astype(str)
                    if camp.sample_column in df.columns
                    else ""
                ),
                "phase": phase,
                "status": status,
                "correctness_passed": correctness,
                "eligible": eligible,
                "reason": [
                    "" if is_eligible else _reason(i)
                    for i, is_eligible in enumerate(eligible)
                ],
            }
        )
        self.eligibility_df = ledger
        n_excluded = int((~eligible).sum())
        if n_excluded:
            reasons = ledger.loc[~eligible, "reason"].value_counts().sort_index()
            summary = "; ".join(
                f"{count}x {reason}" for reason, count in reasons.items()
            )
            _log.warning(
                "eligibility: %d of %d row(s) excluded from calibration (%s); "
                "see eligibility.csv — rows are ledgered, not erased",
                n_excluded,
                len(ledger),
                summary,
            )
        self.runtimes_df = df[eligible].reset_index(drop=True)

    def _ensure_fixtures(self) -> pd.DataFrame:
        if self.fixtures_df is not None:
            return self.fixtures_df
        if self.runtimes_df is None or self.opcounts is None:
            raise RuntimeError(
                "load_runtimes() and load_opcounts() must be called before fitting"
            )
        self._apply_eligibility()
        self.fixtures_df, self.fixture_match_result = build_fixtures_df(
            self.runtimes_df, self.opcounts
        )
        return self.fixtures_df

    def estimate_models(self) -> pd.DataFrame:
        """Fit each ``ModelSpec`` and populate :attr:`estimate_output`."""
        fixtures_df = self._ensure_fixtures()
        self.estimate_output = estimate_models(self.config, fixtures_df)
        assert self.estimate_output is not None
        self.qualification_df = evaluate_qualification(
            self.config, self.estimate_output.planned
        )
        return self.estimate_output.results_df

    def estimate_glue(self) -> pd.DataFrame:
        """Fit the priced glue opcodes and populate :attr:`glue_estimate_output`."""
        if self.estimate_output is None:
            raise RuntimeError("call estimate_models() before estimate_glue()")
        fixtures_df = self._ensure_fixtures()
        self.glue_estimate_output = estimate_glue(self.config, fixtures_df)
        return self.glue_estimate_output.results_df

    def build_proposal(self) -> pd.DataFrame:
        """Aggregate and apply derived params; populate :attr:`proposal_output`."""
        if self.estimate_output is None:
            raise RuntimeError("call estimate_models() before build_proposal()")
        fixtures_df = self._ensure_fixtures()
        self.proposal_output = build_proposal(
            self.config,
            self.estimate_output.results_df,
            self.glue_estimate_output,
            fixtures_df,
            qualification_df=self.qualification_df,
            planned=self.estimate_output.planned,
        )
        # ``build_proposal`` may downgrade adjusted-estimate statuses; keep
        # the authoritative copy on the driver so reports and provenance
        # agree.
        if not self.proposal_output.qualification_df.empty:
            self.qualification_df = self.proposal_output.qualification_df
        return self.proposal_output.new_gas_df

    @property
    def results_df(self) -> pd.DataFrame:
        if self.estimate_output is None:
            raise RuntimeError("call estimate_models() first")
        return self.estimate_output.results_df

    @property
    def glue_results_df(self) -> pd.DataFrame:
        if self.glue_estimate_output is None:
            raise RuntimeError("call estimate_glue() first")
        return self.glue_estimate_output.results_df

    @property
    def proposal_df(self) -> pd.DataFrame:
        if self.proposal_output is None:
            raise RuntimeError("call build_proposal() first")
        return self.proposal_output.new_gas_df

    def write_reports(self, out_dir: Path) -> None:
        """Write every CSV + markdown artifact (and figs, if plots enabled)."""
        if self.proposal_output is None:
            self.build_proposal()
        assert self.estimate_output is not None
        assert self.proposal_output is not None

        out_dir = Path(out_dir)
        from evm_gasfit.provenance import STATUS_FILENAME

        status_path = out_dir / STATUS_FILENAME
        if status_path.exists():
            raise ConfigError(
                f"{status_path} already exists; analysis outputs are immutable — "
                "write to a fresh directory"
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSVs first so plot/markdown writers can co-locate figs.
        self.estimate_output.results_df.to_csv(
            out_dir / "results.csv", index=False, lineterminator="\n"
        )
        self.proposal_output.new_gas_df.to_csv(
            out_dir / "new_gas.csv", index=False, lineterminator="\n"
        )
        self.proposal_output.new_gas_all_df.to_csv(
            out_dir / "new_gas_all_params.csv", index=False, lineterminator="\n"
        )
        if self.qualification_df is not None and not self.qualification_df.empty:
            self.qualification_df.to_csv(
                out_dir / "qualification.csv", index=False, lineterminator="\n"
            )
        if self.eligibility_df is not None:
            self.eligibility_df.to_csv(
                out_dir / "eligibility.csv", index=False, lineterminator="\n"
            )
        comparison_df = build_comparison_df(self.proposal_output, self.config)
        write_comparison_csv(out_dir, comparison_df)
        if self.config.pricing_scenarios:
            write_pricing_scenarios_csv(
                out_dir, build_pricing_scenarios_df(self.proposal_output, self.config)
            )

        glue_enabled = (
            self.config.glue_adjustment.enabled
            and self.glue_estimate_output is not None
        )
        if glue_enabled:
            assert self.glue_estimate_output is not None
            self.glue_estimate_output.results_df.to_csv(
                out_dir / "glue_results.csv", index=False, lineterminator="\n"
            )
            self.proposal_output.glue_opcodes_by_test_df.to_csv(
                out_dir / "glue_opcodes_by_test.csv",
                index=False,
                lineterminator="\n",
            )
            if not self.glue_estimate_output.detection_coverage_df.empty:
                self.glue_estimate_output.detection_coverage_df.to_csv(
                    out_dir / "glue_detection_coverage.csv",
                    index=False,
                    lineterminator="\n",
                )
            if not self.glue_estimate_output.driver_support_df.empty:
                self.glue_estimate_output.driver_support_df.to_csv(
                    out_dir / "glue_driver_support.csv",
                    index=False,
                    lineterminator="\n",
                )

        write_runtime_report(
            out_dir,
            self.estimate_output.results_df,
            self.estimate_output.fits,
            self.config,
        )
        if glue_enabled:
            assert self.glue_estimate_output is not None
            write_glue_report(
                out_dir,
                self.glue_estimate_output.results_df,
                self.glue_estimate_output.fits,
                self.config,
            )
        write_proposal_report(out_dir, self.proposal_output, self.config)
        self._write_meta(out_dir)
        self._write_analysis_status(out_dir)
        _log.removeHandler(self._warning_handler)

    def _write_meta(self, out_dir: Path) -> None:
        from evm_gasfit import __version__

        match = self.fixture_match_result
        dropped = sorted([*match.only_runtimes, *match.only_opcounts]) if match else []
        matched_n = len(match.matched) if match else 0
        in_runtimes_n = matched_n + (len(match.only_runtimes) if match else 0)
        in_opcounts_n = matched_n + (len(match.only_opcounts) if match else 0)
        n_excluded = (
            int((~self.eligibility_df["eligible"]).sum())
            if self.eligibility_df is not None
            else 0
        )
        meta = {
            "evm_gasfit_version": __version__,
            "run_started_at": self.run_started_at.isoformat(),
            "inputs": {
                "config": str(self.config_path) if self.config_path else None,
                "runtimes": str(self.runtimes_path) if self.runtimes_path else None,
                "opcounts": str(self.opcounts_path) if self.opcounts_path else None,
                "manifest": str(self.manifest_path) if self.manifest_path else None,
            },
            "campaign": {
                "manifest_sha256": self.manifest_sha256,
                "eligibility_policy": {
                    "phase_column": self.config.campaign.phase_column,
                    "status_column": self.config.campaign.status_column,
                    "correctness_column": self.config.campaign.correctness_column,
                    "require_correctness_passed": (
                        self.config.campaign.require_correctness_passed
                    ),
                    "eligible_phases": list(self.config.campaign.eligible_phases),
                    "eligible_statuses": list(self.config.campaign.eligible_statuses),
                },
                "rows_excluded_by_eligibility": n_excluded,
                "anchor_rate": self.config.anchor_rate,
            },
            "fixtures": {
                "in_runtimes": in_runtimes_n,
                "in_opcounts": in_opcounts_n,
                "matched": matched_n,
                "dropped": len(dropped),
            },
            "dropped_fixtures": dropped,
            "warnings": list(self._warnings),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _write_analysis_status(self, out_dir: Path) -> None:
        """Write the immutable provenance record, hashed over every artifact."""
        from evm_gasfit import __version__
        from evm_gasfit.provenance import (
            STATUS_FILENAME,
            build_analysis_status,
            planned_models_payload,
            sha256_file,
            write_analysis_status,
        )

        def _hash(path: Path | None) -> str | None:
            if path is None or not path.exists():
                return None
            return sha256_file(path)

        output_hashes: dict[str, str] = {}
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or path.name == STATUS_FILENAME:
                continue
            output_hashes[path.relative_to(out_dir).as_posix()] = sha256_file(path)

        config_document = (
            self.config_path.read_text() if self.config_path is not None else None
        )
        status = build_analysis_status(
            evm_gasfit_version=__version__,
            config=self.config,
            input_paths={
                "config": self.config_path,
                "runtimes": self.runtimes_path,
                "opcounts": self.opcounts_path,
                "manifest": self.manifest_path,
            },
            input_hashes={
                "config": _hash(self.config_path),
                "runtimes": _hash(self.runtimes_path),
                "opcounts": _hash(self.opcounts_path),
                "manifest": _hash(self.manifest_path),
            },
            manifest=self.manifest,
            manifest_sha256=self.manifest_sha256,
            config_document=config_document,
            output_hashes=output_hashes,
            planned_models=planned_models_payload(
                self.qualification_df, self.config.resolved_models
            ),
        )
        write_analysis_status(out_dir, status)
