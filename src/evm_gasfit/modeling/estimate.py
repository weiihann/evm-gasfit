"""Per-spec NNLS estimation entry point.

Consumes the validated :class:`Config` and the shared ``fixtures_df`` built by
``io/fixtures.py``, produces the canonical ``results.csv`` DataFrame plus a
parallel dict of :class:`NNLSResults` objects keyed by fit identity so the
reports layer can render summaries and diagnostics without refitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evm_gasfit.config import Config, ModelSpec
from evm_gasfit.errors import ConfigError, ModelingError

from .nnls import fit_nnls
from .results import NNLSResults

_log = logging.getLogger("evm_gasfit.estimate")


@dataclass
class PlannedFit:
    """One planned ``(spec, model_by-combo, client)`` fit, fitted or not.

    The qualification layer emits a status for *every* planned fit; this
    record is how a fit that never ran (empty spec slice, rank-deficient
    design, solver failure) stays visible with its skip reason instead of
    disappearing from the outputs.
    """

    source_label: str
    test_name: str
    target_opcode: str | None
    group_values: dict[str, str]
    client: str
    fit: NNLSResults | None = None
    skip_reason: str | None = None
    dropped_features: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        return (
            self.source_label,
            self.test_name,
            self.target_opcode or "",
            *[self.group_values[c] for c in sorted(self.group_values)],
            self.client,
        )


@dataclass
class EstimateOutput:
    """Bundle of ``results_df`` and the parallel ``fits`` dict.

    ``fits`` is keyed by
    ``(source_label, test_name, target_opcode, *model_by_values, client_name)``
    so the reports layer can look up the underlying :class:`NNLSResults` for
    each row in ``results_df``. ``source_label`` leads the key so two specs
    sharing test_name + target + model_by (differing only in ``filter_by``)
    don't overwrite each other's fit. ``planned`` carries one record per
    planned fit — including the ones that never produced a row — for the
    qualification layer.
    """

    results_df: pd.DataFrame
    fits: dict[tuple, NNLSResults] = field(default_factory=dict)
    planned: list[PlannedFit] = field(default_factory=list)


def _empty_results_df(config: Config) -> pd.DataFrame:
    """Return the successful-fit schema when a campaign has no eligible rows."""
    model_by_cols = sorted(
        {c for spec in config.resolved_models for c in spec.model_by}
    )
    feature_names: list[str] = []
    for spec in config.resolved_models:
        for name in [*spec.model_params, *spec.setup_params]:
            if name != "target_coef" and name not in feature_names:
                feature_names.append(name)

    columns = [
        "test_name",
        "client_name",
        "target_opcode",
        "source_label",
        *model_by_cols,
        "nobs",
        "intercept_runtime_ms",
        "intercept_pvalue",
        "rsquared",
        "rsquared_adj",
        "target_coef_runtime_ms",
        "target_coef_pvalue",
        "target_coef_conf_int_low",
        "target_coef_conf_int_high",
        "condition_number",
        "n_sessions",
    ]
    for name in feature_names:
        columns.extend(
            [
                f"{name}_runtime_ms",
                f"{name}_pvalue",
                f"{name}_conf_int_low",
                f"{name}_conf_int_high",
            ]
        )
    return pd.DataFrame(columns=columns)


def _apply_filters(df: pd.DataFrame, filter_by: list[str]) -> pd.DataFrame:
    """AND-substring-match ``filter_by`` tokens against ``fixture_name``.

    A ``!``-prefixed token negates: ``!foo`` requires that ``foo`` is absent
    from the fixture name. All tokens are ANDed together.
    """
    if not filter_by:
        return df
    mask = pd.Series(True, index=df.index)
    for token in filter_by:
        if token.startswith("!"):
            mask &= ~df["fixture_name"].str.contains(token[1:], regex=False)
        else:
            mask &= df["fixture_name"].str.contains(token, regex=False)
    return df[mask]


def _materialize_derived(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Materialize each ``fixture_params`` entry as a float column on ``df``."""
    if not spec.fixture_params:
        return df
    df = df.copy()
    for derived_name, fp_spec in spec.fixture_params.items():
        source = fp_spec.source
        if source not in df.columns:
            raise ModelingError(
                f"spec test_name={spec.test_name!r}: fixture_params[{derived_name!r}] "
                f"source column {source!r} is missing on the filtered fixtures"
            )
        source_col = df[source].astype(str)
        if fp_spec.values is not None:
            observed = set(source_col.unique())
            unmapped = observed - set(fp_spec.values)
            if unmapped:
                raise ModelingError(
                    f"spec test_name={spec.test_name!r}: fixture_params[{derived_name!r}] "
                    f"values map omits observed source value(s) {sorted(unmapped)!r}"
                )
            df[derived_name] = source_col.map(fp_spec.values).astype(float)
        else:
            try:
                numeric = source_col.astype(float)
            except (TypeError, ValueError) as exc:
                raise ModelingError(
                    f"spec test_name={spec.test_name!r}: fixture_params[{derived_name!r}] "
                    f"source {source!r} contains non-numeric values; supply a 'values:' map"
                ) from exc
            if fp_spec.transform == "bytes_to_words":
                numeric = np.ceil(numeric.to_numpy() / 32.0).astype(float)
            df[derived_name] = numeric
    return df


def _resolve_target_opcode(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Fill a ``target_opcode`` column per the spec's target rule."""
    df = df.copy()
    if spec.target_operation is not None:
        df["target_opcode"] = spec.target_operation
    else:
        param = spec.target_operation_param
        if param not in df.columns:
            raise ModelingError(
                f"spec test_name={spec.test_name!r}: target_operation_param "
                f"{param!r} is missing on the filtered fixtures"
            )
        if df[param].isna().any():
            missing = df.loc[df[param].isna(), "fixture_name"].tolist()
            raise ModelingError(
                f"spec test_name={spec.test_name!r}: target_operation_param "
                f"{param!r} is null on fixtures: {missing[:5]!r}"
            )
        df["target_opcode"] = df[param].astype(str)
    return df


def _split_baseline_pair(
    df: pd.DataFrame,
    spec: ModelSpec,
    session_column: str | None,
) -> pd.DataFrame:
    """Replace each ``overhead_baseline_False`` row with its delta against the
    matching ``overhead_baseline_True`` control, dropping the ``True`` rows.

    The transform differences runtime, ``opcount``, and every per-opcode
    count. A control may execute fixed production-tail work, so leaving
    ``opcount`` untouched would make the post-pair target-count invariant
    compare variable work with the false row's total count.

    ``overhead_baseline_match_params`` selects the control's shape key. The
    legacy ``None`` setting matches every populated raw parameter except the
    baseline flag. An explicit empty list matches only client and session;
    a nonempty list names the raw parameters that must match. The selected
    ``True`` rows are averaged per key before merging, so repeated controls
    lower baseline noise without fanning out false trials.
    """
    col = spec.overhead_baseline_param
    if col is None:
        return df
    baseline_flags = df[col].astype("string").str.strip().str.casefold()
    false_df = df[baseline_flags == "false"]
    true_df = df[baseline_flags == "true"]
    match_params = spec.overhead_baseline_match_params
    if match_params is None:
        # Only columns with at least one real value in this slice discriminate
        # fixtures; all-NaN columns originate in other benchmark families.
        pair_cols = [
            c
            for c in df.columns
            if c.startswith("param_") and c != col and df[c].notna().any()
        ]
    else:
        missing = [
            c for c in match_params if c not in df.columns or not df[c].notna().any()
        ]
        if missing:
            logical_names = [c.removeprefix("param_") for c in missing]
            raise ConfigError(
                f"spec test_name={spec.test_name!r}: "
                "overhead_baseline_match_params references missing or empty "
                f"fixture parameter(s) {logical_names!r}"
            )
        pair_cols = list(match_params)
    pair_cols.append("client_name")
    if session_column is not None and session_column in df.columns:
        pair_cols.append(session_column)
    # Read attrs before the merge below: DataFrame.attrs is not guaranteed to
    # survive it. ``opcount`` is intentionally explicit because it is not an
    # opcode mnemonic but must track the target-count delta.
    opcode_columns = set(df.attrs.get("opcode_columns", []))
    diff_cols = list(
        dict.fromkeys(
            c
            for c in ["test_runtime_ms", "opcount", *(sorted(opcode_columns))]
            if c in df.columns
        )
    )
    true_baseline = (
        true_df.groupby(pair_cols, dropna=False)[diff_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"{c}__baseline" for c in diff_cols})
    )
    merged = false_df.merge(
        true_baseline,
        on=pair_cols,
        how="left",
        validate="many_to_one",
    )
    unmatched = merged["test_runtime_ms__baseline"].isna()
    if unmatched.any():
        examples = merged.loc[unmatched, "fixture_name"].tolist()[:5]
        raise ConfigError(
            f"spec test_name={spec.test_name!r}: {int(unmatched.sum())} "
            f"overhead_baseline_False fixture(s) have no matching "
            f"overhead_baseline_True counterpart, e.g. {examples!r}"
        )
    for c in diff_cols:
        merged[c] = merged[c] - merged[f"{c}__baseline"]
    return merged.drop(columns=[f"{c}__baseline" for c in diff_cols])


def _enforce_opcount_invariant(df: pd.DataFrame, spec: ModelSpec) -> None:
    """Check ``opcount == row[count_source]`` per the input invariant.

    For ordinary opcode targets the count source is the resolved target opcode.
    For precompile specs the target is a display name with no opcode column, so
    the invariant uses ``PRECOMPILE_0x<40 lowercase hexadecimal address>``:
    the destination-semantic invocation key. It is never a bare ``STATICCALL``
    count, which includes wrapper calls and overstates the target work.
    """
    count_source_override = spec.target_operation_count_source
    for _, row in df.iterrows():
        count_source = count_source_override or row["target_opcode"]
        if count_source not in df.columns:
            raise ConfigError(
                f"fixture {row['fixture_name']!r}: count source {count_source!r} "
                f"has no per-opcode count column"
            )
        expected = row["opcount"]
        actual = row[count_source]
        if pd.isna(actual) or float(expected) != float(actual):
            raise ConfigError(
                f"fixture {row['fixture_name']!r}: opcount={expected} disagrees with "
                f"per-opcode count for {count_source!r}={actual} "
                f"(spec test_name={spec.test_name!r})"
            )
        if float(expected) == 0:
            raise ConfigError(
                f"fixture {row['fixture_name']!r}: opcount=0 for target "
                f"{row['target_opcode']!r} (spec test_name={spec.test_name!r}); "
                f"fixtures that don't execute the target opcode don't belong in "
                f"this group — narrow filter_by or drop the fixture"
            )


def _build_design(
    df: pd.DataFrame,
    spec: ModelSpec,
    session_column: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], pd.Series | None]:
    """Build the design matrix for one (group, client) slice.

    Returns the (renamed) frame, surviving interaction and setup features,
    configured features dropped for having one observed value, and per-row
    session labels (``None`` when the campaign has no session column).
    Sessions ride along on the design frame purely as grouping metadata —
    they are never a regressor.
    """
    design = pd.DataFrame(
        {
            "opcount": df["opcount"].astype(float).to_numpy(),
            "test_runtime_ms": df["test_runtime_ms"].astype(float).to_numpy(),
        },
        index=df.index,
    )
    sessions: pd.Series | None = None
    if session_column is not None and session_column in df.columns:
        sessions = df[session_column].astype(str).reset_index(drop=True)

    def _resolve_source(coef_name: str) -> str:
        # A model_params/setup_params key can reference either a derived
        # column produced by ``_materialize_derived`` (its natural name) or a
        # raw parsed-param column (exposed as ``param_<name>`` by
        # ``build_fixtures_df``).
        if coef_name in df.columns:
            return coef_name
        if f"param_{coef_name}" in df.columns:
            return f"param_{coef_name}"
        raise ModelingError(
            f"spec test_name={spec.test_name!r}: coefficient {coef_name!r} "
            f"has no matching fixture-param column"
        )

    extras: list[str] = []
    dropped_features: list[str] = []
    for coef_name in spec.model_params:
        if coef_name == "target_coef":
            continue
        param_vals = df[_resolve_source(coef_name)].astype(float).to_numpy()
        if len(set(param_vals.tolist())) <= 1:
            _log.warning(
                "spec test_name=%r: dropping extra feature %r — single unique value "
                "across the filtered fixtures",
                spec.test_name,
                coef_name,
            )
            dropped_features.append(coef_name)
            continue
        design[coef_name] = design["opcount"].to_numpy() * param_vals
        extras.append(coef_name)

    # Setup features: n-independent terms whose value is the fixture-param
    # itself (e.g. an input length in words), NOT multiplied by opcount.
    # A setup cost that grows with input length is a real per-workload cost;
    # folding it into the intercept only works when the length never varies,
    # and these features exist precisely for sweeps where it does.
    setup_features: list[str] = []
    for coef_name in spec.setup_params:
        param_vals = df[_resolve_source(coef_name)].astype(float).to_numpy()
        if len(set(param_vals.tolist())) <= 1:
            _log.warning(
                "spec test_name=%r: dropping setup feature %r — single unique "
                "value across the filtered fixtures",
                spec.test_name,
                coef_name,
            )
            dropped_features.append(coef_name)
            continue
        design[coef_name] = param_vals
        setup_features.append(coef_name)
    return design, extras, setup_features, dropped_features, sessions


def _fit_or_skip(
    design: pd.DataFrame,
    features: list[str],
    config: Config,
    spec: ModelSpec,
    client: str,
    group_label: str,
    sessions: pd.Series | None = None,
) -> tuple[NNLSResults | None, str | None]:
    """Run NNLS or log + skip per the §4.2 failure modes.

    Returns ``(fit, None)`` on success and ``(None, reason)`` on skip. The
    reason feeds the qualification ledger so a planned model that produced
    no fit keeps an explicit ``failed`` status instead of vanishing.
    """
    n_features_with_const = len(features) + 1
    if len(design) < n_features_with_const + 1:
        _log.warning(
            "spec test_name=%r group=%s client=%s: nobs=%d < n_features+1=%d, skipping",
            spec.test_name,
            group_label,
            client,
            len(design),
            n_features_with_const + 1,
        )
        return None, "too few observations for the design (nobs < n_features+1)"
    opcount = design["opcount"].to_numpy()
    if len(set(opcount.tolist())) <= 1 or np.all(opcount == 0):
        _log.warning(
            "spec test_name=%r group=%s client=%s: opcount is constant or zero, skipping",
            spec.test_name,
            group_label,
            client,
        )
        return None, "opcount is constant or zero across the filtered rows"
    feature_matrix = design[features].to_numpy(dtype=float)
    design_with_const = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
    if np.linalg.matrix_rank(design_with_const) < design_with_const.shape[1]:
        _log.warning(
            "spec test_name=%r group=%s client=%s: design matrix is rank-deficient, skipping",
            spec.test_name,
            group_label,
            client,
        )
        # Additional repetitions cannot separate proportional regressors —
        # this is the collapsed-design case the plan calls out explicitly.
        return None, "design matrix is rank-deficient (proportional regressors)"
    try:
        return (
            fit_nnls(
                design,
                features=features,
                target="test_runtime_ms",
                n_bootstrap=config.modeling.bootstrap_iterations,
                random_seed=config.modeling.random_seed,
                groups=sessions,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: any numerical failure means this fit attempt is unfit, not a crash
        _log.warning(
            "spec test_name=%r group=%s client=%s: NNLS solver raised %s, skipping",
            spec.test_name,
            group_label,
            client,
            exc,
        )
        return None, f"NNLS solver raised {exc}"


def _build_result_row(
    *,
    spec: ModelSpec,
    client: str,
    target_opcode: str,
    group_values: dict[str, str],
    fit: NNLSResults,
    extras: list[str],
    setup_features: list[str],
    confidence_level: float,
) -> dict[str, object]:
    ci = fit.conf_int(alpha=1.0 - confidence_level)
    row: dict[str, object] = {
        "test_name": spec.test_name,
        "client_name": client,
        "target_opcode": target_opcode,
        # Provenance: the exact resolved spec that produced this fit. Two specs
        # sharing test_name + target + model_by (differing only in filter_by)
        # land on identical key columns, so the aggregator routes rows back to
        # their spec by this label rather than by the key shape.
        "source_label": spec.source_label,
    }
    row.update(group_values)
    row.update(
        {
            "nobs": fit.nobs,
            "intercept_runtime_ms": float(fit.params["const"]),
            "intercept_pvalue": float(fit.pvalues["const"]),
            "rsquared": float(fit.rsquared),
            "rsquared_adj": float(fit.rsquared_adj),
            "target_coef_runtime_ms": float(fit.params["opcount"]),
            "target_coef_pvalue": float(fit.pvalues["opcount"]),
            "target_coef_conf_int_low": float(ci.loc["opcount", 0]),
            "target_coef_conf_int_high": float(ci.loc["opcount", 1]),
            # Conditioning + session evidence, computed from the same design
            # the fit saw. Holdout errors and residual curvature live on the
            # qualification table; these two columns are cheap enough to sit
            # on every row of results.csv.
            "condition_number": float(fit.condition_number),
            "n_sessions": (
                len(np.unique(fit.groups)) if fit.groups is not None else np.nan
            ),
        }
    )
    for extra in extras + setup_features:
        row[f"{extra}_runtime_ms"] = float(fit.params[extra])
        row[f"{extra}_pvalue"] = float(fit.pvalues[extra])
        row[f"{extra}_conf_int_low"] = float(ci.loc[extra, 0])
        row[f"{extra}_conf_int_high"] = float(ci.loc[extra, 1])
    return row


def estimate_models(config: Config, fixtures_df: pd.DataFrame) -> EstimateOutput:
    """Fit one NNLS model per ``(spec, model_by-combo, client)``.

    Args:
        config: The validated configuration whose ``resolved_models`` drives
            iteration order.
        fixtures_df: The shared per-fixture frame produced by
            ``io.fixtures.build_fixtures_df``.

    Returns:
        :class:`EstimateOutput` carrying the canonical ``results.csv`` frame
        (one row per successful fit) and a parallel dict of fit objects.

    Raises:
        ModelingError: If every fit is skipped for a legacy, non-campaign
            input.
    """
    rows: list[dict[str, object]] = []
    fits: dict[tuple, NNLSResults] = {}
    planned: list[PlannedFit] = []
    unmatched: list[ModelSpec] = []
    # Calibration-lane rows (param_campaign_role=calibration) are glue-driver
    # evidence, never target-model input: an overly broad spec selector must
    # not absorb a straight-line driver sweep into a priced target fit. The
    # glue estimator consumes the unfiltered frame.
    # Local import: evm_gasfit.glue's __init__ pulls this module back in
    # (via glue.detect), so a module-level import would be circular.
    from evm_gasfit.glue.lane import split_campaign_lanes

    fixtures_df, calibration_df = split_campaign_lanes(fixtures_df)
    if not calibration_df.empty:
        _log.warning(
            "campaign-lanes: %d calibration row(s) excluded from every target "
            "model fit (glue drivers only); target rows: %d",
            len(calibration_df),
            len(fixtures_df),
        )
    session_column = config.campaign.session_column
    if session_column not in fixtures_df.columns:
        # Legacy three-column CSVs carry no session metadata; pairing and
        # inference fall back to their ordinary non-campaign forms.
        session_column = None

    campaign_columns = (
        config.campaign.phase_column,
        config.campaign.status_column,
        config.campaign.correctness_column,
    )
    campaign_data = all(column in fixtures_df.columns for column in campaign_columns)

    for spec in config.resolved_models:
        slice_df = fixtures_df[fixtures_df["test_name"] == spec.test_name]
        slice_df = _apply_filters(slice_df, spec.filter_by)
        if slice_df.empty:
            _log.warning(
                "spec test_name=%r had no matching fixtures after filter_by=%r; skipping",
                spec.test_name,
                spec.filter_by,
            )
            unmatched.append(spec)
            for client in config.clients:
                planned.append(
                    PlannedFit(
                        source_label=spec.source_label,
                        test_name=spec.test_name,
                        target_opcode=None,
                        group_values={},
                        client=client,
                        skip_reason="no matching fixtures after filter_by",
                    )
                )
            continue

        slice_df = _resolve_target_opcode(slice_df, spec)
        slice_df = _split_baseline_pair(slice_df, spec, session_column)
        _enforce_opcount_invariant(slice_df, spec)
        slice_df = _materialize_derived(slice_df, spec)

        # Validate the model_by columns exist on the slice.
        for col in spec.model_by:
            if col not in slice_df.columns:
                raise ModelingError(
                    f"spec test_name={spec.test_name!r}: model_by column "
                    f"{col!r} is not present on the filtered fixtures"
                )

        # Group iteration. When model_by is empty there is one group: the whole slice.
        if spec.model_by:
            groups = slice_df.groupby(spec.model_by, dropna=False, sort=True)
        else:
            groups = [((), slice_df)]
        if hasattr(groups, "__iter__") and not isinstance(groups, list):
            group_iter = list(groups)
        else:
            group_iter = groups

        for group_key, group_df in group_iter:
            if not spec.model_by:
                group_values: dict[str, str] = {}
            else:
                if not isinstance(group_key, tuple):
                    key_tuple = (group_key,)
                else:
                    key_tuple = group_key
                group_values = {col: val for col, val in zip(spec.model_by, key_tuple)}
            group_label = "/".join(f"{k}={v}" for k, v in group_values.items()) or "all"

            target_opcodes = set(group_df["target_opcode"].unique())
            if len(target_opcodes) != 1:
                raise ModelingError(
                    f"spec test_name={spec.test_name!r} group={group_label}: "
                    f"multiple target_opcodes in one group: {sorted(target_opcodes)!r}"
                )
            target_opcode = next(iter(target_opcodes))

            for client in config.clients:
                client_df = group_df[group_df["client_name"] == client]
                if client_df.empty:
                    planned.append(
                        PlannedFit(
                            source_label=spec.source_label,
                            test_name=spec.test_name,
                            target_opcode=target_opcode,
                            group_values=group_values,
                            client=client,
                            skip_reason="no eligible fixtures for configured client",
                        )
                    )
                    continue
                (
                    design,
                    extras,
                    setup_features,
                    dropped_features,
                    sessions,
                ) = _build_design(client_df, spec, session_column)
                features = ["opcount"] + extras + setup_features
                fit, skip_reason = _fit_or_skip(
                    design, features, config, spec, client, group_label, sessions
                )
                planned.append(
                    PlannedFit(
                        source_label=spec.source_label,
                        test_name=spec.test_name,
                        target_opcode=target_opcode,
                        group_values=group_values,
                        client=client,
                        fit=fit,
                        skip_reason=skip_reason,
                        dropped_features=tuple(dropped_features),
                    )
                )
                if fit is None:
                    continue
                row = _build_result_row(
                    spec=spec,
                    client=client,
                    target_opcode=target_opcode,
                    group_values=group_values,
                    fit=fit,
                    extras=extras,
                    setup_features=setup_features,
                    confidence_level=config.qualification.confidence_level,
                )
                rows.append(row)
                fit_key = (
                    spec.source_label,
                    spec.test_name,
                    target_opcode,
                    *[group_values[c] for c in spec.model_by],
                    client,
                )
                fits[fit_key] = fit

    # A spec that a config names explicitly is an assertion that it should
    # apply, so re-state the zero-match specs as one line after the per-spec
    # warnings have scrolled past. Suite renames land here first.
    if unmatched:
        _log.warning(
            "%d model spec(s) matched no fixtures and were skipped: %s",
            len(unmatched),
            "; ".join(
                f"{spec.source_label} (test_name={spec.test_name!r})"
                for spec in unmatched
            ),
        )

    if not rows:
        if campaign_data:
            return EstimateOutput(
                results_df=_empty_results_df(config),
                fits=fits,
                planned=planned,
            )
        raise ModelingError(
            "every model spec was skipped — no rows produced for results.csv"
        )

    results_df = pd.DataFrame(rows)
    # Deterministic ordering.
    sort_cols = ["test_name", "target_opcode"]
    sort_cols += sorted({c for spec in config.resolved_models for c in spec.model_by})
    sort_cols += ["client_name", "source_label"]
    sort_cols = [c for c in sort_cols if c in results_df.columns]
    results_df = results_df.sort_values(sort_cols, kind="mergesort").reset_index(
        drop=True
    )

    return EstimateOutput(results_df=results_df, fits=fits, planned=planned)
