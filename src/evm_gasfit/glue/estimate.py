"""Four-tier glue-opcode regression.

Pure-glue opcodes are fit one at a time as single-feature NNLS per
``(client, spec)``. Cycle-glue opcodes are fit jointly per client: a single
NNLS over the union of their driver fixtures with one feature per cycle
spec, where each feature is the row-wise sum of that spec's family
members (so DUP1..DUP16 collapse into one ``DUP`` feature).

Mixed-glue opcodes appear both as targets and as glues. They are fit per
``(client, spec)`` with the same single-feature shape as pure glue, but
the LHS is pre-adjusted by subtracting the contribution of every priced
upstream partner: for each partner ``p`` correlated with ``opcount`` on
this spec's driver test (per the detector's ratio table), subtract
``glue_runtime_ms_p · partner_count_per_fixture``. Bootstrap replicates use
the partner's aligned draws, preserving shared session covariance. ``mixed_a``
opcodes allow partners from ``pure ∪ cycle``; ``mixed_b`` opcodes also allow
``mixed_a`` partners. The four-pass order over the tier sequence makes the
dependency static — no topological sort, no cycle detection.

Specs without a driver fixture (``spec.test_name is None``) are skipped
silently; ``validate_inputs`` already emitted the warning at load time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evm_gasfit.config import Config
from evm_gasfit.modeling.nnls import fit_nnls
from evm_gasfit.modeling.results import NNLSResults

from .detect import compute_glue_opcodes_by_test
from .required import (
    PRICED_GLUE_SPECS,
    SPEC_BY_NAME,
    GlueOpcodeSpec,
    validate_inputs,
)

_log = logging.getLogger("evm_gasfit.glue")


@dataclass
class GlueEstimateOutput:
    """Glue results frame plus the per-(client, canonical-name) fits.

    Also carries ``glue_opcodes_by_test_df`` — the detector's per-test
    ratio table — so downstream consumers (proposal aggregator, reports)
    can read it without recomputing.
    """

    results_df: pd.DataFrame
    fits: dict[tuple[str, str], NNLSResults] = field(default_factory=dict)
    glue_opcodes_by_test_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _spec_member_filter(spec: GlueOpcodeSpec) -> set[str] | None:
    """Return the ``opcode``-param values driving this spec, or ``None`` if no filter needed."""
    if spec.test_opcode_filter is not None:
        return {spec.test_opcode_filter}
    if len(spec.members) > 1:
        return set(spec.members)
    return None


def _slice_for_spec(
    fixtures_df: pd.DataFrame, client: str, spec: GlueOpcodeSpec
) -> pd.DataFrame:
    slice_df = fixtures_df[
        (fixtures_df["client_name"] == client)
        & (fixtures_df["test_name"] == spec.test_name)
    ]
    member_filter = _spec_member_filter(spec)
    if member_filter is None:
        return slice_df
    # The parser prefixes raw fixture params with ``param_`` to avoid
    # collisions with opcode mnemonic columns (e.g. ``opcode`` becomes
    # ``param_opcode``); fall back to the bare name when the dataset
    # predates that convention.
    for col in ("param_opcode", "opcode"):
        if col in slice_df.columns:
            return slice_df[slice_df[col].isin(member_filter)]
    return slice_df


def _canonical_count(slice_df: pd.DataFrame, spec: GlueOpcodeSpec) -> np.ndarray:
    cols = [m for m in spec.members if m in slice_df.columns]
    if not cols:
        return np.zeros(len(slice_df), dtype=float)
    return slice_df[cols].astype(float).sum(axis=1).to_numpy()


def _session_groups(frame: pd.DataFrame, config: Config) -> pd.Series | None:
    """Return campaign session clusters when the driver rows carry them."""
    column = config.campaign.session_column
    return frame[column] if column in frame.columns else None


def _pure_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    spec: GlueOpcodeSpec,
) -> NNLSResults | None:
    slice_df = _slice_for_spec(fixtures_df, client, spec)
    if slice_df.empty:
        _log.warning(
            "glue pure-fit skipped: client=%s opcode=%s has no driver fixtures",
            client,
            spec.name,
        )
        return None
    counts = _canonical_count(slice_df, spec)
    if len(set(counts.tolist())) <= 1 or np.all(counts == 0):
        _log.warning(
            "glue pure-fit skipped: client=%s opcode=%s count is constant or zero",
            client,
            spec.name,
        )
        return None
    design = pd.DataFrame(
        {
            spec.name: counts,
            "test_runtime_ms": slice_df["test_runtime_ms"].astype(float).to_numpy(),
        }
    )
    try:
        return fit_nnls(
            design,
            features=[spec.name],
            target="test_runtime_ms",
            n_bootstrap=config.modeling.bootstrap_iterations,
            random_seed=config.modeling.random_seed,
            groups=_session_groups(slice_df, config),
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: any numerical failure means this fit attempt is unfit, not a crash
        _log.warning(
            "glue pure-fit failed: client=%s opcode=%s exc=%s",
            client,
            spec.name,
            exc,
        )
        return None


def _cycle_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    cycle_specs: list[GlueOpcodeSpec],
) -> NNLSResults | None:
    slices: list[pd.DataFrame] = []
    for spec in cycle_specs:
        if spec.test_name is None:
            continue
        slc = _slice_for_spec(fixtures_df, client, spec)
        if not slc.empty:
            slices.append(slc)
    if not slices:
        _log.warning("glue cycle-fit skipped: client=%s no driver rows", client)
        return None
    combined = pd.concat(slices, ignore_index=True)

    feature_names = [spec.name for spec in cycle_specs]
    design_cols: dict[str, np.ndarray] = {
        spec.name: _canonical_count(combined, spec) for spec in cycle_specs
    }
    design_cols["test_runtime_ms"] = (
        combined["test_runtime_ms"].astype(float).to_numpy()
    )
    design = pd.DataFrame(design_cols)
    try:
        return fit_nnls(
            design,
            features=feature_names,
            target="test_runtime_ms",
            n_bootstrap=config.modeling.bootstrap_iterations,
            random_seed=config.modeling.random_seed,
            groups=_session_groups(combined, config),
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: any numerical failure means this fit attempt is unfit, not a crash
        _log.warning("glue cycle-fit failed: client=%s exc=%s", client, exc)
        return None


def _select_partners(
    glue_by_test_df: pd.DataFrame,
    spec: GlueOpcodeSpec,
    allowed_partner_tiers: frozenset[str],
) -> list[str]:
    """Pick which priced glue partners contribute to this mixed fit's LHS.

    Uses the detector's ratio table to find every priced canonical name that
    correlates with ``opcount`` on the spec's driver test/target. Partners
    outside ``allowed_partner_tiers`` are dropped so the four-tier order can
    never be violated even if the detector lists a same-tier or later-tier
    partner.
    """
    if glue_by_test_df.empty:
        return []
    mask = (glue_by_test_df["test_name"] == spec.test_name) & (
        glue_by_test_df["target_opcode"] == spec.name
    )
    if not mask.any():
        return []
    # Preserve detector ordering, dedupe across model_by combos.
    seen: dict[str, None] = {}
    for name in glue_by_test_df.loc[mask, "glue_opcode"].astype(str).tolist():
        seen.setdefault(name, None)

    out: list[str] = []
    for name in seen:
        partner_spec = SPEC_BY_NAME.get(name)
        if partner_spec is None or partner_spec.name == spec.name:
            continue
        if partner_spec.tier not in allowed_partner_tiers:
            continue
        if partner_spec.test_name is None:
            # Priced canonical name with no driver fixture (POP, STOP) is
            # never a viable partner — no fit will ever exist for it.
            continue
        out.append(name)
    return out


def _mixed_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    spec: GlueOpcodeSpec,
    fits: dict[tuple[str, str], NNLSResults],
    glue_by_test_df: pd.DataFrame,
    allowed_partner_tiers: frozenset[str],
    p_threshold: float,
    r2_threshold: float,
) -> NNLSResults | None:
    """Single-feature NNLS with the LHS pre-adjusted by priced upstream partners."""
    slice_df = _slice_for_spec(fixtures_df, client, spec)
    if slice_df.empty:
        _log.warning(
            "glue mixed-fit skipped: client=%s opcode=%s has no driver fixtures",
            client,
            spec.name,
        )
        return None
    counts = _canonical_count(slice_df, spec)
    if len(set(counts.tolist())) <= 1 or np.all(counts == 0):
        _log.warning(
            "glue mixed-fit skipped: client=%s opcode=%s count is constant or zero",
            client,
            spec.name,
        )
        return None

    runtimes = slice_df["test_runtime_ms"].astype(float).to_numpy()
    adjusted = runtimes.copy()
    target_groups = _session_groups(slice_df, config)
    target_sessions = (
        None
        if target_groups is None
        else frozenset(str(group) for group in pd.unique(target_groups))
    )
    bootstrap_partners: list[tuple[np.ndarray, np.ndarray]] = []
    uncertainty_conditional = False
    bootstrap_rng = np.random.default_rng(config.modeling.random_seed)
    # Each candidate partner contributes only if its per-client fit passed
    # both the p-value and R² gates. Missing partners or partners that fail
    # either gate are skipped silently — the omission is already auditable
    # via the partner's row in ``glue_results.csv`` (NaN means no fit,
    # ``p_value`` and ``rsquared`` there record the gates).
    for partner_name in _select_partners(glue_by_test_df, spec, allowed_partner_tiers):
        partner_fit = fits.get((client, partner_name))
        if partner_fit is None:
            continue
        partner_ms = float(partner_fit.params.get(partner_name, float("nan")))
        partner_pval = float(partner_fit.pvalues.get(partner_name, float("nan")))
        partner_r2 = float(partner_fit.rsquared)
        if (
            not np.isfinite(partner_ms)
            or not np.isfinite(partner_pval)
            or not np.isfinite(partner_r2)
        ):
            continue
        if partner_pval >= p_threshold or partner_r2 < r2_threshold:
            continue
        partner_count = _canonical_count(slice_df, SPEC_BY_NAME[partner_name])
        adjusted -= partner_ms * partner_count

        partner_sessions = partner_fit.session_ids
        if partner_fit.uncertainty_conditional:
            uncertainty_conditional = True
            continue
        paired_by_session = (
            target_sessions is not None and target_sessions == partner_sessions
        )
        if target_sessions is None and partner_sessions is None:
            partner_draws = partner_fit.bootstrap_draws(partner_name)
        elif target_sessions is None or partner_sessions is None:
            uncertainty_conditional = True
            continue
        elif paired_by_session:
            partner_draws = partner_fit.bootstrap_draw_matrix(partner_name)
            if len(partner_draws) != config.modeling.bootstrap_iterations:
                uncertainty_conditional = True
                continue
        elif target_sessions & partner_sessions:
            uncertainty_conditional = True
            continue
        else:
            partner_draws = partner_fit.bootstrap_draws(partner_name)
        if len(partner_draws) == 0:
            uncertainty_conditional = True
            continue
        if not paired_by_session:
            partner_draws = partner_draws[
                bootstrap_rng.integers(
                    0,
                    len(partner_draws),
                    size=config.modeling.bootstrap_iterations,
                )
            ]
        bootstrap_partners.append((partner_count, partner_draws))

    bootstrap_target = None
    if bootstrap_partners and not uncertainty_conditional:

        def bootstrap_target(iteration: int) -> np.ndarray | None:
            bootstrap_runtimes = runtimes.copy()
            for partner_count, partner_draws in bootstrap_partners:
                partner_draw = partner_draws[iteration]
                if not np.isfinite(partner_draw):
                    return None
                bootstrap_runtimes -= partner_draw * partner_count
            return bootstrap_runtimes

    design = pd.DataFrame(
        {
            spec.name: counts,
            "test_runtime_ms": adjusted,
        }
    )
    try:
        return fit_nnls(
            design,
            features=[spec.name],
            target="test_runtime_ms",
            n_bootstrap=config.modeling.bootstrap_iterations,
            random_seed=config.modeling.random_seed,
            groups=target_groups,
            bootstrap_target=bootstrap_target,
            uncertainty_conditional=uncertainty_conditional,
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: any numerical failure means this fit attempt is unfit, not a crash
        _log.warning(
            "glue mixed-fit failed: client=%s opcode=%s exc=%s",
            client,
            spec.name,
            exc,
        )
        return None


def _row(client: str, name: str, fit: NNLSResults | None) -> dict[str, object]:
    if fit is None:
        return {
            "client_name": client,
            "glue_opcode": name,
            "nobs": 0,
            "glue_runtime_ms": float("nan"),
            "p_value": float("nan"),
            "rsquared": float("nan"),
        }
    return {
        "client_name": client,
        "glue_opcode": name,
        "nobs": int(fit.nobs),
        "glue_runtime_ms": float(fit.params[name]),
        "p_value": float(fit.pvalues[name]),
        "rsquared": float(fit.rsquared),
    }


_MIXED_A_PARTNER_TIERS: frozenset[str] = frozenset({"pure", "cycle"})
_MIXED_B_PARTNER_TIERS: frozenset[str] = frozenset({"pure", "cycle", "mixed_a"})


def estimate_glue(config: Config, fixtures_df: pd.DataFrame) -> GlueEstimateOutput:
    """Fit one NNLS per (client, canonical glue name) in four ordered passes.

    Pure-glue specs get a single-feature regression each; cycle-glue specs
    share a joint per-client fit; mixed-tier specs get single-feature fits
    with LHS pre-adjusted by upstream partners. Specs whose ``test_name is
    None`` are skipped (no row emitted). Raises ``ConfigError`` if any
    required driver test is absent from ``fixtures_df``.
    """
    validate_inputs(fixtures_df)
    glue_by_test_df = compute_glue_opcodes_by_test(
        fixtures_df,
        config.resolved_models,
        config.glue_adjustment.ratio_corr_eps,
        config.campaign.session_column,
    )

    rows: list[dict[str, object]] = []
    fits: dict[tuple[str, str], NNLSResults] = {}

    active_pure = [
        s for s in PRICED_GLUE_SPECS if s.tier == "pure" and s.test_name is not None
    ]
    active_cycle = [
        s for s in PRICED_GLUE_SPECS if s.tier == "cycle" and s.test_name is not None
    ]
    active_mixed_a = [
        s for s in PRICED_GLUE_SPECS if s.tier == "mixed_a" and s.test_name is not None
    ]
    active_mixed_b = [
        s for s in PRICED_GLUE_SPECS if s.tier == "mixed_b" and s.test_name is not None
    ]

    p_threshold = config.glue_adjustment.glue_contribution_p_value_threshold
    r2_threshold = config.glue_adjustment.glue_contribution_rsquared_threshold
    clients = sorted(fixtures_df["client_name"].unique())
    for client in clients:
        # Tier 1 — pure
        for spec in active_pure:
            fit = _pure_fit(fixtures_df, config, client, spec)
            rows.append(_row(client, spec.name, fit))
            if fit is not None:
                fits[(client, spec.name)] = fit

        # Tier 2 — cycle (joint)
        cycle_fit = _cycle_fit(fixtures_df, config, client, active_cycle)
        for spec in active_cycle:
            if cycle_fit is None:
                rows.append(_row(client, spec.name, None))
                continue
            rows.append(_row(client, spec.name, cycle_fit))
            fits[(client, spec.name)] = cycle_fit

        # Tier 3a — mixed, partners drawn from pure ∪ cycle
        for spec in active_mixed_a:
            fit = _mixed_fit(
                fixtures_df,
                config,
                client,
                spec,
                fits,
                glue_by_test_df,
                _MIXED_A_PARTNER_TIERS,
                p_threshold,
                r2_threshold,
            )
            rows.append(_row(client, spec.name, fit))
            if fit is not None:
                fits[(client, spec.name)] = fit

        # Tier 3b — mixed, partners drawn from pure ∪ cycle ∪ mixed_a
        for spec in active_mixed_b:
            fit = _mixed_fit(
                fixtures_df,
                config,
                client,
                spec,
                fits,
                glue_by_test_df,
                _MIXED_B_PARTNER_TIERS,
                p_threshold,
                r2_threshold,
            )
            rows.append(_row(client, spec.name, fit))
            if fit is not None:
                fits[(client, spec.name)] = fit

    results_df = pd.DataFrame(rows)
    active_names = [
        s.name for s in active_pure + active_cycle + active_mixed_a + active_mixed_b
    ]
    opcode_order = {name: i for i, name in enumerate(active_names)}
    results_df["_order"] = results_df["glue_opcode"].map(opcode_order)
    results_df = (
        results_df.sort_values(["client_name", "_order"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return GlueEstimateOutput(
        results_df=results_df,
        fits=fits,
        glue_opcodes_by_test_df=glue_by_test_df,
    )
