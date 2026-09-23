"""Four-tier glue-opcode regression.

Pure-glue opcodes are fit one at a time as single-feature NNLS per
``(client, spec)``. POP uses a dedicated constant-seed driver (fixed
PUSH0 setup, POP×N, no cleanup): its constant seed rides the intercept,
the slope is identified by its own sweep, and the resulting pure fit
anchors POP for every paired grower driver that subtracts it as a
partner. Cycle-glue opcodes are fit jointly per client: a single NNLS
over the union of their driver fixtures with one feature per cycle spec,
where each feature is the row-wise sum of that spec's family members (so
DUP1..DUP16 collapse into one ``DUP`` feature). The joint design uses
per-driver fixed effects (within-driver demeaning) after pure-tier
partner subtraction, so coefficients are identified by each driver's own
count sweep — never by setup-level differences between drivers.

Mixed-glue opcodes appear both as targets and as glues. They are fit per
``(client, spec)`` with the same single-feature shape as pure glue, but
the LHS is pre-adjusted by subtracting the contribution of every priced
upstream partner correlated with the spec's own driver count *within the
driver slice*: for each partner ``p``, subtract ``glue_runtime_ms_p ·
partner_count_per_row``. Bootstrap replicates use the partner's aligned
draws, preserving shared session covariance. ``mixed_a`` opcodes allow
partners from ``pure ∪ cycle``; ``mixed_b`` opcodes also allow ``mixed_a``
partners. The four-pass order over the tier sequence makes the dependency
static — no topological sort, no cycle detection. Cycle fits subtract
``pure``-tier partners the same way, so a driver's background pure support
(e.g. POP or ISZERO around a DUP loop) is charged to the pure coefficient,
not absorbed into the cycle family's.

Partner selection is slice-local: a partner qualifies when its canonical
count correlates with the spec's own canonical count on the spec's driver
rows. It no longer consults the target-test detector table, which cannot
see calibration-only driver tests at all.

Lane preference
---------------

Driver slices prefer calibration rows: when ``param_campaign_role``
marks any row of a slice as calibration, the slice is restricted to those
rows (clean, straight-line drivers); otherwise the slice keeps whatever
rows the exporter shipped (target-lane drivers for the mixed tier).

Isolation
---------

A glue row is ``isolated`` only when every count-correlated priced support
in its driver slice was actually subtracted. Pure fits sit in tier 1 —
nothing is priced yet — so any correlated priced opcode in a pure driver
marks the fit non-isolated; the calibration drivers must be straight-line.
Non-isolated rows keep their point estimate (labeled research output) but
downstream adjustment treats their opcode as unpriced and blocks the
target's isolated recommendation rather than subtracting a contaminated
coefficient.

Specs without a driver fixture (``spec.test_name is None``) are skipped
silently; ``validate_inputs`` already emitted the warning at load time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evm_gasfit.config import Config
from evm_gasfit.modeling.nnls import fit_nnls
from evm_gasfit.modeling.results import NNLSResults

from .detect import (
    _RATIO_FLOOR,
    _ols_slope,
    compute_detection_coverage,
    compute_glue_opcodes_by_test,
)
from .lane import CAMPAIGN_ROLE_COLUMN
from .required import (
    MEMBER_TO_CANONICAL,
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
    ratio table — plus ``detection_coverage_df`` (per-target-group detection
    status) and ``driver_support_df`` (every count-correlated supporting
    opcode measured on each calibration driver slice) so downstream
    consumers (proposal aggregator, adjuster, reports) can read them
    without recomputing.
    """

    results_df: pd.DataFrame
    fits: dict[tuple[str, str], NNLSResults] = field(default_factory=dict)
    glue_opcodes_by_test_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    detection_coverage_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_support_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _PartnerPlan:
    """One gate-passing partner, ready for LHS subtraction."""

    name: str
    count: np.ndarray
    ms: float
    draws: np.ndarray | None  # per-bootstrap draws, aligned to target or marginal
    aligned: bool  # True when draws share the target's session clusters


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
    if member_filter is not None:
        # The parser prefixes raw fixture params with ``param_`` to avoid
        # collisions with opcode mnemonic columns (e.g. ``opcode`` becomes
        # ``param_opcode``); fall back to the bare name when the dataset
        # predates that convention.
        for col in ("param_opcode", "opcode"):
            if col in slice_df.columns:
                slice_df = slice_df[slice_df[col].isin(member_filter)]
                break
    # Calibration rows win when present: they are the clean straight-line
    # drivers this spec exists to consume, and mixing target-lane rows of
    # the same test (biased by wrapper work) would contaminate the driver.
    if CAMPAIGN_ROLE_COLUMN in slice_df.columns:
        role = slice_df[CAMPAIGN_ROLE_COLUMN].astype("string").str.strip().str.lower()
        calibration = slice_df[role == "calibration"]
        if not calibration.empty:
            slice_df = calibration
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


def _support_evidence(
    slice_df: pd.DataFrame,
    spec: GlueOpcodeSpec,
    eps: float,
) -> dict[str, tuple[float, float]]:
    """Return correlated observed support as ``canonical -> (corr, ratio)``."""
    agg = slice_df.drop_duplicates(subset="fixture_name")
    own = _canonical_count(agg, spec)
    if len(own) < 2 or np.std(own) == 0:
        return {}
    own_family = set(spec.members)
    opcode_columns = agg.attrs.get("opcode_columns", [])
    members_by_canonical: dict[str, list[str]] = {}
    for column in opcode_columns:
        if (
            column not in agg.columns
            or column in {"opcount", *own_family}
            or column.startswith("PRECOMPILE_")
            or not pd.api.types.is_numeric_dtype(agg[column])
        ):
            continue
        canonical = MEMBER_TO_CANONICAL.get(column, column)
        members_by_canonical.setdefault(canonical, []).append(column)
    out: dict[str, tuple[float, float]] = {}
    for canonical, columns in members_by_canonical.items():
        counts = agg[columns].astype(float).sum(axis=1).to_numpy()
        if np.std(counts) == 0:
            continue
        corr = float(np.corrcoef(counts, own)[0, 1])
        ratio = _ols_slope(own.astype(float), counts.astype(float))
        if corr >= (1 - eps) and ratio >= _RATIO_FLOOR:
            out[canonical] = (corr, ratio)
    return out


def _support_ratios(
    slice_df: pd.DataFrame,
    spec: GlueOpcodeSpec,
    eps: float,
) -> dict[str, float]:
    """Return every count-correlated observed support on a driver slice."""
    return {name: ratio for name, (_, ratio) in _support_evidence(slice_df, spec, eps).items()}


def _correlated_support(
    slice_df: pd.DataFrame,
    spec: GlueOpcodeSpec,
    eps: float,
    allowed_specs: list[GlueOpcodeSpec] | None = None,
) -> dict[str, float]:
    """Map correlated support to per-driver-count ratios.

    ``allowed_specs`` is only a partner-selection filter. Isolation callers
    omit it so unpriced and later-tier support remains visible and blocking.
    """
    support = _support_ratios(slice_df, spec, eps)
    if allowed_specs is None:
        return support
    allowed = {partner.name for partner in allowed_specs}
    return {name: ratio for name, ratio in support.items() if name in allowed}



def _plan_partners(
    slice_df: pd.DataFrame,
    config: Config,
    client: str,
    partner_names: list[str],
    fits: dict[tuple[str, str], NNLSResults],
    p_threshold: float,
    r2_threshold: float,
) -> tuple[list[_PartnerPlan], list[str], bool]:
    """Resolve detected partners into subtraction plans.

    Returns ``(plans, unsubscribed, uncertainty_conditional)``. A partner is
    *unsubscribed* when it was detected but its per-client fit is missing or
    failed either quality gate — its cost stays inside this fit's LHS, so
    the resulting coefficient is not an isolated cost. Draw pairing mirrors
    the target adjuster: shared session sets use aligned cluster draws;
    disjoint sets sample marginal draws; anything unsynchronizable marks
    the interval conditional.
    """
    target_groups = _session_groups(slice_df, config)
    target_sessions = (
        None
        if target_groups is None
        else frozenset(str(group) for group in pd.unique(target_groups))
    )
    plans: list[_PartnerPlan] = []
    unsubscribed: list[str] = []
    conditional = False
    rng = np.random.default_rng(config.modeling.random_seed)
    for partner_name in partner_names:
        partner_fit = fits.get((client, partner_name))
        if partner_fit is None:
            unsubscribed.append(partner_name)
            continue
        partner_ms = float(partner_fit.params.get(partner_name, float("nan")))
        partner_pval = float(partner_fit.pvalues.get(partner_name, float("nan")))
        partner_r2 = float(partner_fit.rsquared)
        if (
            not np.isfinite(partner_ms)
            or not np.isfinite(partner_pval)
            or not np.isfinite(partner_r2)
            or partner_pval >= p_threshold
            or partner_r2 < r2_threshold
        ):
            unsubscribed.append(partner_name)
            continue
        partner_count = _canonical_count(slice_df, SPEC_BY_NAME[partner_name])
        if partner_fit.uncertainty_conditional:
            conditional = True
            plans.append(
                _PartnerPlan(partner_name, partner_count, partner_ms, None, False)
            )
            continue
        partner_sessions = partner_fit.session_ids
        draws: np.ndarray | None = None
        aligned = False
        if target_sessions is None and partner_sessions is None:
            draws = partner_fit.bootstrap_draws(partner_name)
        elif target_sessions is None or partner_sessions is None:
            conditional = True
        elif target_sessions == partner_sessions:
            draws = partner_fit.bootstrap_draw_matrix(partner_name)
            if len(draws) != config.modeling.bootstrap_iterations:
                conditional = True
                draws = None
            else:
                aligned = True
        elif target_sessions & partner_sessions:
            conditional = True
        else:
            draws = partner_fit.bootstrap_draws(partner_name)
        if draws is not None and len(draws) == 0:
            conditional = True
            draws = None
        if draws is not None and not aligned:
            draws = draws[
                rng.integers(0, len(draws), size=config.modeling.bootstrap_iterations)
            ]
        plans.append(
            _PartnerPlan(partner_name, partner_count, partner_ms, draws, aligned)
        )
    return plans, unsubscribed, conditional


def _apply_plans(runtimes: np.ndarray, plans: list[_PartnerPlan]) -> np.ndarray:
    adjusted = runtimes.copy()
    for plan in plans:
        adjusted -= plan.ms * plan.count
    return adjusted


def _bootstrap_target_factory(
    runtimes: np.ndarray, plans: list[_PartnerPlan]
) -> Callable[[int], np.ndarray | None]:
    """Per-iteration adjusted-LHS closure for ``fit_nnls`` bootstrap."""

    def bootstrap_target(iteration: int) -> np.ndarray | None:
        out = runtimes.copy()
        for plan in plans:
            if plan.draws is None:
                return None
            draw = plan.draws[iteration]
            if not np.isfinite(draw):
                return None
            out = out - draw * plan.count
        return out

    return bootstrap_target


def _pure_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    spec: GlueOpcodeSpec,
    eps: float,
) -> tuple[NNLSResults | None, list[str]]:
    """Single-feature fit plus the isolation verdict for a pure driver."""
    slice_df = _slice_for_spec(fixtures_df, client, spec)
    if slice_df.empty:
        _log.warning(
            "glue pure-fit skipped: client=%s opcode=%s has no driver fixtures",
            client,
            spec.name,
        )
        return None, []
    counts = _canonical_count(slice_df, spec)
    if len(set(counts.tolist())) <= 1 or np.all(counts == 0):
        _log.warning(
            "glue pure-fit skipped: client=%s opcode=%s count is constant or zero",
            client,
            spec.name,
        )
        return None, []
    # Tier 1 has nothing priced to subtract: every correlated observed
    # opcode support makes this coefficient a bundle, not an isolated cost.
    contaminating = sorted(_correlated_support(slice_df, spec, eps))
    if contaminating:
        _log.warning(
            "glue pure-fit not isolated: client=%s opcode=%s driver carries "
            "count-correlated support %s; straight-line driver required",
            client,
            spec.name,
            contaminating,
        )
    design = pd.DataFrame(
        {
            spec.name: counts,
            "test_runtime_ms": slice_df["test_runtime_ms"].astype(float).to_numpy(),
        }
    )
    try:
        fit = fit_nnls(
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
        return None, contaminating
    return fit, contaminating


def _cycle_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    cycle_specs: list[GlueOpcodeSpec],
    usable_fits: dict[tuple[str, str], NNLSResults],
    p_threshold: float,
    r2_threshold: float,
    eps: float,
) -> tuple[NNLSResults | None, dict[str, list[str]], set[str]]:
    """Joint per-client fit over the union of cycle driver rows.

    Returns ``(fit, unmodeled_by_spec, fitted_names)`` where
    ``unmodeled_by_spec`` maps each member spec to the priced pure-tier
    support detected on its driver block but not subtracted (missing or
    gated partner fits), and ``fitted_names`` lists the members that
    entered the joint design. A member whose own count never varies on
    its block is dropped from the joint design (its row is emitted unfit)
    — a zero column is unidentifiable and would poison the whole solve.
    """
    pure_specs = [s for s in PRICED_GLUE_SPECS if s.tier == "pure" and s.test_name]
    blocks: list[pd.DataFrame] = []
    block_specs: list[GlueOpcodeSpec] = []
    for spec in cycle_specs:
        if spec.test_name is None:
            continue
        slc = _slice_for_spec(fixtures_df, client, spec)
        if slc.empty:
            continue
        own = _canonical_count(slc, spec)
        if len(set(own.tolist())) <= 1 or np.all(own == 0):
            _log.warning(
                "glue cycle-fit member unfit: client=%s opcode=%s count is "
                "constant or zero on its driver block; dropped from joint fit",
                client,
                spec.name,
            )
            continue
        blocks.append(slc)
        block_specs.append(spec)
    unmodeled: dict[str, list[str]] = {spec.name: [] for spec in cycle_specs}
    fitted_names = {s.name for s in block_specs}
    if not blocks:
        _log.warning("glue cycle-fit skipped: client=%s no driver rows", client)
        return None, unmodeled, set()
    combined = pd.concat(blocks, ignore_index=True)
    combined.attrs = dict(fixtures_df.attrs)

    feature_names = [spec.name for spec in block_specs]
    design_cols: dict[str, np.ndarray] = {
        spec.name: _canonical_count(combined, spec) for spec in block_specs
    }

    # Pure-tier partner subtraction per driver block.
    adjusted_blocks: list[np.ndarray] = []
    plan_blocks: list[list[_PartnerPlan]] = []
    conditional = False
    for spec, slc in zip(block_specs, blocks):
        block_runtimes = slc["test_runtime_ms"].astype(float).to_numpy()
        support = _correlated_support(slc, spec, eps)
        allowed_names = {partner.name for partner in pure_specs}
        detected = sorted(name for name in support if name in allowed_names)
        unmodeled_support = sorted(
            name
            for name in support
            if name not in allowed_names
            and name not in fitted_names
            and not (name == "STOP" and spec.name in {"CALL", "STATICCALL"})
        )
        plans, unsubscribed, block_conditional = _plan_partners(
            slc,
            config,
            client,
            detected,
            usable_fits,
            p_threshold,
            r2_threshold,
        )
        conditional = conditional or block_conditional
        block_unmodeled = sorted(set(unmodeled_support) | set(unsubscribed))
        if block_unmodeled:
            unmodeled[spec.name] = block_unmodeled
            _log.warning(
                "glue cycle-fit not isolated: client=%s opcode=%s driver support "
                "%s detected but not subtractable (partner fit missing, gated, "
                "or unpriced)",
                client,
                spec.name,
                block_unmodeled,
            )
        adjusted_blocks.append(_apply_plans(block_runtimes, plans))
        plan_blocks.append(plans)
    adjusted_runtimes = np.concatenate(adjusted_blocks)

    # Per-driver fixed effects: within-block demeaning of every feature and
    # of the adjusted LHS. A single shared intercept would leave each
    # driver's fixed setup offset (seed pushes, top-level STOP, harness
    # tails) to be absorbed by whatever between-block variation exists —
    # letting setup differences masquerade as correlated per-count costs
    # (the stack-conservation nullspace of balanced paired growers). After
    # the within transform, coefficients are identified purely by each
    # driver's own count sweep.
    block_lengths = [len(b) for b in adjusted_blocks]
    starts = np.cumsum([0] + block_lengths)

    def _demean(vector: np.ndarray) -> np.ndarray:
        out = vector.astype(float).copy()
        for lo, hi in zip(starts[:-1], starts[1:]):
            out[lo:hi] -= out[lo:hi].mean()
        return out

    design_cols_demeaned = {
        name: _demean(values) for name, values in design_cols.items()
    }
    demeaned_runtimes = _demean(adjusted_runtimes)

    design = pd.DataFrame(
        {**design_cols_demeaned, "test_runtime_ms": demeaned_runtimes}
    )
    design = design[feature_names + ["test_runtime_ms"]]
    design_matrix = np.column_stack(
        [np.ones(len(design)), design[feature_names].to_numpy(dtype=float)]
    )
    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        _log.warning(
            "glue cycle-fit skipped: client=%s joint design is rank-deficient "
            "(proportional driver features)",
            client,
        )
        return None, unmodeled, set()

    block_runtimes_raw = [
        slc["test_runtime_ms"].astype(float).to_numpy() for slc in blocks
    ]
    bootstrap_target = None
    if any(plan_blocks):
        factories = [
            _bootstrap_target_factory(runtimes, plans)
            for runtimes, plans in zip(block_runtimes_raw, plan_blocks)
        ]

        def bootstrap_target(iteration: int) -> np.ndarray | None:
            parts: list[np.ndarray] = []
            for factory in factories:
                block = factory(iteration)
                if block is None:
                    return None
                block = block - block.mean()  # same within transform
                parts.append(block)
            return np.concatenate(parts)

    try:
        fit = fit_nnls(
            design,
            features=feature_names,
            target="test_runtime_ms",
            n_bootstrap=config.modeling.bootstrap_iterations,
            random_seed=config.modeling.random_seed,
            groups=_session_groups(combined, config),
            bootstrap_target=bootstrap_target,
            uncertainty_conditional=conditional,
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose
        _log.warning("glue cycle-fit failed: client=%s exc=%s", client, exc)
        return None, unmodeled, set()
    all_unmodeled = sorted({name for names in unmodeled.values() for name in names})
    if all_unmodeled:
        # A coupled solve shares every coefficient across all blocks. One
        # contaminated block therefore invalidates every member's published
        # usability, even when the other blocks looked clean.
        unmodeled = {spec.name: all_unmodeled for spec in cycle_specs}
    return fit, unmodeled, fitted_names

def _mixed_fit(
    fixtures_df: pd.DataFrame,
    config: Config,
    client: str,
    spec: GlueOpcodeSpec,
    usable_fits: dict[tuple[str, str], NNLSResults],
    allowed_partner_tiers: frozenset[str],
    p_threshold: float,
    r2_threshold: float,
    eps: float,
) -> tuple[NNLSResults | None, list[str]]:
    """Single-feature NNLS with the LHS pre-adjusted by priced upstream partners."""
    slice_df = _slice_for_spec(fixtures_df, client, spec)
    if slice_df.empty:
        _log.warning(
            "glue mixed-fit skipped: client=%s opcode=%s has no driver fixtures",
            client,
            spec.name,
        )
        return None, []
    counts = _canonical_count(slice_df, spec)
    if len(set(counts.tolist())) <= 1 or np.all(counts == 0):
        _log.warning(
            "glue mixed-fit skipped: client=%s opcode=%s count is constant or zero",
            client,
            spec.name,
        )
        return None, []

    allowed = [
        s for s in PRICED_GLUE_SPECS if s.tier in allowed_partner_tiers and s.test_name
    ]
    support = _correlated_support(slice_df, spec, eps)
    allowed_names = {partner.name for partner in allowed}
    detected = sorted(name for name in support if name in allowed_names)
    unmodeled = sorted(
        name
        for name in support
        if name not in allowed_names
        and not (name == "STOP" and spec.name in {"CALL", "STATICCALL"})
    )
    plans, unsubscribed, conditional = _plan_partners(
        slice_df,
        config,
        client,
        detected,
        usable_fits,
        p_threshold,
        r2_threshold,
    )
    unsubscribed = sorted(set(unsubscribed) | set(unmodeled))
    if unsubscribed:
        _log.warning(
            "glue mixed-fit not isolated: client=%s opcode=%s driver support "
            "%s detected but not subtractable (partner fit missing, gated, "
            "or unpriced)",
            client,
            spec.name,
            unsubscribed,
        )
    runtimes = slice_df["test_runtime_ms"].astype(float).to_numpy()
    adjusted = _apply_plans(runtimes, plans)
    bootstrap_target = None
    if plans and not conditional:
        bootstrap_target = _bootstrap_target_factory(runtimes, plans)

    design = pd.DataFrame(
        {
            spec.name: counts,
            "test_runtime_ms": adjusted,
        }
    )
    try:
        fit = fit_nnls(
            design,
            features=[spec.name],
            target="test_runtime_ms",
            n_bootstrap=config.modeling.bootstrap_iterations,
            random_seed=config.modeling.random_seed,
            groups=_session_groups(slice_df, config),
            bootstrap_target=bootstrap_target,
            uncertainty_conditional=conditional,
        )
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: any numerical failure means this fit attempt is unfit, not a crash
        _log.warning(
            "glue mixed-fit failed: client=%s opcode=%s exc=%s",
            client,
            spec.name,
            exc,
        )
        return None, unsubscribed
    return fit, unsubscribed


def _row(
    client: str,
    spec: GlueOpcodeSpec,
    fit: NNLSResults | None,
    unmodeled: list[str] | None,
) -> dict[str, object]:
    if fit is None:
        return {
            "client_name": client,
            "glue_opcode": spec.name,
            "tier": spec.tier,
            "nobs": 0,
            "glue_runtime_ms": float("nan"),
            "p_value": float("nan"),
            "rsquared": float("nan"),
            "isolated": False if unmodeled else float("nan"),
            "unmodeled_partners": ";".join(sorted(unmodeled or [])),
            "n_sessions": float("nan"),
            "condition_number": float("nan"),
        }
    groups = fit.groups
    return {
        "client_name": client,
        "glue_opcode": spec.name,
        "tier": spec.tier,
        "nobs": int(fit.nobs),
        "glue_runtime_ms": float(fit.params[spec.name]),
        "p_value": float(fit.pvalues.get(spec.name, float("nan"))),
        "rsquared": float(fit.rsquared),
        "isolated": not unmodeled,
        "unmodeled_partners": ";".join(sorted(unmodeled or [])),
        "n_sessions": (
            float(len(np.unique(groups))) if groups is not None else float("nan")
        ),
        "condition_number": float(fit.condition_number),
    }


_DRIVER_SUPPORT_COLUMNS = [
    "glue_opcode",
    "test_name",
    "support_opcode",
    "corr",
    "ratio_per_driver_count",
]


def compute_driver_support(
    fixtures_df: pd.DataFrame,
    eps: float,
) -> pd.DataFrame:
    """Every count-correlated supporting opcode on each glue driver slice.

    One row per ``(glue spec, support opcode)`` where the support count
    co-varies with the driver's own canonical count (Pearson ≥ 1 − eps,
    OLS per-count ratio ≥ floor). This is the explicit supporting-count
    model of every calibration driver — priced or not — that wrapper-bundle
    accounting checks unpriced target-side candidates against (e.g. a
    STOP-only callee contributes ``STOP = 1 × STATICCALL`` on the
    STATICCALL driver, so a 1:1 STOP:STATICCALL target wrapper is covered
    by the STATICCALL subtraction) and that ``glue_driver_support.csv``
    publishes for review. Counts are fixture properties, so the table is
    computed once from the first client that carries the driver.
    """
    clients = sorted(fixtures_df["client_name"].unique())
    rows: list[dict[str, object]] = []
    for spec in PRICED_GLUE_SPECS:
        if spec.test_name is None:
            continue
        for client in clients:
            slice_df = _slice_for_spec(fixtures_df, client, spec)
            if slice_df.empty:
                continue
            evidence = _support_evidence(slice_df, spec, eps)
            for canonical, (corr, ratio) in sorted(evidence.items()):
                rows.append(
                    {
                        "glue_opcode": spec.name,
                        "test_name": spec.test_name,
                        "support_opcode": canonical,
                        "corr": corr,
                        "ratio_per_driver_count": ratio,
                    }
                )
            break  # counts are fixture properties; one client suffices
    df = pd.DataFrame(rows, columns=_DRIVER_SUPPORT_COLUMNS)
    if df.empty:
        return df
    return (
        df.sort_values(["glue_opcode", "support_opcode"], kind="mergesort")
        .drop_duplicates()
        .reset_index(drop=True)
    )


_MIXED_A_PARTNER_TIERS: frozenset[str] = frozenset({"pure", "cycle"})
_MIXED_B_PARTNER_TIERS: frozenset[str] = frozenset({"pure", "cycle", "mixed_a"})

def estimate_glue(config: Config, fixtures_df: pd.DataFrame) -> GlueEstimateOutput:
    """Fit one NNLS per (client, canonical glue name) in four ordered passes.

    Pure-glue specs get a single-feature regression each; cycle-glue specs
    share a joint per-client fit (with pure-tier partner subtraction);
    mixed-tier specs get single-feature fits with LHS pre-adjusted by
    upstream partners. Specs whose ``test_name is None`` are skipped (no
    row emitted). Raises ``ConfigError`` if any required driver test is
    absent from ``fixtures_df``.
    """
    validate_inputs(fixtures_df)
    glue_by_test_df = compute_glue_opcodes_by_test(
        fixtures_df,
        config.resolved_models,
        config.glue_adjustment.ratio_corr_eps,
        config.campaign.session_column,
    )
    detection_coverage_df = compute_detection_coverage(
        fixtures_df,
        config.resolved_models,
        config.glue_adjustment.ratio_corr_eps,
        config.campaign.session_column,
    )
    driver_support_df = compute_driver_support(
        fixtures_df, config.glue_adjustment.ratio_corr_eps
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
    eps = config.glue_adjustment.ratio_corr_eps
    clients = sorted(fixtures_df["client_name"].unique())
    usable_fits: dict[tuple[str, str], NNLSResults] = {}
    for client in clients:
        # Tier 1 — pure
        for spec in active_pure:
            fit, contaminating = _pure_fit(fixtures_df, config, client, spec, eps)
            rows.append(_row(client, spec, fit, contaminating))
            if fit is not None:
                fits[(client, spec.name)] = fit
                if not contaminating:
                    usable_fits[(client, spec.name)] = fit

        # Tier 2 — cycle (joint)
        cycle_fit, unmodeled, cycle_fitted = _cycle_fit(
            fixtures_df,
            config,
            client,
            active_cycle,
            usable_fits,
            p_threshold,
            r2_threshold,
            eps,
        )
        for spec in active_cycle:
            if cycle_fit is None or spec.name not in cycle_fitted:
                rows.append(_row(client, spec, None, unmodeled.get(spec.name)))
                continue
            rows.append(_row(client, spec, cycle_fit, unmodeled.get(spec.name)))
            fits[(client, spec.name)] = cycle_fit
            if not unmodeled.get(spec.name):
                usable_fits[(client, spec.name)] = cycle_fit

        # Tier 3a — mixed, partners drawn from pure ∪ cycle
        for spec in active_mixed_a:
            fit, unsubscribed = _mixed_fit(
                fixtures_df,
                config,
                client,
                spec,
                usable_fits,
                _MIXED_A_PARTNER_TIERS,
                p_threshold,
                r2_threshold,
                eps,
            )
            rows.append(_row(client, spec, fit, unsubscribed))
            if fit is not None:
                fits[(client, spec.name)] = fit
                if not unsubscribed:
                    usable_fits[(client, spec.name)] = fit
        # Tier 3b — mixed, partners drawn from pure ∪ cycle ∪ mixed_a
        for spec in active_mixed_b:
            fit, unsubscribed = _mixed_fit(
                fixtures_df,
                config,
                client,
                spec,
                usable_fits,
                _MIXED_B_PARTNER_TIERS,
                p_threshold,
                r2_threshold,
                eps,
            )
            rows.append(_row(client, spec, fit, unsubscribed))
            if fit is not None:
                fits[(client, spec.name)] = fit
                if not unsubscribed:
                    usable_fits[(client, spec.name)] = fit
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
        detection_coverage_df=detection_coverage_df,
        driver_support_df=driver_support_df,
    )
