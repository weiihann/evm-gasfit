"""Priced-glue opcode specs and runtime-input validation.

Each spec binds a canonical glue name (``ISZERO``, ``DUP``, ``ADD``,
``KECCAK256``, ...) to the driver fixture that produces its runtime
estimate. Family opcodes (``DUP``, ``SWAP``, ``PUSH``) are fit jointly
over all members and share a single coefficient: ``DUP1``..``DUP16`` all
map to the canonical ``DUP`` estimate. Mixed opcodes appear both as
targets and as glues; their fits run after the pure/cycle tiers and
subtract the partner contributions on the LHS. ``POP`` and ``STOP`` are
tracked as priced glue but have no driver fixture yet, so they are
skipped at fit time with a warning rather than an error.

Fit ordering follows the static tier sequence:

1. ``pure`` — independent single-feature NNLS per ``(client, spec)``.
2. ``cycle`` — one joint NNLS per client with one feature per cycle spec.
3. ``mixed_a`` — single-feature NNLS per ``(client, spec)`` with the LHS
   pre-adjusted using partners drawn from ``pure ∪ cycle``.
4. ``mixed_b`` — same as ``mixed_a`` but partners may also come from
   ``mixed_a``.

The tier sequence is the dependency declaration. Within-test partner
selection (which specific pure/cycle/mixed-a ops show up around a given
target) comes from the existing detector ``compute_glue_opcodes_by_test``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from evm_gasfit.errors import ConfigError

_log = logging.getLogger("evm_gasfit.glue")


@dataclass(frozen=True)
class GlueOpcodeSpec:
    """One priced-glue family.

    Attributes:
        name: Canonical glue name emitted on every output frame
            (``glue_results_df``, ``glue_opcodes_by_test_df``). For families
            this is the family prefix (``DUP``, ``SWAP``, ``PUSH``); for
            singletons it is the opcode mnemonic.
        tier: Drives the fit routine. ``"pure"`` opcodes get a single-feature
            NNLS each; ``"cycle"`` opcodes share a joint per-client fit;
            ``"mixed_a"`` and ``"mixed_b"`` opcodes get single-feature fits
            with the LHS pre-adjusted using priced partners from earlier
            tiers (mixed_a allows pure+cycle; mixed_b allows pure+cycle+
            mixed_a).
        test_name: ``test_name`` value to filter driver fixtures by. ``None``
            when no driver fixture exists yet — the spec is then skipped at
            fit time and ``validate_inputs`` does not require it.
        members: Opcode mnemonics in the fixture column space that belong to
            this family. ``("STATICCALL",)`` for singletons; the full member
            tuple for families. Row-wise summed into a synthetic
            ``name``-keyed column when building the design matrix.
        test_opcode_filter: Optional disambiguator when ``test_name`` is
            shared across specs (``MLOAD`` vs the other ops in
            ``test_memory_access``; ``PUSH0`` vs ``PUSH1..PUSH32`` inside
            ``test_push``; every individual mixed opcode living inside
            ``test_arithmetic``/``test_bitwise``/``test_comparison``/
            ``test_memory_access``). ``None`` when ``members`` alone is
            enough.
        required: ``False`` for specs without a driver fixture (STOP, which
            terminates execution and can never appear in a per-count sweep);
            ``validate_inputs`` will not raise on their absence.
    """

    name: str
    tier: Literal["pure", "cycle", "mixed_a", "mixed_b"]
    test_name: str | None
    members: tuple[str, ...]
    test_opcode_filter: str | None = None
    required: bool = True


_DUP_MEMBERS: tuple[str, ...] = tuple(f"DUP{i}" for i in range(1, 17))
_SWAP_MEMBERS: tuple[str, ...] = tuple(f"SWAP{i}" for i in range(1, 17))
_PUSH_MEMBERS: tuple[str, ...] = tuple(f"PUSH{i}" for i in range(1, 33))


# Driver test names follow the calibration exporter's straight-line module
# (tests/benchmark/compute/calibration/test_glue.py). Pure/cycle drivers use
# ``*_straight`` names where the target corpus already ships identically
# named-but-biased workload tests: gasfit slices drivers by ``test_name``
PRICED_GLUE_SPECS: tuple[GlueOpcodeSpec, ...] = (
    # Pure glue — clean straight-line single-opcode sweeps; single-feature
    # fits are identified because the exporter guarantees no count-correlated
    # supporting opcodes in these drivers. POP uses a dedicated constant-seed
    # driver (fixed 512×PUSH0 setup, POP×N with N ≤ 512, no stack cleanup,
    # witness tail): the constant seed rides the intercept and the POP slope
    # is identified by its own sweep, so the pure fit anchors POP for every
    # paired grower driver that subtracts it as a partner.
    GlueOpcodeSpec("ISZERO", "pure", "test_iszero_straight", ("ISZERO",)),
    GlueOpcodeSpec("JUMPDEST", "pure", "test_jumpdests_straight", ("JUMPDEST",)),
    GlueOpcodeSpec(
        "POP", "pure", "test_pop", ("POP",), test_opcode_filter="POP"
    ),
    GlueOpcodeSpec("STOP", "pure", None, ("STOP",), required=False),
    GlueOpcodeSpec("SWAP", "pure", "test_swap_straight", _SWAP_MEMBERS),
    # Cycle glue — jointly fit over the union of their driver rows, with
    # pure-tier partners (notably POP) subtracted from each driver block's
    # LHS first and per-driver fixed effects removing setup-level offsets:
    # coefficients are identified by within-driver count variation only, so
    # differing fixed setups across drivers can never masquerade as
    # correlated per-count costs. CALL lives here because its warm-call
    # driver carries its own sweep; its DUP/POP support is handled by the
    # joint design and pure anchoring respectively.
    GlueOpcodeSpec("CALL", "cycle", "test_call_warm", ("CALL",), test_opcode_filter="CALL"),
    GlueOpcodeSpec(
        "CALLDATASIZE", "cycle", "test_calldatasize", ("CALLDATASIZE",)
    ),
    GlueOpcodeSpec("DUP", "cycle", "test_dup_straight", _DUP_MEMBERS),
    GlueOpcodeSpec("GAS", "cycle", "test_gas_op_straight", ("GAS",)),
    GlueOpcodeSpec(
        "MLOAD", "cycle", "test_memory_access", ("MLOAD",), test_opcode_filter="MLOAD"
    ),
    GlueOpcodeSpec("PUSH", "cycle", "test_push_straight", _PUSH_MEMBERS),
    GlueOpcodeSpec(
        "PUSH0", "cycle", "test_push_straight", ("PUSH0",), test_opcode_filter="PUSH0"
    ),
    GlueOpcodeSpec(
        "STATICCALL",
        "cycle",
        "test_ext_account_query_warm",
        ("STATICCALL",),
        test_opcode_filter="STATICCALL",
    ),
    # Mixed A — partners drawn from pure + cycle
    GlueOpcodeSpec(
        "ADD", "mixed_a", "test_arithmetic", ("ADD",), test_opcode_filter="ADD"
    ),
    GlueOpcodeSpec(
        "AND", "mixed_a", "test_bitwise", ("AND",), test_opcode_filter="AND"
    ),
    GlueOpcodeSpec(
        "CALLDATACOPY",
        "mixed_a",
        "test_calldatacopy_from_origin",
        ("CALLDATACOPY",),
    ),
    GlueOpcodeSpec("CALLDATALOAD", "mixed_a", "test_calldataload", ("CALLDATALOAD",)),
    GlueOpcodeSpec(
        "DIV", "mixed_a", "test_arithmetic", ("DIV",), test_opcode_filter="DIV"
    ),
    GlueOpcodeSpec(
        "EXP", "mixed_a", "test_arithmetic", ("EXP",), test_opcode_filter="EXP"
    ),
    GlueOpcodeSpec(
        "GT", "mixed_a", "test_comparison", ("GT",), test_opcode_filter="GT"
    ),
    GlueOpcodeSpec("JUMPI", "mixed_a", "test_jumpi_fallthrough", ("JUMPI",)),
    GlueOpcodeSpec(
        "LT", "mixed_a", "test_comparison", ("LT",), test_opcode_filter="LT"
    ),
    GlueOpcodeSpec(
        "MSTORE",
        "mixed_a",
        "test_memory_access",
        ("MSTORE",),
        test_opcode_filter="MSTORE",
    ),
    GlueOpcodeSpec(
        "MSTORE8",
        "mixed_a",
        "test_memory_access",
        ("MSTORE8",),
        test_opcode_filter="MSTORE8",
    ),
    GlueOpcodeSpec(
        "MUL", "mixed_a", "test_arithmetic", ("MUL",), test_opcode_filter="MUL"
    ),
    GlueOpcodeSpec("PC", "mixed_a", "test_pc_op", ("PC",)),
    GlueOpcodeSpec(
        "RETURNDATASIZE",
        "mixed_a",
        "test_returndatasize_nonzero",
        ("RETURNDATASIZE",),
    ),
    GlueOpcodeSpec("SELFBALANCE", "mixed_a", "test_selfbalance", ("SELFBALANCE",)),
    GlueOpcodeSpec(
        "SUB", "mixed_a", "test_arithmetic", ("SUB",), test_opcode_filter="SUB"
    ),
    # Mixed B — partners also drawn from mixed_a (JUMP → ADD, PC; KECCAK256 → MSTORE)
    GlueOpcodeSpec("JUMP", "mixed_b", "test_jump_benchmark", ("JUMP",)),
    GlueOpcodeSpec(
        "KECCAK256", "mixed_b", "test_keccak_diff_mem_msg_sizes", ("KECCAK256",)
    ),
)


# Canonical-name views, kept for callers that only care about names.
PURE_GLUE_OPCODES: list[str] = [s.name for s in PRICED_GLUE_SPECS if s.tier == "pure"]
CYCLE_GLUE_OPCODES: list[str] = [s.name for s in PRICED_GLUE_SPECS if s.tier == "cycle"]
MIXED_A_GLUE_OPCODES: list[str] = [
    s.name for s in PRICED_GLUE_SPECS if s.tier == "mixed_a"
]
MIXED_B_GLUE_OPCODES: list[str] = [
    s.name for s in PRICED_GLUE_SPECS if s.tier == "mixed_b"
]
PRICED_GLUE_OPCODES: list[str] = (
    PURE_GLUE_OPCODES + CYCLE_GLUE_OPCODES + MIXED_A_GLUE_OPCODES + MIXED_B_GLUE_OPCODES
)


def _build_member_to_canonical() -> dict[str, str]:
    mapping: dict[str, str] = {}
    seen_names: set[str] = set()
    for spec in PRICED_GLUE_SPECS:
        if spec.name in seen_names:
            raise RuntimeError(
                f"duplicate canonical glue name {spec.name!r} — spec table "
                "must list each name exactly once"
            )
        seen_names.add(spec.name)
        for member in spec.members:
            existing = mapping.get(member)
            if existing is not None and existing != spec.name:
                raise RuntimeError(
                    f"glue member {member!r} mapped to both {existing!r} and "
                    f"{spec.name!r} — spec table is inconsistent"
                )
            mapping[member] = spec.name
    return mapping


# Member opcode mnemonic → canonical family name (e.g. "DUP3" → "DUP").
MEMBER_TO_CANONICAL: dict[str, str] = _build_member_to_canonical()

# Canonical family name → member mnemonics (e.g. "DUP" → DUP1..DUP16).
# Singletons map to themselves. Used for group self-exclusion: a target that
# is itself a family member never has its own family subtracted as glue.
CANONICAL_TO_MEMBERS: dict[str, tuple[str, ...]] = {
    spec.name: spec.members for spec in PRICED_GLUE_SPECS
}

# Canonical-name → spec lookup, used by the mixed-tier fit to find partner specs.
SPEC_BY_NAME: dict[str, GlueOpcodeSpec] = {s.name: s for s in PRICED_GLUE_SPECS}

# Per-count cost of these glue opcodes depends on an input size, so a
# coefficient measured on a driver only transfers to a target group with a
# provably matching size shape. Values are the ``param_``-prefixed fixture
# columns carrying the size. A partner is shape-verified only when driver
# and target declare the same single size value; anything else (differing
# values, sweeping sizes, or one-sided declaration) marks the subtraction
# unverified and blocks the adjusted recommendation.
SHAPE_PARAM_BY_SPEC: dict[str, tuple[str, ...]] = {
    "CALLDATACOPY": ("param_calldata_size", "param_mem_size"),
    "CALLDATALOAD": ("param_calldata_size",),
    "MLOAD": ("param_mem_size",),
    "MSTORE": ("param_mem_size",),
    "MSTORE8": ("param_mem_size",),
}


def validate_inputs(fixtures_df: pd.DataFrame) -> None:
    """Validate that every required driver fixture is present.

    A pure or cycle spec marked ``required=True`` must appear in
    ``fixtures_df["test_name"]``; missing ones raise ``ConfigError``.
    Optional specs whose driver is absent are skipped silently — the fit
    loop will not produce a row for them.

    Mixed-tier specs (``mixed_a``/``mixed_b``) are never required: their
    canonical names overlap with modelspec targets, so the driver
    fixtures come from whichever model tests the user configured. A
    missing mixed-tier driver yields no row and no warning — it's
    treated as a no-op rather than a misconfiguration.

    Args:
        fixtures_df: Shared frame built by ``build_fixtures_df``.
    """
    present = set(fixtures_df["test_name"].unique())
    missing_required = sorted(
        {
            spec.test_name
            for spec in PRICED_GLUE_SPECS
            if spec.required
            and spec.tier in ("pure", "cycle")
            and spec.test_name is not None
            and spec.test_name not in present
        }
    )
    if missing_required:
        raise ConfigError(
            "glue adjustment enabled but the runtime inputs are missing driver "
            f"fixtures for test_name(s): {missing_required!r}"
        )

    missing_optional = sorted(
        {
            spec.test_name
            for spec in PRICED_GLUE_SPECS
            if not spec.required
            and spec.tier in ("pure", "cycle")
            and spec.test_name is not None
            and spec.test_name not in present
        }
    )
    if missing_optional:
        _log.warning(
            "glue adjustment: optional driver fixtures missing for test_name(s): "
            "%r; corresponding opcodes will not be priced as glue",
            missing_optional,
        )
