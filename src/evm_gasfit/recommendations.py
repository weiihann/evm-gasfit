"""Glue-enabled 600M Osaka compute pricing: config generation and
evidence-linked minimal-change recommendations.

This module turns a freshly exported Osaka compute ``workload.json``
(protocol v2) and a completed evm-gasfit analysis of it into pricing
recommendations. It is deliberately importable with the standard library
only; evm-gasfit itself is imported lazily where its machinery is used.

CLI
---
::

    python -m evm_gasfit.recommendations create-config \\
        --workload WORKLOAD.json --client evm2 --out analysis-gasfit.yaml \\
        [--anchor-rate 600000000] [--min-sessions 4]

    python -m evm_gasfit.recommendations build \\
        --workload WORKLOAD.json --analysis ANALYSIS_DIR --out OUT_DIR \\
        [--diagnostics SAMPLES.jsonl] [--anchor-rate 600000000]

``create-config`` emits a comprehensive per-variant, glue-enabled analysis
configuration: one model per ready ``campaign_role='target'`` variant
(calibration-lane cases are never modeled and never priced), glue adjustment
enabled, the frozen 500M-era strict qualification gates, bootstrap 1000,
seed 20260922, and a named 600M/0%-margin pricing scenario. A sidecar
``<out>.variants.json`` records the variant -> parameter/group/current-charge
mapping used later by ``build`` and by campaign audits.

``build`` joins the workload with the completed analysis outputs
(``results.csv``, ``qualification.csv``, ``new_gas_all_params.csv``, optional
glue CSVs, ``analysis_status.json``) and, when ``--diagnostics`` is given,
the campaign's terminal diagnostic records, and writes
``recommendations.json`` plus ``recommendations.csv`` covering every selected
variant (including unsupported ones) and every pricing group.

Policy (frozen in the experiment's ``policy.json``)
---------------------------------------------------
- anchor 600,000,000 gas/s (0.6 gas/ns), 0% extra margin;
- 95% adjusted upper bound as the conservative statistic;
- no decreases; an increase candidate requires a qualified lower-bound evidence of >= 2x the current charge; sensitivity reports vary the discrepancy threshold (1.5x/2x/3x) at the fixed 600M anchor;
- candidates are rounded upward to two significant digits;
- existing formula shapes are preserved where defensible (uniform scale);
  incomplete glue/fit coverage never silently becomes a deployable
  worst-case recommendation -- an evidence-supported candidate is still
  reported, clearly labeled, together with the exact missing coverage.

Current-charge ground truth
---------------------------
Every current gas value or input-dependent formula below is transcribed
from the pinned execution-specs Osaka sources (cited per constant); no
calldata-size proxies or guesses are used. A cross-check test asserts
equality with ``evm_gasfit.defaults.get_gas_costs("osaka")`` for every
shared field when that import is available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger("evm_gasfit.recommendations")
DEFAULT_ANCHOR_RATE = 600_000_000.0  # 0.6 gas/ns; experiment policy anchor.
INCREASE_THRESHOLD = 2.0  # qualified lower-bound / current charge.
SENSITIVITY_THRESHOLDS = (1.5, 2.0, 3.0)
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 20260922
MIN_SESSIONS_DEFAULT = 4  # handoff section 7 gate; the campaign schedule owns 8.
QUALIFICATION_GATES = {
    "confidence_level": 0.95,
    "max_condition_number": 1e8,
    "max_residual_curvature_r2": 0.1,
    "max_relative_uncertainty": 0.5,
    "max_holdout_error": 0.25,
    "min_sessions": MIN_SESSIONS_DEFAULT,
    "enforce_fit_quality": True,
    "block_unqualified": True,
}
PRICING_SCENARIO_NAME = "osaka-600m"
ALLOWED_FAMILIES = frozenset(
    {
        "arithmetic",
        "bitwise",
        "comparison",
        "stack",
        "control_flow",
        "keccak",
        "precompile",
    }
)
# Terminal work-size token of an exported case id: fixed-count exports end in
# ``-opcount_<n>K``, gas-budget (EIP-7904 layout) exports in
# ``-benchmark-gas-value_<n>M``. Stripping it yields the variant identity.
COUNT_TOKEN_RE = re.compile(r"-(?:opcount|benchmark-gas-value)_[^\]]+")
# evm2's worker (crates/cli/Cargo.toml:35) enables alloy-primitives'
# ``keccak-cache-global``; alloy v1.7.3's process-global keccak cache accepts
# inputs of at most MAX_INPUT_LEN = 128 - 32 - 1 - 8 = 87 bytes
# (crates/primitives/src/utils/keccak_cache.rs). Inputs of 1..87 bytes hashed
# repeatedly within one worker process can be served warm from that cache --
# the per-session warmup reset does not clear it -- so those slopes are NOT
# worst-case evidence. Empty input and inputs > 87 bytes bypass the cache.
# Sources: alloy-rs/core v1.7.3 keccak_cache.rs; evm2
# crates/evm2/src/interpreter/instructions/crypto.rs.
KECCAK_CACHE_MAX_INPUT_BYTES = 87

# ---------------------------------------------------------------------------
# Osaka current-charge tables.
#
# Transcribed from src/ethereum/forks/osaka/vm/gas.py (GasCosts) at the
# pinned execution-specs revision. Keep in sync with that file; the
# cross-check test compares against evm_gasfit.defaults when importable.
# ---------------------------------------------------------------------------

_TIER = {"BASE": 2, "VERY_LOW": 3, "LOW": 5, "MID": 8, "HIGH": 10}

OSAKA_GAS_COSTS: dict[str, int] = {
    # Static opcodes (vm/gas.py "Static Opcodes").
    "OPCODE_ADD": _TIER["VERY_LOW"],
    "OPCODE_SUB": _TIER["VERY_LOW"],
    "OPCODE_MUL": _TIER["LOW"],
    "OPCODE_DIV": _TIER["LOW"],
    "OPCODE_SDIV": _TIER["LOW"],
    "OPCODE_SIGNEXTEND": _TIER["LOW"],
    "OPCODE_MOD": _TIER["LOW"],
    "OPCODE_SMOD": _TIER["LOW"],
    "OPCODE_ADDMOD": _TIER["MID"],
    "OPCODE_MULMOD": _TIER["MID"],
    "OPCODE_LT": _TIER["VERY_LOW"],
    "OPCODE_GT": _TIER["VERY_LOW"],
    "OPCODE_SLT": _TIER["VERY_LOW"],
    "OPCODE_SGT": _TIER["VERY_LOW"],
    "OPCODE_EQ": _TIER["VERY_LOW"],
    "OPCODE_ISZERO": _TIER["VERY_LOW"],
    "OPCODE_AND": _TIER["VERY_LOW"],
    "OPCODE_OR": _TIER["VERY_LOW"],
    "OPCODE_XOR": _TIER["VERY_LOW"],
    "OPCODE_NOT": _TIER["VERY_LOW"],
    "OPCODE_BYTE": _TIER["VERY_LOW"],
    "OPCODE_SHL": _TIER["VERY_LOW"],
    "OPCODE_SHR": _TIER["VERY_LOW"],
    "OPCODE_SAR": _TIER["VERY_LOW"],
    "OPCODE_CLZ": _TIER["LOW"],
    "OPCODE_JUMP": _TIER["MID"],
    "OPCODE_JUMPI": _TIER["HIGH"],
    "OPCODE_JUMPDEST": 1,
    "OPCODE_PC": _TIER["BASE"],
    "OPCODE_GAS": _TIER["BASE"],
    "OPCODE_PUSH": _TIER["VERY_LOW"],  # PUSH1..PUSH32 (PUSH0 below)
    "OPCODE_PUSH0": _TIER["BASE"],
    "OPCODE_DUP": _TIER["VERY_LOW"],
    "OPCODE_SWAP": _TIER["VERY_LOW"],
    # Dynamic opcodes used here.
    "OPCODE_EXP_BASE": 10,  # vm/instructions/arithmetic.py exp()
    "OPCODE_EXP_PER_BYTE": 50,
    "OPCODE_KECCAK256_BASE": 30,  # vm/instructions/keccak.py keccak256()
    "OPCODE_KECCAK256_PER_WORD": 6,
    # Precompiles (vm/gas.py "Precompiles").
    "PRECOMPILE_ECRECOVER": 3000,
    "PRECOMPILE_P256VERIFY": 6900,
    "PRECOMPILE_SHA256_BASE": 60,  # precompiled_contracts/sha256.py
    "PRECOMPILE_SHA256_PER_WORD": 12,
    "PRECOMPILE_RIPEMD160_BASE": 600,  # precompiled_contracts/ripemd160.py
    "PRECOMPILE_RIPEMD160_PER_WORD": 120,
    "PRECOMPILE_IDENTITY_BASE": 15,  # precompiled_contracts/identity.py
    "PRECOMPILE_IDENTITY_PER_WORD": 3,
    "PRECOMPILE_BLAKE2F_PER_ROUND": 1,  # precompiled_contracts/blake2f.py
    "PRECOMPILE_POINT_EVALUATION": 50000,
    "PRECOMPILE_BLS_G1ADD": 375,
    "PRECOMPILE_BLS_G1MUL": 12000,  # MSM per-point base (EIP-2537)
    "PRECOMPILE_BLS_G1MAP": 5500,
    "PRECOMPILE_BLS_G2ADD": 600,
    "PRECOMPILE_BLS_G2MUL": 22500,
    "PRECOMPILE_BLS_G2MAP": 23800,
    "PRECOMPILE_ECADD": 150,
    "PRECOMPILE_ECMUL": 6000,
    "PRECOMPILE_ECPAIRING_BASE": 45000,  # precompiled_contracts/alt_bn128.py
    "PRECOMPILE_ECPAIRING_PER_POINT": 34000,
}

# BLS12-381 pairing charges with inline constants, not GasCosts fields:
# src/ethereum/forks/osaka/vm/precompiled_contracts/bls12_381/
# bls12_381_pairing.py:45 ``gas_cost = Uint(32600 * k + 37700)``.
BLS12_PAIRING_PER_POINT = 32600
BLS12_PAIRING_BASE = 37700

# MODEXP (precompiled_contracts/modexp.py): gas = max(500, complexity *
# iterations); operand lengths capped at 1024 bytes by EIP-7823 (the length
# check raises before charge_gas, so oversized input is a 0-charge halt).
MODEXP_MIN_GAS = 500
MODEXP_MAX_OPERAND_BYTES = 1024

# Osaka precompile addresses (precompiled_contracts/__init__.py:38-55).
PRECOMPILE_ADDRESSES: dict[str, str] = {
    "ECRECOVER": "0x0000000000000000000000000000000000000001",
    "SHA2-256": "0x0000000000000000000000000000000000000002",
    "RIPEMD-160": "0x0000000000000000000000000000000000000003",
    "IDENTITY": "0x0000000000000000000000000000000000000004",
    "MODEXP": "0x0000000000000000000000000000000000000005",
    "BN128_ADD": "0x0000000000000000000000000000000000000006",
    "BN128_MUL": "0x0000000000000000000000000000000000000007",
    "BN128_PAIRING": "0x0000000000000000000000000000000000000008",
    "BLAKE2F": "0x0000000000000000000000000000000000000009",
    "POINT_EVALUATION": "0x000000000000000000000000000000000000000a",
    "BLS12_G1ADD": "0x000000000000000000000000000000000000000b",
    "BLS12_G1MSM": "0x000000000000000000000000000000000000000c",
    "BLS12_G2ADD": "0x000000000000000000000000000000000000000d",
    "BLS12_G2MSM": "0x000000000000000000000000000000000000000e",
    "BLS12_PAIRING": "0x000000000000000000000000000000000000000f",
    "BLS12_MAP_FP_TO_G1": "0x0000000000000000000000000000000000000010",
    "BLS12_MAP_FP2_TO_G2": "0x0000000000000000000000000000000000000011",
    "P256VERIFY": "0x0000000000000000000000000000000000000100",
}

# BLS12-381 MSM discount tables (per-mille), transcribed from
# src/ethereum/forks/osaka/vm/precompiled_contracts/bls12_381/__init__.py
# (G1_K_DISCOUNT, G2_K_DISCOUNT, G1_MAX_DISCOUNT=519, G2_MAX_DISCOUNT=524,
# MULTIPLIER=1000). gas = k * MUL * discount(k) // 1000.
G1_K_DISCOUNT: tuple[int, ...] = (
    1000,
    949,
    848,
    797,
    764,
    750,
    738,
    728,
    719,
    712,
    705,
    698,
    692,
    687,
    682,
    677,
    673,
    669,
    665,
    661,
    658,
    654,
    651,
    648,
    645,
    642,
    640,
    637,
    635,
    632,
    630,
    627,
    625,
    623,
    621,
    619,
    617,
    615,
    613,
    611,
    609,
    608,
    606,
    604,
    603,
    601,
    599,
    598,
    596,
    595,
    593,
    592,
    591,
    589,
    588,
    586,
    585,
    584,
    582,
    581,
    580,
    579,
    577,
    576,
    575,
    574,
    573,
    572,
    570,
    569,
    568,
    567,
    566,
    565,
    564,
    563,
    562,
    561,
    560,
    559,
    558,
    557,
    556,
    555,
    554,
    553,
    552,
    551,
    550,
    549,
    548,
    547,
    547,
    546,
    545,
    544,
    543,
    542,
    541,
    540,
    540,
    539,
    538,
    537,
    536,
    536,
    535,
    534,
    533,
    532,
    532,
    531,
    530,
    529,
    528,
    528,
    527,
    526,
    525,
    525,
    524,
    523,
    522,
    522,
    521,
    520,
    520,
    519,
)
G2_K_DISCOUNT: tuple[int, ...] = (
    1000,
    1000,
    923,
    884,
    855,
    832,
    812,
    796,
    782,
    770,
    759,
    749,
    740,
    732,
    724,
    717,
    711,
    704,
    699,
    693,
    688,
    683,
    679,
    674,
    670,
    666,
    663,
    659,
    655,
    652,
    649,
    646,
    643,
    640,
    637,
    634,
    632,
    629,
    627,
    624,
    622,
    620,
    618,
    615,
    613,
    611,
    609,
    607,
    606,
    604,
    602,
    600,
    598,
    597,
    595,
    593,
    592,
    590,
    589,
    587,
    586,
    584,
    583,
    582,
    580,
    579,
    578,
    576,
    575,
    574,
    573,
    571,
    570,
    569,
    568,
    567,
    566,
    565,
    563,
    562,
    561,
    560,
    559,
    558,
    557,
    556,
    555,
    554,
    553,
    552,
    552,
    551,
    550,
    549,
    548,
    547,
    546,
    545,
    545,
    544,
    543,
    542,
    541,
    541,
    540,
    539,
    538,
    537,
    537,
    536,
    535,
    535,
    534,
    533,
    532,
    532,
    531,
    530,
    530,
    529,
    528,
    528,
    527,
    526,
    526,
    525,
    524,
    524,
)
G1_MAX_DISCOUNT = 519
G2_MAX_DISCOUNT = 524
MSM_DISCOUNT_MULTIPLIER = 1000

# Groups whose current charge is a constant.
GROUP_KIND_FIXED = "fixed"
# base + per_unit * units (KECCAK per word, SHA256 per word, EXP per byte,
# pairing per point, BLAKE2F per round).
GROUP_KIND_LINEAR = "linear"
# k * MUL * discount(k) // 1000 (BLS12 G1/G2 MSM, EIP-2537).
GROUP_KIND_MSM = "msm"
# max(500, complexity(base,mod) * iterations(exp)) (EIP-7883 MODEXP).
GROUP_KIND_MODEXP = "modexp"
# MODEXP input-size rejection path: halts before charging (0 gas).
GROUP_KIND_REJECTION = "rejection"
GROUP_KIND_UNSUPPORTED = "unsupported"

_FIXED_OPCODE_GROUPS: dict[str, tuple[str, str]] = {
    # target_operation -> (pricing_group, cost field)
    "ADD": ("OPCODE_ADD", "OPCODE_ADD"),
    "SUB": ("OPCODE_SUB", "OPCODE_SUB"),
    "MUL": ("OPCODE_MUL", "OPCODE_MUL"),
    "DIV": ("OPCODE_DIV", "OPCODE_DIV"),
    "SDIV": ("OPCODE_SDIV", "OPCODE_SDIV"),
    "SIGNEXTEND": ("OPCODE_SIGNEXTEND", "OPCODE_SIGNEXTEND"),
    "MOD": ("OPCODE_MOD", "OPCODE_MOD"),
    "SMOD": ("OPCODE_SMOD", "OPCODE_SMOD"),
    "ADDMOD": ("OPCODE_ADDMOD", "OPCODE_ADDMOD"),
    "MULMOD": ("OPCODE_MULMOD", "OPCODE_MULMOD"),
    "AND": ("OPCODE_AND", "OPCODE_AND"),
    "OR": ("OPCODE_OR", "OPCODE_OR"),
    "XOR": ("OPCODE_XOR", "OPCODE_XOR"),
    "BYTE": ("OPCODE_BYTE", "OPCODE_BYTE"),
    "SHL": ("OPCODE_SHL", "OPCODE_SHL"),
    "SHR": ("OPCODE_SHR", "OPCODE_SHR"),
    "SAR": ("OPCODE_SAR", "OPCODE_SAR"),
    "NOT": ("OPCODE_NOT", "OPCODE_NOT"),
    "CLZ": ("OPCODE_CLZ", "OPCODE_CLZ"),
    "LT": ("OPCODE_LT", "OPCODE_LT"),
    "GT": ("OPCODE_GT", "OPCODE_GT"),
    "SLT": ("OPCODE_SLT", "OPCODE_SLT"),
    "SGT": ("OPCODE_SGT", "OPCODE_SGT"),
    "EQ": ("OPCODE_EQ", "OPCODE_EQ"),
    "ISZERO": ("OPCODE_ISZERO", "OPCODE_ISZERO"),
    "JUMP": ("OPCODE_JUMP", "OPCODE_JUMP"),
    "JUMPI": ("OPCODE_JUMPI", "OPCODE_JUMPI"),
    "JUMPDEST": ("OPCODE_JUMPDEST", "OPCODE_JUMPDEST"),
    "PC": ("OPCODE_PC", "OPCODE_PC"),
    "GAS": ("OPCODE_GAS", "OPCODE_GAS"),
    "PUSH0": ("OPCODE_PUSH0", "OPCODE_PUSH0"),
    "ECRECOVER": ("PRECOMPILE_ECRECOVER", "PRECOMPILE_ECRECOVER"),
    "BN128_ADD": ("PRECOMPILE_ECADD", "PRECOMPILE_ECADD"),
    "BN128_MUL": ("PRECOMPILE_ECMUL", "PRECOMPILE_ECMUL"),
    "P256VERIFY": ("PRECOMPILE_P256VERIFY", "PRECOMPILE_P256VERIFY"),
    "POINT_EVALUATION": (
        "PRECOMPILE_POINT_EVALUATION",
        "PRECOMPILE_POINT_EVALUATION",
    ),
    "BLS12_G1ADD": ("PRECOMPILE_BLS_G1ADD", "PRECOMPILE_BLS_G1ADD"),
    "BLS12_G2ADD": ("PRECOMPILE_BLS_G2ADD", "PRECOMPILE_BLS_G2ADD"),
    "BLS12_MAP_FP_TO_G1": (
        "PRECOMPILE_BLS_G1MAP",
        "PRECOMPILE_BLS_G1MAP",
    ),  # Osaka address 0x10
    "BLS12_MAP_FP2_TO_G2": (
        "PRECOMPILE_BLS_G2MAP",
        "PRECOMPILE_BLS_G2MAP",
    ),  # Osaka address 0x11
}


# ---------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------


def ceil_2_significant(value: float) -> float:
    """Round ``value`` upward to at most two significant digits.

    4.569 -> 4.6, 100.0 -> 100, 101.0 -> 110, 0.851 -> 0.86.
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot round non-finite value {value!r}")
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    factor = 10.0 ** (exponent - 1)
    scaled = value / factor
    # Tolerate float noise (e.g. 46.00000000000001) before ceiling.
    units = math.ceil(scaled - 1e-9)
    return units * factor


def ceil_int(value: float) -> int:
    """Ceiling of ``value`` as an integer gas charge."""
    return int(math.ceil(value - 1e-9))


def gas_from_ns(ns: float, anchor_rate: float) -> float:
    """Convert a runtime in nanoseconds to gas at ``anchor_rate`` gas/s."""
    return ns * anchor_rate / 1e9


def ns_from_ms(ms: float) -> float:
    return float(ms) * 1e6


def _variant_of(case_id: str) -> str:
    """Strip the terminal work-size token from an exported case id."""
    return COUNT_TOKEN_RE.sub("", case_id)


def variant_param(variant_id: str, target_operation: str) -> str:
    """Deterministic per-variant gasfit parameter name."""
    suffix = hashlib.sha256(variant_id.encode()).hexdigest()[:12]
    return f"WORKLOAD_{target_operation}_{suffix}"


def _filter_prefix(variant_id: str) -> str:
    """``filter_by`` substring selecting exactly this variant's cases.

    The exported id is ``<variant>-<work-size token>``; the trailing ``-`` keeps
    prefix-adjacent variants (``mod_32_exp_3`` vs ``mod_32_exp_32``) apart.
    """
    return variant_id[:-1] + "-" if variant_id.endswith("]") else variant_id + "-"


def _words(nbytes: int) -> int:
    return (nbytes + 31) // 32


def _parse_calldata_len(value: Any) -> int | None:
    """Input length in bytes from an exported calldata field (0 when empty)."""
    if isinstance(value, list):
        return sum(_parse_calldata_len(item) or 0 for item in value)
    if not isinstance(value, str):
        return None
    try:
        return len(bytes.fromhex(value[2:] if value.startswith("0x") else value))
    except ValueError:
        return None


def _explicit_txdata_lengths(cases: Sequence[Mapping[str, Any]]) -> list[int] | None:
    """Return validated full transaction input lengths from exported cases."""
    lengths: list[int] = []
    for case in cases:
        transactions = case.get("transactions")
        if not isinstance(transactions, list) or not transactions:
            return None
        for transaction in transactions:
            if not isinstance(transaction, Mapping):
                return None
            data = transaction.get("data")
            if not isinstance(data, str) or not data.startswith("0x"):
                return None
            raw = data[2:]
            if len(raw) % 2:
                return None
            try:
                length = len(bytes.fromhex(raw))
            except ValueError:
                return None
            if length <= 0:
                return None
            lengths.append(length)
    if not lengths or len(set(lengths)) != 1:
        return None
    return lengths


# ---------------------------------------------------------------------------
# MODEXP current gas (precompiled_contracts/modexp.py, EIP-7883).
# ---------------------------------------------------------------------------

_MODEXP_INPUT_PATTERN = (
    r"base=(b'(?:\\.|[^'\\])*'|b\"(?:\\.|[^\"\\])*\")\s+"
    r"exponent=(b'(?:\\.|[^'\\])*'|b\"(?:\\.|[^\"\\])*\")\s+"
    r"modulus=(b'(?:\\.|[^'\\])*'|b\"(?:\\.|[^\"\\])*\")"
)
_MODEXP_INPUT_RE = re.compile(_MODEXP_INPUT_PATTERN)


def parse_modexp_operands(
    mod_exp_input: Any,
) -> tuple[bytes, bytes, bytes] | None:
    """Extract ``(base, exponent, modulus)`` bytes from the exported repr."""
    if not isinstance(mod_exp_input, str):
        return None
    match = _MODEXP_INPUT_RE.search(mod_exp_input)
    if match is None:
        return None
    try:
        return tuple(ast.literal_eval(g) for g in match.groups())  # type: ignore[return-value]
    except (ValueError, SyntaxError):
        return None


def modexp_complexity(base_len: int, mod_len: int) -> int:
    max_len = max(base_len, mod_len)
    if max_len > 32:
        words = (max_len + 7) // 8
        return 2 * words * words
    return 16


def modexp_iterations(exponent: bytes) -> int:
    exp_len = len(exponent)
    head = int.from_bytes(exponent[:32], "big")
    if exp_len <= 32:
        if head == 0:
            count = 0
        else:
            count = head.bit_length() - 1
    else:
        length_part = 16 * (exp_len - 32)
        bits_part = head.bit_length() - 1 if head > 0 else 0
        count = length_part + bits_part
    return max(count, 1)


def modexp_gas(base: bytes, exponent: bytes, modulus: bytes) -> int:
    return max(
        MODEXP_MIN_GAS,
        modexp_complexity(len(base), len(modulus)) * modexp_iterations(exponent),
    )


def bls_msm_gas(k: int, mul_gas: int, table: tuple[int, ...], max_discount: int) -> int:
    """``k * MUL * discount(k) // 1000`` (EIP-2537 discount schedule)."""
    discount = table[k - 1] if 1 <= k <= len(table) else max_discount
    return k * mul_gas * discount // MSM_DISCOUNT_MULTIPLIER


# ---------------------------------------------------------------------------
# Variant classification.
# ---------------------------------------------------------------------------


@dataclass
class VariantInfo:
    """One selected variant and its current-pricing ground truth."""

    variant_id: str
    family: str
    target_operation: str | None
    status: str
    pricing_group: str | None = None
    group_kind: str = GROUP_KIND_UNSUPPORTED
    # Current charge per call (int gas), None when unresolved/unsupported.
    current_charge_gas: int | None = None
    # Linear groups: unit count per call (words / bytes / pairs / rounds).
    units: int | None = None
    # Linear groups: (base param, base gas, per-unit param, per-unit gas).
    linear_params: tuple[str, int, str, int] | None = None
    # MSM groups: (mul param, mul gas, table, max discount).
    input_bytes: int | None = None  # raw input size where meaningful
    keccak_cache_affected: bool = False
    keccak_input_bytes: int | None = None
    reasons: list[str] = field(default_factory=list)
    case_ids: list[str] = field(default_factory=list)
    count_key: str | None = None
    # MODEXP: verified complexity x iterations product before the 500-gas floor.
    modexp_product: int | None = None


def _tokens(variant_id: str) -> list[str]:
    """Parametrize tokens of the variant's pytest id."""
    bracket = variant_id.split("::")[-1]
    inner = bracket[bracket.find("[") + 1 : -1]
    return inner.split("-") if inner else []


def _token_value(tokens: Sequence[str], key: str) -> str | None:
    prefix = key + "_"
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _case_source(case: Mapping[str, Any]) -> Mapping[str, Any]:
    src = case.get("parameters", {}).get("source_parameters", {})
    return src if isinstance(src, Mapping) else {}


def classify_variant(
    variant_id: str, cases: Sequence[Mapping[str, Any]]
) -> VariantInfo:
    """Classify one variant into its pricing group and current charge.

    Every formula is evaluated from the variant's exported input shape and
    the Osaka rules transcribed above. Dynamic input shapes that the export
    does not resolve are reported unresolved -- never guessed.
    """
    first = cases[0]
    info = VariantInfo(
        variant_id=variant_id,
        family=first.get("family", ""),
        target_operation=first.get("target_operation"),
        status=first.get("status", ""),
        case_ids=[str(c["id"]) for c in cases],
        count_key=first.get("parameters", {}).get("target_count_key"),
    )
    op = info.target_operation or ""
    tokens = _tokens(variant_id)
    src = _case_source(first)
    test_name = variant_id.split("::")[-1].split("[")[0]
    if info.status != "ready":
        info.group_kind = GROUP_KIND_UNSUPPORTED
        if test_name == "test_clz_diff":
            info.pricing_group = "OPCODE_CLZ"
            info.current_charge_gas = OSAKA_GAS_COSTS["OPCODE_CLZ"]
        elif test_name == "test_p256verify_uncachable":
            info.pricing_group = "PRECOMPILE_P256VERIFY"
            info.current_charge_gas = OSAKA_GAS_COSTS["PRECOMPILE_P256VERIFY"]
        elif test_name.startswith("test_modexp"):
            info.pricing_group = "PRECOMPILE_MODEXP"
        elif op in _FIXED_OPCODE_GROUPS:
            info.pricing_group, cost_field = _FIXED_OPCODE_GROUPS[op]
            info.current_charge_gas = OSAKA_GAS_COSTS[cost_field]
        elif op == "MODEXP":
            info.pricing_group = "PRECOMPILE_MODEXP"
        info.reasons.append(
            str(first.get("reason") or first.get("unsupported_reason") or "unsupported")
        )
        return info

    if op in _FIXED_OPCODE_GROUPS:
        group, cost_field = _FIXED_OPCODE_GROUPS[op]
        info.pricing_group = group
        info.group_kind = GROUP_KIND_FIXED
        info.current_charge_gas = OSAKA_GAS_COSTS[cost_field]
        return info

    if op == "KECCAK256":
        msg_bytes: int | None = None
        if test_name == "test_keccak":
            length = _parse_calldata_len(src.get("mem_alloc", ""))
            msg_bytes = length
            note = "offset/mem_update do not affect the keccak charge"
        elif test_name == "test_keccak_diff_mem_msg_sizes":
            msg_bytes = src.get("msg_size")
            note = "memory pre-sizing does not affect the keccak charge"
        elif test_name == "test_keccak_workload_witness":
            msg_bytes = src.get("input_length")
            note = "witness workload; msg size from input_length"
        else:  # test_keccak_max_permutations
            msg_bytes = src.get("optimal_input_length")
            note = (
                "optimal attack-block input length is resolved by the "
                "generator at fill time; the exporter must record it "
                "(source_parameters.optimal_input_length or equivalent "
                "resolved-input metadata)"
            )
        if not isinstance(msg_bytes, int) or msg_bytes < 0:
            info.pricing_group = "OPCODE_KECCAK256"
            info.group_kind = GROUP_KIND_LINEAR
            info.linear_params = (
                "OPCODE_KECCAK256_BASE",
                OSAKA_GAS_COSTS["OPCODE_KECCAK256_BASE"],
                "OPCODE_KECCAK256_PER_WORD",
                OSAKA_GAS_COSTS["OPCODE_KECCAK256_PER_WORD"],
            )
            info.reasons.append("unresolved input length: " + note)
            return info
        info.pricing_group = "OPCODE_KECCAK256"
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (
            "OPCODE_KECCAK256_BASE",
            OSAKA_GAS_COSTS["OPCODE_KECCAK256_BASE"],
            "OPCODE_KECCAK256_PER_WORD",
            OSAKA_GAS_COSTS["OPCODE_KECCAK256_PER_WORD"],
        )
        info.units = _words(msg_bytes)
        info.input_bytes = msg_bytes
        info.keccak_input_bytes = msg_bytes
        info.keccak_cache_affected = 1 <= msg_bytes <= KECCAK_CACHE_MAX_INPUT_BYTES
        info.current_charge_gas = (
            OSAKA_GAS_COSTS["OPCODE_KECCAK256_BASE"]
            + OSAKA_GAS_COSTS["OPCODE_KECCAK256_PER_WORD"] * info.units
        )
        info.reasons.append(note)
        return info

    if op == "EXP":
        base_gas = OSAKA_GAS_COSTS["OPCODE_EXP_BASE"]
        per_byte = OSAKA_GAS_COSTS["OPCODE_EXP_PER_BYTE"]
        info.pricing_group = "OPCODE_EXP"
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (
            "OPCODE_EXP_BASE",
            base_gas,
            "OPCODE_EXP_PER_BYTE",
            per_byte,
        )
        exponent_bytes: int | None = None
        if test_name == "test_arithmetic":
            # opcode_EXP variant: exponent is 2**256 - 1 (test_arithmetic.py
            # parametrization), i.e. 32 bytes.
            exponent_bytes = 32
        elif test_name == "test_exp_bench_arithmetic":
            # DUP2; EXP feeds each result into the next exponent. The
            # parameter is only the seed, not a fixed per-operation size.
            info.reasons.append(
                "EXP exponent evolves within the unrolled attack block; "
                "a fixed per-call charge cannot be inferred from its seed"
            )
            return info
        if exponent_bytes is None:
            info.reasons.append("unresolved exponent length for EXP variant")
            return info
        info.units = exponent_bytes
        info.input_bytes = exponent_bytes
        info.current_charge_gas = base_gas + per_byte * exponent_bytes
        return info

    if op in ("SHA2-256", "RIPEMD-160", "IDENTITY"):
        shapes = {
            "SHA2-256": (
                "PRECOMPILE_SHA256",
                OSAKA_GAS_COSTS["PRECOMPILE_SHA256_BASE"],
                OSAKA_GAS_COSTS["PRECOMPILE_SHA256_PER_WORD"],
            ),
            "RIPEMD-160": (
                "PRECOMPILE_RIPEMD160",
                OSAKA_GAS_COSTS["PRECOMPILE_RIPEMD160_BASE"],
                OSAKA_GAS_COSTS["PRECOMPILE_RIPEMD160_PER_WORD"],
            ),
            "IDENTITY": (
                "PRECOMPILE_IDENTITY",
                OSAKA_GAS_COSTS["PRECOMPILE_IDENTITY_BASE"],
                OSAKA_GAS_COSTS["PRECOMPILE_IDENTITY_PER_WORD"],
            ),
        }
        group, base_gas, per_word = shapes[op]
        info.pricing_group = group
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (group + "_BASE", base_gas, group + "_PER_WORD", per_word)
        size: int | None = None
        if "size" in src and isinstance(src["size"], int):
            size = src["size"]
        elif op == "IDENTITY" and test_name == "test_identity":
            size = src.get("optimal_input_length")
            info.reasons.append(
                "optimal input length is resolved by the generator at fill "
                "time; the exporter must record it (resolved-input metadata)"
            )
        elif test_name in ("test_sha256", "test_ripemd160"):
            size = src.get("optimal_input_length")
            info.reasons.append(
                "optimal input length is resolved by the generator at fill "
                "time; the exporter must record it (resolved-input metadata)"
            )
        if not isinstance(size, int) or size < 0:
            return info
        info.units = _words(size)
        info.input_bytes = size
        info.current_charge_gas = base_gas + per_word * info.units
        return info

    if op == "MODEXP":
        if test_name == "test_modexp_length_above_upper_bound":
            info.pricing_group = "MODEXP_INPUT_REJECTION"
            info.group_kind = GROUP_KIND_REJECTION
            info.current_charge_gas = 0
            info.reasons.append(
                "EIP-7823 operand-length check halts before charge_gas; "
                "the rejection path charges 0 gas and is not a priced input"
            )
            return info
        operands = parse_modexp_operands(src.get("mod_exp_input"))
        info.pricing_group = "PRECOMPILE_MODEXP"
        info.group_kind = GROUP_KIND_MODEXP
        if operands is None:
            info.reasons.append(
                "modexp operand bytes unavailable in the export; cannot "
                "evaluate the EIP-7883 formula"
            )
            return info
        base, exponent, modulus = operands
        info.modexp_product = modexp_complexity(
            len(base), len(modulus)
        ) * modexp_iterations(exponent)
        info.input_bytes = 96 + len(base) + len(exponent) + len(modulus)
        info.current_charge_gas = max(MODEXP_MIN_GAS, info.modexp_product)
        return info

    if op == "BLAKE2F":
        per_round = OSAKA_GAS_COSTS["PRECOMPILE_BLAKE2F_PER_ROUND"]
        info.pricing_group = "PRECOMPILE_BLAKE2F"
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (
            "PRECOMPILE_BLAKE2F_BASE",
            0,
            "PRECOMPILE_BLAKE2F_PER_ROUND",
            per_round,
        )
        rounds: int | None = None
        if "blake2f_zero_rounds" in tokens:
            rounds = 0
        elif "blake2f" in tokens:
            rounds = 0xFFFF
        else:
            raw = _token_value(tokens, "num_rounds")
            try:
                rounds = int(raw)
            except (TypeError, ValueError):
                rounds = None
        if rounds is None or rounds < 0:
            info.reasons.append("unresolved BLAKE2F rounds")
            return info
        info.units = rounds
        info.input_bytes = 213
        info.current_charge_gas = rounds * per_round
        return info

    if op == "BN128_PAIRING":
        base_gas = OSAKA_GAS_COSTS["PRECOMPILE_ECPAIRING_BASE"]
        per_point = OSAKA_GAS_COSTS["PRECOMPILE_ECPAIRING_PER_POINT"]
        info.pricing_group = "PRECOMPILE_ECPAIRING"
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (
            "PRECOMPILE_ECPAIRING_BASE",
            base_gas,
            "PRECOMPILE_ECPAIRING_PER_POINT",
            per_point,
        )
        pairs: int | None = None
        if "calldata" in src:
            calldata_len = _parse_calldata_len(src["calldata"])
            if calldata_len is not None:
                pairs = calldata_len // 192
        if pairs is None and isinstance(src.get("num_pairs"), int):
            pairs = src["num_pairs"]
        if pairs is None and test_name == "test_bn128_pairings_amortized":
            lengths = _explicit_txdata_lengths(cases)
            if lengths and lengths[0] % 192 == 0:
                pairs = lengths[0] // 192
        if pairs is None or pairs < 0:
            info.reasons.append("unresolved bn128 pair count")
            return info
        info.units = pairs
        info.input_bytes = pairs * 192
        info.current_charge_gas = base_gas + per_point * pairs
        return info

    if op == "BLS12_PAIRING":
        info.pricing_group = "PRECOMPILE_BLS12_PAIRING"
        info.group_kind = GROUP_KIND_LINEAR
        info.linear_params = (
            "PRECOMPILE_BLS12_PAIRING_BASE",
            BLS12_PAIRING_BASE,
            "PRECOMPILE_BLS12_PAIRING_PER_POINT",
            BLS12_PAIRING_PER_POINT,
        )
        pairs: int | None = None
        if "calldata" in src:
            calldata_len = _parse_calldata_len(src["calldata"])
            if calldata_len is not None:
                pairs = calldata_len // 384
        if pairs is None and isinstance(src.get("num_pairs"), int):
            pairs = src["num_pairs"]
        if pairs is None or pairs < 0:
            info.reasons.append("unresolved bls12 pair count")
            return info
        info.units = pairs
        info.input_bytes = pairs * 384
        info.current_charge_gas = BLS12_PAIRING_BASE + BLS12_PAIRING_PER_POINT * pairs
        return info

    if op in ("BLS12_G1MSM", "BLS12_G2MSM"):
        if op == "BLS12_G1MSM":
            group = "PRECOMPILE_BLS_G1MSM"
            mul_gas = OSAKA_GAS_COSTS["PRECOMPILE_BLS_G1MUL"]
            table, max_discount = G1_K_DISCOUNT, G1_MAX_DISCOUNT
        else:
            group = "PRECOMPILE_BLS_G2MSM"
            mul_gas = OSAKA_GAS_COSTS["PRECOMPILE_BLS_G2MUL"]
            table, max_discount = G2_K_DISCOUNT, G2_MAX_DISCOUNT
        info.pricing_group = group
        info.group_kind = GROUP_KIND_MSM
        info.msm_params = (group + "_MUL", mul_gas, table, max_discount)
        k: int | None = None
        if isinstance(src.get("k"), int):
            k = src["k"]
        elif "calldata" in src:
            calldata_len = _parse_calldata_len(src["calldata"])
            if calldata_len is not None:
                point_size = 160 if op == "BLS12_G1MSM" else 288
                k = calldata_len // point_size if calldata_len > 0 else None
        elif test_name == "test_bls12_381_uncachable":
            lengths = _explicit_txdata_lengths(cases)
            point_size = 160 if op == "BLS12_G1MSM" else 288
            if lengths and lengths[0] % point_size == 0:
                k = lengths[0] // point_size
        if k is None or k < 1:
            info.reasons.append("unresolved msm length k")
            return info
        info.units = k
        info.current_charge_gas = bls_msm_gas(k, mul_gas, table, max_discount)
        return info

    # PUSH1..PUSH32, DUP1..16, SWAP1..16 (incl. truncated-data variants).
    if re.fullmatch(r"PUSH([1-9]|[12][0-9]|3[0-2])", op):
        info.pricing_group = "OPCODE_PUSH"
        info.group_kind = GROUP_KIND_FIXED
        info.current_charge_gas = OSAKA_GAS_COSTS["OPCODE_PUSH"]
        return info
    if re.fullmatch(r"DUP([1-9]|1[0-6])", op):
        info.pricing_group = "OPCODE_DUP"
        info.group_kind = GROUP_KIND_FIXED
        info.current_charge_gas = OSAKA_GAS_COSTS["OPCODE_DUP"]
        return info
    if re.fullmatch(r"SWAP([1-9]|1[0-6])", op):
        info.pricing_group = "OPCODE_SWAP"
        info.group_kind = GROUP_KIND_FIXED
        info.current_charge_gas = OSAKA_GAS_COSTS["OPCODE_SWAP"]
        return info

    info.group_kind = GROUP_KIND_UNSUPPORTED
    info.reasons.append(
        f"no current-charge mapping for target operation {op!r}; refusing "
        "to guess -- extend the catalog from the Osaka sources"
    )
    return info


# ---------------------------------------------------------------------------
# Workload loading.
# ---------------------------------------------------------------------------


class WorkloadError(Exception):
    """Raised for malformed or out-of-scope workload exports."""


def load_workload(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or "cases" not in raw:
        raise WorkloadError(f"{path}: not a workload export (missing 'cases')")
    if raw.get("schema_version") != 2:
        raise WorkloadError(
            f"{path}: expected protocol schema_version 2, got "
            f"{raw.get('schema_version')!r}"
        )
    if raw.get("fork") != "Osaka":
        raise WorkloadError(f"{path}: expected fork 'Osaka', got {raw.get('fork')!r}")
    return raw


def split_case_roles(cases: Iterable[Mapping[str, Any]]) -> tuple[list, list]:
    """Split exported cases into (target, calibration) by campaign_role.

    The role defaults to ``target`` for legacy exports; calibration cases
    must always be explicitly tagged.
    """
    target: list[Mapping[str, Any]] = []
    calibration: list[Mapping[str, Any]] = []
    for case in cases:
        role = case.get("parameters", {}).get("campaign_role", "target")
        if role == "calibration":
            calibration.append(case)
        elif role == "target":
            target.append(case)
        else:
            raise WorkloadError(
                f"case {case.get('id')!r}: invalid campaign_role {role!r} "
                "(expected 'target' or 'calibration')"
            )
    return target, calibration


def group_variants(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group exported cases by variant id (opcount token stripped)."""
    variants: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        variants.setdefault(_variant_of(str(case["id"])), []).append(case)
    return variants


def classify_workload(workload: Mapping[str, Any]) -> dict[str, VariantInfo]:
    """Classify every selected target variant (ready and unsupported)."""
    target_cases, _calibration_cases = split_case_roles(workload["cases"])
    families = {c.get("family") for c in target_cases if c.get("family")}
    unknown = sorted(families - ALLOWED_FAMILIES)
    if unknown:
        raise WorkloadError(
            f"workload contains families outside the compute allowlist: "
            f"{unknown}; refusing to widen scope"
        )
    infos: dict[str, VariantInfo] = {}
    for variant_id, variant_cases in group_variants(target_cases).items():
        infos[variant_id] = classify_variant(variant_id, variant_cases)
    return infos


# ---------------------------------------------------------------------------
# create-config.
# ---------------------------------------------------------------------------


def build_analysis_config(
    workload: Mapping[str, Any],
    *,
    client: str,
    anchor_rate: float = DEFAULT_ANCHOR_RATE,
    min_sessions: int = MIN_SESSIONS_DEFAULT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the gasfit analysis config and its variant sidecar map.

    Returns ``(config, sidecar)``. One model per ready target variant;
    calibration-lane cases are never selected by any model.
    """
    target_cases, calibration_cases = split_case_roles(workload["cases"])
    target_variants = group_variants(target_cases)
    calibration_ids = [str(c["id"]) for c in calibration_cases]

    models: list[dict[str, Any]] = []
    new_params: dict[str, Any] = {}
    sidecar_variants: list[dict[str, Any]] = []
    seen_params: dict[str, str] = {}
    seen_prefixes: dict[str, str] = {}

    for variant_id in sorted(target_variants):
        cases = target_variants[variant_id]
        ready = [c for c in cases if c.get("status") == "ready"]
        info = classify_variant(variant_id, cases)
        if not ready:
            sidecar_variants.append(_sidecar_row(info, None, None))
            continue
        prefix = _filter_prefix(variant_id)
        collision = seen_prefixes.get(prefix)
        if collision is not None:
            raise WorkloadError(
                f"variant filter prefix {prefix!r} collides: {collision} vs "
                f"{variant_id}"
            )
        seen_prefixes[prefix] = variant_id
        # Calibration exclusion guard: the filter must never select a
        # calibration-lane case.
        if any(cid.startswith(prefix) for cid in calibration_ids):
            raise WorkloadError(
                f"variant filter prefix {prefix!r} would match a "
                "calibration-lane case; calibration rows must stay outside "
                "every target model"
            )
        matched = [
            cid for cid in (str(c["id"]) for c in ready) if cid.startswith(prefix)
        ]
        if len(matched) != len(ready):
            raise WorkloadError(
                f"variant filter prefix {prefix!r} matched {len(matched)} of "
                f"{len(ready)} ready cases for {variant_id}"
            )
        op = ready[0].get("target_operation")
        if not op:
            raise WorkloadError(f"ready variant {variant_id} lacks target_operation")
        param = variant_param(variant_id, op)
        if param in seen_params:
            raise WorkloadError(f"duplicate generated parameter {param!r}")
        seen_params[param] = variant_id
        new_params[param] = None
        model: dict[str, Any] = {
            "test_name": variant_id.split("::")[-1].split("[")[0],
            "target_operation": op,
            "filter_by": [prefix],
            "model_params": {"target_coef": param},
        }
        count_key = ready[0].get("parameters", {}).get("target_count_key", "")
        if isinstance(count_key, str) and count_key.startswith("PRECOMPILE_"):
            model["target_operation_count_source"] = count_key
        models.append(model)
        sidecar_variants.append(_sidecar_row(info, param, prefix))

    if not models:
        raise WorkloadError("no ready target variants found; nothing to model")

    qualification = dict(QUALIFICATION_GATES)
    qualification["min_sessions"] = min_sessions
    config = {
        "version": 1,
        "clients": [client],
        "anchor_rate": int(anchor_rate),
        "gas_costs": {"fork": "osaka"},
        "output": {"plots": False},
        "modeling": {
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_seed": RANDOM_SEED,
        },
        "glue_adjustment": {"enabled": True},
        "qualification": qualification,
        "campaign": {
            "eligible_phases": ["qualification"],
            "eligible_statuses": ["executed"],
            "require_correctness_passed": True,
        },
        "pricing_scenarios": [
            {
                "name": PRICING_SCENARIO_NAME,
                "anchor_rate": int(anchor_rate),
                "margin_pct": 0.0,
            }
        ],
        "models": {"custom": models},
        "new_params": new_params,
    }
    sidecar = {
        "workload_schema_version": workload.get("schema_version"),
        "fork": workload.get("fork"),
        "anchor_rate": int(anchor_rate),
        "campaign_roles": {
            "target_cases": len(target_cases),
            "calibration_cases": len(calibration_cases),
            "calibration_priced": 0,
        },
        "variants": sidecar_variants,
    }
    return config, sidecar


def _sidecar_row(
    info: VariantInfo, param: str | None, prefix: str | None
) -> dict[str, Any]:
    return {
        "variant_id": info.variant_id,
        "family": info.family,
        "target_operation": info.target_operation,
        "status": info.status,
        "campaign_role": "target",
        "pricing_group": info.pricing_group,
        "group_kind": info.group_kind,
        "current_charge_gas": info.current_charge_gas,
        "units": info.units,
        "input_bytes": info.input_bytes,
        "keccak_cache_affected": info.keccak_cache_affected,
        "model_param": param,
        "filter_prefix": prefix,
        "target_operation_count_source": info.count_key
        if info.count_key and info.count_key.startswith("PRECOMPILE_")
        else None,
        "reasons": info.reasons,
    }


def _validate_config_document(config: dict[str, Any], path: Path) -> str:
    """Round-trip the generated config through gasfit's own validator."""
    try:
        from evm_gasfit.config import load_config  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - analyzer env has it.
        return (
            f"SKIPPED (evm_gasfit not importable here: {exc}); validate on "
            "the analyzer image before running the campaign"
        )
    import tempfile  # noqa: PLC0415

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        load_config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    del path
    return "validated with evm_gasfit.config.load_config"


def cmd_create_config(args: argparse.Namespace) -> int:
    workload = load_workload(Path(args.workload))
    config, sidecar = build_analysis_config(
        workload,
        client=args.client,
        anchor_rate=args.anchor_rate,
        min_sessions=args.min_sessions,
    )
    out = Path(args.out)
    if out.exists():
        raise WorkloadError(f"{out} already exists; write to a fresh path")
    out.write_text(json.dumps(config, indent=2) + "\n")
    sidecar_path = out.with_name(out.name + ".variants.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    ready = sum(1 for v in sidecar["variants"] if v["model_param"])
    unresolved = sorted(
        v["variant_id"]
        for v in sidecar["variants"]
        if v["status"] == "ready" and v["current_charge_gas"] is None
    )
    validation = _validate_config_document(config, out)
    print(
        f"created {out} with {ready} per-variant glue-enabled target models "
        f"({sidecar['campaign_roles']['calibration_cases']} calibration "
        f"cases present, excluded from every model); sidecar {sidecar_path}; "
        f"validation: {validation}"
    )
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} ready variant(s) lack a resolved "
            "current-charge input shape and will be reported as coverage "
            "gaps rather than guessed:",
            file=sys.stderr,
        )
        for variant_id in unresolved:
            print(f"  - {variant_id}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# build: analysis-output loading.
# ---------------------------------------------------------------------------


class BuildError(Exception):
    """Raised when the analysis outputs are missing or inconsistent."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BuildError(f"missing analysis output {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _load_analysis_status(analysis_dir: Path) -> dict[str, Any]:
    path = analysis_dir / "analysis_status.json"
    if not path.is_file():
        raise BuildError(
            f"missing {path}; the campaign archive must retain the gasfit "
            "provenance record so recommendations can be evidence-linked"
        )
    status = json.loads(path.read_text())
    if not isinstance(status, dict):
        raise BuildError(f"{path}: expected a JSON object")
    return status


def _recover_models(
    analysis_status: Mapping[str, Any], analysis_dir: Path
) -> list[dict[str, Any]]:
    """Recover custom models from the embedded, hash-verified config."""
    inputs = analysis_status.get("inputs")
    config = inputs.get("config") if isinstance(inputs, Mapping) else None
    if not isinstance(config, Mapping):
        raise BuildError(
            f"cannot recover the analysis config from {analysis_dir}; "
            "analysis_status.json must contain inputs.config provenance"
        )
    content = config.get("content")
    expected_sha256 = config.get("sha256")
    if not isinstance(content, str) or not isinstance(expected_sha256, str):
        raise BuildError(
            f"{analysis_dir}/analysis_status.json: inputs.config must contain "
            "embedded string content and sha256"
        )
    actual_sha256 = hashlib.sha256(content.encode()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise BuildError(
            f"{analysis_dir}/analysis_status.json: embedded config sha256 "
            f"{actual_sha256} does not match recorded {expected_sha256}"
        )
    try:
        import yaml

        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise BuildError(
            f"{analysis_dir}/analysis_status.json: embedded config is not "
            f"valid YAML/JSON: {exc}"
        ) from exc
    if isinstance(parsed, Mapping):
        models_document = parsed.get("models")
        if isinstance(models_document, Mapping):
            models = models_document.get("custom")
            if isinstance(models, list) and models:
                return models
    raise BuildError(
        f"{analysis_dir}/analysis_status.json: embedded config has no "
        "non-empty models.custom list"
    )


@dataclass
class VariantEvidence:
    """Per-variant evidence extracted from the completed analysis."""

    source_label: str = ""
    raw_point_ms: float | None = None
    raw_low_ms: float | None = None
    raw_high_ms: float | None = None
    adjusted_point_ms: float | None = None
    adjusted_low_ms: float | None = None
    adjusted_high_ms: float | None = None
    glue_adjustment_ms: float | None = None
    glue_interval_conditional: bool | False = False
    qualification_status: str = ""
    adjusted_estimate_status: str = ""
    qualification_reasons: str = ""
    glue_coverage_complete: bool | None = None
    glue_priced: str = ""
    glue_unpriced: str = ""
    glue_bundled: str = ""
    glue_detection_status: str = ""
    selected_client: str = ""
    model_param: str = ""


def _collect_evidence(
    analysis_dir: Path,
    analysis_status: Mapping[str, Any],
    infos: Mapping[str, VariantInfo],
) -> dict[str, VariantEvidence]:
    """Join results/qualification/new_gas rows back onto variants."""
    models = _recover_models(analysis_status, analysis_dir)
    label_by_variant: dict[str, str] = {}
    for index, model in enumerate(models):
        prefix = (model.get("filter_by") or [""])[0]
        matches = [v for v in infos if (_filter_prefix(v)) == prefix]
        if len(matches) != 1:
            # Fall back to substring matching (identical in practice).
            matches = [v for v in infos if v.startswith(prefix)]
        if len(matches) != 1:
            raise BuildError(
                f"analysis model models.custom[{index}] filter {prefix!r} "
                f"matches {len(matches)} workload variants"
            )
        label_by_variant[matches[0]] = f"models.custom[{index}]"

    results = _read_csv(analysis_dir / "results.csv")
    by_label: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in results:
        by_label.setdefault(row.get("source_label", ""), {}).setdefault(
            row.get("client_name", ""), []
        ).append(row)
    qualification = _read_csv(analysis_dir / "qualification.csv")
    qual_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in qualification:
        qual_by_key[(row.get("source_label", ""), row.get("client_name", ""))] = row
    proposal_rows = _read_csv(analysis_dir / "new_gas_all_params.csv")
    proposal_by_param: dict[str, list[dict[str, str]]] = {}
    for row in proposal_rows:
        proposal_by_param.setdefault(row.get("gas_param", ""), []).append(row)

    evidence: dict[str, VariantEvidence] = {}
    for variant_id, info in infos.items():
        if info.status != "ready":
            continue
        label = label_by_variant.get(variant_id)
        if label is None:
            continue
        ev = VariantEvidence(source_label=label)
        per_client = by_label.get(label, {})
        # Raw slope: worst case (max point) across clients.
        best_client, best_point = None, -1.0
        for client, rows in sorted(per_client.items()):
            for row in rows:
                point = _f(row.get("target_coef_runtime_ms"))
                if point is not None and point > best_point:
                    best_point, best_client = point, (client, row)
        if best_client is not None:
            client, row = best_client
            ev.selected_client = client
            ev.raw_point_ms = _f(row.get("target_coef_runtime_ms"))
            ev.raw_low_ms = _f(row.get("target_coef_conf_int_low"))
            ev.raw_high_ms = _f(row.get("target_coef_conf_int_high"))
            qual = qual_by_key.get((label, client))
            if qual:
                ev.qualification_status = qual.get("status", "")
                ev.adjusted_estimate_status = qual.get("adjusted_estimate_status", "")
                ev.qualification_reasons = qual.get("reasons", "")
        # Adjusted slope + glue coverage from the proposal frame keyed by the
        # deterministic per-variant parameter.
        op = info.target_operation or ""
        param = variant_param(variant_id, op)
        ev.model_param = param
        candidates = proposal_by_param.get(param, [])
        adj_point, adj_row = -1.0, None
        for row in candidates:
            point = _f(row.get("runtime_ms"))
            if point is not None and point > adj_point:
                adj_point, adj_row = point, row
        if adj_row is not None:
            ev.adjusted_point_ms = _f(adj_row.get("runtime_ms"))
            ev.adjusted_low_ms = _f(adj_row.get("conf_int_low"))
            ev.adjusted_high_ms = _f(adj_row.get("conf_int_high"))
            ev.glue_adjustment_ms = _f(adj_row.get("glue_adjustment"))
            conditional = adj_row.get("glue_interval_conditional", "")
            ev.glue_interval_conditional = str(conditional).strip().lower() in {
                "true",
                "1",
            }
            if not ev.qualification_status:
                ev.qualification_status = adj_row.get("qualification_status", "")
            ev.glue_priced = adj_row.get("glue_priced_opcodes", "")
            ev.glue_unpriced = adj_row.get("glue_unpriced_opcodes", "")
            ev.glue_bundled = adj_row.get("glue_bundled_opcodes", "")
            ev.glue_detection_status = adj_row.get("glue_detection_status", "")
            coverage = adj_row.get("glue_coverage_complete", None)
            if coverage is None or coverage == "":
                ev.glue_coverage_complete = None
            else:
                ev.glue_coverage_complete = str(coverage).strip().lower() in {
                    "true",
                    "1",
                }
        evidence[variant_id] = ev
    return evidence


def _coverage_blockers(info: VariantInfo, ev: VariantEvidence | None) -> list[str]:
    """Exact reasons the variant cannot support a deployable price."""
    blockers: list[str] = []
    if ev is None:
        return ["no analysis evidence for this variant"]
    if ev.qualification_status != "qualified":
        blockers.append(
            f"qualification status {ev.qualification_status or 'missing'}"
            + (f" ({ev.qualification_reasons})" if ev.qualification_reasons else "")
        )
    if ev.adjusted_estimate_status != "qualified":
        blockers.append(
            f"adjusted estimate status {ev.adjusted_estimate_status or 'missing'}"
        )
    if ev.glue_coverage_complete is None:
        blockers.append(
            "glue coverage unknown (analysis lacks glue_coverage_complete; "
            "missing driver coverage is real, not hypothetical)"
        )
    elif not ev.glue_coverage_complete:
        unpriced = ev.glue_unpriced or "unlisted"
        blockers.append(
            f"glue coverage incomplete: detected supporting opcode(s) not "
            f"isolable: {unpriced}"
        )
    if ev.glue_interval_conditional:
        blockers.append("glue interval conditional (uncertainty not propagated)")
    adjusted_values = (
        ev.adjusted_point_ms,
        ev.adjusted_low_ms,
        ev.adjusted_high_ms,
    )
    if any(
        value is None or not math.isfinite(value) or value <= 0
        for value in adjusted_values
    ):
        blockers.append(
            "adjusted point and both 95% CI bounds must be finite and positive"
        )
    elif not (ev.adjusted_low_ms <= ev.adjusted_point_ms <= ev.adjusted_high_ms):
        blockers.append("adjusted confidence bounds are not ordered around point")
    if info.current_charge_gas is None:
        blockers.append("current charge unresolved: " + "; ".join(info.reasons))
    if info.keccak_cache_affected:
        blockers.append(
            f"keccak input {info.keccak_input_bytes}B <= "
            f"{KECCAK_CACHE_MAX_INPUT_BYTES}B: evm2's process-global "
            "alloy keccak cache can serve these warm within a session; "
            "not worst-case evidence"
        )
    return blockers


def _coverage_word(ev: VariantEvidence | None) -> str | None:
    if ev is None:
        return None
    if ev.glue_coverage_complete is None:
        return "unknown"
    return "complete" if ev.glue_coverage_complete else "incomplete"


@dataclass
class BudgetEvidence:
    """Exact per-variant marginal charged-gas evidence."""

    counts: list[int] = field(default_factory=list)
    charged_gas: list[int] = field(default_factory=list)
    total_marginal_gas: Fraction | None = None
    other_marginal_gas: Fraction | None = None
    affine: bool = False
    sample_ids: list[str] = field(default_factory=list)
    abstain_reason: str = ""


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record.get("phase") == "diagnostic"
                and record.get("status") == "executed"
                and record.get("correctness_passed") is True
            ):
                records.append(record)
    return records


def _budget_evidence(
    info: VariantInfo,
    diagnostics: Mapping[str, list[Mapping[str, Any]]],
) -> BudgetEvidence:
    """Prove exact affine charged gas and derive non-target marginal gas."""
    out = BudgetEvidence()
    points: dict[int, tuple[int, str]] = {}
    for row in diagnostics.get(info.variant_id, []):
        count = row.get("target_count")
        gas = row.get("charged_gas")
        if not isinstance(count, int) or not isinstance(gas, int):
            continue
        sample_id = str(row.get("sample_id", ""))
        previous = points.get(count)
        if previous is not None and previous[0] != gas:
            out.abstain_reason = (
                f"conflicting duplicate charged-gas rows for target_count {count}"
            )
            return out
        points.setdefault(count, (gas, sample_id))
    if len(points) < 2:
        out.abstain_reason = (
            "insufficient count grid: need >= 2 executed diagnostic points "
            "with charged gas"
        )
        return out
    counts = sorted(points)
    out.counts = counts
    out.charged_gas = [points[c][0] for c in counts]
    out.sample_ids = [points[c][1] for c in counts]
    slopes = {
        Fraction(out.charged_gas[i + 1] - out.charged_gas[i], counts[i + 1] - counts[i])
        for i in range(len(counts) - 1)
    }
    if len(slopes) != 1:
        out.abstain_reason = (
            "charged gas is not exactly affine in the target count over the "
            "sampled grid (per-step marginal gas varies); abstaining"
        )
        return out
    out.affine = True
    out.total_marginal_gas = slopes.pop()
    if info.current_charge_gas is None:
        out.abstain_reason = "current target charge unresolved"
        return out
    out.other_marginal_gas = out.total_marginal_gas - info.current_charge_gas
    if out.other_marginal_gas < 0:
        out.affine = False
        out.abstain_reason = (
            "negative non-target marginal gas after subtracting current target "
            "charge; budget evidence is contradictory"
        )
    return out


def _budget_need_gas(
    info: VariantInfo,
    budget: BudgetEvidence,
    ev: VariantEvidence | None,
    anchor_rate: float,
) -> float | None:
    """Return residual target charge only for fully qualified raw evidence."""
    if not budget.affine or budget.other_marginal_gas is None:
        if not budget.abstain_reason:
            budget.abstain_reason = "charged gas is not exact affine evidence"
        return None
    if ev is None:
        budget.abstain_reason = "missing raw qualification evidence"
        return None
    if ev.qualification_status != "qualified":
        budget.abstain_reason = (
            f"raw qualification status {ev.qualification_status or 'missing'}"
        )
        return None
    raw_values = (ev.raw_point_ms, ev.raw_low_ms, ev.raw_high_ms)
    if any(
        value is None or not math.isfinite(value) or value <= 0 for value in raw_values
    ):
        budget.abstain_reason = (
            "raw point and both confidence bounds must be finite and positive"
        )
        return None
    assert ev.raw_point_ms is not None
    assert ev.raw_low_ms is not None
    assert ev.raw_high_ms is not None
    if not ev.raw_low_ms <= ev.raw_point_ms <= ev.raw_high_ms:
        budget.abstain_reason = "raw confidence bounds are not ordered around point"
        return None
    if info.current_charge_gas is None:
        budget.abstain_reason = "current target charge unresolved"
        return None
    if info.keccak_cache_affected:
        budget.abstain_reason = "short-input keccak cache risk excludes budget evidence"
        return None
    raw_upper_gas = gas_from_ns(ns_from_ms(ev.raw_high_ms), anchor_rate)
    residual = raw_upper_gas - float(budget.other_marginal_gas)
    return float(max(info.current_charge_gas, ceil_2_significant(max(0.0, residual))))


# ---------------------------------------------------------------------------
# Group aggregation.
# ---------------------------------------------------------------------------


@dataclass
class GroupResult:
    """Aggregated judgment for one pricing group."""

    group: str
    kind: str
    variants: list[str] = field(default_factory=list)


def _linear_current(info: VariantInfo) -> float | None:
    if info.linear_params is None or info.units is None:
        return None
    _bp, base, _pp, per_unit = info.linear_params
    return float(base + per_unit * info.units)


def _group_candidate_fixed(
    group_variants: Sequence[tuple[VariantInfo, VariantEvidence | None, float | None]],
    current: int,
) -> dict[str, Any]:
    """Worst-case candidate for a fixed-charge group."""
    upper = max((g for (_i, _e, g) in group_variants if g is not None), default=None)
    candidate = None
    if upper is not None:
        candidate = max(float(current), ceil_2_significant(upper))
    return {
        "uniform_scale": None if candidate is None else candidate / current,
        "params": None
        if candidate is None
        else {"value": ceil_int(candidate), "exact_2sig": candidate},
        "upper_bound_gas": upper,
    }


def _group_candidate_modexp(
    group_variants: Sequence[tuple[VariantInfo, VariantEvidence | None, float | None]],
    anchor_rate: float,
) -> dict[str, Any]:
    """Build a MODEXP product multiplier while preserving its gas floor."""
    requirements: dict[str, int | None] = {}
    above_floor: dict[str, int] = {}
    for info, _ev, required in group_variants:
        rounded_required = None if required is None else ceil_int(required)
        requirements[info.variant_id] = rounded_required
        if rounded_required is not None and rounded_required > MODEXP_MIN_GAS:
            above_floor[info.variant_id] = rounded_required
    missing_for_requirement = [
        info.variant_id
        for info, _ev, _required in group_variants
        if info.variant_id in above_floor
        and (info.modexp_product is None or info.modexp_product <= 0)
    ]
    if missing_for_requirement:
        return {
            "abstained": True,
            "unsupported_shape": "unresolved modexp product",
            "unsupported_variants": missing_for_requirement,
            "reason": (
                "an above-floor MODEXP requirement needs verified "
                "complexity x iterations product"
            ),
        }
    ratios: dict[str, float] = {}
    for info, _ev, _required in group_variants:
        required = above_floor.get(info.variant_id)
        if (
            required is not None
            and info.modexp_product is not None
            and info.modexp_product > 0
        ):
            ratios[info.variant_id] = required / info.modexp_product
    multiplier = max(1.0, ceil_2_significant(max(ratios.values(), default=1.0)))
    missing_for_evaluation = [
        info.variant_id
        for info, _ev, _required in group_variants
        if (info.modexp_product is None or info.modexp_product <= 0) and multiplier > 1
    ]
    if missing_for_evaluation:
        return {
            "abstained": True,
            "unsupported_shape": "unresolved modexp product",
            "unsupported_variants": missing_for_evaluation,
            "reason": (
                "the scaled MODEXP formula cannot evaluate a candidate "
                "charge without a verified complexity x iterations product"
            ),
        }
    candidate_charge_gas = {
        info.variant_id: (
            MODEXP_MIN_GAS
            if info.modexp_product is None or info.modexp_product <= 0
            else max(MODEXP_MIN_GAS, ceil_int(multiplier * info.modexp_product))
        )
        for info, _ev, _required in group_variants
    }
    overpricing: dict[str, float] = {}
    for info, ev, _required in group_variants:
        if ev is None or ev.adjusted_point_ms is None:
            continue
        point_gas = gas_from_ns(ns_from_ms(ev.adjusted_point_ms), anchor_rate)
        if point_gas > 0:
            overpricing[info.variant_id] = (
                candidate_charge_gas[info.variant_id] / point_gas
            )
    return {
        "uniform_scale": None,
        "product_multiplier": multiplier,
        "params": {
            "PRECOMPILE_MODEXP_MULTIPLIER": multiplier,
            "PRECOMPILE_MODEXP_MIN_GAS": MODEXP_MIN_GAS,
        },
        "candidate_charge_gas": candidate_charge_gas,
        "variants": requirements,
        "overpricing": (
            {
                "min": min(overpricing.values()),
                "max": max(overpricing.values()),
                "per_variant": overpricing,
            }
            if overpricing
            else None
        ),
    }


def _group_candidate_scaled(
    group_variants: Sequence[tuple[VariantInfo, VariantEvidence | None, float | None]],
    anchor_rate: float,
) -> dict[str, Any]:
    """Uniform-scale candidate preserving the existing formula shape.

    Returns the scale, the per-variant ratio evidence, and the conservative
    overpricing spread of the uniform scale across sampled inputs.
    """

    if group_variants and group_variants[0][0].group_kind == GROUP_KIND_MODEXP:
        return _group_candidate_modexp(group_variants, anchor_rate)
    zero_charge_variants = [
        info.variant_id
        for info, _ev, upper in group_variants
        if info.current_charge_gas == 0 and upper is not None and upper > 0
    ]
    if zero_charge_variants:
        return {
            "abstained": True,
            "unsupported_shape": "zero-charge linear variant",
            "unsupported_variants": zero_charge_variants,
            "reason": (
                "no finite uniform scale of the existing formula covers a "
                "positive upper bound at zero current charge"
            ),
        }
    ratios: dict[str, float] = {}
    for info, _ev, upper in group_variants:
        if upper is None or not info.current_charge_gas:
            continue
        ratios[info.variant_id] = upper / info.current_charge_gas
    if not ratios:
        return {
            "uniform_scale": None,
            "ratios": {},
            "params": None,
            "overpricing": None,
        }
    scale = max(1.0, ceil_2_significant(max(ratios.values())))
    overpricing: dict[str, float] = {}
    for info, ev, _upper in group_variants:
        if ev is None or ev.adjusted_point_ms is None:
            continue
        point_gas = gas_from_ns(ns_from_ms(ev.adjusted_point_ms), anchor_rate)
        if info.current_charge_gas and point_gas > 0:
            overpricing[info.variant_id] = scale * info.current_charge_gas / point_gas
    spread = None
    if overpricing:
        spread = {
            "min": min(overpricing.values()),
            "max": max(overpricing.values()),
            "per_variant": overpricing,
        }
    return {
        "uniform_scale": scale,
        "ratios": ratios,
        "params": None,
        "overpricing": spread,
        "note": (
            "uniform scale preserves the existing formula shape; the spread "
            "shows how much the conservative scale overprices the "
            "cheapest-to-serve sampled inputs relative to their adjusted "
            "point estimates"
        ),
    }


def _linear_alternative(
    group_variants: Sequence[tuple[VariantInfo, VariantEvidence | None, float | None]],
    anchor_rate: float,
    base_param: str,
    base_gas: int,
    per_unit_param: str,
    per_unit_gas: int,
) -> dict[str, Any] | None:
    """Evidence-supported base+per-unit alternative candidate.

    Requires >= 2 distinct unit magnitudes with usable evidence. The
    per-unit coefficient is the max pairwise point slope; the base is the
    envelope residual of per-variant upper bounds. The result covers every
    sampled input's upper bound (a sampled-domain envelope), which is NOT a
    joint 95% guarantee because the per-variant intervals are correlated
    point estimates.
    """
    points: list[tuple[VariantInfo, float, float | None]] = []
    for info, ev, _upper in group_variants:
        if ev is None or ev.adjusted_point_ms is None or info.units is None:
            continue
        point_gas = gas_from_ns(ns_from_ms(ev.adjusted_point_ms), anchor_rate)
        upper_gas = (
            None
            if ev.adjusted_high_ms is None
            else gas_from_ns(ns_from_ms(ev.adjusted_high_ms), anchor_rate)
        )
        points.append((info, point_gas, upper_gas))
    units = {info.units for info, _p, _u in points if info.units is not None}
    if len(units) < 2:
        return None
    per_unit = 0.0
    for i, (info_a, point_a, _ua) in enumerate(points):
        for info_b, point_b, _ub in points[i + 1 :]:
            du = abs((info_b.units or 0) - (info_a.units or 0))
            if du > 0:
                per_unit = max(per_unit, abs(point_b - point_a) / du)
    if per_unit <= 0:
        return None
    per_unit_candidate = ceil_2_significant(per_unit)
    base_residual = 0.0
    for info, _point, upper in points:
        if upper is None or info.units is None:
            continue
        base_residual = max(base_residual, upper - per_unit_candidate * info.units)
    base_candidate = ceil_2_significant(max(0.0, base_residual))
    return {
        "params": {
            base_param: ceil_int(max(base_gas, base_candidate)),
            per_unit_param: ceil_int(max(per_unit_gas, per_unit_candidate)),
        },
        "exact_2sig": {
            base_param: max(float(base_gas), base_candidate),
            per_unit_param: max(float(per_unit_gas), per_unit_candidate),
        },
        "envelope": (
            "sampled-domain envelope of per-variant 95% upper bounds; the "
            "per-variant intervals are correlated point estimates, so this "
            "is not a joint 95% guarantee"
        ),
    }


def _apply_resolved_inputs(
    workload: Mapping[str, Any],
    infos: Mapping[str, VariantInfo],
    sidecar: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply a reviewed input-length sidecar without overriding known prices."""
    if sidecar.get("schema_version") != 1:
        raise BuildError("resolved-input sidecar schema_version must be 1")
    expected_hash = hashlib.sha256(
        json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if sidecar.get("canonical_workload_sha256") != expected_hash:
        raise BuildError("resolved-input sidecar workload hash does not match")
    entries = sidecar.get("inputs")
    if not isinstance(entries, list):
        raise BuildError("resolved-input sidecar inputs must be a list")
    seen: set[str] = set()
    annotations: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BuildError("resolved-input sidecar entry must be an object")
        variant_id = entry.get("variant_id")
        if not isinstance(variant_id, str) or variant_id in seen:
            raise BuildError("resolved-input sidecar variants must be unique strings")
        seen.add(variant_id)
        info = infos.get(variant_id)
        if info is None:
            raise BuildError(
                f"resolved-input sidecar has unknown variant {variant_id!r}"
            )
        if (
            info.status != "ready"
            or info.pricing_group
            not in {
                "OPCODE_KECCAK256",
                "PRECOMPILE_IDENTITY",
                "PRECOMPILE_SHA256",
                "PRECOMPILE_RIPEMD160",
            }
            or info.linear_params is None
        ):
            raise BuildError(f"resolved-input sidecar cannot resolve {variant_id!r}")
        if info.units is not None or info.input_bytes is not None:
            raise BuildError(
                f"resolved-input sidecar overrides known price {variant_id!r}"
            )
        case_ids = entry.get("case_ids")
        if not isinstance(case_ids, list) or set(case_ids) != set(info.case_ids):
            raise BuildError(
                f"resolved-input sidecar case binding mismatch for {variant_id!r}"
            )
        input_bytes = entry.get("input_bytes")
        if (
            isinstance(input_bytes, bool)
            or not isinstance(input_bytes, int)
            or input_bytes < 0
        ):
            raise BuildError(
                f"resolved-input sidecar input_bytes invalid for {variant_id!r}"
            )
        base_param, base_gas, _unit_param, per_unit = info.linear_params
        del base_param
        info.units = _words(input_bytes)
        info.input_bytes = input_bytes
        info.keccak_input_bytes = (
            input_bytes if info.pricing_group == "OPCODE_KECCAK256" else None
        )
        info.keccak_cache_affected = (
            info.pricing_group == "OPCODE_KECCAK256"
            and 1 <= input_bytes <= KECCAK_CACHE_MAX_INPUT_BYTES
        )
        info.current_charge_gas = base_gas + per_unit * info.units
        info.reasons = [
            reason
            for reason in info.reasons
            if not reason.startswith(
                ("unresolved input length", "optimal input length")
            )
        ]
        evidence = entry.get("evidence")
        if not isinstance(evidence, str) or not evidence:
            raise BuildError(
                f"resolved-input sidecar evidence missing for {variant_id!r}"
            )
        info.reasons.append("resolved input sidecar: " + evidence)
        annotations.append(
            {"variant_id": variant_id, "input_bytes": input_bytes, "evidence": evidence}
        )
    for variant_id, info in infos.items():
        if (
            info.status == "ready"
            and info.pricing_group
            in {
                "OPCODE_KECCAK256",
                "PRECOMPILE_IDENTITY",
                "PRECOMPILE_SHA256",
                "PRECOMPILE_RIPEMD160",
            }
            and info.units is None
            and variant_id not in seen
        ):
            raise BuildError(f"resolved-input sidecar missing variant {variant_id!r}")
    return annotations


def build_recommendations(
    workload: Mapping[str, Any],
    analysis_dir: Path,
    diagnostics_path: Path | None,
    anchor_rate: float = DEFAULT_ANCHOR_RATE,
    resolved_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full recommendation document."""
    analysis_dir = Path(analysis_dir)
    infos = classify_workload(workload)
    resolved_annotations = (
        _apply_resolved_inputs(workload, infos, resolved_inputs)
        if resolved_inputs is not None
        else []
    )
    analysis_status = _load_analysis_status(analysis_dir)
    evidence = _collect_evidence(analysis_dir, analysis_status, infos)
    diagnostics: dict[str, list[Mapping[str, Any]]] = {}
    if diagnostics_path is not None:
        for record in load_diagnostics(diagnostics_path):
            variant = _variant_of(str(record.get("case_id", "")))
            diagnostics.setdefault(variant, []).append(record)

    # ---- per-variant rows -------------------------------------------------
    rows: list[dict[str, Any]] = []
    for variant_id in sorted(infos):
        rows.append(
            _variant_row(
                infos[variant_id],
                evidence.get(variant_id),
                _budget_evidence(infos[variant_id], diagnostics)
                if diagnostics
                else BudgetEvidence(
                    abstain_reason="diagnostics not supplied; budget-aware "
                    "lane unavailable"
                ),
                evidence.get(variant_id),
                anchor_rate,
            )
        )

    # ---- per-group judgments ---------------------------------------------
    groups: dict[str, list[VariantInfo]] = {}
    for info in infos.values():
        if info.pricing_group:
            groups.setdefault(info.pricing_group, []).append(info)
    group_reports: list[dict[str, Any]] = []
    for group in sorted(groups):
        group_reports.append(
            _group_report(
                group, groups[group], infos, evidence, diagnostics, anchor_rate
            )
        )

    summary = {
        "anchor_rate": anchor_rate,
        "increase_threshold": INCREASE_THRESHOLD,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "variants_total": len(rows),
        "variants_ready": sum(1 for r in rows if r["status"] == "ready"),
        "variants_unsupported": sum(
            1 for r in rows if r["group_kind"] == GROUP_KIND_UNSUPPORTED
        ),
        "groups_total": len(group_reports),
        "groups_increase_candidate": sum(
            1 for g in group_reports if g["decision"] == "increase_candidate"
        ),
        "groups_keep": sum(
            1 for g in group_reports if g["decision"].startswith("keep")
        ),
        "groups_blocked": sum(
            1 for g in group_reports if g["decision"].startswith("blocked")
        ),
    }
    return {
        "policy": {
            "anchor_rate": anchor_rate,
            "margin_pct": 0,
            "interval_policy": "95% adjusted upper bound",
            "change_policy": (
                "no decreases; increase candidates only for >= "
                f"{INCREASE_THRESHOLD}x discrepancy supported by qualified "
                "lower bound; uniform scale of the existing formula shape "
                "preferred; base+per-unit alternative reported where >= 2 "
                "distinct input magnitudes carry qualified evidence"
            ),
            "rounding": "round upward to two significant digits",
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "budget_lane": (
                "exact charged-gas accounting from executed diagnostic "
                "records: other marginal gas = total marginal charged gas - "
                "current target charge; target budget need = max(current, "
                "ceil2sig(raw workload 95% upper bound at anchor - other "
                "marginal gas)); requires exact affinity of charged gas in "
                "the target count, else abstains. Raw workload estimates "
                "are never relabeled as isolated prices."
            ),
        },
        "workload": {
            "schema_version": workload.get("schema_version"),
            "fork": workload.get("fork"),
            "generator": workload.get("generator"),
        },
        "analysis": {
            "directory": str(analysis_dir),
            "evm_gasfit_version": analysis_status.get("evm_gasfit_version"),
            "manifest_sha256": (analysis_status.get("campaign", {}) or {}).get(
                "manifest_sha256"
            ),
        },
        "limitations": [
            {
                "id": "keccak-global-cache",
                "description": (
                    "evm2 enables alloy-primitives keccak-cache-global "
                    "(crates/cli/Cargo.toml:35); alloy v1.7.3's process-"
                    "global cache holds inputs of at most "
                    f"{KECCAK_CACHE_MAX_INPUT_BYTES} bytes "
                    "(MAX_INPUT_LEN = 128 - 32 - 1 - 8). Repeated "
                    "deterministic inputs of 1..87 bytes can be served warm "
                    "across samples within a worker session, so those "
                    "slopes are not worst-case evidence and never drive "
                    "increase candidates here."
                ),
                "sources": [
                    "alloy-rs/core v1.7.3 crates/primitives/src/utils/keccak_cache.rs",
                    "evm2 crates/evm2/src/interpreter/instructions/crypto.rs",
                ],
            },
            {
                "id": "envelope-not-joint-ci",
                "description": (
                    "Group candidates that combine several per-variant 95% "
                    "upper bounds are sampled-domain envelopes of "
                    "correlated point estimates, not joint 95% guarantees; "
                    "they are labeled as such wherever emitted."
                ),
            },
            {
                "id": "blake2f-shape-spread",
                "description": (
                    "BLAKE2F (and other base+per-unit groups) preserve "
                    "their formula shape under a uniform scale; the "
                    "reported overpricing spread quantifies how much that "
                    "conservative scale overcharges the cheapest sampled "
                    "inputs, and an evidence-supported base+per-unit "
                    "alternative is reported when >= 2 distinct input "
                    "magnitudes carry qualified evidence."
                ),
            },
        ],
        "resolved_inputs": resolved_annotations,
        "summary": summary,
        "pricing_groups": group_reports,
        "variants": rows,
    }


def _variant_row(
    info: VariantInfo,
    ev: VariantEvidence | None,
    budget: BudgetEvidence,
    budget_ev: VariantEvidence | None,
    anchor_rate: float,
) -> dict[str, Any]:
    blockers = _coverage_blockers(info, ev)
    row: dict[str, Any] = {
        "variant_id": info.variant_id,
        "family": info.family,
        "target_operation": info.target_operation,
        "status": info.status,
        "pricing_group": info.pricing_group,
        "group_kind": info.group_kind,
        "units_per_call": info.units,
        "input_bytes": info.input_bytes,
        "current_charge_gas": info.current_charge_gas,
        "model_param": ev.model_param if ev else None,
        "evidence_source_ids": _source_ids(info, ev, budget),
        "keccak_cache_affected": info.keccak_cache_affected,
        "reasons": list(info.reasons),
    }
    if info.group_kind == GROUP_KIND_UNSUPPORTED:
        row.update(
            {
                "qualification": None,
                "glue_coverage": None,
                "raw_ns": None,
                "raw_ci95_ns": None,
                "adjusted_ns": None,
                "adjusted_ci95_ns": None,
                "equivalent_gas_600M": None,
                "conservative_candidate_gas": None,
                "budget_aware": {
                    "decision": "unsupported",
                    "coverage": "unsupported variant",
                    "abstain_reason": "; ".join(info.reasons),
                },
                "budget_aware_candidate_gas": None,
                "policy_decision": "unsupported",
                "policy_reason": "; ".join(info.reasons),
                "missing_coverage": ["not executable under the campaign"],
            }
        )
        return row
    if info.group_kind == GROUP_KIND_REJECTION:
        row.update(
            {
                "qualification": ev.qualification_status if ev else None,
                "glue_coverage": _coverage_word(ev),
                "raw_ns": _ns(ev.raw_point_ms) if ev else None,
                "raw_ci95_ns": _ci(ev.raw_low_ms, ev.raw_high_ms) if ev else None,
                "adjusted_ns": None,
                "adjusted_ci95_ns": None,
                "equivalent_gas_600M": None,
                "conservative_candidate_gas": None,
                "budget_aware": {
                    "decision": "not-priced-rejection-path",
                    "coverage": "unsupported pricing shape",
                    "abstain_reason": (
                        "rejection path is not a charged pricing workload"
                    ),
                },
                "budget_aware_candidate_gas": None,
                "policy_decision": "not-priced-rejection-path",
                "policy_reason": (
                    "EIP-7823 oversized-operand rejection halts before "
                    "charging; the rejection path is not a priced input and "
                    "never contributes to worst-case coverage"
                ),
                "missing_coverage": [],
            }
        )
        return row

    def gas(ms: float | None) -> float | None:
        return None if ms is None else gas_from_ns(ns_from_ms(ms), anchor_rate)

    lower_gas = gas(ev.adjusted_low_ms) if ev else None
    upper_gas = gas(ev.adjusted_high_ms) if ev else None
    point_gas = gas(ev.adjusted_point_ms) if ev else None
    ratio = (
        lower_gas / info.current_charge_gas
        if lower_gas is not None
        and info.current_charge_gas
        and not info.keccak_cache_affected
        else None
    )
    candidate = None
    if upper_gas is not None and info.current_charge_gas:
        candidate = max(float(info.current_charge_gas), ceil_2_significant(upper_gas))
    budget_need = _budget_need_gas(info, budget, budget_ev, anchor_rate)
    budget_ratio: float | None = None
    if (
        budget_need is not None
        and budget_ev is not None
        and budget_ev.raw_low_ms is not None
        and budget.other_marginal_gas is not None
        and info.current_charge_gas
    ):
        budget_ratio = (
            gas_from_ns(ns_from_ms(budget_ev.raw_low_ms), anchor_rate)
            - float(budget.other_marginal_gas)
        ) / info.current_charge_gas
    if info.current_charge_gas is None:
        decision = "blocked-coverage"
        reason = "; ".join(blockers) or "current charge unresolved"
    elif blockers:
        decision = "blocked-coverage"
        reason = "; ".join(blockers)
    elif ratio is not None and ratio >= INCREASE_THRESHOLD:
        decision = "increase-candidate-evidence"
        reason = (
            f"qualified adjusted lower bound {lower_gas:.1f} gas = "
            f"{ratio:.2f}x current {info.current_charge_gas}"
        )
    else:
        decision = "evidence-holds-current"
        reason = (
            "qualified adjusted upper bound "
            f"{upper_gas:.1f} gas <= 2-sig candidate {candidate:.1f}; "
            "current charge not contradicted at the 95% adjusted upper "
            "bound (keep; adequacy for pricing is a group-level judgment)"
        )
    row.update(
        {
            "qualification": ev.qualification_status if ev else None,
            "adjusted_estimate_status": ev.adjusted_estimate_status if ev else None,
            "glue_coverage": _coverage_word(ev),
            "glue_priced_opcodes": ev.glue_priced if ev else "",
            "glue_unpriced_opcodes": ev.glue_unpriced if ev else "",
            "glue_interval_conditional": (ev.glue_interval_conditional if ev else None),
            "raw_ns": _ns(ev.raw_point_ms) if ev else None,
            "raw_ci95_ns": _ci(ev.raw_low_ms, ev.raw_high_ms) if ev else None,
            "adjusted_ns": _ns(ev.adjusted_point_ms) if ev else None,
            "adjusted_ci95_ns": _ci(ev.adjusted_low_ms, ev.adjusted_high_ms)
            if ev
            else None,
            "equivalent_gas_600M": point_gas,
            "adjusted_lower_gas": lower_gas,
            "adjusted_upper_gas": upper_gas,
            "lower_bound_ratio_vs_current": ratio,
            "conservative_candidate_gas": candidate,
            "budget_aware": {
                "affine_charged_gas": budget.affine,
                "counts": budget.counts,
                "charged_gas": budget.charged_gas,
                "total_marginal_gas": (
                    None
                    if budget.total_marginal_gas is None
                    else float(budget.total_marginal_gas)
                ),
                "other_marginal_gas": (
                    None
                    if budget.other_marginal_gas is None
                    else float(budget.other_marginal_gas)
                ),
                "target_budget_need_gas": budget_need,
                "raw_lower_residual_ratio_vs_current": budget_ratio,
                "increase_trigger": (
                    budget_ratio is not None and budget_ratio >= INCREASE_THRESHOLD
                ),
                "threshold_sensitivity": {
                    f"{threshold}x": (
                        budget_ratio is not None and budget_ratio >= threshold
                    )
                    for threshold in SENSITIVITY_THRESHOLDS
                },
                "coverage": "qualified exact-affine raw evidence"
                if budget_need is not None
                else "inconclusive: " + (budget.abstain_reason or "missing evidence"),
                "abstain_reason": budget.abstain_reason,
                "conditions": (
                    "raw workload lower/upper bounds at the fixed anchor; "
                    "exact charged-gas residual; separate from adjusted "
                    "isolated evidence"
                ),
                "sample_ids": budget.sample_ids,
            },
            "budget_aware_candidate_gas": budget_need,
            "policy_decision": decision,
            "policy_reason": reason,
            "missing_coverage": blockers,
        }
    )
    return row


def _source_ids(
    info: VariantInfo, ev: VariantEvidence | None, budget: BudgetEvidence
) -> list[str]:
    ids: list[str] = []
    if ev and ev.model_param:
        ids.append(f"gas_param:{ev.model_param}")
    if ev and ev.source_label:
        ids.append(f"qualification:{ev.source_label}")
    if budget.sample_ids:
        ids.append(f"diagnostics:{budget.sample_ids[0]}..{budget.sample_ids[-1]}")
    if info.case_ids:
        ids.append(f"workload:{info.case_ids[0]}")
    return ids


def _ns(ms: float | None) -> float | None:
    return None if ms is None else ns_from_ms(ms)


def _ci(low_ms: float | None, high_ms: float | None) -> list[float] | None:
    if low_ms is None and high_ms is None:
        return None
    return [
        ns_from_ms(low_ms) if low_ms is not None else None,
        ns_from_ms(high_ms) if high_ms is not None else None,
    ]


def _group_report(
    group: str,
    group_infos: Sequence[VariantInfo],
    infos: Mapping[str, VariantInfo],
    evidence: Mapping[str, VariantEvidence],
    diagnostics: Mapping[str, list[Mapping[str, Any]]],
    anchor_rate: float,
) -> dict[str, Any]:
    ready = [i for i in group_infos if i.status == "ready"]
    contributing: list[tuple[VariantInfo, VariantEvidence | None, float | None]] = []
    blockers: dict[str, list[str]] = {}
    for info in group_infos:
        if info.status != "ready":
            blockers[info.variant_id] = [
                "selected variant unsupported or inconclusive: "
                + ("; ".join(info.reasons) or "unsupported")
            ]
    for info in ready:
        ev = evidence.get(info.variant_id)
        blockers[info.variant_id] = _coverage_blockers(info, ev)
        upper = (
            gas_from_ns(ns_from_ms(ev.adjusted_high_ms), anchor_rate)
            if ev and ev.adjusted_high_ms is not None
            else None
        )
        contributing.append((info, ev, upper))
    usable = [
        (info, ev, upper)
        for info, ev, upper in contributing
        if not blockers[info.variant_id] and info.current_charge_gas
    ]
    # Cached-path variants cannot support worst-case evidence. They remain
    # explicit group coverage blockers, while uncached evidence can still
    # produce a clearly evidence-supported (non-deployable) candidate.
    cached = [info.variant_id for info in ready if info.keccak_cache_affected]
    missing = {
        variant_id: reasons for variant_id, reasons in blockers.items() if reasons
    }

    decision: str
    reason: str
    deployable = not missing and bool(usable)
    increase = False
    if group in ("MODEXP_INPUT_REJECTION",):
        decision = "not-priced"
        reason = "input-size rejection path charges 0 gas (EIP-7823)"
        deployable = False
    elif not ready:
        decision = "unsupported"
        reason = "no ready variants"
        deployable = False
    elif not usable:
        decision = "blocked-coverage"
        reason = "no variant carries deployable qualified+covered evidence"
        deployable = False
    else:
        increase = _trigger_increase(usable, anchor_rate, INCREASE_THRESHOLD)
        all_adequate = all(
            upper is not None
            and info.current_charge_gas is not None
            and upper <= info.current_charge_gas
            for info, _ev, upper in usable
        ) and len(usable) + len(cached) == len(ready)
        if increase:
            decision = "increase_candidate" if deployable else "blocked-coverage"
            reason = (
                "qualified adjusted lower bound >= "
                f"{INCREASE_THRESHOLD}x current for at least one variant"
                + (
                    ""
                    if deployable
                    else "; coverage unresolved, candidate is evidence-supported but not deployable"
                )
            )
        elif all_adequate:
            decision = "keep_proven_adequate" if deployable else "blocked-coverage"
            reason = (
                "every uncached variant's qualified adjusted 95% upper "
                "bound is <= its current charge: proven adequate at the "
                "anchor over the sampled uncached domain"
                + ("" if deployable else "; coverage unresolved")
            )
        else:
            decision = "keep_by_policy" if deployable else "blocked-coverage"
            reason = (
                "no >= 2x qualified lower-bound discrepancy; current charge "
                "kept by policy (not a proven 600M adequacy claim)"
                + ("" if deployable else "; coverage unresolved")
            )
    if cached:
        reason += (
            f"; {len(cached)} short-input variant(s) (1..87 bytes) are "
            "cache-affected in this worker and excluded from worst-case "
            "evidence; coverage remains unresolved"
        )

    report: dict[str, Any] = {
        "pricing_group": group,
        "kind": ready[0].group_kind if ready else GROUP_KIND_UNSUPPORTED,
        "variants": [i.variant_id for i in group_infos],
        "ready_variants": [i.variant_id for i in ready],
        "decision": decision,
        "deployable": deployable,
        "reason": reason,
        "missing_coverage": missing,
        "inherent_keccak_cache_variants": cached,
        "increase_trigger": increase,
        "sensitivity_triggers": {
            f"{threshold}x": _trigger_increase(usable, anchor_rate, threshold)
            for threshold in SENSITIVITY_THRESHOLDS
        },
    }

    # A positive upper bound at zero current charge cannot be covered by a
    # finite uniform scale of a base+per-unit shape.
    zero_shape = [
        info.variant_id
        for info, _ev, upper in contributing
        if not blockers[info.variant_id]
        and info.current_charge_gas == 0
        and upper is not None
        and upper > 0
    ]
    if zero_shape:
        report["isolated_candidate"] = {
            "abstained": True,
            "unsupported_shape": "zero-charge linear variant",
            "unsupported_variants": zero_shape,
            "reason": "uniform scale cannot cover positive cost at zero charge",
        }
    elif usable and ready[0].group_kind == GROUP_KIND_FIXED:
        report["isolated_candidate"] = _group_candidate_fixed(
            usable, int(ready[0].current_charge_gas or 0)
        )
    elif usable:
        report["isolated_candidate"] = _group_candidate_scaled(usable, anchor_rate)
        info0 = ready[0]
        if info0.group_kind == GROUP_KIND_LINEAR and info0.linear_params:
            base_param, base_gas, per_param, per_gas = info0.linear_params
            alternative = _linear_alternative(
                usable, anchor_rate, base_param, base_gas, per_param, per_gas
            )
            if alternative is not None:
                report["isolated_candidate"]["base_per_unit_alternative"] = alternative

    isolated_candidate = report.get("isolated_candidate")
    if isinstance(isolated_candidate, dict) and isolated_candidate.get("abstained"):
        report["decision"] = "blocked-shape"
        report["deployable"] = False
        report["reason"] = (
            reason
            + "; formula candidate abstained: "
            + str(isolated_candidate.get("reason", "unsupported shape"))
        )

    # Budget-aware lane candidates. This lane is intentionally separate from
    # the isolated adjusted-estimate lane.
    budget_rows: list[tuple[VariantInfo, float]] = []
    budget_meta: dict[str, dict[str, Any]] = {}
    for info in group_infos:
        if info.status != "ready":
            budget_meta[info.variant_id] = {
                "eligible": False,
                "need_gas": None,
                "increase_trigger": False,
                "threshold_sensitivity": {
                    f"{threshold}x": False for threshold in SENSITIVITY_THRESHOLDS
                },
                "coverage": "unsupported variant: "
                + ("; ".join(info.reasons) or "unsupported"),
            }
    for info in ready:
        ev = evidence.get(info.variant_id)
        budget = (
            _budget_evidence(info, diagnostics)
            if diagnostics
            else BudgetEvidence(abstain_reason="diagnostics not supplied")
        )
        need = _budget_need_gas(info, budget, ev, anchor_rate)
        trigger = False
        if need is not None:
            if info.current_charge_gas:
                assert ev is not None and ev.raw_low_ms is not None
                assert budget.other_marginal_gas is not None
                lower_residual = gas_from_ns(
                    ns_from_ms(ev.raw_low_ms), anchor_rate
                ) - float(budget.other_marginal_gas)
                trigger = lower_residual / info.current_charge_gas >= INCREASE_THRESHOLD
            budget_rows.append((info, need))
        budget_meta[info.variant_id] = {
            "eligible": need is not None,
            "need_gas": need,
            "increase_trigger": trigger,
            "threshold_sensitivity": {
                f"{threshold}x": (
                    False
                    if need is None or not info.current_charge_gas
                    else (
                        gas_from_ns(ns_from_ms(ev.raw_low_ms), anchor_rate)
                        - float(budget.other_marginal_gas)
                    )
                    / info.current_charge_gas
                    >= threshold
                )
                for threshold in SENSITIVITY_THRESHOLDS
            },
            "coverage": "qualified exact-affine raw evidence"
            if need is not None
            else "inconclusive: " + (budget.abstain_reason or "missing evidence"),
        }
    budget_coverage_complete = bool(ready) and all(
        item["eligible"] for item in budget_meta.values()
    )
    budget_increase = any(item["increase_trigger"] for item in budget_meta.values())
    report.update(
        {
            "budget_coverage": {
                "complete": budget_coverage_complete,
                "per_variant": budget_meta,
            },
            "budget_increase_trigger": budget_increase,
            "budget_sensitivity_triggers": {
                f"{threshold}x": any(
                    item["threshold_sensitivity"][f"{threshold}x"]
                    for item in budget_meta.values()
                )
                for threshold in SENSITIVITY_THRESHOLDS
            },
            "budget_decision": (
                "increase_candidate"
                if budget_increase and budget_coverage_complete
                else "blocked-coverage"
                if budget_increase
                else "keep_by_policy"
                if budget_coverage_complete
                else "blocked-coverage"
            ),
        }
    )

    if group == "MODEXP_INPUT_REJECTION":
        report["budget_decision"] = "not-priced"
    if budget_rows and ready[0].group_kind != GROUP_KIND_REJECTION:
        if ready[0].group_kind == GROUP_KIND_FIXED:
            value = max(need for _i, need in budget_rows)
            report["budget_aware_candidate"] = {
                "value": ceil_int(value),
                "exact_2sig": value,
                "variants": {i.variant_id: need for i, need in budget_rows},
                "conditions": (
                    "target budget need uses the qualified raw 95% upper "
                    "bound at the fixed anchor minus other marginal gas"
                ),
            }
        elif ready[0].group_kind == GROUP_KIND_MODEXP:
            report["budget_aware_candidate"] = _group_candidate_modexp(
                [(info, None, need) for info, need in budget_rows],
                anchor_rate,
            )
        elif ready[0].group_kind == GROUP_KIND_LINEAR and ready[0].linear_params:
            base_param, base_gas, per_param, per_gas = ready[0].linear_params
            zero_shape = [
                info.variant_id
                for info, need in budget_rows
                if info.current_charge_gas == 0 and need > 0
            ]
            if zero_shape:
                report["budget_aware_candidate"] = {
                    "abstained": True,
                    "unsupported_shape": "round-only zero-charge variant",
                    "variants": {i.variant_id: need for i, need in budget_rows},
                    "reason": (
                        "round-only base+per-unit formula cannot cover a "
                        "positive budget need at zero current charge"
                    ),
                    "unsupported_variants": zero_shape,
                }
            else:
                scales = [
                    need / info.current_charge_gas
                    for info, need in budget_rows
                    if info.current_charge_gas
                ]
                scale = max(1.0, max(scales, default=1.0))
                scale = ceil_2_significant(scale)
                scaled_base = ceil_2_significant(base_gas * scale)
                scaled_per = ceil_2_significant(per_gas * scale)
                report["budget_aware_candidate"] = {
                    "uniform_scale": scale,
                    "params": {
                        base_param: ceil_int(scaled_base),
                        per_param: ceil_int(scaled_per),
                    },
                    "exact_2sig": {
                        base_param: scaled_base,
                        per_param: scaled_per,
                    },
                    "variants": {i.variant_id: need for i, need in budget_rows},
                    "conditions": (
                        "uniform scale preserves the existing base+per-unit "
                        "formula; both coefficients are rounded upward to "
                        "two significant digits"
                    ),
                }
        else:
            scales = [
                need / info.current_charge_gas
                for info, need in budget_rows
                if info.current_charge_gas
            ]
            scale = ceil_2_significant(max(1.0, max(scales, default=1.0)))
            report["budget_aware_candidate"] = {
                "uniform_scale": scale,
                "variants": {i.variant_id: need for i, need in budget_rows},
                "conditions": (
                    "uniform scale preserves the existing formula shape; "
                    "raw slope used, never relabeled isolated"
                ),
            }
    elif ready and ready[0].group_kind not in (
        GROUP_KIND_REJECTION,
        GROUP_KIND_UNSUPPORTED,
    ):
        report["budget_aware_candidate"] = {
            "abstained": True,
            "reason": "no variant satisfied budget-lane qualification, "
            "affinity, current-charge, and cache-risk conditions",
        }

    budget_candidate = report.get("budget_aware_candidate")
    if (
        isinstance(budget_candidate, dict)
        and budget_candidate.get("abstained")
        and budget_candidate.get("unsupported_shape")
    ):
        report["budget_decision"] = "blocked-shape"
    return report


def _trigger_increase(
    usable: Sequence[tuple[VariantInfo, VariantEvidence | None, float | None]],
    anchor_rate: float,
    threshold: float,
) -> bool:
    for info, ev, _upper in usable:
        if ev is None or ev.adjusted_low_ms is None:
            continue
        if not info.current_charge_gas:
            continue
        lower_gas = gas_from_ns(ns_from_ms(ev.adjusted_low_ms), anchor_rate)
        if lower_gas / info.current_charge_gas >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# build: outputs and CLI.
# ---------------------------------------------------------------------------
_CSV_COLUMNS = [
    "variant_id",
    "family",
    "target_operation",
    "status",
    "pricing_group",
    "group_kind",
    "units_per_call",
    "input_bytes",
    "current_charge_gas",
    "qualification",
    "adjusted_estimate_status",
    "glue_coverage",
    "raw_ns",
    "raw_ci95_low_ns",
    "raw_ci95_high_ns",
    "adjusted_ns",
    "adjusted_ci95_low_ns",
    "adjusted_ci95_high_ns",
    "equivalent_gas_600M",
    "adjusted_lower_gas",
    "adjusted_upper_gas",
    "lower_bound_ratio_vs_current",
    "conservative_candidate_gas",
    "budget_aware_candidate_gas",
    "policy_decision",
    "policy_reason",
    "evidence_source_ids",
    "missing_coverage",
    "keccak_cache_affected",
]


def _write_csv(document: Mapping[str, Any], path: Path) -> None:
    def cell(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, list):
            return ";".join(str(v) for v in value)
        return value

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_COLUMNS)
        for row in document["variants"]:
            raw_ci = row.get("raw_ci95_ns") or [None, None]
            adj_ci = row.get("adjusted_ci95_ns") or [None, None]
            record = {
                **row,
                "raw_ci95_low_ns": raw_ci[0],
                "raw_ci95_high_ns": raw_ci[1],
                "adjusted_ci95_low_ns": adj_ci[0],
                "adjusted_ci95_high_ns": adj_ci[1],
            }
            writer.writerow([cell(record.get(col)) for col in _CSV_COLUMNS])


def cmd_build(args: argparse.Namespace) -> int:
    workload = load_workload(Path(args.workload))
    resolved_inputs = None
    if args.resolved_inputs:
        resolved_inputs = json.loads(Path(args.resolved_inputs).read_text())
        if not isinstance(resolved_inputs, dict):
            raise BuildError("resolved-input sidecar must contain a JSON object")
    document = build_recommendations(
        workload,
        Path(args.analysis),
        Path(args.diagnostics) if args.diagnostics else None,
        anchor_rate=args.anchor_rate,
        resolved_inputs=resolved_inputs,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "recommendations.json"
    csv_path = out_dir / "recommendations.csv"
    if json_path.exists() or csv_path.exists():
        raise BuildError(
            f"{json_path} or {csv_path} already exists; write to a fresh directory"
        )
    json_path.write_text(json.dumps(document, indent=2) + "\n")
    _write_csv(document, csv_path)
    summary = document["summary"]
    print(
        f"wrote {json_path} and {csv_path}: "
        f"{summary['variants_total']} variants "
        f"({summary['variants_unsupported']} unsupported), "
        f"{summary['groups_total']} pricing groups "
        f"({summary['groups_increase_candidate']} increase candidates, "
        f"{summary['groups_blocked']} blocked)"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evm_gasfit.recommendations",
        description=(
            "Osaka compute pricing: per-variant glue-enabled analysis config "
            "generation and evidence-linked minimal-change recommendations"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser(
        "create-config",
        help="Create the per-variant glue-enabled 600M analysis config",
    )
    create.add_argument("--workload", required=True, type=Path)
    create.add_argument(
        "--client",
        required=True,
        help="Engine whose runtimes are analyzed (the runtimes CSV client_name)",
    )
    create.add_argument("--out", required=True, type=Path)
    create.add_argument("--anchor-rate", type=float, default=DEFAULT_ANCHOR_RATE)
    create.add_argument("--min-sessions", type=int, default=MIN_SESSIONS_DEFAULT)
    create.set_defaults(func=cmd_create_config)

    build = sub.add_parser(
        "build",
        help="Build recommendation JSON+CSV from a completed analysis",
    )
    build.add_argument("--workload", required=True, type=Path)
    build.add_argument("--analysis", required=True, type=Path)
    build.add_argument("--diagnostics", required=False, type=Path, default=None)
    build.add_argument(
        "--resolved-inputs",
        required=False,
        type=Path,
        default=None,
        help="reviewed resolved-input sidecar for optimized linear variants",
    )
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--anchor-rate", type=float, default=DEFAULT_ANCHOR_RATE)
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (exit codes 0/1/2 mirroring ``evm-gasfit run``)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (WorkloadError, BuildError) as exc:
        _log.error("%s", exc)
        return 1
    except OSError as exc:
        _log.error("io error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
