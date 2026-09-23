"""Bundled :class:`ModelSpec` presets.

A preset is a frozen, named model recipe that users can list under
``models.presets`` in their YAML config to avoid copy-pasting the full spec.
Selecting a preset is equivalent to pasting its literal into ``models.custom``.
"""

from __future__ import annotations

from evm_gasfit.config import FixtureParamSpec, ModelSpec


# Helper: per-word size fixture-param via ``ceil(x / 32)``.
def _bytes_to_words(source: str) -> FixtureParamSpec:
    return FixtureParamSpec(source=source, transform="bytes_to_words")


PRESETS: dict[str, ModelSpec] = {
    # -------------------------------------------------------------------
    # Arithmetic (15)
    # -------------------------------------------------------------------
    "arithmetic_add": ModelSpec(
        test_name="test_arithmetic",
        target_operation="ADD",
        filter_by=["opcode_ADD-"],
        model_params={"target_coef": "OPCODE_ADD"},
    ),
    "arithmetic_sub": ModelSpec(
        test_name="test_arithmetic",
        target_operation="SUB",
        filter_by=["opcode_SUB"],
        model_params={"target_coef": "OPCODE_SUB"},
    ),
    "arithmetic_mul": ModelSpec(
        test_name="test_arithmetic",
        target_operation="MUL",
        filter_by=["opcode_MUL-"],
        model_params={"target_coef": "OPCODE_MUL"},
    ),
    "arithmetic_div": ModelSpec(
        test_name="test_arithmetic",
        target_operation="DIV",
        filter_by=["opcode_DIV"],
        model_params={"target_coef": "OPCODE_DIV"},
    ),
    "arithmetic_sdiv": ModelSpec(
        test_name="test_arithmetic",
        target_operation="SDIV",
        filter_by=["opcode_SDIV"],
        model_params={"target_coef": "OPCODE_SDIV"},
    ),
    "arithmetic_signextend": ModelSpec(
        test_name="test_arithmetic",
        target_operation="SIGNEXTEND",
        filter_by=["opcode_SIGNEXTEND"],
        model_params={"target_coef": "OPCODE_SIGNEXTEND"},
    ),
    "arithmetic_exp": ModelSpec(
        test_name="test_arithmetic",
        target_operation="EXP",
        filter_by=["opcode_EXP"],
        model_params={"target_coef": "OPCODE_EXP_BASE"},
    ),
    "arithmetic_mod": ModelSpec(
        test_name="test_arithmetic",
        target_operation="MOD",
        filter_by=["opcode_MOD"],
        model_params={"target_coef": "OPCODE_MOD"},
    ),
    "arithmetic_mod_bits": ModelSpec(
        test_name="test_mod",
        target_operation="MOD",
        filter_by=["opcode_MOD"],
        model_by=["mod_bits"],
        model_params={"target_coef": "OPCODE_MOD"},
    ),
    "arithmetic_smod": ModelSpec(
        test_name="test_arithmetic",
        target_operation="SMOD",
        filter_by=["opcode_SMOD"],
        model_params={"target_coef": "OPCODE_SMOD"},
    ),
    "arithmetic_smod_bits": ModelSpec(
        test_name="test_mod",
        target_operation="SMOD",
        filter_by=["opcode_SMOD"],
        model_by=["mod_bits"],
        model_params={"target_coef": "OPCODE_SMOD"},
    ),
    "arithmetic_addmod": ModelSpec(
        test_name="test_arithmetic",
        target_operation="ADDMOD",
        filter_by=["opcode_ADDMOD"],
        model_params={"target_coef": "OPCODE_ADDMOD"},
    ),
    "arithmetic_addmod_bits": ModelSpec(
        test_name="test_mod_arithmetic",
        target_operation="ADDMOD",
        filter_by=["opcode_ADDMOD"],
        model_by=["mod_bits"],
        model_params={"target_coef": "OPCODE_ADDMOD"},
    ),
    "arithmetic_mulmod": ModelSpec(
        test_name="test_arithmetic",
        target_operation="MULMOD",
        filter_by=["opcode_MULMOD"],
        model_params={"target_coef": "OPCODE_MULMOD"},
    ),
    "arithmetic_mulmod_bits": ModelSpec(
        test_name="test_mod_arithmetic",
        target_operation="MULMOD",
        filter_by=["opcode_MULMOD"],
        model_by=["mod_bits"],
        model_params={"target_coef": "OPCODE_MULMOD"},
    ),
    # -------------------------------------------------------------------
    # Bitwise (9)
    # -------------------------------------------------------------------
    "bitwise_and": ModelSpec(
        test_name="test_bitwise",
        target_operation="AND",
        filter_by=["opcode_AND"],
        model_params={"target_coef": "OPCODE_AND"},
    ),
    "bitwise_or": ModelSpec(
        test_name="test_bitwise",
        target_operation="OR",
        filter_by=["opcode_OR"],
        model_params={"target_coef": "OPCODE_OR"},
    ),
    "bitwise_xor": ModelSpec(
        test_name="test_bitwise",
        target_operation="XOR",
        filter_by=["opcode_XOR"],
        model_params={"target_coef": "OPCODE_XOR"},
    ),
    "bitwise_byte": ModelSpec(
        test_name="test_bitwise",
        target_operation="BYTE",
        filter_by=["opcode_BYTE"],
        model_params={"target_coef": "OPCODE_BYTE"},
    ),
    "bitwise_shl": ModelSpec(
        test_name="test_bitwise",
        target_operation="SHL",
        filter_by=["opcode_SHL"],
        model_params={"target_coef": "OPCODE_SHL"},
    ),
    "bitwise_shr": ModelSpec(
        test_name="test_bitwise",
        target_operation="SHR",
        filter_by=["opcode_SHR"],
        model_params={"target_coef": "OPCODE_SHR"},
    ),
    "bitwise_sar": ModelSpec(
        test_name="test_bitwise",
        target_operation="SAR",
        filter_by=["opcode_SAR"],
        model_params={"target_coef": "OPCODE_SAR"},
    ),
    "bitwise_not": ModelSpec(
        test_name="test_not_op",
        target_operation="NOT",
        model_params={"target_coef": "OPCODE_NOT"},
    ),
    "bitwise_clz": ModelSpec(
        test_name="test_clz_same",
        target_operation="CLZ",
        model_params={"target_coef": "OPCODE_CLZ"},
    ),
    # -------------------------------------------------------------------
    # Comparison (6)
    # -------------------------------------------------------------------
    "comparison_lt": ModelSpec(
        test_name="test_comparison",
        target_operation="LT",
        filter_by=["opcode_LT"],
        model_params={"target_coef": "OPCODE_LT"},
    ),
    "comparison_gt": ModelSpec(
        test_name="test_comparison",
        target_operation="GT",
        filter_by=["opcode_GT"],
        model_params={"target_coef": "OPCODE_GT"},
    ),
    "comparison_slt": ModelSpec(
        test_name="test_comparison",
        target_operation="SLT",
        filter_by=["opcode_SLT"],
        model_params={"target_coef": "OPCODE_SLT"},
    ),
    "comparison_sgt": ModelSpec(
        test_name="test_comparison",
        target_operation="SGT",
        filter_by=["opcode_SGT"],
        model_params={"target_coef": "OPCODE_SGT"},
    ),
    "comparison_eq": ModelSpec(
        test_name="test_comparison",
        target_operation="EQ",
        filter_by=["opcode_EQ"],
        model_params={"target_coef": "OPCODE_EQ"},
    ),
    "comparison_iszero": ModelSpec(
        test_name="test_iszero",
        target_operation="ISZERO",
        model_params={"target_coef": "OPCODE_ISZERO"},
    ),
    # -------------------------------------------------------------------
    # Stack (4)
    # -------------------------------------------------------------------
    "stack_push0": ModelSpec(
        test_name="test_push",
        target_operation="PUSH0",
        filter_by=["opcode_PUSH0-"],
        model_params={"target_coef": "OPCODE_PUSH0"},
    ),
    "stack_push": ModelSpec(
        test_name="test_push",
        target_operation_param="opcode",
        filter_by=["opcode_PUSH"],
        model_by=["opcode"],
        model_params={"target_coef": "OPCODE_PUSH"},
    ),
    "stack_dup": ModelSpec(
        test_name="test_dup",
        target_operation_param="opcode",
        model_by=["opcode"],
        model_params={"target_coef": "OPCODE_DUP"},
    ),
    "stack_swap": ModelSpec(
        test_name="test_swap",
        target_operation_param="opcode",
        model_by=["opcode"],
        model_params={"target_coef": "OPCODE_SWAP"},
    ),
    # -------------------------------------------------------------------
    # Control flow (5)
    # -------------------------------------------------------------------
    "control_flow_jump": ModelSpec(
        test_name="test_jump_benchmark",
        target_operation="JUMP",
        model_params={"target_coef": "OPCODE_JUMP"},
    ),
    "control_flow_jumpi": ModelSpec(
        test_name="test_jumpi_fallthrough",
        target_operation="JUMPI",
        model_params={"target_coef": "OPCODE_JUMPI"},
    ),
    "control_flow_jumpdest": ModelSpec(
        test_name="test_jumpdests",
        target_operation="JUMPDEST",
        model_params={"target_coef": "OPCODE_JUMPDEST"},
    ),
    "control_flow_pc": ModelSpec(
        test_name="test_pc_op",
        target_operation="PC",
        model_params={"target_coef": "OPCODE_PC"},
    ),
    "control_flow_gas": ModelSpec(
        test_name="test_gas_op",
        target_operation="GAS",
        model_params={"target_coef": "OPCODE_GAS"},
    ),
    # -------------------------------------------------------------------
    # Block & TX context (11)
    # -------------------------------------------------------------------
    "block_basefee": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="BASEFEE",
        filter_by=["opcode_BASEFEE"],
        model_params={"target_coef": "OPCODE_BASEFEE"},
    ),
    "block_blobbasefee": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="BLOBBASEFEE",
        filter_by=["opcode_BLOBBASEFEE"],
        model_params={"target_coef": "OPCODE_BLOBBASEFEE"},
    ),
    "block_chainid": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="CHAINID",
        filter_by=["opcode_CHAINID"],
        model_params={"target_coef": "OPCODE_CHAINID"},
    ),
    "block_coinbase": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="COINBASE",
        filter_by=["opcode_COINBASE"],
        model_params={"target_coef": "OPCODE_COINBASE"},
    ),
    "block_gaslimit": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="GASLIMIT",
        filter_by=["opcode_GASLIMIT"],
        model_params={"target_coef": "OPCODE_GASLIMIT"},
    ),
    "block_number": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="NUMBER",
        filter_by=["opcode_NUMBER"],
        model_params={"target_coef": "OPCODE_NUMBER"},
    ),
    "block_prevrandao": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="PREVRANDAO",
        filter_by=["opcode_PREVRANDAO"],
        model_params={"target_coef": "OPCODE_PREVRANDAO"},
    ),
    "block_timestamp": ModelSpec(
        test_name="test_block_context_ops",
        target_operation="TIMESTAMP",
        filter_by=["opcode_TIMESTAMP"],
        model_params={"target_coef": "OPCODE_TIMESTAMP"},
    ),
    "block_blockhash": ModelSpec(
        test_name="test_blockhash",
        target_operation="BLOCKHASH",
        model_by=["block"],
        model_params={"target_coef": "OPCODE_BLOCKHASH"},
    ),
    "tx_gasprice": ModelSpec(
        test_name="test_call_frame_context_ops",
        target_operation="GASPRICE",
        filter_by=["opcode_GASPRICE"],
        model_params={"target_coef": "OPCODE_GASPRICE"},
    ),
    "tx_origin": ModelSpec(
        test_name="test_call_frame_context_ops",
        target_operation="ORIGIN",
        filter_by=["opcode_ORIGIN"],
        model_params={"target_coef": "OPCODE_ORIGIN"},
    ),
    # -------------------------------------------------------------------
    # Call context (8)
    # -------------------------------------------------------------------
    "call_address": ModelSpec(
        test_name="test_call_frame_context_ops",
        target_operation="ADDRESS",
        filter_by=["opcode_ADDRESS"],
        model_params={"target_coef": "OPCODE_ADDRESS"},
    ),
    "call_caller": ModelSpec(
        test_name="test_call_frame_context_ops",
        target_operation="CALLER",
        filter_by=["opcode_CALLER"],
        model_params={"target_coef": "OPCODE_CALLER"},
    ),
    "call_callvalue": ModelSpec(
        test_name="test_callvalue_from_origin",
        target_operation="CALLVALUE",
        model_params={"target_coef": "OPCODE_CALLVALUE"},
    ),
    "call_calldataload": ModelSpec(
        test_name="test_calldataload",
        target_operation="CALLDATALOAD",
        model_by=["calldata_size"],
        model_params={"target_coef": "OPCODE_CALLDATALOAD"},
    ),
    "call_calldatasize": ModelSpec(
        test_name="test_calldatasize",
        target_operation="CALLDATASIZE",
        model_by=["calldata_size"],
        model_params={"target_coef": "OPCODE_CALLDATASIZE"},
    ),
    "call_returndatasize": ModelSpec(
        test_name="test_returndatasize_nonzero",
        target_operation="RETURNDATASIZE",
        model_by=["returned_size"],
        model_params={"target_coef": "OPCODE_RETURNDATASIZE"},
    ),
    "call_calldatacopy": ModelSpec(
        test_name="test_calldatacopy_from_origin",
        target_operation="CALLDATACOPY",
        model_by=["calldata_size", "mem_size"],
        fixture_params={"calldata_words": _bytes_to_words("calldata_size")},
        model_params={
            "target_coef": "OPCODE_CALLDATACOPY_BASE",
            "calldata_words": "OPCODE_CALLDATACOPY_PER_WORD",
        },
    ),
    "call_returndatacopy": ModelSpec(
        test_name="test_returndatacopy",
        target_operation="RETURNDATACOPY",
        model_by=["return_size", "mem_size"],
        fixture_params={"return_words": _bytes_to_words("return_size")},
        model_params={
            "target_coef": "OPCODE_RETURNDATACOPY_BASE",
            "return_words": "OPCODE_RETURNDATACOPY_PER_WORD",
        },
    ),
    # -------------------------------------------------------------------
    # Memory (5)
    # -------------------------------------------------------------------
    "memory_mload": ModelSpec(
        test_name="test_memory_access",
        target_operation="MLOAD",
        filter_by=["opcode_MLOAD"],
        model_by=["mem_size"],
        model_params={"target_coef": "OPCODE_MLOAD_BASE"},
    ),
    "memory_mstore": ModelSpec(
        test_name="test_memory_access",
        target_operation="MSTORE",
        filter_by=["opcode_MSTORE-"],
        model_by=["mem_size"],
        model_params={"target_coef": "OPCODE_MSTORE_BASE"},
    ),
    "memory_mstore8": ModelSpec(
        test_name="test_memory_access",
        target_operation="MSTORE8",
        filter_by=["opcode_MSTORE8"],
        model_by=["mem_size"],
        model_params={"target_coef": "OPCODE_MSTORE8_BASE"},
    ),
    "memory_msize": ModelSpec(
        test_name="test_msize",
        target_operation="MSIZE",
        model_params={"target_coef": "OPCODE_MSIZE"},
    ),
    "memory_mcopy": ModelSpec(
        test_name="test_mcopy",
        target_operation="MCOPY",
        model_by=["copy_size", "mem_size"],
        fixture_params={"copy_words": _bytes_to_words("copy_size")},
        model_params={
            "target_coef": "OPCODE_MCOPY_BASE",
            "copy_words": "OPCODE_MCOPY_PER_WORD",
        },
    ),
    # -------------------------------------------------------------------
    # Account / storage / state (12)
    # -------------------------------------------------------------------
    "warm_storage_access_sload": ModelSpec(
        test_name="test_sload_same_key_benchmark",
        target_operation="SLOAD",
        model_params={"target_coef": "WARM_ACCESS"},
    ),
    "warm_account_access": ModelSpec(
        test_name="test_ext_account_query_warm",
        target_operation_param="opcode",
        model_by=["opcode"],
        model_params={"target_coef": "WARM_ACCESS"},
    ),
    "cold_storage_sload": ModelSpec(
        test_name="test_sload_bloated",
        target_operation="SLOAD",
        filter_by=["CacheStrategy.NO_CACHE"],
        model_by=["existing_slots"],
        model_params={"target_coef": "COLD_STORAGE_ACCESS"},
    ),
    # SSTORE is split into a read-only access fit (write_new_value=False) and a
    # combined access+write fit (write_new_value=True). The write *delta* is
    # recovered downstream via a derived ``STORAGE_WRITE = max(0,
    # COLD_STORAGE_WRITE - COLD_STORAGE_ACCESS)`` so the combined cold-write op
    # is bounded by a single worst-case client rather than the sum of two
    # independent per-param maxima.
    "cold_storage_sstore_access": ModelSpec(
        test_name="test_sstore_bloated",
        target_operation="SSTORE",
        filter_by=["CacheStrategy.NO_CACHE", "write_new_value_False"],
        model_by=["existing_slots"],
        model_params={"target_coef": "COLD_STORAGE_ACCESS"},
    ),
    "cold_storage_sstore_write": ModelSpec(
        test_name="test_sstore_bloated",
        target_operation="SSTORE",
        filter_by=["CacheStrategy.NO_CACHE", "write_new_value_True"],
        model_by=["existing_slots"],
        model_params={"target_coef": "COLD_STORAGE_WRITE"},
    ),
    # Account access mirrors SSTORE: value_sent=0 isolates the cold access,
    # value_sent=1 measures the combined access+write, and ``ACCOUNT_WRITE`` is
    # derived as the worst write delta across the nocode and code contexts.
    #
    # The nocode presets (EOA / non-existing accounts) have no
    # ``overhead_baseline_True`` counterpart fixture at all — every one of
    # their fixtures is already ``overhead_baseline_False`` — so there is no
    # baseline variant to drop and no filter_by token for it; target_coef
    # relies on the existing per-opcode glue adjustment, same as any other
    # spec.
    "cold_account_nocode_access": ModelSpec(
        test_name="test_account_access",
        target_operation_param="opcode",
        filter_by=[
            "CacheStrategy.NO_CACHE",
            "!AccountMode.EXISTING_CONTRACT",
            "value_sent_0",
        ],
        model_by=["opcode", "account_mode"],
        model_params={"target_coef": "COLD_ACCOUNT_NOCODE_ACCESS"},
    ),
    "cold_account_nocode_write": ModelSpec(
        test_name="test_account_access",
        target_operation_param="opcode",
        filter_by=[
            "CacheStrategy.NO_CACHE",
            "!AccountMode.EXISTING_CONTRACT",
            "value_sent_1",
        ],
        model_by=["opcode", "account_mode"],
        model_params={"target_coef": "COLD_ACCOUNT_NOCODE_WRITE"},
    ),
    # The code presets (EXISTING_CONTRACT_* modes) DO have an
    # overhead_baseline_True counterpart for every fixture, for all 8 opcodes
    # (BALANCE/EXTCODEHASH/EXTCODESIZE and the CALL family) — the delta cancels
    # any contaminant with an identical count on both sides (keccak) while the
    # per-opcode glue detector, which now runs on the same delta rather than
    # skipping baseline-paired specs (glue/detect.py), still catches anything
    # that scales with the target's own opcount instead (the call family's
    # GAS/PUSH-family/POP calling-convention setup, which the True baseline
    # also drops). So a single preset covers the whole opcode universe.
    # NON_EXISTING_ACCOUNT is excluded because that mode has no
    # overhead_baseline_True counterpart at all — only the 4
    # EXISTING_CONTRACT_* modes do. EXISTING_CONTRACT_JUMPDEST is excluded too:
    # that mode points the target opcode at a contract whose code is a single
    # JUMPDEST, which is not representative of the cold account access being
    # priced and was winning the worst-case selection for several
    # (param, client) pairs, including COLD_ACCOUNT_CODE_ACCESS itself.
    "cold_account_code_access": ModelSpec(
        test_name="test_account_access",
        target_operation_param="opcode",
        filter_by=[
            "CacheStrategy.NO_CACHE",
            "!AccountMode.EXISTING_EOA",
            "!AccountMode.NON_EXISTING_ACCOUNT",
            "!AccountMode.EXISTING_CONTRACT_JUMPDEST",
            "value_sent_0",
        ],
        model_by=["opcode", "account_mode"],
        model_params={"target_coef": "COLD_ACCOUNT_CODE_ACCESS"},
        overhead_baseline_param="overhead_baseline",
    ),
    # No BALANCE/EXTCODEHASH/EXTCODESIZE side: those three never carry a value
    # (only CALL/CALLCODE do), so a `value_sent_1` slice restricted to them
    # always matches zero fixtures — COLD_ACCOUNT_CODE_WRITE is priced from
    # the call-family opcodes alone.
    "cold_account_code_write": ModelSpec(
        test_name="test_account_access",
        target_operation_param="opcode",
        filter_by=[
            "CacheStrategy.NO_CACHE",
            "!AccountMode.EXISTING_EOA",
            "!AccountMode.NON_EXISTING_ACCOUNT",
            "!AccountMode.EXISTING_CONTRACT_JUMPDEST",
            "value_sent_1",
            "!opcode_BALANCE",
            "!opcode_EXTCODEHASH",
            "!opcode_EXTCODESIZE",
        ],
        model_by=["opcode", "account_mode"],
        model_params={"target_coef": "COLD_ACCOUNT_CODE_WRITE"},
        overhead_baseline_param="overhead_baseline",
    ),
    "account_codecopy": ModelSpec(
        test_name="test_codecopy_benchmark",
        target_operation="CODECOPY",
        model_by=["code_size", "mem_size"],
        fixture_params={"code_words": _bytes_to_words("code_size")},
        model_params={
            "target_coef": "OPCODE_CODECOPY_BASE",
            "code_words": "OPCODE_CODECOPY_PER_WORD",
        },
    ),
    "account_codesize": ModelSpec(
        test_name="test_codesize",
        target_operation="CODESIZE",
        model_params={"target_coef": "OPCODE_CODESIZE"},
    ),
    "account_selfbalance": ModelSpec(
        test_name="test_selfbalance",
        target_operation="SELFBALANCE",
        model_params={"target_coef": "OPCODE_SELFBALANCE"},
    ),
    # -------------------------------------------------------------------
    # Storage tload/tstore (2)
    # -------------------------------------------------------------------
    "storage_tload": ModelSpec(
        test_name="test_tload",
        target_operation="TLOAD",
        model_params={"target_coef": "OPCODE_TLOAD"},
    ),
    "storage_tstore": ModelSpec(
        test_name="test_tstore",
        target_operation="TSTORE",
        model_params={"target_coef": "OPCODE_TSTORE"},
    ),
    # -------------------------------------------------------------------
    # Hashing (1)
    # -------------------------------------------------------------------
    "keccak": ModelSpec(
        test_name="test_keccak_diff_mem_msg_sizes",
        target_operation="KECCAK256",
        model_by=["mem_size"],
        fixture_params={"msg_words": _bytes_to_words("msg_size")},
        model_params={
            "target_coef": "OPCODE_KECCAK256_BASE",
            "msg_words": "OPCODE_KECCAK256_PER_WORD",
        },
    ),
    # -------------------------------------------------------------------
    # System (1)
    # -------------------------------------------------------------------
    "system_create": ModelSpec(
        test_name="test_create",
        target_operation_param="opcode",
        model_by=["opcode"],
        model_params={"target_coef": "OPCODE_CREATE_BASE"},
    ),
    # -------------------------------------------------------------------
    # Precompiles (19)
    # -------------------------------------------------------------------
    "precompile_ecrecover": ModelSpec(
        test_name="test_ecrecover",
        target_operation="ECRECOVER",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000001",
        filter_by=["ecrecover"],
        model_params={"target_coef": "PRECOMPILE_ECRECOVER"},
    ),
    "precompile_sha256_fixed": ModelSpec(
        test_name="test_sha256_fixed_size",
        target_operation="SHA256",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000002",
        filter_by=["sha256"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_SHA256_BASE",
            "size_words": "PRECOMPILE_SHA256_PER_WORD",
        },
    ),
    "precompile_sha256_uncachable": ModelSpec(
        test_name="test_sha256_uncachable",
        target_operation="SHA256",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000002",
        filter_by=["sha256"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_SHA256_BASE",
            "size_words": "PRECOMPILE_SHA256_PER_WORD",
        },
    ),
    "precompile_ripemd160_fixed": ModelSpec(
        test_name="test_ripemd160_fixed_size",
        target_operation="RIPEMD160",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000003",
        filter_by=["ripemd160"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_RIPEMD160_BASE",
            "size_words": "PRECOMPILE_RIPEMD160_PER_WORD",
        },
    ),
    "precompile_ripemd160_uncachable": ModelSpec(
        test_name="test_ripemd160_uncachable",
        target_operation="RIPEMD160",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000003",
        filter_by=["ripemd160"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_RIPEMD160_BASE",
            "size_words": "PRECOMPILE_RIPEMD160_PER_WORD",
        },
    ),
    "precompile_identity_fixed": ModelSpec(
        test_name="test_identity_fixed_size",
        target_operation="IDENTITY",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000004",
        filter_by=["identity"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_IDENTITY_BASE",
            "size_words": "PRECOMPILE_IDENTITY_PER_WORD",
        },
    ),
    "precompile_identity_uncachable": ModelSpec(
        test_name="test_identity_uncachable",
        target_operation="IDENTITY",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000004",
        filter_by=["identity"],
        fixture_params={"size_words": _bytes_to_words("size")},
        model_params={
            "target_coef": "PRECOMPILE_IDENTITY_BASE",
            "size_words": "PRECOMPILE_IDENTITY_PER_WORD",
        },
    ),
    "precompile_blake2f": ModelSpec(
        test_name="test_blake2f_benchmark",
        target_operation="BLAKE2F",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000009",
        filter_by=["blake2f"],
        model_params={
            "target_coef": "PRECOMPILE_BLAKE2F_BASE",
            "num_rounds": "PRECOMPILE_BLAKE2F_PER_ROUND",
        },
    ),
    "precompile_blake2f_uncachable": ModelSpec(
        test_name="test_blake2f_uncachable",
        target_operation="BLAKE2F",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000009",
        filter_by=["blake2f"],
        model_params={
            "target_coef": "PRECOMPILE_BLAKE2F_BASE",
            "num_rounds": "PRECOMPILE_BLAKE2F_PER_ROUND",
        },
    ),
    "precompile_p256verify": ModelSpec(
        test_name="test_p256verify",
        target_operation="P256VERIFY",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000100",
        filter_by=["p256verify"],
        model_params={"target_coef": "PRECOMPILE_P256VERIFY"},
    ),
    "precompile_p256verify_uncachable": ModelSpec(
        test_name="test_p256verify_uncachable",
        target_operation="P256VERIFY",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000100",
        filter_by=["p256verify"],
        model_params={"target_coef": "PRECOMPILE_P256VERIFY"},
    ),
    "precompile_point_evaluation": ModelSpec(
        test_name="test_point_evaluation",
        target_operation="POINT_EVALUATION",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000a",
        filter_by=["point_evaluation"],
        model_params={"target_coef": "PRECOMPILE_POINT_EVALUATION"},
    ),
    "precompile_point_evaluation_uncachable": ModelSpec(
        test_name="test_point_evaluation_uncachable",
        target_operation="POINT_EVALUATION",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000a",
        filter_by=["point_evaluation"],
        model_params={"target_coef": "PRECOMPILE_POINT_EVALUATION"},
    ),
    "precompile_bn128_add": ModelSpec(
        test_name="test_alt_bn128",
        target_operation="ECADD",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000006",
        # The add-family covers four EEST variants (`bn128_add`,
        # `bn128_add_negative`, `bn128_add_infinities`, `bn128_double`); the
        # selectors must not reach the mul-family or the pairing variants
        # (`bn128_one_pairing`, `bn128_two_pairings`) that share the
        # `test_alt_bn128` test_name — a bare `bn128_` prefix matches them.
        # The `bn128` fixture-param (parsed from the variant token) carries
        # the variant label so NNLS fits one model per variant.
        filter_by=["bn128_add", "bn128_double"],
        model_by=["bn128"],
        model_params={"target_coef": "PRECOMPILE_ECADD"},
    ),
    "precompile_bn128_mul": ModelSpec(
        test_name="test_alt_bn128",
        target_operation="ECMUL",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000007",
        filter_by=["bn128_mul_"],
        model_params={"target_coef": "PRECOMPILE_ECMUL"},
    ),
    "precompile_bn128_add_uncachable": ModelSpec(
        test_name="test_alt_bn128_uncachable",
        target_operation="ECADD",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000006",
        filter_by=["ec_add"],
        model_params={"target_coef": "PRECOMPILE_ECADD"},
    ),
    "precompile_bn128_mul_uncachable": ModelSpec(
        test_name="test_alt_bn128_uncachable",
        target_operation="ECMUL",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000007",
        filter_by=["ec_mul_"],
        model_params={"target_coef": "PRECOMPILE_ECMUL"},
    ),
    "precompile_bn128_pairing": ModelSpec(
        test_name="test_alt_bn128_benchmark",
        target_operation="ECPAIRING",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000008",
        filter_by=["num_pairs"],
        model_params={
            "target_coef": "PRECOMPILE_ECPAIRING_BASE",
            "num_pairs": "PRECOMPILE_ECPAIRING_PER_POINT",
        },
    ),
    "precompile_bn128_pairing_alt": ModelSpec(
        test_name="test_ec_pairing",
        target_operation="ECPAIRING",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000008",
        model_params={
            "target_coef": "PRECOMPILE_ECPAIRING_BASE",
            "num_pairs": "PRECOMPILE_ECPAIRING_PER_POINT",
        },
    ),
    # -------------------------------------------------------------------
    # BLS12-381 (6)
    # -------------------------------------------------------------------
    "precompile_bls_g1add": ModelSpec(
        test_name="test_bls12_381",
        target_operation="BLS12_G1ADD",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000b",
        filter_by=["bls12_g1add"],
        model_params={"target_coef": "PRECOMPILE_BLS_G1ADD"},
    ),
    "precompile_bls_g2add": ModelSpec(
        test_name="test_bls12_381",
        target_operation="BLS12_G2ADD",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000d",
        filter_by=["bls12_g2add"],
        model_params={"target_coef": "PRECOMPILE_BLS_G2ADD"},
    ),
    "precompile_bls_fp_to_g1": ModelSpec(
        test_name="test_bls12_381",
        target_operation="BLS12_MAP_FP_TO_G1",
        # EIP-2537 assigns the G1/G2 maps to addresses 0x10 and 0x11.
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000010",
        filter_by=["bls12_fp_to_g1"],
        model_params={"target_coef": "PRECOMPILE_BLS_G1MAP"},
    ),
    "precompile_bls_fp_to_g2": ModelSpec(
        test_name="test_bls12_381",
        target_operation="BLS12_MAP_FP2_TO_G2",
        target_operation_count_source="PRECOMPILE_0x0000000000000000000000000000000000000011",
        filter_by=["bls12_fp_to_g2"],
        model_params={"target_coef": "PRECOMPILE_BLS_G2MAP"},
    ),
    "precompile_bls_g1msm": ModelSpec(
        test_name="test_bls12_g1_msm",
        target_operation="BLS12_G1MSM",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000c",
        filter_by=["bls12_g1msm"],
        model_by=["k"],
        model_params={"target_coef": "PRECOMPILE_BLS_G1MUL"},
    ),
    "precompile_bls_g2msm": ModelSpec(
        test_name="test_bls12_g2_msm",
        target_operation="BLS12_G2MSM",
        target_operation_count_source="PRECOMPILE_0x000000000000000000000000000000000000000e",
        filter_by=["bls12_g2msm"],
        model_by=["k"],
        model_params={"target_coef": "PRECOMPILE_BLS_G2MUL"},
    ),
}


def get_preset(name: str) -> ModelSpec:
    """Return the preset registered under ``name``.

    Raises:
        KeyError: when ``name`` is not a known preset.
    """
    return PRESETS[name]
